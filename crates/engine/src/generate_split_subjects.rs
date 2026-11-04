//! Split into Layers (since 2026-11-02; peeling since 2026-11-03): SAM 3.1 finds what the
//! picture is of, term by term (the people, the animals, the things the vision-language model
//! names, or the caller's list), each instance becomes its own layer with the picture's pixels
//! and the detector's (edge-refined) coverage as alpha, and the layers stack in depth order,
//! read from who hides whom ([`crate::generate_split_order`]). With `complete` the subjects
//! are peeled off nearest first: each one's hole is repainted by the generative removal before
//! the next is cut, so the fill paints what each subject hid (the lap behind a table, the
//! shoulder behind a nearer person), every layer is whole, and what is left at the end is the
//! background with nobody in it. Without it one fill repaints every subject at once and each
//! layer holds its visible part. On the owner's portrait the one-fill split did 5 of 5 seeds
//! cleanly in about 4.6 s on the RTX 5090, where Qwen-Image-Layered, which this replaced,
//! separated 5 of 10 seeds at about 12 s (measured 2026-11-02; the owner chose SAM alone).

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use photocraft_doc::{Document, LayerId};
use photocraft_genai::GenerativeBackend;
use photocraft_geom::Rect;

use crate::generate_split_order::{self as order, Shape};
use crate::jobs::JobCtx;
use crate::select_ml_cmds::{Instance, SubjectFinder};
use crate::{EngineError, Result};

use super::{
    BACKGROUND_PROMPT, Common, DEFAULT_MATTE_TEMPLATE, FillJob, GenerativeInfo, Gray8, JobProgress, Layer, MATTE_DEFAULT_PROMPT, MAX_GROW, PEEL_PROMPT,
    PREFILL, REMOVE_BG, REMOVE_MIN_REQUEST, Request, Rgba8, SPLIT, Session, TONE_MATCH_MAX, Task, Tuning, Value, at_least, background_job, box_blur, dilate,
    embed_coverage, fill_rect, fill_region, gen_err, json, list_things, matte_request_size, name_selected, plan_common, removal_job, removal_prompt,
    remove_grow, request_size, resize_gray8, resize_rgba8, resolve_auto, set_generative_info, write_rgba8,
};

/// The most subjects one split gives their own layer (`layers` - 1 at most): eleven filled up
/// on the café photo (seven people, three cups and the cards) before its table was searched
/// (measured in the app 2026-11-04).
pub(super) const MAX_SUBJECTS: usize = 19;

pub(super) struct SubjectsPlan {
    /// What to cut out (`prompt`, a list of terms); none: the people, the animals, the things
    /// the vision-language model names, then Select Subject's prompts in turn.
    pub prompt: Option<String>,
    /// At most this many subject layers (`layers` - 1, the background being one).
    pub subjects: usize,
    pub seed: Option<u64>,
    pub name: Option<String>,
    pub edges: Edges,
    /// Peel the subjects off nearest first and fill behind each (`complete`).
    pub complete: bool,
    /// Find the subjects again on the filled picture after each removal (`sweep`), instead of
    /// finding them all first.
    pub sweep: bool,
    pub order: Order,
    pub group: Group,
}

/// Where the subjects' soft edges (hair) come from (`edges`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Edges {
    /// `auto`: ViTMatte when the server has it, else the matte when it agrees with the
    /// detector, else `refine`.
    Auto,
    /// `vitmatte`: ViTMatte inside a band around the detector's mask (hair strand by strand),
    /// else as `auto`.
    Vitmatte,
    /// `matte`: Remove Background's matte model around the detector's mask, else `refine`.
    Matte,
    /// `refine`: the detector's hard mask, edge refined from the pixels (Select Subject's).
    Refine,
    /// `hard`: the detector's hard mask as it is.
    Hard,
}

impl Edges {
    pub(super) fn parse(v: Option<&str>) -> std::result::Result<Edges, String> {
        Ok(match v.map(str::trim).unwrap_or("auto") {
            "" | "auto" => Edges::Auto,
            "vitmatte" => Edges::Vitmatte,
            "matte" => Edges::Matte,
            "refine" => Edges::Refine,
            "hard" => Edges::Hard,
            other => return Err(format!("`edges` must be auto, vitmatte, matte, refine or hard (got `{other}`)")),
        })
    }

    fn name(self) -> &'static str {
        match self {
            Edges::Auto | Edges::Vitmatte => "vitmatte",
            Edges::Matte => "matte",
            Edges::Refine => "refine",
            Edges::Hard => "hard",
        }
    }
}

/// How the layers stack (`order`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Order {
    /// `auto`: who hides whom, read from the masks; then the lower bottom edge; then size.
    Auto,
    /// `size`: the largest lowest (the rule until 2026-11-03).
    Size,
    /// `found`: as the detector found them, the first lowest.
    Found,
}

impl Order {
    pub(super) fn parse(v: Option<&str>) -> std::result::Result<Order, String> {
        Ok(match v.map(str::trim).unwrap_or("auto") {
            "" | "auto" => Order::Auto,
            "size" => Order::Size,
            "found" => Order::Found,
            other => return Err(format!("`order` must be auto, size or found (got `{other}`)")),
        })
    }

    fn name(self) -> &'static str {
        match self {
            Order::Auto => "auto",
            Order::Size => "size",
            Order::Found => "found",
        }
    }
}

/// Which instances share a layer (`group`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Group {
    /// `auto`: two instances whose masks overlap by half of the smaller.
    Auto,
    /// `touching`: any two whose masks touch.
    Touching,
    /// `none`: each its own layer.
    None,
}

impl Group {
    pub(super) fn parse(v: Option<&str>) -> std::result::Result<Group, String> {
        Ok(match v.map(str::trim).unwrap_or("auto") {
            "" | "auto" => Group::Auto,
            "touching" => Group::Touching,
            "none" => Group::None,
            other => return Err(format!("`group` must be auto, touching or none (got `{other}`)")),
        })
    }
}

/// One term of the prompt: what to ask the detector for, and at most how many of it.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Term {
    pub name: String,
    pub max: Option<usize>,
}

/// The prompt as a list of terms: split on commas, semicolons and " and "; `cup:3` asks for at
/// most three; duplicates (with or without "the") and empties go. The words stay as typed for
/// the detector ("the lighthouse"); the layer and the removal drop the article.
pub(super) fn parse_terms(prompt: &str) -> Vec<Term> {
    let mut out: Vec<Term> = Vec::new();
    for raw in prompt.split([',', ';', '\n']).flat_map(|p| p.split(" and ")) {
        let raw = raw.trim();
        let (name, max) = match raw.rsplit_once(':') {
            Some((n, c)) if c.trim().parse::<usize>().is_ok() => (n.trim(), c.trim().parse::<usize>().ok().filter(|n| *n > 0)),
            _ => (raw, None),
        };
        let name = name.trim().trim_matches(|c: char| c == '.' || c == '"');
        if name.is_empty() || out.iter().any(|t| bare(&t.name).eq_ignore_ascii_case(bare(name))) {
            continue;
        }
        out.push(Term { name: name.to_string(), max });
    }
    out
}

/// `term` without a leading article.
fn bare(term: &str) -> &str {
    let t = term.trim();
    for a in ["the ", "The ", "a ", "A ", "an ", "An "] {
        if let Some(rest) = t.strip_prefix(a) {
            return rest.trim();
        }
    }
    t
}

/// A subject on its way to a layer.
pub(super) struct Subject {
    /// The detector's coverage (hard), grown by what the fills completed.
    pub(super) cov: Vec<f32>,
    pub(super) bounds: Rect,
    pub(super) pixels: u64,
    /// What it was found as ("person", "cup", "the main subject").
    pub(super) term: String,
    /// The caller named the term.
    pub(super) asked: bool,
    /// Its soft alpha (hair), from `edges`.
    pub(super) alpha: Vec<f32>,
    /// The fill completed what nearer subjects hid (or nothing hid it).
    pub(super) completed: bool,
    /// Its mask grew since its alpha was matted.
    pub(super) grown: bool,
}

impl Subject {
    fn shape(&self) -> Shape<'_> {
        Shape { cov: &self.cov, bounds: self.bounds, pixels: self.pixels }
    }

    /// The detector's word for the check: a kind of thing, never a generic prompt.
    fn check_term(&self) -> &str {
        check_prompt(&self.term)
    }

    /// What the removal names ("Remove the person"): the term, or "person" for a generic one.
    fn removal_name(&self) -> String {
        bare(self.check_term()).to_string()
    }
}

pub(super) fn run(s: &mut Session, plan: SubjectsPlan) -> Result<Value> {
    let SubjectsPlan { prompt, subjects, seed, name, edges, complete, sweep, order: ordering, group } = plan;
    let finder = crate::select_ml_cmds::subject_finder(s, prompt.as_deref())?;
    let terms: Option<Vec<Term>> = prompt.as_deref().map(parse_terms).filter(|t| !t.is_empty());
    let tuning = Tuning { fast_draft: s.prefs().integrations.fast_draft, ..Tuning::default() };
    // Each hole's tries: the removal (clean on most pictures), then the grey-box fills. Behind a
    // subject with others still behind it, the scene continued; behind the last, the backdrop.
    // Not Krea 2 AnyPaint, though it is quick and keeps every pixel around the hole: as the
    // first try on the café photo it painted a second copy of the woman beside her and a man
    // who was never there, swapped one seed's room, and turned a plain studio backdrop into an
    // invented room; and every switch between it and the 2.1 family reloaded the other model,
    // about 60 s a split. One model family serves the whole split (measured in the app
    // 2026-11-04).
    let mut job = removal_job(s, "Split into Layers", 0, false, false, if complete { PEEL_VARIATIONS } else { FLAT_VARIATIONS }, seed, true, tuning.clone())?;
    // A split's fills are judged by what they painted, so the membrane moves them at most
    // [`TONE_MATCH_MAX`] levels (the Remove tools match all the way).
    job.tone_most = TONE_MATCH_MAX / 255.0;
    let peel_box = background_job(s, "Split into Layers", seed, tuning.clone(), PEEL_PROMPT).ok();
    let last_box = background_job(s, "Split into Layers", seed, tuning, BACKGROUND_PROMPT).ok();
    let base = name;
    // The matte request (Remove Background's model), planned now; used when the server has it.
    let matte = plan_common(
        s,
        REMOVE_BG,
        &json!({"prompt": MATTE_DEFAULT_PROMPT, "template": DEFAULT_MATTE_TEMPLATE, "seed": seed}),
        DEFAULT_MATTE_TEMPLATE,
        Task::Matte,
        "",
    )
    .ok();
    let budget = subjects.clamp(1, MAX_SUBJECTS);
    crate::jobs::edit_job(
        s,
        "Split into Layers",
        move |doc, active, ctx| {
            let started = std::time::Instant::now();
            let area = doc.bounds();
            let (w, h) = (area.width(), area.height());
            let (wu, hu) = (w as usize, h as usize);
            // The picture as it is, for the subjects' pixels and the detector.
            let picture = photocraft_compose::render(doc, area).to_rgba8().pixels;
            let template_id = job.template_id.clone();
            if sweep {
                let swept = sweep_subjects(
                    doc,
                    active,
                    ctx,
                    area,
                    &finder,
                    &job,
                    matte.as_ref(),
                    &picture,
                    terms.as_deref(),
                    budget,
                    (group, ordering, edges),
                    (peel_box.as_ref(), last_box.as_ref()),
                )?;
                let Swept { subjects, cut, background, left_in, tries, found_by, used, fill_ms, fill_seed, generations } = swept;
                ctx.progress(0.95, "Placing");
                let left: Vec<Vec<f32>> = left_in.iter().map(|&i| hole_of(&subjects[i])).collect();
                let behind_left = left_behind(&finder, &background, area, &left, ctx)?;
                let (ids, placed, incomplete) =
                    place_all(doc, active, area, base.as_deref(), &found_by, &template_id, fill_seed, &subjects, cut, background, &left_in)?;
                return Ok((
                    ids,
                    placed,
                    incomplete,
                    tries,
                    found_by,
                    template_id,
                    fill_seed,
                    (fill_ms, generations),
                    started.elapsed().as_millis() as u64,
                    (w, h),
                    used.name(),
                    ordering.name(),
                    behind_left,
                ));
            }
            ctx.progress(0.0, "Finding subjects");
            let (mut found, found_by) = find_subjects(&finder, job.thinker.as_ref(), &picture, area, terms.as_deref(), budget, job.seed, group, ctx)?;
            ctx.check()?;
            found.truncate(budget);
            if found.is_empty() {
                return Err(EngineError::Other(format!("nothing matching \"{found_by}\" was found")));
            }
            let found_by = kept_terms(&found_by, &found);
            // Each subject's alpha: ViTMatte inside a band around the detector's mask (hair,
            // strand by strand), else the matte model's soft alpha around it, else the mask with
            // the edge refined from the pixels, as Select Subject does (or as `edges` asks).
            let backend = Arc::clone(&job.backend);
            let used = matte_all(backend.as_ref(), matte.as_ref(), doc, *active, &picture, area, &mut found, None, edges, ctx)?;
            ctx.check()?;
            // Front to back.
            ctx.progress(0.2, "Ordering by depth");
            let mut subjects: Vec<Subject> = in_depth_order(found, ordering, wu, hu);
            let below = *active;
            let mut fill_ms = 0u64;
            let mut fill_seed = 0u64;
            let mut generations = 0u32;
            // The subjects (front-to-back indices) the fill could not clear: left in the background.
            let mut left_in: Vec<usize> = Vec::new();
            // Each subject's layer pixels, front to back, and the background.
            let mut cut: Vec<Vec<u8>> = Vec::with_capacity(subjects.len());
            // Every fill tried, in order, with the check's verdict (`tries` in the result): what
            // was refused and why can be audited (round 4's cup refusals could not be, 2026-11-04).
            let mut tries: Vec<Value> = Vec::new();
            let background: Vec<u8> = if complete {
                // The picture as it stands: each peel takes one subject out of it.
                let mut p = picture.clone();
                let n = subjects.len();
                for i in 0..n {
                    let (lo, hi) = (0.22 + 0.72 * i as f32 / n as f32, 0.22 + 0.72 * (i + 1) as f32 / n as f32);
                    let label = layer_label(&subjects, i);
                    ctx.progress(lo, &format!("Filling behind {label} ({} of {n})", i + 1));
                    // Its layer: the picture now (nearer subjects gone) under its alpha.
                    cut.push(under_alpha(&p, &subjects[i].alpha));
                    let hole = hole_of(&subjects[i]);
                    let (rest, this) = subjects.split_at_mut(i);
                    let _ = rest;
                    let (this, behind) = this.split_at_mut(1);
                    let this = &this[0];
                    let last = behind.is_empty();
                    let mut removal = job.clone();
                    removal.rewritten = Some(removal_prompt(&this.removal_name()));
                    removal.object = Some(this.removal_name());
                    let box_job = if last { last_box.as_ref() } else { peel_box.as_ref() };
                    let (hole_rect, _) = crate::select_ml_cmds::coverage_bounds(&hole, area);
                    let grow = remove_grow(hole_rect).min(MAX_GROW) as usize;
                    let grown_hole = dilate(&hole, wu, hu, grow);
                    let asks = asks_for(this.check_term(), behind.iter());
                    let behind_shapes: Vec<(String, Vec<f32>, u64)> = behind.iter().map(|b| (b.check_term().to_string(), b.cov.clone(), b.pixels)).collect();
                    let finder_ref = &finder;
                    let grown_ref = &grown_hole;
                    let hole_ref = &hole;
                    let before = &p;
                    let cut = (this.check_term(), this.cov.as_slice(), this.pixels);
                    let mut check = |filled: &[u8], touched: &[f32], patched: bool| -> Result<Verdict<Vec<(String, Instance)>>> {
                        let mut seen: Vec<(String, Instance)> = Vec::new();
                        for (term, at_most) in &asks {
                            for inst in seen_of(finder_ref, filled, area, term, *at_most, ctx)? {
                                seen.push((term.clone(), inst));
                            }
                        }
                        // How much of a detection's part of the hole still shows what was cut.
                        let looks = |cov: &[f32]| kept_share(before, filled, &both(cov, hole_ref), wu, hu);
                        let own = seen
                            .iter()
                            .find_map(|(term, (cov, b, _))| judge(term, cov, *b, hole_ref, grown_ref, touched, cut, &behind_shapes, &looks, wu, patched));
                        Ok((seen, own))
                    };
                    // The subjects still behind keep their own pixels through this fill.
                    let protect = protect_mask(behind, wu * hu);
                    // A person's silhouette is watched by its pixels too (a cup's is not: a cup on
                    // a table of its own colour looks kept whatever the fill did).
                    let watch = creature(this.check_term()).then_some(hole.as_slice());
                    let thing = !creature(this.check_term());
                    let HoleFill { outcome, tries: hole_tries, ms, generations: made } =
                        fill_hole(doc, active, below, ctx, area, &p, &hole, Some(&protect), watch, thing, &removal, box_job, (lo, hi), &mut check)?;
                    tries.extend(hole_tries.into_iter().map(|mut t| {
                        t["behind"] = json!(label.as_str());
                        t
                    }));
                    *active = below;
                    fill_ms += ms;
                    generations += made;
                    let Some((filled, seed_used, seen)) = outcome else {
                        left_in.push(i);
                        continue;
                    };
                    fill_seed = seed_used;
                    // The subjects behind, completed where the fill showed more of them.
                    let any_grew = grow_behind(behind, &seen, &grown_hole, area);
                    p = filled;
                    if any_grew {
                        let (_, rest) = subjects.split_at_mut(i + 1);
                        matte_all(backend.as_ref(), matte.as_ref(), doc, *active, &p, area, rest, None, edges, ctx)?;
                        for b in rest.iter_mut() {
                            b.grown = false;
                        }
                    }
                    ctx.check()?;
                }
                if left_in.len() == subjects.len() {
                    return Err(could_not_clear(&subjects[0].removal_name(), &tries));
                }
                // A subject is whole unless a nearer one that touches it could not be cleared.
                let shapes: Vec<Shape<'_>> = subjects.iter().map(Subject::shape).collect();
                let whole: Vec<bool> =
                    (0..subjects.len()).map(|j| !left_in.iter().any(|&i| i < j && order::touching(&shapes[i], &shapes[j], wu, hu))).collect();
                for (sub, w) in subjects.iter_mut().zip(whole) {
                    sub.completed = w;
                }
                p
            } else {
                // One hole, one fill: every subject at once, as the Remove tool would.
                for sub in &subjects {
                    cut.push(under_alpha(&picture, &sub.alpha));
                }
                let mut hole = vec![0.0f32; wu * hu];
                for sub in &subjects {
                    for (d, c) in hole.iter_mut().zip(hole_of(sub)) {
                        *d = d.max(c);
                    }
                }
                // The removal names what it removes ("Remove the woman…"): the 2.1 route is an
                // edit model with no noise mask, and told only "the object" it changed nothing (the
                // woman stayed in the background, measured in the app 2026-11-02). A person is
                // named "person": the vision-language model named one by their clothes ("the blue
                // tank top") and the edit model, told to remove the top, kept the person and
                // repainted their body on every seed (measured in the app 2026-11-03).
                let name = match subjects.iter().find(|s| s.check_term() == PERSON) {
                    Some(_) => PERSON.to_string(),
                    None if subjects.iter().any(|s| s.asked) => {
                        let mut names: Vec<String> = Vec::new();
                        for s in &subjects {
                            if !names.contains(&s.term) {
                                names.push(s.term.clone());
                            }
                        }
                        names.join(" and ")
                    }
                    None => name_selected(doc, &job, area, &hole, ctx)?.unwrap_or_else(|| subjects[0].removal_name()),
                };
                let name = name.trim().strip_prefix("the ").unwrap_or(name.trim()).to_string();
                ctx.check()?;
                let mut removal = job.clone();
                removal.rewritten = Some(removal_prompt(&name));
                removal.object = Some(name.clone());
                let mut asks: Vec<(String, usize)> = Vec::new();
                for sub in &subjects {
                    match asks.iter_mut().find(|(t, _)| t == sub.check_term()) {
                        Some((_, n)) => *n += 1,
                        None => asks.push((sub.check_term().to_string(), 1)),
                    }
                }
                if !asks.iter().any(|(t, _)| t == PERSON) {
                    asks.push((PERSON.to_string(), 1));
                }
                let finder_ref = &finder;
                let (hole_rect, holed) = crate::select_ml_cmds::coverage_bounds(&hole, area);
                let grown_hole = dilate(&hole, wu, hu, remove_grow(hole_rect).min(MAX_GROW) as usize);
                let mut check = |filled: &[u8], touched: &[f32], _: bool| -> Result<Verdict<()>> {
                    for (term, at_most) in &asks {
                        for (cov, b, _) in seen_of(finder_ref, filled, area, term, *at_most, ctx)? {
                            if mostly_in_hole(&cov, &grown_hole, LEFT_IN_HOLE) {
                                return Ok(((), Some(format!("the {term} still in the hole"))));
                            }
                            if creature(term)
                                && let Some(why) = someone_new(term, &cov, b, touched, (term, &hole, holed), &[])
                            {
                                return Ok(((), Some(why)));
                            }
                        }
                    }
                    Ok(((), None))
                };
                let HoleFill { outcome, tries: hole_tries, ms, generations: made } =
                    fill_hole(doc, active, below, ctx, area, &picture, &hole, None, None, false, &removal, last_box.as_ref(), (0.3, 0.9), &mut check)?;
                tries.extend(hole_tries.into_iter().map(|mut t| {
                    t["behind"] = json!("all");
                    t
                }));
                *active = below;
                fill_ms += ms;
                generations += made;
                let Some((filled, seed_used, ())) = outcome else {
                    return Err(could_not_clear(&name, &tries));
                };
                fill_seed = seed_used;
                for sub in subjects.iter_mut() {
                    sub.completed = false;
                }
                filled
            };
            ctx.progress(0.95, "Placing");
            let left: Vec<Vec<f32>> = left_in.iter().map(|&i| hole_of(&subjects[i])).collect();
            let behind_left = left_behind(&finder, &background, area, &left, ctx)?;
            let (ids, placed, incomplete) =
                place_all(doc, active, area, base.as_deref(), &found_by, &template_id, fill_seed, &subjects, cut, background, &left_in)?;
            Ok((
                ids,
                placed,
                incomplete,
                tries,
                found_by,
                template_id,
                fill_seed,
                (fill_ms, generations),
                started.elapsed().as_millis() as u64,
                (w, h),
                used.name(),
                ordering.name(),
                behind_left,
            ))
        },
        move |(ids, placed, incomplete, tries, found_by, template_id, seed, (fill_ms, generations), total_ms, (w, h), matted, ordered, left_behind)| {
            json!({
                "layers": ids.iter().map(|l| l.0).collect::<Vec<_>>(), "count": ids.len(),
                "subjects": placed, "incomplete": incomplete, "fills": tries.len(), "generations": generations, "tries": tries, "order": ordered,
                "foundBy": found_by, "template": template_id, "seed": seed, "leftBehind": left_behind,
                "width": w, "height": h, "fillMs": fill_ms, "ms": total_ms, "edges": matted,
            })
        },
    )
}

/// What a sweep made: the subjects front to back (as cut), each one's layer pixels, the
/// background, who was left in it, every try's record, what found them, the edges used and
/// the fills' time, last seed and generations.
struct Swept {
    subjects: Vec<Subject>,
    cut: Vec<Vec<u8>>,
    background: Vec<u8>,
    left_in: Vec<usize>,
    tries: Vec<Value>,
    found_by: String,
    used: Edges,
    fill_ms: u64,
    fill_seed: u64,
    generations: u32,
}

/// The split as a sweep (`sweep`): the subjects are found on the picture as it stands, the
/// nearest is cut out and its hole filled, and the picture is searched again, until nothing is
/// found or the budget is spent (the owner's "a batch subject select and remove until there is
/// nothing left", 2026-11-04). Whoever stands behind is found anew on the filled picture, whole
/// where the fill completed them, instead of grown from the first detection by what joins it.
///
/// The first pass's subjects, front to back, are the sweep's list. Each later pass looks for
/// them again (the people, the animals, the things the vision-language model names, or the
/// caller's terms) and for anything new; a find that holds half of a listed subject's mask, or
/// that the mask holds half of, is that subject seen again and refreshes its mask (the union of
/// the two); a find that is two of them at once (the detector joined neighbours) is neither; a
/// new find mostly inside the filled holes is the fill's own work (a grey blob, a sliver of
/// trousers) and dropped, the rest join the list after it, in depth order. The nearest on the
/// list is cut at its turn, by its fresh mask or, when the detector misses it, by its first one,
/// so nobody the first pass saw stays in the background (the striped man, the waiter and the
/// man at the frame's edge, found again mostly inside their neighbours' filled holes, were
/// dropped as debris and stayed, 3 of 3 seeds; the pass order alone had stacked the striped man
/// over the fedora man he stands behind; both measured in the app 2026-11-04). Everyone on the
/// list keeps their own pixels through each fill. A subject no fill could clear stays in the
/// background and is not looked for again; a subject is matted (hair) when its turn comes, and
/// only then.
#[allow(clippy::too_many_arguments)]
fn sweep_subjects(
    doc: &mut Document,
    active: &mut Option<LayerId>,
    ctx: &JobCtx,
    area: Rect,
    finder: &SubjectFinder,
    job: &FillJob,
    matte: Option<&Common>,
    picture: &[u8],
    terms: Option<&[Term]>,
    budget: usize,
    (group, ordering, edges): (Group, Order, Edges),
    (peel_box, last_box): (Option<&FillJob>, Option<&FillJob>),
) -> Result<Swept> {
    let (w, h) = (area.width(), area.height());
    let (wu, hu) = (w as usize, h as usize);
    let below = *active;
    let backend = Arc::clone(&job.backend);
    ctx.progress(0.0, "Finding subjects");
    let (first, found_by) = find_subjects(finder, job.thinker.as_ref(), picture, area, terms, budget, job.seed, group, ctx)?;
    ctx.check()?;
    let found_by = kept_terms(&found_by, &first);
    // What the passes after the first look for: the caller's terms with their caps, else the
    // kinds the first pass found.
    let again: Vec<Term> = match terms {
        Some(t) => t.to_vec(),
        None => found_by.split(", ").map(|t| Term { name: t.to_string(), max: None }).collect(),
    };
    let mut pending: Vec<Subject> = in_depth_order(first, ordering, wu, hu);
    pending.truncate(budget);
    let mut done: Vec<Subject> = Vec::new();
    let mut cut: Vec<Vec<u8>> = Vec::new();
    let mut left_in: Vec<usize> = Vec::new();
    let mut given_up: Vec<Vec<f32>> = Vec::new();
    // The holes the fills have painted so far: a new find mostly inside them is the fill's own
    // work, not a subject.
    let mut painted = vec![0.0f32; wu * hu];
    let mut tries: Vec<Value> = Vec::new();
    let mut used = edges;
    let (mut fill_ms, mut fill_seed, mut generations) = (0u64, 0u64, 0u32);
    let mut p = picture.to_vec();
    let mut pass = 0usize;
    while done.len() < budget {
        pass += 1;
        if pass > 1 {
            // Look again on the picture as it stands: for everyone on the list, and for more
            // within the budget and each kind's cap (counted over every pass: three cups in
            // all, not three more each time).
            let room = budget - done.len() - pending.len();
            let mut found: Vec<Subject> = Vec::new();
            for t in &again {
                ctx.check()?;
                let listed = pending.iter().filter(|s| s.term.eq_ignore_ascii_case(&t.name)).count();
                let have = listed + done.iter().filter(|s| s.term.eq_ignore_ascii_case(&t.name)).count();
                let more = t.max.map_or(if creature(&t.name) { room } else { THINGS_EACH.saturating_sub(have) }, |m| m.saturating_sub(have)).min(room);
                let at_most = listed + more;
                if at_most == 0 {
                    continue;
                }
                for (cov, bounds, pixels) in finder.find_in(&p, area, &t.name, at_most, ctx)? {
                    found.push(Subject {
                        cov,
                        bounds,
                        pixels,
                        term: t.name.clone(),
                        asked: terms.is_some(),
                        alpha: Vec::new(),
                        completed: false,
                        grown: false,
                    });
                }
                clean(&mut found, wu, hu, group);
            }
            // Not what was given up on: it stays in the background.
            found.retain(|s| !given_up.iter().any(|g| overlap(&s.cov, g) * 2 >= s.pixels));
            let mut fresh: Vec<Option<Vec<f32>>> = vec![None; pending.len()];
            let mut new: Vec<Subject> = Vec::new();
            for s in found {
                let same: Vec<usize> = pending
                    .iter()
                    .enumerate()
                    .filter(|(_, q)| q.term.eq_ignore_ascii_case(&s.term) && same_subject(&s.cov, s.pixels, &q.cov, q.pixels))
                    .map(|(i, _)| i)
                    .collect();
                match same.as_slice() {
                    [i] => {
                        let m = fresh[*i].get_or_insert_with(|| pending[*i].cov.clone());
                        for (d, c) in m.iter_mut().zip(&s.cov) {
                            *d = d.max(*c);
                        }
                    }
                    [] => new.push(s),
                    _ => {}
                }
            }
            for (q, f) in pending.iter_mut().zip(fresh) {
                if let Some(cov) = f {
                    let (bounds, pixels) = crate::select_ml_cmds::coverage_bounds(&cov, area);
                    q.cov = cov;
                    q.bounds = bounds;
                    q.pixels = pixels;
                }
            }
            new.retain(|s| (overlap(&s.cov, &painted) as f32) < DEBRIS_IN_PAINTED * s.pixels as f32);
            new.retain(|s| {
                !pending.iter().any(|q| {
                    let both = overlap(&s.cov, &q.cov);
                    both as f64 >= PART_INSIDE * s.pixels.max(1) as f64 || both as f64 >= DUPLICATE_IOU * (s.pixels + q.pixels - both).max(1) as f64
                })
            });
            let mut new = in_depth_order(new, ordering, wu, hu);
            new.truncate(room);
            pending.extend(new);
        }
        if pending.is_empty() {
            break;
        }
        let k = done.len();
        let (lo, hi) = (0.1 + 0.85 * k as f32 / budget as f32, 0.1 + 0.85 * (k + 1) as f32 / budget as f32);
        ctx.progress(lo, &format!("Cutting out the {} ({} of up to {budget})", bare(pending[0].check_term()), k + 1));
        // The nearest gets its soft edge now (its band kept off the others' bodies); the rest
        // wait for their turn.
        let u = matte_all(backend.as_ref(), matte, doc, *active, &p, area, &mut pending, Some(0), edges, ctx)?;
        if k == 0 {
            used = u;
        }
        ctx.check()?;
        let this = pending.remove(0);
        cut.push(under_alpha(&p, &this.alpha));
        let hole = hole_of(&this);
        let mut removal = job.clone();
        removal.rewritten = Some(removal_prompt(&this.removal_name()));
        removal.object = Some(this.removal_name());
        let box_job = if pending.is_empty() { last_box } else { peel_box };
        let (hole_rect, _) = crate::select_ml_cmds::coverage_bounds(&hole, area);
        let grown_hole = dilate(&hole, wu, hu, remove_grow(hole_rect).min(MAX_GROW) as usize);
        let asks = asks_for(this.check_term(), pending.iter());
        let behind_shapes: Vec<(String, Vec<f32>, u64)> = pending.iter().map(|b| (b.check_term().to_string(), b.cov.clone(), b.pixels)).collect();
        let before = &p;
        let cut_shape = (this.check_term(), this.cov.as_slice(), this.pixels);
        let mut check = |filled: &[u8], touched: &[f32], patched: bool| -> Result<Verdict<Vec<(String, Instance)>>> {
            let mut seen: Vec<(String, Instance)> = Vec::new();
            for (term, at_most) in &asks {
                for inst in seen_of(finder, filled, area, term, *at_most, ctx)? {
                    seen.push((term.clone(), inst));
                }
            }
            let looks = |cov: &[f32]| kept_share(before, filled, &both(cov, &hole), wu, hu);
            let own =
                seen.iter().find_map(|(term, (cov, b, _))| judge(term, cov, *b, &hole, &grown_hole, touched, cut_shape, &behind_shapes, &looks, wu, patched));
            Ok((seen, own))
        };
        // Everyone still on the list keeps their own pixels through this fill.
        let protect = protect_mask(&pending, wu * hu);
        let watch = creature(this.check_term()).then_some(hole.as_slice());
        let thing = !creature(this.check_term());
        let HoleFill { outcome, tries: hole_tries, ms, generations: made } =
            fill_hole(doc, active, below, ctx, area, &p, &hole, Some(&protect), watch, thing, &removal, box_job, (lo, hi), &mut check)?;
        // Named by its layer once every subject is known.
        tries.extend(hole_tries.into_iter().map(|mut t| {
            t["behind"] = json!(k);
            t
        }));
        *active = below;
        fill_ms += ms;
        generations += made;
        match outcome {
            Some((filled, seed_used, seen)) => {
                // Those on the list, completed where the fill showed more of them (as the peel
                // does), so one the next pass misses is still cut whole.
                grow_behind(&mut pending, &seen, &grown_hole, area);
                for b in pending.iter_mut() {
                    b.grown = false;
                }
                p = filled;
                fill_seed = seed_used;
                for (d, g) in painted.iter_mut().zip(&hole) {
                    *d = d.max(*g);
                }
            }
            None => {
                left_in.push(k);
                given_up.push(hole);
            }
        }
        done.push(this);
        ctx.check()?;
    }
    if done.is_empty() {
        return Err(EngineError::Other(format!("nothing matching \"{found_by}\" was found")));
    }
    if left_in.len() == done.len() {
        return Err(could_not_clear(&done[0].removal_name(), &tries));
    }
    // A subject is whole unless a nearer one that touches it could not be cleared.
    let shapes: Vec<Shape<'_>> = done.iter().map(Subject::shape).collect();
    let whole: Vec<bool> = (0..done.len()).map(|j| !left_in.iter().any(|&i| i < j && order::touching(&shapes[i], &shapes[j], wu, hu))).collect();
    for (sub, w) in done.iter_mut().zip(whole) {
        sub.completed = w;
    }
    let labels = subjects_labels(&done);
    for t in &mut tries {
        if let Some(k) = t["behind"].as_u64() {
            t["behind"] = json!(labels[k as usize]);
        }
    }
    Ok(Swept { subjects: done, cut, background: p, left_in, tries, found_by, used, fill_ms, fill_seed, generations })
}

/// The layers: the background lowest, then the subjects back to front (the nearest on top),
/// named "{base} k/n" or by their terms; the ids, each subject's record and the names of those
/// left in the background.
#[allow(clippy::too_many_arguments)]
fn place_all(
    doc: &mut Document,
    active: &mut Option<LayerId>,
    area: Rect,
    base: Option<&str>,
    found_by: &str,
    template_id: &str,
    fill_seed: u64,
    subjects: &[Subject],
    mut cut: Vec<Vec<u8>>,
    background: Vec<u8>,
    left_in: &[usize],
) -> Result<(Vec<LayerId>, Vec<Value>, Vec<String>)> {
    let (w, h) = (area.width(), area.height());
    // The background lowest, then the subjects back to front: the nearest on top.
    let count = 1 + subjects.len();
    let labels = subjects_labels(subjects);
    let mut ids: Vec<LayerId> = Vec::with_capacity(count);
    let mut placed: Vec<Value> = Vec::with_capacity(subjects.len());
    let place = |doc: &mut Document, active: &mut Option<LayerId>, pixels: Vec<u8>, layer_name: String| -> Result<LayerId> {
        let mut layer = Layer::raster(layer_name.clone(), doc.pixel_format());
        write_rgba8(&mut layer, &Rgba8::new(w, h, pixels).map_err(gen_err)?, area)?;
        set_generative_info(
            &mut layer,
            &GenerativeInfo {
                command: SPLIT.to_string(),
                prompt: found_by.to_string(),
                template: template_id.to_string(),
                seed: fill_seed,
                rect: [area.x0, area.y0, w as i32, h as i32],
                name: layer_name,
                ..GenerativeInfo::default()
            },
        );
        let nid = doc.insert_above(*active, layer);
        *active = Some(nid);
        Ok(nid)
    };
    let bg_name = match base {
        Some(b) => format!("{b} 1/{count}"),
        // Not a second "Background" beside the document's own.
        None if doc.layers.iter().any(|l| l.name == "Background") => "Split background".to_string(),
        None => "Background".to_string(),
    };
    ids.push(place(doc, active, background, bg_name)?);
    for (k, i) in (0..subjects.len()).rev().enumerate() {
        let sub = &subjects[i];
        let layer_name = match base {
            Some(b) => format!("{b} {}/{count}", k + 2),
            None => labels[i].clone(),
        };
        let pixels = std::mem::take(&mut cut[i]);
        let id = place(doc, active, pixels, layer_name.clone())?;
        ids.push(id);
        placed.push(json!({
            "layer": id.0, "name": layer_name, "term": sub.term, "pixels": sub.pixels, "completed": sub.completed,
            "bounds": [sub.bounds.x0, sub.bounds.y0, sub.bounds.width(), sub.bounds.height()],
        }));
    }
    let incomplete: Vec<String> = left_in
        .iter()
        .map(|&i| match base {
            Some(b) => format!("{b} {}/{count}", subjects.len() - i + 1),
            None => labels[i].clone(),
        })
        .collect();
    Ok((ids, placed, incomplete))
}

/// The terms of `found_by` that kept an instance through the cleaning (a term whose every
/// instance was a speck is not in the result); all of them when none did.
fn kept_terms(found_by: &str, found: &[Subject]) -> String {
    let kept: Vec<&str> = found_by.split(", ").filter(|t| found.iter().any(|s| s.term.split(" and ").any(|x| x.eq_ignore_ascii_case(t)))).collect();
    if kept.is_empty() { found_by.to_string() } else { kept.join(", ") }
}

/// The subjects front to back by `ordering`: who hides whom from the masks, the smallest
/// first (`size`: the largest ends lowest), or the detector's order reversed (`found`).
fn in_depth_order(found: Vec<Subject>, ordering: Order, w: usize, h: usize) -> Vec<Subject> {
    let front_first = front_first(&found, ordering, w, h);
    let mut subjects: Vec<Subject> = Vec::with_capacity(found.len());
    let mut taken: Vec<Option<Subject>> = found.into_iter().map(Some).collect();
    for i in front_first {
        if let Some(sub) = taken.get_mut(i).and_then(Option::take) {
            subjects.push(sub);
        }
    }
    subjects
}

/// The indices of `found` front to back by `ordering` (see [`in_depth_order`]).
fn front_first(found: &[Subject], ordering: Order, w: usize, h: usize) -> Vec<usize> {
    match ordering {
        Order::Auto => order::front_to_back(&found.iter().map(Subject::shape).collect::<Vec<_>>(), w, h),
        Order::Size => {
            let mut ix: Vec<usize> = (0..found.len()).collect();
            ix.sort_by_key(|i| found[*i].pixels);
            ix
        }
        Order::Found => (0..found.len()).rev().collect(),
    }
}

/// Whether two masks are one subject seen twice: either holds half of the other's pixels (a
/// sweep's later find against the first pass's mask: the whole of someone the fill completed
/// against their visible part, or a part of them against the whole).
pub(super) fn same_subject(a: &[f32], a_pixels: u64, b: &[f32], b_pixels: u64) -> bool {
    let smaller = a_pixels.min(b_pixels);
    smaller > 0 && overlap(a, b) * 2 >= smaller
}

/// Where the subjects still `behind` stand (their holes), one value per pixel: what a nearer
/// subject's fill must leave alone.
fn protect_mask(behind: &[Subject], n: usize) -> Vec<f32> {
    let mut protect = vec![0.0f32; n];
    for b in behind {
        for (d, c) in protect.iter_mut().zip(hole_of(b)) {
            *d = d.max(c);
        }
    }
    protect
}

/// The subjects `behind` grown into `grown_hole` where a detection of their kind that continues
/// them (`seen`) shows more of them; whether any grew enough to count as completed. Only what a
/// detection adds next to the subject counts; a patch of the fill the detector took for it
/// stays out (a detached dark block under the woman's layer, measured in the app 2026-11-03).
fn grow_behind(behind: &mut [Subject], seen: &[(String, Instance)], grown_hole: &[f32], area: Rect) -> bool {
    let (w, h) = (area.width() as usize, area.height() as usize);
    let mut any_grew = false;
    for b in behind.iter_mut() {
        let best =
            seen.iter().filter(|(t, (cov, _, _))| t == b.check_term() && continues(cov, &b.cov, b.pixels)).max_by_key(|(_, (cov, _, _))| overlap(cov, &b.cov));
        let Some((_, (cov, _, _))) = best else { continue };
        let wanted: Vec<bool> = b.cov.iter().zip(cov).zip(grown_hole).map(|((d, c), g)| *c >= 0.5 && *g >= 0.5 && *d < 0.5).collect();
        let added = joined_to(&wanted, &b.cov, w, h);
        let mut grew = 0u64;
        for (d, a) in b.cov.iter_mut().zip(&added) {
            if *a {
                *d = 1.0;
                grew += 1;
            }
        }
        if grew * 50 >= b.pixels {
            let (bounds, pixels) = crate::select_ml_cmds::coverage_bounds(&b.cov, area);
            b.bounds = bounds;
            b.pixels = pixels;
            b.grown = true;
            b.completed = true;
            any_grew = true;
        }
    }
    any_grew
}

/// The subjects' layer names in their (front to back) order: "{Term} {k}", numbered back to
/// front within a term, the number dropped when the term has one.
fn subjects_labels(subjects: &[Subject]) -> Vec<String> {
    let n = subjects.len();
    let mut labels = vec![String::new(); n];
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for s in subjects {
        *counts.entry(s.term.clone()).or_default() += 1;
    }
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for i in (0..n).rev() {
        let s = &subjects[i];
        let k = seen.entry(s.term.clone()).or_default();
        *k += 1;
        let term = title_case(&s.term);
        labels[i] = if counts.get(&s.term).copied().unwrap_or(1) > 1 { format!("{term} {k}") } else { term };
    }
    labels
}

fn layer_label(subjects: &[Subject], i: usize) -> String {
    subjects_labels(subjects).swap_remove(i)
}

/// "the main subject" → "Main subject", "cup" → "Cup".
fn title_case(term: &str) -> String {
    let t = bare(term);
    let mut c = t.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

/// `picture` under `alpha`: the pixels, their alpha scaled.
fn under_alpha(picture: &[u8], alpha: &[f32]) -> Vec<u8> {
    picture.as_chunks::<4>().0.iter().zip(alpha).flat_map(|(p, a)| [p[0], p[1], p[2], (f32::from(p[3]) * a.clamp(0.0, 1.0)).round() as u8]).collect()
}

/// A subject's hole: wherever its mask or its alpha shows it (the soft fringe is still the
/// subject's), all or nothing, as the Remove tool's.
fn hole_of(sub: &Subject) -> Vec<f32> {
    // A subject not matted yet (a sweep's, waiting for its turn) has its mask alone.
    let alpha = sub.alpha.iter().copied().chain(std::iter::repeat(0.0));
    sub.cov.iter().zip(alpha).map(|(c, a)| if c.max(a) > HOLE_MIN { 1.0 } else { 0.0 }).collect()
}

/// Whether a detection sits mostly in the hole and covers enough of it: `share` of the
/// detection inside the hole and at least [`LEFT_MIN_SHARE`] of the hole (a person beside the
/// hole overlaps it little; a speck is nothing). A detection over nearly the whole frame is the
/// detector's quirk, not a subject (it flagged one clean backdrop fill whole, 2026-11-03).
fn mostly_in_hole(cov: &[f32], hole: &[f32], share: f32) -> bool {
    let holed = hole.iter().filter(|v| **v >= 0.5).count();
    if holed == 0 {
        return false;
    }
    let (mut on, mut inside) = (0usize, 0usize);
    for (c, h) in cov.iter().zip(hole) {
        if *c >= 0.5 {
            on += 1;
            if *h >= 0.5 {
                inside += 1;
            }
        }
    }
    if on as f32 > WHOLE_FRAME * cov.len() as f32 {
        return false;
    }
    on > 0 && inside as f32 >= share * on as f32 && inside as f32 >= LEFT_MIN_SHARE * holed as f32
}

/// How many pixels two masks share.
fn overlap(a: &[f32], b: &[f32]) -> u64 {
    a.iter().zip(b).filter(|(x, y)| **x >= 0.5 && **y >= 0.5).count() as u64
}

/// Whether a detection is a subject seen again (its completed part included): it holds most of
/// the subject's mask.
fn continues(detection: &[f32], subject: &[f32], subject_pixels: u64) -> bool {
    subject_pixels > 0 && overlap(detection, subject) as f32 >= CONTINUES * subject_pixels as f32
}

/// Why a detection on a fill's result is the fill's own work (the subject painted back, or
/// someone new), or `None` when it was there before. It sits mostly in what the fill repainted
/// (a person: the hole widened by the grow; a thing: its own spot, as a saucer beside a cup
/// overlaps the grow and not the cup), and continues no subject behind by at most
/// [`COMPLETION_MAX`] times that subject's visible pixels (its hidden part completed). A second
/// copy of the woman in blue, painted beside her, covered the sliver of the man at the frame's
/// edge and passed as his completion (measured in the app 2026-11-04). A completion that still
/// `looks` like what was cut ([`KEPT_LOOKS`] of its pixels in the hole as they were) is the
/// subject painted back over whoever stands behind: the front-left man of the café photo came
/// back almost pixel for pixel and passed (measured in the app 2026-11-04). `cut` is the subject
/// whose hole this is (its kind, mask and pixels).
#[allow(clippy::too_many_arguments)]
pub(super) fn fills_own_work(
    term: &str,
    cov: &[f32],
    hole: &[f32],
    grown: &[f32],
    cut: (&str, &[f32], u64),
    behind: &[(String, Vec<f32>, u64)],
    looks: &dyn Fn(&[f32]) -> f32,
    w: usize,
) -> Option<String> {
    let (region, share) = if creature(term) { (grown, LEFT_IN_HOLE) } else { (hole, THING_IN_HOLE) };
    if !mostly_in_hole(cov, region, share) {
        return None;
    }
    let (cut_term, cut_cov, cut_pixels) = cut;
    let continued = behind.iter().filter(|(t, bc, bn)| t == term && continues(cov, bc, *bn)).collect::<Vec<_>>();
    if continued.iter().any(|(_, bc, bn)| added_in(cov, region, bc) <= *bn * COMPLETION_MAX) {
        let kept = looks(cov);
        return (kept >= KEPT_LOOKS).then(|| {
            if term == cut_term {
                format!("the {term} painted back over the one behind ({:.0} % of it as it was)", kept * 100.0)
            } else {
                format!("the {cut_term} still there under the {term} behind ({:.0} % of it as it was)", kept * 100.0)
            }
        });
    }
    if term == cut_term && overlap(cov, cut_cov) * 2 >= cut_pixels {
        return Some(format!("the {term} painted back"));
    }
    if !continued.is_empty() {
        return Some(format!("a {term} over the one behind, more than a completion"));
    }
    // A thing seen apart from the one behind it that it joins (the table's stretch across the
    // hole, found as a table of its own) is that thing's completion, not a new one: "a table in
    // the hole that was not there" refused five strip fills behind the people at the café table
    // (measured in the app 2026-11-04). A person is never excused this way: a second copy of
    // the woman painted beside her would pass as the man at the frame's edge.
    if !creature(term) && behind.iter().any(|(t, bc, _)| t == term && near(cov, bc, w, JOIN_REACH as usize)) {
        return None;
    }
    Some(format!("a {term} in the hole that was not there"))
}

/// Why a detection of `term` (a person or an animal) on a fill's result is someone the fill
/// painted: it stands mostly ([`NEW_INSIDE`]) where the fill was laid in (`touched`), and it is
/// neither the subject whose hole this is (painted back) nor someone behind seen again (a part of
/// them, or their completion by at most [`COMPLETION_MAX`] times their visible pixels). `None`
/// when it was there before. Small figures the frame-wide check never saw (ranked below the
/// people still there, or under its share of the hole): six invented diners and office people
/// on the café backgrounds, and one on the portrait's (measured in the app 2026-11-04).
pub(super) fn someone_new(
    term: &str,
    cov: &[f32],
    bounds: Rect,
    touched: &[f32],
    cut: (&str, &[f32], u64),
    behind: &[(String, Vec<f32>, u64)],
) -> Option<String> {
    let on = cov.iter().filter(|c| **c >= 0.5).count() as u64;
    let inside = overlap(cov, touched);
    if on == 0 || (inside as f32) < NEW_INSIDE * on as f32 {
        return None;
    }
    let (cut_term, cut_cov, cut_pixels) = cut;
    if term == cut_term && overlap(cov, cut_cov) * 2 >= cut_pixels {
        return Some(format!("the {term} painted back"));
    }
    if behind
        .iter()
        .any(|(t, bc, bn)| t == term && (overlap(cov, bc) * 2 >= on || (continues(cov, bc, *bn) && added_in(cov, touched, bc) <= *bn * COMPLETION_MAX)))
    {
        return None;
    }
    Some(format!("a {term} painted into the hole ({}×{} px)", bounds.width(), bounds.height()))
}

/// Why a detection on a fill's result is the fill's own work: [`fills_own_work`] (the subject
/// painted back, more than a completion, a thing on its spot), then, for a person or an animal,
/// [`someone_new`] wherever the fill painted at all. With `patched` (a thing's patch fill, which
/// copies the picture around the hole and so shows the detector the thing again by construction)
/// a thing's detection does not count.
#[allow(clippy::too_many_arguments)]
fn judge(
    term: &str,
    cov: &[f32],
    bounds: Rect,
    hole: &[f32],
    grown: &[f32],
    touched: &[f32],
    cut: (&str, &[f32], u64),
    behind: &[(String, Vec<f32>, u64)],
    looks: &dyn Fn(&[f32]) -> f32,
    w: usize,
    patched: bool,
) -> Option<String> {
    if (creature(term) || !patched)
        && let Some(why) = fills_own_work(term, cov, hole, grown, cut, behind, looks, w)
    {
        return Some(why);
    }
    if creature(term) {
        return someone_new(term, cov, bounds, touched, cut, behind);
    }
    None
}

/// What the check asks the detector for after a fill: the subject's kind (painted back, and
/// one more for someone new of its kind), the kinds still behind it (their completed parts),
/// and people in any case (anyone painted in). A person's or animal's kind is asked for
/// [`NEW_AT_MOST`] at least, down to [`new_floor`].
fn asks_for<'a>(cut_term: &str, behind: impl Iterator<Item = &'a Subject>) -> Vec<(String, usize)> {
    let mut asks: Vec<(String, usize)> = vec![(cut_term.to_string(), 2)];
    for b in behind {
        match asks.iter_mut().find(|(t, _)| t == b.check_term()) {
            Some((_, n)) => *n += 1,
            None => asks.push((b.check_term().to_string(), 2)),
        }
    }
    if !asks.iter().any(|(t, _)| t == PERSON) {
        asks.push((PERSON.to_string(), 1));
    }
    asks
}

/// The error when no fill cleared any subject, with every try's verdict: "could not clear the
/// person out of the background on 6 tries (removal kept what stood there (78 % of it as it
/// was); …), so nothing was changed".
fn could_not_clear(what: &str, tries: &[Value]) -> EngineError {
    let why: Vec<String> = tries.iter().map(|t| format!("{} {}", t["fill"].as_str().unwrap_or("fill"), t["verdict"].as_str().unwrap_or("refused"))).collect();
    EngineError::Other(format!(
        "Split into Layers could not clear the {what} out of the background on {} tries ({}), so nothing was changed",
        tries.len(),
        why.join("; ")
    ))
}

/// The smallest detection (pixels) that counts as someone painted in, on a `w`×`h` picture.
fn new_floor(w: usize, h: usize) -> u64 {
    (((w * h) as f64 * NEW_MIN_SHARE).ceil() as u64).max(NEW_MIN_PIXELS)
}

/// The instances of `term` the check looks at on a fill's result: every one down to the floor
/// for a person or an animal, the largest `at_most` for a thing.
fn seen_of(finder: &SubjectFinder, filled: &[u8], area: Rect, term: &str, at_most: usize, ctx: &JobCtx) -> Result<Vec<Instance>> {
    let (wu, hu) = (area.width() as usize, area.height() as usize);
    if creature(term) {
        finder.find_all_in(filled, area, term, at_most.max(NEW_AT_MOST), new_floor(wu, hu), ctx)
    } else {
        finder.find_in(filled, area, term, at_most, ctx)
    }
}

/// Whether any pixel of `a` lies within `reach` pixels of one of `b` (both over rows of `w`).
fn near(a: &[f32], b: &[f32], w: usize, reach: usize) -> bool {
    if w == 0 {
        return false;
    }
    let h = a.len() / w;
    let r = reach as i64;
    a.iter().enumerate().filter(|(_, v)| **v >= 0.5).any(|(k, _)| {
        let (x, y) = ((k % w) as i64, (k / w) as i64);
        (-r..=r).any(|dy| {
            (-r..=r).any(|dx| {
                let (nx, ny) = (x + dx, y + dy);
                nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h && b[ny as usize * w + nx as usize] >= 0.5
            })
        })
    })
}

/// People the detector still finds on the finished background, apart from the subjects left in
/// it (`left`, their holes): their bounds, `leftBehind` in the result, so a sheet says at once
/// whether anyone was invented or missed (the app's own detector found one on every round-6
/// café background the per-fill checks had passed; measured 2026-11-04).
fn left_behind(finder: &SubjectFinder, background: &[u8], area: Rect, left: &[Vec<f32>], ctx: &JobCtx) -> Result<Vec<Value>> {
    let (wu, hu) = (area.width() as usize, area.height() as usize);
    let mut out = Vec::new();
    for (cov, b, n) in finder.find_all_in(background, area, PERSON, NEW_AT_MOST, new_floor(wu, hu), ctx)? {
        if left.iter().any(|hole| overlap(&cov, hole) * 2 >= n) {
            continue;
        }
        out.push(json!([b.x0, b.y0, b.width(), b.height()]));
    }
    Ok(out)
}

/// Whether a term names a person or an animal (judged over the widened hole and by their
/// pixels) rather than a thing (judged on its own spot).
fn creature(term: &str) -> bool {
    matches!(
        bare(term).to_ascii_lowercase().as_str(),
        PERSON | ANIMAL | "people" | "man" | "woman" | "child" | "boy" | "girl" | "baby" | "face" | "dog" | "cat" | "horse" | "bird" | "cow" | "sheep"
    )
}

/// Where both masks are on (at 0.5 and over).
fn both(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| if *x >= 0.5 && *y >= 0.5 { 1.0 } else { 0.0 }).collect()
}

/// How much of what stood in `region` a fill left as it was: the share of the region's blocks
/// ([`KEPT_BLOCK`] px square, at least half of each in the region) whose mean colour `after`
/// keeps within [`KEPT_TOLERANCE`] of `before`. Blocks, not pixels, so a fill that redraws the
/// subject a little moved or softened still counts as keeping it.
pub(super) fn kept_share(before: &[u8], after: &[u8], region: &[f32], w: usize, h: usize) -> f32 {
    let (mut blocks, mut kept) = (0u32, 0u32);
    for by in (0..h).step_by(KEPT_BLOCK) {
        for bx in (0..w).step_by(KEPT_BLOCK) {
            let (mut n, mut sum_before, mut sum_after) = (0u32, [0u64; 3], [0u64; 3]);
            for y in by..(by + KEPT_BLOCK).min(h) {
                for x in bx..(bx + KEPT_BLOCK).min(w) {
                    let k = y * w + x;
                    if region.get(k).is_none_or(|r| *r < 0.5) {
                        continue;
                    }
                    n += 1;
                    for (c, (sb, sa)) in sum_before.iter_mut().zip(sum_after.iter_mut()).enumerate() {
                        *sb += u64::from(before[k * 4 + c]);
                        *sa += u64::from(after[k * 4 + c]);
                    }
                }
            }
            if (n as usize) * 2 < KEPT_BLOCK * KEPT_BLOCK {
                continue;
            }
            blocks += 1;
            let moved = sum_before.iter().zip(&sum_after).map(|(b, a)| (*b as f32 - *a as f32).abs() / n as f32).sum::<f32>() / 3.0;
            if moved <= KEPT_TOLERANCE {
                kept += 1;
            }
        }
    }
    if blocks == 0 { 0.0 } else { kept as f32 / blocks as f32 }
}

/// The pixels of `detection` inside `region` that `subject` did not hold: what a fill added.
fn added_in(detection: &[f32], region: &[f32], subject: &[f32]) -> u64 {
    detection.iter().zip(region).zip(subject).filter(|((d, r), s)| **d >= 0.5 && **r >= 0.5 && **s < 0.5).count() as u64
}

/// The pixels of `wanted` joined (4-neighbours) to a pixel within [`JOIN_REACH`] of `seed` (at
/// 0.5 and over): the part of a detection that continues the subject, not a patch of its own.
pub(super) fn joined_to(wanted: &[bool], seed: &[f32], w: usize, h: usize) -> Vec<bool> {
    let mut out = vec![false; wanted.len()];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (k, want) in wanted.iter().enumerate() {
        if !want {
            continue;
        }
        let (x, y) = ((k % w) as i64, (k / w) as i64);
        let near = (-JOIN_REACH..=JOIN_REACH).any(|dy| {
            (-JOIN_REACH..=JOIN_REACH).any(|dx| {
                let (nx, ny) = (x + dx, y + dy);
                nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h && seed[ny as usize * w + nx as usize] >= 0.5
            })
        });
        if near {
            out[k] = true;
            queue.push_back(k);
        }
    }
    while let Some(k) = queue.pop_front() {
        let (x, y) = (k % w, k / w);
        let mut step = |n: usize| {
            if wanted[n] && !out[n] {
                out[n] = true;
                queue.push_back(n);
            }
        };
        if x > 0 {
            step(k - 1);
        }
        if x + 1 < w {
            step(k + 1);
        }
        if y > 0 {
            step(k - w);
        }
        if y + 1 < h {
            step(k + w);
        }
    }
    out
}

/// What a fill left: the picture with the hole filled, the fill's seed, and what the check saw
/// in it.
type Filled<T> = (Vec<u8>, u64, T);

/// A check's verdict on a fill: what it saw, and why the fill is rejected, if it is.
type Verdict<T> = (T, Option<String>);

/// A check on a fill: given the filled picture, where the fill was laid at all (one value per
/// canvas pixel) and whether this try was the classical patch, it gives its [`Verdict`].
type Check<'a, T> = dyn FnMut(&[u8], &[f32], bool) -> Result<Verdict<T>> + 'a;

/// What a hole's tries came to: the fill accepted (`None` when every try was refused), every
/// try's record, the time all the tries took and the model generations they asked for.
struct HoleFill<T> {
    outcome: Option<Filled<T>>,
    tries: Vec<Value>,
    ms: u64,
    generations: u32,
}

/// Fill `hole` (one value per canvas pixel) in `picture`, the document's composite standing in
/// for it through a work layer. The removal first (quick, and clean on most pictures), then,
/// after it paints a subject back, the grey-box fill of the hole's bounds (in strips for a big
/// hole, see [`step_of`]) laid only over the hole (widened and feathered) and tone matched, and
/// the classical patch fill as the last try. A result that kept `watch` (a silhouette) as it
/// was, [`KEPT_WHOLE`] of it by its pixels, or left the hole blank or a shape painted over it
/// ([`blank_or_ghost`]), is rejected before the check sees it; then `check` judges it (the
/// detector looking for the subject painted back or someone new, over the picture and over
/// where the fill was laid at all, `touched`) and keeps what it saw; a `thing`'s patch fill,
/// which copies the picture around the hole and so shows the detector the thing again by
/// construction, is judged by the pixels and for people only. The first accepted comes back with
/// its seed, `None` after [`FILL_TRIES`] rejected tries and the patch; with it every try's record
/// (`fill`: removal, fill, strips or patch; `generations`: the model fills it asked for;
/// `verdict`: accepted, or why not), the time and the generations of them all.
#[allow(clippy::too_many_arguments)]
fn fill_hole<T>(
    doc: &mut Document,
    active: &mut Option<LayerId>,
    below: Option<LayerId>,
    ctx: &JobCtx,
    area: Rect,
    picture: &[u8],
    hole: &[f32],
    protect: Option<&[f32]>,
    watch: Option<&[f32]>,
    thing: bool,
    removal: &FillJob,
    box_job: Option<&FillJob>,
    (lo, hi): (f32, f32),
    check: &mut Check<'_, T>,
) -> Result<HoleFill<T>> {
    let (w, h) = (area.width(), area.height());
    let (wu, hu) = (w as usize, h as usize);
    let (hole_rect, holed) = crate::select_ml_cmds::coverage_bounds(hole, area);
    let mut done = HoleFill { outcome: None, tries: Vec::with_capacity(FILL_TRIES as usize + 1), ms: 0, generations: 0 };
    if holed == 0 {
        return Ok(done);
    }
    let grow = remove_grow(hole_rect).min(MAX_GROW);
    // A big hole (over [`BIG_HOLE_SIDE`] a side) gets the grey box in strips: one fill over a
    // third of the frame came back as the grey box it was given, on 6 of 8 café backgrounds
    // (measured in the app 2026-11-04).
    let big = hole_rect.width().max(hole_rect.height()) > BIG_HOLE_SIDE;
    // The ring just outside the hole: what a fill is compared with for texture and brightness.
    let ring: Vec<f32> = dilate(hole, wu, hu, RING_WIDTH).iter().zip(hole).map(|(d, h)| if *d >= 0.5 && *h < 0.5 { 1.0 } else { 0.0 }).collect();
    // A plain ring (a backdrop): the removal is the fill that works there, and strips have
    // nothing to continue (images.jpg's terracotta came back as two strips, grey over white,
    // with a hard seam; measured in the app 2026-11-04).
    let plain = texture_and_luma(picture, &ring, wu, hu, texture_step(wu, hu)).is_none_or(|(t, _)| t < PLAIN_TEXTURE);
    let rect = fill_rect(hole_rect.inflate(grow as i32).intersect(&area), removal.tuning.remove_margin, area)?;
    let rect = at_least(rect, REMOVE_MIN_REQUEST, area);
    // The box the background fill repaints (the hole's bounds widened by the removal's grow)
    // and how its result is laid over the picture: only over the hole, widened and feathered,
    // so the picture outside keeps its own pixels.
    let boxed = hole_rect.inflate(grow as i32).intersect(&area);
    let box_hole: Vec<f32> = (0..wu * hu)
        .map(|k| {
            let (x, y) = (area.x0 + (k % wu) as i32, area.y0 + (k / wu) as i32);
            if x >= boxed.x0 && x < boxed.x1 && y >= boxed.y0 && y < boxed.y1 { 1.0 } else { 0.0 }
        })
        .collect();
    // How a result is laid back into the picture: over the hole widened by the removal's grow
    // and feathered, so the subject's halo and shadow go with it; but only a tight band where
    // another subject still stands (`protect`), so its pixels are its own when its turn comes
    // (a nearer subject's fill repainted its neighbours into grey silhouettes, measured in the
    // app 2026-11-03).
    let feather = (grow as usize / 2).max(1);
    let wide = dilate(hole, wu, hu, (grow as usize).max(1));
    let wide = box_blur(&box_blur(&wide, wu, hu, feather, true), wu, hu, feather, false);
    let over: Vec<f32> = match protect {
        Some(keep) => {
            let tight = dilate(hole, wu, hu, TIGHT_BAND);
            let tight = box_blur(&box_blur(&tight, wu, hu, 1, true), wu, hu, 1, false);
            let keep = dilate(keep, wu, hu, PROTECT_GROW);
            wide.iter().zip(&tight).zip(&keep).zip(hole).map(|(((wd, t), k), hl)| if *k >= 0.5 { t.max(*hl) } else { *wd }).collect()
        }
        None => wide,
    };
    // Wherever a result is laid in at all (the feather included): what the check looks over for
    // anyone the fill painted.
    let touched: Vec<f32> = over.iter().map(|a| if *a > TOUCHED { 1.0 } else { 0.0 }).collect();
    // The picture as it stands, as a layer the fill sees as the composite.
    let mut work = Layer::raster("Split work".to_string(), doc.pixel_format());
    write_rgba8(&mut work, &Rgba8::new(w, h, picture.to_vec()).map_err(gen_err)?, area)?;
    let work = doc.insert_above(below, work);
    // Every try, then the classical patch fill as the last: a step, its result (the picture
    // with the fill laid in), the seed it ran, its time and its generations.
    for attempt in 0..=FILL_TRIES {
        let span = hi - lo;
        ctx.progress(lo + span * 0.8 * attempt as f32 / (FILL_TRIES + 1) as f32, "");
        let step = if attempt == FILL_TRIES { Step::Patch } else { step_of(attempt, big, plain, box_job.is_some()) };
        let kind = step.name();
        let progress = (lo + span * 0.8 * attempt as f32 / (FILL_TRIES + 1) as f32, lo + span * 0.8 * (attempt as f32 + 0.9) / (FILL_TRIES + 1) as f32);
        let result = match step {
            Step::Patch => Ok(patch_fill(picture, hole, protect, wu, hu, removal.seed).map(|f| (f, removal.seed, 0u64, 0u32))),
            Step::Strips => {
                let mut try_job = box_job.cloned().unwrap_or_else(|| removal.clone());
                try_job.seed = try_job.seed.wrapping_add(u64::from(attempt) * 1000);
                fill_in_strips(doc, active, work, ctx, area, picture, &box_hole, boxed, grow, &try_job, progress).map(Some)
            }
            Step::Removal | Step::Box => {
                let try_box = step == Step::Box;
                let mut try_job = if try_box { box_job.cloned().unwrap_or_else(|| removal.clone()) } else { removal.clone() };
                try_job.seed = try_job.seed.wrapping_add(u64::from(attempt) * 1000);
                if !try_box {
                    try_job.grow = grow;
                }
                let coverage = embed_coverage(area, if try_box { &box_hole } else { hole }, rect);
                *active = Some(work);
                one_fill(doc, active, ctx, area, try_job, rect, coverage).map(Some)
            }
        };
        let (filled, fill_seed, took, made) = match result {
            Ok(Some(r)) => r,
            Ok(None) => {
                done.tries.push(json!({"fill": kind, "generations": 0, "verdict": "nothing to fill from"}));
                continue;
            }
            Err(e) => {
                doc.remove(work);
                *active = below;
                return Err(e);
            }
        };
        done.ms += took;
        done.generations += made;
        let record = |verdict: &str| json!({"fill": kind, "generations": made, "verdict": verdict});
        // The result over the picture along `over`. A box fill's tone is matched to the picture
        // where both show the same thing (in the box, outside the hole): a fill a few levels
        // darker left the person's outline as a shadow on the curtain (measured in the app
        // 2026-11-03). The removal's result is tone matched by its own route; the patch fill
        // copies the picture's own pixels.
        let tone = matches!(step, Step::Box | Step::Strips).then(|| ToneGrid::new(picture, &filled, |k| box_hole[k] >= 0.5 && over[k] < 0.02, wu, hu));
        let background: Vec<u8> = picture
            .as_chunks::<4>()
            .0
            .iter()
            .zip(filled.as_chunks::<4>().0)
            .zip(&over)
            .enumerate()
            .flat_map(|(k, ((p, f), a))| {
                let a = a.clamp(0.0, 1.0);
                let t = match &tone {
                    Some(tone) if a > 0.0 => tone.at(k % wu, k / wu),
                    _ => [0.0; 3],
                };
                std::array::from_fn::<u8, 4, _>(|c| {
                    let f = if c < 3 { (f32::from(f[c]) + t[c]).clamp(0.0, 255.0) } else { f32::from(f[c]) };
                    (f32::from(p[c]) * (1.0 - a) + f * a).round() as u8
                })
            })
            .collect();
        ctx.progress(progress.1, "Checking the fill");
        if let Some(silhouette) = watch {
            let kept = kept_share(picture, &background, silhouette, wu, hu);
            if kept >= KEPT_WHOLE {
                done.tries.push(record(&format!("kept what stood there ({:.0} % of it as it was)", kept * 100.0)));
                continue;
            }
        }
        // A blank fill (the grey box as it was) or a smear where the subject stood passes a
        // detector and the pixels alike: judged against the picture around the hole.
        if let Some(why) = blank_or_ghost(picture, &background, hole, &ring, wu, hu) {
            done.tries.push(record(&why));
            continue;
        }
        match check(&background, &touched, step == Step::Patch && thing) {
            Ok((seen, None)) => {
                done.tries.push(record("accepted"));
                done.outcome = Some((background, fill_seed, seen));
                break;
            }
            Ok((_, Some(why))) => {
                done.tries.push(record(&why));
                continue;
            }
            Err(e) => {
                doc.remove(work);
                *active = below;
                return Err(e);
            }
        }
    }
    doc.remove(work);
    *active = below;
    Ok(done)
}

/// One try of a hole's fill.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    /// The named removal on the hole's own shape.
    Removal,
    /// The grey-box fill of the hole's bounds.
    Box,
    /// The grey box in strips along its longer side ([`fill_in_strips`]).
    Strips,
    /// The classical content-aware fill from the hole's surroundings ([`patch_fill`]).
    Patch,
}

impl Step {
    fn name(self) -> &'static str {
        match self {
            Step::Removal => "removal",
            Step::Box => "fill",
            Step::Strips => "strips",
            Step::Patch => "patch",
        }
    }
}

/// Which fill try `attempt` of a hole is. A small hole: the removal, then the grey-box fills. A
/// big one (`big`): the box in strips three times, then the box whole; a big person's removal
/// is skipped (it painted every big person of the café photo back, eight of eight, measured in
/// the app 2026-11-04), a big thing's (a table) comes first, as it often clears. Without a box
/// fill to call on, the removal every time.
fn step_of(attempt: u32, big: bool, plain: bool, boxed: bool) -> Step {
    if !boxed {
        return Step::Removal;
    }
    // The removal first every time: quick (a second or two against six to eight a strip), and
    // the fill that works on a plain backdrop; a big person skipping it lost images.jpg's
    // backdrop to strips (measured in the app 2026-11-04). On a plain ring it gets a second
    // seed before the box.
    if plain {
        return if attempt <= 1 { Step::Removal } else { Step::Box };
    }
    if !big {
        return if attempt == 0 { Step::Removal } else { Step::Box };
    }
    match attempt {
        0 => Step::Removal,
        1 | 2 | 4 => Step::Strips,
        _ => Step::Box,
    }
}

/// One fill of `job` over `coverage` in `rect`, run above the active layer: the composite with
/// its result, the seed it ran, its time and the generations it made. Its layers are taken out
/// again.
fn one_fill(
    doc: &mut Document,
    active: &mut Option<LayerId>,
    ctx: &JobCtx,
    area: Rect,
    job: FillJob,
    rect: Rect,
    coverage: Vec<f32>,
) -> Result<(Vec<u8>, u64, u64, u32)> {
    let (made, ..) = fill_region(doc, active, ctx, job, rect, coverage, None)?;
    let checked = ctx.check();
    let filled = photocraft_compose::render(doc, area).to_rgba8().pixels;
    let seed = made.first().map_or(0, |m| m.1);
    let ms = made.iter().map(|m| m.3).sum::<u64>();
    let generations = made.len() as u32;
    for (id, ..) in &made {
        doc.remove(*id);
    }
    checked?;
    Ok((filled, seed, ms, generations))
}

/// The grey box (`box_hole`, bounds `boxed`) filled in strips along its longer side, each
/// strip seeing the picture with the box grey and the strips before it filled (and overlapping
/// the last by [`STRIP_OVERLAP`] px, where the two are blended): a fill asked for a third of the
/// frame at once handed the grey box back, where the cups and cards came out right (measured in
/// the app 2026-11-04). `work` is the layer the fills see as the picture; it holds `picture`
/// again at the end. The picture with the box filled, the last strip's seed, the fills' time
/// and their count.
#[allow(clippy::too_many_arguments)]
fn fill_in_strips(
    doc: &mut Document,
    active: &mut Option<LayerId>,
    work: LayerId,
    ctx: &JobCtx,
    area: Rect,
    picture: &[u8],
    box_hole: &[f32],
    boxed: Rect,
    grow: u32,
    job: &FillJob,
    (lo, hi): (f32, f32),
) -> Result<(Vec<u8>, u64, u64, u32)> {
    let (w, h) = (area.width(), area.height());
    let wu = w as usize;
    let tall = boxed.height() >= boxed.width();
    let long = if tall { boxed.height() } else { boxed.width() } as usize;
    let n = long.div_ceil(STRIP_LEN).clamp(2, MAX_STRIPS);
    let step = long.div_ceil(n);
    // Where the box starts along its long side, in the picture's rows or columns.
    let origin = if tall { boxed.y0 - area.y0 } else { boxed.x0 - area.x0 } as usize;
    let mut wp = picture.to_vec();
    for (k, c) in box_hole.iter().enumerate() {
        if *c >= 0.5 {
            wp[k * 4..k * 4 + 3].copy_from_slice(&PREFILL[..3]);
        }
    }
    let show = |doc: &mut Document, px: &[u8]| -> Result<()> {
        let layer = doc.layer_mut(work).ok_or(EngineError::NoLayer(work))?;
        write_rgba8(layer, &Rgba8::new(w, h, px.to_vec()).map_err(gen_err)?, area)
    };
    let (mut seed, mut ms, mut generations) = (0u64, 0u64, 0u32);
    for i in 0..n {
        ctx.check().inspect_err(|_| {
            let _ = show(doc, picture);
        })?;
        ctx.progress(lo + (hi - lo) * i as f32 / n as f32, &format!("Filling strip {} of {n}", i + 1));
        let a = (i * step).saturating_sub(if i > 0 { STRIP_OVERLAP } else { 0 });
        let b = ((i + 1) * step).min(long);
        let strip = if tall {
            Rect::new(boxed.x0, boxed.y0 + a as i32, boxed.x1, boxed.y0 + b as i32)
        } else {
            Rect::new(boxed.x0 + a as i32, boxed.y0, boxed.x0 + b as i32, boxed.y1)
        };
        let mask: Vec<f32> = box_hole
            .iter()
            .enumerate()
            .map(|(k, c)| {
                let (x, y) = (area.x0 + (k % wu) as i32, area.y0 + (k / wu) as i32);
                if *c >= 0.5 && x >= strip.x0 && x < strip.x1 && y >= strip.y0 && y < strip.y1 { 1.0 } else { 0.0 }
            })
            .collect();
        let rect = fill_rect(strip.inflate(grow as i32).intersect(&area), job.tuning.remove_margin, area)?;
        let rect = at_least(rect, REMOVE_MIN_REQUEST, area);
        let coverage = embed_coverage(area, &mask, rect);
        show(doc, &wp)?;
        *active = Some(work);
        let mut j = job.clone();
        j.seed = j.seed.wrapping_add(i as u64 * 7);
        let (filled, s, t, g) = match one_fill(doc, active, ctx, area, j, rect, coverage) {
            Ok(r) => r,
            Err(e) => {
                let _ = show(doc, picture);
                return Err(e);
            }
        };
        seed = s;
        ms += t;
        generations += g;
        // The strip into the box; over the overlap with the strip before it, blended from that
        // one into this (a hard step at every strip's edge, measured in the app 2026-11-04).
        for (k, m) in mask.iter().enumerate() {
            if *m < 0.5 {
                continue;
            }
            let along = (if tall { k / wu } else { k % wu }).saturating_sub(origin);
            let t = if i > 0 && along < a + STRIP_OVERLAP { (along.saturating_sub(a) as f32 + 0.5) / STRIP_OVERLAP as f32 } else { 1.0 };
            for c in 0..4 {
                wp[k * 4 + c] = (f32::from(wp[k * 4 + c]) * (1.0 - t) + f32::from(filled[k * 4 + c]) * t).round() as u8;
            }
        }
    }
    show(doc, picture)?;
    Ok((wp, seed, ms, generations))
}

/// The hole filled from its surroundings by the classical content-aware fill (patches copied
/// from around it, never from the hole or from where another subject stands, `protect`), at a
/// scale that keeps the work under [`PATCH_PIXELS`]: the last try when every fill was refused,
/// so a subject whose hole no model would paint is still out of the background. `None` when
/// nothing around the hole can serve as a source.
fn patch_fill(picture: &[u8], hole: &[f32], protect: Option<&[f32]>, w: usize, h: usize, seed: u64) -> Option<Vec<u8>> {
    use photocraft_algo::content_aware::{FillOptions, fill};
    let area = Rect::new(0, 0, w as i32, h as i32);
    let (bounds, holed) = crate::select_ml_cmds::coverage_bounds(hole, area);
    if holed == 0 {
        return None;
    }
    let margin = (bounds.width().max(bounds.height()) / 2).max(64);
    let crop = bounds.inflate(margin as i32).intersect(&area);
    let (cw, ch) = (crop.width() as usize, crop.height() as usize);
    if cw == 0 || ch == 0 {
        return None;
    }
    let k = (((cw * ch) as f64 / PATCH_PIXELS as f64).sqrt().ceil() as usize).max(1);
    let (sw, sh) = (cw.div_ceil(k), ch.div_ceil(k));
    let mut img = vec![0.0f32; sw * sh * 3];
    let mut count = vec![0u32; sw * sh];
    let mut in_hole = vec![false; sw * sh];
    let mut kept_out = vec![false; sw * sh];
    for y in 0..ch {
        for x in 0..cw {
            let px = (crop.y0 as usize + y) * w + crop.x0 as usize + x;
            let i = (y / k) * sw + x / k;
            for c in 0..3 {
                img[i * 3 + c] += f32::from(picture[px * 4 + c]) / 255.0;
            }
            count[i] += 1;
            in_hole[i] |= hole[px] >= 0.5;
            kept_out[i] |= protect.is_some_and(|p| p[px] >= 0.5);
        }
    }
    for i in 0..sw * sh {
        let n = count[i].max(1) as f32;
        for c in 0..3 {
            img[i * 3 + c] /= n;
        }
    }
    let source: Vec<bool> = (0..sw * sh).map(|i| !in_hole[i] && !kept_out[i] && count[i] > 0).collect();
    if !source.iter().any(|s| *s) {
        return None;
    }
    let filled = fill(sw, sh, 3, &img, &in_hole, &source, &FillOptions { seed, ..FillOptions::default() });
    let mut out = picture.to_vec();
    for y in 0..ch {
        for x in 0..cw {
            let px = (crop.y0 as usize + y) * w + crop.x0 as usize + x;
            if hole[px] < 0.5 {
                continue;
            }
            let fx = ((x as f32 + 0.5) / k as f32 - 0.5).clamp(0.0, (sw - 1) as f32);
            let fy = ((y as f32 + 0.5) / k as f32 - 0.5).clamp(0.0, (sh - 1) as f32);
            let (x0, y0) = (fx as usize, fy as usize);
            let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
            let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
            for c in 0..3 {
                let at = |xx: usize, yy: usize| filled[(yy * sw + xx) * 3 + c];
                let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
                let bottom = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
                out[px * 4 + c] = ((top * (1.0 - ty) + bottom * ty).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
    Some(out)
}

/// Why a fill is no fill, judged against the picture in the `ring` around the hole: it left the
/// hole blank where the picture has texture (the grey box handed back nearly as it was, which a
/// detector and the pixel check both pass; 6 of 8 café backgrounds, measured in the app
/// 2026-11-04), or painted a dark or light shape over it (a silhouette-shaped smear where
/// someone stood). The hole as a whole first, then by cells of [`BLANK_CELL`] px, each against
/// the ring around it: a pale box over a third of a hole with fog over the rest, and a flat
/// white quarter, passed the whole hole's averages (measured in the app 2026-11-04). `None` on a
/// plain backdrop, whose fill is rightly plain (though not another brightness), and when the fill
/// has texture and brightness like its surroundings.
pub(super) fn blank_or_ghost(before: &[u8], after: &[u8], hole: &[f32], ring: &[f32], w: usize, h: usize) -> Option<String> {
    let step = texture_step(w, h);
    let (ring_tex, ring_luma) = texture_and_luma(before, ring, w, h, step)?;
    let (hole_tex, hole_luma) = texture_and_luma(after, hole, w, h, step)?;
    let d = hole_luma - ring_luma;
    if ring_tex < PLAIN_TEXTURE {
        return (d.abs() >= GHOST_LUMA).then(|| ghost_words(d, 1.0));
    }
    if hole_tex < BLANK_SHARE * ring_tex {
        return Some(format!("left the hole blank (texture {hole_tex:.3} against {ring_tex:.3} around it)"));
    }
    if d.abs() >= GHOST_LUMA {
        return Some(ghost_words(d, 1.0));
    }
    // By cells over the hole's bounds (the ring's width around them): a cell of the hole against
    // the ring in the cells about it, or the whole ring where none is near.
    let area = Rect::new(0, 0, w as i32, h as i32);
    let (bounds, holed) = crate::select_ml_cmds::coverage_bounds(hole, area);
    let grid = bounds.inflate(RING_WIDTH as i32).intersect(&area);
    let (gw, gh) = ((grid.width() as usize).div_ceil(BLANK_CELL).max(1), (grid.height() as usize).div_ceil(BLANK_CELL).max(1));
    let hole_cells = cell_stats(after, hole, w, h, grid, (gw, gh), step);
    let ring_cells = cell_stats(before, ring, w, h, grid, (gw, gh), step);
    let (mut blank, mut ghost, mut shade) = (0u64, 0u64, 0.0f32);
    for cy in 0..gh {
        for cx in 0..gw {
            let cell = &hole_cells[cy * gw + cx];
            let Some((tex, luma)) = cell.values() else { continue };
            let mut around = CellStats::default();
            for y in cy.saturating_sub(1)..(cy + 2).min(gh) {
                for x in cx.saturating_sub(1)..(cx + 2).min(gw) {
                    around.add(&ring_cells[y * gw + x]);
                }
            }
            let (r_tex, r_luma) = around.values().unwrap_or((ring_tex, ring_luma));
            if r_tex >= PLAIN_TEXTURE && tex < BLANK_SHARE * r_tex {
                blank += u64::from(cell.ln);
            } else if (luma - r_luma).abs() >= GHOST_LUMA {
                ghost += u64::from(cell.ln);
                shade += (luma - r_luma) * cell.ln as f32;
            }
        }
    }
    if blank as f32 >= BLANK_CELLS * holed as f32 {
        return Some(format!("left {:.0} % of the hole blank (texture {hole_tex:.3} against {ring_tex:.3} around it)", 100.0 * blank as f32 / holed as f32));
    }
    if ghost > 0 && ghost as f32 >= BLANK_CELLS * holed as f32 {
        return Some(ghost_words(shade / ghost as f32, ghost as f32 / holed as f32));
    }
    None
}

/// "painted a dark shape over the hole (-40 % in brightness)", or over `share` of it.
fn ghost_words(d: f32, share: f32) -> String {
    let shape = if d < 0.0 { "dark" } else { "light" };
    if share >= 0.999 {
        format!("painted a {shape} shape over the hole ({:+.0} % in brightness)", d * 100.0)
    } else {
        format!("painted a {shape} shape over {:.0} % of the hole ({:+.0} % in brightness)", share * 100.0, d * 100.0)
    }
}

/// The texture and luminance sums of a region's pixels (in one cell, or in all).
#[derive(Clone, Copy, Default)]
struct CellStats {
    tex: f32,
    tn: u32,
    lum: f32,
    ln: u32,
}

impl CellStats {
    fn add(&mut self, o: &CellStats) {
        self.tex += o.tex;
        self.tn += o.tn;
        self.lum += o.lum;
        self.ln += o.ln;
    }

    /// The mean texture (luminance step between neighbours) and luminance, 0..1; `None` with too
    /// few pixels to judge.
    fn values(&self) -> Option<(f32, f32)> {
        (self.ln >= JUDGE_MIN_PIXELS && self.tn >= JUDGE_MIN_PIXELS).then(|| (self.tex / self.tn as f32, self.lum / self.ln as f32))
    }
}

/// Per cell of `grid` ([`BLANK_CELL`] px square, `(gw, gh)` of them), the sums over `region`'s
/// pixels in `img`: luminance, and the luminance step to the neighbour `step` px right and down
/// where that is in the region too.
fn cell_stats(img: &[u8], region: &[f32], w: usize, h: usize, grid: Rect, (gw, gh): (usize, usize), step: usize) -> Vec<CellStats> {
    let luma = |k: usize| (0.299 * f32::from(img[k * 4]) + 0.587 * f32::from(img[k * 4 + 1]) + 0.114 * f32::from(img[k * 4 + 2])) / 255.0;
    let mut cells = vec![CellStats::default(); gw * gh];
    let (gx0, gy0) = (grid.x0.max(0) as usize, grid.y0.max(0) as usize);
    for y in gy0..(grid.y1.max(0) as usize).min(h) {
        for x in gx0..(grid.x1.max(0) as usize).min(w) {
            let k = y * w + x;
            if region[k] < 0.5 {
                continue;
            }
            let (cx, cy) = (((x - gx0) / BLANK_CELL).min(gw - 1), ((y - gy0) / BLANK_CELL).min(gh - 1));
            let cell = &mut cells[cy * gw + cx];
            let l = luma(k);
            cell.lum += l;
            cell.ln += 1;
            if x + step < w && region[k + step] >= 0.5 {
                cell.tex += (l - luma(k + step)).abs();
                cell.tn += 1;
            }
            if y + step < h && region[k + step * w] >= 0.5 {
                cell.tex += (l - luma(k + step * w)).abs();
                cell.tn += 1;
            }
        }
    }
    cells
}

/// The mean luminance step between pixels `step` apart in `region` (its texture) and its mean
/// luminance, 0..1; `None` for a region too small to judge.
fn texture_and_luma(img: &[u8], region: &[f32], w: usize, h: usize, step: usize) -> Option<(f32, f32)> {
    let area = Rect::new(0, 0, w as i32, h as i32);
    let (bounds, _) = crate::select_ml_cmds::coverage_bounds(region, area);
    if bounds.is_empty() {
        return None;
    }
    let (gw, gh) = ((bounds.width() as usize).div_ceil(BLANK_CELL).max(1), (bounds.height() as usize).div_ceil(BLANK_CELL).max(1));
    let mut all = CellStats::default();
    for c in cell_stats(img, region, w, h, bounds, (gw, gh), step) {
        all.add(&c);
    }
    all.values()
}

/// How far apart the pixels compared for texture are: 1 up to [`TEXTURE_PIXELS`] a frame, more
/// on a larger picture, so a 12-megapixel ring's grain is not held against a fill generated at
/// a megapixel and scaled up.
fn texture_step(w: usize, h: usize) -> usize {
    (((w * h) as f32 / TEXTURE_PIXELS as f32).sqrt().ceil() as usize).max(1)
}

/// The subjects of the picture, found term by term, and what found them (the terms joined).
/// With `terms`: each in turn. Without: the people, then the animals, then the foreground
/// things the vision-language model names (one of each, then more of each), then Select
/// Subject's prompts in turn until one finds anything. Each term gets what is left of the budget once the finds so far are cleaned
/// ([`clean`]): the budget counts layers, not detections (seven people came back as eleven, and
/// the table behind them was never asked for; measured in the app 2026-11-03).
#[allow(clippy::too_many_arguments)]
fn find_subjects(
    finder: &SubjectFinder,
    thinker: &dyn GenerativeBackend,
    picture: &[u8],
    area: Rect,
    terms: Option<&[Term]>,
    budget: usize,
    seed: u64,
    group: Group,
    ctx: &JobCtx,
) -> Result<(Vec<Subject>, String)> {
    let (wu, hu) = (area.width() as usize, area.height() as usize);
    let mut out: Vec<Subject> = Vec::new();
    let mut by: Vec<String> = Vec::new();
    let take = |term: &str, asked: bool, at_most: usize, out: &mut Vec<Subject>, by: &mut Vec<String>, ctx: &JobCtx| -> Result<()> {
        if at_most == 0 {
            return Ok(());
        }
        let found = finder.find_in(picture, area, term, at_most, ctx)?;
        if !found.is_empty() && !by.iter().any(|b| b == term) {
            by.push(term.to_string());
        }
        for (cov, bounds, pixels) in found {
            out.push(Subject { cov, bounds, pixels, term: term.to_string(), asked, alpha: Vec::new(), completed: false, grown: false });
        }
        clean(out, wu, hu, group);
        Ok(())
    };
    match terms {
        Some(terms) => {
            for t in terms {
                ctx.check()?;
                let left = budget.saturating_sub(out.len());
                take(&t.name, true, t.max.map_or(left, |m| m.min(left)), &mut out, &mut by, ctx)?;
            }
            if out.is_empty() {
                let shown = terms.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", ");
                return Err(EngineError::Other(format!("nothing matching \"{shown}\" was found")));
            }
        }
        None => {
            take(PERSON, false, budget, &mut out, &mut by, ctx)?;
            ctx.check()?;
            let left = budget.saturating_sub(out.len());
            take(ANIMAL, false, left, &mut out, &mut by, ctx)?;
            ctx.check()?;
            let left = budget.saturating_sub(out.len());
            if left > 0 {
                ctx.progress(0.04, "Naming the things in front");
                let (w, h) = (area.width(), area.height());
                let (rw, rh) = request_size(w, h);
                let seen = Rgba8::new(w, h, picture.to_vec()).map_err(gen_err)?;
                let seen = if (rw, rh) == (w, h) { seen } else { resize_rgba8(&seen, rw, rh)? };
                let things = list_things(thinker, &seen, left.min(MAX_THINGS), seed, &JobProgress::span(ctx, 0.04, 0.06));
                // One of each thing first, then up to THINGS_EACH of each while the budget lasts:
                // the table before a third cup (the café photo's budget went to Cup 1 to 3 and the
                // table was never searched, measured in the app 2026-11-04).
                let mut found_one = vec![false; things.len()];
                for pass in [1usize, THINGS_EACH] {
                    for (thing, found) in things.iter().zip(found_one.iter_mut()) {
                        ctx.check()?;
                        let left = budget.saturating_sub(out.len());
                        if left == 0 {
                            break;
                        }
                        let have = out.iter().filter(|s| s.term == *thing).count();
                        if (pass > 1 && !*found) || have >= pass {
                            continue;
                        }
                        let before = out.len();
                        take(thing, false, have + (pass - have).min(left), &mut out, &mut by, ctx)?;
                        *found |= out.len() > before;
                    }
                }
            }
            if out.is_empty() {
                for p in crate::select_ml_cmds::SUBJECT_FALLBACKS {
                    ctx.check()?;
                    take(p, false, budget, &mut out, &mut by, ctx)?;
                    if !out.is_empty() {
                        break;
                    }
                }
            }
            if out.is_empty() {
                return Err(EngineError::Other("nothing matching \"the main subject\" was found".into()));
            }
        }
    }
    Ok((out, by.join(", ")))
}

/// Drop what is not a subject of its own: specks, whole-frame detections, near-duplicates,
/// parts inside wholes (unless the caller asked for the part by name, when it is cut out of the
/// whole); then, by `group`, join instances that overlap into one.
fn clean(found: &mut Vec<Subject>, w: usize, h: usize, group: Group) {
    let frame = (w * h) as u64;
    // Specks: under [`SPECK_SHARE`] of the largest instance of the same term (a cup is far
    // smaller than a person and still a layer; judged against the largest of all, the cards and
    // two cups of the café photo went, measured in the app 2026-11-03), or under
    // [`FRAME_SHARE`] of the frame; and whole-frame detections.
    let largest_of = |term: &str, found: &[Subject]| found.iter().filter(|s| s.term.eq_ignore_ascii_case(term)).map(|s| s.pixels).max().unwrap_or(0);
    let keep: Vec<bool> = found
        .iter()
        .map(|s| s.pixels * SPECK_SHARE >= largest_of(&s.term, found) && s.pixels * FRAME_SHARE >= frame && (s.pixels as f32) < WHOLE_FRAME * frame as f32)
        .collect();
    let mut k = 0;
    found.retain(|_| {
        k += 1;
        keep[k - 1]
    });
    // Pairs, earlier against later: the earlier (people before things, larger first) wins.
    let mut i = 0;
    while i < found.len() {
        let mut j = i + 1;
        while j < found.len() {
            let both = overlap(&found[i].cov, &found[j].cov);
            let (small, large) = if found[i].pixels <= found[j].pixels { (j, i) } else { (i, j) };
            let (sp, lp) = (found[small].pixels.max(1), found[large].pixels.max(1));
            let iou = both as f64 / (sp + lp - both) as f64;
            let inside = both as f64 / sp as f64;
            let same_term = found[i].term.eq_ignore_ascii_case(&found[j].term);
            if iou >= DUPLICATE_IOU {
                found.remove(j);
                continue;
            }
            if inside >= PART_INSIDE {
                if found[small].asked && !same_term {
                    // The hat on the man: both stay, the hat cut out of the man.
                    let (sc, lc) = (found[small].cov.clone(), &mut found[large].cov);
                    for (l, s) in lc.iter_mut().zip(&sc) {
                        if *s >= 0.5 {
                            *l = 0.0;
                        }
                    }
                    let area = Rect::new(0, 0, w as i32, h as i32);
                    let (b, n) = crate::select_ml_cmds::coverage_bounds(&found[large].cov, area);
                    found[large].bounds = b;
                    found[large].pixels = n;
                } else {
                    found.remove(small);
                    if small == i {
                        j = i + 1;
                        continue;
                    }
                    continue;
                }
            }
            let joined = match group {
                Group::None => false,
                Group::Auto => inside >= GROUP_OVERLAP,
                Group::Touching => both > 0 || order::touching(&found[i].shape(), &found[j].shape(), w, h),
            };
            if joined {
                let (jc, jt) = (found[j].cov.clone(), found[j].term.clone());
                let keep = &mut found[i];
                for (d, c) in keep.cov.iter_mut().zip(&jc) {
                    *d = d.max(*c);
                }
                if !keep.term.eq_ignore_ascii_case(&jt) && !keep.term.contains(&jt) {
                    keep.term = format!("{} and {}", keep.term, jt);
                }
                let area = Rect::new(0, 0, w as i32, h as i32);
                let (b, n) = crate::select_ml_cmds::coverage_bounds(&keep.cov, area);
                keep.bounds = b;
                keep.pixels = n;
                found.remove(j);
                continue;
            }
            j += 1;
        }
        i += 1;
    }
}

/// Every subject's alpha from `edges`, in place; which source served.
#[allow(clippy::too_many_arguments)]
fn matte_all(
    backend: &dyn GenerativeBackend,
    matte: Option<&Common>,
    doc: &Document,
    active: Option<LayerId>,
    picture: &[u8],
    area: Rect,
    subjects: &mut [Subject],
    only: Option<usize>,
    edges: Edges,
    ctx: &JobCtx,
) -> Result<Edges> {
    let (w, h) = (area.width(), area.height());
    let instances: Vec<Instance> = subjects.iter().map(|s| (s.cov.clone(), s.bounds, s.pixels)).collect();
    let mut used = edges;
    let mut alphas = None;
    if matches!(edges, Edges::Auto | Edges::Vitmatte) {
        ctx.progress(0.08, "Matting hair");
        alphas = crate::select_ml_cmds::vitmatte_alphas(backend, picture, area, &instances, only, ctx)?;
        if alphas.is_none() {
            // Once more: the first split of a session got nothing back from ViTMatte and fell
            // back to the whole-frame matte and then the refined mask (measured in the app
            // 2026-11-04, cold runs only).
            ctx.check()?;
            alphas = crate::select_ml_cmds::vitmatte_alphas(backend, picture, area, &instances, only, ctx)?;
        }
        used = Edges::Vitmatte;
    }
    ctx.check()?;
    if alphas.is_none()
        && matches!(edges, Edges::Auto | Edges::Vitmatte | Edges::Matte)
        && let Some(m) = matte
    {
        ctx.progress(0.08, "Matting edges");
        let name = subjects.first().map_or(PERSON, |s| s.check_term()).to_string();
        alphas = matte_alphas(backend, m, picture, (w, h), &instances, &name, ctx)?;
        used = Edges::Matte;
    }
    ctx.check()?;
    let alphas: Vec<Vec<f32>> = match alphas {
        Some(a) => a,
        None if used != Edges::Hard => {
            used = Edges::Refine;
            ctx.progress(0.15, "Refining edges");
            instances.iter().map(|(cov, _, _)| crate::select_ml_cmds::refine_coverage(doc, active, true, cov.clone(), area)).collect()
        }
        None => instances.iter().map(|(cov, _, _)| cov.clone()).collect(),
    };
    for (s, a) in subjects.iter_mut().zip(alphas) {
        s.alpha = a;
    }
    Ok(used)
}

/// The removal runs this many variations a try when peeling (the detector's check and the
/// seeds that follow stand in for the judge)...
const PEEL_VARIATIONS: u32 = 1;
/// ...and this many on the one-fill split (the judge picks the more natural).
const FLAT_VARIATIONS: u32 = 2;
/// The most things the vision-language model is asked to name...
const MAX_THINGS: usize = 6;
/// ...and the most instances of each it is asked for (three: the café photo's third cup was lost
/// at two, measured in the app 2026-11-03).
const THINGS_EACH: usize = 3;
/// The detector's word for an animal.
const ANIMAL: &str = "animal";
/// Two instances with a mask IoU at or above this are one.
const DUPLICATE_IOU: f64 = 0.7;
/// A smaller instance with this share of its mask inside a larger one is a part of it.
const PART_INSIDE: f64 = 0.85;
/// `group: auto` joins two instances when the smaller overlaps the larger by this share.
const GROUP_OVERLAP: f64 = 0.5;
/// A detection after a fill that holds this share of a subject's mask is the subject seen
/// again (with what the fill completed)...
const CONTINUES: f32 = 0.6;
/// ...as long as what the fill added to it is at most this many times its visible pixels; more
/// is someone new standing over it.
const COMPLETION_MAX: u64 = 2;
/// Blocks of a silhouette compared for what a fill kept of it, in pixels square.
const KEPT_BLOCK: usize = 8;
/// A block is kept when its mean colour moved at most this far (of 255): a fill's re-encoding
/// shifts a few levels, a repainted subject a few more; what stood behind it is another colour.
const KEPT_TOLERANCE: f32 = 24.0;
/// A fill that kept this share of a person's silhouette left them where they stood (or painted
/// them back as they were): rejected by the pixels, whatever the detector saw. The detector's
/// verdict alone let the front-left man of the café photo through, painted back almost pixel
/// for pixel (measured in the app 2026-11-04).
const KEPT_WHOLE: f32 = 0.75;
/// A detection in the hole that continues someone behind is still the cut subject painted back
/// when this share of its blocks in the hole kept their colour: a completion shows the one
/// behind, not what was cut.
const KEPT_LOOKS: f32 = 0.4;
/// Of a thing's detection, the share inside the thing's own hole (not its grow) at which it is
/// the thing painted back: a saucer or glass beside a cup overlaps its grow, not its spot.
const THING_IN_HOLE: f32 = 0.5;
/// A hole with a side over this many pixels is big: its grey box is filled in strips.
const BIG_HOLE_SIDE: u32 = 400;
/// A strip of the grey box is at most this long along the box's longer side...
const STRIP_LEN: usize = 400;
/// ...and there are at most this many of them...
const MAX_STRIPS: usize = 5;
/// Strips overlap by this many pixels, blended across them, so each sees the one before it and
/// no step shows at their edge.
const STRIP_OVERLAP: usize = 48;
/// The most pixels the classical patch fill works on (the crop around the hole scaled down).
const PATCH_PIXELS: usize = 400_000;
/// The ring around a hole a fill is judged against, in pixels.
const RING_WIDTH: usize = 12;
/// A ring with less texture than this (the mean luminance step between pixels [`texture_step`]
/// apart) is a plain backdrop: its fill is rightly plain, and strips have nothing to continue.
/// The café's walls run 0.010 to 0.013; at 0.012 the flat white fill of one hole was exempt
/// (measured in the app 2026-11-04).
const PLAIN_TEXTURE: f32 = 0.004;
/// Texture is measured between pixels one apart up to this many pixels a frame, further apart
/// on a larger picture ([`texture_step`]).
const TEXTURE_PIXELS: usize = 1_000_000;
/// The cells (px square) a fill is judged by for a part of the hole left blank or painted over.
const BLANK_CELL: usize = 32;
/// The share of the hole's pixels in blank (or painted-over) cells at which the fill is refused.
const BLANK_CELLS: f32 = 0.25;
/// A result is laid in wherever its feathered coverage is above this: the check for anyone the
/// fill painted looks over all of it.
const TOUCHED: f32 = 0.02;
/// A detection of a person or an animal with this share of its pixels where the fill painted
/// is someone the fill painted (one standing next to the hole overlaps its feathered edge less).
const NEW_INSIDE: f32 = 0.6;
/// The smallest detection, as a share of the frame, that counts as someone painted in: a
/// 30×71 px figure on the 1800×1201 café photo is a person (measured in the app 2026-11-04).
const NEW_MIN_SHARE: f64 = 0.0005;
/// ...and in pixels, on a small picture.
const NEW_MIN_PIXELS: u64 = 16;
/// How many instances of a person's or animal's kind the check asks for after a fill: enough
/// for everyone still in the picture and the small figures a fill may paint beside them.
const NEW_AT_MOST: usize = 16;
/// A fill whose texture is under this share of the ring's left the hole blank.
const BLANK_SHARE: f32 = 0.15;
/// A fill whose mean brightness is this far (0..1) from the ring's painted a shape over the hole.
const GHOST_LUMA: f32 = 0.3;
/// The pixels a region needs to be judged for texture and brightness.
const JUDGE_MIN_PIXELS: u32 = 64;
/// A sweep's later find with this share of its pixels inside what the fills painted is the
/// fills' own work (a blob, a sliver), not a subject: a completion adds at most
/// [`COMPLETION_MAX`] times what showed, two thirds.
const DEBRIS_IN_PAINTED: f32 = 0.7;
/// A speck: under this share (1/n) of the largest instance of its term.
const SPECK_SHARE: u64 = 100;
/// A speck: under this share (1/n) of the frame.
const FRAME_SHARE: u64 = 2000;
/// The band around a subject's hole its fill may repaint where another subject stands.
const TIGHT_BAND: usize = 2;
/// How far around another subject's mask its pixels are kept through a fill.
const PROTECT_GROW: usize = 2;
/// How close to a subject's mask a detection's pixel must be to join it.
const JOIN_REACH: i64 = 2;

/// What the detector is asked for on the filled background: the prompt that found the
/// subjects when it names a kind of thing ("person", the caller's prompt), else "person". The
/// generic prompts ("the main subject") find something on any picture: on a clean backdrop
/// they rejected all three fills (measured in the app 2026-11-03), and the failure this check
/// is for is a person painted back.
fn check_prompt(found_by: &str) -> &str {
    match found_by {
        "the main subject" | "the foreground object" | "object" => PERSON,
        p => p,
    }
}

/// The detector's and the removal's word for a person.
const PERSON: &str = "person";

/// Cells of [`ToneGrid`], in pixels.
const TONE_CELL: usize = 64;

/// A smooth per-channel correction (picture minus fill) measured where `known` says both show
/// the same backdrop, averaged over cells and carried across the rest (the subjects) as a
/// membrane: the fill's tone matched to its surroundings with no step at their edge. At most
/// [`TONE_MATCH_MAX`] a channel: a fill in other colours altogether keeps them.
struct ToneGrid {
    gw: usize,
    gh: usize,
    v: Vec<[f32; 3]>,
}

impl ToneGrid {
    fn new(picture: &[u8], filled: &[u8], known: impl Fn(usize) -> bool, w: usize, h: usize) -> Self {
        let (gw, gh) = (w.div_ceil(TONE_CELL).max(1), h.div_ceil(TONE_CELL).max(1));
        let mut sum = vec![[0.0f64; 3]; gw * gh];
        let mut count = vec![0u32; gw * gh];
        for k in 0..w * h {
            if !known(k) {
                continue;
            }
            let g = (k / w / TONE_CELL) * gw + (k % w) / TONE_CELL;
            for c in 0..3 {
                sum[g][c] += f64::from(picture[k * 4 + c]) - f64::from(filled[k * 4 + c]);
            }
            count[g] += 1;
        }
        let fixed: Vec<bool> = count.iter().map(|n| *n >= 16).collect();
        let (mut all, mut n) = ([0.0f64; 3], 0u64);
        for (s, c) in sum.iter().zip(&count) {
            for i in 0..3 {
                all[i] += s[i];
            }
            n += u64::from(*c);
        }
        let mean: [f32; 3] = std::array::from_fn(|i| if n > 0 { (all[i] / n as f64) as f32 } else { 0.0 });
        let mut v: Vec<[f32; 3]> =
            (0..gw * gh).map(|g| if fixed[g] { std::array::from_fn(|i| (sum[g][i] / f64::from(count[g])) as f32) } else { mean }).collect();
        // The cells with no backdrop take their neighbours' average until it settles.
        for _ in 0..(gw + gh) * 40 {
            let prev = v.clone();
            for gy in 0..gh {
                for gx in 0..gw {
                    let g = gy * gw + gx;
                    if fixed[g] {
                        continue;
                    }
                    let mut acc = [0.0f32; 3];
                    let mut m = 0.0f32;
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let (x, y) = (gx as i32 + dx, gy as i32 + dy);
                        if x < 0 || y < 0 || x >= gw as i32 || y >= gh as i32 {
                            continue;
                        }
                        let q = prev[y as usize * gw + x as usize];
                        for i in 0..3 {
                            acc[i] += q[i];
                        }
                        m += 1.0;
                    }
                    if m > 0.0 {
                        v[g] = std::array::from_fn(|i| acc[i] / m);
                    }
                }
            }
        }
        for cell in &mut v {
            for c in cell.iter_mut() {
                *c = c.clamp(-TONE_MATCH_MAX, TONE_MATCH_MAX);
            }
        }
        ToneGrid { gw, gh, v }
    }

    /// The correction at pixel (`x`, `y`), bilinear between cell centres.
    fn at(&self, x: usize, y: usize) -> [f32; 3] {
        let fx = ((x as f32 + 0.5) / TONE_CELL as f32 - 0.5).clamp(0.0, (self.gw - 1) as f32);
        let fy = ((y as f32 + 0.5) / TONE_CELL as f32 - 0.5).clamp(0.0, (self.gh - 1) as f32);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.gw - 1), (y0 + 1).min(self.gh - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let g = |x: usize, y: usize| self.v[y * self.gw + x];
        std::array::from_fn(|i| {
            let top = g(x0, y0)[i] * (1.0 - tx) + g(x1, y0)[i] * tx;
            let bottom = g(x0, y1)[i] * (1.0 - tx) + g(x1, y1)[i] * tx;
            top * (1.0 - ty) + bottom * ty
        })
    }
}

/// The share of the frame over which a detection counts as the whole frame (ignored).
const WHOLE_FRAME: f32 = 0.9;

/// Fills tried before Split gives up on a background with the subject painted back.
const FILL_TRIES: u32 = 5;

/// Of a detection on the filled background, the share inside the hole at which it counts as
/// the subject painted back (a person beside the hole overlaps it little).
const LEFT_IN_HOLE: f32 = 0.3;

/// The smallest share of the hole a detection inside it must cover to count: a quarter let a
/// person in the hole's corner through (about 18 % of it, on all three seeds of the
/// 12-megapixel portrait, measured in the app 2026-11-03); a few percent still ignores specks.
const LEFT_MIN_SHARE: f32 = 0.03;

/// The subjects' alphas from the matte model (Remove Background's), or `None` when the server
/// lacks it or its matte disagrees with the detector (kept little of the subjects, or much of
/// the rest). `only` is ViTMatte's: here every subject is cut from the one matte. The detector's masks are hard-edged and clip hair (the owner, 2026-11-02); the
/// matte is soft alpha. Each subject takes the matte within a band around its own mask (a
/// fraction of its size, so stray hair stays and other things the matte kept do not), its
/// mask's interior whole, and a pixel near two subjects goes to the nearer.
fn matte_alphas(
    backend: &dyn GenerativeBackend,
    plan: &Common,
    picture: &[u8],
    (w, h): (u32, u32),
    instances: &[crate::select_ml_cmds::Instance],
    name: &str,
    ctx: &JobCtx,
) -> Result<Option<Vec<Vec<f32>>>> {
    if resolve_auto(backend, &[DEFAULT_MATTE_TEMPLATE], "") != DEFAULT_MATTE_TEMPLATE || instances.is_empty() {
        return Ok(None);
    }
    let wu = w as usize;
    let image = Rgba8::new(w, h, picture.to_vec()).map_err(gen_err)?;
    let (rw, rh) = matte_request_size(w, h);
    let image = if (rw, rh) == (w, h) { image } else { resize_rgba8(&image, rw, rh)? };
    let req = Request {
        template: DEFAULT_MATTE_TEMPLATE.to_string(),
        prompt: name.to_string(),
        negative: plan.negative.clone(),
        seed: plan.seed,
        steps: plan.steps,
        guidance: plan.guidance,
        image: Some(image),
        mask: None,
        models: plan.models.clone(),
        size: None,
        params: BTreeMap::new(),
        references: Vec::new(),
    };
    // A failed matte is not a failed split: the refined masks still serve.
    let Ok(resp) = backend.run(&req, &JobProgress::span(ctx, 0.05, 0.3)) else { return Ok(None) };
    let Some(out) = resp.images.into_iter().next() else { return Ok(None) };
    let plane = Gray8::new(out.width, out.height, out.data.as_chunks::<4>().0.iter().map(|p| p[3]).collect()).map_err(gen_err)?;
    let plane = if (plane.width, plane.height) == (w, h) { plane } else { resize_gray8(&plane, w, h)? };
    let matte: Vec<f32> = plane.data.iter().map(|v| f32::from(*v) / 255.0).collect();
    // Per subject, within a window around it: the distance to its mask (outside) and into it
    // (inside), and how far the matte decides (`reach`).
    struct Near {
        win: Rect,
        outside: Vec<f32>,
        inside: Vec<f32>,
        reach: f32,
    }
    let area = Rect::new(0, 0, w as i32, h as i32);
    let mut near = Vec::with_capacity(instances.len());
    for (cov, bounds, _) in instances {
        let reach = (bounds.width().max(bounds.height()) as f32 * BAND).max(MIN_BAND);
        let win = bounds.inflate(reach.ceil() as i32 + 2).intersect(&area);
        let (ww, wh) = (win.width() as usize, win.height() as usize);
        let mut on = Vec::with_capacity(ww * wh);
        for y in win.y0..win.y1 {
            let row = y as usize * wu;
            on.extend(cov[row + win.x0 as usize..row + win.x1 as usize].iter().map(|c| *c >= 0.5));
        }
        let off: Vec<bool> = on.iter().map(|b| !*b).collect();
        let (outside, inside) = (photocraft_algo::selection::edt(&on, ww, wh), photocraft_algo::selection::edt(&off, ww, wh));
        if outside.len() != ww * wh || inside.len() != ww * wh {
            return Ok(None);
        }
        near.push(Near { win, outside, inside, reach });
    }
    // The nearest subject's distance at each pixel; infinite where none is near.
    let mut nearest = vec![f32::INFINITY; matte.len()];
    for n in &near {
        let ww = n.win.width() as usize;
        for (j, d) in n.outside.iter().enumerate() {
            let k = (n.win.y0 as usize + j / ww) * wu + n.win.x0 as usize + j % ww;
            if *d < n.reach {
                nearest[k] = nearest[k].min(*d);
            }
        }
    }
    let mut alphas = vec![vec![0.0f32; matte.len()]; instances.len()];
    let (mut kept, mut core) = (0.0f64, 0usize);
    for (n, alpha) in near.iter().zip(alphas.iter_mut()) {
        let ww = n.win.width() as usize;
        for (j, (d_out, d_in)) in n.outside.iter().zip(&n.inside).enumerate() {
            let k = (n.win.y0 as usize + j / ww) * wu + n.win.x0 as usize + j % ww;
            if *d_out >= n.reach || (*d_out > 0.0 && *d_out > nearest[k] + 0.5) {
                continue;
            }
            if *d_in > n.reach * 0.5 {
                // Deep inside the mask: solid, whatever the matte made of it.
                alpha[k] = 1.0;
                kept += f64::from(matte[k]);
                core += 1;
            } else {
                alpha[k] = matte[k] * ((n.reach - d_out) / (n.reach * 0.5)).clamp(0.0, 1.0);
            }
        }
    }
    let (mut stray, mut rest) = (0.0f64, 0usize);
    for (m, d) in matte.iter().zip(&nearest) {
        if d.is_infinite() {
            stray += f64::from(*m);
            rest += 1;
        }
    }
    // The matte must agree with the detector: most of the subjects' interiors kept, little of
    // what lies well away from them.
    let kept_ok = core == 0 || kept / core as f64 >= MIN_KEPT;
    let stray_ok = rest == 0 || stray / rest as f64 <= MAX_STRAY;
    Ok((kept_ok && stray_ok).then_some(alphas))
}

/// Soft-mask coverage above this is part of the hole the removal repaints.
const HOLE_MIN: f32 = 0.1;

/// The band around a subject's mask the matte decides, as a fraction of its larger side...
const BAND: f32 = 0.04;
/// ...and at least this many pixels.
const MIN_BAND: f32 = 6.0;
/// The matte's least mean over the subjects' interiors to be trusted.
const MIN_KEPT: f64 = 0.6;
/// The matte's most mean over the rest (well away from every subject) to be trusted.
const MAX_STRAY: f64 = 0.2;

#[cfg(test)]
mod check_tests {
    use super::check_prompt;

    #[test]
    fn the_fill_check_asks_for_a_kind_of_thing() {
        assert_eq!(check_prompt("the main subject"), "person");
        assert_eq!(check_prompt("the foreground object"), "person");
        assert_eq!(check_prompt("object"), "person");
        assert_eq!(check_prompt("person"), "person");
        assert_eq!(check_prompt("the lighthouse"), "the lighthouse");
    }
}

#[cfg(test)]
mod tone_tests {
    use super::ToneGrid;

    #[test]
    fn the_tone_grid_carries_the_ring_offset_across_the_hole() {
        // A 256x256 picture of 100 and a fill of 90: the backdrop is known outside the middle
        // square only; the correction is +10 everywhere, the middle included.
        let (w, h) = (256usize, 256usize);
        let picture = vec![100u8; w * h * 4];
        let filled = vec![90u8; w * h * 4];
        let inside = |k: usize| (64..192).contains(&(k % w)) && (64..192).contains(&(k / w));
        let tone = ToneGrid::new(&picture, &filled, |k| !inside(k), w, h);
        for (x, y) in [(0, 0), (128, 128), (100, 200), (255, 255)] {
            let t = tone.at(x, y);
            assert!(t.iter().all(|v| (v - 10.0).abs() < 0.5), "{x},{y}: {t:?}");
        }
    }
}
