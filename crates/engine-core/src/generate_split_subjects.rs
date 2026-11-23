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

use std::collections::BTreeMap;
use std::sync::Arc;

use photocraft_doc::{Document, LayerId};
use photocraft_geom::Rect;

use crate::generate_split_order::{self as order, Shape};
use crate::jobs::JobCtx;
use crate::select_ml_cmds::{Instance, SubjectFinder};
use crate::{EngineError, Result};

use super::{
    BACKGROUND_PROMPT, Common, DEFAULT_MATTE_TEMPLATE, FillJob, GenerativeInfo, Gray8, JobProgress, Layer, MATTE_DEFAULT_PROMPT, MAX_GROW, PEEL_PROMPT,
    PREFILL, REMOVE_BG, REMOVE_MIN_REQUEST, Request, Rgba8, SPLIT, Session, TONE_MATCH_MAX, Task, Tuning, Value, at_least, background_job, box_blur, dilate,
    embed_coverage, fill_rect, fill_region, gen_err, gray8_at, json, list_things, matte_request_size, name_selected, plan_common, removal_job, removal_prompt,
    remove_grow, request_size, resize_rgba8, resolve_auto, sees, set_generative_info, split_removal, write_rgba8,
};

/// The simple split: progressive subject removal, nearest first, until the detector finds nothing.
#[path = "generate_split_simple.rs"]
pub(super) mod simple;

// Split's parts, one file each, their items re-exported here.
#[path = "generate_split_find.rs"]
mod find;
use find::{HOLE_MIN, clean, find_subjects, matte_all};
#[path = "generate_split_fill.rs"]
mod fill;
use fill::{HoleFill, Verdict, fill_hole, one_fill};
#[path = "generate_split_checks.rs"]
mod checks;
pub(in crate::generate_cmds) use checks::*;

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
    /// Keep every refused fill as a hidden layer under the background (`keepTries`).
    pub keep_tries: bool,
    pub order: Order,
    pub group: Group,
    /// Diagnostic, simple method only (`firstPassOnly`, off by default): place the first pass's
    /// subjects under their masks and stop, no removal.
    pub first_pass_only: bool,
    /// How finely the simple method splits (`detail`): which subjects are big enough for a layer
    /// of their own, and whether a subject inside another gets one.
    pub detail: Detail,
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

/// How finely Split into Layers (simple method) splits a picture (`detail`), the dialog's scale
/// from rough to fine, in place of a layer count (the owner, 2026-11-05: the count "doesn't
/// work"; the default of 3 cut two subjects and left the rest in the background).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Detail {
    /// `rough`: the big subjects only (1 % of the frame or more); what lies inside one (a face,
    /// a glove) stays in its layer.
    Rough,
    /// `medium` (the default): subjects of 0.1 % of the frame or more, each part of one on its
    /// own layer above it.
    Medium,
    /// `fine`: everything the detector finds, parts included.
    Fine,
}

impl Detail {
    pub(super) fn parse(v: Option<&str>) -> std::result::Result<Detail, String> {
        Ok(match v.map(str::trim).unwrap_or("medium") {
            "" | "medium" => Detail::Medium,
            "rough" => Detail::Rough,
            "fine" => Detail::Fine,
            other => return Err(format!("`detail` must be rough, medium or fine (got `{other}`)")),
        })
    }

    /// The smallest subject given a layer, as a share of the frame's pixels.
    pub(super) fn min_share(self) -> f64 {
        match self {
            Detail::Rough => 0.01,
            Detail::Medium => 0.001,
            Detail::Fine => 0.0,
        }
    }

    /// Whether a subject inside another (a part) gets its own layer.
    pub(super) fn part_layers(self) -> bool {
        self != Detail::Rough
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Detail::Rough => "rough",
            Detail::Medium => "medium",
            Detail::Fine => "fine",
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
    let SubjectsPlan { prompt, subjects, seed, name, edges, complete, sweep, keep_tries, order: ordering, group, .. } = plan;
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
    let mut job = removal_job(s, "Split into Layers", split_removal(if complete { PEEL_VARIATIONS } else { FLAT_VARIATIONS }, seed, tuning.clone()))?;
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
                    keep_tries,
                    (group, ordering, edges),
                    (peel_box.as_ref(), last_box.as_ref()),
                )?;
                let Swept {
                    subjects,
                    cut,
                    background,
                    left_in,
                    tries,
                    found_by,
                    used,
                    fill_ms,
                    fill_seed,
                    generations,
                    already,
                    refused,
                    picture_tex,
                    by_tiles,
                } = swept;
                ctx.progress(0.95, "Placing");
                let left: Vec<Vec<f32>> = left_in.iter().map(|&i| hole_of(&subjects[i])).collect();
                let behind_left = left_behind(&finder, &background, area, &left, &already, picture_tex, by_tiles, ctx)?;
                let detector = finder.detector_stats();
                let (ids, placed, incomplete, refused_ids) =
                    place_all(doc, active, area, base.as_deref(), &found_by, &template_id, fill_seed, &subjects, cut, background, &left_in, refused)?;
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
                    refused_ids,
                    detector,
                    boxes_of(&already, area),
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
            // Who was in the picture before any fill, apart from the subjects: no fill's work.
            // On a plain backdrop the detector is not shown the picture tile by tile.
            let by_tiles = !plain_backdrop(&picture, &subjects, wu, hu);
            let already = already_there(&finder, &picture, area, &creature_terms(&subjects), &subjects, by_tiles, ctx)?;
            // The picture's texture: what a figure a fill painted is held against (fog is less).
            let picture_tex = picture_texture(&picture, wu, hu);
            // The refused fills, kept on request (`keepTries`), named for their subject and try.
            let mut refused: Vec<(String, Vec<u8>)> = Vec::new();
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
                    cut.push(under_alpha(&p, &subjects[i].alpha, wu));
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
                    let already_ref = &already;
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
                        // Whether a detection has no more detail than fog (then it is no figure).
                        let fog = |cov: &[f32]| fog_counted(finder_ref, filled, cov, wu, hu, picture_tex);
                        // The pictures in the result, looked for once someone new is about to
                        // be refused: whoever is in one is a picture of a person.
                        let mut pictures = Pictures::new(finder_ref, filled, area, ctx);
                        let mut own = None;
                        for (term, (cov, b, _)) in &seen {
                            if let Some(why) = judge(term, cov, *b, hole_ref, grown_ref, touched, cut, &behind_shapes, already_ref, &looks, &fog, wu, patched) {
                                if says_someone_new(&why) && pictures.holds(cov, *b)? {
                                    continue;
                                }
                                own = Some((why, *b));
                                break;
                            }
                        }
                        // Then, tile by tile over what the fill touched, anyone too small to find
                        // in the whole frame.
                        let own = match own {
                            Some(why) => Some(why),
                            None => anyone_small(finder_ref, filled, area, touched, &asks, cut, &behind_shapes, already_ref, &fog, &mut pictures, ctx)?,
                        };
                        Ok((seen, own))
                    };
                    // The subjects still behind keep their own pixels through this fill.
                    let protect = protect_mask(behind, wu * hu);
                    // A person's silhouette is watched by its pixels too (a cup's is not: a cup on
                    // a table of its own colour looks kept whatever the fill did).
                    let watch = creature(this.check_term()).then_some(hole.as_slice());
                    let thing = !creature(this.check_term());
                    let keep = keep_tries;
                    let HoleFill { outcome, tries: hole_tries, ms, generations: made, refused: hole_refused } =
                        fill_hole(doc, active, below, ctx, area, &p, &hole, Some(&protect), watch, thing, keep, &removal, box_job, (lo, hi), &mut check)?;
                    tries.extend(hole_tries.into_iter().map(|mut t| {
                        t["behind"] = json!(label.as_str());
                        t
                    }));
                    refused.extend(hole_refused.into_iter().map(|(why, pixels)| (format!("{label}: {why}"), pixels)));
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
                    cut.push(under_alpha(&picture, &sub.alpha, wu));
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
                    let fog = |cov: &[f32]| fog_counted(finder_ref, filled, cov, wu, hu, picture_tex);
                    let mut pictures = Pictures::new(finder_ref, filled, area, ctx);
                    for (term, at_most) in &asks {
                        for (cov, b, _) in seen_of(finder_ref, filled, area, term, *at_most, ctx)? {
                            if mostly_in_hole(&cov, &grown_hole, LEFT_IN_HOLE) {
                                return Ok(((), Some((format!("the {term} still in the hole"), b))));
                            }
                            if creature(term)
                                && let Some(why) = someone_new(term, &cov, b, touched, (term, &hole, holed), &[], &already, wu)
                                && !fog(&cov)
                                && !(says_someone_new(&why) && pictures.holds(&cov, b)?)
                            {
                                return Ok(((), Some((why, b))));
                            }
                        }
                    }
                    Ok(((), anyone_small(finder_ref, filled, area, touched, &asks, (PERSON, &hole, holed), &[], &already, &fog, &mut pictures, ctx)?))
                };
                let HoleFill { outcome, tries: hole_tries, ms, generations: made, refused: hole_refused } = fill_hole(
                    doc,
                    active,
                    below,
                    ctx,
                    area,
                    &picture,
                    &hole,
                    None,
                    None,
                    false,
                    keep_tries,
                    &removal,
                    last_box.as_ref(),
                    (0.3, 0.9),
                    &mut check,
                )?;
                tries.extend(hole_tries.into_iter().map(|mut t| {
                    t["behind"] = json!("all");
                    t
                }));
                refused.extend(hole_refused.into_iter().map(|(why, pixels)| (format!("all: {why}"), pixels)));
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
            let behind_left = left_behind(&finder, &background, area, &left, &already, picture_tex, by_tiles, ctx)?;
            let detector = finder.detector_stats();
            let (ids, placed, incomplete, refused_ids) =
                place_all(doc, active, area, base.as_deref(), &found_by, &template_id, fill_seed, &subjects, cut, background, &left_in, refused)?;
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
                refused_ids,
                detector,
                boxes_of(&already, area),
            ))
        },
        move |(
            ids,
            placed,
            incomplete,
            tries,
            found_by,
            template_id,
            seed,
            (fill_ms, generations),
            total_ms,
            (w, h),
            matted,
            ordered,
            left_behind,
            refused_ids,
            detector,
            already,
        )| {
            json!({
                "layers": ids.iter().map(|l| l.0).collect::<Vec<_>>(), "count": ids.len(),
                "subjects": placed, "incomplete": incomplete, "fills": tries.len(), "generations": generations, "tries": tries, "order": ordered,
                "foundBy": found_by, "template": template_id, "seed": seed, "leftBehind": left_behind,
                "refusedLayers": refused_ids.iter().map(|l| l.0).collect::<Vec<_>>(),
                "detector": detector, "already": already,
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
    /// Who was in the picture before any fill, apart from the subjects ([`already_there`]).
    already: Vec<Figure>,
    /// The refused fills kept on request, named for their subject and try.
    refused: Vec<(String, Vec<u8>)>,
    /// The picture's texture ([`picture_texture`]).
    picture_tex: f32,
    /// Whether the detector is shown the picture tile by tile (not on a plain backdrop).
    by_tiles: bool,
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
    keep: bool,
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
    // Who was in the picture before any fill, apart from the subjects: no fill's work.
    let by_tiles = !plain_backdrop(picture, &pending, wu, hu);
    let already = already_there(finder, picture, area, &creature_terms(&pending), &pending, by_tiles, ctx)?;
    let picture_tex = picture_texture(picture, wu, hu);
    let mut refused: Vec<(usize, String, Vec<u8>)> = Vec::new();
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
                    [] => {
                        // Their part apart from their mask, uncovered by the removals before
                        // them ([`part_below`]), is theirs: into their mask while they wait
                        // their turn, no subject of its own once they are gone.
                        let part_of = |q: &Subject| creature(&q.term) && q.term.eq_ignore_ascii_case(&s.term) && part_below(&s, q);
                        if let Some(i) = pending.iter().position(&part_of) {
                            let m = fresh[i].get_or_insert_with(|| pending[i].cov.clone());
                            for (d, c) in m.iter_mut().zip(&s.cov) {
                                *d = d.max(*c);
                            }
                        } else if !done.iter().any(&part_of) {
                            new.push(s);
                        }
                    }
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
        cut.push(under_alpha(&p, &this.alpha, wu));
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
            let fog = |cov: &[f32]| fog_counted(finder, filled, cov, wu, hu, picture_tex);
            let mut pictures = Pictures::new(finder, filled, area, ctx);
            let mut own = None;
            for (term, (cov, b, _)) in &seen {
                if let Some(why) = judge(term, cov, *b, &hole, &grown_hole, touched, cut_shape, &behind_shapes, &already, &looks, &fog, wu, patched) {
                    if says_someone_new(&why) && pictures.holds(cov, *b)? {
                        continue;
                    }
                    own = Some((why, *b));
                    break;
                }
            }
            let own = match own {
                Some(why) => Some(why),
                None => anyone_small(finder, filled, area, touched, &asks, cut_shape, &behind_shapes, &already, &fog, &mut pictures, ctx)?,
            };
            Ok((seen, own))
        };
        // Everyone still on the list keeps their own pixels through this fill.
        let protect = protect_mask(&pending, wu * hu);
        let watch = creature(this.check_term()).then_some(hole.as_slice());
        let thing = !creature(this.check_term());
        let HoleFill { outcome, tries: hole_tries, ms, generations: made, refused: hole_refused } =
            fill_hole(doc, active, below, ctx, area, &p, &hole, Some(&protect), watch, thing, keep, &removal, box_job, (lo, hi), &mut check)?;
        // Named by its layer once every subject is known.
        tries.extend(hole_tries.into_iter().map(|mut t| {
            t["behind"] = json!(k);
            t
        }));
        refused.extend(hole_refused.into_iter().map(|(why, pixels)| (k, why, pixels)));
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
    let refused: Vec<(String, Vec<u8>)> = refused.into_iter().map(|(k, why, pixels)| (format!("{}: {why}", labels[k]), pixels)).collect();
    Ok(Swept { subjects: done, cut, background: p, left_in, tries, found_by, used, fill_ms, fill_seed, generations, already, refused, picture_tex, by_tiles })
}

/// What [`place_all`] made: the layers' ids (the background first, then the subjects back to
/// front), the `subjects` of the result, the names of the subjects left in the background
/// (`incomplete`) and the ids of the refused fills kept as layers (`refusedLayers`).
type Placed = (Vec<LayerId>, Vec<Value>, Vec<String>, Vec<LayerId>);

/// The layers: the background lowest, then the subjects back to front (the nearest on top),
/// named "{base} k/n" or by their terms; the ids, each subject's record and the names of those
/// left in the background. The `refused` fills (`keepTries`), named, go under the background,
/// hidden; their ids come last.
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
    refused: Vec<(String, Vec<u8>)>,
) -> Result<Placed> {
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
    let mut refused_ids: Vec<LayerId> = Vec::with_capacity(refused.len());
    for (why, pixels) in refused {
        let id = place(doc, active, pixels, why)?;
        if let Some(layer) = doc.layer_mut(id) {
            layer.visible = false;
        }
        refused_ids.push(id);
    }
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
    Ok((ids, placed, incomplete, refused_ids))
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
/// them (`seen`), or their completed part seen on its own ([`fragment_of`]), shows more of them;
/// whether any grew enough to count as completed. Only what a
/// detection adds next to the subject counts; a patch of the fill the detector took for it
/// stays out (a detached dark block under the woman's layer, measured in the app 2026-11-03).
fn grow_behind(behind: &mut [Subject], seen: &[(String, Instance)], grown_hole: &[f32], area: Rect) -> bool {
    let (w, h) = (area.width() as usize, area.height() as usize);
    let mut any_grew = false;
    for b in behind.iter_mut() {
        let shape = [(b.check_term(), b.cov.as_slice(), b.pixels)];
        let best = seen
            .iter()
            .filter(|(t, (cov, _, _))| t == b.check_term() && (continues(cov, &b.cov, b.pixels) || fragment_of(t, cov, grown_hole, shape, w)))
            .max_by_key(|(_, (cov, _, n))| (overlap(cov, &b.cov), *n));
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

/// `picture` (RGBA8, `w` wide) under `alpha`: the pixels, their alpha scaled.
fn under_alpha(picture: &[u8], alpha: &[f32], w: usize) -> Vec<u8> {
    // The soft edge takes the subject's own colour (the backdrop's tint solved out), so hair
    // does not carry the old background onto whatever the layer is laid over.
    let mut colours = picture.to_vec();
    if w > 0 {
        photocraft_algo::matting::unmix_rgba8(&mut colours, w, alpha.len() / w, alpha);
    }
    colours.as_chunks::<4>().0.iter().zip(alpha).flat_map(|(p, a)| [p[0], p[1], p[2], (f32::from(p[3]) * a.clamp(0.0, 1.0)).round() as u8]).collect()
}

/// A subject's hole: wherever its mask or its alpha shows it (the soft fringe is still the
/// subject's), all or nothing, as the Remove tool's.
fn hole_of(sub: &Subject) -> Vec<f32> {
    // A subject not matted yet (a sweep's, waiting for its turn) has its mask alone.
    let alpha = sub.alpha.iter().copied().chain(std::iter::repeat(0.0));
    sub.cov.iter().zip(alpha).map(|(c, a)| if c.max(a) > HOLE_MIN { 1.0 } else { 0.0 }).collect()
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
/// How much of a detection must be a subject behind's own pixels and their completed part joined
/// to them for it to be their fragment ([`fragment_of`]).
const FRAGMENT_SHARE: f32 = 0.9;
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
/// A tile of the picture the detector is shown on its own ([`found_by_tiles`]): at least this
/// long a side, else a third of the picture's longer side ([`ZOOM_ACROSS`]), widened by
/// [`ZOOM_MARGIN`] for its context.
const ZOOM_TILE: usize = 640;
const ZOOM_ACROSS: usize = 3;
const ZOOM_MARGIN: usize = 32;
/// The smallest find a tile's look holds against a fill: its mask's pixels and its bounds' sides.
const ZOOM_MIN_PIXELS: u64 = 100;
const ZOOM_MIN_SIDE: u32 = 8;
/// A shape painted over part of the hole ([`ghost_cluster`]): this many ghost cells joined, at
/// least, and this share of the hole's pixels.
/// A detection with texture under this share of the picture's is fog, not a figure.
const FIGURE_TEXTURE: f32 = 0.4;
const GHOST_CLUSTER_CELLS: usize = 6;
const GHOST_CLUSTER_SHARE: f32 = 0.02;
/// The tries before the patch for a big hole with texture around it: the removal, the box in
/// strips twice, the box whole.
const BIG_TRIES: u32 = 4;
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
/// What the detector is asked for when a person or an animal is about to be refused as someone
/// the fill painted: the framed pictures in the result ([`Pictures`])...
const PICTURE_TERMS: [&str; 2] = ["framed picture", "picture frame"];
/// ...at most this many of each...
const PICTURES_AT_MOST: usize = 4;
/// ...each at most this share of the frame (a doorway or a window with someone in it is
/// bigger; the café's framed photographs are a third of a percent)...
const PICTURE_MOST: f32 = 0.02;
/// ...and at least this many times the figure's own bounds (the figure handed back as the
/// picture is no picture).
const PICTURE_OVER: f32 = 1.25;
/// A figure with this share of its pixels inside a picture's bounds is in the picture.
const IN_PICTURE: f32 = 0.8;
/// A sweep's later find of a creature kind is a subject's part apart from their mask
/// ([`part_below`]) when its top is below their middle and at most this share of their height
/// under their bottom...
const PART_GAP: f32 = 0.25;
/// ...and it lies within their width widened by this share of it each side.
const PART_SIDE: f32 = 0.25;

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
