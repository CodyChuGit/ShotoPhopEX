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
    REMOVE_BG, REMOVE_MIN_REQUEST, Request, Rgba8, SPLIT, Session, Task, Tuning, Value, at_least, background_job, box_blur, dilate, embed_coverage, fill_rect,
    fill_region, gen_err, json, list_things, matte_request_size, name_selected, plan_common, removal_job, removal_prompt, remove_grow, request_size,
    resize_gray8, resize_rgba8, resolve_auto, set_generative_info, write_rgba8,
};

/// The most subjects one split gives their own layer (`layers` - 1 at most).
pub(super) const MAX_SUBJECTS: usize = 11;

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
struct Subject {
    /// The detector's coverage (hard), grown by what the fills completed.
    cov: Vec<f32>,
    bounds: Rect,
    pixels: u64,
    /// What it was found as ("person", "cup", "the main subject").
    term: String,
    /// The caller named the term.
    asked: bool,
    /// Its soft alpha (hair), from `edges`.
    alpha: Vec<f32>,
    /// The fill completed what nearer subjects hid (or nothing hid it).
    completed: bool,
    /// Its mask grew since its alpha was matted.
    grown: bool,
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
    let SubjectsPlan { prompt, subjects, seed, name, edges, complete, order: ordering, group } = plan;
    let finder = crate::select_ml_cmds::subject_finder(s, prompt.as_deref())?;
    let terms: Option<Vec<Term>> = prompt.as_deref().map(parse_terms).filter(|t| !t.is_empty());
    let tuning = Tuning { fast_draft: s.prefs().integrations.fast_draft, ..Tuning::default() };
    // The removal (the quick, clean route on most pictures), then the grey-box fills: behind a
    // subject with others still behind it, the scene continued; behind the last, the backdrop.
    let job = removal_job(s, "Split into Layers", 0, false, false, if complete { PEEL_VARIATIONS } else { FLAT_VARIATIONS }, seed, true, tuning.clone())?;
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
            ctx.progress(0.0, "Finding subjects");
            let (mut found, found_by) = find_subjects(&finder, job.thinker.as_ref(), &picture, area, terms.as_deref(), budget, job.seed, ctx)?;
            ctx.check()?;
            clean(&mut found, wu, hu, group);
            found.truncate(budget);
            if found.is_empty() {
                return Err(EngineError::Other(format!("nothing matching \"{found_by}\" was found")));
            }
            let found_by = kept_terms(&found_by, &found);
            // Each subject's alpha: ViTMatte inside a band around the detector's mask (hair,
            // strand by strand), else the matte model's soft alpha around it, else the mask with
            // the edge refined from the pixels, as Select Subject does (or as `edges` asks).
            let backend = Arc::clone(&job.backend);
            let used = matte_all(backend.as_ref(), matte.as_ref(), doc, *active, &picture, area, &mut found, edges, ctx)?;
            ctx.check()?;
            // Front to back.
            ctx.progress(0.2, "Ordering by depth");
            let mut subjects: Vec<Subject> = in_depth_order(found, ordering, wu, hu);
            let template_id = job.template_id.clone();
            let below = *active;
            let mut fill_ms = 0u64;
            let mut fill_seed = 0u64;
            // The subjects (front-to-back indices) the fill could not clear: left in the background.
            let mut left_in: Vec<usize> = Vec::new();
            // Each subject's layer pixels, front to back, and the background.
            let mut cut: Vec<Vec<u8>> = Vec::with_capacity(subjects.len());
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
                    // What the check asks the detector for: this subject's kind (painted back?)
                    // and the kinds still behind it (their completed parts).
                    let mut asks: Vec<(String, usize)> = vec![(this.check_term().to_string(), 1)];
                    for b in behind.iter() {
                        match asks.iter_mut().find(|(t, _)| t == b.check_term()) {
                            Some((_, n)) => *n += 1,
                            None => asks.push((b.check_term().to_string(), 2)),
                        }
                    }
                    let behind_shapes: Vec<(String, Vec<f32>, u64)> = behind.iter().map(|b| (b.check_term().to_string(), b.cov.clone(), b.pixels)).collect();
                    let finder_ref = &finder;
                    let hole_ref = &hole;
                    let mut check = |filled: &[u8]| -> Result<Option<Vec<(String, Instance)>>> {
                        let mut seen: Vec<(String, Instance)> = Vec::new();
                        for (term, at_most) in &asks {
                            for inst in finder_ref.find_in(filled, area, term, *at_most, ctx)? {
                                seen.push((term.clone(), inst));
                            }
                        }
                        let painted_back = seen.iter().any(|(term, (cov, _, _))| {
                            mostly_in_hole(cov, hole_ref) && !behind_shapes.iter().any(|(t, bc, bn)| t == term && continues(cov, bc, *bn))
                        });
                        Ok(if painted_back { None } else { Some(seen) })
                    };
                    // The subjects still behind keep their own pixels through this fill.
                    let protect = protect_mask(behind, wu * hu);
                    let outcome = fill_hole(doc, active, below, ctx, area, &p, &hole, Some(&protect), &removal, box_job, (lo, hi), &mut check)?;
                    *active = below;
                    let Some((filled, seed_used, ms, seen)) = outcome else {
                        left_in.push(i);
                        continue;
                    };
                    fill_ms += ms;
                    fill_seed = seed_used;
                    // The subjects behind, completed where the fill showed more of them.
                    let any_grew = grow_behind(behind, &seen, &grown_hole, area);
                    p = filled;
                    if any_grew {
                        let (_, rest) = subjects.split_at_mut(i + 1);
                        matte_all(backend.as_ref(), matte.as_ref(), doc, *active, &p, area, rest, edges, ctx)?;
                        for b in rest.iter_mut() {
                            b.grown = false;
                        }
                    }
                    ctx.check()?;
                }
                if left_in.len() == subjects.len() {
                    return Err(EngineError::Other(format!(
                        "Split into Layers could not clear the {} out of the background: the fill painted it back on {FILL_TRIES} tries, so nothing was changed",
                        subjects[0].removal_name()
                    )));
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
                let finder_ref = &finder;
                let hole_ref = &hole;
                let mut check = |filled: &[u8]| -> Result<Option<()>> {
                    for (term, at_most) in &asks {
                        for (cov, _, _) in finder_ref.find_in(filled, area, term, *at_most, ctx)? {
                            if mostly_in_hole(&cov, hole_ref) {
                                return Ok(None);
                            }
                        }
                    }
                    Ok(Some(()))
                };
                let outcome = fill_hole(doc, active, below, ctx, area, &picture, &hole, None, &removal, last_box.as_ref(), (0.3, 0.9), &mut check)?;
                *active = below;
                let Some((filled, seed_used, ms, ())) = outcome else {
                    return Err(EngineError::Other(format!(
                        "Split into Layers could not clear the {name} out of the background: the fill painted it back on {FILL_TRIES} tries, so nothing was changed"
                    )));
                };
                fill_ms += ms;
                fill_seed = seed_used;
                for sub in subjects.iter_mut() {
                    sub.completed = false;
                }
                filled
            };
            ctx.progress(0.95, "Placing");
            // The background lowest, then the subjects back to front: the nearest on top.
            let count = 1 + subjects.len();
            let labels = subjects_labels(&subjects);
            let mut ids: Vec<LayerId> = Vec::with_capacity(count);
            let mut placed: Vec<Value> = Vec::with_capacity(subjects.len());
            let place = |doc: &mut Document, active: &mut Option<LayerId>, pixels: Vec<u8>, layer_name: String| -> Result<LayerId> {
                let mut layer = Layer::raster(layer_name.clone(), doc.pixel_format());
                write_rgba8(&mut layer, &Rgba8::new(w, h, pixels).map_err(gen_err)?, area)?;
                set_generative_info(
                    &mut layer,
                    &GenerativeInfo {
                        command: SPLIT.to_string(),
                        prompt: found_by.clone(),
                        template: template_id.clone(),
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
            let bg_name = match &base {
                Some(b) => format!("{b} 1/{count}"),
                // Not a second "Background" beside the document's own.
                None if doc.layers.iter().any(|l| l.name == "Background") => "Split background".to_string(),
                None => "Background".to_string(),
            };
            ids.push(place(doc, active, background, bg_name)?);
            for (k, i) in (0..subjects.len()).rev().enumerate() {
                let sub = &subjects[i];
                let layer_name = match &base {
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
                .map(|&i| match &base {
                    Some(b) => format!("{b} {}/{count}", subjects.len() - i + 1),
                    None => labels[i].clone(),
                })
                .collect();
            Ok((ids, placed, incomplete, found_by, template_id, fill_seed, fill_ms, started.elapsed().as_millis() as u64, (w, h), used.name(), ordering.name()))
        },
        move |(ids, placed, incomplete, found_by, template_id, seed, fill_ms, total_ms, (w, h), matted, ordered)| {
            json!({
                "layers": ids.iter().map(|l| l.0).collect::<Vec<_>>(), "count": ids.len(),
                "subjects": placed, "incomplete": incomplete, "order": ordered,
                "foundBy": found_by, "template": template_id, "seed": seed,
                "width": w, "height": h, "fillMs": fill_ms, "ms": total_ms, "edges": matted,
            })
        },
    )
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
    let front_first: Vec<usize> = match ordering {
        Order::Auto => order::front_to_back(&found.iter().map(Subject::shape).collect::<Vec<_>>(), w, h),
        Order::Size => {
            let mut ix: Vec<usize> = (0..found.len()).collect();
            ix.sort_by_key(|i| found[*i].pixels);
            ix
        }
        Order::Found => (0..found.len()).rev().collect(),
    };
    let mut subjects: Vec<Subject> = Vec::with_capacity(found.len());
    let mut taken: Vec<Option<Subject>> = found.into_iter().map(Some).collect();
    for i in front_first {
        if let Some(sub) = taken.get_mut(i).and_then(Option::take) {
            subjects.push(sub);
        }
    }
    subjects
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
    sub.cov.iter().zip(&sub.alpha).map(|(c, a)| if c.max(*a) > HOLE_MIN { 1.0 } else { 0.0 }).collect()
}

/// Whether a detection sits mostly in the hole and covers enough of it: a detection mostly
/// inside the hole and at least [`LEFT_MIN_SHARE`] of it (a person beside the hole overlaps it
/// little; a speck is nothing). A detection over nearly the whole frame is the detector's
/// quirk, not a subject (it flagged one clean backdrop fill whole, 2026-11-03).
fn mostly_in_hole(cov: &[f32], hole: &[f32]) -> bool {
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
    on > 0 && inside as f32 >= LEFT_IN_HOLE * on as f32 && inside as f32 >= LEFT_MIN_SHARE * holed as f32
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

/// What a fill left: the picture with the hole filled, the fill's seed and time, and what the
/// check saw in it.
type Filled<T> = (Vec<u8>, u64, u64, T);

/// Fill `hole` (one value per canvas pixel) in `picture`, the document's composite standing in
/// for it through a work layer: the removal first (quick, and clean on most pictures), then,
/// after it paints a subject back, the grey-box fill of the hole's bounds laid only over the
/// hole (widened and feathered) and tone matched. `check` judges each result (the detector
/// looking for the subject painted back) and keeps what it saw; the first it accepts comes back
/// with its seed and the fill time. `None` after [`FILL_TRIES`] rejected tries.
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
    removal: &FillJob,
    box_job: Option<&FillJob>,
    (lo, hi): (f32, f32),
    check: &mut dyn FnMut(&[u8]) -> Result<Option<T>>,
) -> Result<Option<Filled<T>>> {
    let (w, h) = (area.width(), area.height());
    let (wu, hu) = (w as usize, h as usize);
    let (hole_rect, holed) = crate::select_ml_cmds::coverage_bounds(hole, area);
    if holed == 0 {
        return Ok(None);
    }
    let grow = remove_grow(hole_rect).min(MAX_GROW);
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
    // The picture as it stands, as a layer the fill sees as the composite.
    let mut work = Layer::raster("Split work".to_string(), doc.pixel_format());
    write_rgba8(&mut work, &Rgba8::new(w, h, picture.to_vec()).map_err(gen_err)?, area)?;
    let work = doc.insert_above(below, work);
    let mut ms = 0u64;
    let mut outcome = None;
    for attempt in 0..FILL_TRIES {
        let span = hi - lo;
        ctx.progress(lo + span * 0.8 * attempt as f32 / FILL_TRIES as f32, "");
        let try_box = attempt > 0 && box_job.is_some();
        let mut try_job = if try_box { box_job.cloned().unwrap_or_else(|| removal.clone()) } else { removal.clone() };
        try_job.seed = try_job.seed.wrapping_add(u64::from(attempt) * 1000);
        if !try_box {
            try_job.grow = grow;
        }
        let coverage = embed_coverage(area, if try_box { &box_hole } else { hole }, rect);
        *active = Some(work);
        let result = fill_region(doc, active, ctx, try_job, rect, coverage, None);
        let (made, ..) = match result {
            Ok(r) => r,
            Err(e) => {
                doc.remove(work);
                *active = below;
                return Err(e);
            }
        };
        ctx.check().inspect_err(|_| {
            for (id, ..) in &made {
                doc.remove(*id);
            }
            doc.remove(work);
            *active = below;
        })?;
        ms += made.iter().map(|m| m.3).sum::<u64>();
        // The picture with the fill over the hole, whole.
        let filled = photocraft_compose::render(doc, area).to_rgba8().pixels;
        let fill_seed = made.first().map_or(0, |m| m.1);
        for (id, ..) in &made {
            doc.remove(*id);
        }
        // The result over the picture along `over`. A box fill's tone is matched to the picture
        // where both show the same thing (in the box, outside the hole): a fill a few levels
        // darker left the person's outline as a shadow on the curtain (measured in the app
        // 2026-11-03). The removal's result is tone matched by its own route.
        let tone = try_box.then(|| ToneGrid::new(picture, &filled, |k| box_hole[k] >= 0.5 && over[k] < 0.02, wu, hu));
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
        ctx.progress(lo + span * 0.8 * (attempt as f32 + 0.9) / FILL_TRIES as f32, "Checking the fill");
        match check(&background) {
            Ok(Some(seen)) => {
                outcome = Some((background, fill_seed, ms, seen));
                break;
            }
            Ok(None) => continue,
            Err(e) => {
                doc.remove(work);
                *active = below;
                return Err(e);
            }
        }
    }
    doc.remove(work);
    *active = below;
    Ok(outcome)
}

/// The subjects of the picture, found term by term, and what found them (the terms joined).
/// With `terms`: each in turn. Without: the people, then the animals, then the foreground
/// things the vision-language model names, then Select Subject's prompts in turn until one
/// finds anything. Each term gets what is left of the budget.
#[allow(clippy::too_many_arguments)]
fn find_subjects(
    finder: &SubjectFinder,
    thinker: &dyn GenerativeBackend,
    picture: &[u8],
    area: Rect,
    terms: Option<&[Term]>,
    budget: usize,
    seed: u64,
    ctx: &JobCtx,
) -> Result<(Vec<Subject>, String)> {
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
                for thing in things {
                    ctx.check()?;
                    let left = budget.saturating_sub(out.len());
                    if left == 0 {
                        break;
                    }
                    take(&thing, false, left.min(THINGS_EACH), &mut out, &mut by, ctx)?;
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
    edges: Edges,
    ctx: &JobCtx,
) -> Result<Edges> {
    let (w, h) = (area.width(), area.height());
    let instances: Vec<Instance> = subjects.iter().map(|s| (s.cov.clone(), s.bounds, s.pixels)).collect();
    let mut used = edges;
    let mut alphas = None;
    if matches!(edges, Edges::Auto | Edges::Vitmatte) {
        ctx.progress(0.08, "Matting hair");
        alphas = crate::select_ml_cmds::vitmatte_alphas(backend, picture, area, &instances, ctx)?;
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
/// ...and the most instances of each it is asked for.
const THINGS_EACH: usize = 2;
/// The detector's word for an animal.
const ANIMAL: &str = "animal";
/// Two instances with a mask IoU at or above this are one.
const DUPLICATE_IOU: f64 = 0.7;
/// A smaller instance with this share of its mask inside a larger one is a part of it.
const PART_INSIDE: f64 = 0.85;
/// `group: auto` joins two instances when the smaller overlaps the larger by this share.
const GROUP_OVERLAP: f64 = 0.5;
/// A detection after a fill that holds this share of a subject's mask is the subject seen
/// again (with what the fill completed).
const CONTINUES: f32 = 0.6;
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
/// membrane: the fill's tone matched to its surroundings with no step at their edge.
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
/// the rest). The detector's masks are hard-edged and clip hair (the owner, 2026-11-02); the
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
