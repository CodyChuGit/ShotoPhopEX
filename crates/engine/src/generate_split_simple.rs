//! Split into Layers, the simple way (`method: "simple"`, the default since 2026-11-04):
//! progressive subject removal. The detector looks at the picture, the nearest subject it finds
//! is cut onto its own layer and taken out by the generative removal (as Remove would), and the
//! detector looks again at what is left; nearest first, until it finds nothing more or the layer
//! budget is spent. What remains is the background.
//!
//! Each subject: the removal, then (while the detector still finds the subject in its hole) the
//! grey-box fill of the hole's bounds with the next seed, [`TRIES`] tries at most; a subject no
//! try clears stays in the background and is named in `incomplete`. No strips, no patch, no
//! rules about who a fill may paint: whatever the detector still finds is simply the next
//! subject. Once the layers left would not take every person the detector sees, the people go
//! first. The rule-heavy peel
//! (`method: "classic"`) went eleven rounds on the café photo without the backgrounds getting
//! better (the owner's call, 2026-11-04).

use std::sync::Arc;

use photocraft_doc::{Document, LayerId};
use photocraft_geom::Rect;

use super::{
    DEFAULT_MATTE_TEMPLATE, FillJob, Group, JobCtx, LEFT_IN_HOLE, Layer, MATTE_DEFAULT_PROMPT, MAX_GROW, MAX_SUBJECTS, NEW_AT_MOST, Order, PEEL_PROMPT,
    PEEL_VARIATIONS, PERSON, REMOVE_BG, REMOVE_MIN_REQUEST, Result, Rgba8, Session, Subject, SubjectFinder, SubjectsPlan, Task, Term, ToneGrid, Tuning, Value,
    at_least, background_job, bare, box_blur, clean, could_not_clear, creature, dilate, embed_coverage, fill_rect, find_subjects, gen_err, hole_of,
    in_depth_order, json, kept_terms, matte_all, mostly_in_hole, one_fill, overlap, parse_terms, place_all, plan_common, removal_job, removal_prompt,
    remove_grow, subjects_labels, title_case, under_alpha, write_rgba8,
};

/// How many times the removal is asked to take one subject out before it is left in.
const TRIES: u32 = 3;

/// Passes beyond the layer budget the loop may spend on subjects that could not be cleared
/// (each such pass makes a layer too, so this only bounds a run of failures).
const EXTRA_PASSES: usize = 4;

/// What one subject's removal came to: the picture without it (`None`: still there after every
/// try), the seed that cleared it, every try's record, the fills' time and their generations.
struct Removed {
    picture: Option<Vec<u8>>,
    seed: u64,
    tries: Vec<Value>,
    ms: u64,
    generations: u32,
}

pub(in crate::generate_cmds) fn run(s: &mut Session, plan: SubjectsPlan) -> Result<Value> {
    let SubjectsPlan { prompt, subjects, seed, name, edges, group, .. } = plan;
    let finder = crate::select_ml_cmds::subject_finder(s, prompt.as_deref())?;
    let terms: Option<Vec<Term>> = prompt.as_deref().map(parse_terms).filter(|t| !t.is_empty());
    let tuning = Tuning { fast_draft: s.prefs().integrations.fast_draft, ..Tuning::default() };
    let job = removal_job(s, "Split into Layers", 0, false, false, PEEL_VARIATIONS, seed, true, tuning.clone())?;
    // The grey-box fill, the second and third try of a subject the removal kept.
    let box_job = background_job(s, "Split into Layers", seed, tuning, PEEL_PROMPT).ok();
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
    let base = name;
    crate::jobs::edit_job(
        s,
        "Split into Layers",
        move |doc, active, ctx| {
            let started = std::time::Instant::now();
            let area = doc.bounds();
            let (w, h) = (area.width(), area.height());
            let (wu, hu) = (w as usize, h as usize);
            let picture = photocraft_compose::render(doc, area).to_rgba8().pixels;
            let template_id = job.template_id.clone();
            let backend = Arc::clone(&job.backend);
            let below = *active;
            ctx.progress(0.0, "Finding subjects");
            let (first, found_by) = find_subjects(&finder, job.thinker.as_ref(), &picture, area, terms.as_deref(), budget, job.seed, group, ctx)?;
            ctx.check()?;
            let found_by = kept_terms(&found_by, &first);
            // What each later look asks the detector for: the terms that found the subjects the
            // first time (the vision-language model names the things once).
            let looks: Vec<(String, usize)> = match terms.as_deref() {
                Some(t) => t.iter().map(|t| (t.name.clone(), t.max.unwrap_or(budget))).collect(),
                None => found_by.split(", ").filter(|t| !t.is_empty()).map(|t| (t.to_string(), budget)).collect(),
            };
            let asked = terms.is_some();
            // The picture as it stands: each removal takes the nearest subject out of it.
            let mut p = picture;
            let mut taken: Vec<Subject> = Vec::new();
            let mut cut: Vec<Vec<u8>> = Vec::new();
            let mut left_in: Vec<usize> = Vec::new();
            // The holes no try could clear: what the detector finds there again is not looked at.
            let mut stuck: Vec<Vec<f32>> = Vec::new();
            let mut tries: Vec<Value> = Vec::new();
            let (mut fill_ms, mut generations, mut fill_seed) = (0u64, 0u32, 0u64);
            let mut used = edges;
            let mut found = first;
            let mut passes = 0usize;
            loop {
                found.retain(|f| !stuck.iter().any(|hole| mostly_in_hole(&f.cov, hole, LEFT_IN_HOLE)));
                if found.is_empty() || taken.len() >= budget || passes >= budget + EXTRA_PASSES {
                    break;
                }
                passes += 1;
                // The nearest of what the detector sees now; the people (and animals) first once
                // the layers left would not take them all (the café photo's cups and cards, nearer
                // than the people behind the table, took the budget before the people, measured
                // in the app 2026-11-04).
                let creatures = found.iter().filter(|f| creature(f.check_term())).count();
                if creatures > 0 && budget.saturating_sub(taken.len()) <= creatures {
                    found.retain(|f| creature(f.check_term()));
                }
                let Some(mut front) = in_depth_order(found, Order::Auto, wu, hu).into_iter().next() else {
                    found = Vec::new();
                    break;
                };
                let done = taken.len();
                let (lo, hi) = (0.1 + 0.85 * done as f32 / budget as f32, 0.1 + 0.85 * (done + 1) as f32 / budget as f32);
                ctx.progress(lo, &format!("Removing {} ({} of at most {budget})", title_case(&front.term), done + 1));
                front.asked = asked;
                used = matte_all(backend.as_ref(), matte.as_ref(), doc, *active, &p, area, std::slice::from_mut(&mut front), None, edges, ctx)?;
                let hole = hole_of(&front);
                // Its layer: the picture now (every nearer subject gone) under its alpha.
                cut.push(under_alpha(&p, &front.alpha));
                let removed = remove_one(doc, active, below, ctx, area, &p, &hole, &front, &finder, &job, box_job.as_ref(), done, (lo, hi))?;
                fill_ms += removed.ms;
                generations += removed.generations;
                tries.extend(removed.tries.into_iter().map(|mut t| {
                    t["subject"] = json!(done);
                    t
                }));
                match removed.picture {
                    Some(filled) => {
                        p = filled;
                        fill_seed = removed.seed;
                        front.completed = true;
                    }
                    None => {
                        left_in.push(done);
                        stuck.push(dilate(&hole, wu, hu, 2));
                    }
                }
                taken.push(front);
                ctx.check()?;
                // Look again at what is left.
                let left = budget.saturating_sub(taken.len());
                found = if left == 0 { look(&finder, &p, area, &looks, 1, group, ctx)? } else { look(&finder, &p, area, &looks, left, group, ctx)? };
            }
            if left_in.len() == taken.len() {
                let name = taken.first().map_or_else(|| PERSON.to_string(), Subject::removal_name);
                return Err(could_not_clear(&name, &tries));
            }
            // Whoever the detector still finds once the budget is spent (or a run of failures
            // ended the loop), apart from the subjects left in.
            found.retain(|f| !stuck.iter().any(|hole| mostly_in_hole(&f.cov, hole, LEFT_IN_HOLE)));
            let left_behind: Vec<Value> = found.iter().map(|f| json!([f.bounds.x0, f.bounds.y0, f.bounds.width(), f.bounds.height()])).collect();
            // Each try names its subject by its layer's name.
            let labels = subjects_labels(&taken);
            for t in &mut tries {
                let i = t["subject"].as_u64().unwrap_or(0) as usize;
                t["behind"] = json!(labels.get(i).cloned().unwrap_or_default());
                if let Some(o) = t.as_object_mut() {
                    o.remove("subject");
                }
            }
            ctx.progress(0.95, "Placing");
            let detector = finder.detector_stats();
            let (ids, placed, incomplete, _) =
                place_all(doc, active, area, base.as_deref(), &found_by, &template_id, fill_seed, &taken, cut, p, &left_in, Vec::new())?;
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
                left_behind,
                detector,
                passes,
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
            left_behind,
            detector,
            passes,
        )| {
            json!({
                "method": "simple", "passes": passes,
                "layers": ids.iter().map(|l| l.0).collect::<Vec<_>>(), "count": ids.len(),
                "subjects": placed, "incomplete": incomplete, "fills": tries.len(), "generations": generations, "tries": tries, "order": "auto",
                "foundBy": found_by, "template": template_id, "seed": seed, "leftBehind": left_behind, "refusedLayers": [],
                "detector": detector, "already": [],
                "width": w, "height": h, "fillMs": fill_ms, "ms": total_ms, "edges": matted,
            })
        },
    )
}

/// Whether a detection after a removal is the removed subject still there (or painted back): it
/// lies mostly inside the hole (widened by the removal's grow) and covers half of it. Someone
/// behind whom the removal showed more of reaches well outside the hole and is the next subject.
fn still_there(cov: &[f32], hole: &[f32], grown: &[f32]) -> bool {
    let on = cov.iter().filter(|c| **c >= 0.5).count() as u64;
    let holed = hole.iter().filter(|c| **c >= 0.5).count() as u64;
    on > 0 && holed > 0 && overlap(cov, grown) as f32 >= STILL_INSIDE * on as f32 && overlap(cov, hole) as f32 >= STILL_COVERS * holed as f32
}

/// A detection this much inside the widened hole...
const STILL_INSIDE: f32 = 0.6;
/// ...and over this much of the hole is the subject still there.
const STILL_COVERS: f32 = 0.5;

/// The subjects the detector finds in `picture` for `looks` (each term at most its count and
/// `left` in all), cleaned as the first look's are.
fn look(finder: &SubjectFinder, picture: &[u8], area: Rect, looks: &[(String, usize)], left: usize, group: Group, ctx: &JobCtx) -> Result<Vec<Subject>> {
    let (wu, hu) = (area.width() as usize, area.height() as usize);
    let mut out: Vec<Subject> = Vec::new();
    for (term, most) in looks {
        ctx.check()?;
        for (cov, bounds, pixels) in finder.find_in(picture, area, term, left.min(*most).max(1), ctx)? {
            out.push(Subject { cov, bounds, pixels, term: term.clone(), asked: false, alpha: Vec::new(), completed: false, grown: false });
        }
    }
    clean(&mut out, wu, hu, group);
    Ok(out)
}

/// Take `sub` (its `hole`) out of `picture` with the named removal, the result laid back over the
/// hole widened by the removal's grow and feathered; tried again with the next seed while the
/// detector still finds a `sub`'s kind mostly in the hole.
#[allow(clippy::too_many_arguments)]
fn remove_one(
    doc: &mut Document,
    active: &mut Option<LayerId>,
    below: Option<LayerId>,
    ctx: &JobCtx,
    area: Rect,
    picture: &[u8],
    hole: &[f32],
    sub: &Subject,
    finder: &SubjectFinder,
    job: &FillJob,
    box_job: Option<&FillJob>,
    nth: usize,
    (lo, hi): (f32, f32),
) -> Result<Removed> {
    let (w, h) = (area.width(), area.height());
    let (wu, hu) = (w as usize, h as usize);
    let mut out = Removed { picture: None, seed: 0, tries: Vec::new(), ms: 0, generations: 0 };
    let (hole_rect, holed) = crate::select_ml_cmds::coverage_bounds(hole, area);
    if holed == 0 {
        out.picture = Some(picture.to_vec());
        return Ok(out);
    }
    let grow = remove_grow(hole_rect).min(MAX_GROW);
    let grown = dilate(hole, wu, hu, grow as usize);
    let rect = fill_rect(hole_rect.inflate(grow as i32).intersect(&area), job.tuning.remove_margin, area)?;
    let rect = at_least(rect, REMOVE_MIN_REQUEST, area);
    // Laid over the hole widened and feathered, so the subject's halo and shadow go with it.
    let feather = (grow as usize / 2).max(1);
    let over = box_blur(&box_blur(&dilate(hole, wu, hu, (grow as usize).max(1)), wu, hu, feather, true), wu, hu, feather, false);
    let name = sub.removal_name();
    let term = sub.check_term().to_string();
    let mut removal = job.clone();
    removal.rewritten = Some(removal_prompt(&name));
    removal.object = Some(name);
    removal.grow = grow;
    // The picture as it stands, as a layer the fill sees as the composite.
    let mut work = Layer::raster("Split work".to_string(), doc.pixel_format());
    write_rgba8(&mut work, &Rgba8::new(w, h, picture.to_vec()).map_err(gen_err)?, area)?;
    let work = doc.insert_above(below, work);
    // The grey-box fill's box: the hole's bounds widened by the grow. The model sees a grey box
    // there, not the subject; the edit removal, seeing a big subject, keeps it (both people of
    // the café photo stayed after three removals, measured in the app 2026-11-04).
    let boxed = hole_rect.inflate(grow as i32).intersect(&area);
    let box_hole: Vec<f32> = (0..wu * hu)
        .map(|k| {
            let (x, y) = (area.x0 + (k % wu) as i32, area.y0 + (k / wu) as i32);
            if x >= boxed.x0 && x < boxed.x1 && y >= boxed.y0 && y < boxed.y1 { 1.0 } else { 0.0 }
        })
        .collect();
    let tried = (|| -> Result<()> {
        for attempt in 0..TRIES {
            ctx.progress(lo + (hi - lo) * attempt as f32 / TRIES as f32, "");
            // The removal first; then the grey-box fill (the removal again without one).
            let boxed_try = attempt > 0 && box_job.is_some();
            let mut try_job = match box_job {
                Some(b) if boxed_try => b.clone(),
                _ => removal.clone(),
            };
            try_job.seed = job.seed.wrapping_add(nth as u64 * 101).wrapping_add(u64::from(attempt) * 1000);
            let kind = if boxed_try { "fill" } else { "removal" };
            *active = Some(work);
            let coverage = embed_coverage(area, if boxed_try { &box_hole } else { hole }, rect);
            let (filled, seed, ms, made) = one_fill(doc, active, ctx, area, try_job, rect, coverage)?;
            out.ms += ms;
            out.generations += made;
            // A box fill's tone is matched to the picture where both show the same thing (in the
            // box, outside the hole); the removal's is matched by its own route.
            let tone = boxed_try.then(|| ToneGrid::new(picture, &filled, |k| box_hole[k] >= 0.5 && over[k] < 0.02, wu, hu));
            let laid: Vec<u8> = picture
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
            ctx.progress(lo + (hi - lo) * (attempt as f32 + 0.8) / TRIES as f32, "Checking the removal");
            let still = finder.find_in(&laid, area, &term, NEW_AT_MOST, ctx)?.into_iter().find(|(cov, ..)| still_there(cov, hole, &grown));
            match still {
                None => {
                    out.tries.push(json!({"fill": kind, "generations": made, "verdict": "accepted"}));
                    out.picture = Some(laid);
                    out.seed = seed;
                    return Ok(());
                }
                Some((_, b, _)) => {
                    out.tries.push(json!({"fill": kind, "generations": made, "verdict": format!("the {} still there", bare(&term)), "at": [b.x0, b.y0, b.width(), b.height()]}));
                }
            }
            ctx.check()?;
        }
        Ok(())
    })();
    doc.remove(work);
    *active = below;
    tried?;
    Ok(out)
}
