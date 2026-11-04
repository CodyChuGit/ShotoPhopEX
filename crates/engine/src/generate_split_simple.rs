//! Split into Layers, the simple way (`method: "simple"`, the default since 2026-11-04):
//! progressive subject removal. The detector finds the subjects on the original photo (a mask
//! reaching across the whole frame is the background, not a subject); the nearest is cut onto
//! its own layer and taken out by the generative removal (as Remove would); the detector finds
//! the subjects still to come again on what is left (their masks now, or that they are gone),
//! never adding new ones; nearest first, until none is left or the layer budget is spent. What
//! remains is the background.
//!
//! Each subject: the plain removal, tried again with the next seed while the detector still
//! finds the subject in its hole or the fill left the hole blank or smeared, [`TRIES`] tries at
//! most; no grey-box fill, no strips. A subject no try clears stays in the
//! background and is named in `incomplete`. No patch, no rules about who a fill may paint. Once
//! the layers left would not take every person still to come, the people go first. The
//! rule-heavy peel
//! (`method: "classic"`) went eleven rounds on the café photo without the backgrounds getting
//! better (the owner's call, 2026-11-04).

use std::sync::Arc;

use photocraft_doc::{Document, LayerId};
use photocraft_geom::Rect;

use super::{
    DEFAULT_MATTE_TEMPLATE, FillJob, JobCtx, Layer, MATTE_DEFAULT_PROMPT, MAX_GROW, MAX_SUBJECTS, NEW_AT_MOST, Order, PEEL_VARIATIONS, PERSON, REMOVE_BG,
    REMOVE_MIN_REQUEST, RING_WIDTH, Result, Rgba8, Session, Subject, SubjectFinder, SubjectsPlan, Task, Term, Tuning, Value, at_least, bare, blank_or_ghost,
    box_blur, could_not_clear, creature, dilate, embed_coverage, fill_rect, find_subjects, front_first, gen_err, hole_of, json, kept_terms, matte_all,
    one_fill, overlap, parse_terms, place_all, plan_common, removal_job, removal_prompt, remove_grow, same_subject, subjects_labels, title_case, under_alpha,
    write_rgba8,
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
    let job = removal_job(s, "Split into Layers", 0, false, false, PEEL_VARIATIONS, seed, true, tuning)?;
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
            let asked = terms.is_some();
            // The subjects: what the detector found on the ORIGINAL photo, less any mask that is
            // the background (its bounds reach across nearly the whole frame both ways). Later
            // looks only find these again; they never add a subject from the painted picture (a
            // mask over nearly all of the woman photo, found once she was gone, was removed as a
            // subject and the truck and road became an invented white room, measured in the app
            // 2026-11-04).
            let (mut pending, background_like): (Vec<Subject>, Vec<Subject>) = first.into_iter().partition(|f| !is_background(f, wu, hu));
            let dropped: Vec<Value> = background_like
                .iter()
                .map(|f| json!({"term": f.term, "bounds": [f.bounds.x0, f.bounds.y0, f.bounds.width(), f.bounds.height()], "pixels": f.pixels}))
                .collect();
            // Subjects the detector no longer found before their turn (painted over by a nearer
            // subject's removal): no layer of their own.
            let mut lost: Vec<Value> = Vec::new();
            // The picture as it stands: each removal takes the nearest subject out of it.
            let mut p = picture;
            let mut taken: Vec<Subject> = Vec::new();
            let mut cut: Vec<Vec<u8>> = Vec::new();
            let mut left_in: Vec<usize> = Vec::new();
            let mut tries: Vec<Value> = Vec::new();
            let (mut fill_ms, mut generations, mut fill_seed) = (0u64, 0u32, 0u64);
            let mut used = edges;
            let mut passes = 0usize;
            loop {
                if pending.is_empty() || taken.len() >= budget || passes >= budget + EXTRA_PASSES {
                    break;
                }
                passes += 1;
                // The nearest of the subjects still to come; the people (and animals) first once
                // the layers left would not take them all (the café photo's cups and cards, nearer
                // than the people behind the table, took the budget before the people, measured
                // in the app 2026-11-04).
                let order = front_first(&pending, Order::Auto, wu, hu);
                let creatures = pending.iter().filter(|f| creature(f.check_term())).count();
                let people_first = creatures > 0 && budget.saturating_sub(taken.len()) <= creatures;
                let Some(next) = order.iter().copied().find(|i| !people_first || pending.get(*i).is_some_and(|f| creature(f.check_term()))) else {
                    break;
                };
                if next >= pending.len() {
                    break;
                }
                let mut front = pending.remove(next);
                let done = taken.len();
                let (lo, hi) = (0.1 + 0.85 * done as f32 / budget as f32, 0.1 + 0.85 * (done + 1) as f32 / budget as f32);
                ctx.progress(lo, &format!("Removing {} ({} of at most {budget})", title_case(&front.term), done + 1));
                front.asked = asked;
                used = matte_all(backend.as_ref(), matte.as_ref(), doc, *active, &p, area, std::slice::from_mut(&mut front), None, edges, ctx)?;
                let hole = hole_of(&front);
                // Its layer: the picture now (every nearer subject gone) under its alpha.
                cut.push(under_alpha(&p, &front.alpha));
                let removed = remove_one(doc, active, below, ctx, area, &p, &hole, &front, &finder, &job, done, (lo, hi))?;
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
                    None => left_in.push(done),
                }
                taken.push(front);
                ctx.check()?;
                // Look again, for the listed subjects still to come only: is each still there, and
                // what is its mask now (a removal may show more of someone it hid).
                if !pending.is_empty() && taken.len() < budget {
                    refind(&finder, &p, area, &mut pending, &mut lost, ctx)?;
                }
            }
            if left_in.len() == taken.len() {
                let name = taken.first().map_or_else(|| PERSON.to_string(), Subject::removal_name);
                return Err(could_not_clear(&name, &tries));
            }
            // The listed subjects the budget did not reach, still in the background.
            let left_behind: Vec<Value> = pending.iter().map(|f| json!([f.bounds.x0, f.bounds.y0, f.bounds.width(), f.bounds.height()])).collect();
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
                (passes, dropped, lost),
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
            (passes, dropped, lost),
        )| {
            json!({
                "method": "simple", "passes": passes, "droppedAsBackground": dropped, "lost": lost,
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

/// A mask whose bounds reach across at least this share of the frame's width AND of its height
/// is the background, not a subject: the woman photo's "animal" found once she was gone spanned
/// 832×1248 of 832×1248; the subjects that must stay span at most 100 % × 94.5 % (images.jpg's
/// woman), 93 % × 79 % (the woman at the truck) and 77 % × 87 % (the cat astronaut).
const BACKGROUND_SPAN: f32 = 0.98;

/// Whether a found mask is the background ([`BACKGROUND_SPAN`] both ways).
fn is_background(s: &Subject, w: usize, h: usize) -> bool {
    w > 0 && h > 0 && s.bounds.width() as f32 >= BACKGROUND_SPAN * w as f32 && s.bounds.height() as f32 >= BACKGROUND_SPAN * h as f32
}

/// Each subject still to come (`pending`), found again in `picture` as it stands: the detector is
/// asked for its kind, and the find that is the same subject ([`same_subject`]: either holds half
/// of the other; the best overlap first, each find matched once) becomes its mask now. A subject
/// with no such find is gone from the picture (a nearer one's removal painted over it): it leaves
/// the list, named in `lost`. Nothing the detector finds is ever added.
fn refind(finder: &SubjectFinder, picture: &[u8], area: Rect, pending: &mut Vec<Subject>, lost: &mut Vec<Value>, ctx: &JobCtx) -> Result<()> {
    let mut kinds: Vec<String> = Vec::new();
    for s in pending.iter() {
        let k = s.check_term().to_string();
        if !kinds.contains(&k) {
            kinds.push(k);
        }
    }
    let mut keep = vec![true; pending.len()];
    for kind in kinds {
        ctx.check()?;
        let mine: Vec<usize> = (0..pending.len()).filter(|i| pending[*i].check_term() == kind).collect();
        let finds = finder.find_in(picture, area, &kind, (mine.len() + REFIND_SPARE).min(NEW_AT_MOST.max(mine.len())), ctx)?;
        let mut pairs: Vec<(u64, usize, usize)> = Vec::new();
        for &i in &mine {
            for (d, (cov, _, pixels)) in finds.iter().enumerate() {
                if same_subject(&pending[i].cov, pending[i].pixels, cov, *pixels) {
                    pairs.push((overlap(&pending[i].cov, cov), i, d));
                }
            }
        }
        pairs.sort_by_key(|p| std::cmp::Reverse(p.0));
        let (mut got, mut used) = (Vec::new(), Vec::new());
        for (_, i, d) in pairs {
            if got.contains(&i) || used.contains(&d) {
                continue;
            }
            if let (Some(s), Some((cov, bounds, pixels))) = (pending.get_mut(i), finds.get(d)) {
                s.cov = cov.clone();
                s.bounds = *bounds;
                s.pixels = *pixels;
                s.alpha = Vec::new();
            }
            got.push(i);
            used.push(d);
        }
        for i in mine {
            if !got.contains(&i)
                && let (Some(k), Some(s)) = (keep.get_mut(i), pending.get(i))
            {
                *k = false;
                lost.push(json!({"term": s.term, "bounds": [s.bounds.x0, s.bounds.y0, s.bounds.width(), s.bounds.height()]}));
            }
        }
    }
    let mut k = 0;
    pending.retain(|_| {
        k += 1;
        keep.get(k - 1).copied().unwrap_or(false)
    });
    Ok(())
}

/// Finds asked for beyond the subjects of a kind still to come, so each can be matched.
const REFIND_SPARE: usize = 3;

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
    // The ring just outside the hole: what a fill is held against for blanks and smears.
    let ring: Vec<f32> = dilate(hole, wu, hu, RING_WIDTH).iter().zip(hole).map(|(d, h)| if *d >= 0.5 && *h < 0.5 { 1.0 } else { 0.0 }).collect();
    let tried = (|| -> Result<()> {
        for attempt in 0..TRIES {
            ctx.progress(lo + (hi - lo) * attempt as f32 / TRIES as f32, "");
            // The plain removal every try, with the next seed: no grey box, no strips (the grey-box
            // fills painted the white walls and the invented rooms and people, measured in the
            // app 2026-11-04; the owner's call).
            let mut try_job = removal.clone();
            try_job.seed = job.seed.wrapping_add(nth as u64 * 101).wrapping_add(u64::from(attempt) * 1000);
            let kind = "removal";
            let span = (lo + (hi - lo) * attempt as f32 / TRIES as f32, lo + (hi - lo) * (attempt as f32 + 0.8) / TRIES as f32);
            *active = Some(work);
            let (filled, seed, ms, made) = one_fill(doc, active, ctx, area, try_job, rect, embed_coverage(area, hole, rect))?;
            out.ms += ms;
            out.generations += made;
            let laid: Vec<u8> = picture
                .as_chunks::<4>()
                .0
                .iter()
                .zip(filled.as_chunks::<4>().0)
                .zip(&over)
                .flat_map(|((p, f), a)| {
                    let a = a.clamp(0.0, 1.0);
                    std::array::from_fn::<u8, 4, _>(|c| (f32::from(p[c]) * (1.0 - a) + f32::from(f[c]) * a).round() as u8)
                })
                .collect();
            ctx.progress(span.1, "Checking the removal");
            // A fill that left the hole blank (the grey box handed back) or a smear where the
            // subject stood is no removal, whoever the detector finds in it.
            if let Some(why) = blank_or_ghost(picture, &laid, hole, &ring, wu, hu) {
                out.tries.push(json!({"fill": kind, "generations": made, "verdict": why}));
                ctx.check()?;
                continue;
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn subject(x: i32, y: i32, w: i32, h: i32) -> Subject {
        Subject {
            cov: Vec::new(),
            bounds: Rect::new(x, y, x + w, y + h),
            pixels: (w * h) as u64,
            term: "person".into(),
            asked: false,
            alpha: Vec::new(),
            completed: false,
            grown: false,
        }
    }

    #[test]
    fn a_mask_across_the_whole_frame_is_the_background() {
        // The woman photo's "animal" after she was gone: the whole 832×1248 frame.
        assert!(is_background(&subject(0, 0, 832, 1248), 832, 1248));
        // The subjects that must stay: images.jpg's woman (452×641 of 452×678), the woman at the
        // truck (773×988 of 832×1248), the cat astronaut (646×1081 of 832×1248).
        assert!(!is_background(&subject(0, 37, 452, 641), 452, 678));
        assert!(!is_background(&subject(59, 203, 773, 988), 832, 1248));
        assert!(!is_background(&subject(101, 107, 646, 1081), 832, 1248));
        // Across the whole width but not the height (a road, a table edge): a subject.
        assert!(!is_background(&subject(0, 600, 832, 400), 832, 1248));
        assert!(!is_background(&subject(0, 0, 0, 0), 0, 0), "no frame, no background");
    }
}
