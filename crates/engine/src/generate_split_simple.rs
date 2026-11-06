//! Split into Layers, the simple way (`method: "simple"`, the default since 2026-11-04):
//! progressive subject removal. The detector finds the subjects on the original photo (a mask
//! reaching across the whole frame is the background, not a subject); the nearest is cut onto
//! its own layer and taken out by the generative removal (as Remove would); the detector finds
//! the subjects still to come again on what is left (their masks now, or that they are gone),
//! never adding new ones; nearest first, until none is left or the layer budget is spent. What
//! remains is the background. A subject mostly inside a larger one (a face in an astronaut) is
//! a part of it: never removed on its own, it gets its own layer, above the whole's, when the
//! whole is cut, and the whole's removal takes it too.
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
    /// The refused fills, kept on request (`keepTries`): each named for its try and verdict.
    refused: Vec<(String, Vec<u8>)>,
    /// Where the result was laid into the picture (one value per pixel; empty when none was).
    pasted: Vec<f32>,
}

/// The share of a pasted pixel above which it counts as repainted.
const PASTED: f32 = 0.02;

/// A subject found inside another (its whole): never removed on its own, cut onto its own layer
/// when its whole is cut. `parent` is where its direct whole stands in the group (0: the whole
/// the group is removed as, k: the group's k-th part).
struct Part {
    sub: Subject,
    parent: usize,
}

/// A subject with at least this share of its mask inside a larger subject's mask is a part of
/// it: the cat's face, helmet, gloves and boots lie 99–100 % inside the astronaut, the rocket in
/// its hand 88 % (first-pass masks, measured 2026-11-05).
const PART_SHARE: f32 = 0.8;

pub(in crate::generate_cmds) fn run(s: &mut Session, plan: SubjectsPlan) -> Result<Value> {
    let SubjectsPlan { prompt, subjects, seed, name, edges, group, keep_tries, first_pass_only, detail, .. } = plan;
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
            let (pending, background_like): (Vec<Subject>, Vec<Subject>) = first.into_iter().partition(|f| !is_background(f, wu, hu));
            let dropped: Vec<Value> = background_like
                .iter()
                .map(|f| json!({"term": f.term, "bounds": [f.bounds.x0, f.bounds.y0, f.bounds.width(), f.bounds.height()], "pixels": f.pixels}))
                .collect();
            // Subjects under the detail's smallest share of the frame stay in the background
            // (`detail`: rough keeps the big ones only).
            let min_px = (detail.min_share() * (wu * hu) as f64).ceil() as u64;
            let (pending, small): (Vec<Subject>, Vec<Subject>) = pending.into_iter().partition(|f| f.pixels >= min_px);
            let small: Vec<Value> = small
                .iter()
                .map(|f| json!({"term": f.term, "bounds": [f.bounds.x0, f.bounds.y0, f.bounds.width(), f.bounds.height()], "pixels": f.pixels}))
                .collect();
            // Diagnostic (`firstPassOnly`, off by default): the first pass's subjects as layers of
            // the picture under their hard masks, nearest on top, over the picture unchanged; no
            // matting, no removal.
            if first_pass_only {
                let order = front_first(&pending, Order::Auto, wu, hu);
                let mut slots: Vec<Option<Subject>> = pending.into_iter().map(Some).collect();
                let ordered: Vec<Subject> = order.into_iter().filter_map(|i| slots.get_mut(i).and_then(Option::take)).collect();
                let cut: Vec<Vec<u8>> = ordered.iter().map(|s| under_alpha(&picture, &s.cov)).collect();
                let (ids, placed, incomplete, refused_ids) =
                    place_all(doc, active, area, base.as_deref(), &found_by, &template_id, 0, &ordered, cut, picture, &[], Vec::new())?;
                let detector = finder.detector_stats();
                return Ok((
                    ids,
                    placed,
                    incomplete,
                    Vec::new(),
                    found_by,
                    template_id,
                    0,
                    (0, 0),
                    started.elapsed().as_millis() as u64,
                    (w, h),
                    "hard",
                    Vec::new(),
                    detector,
                    (0, dropped, small, Vec::new(), refused_ids, Vec::new(), json!({"found": [], "layers": []})),
                ));
            }
            // The parts (a subject mostly inside a larger one) leave the list: each is cut with its
            // whole and never removed on its own.
            let (mut pending, mut pending_parts, part_records) = split_parts(pending);
            // Rough: a part stays in its whole's layer (and goes with it).
            if !detail.part_layers() {
                pending_parts.iter_mut().for_each(Vec::clear);
            }
            // Each part's layer and its direct whole's, as indices into `taken`.
            let mut part_layers: Vec<(usize, usize)> = Vec::new();
            // Subjects the detector no longer found before their turn (painted over by a nearer
            // subject's removal): no turn of their own, a layer cut from the picture before that
            // removal (`layered`) while the budget lasts.
            let mut lost: Vec<Value> = Vec::new();
            // Where accepted removals repainted the picture so far, and every mask swap (or refused one).
            let mut repainted = vec![false; wu * hu];
            let mut swaps: Vec<Value> = Vec::new();
            // The picture as it stands: each removal takes the nearest subject out of it.
            let mut p = picture;
            let mut taken: Vec<Subject> = Vec::new();
            let mut cut: Vec<Vec<u8>> = Vec::new();
            let mut left_in: Vec<usize> = Vec::new();
            let mut tries: Vec<Value> = Vec::new();
            // The refused fills, kept on request (`keepTries`): subject index, name, pixels.
            let mut refused: Vec<(usize, String, Vec<u8>)> = Vec::new();
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
                if next >= pending.len() || next >= pending_parts.len() {
                    break;
                }
                let front = pending.remove(next);
                let mut parts = pending_parts.remove(next);
                let done = taken.len();
                let (lo, hi) = (0.1 + 0.85 * done as f32 / budget as f32, 0.1 + 0.85 * (done + 1) as f32 / budget as f32);
                ctx.progress(lo, &format!("Removing {} ({} of at most {budget})", title_case(&front.term), done + 1));
                // Its parts count toward the layers: the largest first (a part's whole before it),
                // as many as the layers left take; the rest stay in the whole's layer.
                let over_budget = parts.split_off(parts.len().min(budget.saturating_sub(done + 1)));
                let parents: Vec<usize> = std::iter::once(0).chain(parts.iter().map(|q| q.parent)).collect();
                let mut group: Vec<Subject> = std::iter::once(front).chain(parts.into_iter().map(|q| q.sub)).collect();
                for g in &mut group {
                    g.asked = asked;
                }
                used = matte_all(backend.as_ref(), matte.as_ref(), doc, *active, &p, area, &mut group, None, edges, ctx)?;
                // The hole: the whole and every part (none pokes out and stays in the background).
                let hole = group_hole(&group, &over_budget.iter().map(|q| &q.sub).collect::<Vec<_>>());
                // The layers: the picture now (every nearer subject gone) under each one's alpha,
                // less what its parts above it show; the parts on top, the smallest highest.
                let mut cuts = cut_group(&p, &group, &parents);
                let whole_at = done + group.len() - 1;
                for k in (1..group.len()).rev() {
                    part_layers.push((done + group.len() - 1 - k, done + group.len() - 1 - parents.get(k).copied().unwrap_or(0)));
                }
                let mut front = group.remove(0);
                while let Some(q) = group.pop() {
                    taken.push(q);
                    cut.push(cuts.pop().unwrap_or_default());
                }
                cut.push(cuts.pop().unwrap_or_default());
                // (`passes - 1`: the n-th removal's seeds, as before parts had layers.)
                let removed = remove_one(doc, active, below, ctx, area, &p, &hole, &front, &finder, &job, keep_tries, passes - 1, (lo, hi))?;
                refused.extend(removed.refused.into_iter().map(|(why, px)| (whole_at, why, px)));
                fill_ms += removed.ms;
                generations += removed.generations;
                tries.extend(removed.tries.into_iter().map(|mut t| {
                    t["subject"] = json!(whole_at);
                    t
                }));
                let cleared = removed.picture.is_some();
                for q in taken.iter_mut().skip(done) {
                    q.completed = cleared;
                }
                // The picture before this removal, while it is looked at again below.
                let mut before: Option<Vec<u8>> = None;
                match removed.picture {
                    Some(filled) => {
                        before = Some(std::mem::replace(&mut p, filled));
                        for (r, a) in repainted.iter_mut().zip(&removed.pasted) {
                            *r |= *a > PASTED;
                        }
                        fill_seed = removed.seed;
                        front.completed = true;
                    }
                    None => left_in.extend(done..=whole_at),
                }
                taken.push(front);
                ctx.check()?;
                // Look again, for the listed subjects still to come only: is each still there, and
                // what is its mask now (a removal may show more of someone it hid).
                if !pending.is_empty() && taken.len() < budget {
                    let mut gone = refind(&finder, &p, area, (&mut pending, &mut pending_parts), &mut lost, &repainted, &mut swaps, ctx)?;
                    // A subject no longer found was painted over by this removal: it is gone from
                    // the background, so it gets a layer of its own, cut from the picture as it
                    // stood before that removal, behind the one removed (the café's middle man
                    // vanished from the split altogether, no layer and not in the background,
                    // measured in the app 2026-11-06).
                    if !gone.is_empty() {
                        let source = before.as_deref().unwrap_or(&p);
                        used = matte_all(backend.as_ref(), matte.as_ref(), doc, *active, source, area, &mut gone, None, edges, ctx)?;
                        // `refind` named them last in `lost`, in this order.
                        let first = lost.len().saturating_sub(gone.len());
                        for (k, g) in gone.into_iter().enumerate() {
                            if taken.len() >= budget {
                                break;
                            }
                            if let Some(record) = lost.get_mut(first + k) {
                                record["layered"] = json!(true);
                            }
                            cut.push(under_alpha(source, &g.alpha));
                            taken.push(g);
                        }
                    }
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
            let refused: Vec<(String, Vec<u8>)> =
                refused.into_iter().map(|(i, why, px)| (format!("{}: {why}", labels.get(i).cloned().unwrap_or_default()), px)).collect();
            ctx.progress(0.95, "Placing");
            let detector = finder.detector_stats();
            let parts = json!({
                "found": part_records,
                "layers": part_layers.iter().map(|(q, w)| json!({"part": labels.get(*q), "whole": labels.get(*w)})).collect::<Vec<_>>(),
            });
            let (ids, placed, incomplete, refused_ids) =
                place_all(doc, active, area, base.as_deref(), &found_by, &template_id, fill_seed, &taken, cut, p, &left_in, refused)?;
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
                (passes, dropped, small, lost, refused_ids, swaps, parts),
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
            (passes, dropped, small, lost, refused_ids, swaps, parts),
        )| {
            json!({
                "method": "simple", "detail": detail.name(), "passes": passes, "droppedAsBackground": dropped, "tooSmall": small, "lost": lost, "swaps": swaps, "parts": parts,
                "layers": ids.iter().map(|l| l.0).collect::<Vec<_>>(), "count": ids.len(),
                "subjects": placed, "incomplete": incomplete, "fills": tries.len(), "generations": generations, "tries": tries, "order": "auto",
                "foundBy": found_by, "template": template_id, "seed": seed, "leftBehind": left_behind, "refusedLayers": refused_ids.iter().map(|l| l.0).collect::<Vec<_>>(),
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
/// the list, named in `lost`, and is returned (in the order it was named there). Nothing the
/// detector finds is ever added.
#[allow(clippy::too_many_arguments)]
fn refind(
    finder: &SubjectFinder,
    picture: &[u8],
    area: Rect,
    (pending, parts): (&mut Vec<Subject>, &mut Vec<Vec<Part>>),
    lost: &mut Vec<Value>,
    repainted: &[bool],
    swaps: &mut Vec<Value>,
    ctx: &JobCtx,
) -> Result<Vec<Subject>> {
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
                // The subject is found; its mask is swapped for the find's only by [`may_swap`].
                let b = |r: Rect| json!([r.x0, r.y0, r.width(), r.height()]);
                let failed = may_swap(&s.cov, cov, repainted);
                let mut record = json!({"term": s.term, "old": b(s.bounds), "oldPixels": s.pixels, "new": b(*bounds), "newPixels": *pixels});
                if failed.is_empty() {
                    let changed = s.bounds != *bounds || s.pixels != *pixels;
                    s.cov = cov.clone();
                    s.bounds = *bounds;
                    s.pixels = *pixels;
                    s.alpha = Vec::new();
                    record["swapped"] = json!(true);
                    if changed {
                        swaps.push(record);
                    }
                } else {
                    record["swapped"] = json!(false);
                    record["failed"] = json!(failed);
                    swaps.push(record);
                }
            }
            got.push(i);
            used.push(d);
        }
        for i in mine {
            if !got.contains(&i)
                && let (Some(k), Some(s)) = (keep.get_mut(i), pending.get(i))
            {
                *k = false;
                let n = parts.get(i).map_or(0, Vec::len);
                lost.push(json!({"term": s.term, "bounds": [s.bounds.x0, s.bounds.y0, s.bounds.width(), s.bounds.height()], "parts": n}));
            }
        }
    }
    // The lost ones in the order `lost` names them (kind by kind), out of the list.
    let mut order: Vec<usize> = Vec::new();
    let mut kinds_seen: Vec<String> = Vec::new();
    for s in pending.iter() {
        let k = s.check_term().to_string();
        if !kinds_seen.contains(&k) {
            kinds_seen.push(k);
        }
    }
    for kind in &kinds_seen {
        order.extend((0..pending.len()).filter(|i| !keep.get(*i).copied().unwrap_or(true) && pending[*i].check_term() == kind));
    }
    let mut slots: Vec<Option<Subject>> = std::mem::take(pending).into_iter().map(Some).collect();
    let gone: Vec<Subject> = order.iter().filter_map(|i| slots.get_mut(*i).and_then(Option::take)).collect();
    *pending = slots.into_iter().flatten().collect();
    let mut k = 0;
    parts.retain(|_| {
        k += 1;
        keep.get(k - 1).copied().unwrap_or(false)
    });
    Ok(gone)
}

/// The subjects split into wholes and their parts: a subject with at least [`PART_SHARE`] of its
/// mask inside a larger one's is a part of the smallest such (its direct whole), and goes with
/// the whole that whole is part of in turn (a face in a helmet in an astronaut goes with the
/// astronaut). Each whole's parts, largest first (a part's direct whole before it); and a record
/// of each part and its direct whole.
fn split_parts(found: Vec<Subject>) -> (Vec<Subject>, Vec<Vec<Part>>, Vec<Value>) {
    let n = found.len();
    let on: Vec<u64> = found.iter().map(|s| s.cov.iter().filter(|v| **v >= 0.5).count() as u64).collect();
    let whole_of: Vec<Option<usize>> = (0..n)
        .map(|i| {
            let mine = on.get(i).copied().unwrap_or(0);
            if mine == 0 {
                return None;
            }
            (0..n)
                .filter(|&j| j != i && on.get(j).is_some_and(|&o| o > mine))
                .filter(|&j| match (found.get(i), found.get(j)) {
                    (Some(a), Some(b)) => overlap(&a.cov, &b.cov) as f32 >= PART_SHARE * mine as f32,
                    _ => false,
                })
                .min_by_key(|&j| (on.get(j).copied().unwrap_or(0), j))
        })
        .collect();
    // Wholes are strictly larger, so the chain ends; bounded all the same.
    let top = |mut i: usize| -> usize {
        for _ in 0..=n {
            match whole_of.get(i).copied().flatten() {
                Some(j) => i = j,
                None => break,
            }
        }
        i
    };
    let b = |r: Rect| json!([r.x0, r.y0, r.width(), r.height()]);
    let records: Vec<Value> = (0..n)
        .filter_map(|i| {
            let (part, whole) = (found.get(i)?, found.get(whole_of.get(i).copied().flatten()?)?);
            Some(json!({"term": part.term, "bounds": b(part.bounds), "pixels": part.pixels, "whole": {"term": whole.term, "bounds": b(whole.bounds)}}))
        })
        .collect();
    let mut slots: Vec<Option<Subject>> = found.into_iter().map(Some).collect();
    let (mut wholes, mut parts) = (Vec::new(), Vec::new());
    for t in (0..n).filter(|i| whole_of.get(*i).copied().flatten().is_none()) {
        let mut members: Vec<usize> = (0..n).filter(|&i| i != t && whole_of.get(i).copied().flatten().is_some() && top(i) == t).collect();
        members.sort_by_key(|&i| (std::cmp::Reverse(on.get(i).copied().unwrap_or(0)), i));
        let at = |i: usize| -> usize { members.iter().position(|m| *m == i).map_or(0, |p| p + 1) };
        let group: Vec<Part> = members
            .iter()
            .filter_map(|&m| {
                let parent = whole_of.get(m).copied().flatten().map_or(0, at);
                slots.get_mut(m).and_then(Option::take).map(|sub| Part { sub, parent })
            })
            .collect();
        if let Some(w) = slots.get_mut(t).and_then(Option::take) {
            wholes.push(w);
            parts.push(group);
        }
    }
    (wholes, parts, records)
}

/// The hole of a whole and its parts: wherever any of `group` (matted) or the `unmatted` parts
/// shows.
fn group_hole(group: &[Subject], unmatted: &[&Subject]) -> Vec<f32> {
    let mut hole: Vec<f32> = Vec::new();
    for s in group.iter().chain(unmatted.iter().copied()) {
        let h = hole_of(s);
        if hole.is_empty() {
            hole = h;
        } else {
            for (d, c) in hole.iter_mut().zip(h) {
                *d = d.max(c);
            }
        }
    }
    hole
}

/// Each of `group` (a whole, then its parts) cut from `picture` under its alpha, less what its
/// direct parts (`parents[k]`: the group index of part k's direct whole) show over it, so the
/// stack composites as the whole did: a layer of alpha a under parts showing u keeps
/// (a − u) / (1 − u).
fn cut_group(picture: &[u8], group: &[Subject], parents: &[usize]) -> Vec<Vec<u8>> {
    group
        .iter()
        .enumerate()
        .map(|(k, s)| {
            // What the direct parts above it leave showing, Π (1 − their alpha).
            let mut through = vec![1.0f32; s.alpha.len()];
            for (c, part) in group.iter().enumerate().skip(1) {
                if c != k && parents.get(c) == Some(&k) {
                    for (q, a) in through.iter_mut().zip(&part.alpha) {
                        *q *= 1.0 - a.clamp(0.0, 1.0);
                    }
                }
            }
            let alpha: Vec<f32> =
                s.alpha.iter().zip(&through).map(|(a, q)| if *q <= 1e-4 { 0.0 } else { ((a.clamp(0.0, 1.0) - (1.0 - q)) / q).clamp(0.0, 1.0) }).collect();
            under_alpha(picture, &alpha)
        })
        .collect()
}

/// Whether a re-found detection (`new`) may replace a pending subject's mask (`old`): the rules it
/// fails, none when it may. (a) It covers at least [`SWAP_COVERS`] of the subject's mask; (b) of
/// its pixels outside that mask, at least [`SWAP_IN_REPAINTED`] lie where accepted removals of this
/// split repainted (`repainted`): a subject grows only into what a nearer one's removal showed of
/// it. On the cat picture the cat's face was swapped for a detection of the whole astronaut, and
/// the astronaut for its legs alone, by the looser same-subject match (measured 2026-11-05).
fn may_swap(old: &[f32], new: &[f32], repainted: &[bool]) -> Vec<&'static str> {
    let old_px = old.iter().filter(|v| **v >= 0.5).count() as u64;
    let mut failed = Vec::new();
    if old_px == 0 || (overlap(old, new) as f32) < SWAP_COVERS * old_px as f32 {
        failed.push("covers less than 80 % of the subject");
    }
    let (mut outside, mut in_repainted) = (0u64, 0u64);
    for (k, (o, n)) in old.iter().zip(new).enumerate() {
        if *n >= 0.5 && *o < 0.5 {
            outside += 1;
            if repainted.get(k).copied().unwrap_or(false) {
                in_repainted += 1;
            }
        }
    }
    if outside > 0 && (in_repainted as f32) < SWAP_IN_REPAINTED * outside as f32 {
        failed.push("grows outside the repainted areas");
    }
    failed
}

const SWAP_COVERS: f32 = 0.8;
const SWAP_IN_REPAINTED: f32 = 0.9;

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
    keep: bool,
    nth: usize,
    (lo, hi): (f32, f32),
) -> Result<Removed> {
    let (w, h) = (area.width(), area.height());
    let (wu, hu) = (w as usize, h as usize);
    let mut out = Removed { picture: None, seed: 0, tries: Vec::new(), ms: 0, generations: 0, refused: Vec::new(), pasted: Vec::new() };
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
            // A fill that left the hole blank is no removal, whoever the detector finds in it.
            if let Some(why) = blank_refusal(picture, &laid, hole, &ring, wu, hu) {
                out.tries.push(json!({"fill": kind, "generations": made, "verdict": why}));
                if keep {
                    out.refused.push((format!("{kind} refused ({why})"), laid));
                }
                ctx.check()?;
                continue;
            }
            let still = finder.find_in(&laid, area, &term, NEW_AT_MOST, ctx)?.into_iter().find(|(cov, ..)| still_there(cov, hole, &grown));
            match still {
                None => {
                    out.tries.push(json!({"fill": kind, "generations": made, "verdict": "accepted"}));
                    out.picture = Some(laid);
                    out.pasted = over.clone();
                    out.seed = seed;
                    return Ok(());
                }
                Some((_, b, _)) => {
                    let why = format!("the {} still there", bare(&term));
                    out.tries.push(json!({"fill": kind, "generations": made, "verdict": why, "at": [b.x0, b.y0, b.width(), b.height()]}));
                    if keep {
                        out.refused.push((format!("{kind} refused ({why})"), laid));
                    }
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

/// Why a removal's result is refused by its pixels: only when it left the hole blank
/// ([`blank_or_ghost`]'s "left … blank" verdicts). A dark or light shape over the hole is not
/// refused here: the tree trunk a removal continued through the yellow boy's hole on the kids
/// photo was refused three times as a dark shape and he stayed in the background (the owner's
/// call, 2026-11-05). The classic method still refuses both.
fn blank_refusal(before: &[u8], after: &[u8], hole: &[f32], ring: &[f32], w: usize, h: usize) -> Option<String> {
    blank_or_ghost(before, after, hole, ring, w, h).filter(|why| why.starts_with("left ") && why.contains("blank"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 200×200 mask of the rectangle x0..x1 × y0..y1.
    fn rect_mask(x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<f32> {
        (0..200 * 200).map(|k| if (x0..x1).contains(&(k % 200)) && (y0..y1).contains(&(k / 200)) { 1.0 } else { 0.0 }).collect()
    }

    #[test]
    fn a_small_subject_keeps_its_mask_next_to_a_much_larger_detection() {
        // The cat's face (20×20) re-found as the whole astronaut (160×180) after the rocket was
        // removed elsewhere: the find covers the face but grows far outside what was repainted.
        let face = rect_mask(90, 30, 110, 50);
        let astronaut = rect_mask(20, 10, 180, 190);
        let repainted: Vec<bool> = rect_mask(0, 0, 20, 60).iter().map(|v| *v > 0.5).collect();
        assert_eq!(may_swap(&face, &astronaut, &repainted), vec!["grows outside the repainted areas"]);
    }

    #[test]
    fn a_big_subject_keeps_its_full_mask_when_found_again_as_its_lower_part() {
        // The astronaut (160×180) re-found as its legs only (160×60): covers a third of it.
        let astronaut = rect_mask(20, 10, 180, 190);
        let legs = rect_mask(20, 130, 180, 190);
        assert_eq!(may_swap(&astronaut, &legs, &vec![false; 200 * 200]), vec!["covers less than 80 % of the subject"]);
    }

    #[test]
    fn a_half_hidden_subject_grows_into_the_hole_a_removal_repainted() {
        // Someone seen only on the left (x 40..100) behind a nearer one at x 100..160, whose
        // removal repainted x 96..164: found again whole (x 40..160), it takes the new mask.
        let visible = rect_mask(40, 40, 100, 160);
        let whole = rect_mask(40, 40, 160, 160);
        let repainted: Vec<bool> = rect_mask(96, 30, 164, 170).iter().map(|v| *v > 0.5).collect();
        assert!(may_swap(&visible, &whole, &repainted).is_empty());
        // The same find with nothing repainted there grows into the picture itself: refused.
        assert_eq!(may_swap(&visible, &whole, &vec![false; 200 * 200]), vec!["grows outside the repainted areas"]);
    }

    #[test]
    fn a_dark_shape_in_the_hole_is_accepted_and_a_blank_fill_refused() {
        // A 64×64 picture of 4-px stripes; the hole is the middle 24×24, the ring the 12 px
        // around it (the scene of `a_blank_fill_and_a_smear_are_no_fill`).
        let (w, h) = (64usize, 64usize);
        let stripes = |k: usize| -> [u8; 4] { if (k / w / 4).is_multiple_of(2) { [60u8, 60, 60, 255] } else { [180, 180, 180, 255] } };
        let before: Vec<u8> = (0..w * h).flat_map(stripes).collect();
        let in_hole = |k: usize| (20..44).contains(&(k % w)) && (20..44).contains(&(k / w));
        let in_ring = |k: usize| !in_hole(k) && (8..56).contains(&(k % w)) && (8..56).contains(&(k / w));
        let hole: Vec<f32> = (0..w * h).map(|k| if in_hole(k) { 1.0 } else { 0.0 }).collect();
        let ring: Vec<f32> = (0..w * h).map(|k| if in_ring(k) { 1.0 } else { 0.0 }).collect();
        let with = |paint: &dyn Fn(usize) -> [u8; 4]| -> Vec<u8> { (0..w * h).flat_map(|k| if in_hole(k) { paint(k) } else { stripes(k) }).collect() };
        // A dark textured shape through the hole (like the continued trunk): the classic check
        // refuses it, the simple method accepts it.
        let dark = with(&|k| if (k / w).is_multiple_of(2) { [10u8, 10, 10, 255] } else { [40, 40, 40, 255] });
        assert!(blank_or_ghost(&before, &dark, &hole, &ring, w, h).is_some_and(|v| v.starts_with("painted a dark shape")));
        assert_eq!(blank_refusal(&before, &dark, &hole, &ring, w, h), None, "a dark shape is not refused");
        // A flat grey where the picture has stripes is still refused.
        let blank = with(&|_| [128u8, 128, 128, 255]);
        assert!(blank_refusal(&before, &blank, &hole, &ring, w, h).is_some_and(|v| v.starts_with("left the hole blank")));
        // A fill that continues the stripes passes both.
        assert_eq!(blank_refusal(&before, &with(&stripes), &hole, &ring, w, h), None);
    }

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

    /// A subject over x0..x1 × y0..y1 of a 200×200 frame, its alpha its hard mask.
    fn masked(x0: usize, y0: usize, x1: usize, y1: usize) -> Subject {
        let cov = rect_mask(x0, y0, x1, y1);
        let mut s = subject(x0 as i32, y0 as i32, (x1 - x0) as i32, (y1 - y0) as i32);
        s.alpha = cov.clone();
        s.cov = cov;
        s
    }

    #[test]
    fn subjects_inside_a_larger_one_are_its_parts_nested_by_their_smallest_whole() {
        // The cat picture's shape: a face in a helmet in an astronaut, a glove 85 % inside him,
        // and a rocket beside him (a whole of its own).
        let astronaut = masked(20, 10, 180, 190);
        let helmet = masked(60, 10, 140, 70);
        let face = masked(80, 20, 120, 50);
        let glove = masked(14, 100, 54, 140); // 34 of its 40 columns inside: 85 %
        let rocket = masked(150, 150, 199, 199); // 30 of 49 columns, 40 of 49 rows inside: 50 %
        let (wholes, parts, records) = split_parts(vec![face, astronaut, rocket, helmet, glove]);
        assert_eq!(wholes.len(), 2, "the astronaut and the rocket");
        assert_eq!((wholes[0].bounds.x0, wholes[1].bounds.x0), (20, 150));
        let mine: Vec<(i32, usize)> = parts[0].iter().map(|p| (p.sub.bounds.x0, p.parent)).collect();
        // Largest first: the helmet (whole: the astronaut, 0), the glove (0), the face (whole:
        // the helmet, the group's 1st part).
        assert_eq!(mine, vec![(60, 0), (14, 0), (80, 1)]);
        assert!(parts[1].is_empty());
        assert_eq!(records.len(), 3, "{records:?}");
        assert!(records.iter().any(|r| r["bounds"][0] == 80 && r["whole"]["bounds"][0] == 60), "{records:?}");
    }

    #[test]
    fn a_part_is_cut_out_of_its_whole_and_the_stack_composites_as_the_whole() {
        // A whole with a soft edge band, and a part inside it with a soft edge.
        let mut whole = masked(20, 20, 180, 180);
        let mut part = masked(60, 60, 100, 100);
        for (k, a) in part.alpha.iter_mut().enumerate() {
            if (k % 200 == 60 || k % 200 == 99) && *a > 0.0 {
                *a = 0.5;
            }
        }
        whole.alpha[100 * 200 + 20] = 0.4;
        let picture: Vec<u8> = (0..200 * 200).flat_map(|k| [(k % 251) as u8, 90, 200, 255]).collect();
        let cuts = cut_group(&picture, &[whole, part], &[0, 0]);
        let a = |c: &Vec<u8>, x: usize, y: usize| f32::from(c[(y * 200 + x) * 4 + 3]) / 255.0;
        assert_eq!(a(&cuts[0], 80, 80), 0.0, "a gap in the whole where the part is");
        assert_eq!(a(&cuts[1], 80, 80), 1.0, "the part whole on its layer");
        assert_eq!(a(&cuts[0], 150, 150), 1.0, "the rest of the whole stays");
        assert!((a(&cuts[0], 20, 100) - 0.4).abs() < 0.01, "the whole's own soft edge stays");
        // Over the part's soft edge the part (0.5) over the whole (1.0 there) is opaque again:
        // no seam.
        let (pa, wa) = (a(&cuts[1], 60, 80), a(&cuts[0], 60, 80));
        assert!((pa + wa * (1.0 - pa) - 1.0).abs() < 0.01, "part {pa} over whole {wa}");
        assert_eq!(cuts[0][(80 * 200 + 150) * 4], picture[(80 * 200 + 150) * 4], "the picture's own pixels");
    }

    #[test]
    fn a_part_that_sticks_out_of_its_whole_is_in_the_hole() {
        // A part 85 % inside the whole, its top 15 % above it: the hole takes all of it, whether
        // it was matted (a layer of its own) or not (no layer left for it).
        let whole = masked(40, 60, 160, 180);
        let part = masked(80, 40, 120, 160); // rows 40..60 of 40..160 outside: 83 % inside
        for hole in [group_hole(&[masked(40, 60, 160, 180), part], &[]), group_hole(std::slice::from_ref(&whole), &[&masked(80, 40, 120, 160)])] {
            assert_eq!(hole[45 * 200 + 100], 1.0, "the part's top, outside the whole");
            assert_eq!(hole[100 * 200 + 50], 1.0, "the whole");
            assert_eq!(hole[45 * 200 + 50], 0.0, "nothing else");
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
