//! Split into Layers' detection: finding the subjects term by term, naming lookalikes, cleaning
//! the detections and the matte that refines their alpha.

use std::collections::BTreeMap;

use photocraft_doc::{Document, LayerId};
use photocraft_genai::GenerativeBackend;
use photocraft_geom::Rect;

use crate::jobs::JobCtx;
use crate::select_ml_cmds::{Instance, SubjectFinder};
use crate::{EngineError, Result, generate_split_order as order};

use super::{
    ANIMAL, Common, DEFAULT_MATTE_TEMPLATE, DUPLICATE_IOU, Edges, FRAME_SHARE, GROUP_OVERLAP, Gray8, Group, JobProgress, MAX_THINGS, PART_INSIDE, PERSON,
    Request, Rgba8, SPECK_SHARE, Subject, THINGS_EACH, Term, WHOLE_FRAME, bare, crop_rgba, gen_err, gray8_at, list_things, matte_request_size, overlap,
    request_size, resize_rgba8, resolve_auto, sees,
};

/// The subjects of the picture, found term by term, and what found them (the terms joined).
/// With `terms`: each in turn. Without: the people, then the animals, then the foreground
/// things the vision-language model names (one of each, then more of each), then Select
/// Subject's prompts in turn until one finds anything. Each term gets what is left of the budget once the finds so far are cleaned
/// ([`clean`]): the budget counts layers, not detections (seven people came back as eleven, and
/// the table behind them was never asked for; measured in the app 2026-11-03).
#[allow(clippy::too_many_arguments)]
pub(super) fn find_subjects(
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
            name_lookalikes(thinker, picture, area, &mut out, seed, ctx)?;
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

/// Each person and animal the detector found, checked by the vision-language model on its own
/// crop: the detector's "person" and "animal" also fire on things of their shape (a duck was a
/// "Person" and a pear an "Animal" in the app, 2026-11-06 and 2026-11-12). One the model says
/// is no real person (animal) is renamed by what the model calls it ([`list_things`] on the
/// crop, which names things, not people or animals), so its layer is "Pear" and its removal asks
/// for a pear. One the model sees, does not answer about, or cannot name keeps its word.
pub(super) fn name_lookalikes(thinker: &dyn GenerativeBackend, picture: &[u8], area: Rect, out: &mut Vec<Subject>, seed: u64, ctx: &JobCtx) -> Result<()> {
    let progress = JobProgress::span(ctx, 0.03, 0.04);
    let mut gone = vec![false; out.len()];
    for (s, gone) in out.iter_mut().zip(gone.iter_mut()) {
        let word = bare(&s.term).to_ascii_lowercase();
        if s.asked || (word != PERSON && word != ANIMAL) {
            continue;
        }
        ctx.check()?;
        let Some(view) = lookalike_view(picture, area, s.bounds) else { continue };
        let question = format!("Is there a real {word} (not a picture, toy, statue or object shaped like one) in this photo?");
        if sees(thinker, &view, &question, seed, &progress) != Some(false) {
            continue;
        }
        match list_things(thinker, &view, 1, seed, &progress).into_iter().next() {
            Some(name) => s.term = name,
            // No real person and nothing the model can name (a speck it called "None", in the
            // app 2026-11-13): no subject, it stays in the background.
            None => *gone = true,
        }
    }
    let mut k = 0;
    out.retain(|_| {
        k += 1;
        !gone[k - 1]
    });
    Ok(())
}

/// A subject's bounds with a margin of [`LOOKALIKE_MARGIN`] of their size (within `area`),
/// cut from `picture`: what the vision-language model is shown to say what the subject is.
pub(super) fn lookalike_view(picture: &[u8], area: Rect, bounds: Rect) -> Option<Rgba8> {
    let (mx, my) = ((bounds.width() as f32 * LOOKALIKE_MARGIN) as i32, (bounds.height() as f32 * LOOKALIKE_MARGIN) as i32);
    let r = Rect::new(bounds.x0 - mx, bounds.y0 - my, bounds.x1 + mx, bounds.y1 + my).intersect(&area);
    if r.is_empty() {
        return None;
    }
    Rgba8::new(r.width(), r.height(), crop_rgba(picture, area, r)).ok()
}

/// The margin around a subject shown to the vision-language model, as a share of its size.
pub(super) const LOOKALIKE_MARGIN: f32 = 0.15;

/// Drop what is not a subject of its own: specks, whole-frame detections, near-duplicates,
/// parts inside wholes (unless the caller asked for the part by name, when it is cut out of the
/// whole); then, by `group`, join instances that overlap into one.
pub(super) fn clean(found: &mut Vec<Subject>, w: usize, h: usize, group: Group) {
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
pub(super) fn matte_all(
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
    for (i, (s, mut a)) in subjects.iter_mut().zip(alphas).enumerate() {
        // A matted alpha loses the specks and islands the matting left about the picture (a
        // subject's own separate parts are large against it and stay).
        if used != Edges::Hard && only.is_none_or(|o| o == i) {
            crate::select_ml_cmds::despeckle_alpha(&mut a, area, None);
        }
        s.alpha = a;
    }
    Ok(used)
}

/// The subjects' alphas from the matte model (Remove Background's), or `None` when the server
/// lacks it or its matte disagrees with the detector (kept little of the subjects, or much of
/// the rest). `only` is ViTMatte's: here every subject is cut from the one matte. The detector's masks are hard-edged and clip hair (the owner, 2026-11-02); the
/// matte is soft alpha. Each subject takes the matte within a band around its own mask (a
/// fraction of its size, so stray hair stays and other things the matte kept do not), its
/// mask's interior whole, and a pixel near two subjects goes to the nearer.
pub(super) fn matte_alphas(
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
    let plane = gray8_at(plane, w, h)?;
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
pub(super) const HOLE_MIN: f32 = 0.1;

/// The band around a subject's mask the matte decides, as a fraction of its larger side...
pub(super) const BAND: f32 = 0.04;

/// ...and at least this many pixels.
pub(super) const MIN_BAND: f32 = 6.0;

/// The matte's least mean over the subjects' interiors to be trusted.
pub(super) const MIN_KEPT: f64 = 0.6;

/// The matte's most mean over the rest (well away from every subject) to be trusted.
pub(super) const MAX_STRAY: f64 = 0.2;
