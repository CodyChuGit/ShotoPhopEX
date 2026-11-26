//! Split into Layers' hole fills: the tries in order (removal, grey box, strips, the
//! classical patch) and the tone grid that matches a fill to its surroundings.

use photocraft_doc::{Document, LayerId};
use photocraft_geom::Rect;

use crate::jobs::JobCtx;
use crate::{EngineError, Result};

use super::{
    BIG_HOLE_SIDE, BIG_TRIES, FILL_TRIES, FillJob, KEPT_WHOLE, Layer, MAX_GROW, MAX_STRIPS, PATCH_PIXELS, PLAIN_TEXTURE, PREFILL, PROTECT_GROW,
    REMOVE_MIN_REQUEST, RING_WIDTH, Rgba8, STRIP_LEN, STRIP_OVERLAP, TIGHT_BAND, TONE_MATCH_MAX, TOUCHED, Value, at_least, blank_or_ghost, box_blur, dilate,
    embed_coverage, fill_rect, fill_region, gen_err, json, kept_share, remove_grow, texture_and_luma, texture_step, tolerable, write_rgba8,
};

/// What a fill left: the picture with the hole filled, the fill's seed, and what the check saw
/// in it.
pub(super) type Filled<T> = (Vec<u8>, u64, T);

/// A check's verdict on a fill: what it saw, and why the fill is rejected, if it is, with the
/// bounds of the detection the refusal names (`at` in the try's record).
pub(super) type Verdict<T> = (T, Option<(String, Rect)>);

/// A check on a fill: given the filled picture, where the fill was laid at all (one value per
/// canvas pixel) and whether this try was the classical patch, it gives its [`Verdict`].
pub(super) type Check<'a, T> = dyn FnMut(&[u8], &[f32], bool) -> Result<Verdict<T>> + 'a;

/// What a hole's tries came to: the fill accepted (`None` when every try was refused), every
/// try's record, the time all the tries took, the model generations they asked for, and the
/// refused fills themselves when asked to keep them (each named for its try and verdict).
pub(super) struct HoleFill<T> {
    pub(super) outcome: Option<Filled<T>>,
    pub(super) tries: Vec<Value>,
    pub(super) ms: u64,
    pub(super) generations: u32,
    pub(super) refused: Vec<(String, Vec<u8>)>,
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
/// `verdict`: accepted, or why not; `at`: the bounds of the detection a refusal names), the time
/// and the generations of them all. When every try
/// is refused and one of them only for a thing painted where none was ([`tolerable`]), that one
/// is taken after all, "accepted as the last resort". With `keep`, the refused fills come back
/// too.
#[allow(clippy::too_many_arguments)]
pub(super) fn fill_hole<T>(
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
    keep: bool,
    removal: &FillJob,
    box_job: Option<&FillJob>,
    (lo, hi): (f32, f32),
    check: &mut Check<'_, T>,
) -> Result<HoleFill<T>> {
    let (w, h) = (area.width(), area.height());
    let (wu, hu) = (w as usize, h as usize);
    let (hole_rect, holed) = crate::select_ml_cmds::coverage_bounds(hole, area);
    let mut done = HoleFill { outcome: None, tries: Vec::with_capacity(FILL_TRIES as usize + 1), ms: 0, generations: 0, refused: Vec::new() };
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
    // A big hole with texture around it gets fewer tries before the patch: a strips try is four
    // generations, and a refused big hole cost its whole ladder, 17 generations, in round 7.
    let tries = if big && !plain { BIG_TRIES } else { FILL_TRIES };
    // The first refusal the hole may fall back on ([`tolerable`]): its record's index, its
    // verdict and its result.
    let mut fallback: Option<(usize, String, Filled<T>)> = None;
    for attempt in 0..=tries {
        let span = hi - lo;
        ctx.progress(lo + span * 0.8 * attempt as f32 / (tries + 1) as f32, "");
        let step = if attempt == tries { Step::Patch } else { step_of(attempt, big, plain, box_job.is_some()) };
        let kind = step.name();
        let progress = (lo + span * 0.8 * attempt as f32 / (tries + 1) as f32, lo + span * 0.8 * (attempt as f32 + 0.9) / (tries + 1) as f32);
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
        // By the pixels first: the silhouette kept, then a blank fill (the grey box as it was)
        // or a smear where the subject stood, which pass a detector and the pixels alike and
        // are judged against the picture around the hole.
        let by_pixels = watch
            .map(|silhouette| kept_share(picture, &background, silhouette, wu, hu))
            .filter(|kept| *kept >= KEPT_WHOLE)
            .map(|kept| format!("kept what stood there ({:.0} % of it as it was)", kept * 100.0))
            .or_else(|| blank_or_ghost(picture, &background, hole, &ring, wu, hu));
        if let Some(why) = by_pixels {
            done.tries.push(record(&why));
            if keep {
                done.refused.push((format!("{kind} refused ({why})"), background));
            }
            continue;
        }
        match check(&background, &touched, step == Step::Patch && thing) {
            Ok((seen, None)) => {
                done.tries.push(record("accepted"));
                done.outcome = Some((background, fill_seed, seen));
                break;
            }
            Ok((seen, Some((why, at)))) => {
                let mut t = record(&why);
                t["at"] = json!([at.x0, at.y0, at.width(), at.height()]);
                done.tries.push(t);
                if fallback.is_none() && tolerable(&why) {
                    fallback = Some((done.tries.len() - 1, why, (background, fill_seed, seen)));
                } else if keep {
                    done.refused.push((format!("{kind} refused ({why})"), background));
                }
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
    // The one refusal a hole falls back on when every try is refused: a thing painted where none
    // was (a table behind the fedora man, round 7: six refusals, and he stayed in the background
    // in two seeds of three). The room gains a table, but the subject leaves the background. A
    // person painted back or in, a shape over the hole or a blank never.
    match fallback {
        Some((at, why, filled)) if done.outcome.is_none() => {
            done.tries[at]["verdict"] = json!(format!("accepted as the last resort ({why})"));
            done.outcome = Some(filled);
        }
        Some((at, why, (background, _, _))) if keep => {
            let kind = done.tries[at]["fill"].as_str().unwrap_or("").to_string();
            done.refused.push((format!("{kind} refused ({why})"), background));
        }
        _ => {}
    }
    Ok(done)
}

/// One try of a hole's fill.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Step {
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
    pub(super) fn name(self) -> &'static str {
        match self {
            Step::Removal => "removal",
            Step::Box => "fill",
            Step::Strips => "strips",
            Step::Patch => "patch",
        }
    }
}

/// Which fill try `attempt` of a hole is. The removal first, every time; then, for a small hole
/// or a plain ring, the grey-box fills; for a big hole with texture around it (`big`), the box
/// in strips twice and then whole ([`BIG_TRIES`] tries in all before the patch). Without a box
/// fill to call on, the removal every time.
pub(super) fn step_of(attempt: u32, big: bool, plain: bool, boxed: bool) -> Step {
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
        1 | 2 => Step::Strips,
        _ => Step::Box,
    }
}

/// One fill of `job` over `coverage` in `rect`, run above the active layer: the composite with
/// its result, the seed it ran, its time and the generations it made. Its layers are taken out
/// again.
pub(super) fn one_fill(
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
pub(super) fn fill_in_strips(
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
pub(super) fn patch_fill(picture: &[u8], hole: &[f32], protect: Option<&[f32]>, w: usize, h: usize, seed: u64) -> Option<Vec<u8>> {
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

/// Cells of [`ToneGrid`], in pixels.
pub(super) const TONE_CELL: usize = 64;

/// A smooth per-channel correction (picture minus fill) measured where `known` says both show
/// the same backdrop, averaged over cells and carried across the rest (the subjects) as a
/// membrane: the fill's tone matched to its surroundings with no step at their edge. At most
/// [`TONE_MATCH_MAX`] a channel: a fill in other colours altogether keeps them.
pub(super) struct ToneGrid {
    pub(super) gw: usize,
    pub(super) gh: usize,
    pub(super) v: Vec<[f32; 3]>,
}

impl ToneGrid {
    pub(super) fn new(picture: &[u8], filled: &[u8], known: impl Fn(usize) -> bool, w: usize, h: usize) -> Self {
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
    pub(super) fn at(&self, x: usize, y: usize) -> [f32; 3] {
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
