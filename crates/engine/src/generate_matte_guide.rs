//! Remove Background / High Quality Select Subject on the matte model: the detector picks
//! *which* thing, the matte model gives its soft edge.
//!
//! Qwen-Image-2.1's matte decides on its own what the subject is. On a scene with no single
//! subject it keeps too much (a kitchen's fridge came back with the counter, the cabinets and
//! half the fruit bowl) or too little (asked for "the fridge", only its door), and its
//! background is a speckle of 1 to 4 out of 255 instead of 0 (RTX 3090 release test,
//! 2026-11-14). SAM 3.1 is reliable at the *which*: the matte is kept only near what the
//! detector finds for the same name, the detector's solid inside is filled in, and the faint
//! speckle is cleared.

use crate::Result;
use crate::jobs::JobCtx;
use photocraft_geom::Rect;

/// How far (a share of the picture's diagonal) the matte may reach past the detector's mask:
/// room for hair and soft edges the hard mask cuts, not for a neighbouring object.
const REACH: f64 = 0.02;
/// How far inside the detector's mask its coverage counts as solid (a share of the diagonal).
const INSET: f64 = 0.01;
/// Matte values this close to 0 or 1 (out of 1) are the model's speckle, not coverage.
const SPECKLE: f32 = 6.0 / 255.0;

/// The detector's instances of `name` (or, with no name, Select Subject's prompts in turn) in
/// `picture` (RGBA8 over `area`), joined; None when it finds nothing.
pub(crate) fn guide_for(
    finder: &crate::select_ml_cmds::SubjectFinder,
    picture: &[u8],
    area: Rect,
    name: Option<&str>,
    ctx: &JobCtx,
) -> Result<Option<Vec<f32>>> {
    let names: Vec<&str> = match name.map(str::trim).filter(|n| !n.is_empty()) {
        Some(n) => vec![n],
        None => crate::select_ml_cmds::SUBJECT_FALLBACKS.iter().take(2).copied().collect(),
    };
    for n in names {
        ctx.check()?;
        let found = finder.find_in(picture, area, n, GUIDE_AT_MOST, ctx)?;
        if found.is_empty() {
            continue;
        }
        let mut joined = vec![0.0f32; picture.len() / 4];
        for (cov, ..) in &found {
            for (j, c) in joined.iter_mut().zip(cov) {
                *j = j.max(*c);
            }
        }
        return Ok(Some(joined));
    }
    Ok(None)
}

/// How many detector instances a guide joins (a named "the plates" may be several).
const GUIDE_AT_MOST: usize = 3;

/// `matte` (one coverage per pixel of a `w × h` picture) kept only within [`REACH`] of the
/// detector's `guide`, with the guide's inside (deeper than [`INSET`]) solid. An empty matte or
/// guide leaves the matte as it is (an empty matte still reports "kept nothing").
pub(crate) fn apply_guide(matte: &mut [f32], guide: &[f32], w: usize, h: usize) {
    if matte.len() != w * h || guide.len() != w * h || !matte.iter().any(|c| *c > SPECKLE) {
        return;
    }
    let found: Vec<bool> = guide.iter().map(|g| *g >= 0.5).collect();
    if !found.iter().any(|f| *f) {
        return;
    }
    let diag = ((w * w + h * h) as f64).sqrt();
    let reach = ((diag * REACH) as usize).max(4);
    let inset = ((diag * INSET) as usize).max(2);
    let near = photocraft_algo::remove::dilate(w, h, &found, reach);
    let outside: Vec<bool> = found.iter().map(|f| !f).collect();
    let rim = photocraft_algo::remove::dilate(w, h, &outside, inset);
    for (i, c) in matte.iter_mut().enumerate() {
        if !near[i] {
            *c = 0.0;
        } else if !rim[i] {
            *c = 1.0;
        }
    }
}

/// The matte model's near-0 speckle cleared and its near-1 made solid.
pub(crate) fn clear_speckle(matte: &mut [f32]) {
    for c in matte.iter_mut() {
        if *c < SPECKLE {
            *c = 0.0;
        } else if *c > 1.0 - SPECKLE {
            *c = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(w: usize, h: usize, x0: usize, y0: usize, x1: usize, y1: usize, v: f32) -> Vec<f32> {
        (0..w * h).map(|i| if (x0..x1).contains(&(i % w)) && (y0..y1).contains(&(i / w)) { v } else { 0.0 }).collect()
    }

    #[test]
    fn the_matte_is_cut_to_what_the_detector_found_and_its_inside_filled() {
        let (w, h) = (200, 100);
        // The matte keeps the subject (x 40..80) and a neighbour (x 140..180) and misses the
        // subject's left half; the detector found only the subject.
        let mut matte = rect(w, h, 60, 20, 80, 80, 0.9);
        for (m, n) in matte.iter_mut().zip(rect(w, h, 140, 20, 180, 80, 1.0)) {
            *m = m.max(n);
        }
        let guide = rect(w, h, 40, 20, 80, 80, 1.0);
        apply_guide(&mut matte, &guide, w, h);
        assert_eq!(matte[50 * w + 160], 0.0, "the neighbour is dropped");
        assert_eq!(matte[50 * w + 50], 1.0, "the missed half is filled from the detector");
        assert_eq!(matte[50 * w + 70], 1.0);
        // The edge band keeps the matte's own soft value.
        assert!((matte[21 * w + 70] - 0.9).abs() < 1e-6, "{}", matte[21 * w + 70]);
    }

    #[test]
    fn an_empty_guide_or_matte_changes_nothing_and_speckle_clears() {
        let (w, h) = (20, 10);
        let mut matte = rect(w, h, 5, 2, 15, 8, 0.5);
        let before = matte.clone();
        apply_guide(&mut matte, &vec![0.0; w * h], w, h);
        assert_eq!(matte, before);
        let mut empty = vec![0.0; w * h];
        apply_guide(&mut empty, &rect(w, h, 5, 2, 15, 8, 1.0), w, h);
        assert!(empty.iter().all(|c| *c == 0.0), "an empty matte stays empty");
        let mut speck = vec![2.0 / 255.0, 0.5, 253.0 / 255.0];
        clear_speckle(&mut speck);
        assert_eq!(speck, vec![0.0, 0.5, 1.0]);
    }
}
