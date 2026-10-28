//! Realign to source: a model that regenerates the whole picture (Qwen-Image-2.1's edits and
//! fills, any edit by instruction) hands it back a few pixels off, sometimes a few percent
//! zoomed, differently per seed (AusBoss measured 24 px median on 2.1 restyles, FLUX 3 Klein
//! up to 200 px, `research/notes/ausboss-catalogue-2026-10.md` §2.1). Stacked on the original
//! as a masked layer, the shift shows as a doubled edge at the mask and a seam at the border.
//! This finds the translation that best matches the result to the original outside the edited
//! area (coarse search on a quarter-scale luma, refined at full scale) and moves the result
//! back, the strip it leaves at the border taken from the original.

use photocraft_genai::{Gray8, Rgba8};
use rayon::prelude::*;

/// Mask values below this count as "untouched picture" the alignment may measure on.
const OUTSIDE: u8 = 16;
/// The coarse search runs on the picture shrunk by this factor.
const COARSE: usize = 4;
/// Nothing is realigned when fewer than this many pixels are outside the mask.
const MIN_SAMPLES: usize = 2048;

/// Luma `(w, h)` of `img` shrunk by `f` (box mean), with the share of each cell that lies
/// outside `mask` (1 when there is no mask).
fn luma_down(img: &Rgba8, mask: Option<&Gray8>, f: usize) -> (Vec<f32>, Vec<f32>, usize, usize) {
    let (w, h) = (img.width as usize / f, img.height as usize / f);
    let mut luma = vec![0f32; w * h];
    let mut weight = vec![0f32; w * h];
    let iw = img.width as usize;
    for y in 0..h {
        for x in 0..w {
            let (mut sum, mut outside) = (0f32, 0f32);
            for dy in 0..f {
                for dx in 0..f {
                    let i = (y * f + dy) * iw + x * f + dx;
                    if let Some(p) = img.data.get(i * 4..i * 4 + 3) {
                        sum += 0.299 * f32::from(p[0]) + 0.587 * f32::from(p[1]) + 0.114 * f32::from(p[2]);
                    }
                    let m = mask.and_then(|m| m.data.get(i)).copied().unwrap_or(0);
                    if m < OUTSIDE {
                        outside += 1.0;
                    }
                }
            }
            let n = (f * f) as f32;
            luma[y * w + x] = sum / n;
            weight[y * w + x] = outside / n;
        }
    }
    (luma, weight, w, h)
}

/// Mean absolute luma difference between `a` and `b` shifted by `(dx, dy)` (a pixel of `b` at
/// `(x, y)` is compared with `a` at `(x + dx, y + dy)`), over the cells both cover, weighted by
/// `wa`; `None` when too few cells overlap.
#[allow(clippy::too_many_arguments)]
fn cost(a: &[f32], wa: &[f32], b: &[f32], w: usize, h: usize, dx: i32, dy: i32, min_cells: usize) -> Option<f32> {
    let (mut sum, mut n) = (0f32, 0f32);
    let (x0, x1) = (0.max(-dx) as usize, (w as i32).min(w as i32 - dx) as usize);
    let (y0, y1) = (0.max(-dy) as usize, (h as i32).min(h as i32 - dy) as usize);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    for y in y0..y1 {
        for x in x0..x1 {
            let ia = (y as i32 + dy) as usize * w + (x as i32 + dx) as usize;
            let ib = y * w + x;
            let wt = wa[ia] * wa[ib];
            if wt > 0.0 {
                sum += wt * (a[ia] - b[ib]).abs();
                n += wt;
            }
        }
    }
    (n >= min_cells as f32).then(|| sum / n)
}

/// The shift `(dx, dy)` that lays `result` over `original` best outside `mask` (the result's
/// pixel at `(x, y)` matches the original at `(x + dx, y + dy)`), searched within `±max_shift`
/// pixels; `(0, 0)` when the sizes differ, the mask leaves too little to measure, or no shift
/// beats none by a margin.
pub(crate) fn drift(result: &Rgba8, original: &Rgba8, mask: Option<&Gray8>, max_shift: u32) -> (i32, i32) {
    if (result.width, result.height) != (original.width, original.height) || result.width < 32 || result.height < 32 {
        return (0, 0);
    }
    if let Some(m) = mask
        && (m.width, m.height) != (result.width, result.height)
    {
        return (0, 0);
    }
    let (la, wa, cw, ch) = luma_down(original, mask, COARSE);
    let (lb, _, _, _) = luma_down(result, None, COARSE);
    let outside: f32 = wa.iter().sum();
    if (outside as usize) * COARSE * COARSE < MIN_SAMPLES {
        return (0, 0);
    }
    let reach = (max_shift as i32 / COARSE as i32).max(1);
    let min_cells = (outside * 0.25) as usize;
    let Some(c0) = cost(&la, &wa, &lb, cw, ch, 0, 0, min_cells) else { return (0, 0) };
    let mut best = (0i32, 0i32, c0);
    // Each candidate keeps its original sum order; ordered collection preserves tie breaks.
    let coarse: Vec<_> = (-reach..=reach)
        .into_par_iter()
        .map(|dy| (-reach..=reach).map(|dx| (dx, dy, cost(&la, &wa, &lb, cw, ch, dx, dy, min_cells))).collect::<Vec<_>>())
        .collect();
    for row in coarse {
        for (dx, dy, c) in row {
            if let Some(c) = c
                && c < best.2
            {
                best = (dx, dy, c);
            }
        }
    }
    // Refine at full scale around the coarse answer.
    let (fa, fwa, fw, fh) = luma_down(original, mask, 1);
    let (fb, _, _, _) = luma_down(result, None, 1);
    let min_full = (fwa.iter().sum::<f32>() * 0.25) as usize;
    let Some(full0) = cost(&fa, &fwa, &fb, fw, fh, 0, 0, min_full) else { return (0, 0) };
    let (cx, cy) = (best.0 * COARSE as i32, best.1 * COARSE as i32);
    let mut fine = (0i32, 0i32, full0);
    let fine_rows: Vec<_> = ((cy - COARSE as i32)..=(cy + COARSE as i32))
        .into_par_iter()
        .map(|dy| {
            ((cx - COARSE as i32)..=(cx + COARSE as i32))
                .filter(|dx| dx.abs() <= max_shift as i32 && dy.abs() <= max_shift as i32)
                .map(|dx| (dx, dy, cost(&fa, &fwa, &fb, fw, fh, dx, dy, min_full)))
                .collect::<Vec<_>>()
        })
        .collect();
    for row in fine_rows {
        for (dx, dy, c) in row {
            if let Some(c) = c
                && c < fine.2
            {
                fine = (dx, dy, c);
            }
        }
    }
    // A shift has to be a real improvement, not noise: a tenth off the unshifted cost.
    if fine.2 < full0 * 0.9 { (fine.0, fine.1) } else { (0, 0) }
}

/// `result` moved by `(dx, dy)` so that it lies over `original` (see [`drift`]); the strip the
/// move leaves at the border is the original's pixels. `result` itself at `(0, 0)`.
pub(crate) fn shifted(result: &Rgba8, original: &Rgba8, (dx, dy): (i32, i32)) -> Rgba8 {
    if (dx, dy) == (0, 0) || (result.width, result.height) != (original.width, original.height) {
        return result.clone();
    }
    let (w, h) = (result.width as i32, result.height as i32);
    let mut data = original.data.clone();
    for y in 0..h {
        for x in 0..w {
            // The pixel that belongs at (x, y) is the result's at (x - dx, y - dy).
            let (sx, sy) = (x - dx, y - dy);
            if sx < 0 || sy < 0 || sx >= w || sy >= h {
                continue;
            }
            let (di, si) = (((y * w + x) * 4) as usize, ((sy * w + sx) * 4) as usize);
            if let (Some(d), Some(s)) = (data.get_mut(di..di + 4), result.data.get(si..si + 4)) {
                d.copy_from_slice(s);
            }
        }
    }
    Rgba8::new(result.width, result.height, data).unwrap_or_else(|_| result.clone())
}

/// The largest shift realignment looks for: a share of the picture's shorter side.
pub(crate) fn max_shift(width: u32, height: u32) -> u32 {
    (width.min(height) / 12).clamp(4, 96)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The previous search is the oracle for sum order and equal-cost candidate selection.
    fn sequential_drift(result: &Rgba8, original: &Rgba8, mask: Option<&Gray8>, max_shift: u32) -> (i32, i32) {
        if (result.width, result.height) != (original.width, original.height) || result.width < 32 || result.height < 32 {
            return (0, 0);
        }
        if let Some(m) = mask
            && (m.width, m.height) != (result.width, result.height)
        {
            return (0, 0);
        }
        let (la, wa, cw, ch) = luma_down(original, mask, COARSE);
        let (lb, _, _, _) = luma_down(result, None, COARSE);
        let outside: f32 = wa.iter().sum();
        if (outside as usize) * COARSE * COARSE < MIN_SAMPLES {
            return (0, 0);
        }
        let reach = (max_shift as i32 / COARSE as i32).max(1);
        let min_cells = (outside * 0.25) as usize;
        let Some(c0) = cost(&la, &wa, &lb, cw, ch, 0, 0, min_cells) else { return (0, 0) };
        let mut best = (0i32, 0i32, c0);
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                if let Some(c) = cost(&la, &wa, &lb, cw, ch, dx, dy, min_cells)
                    && c < best.2
                {
                    best = (dx, dy, c);
                }
            }
        }
        // Refine at full scale around the coarse answer.
        let (fa, fwa, fw, fh) = luma_down(original, mask, 1);
        let (fb, _, _, _) = luma_down(result, None, 1);
        let min_full = (fwa.iter().sum::<f32>() * 0.25) as usize;
        let Some(full0) = cost(&fa, &fwa, &fb, fw, fh, 0, 0, min_full) else { return (0, 0) };
        let (cx, cy) = (best.0 * COARSE as i32, best.1 * COARSE as i32);
        let mut fine = (0i32, 0i32, full0);
        for dy in (cy - COARSE as i32)..=(cy + COARSE as i32) {
            for dx in (cx - COARSE as i32)..=(cx + COARSE as i32) {
                if dx.abs() > max_shift as i32 || dy.abs() > max_shift as i32 {
                    continue;
                }
                if let Some(c) = cost(&fa, &fwa, &fb, fw, fh, dx, dy, min_full)
                    && c < fine.2
                {
                    fine = (dx, dy, c);
                }
            }
        }
        // A shift has to be a real improvement, not noise: a tenth off the unshifted cost.
        if fine.2 < full0 * 0.9 { (fine.0, fine.1) } else { (0, 0) }
    }

    #[test]
    fn parallel_search_preserves_the_sequential_alignment() {
        let original = picture(256, 192);
        let mut mask = vec![0; 256 * 192];
        for (i, v) in mask.iter_mut().enumerate() {
            *v = if i % 7 == 0 { 255 } else { 0 };
        }
        let mask = Gray8::new(256, 192, mask).unwrap();
        for delta in [(0, 0), (7, -5), (-12, 9)] {
            let result = moved(&original, delta.0, delta.1);
            for mask in [None, Some(&mask)] {
                assert_eq!(drift(&result, &original, mask, 16), sequential_drift(&result, &original, mask, 16));
            }
        }
        let flat = Rgba8::new(256, 192, vec![128; 256 * 192 * 4]).unwrap();
        assert_eq!(drift(&flat, &flat, None, 16), sequential_drift(&flat, &flat, None, 16));
    }

    #[test]
    #[ignore = "release CPU timing at tile and 24 MP sizes"]
    fn alignment_performance() {
        for (w, h) in [(992, 992), (6000, 4000)] {
            let original = picture(w, h);
            let result = moved(&original, 7, -5);
            let start = std::time::Instant::now();
            let before = sequential_drift(&result, &original, None, max_shift(w, h));
            let before_ms = start.elapsed().as_secs_f64() * 1000.0;
            let start = std::time::Instant::now();
            let after = drift(&result, &original, None, max_shift(w, h));
            eprintln!("alignment {w}x{h}: sequential {before_ms:.3} ms, parallel {:.3} ms", start.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(before, after);
        }
    }

    /// A textured picture: smoothed noise, so that a shift is unambiguous in both directions
    /// (a striped test picture aliased along its stripes).
    fn picture(w: u32, h: u32) -> Rgba8 {
        let (wu, hu) = (w as usize, h as usize);
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let noise: Vec<f32> = (0..wu * hu)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect();
        let mut data = Vec::with_capacity(wu * hu * 4);
        for yy in 0..hu {
            for xx in 0..wu {
                let mut s = 0f32;
                for dy in 0..3 {
                    for dx in 0..3 {
                        s += noise[((yy + dy) % hu) * wu + (xx + dx) % wu];
                    }
                }
                let u = (s / 9.0 * 255.0) as u8;
                data.extend_from_slice(&[u, 255 - u, (u / 2).wrapping_add(90), 255]);
            }
        }
        Rgba8::new(w, h, data).unwrap()
    }

    /// `img` moved by `(dx, dy)` (its pixel at `(x, y)` lands at `(x + dx, y + dy)`), the gap grey.
    fn moved(img: &Rgba8, dx: i32, dy: i32) -> Rgba8 {
        let (w, h) = (img.width as i32, img.height as i32);
        let mut data = vec![128u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = (x - dx, y - dy);
                if sx >= 0 && sy >= 0 && sx < w && sy < h {
                    let (di, si) = (((y * w + x) * 4) as usize, ((sy * w + sx) * 4) as usize);
                    data[di..di + 4].copy_from_slice(&img.data[si..si + 4]);
                }
            }
        }
        Rgba8::new(img.width, img.height, data).unwrap()
    }

    #[test]
    fn a_shifted_result_is_measured_and_moved_back() {
        let original = picture(160, 120);
        // Within `max_shift` (10 px on this size); a bigger drift is left where it is.
        for (dx, dy) in [(5, -3), (-9, 7), (0, 0), (8, 0)] {
            let result = moved(&original, dx, dy);
            // The result's pixel at (x, y) is the original's at (x - dx, y - dy): the drift is (-dx, -dy).
            let d = drift(&result, &original, None, max_shift(160, 120));
            assert_eq!(d, (-dx, -dy), "shift ({dx}, {dy})");
            let back = shifted(&result, &original, d);
            // Inside the picture the moved-back result is the original again.
            let (w, h) = (160usize, 120usize);
            let mut diff = 0usize;
            for y in 20..h - 20 {
                for x in 20..w - 20 {
                    let i = (y * w + x) * 4;
                    if back.data[i..i + 3] != original.data[i..i + 3] {
                        diff += 1;
                    }
                }
            }
            assert_eq!(diff, 0, "shift ({dx}, {dy}): {diff} pixels differ after realignment");
        }
    }

    #[test]
    fn the_edited_area_is_left_out_of_the_measure_and_small_pictures_are_left_alone() {
        let original = picture(160, 120);
        let result = moved(&original, 4, 2);
        // A mask over the left half: the measure uses the right half alone and still finds the shift.
        let mut m = vec![0u8; 160 * 120];
        for y in 0..120 {
            for x in 0..80 {
                m[y * 160 + x] = 255;
            }
        }
        let mask = Gray8::new(160, 120, m).unwrap();
        assert_eq!(drift(&result, &original, Some(&mask), 16), (-4, -2));
        // A mask covering nearly everything leaves too little to measure.
        let mask = Gray8::new(160, 120, vec![255u8; 160 * 120]).unwrap();
        assert_eq!(drift(&result, &original, Some(&mask), 16), (0, 0));
        // Different sizes, a tiny picture: no realignment.
        assert_eq!(drift(&result, &picture(100, 120), None, 16), (0, 0));
        assert_eq!(drift(&picture(16, 16), &picture(16, 16), None, 4), (0, 0));
        assert_eq!(max_shift(1024, 768), 64);
        assert_eq!(max_shift(64, 64), 5);
        assert_eq!(max_shift(4000, 4000), 96);
    }
}
