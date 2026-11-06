//! The yardsticks (plan §12.1). Pictures: PSNR and SSIM on luma. Masks: IoU and the boundary
//! F-score within a pixel tolerance. Thresholds live in [`Verdict`] so every report applies the
//! same ones.

/// Peak signal-to-noise ratio in dB between two luma planes in `0..=1` (`f32::INFINITY` when
/// identical; `None` when the sizes differ).
pub fn psnr(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mse = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len() as f32;
    Some(if mse <= 0.0 { f32::INFINITY } else { 10.0 * (1.0 / mse).log10() })
}

/// Global SSIM on luma: the mean over 8 × 8 windows (stride 4) of the standard formula with
/// `k1 = 0.01`, `k2 = 0.03` on a `0..=1` range. `None` when the sizes differ or the picture is
/// smaller than a window.
pub fn ssim(a: &[f32], b: &[f32], width: usize, height: usize) -> Option<f32> {
    const WIN: usize = 8;
    const STRIDE: usize = 4;
    if a.len() != width * height || b.len() != a.len() || width < WIN || height < WIN {
        return None;
    }
    let (c1, c2) = (0.01f32 * 0.01, 0.03f32 * 0.03);
    let (mut sum, mut count) = (0.0f64, 0u64);
    let mut y = 0;
    while y + WIN <= height {
        let mut x = 0;
        while x + WIN <= width {
            let (mut ma, mut mb) = (0.0f32, 0.0f32);
            for j in 0..WIN {
                for i in 0..WIN {
                    let k = (y + j) * width + x + i;
                    ma += a.get(k).copied().unwrap_or(0.0);
                    mb += b.get(k).copied().unwrap_or(0.0);
                }
            }
            let n = (WIN * WIN) as f32;
            ma /= n;
            mb /= n;
            let (mut va, mut vb, mut cov) = (0.0f32, 0.0f32, 0.0f32);
            for j in 0..WIN {
                for i in 0..WIN {
                    let k = (y + j) * width + x + i;
                    let da = a.get(k).copied().unwrap_or(0.0) - ma;
                    let db = b.get(k).copied().unwrap_or(0.0) - mb;
                    va += da * da;
                    vb += db * db;
                    cov += da * db;
                }
            }
            va /= n - 1.0;
            vb /= n - 1.0;
            cov /= n - 1.0;
            let s = ((2.0 * ma * mb + c1) * (2.0 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            sum += f64::from(s);
            count += 1;
            x += STRIDE;
        }
        y += STRIDE;
    }
    if count == 0 { None } else { Some((sum / count as f64) as f32) }
}

/// Intersection over union of two coverages thresholded at 0.5 (`None` when the sizes differ;
/// `Some(1.0)` when both are empty).
pub fn iou(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let (mut i, mut u) = (0u64, 0u64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x >= 0.5, *y >= 0.5);
        i += u64::from(x && y);
        u += u64::from(x || y);
    }
    Some(if u == 0 { 1.0 } else { i as f32 / u as f32 })
}

/// Boundary pixels of a thresholded coverage: inside pixels with an outside 4-neighbour (or on
/// the picture's edge).
fn boundary(c: &[f32], w: usize, h: usize) -> Vec<bool> {
    let inside = |x: isize, y: isize| -> bool {
        if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
            return false;
        }
        c.get(y as usize * w + x as usize).is_some_and(|v| *v >= 0.5)
    };
    let mut out = vec![false; w * h];
    for y in 0..h as isize {
        for x in 0..w as isize {
            if inside(x, y)
                && !(inside(x - 1, y) && inside(x + 1, y) && inside(x, y - 1) && inside(x, y + 1))
                && let Some(o) = out.get_mut(y as usize * w + x as usize)
            {
                *o = true;
            }
        }
    }
    out
}

/// Boundary F-score: precision and recall of `a`'s boundary against `b`'s within `tol` pixels
/// (Chebyshev), as in the DAVIS benchmark's F measure. `Some(1.0)` when neither has a boundary.
pub fn boundary_f(a: &[f32], b: &[f32], w: usize, h: usize, tol: usize) -> Option<f32> {
    if a.len() != w * h || b.len() != a.len() {
        return None;
    }
    let (ba, bb) = (boundary(a, w, h), boundary(b, w, h));
    let near = |set: &[bool], x: usize, y: usize| -> bool {
        let (x0, y0) = (x.saturating_sub(tol), y.saturating_sub(tol));
        let (x1, y1) = ((x + tol).min(w - 1), (y + tol).min(h - 1));
        (y0..=y1).any(|yy| (x0..=x1).any(|xx| set.get(yy * w + xx).copied().unwrap_or(false)))
    };
    let count = |from: &[bool], to: &[bool]| -> (u64, u64) {
        let (mut hit, mut total) = (0u64, 0u64);
        for y in 0..h {
            for x in 0..w {
                if from.get(y * w + x).copied().unwrap_or(false) {
                    total += 1;
                    hit += u64::from(near(to, x, y));
                }
            }
        }
        (hit, total)
    };
    let (pa, ta) = count(&ba, &bb);
    let (pb, tb) = count(&bb, &ba);
    if ta == 0 && tb == 0 {
        return Some(1.0);
    }
    if ta == 0 || tb == 0 {
        return Some(0.0);
    }
    let (precision, recall) = (pa as f32 / ta as f32, pb as f32 / tb as f32);
    Some(if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) })
}

/// Bounding-box IoU for `[x, y, w, h]` boxes (the SAM suite's first yardstick while the engines
/// report instance bounds rather than masks).
pub fn bounds_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let (ax1, ay1, bx1, by1) = (a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]);
    let iw = (ax1.min(bx1) - a[0].max(b[0])).max(0.0);
    let ih = (ay1.min(by1) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let union = a[2] * a[3] + b[2] * b[3] - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// The plan's thresholds, applied the same way in every report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    /// No comparison was possible (a missing or mismatched output).
    None,
}

impl Verdict {
    /// Pictures at equal seed: PSNR ≥ 30 dB **or** SSIM ≥ 0.90 (samplers may legitimately
    /// diverge; the softer yardstick decides and a human looks at the contact sheet).
    pub fn picture(psnr: Option<f32>, ssim: Option<f32>) -> Self {
        match (psnr, ssim) {
            (None, None) => Self::None,
            (p, s) => {
                if p.is_some_and(|p| p >= 30.0) || s.is_some_and(|s| s >= 0.90) {
                    Self::Pass
                } else {
                    Self::Fail
                }
            }
        }
    }

    /// Masks: IoU ≥ 0.97 and boundary F ≥ 0.95.
    pub fn mask(iou: Option<f32>, f: Option<f32>) -> Self {
        match (iou, f) {
            (Some(i), Some(f)) => {
                if i >= 0.97 && f >= 0.95 {
                    Self::Pass
                } else {
                    Self::Fail
                }
            }
            _ => Self::None,
        }
    }

    pub fn glyph(self) -> &'static str {
        match self {
            Self::Pass => "✓",
            Self::Fail => "✗",
            Self::None => "–",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_pictures_score_perfectly_and_noise_scores_lower() {
        let w = 32;
        let a: Vec<f32> = (0..w * w).map(|i| ((i % w) as f32) / w as f32).collect();
        assert_eq!(psnr(&a, &a), Some(f32::INFINITY));
        assert!((ssim(&a, &a, w, w).unwrap_or(0.0) - 1.0).abs() < 1e-4);
        let b: Vec<f32> = a.iter().enumerate().map(|(i, v)| (v + if i % 2 == 0 { 0.05 } else { -0.05 }).clamp(0.0, 1.0)).collect();
        let p = psnr(&a, &b).unwrap_or(0.0);
        assert!((25.0..27.0).contains(&p), "{p}");
        assert!(ssim(&a, &b, w, w).unwrap_or(1.0) < 0.99);
        assert_eq!(psnr(&a, &b[..10]), None);
        assert_eq!(Verdict::picture(Some(31.0), Some(0.5)), Verdict::Pass);
        assert_eq!(Verdict::picture(Some(20.0), Some(0.5)), Verdict::Fail);
        assert_eq!(Verdict::picture(None, None), Verdict::None);
    }

    #[test]
    fn masks_compare_by_area_and_boundary() {
        let (w, h) = (20usize, 20usize);
        let square = |x0: usize, y0: usize, x1: usize, y1: usize| -> Vec<f32> {
            (0..w * h).map(|i| if (x0..x1).contains(&(i % w)) && (y0..y1).contains(&(i / w)) { 1.0 } else { 0.0 }).collect()
        };
        let a = square(5, 5, 15, 15);
        let b = square(6, 5, 16, 15);
        assert_eq!(iou(&a, &a), Some(1.0));
        let i = iou(&a, &b).unwrap_or(0.0);
        assert!((i - 90.0 / 110.0).abs() < 1e-5, "{i}");
        assert_eq!(boundary_f(&a, &a, w, h, 0), Some(1.0));
        assert_eq!(boundary_f(&a, &b, w, h, 1), Some(1.0), "one pixel off is within tolerance 1");
        assert!(boundary_f(&a, &b, w, h, 0).unwrap_or(1.0) < 1.0);
        let empty = vec![0.0; w * h];
        assert_eq!(iou(&empty, &empty), Some(1.0));
        assert_eq!(boundary_f(&a, &empty, w, h, 2), Some(0.0));
        assert_eq!(Verdict::mask(Some(0.98), Some(0.96)), Verdict::Pass);
        assert_eq!(Verdict::mask(Some(0.98), Some(0.5)), Verdict::Fail);
        assert!((bounds_iou([0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 10.0, 10.0]) - 50.0 / 150.0).abs() < 1e-6);
    }
}
