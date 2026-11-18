//! Finishing a cut-out's soft edge, with no model: the straight colour behind each
//! half-covered pixel, a garbage matte that clears stray specks, and green / blue screen spill.
//!
//! * **Foreground estimation** follows D. Germer, T. Stadie, S. Conrad and R. Klette, "Fast
//!   Multi-Level Foreground Estimation" (ICPR 2020). An edge pixel's colour is a mix
//!   `I = αF + (1 − α)B` of the subject `F` and what was behind it `B`; both are solved for at
//!   once with a smoothness term that is weaker across alpha edges, coarse to fine: the picture
//!   is shrunk to 1 × 1 and grown back level by level, each level starting from the one below
//!   and refined by a few sweeps. Sweeps run in red-black (checkerboard) order so the rows can
//!   be solved in parallel. Colours are mixed in linear light, as light mixes.
//! * **Despeckle** keeps the alpha's large connected parts and fades out everything a picture-
//!   scaled distance away from them (a soft garbage matte), so stray specks and floating
//!   islands vanish while the subject's own soft edge stays as it was.
//! * **Despill** pulls the screen channel down to the other two where it stands above them and
//!   gives half of what it took to each of the others, so brightness holds.

use photocraft_color::convert::{linear_to_srgb, srgb_to_linear};

use super::box_mean;

/// Foreground estimation settings (the paper's reference implementation's defaults).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForegroundParams {
    /// Smoothness everywhere (`ε`).
    pub regularization: f32,
    /// Extra smoothness where alpha changes (`ω`): F and B flow along the edge, not across it.
    pub gradient_weight: f32,
    /// Sweeps on levels no larger than `small_size` on either side...
    pub small_iterations: usize,
    /// ...and on the larger ones.
    pub big_iterations: usize,
    pub small_size: usize,
}

impl Default for ForegroundParams {
    fn default() -> Self {
        Self { regularization: 1e-5, gradient_weight: 1.0, small_iterations: 10, big_iterations: 2, small_size: 32 }
    }
}

/// The foreground `F` and background `B` colours behind every pixel of `img` (linear RGB, 0–1)
/// under `alpha`, so that `img ≈ αF + (1 − α)B` with both smooth. Values stay within 0–1.
pub fn estimate_foreground(img: &[[f32; 3]], alpha: &[f32], w: usize, h: usize, p: &ForegroundParams) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let n = w * h;
    if n == 0 || img.len() < n || alpha.len() < n {
        return (Vec::new(), Vec::new());
    }
    // The coarsest level is one pixel: the mean colour of the clearly opaque and the clearly
    // clear pixels.
    let (mut f0, mut b0, mut nf, mut nb) = ([0.0f64; 3], [0.0f64; 3], 0.0f64, 0.0f64);
    for (q, a) in img[..n].iter().zip(&alpha[..n]) {
        if *a > 0.9 {
            (0..3).for_each(|c| f0[c] += f64::from(q[c]));
            nf += 1.0;
        }
        if *a < 0.1 {
            (0..3).for_each(|c| b0[c] += f64::from(q[c]));
            nb += 1.0;
        }
    }
    let mut fb: Vec<[f32; 6]> = vec![[
        (f0[0] / (nf + 1e-5)) as f32,
        (f0[1] / (nf + 1e-5)) as f32,
        (f0[2] / (nf + 1e-5)) as f32,
        (b0[0] / (nb + 1e-5)) as f32,
        (b0[1] / (nb + 1e-5)) as f32,
        (b0[2] / (nb + 1e-5)) as f32,
    ]];
    let (mut pw, mut ph) = (1usize, 1usize);
    let levels = (w.max(h) as f64).log2().ceil() as i32;
    for level in 0..=levels {
        // Each level is `size^(level / levels)` on a side, so the sides grow together.
        let (lw, lh) = if levels == 0 {
            (w, h)
        } else {
            let t = f64::from(level) / f64::from(levels);
            (((w as f64).powf(t).round() as usize).max(1), ((h as f64).powf(t).round() as usize).max(1))
        };
        let (im, al) = if (lw, lh) == (w, h) { (img[..n].to_vec(), alpha[..n].to_vec()) } else { (nearest(img, w, h, lw, lh), nearest(alpha, w, h, lw, lh)) };
        let mut cur = nearest(&fb, pw, ph, lw, lh);
        let iterations = if lw <= p.small_size && lh <= p.small_size { p.small_iterations } else { p.big_iterations };
        let mut next = cur.clone();
        let par = lw * lh >= 16384;
        for _ in 0..iterations {
            // Red pixels from black neighbours, then black from red: each half is independent.
            for colour in 0..2 {
                sweep(&cur, &mut next, &im, &al, lw, lh, colour, p, par);
                std::mem::swap(&mut cur, &mut next);
            }
        }
        fb = cur;
        (pw, ph) = (lw, lh);
    }
    fb.into_iter().map(|v| ([v[0], v[1], v[2]], [v[3], v[4], v[5]])).unzip()
}

/// Nearest-neighbour resize (the reference's: source `x · w_src / w_dst`).
fn nearest<T: Copy + Send + Sync>(src: &[T], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(dw * dh);
    for y in 0..dh {
        let sy = (y * sh / dh).min(sh - 1);
        for x in 0..dw {
            out.push(src[sy * sw + (x * sw / dw).min(sw - 1)]);
        }
    }
    out
}

/// One half-sweep: every pixel of checkerboard `colour` solves its 2 × 2 system for F and B
/// from its own colour and its four neighbours (edge-clamped); the rest are copied.
#[allow(clippy::too_many_arguments)]
fn sweep(cur: &[[f32; 6]], next: &mut [[f32; 6]], im: &[[f32; 3]], al: &[f32], w: usize, h: usize, colour: usize, p: &ForegroundParams, par: bool) {
    crate::par::chunks_mut(next, w, par, |y, row| {
        let (up, down) = (y.saturating_sub(1), (y + 1).min(h - 1));
        for (x, out) in row.iter_mut().enumerate() {
            let k = y * w + x;
            if (x + y) % 2 != colour {
                *out = cur[k];
                continue;
            }
            let a0 = al[k];
            let a1 = 1.0 - a0;
            let (mut a00, a01, mut a11) = (a0 * a0, a0 * a1, a1 * a1);
            let q = im[k];
            let mut b = [a0 * q[0], a0 * q[1], a0 * q[2], a1 * q[0], a1 * q[1], a1 * q[2]];
            for j in [y * w + x.saturating_sub(1), y * w + (x + 1).min(w - 1), up * w + x, down * w + x] {
                let d = p.regularization + p.gradient_weight * (a0 - al[j]).abs();
                a00 += d;
                a11 += d;
                for (bc, v) in b.iter_mut().zip(cur[j]) {
                    *bc += d * v;
                }
            }
            let inv = 1.0 / (a00 * a11 - a01 * a01);
            let (m00, m01, m11) = (inv * a11, -inv * a01, inv * a00);
            for c in 0..3 {
                out[c] = (m00 * b[c] + m01 * b[c + 3]).clamp(0.0, 1.0);
                out[c + 3] = (m01 * b[c] + m11 * b[c + 3]).clamp(0.0, 1.0);
            }
        }
    });
}

/// How far around the soft pixels [`unmix_rgba8`] reads, at least.
const UNMIX_MARGIN: usize = 16;

/// A cut-out's soft edge without the old background showing through: every pixel of `px`
/// (sRGB-encoded RGBA8, `w × h`) whose `alpha` is neither 0 nor 1 takes the estimated
/// foreground colour ([`estimate_foreground`], worked in linear light). Its own alpha byte and
/// every solid or clear pixel stay as they were. Only the box around the soft pixels (and a
/// margin) is solved.
pub fn unmix_rgba8(px: &mut [u8], w: usize, h: usize, alpha: &[f32]) {
    let n = w * h;
    if n == 0 || px.len() < n * 4 || alpha.len() < n {
        return;
    }
    let soft = |a: f32| a > 0.002 && a < 0.998;
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for y in 0..h {
        for x in 0..w {
            if soft(alpha[y * w + x]) {
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
            }
        }
    }
    if x0 >= x1 {
        return;
    }
    let margin = UNMIX_MARGIN.max((x1 - x0).max(y1 - y0) / 16);
    let (x0, y0, x1, y1) = (x0.saturating_sub(margin), y0.saturating_sub(margin), (x1 + margin).min(w), (y1 + margin).min(h));
    let (cw, ch) = (x1 - x0, y1 - y0);
    let lut: Vec<f32> = (0..256).map(|v| srgb_to_linear(v as f32 / 255.0)).collect();
    let mut img = Vec::with_capacity(cw * ch);
    let mut al = Vec::with_capacity(cw * ch);
    for y in y0..y1 {
        for x in x0..x1 {
            let k = y * w + x;
            img.push([lut[px[k * 4] as usize], lut[px[k * 4 + 1] as usize], lut[px[k * 4 + 2] as usize]]);
            al.push(alpha[k].clamp(0.0, 1.0));
        }
    }
    let (fg, _) = estimate_foreground(&img, &al, cw, ch, &ForegroundParams::default());
    for y in y0..y1 {
        for x in x0..x1 {
            let (k, j) = (y * w + x, (y - y0) * cw + x - x0);
            if soft(al[j]) {
                for c in 0..3 {
                    px[k * 4 + c] = (linear_to_srgb(fg[j][c]) * 255.0).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
}

/// Despeckle settings. An island is dropped only when it is small against the picture *and*
/// against the alpha's largest part, so a legitimately separate part (a held cup) stays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DespeckleParams {
    /// Islands under this share of the picture's pixels may go...
    pub min_area_fraction: f32,
    /// ...and never fewer than this many pixels...
    pub min_area_floor: usize,
    /// ...but only if they are also under this share of the largest part.
    pub max_share_of_largest: f32,
    /// How far from what is kept the alpha is left untouched, as a share of the picture's
    /// longer side...
    pub radius_fraction: f32,
    /// ...and at least this many pixels.
    pub radius_floor: f32,
}

impl Default for DespeckleParams {
    fn default() -> Self {
        Self { min_area_fraction: 0.0005, min_area_floor: 16, max_share_of_largest: 0.05, radius_fraction: 0.02, radius_floor: 8.0 }
    }
}

/// Clears stray specks and floating islands from a soft alpha (`w × h`, 0–1), in place:
/// the alpha is cut at 0.5 into 8-connected parts, the small ones are dropped (see
/// [`DespeckleParams`]), what is kept is grown by the radius plus a soft falloff and blurred,
/// and the alpha is multiplied by that garbage matte. Within the radius of a kept part the
/// matte is exactly 1, so the subject's own soft edge is untouched. An alpha with nothing at
/// or above 0.5 is left alone. Returns how many islands were dropped.
pub fn despeckle(alpha: &mut [f32], w: usize, h: usize, p: &DespeckleParams) -> usize {
    despeckle_inner(alpha, w, h, p, None)
}

/// [`despeckle`], but a part touching any `keep` pixel is never dropped however small it is
/// (a detector's own mask: only what matting added around it can go).
pub fn despeckle_keeping(alpha: &mut [f32], w: usize, h: usize, p: &DespeckleParams, keep: &[bool]) -> usize {
    despeckle_inner(alpha, w, h, p, Some(keep).filter(|k| k.len() >= w * h))
}

fn despeckle_inner(alpha: &mut [f32], w: usize, h: usize, p: &DespeckleParams, keep: Option<&[bool]>) -> usize {
    let n = w * h;
    if n == 0 || alpha.len() < n {
        return 0;
    }
    // Everything happens inside the box of what is not clear, grown by the matte's reach.
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for y in 0..h {
        for x in 0..w {
            if alpha[y * w + x] > 0.0 {
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
            }
        }
    }
    if x0 >= x1 {
        return 0;
    }
    let radius = (w.max(h) as f32 * p.radius_fraction).max(p.radius_floor).ceil() as usize;
    // Three box passes of radius `rb` approximate a Gaussian with σ ≈ rb and reach 3 · rb, so
    // growing by `radius + 3 · rb` leaves the matte at exactly 1 within `radius`.
    let rb = (radius / 2).max(1);
    let grow = radius + 3 * rb;
    let (x0, y0, x1, y1) = (x0.saturating_sub(grow), y0.saturating_sub(grow), (x1 + grow).min(w), (y1 + grow).min(h));
    let (cw, ch) = (x1 - x0, y1 - y0);
    let on: Vec<bool> = (y0..y1).flat_map(|y| (x0..x1).map(move |x| y * w + x)).map(|k| alpha[k] >= 0.5).collect();
    let (labels, sizes) = crate::segment::components(&on, cw, ch);
    let largest = sizes.iter().skip(1).copied().max().unwrap_or(0);
    if largest == 0 {
        return 0;
    }
    let small = ((n as f64 * f64::from(p.min_area_fraction)).ceil() as usize).max(p.min_area_floor);
    let share = (largest as f64 * f64::from(p.max_share_of_largest)).ceil() as usize;
    let mut dropped: Vec<bool> = sizes.iter().map(|s| *s < small && *s < share).collect();
    if let Some(keep) = keep {
        for (j, l) in labels.iter().enumerate() {
            if *l != 0 && keep[(y0 + j / cw) * w + x0 + j % cw] {
                dropped[*l as usize] = false;
            }
        }
    }
    let removed = dropped.iter().skip(1).filter(|d| **d).count();
    let kept: Vec<f32> = labels.iter().map(|l| if *l != 0 && !dropped[*l as usize] { 1.0 } else { 0.0 }).collect();
    let mut matte = super::morph(&kept, cw, ch, grow, true);
    for _ in 0..3 {
        matte = box_mean(&matte, cw, ch, rb);
    }
    for y in 0..h {
        for x in 0..w {
            let inside = (x0..x1).contains(&x) && (y0..y1).contains(&y);
            let m = if inside { matte[(y - y0) * cw + x - x0] } else { 0.0 };
            let a = &mut alpha[y * w + x];
            // Box sums leave rounding dust on the plateau.
            if m < 0.999 {
                *a *= m.max(0.0);
            }
        }
    }
    removed
}

/// Which backing screen a picture was shot on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    Green,
    Blue,
}

/// What the screen channel is held down to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpillLimit {
    /// The mean of the other two channels (the stronger, usual choice).
    Average,
    /// The larger of the other two (gentler: yellows and cyans keep more of the screen colour).
    Max,
}

/// One pixel despilled (any RGB encoding; the values are used as given): where the screen
/// channel stands above the limit it comes down to it and each other channel gains half of
/// what it lost, so `r + g + b` is unchanged. `strength` (0–1) blends from the original.
pub fn despill_pixel(rgb: [f32; 3], screen: Screen, limit: SpillLimit, strength: f32) -> [f32; 3] {
    let (s, o1, o2) = match screen {
        Screen::Green => (1, 0, 2),
        Screen::Blue => (2, 0, 1),
    };
    let cap = match limit {
        SpillLimit::Average => (rgb[o1] + rgb[o2]) * 0.5,
        SpillLimit::Max => rgb[o1].max(rgb[o2]),
    };
    let spill = (rgb[s] - cap) * strength.clamp(0.0, 1.0);
    // Also no change when a sample is NaN.
    if spill.is_nan() || spill <= 0.0 {
        return rgb;
    }
    let mut out = rgb;
    out[s] -= spill;
    out[o1] += spill * 0.5;
    out[o2] += spill * 0.5;
    out
}

/// [`despill_pixel`] over straight RGBA pixels (alpha untouched).
pub fn despill(px: &mut [[f32; 4]], screen: Screen, limit: SpillLimit, strength: f32) {
    for q in px.iter_mut() {
        let [r, g, b] = despill_pixel([q[0], q[1], q[2]], screen, limit, strength);
        (q[0], q[1], q[2]) = (r, g, b);
    }
}

/// The screen a picture was shot on, from its pixels where the rough `alpha` says background
/// (below 0.3): blue when their mean blue is clearly (0.05) above their mean green, else
/// green, as it is when too little background (under 1% of the picture) shows.
pub fn detect_screen(rgb: &[[f32; 3]], alpha: &[f32]) -> Screen {
    let (mut g, mut b, mut count) = (0.0f64, 0.0f64, 0usize);
    for (q, a) in rgb.iter().zip(alpha) {
        if *a < 0.3 {
            g += f64::from(q[1]);
            b += f64::from(q[2]);
            count += 1;
        }
    }
    if count == 0 || (count as f64) < rgb.len().min(alpha.len()) as f64 * 0.01 {
        return Screen::Green;
    }
    let (g, b) = (g / count as f64, b / count as f64);
    if (g - b).abs() < 0.05 || g >= b { Screen::Green } else { Screen::Blue }
}

#[cfg(test)]
mod tests;
