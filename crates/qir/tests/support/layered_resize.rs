//! PhotoCraft's Lanczos-3 resampler, retained here so GPU parity uses the app's pixels.
use image::RgbaImage;

pub fn resize(src: &RgbaImage, w: u32, h: u32) -> RgbaImage {
    let data = resample_u8(src.as_raw(), src.width() as usize, src.height() as usize, 4, w as usize, h as usize, LANCZOS);
    RgbaImage::from_raw(w, h, data).unwrap()
}

fn lanczos3(x: f32) -> f32 {
    let x = x.abs();
    if x < 1e-6 {
        1.0
    } else if x >= 3.0 {
        0.0
    } else {
        let px = std::f32::consts::PI * x;
        3.0 * px.sin() * (px / 3.0).sin() / (px * px)
    }
}

/// A resampling kernel and its radius in source pixels (at 1:1).
#[derive(Clone, Copy)]
struct Filter {
    kernel: fn(f32) -> f32,
    radius: f32,
}
const LANCZOS: Filter = Filter { kernel: lanczos3, radius: 3.0 };

/// Per-output-index taps of a separable resampler: (first source index, normalised weights).
fn taps(src_len: usize, dst_len: usize, f: Filter) -> Vec<(usize, Vec<f32>)> {
    let scale = src_len as f32 / dst_len as f32;
    let widen = scale.max(1.0);
    let support = f.radius * widen;
    (0..dst_len)
        .map(|i| {
            let centre = (i as f32 + 0.5) * scale - 0.5;
            let lo = ((centre - support).floor().max(0.0)) as usize;
            let hi = ((centre + support).ceil() as usize).min(src_len.saturating_sub(1));
            let mut weights: Vec<f32> = (lo..=hi).map(|j| (f.kernel)((j as f32 - centre) / widen)).collect();
            let sum: f32 = weights.iter().sum();
            if sum.abs() > 1e-6 {
                for w in &mut weights {
                    *w /= sum;
                }
            } else if let Some(w) = weights.first_mut() {
                *w = 1.0;
            }
            (lo, weights)
        })
        .collect()
}

/// Separable resample of interleaved 8-bit samples (`ch` per pixel) from `sw`×`sh` to `dw`×`dh`.
fn resample_u8(src: &[u8], sw: usize, sh: usize, ch: usize, dw: usize, dh: usize, f: Filter) -> Vec<u8> {
    let xt = taps(sw, dw, f);
    let yt = taps(sh, dh, f);
    // Horizontal pass into f32 rows, then vertical.
    let mut mid = vec![0.0f32; dw * sh * ch];
    for y in 0..sh {
        let row = &src[y * sw * ch..((y + 1) * sw * ch).min(src.len())];
        for (x, (lo, ws)) in xt.iter().enumerate() {
            let out = &mut mid[(y * dw + x) * ch..(y * dw + x + 1) * ch];
            for (k, w) in ws.iter().enumerate() {
                let start = ((lo + k) * ch).min(row.len().saturating_sub(ch));
                for (o, p) in out.iter_mut().zip(row.iter().skip(start)) {
                    *o += w * f32::from(*p);
                }
            }
        }
    }
    let mut dst = vec![0u8; dw * dh * ch];
    for (y, (lo, ws)) in yt.iter().enumerate() {
        for x in 0..dw {
            for c in 0..ch {
                let mut acc = 0.0f32;
                for (k, w) in ws.iter().enumerate() {
                    acc += w * mid.get(((lo + k) * dw + x) * ch + c).copied().unwrap_or(0.0);
                }
                if let Some(d) = dst.get_mut((y * dw + x) * ch + c) {
                    *d = (acc + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    dst
}
