//! ComfyUI image and soft-mask preparation for Qwen-Image-Edit-2511.
use crate::Result;
use candle_core::{DType, Device, Tensor};

/// FluxKontext's fixed (width, height) choices, in ComfyUI order.
pub const KONTEXT_RESOLUTIONS: [(usize, usize); 17] = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
];

fn dimensions(pixels: &Tensor) -> Result<(usize, usize)> {
    let (h, w, c) = pixels.dims3()?;
    anyhow::ensure!(h > 0 && w > 0 && c == 3 && pixels.dtype() == DType::F32 && matches!(pixels.device(), Device::Cpu), "expected nonempty CPU f32 HWC RGB");
    Ok((h, w))
}

/// EditPlus VAE reference pixels: about 1024^2, ties-to-even multiples of eight, area pooling.
pub fn resize_vae_reference(pixels: &Tensor) -> Result<Tensor> {
    let (h, w) = dimensions(pixels)?;
    let scale = (1024.0 * 1024.0 / (h as f64 * w as f64)).sqrt();
    let oh = (h as f64 * scale / 8.0).round_ties_even() as usize * 8;
    let ow = (w as f64 * scale / 8.0).round_ties_even() as usize * 8;
    anyhow::ensure!(oh > 0 && ow > 0, "invalid EditPlus aspect ratio");
    let src = pixels.to_vec3::<f32>()?;
    let mut out = vec![0.; oh * ow * 3];
    for y in 0..oh {
        let (a, b) = (y * h / oh, ((y + 1) * h).div_ceil(oh));
        for x in 0..ow {
            let (c, d) = (x * w / ow, ((x + 1) * w).div_ceil(ow));
            for ch in 0..3 {
                let mut sum = 0f32;
                for row in &src[a..b] {
                    for p in &row[c..d] {
                        sum += p[ch];
                    }
                }
                out[(y * ow + x) * 3 + ch] = sum / ((b - a) * (d - c)) as f32;
            }
        }
    }
    Ok(Tensor::from_vec(out, (oh, ow, 3), pixels.device())?)
}

/// VAEEncode removes edge pixels to spatial compression multiples; it never resizes them.
pub fn crop_vae_target(pixels: &Tensor) -> Result<Tensor> {
    let (h, w) = dimensions(pixels)?;
    anyhow::ensure!(h >= 8 && w >= 8, "target is smaller than VAE compression");
    Ok(pixels.narrow(0, (h % 8) / 2, h / 8 * 8)?.narrow(1, (w % 8) / 2, w / 8 * 8)?.contiguous()?)
}

/// [`kontext_geometry`]'s result: the chosen (height, width) and the crop (x, y, width, height).
pub type KontextGeometry = ((usize, usize), (usize, usize, usize, usize));

/// Chosen size and symmetric crop (x, y, width, height), including Python's tie rounding.
pub fn kontext_geometry(h: usize, w: usize) -> Result<KontextGeometry> {
    anyhow::ensure!(h > 0 && w > 0, "empty Kontext image");
    let aspect = w as f64 / h as f64;
    let &(ow, oh) = KONTEXT_RESOLUTIONS
        .iter()
        .min_by(|&&(aw, ah), &&(bw, bh)| {
            (aspect - aw as f64 / ah as f64).abs().total_cmp(&(aspect - bw as f64 / bh as f64).abs()).then(aw.cmp(&bw)).then(ah.cmp(&bh))
        })
        .ok_or_else(|| anyhow::anyhow!("empty resolution table"))?;
    let new_aspect = ow as f64 / oh as f64;
    let x = if aspect > new_aspect { ((w as f64 - w as f64 * (new_aspect / aspect)) / 2.).round_ties_even() as usize } else { 0 };
    let y = if aspect < new_aspect { ((h as f64 - h as f64 * (aspect / new_aspect)) / 2.).round_ties_even() as usize } else { 0 };
    anyhow::ensure!(2 * x < w && 2 * y < h, "empty Kontext crop");
    Ok(((ow, oh), (x, y, w - 2 * x, h - 2 * y)))
}

// Pillow's RGB resampler normalizes Lanczos coefficients to signed 22-bit fixed point.
fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 3.0 * filter_scale;
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5) as isize).max(0) as usize;
            let hi = ((center + support + 0.5) as usize).min(input);
            let weights: Vec<f64> = (lo..hi)
                .map(|j| {
                    let x = (j as f64 - center + 0.5) / filter_scale;
                    if x == 0. {
                        1.
                    } else if !(-3.0..3.0).contains(&x) {
                        0.
                    } else {
                        let p = x * std::f64::consts::PI;
                        (p.sin() / p) * ((p / 3.).sin() / (p / 3.))
                    }
                })
                .collect();
            let sum: f64 = weights.iter().sum();
            (lo, weights.into_iter().map(|v| (v / sum * (1u64 << 22) as f64).round() as i32).collect())
        })
        .collect()
}

/// Kontext's center crop and Pillow-compatible, quantized RGB Lanczos resize.
pub fn resize_kontext(pixels: &Tensor) -> Result<Tensor> {
    let (h, w) = dimensions(pixels)?;
    let ((ow, oh), (x, y, cw, ch)) = kontext_geometry(h, w)?;
    let crop = pixels.narrow(0, y, ch)?.narrow(1, x, cw)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let src: Vec<u8> = crop.iter().map(|v| (255.0 * v).clamp(0., 255.) as u8).collect();
    let horizontal = coefficients(cw, ow);
    let vertical = coefficients(ch, oh);
    let clip = |s: i64| ((s + (1 << 21)) >> 22).clamp(0, 255) as u8;
    let mut tmp = vec![0u8; ch * ow * 3];
    for y in 0..ch {
        for (x, (lo, ws)) in horizontal.iter().enumerate() {
            for c in 0..3 {
                let s = ws.iter().enumerate().map(|(j, k)| src[(y * cw + lo + j) * 3 + c] as i64 * *k as i64).sum();
                tmp[(y * ow + x) * 3 + c] = clip(s);
            }
        }
    }
    let mut out = vec![0f32; oh * ow * 3];
    for (y, (lo, ws)) in vertical.iter().enumerate() {
        for x in 0..ow {
            for c in 0..3 {
                let s = ws.iter().enumerate().map(|(j, k)| tmp[((lo + j) * ow + x) * 3 + c] as i64 * *k as i64).sum();
                out[(y * ow + x) * 3 + c] = clip(s) as f32 / 255.;
            }
        }
    }
    Ok(Tensor::from_vec(out, (oh, ow, 3), pixels.device())?)
}

/// SetLatentNoiseMask uses PNG red, preserving soft values; output is BCHW or BC1HW.
pub fn latent_noise_mask(png: &image::RgbImage, shape: &[usize]) -> Result<Tensor> {
    anyhow::ensure!(shape.len() == 4 || (shape.len() == 5 && shape[2] == 1), "expected BCHW or BC1HW mask shape");
    anyhow::ensure!(shape.iter().all(|n| *n > 0) && png.width() > 0 && png.height() > 0, "empty mask");
    let (h, w) = (png.height() as usize, png.width() as usize);
    let (oh, ow) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let mut plane = Vec::with_capacity(oh * ow);
    for y in 0..oh {
        let sy = ((y as f32 + 0.5) * (h as f64 / oh as f64) as f32 - 0.5).max(0.);
        let ya = (sy as usize).min(h - 1);
        let yb = (ya + 1).min(h - 1);
        let dy = sy - ya as f32;
        for x in 0..ow {
            let sx = ((x as f32 + 0.5) * (w as f64 / ow as f64) as f32 - 0.5).max(0.);
            let xa = (sx as usize).min(w - 1);
            let xb = (xa + 1).min(w - 1);
            let dx = sx - xa as f32;
            let at = |y, x| png.get_pixel(x as u32, y as u32)[0] as f32 / 255.;
            let top = (1. - dx) * at(ya, xa) + dx * at(ya, xb);
            let bottom = (1. - dx) * at(yb, xa) + dx * at(yb, xb);
            // Torch's 2D kernel sums four weighted corners; its 3D kernel nests lerps.
            plane.push(if shape.len() == 4 {
                ((1. - dy) * (1. - dx) * at(ya, xa) + (1. - dy) * dx * at(ya, xb)) + dy * (1. - dx) * at(yb, xa) + dy * dx * at(yb, xb)
            } else {
                (1. - dy) * top + dy * bottom
            });
        }
    }
    let mut out = Vec::with_capacity(shape.iter().product());
    for _ in 0..shape[0] * shape[1] {
        out.extend_from_slice(&plane);
    }
    Ok(Tensor::from_vec(out, shape, &Device::Cpu)?)
}
