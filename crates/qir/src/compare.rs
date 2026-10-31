//! `qir compare`: the parity and regression gates, in Rust. These replace the Python helpers
//! that used to sit in `scripts/` and `tools/` (`compare_png.py`, `compare_latent.py`,
//! `compare_dit.py`, `layered_compare.py`, `lora_compare.py`, `qwen21_ref_compare.py`), so
//! checking a run against a reference needs nothing but this binary. Each prints one line or
//! one JSON object per item and returns whether the gate passed.

use anyhow::{bail, ensure, Context};
use candle_core::{DType, Device, Tensor};
use std::path::Path;

use crate::Result;

/// PSNR (dB, over 8-bit values) and the largest byte difference between two same-size PNGs.
/// RGB only when both pictures are opaque everywhere, RGBA otherwise.
pub fn png(a: &Path, b: &Path) -> Result<(f64, u8)> {
    let (a, b) = (rgba(a)?, rgba(b)?);
    ensure!(a.dimensions() == b.dimensions(), "size mismatch {:?} vs {:?}", a.dimensions(), b.dimensions());
    let opaque = a.pixels().chain(b.pixels()).all(|p| p[3] == 255);
    let channels = if opaque { 3 } else { 4 };
    let (mut sum, mut max, mut n) = (0f64, 0u8, 0usize);
    for (p, q) in a.pixels().zip(b.pixels()) {
        for c in 0..channels {
            let d = p[c].abs_diff(q[c]);
            sum += f64::from(d) * f64::from(d);
            max = max.max(d);
            n += 1;
        }
    }
    let mse = sum / n as f64;
    let psnr = if mse == 0. { f64::INFINITY } else { 10. * (255f64.powi(2) / mse).log10() };
    Ok((psnr, max))
}

/// Cosine, max |a-b| and MSE of two flat vectors.
pub fn stats(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let cos = dot / (norm(a) * norm(b) + 1e-12);
    let maxabs = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0., f64::max);
    let mse = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64;
    (cos, maxabs, mse)
}

/// One named tensor of a safetensors file, as f64 with its shape.
pub fn tensor(path: &Path, key: &str) -> Result<(Vec<usize>, Vec<f64>)> {
    let mut all = candle_core::safetensors::load(path, &Device::Cpu).with_context(|| format!("reading {}", path.display()))?;
    let t: Tensor = all.remove(key).with_context(|| format!("{}: no `{key}` tensor", path.display()))?;
    let shape = t.dims().to_vec();
    Ok((shape, t.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?))
}

/// The `dit-forward` oracle gate: overall cosine and MSE of the `output` tensors, and the mean
/// per-token cosine over the last `image_tokens` rows. A leading batch of 1 is dropped.
pub fn dit(ours: &Path, oracle: &Path, image_tokens: usize) -> Result<(f64, f64, f64)> {
    let (sa, a) = tensor(ours, "output")?;
    let (sb, b) = tensor(oracle, "output")?;
    let rows = |s: &[usize]| if s.len() == 3 { &s[1..] } else { s }.to_vec();
    let (sa, sb) = (rows(&sa), rows(&sb));
    ensure!(sa == sb && sa.len() == 2, "shape mismatch: ours {sa:?} vs oracle {sb:?}");
    let (cos, _, mse) = stats(&a, &b);
    let width = sa[1];
    let tokens = image_tokens.min(sa[0]);
    let start = (sa[0] - tokens) * width;
    let per_token: f64 = a[start..].chunks(width).zip(b[start..].chunks(width)).map(|(x, y)| stats(x, y).0).sum::<f64>() / tokens as f64;
    Ok((cos, mse, per_token))
}

/// One Layered layer against its reference: RGB mean error over the pixels visible in either
/// (alpha above `visible`), over all pixels, alpha mean error and the alpha>127 mask IoU.
#[derive(Debug, serde::Serialize)]
pub struct Layer {
    pub layer: usize,
    pub rgb_mae: f64,
    pub rgb_mae_all: f64,
    pub alpha_mae: f64,
    pub alpha_iou: f64,
    pub identical: bool,
}

pub fn layer(index: usize, a: &Path, b: &Path, visible: u8, pad: bool) -> Result<Layer> {
    let (mut a, mut b) = (rgba(a)?, rgba(b)?);
    if pad {
        // Top-left placement on a common transparent canvas; never a resize.
        let (w, h) = (a.width().max(b.width()), a.height().max(b.height()));
        let canvas = |img: &image::RgbaImage| {
            let mut out = image::RgbaImage::new(w, h);
            image::imageops::replace(&mut out, img, 0, 0);
            out
        };
        (a, b) = (canvas(&a), canvas(&b));
    }
    ensure!(a.dimensions() == b.dimensions(), "layer {index}: size mismatch {:?} vs {:?}", a.dimensions(), b.dimensions());
    let (mut rgb, mut rgb_n, mut rgb_all, mut alpha, mut inter, mut union) = (0f64, 0usize, 0f64, 0f64, 0usize, 0usize);
    for (p, q) in a.pixels().zip(b.pixels()) {
        let d: f64 = (0..3).map(|c| f64::from(p[c].abs_diff(q[c]))).sum();
        rgb_all += d;
        if p[3] > visible || q[3] > visible {
            rgb += d;
            rgb_n += 3;
        }
        alpha += f64::from(p[3].abs_diff(q[3]));
        let (ma, mb) = (p[3] > 127, q[3] > 127);
        inter += usize::from(ma && mb);
        union += usize::from(ma || mb);
    }
    let n = (a.width() * a.height()) as f64;
    Ok(Layer {
        layer: index,
        rgb_mae: if rgb_n == 0 { 0. } else { rgb / rgb_n as f64 },
        rgb_mae_all: rgb_all / (3. * n),
        alpha_mae: alpha / n,
        alpha_iou: if union == 0 { 1. } else { inter as f64 / union as f64 },
        identical: a == b,
    })
}

/// One raw `.rgba` artifact against its namesake: RGB and all-byte mean error, largest byte.
#[derive(Debug, serde::Serialize)]
pub struct Raw {
    pub image: String,
    pub identical: bool,
    pub mae_rgb_bytes: f64,
    pub mae_bytes: f64,
    pub max_byte: u8,
    pub pass: bool,
}

/// Every `*.rgba` in `after` that has a namesake in `before`; `limit` is the allowed RGB mean
/// error in bytes. Fails on a missing pair only when nothing matched at all.
pub fn raw_dirs(before: &Path, after: &Path, limit: f64) -> Result<Vec<Raw>> {
    let mut names: Vec<_> = std::fs::read_dir(after)
        .with_context(|| format!("reading {}", after.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "rgba"))
        .collect();
    names.sort();
    let mut out = Vec::new();
    for path in names {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let Ok(a) = std::fs::read(before.join(&name)) else { continue };
        let b = std::fs::read(&path)?;
        ensure!(!a.is_empty() && a.len() == b.len() && a.len() % 4 == 0, "{name}: mismatched or empty images");
        let (mut rgb, mut all, mut max) = (0f64, 0f64, 0u8);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            let d = x.abs_diff(*y);
            all += f64::from(d);
            if i % 4 != 3 {
                rgb += f64::from(d);
            }
            max = max.max(d);
        }
        let mae_rgb_bytes = rgb / (a.len() / 4 * 3) as f64;
        out.push(Raw { image: name, identical: a == b, mae_rgb_bytes, mae_bytes: all / a.len() as f64, max_byte: max, pass: mae_rgb_bytes <= limit });
    }
    if out.is_empty() {
        bail!("no matching *.rgba files in {} and {}", before.display(), after.display());
    }
    Ok(out)
}

fn rgba(path: &Path) -> Result<image::RgbaImage> {
    Ok(image::open(path).with_context(|| format!("reading {}", path.display()))?.to_rgba8())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_of_equal_and_opposite_vectors() {
        let a = [1., 2., 3.];
        let (cos, maxabs, mse) = stats(&a, &a);
        assert!((cos - 1.).abs() < 1e-9 && maxabs == 0. && mse == 0.);
        let b = [-1., -2., -3.];
        let (cos, maxabs, mse) = stats(&a, &b);
        assert!((cos + 1.).abs() < 1e-9 && maxabs == 6. && (mse - 56. / 3.).abs() < 1e-9);
    }

    #[test]
    fn png_psnr_and_layer_and_raw() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("qir-compare-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("before"))?;
        std::fs::create_dir_all(dir.join("after"))?;
        let mut a = image::RgbaImage::from_pixel(4, 4, image::Rgba([10, 20, 30, 255]));
        a.save(dir.join("a.png"))?;
        assert_eq!(png(&dir.join("a.png"), &dir.join("a.png"))?, (f64::INFINITY, 0));
        a.put_pixel(0, 0, image::Rgba([12, 20, 30, 255]));
        a.save(dir.join("b.png"))?;
        // One channel off by 2 out of 4·4·3 opaque values: MSE 4/48.
        let (psnr, max) = png(&dir.join("a.png"), &dir.join("b.png"))?;
        assert_eq!(max, 2);
        assert!((psnr - 10. * (255f64.powi(2) * 12.).log10()).abs() < 1e-9);
        let l = layer(1, &dir.join("a.png"), &dir.join("b.png"), 127, false)?;
        assert!(!l.identical && l.alpha_iou == 1. && (l.rgb_mae - 2. / 48.).abs() < 1e-12);

        std::fs::write(dir.join("before/x.rgba"), [0u8, 0, 0, 255, 9, 9, 9, 255])?;
        std::fs::write(dir.join("after/x.rgba"), [3u8, 0, 0, 0, 9, 9, 9, 255])?;
        let r = raw_dirs(&dir.join("before"), &dir.join("after"), 1.)?;
        assert_eq!(r.len(), 1);
        assert!((r[0].mae_rgb_bytes - 0.5).abs() < 1e-12 && r[0].max_byte == 255 && r[0].pass);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
