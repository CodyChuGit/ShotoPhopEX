//! Picture to 3D gaussian splat with Idol, following AusBoss's "Idol Picture to 3D" ComfyUI
//! workflow: the object cut out (its alpha), squared and centred on black at 1024
//! (`TripoSplatPreprocessImage`), read by DINOv3 and the FLUX.2 VAE
//! (`TripoSplatConditioning`), 50 Euler steps of the flow model at CFG 4 (shift 3, the
//! `simple` schedule, torch's CPU noise for the seed), then the octree and gaussian decoders
//! (`VAEDecodeTripoSplat`). The four models take turns on the card: the encoders load, read
//! the picture and go before the flow model loads, and the decoder loads after it goes.

use std::path::{Path, PathBuf};

use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use image::RgbaImage;

use super::decoder::{IdolDecoder, Splat, GAUSSIANS_PER_POINT};
use super::dino::Dino;
use super::dit::{IdolDit, CAM_CHANNELS, LATENT_CHANNELS, TOKENS};
use crate::comfy::{FlowSchedule, Mt19937, Shift};
use crate::moss::vae::Flux2Encoder;
use crate::moss::weights::Weights;
use crate::pipeline::Cancelled;
use crate::Result;

/// The side the object is squared to.
pub const SIZE: usize = 1024;
/// The workflow's alpha erosion (`erode_radius`).
pub const ERODE: usize = 6;
/// `ModelSamplingDiscreteFlow` for TripoSplat (`sampling_settings.shift = 3`, multiplier 1000).
pub const SHIFT: Shift = Shift::Snr { shift: 3.0, multiplier: 1000.0, timesteps: 1000 };

/// Idol's files in a ComfyUI-layout `models` folder (Comfy-Org's names).
#[derive(Clone, Debug)]
pub struct IdolFiles {
    pub dit: PathBuf,
    pub decoder: PathBuf,
    pub dino: PathBuf,
    /// A FLUX.2 VAE with its encoder in diffusers' layout (`flux2-vae_small`).
    pub vae: PathBuf,
}

/// The first of `names` under `root/<dir>` for each part, or `None` if a part is missing.
pub fn find_files(root: &Path) -> Option<IdolFiles> {
    let pick = |dirs: &[&str], names: &[&str]| -> Option<PathBuf> {
        dirs.iter().flat_map(|d| names.iter().map(move |n| root.join(d).join(n))).find(|p| p.is_file())
    };
    Some(IdolFiles {
        dit: pick(&["diffusion_models", "unet"], &["idol_fp16.safetensors", "idol.safetensors"])?,
        decoder: pick(&["vae"], &["idol_decoder_fp16.safetensors", "idol_decoder.safetensors"])?,
        dino: pick(&["clip_vision"], &["dino_v3_vit_h_bf16.safetensors", "dino_v3_vit_h.safetensors"])?,
        vae: pick(&["vae"], &["flux2-vae_small.safetensors"])?,
    })
}

/// What to make.
#[derive(Clone, Debug)]
pub struct Request {
    /// The object on a transparent background (straight alpha).
    pub picture: RgbaImage,
    pub seed: u64,
    pub steps: usize,
    pub cfg: f32,
    /// Rounded to a multiple of 32, within 32 768 ..= 1 048 576.
    pub num_gaussians: usize,
}

impl Request {
    pub fn new(picture: RgbaImage, seed: u64) -> Self {
        Self { picture, seed, steps: 50, cfg: 4.0, num_gaussians: 262_144 }
    }
}

/// `TripoSplatPreprocessImage`: `(SIZE · SIZE · 3)` RGB in `[0, 1]`, the object squared at
/// 1.2× its eroded alpha's box and composited on black.
pub fn preprocess(picture: &RgbaImage, erode: usize) -> Result<Vec<f32>> {
    use image::imageops::{resize, FilterType};
    let (w, h) = picture.dimensions();
    let s = SIZE as f64 / w.min(h) as f64;
    let (rw, rh) = (((w as f64 * s).round() as u32).max(1), ((h as f64 * s).round() as u32).max(1));
    let img = if (rw, rh) == (w, h) { picture.clone() } else { resize(picture, rw, rh, FilterType::Lanczos3) };
    let (rw, rh) = (rw as usize, rh as usize);
    let mut rgba: Vec<[f32; 4]> = img.pixels().map(|p| p.0.map(|v| f32::from(v) / 255.0)).collect();
    if erode > 0 {
        let alpha: Vec<f32> = rgba.iter().map(|p| p[3]).collect();
        let eroded = min_filter(&alpha, rw, rh, erode);
        for (p, a) in rgba.iter_mut().zip(eroded) {
            p[3] = a;
        }
    }
    let (mut x0, mut x1, mut y0, mut y1) = (usize::MAX, 0, usize::MAX, 0);
    for y in 0..rh {
        for x in 0..rw {
            if rgba[y * rw + x][3] > 0.0 {
                (x0, x1, y0, y1) = (x0.min(x), x1.max(x), y0.min(y), y1.max(y));
            }
        }
    }
    anyhow::ensure!(x0 != usize::MAX, "the picture has no object: its alpha is empty");
    let (cx, cy) = ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0);
    let half = (x1 - x0).max(y1 - y0) as f64 / 2.0 * 1.2;
    // Python's int(): toward zero.
    let (left, upper, right, lower) = ((cx - half) as i64, (cy - half) as i64, (cx + half) as i64, (cy + half) as i64);
    let (cw, ch) = ((right - left) as u32, (lower - upper) as u32);
    let mut crop = RgbaImage::new(cw.max(1), ch.max(1));
    for (x, y, px) in crop.enumerate_pixels_mut() {
        let (sx, sy) = (x as i64 + left, y as i64 + upper);
        if sx >= 0 && sy >= 0 && (sx as usize) < rw && (sy as usize) < rh {
            let p = rgba[sy as usize * rw + sx as usize];
            px.0 = p.map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    let sq = resize(&crop, SIZE as u32, SIZE as u32, FilterType::Lanczos3);
    let mut out = Vec::with_capacity(SIZE * SIZE * 3);
    for p in sq.pixels() {
        let a = f32::from(p.0[3]) / 255.0;
        out.extend((0..3).map(|c| f32::from(p.0[c]) / 255.0 * a));
    }
    Ok(out)
}

/// Morphological erosion: the minimum over a `(2r+1)²` window (`-max_pool2d(-a)`, the window
/// clipped at the border).
fn min_filter(a: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut rows = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let (lo, hi) = (x.saturating_sub(r), (x + r).min(w - 1));
            rows[y * w + x] = a[y * w + lo..=y * w + hi].iter().cloned().fold(f32::INFINITY, f32::min);
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        let (lo, hi) = (y.saturating_sub(r), (y + r).min(h - 1));
        for x in 0..w {
            out[y * w + x] = (lo..=hi).map(|yy| rows[yy * w + x]).fold(f32::INFINITY, f32::min);
        }
    }
    out
}

/// torch's scalar `randn` path (tensors under 16 elements): Box–Muller in f64 from 53-bit
/// uniforms, the sine half cached for the next draw.
fn randn_small(g: &mut Mt19937, n: usize, cache: &mut Option<f64>) -> Vec<f32> {
    let uniform = |g: &mut Mt19937| {
        let hi = u64::from(g.next_u32());
        let lo = u64::from(g.next_u32());
        (((hi << 32) | lo) & ((1u64 << 53) - 1)) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            if let Some(v) = cache.take() {
                return v as f32;
            }
            let u1 = uniform(g);
            let u2 = uniform(g);
            let r = (-2.0 * (-u2).ln_1p()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u1;
            *cache = Some(r * theta.sin());
            (r * theta.cos()) as f32
        })
        .collect()
}

/// ComfyUI's noise for Idol's nested latent: `randn(1, 8192, 16)` then `randn(1, 1, 5)` on one
/// generator seeded with `seed`.
pub fn noise(seed: u64) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut g = Mt19937::new(seed);
    let z = crate::comfy::randn_with(&mut g, TOKENS * LATENT_CHANNELS).map_err(anyhow::Error::msg)?;
    let cam = randn_small(&mut g, CAM_CHANNELS, &mut None);
    Ok((z, cam))
}

/// The run's device and dtype: CUDA in f16 (the files' dtype, as ComfyUI runs them), else
/// the CPU in f32.
pub fn device() -> Result<(Device, DType)> {
    let dev = crate::device::best_device()?;
    Ok(if dev.is_cuda() { (dev, DType::F16) } else { (dev, DType::F32) })
}

pub struct IdolPipeline {
    files: IdolFiles,
    dev: Device,
    dtype: DType,
}

impl IdolPipeline {
    pub fn new(files: IdolFiles) -> Result<Self> {
        let (dev, dtype) = device()?;
        Ok(Self { files, dev, dtype })
    }

    pub fn with_device(files: IdolFiles, dev: Device, dtype: DType) -> Self {
        Self { files, dev, dtype }
    }

    /// `(feature1, ref latent)` for the preprocessed `rgb (SIZE·SIZE·3)`.
    pub fn encode(&self, rgb: &[f32]) -> Result<(Tensor, Tensor)> {
        let img = Tensor::from_slice(rgb, (1, SIZE, SIZE, 3), &self.dev)?.permute((0, 3, 1, 2))?.contiguous()?;
        let feat1 = {
            // DINOv3 runs in f32, as ComfyUI's manual_cast runs it for a float32 picture.
            let w = Weights::open(&self.files.dino, &self.dev, DType::F32, false).context("DINOv3 ViT-H")?;
            Dino::load(&w)?.features(&img)?
        };
        let feat2 = {
            let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&self.files.vae), self.dtype, &self.dev)? };
            Flux2Encoder::load(vb).context("FLUX.2 VAE encoder")?.encode(&((img * 2.0)? - 1.0)?)?
        };
        Ok((feat1, feat2))
    }

    /// Sample the shape code `(1, 8192, 16)` f32.
    pub fn sample(&self, feat1: &Tensor, feat2: &Tensor, req: &Request, progress: &mut dyn FnMut(f32, &str), cancel: &dyn Fn() -> bool) -> Result<Tensor> {
        let w = Weights::open(&self.files.dit, &self.dev, self.dtype, false).context("Idol flow model")?;
        let dit = IdolDit::load(&w)?;
        let ctx_pos = dit.refine_context(&dit.context(feat1, feat2)?)?;
        let ctx_neg = dit.refine_context(&dit.context(&feat1.zeros_like()?, &feat2.zeros_like()?)?)?;
        let sched = FlowSchedule::simple(req.steps, SHIFT).map_err(anyhow::Error::msg)?;
        let (nz, nc) = noise(req.seed)?;
        let s0 = sched.sigma(0);
        let mut z = (Tensor::from_vec(nz, (1, TOKENS, LATENT_CHANNELS), &self.dev)? * f64::from(s0))?;
        let mut cam = (Tensor::from_vec(nc, (1, 1, CAM_CHANNELS), &self.dev)? * f64::from(s0))?;
        let n = sched.steps();
        for i in 0..n {
            if cancel() {
                return Err(Cancelled.into());
            }
            let t = sched.sigma(i) * 1000.0;
            let (vz, vc) = dit.forward(&z, &cam, t, &ctx_pos)?;
            let (vz, vc) = if req.cfg == 1.0 {
                (vz, vc)
            } else {
                let (uz, uc) = dit.forward(&z, &cam, t, &ctx_neg)?;
                let g = f64::from(req.cfg);
                ((&uz + ((vz - &uz)? * g)?)?, (&uc + ((vc - &uc)? * g)?)?)
            };
            let dt = f64::from(sched.dt(i));
            z = (z + (vz * dt)?)?;
            cam = (cam + (vc * dt)?)?;
            progress((i + 1) as f32 / n as f32, "Shaping");
        }
        Ok(z)
    }

    /// Decode the shape code into gaussians.
    pub fn decode(&self, z: &Tensor, req: &Request) -> Result<Splat> {
        let w = Weights::open(&self.files.decoder, &self.dev, self.dtype, false).context("Idol decoder")?;
        let n = req.num_gaussians.clamp(32_768, 1_048_576);
        let n = (n as f64 / GAUSSIANS_PER_POINT as f64).round() as usize * GAUSSIANS_PER_POINT;
        IdolDecoder::load(&w)?.decode(z, n, req.seed)
    }

    /// The whole run; `progress` hears each stage, `cancel` is read between steps.
    pub fn generate(&self, req: &Request, progress: &mut dyn FnMut(f32, &str), cancel: &dyn Fn() -> bool) -> Result<Splat> {
        progress(0.0, "Squaring the object");
        let rgb = preprocess(&req.picture, ERODE)?;
        progress(0.02, "Reading the picture");
        let (f1, f2) = self.encode(&rgb)?;
        let z = self.sample(&f1, &f2, req, &mut |p, s| progress(0.08 + 0.84 * p, s), cancel)?;
        progress(0.93, "Placing the splats");
        let s = self.decode(&z, req)?;
        progress(1.0, "Done");
        Ok(s)
    }
}
