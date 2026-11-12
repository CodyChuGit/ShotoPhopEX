//! FLUX 3 Image Klein, resident on CUDA: Qwen3-VL-4B's language tower as the text encoder, the
//! DiT and the FLUX.2 VAE (encoder and decoder). The conventions of BFL's reference, as
//! `gn-mlx`'s Klein runs them on a Mac: torch's CPU noise for the seed (`(h·w, 128)`, rounded to
//! bf16), the flow map's 16 Euler steps on `α·t / (1 + (α − 1)·t)` with no CFG (Base: 50 with CFG
//! against the empty prompt), references told apart by their RoPE layer `-(1 + i)`. The prompt
//! upsampler (a 9B model) is not run; prompts go to the encoder as typed.
//!
//! On a card under [`LOW_VRAM_BELOW`] the encoder and the DiT are never on it together (8 GB
//! mode): the encoder loads, encodes and goes before the DiT comes back.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use super::dit::{self, KleinDit, CHANNELS};
use crate::model::text_encoder::{QwenTextEncoder, TextConfig};
use crate::moss::vae::{self, Flux2Decoder, Flux2Encoder};
use crate::moss::weights::Weights;
use crate::pipeline::{Cancelled, Picture, RefImage, Request, Timings};
use crate::Result;

/// The schedule's shift. mflux's 6.93 (good on the Mac); BFL's `Pipeline.generate` defaults to
/// 16. `QIR_KLEIN_ALPHA` overrides it for comparisons.
pub const ALPHA: f32 = 6.93;
pub const MAX_REFERENCES: usize = 10;
/// An edit's output when the request names no size: the first picture's aspect at about this
/// many pixels (AusBoss's Klein edit workflow scales the picture to 1 MP), never larger than the
/// picture.
pub const EDIT_PIXELS: usize = 1024 * 1024;
/// A fill's reference side as a share of the output's.
const FILL_REFERENCE_SHARE: f32 = 0.5;
/// Each further reference at most this many pixels.
const REFERENCE_PIXELS: usize = 1024 * 1024;
/// Cards with less memory than this run in 8 GB mode (the int8 DiT 5.8 GB and the fp8 encoder
/// about 4.5 GB do not fit together with room for the activations).
pub const LOW_VRAM_BELOW: usize = 15 << 30;
/// The hidden states the DiT reads (after this many layers), stacked into 20480-wide tokens.
const TAPS: [usize; 8] = [4, 8, 12, 16, 20, 24, 28, 32];
const PAD_MULTIPLE: usize = 80;
const MAX_LENGTH: usize = 4096;
/// Qwen's `<|endoftext|>`, its pad token.
const PAD_TOKEN: &str = "<|endoftext|>";

/// ComfyUI's file names, first found wins: ConvRot INT8 before bf16.
const DIT: &[&str] = &["flux3-image-klein-int8-convrot.safetensors", "flux3-image-klein-bf16.safetensors"];
const DIT_BASE: &[&str] = &["flux3-image-klein-base-int8-convrot.safetensors", "flux3-image-klein-base-bf16.safetensors"];
/// Qwen3-VL-4B: Krea 2's fp8 file first (4.5 GB on the card), then BFL's bf16 one.
const TEXT: &[&str] = &["qwen3vl_4b_fp8_scaled.safetensors", "qwen3vl_4b_bf16.safetensors"];
/// The FLUX.2 VAE in diffusers' names (Comfy-Org's `flux2-vae_small`, which Get AI Models
/// installs). BFL's own `flux2-vae.safetensors` keeps the original names, which
/// [`Flux2Decoder`] does not read, so it is not looked for.
const VAE: &[&str] = &["flux2-vae_small.safetensors"];

/// Qwen3-VL's chat template for one user message, ready for the answer.
pub fn template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

/// The prompt's length padded to the reference's multiple of 80, at most 4096.
pub fn padded_len(real: usize) -> usize {
    real.div_ceil(PAD_MULTIPLE).saturating_mul(PAD_MULTIPLE).clamp(PAD_MULTIPLE, MAX_LENGTH)
}

/// The schedule: `steps + 1` times from 1 to 0.
pub fn timesteps(steps: usize, alpha: f32) -> Vec<f32> {
    (0..=steps)
        .map(|i| {
            let t = 1.0 - i as f32 / steps as f32;
            alpha * t / (1.0 + (alpha - 1.0) * t)
        })
        .collect()
}

/// `(w, h)` at about `pixels` keeping the aspect, never larger than the picture, sides multiples
/// of 16.
pub fn fit(w: usize, h: usize, pixels: usize) -> (usize, usize) {
    let scale = (pixels as f64 / (w.max(1) * h.max(1)) as f64).sqrt().min(1.0);
    let side = |v: usize| (((v as f64 * scale / 16.0).round() as usize) * 16).max(16);
    (side(w), side(h))
}

/// Klein's files in a ComfyUI-layout `models` folder.
#[derive(Clone, Debug)]
pub struct KleinFiles {
    pub dit: PathBuf,
    pub text: PathBuf,
    pub vae: PathBuf,
    /// The folder with Qwen's `vocab.json`, `merges.txt` and `tokenizer_config.json`.
    pub tokenizer: PathBuf,
}

/// `names` in `dir` or one of its subfolders (sorted, so the choice is stable).
fn find(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    let mut dirs = vec![dir.to_path_buf()];
    if let Ok(rd) = std::fs::read_dir(dir) {
        let mut subs: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect();
        subs.sort();
        dirs.extend(subs);
    }
    names.iter().find_map(|n| dirs.iter().map(|d| d.join(n)).find(|p| p.is_file()))
}

/// Klein under `root` (the distilled model, or Base), when every file is there.
pub fn find_files(root: &Path, base: bool) -> Option<KleinFiles> {
    Some(KleinFiles {
        dit: find(&root.join("diffusion_models"), if base { DIT_BASE } else { DIT })?,
        text: find(&root.join("text_encoders"), TEXT)?,
        vae: find(&root.join("vae"), VAE)?,
        tokenizer: crate::comfy_files::tokenizer_dir(root)?,
    })
}

/// A prompt's encoding, kept for the next request with the same prompt.
struct Cached {
    prompt: String,
    hidden: Tensor,
}

pub struct KleinPipeline {
    dev: Device,
    dtype: DType,
    tok: Tokenizer,
    pad_id: u32,
    files: KleinFiles,
    low_vram: bool,
    te: Mutex<Option<Arc<QwenTextEncoder>>>,
    dit: Mutex<Option<Arc<KleinDit>>>,
    vae: Flux2Decoder,
    encoder: Flux2Encoder,
    encoded: Mutex<Vec<Cached>>,
}

impl KleinPipeline {
    pub fn load(files: &KleinFiles, progress: &mut dyn FnMut(f32, &str)) -> Result<Self> {
        let dev = crate::device::best_device()?;
        let dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
        let tok = crate::comfy_files::qwen_tokenizer(&files.tokenizer)?;
        let pad_id = tok.token_to_id(PAD_TOKEN).context("the tokenizer has no <|endoftext|>")?;
        let low_vram = low_vram_mode(&dev);
        tracing::info!(low_vram, "Klein memory mode");
        let te = Mutex::new(None);
        if !low_vram {
            progress(0.0, "Loading the text encoder");
            *te.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(load_text(files, &dev, dtype)?));
        }
        // In 8 GB mode both wait for the first picture: the encoder runs first, and a DiT loaded
        // now would only be dropped for it and loaded again.
        let d = Mutex::new(None);
        if !low_vram {
            progress(0.5, "Loading the transformer");
            *d.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(load_dit(files, &dev, dtype)?));
        }
        progress(0.9, "Loading the VAE");
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[&files.vae], dtype, &dev)? };
        let vae = Flux2Decoder::load(vb.clone()).context("the FLUX.2 VAE")?;
        let encoder = Flux2Encoder::load(vb).context("the FLUX.2 VAE file has no encoder")?;
        progress(1.0, "Loaded");
        Ok(Self { dev, dtype, tok, pad_id, files: files.clone(), low_vram, te, dit: d, vae, encoder, encoded: Mutex::new(Vec::new()) })
    }

    pub fn is_cuda(&self) -> bool {
        self.dev.is_cuda()
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// The dtype LoRA files load in.
    pub fn lora_dtype(&self) -> DType {
        self.dtype
    }

    pub fn low_vram(&self) -> bool {
        self.low_vram
    }

    /// The text encoder, loaded when it is not on the card (in 8 GB mode the DiT goes first).
    fn text_encoder(&self) -> Result<Arc<QwenTextEncoder>> {
        let mut te = self.te.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = te.as_ref() {
            return Ok(Arc::clone(t));
        }
        let mut dit = self.dit.lock().unwrap_or_else(|e| e.into_inner());
        if self.low_vram && dit.take().is_some() {
            crate::device::trim_pool(&self.dev);
        }
        let t = Arc::new(load_text(&self.files, &self.dev, self.dtype)?);
        *te = Some(Arc::clone(&t));
        Ok(t)
    }

    /// The DiT, loaded when it is not on the card (in 8 GB mode the encoder goes first).
    fn transformer(&self) -> Result<Arc<KleinDit>> {
        let mut te = self.te.lock().unwrap_or_else(|e| e.into_inner());
        let mut dit = self.dit.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(d) = dit.as_ref() {
            return Ok(Arc::clone(d));
        }
        if self.low_vram && te.take().is_some() {
            crate::device::trim_pool(&self.dev);
        }
        let d = Arc::new(load_dit(&self.files, &self.dev, self.dtype)?);
        *dit = Some(Arc::clone(&d));
        Ok(d)
    }

    /// The prompt's tokens `(L, 20480)` in the run dtype, `L` padded to a multiple of 80.
    pub fn encode(&self, prompt: &str) -> Result<Tensor> {
        let enc = self.tok.encode(template(prompt), false).map_err(|e| anyhow::anyhow!("encode the prompt: {e}"))?;
        let mut ids = enc.get_ids().to_vec();
        ids.truncate(MAX_LENGTH);
        let pads = padded_len(ids.len()) - ids.len();
        let te = self.text_encoder()?;
        let states = te.forward_taps_padded(&ids, self.pad_id, pads, &TAPS)?;
        // Each token's eight hidden states side by side, the earliest first.
        let refs: Vec<&Tensor> = states.iter().collect();
        Ok(Tensor::cat(&refs, 2)?.squeeze(0)?.to_dtype(self.dtype)?)
    }

    /// `prompts`' encodings, cached (the last two kept).
    fn encodings(&self, prompts: &[&str]) -> Result<Vec<Tensor>> {
        let mut out = Vec::with_capacity(prompts.len());
        for p in prompts {
            let hit = self.encoded.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|c| c.prompt == *p).map(|c| c.hidden.clone());
            let h = match hit {
                Some(h) => h,
                None => {
                    let h = self.encode(p)?;
                    let mut slot = self.encoded.lock().unwrap_or_else(|e| e.into_inner());
                    slot.push(Cached { prompt: (*p).to_string(), hidden: h.clone() });
                    let extra = slot.len().saturating_sub(2);
                    slot.drain(..extra);
                    h
                }
            };
            out.push(h);
        }
        Ok(out)
    }

    /// A picture resized to `(w, h)` (Lanczos) and encoded: tokens `(h/16 · w/16, 128)` f32.
    fn reference(&self, r: &RefImage, w: usize, h: usize) -> Result<Tensor> {
        let img = image::RgbaImage::from_raw(u32::try_from(r.width)?, u32::try_from(r.height)?, r.rgba.clone())
            .context("a picture's pixels do not match its size")?;
        let img = if (w, h) == (r.width, r.height) {
            img
        } else {
            image::imageops::resize(&img, u32::try_from(w)?, u32::try_from(h)?, image::imageops::FilterType::Lanczos3)
        };
        let z = self.encoder.encode(&vae::from_rgba_u8(img.as_raw(), w, h, &self.dev)?)?;
        dit::to_tokens(&z)
    }

    /// Generative Expand as AusBoss's Klein Outpaint workflow runs it (the LoRA in `req.loras`, at
    /// 1.5): the canvas with `mask`'s white flat grey (#808080) is the reference at the output's
    /// full size, the picture is sampled whole with no noise mask, and the canvas' own pixels go
    /// back outside the mask after decoding. Keeping the latent every step instead (the fill)
    /// leaves the new areas out of line with the picture: hard seams and a horizon that jumps.
    pub fn outpaint(&self, req: &Request, mask: &RefImage, progress: &mut dyn FnMut(f32, &str), cancel: &dyn Fn() -> bool) -> Result<Picture> {
        let canvas = req.references.first().context("an expand needs its canvas")?;
        let mut ask = req.clone();
        ask.references[0] = greyed(canvas, mask)?;
        let mut pic = self.generate(&ask, None, progress, cancel)?;
        let (w, h) = (pic.width, pic.height);
        let fit = |r: &RefImage, filter| -> Result<image::RgbaImage> {
            let img = image::RgbaImage::from_raw(u32::try_from(r.width)?, u32::try_from(r.height)?, r.rgba.clone())
                .context("a picture's pixels do not match its size")?;
            Ok(if img.dimensions() == (w, h) { img } else { image::imageops::resize(&img, w, h, filter) })
        };
        let (orig, m) = (fit(canvas, image::imageops::FilterType::Lanczos3)?, fit(mask, image::imageops::FilterType::Triangle)?);
        for ((px, o), mp) in pic.rgba.chunks_exact_mut(4).zip(orig.pixels()).zip(m.pixels()) {
            let a = f32::from(mp.0[0]) / 255.0;
            for (c, oc) in px.iter_mut().zip(o.0).take(3) {
                *c = (f32::from(*c) * a + f32::from(oc) * (1.0 - a)).round() as u8;
            }
        }
        Ok(pic)
    }

    /// Make one picture. `req.references`: the pictures (the first is the one edited and sets
    /// the canvas); `mask`, for a fill, white where the model paints (the first picture is kept
    /// elsewhere). `req.steps` 0 takes the checkpoint's own (16, or 50 for Base);
    /// `req.guidance` only matters for Base. `progress` hears every step, `cancel` is read
    /// between steps.
    pub fn generate(&self, req: &Request, mask: Option<&RefImage>, progress: &mut dyn FnMut(f32, &str), cancel: &dyn Fn() -> bool) -> Result<Picture> {
        crate::device::bind_thread(&self.dev);
        anyhow::ensure!(req.steps <= 200, "{} steps (1..200)", req.steps);
        anyhow::ensure!(req.references.len() <= MAX_REFERENCES, "Klein takes at most {MAX_REFERENCES} pictures ({} given)", req.references.len());
        anyhow::ensure!(mask.is_none() || !req.references.is_empty(), "a fill needs its picture");
        let (w, h) = match req.references.first() {
            Some(p) if req.width == 0 || req.height == 0 => fit(p.width, p.height, EDIT_PIXELS),
            _ => (req.width, req.height),
        };
        anyhow::ensure!(w > 0 && h > 0 && w.is_multiple_of(16) && h.is_multiple_of(16), "the picture's sides must be positive multiples of 16 ({w}x{h} asked)");
        anyhow::ensure!(w * h <= 2048 * 2048, "{w}x{h} is too large");
        let check = || -> Result<()> {
            if cancel() {
                Err(Cancelled.into())
            } else {
                Ok(())
            }
        };
        let mut timings = Timings::default();

        check()?;
        progress(0.02, "Encoding the prompt");
        let t0 = Instant::now();
        // Base needs the empty prompt too; which checkpoint it is shows in the file's name.
        let base = self.files.dit.file_name().is_some_and(|n| n.to_string_lossy().contains("-base-"));
        let cfg = base && req.guidance != 1.0;
        let prompts: Vec<&str> = if cfg { vec![&req.prompt, ""] } else { vec![&req.prompt] };
        let enc = self.encodings(&prompts)?;
        let dit = self.transformer()?;
        anyhow::ensure!(dit.is_flow_map() != base, "the DiT file's name and its contents disagree on whether it is Base");
        let unmatched = dit.set_loras(&req.loras)?;
        if !unmatched.is_empty() {
            tracing::warn!(modules = unmatched.len(), first = %unmatched[0], "LoRA modules FLUX 3 Image Klein does not have (a file for another model?)");
            anyhow::ensure!(
                unmatched.len() < req.loras.layers.iter().map(|(f, _)| f.entries.len()).sum::<usize>(),
                "none of the LoRAs' modules is FLUX 3 Image Klein's (a file for another model?)"
            );
        }

        // The references: the first at the output's size (it sets the canvas), the rest under 1 MP.
        progress(0.04, "Reading the pictures");
        let mut cond_tokens = Vec::new();
        let mut cond_ids: Vec<[f32; 4]> = Vec::new();
        // Fill: the picture's own latent (the area kept), and the picture the model reads with the
        // area flat grey at half the side (it paints the area back).
        let mut original = None;
        for (i, r) in req.references.iter().enumerate() {
            let (rw, rh) = if i == 0 { (w, h) } else { fit(r.width, r.height, REFERENCE_PIXELS) };
            if let (Some(m), 0) = (mask, i) {
                original = Some(self.reference(r, rw, rh)?);
                let side = |v: usize| (((v as f32 * FILL_REFERENCE_SHARE) / 16.0).round() as usize * 16).max(16);
                let (gw, gh) = (side(rw), side(rh));
                cond_tokens.push(self.reference(&greyed(r, m)?, gw, gh)?);
                cond_ids.extend(dit::grid_ids(gh / 16, gw / 16, -1.0));
                continue;
            }
            cond_tokens.push(self.reference(r, rw, rh)?);
            cond_ids.extend(dit::grid_ids(rh / 16, rw / 16, -(1.0 + i as f32)));
        }
        let cond = if cond_tokens.is_empty() { None } else { Some(Tensor::cat(&cond_tokens, 0)?) };

        let (lh, lw) = (h / 16, w / 16);
        let image_ids = dit::grid_ids(lh, lw, 0.0);
        let noise = crate::comfy::torch_randn(req.seed, lh * lw * CHANNELS).map_err(anyhow::Error::msg)?;
        let noise: Vec<f32> = noise.iter().map(|&v| dit::bf16_round(v)).collect();
        let mut x = Tensor::from_vec(noise, (lh * lw, CHANNELS), &self.dev)?;
        // Fill: outside the mask the picture's own latent, noised to each step's time, is put back
        // after every step (the last leaves it exact); inside, the model paints.
        let keep = match (mask, &original) {
            (Some(m), Some(z0)) => Some((z0.clone(), mask_tokens(m, lw, lh, &self.dev)?, x.clone())),
            _ => None,
        };
        let cond_ref = cond.as_ref().map(|c| (c, cond_ids.as_slice()));
        let preps = enc.iter().map(|e| dit.prepare(&image_ids, cond_ref, e)).collect::<Result<Vec<_>>>()?;
        self.dev.synchronize()?;
        timings.encode_s = t0.elapsed().as_secs_f32();

        let steps = if req.steps == 0 {
            if base {
                50
            } else {
                16
            }
        } else {
            req.steps
        };
        let alpha = std::env::var("QIR_KLEIN_ALPHA").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(ALPHA);
        let sched = timesteps(steps, alpha);
        let t0 = Instant::now();
        for i in 0..steps {
            check()?;
            let (t, t1) = (sched[i], sched[i + 1]);
            let target = (!base).then_some(t1);
            let v = dit.forward(&x, &preps[0], t, target)?;
            let v = match preps.get(1) {
                Some(neg) => {
                    let u = dit.forward(&x, neg, t, target)?;
                    (&u + ((v - &u)? * f64::from(req.guidance))?)?
                }
                None => v,
            };
            x = (x + (v * f64::from(t1 - t))?)?;
            if let Some((z0, m, eps)) = &keep {
                let known = ((z0 * f64::from(1.0 - t1))? + (eps * f64::from(t1))?)?;
                x = (x.broadcast_mul(m)? + known.broadcast_mul(&(1.0 - m)?)?)?;
            }
            progress(0.05 + 0.85 * (i + 1) as f32 / steps as f32, "Sampling");
        }
        self.dev.synchronize()?;
        timings.denoise_s = t0.elapsed().as_secs_f32();

        check()?;
        progress(0.92, "Decoding");
        if self.low_vram {
            crate::device::trim_pool(&self.dev);
        }
        let t0 = Instant::now();
        let img = self.vae.decode(&dit::from_tokens(&x, lh, lw)?)?;
        self.dev.synchronize()?;
        timings.decode_s = t0.elapsed().as_secs_f32();
        let (pw, ph, rgba) = vae::to_rgba_u8(&img)?;
        tracing::info!(
            encode_s = timings.encode_s,
            denoise_s = timings.denoise_s,
            decode_s = timings.decode_s,
            width = pw,
            height = ph,
            steps,
            "Klein timings"
        );
        progress(1.0, "Done");
        Ok(Picture { width: pw as u32, height: ph as u32, rgba, timings })
    }
}

/// `r` with the mask's white made flat grey (#808080), blended by the mask's value; a mask at
/// another size is resized to the picture's.
fn greyed(r: &RefImage, m: &RefImage) -> Result<RefImage> {
    let mask =
        image::RgbaImage::from_raw(u32::try_from(m.width)?, u32::try_from(m.height)?, m.rgba.clone()).context("the mask's pixels do not match its size")?;
    let mask = if (m.width, m.height) == (r.width, r.height) {
        mask
    } else {
        image::imageops::resize(&mask, u32::try_from(r.width)?, u32::try_from(r.height)?, image::imageops::FilterType::Triangle)
    };
    let mut rgba = r.rgba.clone();
    for (px, mp) in rgba.chunks_exact_mut(4).zip(mask.pixels()) {
        let a = f32::from(mp.0[0]) / 255.0;
        for c in px.iter_mut().take(3) {
            *c = (f32::from(*c) * (1.0 - a) + 128.0 * a).round() as u8;
        }
    }
    Ok(RefImage { width: r.width, height: r.height, rgba, offset: None })
}

/// A fill mask on the latent grid `(lh·lw, 1)` f32 in 0..1 (1 is painted): averaged over each
/// latent pixel's 16×16 and doubled, so an edge pixel the mask touches is painted too.
fn mask_tokens(m: &RefImage, lw: usize, lh: usize, dev: &Device) -> Result<Tensor> {
    let img =
        image::RgbaImage::from_raw(u32::try_from(m.width)?, u32::try_from(m.height)?, m.rgba.clone()).context("the mask's pixels do not match its size")?;
    let small = image::imageops::resize(&img, u32::try_from(lw)?, u32::try_from(lh)?, image::imageops::FilterType::Triangle);
    let v: Vec<f32> = small.pixels().map(|p| (f32::from(p.0[0]) / 255.0 * 2.0).min(1.0)).collect();
    Ok(Tensor::from_vec(v, (lh * lw, 1), dev)?)
}

/// 8 GB mode on a card under [`LOW_VRAM_BELOW`], or as `QIR_KLEIN_LOW_VRAM=1` / `0` says.
fn low_vram_mode(dev: &Device) -> bool {
    match std::env::var("QIR_KLEIN_LOW_VRAM").ok().as_deref().map(str::trim) {
        Some("1") => true,
        Some("0") => false,
        _ => crate::device::free_vram(dev).is_some_and(|(_, total)| total < LOW_VRAM_BELOW),
    }
}

/// Qwen3-VL-4B's language tower: ComfyUI's repack (rooted at `model.`) or BFL's file (rooted at
/// `model.language_model.`); the vision tower is never read.
fn load_text(files: &KleinFiles, dev: &Device, dtype: DType) -> Result<QwenTextEncoder> {
    let t0 = Instant::now();
    let vb = crate::mapped::var_builder(std::slice::from_ref(&files.text), dtype, dev, |n| n.starts_with("model.") && !n.starts_with("model.visual."))?;
    let root = if vb.contains_tensor("model.language_model.embed_tokens.weight") { vb.pp("model").pp("language_model") } else { vb.pp("model") };
    let te = QwenTextEncoder::load_at(&TextConfig::qwen3vl_4b(), false, root).context("the Qwen3-VL-4B encoder")?;
    tracing::info!(load_s = t0.elapsed().as_secs_f32(), "Klein text encoder loaded");
    Ok(te)
}

fn load_dit(files: &KleinFiles, dev: &Device, dtype: DType) -> Result<KleinDit> {
    let t0 = Instant::now();
    let dit = KleinDit::load(&Weights::open(&files.dit, dev, dtype, true)?).context("the FLUX 3 Image Klein DiT")?;
    tracing::info!(load_s = t0.elapsed().as_secs_f32(), "Klein DiT loaded");
    Ok(dit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_padding_and_sizes() {
        let s = timesteps(16, ALPHA);
        assert_eq!(s.len(), 17);
        assert!((s[0] - 1.0).abs() < 1e-6 && s[16].abs() < 1e-6);
        assert!(s.windows(2).all(|w| w[1] < w[0]));
        assert_eq!((padded_len(1), padded_len(80), padded_len(81), padded_len(5000)), (80, 80, 160, 4096));
        assert_eq!(fit(1024, 1024, EDIT_PIXELS), (1024, 1024));
        assert_eq!(fit(4000, 3000, EDIT_PIXELS), (1184, 880));
        assert_eq!(template("a cat"), "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n");
    }
}
