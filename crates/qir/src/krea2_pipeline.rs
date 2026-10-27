//! Krea 2 Turbo, resident (plan Phase 4): the Qwen3-VL-4B encoder, the single-stream DiT and
//! the Wan 2.1 VAE loaded once, text to image with ComfyUI's conventions (its noise for the
//! seed, the `simple` schedule over the model's sigma table with shift 1.15, eight steps, the
//! Wan21 latent statistics).
//!
//! The weights folder (bring your own: `qwen-image-rs convert-fp8` rewrites ComfyUI's fp8
//! repacks): `transformer/*.safetensors` (keys `blocks.*`, `txtfusion.*`, …),
//! `text_encoder/*.safetensors` (keys `model.*`), `vae/*.safetensors` (Wan's layout) and
//! `tokenizer/tokenizer.json` (Qwen3-VL's).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use crate::Result;
use crate::convrot_cache::{CacheSpec, Policy};
use crate::loader::WeightSet;
use crate::model::krea2::{self, EditRef, Krea2Config, Krea2Dit};
use crate::model::text_encoder::{QwenTextEncoder, TextConfig};
use crate::model::vision::{self, VisionConfig, VisionTower};
use crate::model::wan_vae::{self, WanVae};
use crate::pipeline::{Cancelled, Picture, RefImage, Request, Timings};

/// The longest side of the picture the encoder sees beside an edit's instruction (the
/// comfyui-krea2edit `grounding_px` default; the LoRAs trained at 384–768).
pub const GROUNDING_PX: u32 = 768;

/// How the pipeline is loaded.
#[derive(Clone, Debug)]
pub struct Krea2Options {
    /// ConvRot W8A8 INT8 for the DiT's block linears (the `convrot` feature): 12 GB instead of
    /// 24, and the encoder stays resident beside it.
    pub convrot: bool,
    /// Where the prequantized ConvRot DiT is cached (the quantisation is 11 s of each cold
    /// start otherwise).
    pub cache: CacheSpec,
}

impl Krea2Options {
    /// ConvRot on when the build has it, the cache where the environment says.
    pub fn defaults() -> Self {
        Self { convrot: cfg!(feature = "convrot"), cache: CacheSpec::from_cli(None, false, false) }
    }
}

/// The folder layout the pipeline reads.
pub fn is_snapshot(dir: &Path) -> bool {
    dir.join("tokenizer/tokenizer.json").is_file() && dir.join("text_encoder").is_dir() && dir.join("transformer").is_dir() && dir.join("vae").is_dir()
}

pub struct Krea2Pipeline {
    dev: Device,
    dtype: DType,
    tok: Tokenizer,
    model: PathBuf,
    te: std::sync::Mutex<Option<QwenTextEncoder>>,
    /// The encoder's vision tower, loaded on the first edit (f32, ~1.7 GB).
    vision: std::sync::Mutex<Option<VisionTower>>,
    dit: Krea2Dit,
    vae: WanVae,
    /// Drop the encoder after each prompt (the bf16 DiT leaves no room for it; it loads again
    /// in two seconds for the next prompt).
    evict_te: bool,
}

impl Krea2Pipeline {
    pub fn load(model: &Path, opts: Krea2Options, progress: &mut dyn FnMut(f32, &str)) -> Result<Self> {
        if !is_snapshot(model) {
            anyhow::bail!("{} is not a Krea 2 weights folder (tokenizer/, text_encoder/, transformer/, vae/)", model.display());
        }
        let dev = crate::device::best_device()?;
        let dtype = if matches!(dev, Device::Cuda(_)) { DType::BF16 } else { DType::F32 };
        let tok = Tokenizer::from_file(model.join("tokenizer/tokenizer.json")).map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

        // ComfyUI's fp8_scaled file as it is: 13 GB, run with its own fp8 recipe (`crate::fp8`).
        let raw = WeightSet::resolve(&model.join("transformer"))?.files;
        let fp8 = crate::fp8::Fp8Plan::from_files(&raw)?;
        let fp8_text = crate::fp8::Fp8Plan::from_files(&WeightSet::resolve(&model.join("text_encoder"))?.files)?.is_some();
        anyhow::ensure!((fp8.is_none() && !fp8_text) || crate::fp8::supported_on(&dev), "{} holds fp8_scaled weights and this build has no fp8 kernels (the `fp8` feature)", model.display());
        // Without ConvRot or fp8 the DiT is 24 GB in bf16: the encoder comes and goes per prompt.
        let evict_te = fp8.is_none() && (!opts.convrot || !cfg!(feature = "convrot"));
        let te = if evict_te {
            None
        } else {
            progress(0.0, "Loading the text encoder");
            let t0 = Instant::now();
            let te = load_text_encoder(model, &dev, dtype)?;
            tracing::info!(load_s = t0.elapsed().as_secs_f32(), "text encoder loaded");
            Some(te)
        };

        progress(0.3, "Loading the transformer");
        let t0 = Instant::now();
        let files = if fp8.is_some() { raw } else { crate::convrot_cache::resolve_files(&model.join("transformer"), opts.convrot, &opts.cache, &Policy::krea2())? };
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, &dev)? };
        let dit = Krea2Dit::load_with(&Krea2Config::default(), opts.convrot, fp8.as_ref(), vb)?;
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), convrot = opts.convrot && fp8.is_none(), fp8 = fp8.is_some(), full_precision = fp8.as_ref().map_or(0, |p| p.full_precision.len()), "DiT loaded");

        progress(0.9, "Loading the VAE");
        let t0 = Instant::now();
        let files = WeightSet::resolve(&model.join("vae"))?.files;
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, &dev)? };
        let vae = WanVae::load(vb)?;
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "VAE loaded");
        progress(1.0, "Loaded");
        Ok(Self { dev, dtype, tok, model: model.to_path_buf(), te: std::sync::Mutex::new(te), vision: std::sync::Mutex::new(None), dit, vae, evict_te })
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self.dev, Device::Cuda(_))
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// The dtype the DiT runs in (what a LoRA's tensors are put in).
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The conditioning `(1, tokens, 12, 2560)` for a prompt, grounded on `picture` when one is
    /// given (an edit: the picture through the vision tower, in view before the instruction).
    fn condition(&self, prompt: &str, picture: Option<&RefImage>) -> Result<Tensor> {
        let feats = match picture {
            Some(p) => Some(self.grounding_features(p)?),
            None => None,
        };
        let text = if feats.is_some() { krea2::edit_template(prompt) } else { krea2::template(prompt) };
        let ids: Vec<u32> = self.tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?.get_ids().to_vec();
        let mut slot = self.te.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(load_text_encoder(&self.model, &self.dev, self.dtype)?);
        }
        let te = slot.as_ref().ok_or_else(|| anyhow::anyhow!("no text encoder"))?;
        let cond = match &feats {
            Some(f) => krea2::condition_with_images(te, &ids, &[f])?,
            None => krea2::condition(te, &ids, &self.dev)?,
        }
        .unsqueeze(0)?;
        if self.evict_te {
            *slot = None;
            let _ = self.dev.synchronize();
        }
        Ok(cond)
    }

    /// The picture as the encoder sees it: its longest side brought to [`GROUNDING_PX`] at
    /// most, then to the tower's grid (ComfyUI's `process_qwen2vl_images`), through the 4B
    /// vision tower (loaded on first use).
    fn grounding_features(&self, p: &RefImage) -> Result<vision::VisionFeatures> {
        let img = rgb_image(p)?;
        let (w, h) = img.dimensions();
        let img = if w.max(h) > GROUNDING_PX {
            let s = f64::from(GROUNDING_PX) / f64::from(w.max(h));
            let (nw, nh) = (((f64::from(w) * s).round() as u32).max(1), ((f64::from(h) * s).round() as u32).max(1));
            image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle)
        } else {
            img
        };
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb = Tensor::from_vec(img.into_raw(), (h, w, 3), &self.dev)?.permute((2, 0, 1))?.to_dtype(DType::F32)?.affine(1.0 / 255.0, 0.0)?;
        let (vw, vh) = vision::vit_size(w, h, 3136, 12_845_056);
        let rgb = if (vw, vh) == (w, h) { rgb } else { vision::resize_bilinear(&rgb, vh, vw)? };
        let mut slot = self.vision.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            let t0 = Instant::now();
            let files = WeightSet::resolve(&self.model.join("text_encoder"))?.files;
            let vb32 = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, DType::F32, &self.dev)? };
            *slot = Some(VisionTower::load(&VisionConfig::qwen3vl_4b(), vb32.pp("model").pp("visual")).context("the Qwen3-VL-4B vision tower (an encoder file without it cannot edit)")?);
            tracing::info!(load_s = t0.elapsed().as_secs_f32(), "vision tower loaded");
        }
        let tower = slot.as_ref().ok_or_else(|| anyhow::anyhow!("no vision tower"))?;
        let (patches, gh, gw) = vision::patchify(&rgb, tower.config())?;
        tower.forward(&patches, gh, gw)
    }

    /// An edit's picture for the DiT: fitted to the target `tw × th` (centre-cropped to its
    /// aspect, then resampled; the comfyui-krea2edit `fit` path at a matched aspect) and
    /// through the VAE's encoder, in the sampler's space `(1, 16, th/8, tw/8)`.
    fn picture_latent(&self, p: &RefImage, tw: usize, th: usize) -> Result<Tensor> {
        anyhow::ensure!(self.vae.can_encode(), "this VAE file has no encoder; an edit needs it");
        let img = rgb_image(p)?;
        let (iw, ih) = (img.width() as f64, img.height() as f64);
        let s = (th as f64 / ih).max(tw as f64 / iw);
        let (cw, ch) = (((tw as f64 / s).round() as u32).clamp(1, img.width()), ((th as f64 / s).round() as u32).clamp(1, img.height()));
        let (x0, y0) = ((img.width() - cw) / 2, (img.height() - ch) / 2);
        let crop = image::imageops::crop_imm(&img, x0, y0, cw, ch).to_image();
        let fitted = if (cw as usize, ch as usize) == (tw, th) { crop } else { image::imageops::resize(&crop, tw as u32, th as u32, image::imageops::FilterType::CatmullRom) };
        let x = Tensor::from_vec(fitted.into_raw(), (th, tw, 3), &self.dev)?.permute((2, 0, 1))?.unsqueeze(0)?.to_dtype(DType::F32)?.affine(2.0 / 255.0, -1.0)?;
        self.vae.encode(&x)
    }

    /// Make one picture; `progress` hears every step, `cancel` is read between steps.
    pub fn generate(&self, req: &Request, progress: &mut dyn FnMut(f32, &str), cancel: &dyn Fn() -> bool) -> Result<Picture> {
        crate::device::bind_thread(&self.dev);
        anyhow::ensure!(req.steps > 0, "at least one step");
        // An edit: the first reference is the picture (one picture; the krea2_edit LoRAs take up
        // to two, a scene and a subject, which nothing asks for yet). Its size is the target's
        // when the request names none, floored to the 16-pixel grid.
        anyhow::ensure!(req.references.len() <= 1, "a Krea 2 edit takes one picture ({} given)", req.references.len());
        let picture = req.references.first();
        let (w, h) = match picture {
            Some(p) if req.width == 0 || req.height == 0 => (p.width / 16 * 16, p.height / 16 * 16),
            _ => (req.width, req.height),
        };
        anyhow::ensure!(w > 0 && h > 0 && w.is_multiple_of(16) && h.is_multiple_of(16), "the picture's sides must be positive multiples of 16 ({w}x{h} asked)");
        let unmatched = self.dit.set_loras(&req.loras)?;
        if !unmatched.is_empty() {
            tracing::warn!(modules = unmatched.len(), first = unmatched[0], "LoRA modules this model does not have (a file for another model?)");
        }
        let check = |cancel: &dyn Fn() -> bool| -> Result<()> {
            if cancel() {
                Err(Cancelled.into())
            } else {
                Ok(())
            }
        };
        let mut timings = Timings::default();

        check(cancel)?;
        progress(0.02, "Encoding the prompt");
        let t0 = Instant::now();
        let cond = self.condition(&req.prompt, picture)?;
        let edit = match picture {
            Some(p) => Some(EditRef { latent: self.picture_latent(p, w, h)?.to_dtype(self.dtype)?, boost: req.ref_boost }),
            None => None,
        };
        self.dev.synchronize()?;
        timings.encode_s = t0.elapsed().as_secs_f32();
        crate::profile::report("krea2 encode");

        // ComfyUI's noise on the (1, 16, h, w) latent, in the sampler's normalised space; the
        // `simple` schedule over the model's table; Euler with the velocity the DiT returns.
        let (lh, lw) = (h / wan_vae::SCALE, w / wan_vae::SCALE);
        let schedule = crate::comfy::FlowSchedule::simple(req.steps, crate::comfy::FlowSchedule::KREA2).map_err(anyhow::Error::msg)?;
        let noise = crate::comfy::torch_randn(req.seed, wan_vae::LATENT_CHANNELS * lh * lw).map_err(anyhow::Error::msg)?;
        let mut x = Tensor::from_vec(noise, (1, wan_vae::LATENT_CHANNELS, lh, lw), &self.dev)?;
        let t0 = Instant::now();
        for i in 0..req.steps {
            check(cancel)?;
            let v = self.dit.forward_edit(&x.to_dtype(self.dtype)?, schedule.sigma(i), &cond, edit.as_ref())?.to_dtype(DType::F32)?;
            x = (x + (v * f64::from(schedule.dt(i)))?)?;
            progress(0.05 + 0.85 * (i + 1) as f32 / req.steps as f32, "Sampling");
        }
        self.dev.synchronize()?;
        timings.denoise_s = t0.elapsed().as_secs_f32();
        crate::profile::report("krea2 sampling");

        check(cancel)?;
        progress(0.92, "Decoding");
        let t0 = Instant::now();
        let img = self.vae.decode(&self.vae.denormalize(&x)?)?;
        self.dev.synchronize()?;
        timings.decode_s = t0.elapsed().as_secs_f32();
        let (pw, ph, rgba) = wan_vae::to_rgba_u8(&img)?;
        tracing::info!(encode_s = timings.encode_s, denoise_s = timings.denoise_s, decode_s = timings.decode_s, width = pw, height = ph, steps = req.steps, "timings");
        progress(1.0, "Done");
        Ok(Picture { width: pw as u32, height: ph as u32, rgba, timings })
    }
}

/// A reference's pixels as RGB (its alpha dropped, as ComfyUI's `LoadImage` hands a picture to
/// the VAE and the encoder).
fn rgb_image(p: &RefImage) -> Result<image::RgbImage> {
    let rgba: image::RgbaImage =
        image::ImageBuffer::from_raw(p.width as u32, p.height as u32, p.rgba.clone()).ok_or_else(|| anyhow::anyhow!("a picture of {}x{} with {} bytes", p.width, p.height, p.rgba.len()))?;
    Ok(image::DynamicImage::ImageRgba8(rgba).to_rgb8())
}

fn load_text_encoder(model: &Path, dev: &Device, dtype: DType) -> Result<QwenTextEncoder> {
    let files = WeightSet::resolve(&model.join("text_encoder"))?.files;
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, dev)? };
    QwenTextEncoder::load_at(&TextConfig::qwen3vl_4b(), false, vb.pp("model")).context("the Qwen3-VL-4B encoder")
}
