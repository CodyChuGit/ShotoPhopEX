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
use crate::model::krea2::{self, Krea2Config, Krea2Dit};
use crate::model::text_encoder::{QwenTextEncoder, TextConfig};
use crate::model::wan_vae::{self, WanVae};
use crate::pipeline::{Cancelled, Picture, Request, Timings};

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
        Ok(Self { dev, dtype, tok, model: model.to_path_buf(), te: std::sync::Mutex::new(te), dit, vae, evict_te })
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

    /// The conditioning `(1, tokens, 12, 2560)` for a prompt.
    fn condition(&self, prompt: &str) -> Result<Tensor> {
        let ids: Vec<u32> = self.tok.encode(krea2::template(prompt), false).map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?.get_ids().to_vec();
        let mut slot = self.te.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(load_text_encoder(&self.model, &self.dev, self.dtype)?);
        }
        let te = slot.as_ref().ok_or_else(|| anyhow::anyhow!("no text encoder"))?;
        let cond = krea2::condition(te, &ids, &self.dev)?.unsqueeze(0)?;
        if self.evict_te {
            *slot = None;
            let _ = self.dev.synchronize();
        }
        Ok(cond)
    }

    /// Make one picture; `progress` hears every step, `cancel` is read between steps.
    pub fn generate(&self, req: &Request, progress: &mut dyn FnMut(f32, &str), cancel: &dyn Fn() -> bool) -> Result<Picture> {
        crate::device::bind_thread(&self.dev);
        anyhow::ensure!(req.steps > 0, "at least one step");
        anyhow::ensure!(req.references.is_empty(), "Krea 2 text to image takes no reference pictures yet");
        let (w, h) = (req.width, req.height);
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
        let cond = self.condition(&req.prompt)?;
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
            let v = self.dit.forward(&x.to_dtype(self.dtype)?, schedule.sigma(i), &cond)?.to_dtype(DType::F32)?;
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

fn load_text_encoder(model: &Path, dev: &Device, dtype: DType) -> Result<QwenTextEncoder> {
    let files = WeightSet::resolve(&model.join("text_encoder"))?.files;
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, dev)? };
    QwenTextEncoder::load_at(&TextConfig::qwen3vl_4b(), false, vb.pp("model")).context("the Qwen3-VL-4B encoder")
}
