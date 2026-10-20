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
}

impl Krea2Options {
    /// ConvRot on when the build has it.
    pub fn defaults() -> Self {
        Self { convrot: cfg!(feature = "convrot") }
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
}

impl Krea2Pipeline {
    pub fn load(model: &Path, opts: Krea2Options, progress: &mut dyn FnMut(f32, &str)) -> Result<Self> {
        if !is_snapshot(model) {
            anyhow::bail!("{} is not a Krea 2 weights folder (tokenizer/, text_encoder/, transformer/, vae/)", model.display());
        }
        let dev = crate::device::best_device()?;
        let dtype = if matches!(dev, Device::Cuda(_)) { DType::BF16 } else { DType::F32 };
        let tok = Tokenizer::from_file(model.join("tokenizer/tokenizer.json")).map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

        progress(0.0, "Loading the text encoder");
        let t0 = Instant::now();
        let te = load_text_encoder(model, &dev, dtype)?;
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "text encoder loaded");

        progress(0.3, "Loading the transformer");
        let t0 = Instant::now();
        let files = WeightSet::resolve(&model.join("transformer"))?.files;
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, &dev)? };
        let dit = Krea2Dit::load(&Krea2Config::default(), opts.convrot, vb)?;
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), convrot = opts.convrot, "DiT loaded");

        progress(0.9, "Loading the VAE");
        let t0 = Instant::now();
        let files = WeightSet::resolve(&model.join("vae"))?.files;
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, &dev)? };
        let vae = WanVae::load(vb)?;
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "VAE loaded");
        progress(1.0, "Loaded");
        Ok(Self { dev, dtype, tok, model: model.to_path_buf(), te: std::sync::Mutex::new(Some(te)), dit, vae })
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self.dev, Device::Cuda(_))
    }

    /// The conditioning `(1, tokens, 12, 2560)` for a prompt.
    fn condition(&self, prompt: &str) -> Result<Tensor> {
        let ids: Vec<u32> = self.tok.encode(krea2::template(prompt), false).map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?.get_ids().to_vec();
        let mut slot = self.te.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(load_text_encoder(&self.model, &self.dev, self.dtype)?);
        }
        let te = slot.as_ref().ok_or_else(|| anyhow::anyhow!("no text encoder"))?;
        Ok(krea2::condition(te, &ids, &self.dev)?.unsqueeze(0)?)
    }

    /// Make one picture; `progress` hears every step, `cancel` is read between steps.
    pub fn generate(&self, req: &Request, progress: &mut dyn FnMut(f32, &str), cancel: &dyn Fn() -> bool) -> Result<Picture> {
        anyhow::ensure!(req.steps > 0, "at least one step");
        anyhow::ensure!(req.references.is_empty(), "Krea 2 text to image takes no reference pictures yet");
        let (w, h) = (req.width, req.height);
        anyhow::ensure!(w > 0 && h > 0 && w.is_multiple_of(16) && h.is_multiple_of(16), "the picture's sides must be positive multiples of 16 ({w}x{h} asked)");
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
