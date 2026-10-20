//! A resident text-to-image pipeline: the tokenizer, the text encoder, the DiT and the VAE
//! loaded once and kept on the device, then `prompt → picture` as many times as asked, with
//! progress reports and cancellation between steps. What the `generate` subcommand does per
//! process, as a library the engine worker keeps alive (genai-native plan Phase 2).
//!
//! The sampling follows ComfyUI's conventions by default ([`crate::comfy`]: its CPU noise for
//! the seed, the `simple` flow schedule, its timestep rounding), so a picture matches the
//! ComfyUI route's for the same prompt, seed and steps; [`Schedule::Diffusers`] keeps the
//! reference pipeline's resolution-shifted schedule and candle's device RNG.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use crate::Result;
use crate::convrot_cache::CacheSpec;
use crate::loader::WeightSet;
use crate::model::config::VaeConfig;
use crate::model::dit::QwenImageDit;
use crate::model::scheduler::{FlowConfig, FlowMatchEuler};
use crate::model::text_encoder::{QwenTextEncoder, TextConfig, prompt as tmpl};
use crate::model::vae::{self, QwenImageVae, VaeTiling};

/// How the pipeline is loaded.
#[derive(Clone, Debug)]
pub struct Options {
    /// ConvRot W8A8 INT8 for the DiT's linears (needs the `convrot` feature; ignored without).
    pub convrot: bool,
    /// Where the prequantized ConvRot DiT is cached.
    pub cache: CacheSpec,
    /// Q8_0 the text encoder on load (about half its 16 GB; a 24 GB card needs it).
    pub quant_text: bool,
    /// A pre-quantized GGUF text encoder instead of the snapshot's bf16 shards.
    pub text_gguf: Option<PathBuf>,
    pub vae_tile: VaeTiling,
    /// Decode in f32 (bf16 on CUDA otherwise).
    pub vae_f32: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            convrot: cfg!(feature = "convrot"),
            cache: CacheSpec::from_cli(None, false, false),
            quant_text: false,
            text_gguf: None,
            vae_tile: VaeTiling::Auto,
            vae_f32: false,
        }
    }
}

/// Which sampling conventions a generation follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    /// ComfyUI's: `torch.manual_seed` noise, the `simple` scheduler, bf16 timestep rounding.
    Comfy,
    /// diffusers': candle's device RNG, the resolution-shifted `FlowMatchEulerDiscreteScheduler`.
    Diffusers,
}

/// One text-to-image generation.
#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: String,
    /// Used only when `guidance > 1` (true CFG, two DiT passes per step).
    pub negative: String,
    /// Pixels; multiples of 16.
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub seed: u64,
    pub guidance: f32,
    pub schedule: Schedule,
}

impl Request {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: 25,
            seed: 0,
            guidance: 1.0,
            schedule: Schedule::Comfy,
        }
    }
}

/// Seconds spent in each phase of one generation (device-synchronised).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timings {
    pub encode_s: f32,
    pub denoise_s: f32,
    pub decode_s: f32,
}

/// A finished picture: straight RGBA8, row-major.
#[derive(Clone, Debug)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub timings: Timings,
}

/// The error a cancelled generation returns (look for it with `Error::downcast_ref`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

/// The snapshot layout the pipeline reads (`processor/`, `text_encoder/`, `transformer/`,
/// `vae/`, as Hugging Face publishes Qwen-Image-2.1).
pub fn is_snapshot(dir: &Path) -> bool {
    dir.join("processor/tokenizer.json").is_file()
        && dir.join("text_encoder").is_dir()
        && dir.join("transformer").is_dir()
        && dir.join("vae/config.json").is_file()
}

/// The three models, resident.
pub struct Pipeline {
    dev: Device,
    dtype: DType,
    tok: Tokenizer,
    /// Tokens of the system turn, dropped from the encoder's output.
    drop: usize,
    te: QwenTextEncoder,
    dit: QwenImageDit,
    vae: QwenImageVae,
    opts: Options,
}

impl Pipeline {
    /// Load everything from `model` (a snapshot directory) onto the best device. `progress`
    /// hears each phase.
    pub fn load(model: &Path, opts: Options, progress: &mut dyn FnMut(f32, &str)) -> Result<Self> {
        if !is_snapshot(model) {
            anyhow::bail!(
                "{} is not a Qwen-Image-2.1 snapshot (processor/, text_encoder/, transformer/, vae/)",
                model.display()
            );
        }
        let dev = crate::device::best_device()?;
        let dtype = if matches!(dev, Device::Cuda(_)) {
            DType::BF16
        } else {
            DType::F32
        };
        let tok = Tokenizer::from_file(model.join("processor/tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        let sysp = format!("<|im_start|>system\n{}<|im_end|>\n", tmpl::SYS_PROMPT);
        let drop = tok
            .encode(sysp, false)
            .map_err(|e| anyhow::anyhow!("encode system turn: {e}"))?
            .get_ids()
            .len();

        progress(0.0, "Loading the text encoder");
        let t0 = Instant::now();
        let te = load_text_encoder(model, &dev, dtype, opts.quant_text, opts.text_gguf.as_deref())?;
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "text encoder loaded");

        progress(0.4, "Loading the transformer");
        let t0 = Instant::now();
        let files = crate::convrot_cache::resolve_dit_files(&model.join("transformer"), opts.convrot, &opts.cache)?;
        let dit_vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, &dev)? };
        let dit = QwenImageDit::load(32, 64, false, opts.convrot, dit_vb)?;
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "DiT loaded");

        progress(0.8, "Loading the VAE");
        let t0 = Instant::now();
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(model.join("vae/config.json")).context("vae/config.json")?,
        )?;
        let f32vec = |k: &str| -> Vec<f32> {
            cfg[k].as_array().map_or(vec![], |a| {
                a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()
            })
        };
        let vae_dt = if opts.vae_f32 { DType::F32 } else { dtype };
        let vae_files = WeightSet::resolve(&model.join("vae"))?.files;
        let vae_vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&vae_files, vae_dt, &dev)? };
        let vae = QwenImageVae::load(&VaeConfig::default(), &f32vec("latents_mean"), &f32vec("latents_std"), vae_vb)?;
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "VAE loaded");
        progress(1.0, "Loaded");
        Ok(Self { dev, dtype, tok, drop, te, dit, vae, opts })
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self.dev, Device::Cuda(_))
    }

    pub fn options(&self) -> &Options {
        &self.opts
    }

    /// The prompt's embedding `(1, tokens, 4096)` without the system turn.
    fn encode(&self, text: &str) -> Result<Tensor> {
        let ids: Vec<u32> = self
            .tok
            .encode(tmpl::t2i_template(text), false)
            .map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?
            .get_ids()
            .to_vec();
        let seq = ids.len();
        let hidden = self.te.forward(&Tensor::from_vec(ids, (1, seq), &self.dev)?)?;
        Ok(hidden.narrow(1, self.drop, seq - self.drop)?.contiguous()?)
    }

    /// Make one picture. `progress` hears every step; `cancel` is read between steps and a
    /// true answer ends the run with [`Cancelled`].
    pub fn generate(
        &self,
        req: &Request,
        progress: &mut dyn FnMut(f32, &str),
        cancel: &dyn Fn() -> bool,
    ) -> Result<Picture> {
        let (w, h) = (req.width, req.height);
        if w == 0 || h == 0 || !w.is_multiple_of(16) || !h.is_multiple_of(16) {
            anyhow::bail!("the picture's sides must be positive multiples of 16 ({w}x{h} asked)");
        }
        if req.steps == 0 {
            anyhow::bail!("at least one step");
        }
        let check = |cancel: &dyn Fn() -> bool| -> Result<()> {
            if cancel() { Err(Cancelled.into()) } else { Ok(()) }
        };
        let mut timings = Timings::default();

        // 1. Text.
        check(cancel)?;
        progress(0.0, "Encoding the prompt");
        let t0 = Instant::now();
        let cond = self.encode(&req.prompt)?;
        let neg = if req.guidance > 1.0 { Some(self.encode(&req.negative)?) } else { None };
        self.dev.synchronize()?;
        timings.encode_s = t0.elapsed().as_secs_f32();

        // 2. Noise and schedule, then the Euler loop.
        let (lw, lh) = (w / 16, h / 16);
        let img_seq = lw * lh;
        let (tsteps, dts): (Vec<f32>, Vec<f32>) = match req.schedule {
            Schedule::Comfy => {
                use crate::comfy::{FlowSchedule, comfy_timestep};
                let s = FlowSchedule::simple(req.steps, FlowSchedule::QWEN_IMAGE_21).map_err(anyhow::Error::msg)?;
                ((0..req.steps).map(|i| comfy_timestep(s.sigma(i))).collect(), (0..req.steps).map(|i| s.dt(i)).collect())
            }
            Schedule::Diffusers => {
                let s = FlowMatchEuler::new(&FlowConfig::default(), req.steps, img_seq);
                (s.timesteps().iter().map(|t| (*t / 1000.0) as f32).collect(), (0..req.steps).map(|i| s.dt(i) as f32).collect())
            }
        };
        let t0 = Instant::now();
        let mut latents = match req.schedule {
            Schedule::Comfy => {
                let noise = crate::comfy::noise_packed(req.seed, lh, lw).map_err(anyhow::Error::msg)?;
                Tensor::from_vec(noise, (1, img_seq, 64), &self.dev)?
            }
            Schedule::Diffusers => {
                self.dev.set_seed(req.seed)?;
                Tensor::randn(0f32, 1f32, (1, img_seq, 64), &self.dev)?
            }
        }
        .to_dtype(DType::F32)?;
        for (i, (t, dt)) in tsteps.iter().zip(&dts).enumerate() {
            check(cancel)?;
            let tt = Tensor::from_vec(vec![*t], (1,), &self.dev)?;
            let v = self.velocity(&latents.to_dtype(self.dtype)?, &cond, neg.as_ref(), req.guidance, &tt, (lh, lw), img_seq)?;
            latents = (latents + (v * *dt as f64)?)?;
            progress(0.05 + 0.85 * (i + 1) as f32 / req.steps as f32, "Sampling");
        }
        self.dev.synchronize()?;
        timings.denoise_s = t0.elapsed().as_secs_f32();

        // 3. Pixels.
        check(cancel)?;
        progress(0.92, "Decoding");
        let t0 = Instant::now();
        let z = vae::unpack_latents_hw(&latents, lh, lw)?;
        let img = self.vae.decode_auto(&z, self.opts.vae_tile)?;
        self.dev.synchronize()?;
        timings.decode_s = t0.elapsed().as_secs_f32();
        let (pw, ph, rgba) = vae::to_rgba_u8(&img)?;
        tracing::info!(
            encode_s = timings.encode_s,
            denoise_s = timings.denoise_s,
            decode_s = timings.decode_s,
            width = pw,
            height = ph,
            steps = req.steps,
            "timings"
        );
        progress(1.0, "Done");
        Ok(Picture { width: pw as u32, height: ph as u32, rgba, timings })
    }

    /// The DiT's velocity over the image tokens, with true CFG when `guidance > 1` and a
    /// negative embedding is given.
    #[allow(clippy::too_many_arguments)]
    fn velocity(
        &self,
        latents: &Tensor,
        cond: &Tensor,
        neg: Option<&Tensor>,
        guidance: f32,
        tt: &Tensor,
        (h, w): (usize, usize),
        img_seq: usize,
    ) -> Result<Tensor> {
        let joint = self.dit.forward(latents, cond, tt, h, w)?;
        let (_b, jl, _) = joint.dims3()?;
        let v_cond = joint.narrow(1, jl - img_seq, img_seq)?.to_dtype(DType::F32)?;
        if guidance > 1.0 {
            if let Some(neg) = neg {
                let ju = self.dit.forward(latents, neg, tt, h, w)?;
                let (_b, jlu, _) = ju.dims3()?;
                let v_uncond = ju.narrow(1, jlu - img_seq, img_seq)?.to_dtype(DType::F32)?;
                return Ok((&v_uncond + ((&v_cond - &v_uncond)? * guidance as f64)?)?);
            }
        }
        Ok(v_cond)
    }
}

/// The text encoder: a pre-quantized GGUF when given (Q8_0, no bf16 transient), else the
/// snapshot's bf16 shards, quantized on load when asked.
pub fn load_text_encoder(
    model: &Path,
    dev: &Device,
    dtype: DType,
    quant_text: bool,
    text_gguf: Option<&Path>,
) -> Result<QwenTextEncoder> {
    if let Some(g) = text_gguf {
        let qvb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(g, dev)?;
        QwenTextEncoder::load_gguf(&TextConfig::default(), qvb)
    } else {
        let files = WeightSet::resolve(&model.join("text_encoder"))?.files;
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, dev)? };
        QwenTextEncoder::load(&TextConfig::default(), quant_text, vb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_snapshot_needs_its_four_parts() {
        let d = std::env::temp_dir().join(format!("qir-pipeline-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        assert!(!is_snapshot(&d));
        for sub in ["processor", "text_encoder", "transformer", "vae"] {
            std::fs::create_dir_all(d.join(sub)).unwrap();
        }
        assert!(!is_snapshot(&d), "the files matter, not the folders");
        std::fs::write(d.join("processor/tokenizer.json"), "{}").unwrap();
        std::fs::write(d.join("vae/config.json"), "{}").unwrap();
        assert!(is_snapshot(&d));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn request_defaults_follow_the_comfy_template() {
        let r = Request::new("a fox");
        assert_eq!((r.width, r.height, r.steps, r.guidance, r.schedule), (1024, 1024, 25, 1.0, Schedule::Comfy));
        assert_eq!(Cancelled.to_string(), "cancelled");
        let e: anyhow::Error = Cancelled.into();
        assert!(e.downcast_ref::<Cancelled>().is_some());
    }
}
