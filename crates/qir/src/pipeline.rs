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
use candle_core::{DType, Device, IndexOp, Tensor};
use tokenizers::Tokenizer;

use crate::convrot_cache::CacheSpec;
use crate::loader::WeightSet;
use crate::model::config::VaeConfig;
use crate::model::dit::QwenImageDit;
use crate::model::dit::{PrefixCache, RefLatent};
use crate::model::scheduler::{FlowConfig, FlowMatchEuler};
use crate::model::text_encoder::{self, prompt as tmpl, QwenTextEncoder, TextConfig};
use crate::model::vae::{self, QwenImageVae, VaeTiling};
use crate::model::vision::{self, VisionConfig, VisionFeatures, VisionTower};
use crate::Result;

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
    /// Load the vision tower (2.4 GB in f32) so requests may carry reference pictures.
    pub vision: bool,
    /// With references: run the text and reference rows through the DiT once per generation
    /// and keep their keys and values (`dit::PrefixCache`, ComfyUI's `QwenImage21Cache`), so
    /// every step runs the target's rows alone. Off, each step runs the whole joint sequence
    /// (the plain reference for the cache, several times slower).
    pub prefix_cache: bool,
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
            vision: true,
            prefix_cache: true,
        }
    }
}

/// A reference picture: straight RGBA8, row-major (any size; the pipeline resizes it the way
/// ComfyUI's `TextEncodeQwenImage21` does at `resolution 0`, to multiples of 32).
/// A reference's packed normalised latent `(1, h·w, 64)` with its latent grid.
type PackedLatent = (Tensor, usize, usize);

#[derive(Clone, Debug)]
pub struct RefImage {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
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
    /// Pixels; multiples of 16. With references, 0 means the first reference's size (what
    /// ComfyUI's edit template does: its empty latent is sized to `image_1`).
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub seed: u64,
    pub guidance: f32,
    pub schedule: Schedule,
    /// Reference pictures in prompt order (`<image1>`, `<image2>`, …): the picture being edited
    /// first. Empty for text to image.
    pub references: Vec<RefImage>,
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
            references: Vec::new(),
        }
    }
}

/// `TextEncodeQwenImage21` at `resolution 0`: each side rounded to the nearest multiple of
/// 32, at least 32.
pub fn ref_size(width: usize, height: usize) -> (usize, usize) {
    let r = |v: usize| (((v as f64 / 32.0).round() as usize) * 32).max(32);
    (r(width), r(height))
}

/// A reference picture resized to `ref_size` (Lanczos, as the node) and split into what the
/// two consumers see: RGBA in `[-1, 1]` for the VAE, RGB over white in `[0, 1]` for the
/// vision tower.
fn prepare_ref(r: &RefImage, dev: &Device) -> Result<(Tensor, Tensor, (usize, usize))> {
    let (w, h) = ref_size(r.width, r.height);
    let img: image::RgbaImage = image::ImageBuffer::from_raw(r.width as u32, r.height as u32, r.rgba.clone())
        .ok_or_else(|| anyhow::anyhow!("reference of {}x{} with {} bytes", r.width, r.height, r.rgba.len()))?;
    let img = if (w, h) == (r.width, r.height) { img } else { image::imageops::resize(&img, w as u32, h as u32, image::imageops::FilterType::Lanczos3) };
    let rgba = vae::from_rgba_u8(w, h, img.as_raw(), dev)?; // (1, 4, H, W) in [-1, 1]
    let px = rgba.i(0)?; // (4, H, W)
    let rgb01 = ((px.narrow(0, 0, 3)? + 1.0)? * 0.5)?;
    let a01 = ((px.narrow(0, 3, 1)? + 1.0)? * 0.5)?;
    let over_white = (rgb01.broadcast_mul(&a01)? + (a01.neg()? + 1.0)?.broadcast_as((3, h, w))?)?;
    Ok((rgba, over_white, (w, h)))
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
    vision: Option<VisionTower>,
    opts: Options,
}

impl Pipeline {
    /// Load everything from `model` (a snapshot directory) onto the best device. `progress`
    /// hears each phase.
    pub fn load(model: &Path, opts: Options, progress: &mut dyn FnMut(f32, &str)) -> Result<Self> {
        if !is_snapshot(model) {
            anyhow::bail!("{} is not a Qwen-Image-2.1 snapshot (processor/, text_encoder/, transformer/, vae/)", model.display());
        }
        let dev = crate::device::best_device()?;
        let dtype = if matches!(dev, Device::Cuda(_)) { DType::BF16 } else { DType::F32 };
        let tok = Tokenizer::from_file(model.join("processor/tokenizer.json")).map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        let sysp = format!("<|im_start|>system\n{}<|im_end|>\n", tmpl::SYS_PROMPT);
        let drop = tok.encode(sysp, false).map_err(|e| anyhow::anyhow!("encode system turn: {e}"))?.get_ids().len();

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
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(model.join("vae/config.json")).context("vae/config.json")?)?;
        let f32vec = |k: &str| -> Vec<f32> { cfg[k].as_array().map_or(vec![], |a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()) };
        let vae_dt = if opts.vae_f32 { DType::F32 } else { dtype };
        let vae_files = WeightSet::resolve(&model.join("vae"))?.files;
        let vae_vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&vae_files, vae_dt, &dev)? };
        let vae = QwenImageVae::load(&VaeConfig::default(), &f32vec("latents_mean"), &f32vec("latents_std"), vae_vb)?;
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "VAE loaded");

        let vision = if opts.vision {
            progress(0.9, "Loading the vision tower");
            let t0 = Instant::now();
            let files = WeightSet::resolve(&model.join("text_encoder"))?.files;
            let vb32 = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, DType::F32, &dev)? };
            let tower = VisionTower::load(&VisionConfig::default(), vb32.pp("model").pp("visual"))?;
            tracing::info!(load_s = t0.elapsed().as_secs_f32(), "vision tower loaded");
            Some(tower)
        } else {
            None
        };
        progress(1.0, "Loaded");
        Ok(Self { dev, dtype, tok, drop, te, dit, vae, vision, opts })
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
        let ids: Vec<u32> = self.tok.encode(tmpl::t2i_template(text), false).map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?.get_ids().to_vec();
        let seq = ids.len();
        let hidden = self.te.forward(&Tensor::from_vec(ids, (1, seq), &self.dev)?)?;
        Ok(hidden.narrow(1, self.drop, seq - self.drop)?.contiguous()?)
    }

    /// The prompt's conditioning with pictures: the tower's features spliced into the prompt,
    /// the picture rows dropped again on the way out, and where each reference latent goes.
    fn encode_with_images(&self, text: &str, feats: &[VisionFeatures]) -> Result<(Tensor, Vec<usize>)> {
        let ids: Vec<u32> =
            self.tok.encode(tmpl::edit_template(text, feats.len()), false).map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?.get_ids().to_vec();
        let refs: Vec<&VisionFeatures> = feats.iter().collect();
        let (hidden, spans) = self.te.forward_with_images(&ids, &refs)?;
        text_encoder::condition(&hidden, &spans, self.drop)
    }

    /// Each reference through the VAE encoder (its packed normalised latent and grid) and the
    /// vision tower.
    fn prepare_references(&self, refs: &[RefImage]) -> Result<(Vec<PackedLatent>, Vec<VisionFeatures>)> {
        let Some(tower) = &self.vision else { anyhow::bail!("this pipeline was loaded without the vision tower; references need it") };
        anyhow::ensure!(self.vae.can_encode(), "this VAE weight set has no encoder; references need it");
        let mut latents = Vec::with_capacity(refs.len());
        let mut feats = Vec::with_capacity(refs.len());
        for r in refs {
            let (rgba, rgb, (w, h)) = prepare_ref(r, &self.dev)?;
            let z = self.vae.encode(&rgba)?; // (1, 64, h/16, w/16) f32
            let (lh, lw) = (h / 16, w / 16);
            let packed = z.reshape((1, 64, lh * lw))?.transpose(1, 2)?.contiguous()?.to_dtype(self.dtype)?;
            latents.push((packed, lh, lw));
            let (vw, vh) = vision::vit_size(w, h, 3136, 12_845_056);
            let rgb = if (vw, vh) == (w, h) { rgb } else { vision::resize_bilinear(&rgb, vh, vw)? };
            let (patches, gh, gw) = vision::patchify(&rgb, tower.config())?;
            feats.push(tower.forward(&patches, gh, gw)?);
        }
        Ok((latents, feats))
    }

    /// Make one picture. `progress` hears every step; `cancel` is read between steps and a
    /// true answer ends the run with [`Cancelled`].
    pub fn generate(&self, req: &Request, progress: &mut dyn FnMut(f32, &str), cancel: &dyn Fn() -> bool) -> Result<Picture> {
        if req.steps == 0 {
            anyhow::bail!("at least one step");
        }
        let check = |cancel: &dyn Fn() -> bool| -> Result<()> {
            if cancel() {
                Err(Cancelled.into())
            } else {
                Ok(())
            }
        };
        let mut timings = Timings::default();

        // 1. The references (their latents and vision features), then the text.
        check(cancel)?;
        let t0 = Instant::now();
        let (ref_latents, feats) = if req.references.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            progress(0.0, "Reading the references");
            self.prepare_references(&req.references)?
        };
        // The target: the request's size, or the first reference's when it says nothing.
        let (w, h) = match (req.width, req.height, ref_latents.first()) {
            (0, 0, Some((_, lh, lw))) => (lw * 16, lh * 16),
            (w, h, _) => (w, h),
        };
        if w == 0 || h == 0 || !w.is_multiple_of(16) || !h.is_multiple_of(16) {
            anyhow::bail!("the picture's sides must be positive multiples of 16 ({w}x{h} asked)");
        }
        check(cancel)?;
        progress(0.02, "Encoding the prompt");
        let (cond, slots) = if feats.is_empty() { (self.encode(&req.prompt)?, Vec::new()) } else { self.encode_with_images(&req.prompt, &feats)? };
        let neg = if req.guidance > 1.0 {
            Some(if feats.is_empty() { (self.encode(&req.negative)?, Vec::new()) } else { self.encode_with_images(&req.negative, &feats)? })
        } else {
            None
        };
        let refs: Vec<RefLatent> = ref_latents.iter().map(|(t, lh, lw)| RefLatent { packed: t, h: *lh, w: *lw }).collect();
        // With references, the prefix (text and reference rows) through the DiT once; each
        // step then runs the target's rows against the cached keys and values.
        let (lw, lh) = (w / 16, h / 16);
        let caches: Option<(PrefixCache, Option<PrefixCache>)> = if self.opts.prefix_cache && !refs.is_empty() {
            check(cancel)?;
            progress(0.03, "Reading the references into the model");
            let c = self.dit.prefix_cache(&cond, lh, lw, &refs, &slots)?;
            let n = match &neg {
                Some((n, ns)) => Some(self.dit.prefix_cache(n, lh, lw, &refs, ns)?),
                None => None,
            };
            Some((c, n))
        } else {
            None
        };
        self.dev.synchronize()?;
        timings.encode_s = t0.elapsed().as_secs_f32();

        // 2. Noise and schedule, then the Euler loop.
        let img_seq = lw * lh;
        let (tsteps, dts): (Vec<f32>, Vec<f32>) = match req.schedule {
            Schedule::Comfy => {
                use crate::comfy::{comfy_timestep, FlowSchedule};
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
            let v = match &caches {
                Some((c, n)) => self.velocity_cached(&latents.to_dtype(self.dtype)?, c, n.as_ref(), req.guidance, &tt)?,
                None => self.velocity(
                    &latents.to_dtype(self.dtype)?,
                    (&cond, &slots),
                    neg.as_ref().map(|(n, s)| (n, s.as_slice())),
                    req.guidance,
                    &tt,
                    (lh, lw),
                    &refs,
                )?,
            };
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

    /// The DiT's velocity over the target's tokens, with true CFG when `guidance > 1` and a
    /// negative conditioning is given. `cond` and `neg` carry their image slots.
    #[allow(clippy::too_many_arguments)]
    fn velocity(
        &self,
        latents: &Tensor,
        cond: (&Tensor, &[usize]),
        neg: Option<(&Tensor, &[usize])>,
        guidance: f32,
        tt: &Tensor,
        (h, w): (usize, usize),
        refs: &[RefLatent],
    ) -> Result<Tensor> {
        let img_seq = h * w;
        let joint = self.dit.forward_refs(latents, cond.0, tt, h, w, refs, cond.1)?;
        let (_b, jl, _) = joint.dims3()?;
        let v_cond = joint.narrow(1, jl - img_seq, img_seq)?.to_dtype(DType::F32)?;
        if guidance > 1.0 {
            if let Some((neg, nslots)) = neg {
                let ju = self.dit.forward_refs(latents, neg, tt, h, w, refs, nslots)?;
                let (_b, jlu, _) = ju.dims3()?;
                let v_uncond = ju.narrow(1, jlu - img_seq, img_seq)?.to_dtype(DType::F32)?;
                return Ok((&v_uncond + ((&v_cond - &v_uncond)? * guidance as f64)?)?);
            }
        }
        Ok(v_cond)
    }

    /// [`Self::velocity`] against the prefix caches: the target's rows alone through the DiT.
    fn velocity_cached(&self, latents: &Tensor, cond: &PrefixCache, neg: Option<&PrefixCache>, guidance: f32, tt: &Tensor) -> Result<Tensor> {
        let v_cond = self.dit.forward_cached(latents, tt, cond)?.to_dtype(DType::F32)?;
        if guidance > 1.0 {
            if let Some(neg) = neg {
                let v_uncond = self.dit.forward_cached(latents, tt, neg)?.to_dtype(DType::F32)?;
                return Ok((&v_uncond + ((&v_cond - &v_uncond)? * guidance as f64)?)?);
            }
        }
        Ok(v_cond)
    }
}

/// The text encoder: a pre-quantized GGUF when given (Q8_0, no bf16 transient), else the
/// snapshot's bf16 shards, quantized on load when asked.
pub fn load_text_encoder(model: &Path, dev: &Device, dtype: DType, quant_text: bool, text_gguf: Option<&Path>) -> Result<QwenTextEncoder> {
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
