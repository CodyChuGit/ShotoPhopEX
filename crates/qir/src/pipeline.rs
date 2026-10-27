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
use crate::model::dit::{self, PrefixCache, RefLatent};
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
    /// When the prefix cache of a request would not fit beside the resident models (the bf16
    /// text encoder alone holds 16 GB), drop the text encoder after it has encoded the prompt
    /// and load it again for the next request (about 7 s). Off, the driver pages instead,
    /// which made a two-reference edit twenty times slower on a 32 GB card.
    pub evict_text_encoder: bool,
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
            evict_text_encoder: true,
        }
    }
}

/// A reference picture: straight RGBA8, row-major (any size; the pipeline resizes it the way
/// ComfyUI's `TextEncodeQwenImage21` does at `resolution 0`, to multiples of 32).
/// A reference's packed normalised latent `(1, h·w, 64)` with its latent grid and its offset on
/// the target's grid (see [`RefImage::offset`]).
type PackedLatent = (Tensor, usize, usize, Option<(usize, usize)>);

#[derive(Clone, Debug)]
pub struct RefImage {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
    /// Where the picture sits inside the target, `(x, y)` in pixels, multiples of 16 (an
    /// expand's picture inside the larger canvas): its latent tokens then take the positions of
    /// the target tokens under them, and the picture is cropped to the latent grid rather than
    /// resized to ComfyUI's multiples of 32. `None`: a reference centred on the target, as
    /// ComfyUI places every reference.
    pub offset: Option<(usize, usize)>,
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
    /// LoRA adapters on the DiT for this request, with their strengths (`crate::lora`); empty
    /// runs the base model. Set on the resident model before sampling, so a request's adapters
    /// never outlive it.
    pub loras: crate::lora::LoraStack,
    /// An edit's pull toward its picture on the models that take one as in-context tokens (Krea 2's
    /// edit LoRAs: the target's attention to the picture multiplied by this; 1 = none).
    pub ref_boost: f32,
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
            loras: crate::lora::LoraStack::default(),
            ref_boost: 1.0,
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
    let img: image::RgbaImage = image::ImageBuffer::from_raw(r.width as u32, r.height as u32, r.rgba.clone())
        .ok_or_else(|| anyhow::anyhow!("reference of {}x{} with {} bytes", r.width, r.height, r.rgba.len()))?;
    let (w, h, img) = if r.offset.is_some() {
        // Placed on the target's grid: cropped to whole latent tokens, never resampled.
        let (w, h) = ((r.width / 16).max(1) * 16, (r.height / 16).max(1) * 16);
        anyhow::ensure!(w <= r.width && h <= r.height, "a placed reference needs at least 16x16 pixels");
        (w, h, if (w, h) == (r.width, r.height) { img } else { image::imageops::crop_imm(&img, 0, 0, w as u32, h as u32).to_image() })
    } else {
        let (w, h) = ref_size(r.width, r.height);
        (w, h, if (w, h) == (r.width, r.height) { img } else { image::imageops::resize(&img, w as u32, h as u32, image::imageops::FilterType::Lanczos3) })
    };
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

/// Where the models come from: Hugging Face's snapshot folder, or a ComfyUI `models` folder
/// holding ComfyUI's single files ([`crate::comfy_files`], Rust rebuild R4).
#[derive(Clone, Debug)]
pub enum Source {
    Snapshot(PathBuf),
    Comfy(crate::comfy_files::Qwen21Files),
}

impl Source {
    /// `dir` as a snapshot, else as a ComfyUI `models` folder with the 2.1 files.
    pub fn find(dir: &Path) -> Option<Self> {
        if is_snapshot(dir) {
            Some(Self::Snapshot(dir.to_path_buf()))
        } else {
            crate::comfy_files::find_qwen21(dir).map(Self::Comfy)
        }
    }

    fn tokenizer(&self) -> Result<Tokenizer> {
        match self {
            Self::Snapshot(d) => Tokenizer::from_file(d.join("processor/tokenizer.json")).map_err(|e| anyhow::anyhow!("load tokenizer: {e}")),
            Self::Comfy(f) => crate::comfy_files::qwen_tokenizer(&f.tokenizer),
        }
    }

    /// The text encoder's files (its vision tower and language-model head live there too).
    fn text_files(&self) -> Result<Vec<PathBuf>> {
        match self {
            Self::Snapshot(d) => Ok(WeightSet::resolve(&d.join("text_encoder"))?.files),
            Self::Comfy(f) => Ok(vec![f.text.clone()]),
        }
    }

    /// The language model's root in those files (ComfyUI's repack drops `language_model`).
    fn lm_root(&self) -> &'static str {
        match self {
            Self::Snapshot(_) => "model.language_model",
            Self::Comfy(_) => "model",
        }
    }
}

/// The three models, resident.
pub struct Pipeline {
    dev: Device,
    dtype: DType,
    tok: Tokenizer,
    /// Tokens of the system turn, dropped from the encoder's output.
    drop: usize,
    /// The models' source, for loading the text encoder again after an eviction.
    src: Source,
    /// The text encoder; `None` between an eviction (`Options::evict_text_encoder`) and the
    /// next prompt.
    te: std::sync::Mutex<Option<QwenTextEncoder>>,
    dit: QwenImageDit,
    vae: QwenImageVae,
    vision: Option<VisionTower>,
    opts: Options,
    /// The final norm and language-model head, for [`Pipeline::chat`]; loaded on first use.
    head: std::sync::Mutex<Option<ChatHead>>,
}

/// The text encoder's last two pieces, which conditioning never uses and a chat does.
struct ChatHead {
    norm: candle_nn::RmsNorm,
    head: candle_core::quantized::QMatMul,
}

/// `<|im_end|>` and `<|endoftext|>`: a reply ends at either.
const IM_END: u32 = 151_645;
const END_OF_TEXT: u32 = 151_643;

/// How [`Pipeline::chat`] samples (the app's enhance templates: temperature 0.3, top-k 40,
/// top-p 0.9, at most 120 tokens).
#[derive(Clone, Debug)]
pub struct ChatOptions {
    pub max_new: usize,
    /// 0 picks the likeliest token every time.
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self { max_new: 120, temperature: 0.3, top_k: 40, top_p: 0.9, seed: 0 }
    }
}

/// A small seeded generator for the sampling (xorshift64*).
struct ChatRng(u64);

impl ChatRng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15 | 1)
    }

    /// Uniform in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// The next token from `logits`: the argmax at temperature 0, else top-k then top-p over the
/// tempered softmax, drawn with `rng`.
fn sample(logits: &[f32], opts: &ChatOptions, rng: &mut ChatRng) -> u32 {
    let argmax = || logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0 as u32;
    if opts.temperature <= 0.0 || logits.is_empty() {
        return argmax();
    }
    let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().filter(|(_, v)| v.is_finite()).collect();
    let k = opts.top_k.clamp(1, top.len().max(1));
    top.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    top.truncate(k);
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    let Some(&(_, max)) = top.first() else { return argmax() };
    let mut probs: Vec<f32> = top.iter().map(|(_, v)| ((v - max) / opts.temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);
    // Keep the smallest prefix whose mass reaches top-p.
    let mut mass = 0.0;
    let mut keep = probs.len();
    for (i, p) in probs.iter().enumerate() {
        mass += p;
        if mass >= opts.top_p {
            keep = i + 1;
            break;
        }
    }
    let total: f32 = probs[..keep].iter().sum();
    let mut r = rng.next_f32() * total;
    for (i, p) in probs[..keep].iter().enumerate() {
        if r < *p {
            return top[i].0 as u32;
        }
        r -= p;
    }
    top[keep - 1].0 as u32
}

impl Pipeline {
    /// Load everything from `model` (a snapshot directory, or a ComfyUI `models` folder with
    /// ComfyUI's 2.1 files) onto the best device. `progress` hears each phase.
    pub fn load(model: &Path, opts: Options, progress: &mut dyn FnMut(f32, &str)) -> Result<Self> {
        let Some(src) = Source::find(model) else {
            anyhow::bail!(
                "{} is neither a Qwen-Image-2.1 snapshot (processor/, text_encoder/, transformer/, vae/) nor a ComfyUI models folder with its files \
                 (diffusion_models/qwen_image_2.1_bf16, text_encoders/qwen3vl_8b_bf16, vae/qwen_image_2.1_vae_bf16, and ComfyUI's qwen25_tokenizer)",
                model.display()
            );
        };
        let dev = crate::device::best_device()?;
        let dtype = if matches!(dev, Device::Cuda(_)) { DType::BF16 } else { DType::F32 };
        let tok = src.tokenizer()?;
        let sysp = format!("<|im_start|>system\n{}<|im_end|>\n", tmpl::SYS_PROMPT);
        let drop = tok.encode(sysp, false).map_err(|e| anyhow::anyhow!("encode system turn: {e}"))?.get_ids().len();

        progress(0.0, "Loading the text encoder");
        let t0 = Instant::now();
        let te = load_text_from(&src, &dev, dtype, opts.quant_text, opts.text_gguf.as_deref())?;
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "text encoder loaded");

        progress(0.4, "Loading the transformer");
        let t0 = Instant::now();
        let files = match &src {
            Source::Snapshot(d) => crate::convrot_cache::resolve_dit_files(&d.join("transformer"), opts.convrot, &opts.cache)?,
            Source::Comfy(f) => crate::convrot_cache::resolve_file_set(&f.dit, vec![f.dit.clone()], opts.convrot, &opts.cache, &crate::convrot_cache::Policy::qwen_image_21())?,
        };
        let dit_vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, &dev)? };
        let dit = QwenImageDit::load(32, 64, false, opts.convrot, dit_vb)?;
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "DiT loaded");

        progress(0.8, "Loading the VAE");
        let t0 = Instant::now();
        let vae_dt = if opts.vae_f32 { DType::F32 } else { dtype };
        let vae = match &src {
            Source::Snapshot(d) => {
                let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("vae/config.json")).context("vae/config.json")?)?;
                let f32vec = |k: &str| -> Vec<f32> { cfg[k].as_array().map_or(vec![], |a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()) };
                let vae_files = WeightSet::resolve(&d.join("vae"))?.files;
                let vae_vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&vae_files, vae_dt, &dev)? };
                QwenImageVae::load(&VaeConfig::default(), &f32vec("latents_mean"), &f32vec("latents_std"), vae_vb)?
            }
            Source::Comfy(f) => {
                use crate::comfy_files::{qwen21_vae_name, Renamed, QWEN21_LATENTS_MEAN, QWEN21_LATENTS_STD};
                let vb = unsafe { Renamed::new(std::slice::from_ref(&f.vae), qwen21_vae_name)? }.var_builder(vae_dt, &dev);
                QwenImageVae::load(&VaeConfig::default(), &QWEN21_LATENTS_MEAN, &QWEN21_LATENTS_STD, vb)?
            }
        };
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "VAE loaded");

        let vision = if opts.vision {
            progress(0.9, "Loading the vision tower");
            let t0 = Instant::now();
            let files = src.text_files()?;
            let vb32 = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, DType::F32, &dev)? };
            let tower = VisionTower::load(&VisionConfig::default(), vb32.pp("model").pp("visual"))?;
            tracing::info!(load_s = t0.elapsed().as_secs_f32(), "vision tower loaded");
            Some(tower)
        } else {
            None
        };
        progress(1.0, "Loaded");
        Ok(Self { dev, dtype, tok, drop, src, te: std::sync::Mutex::new(Some(te)), dit, vae, vision, opts, head: std::sync::Mutex::new(None) })
    }

    /// Run `f` on the text encoder, loading it again first if it was evicted.
    fn with_text_encoder<R>(&self, f: impl FnOnce(&QwenTextEncoder) -> Result<R>) -> Result<R> {
        let mut slot = self.te.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            let t0 = Instant::now();
            *slot = Some(load_text_from(&self.src, &self.dev, self.dtype, self.opts.quant_text, self.opts.text_gguf.as_deref())?);
            tracing::info!(load_s = t0.elapsed().as_secs_f32(), "text encoder loaded again");
        }
        match slot.as_ref() {
            Some(te) => f(te),
            None => anyhow::bail!("no text encoder"),
        }
    }

    /// Is the text encoder on the device right now?
    pub fn text_encoder_resident(&self) -> bool {
        self.te.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    /// The prompt is encoded; make room for the sampling if the device cannot hold the
    /// prefix cache of `prefix_len` rows (plus the step's working set) beside everything
    /// resident: drop the text encoder, which the next prompt loads again.
    fn make_room(&self, prefix_len: usize, target_len: usize) {
        if !self.opts.evict_text_encoder || !self.text_encoder_resident() {
            return;
        }
        // The pool's retained memory is this process's to reuse (the driver counts it as used).
        let Some((free, total)) = crate::device::usable_vram(&self.dev) else { return };
        let cache = self.dit.prefix_cache_bytes(prefix_len);
        // The step's working set: the joint keys and values the target attends to (two
        // copies in flight), the prefix pass's f32 score chunks, and a margin.
        let working = 4 * (prefix_len + target_len) * dit::INNER * 2 + 3 * dit::HEADS * 1024 * prefix_len * 4 + (1 << 30);
        let need = cache + working;
        if free >= need {
            return;
        }
        tracing::info!(free_gb = free as f32 / 1e9, need_gb = need as f32 / 1e9, total_gb = total as f32 / 1e9, "evicting the text encoder for the sampling");
        *self.te.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let _ = self.dev.synchronize();
    }

    /// The dtype the DiT runs in (what a LoRA's tensors are put in).
    pub fn dtype(&self) -> DType {
        self.dtype
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
        let hidden = self.with_text_encoder(|te| te.forward(&Tensor::from_vec(ids, (1, seq), &self.dev)?))?;
        Ok(hidden.narrow(1, self.drop, seq - self.drop)?.contiguous()?)
    }

    /// The prompt's conditioning with pictures: the tower's features spliced into the prompt,
    /// the picture rows dropped again on the way out, and where each reference latent goes.
    fn encode_with_images(&self, text: &str, feats: &[VisionFeatures]) -> Result<(Tensor, Vec<usize>)> {
        let ids: Vec<u32> =
            self.tok.encode(tmpl::edit_template(text, feats.len()), false).map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?.get_ids().to_vec();
        let refs: Vec<&VisionFeatures> = feats.iter().collect();
        let (hidden, spans) = self.with_text_encoder(|te| te.forward_with_images(&ids, &refs))?;
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
            let offset = match r.offset {
                Some((x, y)) => {
                    anyhow::ensure!(x.is_multiple_of(16) && y.is_multiple_of(16), "a reference's offset must be a multiple of 16 pixels ({x}, {y} given)");
                    Some((y / 16, x / 16))
                }
                None => None,
            };
            latents.push((packed, lh, lw, offset));
            let (vw, vh) = vision::vit_size(w, h, 3136, 12_845_056);
            let rgb = if (vw, vh) == (w, h) { rgb } else { vision::resize_bilinear(&rgb, vh, vw)? };
            let (patches, gh, gw) = vision::patchify(&rgb, tower.config())?;
            feats.push(tower.forward(&patches, gh, gw)?);
        }
        Ok((latents, feats))
    }

    /// The final norm and the language-model head of the text encoder, loaded on first use
    /// (the head as Q8_0: 0.6 GB instead of 1.2), for [`Self::chat`].
    fn with_head<R>(&self, f: impl FnOnce(&ChatHead) -> Result<R>) -> Result<R> {
        let mut slot = self.head.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            let t0 = Instant::now();
            let files = self.src.text_files()?;
            let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, self.dtype, &self.dev)? };
            let cfg = TextConfig::default();
            let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp(self.src.lm_root()).pp("norm"))?;
            let w = vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight").context("the text encoder has no lm_head (a text-only repack?)")?;
            let head = if matches!(self.dev, Device::Cuda(_)) {
                candle_core::quantized::QMatMul::from_qtensor(candle_core::quantized::QTensor::quantize(&w, candle_core::quantized::GgmlDType::Q8_0)?)?
            } else {
                candle_core::quantized::QMatMul::Tensor(w)
            };
            tracing::info!(load_s = t0.elapsed().as_secs_f32(), "language-model head loaded");
            *slot = Some(ChatHead { norm, head });
        }
        match slot.as_ref() {
            Some(h) => f(h),
            None => anyhow::bail!("no language-model head"),
        }
    }

    /// Answer as the vision-language model the text encoder is (Qwen3-VL 8B Instruct, with
    /// its language-model head): `system` and `user` as the chat's turns, `image` in view, the
    /// reply sampled token by token (temperature, top-k, top-p with `opts.seed`) until the end
    /// of the turn or `opts.max_new` tokens. What ComfyUI's `TextGenerate` node does with the
    /// same file, so the app's prompt rewriter and judge run in the worker without a second
    /// model on the card. The prompt runs once and every layer keeps its keys and values
    /// ([`text_encoder::KvCache`]), so each token after it is one row through the decoder.
    pub fn chat(&self, system: &str, user: &str, image: Option<&RefImage>, opts: &ChatOptions, cancel: &dyn Fn() -> bool) -> Result<String> {
        crate::device::bind_thread(&self.dev);
        let t0 = Instant::now();
        let feats: Vec<VisionFeatures> = match image {
            Some(r) => {
                let Some(tower) = &self.vision else { anyhow::bail!("this pipeline was loaded without the vision tower") };
                let (_, rgb, (w, h)) = prepare_ref(r, &self.dev)?;
                let (vw, vh) = vision::vit_size(w, h, 3136, 12_845_056);
                let rgb = if (vw, vh) == (w, h) { rgb } else { vision::resize_bilinear(&rgb, vh, vw)? };
                let (patches, gh, gw) = vision::patchify(&rgb, tower.config())?;
                vec![tower.forward(&patches, gh, gw)?]
            }
            None => Vec::new(),
        };
        let vision_block = if feats.is_empty() { "" } else { tmpl::VISION_BLOCK };
        let text = format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{vision_block}{user}<|im_end|>\n<|im_start|>assistant\n");
        let mut ids: Vec<u32> = self.tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode the chat: {e}"))?.get_ids().to_vec();
        let prompt_len = ids.len();
        let refs: Vec<&VisionFeatures> = feats.iter().collect();
        let mut rng = ChatRng::new(opts.seed);
        let mut prefill_s = 0.0f32;
        self.with_text_encoder(|te| {
            let tp = Instant::now();
            let (mut last, mut cache) = te.prefill(&ids, &refs)?; // (1, hidden)
            prefill_s = tp.elapsed().as_secs_f32();
            for _ in 0..opts.max_new {
                if cancel() {
                    return Err(Cancelled.into());
                }
                let logits: Tensor = self.with_head(|h| Ok(candle_core::Module::forward(&h.head, &last.apply(&h.norm)?)?))?;
                let logits = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                let next = sample(&logits, opts, &mut rng);
                if next == IM_END || next == END_OF_TEXT {
                    break;
                }
                ids.push(next);
                last = te.step(next, &mut cache)?;
            }
            Ok(())
        })?;
        let reply = self.tok.decode(&ids[prompt_len..], true).map_err(|e| anyhow::anyhow!("decode the reply: {e}"))?;
        tracing::info!(tokens = ids.len() - prompt_len, prompt_tokens = prompt_len, prefill_s, chat_s = t0.elapsed().as_secs_f32(), "chat");
        Ok(reply.trim().to_string())
    }

    /// Make one picture. `progress` hears every step; `cancel` is read between steps and a
    /// true answer ends the run with [`Cancelled`].
    pub fn generate(&self, req: &Request, progress: &mut dyn FnMut(f32, &str), cancel: &dyn Fn() -> bool) -> Result<Picture> {
        crate::device::bind_thread(&self.dev);
        if req.steps == 0 {
            anyhow::bail!("at least one step");
        }
        // The request's adapters on the DiT (none clears the last request's).
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
            (0, 0, Some((_, lh, lw, None))) => (lw * 16, lh * 16),
            (w, h, _) => (w, h),
        };
        if w == 0 || h == 0 || !w.is_multiple_of(16) || !h.is_multiple_of(16) {
            anyhow::bail!("the picture's sides must be positive multiples of 16 ({w}x{h} asked)");
        }
        for (_, lh, lw, offset) in &ref_latents {
            if let Some((oy, ox)) = offset {
                if ox + lw > w / 16 || oy + lh > h / 16 {
                    anyhow::bail!("a placed reference of {}x{} at ({}, {}) does not fit the {w}x{h} picture", lw * 16, lh * 16, ox * 16, oy * 16);
                }
            }
        }
        check(cancel)?;
        progress(0.02, "Encoding the prompt");
        let (cond, slots) = if feats.is_empty() { (self.encode(&req.prompt)?, Vec::new()) } else { self.encode_with_images(&req.prompt, &feats)? };
        let neg = if req.guidance > 1.0 {
            Some(if feats.is_empty() { (self.encode(&req.negative)?, Vec::new()) } else { self.encode_with_images(&req.negative, &feats)? })
        } else {
            None
        };
        let refs: Vec<RefLatent> = ref_latents.iter().map(|(t, lh, lw, o)| RefLatent { packed: t, h: *lh, w: *lw, offset: *o }).collect();
        // With references, the prefix (text and reference rows) through the DiT once; each
        // step then runs the target's rows against the cached keys and values.
        let (lw, lh) = (w / 16, h / 16);
        let caches: Option<(PrefixCache, Option<PrefixCache>)> = if self.opts.prefix_cache && !refs.is_empty() {
            check(cancel)?;
            progress(0.03, "Reading the references into the model");
            let prefix_len = cond.dim(1)? + ref_latents.iter().map(|(_, lh, lw, _)| lh * lw).sum::<usize>();
            self.make_room(prefix_len * if neg.is_some() { 2 } else { 1 }, lh * lw);
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
    load_text_from(&Source::Snapshot(model.to_path_buf()), dev, dtype, quant_text, text_gguf)
}

/// [`load_text_encoder`] from either source (ComfyUI's single file roots the language model at
/// `model`).
fn load_text_from(src: &Source, dev: &Device, dtype: DType, quant_text: bool, text_gguf: Option<&Path>) -> Result<QwenTextEncoder> {
    if let Some(g) = text_gguf {
        let qvb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(g, dev)?;
        QwenTextEncoder::load_gguf(&TextConfig::default(), qvb)
    } else {
        let files = src.text_files()?;
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, dev)? };
        QwenTextEncoder::load_at(&TextConfig::default(), quant_text, vb.pp(src.lm_root()))
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

    #[test]
    fn sampling_is_greedy_at_zero_and_stays_in_the_top_k_and_top_p_otherwise() {
        let logits = vec![0.0, 5.0, 4.9, -3.0, 1.0];
        let greedy = ChatOptions { temperature: 0.0, ..ChatOptions::default() };
        let mut rng = ChatRng::new(1);
        assert_eq!(sample(&logits, &greedy, &mut rng), 1);
        // Top-k 2: only tokens 1 and 2 are ever drawn, both of them over many draws.
        let opts = ChatOptions { temperature: 1.0, top_k: 2, top_p: 1.0, ..ChatOptions::default() };
        let mut seen = [0usize; 5];
        for _ in 0..400 {
            seen[sample(&logits, &opts, &mut rng) as usize] += 1;
        }
        assert_eq!((seen[0], seen[3], seen[4]), (0, 0, 0), "{seen:?}");
        assert!(seen[1] > 50 && seen[2] > 50, "{seen:?}");
        // A small top-p keeps the likeliest alone.
        let opts = ChatOptions { temperature: 1.0, top_k: 5, top_p: 0.1, ..ChatOptions::default() };
        for _ in 0..50 {
            assert_eq!(sample(&logits, &opts, &mut rng), 1);
        }
        // The same seed draws the same tokens.
        let opts = ChatOptions { temperature: 1.0, top_k: 5, top_p: 1.0, ..ChatOptions::default() };
        let draw = |seed| {
            let mut r = ChatRng::new(seed);
            (0..20).map(|_| sample(&logits, &opts, &mut r)).collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7));
    }
}
