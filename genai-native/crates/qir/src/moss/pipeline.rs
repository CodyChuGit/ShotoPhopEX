//! MOSS-Image text to image, resident: the Qwen3.5-9B encoder, the DiT and the FLUX.2 decoder
//! loaded once, sampled with the official pipeline's conventions (torch's CPU noise for the
//! seed, 50 Euler steps on the shift-5 schedule snapped to the training grid, CFG 3 whose
//! negative branch is the empty prompt with block 22 skipped).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use super::dit::{self, MossConfig, MossDit};
use super::text::{self, Qwen35Text, TextConfig};
use super::vae::{self, Flux2Decoder, Flux2Encoder};
use super::weights::{QLinear, Weights};
use crate::model::text_encoder::mrope_positions;
use crate::model::vision::{self as vit, VisionConfig, VisionTower};
use crate::pipeline::{Cancelled, Picture, RefImage, Request, Timings};
use crate::preview::StepPreview;
use crate::Result;

/// The official defaults.
pub const STEPS: usize = 50;
pub const GUIDANCE: f32 = 3.0;
pub const SHIFT: f64 = 5.0;
/// The block the negative branch leaves out (zero-based).
pub const UNCOND_SKIP: usize = 22;
/// MOSS Edit's references: their long edge at most this (the official `infer.py`'s
/// `max_edge`), sides rounded to 16, never enlarged.
pub const REFERENCE_EDGE: usize = 1024;
pub const LOW_VRAM_REFERENCE_EDGE: usize = 512;
/// Cards with less memory than this run in 12 GB mode ([`low_vram_mode`]).
pub const LOW_VRAM_BELOW: usize = 20 << 30;
/// An edit's output when the request names no size: the picture's aspect at about this many
/// pixels (1 MP, the edit workflow's), sides multiples of 16.
pub const EDIT_PIXELS: usize = 1024 * 1024;

/// How a step moves the latent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampler {
    /// First-order Euler on the velocity (the official pipeline's).
    Euler,
    /// RES4LYF's `res_2s`: a two-stage exponential integrator on the denoised estimate
    /// (midpoint in log-SNR), two model calls a step. The last step to σ = 0 is one call.
    Res2s,
}

/// The sampling recipe of a text-to-image run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    /// The flow schedule's shift (5 official; 1 is linear σ, more steps at low noise).
    pub shift: f64,
    pub sampler: Sampler,
    /// Skip block [`UNCOND_SKIP`] in the negative branch (the official skip-layer guidance).
    pub skip_uncond: bool,
    /// `res_2s`'s ancestral noise, RES4LYF's `eta` in its "hard" mode (0 = the plain ODE,
    /// `res_2s_ode`; the `res_2s` that KSamplerSelect lists runs 0.5 on steps and substeps).
    pub eta: f64,
    /// An edit in ai-toolkit's MOSS edit layout (ostris' ComfyUI-MOSS-Image-Ostris-Edit, the
    /// one its edit LoRAs train on): each prompt opens with `Picture N: ` per reference, the
    /// references sit at the prompt's text index of the RoPE, encoded at the output's size.
    /// MOSS Edit's own layout (text index 0, long edge [`REFERENCE_EDGE`]) otherwise.
    pub ostris_refs: bool,
    /// How much of an [`Self::ostris_refs`] edit is redrawn: 1 starts from noise (the author's
    /// empty latent), less starts from the picture's own latent noised that far (ComfyUI's
    /// KSampler `denoise`).
    pub denoise: f32,
    /// An [`Self::ostris_refs`] edit takes the picture's colours back (the wavelet colour fix).
    pub color_fix: bool,
}

impl Sampling {
    /// The official pipeline: Euler on the shift-5 schedule, block 22 skipped in the negative.
    pub const OFFICIAL: Self = Self { shift: SHIFT, sampler: Sampler::Euler, skip_uncond: true, eta: 0.0, ostris_refs: false, denoise: 1.0, color_fix: false };
    /// The app's MOSS-Image text to image, after RaymondLuxuryYacht's MOSS realism workflow
    /// (Civitai 3035895, 2026-11-04): RES4LYF's `res_2s` (ancestral, eta 0.5), 20 steps, plain CFG
    /// 3 (no skipped block) with a real negative ([`REALISM_NEGATIVE`]). The workflow's schedule
    /// is linear σ (shift 1); in the app (2026-11-11, three prompts) that came out dark and
    /// muddy with broken signs, so the official shift 5 stays. About 30 % faster than 50 Euler steps.
    pub const REALISM: Self = Self { shift: SHIFT, sampler: Sampler::Res2s, skip_uncond: false, eta: 0.5, ostris_refs: false, denoise: 1.0, color_fix: false };
    /// An ai-toolkit edit LoRA's run (NO8D's MOSS HighQuality restore, Civitai 2436859): Euler, no
    /// skipped block, references in ostris' layout, as its author's graph; then, from the in-app
    /// rounds of 2026-11-17 (three photos against the shipped Upscale), 45 % redrawn from the
    /// picture's latent (from noise, 1.0, rewrites text and faces; 0.45 is the best cleanup,
    /// 0.3 keeps faces) and the wavelet colour fix (brightness and colour back to the picture's,
    /// ΔE down a third to a half, detail and edges unchanged).
    pub const OSTRIS_EDIT: Self =
        Self { shift: SHIFT, sampler: Sampler::Euler, skip_uncond: false, eta: 0.0, ostris_refs: true, denoise: 0.45, color_fix: true };

    /// `self` redrawing `denoise` (0..1] of an edit.
    pub fn with_denoise(mut self, denoise: f32) -> Self {
        self.denoise = denoise.clamp(0.01, 1.0);
        self
    }

    /// `self` with the A/B hooks applied: `QIR_MOSS_SHIFT`, `QIR_MOSS_SAMPLER` (`euler` /
    /// `res_2s`), `QIR_MOSS_SKIP` (`0` / `1`), `QIR_MOSS_ETA` (`res_2s`'s noise, 0..1).
    pub fn with_env(mut self) -> Self {
        let var = |k: &str| std::env::var(k).ok().map(|v| v.trim().to_ascii_lowercase()).filter(|v| !v.is_empty());
        if let Some(s) = var("QIR_MOSS_SHIFT").and_then(|v| v.parse::<f64>().ok()).filter(|s| s.is_finite() && *s > 0.0) {
            self.shift = s;
        }
        match var("QIR_MOSS_SAMPLER").as_deref() {
            Some("euler") => self.sampler = Sampler::Euler,
            Some("res_2s" | "res2s") => {
                self.sampler = Sampler::Res2s;
                self.eta = 0.5;
            }
            Some("res_2s_ode" | "res2s_ode") => {
                self.sampler = Sampler::Res2s;
                self.eta = 0.0;
            }
            _ => {}
        }
        if let Some(e) = var("QIR_MOSS_ETA").and_then(|v| v.parse::<f64>().ok()).filter(|e| (0.0..=1.0).contains(e)) {
            self.eta = e;
        }
        if let Some(d) = var("QIR_MOSS_DENOISE").and_then(|v| v.parse::<f32>().ok()).filter(|d| *d > 0.0 && *d <= 1.0) {
            self.denoise = d;
        }
        match var("QIR_MOSS_COLORFIX").as_deref() {
            Some("wavelet" | "1") => self.color_fix = true,
            Some("none" | "0") => self.color_fix = false,
            _ => {}
        }
        match var("QIR_MOSS_SKIP").as_deref() {
            Some("0") => self.skip_uncond = false,
            Some("1") => self.skip_uncond = true,
            _ => {}
        }
        self
    }
}

/// The negative prompt of the realism recipe (the workflow's own).
pub const REALISM_NEGATIVE: &str = "blur, smooth, anime, render, cartoon, 3d";

/// Which fine-tune a template runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// MOSS-Image-SFT, the instruction-tuned release (the default).
    Sft,
    /// MOSS-Image-Base.
    Base,
    /// lodestones' Kromoss (Apache-2.0), a full fine-tune of Base at 1024 px; its 8-step DMD2
    /// LoRA makes it the turbo model.
    Kromoss,
}

/// MOSS-Image's files in a ComfyUI-layout `models` folder (Comfy-Org/MOSS-Image's names; each
/// in its folder or one subfolder below it).
#[derive(Clone, Debug)]
pub struct MossFiles {
    pub dit: PathBuf,
    pub text: PathBuf,
    pub vae: PathBuf,
    pub tokenizer: PathBuf,
}

/// The file names, first found wins: ConvRot INT8 before bf16, the small decoder before the full.
fn dit_names(v: Variant) -> [&'static str; 2] {
    match v {
        Variant::Sft => ["moss_image_sft_int8_convrot.safetensors", "moss_image_sft_bf16.safetensors"],
        Variant::Base => ["moss_image_base_int8_convrot.safetensors", "moss_image_base_bf16.safetensors"],
        Variant::Kromoss => ["kromoss_int8_convrot.safetensors", "kromoss_bf16.safetensors"],
    }
}
const TEXT: &[&str] = &["qwen3.5_9b_int8_convrot.safetensors", "qwen3.5_9b_bf16.safetensors"];
const VAE: &[&str] = &["flux2-vae_small.safetensors", "flux2-vae.safetensors"];
/// Qwen3.5's `tokenizer.json` (its vocabulary is not Qwen2.5's), in its own folder.
const TOKENIZER_DIR: &str = "qwen35_tokenizer";

/// `names` in `dir` or in one of its subfolders (sorted, so the choice is stable).
fn find(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    let mut dirs = vec![dir.to_path_buf()];
    if let Ok(rd) = std::fs::read_dir(dir) {
        let mut subs: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect();
        subs.sort();
        dirs.extend(subs);
    }
    names.iter().find_map(|n| dirs.iter().map(|d| d.join(n)).find(|p| p.is_file()))
}

/// MOSS-Image under `root`, when every file is there.
pub fn find_files(root: &Path, variant: Variant) -> Option<MossFiles> {
    let text = find(&root.join("text_encoders"), TEXT)?;
    let own = root.join("text_encoders").join(TOKENIZER_DIR).join("tokenizer.json");
    let beside = text.parent().map(|d| d.join("tokenizer.json"));
    let tokenizer = std::iter::once(own).chain(beside).find(|p| p.is_file())?;
    Some(MossFiles { dit: find(&root.join("diffusion_models"), &dit_names(variant))?, text, vae: find(&root.join("vae"), VAE)?, tokenizer })
}

/// A prompt's encoding, kept for the next request with the same prompt.
struct Cached {
    prompt: String,
    hidden: Tensor,
}

pub struct MossPipeline {
    dev: Device,
    dtype: DType,
    tok: Tokenizer,
    files: MossFiles,
    /// 12 GB mode: the encoder and the DiT never on the card together (the encoder loads,
    /// encodes and goes before the DiT comes back). Off on cards with room for both.
    low_vram: bool,
    te: Mutex<Option<Arc<Qwen35Text>>>,
    dit: Mutex<Option<Arc<MossDit>>>,
    vae: Flux2Decoder,
    /// MOSS Edit's reference encoder (the small VAE file holds it), when the file has one.
    encoder: Option<Flux2Encoder>,
    positive: Mutex<Option<Cached>>,
    negative: Mutex<Option<Cached>>,
    /// The language-model head, loaded on a first chat ([`Self::chat`]).
    head: Mutex<Option<Arc<QLinear>>>,
    /// The encoder file's Qwen3.5 vision tower (f32), loaded on a first prompt with pictures in
    /// view ([`Sampling::ostris_refs`]).
    vision: Mutex<Option<Arc<VisionTower>>>,
}

impl MossPipeline {
    pub fn load(files: &MossFiles, progress: &mut dyn FnMut(f32, &str)) -> Result<Self> {
        let dev = crate::device::best_device()?;
        let dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
        let tok = Tokenizer::from_file(&files.tokenizer).map_err(|e| anyhow::anyhow!("load {}: {e}", files.tokenizer.display()))?;

        let low_vram = low_vram_mode(&dev);
        tracing::info!(low_vram, "MOSS memory mode");
        let this_te = Mutex::new(None);
        let this_dit = Mutex::new(None);
        if !low_vram {
            progress(0.0, "Loading the text encoder");
            *this_te.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(load_text(files, &dev, dtype)?));
        }
        progress(0.5, "Loading the transformer");
        *this_dit.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(load_dit(files, &dev, dit_dtype(dtype))?));
        progress(0.9, "Loading the VAE");
        // bf16 on CUDA: 60.0 dB against ComfyUI's f32 decode (f32 60.7), faster and half the memory.
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[&files.vae], dtype, &dev)? };
        let vae = Flux2Decoder::load(vb.clone()).context("the FLUX.2 VAE")?;
        let encoder = match Flux2Encoder::load(vb) {
            Ok(e) => Some(e),
            Err(e) => {
                tracing::info!(error = %e, "no FLUX.2 encoder in the VAE file: MOSS Edit is off");
                None
            }
        };
        progress(1.0, "Loaded");
        Ok(Self {
            dev,
            dtype,
            tok,
            files: files.clone(),
            low_vram,
            te: this_te,
            dit: this_dit,
            vae,
            encoder,
            positive: Mutex::new(None),
            negative: Mutex::new(None),
            head: Mutex::new(None),
            vision: Mutex::new(None),
        })
    }

    pub fn is_cuda(&self) -> bool {
        self.dev.is_cuda()
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The token ids of `prompt` in the encoder's template.
    pub fn tokens(&self, prompt: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(text::template(prompt), false).map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?.get_ids().to_vec())
    }

    /// The dtype LoRA files load in for this pipeline's DiT.
    pub fn lora_dtype(&self) -> DType {
        dit_dtype(self.dtype)
    }

    /// Whether the pipeline runs in 12 GB mode.
    pub fn low_vram(&self) -> bool {
        self.low_vram
    }

    /// The text encoder, loaded when it is not on the card (in 12 GB mode the DiT goes first).
    fn text_encoder(&self) -> Result<Arc<Qwen35Text>> {
        // Locked in one order (encoder, then DiT) everywhere.
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

    /// The DiT, loaded when it is not on the card (in 12 GB mode the encoder goes first).
    fn transformer(&self) -> Result<Arc<MossDit>> {
        let mut te = self.te.lock().unwrap_or_else(|e| e.into_inner());
        let mut dit = self.dit.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(d) = dit.as_ref() {
            return Ok(Arc::clone(d));
        }
        if self.low_vram && te.take().is_some() {
            crate::device::trim_pool(&self.dev);
        }
        let d = Arc::new(load_dit(&self.files, &self.dev, dit_dtype(self.dtype))?);
        *dit = Some(Arc::clone(&d));
        Ok(d)
    }

    /// A text answer from the encoder itself, Qwen3.5-9B (thinking off): the reply to `user`
    /// under `system`, sampled as the other chats are ([`crate::pipeline::ChatOptions`]). The
    /// prompt rewriter of a MOSS picture then needs no other model on the card.
    pub fn chat(&self, system: &str, user: &str, opts: &crate::pipeline::ChatOptions, cancel: &dyn Fn() -> bool) -> Result<String> {
        crate::device::bind_thread(&self.dev);
        let t0 = Instant::now();
        let te = self.text_encoder()?;
        let head = self.lm_head()?;
        let (reply, tokens, prompt_tokens) = write_reply(&self.tok, &te, &head, system, user, opts, cancel)?;
        drop(head);
        if self.low_vram {
            // 12 GB mode: the head (1-2 GB) goes again, so the DiT has its room back.
            *self.head.lock().unwrap_or_else(|e| e.into_inner()) = None;
            crate::device::trim_pool(&self.dev);
        }
        tracing::info!(tokens, prompt_tokens, chat_s = t0.elapsed().as_secs_f32(), "moss chat");
        Ok(reply)
    }

    /// The encoder file's `lm_head` (248320 × 4096), on the card from the first chat on.
    fn lm_head(&self) -> Result<Arc<QLinear>> {
        let mut slot = self.head.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = slot.as_ref() {
            return Ok(Arc::clone(h));
        }
        let w = Weights::open(&self.files.text, &self.dev, self.dtype, true)?;
        let t0 = Instant::now();
        let h = Arc::new(load_lm_head(&w)?);
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "MOSS chat head loaded");
        *slot = Some(Arc::clone(&h));
        Ok(h)
    }

    /// The vision tower of the encoder file (`model.visual.*`, Qwen3.5-9B's: Qwen3-VL's ViT
    /// without DeepStack), on the card from the first prompt with pictures on.
    fn vision(&self) -> Result<Arc<VisionTower>> {
        let mut slot = self.vision.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = slot.as_ref() {
            return Ok(Arc::clone(v));
        }
        let t0 = Instant::now();
        let vb = candle_nn::VarBuilder::from_tensors(visual_tensors(&self.files.text, &self.dev)?, DType::F32, &self.dev);
        let cfg = VisionConfig { deepstack_visual_indexes: Vec::new(), ..VisionConfig::default() };
        let v = Arc::new(VisionTower::load(&cfg, vb.pp("model").pp("visual")).context("the encoder file's Qwen3.5 vision tower (model.visual.*)")?);
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "MOSS vision tower loaded");
        *slot = Some(Arc::clone(&v));
        Ok(v)
    }

    /// The encoder's last hidden state `(rows, 4096)` for `prompt` with `pictures` in view, as
    /// ostris' `TextEncodeMOSSImageOstrisEdit` encodes it (the layout ai-toolkit's MOSS edit
    /// LoRAs train on): `Picture N: <|vision_start|><|image_pad|><|vision_end|>` per picture
    /// ahead of the prompt in MOSS's template, each picture downscaled (area, never enlarged)
    /// to fit 384² pixels, then to the ViT's 32-pixel grid (bilinear), its merged rows in
    /// place of `<|image_pad|>` at multimodal RoPE positions.
    pub fn encode_with_pictures(&self, prompt: &str, pictures: &[RefImage]) -> Result<Tensor> {
        let labels: String = (1..=pictures.len()).map(|i| format!("Picture {i}: <|vision_start|><|image_pad|><|vision_end|>")).collect();
        let ids = self.tokens(format!("{labels}{prompt}").trim())?;
        let pad = self.tok.token_to_id("<|image_pad|>").context("the tokenizer has no <|image_pad|>")?;
        let tower = self.vision()?;
        let feats = pictures
            .iter()
            .map(|r| {
                let (patches, gh, gw) = vit_patches(r, tower.config(), &self.dev)?;
                tower.forward_editplus(&patches, gh, gw)
            })
            .collect::<Result<Vec<_>>>()?;
        let te = self.text_encoder()?;
        let text = te.embed_ids(&ids)?;
        let at: Vec<usize> = ids.iter().enumerate().filter(|(_, t)| **t == pad).map(|(i, _)| i).collect();
        anyhow::ensure!(at.len() == feats.len(), "{} <|image_pad|> tokens for {} pictures", at.len(), feats.len());
        let mut parts = Vec::new();
        let mut spans = Vec::new();
        let (mut cursor, mut len) = (0usize, 0usize);
        for (&p, f) in at.iter().zip(&feats) {
            if p > cursor {
                parts.push(text.narrow(0, cursor, p - cursor)?);
                len += p - cursor;
            }
            spans.push((len, f.tokens()));
            parts.push(f.merged.to_dtype(text.dtype())?);
            len += f.tokens();
            cursor = p + 1;
        }
        if cursor < ids.len() {
            parts.push(text.narrow(0, cursor, ids.len() - cursor)?);
            len += ids.len() - cursor;
        }
        let grids: Vec<(usize, usize)> = feats.iter().map(|f| (f.rows, f.cols)).collect();
        let pos = mrope_positions(&spans, &grids, len);
        te.forward_rows(Tensor::cat(&parts, 0)?, &pos)
    }

    /// The encoder's last hidden state `(tokens, 4096)` for `prompt`.
    pub fn encode(&self, prompt: &str) -> Result<Tensor> {
        self.text_encoder()?.forward(&self.tokens(prompt)?)
    }

    fn cached(&self, slot: &Mutex<Option<Cached>>, prompt: &str) -> Result<Tensor> {
        let mut slot = slot.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = slot.as_ref().filter(|c| c.prompt == prompt) {
            return Ok(c.hidden.clone());
        }
        let hidden = self.encode(prompt)?;
        *slot = Some(Cached { prompt: prompt.to_string(), hidden: hidden.clone() });
        Ok(hidden)
    }

    pub fn dit(&self) -> Result<Arc<MossDit>> {
        self.transformer()
    }

    pub fn vae(&self) -> &Flux2Decoder {
        &self.vae
    }

    /// Whether MOSS Edit can run (the VAE file has the encoder).
    pub fn can_edit(&self) -> bool {
        self.encoder.is_some()
    }

    /// A reference resized (Lanczos) to `w × h` (multiples of 16) and encoded.
    pub fn reference_latent_sized(&self, r: &RefImage, w: usize, h: usize) -> Result<Tensor> {
        let enc = self.encoder.as_ref().context("the VAE file has no FLUX.2 encoder (Comfy-Org's flux2-vae_small has one)")?;
        let img = image::RgbaImage::from_raw(r.width as u32, r.height as u32, r.rgba.clone()).context("a reference's pixels do not match its size")?;
        let img = if (w, h) == (r.width, r.height) { img } else { image::imageops::resize(&img, w as u32, h as u32, image::imageops::FilterType::Lanczos3) };
        enc.encode(&vae::from_rgba_u8(img.as_raw(), w, h, &self.dev)?)
    }

    /// A reference as MOSS Edit takes it: its long edge at most `edge` (Lanczos, never enlarged,
    /// each side rounded to 16 independently, as the official `resize_image`), encoded.
    pub fn reference_latent(&self, r: &RefImage, edge: usize) -> Result<Tensor> {
        let (w, h) = reference_size(r.width, r.height, edge);
        self.reference_latent_sized(r, w, h)
    }

    /// Make one picture; `progress` hears every step, `cancel` is read between steps.
    pub fn generate(&self, req: &Request, progress: &mut dyn FnMut(f32, &str), cancel: &dyn Fn() -> bool) -> Result<Picture> {
        self.generate_previewed(req, progress, None, cancel)
    }

    /// [`Self::generate`], showing the denoised estimate `(1, 128, h/16, w/16)` (the FLUX.2
    /// VAE's normalised packed space, `crate::preview::LatentSpace::Flux2`) to `preview` after
    /// the steps it asks for; the picture is the same either way.
    pub fn generate_previewed(
        &self,
        req: &Request,
        progress: &mut dyn FnMut(f32, &str),
        preview: Option<&mut dyn StepPreview>,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Picture> {
        self.generate_sampled(req, Sampling::OFFICIAL.with_env(), progress, preview, cancel)
    }

    /// [`Self::generate_previewed`] with `sampling` for a text to image (an edit keeps its
    /// own skipped-block rule).
    pub fn generate_sampled(
        &self,
        req: &Request,
        sampling: Sampling,
        progress: &mut dyn FnMut(f32, &str),
        preview: Option<&mut dyn StepPreview>,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Picture> {
        self.sample(req, sampling, None, progress, preview, cancel)
    }

    /// Generative Fill (no MOSS inpaint model exists): `req.references[0]` is the picture, `mask`
    /// white where MOSS paints. Sampled from noise at the picture's size; after every step the
    /// latent outside the mask is set back to the picture's own, noised to that step's σ (the
    /// last step leaves it exact), so the new area grows into the picture around it (masked
    /// latent blending, as Klein's fill). `show` picks what the model reads besides the
    /// prompt: nothing (text to image, the turbo's run), or the picture with the area flat grey
    /// as an edit reference (with the MOSS Edit LoRA in `req.loras`). `sampling.denoise` under 1
    /// starts from the picture's latent noised that far instead of from noise.
    pub fn fill_sampled(
        &self,
        req: &Request,
        mask: &RefImage,
        show: FillShows,
        sampling: Sampling,
        progress: &mut dyn FnMut(f32, &str),
        preview: Option<&mut dyn StepPreview>,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Picture> {
        let picture = req.references.first().context("a fill needs its picture")?.clone();
        anyhow::ensure!(req.references.len() == 1, "a MOSS fill takes the picture alone ({} pictures given)", req.references.len());
        let (w, h) = if req.width == 0 || req.height == 0 { edit_size(picture.width, picture.height) } else { (req.width, req.height) };
        let mut ask = req.clone();
        ask.width = w;
        ask.height = h;
        ask.references = match show {
            FillShows::Nothing => Vec::new(),
            FillShows::GreyedPicture => vec![crate::klein::pipeline::greyed(&picture, mask)?],
        };
        self.sample(&ask, sampling, Some(Fill { picture: &picture, mask }), progress, preview, cancel)
    }

    fn sample(
        &self,
        req: &Request,
        sampling: Sampling,
        fill: Option<Fill<'_>>,
        progress: &mut dyn FnMut(f32, &str),
        mut preview: Option<&mut dyn StepPreview>,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Picture> {
        crate::device::bind_thread(&self.dev);
        anyhow::ensure!(req.steps > 0, "at least one step");
        let (w, h) = match req.references.first() {
            Some(p) if req.width == 0 || req.height == 0 => edit_size(p.width, p.height),
            _ => (req.width, req.height),
        };
        anyhow::ensure!(w > 0 && h > 0 && w.is_multiple_of(16) && h.is_multiple_of(16), "the picture's sides must be positive multiples of 16 ({w}x{h} asked)");
        anyhow::ensure!(req.references.len() <= 4, "MOSS Edit takes at most 4 pictures ({} given)", req.references.len());

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
        // ai-toolkit's edit layout names each reference ahead of the prompt.
        let ostris = sampling.ostris_refs && !req.references.is_empty();
        let named = |p: &str| -> String {
            if ostris {
                let mut s: String = (1..=req.references.len()).map(|i| format!("Picture {i}: ")).collect();
                s.push_str(p);
                s
            } else {
                p.to_string()
            }
        };
        // ... and lets the encoder see them (`QIR_MOSS_VISION=0`: the names alone).
        let vision = ostris && std::env::var("QIR_MOSS_VISION").map(|v| v.trim() != "0").unwrap_or(true);
        let encode = |slot: &Mutex<Option<Cached>>, p: &str| -> Result<Tensor> {
            if vision {
                self.encode_with_pictures(p, &req.references)
            } else {
                self.cached(slot, &named(p))
            }
        };
        let cond = encode(&self.positive, &req.prompt)?;
        let cfg = req.guidance != 1.0;
        let uncond = if cfg { Some(encode(&self.negative, &req.negative)?) } else { None };
        // The prompts are in hand: the DiT (back on the card in 12 GB mode) and its adapters.
        let dit = self.transformer()?;
        let unmatched = dit.set_loras(&req.loras)?;
        if !unmatched.is_empty() {
            tracing::warn!(modules = unmatched.len(), first = %unmatched[0], "LoRA modules MOSS-Image does not have (a file for another model?)");
        }
        // MOSS Edit: the pictures encoded and run through the blocks once.
        // A restore from the picture's own latent, noised to `Sampling::denoise` (1 = the
        // author's empty latent), so less of the picture is redrawn.
        let mut start: Option<Tensor> = None;
        let refs = if req.references.is_empty() {
            None
        } else {
            progress(0.04, "Reading the picture");
            // 12 GB mode: references at 512 (the edit workflow's), their cache a quarter the size.
            let default_edge = if self.low_vram { LOW_VRAM_REFERENCE_EDGE } else { REFERENCE_EDGE };
            let edge = std::env::var("QIR_MOSS_REF_EDGE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default_edge);
            if ostris {
                // At the output's size (the author's FluxKontextImageScale, about 1 MP), at the
                // prompt's text index. One cache serves both prompts (a guided run with prompts
                // of different lengths puts the negative's references at the prompt's index).
                let latents = req.references.iter().map(|r| self.reference_latent_sized(r, w, h)).collect::<Result<Vec<_>>>()?;
                start = latents.first().cloned();
                Some(dit.prepare_references_at(&latents, cond.dim(0)?)?)
            } else {
                // The picture being edited (the first reference, the output's aspect) is encoded at
                // the output's size: the reference tokens sit at the same grid positions as the
                // output's, so a picture encoded smaller (768×1024 for an 864×1152 edit) came back as
                // a smaller copy in the top-left of the frame, a second head above the first once
                // laid over the original (Enhance round 5 on the 5090, 2026-11-17). Square pictures
                // at 1024 were already the same size and were fine.
                let latents = req
                    .references
                    .iter()
                    .enumerate()
                    .map(|(i, r)| match same_grid_size(i, r.width, r.height, w, h, edge) {
                        true => self.reference_latent_sized(r, w, h),
                        false => self.reference_latent(r, edge),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Some(dit.prepare_references(&latents)?)
            }
        };
        // The edit's negative branch skips no block (the official edit sampler), unless
        // `QIR_MOSS_EDIT_SKIP` names one (the author's ComfyUI workflow skips 22).
        let skip: Vec<usize> = if refs.is_some() {
            std::env::var("QIR_MOSS_EDIT_SKIP").ok().and_then(|v| v.trim().parse().ok()).into_iter().collect()
        } else if sampling.skip_uncond {
            vec![UNCOND_SKIP]
        } else {
            Vec::new()
        };
        self.dev.synchronize()?;
        timings.encode_s = t0.elapsed().as_secs_f32();

        let (lh, lw) = (h / vae::SCALE, w / vae::SCALE);
        let channels = dit.config().channels;
        let noise = crate::comfy::torch_randn(req.seed, channels * lh * lw).map_err(anyhow::Error::msg)?;
        let mut x = Tensor::from_vec(noise, (1, channels, lh, lw), &self.dev)?;
        // A fill: the picture's own latent (kept outside the mask), the mask on the latent grid
        // and the starting noise (the kept latent is noised with it to each step's σ).
        let keep = match &fill {
            Some(f) => {
                let z0 = self.reference_latent_sized(f.picture, w, h)?.to_dtype(x.dtype())?;
                let m = crate::klein::pipeline::mask_tokens(f.mask, lw, lh, &self.dev)?.reshape((1, 1, lh, lw))?.to_dtype(x.dtype())?;
                start = Some(z0.clone());
                Some((z0, m, x.clone()))
            }
            None => None,
        };
        let denoise = Some(sampling.denoise).filter(|d| (ostris || fill.is_some()) && *d < 1.0);
        let sig = match (denoise, &start) {
            // ComfyUI's KSampler `denoise`: the last `steps` of a `steps / denoise` schedule,
            // the latent noised to its first σ.
            (Some(d), Some(init)) => {
                let full = dit::sigmas(((req.steps as f32) / d).round().max(req.steps as f32) as usize, sampling.shift);
                let sig = full[full.len().saturating_sub(req.steps + 1)..].to_vec();
                let s0 = f64::from(sig[0]);
                x = ((init.to_dtype(x.dtype())? * (1.0 - s0))? + (x * s0)?)?;
                sig
            }
            _ => dit::sigmas(req.steps, sampling.shift),
        };
        let n = sig.len() - 1;
        tracing::info!(shift = sampling.shift, sampler = ?sampling.sampler, eta = sampling.eta, skip = ?skip, steps = n, guidance = req.guidance, "MOSS sampling");
        // The guided velocity at σ (the DiT's timestep is σ·1000).
        let velocity = |x: &Tensor, s: f32| -> Result<Tensor> {
            let t = s * 1000.0;
            Ok(match &uncond {
                Some(u) => {
                    let (pos, neg) = dit.forward_pair_refs(x, t, &cond, u, &skip, refs.as_ref())?;
                    (&neg + ((pos - &neg)? * f64::from(req.guidance))?)?
                }
                None => dit.forward_refs(x, t, &cond, &[], refs.as_ref())?,
            })
        };
        // `res_2s`'s fresh noise: seeded from the request, standardised per channel (RES4LYF's z-score).
        let xdt = x.dtype();
        let fresh = |k: u64| -> Result<Tensor> {
            let seed = req.seed ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let v = crate::comfy::torch_randn(seed, channels * lh * lw).map_err(anyhow::Error::msg)?;
            let t = Tensor::from_vec(v, (1, channels, lh, lw), &self.dev)?;
            let c = t.broadcast_sub(&t.mean_keepdim(3)?.mean_keepdim(2)?)?;
            let sd = c.sqr()?.mean_keepdim(3)?.mean_keepdim(2)?.sqrt()?;
            Ok(c.broadcast_div(&sd)?.to_dtype(xdt)?)
        };
        let t0 = Instant::now();
        for i in 0..n {
            check()?;
            let v = velocity(&x, sig[i])?;
            if i == 0 {
                if let Ok(path) = std::env::var("QIR_MOSS_DUMP") {
                    // A check's hook: the first step's velocity, to compare two builds or paths.
                    v.save_safetensors("v", path.trim())?;
                }
            }
            if let Some(p) = preview.as_deref_mut() {
                if p.wants(i + 1, n) {
                    match (&v * f64::from(sig[i])).and_then(|sv| &x - sv) {
                        Ok(x0) => p.show(i + 1, n, &x0),
                        Err(e) => tracing::warn!(error = %e, "preview skipped"),
                    }
                }
            }
            x = match sampling.sampler {
                Sampler::Res2s if sig[i + 1] > 0.0 => {
                    let s = f64::from(sig[i]);
                    let x0 = (&x - (&v * s)?)?;
                    // The step lands at σ_down, fresh noise takes it to σ' (eta 0: σ_down = σ').
                    let step = ancestral(f64::from(sig[i + 1]), sampling.eta);
                    let st = res2s_step(s, step.down);
                    // The midpoint stage the same way: to its own σ_down, then noised to it.
                    let mid = ancestral(st.mid_sigma, sampling.eta);
                    let (cx, cx0) = exp_step(s, mid.down);
                    let mut x2 = ((&x * cx)? + (&x0 * cx0)?)?;
                    if mid.up > 0.0 {
                        x2 = ((x2 * mid.alpha)? + (fresh(2 * i as u64 + 1)? * mid.up)?)?;
                    }
                    let v2 = velocity(&x2, st.mid_sigma as f32)?;
                    let x0_2 = (&x2 - (v2 * st.mid_sigma)?)?;
                    let out = (((&x * st.x)? + (x0 * st.x0_1)?)? + (x0_2 * st.x0_2)?)?;
                    if step.up > 0.0 {
                        ((out * step.alpha)? + (fresh(2 * i as u64 + 2)? * step.up)?)?
                    } else {
                        out
                    }
                }
                _ => (x + (v * f64::from(sig[i + 1] - sig[i]))?)?,
            };
            if let Some((z0, m, eps)) = &keep {
                let s1 = f64::from(sig[i + 1]);
                let known = ((z0 * (1.0 - s1))? + (eps * s1)?)?;
                x = (x.broadcast_mul(m)? + known.broadcast_mul(&(1.0 - m)?)?)?;
            }
            progress(0.05 + 0.85 * (i + 1) as f32 / n as f32, "Sampling");
        }
        self.dev.synchronize()?;
        timings.denoise_s = t0.elapsed().as_secs_f32();
        crate::profile::report("moss sampling");

        check()?;
        progress(0.92, "Decoding");
        if self.low_vram {
            // The sampler's freed blocks back to the driver before the decoder's activations.
            crate::device::trim_pool(&self.dev);
        }
        let t0 = Instant::now();
        let img = self.vae.decode(&x)?;
        self.dev.synchronize()?;
        timings.decode_s = t0.elapsed().as_secs_f32();
        let (pw, ph, mut rgba) = vae::to_rgba_u8(&img)?;
        // A restore's colours from the picture itself (`Sampling::color_fix`): the
        // author's wavelet colour match, the output's detail on the picture's low frequencies.
        if ostris && sampling.color_fix {
            if let Some(r) = req.references.first() {
                let src = image::RgbaImage::from_raw(r.width as u32, r.height as u32, r.rgba.clone()).context("a reference's pixels do not match its size")?;
                let src = image::imageops::resize(&src, pw as u32, ph as u32, image::imageops::FilterType::Lanczos3);
                wavelet_color_match(&mut rgba, src.as_raw(), pw, ph);
            }
        }
        tracing::info!(
            encode_s = timings.encode_s,
            denoise_s = timings.denoise_s,
            decode_s = timings.decode_s,
            width = pw,
            height = ph,
            steps = n,
            "MOSS timings"
        );
        progress(1.0, "Done");
        Ok(Picture { width: pw as u32, height: ph as u32, rgba, timings })
    }
}

/// What a MOSS fill shows the model besides the prompt ([`MossPipeline::fill_sampled`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillShows {
    /// Nothing: text to image, the picture reaching the area only through the kept latent.
    Nothing,
    /// The picture with the area flat grey, as a MOSS Edit reference.
    GreyedPicture,
}

/// A fill's picture and mask, for the sampler.
#[derive(Clone, Copy)]
struct Fill<'a> {
    picture: &'a RefImage,
    mask: &'a RefImage,
}

/// One `res_2s` step from σ to σ' > 0 as coefficients: the stage at the log-SNR midpoint is
/// `mid_x·x + mid_x0·x̂₀` at `mid_sigma`, the step `x·x + x0_1·x̂₀ + x0_2·x̂₀(mid)`. The flow's
/// α = 1 − σ, λ = ln(α/σ), h = λ' − λ; the data-prediction exponential integrator with RES's
/// weights b₂ = φ₂(−h)/c₂, b₁ = φ₁(−h) − b₂ at c₂ = ½. σ = 1 (λ = −∞) is clamped just below.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Res2sStep {
    pub mid_sigma: f64,
    pub mid_x: f64,
    pub mid_x0: f64,
    pub x: f64,
    pub x0_1: f64,
    pub x0_2: f64,
}

/// RES4LYF's "hard" noise on a rectified flow for a step ending at σ': σ_up = eta·σ', the
/// deterministic part ends at σ_down and `alpha·x(σ_down) + σ_up·ε` is back at σ' (signal
/// 1 − σ', noise σ').
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ancestral {
    pub down: f64,
    pub alpha: f64,
    pub up: f64,
}

pub fn ancestral(next: f64, eta: f64) -> Ancestral {
    if next <= 0.0 || eta <= 0.0 {
        return Ancestral { down: next, alpha: 1.0, up: 0.0 };
    }
    let up = eta.min(0.9999) * next;
    let residual = (next * next - up * up).sqrt();
    let alpha = (1.0 - next) + residual;
    Ancestral { down: residual / alpha, alpha, up }
}

/// The first-order data-prediction step from σ to σ' as `(x, x̂₀)` coefficients.
pub fn exp_step(sigma: f64, next: f64) -> (f64, f64) {
    let s = sigma.clamp(1e-6, 1.0 - 1e-4);
    // e^{−h} = (α σ') / (σ α')
    let e = ((1.0 - s) * next) / (s * (1.0 - next));
    (next / sigma, (1.0 - next) * (1.0 - e))
}

pub fn res2s_step(sigma: f64, next: f64) -> Res2sStep {
    const C2: f64 = 0.5;
    let s = sigma.clamp(1e-6, 1.0 - 1e-4);
    let lambda = |s: f64| ((1.0 - s) / s).ln();
    let h = lambda(next) - lambda(s);
    let phi1 = |z: f64| if z.abs() < 1e-8 { 1.0 } else { z.exp_m1() / z };
    let phi2 = |z: f64| if z.abs() < 1e-5 { 0.5 } else { (z.exp_m1() - z) / (z * z) };
    let mid_lambda = lambda(s) + C2 * h;
    let mid_sigma = 1.0 / (1.0 + mid_lambda.exp());
    let b2 = phi2(-h) / C2;
    let b1 = phi1(-h) - b2;
    Res2sStep {
        mid_sigma,
        mid_x: mid_sigma / sigma,
        mid_x0: (1.0 - mid_sigma) * C2 * h * phi1(-C2 * h),
        x: next / sigma,
        x0_1: (1.0 - next) * h * b1,
        x0_2: (1.0 - next) * h * b2,
    }
}

/// Qwen3.5-9B's reply to `user` under `system` (thinking off), with the tokens it wrote and
/// read: the loop of [`MossPipeline::chat`], for any holder of the encoder and its `lm_head`.
pub(crate) fn write_reply(
    tok: &Tokenizer,
    te: &Qwen35Text,
    head: &QLinear,
    system: &str,
    user: &str,
    opts: &crate::pipeline::ChatOptions,
    cancel: &dyn Fn() -> bool,
) -> Result<(String, usize, usize)> {
    use crate::pipeline::{sample, Cancelled, ChatRng};
    let text = format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let prompt: Vec<u32> = tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode the chat: {e}"))?.get_ids().to_vec();
    let stops: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| tok.token_to_id(t)).collect();
    anyhow::ensure!(!stops.is_empty(), "the Qwen3.5 tokenizer has no <|im_end|>");
    let mut out = Vec::new();
    let mut rng = ChatRng::new(opts.seed);
    let (mut last, mut st) = te.prefill(&prompt)?;
    while out.len() < opts.max_new {
        if cancel() {
            return Err(Cancelled.into());
        }
        let logits = head.forward(&last)?.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let next = sample(&logits, opts, &mut rng);
        if stops.contains(&next) {
            break;
        }
        out.push(next);
        last = te.step(next, &mut st)?;
    }
    let reply = tok.decode(&out, true).map_err(|e| anyhow::anyhow!("decode the reply: {e}"))?;
    Ok((reply.trim().to_string(), out.len(), prompt.len()))
}

/// The Qwen3.5-9B files that can write (those of `names` under `root`'s `text_encoders`, in
/// that order) and their tokenizer, for a model that borrows one as a prompt writer.
pub fn find_writers(root: &Path, names: &[&str]) -> Option<(Vec<PathBuf>, PathBuf)> {
    let dir = root.join("text_encoders");
    let texts: Vec<PathBuf> = names.iter().filter_map(|n| find(&dir, &[n])).collect();
    let own = dir.join(TOKENIZER_DIR).join("tokenizer.json");
    let beside = texts.iter().filter_map(|t| t.parent().map(|d| d.join("tokenizer.json")));
    let tokenizer = std::iter::once(own).chain(beside).find(|p| p.is_file())?;
    (!texts.is_empty()).then_some((texts, tokenizer))
}

/// The `lm_head` of a Qwen3.5-9B file, under any of its roots.
pub(crate) fn load_lm_head(w: &Weights) -> Result<QLinear> {
    let name = ["lm_head", "model.lm_head", "model.language_model.lm_head"]
        .into_iter()
        .find(|n| w.contains(&format!("{n}.weight")))
        .context("this Qwen3.5-9B encoder file has no lm_head, so it cannot write text")?;
    w.linear(name)
}

/// 12 GB mode on a card under (by [`crate::device::card_total`]) [`LOW_VRAM_BELOW`], or as `QIR_MOSS_LOW_VRAM=1` / `0` says.
fn low_vram_mode(dev: &Device) -> bool {
    match std::env::var("QIR_MOSS_LOW_VRAM").ok().as_deref().map(str::trim) {
        Some("1") => true,
        Some("0") => false,
        _ => crate::device::card_total(dev).is_some_and(|total| total < LOW_VRAM_BELOW),
    }
}

fn load_text(files: &MossFiles, dev: &Device, dtype: DType) -> Result<Qwen35Text> {
    let t0 = Instant::now();
    let te = Qwen35Text::load(&TextConfig::qwen35_9b(), &Weights::open(&files.text, dev, dtype, true)?).context("the Qwen3.5-9B encoder")?;
    tracing::info!(load_s = t0.elapsed().as_secs_f32(), "MOSS text encoder loaded");
    Ok(te)
}

/// The DiT's dtype: the run's, or f32 (dense, dequantized, the candle chains) under
/// `QIR_MOSS_DIT_F32=1`: a reference for the bf16 paths, not for use.
fn dit_dtype(run: DType) -> DType {
    if std::env::var("QIR_MOSS_DIT_F32").is_ok_and(|v| v.trim() == "1") {
        DType::F32
    } else {
        run
    }
}

fn load_dit(files: &MossFiles, dev: &Device, dtype: DType) -> Result<MossDit> {
    let t0 = Instant::now();
    let dit = MossDit::load(&MossConfig::default(), &Weights::open(&files.dit, dev, dtype, true)?).context("the MOSS-Image DiT")?;
    tracing::info!(load_s = t0.elapsed().as_secs_f32(), "MOSS DiT loaded");
    Ok(dit)
}

/// A reference's size for MOSS Edit: the long edge at most `edge` (rounded down to 16), each
/// side scaled and rounded to the nearest 16 (at least 16), never enlarged.
pub fn reference_size(w: usize, h: usize, edge: usize) -> (usize, usize) {
    let limit = (edge / 16 * 16).max(16);
    let scale = (limit as f64 / w.max(h).max(1) as f64).min(1.0);
    let side = |v: usize| (((v as f64 * scale / 16.0 + 0.5).floor() as usize) * 16).clamp(16, limit);
    (side(w), side(h))
}

/// Whether reference `index` (of `rw`×`rh` pixels) is encoded at the output's `w`×`h`: the first
/// one, when it has the output's aspect (within 2 %) and the output is no bigger than the
/// reference edge allows in area (so 12 GB mode keeps its smaller references).
pub fn same_grid_size(index: usize, rw: usize, rh: usize, w: usize, h: usize, edge: usize) -> bool {
    let (a, b) = ((rw * h) as f64, (rh * w) as f64);
    index == 0 && rw > 0 && rh > 0 && (a - b).abs() <= 0.02 * a.max(b) && w * h <= edge * edge
}

/// An edit's output size when none is asked: the picture's aspect at about [`EDIT_PIXELS`],
/// sides multiples of 16.
pub fn edit_size(w: usize, h: usize) -> (usize, usize) {
    let scale = (EDIT_PIXELS as f64 / (w.max(1) * h.max(1)) as f64).sqrt();
    let side = |v: usize| (((v as f64 * scale / 16.0).round() as usize) * 16).max(16);
    (side(w), side(h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_and_edit_sizes() {
        assert_eq!(reference_size(4000, 3000, 1024), (1024, 768));
        assert_eq!(reference_size(500, 300, 1024), (496, 304));
        assert_eq!(edit_size(4000, 3000), (1184, 880));
        assert_eq!(edit_size(1024, 1024), (1024, 1024));
        // The edited picture is encoded at the output's size when it has its aspect.
        assert!(same_grid_size(0, 864, 1152, 864, 1152, REFERENCE_EDGE));
        assert!(same_grid_size(0, 1728, 2304, 864, 1152, REFERENCE_EDGE));
        assert!(!same_grid_size(1, 864, 1152, 864, 1152, REFERENCE_EDGE));
        assert!(!same_grid_size(0, 1024, 1024, 864, 1152, REFERENCE_EDGE));
        assert!(!same_grid_size(0, 864, 1152, 864, 1152, LOW_VRAM_REFERENCE_EDGE));
    }

    #[test]
    fn res2s_is_consistent() {
        for (a, b) in [(1.0, 0.95), (0.6, 0.55), (0.1, 0.05), (0.002, 0.001)] {
            let s = res2s_step(a, b);
            // A constant x̂₀ (the exact solution's) lands on (1 − σ')·x̂₀ + σ'·ε.
            let (x0, eps) = (0.7, -1.3);
            let x = (1.0 - a) * x0 + a * eps;
            let out = s.x * x + (s.x0_1 + s.x0_2) * x0;
            let want = (1.0 - b) * x0 + b * eps;
            assert!((out - want).abs() < 2e-3, "{a}->{b}: {out} vs {want}");
            assert!(s.mid_sigma < a && s.mid_sigma > b, "{a}->{b}: midpoint {}", s.mid_sigma);
            let mid = s.mid_x * x + s.mid_x0 * x0;
            let want = (1.0 - s.mid_sigma) * x0 + s.mid_sigma * eps;
            assert!((mid - want).abs() < 2e-3, "{a}->{b}: mid {mid} vs {want}");
            let (cx, cx0) = exp_step(a, b);
            assert!((cx * x + cx0 * x0 - ((1.0 - b) * x0 + b * eps)).abs() < 2e-3);
        }
        for (next, eta) in [(0.9, 0.5), (0.3, 0.5), (0.05, 1.0), (0.4, 0.0)] {
            let a = ancestral(next, eta);
            // Signal and noise land on the flow at σ'.
            assert!((a.alpha * (1.0 - a.down) - (1.0 - next)).abs() < 1e-9);
            assert!(((a.alpha * a.down).powi(2) + a.up * a.up - next * next).abs() < 1e-9);
            assert!(a.down <= next + 1e-12);
        }
    }
}

/// The encoder file's `model.visual.*` tensors in f32 on `dev`: plain ones converted, ComfyUI's
/// INT8 ConvRot linears (`int8_tensorwise` rows with a per-row `weight_scale`, `convrot` in
/// groups of `convrot_groupsize` input features, 64 in Comfy-Org's file) dequantized and
/// rotated back, as comfy-kitchen's `dequantize_int8_convrot_weight` does.
fn visual_tensors(path: &Path, dev: &Device) -> Result<std::collections::HashMap<String, Tensor>> {
    use safetensors::{Dtype, SafeTensors};
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&map)?;
    let as_f32 = |name: &str| -> Result<Vec<f32>> {
        let t = st.tensor(name)?;
        let b = t.data();
        Ok(match t.dtype() {
            Dtype::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            Dtype::BF16 => b.chunks_exact(2).map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)).collect(),
            Dtype::F16 => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            other => anyhow::bail!("{name}: {other:?} is not a float tensor"),
        })
    };
    let mut out = std::collections::HashMap::new();
    for (name, view) in st.tensors() {
        if !name.starts_with("model.visual.") || name.ends_with(".weight_scale") || name.ends_with(".comfy_quant") || name.ends_with(".input_scale") {
            continue;
        }
        let shape = view.shape().to_vec();
        let values = if view.dtype() == Dtype::I8 {
            let module = name.strip_suffix(".weight").with_context(|| format!("{name}: an INT8 tensor that is not a weight"))?;
            anyhow::ensure!(shape.len() == 2, "{name}: an INT8 weight of shape {shape:?}");
            let (n, k) = (shape[0], shape[1]);
            let scale = as_f32(&format!("{module}.weight_scale"))?;
            anyhow::ensure!(scale.len() == n || scale.len() == 1, "{module}.weight_scale: {} values for {n} rows", scale.len());
            let group = match st.tensor(&format!("{module}.comfy_quant")) {
                Ok(q) => {
                    let d: serde_json::Value = serde_json::from_slice(q.data()).with_context(|| format!("{module}.comfy_quant"))?;
                    if d.get("convrot").and_then(serde_json::Value::as_bool).unwrap_or(false) {
                        d.get("convrot_groupsize").and_then(serde_json::Value::as_u64).unwrap_or(256) as usize
                    } else {
                        0
                    }
                }
                Err(_) => 0,
            };
            let mut w: Vec<f32> = view.data().iter().enumerate().map(|(i, &b)| f32::from(b as i8) * scale[if scale.len() == 1 { 0 } else { i / k }]).collect();
            if group > 0 {
                anyhow::ensure!(k.is_multiple_of(group), "{module}: {k} input features in ConvRot groups of {group}");
                let h = regular_hadamard(group)?;
                let mut tmp = vec![0f32; group];
                for row in w.chunks_mut(k) {
                    for g in row.chunks_mut(group) {
                        // W = W_rot · Hᵀ per group (H is symmetric).
                        for (j, t) in tmp.iter_mut().enumerate() {
                            *t = g.iter().zip(&h[j * group..(j + 1) * group]).map(|(a, b)| a * b).sum();
                        }
                        g.copy_from_slice(&tmp);
                    }
                }
            }
            w
        } else {
            as_f32(&name)?
        };
        out.insert(name.clone(), Tensor::from_vec(values, shape, &Device::Cpu)?.to_device(dev)?);
    }
    anyhow::ensure!(!out.is_empty(), "{}: no model.visual.* tensors (an encoder file without the vision tower)", path.display());
    Ok(out)
}

/// comfy-kitchen's `_build_hadamard`: the normalized regular Hadamard of `size` (a power of 4),
/// Kronecker powers of `[[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]`, row-major.
fn regular_hadamard(size: usize) -> Result<Vec<f32>> {
    anyhow::ensure!(size >= 4 && size.is_power_of_two() && size.trailing_zeros() % 2 == 0, "a regular Hadamard of {size} (a power of 4)");
    const H4: [f32; 16] = [1., 1., 1., -1., 1., 1., -1., 1., 1., -1., 1., 1., -1., 1., 1., 1.];
    let mut h = H4.to_vec();
    let mut n = 4;
    while n < size {
        let m = n * 4;
        let mut next = vec![0f32; m * m];
        for (i, a) in h.iter().enumerate() {
            let (ar, ac) = (i / n, i % n);
            for (j, b) in H4.iter().enumerate() {
                let (br, bc) = (j / 4, j % 4);
                next[(ar * 4 + br) * m + ac * 4 + bc] = a * b;
            }
        }
        h = next;
        n = m;
    }
    let norm = 1.0 / (size as f32).sqrt();
    Ok(h.into_iter().map(|v| v * norm).collect())
}

/// A picture as the ViT takes it: area-downscaled (never enlarged) to fit 384² pixels (the
/// edit encoder's `_fit_area`), then bilinearly to the 32-pixel grid (`process_qwen2vl_images`),
/// patchified. Returns `(patches, grid_h, grid_w)`.
fn vit_patches(r: &RefImage, cfg: &VisionConfig, dev: &Device) -> Result<(Tensor, usize, usize)> {
    const MAX_PIXELS: f64 = 384.0 * 384.0;
    let (w, h) = (r.width, r.height);
    anyhow::ensure!(w > 0 && h > 0 && r.rgba.len() == w * h * 4, "a picture of {w}x{h} with {} bytes", r.rgba.len());
    let scale = (MAX_PIXELS / (w * h) as f64).sqrt().min(1.0);
    let (aw, ah) = (((w as f64 * scale).round() as usize).max(1), ((h as f64 * scale).round() as usize).max(1));
    let rgb = area_resize(&r.rgba, w, h, aw, ah);
    let rgb = Tensor::from_vec(rgb, (3, ah, aw), &Device::Cpu)?;
    let (vw, vh) = vit::vit_size(aw, ah, 3136, 12_845_056);
    let rgb = if (vw, vh) == (aw, ah) { rgb } else { vit::resize_bilinear(&rgb, vh, vw)? };
    let (p, gh, gw) = vit::patchify(&rgb, cfg)?;
    Ok((p.to_device(dev)?, gh, gw))
}

/// torch's `interpolate(mode="area")` (adaptive average pooling) of RGBA bytes to `(3, dh, dw)`
/// f32 in `[0, 1]`, alpha dropped.
fn area_resize(rgba: &[u8], w: usize, h: usize, dw: usize, dh: usize) -> Vec<f32> {
    let span = |i: usize, n_src: usize, n_dst: usize| -> (usize, usize) { (i * n_src / n_dst, ((i + 1) * n_src).div_ceil(n_dst)) };
    let mut out = vec![0f32; 3 * dh * dw];
    for y in 0..dh {
        let (y0, y1) = span(y, h, dh);
        for x in 0..dw {
            let (x0, x1) = span(x, w, dw);
            let mut acc = [0f32; 3];
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let o = (sy * w + sx) * 4;
                    for (c, a) in acc.iter_mut().enumerate() {
                        *a += f32::from(rgba[o + c]);
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as f32 * 255.0;
            for (c, a) in acc.iter().enumerate() {
                out[(c * dh + y) * dw + x] = a / n;
            }
        }
    }
    out
}

/// StableSR's wavelet colour fix (`wavelet_reconstruction`, the author's ComfyUI colour match):
/// both pictures split into high and low frequencies by five à-trous blurs (radius 1, 2, 4, 8,
/// 16, the 3×3 binomial kernel, edges replicated); `out` keeps its high frequencies on the
/// low frequencies of `style` (same size, RGBA bytes; alpha left as it is).
pub fn wavelet_color_match(out: &mut [u8], style: &[u8], w: usize, h: usize) {
    let plane = |src: &[u8], c: usize| -> Vec<f32> { (0..w * h).map(|i| f32::from(src[i * 4 + c]) / 255.0).collect() };
    let blur = |x: &[f32], r: usize| -> Vec<f32> {
        const K: [f32; 3] = [0.25, 0.5, 0.25];
        let at = |v: isize, n: usize| v.clamp(0, n as isize - 1) as usize;
        let mut tmp = vec![0f32; w * h];
        for y in 0..h {
            for xx in 0..w {
                tmp[y * w + xx] = (0..3).map(|k| K[k] * x[y * w + at(xx as isize + (k as isize - 1) * r as isize, w)]).sum();
            }
        }
        let mut outp = vec![0f32; w * h];
        for y in 0..h {
            for xx in 0..w {
                outp[y * w + xx] = (0..3).map(|k| K[k] * tmp[at(y as isize + (k as isize - 1) * r as isize, h) * w + xx]).sum();
            }
        }
        outp
    };
    let low = |x: Vec<f32>| -> Vec<f32> { (0..5).fold(x, |cur, i| blur(&cur, 1 << i)) };
    for c in 0..3 {
        let content = plane(out, c);
        let (cl, sl) = (low(content.clone()), low(plane(style, c)));
        for i in 0..w * h {
            let v = content[i] - cl[i] + sl[i];
            out[i * 4 + c] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
}

#[cfg(test)]
mod vision_tests {
    use super::{area_resize, regular_hadamard, wavelet_color_match};

    #[test]
    fn the_regular_hadamard_is_orthogonal_and_symmetric() {
        let n = 64;
        let h = regular_hadamard(n).unwrap();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(h[i * n + j], h[j * n + i]);
                let dot: f32 = (0..n).map(|k| h[i * n + k] * h[j * n + k]).sum();
                assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn wavelet_match_takes_flat_colour_from_the_style() {
        let (w, h) = (8, 8);
        let mut out: Vec<u8> = (0..w * h).flat_map(|_| [200u8, 100, 50, 255]).collect();
        let style: Vec<u8> = (0..w * h).flat_map(|_| [10u8, 20, 30, 255]).collect();
        wavelet_color_match(&mut out, &style, w, h);
        assert!(out.chunks(4).all(|p| p == [10, 20, 30, 255]), "{:?}", &out[..4]);
    }

    #[test]
    fn area_resize_averages_blocks_and_keeps_a_same_size_picture() {
        // 2x2 RGBA -> 1x1: the mean of the four pixels per channel.
        let px = [0u8, 0, 0, 255, 255, 0, 0, 255, 0, 255, 0, 255, 255, 255, 255, 255];
        let one = area_resize(&px, 2, 2, 1, 1);
        assert_eq!(one, vec![0.5, 0.5, 0.25]);
        let same = area_resize(&px, 2, 2, 2, 2);
        assert_eq!(&same[0..4], &[0.0, 1.0, 0.0, 1.0]);
    }
}
