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
}

impl Sampling {
    /// The official pipeline: Euler on the shift-5 schedule, block 22 skipped in the negative.
    pub const OFFICIAL: Self = Self { shift: SHIFT, sampler: Sampler::Euler, skip_uncond: true, eta: 0.0 };
    /// The app's MOSS-Image text to image, after RaymondLuxuryYacht's MOSS realism workflow
    /// (Civitai 3035895, 2026-11-04): RES4LYF's `res_2s` (ancestral, eta 0.5), 20 steps, plain CFG
    /// 3 (no skipped block) with a real negative ([`REALISM_NEGATIVE`]). The workflow's schedule
    /// is linear σ (shift 1); in the app (2026-11-11, three prompts) that came out dark and
    /// muddy with broken signs, so the official shift 5 stays. About 30 % faster than 50 Euler steps.
    pub const REALISM: Self = Self { shift: SHIFT, sampler: Sampler::Res2s, skip_uncond: false, eta: 0.5 };

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
        use crate::pipeline::{sample, Cancelled, ChatRng};
        crate::device::bind_thread(&self.dev);
        let t0 = Instant::now();
        let text = format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        let prompt: Vec<u32> = self.tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode the chat: {e}"))?.get_ids().to_vec();
        let stops: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| self.tok.token_to_id(t)).collect();
        anyhow::ensure!(!stops.is_empty(), "the Qwen3.5 tokenizer has no <|im_end|>");
        let te = self.text_encoder()?;
        let head = self.lm_head()?;
        let mut out = Vec::new();
        let mut rng = ChatRng::new(opts.seed);
        let (mut last, mut st) = te.prefill(&prompt)?;
        let tp = Instant::now();
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
        drop((head, st, last));
        if self.low_vram {
            // 12 GB mode: the head (1-2 GB) goes again, so the DiT has its room back.
            *self.head.lock().unwrap_or_else(|e| e.into_inner()) = None;
            crate::device::trim_pool(&self.dev);
        }
        let reply = self.tok.decode(&out, true).map_err(|e| anyhow::anyhow!("decode the reply: {e}"))?;
        tracing::info!(
            tokens = out.len(),
            prompt_tokens = prompt.len(),
            decode_s = tp.elapsed().as_secs_f32(),
            chat_s = t0.elapsed().as_secs_f32(),
            "moss chat"
        );
        Ok(reply.trim().to_string())
    }

    /// The encoder file's `lm_head` (248320 × 4096), on the card from the first chat on.
    fn lm_head(&self) -> Result<Arc<QLinear>> {
        let mut slot = self.head.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = slot.as_ref() {
            return Ok(Arc::clone(h));
        }
        let w = Weights::open(&self.files.text, &self.dev, self.dtype, true)?;
        let name = ["lm_head", "model.lm_head", "model.language_model.lm_head"]
            .into_iter()
            .find(|n| w.contains(&format!("{n}.weight")))
            .context("this Qwen3.5-9B encoder file has no lm_head, so it cannot write text")?;
        let t0 = Instant::now();
        let h = Arc::new(w.linear(name)?);
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), name, "MOSS chat head loaded");
        *slot = Some(Arc::clone(&h));
        Ok(h)
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

    /// A reference as MOSS Edit takes it: its long edge at most `edge` (Lanczos, never enlarged,
    /// each side rounded to 16 independently, as the official `resize_image`), encoded.
    pub fn reference_latent(&self, r: &RefImage, edge: usize) -> Result<Tensor> {
        let enc = self.encoder.as_ref().context("the VAE file has no FLUX.2 encoder (Comfy-Org's flux2-vae_small has one)")?;
        let (w, h) = reference_size(r.width, r.height, edge);
        let img = image::RgbaImage::from_raw(r.width as u32, r.height as u32, r.rgba.clone()).context("a reference's pixels do not match its size")?;
        let img = if (w, h) == (r.width, r.height) { img } else { image::imageops::resize(&img, w as u32, h as u32, image::imageops::FilterType::Lanczos3) };
        enc.encode(&vae::from_rgba_u8(img.as_raw(), w, h, &self.dev)?)
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
        let cond = self.cached(&self.positive, &req.prompt)?;
        let cfg = req.guidance != 1.0;
        let uncond = if cfg { Some(self.cached(&self.negative, &req.negative)?) } else { None };
        // The prompts are in hand: the DiT (back on the card in 12 GB mode) and its adapters.
        let dit = self.transformer()?;
        let unmatched = dit.set_loras(&req.loras)?;
        if !unmatched.is_empty() {
            tracing::warn!(modules = unmatched.len(), first = %unmatched[0], "LoRA modules MOSS-Image does not have (a file for another model?)");
        }
        // MOSS Edit: the pictures encoded and run through the blocks once.
        let refs = if req.references.is_empty() {
            None
        } else {
            progress(0.04, "Reading the picture");
            // 12 GB mode: references at 512 (the edit workflow's), their cache a quarter the size.
            let default_edge = if self.low_vram { LOW_VRAM_REFERENCE_EDGE } else { REFERENCE_EDGE };
            let edge = std::env::var("QIR_MOSS_REF_EDGE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default_edge);
            let latents = req.references.iter().map(|r| self.reference_latent(r, edge)).collect::<Result<Vec<_>>>()?;
            Some(dit.prepare_references(&latents)?)
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
        let sig = dit::sigmas(req.steps, sampling.shift);
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
        let (pw, ph, rgba) = vae::to_rgba_u8(&img)?;
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

/// 12 GB mode on a card under [`LOW_VRAM_BELOW`], or as `QIR_MOSS_LOW_VRAM=1` / `0` says.
fn low_vram_mode(dev: &Device) -> bool {
    match std::env::var("QIR_MOSS_LOW_VRAM").ok().as_deref().map(str::trim) {
        Some("1") => true,
        Some("0") => false,
        _ => crate::device::free_vram(dev).is_some_and(|(_, total)| total < LOW_VRAM_BELOW),
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
