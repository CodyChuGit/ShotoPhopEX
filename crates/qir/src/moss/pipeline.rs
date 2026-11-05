//! MOSS-Image text to image, resident: the Qwen3.5-9B encoder, the DiT and the FLUX.2 decoder
//! loaded once, sampled with the official pipeline's conventions (torch's CPU noise for the
//! seed, 50 Euler steps on the shift-5 schedule snapped to the training grid, CFG 3 whose
//! negative branch is the empty prompt with block 22 skipped).

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use super::dit::{self, MossConfig, MossDit};
use super::text::{self, Qwen35Text, TextConfig};
use super::vae::{self, Flux2Decoder, Flux2Encoder};
use super::weights::Weights;
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
/// An edit's output when the request names no size: the picture's aspect at about this many
/// pixels (1 MP, the edit workflow's), sides multiples of 16.
pub const EDIT_PIXELS: usize = 1024 * 1024;

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
    te: Qwen35Text,
    dit: MossDit,
    vae: Flux2Decoder,
    /// MOSS Edit's reference encoder (the small VAE file holds it), when the file has one.
    encoder: Option<Flux2Encoder>,
    positive: Mutex<Option<Cached>>,
    negative: Mutex<Option<Cached>>,
}

impl MossPipeline {
    pub fn load(files: &MossFiles, progress: &mut dyn FnMut(f32, &str)) -> Result<Self> {
        let dev = crate::device::best_device()?;
        let dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
        let tok = Tokenizer::from_file(&files.tokenizer).map_err(|e| anyhow::anyhow!("load {}: {e}", files.tokenizer.display()))?;

        progress(0.0, "Loading the text encoder");
        let t0 = Instant::now();
        let te = Qwen35Text::load(&TextConfig::qwen35_9b(), &Weights::open(&files.text, &dev, dtype, true)?).context("the Qwen3.5-9B encoder")?;
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "MOSS text encoder loaded");

        progress(0.5, "Loading the transformer");
        let t0 = Instant::now();
        let dit = MossDit::load(&MossConfig::default(), &Weights::open(&files.dit, &dev, dtype, true)?).context("the MOSS-Image DiT")?;
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "MOSS DiT loaded");

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
        Ok(Self { dev, dtype, tok, te, dit, vae, encoder, positive: Mutex::new(None), negative: Mutex::new(None) })
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

    /// The encoder's last hidden state `(tokens, 4096)` for `prompt`.
    pub fn encode(&self, prompt: &str) -> Result<Tensor> {
        self.te.forward(&self.tokens(prompt)?)
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

    pub fn dit(&self) -> &MossDit {
        &self.dit
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
        let unmatched = self.dit.set_loras(&req.loras)?;
        if !unmatched.is_empty() {
            tracing::warn!(modules = unmatched.len(), first = %unmatched[0], "LoRA modules MOSS-Image does not have (a file for another model?)");
        }
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
        // MOSS Edit: the pictures encoded and run through the blocks once.
        let refs = if req.references.is_empty() {
            None
        } else {
            progress(0.04, "Reading the picture");
            let edge = std::env::var("QIR_MOSS_REF_EDGE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(REFERENCE_EDGE);
            let latents = req.references.iter().map(|r| self.reference_latent(r, edge)).collect::<Result<Vec<_>>>()?;
            Some(self.dit.prepare_references(&latents)?)
        };
        // The edit's negative branch skips no block (the official edit sampler), unless
        // `QIR_MOSS_EDIT_SKIP` names one (the author's ComfyUI workflow skips 22).
        let skip: Vec<usize> = if refs.is_some() {
            std::env::var("QIR_MOSS_EDIT_SKIP").ok().and_then(|v| v.trim().parse().ok()).into_iter().collect()
        } else {
            vec![UNCOND_SKIP]
        };
        self.dev.synchronize()?;
        timings.encode_s = t0.elapsed().as_secs_f32();

        let (lh, lw) = (h / vae::SCALE, w / vae::SCALE);
        let channels = self.dit.config().channels;
        let noise = crate::comfy::torch_randn(req.seed, channels * lh * lw).map_err(anyhow::Error::msg)?;
        let mut x = Tensor::from_vec(noise, (1, channels, lh, lw), &self.dev)?;
        let sig = dit::sigmas(req.steps, SHIFT);
        let n = sig.len() - 1;
        let t0 = Instant::now();
        for i in 0..n {
            check()?;
            let t = sig[i] * 1000.0;
            let v = match &uncond {
                Some(u) => {
                    let (pos, neg) = self.dit.forward_pair_refs(&x, t, &cond, u, &skip, refs.as_ref())?;
                    (&neg + ((pos - &neg)? * f64::from(req.guidance))?)?
                }
                None => self.dit.forward_refs(&x, t, &cond, &[], refs.as_ref())?,
            };
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
            x = (x + (v * f64::from(sig[i + 1] - sig[i]))?)?;
            progress(0.05 + 0.85 * (i + 1) as f32 / n as f32, "Sampling");
        }
        self.dev.synchronize()?;
        timings.denoise_s = t0.elapsed().as_secs_f32();
        crate::profile::report("moss sampling");

        check()?;
        progress(0.92, "Decoding");
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
}
