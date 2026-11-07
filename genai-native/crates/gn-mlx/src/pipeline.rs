//! MOSS-Image on MLX, text to image and MOSS Edit, with `qwen_image_rs::moss::pipeline`'s
//! conventions (torch's CPU noise for the seed, Euler steps on the shift-5 schedule snapped to
//! the training grid, CFG whose negative branch is the empty prompt, block 22 skipped unless
//! editing; an edit's pictures resized and encoded by the FLUX.2 encoder, run once through the
//! DiT at t = 0 and attended at every step) within a 16 GB Mac's memory: the encoder (6-bit) is
//! streamed a layer at a time beside the resident DiT (8-bit, its LoRAs unmerged) and VAE. Converted weights are cached as MLX safetensors
//! (`cache_dir`), so only the first load of a model reads the ComfyUI files.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use mlx_rs::{Array, Dtype};
use qwen_image_rs::moss::dit::{MossConfig, sigmas};
use qwen_image_rs::moss::pipeline::{LOW_VRAM_REFERENCE_EDGE, MossFiles, REFERENCE_EDGE, SHIFT, UNCOND_SKIP, Variant, find_files, reference_size};
use qwen_image_rs::moss::text::{TextConfig, template};
use qwen_image_rs::pipeline::Cancelled;
use tokenizers::Tokenizer;

use crate::convert::{self, Lora, Precision};
use crate::dit::MossDit;
use crate::nn::{Params, limit_cache, shape};
use crate::text::Qwen35Text;
use crate::vae::{self, Flux2Decoder, Flux2Encoder};

/// The run dtype of every model.
const DTYPE: Dtype = Dtype::Float16;
/// The text encoder's: 6-bit (5.6 GB; cos 0.9988 to f16 over 8 layers, 8-bit 0.99993 at 7.4 GB,
/// 4-bit 0.978). 8-bit pushed a 16 GB Mac with a desktop and the editor open into swap.
const TEXT_PRECISION: Precision = Precision::Q6;
/// Pixels per latent pixel.
const SCALE: usize = 16;
/// An edit's output when the request names no size: the picture's aspect at about this many
/// pixels (768²; qir's 1 MP took 16 passes of 35 s on an M1 Air, past the editor's deadline).
const EDIT_PIXELS: usize = 768 * 768;

/// An edit's output size: the picture's aspect at about `pixels`, never larger than the picture
/// (its references are never enlarged, and an output above them lost the scene), sides multiples
/// of 16.
fn edit_size(w: usize, h: usize, pixels: usize) -> (usize, usize) {
    let scale = (pixels as f64 / (w.max(1) * h.max(1)) as f64).sqrt().min(1.0);
    let side = |v: usize| (((v as f64 * scale / 16.0).round() as usize) * 16).max(16);
    (side(w), side(h))
}

/// MOSS Edit takes at most this many pictures.
const MAX_REFERENCES: usize = 4;

/// A picture for MOSS Edit: straight RGBA8.
#[derive(Clone, Debug)]
pub struct RefPicture {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

/// One picture to make.
#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: String,
    pub negative: String,
    /// The output's size; 0 for an edit takes the picture's aspect at about 1 MP.
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// CFG scale; 1 runs the prompt alone.
    pub guidance: f32,
    pub seed: u64,
    /// MOSS Edit's pictures, the one being edited first; none for text to image.
    pub references: Vec<RefPicture>,
    /// Fill's area on the first reference, its size: white is painted, black kept
    /// (FLUX 3 Image Klein only).
    pub mask: Option<RefPicture>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Timings {
    /// Loading (or converting) and running the encoder.
    pub encode_s: f32,
    /// Loading the DiT, when it was not resident, and an edit's reference pass.
    pub load_s: f32,
    pub steps_s: f32,
    pub decode_s: f32,
}

pub struct Picture {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
    pub timings: Timings,
}

/// A LoRA on the DiT (applied unmerged).
#[derive(Clone, Debug, PartialEq)]
pub struct LoraSpec {
    pub path: PathBuf,
    pub strength: f32,
}

/// The DiT held: its variant and precision.
type DitKey = (Variant, Precision);

pub struct MossMlx {
    root: PathBuf,
    /// The encoder, VAE and tokenizer (shared by every variant).
    shared: MossFiles,
    cache_dir: PathBuf,
    tok: Tokenizer,
    vae: Flux2Decoder,
    encoder: Option<Flux2Encoder>,
    dit: Option<(DitKey, MossDit)>,
    /// The LoRAs on the held DiT.
    dit_loras: Vec<LoraSpec>,
    dit_precision: Precision,
    /// The last prompts' encodings.
    encoded: Vec<(String, Array)>,
}

impl MossMlx {
    /// MOSS-Image's files under the ComfyUI `models` folder `root` (the encoder, VAE and
    /// tokenizer of whichever variant is there); converted weights cached under `cache_dir`.
    pub fn new(root: &Path, cache_dir: &Path) -> Result<Self> {
        limit_cache()?;
        let shared = [Variant::Kromoss, Variant::Sft, Variant::Base]
            .into_iter()
            .find_map(|v| find_files(root, v))
            .with_context(|| format!("MOSS-Image's files under {}", root.display()))?;
        let tok = Tokenizer::from_file(&shared.tokenizer).map_err(|e| anyhow::anyhow!("load {}: {e}", shared.tokenizer.display()))?;
        let p = vae::convert(&shared.vae, DTYPE)?;
        let vae = Flux2Decoder::load(&p, DTYPE)?;
        let encoder = Flux2Encoder::load(&p, DTYPE).ok();
        Ok(Self {
            root: root.to_path_buf(),
            shared,
            cache_dir: cache_dir.to_path_buf(),
            tok,
            vae,
            encoder,
            dit: None,
            dit_loras: Vec::new(),
            dit_precision: Precision::Q8,
            encoded: Vec::new(),
        })
    }

    /// Whether MOSS Edit can run (the VAE file has the encoder).
    pub fn can_edit(&self) -> bool {
        self.encoder.is_some()
    }

    /// Hold the DiT in `precision`: 8-bit by default (5.6 GB; 35 dB to the f16 DiT's picture at
    /// 512², 4-bit's drifts to 15 dB: another picture of the prompt), f16 needs 10 GB.
    pub fn set_dit_precision(&mut self, precision: Precision) {
        self.dit_precision = precision;
    }

    /// Drop every model but the VAE (the next request loads what it needs).
    pub fn unload(&mut self) -> Result<()> {
        self.dit = None;
        self.encoded.clear();
        mlx_rs::memory::clear_cache()?;
        Ok(())
    }

    /// `name`'s converted weights from the cache, or `convert` then cached.
    fn cached(&self, name: &str, convert: impl FnOnce() -> Result<Params>) -> Result<Params> {
        let path = self.cache_dir.join(name);
        // An unreadable cache is converted again.
        if path.is_file()
            && let Ok(p) = convert::load(&path)
        {
            return Ok(p);
        }
        let p = convert()?;
        convert::save(&p, &path)?;
        Ok(p)
    }

    fn stem(path: &Path) -> String {
        let s = path.file_stem().map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned());
        s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' }).collect()
    }

    /// The encodings of `prompts` (the encoder loaded, run and dropped if any is new).
    fn encode(&mut self, prompts: &[&str], cancel: &dyn Fn() -> bool) -> Result<Vec<Array>> {
        let missing: Vec<&str> = prompts.iter().copied().filter(|p| !self.encoded.iter().any(|(q, _)| q == p)).collect();
        if !missing.is_empty() {
            // Streamed a layer at a time (its peak one layer and the embedding table, about 1.2
            // GB), so the DiT stays: holding the whole encoder (5.6 GB at 6-bit) beside it put a
            // 16 GB Mac into swap, and dropping the DiT for it cost a reload per prompt.
            let cfg = TextConfig::qwen35_9b();
            let text = self.shared.text.clone();
            let dir = self.cache_dir.join(format!("{}.{}", Self::stem(&text), TEXT_PRECISION.tag()));
            if !dir.join("base.safetensors").is_file() {
                crate::text::convert_streamed(&text, &cfg, DTYPE, TEXT_PRECISION, &dir)?;
            }
            let te = Qwen35Text::streamed(&cfg, crate::text::embedding(&text, &cfg)?, &dir, DTYPE)?;
            for p in missing {
                if cancel() {
                    return Err(Cancelled.into());
                }
                let ids = self.tok.encode(template(p), false).map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?.get_ids().to_vec();
                let h = te.forward(&ids)?;
                h.eval()?;
                self.encoded.push((p.to_string(), h));
            }
            drop(te);
            mlx_rs::memory::clear_cache()?;
            // The last two (a prompt and its negative).
            let extra = self.encoded.len().saturating_sub(2);
            self.encoded.drain(..extra);
        }
        prompts.iter().map(|p| self.encoded.iter().find(|(q, _)| q == p).map(|(_, h)| h.clone()).context("an encoding went missing")).collect()
    }

    /// `variant`'s DiT with `loras` on it (unmerged), loaded unless it is the one held; one
    /// converted cache per variant and precision, whatever the LoRAs.
    fn dit(&mut self, variant: Variant, loras: &[LoraSpec]) -> Result<&MossDit> {
        let key: DitKey = (variant, self.dit_precision);
        if self.dit.as_ref().is_none_or(|(k, _)| *k != key) {
            self.dit = None;
            self.dit_loras.clear();
            mlx_rs::memory::clear_cache()?;
            let files = find_files(&self.root, variant).with_context(|| format!("MOSS-Image's {variant:?} files under {}", self.root.display()))?;
            let cfg = MossConfig::default();
            let precision = self.dit_precision;
            let name = format!("{}.{}.safetensors", Self::stem(&files.dit), precision.tag());
            let params = self.cached(&name, || crate::dit::convert(&files.dit, &cfg, DTYPE, precision, &[]))?;
            self.dit = Some((key, MossDit::load(&cfg, &params, DTYPE)?));
        }
        let Some((_, dit)) = self.dit.as_mut() else { anyhow::bail!("the DiT did not load") };
        if self.dit_loras != loras {
            let opened = loras.iter().map(|l| Lora::open(&l.path, l.strength)).collect::<Result<Vec<_>>>()?;
            let matched = dit.set_loras(&opened)?;
            anyhow::ensure!(loras.is_empty() || matched > 0, "none of the LoRAs' modules is MOSS-Image's (a file for another model?)");
            self.dit_loras = loras.to_vec();
        }
        Ok(dit)
    }

    /// A picture as MOSS Edit takes it: its long edge at most `edge` (Lanczos, never enlarged,
    /// each side rounded to 16), encoded.
    fn reference_latent(&self, r: &RefPicture, edge: usize) -> Result<Array> {
        let enc = self.encoder.as_ref().context("the VAE file has no FLUX.2 encoder (Comfy-Org's flux2-vae_small has one)")?;
        let (w, h) = reference_size(r.width, r.height, edge);
        let img = image::RgbaImage::from_raw(u32::try_from(r.width)?, u32::try_from(r.height)?, r.rgba.clone())
            .context("a picture's pixels do not match its size")?;
        let img = if (w, h) == (r.width, r.height) {
            img
        } else {
            image::imageops::resize(&img, u32::try_from(w)?, u32::try_from(h)?, image::imageops::FilterType::Lanczos3)
        };
        let z = enc.encode(&vae::from_rgba_u8(img.as_raw(), w, h)?)?;
        z.eval()?;
        Ok(z)
    }

    /// Make one picture with `variant`'s DiT and `loras` merged; `progress(step, steps)` before
    /// the first step (0) and after each; `cancel` is read between steps.
    pub fn generate(
        &mut self,
        variant: Variant,
        loras: &[LoraSpec],
        req: &Request,
        progress: &mut dyn FnMut(usize, usize),
        cancel: &dyn Fn() -> bool,
    ) -> Result<Picture> {
        anyhow::ensure!(req.steps > 0 && req.steps <= 200, "{} steps (1..200)", req.steps);
        anyhow::ensure!(req.guidance.is_finite(), "guidance {}", req.guidance);
        anyhow::ensure!(req.references.len() <= MAX_REFERENCES, "MOSS Edit takes at most {MAX_REFERENCES} pictures ({} given)", req.references.len());
        let (w, h) = match req.references.first() {
            Some(p) if req.width == 0 || req.height == 0 => edit_size(p.width, p.height, EDIT_PIXELS),
            _ => (req.width, req.height),
        };
        anyhow::ensure!(
            w > 0 && h > 0 && w.is_multiple_of(SCALE) && h.is_multiple_of(SCALE),
            "the picture's sides must be positive multiples of {SCALE} ({w}x{h} asked)"
        );
        anyhow::ensure!(w * h <= 2048 * 2048, "{w}x{h} is too large");
        let check = || -> Result<()> { if cancel() { Err(Cancelled.into()) } else { Ok(()) } };
        let mut timings = Timings::default();

        let t0 = Instant::now();
        let cfg = req.guidance != 1.0;
        let prompts: Vec<&str> = if cfg { vec![&req.prompt, &req.negative] } else { vec![&req.prompt] };
        let enc = self.encode(&prompts, cancel)?;
        timings.encode_s = t0.elapsed().as_secs_f32();
        check()?;

        let t0 = Instant::now();
        // The references at the output's scale (long edge 512..1024): at 512 under a 768 output the
        // edit kept only the subject (the rest of the picture was lost).
        let edge = w.max(h).clamp(LOW_VRAM_REFERENCE_EDGE, REFERENCE_EDGE);
        let latents = req.references.iter().map(|r| self.reference_latent(r, edge)).collect::<Result<Vec<_>>>()?;
        let dit = self.dit(variant, loras)?;
        let refs = if latents.is_empty() { None } else { Some(dit.prepare_references(&latents)?) };
        // The edit's negative branch skips no block (the official edit sampler).
        let skip: &[usize] = if refs.is_some() { &[] } else { &[UNCOND_SKIP] };
        timings.load_s = t0.elapsed().as_secs_f32();

        let (lh, lw) = (h / SCALE, w / SCALE);
        let channels = dit.config().channels;
        let noise = qwen_image_rs::comfy::torch_randn(req.seed, channels * lh * lw).map_err(anyhow::Error::msg)?;
        let mut x = Array::from_slice(&noise, &shape(&[1, channels, lh, lw])?);
        let sig = sigmas(req.steps, SHIFT);
        let n = sig.len().saturating_sub(1);
        progress(0, n);
        let t0 = Instant::now();
        for i in 0..n {
            check()?;
            let (s0, s1) = (sig[i], sig[i + 1]);
            let pos = dit.forward(&x, s0 * 1000.0, &enc[0], &[], refs.as_ref())?;
            let v = match enc.get(1) {
                Some(neg) => {
                    let neg = dit.forward(&x, s0 * 1000.0, neg, skip, refs.as_ref())?;
                    neg.add(pos.subtract(&neg)?.multiply(Array::from_f32(req.guidance))?)?
                }
                None => pos,
            };
            x = x.add(v.multiply(Array::from_f32(s1 - s0))?)?;
            x.eval()?;
            progress(i + 1, n);
        }
        timings.steps_s = t0.elapsed().as_secs_f32();
        drop(refs);

        check()?;
        let t0 = Instant::now();
        let img = self.vae.decode(&x)?;
        let (pw, ph, rgba) = vae::to_rgba_u8(&img)?;
        timings.decode_s = t0.elapsed().as_secs_f32();
        mlx_rs::memory::clear_cache()?;
        Ok(Picture { width: pw, height: ph, rgba, timings })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn edit_sizes_keep_the_aspect() {
        assert_eq!(super::edit_size(1024, 1024, super::EDIT_PIXELS), (768, 768));
        assert_eq!(super::edit_size(512, 512, super::EDIT_PIXELS), (512, 512));
        assert_eq!(super::edit_size(500, 300, super::EDIT_PIXELS), (496, 304));
        assert_eq!(super::edit_size(4000, 3000, super::EDIT_PIXELS), (880, 672));
        assert_eq!(super::edit_size(0, 0, super::EDIT_PIXELS), (16, 16));
        // The references at the output's long edge never exceed the picture's.
        let (w, h) = super::edit_size(512, 512, super::EDIT_PIXELS);
        assert_eq!(super::reference_size(512, 512, w.max(h).clamp(super::LOW_VRAM_REFERENCE_EDGE, super::REFERENCE_EDGE)), (w, h));
    }
}
