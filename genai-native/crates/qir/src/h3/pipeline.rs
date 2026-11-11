//! P-H3-Image text to image and edits, with PrunaAI's ComfyUI conventions: the raw prompt
//! (pictures as `<Picture n>: ` blocks) through Qwen3-VL-4B and the text connector, torch's CPU
//! noise for the seed over the packed still (one video latent frame and two audio latents),
//! Euler on the 8 fixed sigmas at shift 1 with no CFG, and the tiled VAE decode. The encoder
//! and the VAE take turns beside the resident DiT (int8: 21 GB): the encoder loads for a new
//! prompt and goes, the VAE loads when a picture is read or decoded and stays until the next
//! new prompt, so the card never holds both (peak about 28 GB at 4 MP).

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use super::dit::{H3Dit, AUDIO_CHANNELS, LATENT_CHANNELS};
use super::layout::Layout;
use super::text::{self, H3Text};
use super::vae::{self, H3Vae};
use super::PH3_SIGMAS;
use crate::model::vision::{self, VisionConfig, VisionFeatures, VisionTower};
use crate::moss::weights::Weights;
use crate::pipeline::{Cancelled, Picture, RefImage, Request, Timings};
use crate::Result;

/// The audio latents a still spans.
pub const AUDIO_T: usize = 2;
/// An edit's output when the request names no size: picture 1's aspect within this area
/// (PrunaAI: edits are best at 4 MP).
pub const EDIT_AREA: usize = 2048 * 2048;
/// At most one picture to edit and seven references.
pub const MAX_PICTURES: usize = 8;

/// P-H3-Image's files in a ComfyUI-layout `models` folder (PrunaAI/P-H3-Image-Comfy's names).
#[derive(Clone, Debug)]
pub struct H3Files {
    pub dit: PathBuf,
    pub text: PathBuf,
    pub vae: PathBuf,
    /// ComfyUI's `qwen25_tokenizer` folder.
    pub tokenizer: PathBuf,
}

const DIT: &[&str] = &["p-h3-image_int8_convrot.safetensors", "p-h3-image_bf16.safetensors"];
const TEXT: &[&str] = &["p-h3-image_text_encoder_int8_convrot.safetensors", "p-h3-image_text_encoder_bf16.safetensors"];
const VAE: &[&str] = &["p-h3-image_vae_bf16.safetensors", "p-h3-image_vae_int8_convrot.safetensors"];

/// `names` in `dir` or one subfolder below it, first found wins; `env` names a file instead.
fn find(dir: &Path, names: &[&str], env: &str) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(env).map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(p);
    }
    let mut dirs = vec![dir.to_path_buf()];
    if let Ok(rd) = std::fs::read_dir(dir) {
        let mut subs: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect();
        subs.sort();
        dirs.extend(subs);
    }
    names.iter().find_map(|n| dirs.iter().map(|d| d.join(n)).find(|p| p.is_file()))
}

/// P-H3-Image under `root`, when every file is there. `QIR_H3_DIT`, `QIR_H3_TEXT` and
/// `QIR_H3_VAE` pick other files (the quality comparisons).
pub fn find_files(root: &Path) -> Option<H3Files> {
    Some(H3Files {
        dit: find(&root.join("diffusion_models"), DIT, "QIR_H3_DIT")?,
        text: find(&root.join("text_encoders"), TEXT, "QIR_H3_TEXT")?,
        vae: find(&root.join("vae"), VAE, "QIR_H3_VAE")?,
        tokenizer: crate::comfy_files::tokenizer_dir(root)?,
    })
}

/// A prompt's conditioning, kept for the next request with the same prompt and pictures.
struct Cached {
    key: (String, u64),
    states: Tensor,
    tags: Vec<usize>,
}

pub struct H3Pipeline {
    dev: Device,
    dtype: DType,
    tok: Tokenizer,
    files: H3Files,
    dit: H3Dit,
    vae: Mutex<Option<H3Vae>>,
    cond: Mutex<Option<Cached>>,
}

impl H3Pipeline {
    pub fn load(files: &H3Files, progress: &mut dyn FnMut(f32, &str)) -> Result<Self> {
        let dev = crate::device::best_device()?;
        let dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
        let tok = crate::comfy_files::qwen_tokenizer(&files.tokenizer)?;
        progress(0.0, "Loading the transformer");
        let t0 = Instant::now();
        let dit = H3Dit::load(&Weights::open(&files.dit, &dev, dtype, true)?).context("the P-H3-Image DiT")?;
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "H3 DiT loaded");
        progress(1.0, "Loaded");
        Ok(Self { dev, dtype, tok, files: files.clone(), dit, vae: Mutex::new(None), cond: Mutex::new(None) })
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// The run dtype (LoRA files are read in it).
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// `run` with the VAE on the card (loaded on first use).
    fn with_vae<T>(&self, run: impl FnOnce(&H3Vae) -> Result<T>) -> Result<T> {
        let mut slot = self.vae.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            let t0 = Instant::now();
            *slot = Some(H3Vae::load(&Weights::open(&self.files.vae, &self.dev, self.dtype, true)?).context("the P-H3-Image VAE")?);
            tracing::info!(load_s = t0.elapsed().as_secs_f32(), "H3 VAE loaded");
        }
        run(slot.as_ref().ok_or_else(|| anyhow::anyhow!("no VAE"))?)
    }

    /// The conditioning for `prompt` with `pictures` (already in the encoder's view), the
    /// encoder and the vision tower loaded for it and dropped after.
    fn condition(&self, prompt: &str, pictures: &[RefImage]) -> Result<(Tensor, Vec<usize>)> {
        let key = (prompt.to_string(), pictures_key(pictures));
        let mut slot = self.cond.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = slot.as_ref().filter(|c| c.key == key) {
            return Ok((c.states.clone(), c.tags.clone()));
        }
        // the VAE makes room for the encoder
        *self.vae.lock().unwrap_or_else(|e| e.into_inner()) = None;
        crate::device::trim_pool(&self.dev);
        let t0 = Instant::now();
        let features = if pictures.is_empty() {
            Vec::new()
        } else {
            let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[&self.files.text], DType::F32, &self.dev)? };
            let tower = VisionTower::load(&VisionConfig::qwen3vl_4b(), vb.pp("model").pp("visual")).context("the Qwen3-VL-4B vision tower")?;
            pictures
                .iter()
                .map(|p| -> Result<VisionFeatures> {
                    let (w, h) = text::size_within_area(p.width, p.height, text::VISION_PIXELS);
                    let rgb = resized_rgb(p, w, h)?;
                    let rgb = Tensor::from_vec(rgb, (h, w, 3), &self.dev)?.permute((2, 0, 1))?.to_dtype(DType::F32)?.affine(1.0 / 255.0, 0.0)?;
                    let (vw, vh) = vision::vit_size(w, h, 3136, 12_845_056);
                    let rgb = if (vw, vh) == (w, h) { rgb } else { vision::resize_bilinear(&rgb, vh, vw)? };
                    let (patches, gh, gw) = vision::patchify(&rgb, tower.config())?;
                    tower.forward(&patches, gh, gw)
                })
                .collect::<Result<Vec<_>>>()?
        };
        let te = H3Text::load(&Weights::open(&self.files.text, &self.dev, self.dtype, true)?).context("the P-H3-Image text encoder")?;
        let ids = text::token_ids(&self.tok, prompt, pictures.len())?;
        let c = te.encode(&ids, &features.iter().collect::<Vec<_>>())?;
        drop(te);
        drop(features);
        self.dev.synchronize()?;
        crate::device::trim_pool(&self.dev);
        tracing::info!(encode_s = t0.elapsed().as_secs_f32(), rows = c.tags.len(), "H3 prompt encoded");
        *slot = Some(Cached { key, states: c.states.clone(), tags: c.tags.clone() });
        Ok((c.states, c.tags))
    }

    /// Make one picture; `progress` hears every step, `cancel` is read between steps.
    /// `req.steps` 8 runs the fixed sigmas; any other count the `simple` schedule at shift
    /// [`crate::h3::LORA_SHIFT`] (the 2-step LoRA's).
    pub fn generate(&self, req: &Request, progress: &mut dyn FnMut(f32, &str), cancel: &dyn Fn() -> bool) -> Result<Picture> {
        crate::device::bind_thread(&self.dev);
        anyhow::ensure!(req.references.len() <= MAX_PICTURES, "P-H3-Image takes at most {MAX_PICTURES} pictures ({} given)", req.references.len());
        let check = || -> Result<()> {
            if cancel() {
                Err(Cancelled.into())
            } else {
                Ok(())
            }
        };
        let mut timings = Timings::default();
        let area = if req.width > 0 && req.height > 0 { req.width * req.height } else if req.references.is_empty() { text::IMAGE_SIZE * text::IMAGE_SIZE } else { EDIT_AREA };
        let (w, h) = match req.references.first() {
            Some(p) if req.width == 0 || req.height == 0 => text::size_within_area(p.width, p.height, area),
            _ if req.width == 0 || req.height == 0 => (text::IMAGE_SIZE, text::IMAGE_SIZE),
            _ => (req.width, req.height),
        };
        anyhow::ensure!(w.is_multiple_of(32) && h.is_multiple_of(32), "the picture's sides must be multiples of 32 ({w}x{h} asked)");

        check()?;
        progress(0.02, "Encoding the prompt");
        let t0 = Instant::now();
        let prompt = text::edit_prompt(&req.prompt, !req.references.is_empty());
        let (states, tags) = self.condition(&prompt, &req.references)?;
        // the references through the VAE, each within the output's area
        let mut refs = Vec::new();
        if !req.references.is_empty() {
            progress(0.04, "Reading the pictures");
            for p in &req.references {
                let (rw, rh) = text::size_within_area(p.width, p.height, area);
                let rgb = resized_rgb(p, rw, rh)?;
                let x = Tensor::from_vec(rgb, (rh, rw, 3), &self.dev)?.permute((2, 0, 1))?.to_dtype(DType::F32)?.affine(2.0 / 255.0, -1.0)?;
                refs.push(self.with_vae(|v| v.encode(&x))?);
            }
        }
        let ref_sizes: Vec<(usize, usize)> = refs.iter().map(|r| -> Result<(usize, usize)> { Ok((r.dim(2)?, r.dim(3)?)) }).collect::<Result<_>>()?;
        let (lh, lw) = (h / vae::SCALE, w / vae::SCALE);
        let audio_t = self.dit.config().audio.then_some(AUDIO_T);
        let layout = Layout::new(tags, &ref_sizes, audio_t, lh, lw);
        // the request's adapters, before the text is refined (none takes the previous ones off)
        self.dit.set_loras(&req.loras)?;
        let prepared = self.dit.prepare(&states, layout)?;
        self.dev.synchronize()?;
        timings.encode_s = t0.elapsed().as_secs_f32();

        // ComfyUI's prepare_noise: the video latent's noise, then the audio's, one generator
        let nv = LATENT_CHANNELS * lh * lw;
        let na = AUDIO_CHANNELS * 2 * AUDIO_T;
        let sizes: Vec<usize> = if audio_t.is_some() { vec![nv, na] } else { vec![nv] };
        let mut noise = crate::comfy::torch_randn_seq(req.seed, &sizes).map_err(anyhow::Error::msg)?.into_iter();
        let mut x = Tensor::from_vec(noise.next().unwrap_or_default(), (LATENT_CHANNELS, 1, lh, lw), &self.dev)?;
        let mut a = match audio_t {
            Some(t) => Some(Tensor::from_vec(noise.next().unwrap_or_default(), (AUDIO_CHANNELS, 2, t), &self.dev)?),
            None => None,
        };
        let sig = sigmas(req.steps)?;
        if !req.loras.is_empty() {
            tracing::info!(loras = ?req.loras, steps = sig.len() - 1, "H3 LoRAs on");
        }
        let n = sig.len() - 1;
        let t0 = Instant::now();
        for i in 0..n {
            check()?;
            // the model sees the latent in its dtype and sigma through timestep / 1000
            let s = (sig[i] * 1000.0) / 1000.0;
            let xin = x.to_dtype(self.dtype)?;
            let ain = match &a {
                Some(a) => Some(a.to_dtype(self.dtype)?),
                None => None,
            };
            let (v, va) = self.dit.forward(&prepared, &xin, ain.as_ref(), &refs, req.seed, s.max(1e-6), &sig)?;
            let dt = f64::from(sig[i + 1] - sig[i]);
            x = (x + (v * dt)?)?;
            if let (Some(av), Some(va)) = (a.as_mut(), va) {
                *av = (&*av + (va * dt)?)?;
            }
            progress(0.05 + 0.85 * (i + 1) as f32 / n as f32, "Sampling");
        }
        self.dev.synchronize()?;
        timings.denoise_s = t0.elapsed().as_secs_f32();

        check()?;
        progress(0.92, "Decoding");
        crate::device::trim_pool(&self.dev);
        let t0 = Instant::now();
        let img = self.with_vae(|v| v.decode(&x))?;
        self.dev.synchronize()?;
        timings.decode_s = t0.elapsed().as_secs_f32();
        let rgba = to_rgba_u8(&img)?;
        tracing::info!(encode_s = timings.encode_s, denoise_s = timings.denoise_s, decode_s = timings.decode_s, width = w, height = h, steps = n, "H3 timings");
        progress(1.0, "Done");
        Ok(Picture { width: w as u32, height: h as u32, rgba, timings })
    }
}

/// The sigmas for `steps`: PrunaAI's eight, or the `simple` scheduler at the 2-step LoRA's
/// shift for any other count.
pub fn sigmas(steps: usize) -> Result<Vec<f32>> {
    if steps == 0 || steps == PH3_SIGMAS.len() - 1 {
        return Ok(PH3_SIGMAS.to_vec());
    }
    let shift = crate::comfy::Shift::Snr { shift: super::LORA_SHIFT, multiplier: 1000.0, timesteps: 1000 };
    Ok(crate::comfy::FlowSchedule::simple(steps, shift).map_err(anyhow::Error::msg)?.sigmas)
}

/// A cache key for the pictures (sizes and a sample of their bytes).
fn pictures_key(pictures: &[RefImage]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for p in pictures {
        (p.width, p.height).hash(&mut h);
        p.rgba.hash(&mut h);
    }
    h.finish()
}

/// `p` as RGB bytes at `w x h` (Lanczos, untouched when already that size; ComfyUI's
/// `_resize_to`).
fn resized_rgb(p: &RefImage, w: usize, h: usize) -> Result<Vec<u8>> {
    let img = image::RgbaImage::from_raw(p.width as u32, p.height as u32, p.rgba.clone()).context("a picture's pixels do not match its size")?;
    let rgb = image::DynamicImage::ImageRgba8(img).to_rgb8();
    let rgb = if (w, h) == (p.width, p.height) { rgb } else { image::imageops::resize(&rgb, w as u32, h as u32, image::imageops::FilterType::Lanczos3) };
    Ok(rgb.into_raw())
}

/// `(3, H, W)` in `[0, 1]` -> RGBA bytes.
fn to_rgba_u8(img: &Tensor) -> Result<Vec<u8>> {
    let (_, h, w) = img.dims3()?;
    let px = img.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.permute((1, 2, 0))?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(h * w * 4);
    for c in px.chunks_exact(3) {
        for v in c {
            out.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        out.push(255);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_steps_are_prunas_and_others_use_the_lora_shift() {
        assert_eq!(sigmas(8).unwrap(), PH3_SIGMAS.to_vec());
        let s = sigmas(2).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], 1.0);
        assert_eq!(s[2], 0.0);
        // shift 12 at t = 0.5: 12 * 0.5 / (1 + 11 * 0.5) = 0.923
        assert!((s[1] - 0.923_077).abs() < 1e-4, "{}", s[1]);
    }
}
