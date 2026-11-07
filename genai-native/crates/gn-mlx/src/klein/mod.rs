//! FLUX 3 Image Klein on MLX: text to image and editing with up to 10 reference pictures,
//! from mflux's pre-quantized files (`mflux-community/flux-3-image-klein-mflux-q4` or `-q8`,
//! loaded as they are, no conversion). The VAE is the FLUX.2 VAE MOSS-Image already uses (Klein
//! samples in its latent space; mflux: "identical weights").
//!
//! The reference's conventions: torch's CPU noise for the seed (`(h·w, 128)`, rounded to bf16),
//! the flow map's 16 Euler steps on `α·t / (1 + (α − 1)·t)` with no CFG. Base takes 50 steps
//! with CFG against the empty prompt. References are encoded by the FLUX.2 encoder and told
//! apart by their RoPE layer `-(1 + i)`.
//!
//! The prompt upsampler (a 9B model) is not run. Prompts go to the encoder as typed, as mflux
//! does.

pub mod dit;
pub mod text;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use mlx_rs::{Array, Dtype};
use qwen_image_rs::pipeline::Cancelled;

use crate::convert::Lora;
use crate::nn::{Params, limit_cache};
use crate::pipeline::{LoraSpec, Picture, RefPicture, Request, Timings};
use crate::vae::{self, Flux2Decoder, Flux2Encoder};
use dit::{CHANNELS, KleinDit, Tokens, from_tokens, grid_ids, text_ids, to_tokens};
use text::KleinText;

const DTYPE: Dtype = Dtype::Float16;
/// Pixels per latent pixel.
const SCALE: usize = 16;
/// The schedule's shift. mflux uses 6.93 (its pictures on the Air were good); BFL's
/// `Pipeline.generate` defaults to 16.
pub const ALPHA: f32 = 6.93;
pub const MAX_REFERENCES: usize = 10;
/// An edit's output when the request names no size: the first picture's aspect at about this
/// many pixels, never larger than the picture.
const EDIT_PIXELS: usize = 768 * 768;
/// Each further reference at most this many pixels (the reference's multi-reference cap).
const REFERENCE_PIXELS: usize = 1024 * 1024;

/// mflux's folder names for each part.
const TRANSFORMER: &str = "transformer";
const TEXT_ENCODER: &str = "text_encoder";

/// Every `*.safetensors` in `dir`, merged. Quantized words stay as they are; every other
/// floating tensor (scales, biases, norms; mflux stores them in bf16) goes to `dtype`.
pub fn load_dir(dir: &Path, dtype: Dtype) -> Result<Params> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    anyhow::ensure!(!files.is_empty(), "no .safetensors in {}", dir.display());
    let mut p = Params::new();
    for f in files {
        let part = Array::load_safetensors(&f).with_context(|| format!("read {}", f.display()))?;
        for (name, a) in part {
            let a = match a.dtype() {
                Dtype::Bfloat16 | Dtype::Float32 | Dtype::Float16 if a.dtype() != dtype => a.as_dtype(dtype)?,
                _ => a,
            };
            a.eval()?;
            p.insert(name, a);
        }
    }
    Ok(p)
}

/// The names a LoRA file may give mflux's `module`: mflux's and diffusers' own
/// (`attn.to_out` or `attn.to_out.0`), and the reference's (BFL's and ComfyUI's, which
/// ai-toolkit writes: `content_mode_blocks.image.0.q_proj`, `single_blocks.3.mlp_in`, …).
pub fn lora_names(module: &str) -> Vec<String> {
    let mut out = vec![module.to_string()];
    if let Some(stem) = module.strip_suffix(".attn.to_out") {
        out.push(format!("{stem}.attn.to_out.0"));
    }
    let bfl_part = [
        ("attn.to_q", "q_proj"),
        ("attn.to_k", "k_proj"),
        ("attn.to_v", "v_proj"),
        ("attn.to_out", "attn_out"),
        ("ff.linear_in", "mlp_in"),
        ("ff.linear_out", "mlp_out"),
    ];
    let bfl_block = [
        ("mode_blocks.image_cond.", "content_mode_blocks.image_cond."),
        ("mode_blocks.image.", "content_mode_blocks.image."),
        ("txt_mode_blocks.", "txt_mode_blocks."),
        ("single_transformer_blocks.", "single_blocks."),
    ];
    for (ours, theirs) in bfl_block {
        let Some(rest) = module.strip_prefix(ours) else { continue };
        let Some((index, part)) = rest.split_once('.') else { continue };
        if let Some((_, b)) = bfl_part.iter().find(|(a, _)| *a == part) {
            out.push(format!("{theirs}{index}.{b}"));
        }
        break;
    }
    out
}

/// The schedule: `steps + 1` times from 1 to 0.
pub fn timesteps(steps: usize, alpha: f32) -> Vec<f32> {
    (0..=steps)
        .map(|i| {
            let t = 1.0 - i as f32 / steps as f32;
            alpha * t / (1.0 + (alpha - 1.0) * t)
        })
        .collect()
}

/// `(w, h)` at about `pixels` keeping the aspect, never larger than the picture, sides multiples
/// of 16.
fn fit(w: usize, h: usize, pixels: usize) -> (usize, usize) {
    let scale = (pixels as f64 / (w.max(1) * h.max(1)) as f64).sqrt().min(1.0);
    let side = |v: usize| (((v as f64 * scale / 16.0).round() as usize) * 16).max(16);
    (side(w), side(h))
}

/// `r` with the mask's white made flat grey (#808080), blended by the mask's value; the mask at
/// another size is resized to the picture's.
fn greyed(r: &RefPicture, m: &RefPicture) -> Result<RefPicture> {
    let mask =
        image::RgbaImage::from_raw(u32::try_from(m.width)?, u32::try_from(m.height)?, m.rgba.clone()).context("the mask's pixels do not match its size")?;
    let mask = if (m.width, m.height) == (r.width, r.height) {
        mask
    } else {
        image::imageops::resize(&mask, u32::try_from(r.width)?, u32::try_from(r.height)?, image::imageops::FilterType::Triangle)
    };
    let mut rgba = r.rgba.clone();
    for (px, mp) in rgba.as_chunks_mut::<4>().0.iter_mut().zip(mask.pixels()) {
        let a = f32::from(mp.0[0]) / 255.0;
        for c in px.iter_mut().take(3) {
            *c = (f32::from(*c) * (1.0 - a) + 128.0 * a).round() as u8;
        }
    }
    Ok(RefPicture { width: r.width, height: r.height, rgba })
}

/// A fill mask on the latent grid `(lh·lw, 1)` f32 in 0..1 (1 is painted): averaged over each
/// latent pixel's 16×16 and doubled, so an edge pixel the mask touches is painted too.
fn mask_tokens(m: &RefPicture, lw: usize, lh: usize) -> Result<Array> {
    let img =
        image::RgbaImage::from_raw(u32::try_from(m.width)?, u32::try_from(m.height)?, m.rgba.clone()).context("the mask's pixels do not match its size")?;
    let small = image::imageops::resize(&img, u32::try_from(lw)?, u32::try_from(lh)?, image::imageops::FilterType::Triangle);
    let v: Vec<f32> = small.pixels().map(|p| (f32::from(p.0[0]) / 255.0 * 2.0).min(1.0)).collect();
    Ok(Array::from_slice(&v, &crate::nn::shape(&[lh * lw, 1])?))
}

/// One pass's tokens.
fn tokens<'a>(x: &'a Array, image_ids: &'a [[f32; 4]], cond: Option<(&'a Array, &'a [[f32; 4]])>, text: &'a (Array, Vec<[f32; 4]>)) -> Tokens<'a> {
    Tokens { image: x, image_ids, cond, text: &text.0, text_ids: &text.1 }
}

/// Which checkpoint and which quantization are held.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Held {
    dir: PathBuf,
}

pub struct KleinMlx {
    vae: Flux2Decoder,
    encoder: Flux2Encoder,
    dit: Option<(Held, KleinDit)>,
    dit_loras: Vec<LoraSpec>,
    /// The last prompts' encodings.
    encoded: Vec<(PathBuf, String, Array)>,
}

impl KleinMlx {
    /// `vae_file`: the FLUX.2 VAE (with its encoder, e.g. Comfy-Org's `flux2-vae_small`).
    pub fn new(vae_file: &Path) -> Result<Self> {
        limit_cache()?;
        let p = vae::convert(vae_file, DTYPE)?;
        Ok(Self {
            vae: Flux2Decoder::load(&p, DTYPE)?,
            encoder: Flux2Encoder::load(&p, DTYPE).context("the VAE file has no FLUX.2 encoder")?,
            dit: None,
            dit_loras: Vec::new(),
            encoded: Vec::new(),
        })
    }

    /// Drop the DiT and the encodings (the next request loads what it needs).
    pub fn unload(&mut self) -> Result<()> {
        self.dit = None;
        self.encoded.clear();
        mlx_rs::memory::clear_cache()?;
        Ok(())
    }

    /// The encodings of `prompts` with the text encoder of the checkpoint in `model` (loaded,
    /// run and dropped if any is new, so it never sits beside the DiT's activations).
    fn encode(&mut self, model: &Path, prompts: &[&str], cancel: &dyn Fn() -> bool) -> Result<Vec<Array>> {
        let missing: Vec<&str> = prompts.iter().copied().filter(|p| !self.encoded.iter().any(|(m, q, _)| m == model && q == p)).collect();
        if !missing.is_empty() {
            let te = KleinText::load(&model.join(TEXT_ENCODER), DTYPE)?;
            for p in missing {
                if cancel() {
                    return Err(Cancelled.into());
                }
                let h = te.encode(p)?;
                self.encoded.push((model.to_path_buf(), p.to_string(), h));
            }
            drop(te);
            mlx_rs::memory::clear_cache()?;
            let extra = self.encoded.len().saturating_sub(2);
            self.encoded.drain(..extra);
        }
        prompts
            .iter()
            .map(|p| self.encoded.iter().find(|(m, q, _)| m == model && q == p).map(|(_, _, h)| h.clone()).context("an encoding went missing"))
            .collect()
    }

    /// The DiT in `model` with `loras` on it (unmerged), loaded unless it is the one held.
    fn dit(&mut self, model: &Path, loras: &[LoraSpec]) -> Result<&KleinDit> {
        let key = Held { dir: model.to_path_buf() };
        if self.dit.as_ref().is_none_or(|(k, _)| *k != key) {
            self.dit = None;
            self.dit_loras.clear();
            mlx_rs::memory::clear_cache()?;
            let p = load_dir(&model.join(TRANSFORMER), DTYPE)?;
            self.dit = Some((key, KleinDit::load(&p, DTYPE)?));
        }
        let Some((_, dit)) = self.dit.as_mut() else { anyhow::bail!("the DiT did not load") };
        if self.dit_loras != loras {
            let opened = loras.iter().map(|l| Lora::open(&l.path, l.strength)).collect::<Result<Vec<_>>>()?;
            let matched = dit.set_loras(&opened)?;
            anyhow::ensure!(loras.is_empty() || matched > 0, "none of the LoRAs' modules is FLUX 3 Image Klein's (a file for another model?)");
            self.dit_loras = loras.to_vec();
        }
        Ok(dit)
    }

    /// A picture resized to `(w, h)` (Lanczos) and encoded: tokens `(h/16 · w/16, 128)`.
    fn reference(&self, r: &RefPicture, w: usize, h: usize) -> Result<Array> {
        let img = image::RgbaImage::from_raw(u32::try_from(r.width)?, u32::try_from(r.height)?, r.rgba.clone())
            .context("a picture's pixels do not match its size")?;
        let img = if (w, h) == (r.width, r.height) {
            img
        } else {
            image::imageops::resize(&img, u32::try_from(w)?, u32::try_from(h)?, image::imageops::FilterType::Lanczos3)
        };
        let z = self.encoder.encode(&vae::from_rgba_u8(img.as_raw(), w, h)?)?;
        let t = to_tokens(&z)?.as_dtype(DTYPE)?;
        t.eval()?;
        Ok(t)
    }

    /// Make one picture with the checkpoint in `model` (mflux's folder: `transformer/`,
    /// `text_encoder/`) and `loras`. `req.steps` 0 takes the checkpoint's own (16, or 50 for
    /// Base); `req.guidance` only matters for Base. `progress(step, steps)` runs before the first
    /// step (0) and after each one; `cancel` is checked between steps.
    pub fn generate(
        &mut self,
        model: &Path,
        loras: &[LoraSpec],
        req: &Request,
        progress: &mut dyn FnMut(usize, usize),
        cancel: &dyn Fn() -> bool,
    ) -> Result<Picture> {
        anyhow::ensure!(req.steps <= 200, "{} steps (1..200)", req.steps);
        anyhow::ensure!(req.references.len() <= MAX_REFERENCES, "Klein takes at most {MAX_REFERENCES} pictures ({} given)", req.references.len());
        let (w, h) = match req.references.first() {
            Some(p) if req.width == 0 || req.height == 0 => fit(p.width, p.height, EDIT_PIXELS),
            _ => (req.width, req.height),
        };
        anyhow::ensure!(
            w > 0 && h > 0 && w.is_multiple_of(SCALE) && h.is_multiple_of(SCALE),
            "the picture's sides must be positive multiples of {SCALE} ({w}x{h} asked)"
        );
        anyhow::ensure!(w * h <= 2048 * 2048, "{w}x{h} is too large");
        let check = || -> Result<()> { if cancel() { Err(Cancelled.into()) } else { Ok(()) } };
        let mut timings = Timings::default();

        // The DiT first: whether it is Base decides whether the empty prompt is needed.
        let t0 = Instant::now();
        let flow_map = self.dit(model, loras)?.is_flow_map();
        timings.load_s = t0.elapsed().as_secs_f32();
        let cfg = !flow_map && req.guidance != 1.0;
        let t0 = Instant::now();
        let prompts: Vec<&str> = if cfg { vec![&req.prompt, ""] } else { vec![&req.prompt] };
        let enc = self.encode(model, &prompts, cancel)?;
        timings.encode_s = t0.elapsed().as_secs_f32();
        check()?;

        // The references: the first at the output's size (it sets the canvas), the rest under 1 MP.
        let t0 = Instant::now();
        let mut cond_tokens = Vec::new();
        let mut cond_ids: Vec<[f32; 4]> = Vec::new();
        // Fill: the picture's own latent (the area kept), and the picture the model sees with the
        // area flat grey (shown the area as it was, Klein paints it back).
        let mut original = None;
        for (i, r) in req.references.iter().enumerate() {
            let (rw, rh) = if i == 0 { (w, h) } else { fit(r.width, r.height, REFERENCE_PIXELS) };
            match (&req.mask, i) {
                (Some(m), 0) => {
                    original = Some(self.reference(r, rw, rh)?);
                    cond_tokens.push(self.reference(&greyed(r, m)?, rw, rh)?);
                }
                _ => cond_tokens.push(self.reference(r, rw, rh)?),
            }
            cond_ids.extend(grid_ids(rh / SCALE, rw / SCALE, -(1.0 + i as f32)));
        }
        let cond = if cond_tokens.is_empty() { None } else { Some(mlx_rs::ops::concatenate(&cond_tokens, 0)?) };
        timings.load_s += t0.elapsed().as_secs_f32();

        let Some((_, dit)) = self.dit.as_ref() else { anyhow::bail!("the DiT did not load") };
        let (lh, lw) = (h / SCALE, w / SCALE);
        let image_ids = grid_ids(lh, lw, 0.0);
        let noise = qwen_image_rs::comfy::torch_randn(req.seed, lh * lw * CHANNELS).map_err(anyhow::Error::msg)?;
        let noise: Vec<f32> = noise.iter().map(|&v| dit::bf16_round(v)).collect();
        let mut x = Array::from_slice(&noise, &crate::nn::shape(&[lh * lw, CHANNELS])?);
        // Fill: outside the mask the picture's own latent, noised to each step's time, is put
        // back after every step (the last step leaves it exact); inside, the model paints.
        let keep = match &req.mask {
            Some(m) => {
                let z0 = original.as_ref().context("a fill needs its picture")?.as_dtype(Dtype::Float32)?;
                Some((z0, mask_tokens(m, lw, lh)?, x.clone()))
            }
            None => None,
        };
        let texts: Vec<(Array, Vec<[f32; 4]>)> = enc.iter().map(|e| (e.clone(), text_ids(e.shape()[0] as usize))).collect();

        let steps = if req.steps == 0 { if flow_map { 16 } else { 50 } } else { req.steps };
        let sched = timesteps(steps, ALPHA);
        let cond_ref = cond.as_ref().map(|c| (c, cond_ids.as_slice()));
        // The text's and the references' embedded tokens and every position, once.
        let preps = texts.iter().map(|t| dit.prepare(&tokens(&x, &image_ids, cond_ref, t))).collect::<Result<Vec<_>>>()?;
        progress(0, steps);
        let t0 = Instant::now();
        for i in 0..steps {
            check()?;
            let (t, t1) = (sched[i], sched[i + 1]);
            let target = flow_map.then_some(t1);
            let v = dit.forward(&tokens(&x, &image_ids, cond_ref, &texts[0]), &preps[0], t, target)?;
            let v = match (texts.get(1), preps.get(1)) {
                (Some(neg), Some(prep)) => {
                    let u = dit.forward(&tokens(&x, &image_ids, cond_ref, neg), prep, t, target)?;
                    u.add(v.subtract(&u)?.multiply(Array::from_f32(req.guidance))?)?
                }
                _ => v,
            };
            x = x.add(v.multiply(Array::from_f32(t1 - t))?)?;
            if let Some((z0, m, eps)) = &keep {
                let known = z0.multiply(Array::from_f32(1.0 - t1))?.add(eps.multiply(Array::from_f32(t1))?)?;
                x = x.multiply(m)?.add(known.multiply(Array::from_f32(1.0).subtract(m)?)?)?;
            }
            x.eval()?;
            progress(i + 1, steps);
        }
        timings.steps_s = t0.elapsed().as_secs_f32();

        check()?;
        let t0 = Instant::now();
        let img = self.vae.decode(&from_tokens(&x, lh, lw)?)?;
        let (pw, ph, rgba) = vae::to_rgba_u8(&img)?;
        timings.decode_s = t0.elapsed().as_secs_f32();
        mlx_rs::memory::clear_cache()?;
        Ok(Picture { width: pw, height: ph, rgba, timings })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_runs_from_one_to_zero() {
        let s = timesteps(16, ALPHA);
        assert_eq!(s.len(), 17);
        assert!((s[0] - 1.0).abs() < 1e-6 && s[16].abs() < 1e-6);
        assert!(s.windows(2).all(|w| w[1] < w[0]));
    }

    #[test]
    fn lora_names_cover_bfl_and_diffusers() {
        let n = lora_names("mode_blocks.image.0.attn.to_out");
        assert!(n.contains(&"mode_blocks.image.0.attn.to_out.0".to_string()));
        assert!(n.contains(&"content_mode_blocks.image.0.attn_out".to_string()));
        assert!(lora_names("mode_blocks.image_cond.4.attn.to_q").contains(&"content_mode_blocks.image_cond.4.q_proj".to_string()));
        assert!(lora_names("single_transformer_blocks.27.ff.linear_in").contains(&"single_blocks.27.mlp_in".to_string()));
        assert!(lora_names("txt_mode_blocks.2.ff.linear_out").contains(&"txt_mode_blocks.2.mlp_out".to_string()));
    }

    #[test]
    fn sizes_fit() {
        assert_eq!(fit(1024, 1024, EDIT_PIXELS), (768, 768));
        assert_eq!(fit(512, 512, EDIT_PIXELS), (512, 512));
    }
}
