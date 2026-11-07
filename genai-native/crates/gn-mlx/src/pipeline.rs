//! MOSS-Image text to image on MLX, with `qwen_image_rs::moss::pipeline`'s conventions (torch's
//! CPU noise for the seed, Euler steps on the shift-5 schedule snapped to the training grid,
//! CFG whose negative branch is the empty prompt with block 22 skipped) and a 16 GB Mac's
//! memory rule: the encoder (8-bit) loads, encodes and goes before the DiT (8-bit) is loaded;
//! the DiT and the decoder then stay for the next picture. Converted weights are cached as MLX
//! safetensors (`cache_dir`), so only the first load reads the ComfyUI files.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use mlx_rs::{Array, Dtype};
use qwen_image_rs::moss::dit::{sigmas, MossConfig};
use qwen_image_rs::moss::pipeline::{find_files, MossFiles, Variant, SHIFT, UNCOND_SKIP};
use qwen_image_rs::moss::text::{template, TextConfig};
use tokenizers::Tokenizer;

use crate::convert::{self, Lora, Precision};
use crate::dit::MossDit;
use crate::nn::{limit_cache, Params};
use crate::text::Qwen35Text;
use crate::vae::{self, Flux2Decoder};

/// The run dtype of every model.
const DTYPE: Dtype = Dtype::Float16;
/// Pixels per latent pixel.
const SCALE: usize = 16;

/// One picture to make.
#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: String,
    pub negative: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// CFG scale; 1 runs the prompt alone.
    pub guidance: f32,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Timings {
    /// Loading (or converting) and running the encoder.
    pub encode_s: f32,
    /// Loading the DiT, when it was not resident.
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

/// The LoRA merged into the DiT at load (the turbo model's).
#[derive(Clone, Debug)]
pub struct LoraSpec {
    pub path: PathBuf,
    pub strength: f32,
}

pub struct MossMlx {
    files: MossFiles,
    cache_dir: PathBuf,
    lora: Option<LoraSpec>,
    tok: Tokenizer,
    vae: Flux2Decoder,
    dit: Option<MossDit>,
    dit_precision: Precision,
    /// The last prompts' encodings.
    encoded: Vec<(String, Array)>,
}

impl MossMlx {
    /// The files under the ComfyUI `models` folder `root` (Kromoss for [`Variant::Kromoss`]);
    /// converted weights cached under `cache_dir`.
    pub fn new(root: &Path, variant: Variant, lora: Option<LoraSpec>, cache_dir: &Path) -> Result<Self> {
        limit_cache()?;
        let files = find_files(root, variant).with_context(|| format!("MOSS-Image's files under {}", root.display()))?;
        let tok = Tokenizer::from_file(&files.tokenizer).map_err(|e| anyhow::anyhow!("load {}: {e}", files.tokenizer.display()))?;
        let vae = Flux2Decoder::load(&vae::convert(&files.vae, DTYPE)?, DTYPE)?;
        Ok(Self { files, cache_dir: cache_dir.to_path_buf(), lora, tok, vae, dit: None, dit_precision: Precision::Q8, encoded: Vec::new() })
    }

    /// Hold the DiT in `precision`: 8-bit by default (5.6 GB; 35 dB to the f16 DiT's picture at
    /// 512², 4-bit's drifts to 15 dB: another picture of the prompt), f16 needs 10 GB.
    pub fn set_dit_precision(&mut self, precision: Precision) {
        if precision != self.dit_precision {
            self.dit = None;
            self.dit_precision = precision;
        }
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
        path.file_stem().map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned())
    }

    /// The encodings of `prompts` (the encoder loaded, run and dropped if any is new).
    fn encode(&mut self, prompts: &[&str]) -> Result<Vec<Array>> {
        let missing: Vec<&str> = prompts.iter().copied().filter(|p| !self.encoded.iter().any(|(q, _)| q == p)).collect();
        if !missing.is_empty() {
            // The DiT goes first: the two never share the memory.
            self.dit = None;
            mlx_rs::memory::clear_cache()?;
            let cfg = TextConfig::qwen35_9b();
            let name = format!("{}.{}.safetensors", Self::stem(&self.files.text), Precision::Q8.tag());
            let embed_file = &self.files.text;
            // The INT8 embedding table is read from the file each time (it is not converted).
            let w = qwen_image_rs::moss::weights::Weights::open(embed_file, &candle_core::Device::Cpu, candle_core::DType::F32, false)?;
            let embed = w.prequantized("model.language_model.embed_tokens")?;
            drop(w);
            let params = self.cached(&name, || Ok(crate::text::convert(embed_file, &cfg, DTYPE, Precision::Q8)?.params))?;
            let te = Qwen35Text::load(&cfg, crate::text::TextWeights { params, embed }, DTYPE)?;
            for p in missing {
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

    fn dit(&mut self) -> Result<&MossDit> {
        if self.dit.is_none() {
            let cfg = MossConfig::default();
            let lora_tag = self.lora.as_ref().map_or_else(String::new, |l| format!(".{}-{}", Self::stem(&l.path), l.strength));
            let precision = self.dit_precision;
            let name = format!("{}{lora_tag}.{}.safetensors", Self::stem(&self.files.dit), precision.tag());
            let (dit_path, lora) = (self.files.dit.clone(), self.lora.clone());
            let params = self.cached(&name, || {
                let lora = lora.map(|l| Lora::open(&l.path, l.strength)).transpose()?;
                crate::dit::convert(&dit_path, &cfg, DTYPE, precision, lora.as_ref())
            })?;
            self.dit = Some(MossDit::load(&cfg, &params, DTYPE)?);
        }
        self.dit.as_ref().context("the DiT did not load")
    }

    /// Make one picture; `progress(step, steps)` before the first step (0) and after each.
    pub fn generate(&mut self, req: &Request, progress: &mut dyn FnMut(usize, usize)) -> Result<Picture> {
        anyhow::ensure!(req.steps > 0, "at least one step");
        let (w, h) = (req.width, req.height);
        anyhow::ensure!(w > 0 && h > 0 && w.is_multiple_of(SCALE) && h.is_multiple_of(SCALE), "the picture's sides must be positive multiples of {SCALE} ({w}x{h} asked)");
        anyhow::ensure!(w * h <= 4096 * 4096, "{w}x{h} is too large");
        let mut timings = Timings::default();

        let t0 = Instant::now();
        let cfg = req.guidance != 1.0;
        let prompts: Vec<&str> = if cfg { vec![&req.prompt, &req.negative] } else { vec![&req.prompt] };
        let enc = self.encode(&prompts)?;
        timings.encode_s = t0.elapsed().as_secs_f32();

        let t0 = Instant::now();
        let dit = self.dit()?;
        timings.load_s = t0.elapsed().as_secs_f32();

        let (lh, lw) = (h / SCALE, w / SCALE);
        let channels = dit.config().channels;
        let noise = qwen_image_rs::comfy::torch_randn(req.seed, channels * lh * lw).map_err(anyhow::Error::msg)?;
        let mut x = Array::from_slice(&noise, &crate::nn::shape(&[1, channels, lh, lw])?);
        let sig = sigmas(req.steps, SHIFT);
        let n = sig.len().saturating_sub(1);
        progress(0, n);
        let t0 = Instant::now();
        for i in 0..n {
            let (s0, s1) = (sig[i], sig[i + 1]);
            let pos = dit.forward(&x, s0 * 1000.0, &enc[0], &[])?;
            let v = match enc.get(1) {
                Some(neg) => {
                    let neg = dit.forward(&x, s0 * 1000.0, neg, &[UNCOND_SKIP])?;
                    neg.add(pos.subtract(&neg)?.multiply(Array::from_f32(req.guidance))?)?
                }
                None => pos,
            };
            x = x.add(v.multiply(Array::from_f32(s1 - s0))?)?;
            x.eval()?;
            progress(i + 1, n);
        }
        timings.steps_s = t0.elapsed().as_secs_f32();

        let t0 = Instant::now();
        let img = self.vae.decode(&x)?;
        let (pw, ph, rgba) = vae::to_rgba_u8(&img)?;
        timings.decode_s = t0.elapsed().as_secs_f32();
        mlx_rs::memory::clear_cache()?;
        Ok(Picture { width: pw, height: ph, rgba, timings })
    }
}
