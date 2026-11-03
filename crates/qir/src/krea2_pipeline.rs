//! Krea 2 Turbo, resident (plan Phase 4): the Qwen3-VL-4B encoder, the single-stream DiT and
//! the Wan 2.1 VAE loaded once, text to image with ComfyUI's conventions (its noise for the
//! seed, the `simple` schedule over the model's sigma table with shift 1.15, eight steps, the
//! Wan21 latent statistics).
//!
//! The weights folder (bring your own: `qwen-image-rs convert-fp8` rewrites ComfyUI's fp8
//! repacks): `transformer/*.safetensors` (keys `blocks.*`, `txtfusion.*`, …),
//! `text_encoder/*.safetensors` (keys `model.*`), `vae/*.safetensors` (Wan's layout) and
//! `tokenizer/tokenizer.json` (Qwen3-VL's).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use crate::convrot_cache::{CacheSpec, Policy};
use crate::loader::WeightSet;
use crate::model::krea2::{self, EditRef, Krea2Config, Krea2Dit};
use crate::model::text_encoder::{QwenTextEncoder, TextConfig};
use crate::model::vision::{self, VisionConfig, VisionTower};
use crate::model::wan_vae::{self, WanVae};
use crate::pipeline::{Cancelled, Picture, RefImage, Request, Timings};
use crate::preview::StepPreview;
use crate::Result;

/// The longest side of the picture the encoder sees beside an edit's instruction (the
/// comfyui-krea2edit `grounding_px` default; the LoRAs trained at 384–768).
pub const GROUNDING_PX: u32 = 768;

/// How the pipeline is loaded.
#[derive(Clone, Debug)]
pub struct Krea2Options {
    /// ConvRot W8A8 INT8 for the DiT's block linears (the `convrot` feature): 12 GB instead of
    /// 24, and the encoder stays resident beside it.
    pub convrot: bool,
    /// Where the prequantized ConvRot DiT is cached (the quantisation is 11 s of each cold
    /// start otherwise).
    pub cache: CacheSpec,
}

impl Krea2Options {
    /// ConvRot on when the build has it, the cache where the environment says.
    pub fn defaults() -> Self {
        Self { convrot: cfg!(feature = "convrot"), cache: CacheSpec::from_cli(None, false, false) }
    }
}

/// The folder layout the pipeline reads.
pub fn is_snapshot(dir: &Path) -> bool {
    dir.join("tokenizer/tokenizer.json").is_file() && dir.join("text_encoder").is_dir() && dir.join("transformer").is_dir() && dir.join("vae").is_dir()
}

/// The pipeline's files: from the weights folder above, or from a ComfyUI `models` folder
/// holding ComfyUI's own files ([`crate::comfy_files::find_krea2`], Rust rebuild R4).
struct Files {
    /// The ConvRot cache's key: the transformer folder, or ComfyUI's DiT file.
    dit_key: PathBuf,
    dit: Vec<PathBuf>,
    text: Vec<PathBuf>,
    vae: Vec<PathBuf>,
    /// `tokenizer.json`, or the folder ComfyUI's Qwen tokenizer is assembled from.
    tokenizer: std::result::Result<PathBuf, PathBuf>,
}

impl Files {
    fn find(dir: &Path) -> Result<Option<Self>> {
        if is_snapshot(dir) {
            let tr = dir.join("transformer");
            return Ok(Some(Self {
                dit: WeightSet::resolve(&tr)?.files,
                dit_key: tr,
                text: WeightSet::resolve(&dir.join("text_encoder"))?.files,
                vae: WeightSet::resolve(&dir.join("vae"))?.files,
                tokenizer: Ok(dir.join("tokenizer/tokenizer.json")),
            }));
        }
        Ok(crate::comfy_files::find_krea2(dir).map(|f| Self {
            dit_key: f.dit.clone(),
            dit: vec![f.dit],
            text: vec![f.text],
            vae: vec![f.vae],
            tokenizer: Err(f.tokenizer),
        }))
    }

    fn tokenizer(&self) -> Result<Tokenizer> {
        match &self.tokenizer {
            Ok(file) => Tokenizer::from_file(file).map_err(|e| anyhow::anyhow!("load tokenizer: {e}")),
            Err(dir) => crate::comfy_files::qwen_tokenizer(dir),
        }
    }
}

/// Krea 2 in `dir` (either layout of [`Files`]).
pub fn is_available(dir: &Path) -> bool {
    Files::find(dir).is_ok_and(|f| f.is_some())
}

pub struct Krea2Pipeline {
    dev: Device,
    dtype: DType,
    tok: Tokenizer,
    /// The encoder's files, for loading it again (and its vision tower on the first edit).
    text: Vec<PathBuf>,
    te: std::sync::Mutex<Option<QwenTextEncoder>>,
    /// The encoder's vision tower, loaded on the first edit (f32, ~1.7 GB).
    vision: std::sync::Mutex<Option<VisionTower>>,
    dit: Krea2Dit,
    vae: WanVae,
    /// The stock style template names the RGB Qwen VAE; legacy Krea uses Wan's file.
    style_vae_file: Option<PathBuf>,
    style_vae: std::sync::Mutex<Option<WanVae>>,
    /// Drop the encoder after each prompt (the bf16 DiT leaves no room for it; it loads again
    /// in two seconds for the next prompt).
    evict_te: bool,
    /// The encoder's final norm and language-model head (Qwen3-VL-4B ties it to the token
    /// embedding), loaded on the first [`Self::chat`].
    head: std::sync::Mutex<Option<crate::pipeline::ChatHead>>,
}

impl Krea2Pipeline {
    pub fn load(model: &Path, opts: Krea2Options, progress: &mut dyn FnMut(f32, &str)) -> Result<Self> {
        let Some(files) = Files::find(model)? else {
            anyhow::bail!(
                "{} is neither a Krea 2 weights folder (tokenizer/, text_encoder/, transformer/, vae/) nor a ComfyUI models folder with its files \
                 (diffusion_models/krea2_turbo_fp8_scaled, text_encoders/qwen3vl_4b_fp8_scaled, vae/wan_2.1_vae, and ComfyUI's qwen25_tokenizer)",
                model.display()
            );
        };
        let dev = crate::device::best_device()?;
        let dtype = if matches!(dev, Device::Cuda(_)) { DType::BF16 } else { DType::F32 };
        let tok = files.tokenizer()?;

        // ComfyUI's fp8_scaled file as it is: 13 GB, run with its own fp8 recipe (`crate::fp8`).
        let raw = files.dit.clone();
        let fp8 = crate::fp8::Fp8Plan::from_files(&raw)?;
        let fp8_text = crate::fp8::Fp8Plan::from_files(&files.text)?.is_some();
        anyhow::ensure!(
            (fp8.is_none() && !fp8_text) || crate::fp8::supported_on(&dev),
            "{} holds fp8_scaled weights and this build has no fp8 kernels (the `fp8` feature)",
            model.display()
        );
        // Without ConvRot or fp8 the DiT is 24 GB in bf16: the encoder comes and goes per prompt.
        let evict_te = fp8.is_none() && (!opts.convrot || !cfg!(feature = "convrot"));
        let te = if evict_te {
            None
        } else {
            progress(0.0, "Loading the text encoder");
            let t0 = Instant::now();
            let te = load_text_encoder(&files.text, &dev, dtype)?;
            tracing::info!(load_s = t0.elapsed().as_secs_f32(), "text encoder loaded");
            Some(te)
        };

        progress(0.3, "Loading the transformer");
        let t0 = Instant::now();
        let dit_files =
            if fp8.is_some() { raw } else { crate::convrot_cache::resolve_file_set(&files.dit_key, raw, opts.convrot, &opts.cache, &Policy::krea2())? };
        let vb = crate::mapped::var_builder(&dit_files, dtype, &dev, |_| true)?;
        let dit = Krea2Dit::load_with(&Krea2Config::default(), opts.convrot, fp8.as_ref(), vb)?;
        tracing::info!(
            load_s = t0.elapsed().as_secs_f32(),
            convrot = opts.convrot && fp8.is_none(),
            fp8 = fp8.is_some(),
            full_precision = fp8.as_ref().map_or(0, |p| p.full_precision.len()),
            "DiT loaded"
        );

        progress(0.9, "Loading the VAE");
        let t0 = Instant::now();
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files.vae, dtype, &dev)? };
        let vae = WanVae::load(vb)?;
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "VAE loaded");
        progress(1.0, "Loaded");
        Ok(Self {
            dev,
            dtype,
            tok,
            text: files.text,
            te: std::sync::Mutex::new(te),
            vision: std::sync::Mutex::new(None),
            dit,
            vae,
            style_vae_file: if is_snapshot(model) { None } else { Some(model.join("vae/qwen_image_vae.safetensors")) },
            style_vae: std::sync::Mutex::new(None),
            evict_te,
            head: std::sync::Mutex::new(None),
        })
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self.dev, Device::Cuda(_))
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// The dtype the DiT runs in (what a LoRA's tensors are put in).
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The conditioning `(1, tokens, 12, 2560)` for a prompt, grounded on `picture` when one is
    /// given (an edit: the picture through the vision tower, in view before the instruction).
    fn condition(&self, prompt: &str, picture: Option<&RefImage>) -> Result<Tensor> {
        let feats = match picture {
            Some(p) => Some(self.grounding_features(p)?),
            None => None,
        };
        let text = if feats.is_some() { krea2::edit_template(prompt) } else { krea2::template(prompt) };
        let ids: Vec<u32> = self.tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?.get_ids().to_vec();
        let mut slot = self.te.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(load_text_encoder(&self.text, &self.dev, self.dtype)?);
        }
        let te = slot.as_ref().ok_or_else(|| anyhow::anyhow!("no text encoder"))?;
        let cond = match &feats {
            Some(f) => krea2::condition_with_images(te, &ids, &[f])?,
            None => krea2::condition(te, &ids, &self.dev)?,
        }
        .unsqueeze(0)?;
        if self.evict_te {
            *slot = None;
            let _ = self.dev.synchronize();
        }
        Ok(cond)
    }

    /// Answer as the vision-language model Krea 2's encoder is (Qwen3-VL-4B Instruct, its head
    /// tied to the token embedding): [`crate::pipeline::Pipeline::chat`] on the 4B, `image` in
    /// view as ComfyUI's Qwen-VL processing sizes it. AusBoss's Krea 2 inpaint workflow writes
    /// its prompt with this same file through ComfyUI's `TextGenerate`; answering here keeps
    /// Krea 2 resident instead of swapping in Qwen-Image-2.1's 8B encoder for the rewriter.
    pub fn chat(&self, system: &str, user: &str, image: Option<&RefImage>, opts: &crate::pipeline::ChatOptions, cancel: &dyn Fn() -> bool) -> Result<String> {
        use crate::pipeline::{sample, ChatRng, END_OF_TEXT, IM_END};
        crate::device::bind_thread(&self.dev);
        let t0 = Instant::now();
        let feats = match image {
            Some(p) => vec![self.vision_view(&rgb_image(p)?, None)?],
            None => Vec::new(),
        };
        let vision_block = if feats.is_empty() { "" } else { crate::model::text_encoder::prompt::VISION_BLOCK };
        let text = format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{vision_block}{user}<|im_end|>\n<|im_start|>assistant\n");
        let mut ids: Vec<u32> = self.tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode the chat: {e}"))?.get_ids().to_vec();
        let prompt_len = ids.len();
        let refs: Vec<&vision::VisionFeatures> = feats.iter().collect();
        let mut rng = ChatRng::new(opts.seed);
        let mut head = self.head.lock().unwrap_or_else(|e| e.into_inner());
        if head.is_none() {
            *head = Some(self.load_head()?);
        }
        let h = head.as_ref().ok_or_else(|| anyhow::anyhow!("no language-model head"))?;
        let mut slot = self.te.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(load_text_encoder(&self.text, &self.dev, self.dtype)?);
        }
        let te = slot.as_ref().ok_or_else(|| anyhow::anyhow!("no text encoder"))?;
        let tp = Instant::now();
        let answered: Result<()> = (|| {
            let (mut last, mut cache) = te.prefill(&ids, &refs)?;
            for _ in 0..opts.max_new {
                if cancel() {
                    return Err(Cancelled.into());
                }
                let logits = candle_core::Module::forward(&h.head, &last.apply(&h.norm)?)?;
                let logits = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                let next = sample(&logits, opts, &mut rng);
                if next == IM_END || next == END_OF_TEXT {
                    break;
                }
                ids.push(next);
                last = te.step(next, &mut cache)?;
            }
            Ok(())
        })();
        if self.evict_te {
            *slot = None;
            let _ = self.dev.synchronize();
        }
        answered?;
        let reply = self.tok.decode(&ids[prompt_len..], true).map_err(|e| anyhow::anyhow!("decode the reply: {e}"))?;
        tracing::info!(tokens = ids.len() - prompt_len, prompt_tokens = prompt_len, decode_s = tp.elapsed().as_secs_f32(), chat_s = t0.elapsed().as_secs_f32(), "krea2 chat");
        Ok(reply.trim().to_string())
    }

    /// The 4B's final norm and its head, the token embedding (tied), as Q8_0 on the GPU
    /// (0.4 GB; quantized on every core from a CPU copy, as the 2.1 head is).
    fn load_head(&self) -> Result<crate::pipeline::ChatHead> {
        let t0 = Instant::now();
        let cfg = TextConfig::qwen3vl_4b();
        let vb = crate::mapped::var_builder(&self.text, self.dtype, &self.dev, |n| n == "model.norm.weight")?;
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model").pp("norm"))?;
        let head = if matches!(self.dev, Device::Cuda(_)) {
            let host = crate::mapped::var_builder(&self.text, self.dtype, &Device::Cpu, |n| n == "model.embed_tokens.weight")?;
            let w = host.get((cfg.vocab_size, cfg.hidden_size), "model.embed_tokens.weight").context("the Qwen3-VL-4B token embedding")?;
            candle_core::quantized::QMatMul::from_qtensor(crate::q8::quantize_q8_0(&w, &self.dev)?)?
        } else {
            let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&self.text, self.dtype, &self.dev)? };
            candle_core::quantized::QMatMul::Tensor(vb.get((cfg.vocab_size, cfg.hidden_size), "model.embed_tokens.weight").context("the Qwen3-VL-4B token embedding")?)
        };
        tracing::info!(load_s = t0.elapsed().as_secs_f32(), "krea2 language-model head loaded");
        Ok(crate::pipeline::ChatHead { norm, head })
    }

    /// The picture as the encoder sees it: its longest side brought to [`GROUNDING_PX`] at
    /// most, then to the tower's grid (ComfyUI's `process_qwen2vl_images`), through the 4B
    /// vision tower (loaded on first use).
    fn grounding_features(&self, p: &RefImage) -> Result<vision::VisionFeatures> {
        let img = rgb_image(p)?;
        let (w, h) = img.dimensions();
        let img = if w.max(h) > GROUNDING_PX {
            let s = f64::from(GROUNDING_PX) / f64::from(w.max(h));
            let (nw, nh) = (((f64::from(w) * s).round() as u32).max(1), ((f64::from(h) * s).round() as u32).max(1));
            image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle)
        } else {
            img
        };
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb = Tensor::from_vec(img.into_raw(), (h, w, 3), &self.dev)?.permute((2, 0, 1))?.to_dtype(DType::F32)?.affine(1.0 / 255.0, 0.0)?;
        let (vw, vh) = vision::vit_size(w, h, 3136, 12_845_056);
        let rgb = if (vw, vh) == (w, h) { rgb } else { vision::resize_bilinear(&rgb, vh, vw)? };
        self.vision_features(&rgb, false)
    }

    fn vision_features(&self, rgb: &Tensor, editplus: bool) -> Result<vision::VisionFeatures> {
        let mut slot = self.vision.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            let t0 = Instant::now();
            let vb32 = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&self.text, DType::F32, &self.dev)? };
            *slot = Some(
                VisionTower::load(&VisionConfig::qwen3vl_4b(), vb32.pp("model").pp("visual"))
                    .context("the Qwen3-VL-4B vision tower (an encoder file without it cannot edit)")?,
            );
            tracing::info!(load_s = t0.elapsed().as_secs_f32(), "vision tower loaded");
        }
        let tower = slot.as_ref().ok_or_else(|| anyhow::anyhow!("no vision tower"))?;
        let (patches, gh, gw) = vision::patchify(rgb, tower.config())?;
        if editplus {
            tower.forward_editplus(&patches, gh, gw)
        } else {
            tower.forward(&patches, gh, gw)
        }
    }

    /// An edit's picture for the DiT: fitted to the target `tw × th` (centre-cropped to its
    /// aspect, then resampled; the comfyui-krea2edit `fit` path at a matched aspect) and
    /// through the VAE's encoder, in the sampler's space `(1, 16, th/8, tw/8)`.
    fn picture_latent(&self, p: &RefImage, tw: usize, th: usize) -> Result<Tensor> {
        self.picture_latent_with(&self.vae, p, tw, th)
    }

    fn picture_latent_with(&self, vae: &WanVae, p: &RefImage, tw: usize, th: usize) -> Result<Tensor> {
        anyhow::ensure!(vae.can_encode(), "this VAE file has no encoder; an edit needs it");
        let img = rgb_image(p)?;
        let (iw, ih) = (img.width() as f64, img.height() as f64);
        let s = (th as f64 / ih).max(tw as f64 / iw);
        let (cw, ch) = (((tw as f64 / s).round() as u32).clamp(1, img.width()), ((th as f64 / s).round() as u32).clamp(1, img.height()));
        let (x0, y0) = ((img.width() - cw) / 2, (img.height() - ch) / 2);
        let crop = image::imageops::crop_imm(&img, x0, y0, cw, ch).to_image();
        let fitted = if (cw as usize, ch as usize) == (tw, th) {
            crop
        } else {
            image::imageops::resize(&crop, tw as u32, th as u32, image::imageops::FilterType::CatmullRom)
        };
        let x =
            Tensor::from_vec(fitted.into_raw(), (th, tw, 3), &self.dev)?.permute((2, 0, 1))?.unsqueeze(0)?.to_dtype(DType::F32)?.affine(2.0 / 255.0, -1.0)?;
        vae.encode(&x)
    }

    /// A fill's VAE: the Qwen-Image VAE beside the weights when it is there (the AnyPaint
    /// workflows' `qwen_image_vae`, Wan 2.1's encoder with a sharper decoder), else the model's.
    fn with_fill_vae<T>(&self, run: impl FnOnce(&WanVae) -> Result<T>) -> Result<T> {
        if self.style_vae_file.as_ref().is_some_and(|p| p.is_file()) {
            self.with_style_vae(run)
        } else {
            run(&self.vae)
        }
    }

    fn with_style_vae<T>(&self, run: impl FnOnce(&WanVae) -> Result<T>) -> Result<T> {
        let Some(path) = &self.style_vae_file else { return run(&self.vae) };
        let mut slot = self.style_vae.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            anyhow::ensure!(path.is_file(), "the style template needs {}", path.display());
            let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[path], self.dtype, &self.dev)? };
            *slot = Some(WanVae::load(vb)?);
        }
        run(slot.as_ref().ok_or_else(|| anyhow::anyhow!("no style VAE"))?)
    }

    fn style_condition(&self, prompt: &str, pictures: &[RefImage]) -> Result<(Tensor, Vec<Tensor>)> {
        let mut features = Vec::new();
        let mut references = Vec::new();
        for picture in pictures {
            let (vl, vae) = style_inputs(picture, &self.dev)?;
            features.push(self.vision_features(&vl, true)?);
            references.push(self.with_style_vae(|encoder| encoder.encode(&vae))?.to_dtype(self.dtype)?);
        }
        let ids =
            self.tok.encode(krea2::style_template(prompt, pictures.len()), false).map_err(|e| anyhow::anyhow!("encode style prompt: {e}"))?.get_ids().to_vec();
        let mut slot = self.te.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(load_text_encoder(&self.text, &self.dev, self.dtype)?);
        }
        let te = slot.as_ref().ok_or_else(|| anyhow::anyhow!("no text encoder"))?;
        let cond = krea2::condition_with_images(te, &ids, &features.iter().collect::<Vec<_>>())?.unsqueeze(0)?;
        if self.evict_te {
            *slot = None;
            self.dev.synchronize()?;
        }
        Ok((cond, references))
    }

    /// Generate from one or two style pictures using stock EditPlus semantics.
    pub fn generate_style(&self, req: &Request, progress: &mut dyn FnMut(f32, &str), cancel: &dyn Fn() -> bool) -> Result<Picture> {
        self.generate_style_previewed(req, progress, None, cancel)
    }

    /// [`Self::generate_style`] with previews ([`Self::generate_previewed`]).
    pub fn generate_style_previewed(
        &self,
        req: &Request,
        progress: &mut dyn FnMut(f32, &str),
        preview: Option<&mut dyn StepPreview>,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Picture> {
        anyhow::ensure!((1..=2).contains(&req.references.len()), "Krea style needs one or two references");
        anyhow::ensure!(req.width > 0 && req.height > 0, "Krea style needs explicit width and height");
        self.generate_inner(req, progress, preview, cancel, true)
    }

    /// Make one picture; `progress` hears every step, `cancel` is read between steps.
    pub fn generate(&self, req: &Request, progress: &mut dyn FnMut(f32, &str), cancel: &dyn Fn() -> bool) -> Result<Picture> {
        self.generate_inner(req, progress, None, cancel, false)
    }

    /// [`Self::generate`], showing the denoised estimate `(1, 16, h/8, w/8)` to `preview`
    /// after the steps it asks for; the picture is the same either way.
    pub fn generate_previewed(
        &self,
        req: &Request,
        progress: &mut dyn FnMut(f32, &str),
        preview: Option<&mut dyn StepPreview>,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Picture> {
        self.generate_inner(req, progress, preview, cancel, false)
    }

    fn generate_inner(
        &self,
        req: &Request,
        progress: &mut dyn FnMut(f32, &str),
        mut preview: Option<&mut dyn StepPreview>,
        cancel: &dyn Fn() -> bool,
        style: bool,
    ) -> Result<Picture> {
        crate::device::bind_thread(&self.dev);
        anyhow::ensure!(req.steps > 0, "at least one step");
        // An edit: the first reference is the picture (one picture; the krea2_edit LoRAs take up
        // to two, a scene and a subject, which nothing asks for yet). Its size is the target's
        // when the request names none, floored to the 16-pixel grid.
        anyhow::ensure!(style || req.references.len() <= 1, "a Krea 2 edit takes one picture ({} given)", req.references.len());
        let picture = if style { None } else { req.references.first() };
        let (w, h) = match picture {
            Some(p) if req.width == 0 || req.height == 0 => (p.width / 16 * 16, p.height / 16 * 16),
            _ => (req.width, req.height),
        };
        anyhow::ensure!(w > 0 && h > 0 && w.is_multiple_of(16) && h.is_multiple_of(16), "the picture's sides must be positive multiples of 16 ({w}x{h} asked)");
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

        check(cancel)?;
        progress(0.02, "Encoding the prompt");
        let t0 = Instant::now();
        let (cond, style_refs) = if style { self.style_condition(&req.prompt, &req.references)? } else { (self.condition(&req.prompt, picture)?, Vec::new()) };
        let edit = match picture {
            Some(p) => Some(EditRef { latent: self.picture_latent(p, w, h)?.to_dtype(self.dtype)?, boost: req.ref_boost }),
            None => None,
        };
        self.dev.synchronize()?;
        timings.encode_s = t0.elapsed().as_secs_f32();
        crate::profile::report("krea2 encode");

        // ComfyUI's noise on the (1, 16, h, w) latent, in the sampler's normalised space; the
        // `simple` schedule over the model's table; Euler with the velocity the DiT returns.
        let (lh, lw) = (h / wan_vae::SCALE, w / wan_vae::SCALE);
        let schedule = crate::comfy::FlowSchedule::simple(
            req.steps,
            if style {
                // ModelSamplingFlux interpolates shift from the requested target token count.
                let slope = (1.15 - 0.5) / (4096.0 - 256.0);
                crate::comfy::Shift::Flux { shift: (slope * (w * h) as f64 / 256.0 + 0.5 - slope * 256.0) as f32, timesteps: 10_000 }
            } else {
                crate::comfy::FlowSchedule::KREA2
            },
        )
        .map_err(anyhow::Error::msg)?;
        let noise = crate::comfy::torch_randn(req.seed, wan_vae::LATENT_CHANNELS * lh * lw).map_err(anyhow::Error::msg)?;
        let mut x = Tensor::from_vec(noise, (1, wan_vae::LATENT_CHANNELS, lh, lw), &self.dev)?;
        let t0 = Instant::now();
        for i in 0..req.steps {
            check(cancel)?;
            let input = x.to_dtype(self.dtype)?;
            let v = if style {
                let positive = self.dit.forward_style(&input, schedule.sigma(i), &cond, &style_refs)?.to_dtype(DType::F32)?;
                if req.guidance != 1.0 {
                    let negative = self.dit.forward_style(&input, schedule.sigma(i), &cond.zeros_like()?, &style_refs)?.to_dtype(DType::F32)?;
                    (&negative + ((&positive - &negative)? * f64::from(req.guidance))?)?
                } else {
                    positive
                }
            } else {
                self.dit.forward_edit(&input, schedule.sigma(i), &cond, edit.as_ref())?
            }
            .to_dtype(DType::F32)?;
            if let Some(p) = preview.as_deref_mut() {
                if p.wants(i + 1, req.steps) {
                    match (&v * f64::from(schedule.sigma(i))).and_then(|sv| &x - sv) {
                        Ok(x0) => p.show(i + 1, req.steps, &x0),
                        Err(e) => tracing::warn!(error = %e, "preview skipped"),
                    }
                }
            }
            x = (x + (v * f64::from(schedule.dt(i)))?)?;
            progress(0.05 + 0.85 * (i + 1) as f32 / req.steps as f32, "Sampling");
        }
        self.dev.synchronize()?;
        timings.denoise_s = t0.elapsed().as_secs_f32();
        crate::profile::report("krea2 sampling");

        check(cancel)?;
        progress(0.92, "Decoding");
        let t0 = Instant::now();
        let img = if style { self.with_style_vae(|vae| vae.decode(&vae.denormalize(&x)?))? } else { self.vae.decode(&self.vae.denormalize(&x)?)? };
        self.dev.synchronize()?;
        timings.decode_s = t0.elapsed().as_secs_f32();
        let (pw, ph, rgba) = wan_vae::to_rgba_u8(&img)?;
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

    /// Inpaint with the AnyPaint LoRA (yijunwang2/krea2-anypaint; the recipe of its
    /// `anypaint.py` and of AusBoss's "Krea 2 Inpaint Masked" workflow): `req.references` is
    /// the picture (the crop around the area) and its mask (white = paint). The model is shown
    /// the picture with the area flattened to the median colour of the rest, as a registered
    /// reference over the whole canvas ([`Krea2Dit::registered_kv`]) and, beside the prompt, to
    /// the encoder ("Picture 1: …"); sampling starts from the picture's own latent under a
    /// noise mask (the area widened by `opts.seam_px`), so the known pixels stay where they are.
    /// The result is the picture's size floored to 16 pixels; the caller composites it back.
    pub fn fill_previewed(
        &self,
        req: &Request,
        opts: &FillOptions,
        progress: &mut dyn FnMut(f32, &str),
        mut preview: Option<&mut dyn StepPreview>,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Picture> {
        crate::device::bind_thread(&self.dev);
        anyhow::ensure!(req.steps > 0, "at least one step");
        anyhow::ensure!(req.references.len() == 2, "a Krea 2 fill takes the picture and its mask ({} given)", req.references.len());
        anyhow::ensure!(opts.denoise > 0.0 && opts.denoise <= 1.0, "denoise {} is outside 0..1", opts.denoise);
        let check = |cancel: &dyn Fn() -> bool| -> Result<()> {
            if cancel() {
                Err(Cancelled.into())
            } else {
                Ok(())
            }
        };
        let (src, mask_src) = (&req.references[0], &req.references[1]);
        let (w, h) = (src.width / 16 * 16, src.height / 16 * 16);
        anyhow::ensure!(w >= 16 && h >= 16, "the picture is too small to fill ({}x{})", src.width, src.height);
        let unmatched = self.dit.set_loras(&req.loras)?;
        if !unmatched.is_empty() {
            tracing::warn!(modules = unmatched.len(), first = unmatched[0], "LoRA modules this model does not have (a file for another model?)");
        }
        let mut timings = Timings::default();
        check(cancel)?;
        progress(0.02, "Encoding the prompt");
        let t0 = Instant::now();
        let fill = prepare_fill(src, mask_src, w, h, opts)?;
        let picture = RefImage { width: w, height: h, rgba: rgba_of(&fill.picture), offset: None };
        let feats = self.vision_view(&fill.filled, opts.vision_px)?;
        let ids: Vec<u32> = self
            .tok
            .encode(krea2::template(&format!("Picture 1: <|vision_start|><|image_pad|><|vision_end|>{}", req.prompt)), false)
            .map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?
            .get_ids()
            .to_vec();
        let cond = {
            let mut slot = self.te.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() {
                *slot = Some(load_text_encoder(&self.text, &self.dev, self.dtype)?);
            }
            let te = slot.as_ref().ok_or_else(|| anyhow::anyhow!("no text encoder"))?;
            let cond = krea2::condition_with_images(te, &ids, &[&feats])?.unsqueeze(0)?;
            if self.evict_te {
                *slot = None;
                let _ = self.dev.synchronize();
            }
            cond
        };
        let (rw, rh) = reference_size(w, h, opts.reference_edge);
        let small = image::imageops::resize(&fill.filled, rw as u32, rh as u32, image::imageops::FilterType::Triangle);
        let small = RefImage { width: rw, height: rh, rgba: rgba_of(&small), offset: None };
        let (reference, target) =
            self.with_fill_vae(|vae| Ok((self.picture_latent_with(vae, &small, rw, rh)?, self.picture_latent_with(vae, &picture, w, h)?)))?;
        let (reference, target) = (reference.to_dtype(self.dtype)?, target.to_dtype(DType::F32)?);
        let (lh, lw) = (h / wan_vae::SCALE, w / wan_vae::SCALE);
        let registered = self.dit.registered_kv(&reference, (lh, lw))?;
        let mask = Tensor::from_vec(latent_mask(&fill.noise_mask, lw, lh), (1, 1, lh, lw), &self.dev)?;
        self.dev.synchronize()?;
        timings.encode_s = t0.elapsed().as_secs_f32();

        // ComfyUI's KSampler at `denoise` < 1: the schedule of steps/denoise steps, its tail.
        let full = ((req.steps as f32 / opts.denoise) as usize).max(req.steps);
        let sched = crate::comfy::FlowSchedule::simple(full, crate::comfy::FlowSchedule::KREA2).map_err(anyhow::Error::msg)?;
        let sigmas = sched.sigmas[sched.sigmas.len() - req.steps - 1..].to_vec();
        let noise = crate::comfy::torch_randn(req.seed, wan_vae::LATENT_CHANNELS * lh * lw).map_err(anyhow::Error::msg)?;
        let noise = Tensor::from_vec(noise, (1, wan_vae::LATENT_CHANNELS, lh, lw), &self.dev)?;
        let keep = (1.0 - &mask)?;
        let noised = |sigma: f32| -> Result<Tensor> { Ok(((&noise * f64::from(sigma))? + (&target * f64::from(1.0 - sigma))?)?) };
        let mut x = noised(sigmas[0])?;
        let t0 = Instant::now();
        for i in 0..req.steps {
            check(cancel)?;
            let sigma = sigmas[i];
            let input = (x.broadcast_mul(&mask)? + noised(sigma)?.broadcast_mul(&keep)?)?;
            let v = self.dit.forward_registered(&input.to_dtype(self.dtype)?, sigma, &cond, &registered)?.to_dtype(DType::F32)?;
            let denoised = (&input - (&v * f64::from(sigma))?)?;
            let denoised = (denoised.broadcast_mul(&mask)? + target.broadcast_mul(&keep)?)?;
            if let Some(p) = preview.as_deref_mut() {
                if p.wants(i + 1, req.steps) {
                    p.show(i + 1, req.steps, &denoised);
                }
            }
            let d = ((&x - &denoised)? / f64::from(sigma))?;
            x = (&x + (d * f64::from(sigmas[i + 1] - sigma))?)?;
            progress(0.05 + 0.85 * (i + 1) as f32 / req.steps as f32, "Sampling");
        }
        let x = (x.broadcast_mul(&mask)? + target.broadcast_mul(&keep)?)?;
        self.dev.synchronize()?;
        timings.denoise_s = t0.elapsed().as_secs_f32();
        check(cancel)?;
        progress(0.92, "Decoding");
        let t0 = Instant::now();
        let img = self.with_fill_vae(|vae| vae.decode(&vae.denormalize(&x)?))?;
        self.dev.synchronize()?;
        timings.decode_s = t0.elapsed().as_secs_f32();
        let (pw, ph, rgba) = wan_vae::to_rgba_u8(&img)?;
        tracing::info!(
            encode_s = timings.encode_s,
            denoise_s = timings.denoise_s,
            decode_s = timings.decode_s,
            width = pw,
            height = ph,
            steps = req.steps,
            denoise = opts.denoise,
            "anypaint fill"
        );
        progress(1.0, "Done");
        Ok(Picture { width: pw as u32, height: ph as u32, rgba, timings })
    }

    /// The picture as the encoder sees it beside a fill's prompt: ComfyUI's Qwen-VL processing
    /// (the tower's grid, 3136 to 12.8 M pixels), the longest side first brought to `cap` when
    /// one is given.
    fn vision_view(&self, img: &image::RgbImage, cap: Option<u32>) -> Result<vision::VisionFeatures> {
        let (w, h) = img.dimensions();
        let img = match cap {
            Some(cap) if w.max(h) > cap => {
                let s = f64::from(cap) / f64::from(w.max(h));
                image::imageops::resize(
                    img,
                    ((f64::from(w) * s).round() as u32).max(1),
                    ((f64::from(h) * s).round() as u32).max(1),
                    image::imageops::FilterType::Triangle,
                )
            }
            _ => img.clone(),
        };
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb = Tensor::from_vec(img.into_raw(), (h, w, 3), &self.dev)?.permute((2, 0, 1))?.to_dtype(DType::F32)?.affine(1.0 / 255.0, 0.0)?;
        let (vw, vh) = vision::vit_size(w, h, 3136, 12_845_056);
        let rgb = if (vw, vh) == (w, h) { rgb } else { vision::resize_bilinear(&rgb, vh, vw)? };
        self.vision_features(&rgb, false)
    }
}

/// EditPlus's two area resizes, followed by Qwen3-VL's 32-pixel tower grid.
/// Returns tower RGB in [0,1] and VAE RGB in [-1,1], preserving aspect ratio.
pub fn style_inputs(p: &RefImage, dev: &Device) -> Result<(Tensor, Tensor)> {
    anyhow::ensure!(p.width > 0 && p.height > 0, "empty style picture");
    let rgb = rgb_image(p)?;
    let rgb = (Tensor::from_vec(rgb.into_raw(), (p.height, p.width, 3), dev)?.permute((2, 0, 1))?.to_dtype(DType::F32)? / 255.0)?;
    let size = |side: f64, multiple: f64| {
        let scale = side / ((p.width * p.height) as f64).sqrt();
        (
            ((p.width as f64 * scale / multiple).round_ties_even() * multiple) as usize,
            ((p.height as f64 * scale / multiple).round_ties_even() * multiple) as usize,
        )
    };
    let (vw, vh) = size(384.0, 1.0);
    anyhow::ensure!(vw > 0 && vh > 0, "style picture aspect ratio rounds a vision side to zero");
    let vl = resize_area(&rgb, vh, vw)?;
    let rounded = (((vw as f64 / 32.0).round_ties_even() * 32.0) as usize, ((vh as f64 / 32.0).round_ties_even() * 32.0) as usize);
    let (tw, th) = if (3136..=12_845_056).contains(&(rounded.0 * rounded.1)) { rounded } else { vision::vit_size(vw, vh, 3136, 12_845_056) };
    let vl = if (tw, th) == (vw, vh) { vl } else { vision::resize_bilinear(&vl, th, tw)? };
    let (rw, rh) = size(1024.0, 8.0);
    anyhow::ensure!(rw > 0 && rh > 0, "style picture aspect ratio rounds a VAE side to zero");
    let vae = resize_area(&rgb, rh, rw)?.unsqueeze(0)?.affine(2.0, -1.0)?;
    Ok((vl, vae))
}

/// Torch area interpolation is adaptive average pooling, including when upscaling.
fn resize_area(rgb: &Tensor, dh: usize, dw: usize) -> Result<Tensor> {
    let (channels, sh, sw) = rgb.dims3()?;
    let source = rgb.flatten_all()?.to_vec1::<f32>()?;
    let mut output = vec![0.0f32; channels * dh * dw];
    for c in 0..channels {
        for y in 0..dh {
            let y0 = y * sh / dh;
            let y1 = ((y + 1) * sh).div_ceil(dh);
            for x in 0..dw {
                let x0 = x * sw / dw;
                let x1 = ((x + 1) * sw).div_ceil(dw);
                let mut sum = 0.0;
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        sum += source[(c * sh + sy) * sw + sx];
                    }
                }
                output[(c * dh + y) * dw + x] = sum / (y1 - y0) as f32 / (x1 - x0) as f32;
            }
        }
    }
    Ok(Tensor::from_vec(output, (channels, dh, dw), rgb.device())?)
}

/// A reference's pixels as RGB (its alpha dropped, as ComfyUI's `LoadImage` hands a picture to
/// the VAE and the encoder).
fn rgb_image(p: &RefImage) -> Result<image::RgbImage> {
    let rgba: image::RgbaImage = image::ImageBuffer::from_raw(p.width as u32, p.height as u32, p.rgba.clone())
        .ok_or_else(|| anyhow::anyhow!("a picture of {}x{} with {} bytes", p.width, p.height, p.rgba.len()))?;
    Ok(image::DynamicImage::ImageRgba8(rgba).to_rgb8())
}

fn load_text_encoder(files: &[PathBuf], dev: &Device, dtype: DType) -> Result<QwenTextEncoder> {
    // The language model faulted in on every core first ([`crate::mapped`]), not the vision tower.
    let vb = crate::mapped::var_builder(files, dtype, dev, |n| n.starts_with("model.") && !n.starts_with("model.visual."))?;
    QwenTextEncoder::load_at(&TextConfig::qwen3vl_4b(), false, vb.pp("model")).context("the Qwen3-VL-4B encoder")
}

/// How a Krea 2 AnyPaint fill runs ([`Krea2Pipeline::fill_previewed`]).
#[derive(Clone, Debug)]
pub struct FillOptions {
    /// How much of the area is redrawn: 1 replaces it, about 0.7 keeps the old shape.
    pub denoise: f32,
    /// The band around the area the sampler may also repaint (pixels at the picture's size);
    /// AnyPaint's `SEAM_PX`.
    pub seam_px: u32,
    /// The registered reference's longest side (AnyPaint's `REFERENCE_MAX_EDGE`).
    pub reference_edge: u32,
    /// The encoder's view of the picture, its longest side at most this (none: the picture).
    pub vision_px: Option<u32>,
}

impl Default for FillOptions {
    fn default() -> Self {
        Self { denoise: 1.0, seam_px: ANYPAINT_SEAM_PX, reference_edge: ANYPAINT_REFERENCE_EDGE, vision_px: None }
    }
}

/// AnyPaint's boundary band (`anypaint.py` `SEAM_PX`).
pub const ANYPAINT_SEAM_PX: u32 = 32;
/// AnyPaint's reference size (`anypaint.py` `REFERENCE_MAX_EDGE`).
pub const ANYPAINT_REFERENCE_EDGE: u32 = 384;

/// A fill's pictures at the working size.
pub struct PreparedFill {
    /// The picture as given.
    pub picture: image::RgbImage,
    /// The picture with the area flattened to the median colour of the rest.
    pub filled: image::RgbImage,
    /// The sampler's mask, 0..1 per pixel: the area widened by the seam band.
    pub noise_mask: Vec<f32>,
}

/// The picture and mask at `w × h`, the area (mask over half) flattened to the median colour of
/// what is outside it (mid-grey when nothing is), and the noise mask: the area dilated by
/// `opts.seam_px` (a square window, as AnyPaint's `_dilate` does).
pub fn prepare_fill(src: &RefImage, mask: &RefImage, w: usize, h: usize, opts: &FillOptions) -> Result<PreparedFill> {
    let picture = rgb_image(src)?;
    let picture = if (picture.width() as usize, picture.height() as usize) == (w, h) {
        picture
    } else {
        image::imageops::resize(&picture, w as u32, h as u32, image::imageops::FilterType::CatmullRom)
    };
    let m = rgb_image(mask)?;
    let m = if (m.width() as usize, m.height() as usize) == (w, h) {
        m
    } else {
        image::imageops::resize(&m, w as u32, h as u32, image::imageops::FilterType::Triangle)
    };
    let hole: Vec<bool> = m.pixels().map(|p| p.0[0] > 127).collect();
    anyhow::ensure!(hole.iter().any(|&b| b), "the mask marks nothing to fill");
    let mut channels: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (p, &inside) in picture.pixels().zip(&hole) {
        if !inside {
            for c in 0..3 {
                channels[c].push(p.0[c]);
            }
        }
    }
    let median = |v: &mut Vec<u8>| -> u8 {
        if v.is_empty() {
            return 127;
        }
        let mid = (v.len() - 1) / 2;
        *v.select_nth_unstable(mid).1
    };
    let colour = [median(&mut channels[0]), median(&mut channels[1]), median(&mut channels[2])];
    let mut filled = picture.clone();
    for (p, &inside) in filled.pixels_mut().zip(&hole) {
        if inside {
            p.0 = colour;
        }
    }
    let noise_mask = dilate(&hole, w, h, opts.seam_px as usize).into_iter().map(|b| if b { 1.0 } else { 0.0 }).collect();
    Ok(PreparedFill { picture, filled, noise_mask })
}

/// `mask` (`w × h`) dilated by `r` pixels with a square window, in two separable passes.
fn dilate(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    if r == 0 {
        return mask.to_vec();
    }
    let pass = |src: &[bool], along_x: bool| -> Vec<bool> {
        let mut out = vec![false; w * h];
        let (outer, inner) = if along_x { (h, w) } else { (w, h) };
        for o in 0..outer {
            let at = |i: usize| if along_x { o * w + i } else { i * w + o };
            // The distance to the last set pixel seen, forward then backward.
            let mut last: Option<usize> = None;
            for i in 0..inner {
                if src[at(i)] {
                    last = Some(i);
                }
                if last.is_some_and(|l| i - l <= r) {
                    out[at(i)] = true;
                }
            }
            last = None;
            for i in (0..inner).rev() {
                if src[at(i)] {
                    last = Some(i);
                }
                if last.is_some_and(|l| l - i <= r) {
                    out[at(i)] = true;
                }
            }
        }
        out
    };
    pass(&pass(mask, true), false)
}

/// A pixel mask (`w × h` = `8·lw × 8·lh`) at the latent's size: the mean of each 8×8 cell.
fn latent_mask(mask: &[f32], lw: usize, lh: usize) -> Vec<f32> {
    let (w, s) = (lw * wan_vae::SCALE, wan_vae::SCALE);
    let mut out = vec![0f32; lw * lh];
    for y in 0..lh {
        for x in 0..lw {
            let mut sum = 0.0;
            for dy in 0..s {
                for dx in 0..s {
                    sum += mask[(y * s + dy) * w + x * s + dx];
                }
            }
            out[y * lw + x] = sum / (s * s) as f32;
        }
    }
    out
}

/// AnyPaint's reference size: the longest side brought to `max_edge`, each side the nearest
/// multiple of 16 (at least 16).
pub fn reference_size(w: usize, h: usize, max_edge: u32) -> (usize, usize) {
    let snap = |v: f64| ((v / 16.0).round() as usize * 16).max(16);
    let longest = w.max(h) as f64;
    let s = if max_edge > 0 && longest > f64::from(max_edge) { f64::from(max_edge) / longest } else { 1.0 };
    (snap(w as f64 * s), snap(h as f64 * s))
}

fn rgba_of(img: &image::RgbImage) -> Vec<u8> {
    img.pixels().flat_map(|p| [p.0[0], p.0[1], p.0[2], 255]).collect()
}

#[cfg(test)]
mod fill_tests {
    use super::*;

    fn picture(w: usize, h: usize, f: impl Fn(usize, usize) -> [u8; 4]) -> RefImage {
        let mut rgba = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                rgba.extend_from_slice(&f(x, y));
            }
        }
        RefImage { width: w, height: h, rgba, offset: None }
    }

    #[test]
    fn the_area_takes_the_median_colour_and_the_noise_mask_widens_it() -> Result<()> {
        let src = picture(64, 32, |x, _| if x < 32 { [200, 10, 10, 255] } else { [10, 10, 200, 255] });
        // The area: x in 40..48, every row.
        let mask = picture(64, 32, |x, _| if (40..48).contains(&x) { [255, 255, 255, 255] } else { [0, 0, 0, 255] });
        let opts = FillOptions { seam_px: 4, ..FillOptions::default() };
        let f = prepare_fill(&src, &mask, 64, 32, &opts)?;
        // 32 red columns, 24 blue outside the area: the median is red's... per channel.
        let inside = f.filled.get_pixel(44, 5).0;
        assert_eq!(inside, [200, 10, 10], "median of 32 red and 24 blue columns per channel");
        assert_eq!(f.filled.get_pixel(10, 5).0, [200, 10, 10]);
        assert_eq!(f.filled.get_pixel(60, 5).0, [10, 10, 200], "outside the area untouched");
        let row: Vec<f32> = (0..64).map(|x| f.noise_mask[5 * 64 + x]).collect();
        assert!(row[36..52].iter().all(|&v| v == 1.0) && row[35] == 0.0 && row[52] == 0.0, "{row:?}");
        Ok(())
    }

    #[test]
    fn an_empty_mask_is_refused() {
        let src = picture(32, 32, |_, _| [1, 2, 3, 255]);
        let mask = picture(32, 32, |_, _| [0, 0, 0, 255]);
        assert!(prepare_fill(&src, &mask, 32, 32, &FillOptions::default()).is_err());
    }

    #[test]
    fn the_reference_is_384_on_its_longest_side_in_sixteens() {
        assert_eq!(reference_size(1280, 960, 384), (384, 288));
        assert_eq!(reference_size(1000, 333, 384), (384, 128));
        assert_eq!(reference_size(200, 100, 384), (208, 96));
    }

    #[test]
    fn the_latent_mask_averages_cells() {
        let mut m = vec![0f32; 16 * 8];
        for y in 0..8 {
            for x in 0..4 {
                m[y * 16 + x] = 1.0;
            }
        }
        assert_eq!(latent_mask(&m, 2, 1), vec![0.5, 0.0]);
    }
}
