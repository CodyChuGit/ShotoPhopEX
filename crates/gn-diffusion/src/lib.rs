//! The diffusion models behind `gn-engine`'s worker protocol (plan Phases 2–4): Qwen-Image-2.1
//! and Krea 2 Turbo through `qwen-image-rs`'s resident pipelines. The editor's `RunRequest`
//! with template `qwen-2.1/image` (task `image`) makes a picture from text, `qwen-2.1/edit`
//! (task `edit`) re-renders the picture it carries by instruction with its reference pictures
//! in context, `qwen-2.1/fill` (task `fill`) does the same with the mask as the second
//! picture, `qwen-2.1/expand` places the picture inside the larger canvas, `qwen-2.1/matte`
//! answers RGBA whose alpha is the cut-out, the way the app's ComfyUI graphs do;
//! `krea2-turbo/image` makes a picture with Krea 2 Turbo. All with ComfyUI's conventions (its
//! noise for the seed, its `simple` schedule over each model's sigma table, its reference
//! sizes), so the two routes agree picture for picture.
//!
//! The weights live under the model root: `qwen-image-2.1/` (Hugging Face's layout:
//! `processor/ text_encoder/ transformer/ vae/`) with the INT8 DiT cache under `.qir-cache/`
//! beside it, and `krea2-turbo/` (`tokenizer/ text_encoder/ transformer/ vae/`; the
//! transformer either ComfyUI's `fp8_scaled` repack as it is, run with ComfyUI's own fp8
//! recipe, or bf16 safetensors from `qwen-image-rs convert-fp8`, run with ConvRot). One family's
//! models are resident at a time (the card holds either set, not both); the other family's
//! first request swaps them.

#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use std::sync::Arc;

use gn_engine::protocol::{Blob, DoneInfo, ErrorKind, Health, RunRequest};
use gn_engine::server::{Engine, EngineError};
use qwen_image_rs::convrot_cache::CacheSpec;
use qwen_image_rs::krea2_pipeline::{self, Krea2Options, Krea2Pipeline};
use qwen_image_rs::lora::{LoraFile, LoraStack};
use qwen_image_rs::pipeline::{Cancelled, ChatOptions, Options, Picture, Pipeline, RefImage, Request, Schedule, is_snapshot};
use serde_json::{Value, json};

/// The model root's subdirectory for LoRA files (ComfyUI's `models/loras` layout: one file
/// each); a request may also name a file by its absolute path.
pub const LORA_DIR: &str = "loras";
/// How many LoRA files stay loaded on the device between requests.
const LORA_CACHE: usize = 6;

/// The text-to-image template id this engine answers, as PhotoshopEX names it.
pub const TEMPLATE: &str = "qwen-2.1/image";
/// The edit template: the picture (image 1) and up to nine references (image 2, …).
pub const TEMPLATE_EDIT: &str = "qwen-2.1/edit";
/// The fill template: the picture (image 1) and the selection mask (image 2).
pub const TEMPLATE_FILL: &str = "qwen-2.1/fill";
/// The expand template: the prefilled canvas (image 1) and the mask of the new area (image 2).
pub const TEMPLATE_EXPAND: &str = "qwen-2.1/expand";
/// The matte template: the picture alone, the answer's alpha is the cut-out (the 2.1 VAE
/// decodes RGBA).
pub const TEMPLATE_MATTE: &str = "qwen-2.1/matte";
/// Krea 2 Turbo text to image.
pub const TEMPLATE_KREA2: &str = "krea2-turbo/image";
/// The app's prompt rewriter and judge with the picture in view, and without: answered by the
/// 2.1 text encoder as the vision-language model it is (Qwen3-VL 8B Instruct with its
/// language-model head), so they need no second model on the card.
pub const TEMPLATE_CHAT_EDIT: &str = "enhance/edit";
pub const TEMPLATE_CHAT_IMAGE: &str = "enhance/image";

/// The model families served, one resident at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    QwenImage21,
    Krea2,
}

/// Every template served, with the task each one answers and the family that serves it.
pub const TEMPLATES: [(&str, &str, Family); 8] = [
    (TEMPLATE, "image", Family::QwenImage21),
    (TEMPLATE_EDIT, "edit", Family::QwenImage21),
    (TEMPLATE_FILL, "fill", Family::QwenImage21),
    (TEMPLATE_EXPAND, "expand", Family::QwenImage21),
    (TEMPLATE_MATTE, "matte", Family::QwenImage21),
    (TEMPLATE_KREA2, "image", Family::Krea2),
    (TEMPLATE_CHAT_EDIT, "text", Family::QwenImage21),
    (TEMPLATE_CHAT_IMAGE, "text", Family::QwenImage21),
];
/// The model root's subdirectory for the Qwen-Image-2.1 snapshot.
pub const MODEL_DIR: &str = "qwen-image-2.1";
/// The model root's subdirectory for the prequantized 2.1 DiT.
pub const CACHE_DIR: &str = ".qir-cache";
/// The model root's subdirectory for the Krea 2 Turbo weights.
pub const KREA2_DIR: &str = "krea2-turbo";
/// The pre-quantized Qwen3-VL-8B text encoder inside the 2.1 snapshot folder, used when there.
pub const TEXT_GGUF: &str = "text_encoder.q8_0.gguf";
/// The templates' defaults when a request leaves them at zero (ComfyUI's templates).
pub const DEFAULT_STEPS: u32 = 25;
pub const KREA2_STEPS: u32 = 8;
pub const DEFAULT_SIDE: u32 = 1024;
/// How many pictures an edit may carry: the picture and the template's nine references.
pub const MAX_PICTURES: usize = 10;

/// What is on the device.
enum Resident {
    None,
    Qwen(Pipeline),
    Krea2(Krea2Pipeline),
}

pub struct DiffusionEngine {
    models_root: PathBuf,
    /// Q8_0 the 2.1 text encoder (for cards under 28 GB).
    quant_text: bool,
    resident: Resident,
    last_error: Option<String>,
    /// LoRA files on the device, by path, in the order they were loaded (the oldest goes when
    /// the cache is full); cleared with the models.
    loras: Vec<(PathBuf, Arc<LoraFile>)>,
    /// The card as the driver describes it, asked once when the engine is made.
    gpu: Option<qwen_image_rs::device::Gpu>,
    /// Why a CUDA build serves nothing on this PC (no NVIDIA driver or card, a card older than
    /// the kernels): the engine lists no templates and says so in its health.
    refused: Option<String>,
}

/// The card and, in a CUDA build, the sentence that refuses it (plan-native-engine §10, GPU
/// detection). A CPU build refuses nothing: it runs, slowly, for tests.
fn gpu_and_refusal() -> (Option<qwen_image_rs::device::Gpu>, Option<String>) {
    use qwen_image_rs::device::{built_compute_cap, check, probe};
    let probed = probe();
    if !cfg!(feature = "cuda") {
        return (probed.ok(), None);
    }
    let refused = match &probed {
        Ok(g) => check(g, built_compute_cap()).err(),
        Err(e) => Some(e.clone()),
    };
    let refused = refused.map(|why| format!("The native diffusion models need an NVIDIA RTX card: {why}."));
    (probed.ok(), refused)
}

/// One LoRA a request asks for: `{"file": "name-or-path.safetensors", "strength": 1.0}`.
fn lora_asks(params: &std::collections::BTreeMap<String, Value>) -> Result<Vec<(String, f32)>, EngineError> {
    let Some(v) = params.get("loras") else { return Ok(Vec::new()) };
    let Some(items) = v.as_array() else {
        return Err(EngineError::new(ErrorKind::BadRequest, "`loras` must be a list of {file, strength}"));
    };
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        let file = it.get("file").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty());
        let Some(file) = file else {
            return Err(EngineError::new(ErrorKind::BadRequest, "a LoRA needs its `file`"));
        };
        let strength = it.get("strength").and_then(Value::as_f64).unwrap_or(1.0);
        if !strength.is_finite() || !(-4.0..=4.0).contains(&strength) {
            return Err(EngineError::new(ErrorKind::BadRequest, format!("LoRA strength {strength} is out of the range -4..4")));
        }
        out.push((file.to_string(), strength as f32));
    }
    Ok(out)
}

impl DiffusionEngine {
    pub fn new(models_root: PathBuf, quant_text: bool) -> Self {
        let (gpu, refused) = gpu_and_refusal();
        if let Some(why) = &refused {
            eprintln!("gn-diffusion: {why}");
        }
        Self { models_root, quant_text, resident: Resident::None, last_error: None, loras: Vec::new(), gpu, refused }
    }

    /// Where a LoRA named by a request lives: an absolute path as given, a name under the model
    /// root's `loras/`. A name must stay a plain file name (no folders) so a request cannot read
    /// outside the root.
    fn lora_path(&self, file: &str) -> Result<PathBuf, EngineError> {
        let p = Path::new(file);
        if p.is_absolute() {
            return Ok(p.to_path_buf());
        }
        if p.components().count() != 1 {
            return Err(EngineError::new(ErrorKind::BadRequest, format!("a LoRA is named by its file name or an absolute path, not `{file}`")));
        }
        Ok(self.models_root.join(LORA_DIR).join(p))
    }

    /// A text answer (the app's `enhance/*` templates): the 2.1 pipeline's encoder as Qwen3-VL,
    /// the system turn from `params.system`, the picture (if any) in view, sampled as the app's
    /// ComfyUI templates ask (temperature 0.3, top-k 40, top-p 0.9, at most 120 tokens).
    fn chat(
        &mut self,
        req: &RunRequest,
        blobs: &[Vec<u8>],
        progress: &mut dyn FnMut(f32, &str),
        cancel: &AtomicBool,
        started: Instant,
    ) -> Result<(DoneInfo, Vec<Vec<u8>>), EngineError> {
        let image = match &req.image {
            Some(b) => Some(reference_of(b, blobs, "picture")?),
            None => None,
        };
        let system = req.params.get("system").and_then(Value::as_str).unwrap_or_default().to_string();
        self.ensure_loaded(Family::QwenImage21, progress)?;
        let Resident::Qwen(p) = &self.resident else { return Err(EngineError::new(ErrorKind::Internal, "no pipeline")) };
        progress(0.5, "Thinking");
        let opts = ChatOptions { seed: req.seed, ..ChatOptions::default() };
        let stop = || cancel.load(Ordering::SeqCst);
        let reply = p.chat(&system, &req.prompt, image.as_ref(), &opts, &stop);
        self.give_back();
        let reply = reply.map_err(|e| {
            exit_if_poisoned(&e.to_string());
            EngineError::new(kind_of(&e), e.to_string())
        })?;
        progress(1.0, "Done");
        Ok((DoneInfo { images: Vec::new(), texts: vec![reply], seed: req.seed, elapsed_ms: started.elapsed().as_millis() as u64 }, Vec::new()))
    }

    /// Between requests, the device memory the pool kept from this one goes back to the
    /// driver: the card is shared (ComfyUI, the desktop, other tenants), and 8–11 GB of
    /// retained slack pushed them into paging (`qwen_image_rs::device::trim_pool`).
    fn give_back(&self) {
        match &self.resident {
            Resident::Qwen(p) => qwen_image_rs::device::trim_pool(p.device()),
            Resident::Krea2(p) => qwen_image_rs::device::trim_pool(p.device()),
            Resident::None => {}
        }
    }

    /// The resident pipeline's CUDA context made current on this (the client's) thread,
    /// before anything of the last request's is dropped here
    /// (`qwen_image_rs::device::bind_thread`).
    fn bind(&self) {
        match &self.resident {
            Resident::Qwen(p) => qwen_image_rs::device::bind_thread(p.device()),
            Resident::Krea2(p) => qwen_image_rs::device::bind_thread(p.device()),
            Resident::None => {}
        }
    }

    /// The request's LoRA stack on the device of the resident pipeline (loaded once per file,
    /// kept for the next requests).
    fn stack_for(&mut self, params: &std::collections::BTreeMap<String, Value>) -> Result<LoraStack, EngineError> {
        let asks = lora_asks(params)?;
        if asks.is_empty() {
            return Ok(LoraStack::default());
        }
        let (dev, dtype) = match &self.resident {
            Resident::Qwen(p) => (p.device().clone(), p.dtype()),
            Resident::Krea2(p) => (p.device().clone(), p.dtype()),
            Resident::None => return Err(EngineError::new(ErrorKind::Internal, "no pipeline for the LoRAs")),
        };
        let mut layers = Vec::with_capacity(asks.len());
        for (file, strength) in asks {
            let path = self.lora_path(&file)?;
            let loaded = match self.loras.iter().find(|(p, _)| *p == path) {
                Some((_, f)) => Arc::clone(f),
                None => {
                    if !path.is_file() {
                        return Err(EngineError::new(ErrorKind::ModelMissing, format!("LoRA {} is not installed (looked in {})", file, self.models_root.join(LORA_DIR).display())));
                    }
                    let f = Arc::new(LoraFile::load(&path, &dev, dtype).map_err(|e| EngineError::new(ErrorKind::ModelUnreadable, format!("LoRA {file}: {e}")))?);
                    if self.loras.len() >= LORA_CACHE {
                        self.loras.remove(0);
                    }
                    self.loras.push((path, Arc::clone(&f)));
                    f
                }
            };
            layers.push((loaded, strength));
        }
        Ok(LoraStack { layers })
    }

    pub fn snapshot_dir(&self) -> PathBuf {
        self.models_root.join(MODEL_DIR)
    }

    pub fn krea2_dir(&self) -> PathBuf {
        self.models_root.join(KREA2_DIR)
    }

    fn has_family(&self, family: Family) -> bool {
        match family {
            Family::QwenImage21 => is_snapshot(&self.snapshot_dir()),
            Family::Krea2 => krea2_pipeline::is_snapshot(&self.krea2_dir()),
        }
    }

    /// The templates the weights on disk serve (none on a refused card).
    fn installed(&self) -> Vec<&'static str> {
        if self.refused.is_some() {
            return Vec::new();
        }
        TEMPLATES.iter().filter(|(_, _, f)| self.has_family(*f)).map(|(t, _, _)| *t).collect()
    }

    /// Load `family`'s pipeline if it is not resident, letting the other family's go first.
    fn ensure_loaded(&mut self, family: Family, progress: &mut dyn FnMut(f32, &str)) -> Result<(), EngineError> {
        self.bind();
        match (&self.resident, family) {
            (Resident::Qwen(_), Family::QwenImage21) | (Resident::Krea2(_), Family::Krea2) => return Ok(()),
            (Resident::None, _) => {}
            _ => {
                progress(0.0, "Unloading the other models");
                self.resident = Resident::None;
            }
        }
        let mut report = |f: f32, m: &str| progress(f * 0.3, m);
        let err = |e: anyhow::Error| EngineError::new(kind_of(&e), e.to_string());
        self.resident = match family {
            Family::QwenImage21 => {
                let dir = self.snapshot_dir();
                if !is_snapshot(&dir) {
                    return Err(EngineError::new(
                        ErrorKind::ModelMissing,
                        format!("no Qwen-Image-2.1 snapshot in {} (processor/, text_encoder/, transformer/, vae/)", dir.display()),
                    ));
                }
                // A pre-quantized text encoder beside the snapshot (`qwen-image-rs
                // prequantize-text`, Q8_0 GGUF) loads in seconds; the bf16 shards take about a
                // minute, Q8 on load two, and the encoder is loaded again after every eviction.
                let gguf = dir.join(TEXT_GGUF);
                let opts = Options {
                    cache: CacheSpec::from_cli(Some(&self.models_root.join(CACHE_DIR)), false, false),
                    quant_text: self.quant_text,
                    text_gguf: gguf.is_file().then_some(gguf),
                    ..Options::default()
                };
                Resident::Qwen(Pipeline::load(&dir, opts, &mut report).map_err(err)?)
            }
            Family::Krea2 => {
                let dir = self.krea2_dir();
                if !krea2_pipeline::is_snapshot(&dir) {
                    return Err(EngineError::new(
                        ErrorKind::ModelMissing,
                        format!("no Krea 2 weights in {} (tokenizer/, text_encoder/, transformer/, vae/; convert-fp8 makes them from ComfyUI's files)", dir.display()),
                    ));
                }
                let opts = Krea2Options { cache: CacheSpec::from_cli(Some(&self.models_root.join(CACHE_DIR)), false, false), ..Krea2Options::defaults() };
                Resident::Krea2(Krea2Pipeline::load(&dir, opts, &mut report).map_err(err)?)
            }
        };
        Ok(())
    }
}

/// The failure taxonomy from a pipeline error's text.
/// A CUDA error that leaves the process's context unusable (every later call fails the same
/// way): an illegal address, a failed launch, a misaligned access. Measured 2026-10-26: one
/// illegal address in a sampling step on a full card, and every request after it failed.
pub fn is_sticky_cuda_error(message: &str) -> bool {
    const STICKY: [&str; 6] = [
        "CUDA_ERROR_ILLEGAL_ADDRESS",
        "CUDA_ERROR_LAUNCH_FAILED",
        "CUDA_ERROR_ILLEGAL_INSTRUCTION",
        "CUDA_ERROR_MISALIGNED_ADDRESS",
        "CUDA_ERROR_HARDWARE_STACK_ERROR",
        "CUDA_ERROR_ASSERT",
    ];
    STICKY.iter().any(|s| message.contains(s))
}

/// The worker's way out of a poisoned CUDA context: say why and end the process, so the
/// editor's client sees the channel close and starts a fresh worker for the next request
/// (plan-native-engine §12.2, `EngineCrashed`: respawn). Continuing would fail every request.
fn exit_if_poisoned(message: &str) {
    if is_sticky_cuda_error(message) {
        eprintln!("gn-diffusion: the CUDA context is unusable ({message}); the worker exits so a fresh one takes over");
        std::process::exit(3);
    }
}

fn kind_of(e: &anyhow::Error) -> ErrorKind {
    if e.downcast_ref::<Cancelled>().is_some() {
        return ErrorKind::Cancelled;
    }
    let m = e.to_string().to_ascii_lowercase();
    if m.contains("out of memory") || m.contains("cuda_error_out_of_memory") || (m.contains("alloc") && m.contains("fail")) {
        ErrorKind::OutOfMemory
    } else if m.contains("does not exist") || m.contains("not found") || m.contains("no such file") {
        ErrorKind::ModelMissing
    } else if m.contains("safetensors") || m.contains("tokenizer") || m.contains("config.json") {
        ErrorKind::ModelUnreadable
    } else {
        ErrorKind::Internal
    }
}

/// A request's size: the template's default where it says nothing, rounded down to 16.
fn side(v: u32) -> usize {
    let v = if v == 0 { DEFAULT_SIDE } else { v };
    (v as usize / 16 * 16).max(16)
}

/// The task a served template answers; `None` for a template served elsewhere.
pub fn task_of(template: &str) -> Option<&'static str> {
    TEMPLATES.iter().find(|(t, _, _)| *t == template).map(|(_, task, _)| *task)
}

/// The family that serves a template.
pub fn family_of(template: &str) -> Option<Family> {
    TEMPLATES.iter().find(|(t, _, _)| *t == template).map(|(_, _, f)| *f)
}

/// A blob's bytes, checked against its header.
fn bytes_of<'a>(b: &Blob, blobs: &'a [Vec<u8>], what: &str) -> Result<&'a [u8], EngineError> {
    let bytes = blobs.get(b.index as usize).ok_or_else(|| EngineError::new(ErrorKind::BadRequest, format!("the {what} names blob {} that was not sent", b.index)))?;
    let want = b.width as usize * b.height as usize * usize::from(b.channels);
    if b.width == 0 || b.height == 0 || bytes.len() != want {
        return Err(EngineError::new(
            ErrorKind::BadRequest,
            format!("the {what} is {}x{} with {} channels but carries {} bytes", b.width, b.height, b.channels, bytes.len()),
        ));
    }
    Ok(bytes)
}

/// A picture blob as a reference: RGBA with the alpha made opaque, the way the ComfyUI route's
/// `LoadImage` hands the model its pictures (the colours as they are, the VAE's alpha channel
/// padded opaque; nothing is composited over white).
fn reference_of(b: &Blob, blobs: &[Vec<u8>], what: &str) -> Result<RefImage, EngineError> {
    let bytes = bytes_of(b, blobs, what)?;
    let rgba: Vec<u8> = match b.channels {
        4 => bytes.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        3 => bytes.as_chunks::<3>().0.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        1 => bytes.iter().flat_map(|v| [*v, *v, *v, 255]).collect(),
        c => return Err(EngineError::new(ErrorKind::BadRequest, format!("the {what} has {c} channels (1, 3 or 4 expected)"))),
    };
    Ok(RefImage { width: b.width as usize, height: b.height as usize, rgba, offset: None })
}

/// An expand's reference: the picture cut out of the canvas at the request's `ref_x`, `ref_y`,
/// `ref_w`, `ref_h` (widened to whole latent tokens of 16 px) and placed back at that offset,
/// so the model continues the picture into the rest of the canvas with the picture's tokens at
/// their own positions (the editor masks the result to the new area afterwards).
fn placed_picture(canvas: &RefImage, params: &std::collections::BTreeMap<String, Value>) -> Result<RefImage, EngineError> {
    let p = |k: &str| params.get(k).and_then(Value::as_u64).map(|v| v as usize);
    let (Some(x), Some(y), Some(w), Some(h)) = (p("ref_x"), p("ref_y"), p("ref_w"), p("ref_h")) else {
        return Err(EngineError::new(ErrorKind::BadRequest, "an expand names the picture's place in the canvas (ref_x, ref_y, ref_w, ref_h)"));
    };
    let (x0, y0) = (x / 16 * 16, y / 16 * 16);
    let x1 = ((x + w).div_ceil(16) * 16).min(canvas.width / 16 * 16);
    let y1 = ((y + h).div_ceil(16) * 16).min(canvas.height / 16 * 16);
    if x1 <= x0 || y1 <= y0 {
        return Err(EngineError::new(ErrorKind::BadRequest, format!("the picture's place {x},{y} {w}x{h} lies outside the {}x{} canvas", canvas.width, canvas.height)));
    }
    let (cw, ch) = (x1 - x0, y1 - y0);
    let mut rgba = Vec::with_capacity(cw * ch * 4);
    for row in y0..y1 {
        let start = (row * canvas.width + x0) * 4;
        rgba.extend_from_slice(canvas.rgba.get(start..start + cw * 4).unwrap_or(&[]));
    }
    if rgba.len() != cw * ch * 4 {
        return Err(EngineError::new(ErrorKind::BadRequest, "the canvas is shorter than its header says"));
    }
    Ok(RefImage { width: cw, height: ch, rgba, offset: Some((x0, y0)) })
}

impl Engine for DiffusionEngine {
    fn name(&self) -> String {
        "qir".into()
    }

    fn health(&mut self) -> Health {
        let (resident, cuda, vram) = match &self.resident {
            Resident::None => (Vec::new(), cfg!(feature = "cuda"), None),
            Resident::Qwen(p) => (vec![MODEL_DIR.to_string()], p.is_cuda(), qwen_image_rs::device::free_vram(p.device())),
            Resident::Krea2(p) => (vec![KREA2_DIR.to_string()], p.is_cuda(), qwen_image_rs::device::free_vram(p.device())),
        };
        // The card's name where the driver gave one; the free memory only while a pipeline
        // holds the context (asking without one would create it).
        let device = match (&self.gpu, cuda) {
            (Some(g), true) => g.name.clone(),
            (None, true) => "cuda:0".into(),
            (_, false) => "cpu".into(),
        };
        let detail = self.refused.clone().or_else(|| self.last_error.clone()).unwrap_or_default();
        Health {
            ok: !self.installed().is_empty() && self.last_error.is_none(),
            engine: "qir".into(),
            device,
            vram_total: vram.map(|(_, t)| t as u64).or(self.gpu.as_ref().map(|g| g.total_bytes as u64)),
            vram_free: vram.map(|(f, _)| f as u64),
            resident,
            detail,
        }
    }

    fn run(
        &mut self,
        req: &RunRequest,
        blobs: &[Vec<u8>],
        progress: &mut dyn FnMut(f32, &str),
        cancel: &AtomicBool,
    ) -> Result<(DoneInfo, Vec<Vec<u8>>), EngineError> {
        let started = Instant::now();
        if let Some(why) = &self.refused {
            return Err(EngineError::new(ErrorKind::Unsupported, why.clone()));
        }
        let (Some(task), Some(family)) = (task_of(&req.model), family_of(&req.model)) else {
            let served: Vec<&str> = TEMPLATES.iter().map(|(t, _, _)| *t).collect();
            return Err(EngineError::new(ErrorKind::ModelMissing, format!("{} is not served here ({} are)", req.model, served.join(", "))));
        };
        if req.task != task {
            return Err(EngineError::new(ErrorKind::Unsupported, format!("{} answers `{task}`, not `{}`", req.model, req.task)));
        }
        if req.prompt.trim().is_empty() {
            return Err(EngineError::new(ErrorKind::BadRequest, "a prompt is needed"));
        }
        if task == "text" {
            return self.chat(req, blobs, progress, cancel, started);
        }
        // The pictures the model sees in order: image 1 is the picture being edited, then the
        // mask (a fill) or the references (an edit; a matte has none). The target takes image
        // 1's size, as ComfyUI's `TextEncodeQwenImage21` sizes its latent at `resolution 0`.
        // An expand is the exception: its one reference is the picture cut out of the canvas
        // and placed at its own position, and the target is the whole canvas.
        let mut references: Vec<RefImage> = Vec::new();
        let mut canvas: Option<(usize, usize)> = None;
        match task {
            "edit" | "fill" | "expand" | "matte" => {
                let Some(image) = &req.image else { return Err(EngineError::new(ErrorKind::BadRequest, format!("{} needs the picture", req.model))) };
                let picture = reference_of(image, blobs, "picture")?;
                if task == "expand" {
                    canvas = Some((picture.width / 16 * 16, picture.height / 16 * 16));
                    references.push(placed_picture(&picture, &req.params)?);
                } else {
                    references.push(picture);
                }
                if task == "fill" {
                    let Some(mask) = &req.mask else { return Err(EngineError::new(ErrorKind::BadRequest, format!("{} needs the mask", req.model))) };
                    references.push(reference_of(mask, blobs, "mask")?);
                } else if task == "edit" {
                    for (i, r) in req.references.iter().enumerate() {
                        references.push(reference_of(r, blobs, &format!("reference {}", i + 2))?);
                    }
                }
                if references.len() > MAX_PICTURES {
                    return Err(EngineError::new(ErrorKind::BadRequest, format!("{} pictures; at most {MAX_PICTURES}", references.len())));
                }
            }
            _ => {}
        }
        let default_steps = match family {
            Family::QwenImage21 => DEFAULT_STEPS,
            Family::Krea2 => KREA2_STEPS,
        };
        let result: Result<Picture, EngineError> = (|| {
            self.ensure_loaded(family, progress)?;
            if cancel.load(Ordering::SeqCst) {
                return Err(EngineError::cancelled());
            }
            let (width, height) = match canvas {
                Some(c) => c,
                None if references.is_empty() => (side(req.width), side(req.height)),
                None => (0, 0),
            };
            let loras = self.stack_for(&req.params)?;
            let ask = Request {
                prompt: req.prompt.clone(),
                negative: req.negative.clone(),
                width,
                height,
                steps: if req.steps == 0 { default_steps } else { req.steps } as usize,
                seed: req.seed,
                guidance: if req.guidance > 0.0 { req.guidance } else { 1.0 },
                schedule: Schedule::Comfy,
                references,
                loras,
            };
            let mut report = |f: f32, m: &str| progress(0.3 + 0.7 * f, m);
            let stop = || cancel.load(Ordering::SeqCst);
            let err = |e: anyhow::Error| EngineError::new(kind_of(&e), e.to_string());
            match &self.resident {
                Resident::Qwen(p) => p.generate(&ask, &mut report, &stop).map_err(err),
                Resident::Krea2(p) => p.generate(&ask, &mut report, &stop).map_err(err),
                Resident::None => Err(EngineError::new(ErrorKind::Internal, "no pipeline")),
            }
        })();
        self.give_back();
        let pic = match result {
            Ok(p) => p,
            Err(e) => {
                exit_if_poisoned(&e.message);
                if matches!(e.kind, ErrorKind::ModelUnreadable | ErrorKind::Internal) {
                    self.last_error = Some(e.message.clone());
                }
                if e.kind == ErrorKind::OutOfMemory {
                    // Give the memory back: the next request reloads.
                    self.resident = Resident::None;
                    self.loras.clear();
                }
                return Err(e);
            }
        };
        self.last_error = None;
        let blob = Blob { width: pic.width, height: pic.height, channels: 4, index: 0 };
        Ok((DoneInfo { images: vec![blob], texts: Vec::new(), seed: req.seed, elapsed_ms: started.elapsed().as_millis() as u64 }, vec![pic.rgba]))
    }

    fn free(&mut self) {
        self.bind();
        self.resident = Resident::None;
        self.loras.clear();
    }

    fn index(&mut self, dir: &str) -> Value {
        json!({
            "dir": dir,
            "installed": self.installed(),
            "modelsRoot": self.models_root.to_string_lossy(),
            "snapshot": self.snapshot_dir().to_string_lossy(),
            "krea2": self.krea2_dir().to_string_lossy(),
        })
    }
}

/// Parse the worker's flags: `--models <dir>` (the same root the perception engine reads) and
/// `--text-quant`. `None` without a model root.
pub fn from_args(args: &[String]) -> Option<DiffusionEngine> {
    let mut models: Option<PathBuf> = None;
    let mut quant_text = false;
    let mut i = 0;
    while i < args.len() {
        match args.get(i).map(String::as_str) {
            Some("--models") => {
                models = args.get(i + 1).filter(|v| !v.trim().is_empty()).map(PathBuf::from);
                i += 2;
            }
            Some("--text-quant") => {
                quant_text = true;
                i += 1;
            }
            _ => i += 1,
        }
    }
    Some(DiffusionEngine::new(models?, quant_text))
}

/// Is `dir` a model root holding either family's weights?
pub fn has_models(dir: &Path) -> bool {
    is_snapshot(&dir.join(MODEL_DIR)) || krea2_pipeline::is_snapshot(&dir.join(KREA2_DIR))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn req(model: &str, task: &str) -> RunRequest {
        RunRequest {
            model: model.into(),
            task: task.into(),
            prompt: "a fox".into(),
            negative: String::new(),
            seed: 7,
            steps: 0,
            guidance: 0.0,
            width: 0,
            height: 0,
            image: None,
            mask: None,
            references: Vec::new(),
            params: BTreeMap::new(),
        }
    }

    #[test]
    fn missing_snapshot_is_a_typed_error_not_a_crash() {
        let mut e = DiffusionEngine::new(PathBuf::from("Z:/nowhere/models"), false);
        assert!(!e.health().ok);
        assert_eq!(e.index("")["installed"].as_array().map(Vec::len), Some(0));
        let r = e.run(&req(TEMPLATE, "image"), &[], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::ModelMissing, .. })), "{r:?}");
        let r = e.run(&req(TEMPLATE_KREA2, "image"), &[], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(&r, Err(EngineError { kind: ErrorKind::ModelMissing, message }) if message.contains("Krea 2")), "{r:?}");
        let r = e.run(&req("sam3.1/segment", "segment"), &[], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::ModelMissing, .. })));
        // A task the template does not answer.
        let r = e.run(&req(TEMPLATE, "edit"), &[], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::Unsupported, .. })));
        let r = e.run(&req(TEMPLATE_KREA2, "edit"), &[], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::Unsupported, .. })));
        // The edit and fill templates check their pictures before the models.
        let r = e.run(&req(TEMPLATE_EDIT, "edit"), &[], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(&r, Err(EngineError { kind: ErrorKind::BadRequest, message }) if message.contains("picture")), "{r:?}");
        let mut f = req(TEMPLATE_FILL, "fill");
        f.image = Some(Blob { width: 2, height: 1, channels: 4, index: 0 });
        let r = e.run(&f, &[vec![0; 8]], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(&r, Err(EngineError { kind: ErrorKind::BadRequest, message }) if message.contains("mask")), "{r:?}");
        f.mask = Some(Blob { width: 2, height: 1, channels: 1, index: 1 });
        let r = e.run(&f, &[vec![0; 8], vec![255, 0]], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::ModelMissing, .. })), "{r:?}");
        // A matte takes the picture alone and goes on to the models.
        let mut m = req(TEMPLATE_MATTE, "matte");
        m.image = Some(Blob { width: 2, height: 1, channels: 4, index: 0 });
        let r = e.run(&m, &[vec![0; 8]], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::ModelMissing, .. })), "{r:?}");
        assert!(!has_models(Path::new("Z:/nowhere")));
    }

    /// A refused card (no driver, too old for the kernels): no templates, the sentence in the
    /// health and in every run's answer.
    #[test]
    fn a_refused_card_serves_nothing_and_says_why() {
        let mut e = DiffusionEngine::new(PathBuf::from("Z:/nowhere/models"), false);
        e.refused = Some("The native diffusion models need an NVIDIA RTX card: Card (8.9) is too old.".into());
        let h = e.health();
        assert!(!h.ok && h.detail.contains("Card"), "{h:?}");
        assert!(e.installed().is_empty());
        let r = e.run(&req(TEMPLATE_KREA2, "image"), &[], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(&r, Err(EngineError { kind: ErrorKind::Unsupported, message }) if message.contains("RTX")), "{r:?}");
    }

    #[test]
    fn pictures_become_opaque_references() {
        let blobs = vec![vec![10, 20, 30, 0, 40, 50, 60, 128], vec![255, 0], vec![1, 2, 3]];
        let img = reference_of(&Blob { width: 2, height: 1, channels: 4, index: 0 }, &blobs, "picture").unwrap();
        assert_eq!((img.width, img.height), (2, 1));
        assert_eq!(img.rgba, vec![10, 20, 30, 255, 40, 50, 60, 255], "the colours as they are, alpha opaque");
        let mask = reference_of(&Blob { width: 2, height: 1, channels: 1, index: 1 }, &blobs, "mask").unwrap();
        assert_eq!(mask.rgba, vec![255, 255, 255, 255, 0, 0, 0, 255], "a grey picture of the mask");
        let rgb = reference_of(&Blob { width: 1, height: 1, channels: 3, index: 2 }, &blobs, "reference").unwrap();
        assert_eq!(rgb.rgba, vec![1, 2, 3, 255]);
        // Headers that do not match their bytes, or name a missing blob, are refused.
        let r = reference_of(&Blob { width: 3, height: 1, channels: 4, index: 0 }, &blobs, "picture");
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::BadRequest, .. })));
        let r = reference_of(&Blob { width: 1, height: 1, channels: 4, index: 9 }, &blobs, "picture");
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::BadRequest, .. })));
        let r = reference_of(&Blob { width: 1, height: 1, channels: 2, index: 1 }, &blobs, "picture");
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::BadRequest, .. })));
    }

    #[test]
    fn an_expand_places_the_picture_on_whole_tokens() {
        // A 64x48 canvas whose pixel value is its column; the picture at (20, 0) 30x48.
        let canvas = RefImage { width: 64, height: 48, rgba: (0..48).flat_map(|_| (0..64u8).flat_map(|x| [x, 0, 0, 255])).collect(), offset: None };
        let params: BTreeMap<String, Value> = [("ref_x", 20u64), ("ref_y", 0), ("ref_w", 30), ("ref_h", 48)].into_iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        let p = placed_picture(&canvas, &params).unwrap();
        // 20..50 widened to 16..64: 48 wide, placed at x = 16.
        assert_eq!((p.width, p.height, p.offset), (48, 48, Some((16, 0))));
        assert_eq!((p.rgba[0], p.rgba[4 * 47]), (16, 63), "the first row runs from column 16 to 63");
        assert!(placed_picture(&canvas, &BTreeMap::new()).is_err(), "no place named");
        let off: BTreeMap<String, Value> = [("ref_x", 70u64), ("ref_y", 0), ("ref_w", 10), ("ref_h", 10)].into_iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        assert!(placed_picture(&canvas, &off).is_err(), "outside the canvas");
    }

    #[test]
    fn flags_and_sizes() {
        assert!(from_args(&["--runtime".into(), "a".into()]).is_none());
        let e = from_args(&["--runtime".into(), "a".into(), "--models".into(), "m".into(), "--text-quant".into()]).unwrap();
        assert!(e.quant_text);
        assert_eq!(e.snapshot_dir(), PathBuf::from("m").join(MODEL_DIR));
        assert_eq!(e.krea2_dir(), PathBuf::from("m").join(KREA2_DIR));
        assert_eq!((side(0), side(1024), side(1000), side(5)), (1024, 1024, 992, 16));
        assert_eq!((task_of(TEMPLATE), task_of(TEMPLATE_EDIT), task_of(TEMPLATE_FILL), task_of("krea2-turbo/image")), (Some("image"), Some("edit"), Some("fill"), Some("image")));
        assert_eq!((task_of(TEMPLATE_EXPAND), task_of(TEMPLATE_MATTE), task_of("nope/x")), (Some("expand"), Some("matte"), None));
        assert_eq!((family_of(TEMPLATE_KREA2), family_of(TEMPLATE_FILL), family_of("nope/x")), (Some(Family::Krea2), Some(Family::QwenImage21), None));
        assert_eq!((task_of(TEMPLATE_CHAT_EDIT), task_of(TEMPLATE_CHAT_IMAGE), family_of(TEMPLATE_CHAT_EDIT)), (Some("text"), Some("text"), Some(Family::QwenImage21)));
        assert!(is_sticky_cuda_error(r#"DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was encountered")"#));
        assert!(!is_sticky_cuda_error("DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"), "out of memory is recoverable: the models are dropped instead");
        assert_eq!(kind_of(&anyhow::anyhow!("CUDA_ERROR_OUT_OF_MEMORY")), ErrorKind::OutOfMemory);
        assert_eq!(kind_of(&Cancelled.into()), ErrorKind::Cancelled);
        assert_eq!(kind_of(&anyhow::anyhow!("weight directory does not exist: x")), ErrorKind::ModelMissing);
    }

    #[test]
    fn loras_are_read_from_the_request_and_named_inside_the_root() {
        let p = |v: Value| -> BTreeMap<String, Value> { [("loras".to_string(), v)].into_iter().collect() };
        assert!(lora_asks(&BTreeMap::new()).unwrap().is_empty());
        let asks = lora_asks(&p(json!([{"file": "a.safetensors"}, {"file": "b.safetensors", "strength": 0.5}]))).unwrap();
        assert_eq!(asks, vec![("a.safetensors".to_string(), 1.0), ("b.safetensors".to_string(), 0.5)]);
        assert_eq!(lora_asks(&p(json!("x"))).unwrap_err().kind, ErrorKind::BadRequest);
        assert_eq!(lora_asks(&p(json!([{"strength": 1}]))).unwrap_err().kind, ErrorKind::BadRequest);
        assert_eq!(lora_asks(&p(json!([{"file": "a", "strength": 9}]))).unwrap_err().kind, ErrorKind::BadRequest);
        let e = DiffusionEngine::new(PathBuf::from("root"), false);
        assert_eq!(e.lora_path("a.safetensors").unwrap(), PathBuf::from("root").join(LORA_DIR).join("a.safetensors"));
        assert!(e.lora_path("../a.safetensors").is_err(), "no folders in a name");
        let abs = std::env::temp_dir().join("x.safetensors");
        assert_eq!(e.lora_path(&abs.to_string_lossy()).unwrap(), abs);
        // A LoRA asked for with no pipeline resident, or a file that is not there, is a typed error.
        let mut e = e;
        assert_eq!(e.stack_for(&p(json!([{"file": "a.safetensors"}]))).unwrap_err().kind, ErrorKind::Internal);
        assert!(e.stack_for(&BTreeMap::new()).unwrap().is_empty());
    }
}
