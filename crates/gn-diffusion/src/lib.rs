//! The diffusion models behind `gn-engine`'s worker protocol (plan Phases 2 and 3):
//! Qwen-Image-2.1 through `qwen-image-rs`'s resident pipeline. The editor's `RunRequest` with
//! template `qwen-2.1/image` (task `image`) makes a picture from text, `qwen-2.1/edit` (task
//! `edit`) re-renders the picture it carries by instruction with its reference pictures in
//! context, and `qwen-2.1/fill` (task `fill`) does the same with the mask as the second
//! picture, the way the app's ComfyUI graphs do; all with ComfyUI's conventions (its noise for
//! the seed, its `simple` schedule, its reference sizes), so the two routes agree picture for
//! picture.
//!
//! The snapshot lives under the model root as `qwen-image-2.1/` (Hugging Face's layout:
//! `processor/ text_encoder/ transformer/ vae/`); the INT8 DiT cache under `.qir-cache/` beside
//! it. The models load on the first request and stay on the device.

#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use gn_engine::protocol::{Blob, DoneInfo, ErrorKind, Health, RunRequest};
use gn_engine::server::{Engine, EngineError};
use qwen_image_rs::convrot_cache::CacheSpec;
use qwen_image_rs::pipeline::{Cancelled, Options, Pipeline, RefImage, Request, Schedule, is_snapshot};
use serde_json::{Value, json};

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
/// Every template served, with the task each one answers.
pub const TEMPLATES: [(&str, &str); 5] =
    [(TEMPLATE, "image"), (TEMPLATE_EDIT, "edit"), (TEMPLATE_FILL, "fill"), (TEMPLATE_EXPAND, "expand"), (TEMPLATE_MATTE, "matte")];
/// The model root's subdirectory for the snapshot.
pub const MODEL_DIR: &str = "qwen-image-2.1";
/// The model root's subdirectory for the prequantized DiT.
pub const CACHE_DIR: &str = ".qir-cache";
/// The template's defaults when a request leaves them at zero (ComfyUI's `qwen-2.1/image`).
pub const DEFAULT_STEPS: u32 = 25;
pub const DEFAULT_SIDE: u32 = 1024;
/// How many pictures an edit may carry: the picture and the template's nine references.
pub const MAX_PICTURES: usize = 10;

pub struct DiffusionEngine {
    models_root: PathBuf,
    /// Q8_0 the text encoder (for cards under 28 GB).
    quant_text: bool,
    pipeline: Option<Pipeline>,
    last_error: Option<String>,
}

impl DiffusionEngine {
    pub fn new(models_root: PathBuf, quant_text: bool) -> Self {
        Self { models_root, quant_text, pipeline: None, last_error: None }
    }

    pub fn snapshot_dir(&self) -> PathBuf {
        self.models_root.join(MODEL_DIR)
    }

    fn has_snapshot(&self) -> bool {
        is_snapshot(&self.snapshot_dir())
    }

    /// Load the pipeline if it is not loaded yet.
    fn ensure_loaded(&mut self, progress: &mut dyn FnMut(f32, &str)) -> Result<(), EngineError> {
        if self.pipeline.is_some() {
            return Ok(());
        }
        let dir = self.snapshot_dir();
        if !is_snapshot(&dir) {
            return Err(EngineError::new(
                ErrorKind::ModelMissing,
                format!("no Qwen-Image-2.1 snapshot in {} (processor/, text_encoder/, transformer/, vae/)", dir.display()),
            ));
        }
        let opts =
            Options { cache: CacheSpec::from_cli(Some(&self.models_root.join(CACHE_DIR)), false, false), quant_text: self.quant_text, ..Options::default() };
        let mut report = |f: f32, m: &str| progress(f * 0.3, m);
        let p = Pipeline::load(&dir, opts, &mut report).map_err(|e| EngineError::new(kind_of(&e), e.to_string()))?;
        self.pipeline = Some(p);
        Ok(())
    }
}

/// The failure taxonomy from a pipeline error's text.
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
    TEMPLATES.iter().find(|(t, _)| *t == template).map(|(_, task)| *task)
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
    Ok(RefImage { width: b.width as usize, height: b.height as usize, rgba })
}

impl Engine for DiffusionEngine {
    fn name(&self) -> String {
        "qir".into()
    }

    fn health(&mut self) -> Health {
        let resident = if self.pipeline.is_some() { vec![MODEL_DIR.to_string()] } else { Vec::new() };
        Health {
            ok: self.has_snapshot() && self.last_error.is_none(),
            engine: "qir".into(),
            device: self.pipeline.as_ref().map_or_else(|| "cuda:0".into(), |p| if p.is_cuda() { "cuda:0".into() } else { "cpu".into() }),
            vram_total: None,
            vram_free: None,
            resident,
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
        let Some(task) = task_of(&req.model) else {
            let served: Vec<&str> = TEMPLATES.iter().map(|(t, _)| *t).collect();
            return Err(EngineError::new(ErrorKind::ModelMissing, format!("{} is not served here ({} are)", req.model, served.join(", "))));
        };
        if req.task != task {
            return Err(EngineError::new(ErrorKind::Unsupported, format!("{} answers `{task}`, not `{}`", req.model, req.task)));
        }
        if req.prompt.trim().is_empty() {
            return Err(EngineError::new(ErrorKind::BadRequest, "a prompt is needed"));
        }
        // The pictures the model sees in order: image 1 is the picture being edited, then the
        // mask (a fill or an expand) or the references (an edit; a matte has none). The target
        // takes image 1's size, as ComfyUI's `TextEncodeQwenImage21` sizes its latent at
        // `resolution 0`.
        let mut references: Vec<RefImage> = Vec::new();
        match task {
            "edit" | "fill" | "expand" | "matte" => {
                let Some(image) = &req.image else { return Err(EngineError::new(ErrorKind::BadRequest, format!("{} needs the picture", req.model))) };
                references.push(reference_of(image, blobs, "picture")?);
                if task == "fill" || task == "expand" {
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
        let result: Result<qwen_image_rs::pipeline::Picture, EngineError> = (|| {
            self.ensure_loaded(progress)?;
            let Some(p) = self.pipeline.as_ref() else { return Err(EngineError::new(ErrorKind::Internal, "no pipeline")) };
            if cancel.load(Ordering::SeqCst) {
                return Err(EngineError::cancelled());
            }
            let (width, height) = if references.is_empty() { (side(req.width), side(req.height)) } else { (0, 0) };
            let ask = Request {
                prompt: req.prompt.clone(),
                negative: req.negative.clone(),
                width,
                height,
                steps: if req.steps == 0 { DEFAULT_STEPS } else { req.steps } as usize,
                seed: req.seed,
                guidance: if req.guidance > 0.0 { req.guidance } else { 1.0 },
                schedule: Schedule::Comfy,
                references,
            };
            let mut report = |f: f32, m: &str| progress(0.3 + 0.7 * f, m);
            p.generate(&ask, &mut report, &|| cancel.load(Ordering::SeqCst)).map_err(|e| EngineError::new(kind_of(&e), e.to_string()))
        })();
        let pic = match result {
            Ok(p) => p,
            Err(e) => {
                if matches!(e.kind, ErrorKind::ModelUnreadable | ErrorKind::Internal) {
                    self.last_error = Some(e.message.clone());
                }
                if e.kind == ErrorKind::OutOfMemory {
                    // Give the memory back: the next request reloads.
                    self.pipeline = None;
                }
                return Err(e);
            }
        };
        self.last_error = None;
        let blob = Blob { width: pic.width, height: pic.height, channels: 4, index: 0 };
        Ok((DoneInfo { images: vec![blob], texts: Vec::new(), seed: req.seed, elapsed_ms: started.elapsed().as_millis() as u64 }, vec![pic.rgba]))
    }

    fn free(&mut self) {
        self.pipeline = None;
    }

    fn index(&mut self, dir: &str) -> Value {
        let installed: Vec<&str> = if self.has_snapshot() { TEMPLATES.iter().map(|(t, _)| *t).collect() } else { Vec::new() };
        json!({
            "dir": dir,
            "installed": installed,
            "modelsRoot": self.models_root.to_string_lossy(),
            "snapshot": self.snapshot_dir().to_string_lossy(),
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

/// Is `dir` a model root holding the snapshot?
pub fn has_models(dir: &Path) -> bool {
    is_snapshot(&dir.join(MODEL_DIR))
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
        let r = e.run(&req("sam3.1/segment", "segment"), &[], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::ModelMissing, .. })));
        // A task the template does not answer.
        let r = e.run(&req(TEMPLATE, "edit"), &[], &mut |_, _| {}, &AtomicBool::new(false));
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
    fn flags_and_sizes() {
        assert!(from_args(&["--runtime".into(), "a".into()]).is_none());
        let e = from_args(&["--runtime".into(), "a".into(), "--models".into(), "m".into(), "--text-quant".into()]).unwrap();
        assert!(e.quant_text);
        assert_eq!(e.snapshot_dir(), PathBuf::from("m").join(MODEL_DIR));
        assert_eq!((side(0), side(1024), side(1000), side(5)), (1024, 1024, 992, 16));
        assert_eq!((task_of(TEMPLATE), task_of(TEMPLATE_EDIT), task_of(TEMPLATE_FILL), task_of("krea2-turbo/image")), (Some("image"), Some("edit"), Some("fill"), None));
        assert_eq!((task_of(TEMPLATE_EXPAND), task_of(TEMPLATE_MATTE)), (Some("expand"), Some("matte")));
        assert_eq!(kind_of(&anyhow::anyhow!("CUDA_ERROR_OUT_OF_MEMORY")), ErrorKind::OutOfMemory);
        assert_eq!(kind_of(&Cancelled.into()), ErrorKind::Cancelled);
        assert_eq!(kind_of(&anyhow::anyhow!("weight directory does not exist: x")), ErrorKind::ModelMissing);
    }
}
