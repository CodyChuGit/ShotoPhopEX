//! The diffusion models behind `gn-engine`'s worker protocol (plan Phase 2): Qwen-Image-2.1
//! text to image through `qwen-image-rs`'s resident pipeline. The editor's `RunRequest` with
//! task `image` and template `qwen-2.1/image` makes a picture with ComfyUI's conventions (its
//! noise for the seed, its `simple` schedule), so the two routes agree picture for picture.
//!
//! The snapshot lives under the model root as `qwen-image-2.1/` (Hugging Face's layout:
//! `processor/ text_encoder/ transformer/ vae/`); the INT8 DiT cache under `.qir-cache/` beside
//! it. The three models load on the first request and stay on the device.

#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use gn_engine::protocol::{Blob, DoneInfo, ErrorKind, Health, RunRequest};
use gn_engine::server::{Engine, EngineError};
use qwen_image_rs::convrot_cache::CacheSpec;
use qwen_image_rs::pipeline::{Cancelled, Options, Pipeline, Request, Schedule, is_snapshot};
use serde_json::{Value, json};

/// The template id this engine answers, as PhotoshopEX names it.
pub const TEMPLATE: &str = "qwen-2.1/image";
/// The model root's subdirectory for the snapshot.
pub const MODEL_DIR: &str = "qwen-image-2.1";
/// The model root's subdirectory for the prequantized DiT.
pub const CACHE_DIR: &str = ".qir-cache";
/// The template's defaults when a request leaves them at zero (ComfyUI's `qwen-2.1/image`).
pub const DEFAULT_STEPS: u32 = 25;
pub const DEFAULT_SIDE: u32 = 1024;

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
        _blobs: &[Vec<u8>],
        progress: &mut dyn FnMut(f32, &str),
        cancel: &AtomicBool,
    ) -> Result<(DoneInfo, Vec<Vec<u8>>), EngineError> {
        let started = Instant::now();
        if req.model != TEMPLATE {
            return Err(EngineError::new(ErrorKind::ModelMissing, format!("{} is not served here ({TEMPLATE} is)", req.model)));
        }
        if req.task != "image" {
            return Err(EngineError::new(ErrorKind::Unsupported, format!("this engine makes pictures from text only, not `{}`", req.task)));
        }
        if req.prompt.trim().is_empty() {
            return Err(EngineError::new(ErrorKind::BadRequest, "a prompt is needed"));
        }
        let result: Result<qwen_image_rs::pipeline::Picture, EngineError> = (|| {
            self.ensure_loaded(progress)?;
            let Some(p) = self.pipeline.as_ref() else { return Err(EngineError::new(ErrorKind::Internal, "no pipeline")) };
            if cancel.load(Ordering::SeqCst) {
                return Err(EngineError::cancelled());
            }
            let ask = Request {
                prompt: req.prompt.clone(),
                negative: req.negative.clone(),
                width: side(req.width),
                height: side(req.height),
                steps: if req.steps == 0 { DEFAULT_STEPS } else { req.steps } as usize,
                seed: req.seed,
                guidance: if req.guidance > 0.0 { req.guidance } else { 1.0 },
                schedule: Schedule::Comfy,
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
        let installed: Vec<&str> = if self.has_snapshot() { vec![TEMPLATE] } else { Vec::new() };
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
        let r = e.run(&req(TEMPLATE, "edit"), &[], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::Unsupported, .. })));
        assert!(!has_models(Path::new("Z:/nowhere")));
    }

    #[test]
    fn flags_and_sizes() {
        assert!(from_args(&["--runtime".into(), "a".into()]).is_none());
        let e = from_args(&["--runtime".into(), "a".into(), "--models".into(), "m".into(), "--text-quant".into()]).unwrap();
        assert!(e.quant_text);
        assert_eq!(e.snapshot_dir(), PathBuf::from("m").join(MODEL_DIR));
        assert_eq!((side(0), side(1024), side(1000), side(5)), (1024, 1024, 992, 16));
        assert_eq!(kind_of(&anyhow::anyhow!("CUDA_ERROR_OUT_OF_MEMORY")), ErrorKind::OutOfMemory);
        assert_eq!(kind_of(&Cancelled.into()), ErrorKind::Cancelled);
        assert_eq!(kind_of(&anyhow::anyhow!("weight directory does not exist: x")), ErrorKind::ModelMissing);
    }
}
