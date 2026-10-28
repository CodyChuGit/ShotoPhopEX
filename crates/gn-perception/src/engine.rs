//! The perception models behind `gn-engine`'s worker protocol: the `Engine` the editor's
//! `--engine` process serves once a runtime and a model root are configured.
//!
//! Requests arrive as the editor's `RunRequest` with task `segment`; the template id picks the
//! route (`…/segment` text, `…/segment-point` points, `…/segment-box` a box with optional
//! points) and `params` carry what PhotoshopEX's `select.*` commands send today: `threshold`,
//! `points` / `negatives` as JSON strings of `{x, y}` pixel coordinates in the request picture,
//! `bx by bw bh` as pixels. Every instance comes back as a 1-channel coverage blob.
//!
//! The runtime and the models load on the first request and stay resident; the last picture's
//! embedding is cached by content, so the probe that follows the pointer pays only the prompt.
//!
//! Two backends answer the same requests. With the `candle` feature and SAM 3.1's own
//! checkpoint on disk (ComfyUI's `checkpoints/sam3.1_multiplex_fp16.safetensors`, or one in
//! `sam3.1/`), `gn-segment` runs it on candle: no ONNX Runtime, no cuDNN, no exported graphs
//! (Rust rebuild R1). Otherwise the exported ONNX graphs run through ONNX Runtime.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use gn_engine::protocol::{Blob, DoneInfo, ErrorKind, Health, RunRequest};
use gn_engine::server::{Engine, EngineError};
use serde_json::{Value, json};

#[cfg(feature = "onnx")]
use crate::graph::Error;
use crate::graph::{Device, Instance, Options};
#[cfg(feature = "onnx")]
use crate::runtime::{Runtime, RuntimeConfig, library_name};
#[cfg(feature = "onnx")]
use crate::sam31::{Embedding, Sam31};

/// The template ids this engine answers, as PhotoshopEX names them.
pub const TEMPLATES: [&str; 3] = ["sam3.1/segment", "sam3.1/segment-point", "sam3.1/segment-box"];
/// The model root's subdirectory for SAM 3.1.
pub const SAM31_DIR: &str = "sam3.1";
/// SAM 3.1's checkpoint as ComfyUI names it.
pub const SAM31_CHECKPOINT: &str = "sam3.1_multiplex_fp16.safetensors";

/// The resident model: the candle port or the ONNX graphs.
enum Model {
    #[cfg(feature = "onnx")]
    Ort(Box<Sam31>),
    #[cfg(feature = "candle")]
    Candle(Box<gn_segment::sam::Sam31>),
    /// A build with neither backend never opens a model (every request is ModelMissing).
    #[cfg(not(any(feature = "onnx", feature = "candle")))]
    #[allow(dead_code)]
    None(std::convert::Infallible),
}

/// A picture's embedding, from the backend that made it.
enum Emb {
    #[cfg(feature = "onnx")]
    Ort(Embedding),
    #[cfg(feature = "candle")]
    Candle(Box<gn_segment::sam::Frame>),
    #[cfg(not(any(feature = "onnx", feature = "candle")))]
    #[allow(dead_code)]
    None(std::convert::Infallible),
}

/// The candle backend's files under a models root: the checkpoint and the CLIP tokenizer (the
/// ONNX folder's tokenizer.json, else ComfyUI's `comfy/sd1_tokenizer` beside a ComfyUI
/// `models` folder).
#[cfg(feature = "candle")]
fn candle_files(root: &Path) -> Option<(PathBuf, PathBuf)> {
    let checkpoint = [root.join("checkpoints").join(SAM31_CHECKPOINT), root.join(SAM31_DIR).join(SAM31_CHECKPOINT)].into_iter().find(|p| p.is_file())?;
    let own = root.join(SAM31_DIR).join("tokenizer.json");
    let comfy = root.parent().map(|p| p.join("comfy").join("sd1_tokenizer")).filter(|d| d.join("vocab.json").is_file() && d.join("merges.txt").is_file());
    let tokenizer = if own.is_file() { Some(own) } else { comfy }?;
    Some((checkpoint, tokenizer))
}

#[cfg(not(feature = "candle"))]
fn candle_files(_: &Path) -> Option<(PathBuf, PathBuf)> {
    None
}

pub struct PerceptionEngine {
    /// Directories searched for ONNX Runtime and the CUDA / cuDNN libraries.
    runtime_dirs: Vec<PathBuf>,
    /// The directory holding one subdirectory per model (`sam3.1/`), or a ComfyUI `models` folder.
    models_root: PathBuf,
    device: Device,
    /// SAM 3.1 on candle in f16 on CUDA (`--sam-precision f16`): 1.9x the speed of f32, every
    /// mask at IoU >= 0.993 against ComfyUI's own f32 node (gn-segment F16.md). f32 otherwise.
    half: bool,
    #[cfg(feature = "onnx")]
    rt: Option<Runtime>,
    sam: Option<Model>,
    /// The last picture embedded: a content hash and its embedding.
    cache: Option<(u64, Emb)>,
    last_error: Option<String>,
}

impl PerceptionEngine {
    pub fn new(runtime_dirs: Vec<PathBuf>, models_root: PathBuf, cpu: bool) -> Self {
        let device = if cpu { Device::Cpu } else { Device::Cuda { ordinal: 0 } };
        PerceptionEngine {
            runtime_dirs,
            models_root,
            device,
            half: false,
            #[cfg(feature = "onnx")]
            rt: None,
            sam: None,
            cache: None,
            last_error: None,
        }
    }

    /// SAM 3.1's compute precision on CUDA: f16 when `half`, else f32 (the CPU is always f32).
    pub fn with_half(mut self, half: bool) -> Self {
        self.half = half;
        self
    }

    /// The directory among `runtime_dirs` that holds the runtime library.
    #[cfg(feature = "onnx")]
    fn ort_dir(&self) -> Option<PathBuf> {
        self.runtime_dirs.iter().find(|d| d.join(library_name()).is_file()).cloned()
    }

    /// Built without ONNX Runtime: never one.
    #[cfg(not(feature = "onnx"))]
    fn ort_dir(&self) -> Option<PathBuf> {
        None
    }

    /// Is a backend's model on disk (the checkpoint for candle, the graphs for ONNX Runtime)?
    fn has_model(&self) -> bool {
        has_models(&self.models_root)
    }

    /// Can a backend run (candle needs no runtime library; the ONNX graphs need ONNX Runtime)?
    fn ready(&self) -> bool {
        candle_files(&self.models_root).is_some() || (onnx_layout(&self.models_root) && self.ort_dir().is_some())
    }

    /// Open SAM 3.1 if it is not open yet: on candle when its checkpoint is there, else the
    /// runtime and the ONNX graphs.
    #[cfg_attr(not(any(feature = "onnx", feature = "candle")), allow(unused_variables, unreachable_code))]
    fn ensure_sam(&mut self, progress: &mut dyn FnMut(f32, &str)) -> Result<(), EngineError> {
        #[cfg(feature = "candle")]
        if let Some((checkpoint, tokenizer)) = candle_files(&self.models_root).filter(|_| self.sam.is_none()) {
            progress(0.0, "Loading SAM 3.1");
            let device = match self.device {
                Device::Cuda { ordinal } => candle_core::Device::new_cuda(ordinal as usize).map_err(|e| candle_error(e.into()))?,
                Device::Cpu => candle_core::Device::Cpu,
            };
            let t0 = Instant::now();
            let dtype = if self.half && device.is_cuda() { candle_core::DType::F16 } else { candle_core::DType::F32 };
            let mut sam = gn_segment::sam::Sam31::load_with_dtype(&checkpoint, &device, dtype).map_err(candle_error)?;
            sam.set_tokenizer(&tokenizer).map_err(candle_error)?;
            eprintln!("gn-perception: SAM 3.1 on candle in {dtype:?} ({}) loaded in {:.1} s", checkpoint.display(), t0.elapsed().as_secs_f32());
            self.sam = Some(Model::Candle(Box::new(sam)));
        }
        #[cfg(all(feature = "candle", not(feature = "onnx")))]
        if self.sam.is_none() {
            return Err(EngineError::new(
                ErrorKind::ModelMissing,
                format!("no SAM 3.1 in {}: ComfyUI's checkpoints/{SAM31_CHECKPOINT} is missing", self.models_root.display()),
            ));
        }
        #[cfg(not(any(feature = "onnx", feature = "candle")))]
        return Err(EngineError::new(ErrorKind::ModelMissing, "this build has no SAM 3.1 backend (neither `candle` nor `onnx`)"));
        #[cfg(feature = "onnx")]
        if self.sam.is_none() {
            let dir = self.models_root.join(SAM31_DIR);
            if !Sam31::is_layout(&dir) {
                return Err(EngineError::new(
                    ErrorKind::ModelMissing,
                    format!(
                        "no SAM 3.1 in {}: neither ComfyUI's checkpoints/{SAM31_CHECKPOINT} nor ONNX graphs in {}",
                        self.models_root.display(),
                        dir.display()
                    ),
                ));
            }
            if self.rt.is_none() {
                let Some(ort_dir) = self.ort_dir() else {
                    return Err(EngineError::new(
                        ErrorKind::ModelUnreadable,
                        format!("no {} in the runtime directories {:?}", library_name(), self.runtime_dirs),
                    ));
                };
                progress(0.0, "Loading the runtime");
                let cfg = RuntimeConfig { ort_dir, library_dirs: self.runtime_dirs.clone(), device: self.device, profile_dir: None };
                self.rt = Some(Runtime::open(cfg).map_err(|e| EngineError::new(ErrorKind::ModelUnreadable, e.to_string()))?);
            }
            let Some(rt) = self.rt.as_ref() else { return Err(EngineError::new(ErrorKind::Internal, "no runtime")) };
            progress(0.0, "Loading SAM 3.1");
            self.sam = Some(Model::Ort(Box::new(Sam31::open(rt, &dir).map_err(|e| EngineError::new(kind_of(&e), e.to_string()))?)));
        }
        Ok(())
    }
}

#[cfg(feature = "candle")]
fn candle_error(e: anyhow::Error) -> EngineError {
    EngineError::new(ErrorKind::ModelUnreadable, format!("{e:#}"))
}

#[cfg(feature = "candle")]
fn candle_opts(o: Options) -> gn_segment::sam::Options {
    gn_segment::sam::Options { threshold: o.threshold, max_detections: o.max_detections, refine_iterations: o.refine_iterations }
}

#[cfg(feature = "candle")]
fn from_candle(i: gn_segment::sam::Instance) -> Instance {
    Instance { score: i.score, bbox: i.bbox, mask: i.mask, bounds: i.bounds, pixels: i.pixels }
}

#[cfg(all(feature = "candle", feature = "onnx"))]
fn mismatched() -> EngineError {
    EngineError::new(ErrorKind::Internal, "the cached picture came from the other backend")
}

/// The three routes on whichever backend is resident.
#[cfg_attr(not(any(feature = "onnx", feature = "candle")), allow(unused_variables))]
impl Model {
    fn embed(&mut self, rgba: &[u8], w: usize, h: usize) -> Result<Emb, EngineError> {
        match self {
            #[cfg(feature = "onnx")]
            Model::Ort(s) => s.embed(rgba, w, h).map(Emb::Ort).map_err(|e| EngineError::new(kind_of(&e), e.to_string())),
            #[cfg(feature = "candle")]
            Model::Candle(s) => s.encode(rgba, w, h).map(|f| Emb::Candle(Box::new(f))).map_err(candle_error),
            #[cfg(not(any(feature = "onnx", feature = "candle")))]
            Model::None(n) => match *n {},
        }
    }

    fn detect_text(&mut self, emb: &Emb, prompt: &str, boxes: &[[f32; 4]], opts: Options) -> Result<Vec<Instance>, EngineError> {
        match (self, emb) {
            #[cfg(feature = "onnx")]
            (Model::Ort(s), Emb::Ort(e)) => s.detect_text(e, prompt, boxes, opts).map(|r| r.0).map_err(|e| EngineError::new(kind_of(&e), e.to_string())),
            #[cfg(feature = "candle")]
            (Model::Candle(s), Emb::Candle(f)) => {
                s.detect_text(f, prompt, boxes, candle_opts(opts)).map(|r| r.0.into_iter().map(from_candle).collect()).map_err(candle_error)
            }
            #[cfg(all(feature = "candle", feature = "onnx"))]
            _ => Err(mismatched()),
            #[cfg(not(any(feature = "onnx", feature = "candle")))]
            (Model::None(n), _) => match *n {},
        }
    }

    fn segment_points(&mut self, emb: &Emb, positives: &[[f32; 2]], negatives: &[[f32; 2]], opts: Options) -> Result<Instance, EngineError> {
        match (self, emb) {
            #[cfg(feature = "onnx")]
            (Model::Ort(s), Emb::Ort(e)) => {
                s.segment_points(e, positives, negatives, opts).map(|r| r.0).map_err(|e| EngineError::new(kind_of(&e), e.to_string()))
            }
            #[cfg(feature = "candle")]
            (Model::Candle(s), Emb::Candle(f)) => s.segment_points(f, positives, negatives, candle_opts(opts)).map(|r| from_candle(r.0)).map_err(candle_error),
            #[cfg(all(feature = "candle", feature = "onnx"))]
            _ => Err(mismatched()),
            #[cfg(not(any(feature = "onnx", feature = "candle")))]
            (Model::None(n), _) => match *n {},
        }
    }

    fn segment_box(&mut self, emb: &Emb, bbox: [f32; 4], opts: Options) -> Result<Instance, EngineError> {
        match (self, emb) {
            #[cfg(feature = "onnx")]
            (Model::Ort(s), Emb::Ort(e)) => s.segment_box(e, bbox, opts).map(|r| r.0).map_err(|e| EngineError::new(kind_of(&e), e.to_string())),
            #[cfg(feature = "candle")]
            (Model::Candle(s), Emb::Candle(f)) => s.segment_box(f, bbox, candle_opts(opts)).map(|r| from_candle(r.0)).map_err(candle_error),
            #[cfg(all(feature = "candle", feature = "onnx"))]
            _ => Err(mismatched()),
            #[cfg(not(any(feature = "onnx", feature = "candle")))]
            (Model::None(n), _) => match *n {},
        }
    }
}

#[cfg(feature = "onnx")]
fn kind_of(e: &Error) -> ErrorKind {
    match e {
        Error::Missing(_) => ErrorKind::ModelMissing,
        Error::Runtime(_) | Error::Shape(_) | Error::Tokenizer(_) => ErrorKind::ModelUnreadable,
    }
}

/// FNV-1a over the picture bytes and its size: the embedding cache key.
fn content_hash(bytes: &[u8], w: u32, h: u32) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut step = |b: u8| {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for b in w.to_le_bytes().iter().chain(h.to_le_bytes().iter()) {
        step(*b);
    }
    for b in bytes {
        step(*b);
    }
    hash
}

/// `[{"x": 1, "y": 2}, …]` (a JSON string, as the ComfyUI templates receive it) → pixels.
fn points_param(v: Option<&Value>) -> Vec<[f32; 2]> {
    let parsed: Option<Value> = match v {
        Some(Value::String(s)) => serde_json::from_str(s).ok(),
        Some(other) => Some(other.clone()),
        None => None,
    };
    parsed
        .and_then(|p| p.as_array().cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|p| Some([p.get("x")?.as_f64()? as f32, p.get("y")?.as_f64()? as f32]))
        .collect()
}

fn num(params: &std::collections::BTreeMap<String, Value>, key: &str) -> Option<f32> {
    params.get(key).and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))).map(|f| f as f32)
}

/// A route of the node, from the template id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Text,
    Point,
    Box,
}

fn route_of(model: &str) -> Option<Route> {
    match model.rsplit('/').next().unwrap_or(model) {
        "segment" => Some(Route::Text),
        "segment-point" => Some(Route::Point),
        "segment-box" => Some(Route::Box),
        _ => None,
    }
}

fn coverage_blob(inst: &Instance, w: u32, h: u32, index: u32) -> (Blob, Vec<u8>) {
    let bytes: Vec<u8> = inst.mask.iter().map(|v| if *v > 0.5 { 255 } else { 0 }).collect();
    (Blob { width: w, height: h, channels: 1, index }, bytes)
}

impl Engine for PerceptionEngine {
    fn name(&self) -> String {
        "perception".into()
    }

    fn health(&mut self) -> Health {
        let mut resident = Vec::new();
        if self.sam.is_some() {
            resident.push("sam3.1".to_string());
        }
        let ready = self.ready();
        Health {
            ok: ready && self.last_error.is_none(),
            engine: "perception".into(),
            device: match self.device {
                Device::Cuda { ordinal } => format!("cuda:{ordinal}"),
                Device::Cpu => "cpu".into(),
            },
            vram_total: None,
            vram_free: None,
            resident,
            detail: self.last_error.as_ref().map(|e| format!("SAM 3.1: {e}")).unwrap_or_default(),
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
        if req.task != "segment" {
            return Err(EngineError::new(ErrorKind::Unsupported, format!("this engine serves segmentation only, not `{}`", req.task)));
        }
        let Some(route) = route_of(&req.model) else {
            return Err(EngineError::new(ErrorKind::ModelMissing, format!("{} is not a SAM 3.1 template", req.model)));
        };
        let Some(image) = req.image.as_ref() else {
            return Err(EngineError::new(ErrorKind::BadRequest, "a segmentation needs a picture"));
        };
        if image.channels != 4 {
            return Err(EngineError::new(ErrorKind::BadRequest, format!("the picture has {} channels, not 4", image.channels)));
        }
        let Some(rgba) = blobs.get(image.index as usize) else {
            return Err(EngineError::new(ErrorKind::BadRequest, "the picture's blob is missing"));
        };
        let (w, h) = (image.width, image.height);
        let opts = Options {
            threshold: num(&req.params, "threshold").unwrap_or(0.5),
            max_detections: num(&req.params, "maxDetections").map_or(1, |m| m.max(1.0) as usize),
            refine_iterations: num(&req.params, "refineIterations").map_or(2, |r| r.max(0.0) as usize),
        };
        let positives = points_param(req.params.get("points"));
        let negatives = points_param(req.params.get("negatives"));
        let bbox = match (num(&req.params, "bx"), num(&req.params, "by"), num(&req.params, "bw"), num(&req.params, "bh")) {
            (Some(x), Some(y), Some(bw), Some(bh)) => Some([x, y, x + bw, y + bh]),
            _ => None,
        };

        let result: Result<Vec<Instance>, EngineError> = (|| {
            self.ensure_sam(progress)?;
            // Two fields borrowed apart: the model runs, the cache is read and written.
            let Some(sam) = self.sam.as_mut() else { return Err(EngineError::new(ErrorKind::Internal, "no model")) };
            let cache = &mut self.cache;
            if cancel.load(Ordering::SeqCst) {
                return Err(EngineError::cancelled());
            }
            // The picture's embedding: reused when the same pixels come again.
            let key = content_hash(rgba, w, h);
            let hit = cache.as_ref().is_some_and(|(k, _)| *k == key);
            if !hit {
                progress(0.1, "Reading the picture");
                let emb = sam.embed(rgba, w as usize, h as usize)?;
                *cache = Some((key, emb));
            }
            let Some((_, emb)) = cache.as_ref() else { return Err(EngineError::new(ErrorKind::Internal, "no embedding")) };
            if cancel.load(Ordering::SeqCst) {
                return Err(EngineError::cancelled());
            }
            progress(0.5, "Segmenting");
            let mut out = Vec::new();
            match route {
                Route::Text => {
                    let boxes: Vec<[f32; 4]> = bbox.into_iter().collect();
                    out.extend(sam.detect_text(emb, &req.prompt, &boxes, opts)?);
                }
                Route::Point | Route::Box => {
                    if !positives.is_empty() || !negatives.is_empty() {
                        out.push(sam.segment_points(emb, &positives, &negatives, opts)?);
                    }
                    if let Some(b) = bbox {
                        out.push(sam.segment_box(emb, b, opts)?);
                    }
                    if out.is_empty() {
                        return Err(EngineError::new(ErrorKind::BadRequest, "a point or box segmentation needs points or a box"));
                    }
                }
            }
            Ok(out)
        })();
        let instances = match result {
            Ok(i) => i,
            Err(e) => {
                if matches!(e.kind, ErrorKind::ModelUnreadable | ErrorKind::Internal) {
                    self.last_error = Some(e.message.clone());
                }
                return Err(e);
            }
        };
        self.last_error = None;
        // Empty masks are dropped, as the ComfyUI route reports nothing for them.
        let kept: Vec<&Instance> = instances.iter().filter(|i| i.pixels > 0).collect();
        let mut images = Vec::with_capacity(kept.len());
        let mut bytes = Vec::with_capacity(kept.len());
        for (i, inst) in kept.iter().enumerate() {
            let (b, data) = coverage_blob(inst, w, h, i as u32);
            images.push(b);
            bytes.push(data);
        }
        progress(1.0, "Done");
        Ok((DoneInfo { images, texts: Vec::new(), seed: req.seed, elapsed_ms: started.elapsed().as_millis() as u64 }, bytes))
    }

    fn free(&mut self) {
        self.cache = None;
        self.sam = None;
    }

    /// SAM 3.1's sessions load again in about a second; the diffusion engine beside it in the
    /// worker needs the device more.
    fn yields(&self) -> bool {
        true
    }

    fn index(&mut self, dir: &str) -> Value {
        let installed: Vec<&str> = if self.has_model() { TEMPLATES.to_vec() } else { Vec::new() };
        json!({
            "dir": dir,
            "installed": installed,
            "backend": if candle_files(&self.models_root).is_some() { "candle" } else { "onnxruntime" },
            "modelsRoot": self.models_root.to_string_lossy(),
            "runtime": self.ort_dir().map(|p| p.to_string_lossy().into_owned()),
            "samPrecision": if self.half { "f16" } else { "f32" },
            "runtimeDirs": self.runtime_dirs.iter().map(|p| p.to_string_lossy().into_owned()).collect::<Vec<_>>(),
        })
    }
}

/// Parse the worker's own flags: `--runtime <dir;dir>` (repeatable), `--models <dir>`, `--cpu`.
/// `None` when no runtime or model root was given (the caller then serves its fallback).
pub fn from_args(args: &[String]) -> Option<PerceptionEngine> {
    let mut runtime: Vec<PathBuf> = Vec::new();
    let mut models: Option<PathBuf> = None;
    let mut cpu = false;
    let mut half = false;
    let mut i = 0;
    while i < args.len() {
        match args.get(i).map(String::as_str) {
            Some("--runtime") => {
                if let Some(v) = args.get(i + 1) {
                    runtime.extend(v.split(';').map(str::trim).filter(|s| !s.is_empty()).map(PathBuf::from));
                }
                i += 2;
            }
            Some("--models") => {
                models = args.get(i + 1).filter(|v| !v.trim().is_empty()).map(PathBuf::from);
                i += 2;
            }
            Some("--sam-precision") => {
                half = args.get(i + 1).is_some_and(|v| v.trim().eq_ignore_ascii_case("f16"));
                i += 2;
            }
            Some("--cpu") => {
                cpu = true;
                i += 1;
            }
            _ => i += 1,
        }
    }
    let models = models?;
    if runtime.is_empty() {
        return None;
    }
    Some(PerceptionEngine::new(runtime, models, cpu).with_half(half))
}

/// Is `dir` a usable model root (holds at least one model this engine serves)?
pub fn has_models(dir: &Path) -> bool {
    candle_files(dir).is_some() || onnx_layout(dir)
}

/// Are SAM 3.1's exported graphs under the model root (and can this build run them)?
#[cfg(feature = "onnx")]
fn onnx_layout(root: &Path) -> bool {
    Sam31::is_layout(&root.join(SAM31_DIR))
}

#[cfg(not(feature = "onnx"))]
fn onnx_layout(_: &Path) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn req(model: &str, params: BTreeMap<String, Value>) -> RunRequest {
        RunRequest {
            model: model.into(),
            task: "segment".into(),
            prompt: "fox".into(),
            negative: String::new(),
            seed: 0,
            steps: 0,
            guidance: 0.0,
            width: 0,
            height: 0,
            image: Some(Blob { width: 2, height: 2, channels: 4, index: 0 }),
            mask: None,
            references: Vec::new(),
            params,
        }
    }

    #[test]
    fn params_are_read_the_way_the_templates_send_them() {
        let mut p = BTreeMap::new();
        p.insert("points".into(), Value::String(r#"[{"x": 3, "y": 4}, {"x": 5.5, "y": 6}]"#.into()));
        p.insert("threshold".into(), json!(0.4));
        p.insert("bx".into(), json!(10));
        assert_eq!(points_param(p.get("points")), vec![[3.0, 4.0], [5.5, 6.0]]);
        assert_eq!(points_param(p.get("negatives")), Vec::<[f32; 2]>::new());
        assert_eq!(num(&p, "threshold"), Some(0.4));
        assert_eq!(num(&p, "bx"), Some(10.0));
        assert_eq!(route_of("sam3.1/segment-box"), Some(Route::Box));
        assert_eq!(route_of("krea2-turbo/image"), None);
    }

    #[test]
    fn missing_models_are_a_typed_error_not_a_crash() {
        let mut e = PerceptionEngine::new(vec![PathBuf::from("Z:/nowhere")], PathBuf::from("Z:/nowhere/models"), true);
        assert!(!e.health().ok);
        assert_eq!(e.index("")["installed"].as_array().map(Vec::len), Some(0));
        let r = e.run(&req("sam3.1/segment", BTreeMap::new()), &[vec![0; 16]], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::ModelMissing, .. })), "{r:?}");
        let r = e.run(&req("krea2-turbo/image", BTreeMap::new()), &[vec![0; 16]], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::ModelMissing, .. })));
        let mut other = req("sam3.1/segment", BTreeMap::new());
        other.task = "image".into();
        let r = e.run(&other, &[vec![0; 16]], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::Unsupported, .. })));
    }

    #[test]
    fn flags_make_an_engine_only_when_both_directories_are_given() {
        assert!(from_args(&["--models".into(), "m".into()]).is_none());
        assert!(from_args(&["--runtime".into(), "a;b".into()]).is_none());
        let e = from_args(&[
            "--token".into(),
            "t".into(),
            "--runtime".into(),
            "a;b".into(),
            "--runtime".into(),
            "c".into(),
            "--models".into(),
            "m".into(),
            "--cpu".into(),
        ]);
        let e = e.unwrap_or_else(|| panic!("an engine"));
        assert_eq!(e.runtime_dirs, vec![PathBuf::from("a"), PathBuf::from("b"), PathBuf::from("c")]);
        assert_eq!(e.device, Device::Cpu);
        assert!(!e.half);
        let half = from_args(&["--runtime".into(), "r".into(), "--models".into(), "m".into(), "--sam-precision".into(), "F16".into()]);
        assert!(half.is_some_and(|e| e.half));
    }

    #[test]
    fn content_hash_sees_size_and_bytes() {
        assert_ne!(content_hash(&[0; 4], 1, 1), content_hash(&[0; 4], 2, 2));
        assert_ne!(content_hash(&[0, 0, 0, 255], 1, 1), content_hash(&[0; 4], 1, 1));
        assert_eq!(content_hash(&[7; 8], 2, 1), content_hash(&[7; 8], 2, 1));
    }
}
