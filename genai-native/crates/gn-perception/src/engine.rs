//! The perception models behind `gn-engine`'s worker protocol: the `Engine` the editor's
//! `--engine` process serves once a runtime and a model root are configured.
//!
//! Requests arrive as the editor's `RunRequest` with task `segment`; the template id picks the
//! route (`…/segment` text, `…/segment-point` points, `…/segment-box` a box with optional
//! points) and `params` carry what PhotoshopEX's `select.*` commands send today: `threshold`,
//! `points` / `negatives` as JSON strings of `{x, y}` pixel coordinates in the request picture,
//! `bx by bw bh` as pixels. Every instance comes back as a 1-channel coverage blob.
//!
//! The model loads on the first request and stays resident; the last picture's
//! embedding is cached by content, so the probe that follows the pointer pays only the prompt.
//!
//! With the `candle` feature and SAM 3.1's own checkpoint on disk (ComfyUI's
//! `checkpoints/sam3.1_multiplex_fp16.safetensors`, or one in `sam3.1/`), `gn-segment` runs it
//! on candle (Rust rebuild R1). A build without `candle` answers every request ModelMissing.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use gn_engine::protocol::{Blob, DoneInfo, ErrorKind, Health, RunRequest};
use gn_engine::server::{Engine, EngineError};
use serde_json::{Value, json};

use crate::graph::{Device, Instance, Options};

/// The template ids this engine answers, as PhotoshopEX names them.
pub const TEMPLATES: [&str; 3] = ["sam3.1/segment", "sam3.1/segment-point", "sam3.1/segment-box"];
/// ViTMatte's template: a picture and a trimap (the mask blob: 0 background, 128 unknown,
/// 255 foreground) to an RGBA image whose alpha is the matte.
pub const MATTE_TEMPLATE: &str = "vitmatte/refine";
/// The model root's subdirectory for ViTMatte (transformers' `model.safetensors` + `config.json`).
pub const VITMATTE_DIR: &str = "vitmatte";
/// The model root's subdirectory for SAM 3.1.
pub const SAM31_DIR: &str = "sam3.1";
/// SAM 3.1's checkpoint as ComfyUI names it.
pub const SAM31_CHECKPOINT: &str = "sam3.1_multiplex_fp16.safetensors";

/// The resident model: SAM 3.1 on candle.
enum Model {
    #[cfg(feature = "candle")]
    Candle(Box<gn_segment::sam::Sam31>),
    /// A build without `candle` never opens a model (every request is ModelMissing).
    #[cfg(not(feature = "candle"))]
    #[allow(dead_code)]
    None(std::convert::Infallible),
}

/// A picture's embedding.
enum Emb {
    #[cfg(feature = "candle")]
    Candle(Box<gn_segment::sam::Frame>),
    #[cfg(not(feature = "candle"))]
    #[allow(dead_code)]
    None(std::convert::Infallible),
}

/// The candle backend's files under a models root: the checkpoint and the CLIP tokenizer
/// (`sam3.1/tokenizer.json`, else ComfyUI's `comfy/sd1_tokenizer` beside a ComfyUI `models`
/// folder).
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
    /// The native runtime folders (the CUDA libraries the worker finds on its `PATH`), as given.
    runtime_dirs: Vec<PathBuf>,
    /// The directory holding one subdirectory per model (`sam3.1/`), or a ComfyUI `models` folder.
    models_root: PathBuf,
    device: Device,
    /// SAM 3.1 on candle in f16 on CUDA (`--sam-precision f16`): 1.9x the speed of f32, every
    /// mask at IoU >= 0.993 against ComfyUI's own f32 node (gn-segment F16.md). f32 otherwise.
    half: bool,
    sam: Option<Model>,
    /// SAM 3.1's checkpoint as read (1.6 GB of host memory), kept when the model gives the
    /// device back: reading the file was 0.8 s of the 0.7–1.1 s reload every Remove stroke after a
    /// diffusion run paid (2026-11-01).
    #[cfg(feature = "candle")]
    checkpoint: Option<gn_segment::weights::Checkpoint>,
    /// The last picture embedded: a content hash and its embedding.
    cache: Option<(u64, Emb)>,
    last_error: Option<String>,
    #[cfg(feature = "candle")]
    vitmatte: Option<gn_segment::vitmatte::VitMatte>,
}

impl PerceptionEngine {
    pub fn new(runtime_dirs: Vec<PathBuf>, models_root: PathBuf, cpu: bool) -> Self {
        let device = if cpu {
            Device::Cpu
        } else if cfg!(feature = "metal") {
            Device::Metal
        } else {
            Device::Cuda { ordinal: 0 }
        };
        PerceptionEngine {
            runtime_dirs,
            models_root,
            device,
            half: false,
            sam: None,
            #[cfg(feature = "candle")]
            checkpoint: None,
            cache: None,
            last_error: None,
            #[cfg(feature = "candle")]
            vitmatte: None,
        }
    }

    /// Is ViTMatte on disk for a build that can run it?
    fn has_matte(&self) -> bool {
        cfg!(feature = "candle") && self.models_root.join(VITMATTE_DIR).join("model.safetensors").is_file()
    }

    /// `vitmatte/refine`: the alpha for the picture inside the trimap's unknown band.
    #[cfg_attr(not(feature = "candle"), allow(unused_variables))]
    #[allow(clippy::needless_return)] // one return per build: the candle one, else the error
    fn run_matte(&mut self, req: &RunRequest, blobs: &[Vec<u8>], progress: &mut dyn FnMut(f32, &str)) -> Result<(DoneInfo, Vec<Vec<u8>>), EngineError> {
        let started = Instant::now();
        let (Some(image), Some(mask)) = (req.image.as_ref(), req.mask.as_ref()) else {
            return Err(EngineError::new(ErrorKind::BadRequest, "a matte needs a picture and a trimap"));
        };
        if image.channels != 4 || mask.channels != 1 || (image.width, image.height) != (mask.width, mask.height) {
            return Err(EngineError::new(ErrorKind::BadRequest, "the picture must be RGBA and the trimap one channel of the same size"));
        }
        let (Some(rgba), Some(tri)) = (blobs.get(image.index as usize), blobs.get(mask.index as usize)) else {
            return Err(EngineError::new(ErrorKind::BadRequest, "the picture's or trimap's blob is missing"));
        };
        let (w, h) = (image.width as usize, image.height as usize);
        #[cfg(feature = "candle")]
        {
            if self.vitmatte.is_none() {
                progress(0.0, "Loading ViTMatte");
                let device = self.candle_device()?;
                self.vitmatte = Some(
                    gn_segment::vitmatte::VitMatte::load(self.models_root.join(VITMATTE_DIR), &device)
                        .map_err(|e| EngineError::new(ErrorKind::ModelMissing, format!("{e:#}")))?,
                );
            }
            let Some(m) = self.vitmatte.as_ref() else { return Err(EngineError::new(ErrorKind::Internal, "no model")) };
            progress(0.3, "Matting");
            let alpha = m.matte(rgba, tri, w, h).map_err(candle_error)?;
            let mut out = rgba.clone();
            for (px, a) in out.as_chunks_mut::<4>().0.iter_mut().zip(&alpha) {
                px[3] = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            progress(1.0, "Done");
            return Ok((
                DoneInfo {
                    images: vec![Blob { width: image.width, height: image.height, channels: 4, index: 0 }],
                    texts: Vec::new(),
                    seed: req.seed,
                    elapsed_ms: started.elapsed().as_millis() as u64,
                },
                vec![out],
            ));
        }
        #[cfg(not(feature = "candle"))]
        Err(EngineError::new(ErrorKind::ModelMissing, "this build has no ViTMatte backend (built without `candle`)"))
    }

    /// The candle device for `self.device`: Metal falls back to the CPU when it cannot open.
    #[cfg(feature = "candle")]
    fn candle_device(&self) -> Result<candle_core::Device, EngineError> {
        Ok(match self.device {
            Device::Cuda { ordinal } => candle_core::Device::new_cuda(ordinal as usize).map_err(|e| candle_error(e.into()))?,
            Device::Metal => candle_core::Device::new_metal(0).unwrap_or_else(|e| {
                eprintln!("gn-perception: Metal did not open ({e}); running on the CPU");
                candle_core::Device::Cpu
            }),
            Device::Cpu => candle_core::Device::Cpu,
        })
    }

    /// SAM 3.1's compute precision on CUDA: f16 when `half`, else f32 (the CPU is always f32).
    pub fn with_half(mut self, half: bool) -> Self {
        self.half = half;
        self
    }

    /// Is SAM 3.1's checkpoint (and a tokenizer) on disk for a build that can run it?
    fn has_model(&self) -> bool {
        has_models(&self.models_root)
    }

    /// Open SAM 3.1 on candle if it is not open yet.
    #[cfg_attr(not(feature = "candle"), allow(unused_variables))]
    fn ensure_sam(&mut self, progress: &mut dyn FnMut(f32, &str)) -> Result<(), EngineError> {
        #[cfg(feature = "candle")]
        if let Some((checkpoint, tokenizer)) = candle_files(&self.models_root).filter(|_| self.sam.is_none()) {
            progress(0.0, "Loading SAM 3.1");
            let device = self.candle_device()?;
            let t0 = Instant::now();
            let dtype = if self.half && device.is_cuda() { candle_core::DType::F16 } else { candle_core::DType::F32 };
            if self.checkpoint.is_none() {
                self.checkpoint = Some(gn_segment::weights::Checkpoint::open(&checkpoint).and_then(|c| c.with_compute_dtype(dtype)).map_err(candle_error)?);
            }
            let ck = self.checkpoint.as_ref().ok_or_else(|| EngineError::new(ErrorKind::Internal, "no SAM 3.1 checkpoint"))?;
            let mut sam = gn_segment::sam::Sam31::from_checkpoint(ck, &device).map_err(candle_error)?;
            sam.set_tokenizer(&tokenizer).map_err(candle_error)?;
            eprintln!("gn-perception: SAM 3.1 on candle in {dtype:?} ({}) loaded in {:.1} s", checkpoint.display(), t0.elapsed().as_secs_f32());
            self.sam = Some(Model::Candle(Box::new(sam)));
        }
        if self.sam.is_some() {
            return Ok(());
        }
        #[cfg(feature = "candle")]
        let why = format!("no SAM 3.1 in {}: ComfyUI's checkpoints/{SAM31_CHECKPOINT} is missing", self.models_root.display());
        #[cfg(not(feature = "candle"))]
        let why = String::from("this build has no SAM 3.1 backend (built without `candle`)");
        Err(EngineError::new(ErrorKind::ModelMissing, why))
    }
}

#[cfg(feature = "candle")]
fn candle_error(e: anyhow::Error) -> EngineError {
    EngineError::new(ErrorKind::ModelUnreadable, format!("{e:#}"))
}

#[cfg(feature = "candle")]
fn candle_opts(o: Options) -> gn_segment::sam::Options {
    gn_segment::sam::Options {
        threshold: o.threshold,
        max_detections: o.max_detections,
        refine_iterations: o.refine_iterations,
        soft: o.soft,
        candidate: o.candidate,
    }
}

#[cfg(feature = "candle")]
fn from_candle(i: gn_segment::sam::Instance) -> Instance {
    Instance { score: i.score, bbox: i.bbox, mask: i.mask, bounds: i.bounds, pixels: i.pixels }
}

/// The three routes on the resident model.
#[cfg_attr(not(feature = "candle"), allow(unused_variables))]
impl Model {
    fn embed(&mut self, rgba: &[u8], w: usize, h: usize) -> Result<Emb, EngineError> {
        match self {
            #[cfg(feature = "candle")]
            Model::Candle(s) => s.encode(rgba, w, h).map(|f| Emb::Candle(Box::new(f))).map_err(candle_error),
            #[cfg(not(feature = "candle"))]
            Model::None(n) => match *n {},
        }
    }

    fn detect_text(&mut self, emb: &Emb, prompt: &str, boxes: &[[f32; 4]], opts: Options) -> Result<Vec<Instance>, EngineError> {
        match (self, emb) {
            #[cfg(feature = "candle")]
            (Model::Candle(s), Emb::Candle(f)) => {
                s.detect_text(f, prompt, boxes, candle_opts(opts)).map(|r| r.0.into_iter().map(from_candle).collect()).map_err(candle_error)
            }
            #[cfg(not(feature = "candle"))]
            (Model::None(n), _) => match *n {},
        }
    }

    fn segment_points(&mut self, emb: &Emb, positives: &[[f32; 2]], negatives: &[[f32; 2]], opts: Options) -> Result<Instance, EngineError> {
        match (self, emb) {
            #[cfg(feature = "candle")]
            (Model::Candle(s), Emb::Candle(f)) => s.segment_points(f, positives, negatives, candle_opts(opts)).map(|r| from_candle(r.0)).map_err(candle_error),
            #[cfg(not(feature = "candle"))]
            (Model::None(n), _) => match *n {},
        }
    }

    fn segment_box(&mut self, emb: &Emb, bbox: [f32; 4], opts: Options) -> Result<Instance, EngineError> {
        match (self, emb) {
            #[cfg(feature = "candle")]
            (Model::Candle(s), Emb::Candle(f)) => s.segment_box(f, bbox, candle_opts(opts)).map(|r| from_candle(r.0)).map_err(candle_error),
            #[cfg(not(feature = "candle"))]
            (Model::None(n), _) => match *n {},
        }
    }
}

/// Text-detection refinement passes when the request names none: 2 (ComfyUI's), except on Metal,
/// where each pass re-runs the vision trunk on a crop (about 10 s on a 16 GB M1). There a picture
/// no larger than 1.5x SAM's 1008 px input gets none: on a 512² picture the masks without it were
/// at IoU >= 0.998 against two passes, the prompt 6.5 s instead of 25–35 s (2026-11-07). Larger
/// pictures keep one pass for edge detail at their own resolution.
fn default_refine(device: &Device, w: u32, h: u32) -> usize {
    match device {
        Device::Metal if w.max(h) <= 1512 => 0,
        Device::Metal => 1,
        _ => 2,
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
    // Binary masks come out 0 or 255 as before; soft ones keep their coverage.
    let bytes: Vec<u8> = inst.mask.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
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
        let ready = self.has_model();
        Health {
            ok: ready && self.last_error.is_none(),
            engine: "perception".into(),
            device: match self.device {
                Device::Cuda { ordinal } => format!("cuda:{ordinal}"),
                Device::Cpu => "cpu".into(),
                Device::Metal => "metal".into(),
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
        if req.model == MATTE_TEMPLATE {
            if !self.has_matte() {
                return Err(EngineError::new(ErrorKind::ModelMissing, format!("no ViTMatte in {}", self.models_root.join(VITMATTE_DIR).display())));
            }
            return self.run_matte(req, blobs, progress);
        }
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
            refine_iterations: num(&req.params, "refineIterations").map_or_else(|| default_refine(&self.device, w, h), |r| r.max(0.0) as usize),
            soft: num(&req.params, "softEdge").map_or(0.0, |t| t.clamp(0.0, 64.0)),
            candidate: num(&req.params, "candidate").filter(|c| *c >= 0.0).map(|c| c as usize),
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
        #[cfg(feature = "candle")]
        {
            self.vitmatte = None;
        }
    }

    /// SAM 3.1 loads again in seconds; the diffusion engine beside it in the
    /// worker needs the device more.
    fn yields(&self) -> bool {
        true
    }

    fn index(&mut self, dir: &str) -> Value {
        let mut installed: Vec<&str> = if self.has_model() { TEMPLATES.to_vec() } else { Vec::new() };
        if self.has_matte() {
            installed.push(MATTE_TEMPLATE);
        }
        json!({
            "dir": dir,
            "installed": installed,
            "backend": if cfg!(feature = "candle") { "candle" } else { "none" },
            "modelsRoot": self.models_root.to_string_lossy(),
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
    // `GN_PERCEPTION_CPU=1`: the CPU whatever the build has (to compare with the GPU).
    if std::env::var("GN_PERCEPTION_CPU").is_ok_and(|v| v.trim() == "1") {
        cpu = true;
    }
    let models = models?;
    // A Mac needs no runtime folder (no CUDA): candle runs on Metal or the CPU.
    if runtime.is_empty() && !cfg!(target_os = "macos") {
        return None;
    }
    Some(PerceptionEngine::new(runtime, models, cpu).with_half(half))
}

/// Is `dir` a usable model root (holds at least one model this engine serves)?
pub fn has_models(dir: &Path) -> bool {
    candle_files(dir).is_some()
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
            preview: Default::default(),
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
        // A Mac needs no runtime folder (Metal is part of the system).
        assert_eq!(from_args(&["--models".into(), "m".into()]).is_some(), cfg!(target_os = "macos"));
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
    fn metal_refines_only_large_pictures() {
        assert_eq!(default_refine(&Device::Metal, 512, 512), 0);
        assert_eq!(default_refine(&Device::Metal, 4000, 3000), 1);
        assert_eq!(default_refine(&Device::Cpu, 512, 512), 2);
    }

    #[test]
    fn content_hash_sees_size_and_bytes() {
        assert_ne!(content_hash(&[0; 4], 1, 1), content_hash(&[0; 4], 2, 2));
        assert_ne!(content_hash(&[0, 0, 0, 255], 1, 1), content_hash(&[0; 4], 1, 1));
        assert_eq!(content_hash(&[7; 8], 2, 1), content_hash(&[7; 8], 2, 1));
    }
}
