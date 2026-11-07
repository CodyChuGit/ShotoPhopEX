//! The diffusion models behind `gn-engine`'s worker protocol (plan Phases 2–4): Qwen-Image-2.1
//! and Krea 2 Turbo through `qwen-image-rs`'s resident pipelines. The editor's `RunRequest`
//! with template `qwen-2.1/image` (task `image`) makes a picture from text, `qwen-2.1/edit`
//! (task `edit`) re-renders the picture it carries by instruction with its reference pictures
//! in context, `qwen-2.1/fill` (task `fill`) does the same with the mask as the second
//! picture, `qwen-2.1/expand` places the picture inside the larger canvas, `qwen-2.1/matte`
//! answers RGBA whose alpha is the cut-out, the way the app's ComfyUI graphs do;
//! `krea2-turbo/image` makes a picture with Krea 2 Turbo. All with ComfyUI's conventions (its
//! noise for the seed, its `simple` schedule over each model's sigma table, its reference
//! sizes), so the two routes agree picture for picture. `moss-image/image` (and
//! `moss-image/image-base`) make a picture with MOSS-Image on the official pipeline's
//! conventions instead (`qir::moss`).
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

use gn_engine::protocol::{Blob, DoneInfo, ErrorKind, Health, Preview, RunRequest};
use gn_engine::server::{Engine, EngineError};
use qwen_image_rs::convrot_cache::CacheSpec;
use qwen_image_rs::edit2511_pipeline::{self, Edit2511Pipeline, Edit2511Request};
use qwen_image_rs::krea2_pipeline::{self, Krea2Options, Krea2Pipeline};
use qwen_image_rs::layered_pipeline::LayeredPipeline;
use qwen_image_rs::lora::{LoraFile, LoraStack};
use qwen_image_rs::moss::pipeline::{self as moss, MossPipeline, Variant as MossVariant};
use qwen_image_rs::pipeline::{Cancelled, ChatOptions, Options, Picture, Pipeline, RefImage, Request, Schedule, is_snapshot};
use qwen_image_rs::preview::{Frame, LatentSpace, PreviewMode, Previewer, StepPreview, TINY_DIR, TinyDecoder};
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
/// Generative Expand with AusBoss's Qwen-Image-2.1 Outpaint LoRA (`ausboss/Qwen-Image-2.1-
/// Outpaint-LoRA` v2, Qwen Research licence, BYOM): image 1 is the whole grown canvas with the
/// new area flat `#808080` and the picture where it stays, the target takes its size (an edit
/// at ComfyUI's `resolution 0`), no noise mask; the LoRA keeps the picture in place (its card:
/// 32 of 32 renders against 3 of 16 for plain 2.1) and paints the grey. Served when that file
/// is under the model root's `loras/`.
pub const TEMPLATE_OUTPAINT: &str = "qwen-2.1/outpaint";
/// The LoRA [`TEMPLATE_OUTPAINT`] runs with (added to a request's own LoRAs).
pub const QWEN21_OUTPAINT_LORA: &str = "qwen-image-2.1-outpaint-v2.safetensors";
/// [`TEMPLATE_OUTPAINT`] in six steps with Viggle's Turbo LoRA (`Viggle/Qwen-Image-2.1-viggle-turbo`
/// v0.3, Qwen Research licence, BYOM) stacked under the Outpaint LoRA, on its own sigmas
/// ([`Schedule::ViggleTurbo`]); served when both files are under `loras/`.
pub const TEMPLATE_OUTPAINT_TURBO: &str = "qwen-2.1/outpaint-turbo";
/// The Turbo LoRA [`TEMPLATE_OUTPAINT_TURBO`] adds (unmerged, strength 1, as its card asks).
pub const QWEN21_TURBO_LORA: &str = "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors";
/// The matte template: the picture alone, the answer's alpha is the cut-out (the 2.1 VAE
/// decodes RGBA).
pub const TEMPLATE_MATTE: &str = "qwen-2.1/matte";
/// A new picture at the asked size from one to nine reference pictures (2.1's text to image
/// with the references in its encoder's view): PhotoshopEX's Generate Image with references.
pub const TEMPLATE_IMAGE_REF: &str = "qwen-2.1/image-ref";
/// Krea 2 Turbo text to image.
pub const TEMPLATE_KREA2: &str = "krea2-turbo/image";
/// Krea 2 Turbo generation in the style of one or two reference pictures.
pub const TEMPLATE_KREA2_STYLE: &str = "krea2-turbo/image-style";
/// Ostris's style adapter, required for the style template.
pub const KREA2_STYLE_LORA: &str = "krea2_style_reference.safetensors";
/// Krea 2 Turbo editing a picture by instruction: the picture as in-context tokens and in the
/// encoder's view, with the community krea2_edit General Edit V2 LoRA (conradlocke; personal
/// and non-commercial use, BYOM). Served when that file is under the model root's `loras/`.
pub const TEMPLATE_KREA2_EDIT: &str = "krea2-turbo/edit";
/// The LoRA [`TEMPLATE_KREA2_EDIT`] runs with (added to a request's own LoRAs).
pub const KREA2_EDIT_LORA: &str = "krea2_general_edit_v2.safetensors";
/// Krea 2 Turbo inpainting with the AnyPaint LoRA (yijunwang2/krea2-anypaint; Krea 2
/// Community License, BYOM): the picture with the area flattened to its median colour as a
/// reference registered over the whole canvas and in the encoder's view, the known pixels kept
/// by the sampler's noise mask (AusBoss's "Krea 2 Inpaint Masked" workflow). Served when that
/// file is under the model root's `loras/`.
pub const TEMPLATE_KREA2_FILL: &str = "krea2-turbo/fill";
/// The LoRA [`TEMPLATE_KREA2_FILL`] runs with (added to a request's own LoRAs).
pub const KREA2_ANYPAINT_LORA: &str = "krea2_anypaint_rank32.safetensors";
/// A chat's `params.encoder` that answers on Krea 2's Qwen3-VL-4B (the enhance templates): what
/// AusBoss's Krea 2 inpaint workflow writes its prompt with.
pub const CHAT_ON_KREA2: &str = "krea2";
/// Its card's `ref_boost` for edits at 8 steps and CFG 1; a request's `ref_boost` overrides it.
pub const KREA2_REF_BOOST: f32 = 4.0;
/// The app's prompt rewriter and judge with the picture in view, and without: answered by the
/// 2.1 text encoder as the vision-language model it is (Qwen3-VL 8B Instruct with its
/// language-model head), so they need no second model on the card.
pub const TEMPLATE_CHAT_EDIT: &str = "enhance/edit";
pub const TEMPLATE_CHAT_IMAGE: &str = "enhance/image";
/// Split into Layers: Qwen-Image-Layered (Apache-2.0) from ComfyUI's files (fp8mixed DiT, the
/// Qwen2.5-VL 7B encoder, the RGBA layered VAE), the picture in, one RGBA layer per slice out,
/// background first (`qir::layered_pipeline`, Rust rebuild R3).
pub const TEMPLATE_LAYERS: &str = "qwen-layered/split";
/// Its template's defaults: 20 steps, 3 layers.
pub const LAYERS_STEPS: u32 = 20;
pub const DEFAULT_LAYERS: u64 = 3;
/// MOSS-Image (OpenMOSS, Apache-2.0) text to image from Comfy-Org's files: the SFT release, and
/// the Base one (`qir::moss`).
pub const TEMPLATE_MOSS: &str = "moss-image/image";
pub const TEMPLATE_MOSS_BASE: &str = "moss-image/image-base";
/// The official pipeline's defaults: 50 steps at CFG 3.
pub const MOSS_STEPS: u32 = 50;
pub const MOSS_GUIDANCE: f32 = 3.0;
/// The low-VRAM turbo: lodestones' Kromoss (Apache-2.0, a full fine-tune of MOSS-Image-Base)
/// with its 8-step DMD2 LoRA, at the card's 8 steps and CFG 1 (no negative pass).
pub const TEMPLATE_MOSS_TURBO: &str = "moss-image/turbo";
pub const MOSS_TURBO_LORA: &str = "kromoss-8-steps-dmd2-lora.safetensors";
pub const MOSS_TURBO_STEPS: u32 = 8;
/// akhaliq's MOSS-Image-SFT 4-step LoRA (Apache-2.0; stock Euler, 4 steps, CFG 1): MOSS Turbo
/// on MLX when it is installed.
pub const MOSS_SFT_4STEP_LORA: &str = "moss_image_sft_4step_v2.0.safetensors";
/// MOSS Edit (chinoll/Moss-Image-Edit, a rank-64 LoRA on MOSS-Image-SFT; its code GPLv3, the
/// weights unlicensed): the picture as reference tokens at t = 0. 30 steps at CFG 4 (its
/// workflow's), and a turbo with Kromoss' 8-step LoRA stacked on it at CFG 1.
pub const TEMPLATE_MOSS_EDIT: &str = "moss-image/edit";
pub const TEMPLATE_MOSS_EDIT_TURBO: &str = "moss-image/edit-turbo";
/// The adapter as its repo names it, in its own folder; or renamed to a file of its own.
pub const MOSS_EDIT_LORAS: [&str; 2] = ["moss-image-edit/adapter_model.safetensors", "moss_image_edit_lora.safetensors"];
pub const MOSS_EDIT_STEPS: u32 = 30;
/// FLUX 3 Image Klein on MLX (macOS; mflux-community's pre-quantized files): text to image
/// (the Mac's quality pick next to MOSS Turbo), an edit by
/// instruction (and, with AusBoss's outpaint LoRA in the request, the editor's Expand), and a
/// fill that paints only inside the mask.
pub const TEMPLATE_KLEIN_IMAGE: &str = "flux3-klein/image";
pub const TEMPLATE_KLEIN_EDIT: &str = "flux3-klein/edit";
pub const TEMPLATE_KLEIN_FILL: &str = "flux3-klein/fill";
/// Klein's folders under the model root, the first found wins (q4 first: on an M1 it is as fast
/// as q8 and half the memory).
pub const KLEIN_DIRS: [&str; 2] = ["mlx/flux-3-image-klein-q4", "mlx/flux-3-image-klein-q8"];
/// The FLUX.2 VAE Klein decodes with (MOSS's file).
pub const KLEIN_VAE: &str = "vae/flux2-vae_small.safetensors";
pub const MOSS_EDIT_GUIDANCE: f32 = 4.0;

/// The model families served, one resident at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    QwenImage21,
    Krea2,
    Layered,
    Edit2511,
    Moss(MossVariant),
    /// FLUX 3 Image Klein, MLX only.
    Klein,
}

/// Every template served, with the task each one answers and the family that serves it.
pub const TEMPLATES: [(&str, &str, Family); 36] = [
    (TEMPLATE_KLEIN_IMAGE, "image", Family::Klein),
    (TEMPLATE_KLEIN_EDIT, "edit", Family::Klein),
    (TEMPLATE_KLEIN_FILL, "fill", Family::Klein),
    (TEMPLATE_MOSS_EDIT, "edit", Family::Moss(MossVariant::Sft)),
    (TEMPLATE_MOSS_EDIT_TURBO, "edit", Family::Moss(MossVariant::Sft)),
    (TEMPLATE_MOSS_TURBO, "image", Family::Moss(MossVariant::Kromoss)),
    (TEMPLATE_OUTPAINT_TURBO, "expand", Family::QwenImage21),
    (TEMPLATE_OUTPAINT, "expand", Family::QwenImage21),
    (TEMPLATE_KREA2_FILL, "fill", Family::Krea2),
    (TEMPLATE_MOSS, "image", Family::Moss(MossVariant::Sft)),
    (TEMPLATE_MOSS_BASE, "image", Family::Moss(MossVariant::Base)),
    (TEMPLATE_KREA2_EDIT, "edit", Family::Krea2),
    (TEMPLATE_IMAGE_REF, "image", Family::QwenImage21),
    (TEMPLATE, "image", Family::QwenImage21),
    (TEMPLATE_EDIT, "edit", Family::QwenImage21),
    (TEMPLATE_FILL, "fill", Family::QwenImage21),
    (TEMPLATE_EXPAND, "expand", Family::QwenImage21),
    (TEMPLATE_MATTE, "matte", Family::QwenImage21),
    (TEMPLATE_KREA2, "image", Family::Krea2),
    (TEMPLATE_KREA2_STYLE, "image", Family::Krea2),
    (TEMPLATE_CHAT_EDIT, "text", Family::QwenImage21),
    (TEMPLATE_CHAT_IMAGE, "text", Family::QwenImage21),
    (TEMPLATE_LAYERS, "layers", Family::Layered),
    ("qwen-edit-2511/edit", "edit", Family::Edit2511),
    ("qwen-edit-2511/edit-lightning-8", "edit", Family::Edit2511),
    ("qwen-edit-2511/fill", "fill", Family::Edit2511),
    ("qwen-edit-2511/fill-lightning-4", "fill", Family::Edit2511),
    ("qwen-edit-2511/fill-lightning-8", "fill", Family::Edit2511),
    ("qwen-edit-2511/fill-guided", "fill", Family::Edit2511),
    ("qwen-edit-2511/fill-ref", "fill", Family::Edit2511),
    ("qwen-edit-2511/fill-ref-lightning-8", "fill", Family::Edit2511),
    ("qwen-edit-2511/image-ref", "image", Family::Edit2511),
    ("qwen-edit-2511/expand", "expand", Family::Edit2511),
    ("qwen-edit-2511/expand-lightning-8", "expand", Family::Edit2511),
    ("qwen-edit-2511/remove-lightning-8", "fill", Family::Edit2511),
    ("qwen-edit-2511/angles", "edit", Family::Edit2511),
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

/// What is on the device. One per engine, so the variants' sizes do not matter.
#[allow(clippy::large_enum_variant)]
enum Resident {
    None,
    Qwen(Pipeline),
    Krea2(Krea2Pipeline),
    Layered(LayeredPipeline),
    Edit2511(Edit2511Pipeline),
    #[cfg_attr(feature = "mlx", allow(dead_code))]
    Moss(MossPipeline, MossVariant),
    /// MOSS-Image on MLX (macOS): every variant through one pipeline, its DiT swapped per
    /// request.
    #[cfg(feature = "mlx")]
    Mlx(Box<gn_mlx::pipeline::MossMlx>),
    /// FLUX 3 Image Klein on MLX (macOS).
    #[cfg(feature = "mlx")]
    Klein(Box<gn_mlx::klein::KleinMlx>),
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
    /// The tiny decoder Sharp previews use, loaded on the first Sharp run of its latent space.
    tiny: Option<TinyDecoder>,
}

/// The protocol's preview mode as the pipelines know it.
fn preview_mode(m: gn_engine::protocol::PreviewMode) -> PreviewMode {
    match m {
        gn_engine::protocol::PreviewMode::Off => PreviewMode::Off,
        gn_engine::protocol::PreviewMode::Fast => PreviewMode::Fast,
        gn_engine::protocol::PreviewMode::Sharp => PreviewMode::Sharp,
    }
}

/// The latent space a family samples in: Qwen-Image-2.1's own 64-channel VAE, the Wan 2.1
/// family's (Krea 2, Qwen-Image-Edit-2511), the Layered RGBA VAE's, or FLUX.2's (MOSS-Image).
fn latent_space(f: Family) -> LatentSpace {
    match f {
        Family::QwenImage21 => LatentSpace::QwenImage21,
        Family::Krea2 | Family::Edit2511 => LatentSpace::Wan21,
        Family::Layered => LatentSpace::Layered,
        Family::Moss(_) | Family::Klein => LatentSpace::Flux2,
    }
}

fn engine_preview(f: Frame) -> Preview {
    let n = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
    Preview { step: n(f.step), steps: n(f.steps), output: n(f.output), width: n(f.width), height: n(f.height), rgba: f.rgba }
}

/// One line in the worker's log per previewed run: how many pictures and what they cost.
fn log_previews(mode: PreviewMode, viewer: &Previewer<'_>) {
    if mode != PreviewMode::Off {
        eprintln!("gn-diffusion: previews {mode:?}: {} pictures in {:.1} ms", viewer.shown, viewer.spent.as_secs_f64() * 1000.0);
    }
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

/// A request's params with [`KREA2_EDIT_LORA`] at strength 1 in front of its own `loras`,
/// unless the request already names that file (at its own strength).
fn with_edit_lora(params: &std::collections::BTreeMap<String, Value>) -> std::collections::BTreeMap<String, Value> {
    with_lora_first(params, KREA2_EDIT_LORA)
}

/// A request's params with `file` at strength 1 in front of its own `loras`, unless the
/// request already names that file (at its own strength).
fn with_lora_first(params: &std::collections::BTreeMap<String, Value>, file: &str) -> std::collections::BTreeMap<String, Value> {
    let mut out = params.clone();
    let mut list = params.get("loras").and_then(Value::as_array).cloned().unwrap_or_default();
    let named = list.iter().any(|l| l.get("file").and_then(Value::as_str).is_some_and(|f| Path::new(f.trim()).file_name().is_some_and(|n| n == file)));
    if !named {
        list.insert(0, json!({"file": file, "strength": 1.0}));
    }
    out.insert("loras".into(), Value::Array(list));
    out
}

/// The template model slot is always applied at strength one before user adapters.
fn with_style_lora(params: &std::collections::BTreeMap<String, Value>) -> Result<std::collections::BTreeMap<String, Value>, EngineError> {
    let slot = params.get("models").and_then(|v| v.get("lora")).or_else(|| params.get("lora"));
    let file = match slot {
        Some(v) => v
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| EngineError::new(ErrorKind::BadRequest, "the lora model slot needs a file name"))?,
        None => KREA2_STYLE_LORA,
    };
    let mut list = lora_asks(params)?;
    list.retain(|(name, _)| name != file);
    list.insert(0, (file.to_string(), 1.0));
    let mut out = params.clone();
    out.insert("loras".into(), Value::Array(list.into_iter().map(|(file, strength)| json!({"file":file,"strength":strength})).collect()));
    Ok(out)
}

/// Whether a chat's `encoder` param asks for Krea 2's encoder ([`CHAT_ON_KREA2`]); absent or
/// null is the default (Qwen-Image-2.1's 8B), anything else an error.
fn chat_on_krea2(params: &std::collections::BTreeMap<String, Value>) -> Result<bool, EngineError> {
    match params.get("encoder") {
        None | Some(Value::Null) => Ok(false),
        Some(v) if v.as_str() == Some(CHAT_ON_KREA2) => Ok(true),
        Some(v) => Err(EngineError::new(ErrorKind::BadRequest, format!("encoder {v}: a chat answers on the default encoder or \"{CHAT_ON_KREA2}\""))),
    }
}

/// A Krea 2 fill's `denoise` (0 < d ≤ 1, default 1) and `seam` (the band around the area the
/// sampler may repaint, 0..256 px, default AnyPaint's 32).
fn krea2_fill_options(params: &std::collections::BTreeMap<String, Value>) -> Result<krea2_pipeline::FillOptions, EngineError> {
    let mut o = krea2_pipeline::FillOptions::default();
    if let Some(v) = params.get("denoise") {
        match v.as_f64() {
            Some(d) if d > 0.0 && d <= 1.0 => o.denoise = d as f32,
            _ => return Err(EngineError::new(ErrorKind::BadRequest, format!("denoise {v} is outside 0..1"))),
        }
    }
    if let Some(v) = params.get("seam") {
        match v.as_u64() {
            Some(px) if px <= 256 => o.seam_px = px as u32,
            _ => return Err(EngineError::new(ErrorKind::BadRequest, format!("seam {v} is outside 0..256 pixels"))),
        }
    }
    Ok(o)
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
        Self { models_root, quant_text, resident: Resident::None, last_error: None, loras: Vec::new(), gpu, refused, tiny: None }
    }

    /// For a Sharp preview: `space`'s tiny decoder on the resident pipeline's device, loaded
    /// once from the model root's `vae_approx/`. Without the file (or when it does not load)
    /// Sharp shows Fast; the run goes on either way.
    fn ensure_tiny(&mut self, mode: PreviewMode, space: LatentSpace) {
        if mode != PreviewMode::Sharp {
            return;
        }
        let dev = match &self.resident {
            Resident::Qwen(p) => p.device(),
            Resident::Krea2(p) => p.device(),
            Resident::Layered(p) => p.device(),
            Resident::Edit2511(p) => p.device(),
            Resident::Moss(p, _) => p.device(),
            #[cfg(feature = "mlx")]
            Resident::Mlx(_) | Resident::Klein(_) => return,
            Resident::None => return,
        };
        if self.tiny.as_ref().is_some_and(|t| t.space() == space && t.on(dev)) {
            return;
        }
        // A space no tiny decoder decodes (Layered) shows Fast without a word.
        let Some(name) = space.tiny_file() else { return };
        self.tiny = None;
        let Some(file) = TinyDecoder::file(&self.models_root, space) else {
            eprintln!("gn-diffusion: no {name} under {}; Sharp previews show Fast", self.models_root.join(TINY_DIR).display());
            return;
        };
        match TinyDecoder::load(&file, space, dev) {
            Ok(t) => self.tiny = Some(t),
            Err(e) => eprintln!("gn-diffusion: {} did not load ({e}); Sharp previews show Fast", file.display()),
        }
    }

    /// Where a LoRA named by a request lives: an absolute path as given, a name under the model
    /// root's `loras/`. A name must stay a plain file name (no folders) so a request cannot read
    /// outside the root.
    /// A template's own LoRA `name` under the root's `loras/`, or up to two subfolders below it
    /// (`loras/moss-image/…`, as a junctioned model folder puts it), first found in sorted order.
    fn template_lora(&self, name: &str) -> Option<PathBuf> {
        fn walk(dir: &Path, name: &str, depth: usize) -> Option<PathBuf> {
            let here = dir.join(name);
            if here.is_file() {
                return Some(here);
            }
            if depth == 0 {
                return None;
            }
            let mut subs: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect();
            subs.sort();
            subs.iter().find_map(|d| walk(d, name, depth - 1))
        }
        walk(&self.models_root.join(LORA_DIR), name, 2)
    }

    /// Klein's folder under the model root ([`KLEIN_DIRS`], first found).
    fn klein_dir(&self) -> Option<PathBuf> {
        KLEIN_DIRS.iter().map(|d| self.models_root.join(d)).find(|d| d.join("transformer").is_dir() && d.join("text_encoder").is_dir())
    }

    fn moss_edit_lora(&self) -> Option<PathBuf> {
        MOSS_EDIT_LORAS.iter().find_map(|n| self.template_lora(n))
    }

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
    /// `params.encoder: "krea2"` answers with Krea 2's own Qwen3-VL-4B instead
    /// ([`CHAT_ON_KREA2`]): a Krea 2 fill's rewriter then needs no swap of the resident family.
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
        let on_krea2 = chat_on_krea2(&req.params)?;
        self.ensure_loaded(if on_krea2 { Family::Krea2 } else { Family::QwenImage21 }, progress)?;
        progress(0.5, "Thinking");
        let opts = ChatOptions { seed: req.seed, ..ChatOptions::default() };
        let stop = || cancel.load(Ordering::SeqCst);
        let reply = match &self.resident {
            Resident::Qwen(p) if !on_krea2 => p.chat(&system, &req.prompt, image.as_ref(), &opts, &stop),
            Resident::Krea2(p) if on_krea2 => p.chat(&system, &req.prompt, image.as_ref(), &opts, &stop),
            _ => return Err(EngineError::new(ErrorKind::Internal, "no pipeline")),
        };
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
            Resident::Layered(p) => qwen_image_rs::device::trim_pool(p.device()),
            Resident::Edit2511(p) => qwen_image_rs::device::trim_pool(p.device()),
            Resident::Moss(p, _) => qwen_image_rs::device::trim_pool(p.device()),
            #[cfg(feature = "mlx")]
            Resident::Mlx(_) | Resident::Klein(_) => {}
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
            Resident::Layered(p) => qwen_image_rs::device::bind_thread(p.device()),
            Resident::Edit2511(p) => qwen_image_rs::device::bind_thread(p.device()),
            Resident::Moss(p, _) => qwen_image_rs::device::bind_thread(p.device()),
            #[cfg(feature = "mlx")]
            Resident::Mlx(_) | Resident::Klein(_) => {}
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
            Resident::Edit2511(_) => (qwen_image_rs::edit2511_pipeline::cpu_device(), qwen_image_rs::edit2511_pipeline::adapter_dtype()),
            Resident::Moss(p, _) => (p.device().clone(), p.dtype()),
            #[cfg(feature = "mlx")]
            Resident::Mlx(_) | Resident::Klein(_) => return Err(EngineError::new(ErrorKind::Internal, "MLX applies its LoRAs itself")),
            Resident::Layered(_) | Resident::None => return Err(EngineError::new(ErrorKind::Internal, "no pipeline for the LoRAs")),
        };
        let mut layers = Vec::with_capacity(asks.len());
        for (file, strength) in asks {
            let path = self.lora_path(&file)?;
            let loaded = match self.loras.iter().find(|(p, _)| *p == path) {
                Some((_, f)) => Arc::clone(f),
                None => {
                    if !path.is_file() {
                        return Err(EngineError::new(
                            ErrorKind::ModelMissing,
                            format!("LoRA {} is not installed (looked in {})", file, self.models_root.join(LORA_DIR).display()),
                        ));
                    }
                    let f =
                        Arc::new(LoraFile::load(&path, &dev, dtype).map_err(|e| EngineError::new(ErrorKind::ModelUnreadable, format!("LoRA {file}: {e}")))?);
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

    /// Where 2.1 loads from: the snapshot folder when there, else the models root itself when
    /// it is a ComfyUI `models` folder holding ComfyUI's 2.1 files (R4: no snapshot needed).
    fn qwen21_source(&self) -> Option<PathBuf> {
        let snap = self.snapshot_dir();
        if is_snapshot(&snap) { Some(snap) } else { qwen_image_rs::comfy_files::find_qwen21(&self.models_root).map(|_| self.models_root.clone()) }
    }

    pub fn krea2_dir(&self) -> PathBuf {
        self.models_root.join(KREA2_DIR)
    }

    /// The prequantized-DiT cache: beside the models in the worker's own folders, but never
    /// inside a ComfyUI `models` folder (`source` is the models root then), which belongs to
    /// ComfyUI: the user's local app data instead.
    fn cache_dir(&self, source: &Path) -> PathBuf {
        match std::env::var_os("LOCALAPPDATA").filter(|_| source == self.models_root) {
            Some(d) => PathBuf::from(d).join("genai-native").join("qir-cache"),
            None => self.models_root.join(CACHE_DIR),
        }
    }

    /// Where Krea 2 loads from, as [`Self::qwen21_source`].
    fn krea2_source(&self) -> Option<PathBuf> {
        let dir = self.krea2_dir();
        if krea2_pipeline::is_snapshot(&dir) { Some(dir) } else { qwen_image_rs::comfy_files::find_krea2(&self.models_root).map(|_| self.models_root.clone()) }
    }

    fn has_family(&self, family: Family) -> bool {
        match family {
            Family::QwenImage21 => self.qwen21_source().is_some(),
            Family::Krea2 => self.krea2_source().is_some(),
            Family::Layered => self.layered_available(),
            Family::Edit2511 => self.edit2511_available(),
            Family::Moss(v) => moss::find_files(&self.models_root, v).is_some(),
            Family::Klein => cfg!(feature = "mlx") && self.klein_dir().is_some() && self.models_root.join(KLEIN_VAE).is_file(),
        }
    }

    /// The templates the weights on disk serve (none on a refused card).
    fn installed(&self) -> Vec<&'static str> {
        if self.refused.is_some() {
            return Vec::new();
        }
        let edit2511 = self.edit2511_available();
        TEMPLATES
            .iter()
            .filter(|(t, _, f)| {
                #[cfg(feature = "mlx")]
                if t.starts_with("moss-image/") {
                    return self.mlx_recipe(t).is_ok();
                }
                (if *f == Family::Edit2511 { edit2511 } else { self.has_family(*f) })
                    && (*f != Family::Edit2511
                        || edit2511_defaults(t).is_some_and(|(_, _, names)| names.iter().all(|n| self.models_root.join("loras").join(n).is_file())))
                    && (*t != TEMPLATE_KREA2_EDIT || self.lora_path(KREA2_EDIT_LORA).is_ok_and(|p| p.is_file()))
                    && (*t != TEMPLATE_OUTPAINT || self.lora_path(QWEN21_OUTPAINT_LORA).is_ok_and(|p| p.is_file()))
                    && (*t != TEMPLATE_OUTPAINT_TURBO || [QWEN21_OUTPAINT_LORA, QWEN21_TURBO_LORA].iter().all(|n| self.lora_path(n).is_ok_and(|p| p.is_file())))
                    && (*t != TEMPLATE_KREA2_FILL || self.lora_path(KREA2_ANYPAINT_LORA).is_ok_and(|p| p.is_file()))
                    && (*t != TEMPLATE_MOSS_TURBO || self.template_lora(MOSS_TURBO_LORA).is_some())
                    && (*t != TEMPLATE_MOSS_EDIT || self.moss_edit_lora().is_some())
                    && (*t != TEMPLATE_MOSS_EDIT_TURBO || (self.moss_edit_lora().is_some() && self.template_lora(MOSS_TURBO_LORA).is_some()))
                    && (*t != TEMPLATE_KREA2_STYLE
                        || (self.lora_path(KREA2_STYLE_LORA).is_ok_and(|p| p.is_file())
                            && self.krea2_source().is_some_and(|p| krea2_pipeline::is_snapshot(&p) || p.join("vae/qwen_image_vae.safetensors").is_file())))
            })
            .map(|(t, _, _)| *t)
            .collect()
    }

    /// Load `family`'s pipeline if it is not resident, letting the other family's go first.
    fn ensure_loaded(&mut self, family: Family, progress: &mut dyn FnMut(f32, &str)) -> Result<(), EngineError> {
        self.bind();
        match (&self.resident, family) {
            (Resident::Qwen(_), Family::QwenImage21)
            | (Resident::Krea2(_), Family::Krea2)
            | (Resident::Layered(_), Family::Layered)
            | (Resident::Edit2511(_), Family::Edit2511) => return Ok(()),
            (Resident::Moss(_, held), Family::Moss(v)) if *held == v => return Ok(()),
            #[cfg(feature = "mlx")]
            (Resident::Mlx(_), Family::Moss(_)) => return Ok(()),
            #[cfg(feature = "mlx")]
            (Resident::Klein(_), Family::Klein) => return Ok(()),
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
                let Some(dir) = self.qwen21_source() else {
                    return Err(EngineError::new(
                        ErrorKind::ModelMissing,
                        format!(
                            "no Qwen-Image-2.1 in {}: neither a snapshot in {MODEL_DIR}/ nor ComfyUI's files (diffusion_models/qwen_image_2.1_bf16, \
                             text_encoders/qwen3vl_8b_bf16, vae/qwen_image_2.1_vae_bf16 and ComfyUI's qwen25_tokenizer)",
                            self.models_root.display()
                        ),
                    ));
                };
                // A pre-quantized text encoder beside the snapshot (`qwen-image-rs
                // prequantize-text`, Q8_0 GGUF) loads in seconds; the bf16 shards take about a
                // minute, Q8 on load two, and the encoder is loaded again after every eviction.
                // From ComfyUI's single file under `--text-quant`: the same GGUF, made once into
                // the cache folder.
                let text_gguf = match qwen_image_rs::comfy_files::find_qwen21(&dir).filter(|_| dir == self.models_root) {
                    Some(f) if self.quant_text => {
                        report(0.0, "Preparing the text encoder (first run only)");
                        Some(qwen_image_rs::comfy_files::text_q8_cache(&f.text, "model", &self.cache_dir(&dir)).map_err(err)?)
                    }
                    Some(_) => None,
                    None => Some(dir.join(TEXT_GGUF)).filter(|g| g.is_file()),
                };
                let opts = Options {
                    cache: CacheSpec::from_cli(Some(&self.cache_dir(&dir)), false, false),
                    quant_text: self.quant_text,
                    text_gguf,
                    ..Options::default()
                };
                Resident::Qwen(Pipeline::load(&dir, opts, &mut report).map_err(err)?)
            }
            Family::Krea2 => {
                let Some(dir) = self.krea2_source() else {
                    return Err(EngineError::new(
                        ErrorKind::ModelMissing,
                        format!(
                            "no Krea 2 in {}: neither a weights folder in {KREA2_DIR}/ nor ComfyUI's files (diffusion_models/krea2_turbo_fp8_scaled, \
                             text_encoders/qwen3vl_4b_fp8_scaled, vae/wan_2.1_vae and ComfyUI's qwen25_tokenizer)",
                            self.models_root.display()
                        ),
                    ));
                };
                let opts = Krea2Options { cache: CacheSpec::from_cli(Some(&self.cache_dir(&dir)), false, false), ..Krea2Options::defaults() };
                Resident::Krea2(Krea2Pipeline::load(&dir, opts, &mut report).map_err(err)?)
            }
            Family::Edit2511 => {
                report(0.0, "Loading Qwen-Image-Edit-2511");
                Resident::Edit2511(Edit2511Pipeline::load(&self.models_root).map_err(err)?)
            }
            Family::Layered => {
                if !self.layered_available() {
                    return Err(EngineError::new(
                        ErrorKind::ModelMissing,
                        format!(
                            "no Qwen-Image-Layered in {}: ComfyUI's diffusion_models/qwen_image_layered_fp8mixed, \
                             text_encoders/qwen_2.5_vl_7b_fp8_scaled, vae/qwen_image_layered_vae and ComfyUI's qwen25_tokenizer",
                            self.models_root.display()
                        ),
                    ));
                }
                report(0.0, "Loading Qwen-Image-Layered");
                Resident::Layered(LayeredPipeline::load(&self.models_root).map_err(err)?)
            }
            #[cfg(feature = "mlx")]
            Family::Klein => {
                report(0.5, "Loading the VAE");
                Resident::Klein(Box::new(gn_mlx::klein::KleinMlx::new(&self.models_root.join(KLEIN_VAE)).map_err(err)?))
            }
            #[cfg(not(feature = "mlx"))]
            Family::Klein => return Err(EngineError::new(ErrorKind::Unsupported, "FLUX 3 Image Klein runs on a Mac (MLX) only")),
            #[cfg(feature = "mlx")]
            Family::Moss(_) => {
                report(0.5, "Loading the VAE");
                let cache = self.models_root.join(CACHE_DIR).join("mlx");
                Resident::Mlx(Box::new(gn_mlx::pipeline::MossMlx::new(&self.models_root, &cache).map_err(err)?))
            }
            #[cfg(not(feature = "mlx"))]
            Family::Moss(v) => {
                let Some(files) = moss::find_files(&self.models_root, v) else {
                    return Err(EngineError::new(
                        ErrorKind::ModelMissing,
                        format!(
                            "no MOSS-Image in {}: diffusion_models/{}_int8_convrot (or _bf16), \
                             text_encoders/qwen3.5_9b_int8_convrot (or _bf16), vae/flux2-vae_small (or flux2-vae) and Qwen3.5's \
                             tokenizer.json in text_encoders/qwen35_tokenizer/",
                            self.models_root.display(),
                            match v {
                                MossVariant::Base => "moss_image_base",
                                MossVariant::Sft => "moss_image_sft",
                                MossVariant::Kromoss => "kromoss",
                            }
                        ),
                    ));
                };
                Resident::Moss(MossPipeline::load(&files, &mut report).map_err(err)?, v)
            }
        };
        Ok(())
    }

    fn edit2511_available(&self) -> bool {
        qwen_image_rs::edit2511_files::Edit2511Files::resolve(&self.models_root).is_ok()
            && qwen_image_rs::comfy_files::tokenizer_dir(&self.models_root).is_some()
    }

    fn edit2511(
        &mut self,
        req: &RunRequest,
        blobs: &[Vec<u8>],
        progress: &mut dyn FnMut(f32, &str),
        preview: &mut dyn FnMut(Preview),
        cancel: &AtomicBool,
        started: Instant,
    ) -> Result<(DoneInfo, Vec<Vec<u8>>), EngineError> {
        let prepare_started = Instant::now();
        let mut ask = edit2511_request(req, blobs)?;
        if ["QIR_PROFILE", "QIR_PHASE_PROFILE"].iter().any(|key| std::env::var(key).is_ok_and(|v| v == "1")) {
            eprintln!("2511 phase worker image prepare: {:.6} s", prepare_started.elapsed().as_secs_f64());
        }
        let (_, _, names) = edit2511_defaults(&req.model).ok_or_else(|| EngineError::new(ErrorKind::BadRequest, "unknown 2511 template"))?;
        let mut params = req.params.clone();
        let mut loras: Vec<Value> = names.iter().map(|n| json!({"file": n, "strength": 1.0})).collect();
        if let Some(value) = params.get("loras") {
            let array = value.as_array().ok_or_else(|| EngineError::new(ErrorKind::BadRequest, "loras must be an array"))?;
            loras.extend(array.iter().cloned());
        }
        params.insert("loras".into(), Value::Array(loras));
        let result = (|| {
            self.ensure_loaded(Family::Edit2511, progress)?;
            ask.loras = self.stack_for(&params)?;
            let mode = preview_mode(req.preview);
            self.ensure_tiny(mode, LatentSpace::Wan21);
            let Resident::Edit2511(p) = &self.resident else { return Err(EngineError::new(ErrorKind::Internal, "no 2511 pipeline")) };
            let mut sink = |f: Frame| preview(engine_preview(f));
            let mut viewer = Previewer::new(mode, LatentSpace::Wan21, self.tiny.as_ref(), &mut sink);
            let shown = (mode != PreviewMode::Off).then_some(&mut viewer as &mut dyn StepPreview);
            let out = p
                .generate_previewed(&ask, |i, n| progress(0.3 + 0.7 * i as f32 / n.max(1) as f32, "Editing"), shown, cancel)
                .map_err(|e| if cancel.load(Ordering::SeqCst) { EngineError::cancelled() } else { EngineError::new(kind_of(&e), e.to_string()) });
            log_previews(mode, &viewer);
            out
        })();
        self.give_back();
        let rgb = match result {
            Ok(rgb) => rgb,
            Err(e) => {
                exit_if_poisoned(&e.message);
                if matches!(e.kind, ErrorKind::Internal | ErrorKind::ModelUnreadable) {
                    self.last_error = Some(e.message.clone());
                }
                if e.kind == ErrorKind::OutOfMemory {
                    self.resident = Resident::None;
                }
                return Err(e);
            }
        };
        self.last_error = None;
        let header = Blob { width: rgb.width(), height: rgb.height(), channels: 4, index: 0 };
        let rgba = rgb.as_raw().as_chunks::<3>().0.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect();
        Ok((DoneInfo { images: vec![header], texts: Vec::new(), seed: req.seed, elapsed_ms: started.elapsed().as_millis() as u64 }, vec![rgba]))
    }

    /// Qwen-Image-Layered's three files and the Qwen tokenizer under the models root (ComfyUI's
    /// layout only: it has no snapshot form here).
    fn layered_available(&self) -> bool {
        qwen_image_rs::layered_files::LayeredFiles::resolve(&self.models_root).is_ok() && qwen_image_rs::comfy_files::tokenizer_dir(&self.models_root).is_some()
    }

    /// Split into Layers: the picture (a multiple of 16 on each side, as the app sends it) and
    /// `params.layers`; one RGBA blob per layer, background first. The text encoder and the DiT
    /// take the card in turn inside the pipeline, so the other families are unloaded first.
    fn split(
        &mut self,
        req: &RunRequest,
        blobs: &[Vec<u8>],
        progress: &mut dyn FnMut(f32, &str),
        preview: &mut dyn FnMut(Preview),
        cancel: &AtomicBool,
        started: Instant,
    ) -> Result<(DoneInfo, Vec<Vec<u8>>), EngineError> {
        let Some(image) = &req.image else { return Err(EngineError::new(ErrorKind::BadRequest, format!("{} needs the picture", req.model))) };
        let picture = reference_of(image, blobs, "picture")?;
        if picture.width == 0 || picture.height == 0 || picture.width % 16 != 0 || picture.height % 16 != 0 {
            return Err(EngineError::new(
                ErrorKind::BadRequest,
                format!("the picture is {}x{}; the split takes multiples of 16", picture.width, picture.height),
            ));
        }
        let layers = match req.params.get("layers") {
            None => DEFAULT_LAYERS,
            Some(v) => v
                .as_u64()
                .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
                .filter(|n| (1..=8).contains(n))
                .ok_or_else(|| EngineError::new(ErrorKind::BadRequest, format!("layers {v} is not a whole number from 1 to 8")))?,
        } as usize;
        let steps = if req.steps == 0 { LAYERS_STEPS } else { req.steps } as usize;
        let Some(rgba) = qwen_image_rs::layered_pipeline::RgbaImage::from_raw(picture.width as u32, picture.height as u32, picture.rgba) else {
            return Err(EngineError::new(ErrorKind::BadRequest, "the picture's bytes do not match its size"));
        };
        let result: Result<Vec<_>, EngineError> = (|| {
            self.ensure_loaded(Family::Layered, progress)?;
            let mode = preview_mode(req.preview);
            self.ensure_tiny(mode, LatentSpace::Layered);
            let Resident::Layered(p) = &self.resident else { return Err(EngineError::new(ErrorKind::Internal, "no pipeline")) };
            let mut report = |i: usize, n: usize| progress(0.3 + 0.7 * i as f32 / n.max(1) as f32, "Splitting");
            let mut sink = |f: Frame| preview(engine_preview(f));
            let mut viewer = Previewer::new(mode, LatentSpace::Layered, self.tiny.as_ref(), &mut sink);
            let shown = (mode != PreviewMode::Off).then_some(&mut viewer as &mut dyn StepPreview);
            p.set_step_cache(req.params.get("step_cache").and_then(Value::as_f64).map(|t| t as f32));
            let out = p
                .split_previewed(&rgba, layers, &req.prompt, req.seed, steps, &mut report, shown, cancel)
                .map_err(|e| if cancel.load(Ordering::SeqCst) { EngineError::cancelled() } else { EngineError::new(kind_of(&e), e.to_string()) });
            log_previews(mode, &viewer);
            out
        })();
        self.give_back();
        let out = match result {
            Ok(o) => o,
            Err(e) => {
                exit_if_poisoned(&e.message);
                if matches!(e.kind, ErrorKind::ModelUnreadable | ErrorKind::Internal) {
                    self.last_error = Some(e.message.clone());
                }
                if e.kind == ErrorKind::OutOfMemory {
                    self.resident = Resident::None;
                }
                return Err(e);
            }
        };
        self.last_error = None;
        let images = out.iter().enumerate().map(|(i, l)| Blob { width: l.width(), height: l.height(), channels: 4, index: i as u32 }).collect();
        let bytes = out.into_iter().map(|l| l.into_raw()).collect();
        Ok((DoneInfo { images, texts: Vec::new(), seed: req.seed, elapsed_ms: started.elapsed().as_millis() as u64 }, bytes))
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

/// Template adapter order and sampler defaults, independent of the installed files.
pub fn edit2511_defaults(template: &str) -> Option<(u32, f32, Vec<&'static str>)> {
    use qwen_image_rs::edit2511_files::ADAPTER_NAMES as A;
    let suffix = template.strip_prefix("qwen-edit-2511/")?;
    Some(match suffix {
        "edit" | "fill" | "fill-ref" | "expand" => (40, 4., vec![]),
        "fill-lightning-4" => (4, 1., vec![A[0]]),
        "angles" => (4, 1., vec![A[3], A[0]]),
        "remove-lightning-8" => (8, 1., vec![A[1], A[2]]),
        "edit-lightning-8" | "fill-lightning-8" | "fill-guided" | "fill-ref-lightning-8" | "image-ref" | "expand-lightning-8" => (8, 1., vec![A[1]]),
        _ => return None,
    })
}

/// Resolve graph inputs before loading models so invalid requests fail without touching CUDA.
fn edit2511_request(req: &RunRequest, blobs: &[Vec<u8>]) -> Result<Edit2511Request, EngineError> {
    let bad = |message: &str| EngineError::new(ErrorKind::BadRequest, message);
    let (steps, cfg, _) = edit2511_defaults(&req.model).ok_or_else(|| bad("unknown 2511 template"))?;
    for (key, expected) in [("sampler", "euler"), ("sampler_name", "euler"), ("scheduler", "simple")] {
        if req.params.get(key).is_some_and(|v| v.as_str() != Some(expected)) {
            return Err(bad("2511 supports Euler/simple only"));
        }
    }
    if req.params.get("denoise").is_some_and(|v| v.as_f64() != Some(1.)) {
        return Err(bad("2511 supports denoise 1 only"));
    }
    let rgb = |b: &Blob, what: &str| -> Result<edit2511_pipeline::RgbImage, EngineError> {
        let input = bytes_of(b, blobs, what)?;
        let bytes = match b.channels {
            3 => input.to_vec(),
            4 => input.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect(),
            1 => input.iter().flat_map(|v| [*v, *v, *v]).collect(),
            c => return Err(bad(&format!("the {what} has {c} channels (1, 3 or 4 expected)"))),
        };
        edit2511_pipeline::RgbImage::from_raw(b.width, b.height, bytes).ok_or_else(|| bad("invalid RGB blob"))
    };
    let suffix = req.model.strip_prefix("qwen-edit-2511/").ok_or_else(|| bad("invalid 2511 template"))?;
    let mut references = Vec::new();
    let mut target = None;
    let mask_needed = req.task == "fill" || req.task == "expand";
    let mask = if mask_needed { Some(rgb(req.mask.as_ref().ok_or_else(|| bad("2511 fill/expand needs the mask"))?, "mask")?) } else { None };
    if suffix == "image-ref" {
        if !(1..=3).contains(&req.references.len()) {
            return Err(bad("2511 image-ref needs one to three references"));
        }
        for b in &req.references {
            references.push(rgb(b, "reference")?);
        }
    } else {
        let picture = rgb(req.image.as_ref().ok_or_else(|| bad("2511 needs the picture"))?, "picture")?;
        let mut reference = picture.clone();
        if req.task == "expand" {
            let number = |key| {
                req.params.get(key).and_then(Value::as_u64).and_then(|v| u32::try_from(v).ok()).ok_or_else(|| bad("expand needs ref_x/ref_y/ref_w/ref_h"))
            };
            let (x, y, w, h) = (number("ref_x")?, number("ref_y")?, number("ref_w")?, number("ref_h")?);
            if w == 0 || h == 0 || x.checked_add(w).is_none_or(|v| v > picture.width()) || y.checked_add(h).is_none_or(|v| v > picture.height()) {
                return Err(bad("expand rectangle lies outside canvas"));
            }
            // ImageCrop uses the exact rectangle, without 2.1's token-grid widening or offsets.
            let mut bytes = Vec::with_capacity(w as usize * h as usize * 3);
            for row in y..y + h {
                for col in x..x + w {
                    bytes.extend_from_slice(&picture.get_pixel(col, row).0);
                }
            }
            reference = edit2511_pipeline::RgbImage::from_raw(w, h, bytes).ok_or_else(|| bad("invalid expand crop"))?;
        }
        let scaled = edit2511_pipeline::kontext(&reference).map_err(|e| bad(&e.to_string()))?;
        target = Some(if suffix == "angles" { scaled.clone() } else { picture });
        references.push(scaled);
        if suffix == "fill-guided" {
            let mask = mask.as_ref().ok_or_else(|| bad("guided fill needs mask"))?;
            references.push(edit2511_pipeline::kontext(mask).map_err(|e| bad(&e.to_string()))?);
        }
        if matches!(suffix, "edit" | "edit-lightning-8" | "fill-ref" | "fill-ref-lightning-8") {
            if req.references.len() > 2 {
                return Err(bad("2511 takes at most two additional references"));
            }
            for b in &req.references {
                references.push(rgb(b, "reference")?);
            }
        } else if !req.references.is_empty() {
            return Err(bad("this 2511 template takes no additional references"));
        }
    }
    Ok(Edit2511Request {
        prompt: req.prompt.clone(),
        negative: req.negative.clone(),
        references,
        target,
        mask,
        width: side(req.width),
        height: side(req.height),
        seed: req.seed,
        steps: if req.steps == 0 { steps } else { req.steps } as usize,
        cfg: if req.guidance > 0. { req.guidance } else { cfg },
        loras: LoraStack::default(),
        // The app's Fast draft: a first-block step cache at this threshold.
        step_cache: req.params.get("step_cache").and_then(Value::as_f64).map(|t| t as f32),
    })
}

/// A blob's bytes, checked against its header.
fn bytes_of<'a>(b: &Blob, blobs: &'a [Vec<u8>], what: &str) -> Result<&'a [u8], EngineError> {
    let bytes =
        blobs.get(b.index as usize).ok_or_else(|| EngineError::new(ErrorKind::BadRequest, format!("the {what} names blob {} that was not sent", b.index)))?;
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
        return Err(EngineError::new(
            ErrorKind::BadRequest,
            format!("the picture's place {x},{y} {w}x{h} lies outside the {}x{} canvas", canvas.width, canvas.height),
        ));
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
            Resident::Layered(p) => (vec!["qwen-image-layered".to_string()], p.device().is_cuda(), qwen_image_rs::device::free_vram(p.device())),
            Resident::Edit2511(p) => (vec!["qwen-image-edit-2511".to_string()], p.device().is_cuda(), qwen_image_rs::device::free_vram(p.device())),
            Resident::Moss(p, _) => (vec!["moss-image".to_string()], p.is_cuda(), qwen_image_rs::device::free_vram(p.device())),
            #[cfg(feature = "mlx")]
            Resident::Mlx(_) => (vec!["moss-image".to_string()], false, None),
            #[cfg(feature = "mlx")]
            Resident::Klein(_) => (vec!["flux3-klein".to_string()], false, None),
        };
        // The card's name where the driver gave one; the free memory only while a pipeline
        // holds the context (asking without one would create it).
        let device = match (&self.gpu, cuda) {
            (Some(g), true) => g.name.clone(),
            (None, true) => "cuda:0".into(),
            (_, false) if cfg!(feature = "mlx") => "Apple GPU (MLX)".into(),
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
        self.run_previewed(req, blobs, progress, &mut |_| {}, cancel)
    }

    /// Previews (`req.preview`) from every diffusion family: the step loops hand their
    /// denoised estimate to a [`Previewer`] (the chat templates have none).
    fn run_previewed(
        &mut self,
        req: &RunRequest,
        blobs: &[Vec<u8>],
        progress: &mut dyn FnMut(f32, &str),
        preview: &mut dyn FnMut(Preview),
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
        // A warm-up (the editor at launch): the template's models loaded, nothing generated.
        if req.params.get("warmup").and_then(Value::as_bool) == Some(true) {
            // The VRAM guard: with nothing resident, `need_free` bytes must be free on the card
            // (asked of the driver here; the health knows only once a pipeline holds the context).
            if let (Some(need), Resident::None) = (req.params.get("need_free").and_then(Value::as_u64), &self.resident) {
                let free = qwen_image_rs::device::best_device().ok().and_then(|d| qwen_image_rs::device::free_vram(&d)).map(|(f, _)| f as u64);
                if free.is_none_or(|f| f < need) {
                    let gb = |b: u64| b as f64 / 1e9;
                    let why = format!("skipped: {:.1} GB free on the card, {:.1} GB wanted", free.map_or(0.0, gb), gb(need));
                    return Ok((
                        DoneInfo { images: Vec::new(), texts: vec![why], seed: req.seed, elapsed_ms: started.elapsed().as_millis() as u64 },
                        Vec::new(),
                    ));
                }
            }
            let loaded = self.ensure_loaded(family, progress);
            self.give_back();
            if let Err(e) = loaded {
                if e.kind == ErrorKind::OutOfMemory {
                    self.resident = Resident::None;
                }
                return Err(e);
            }
            if cancel.load(Ordering::SeqCst) {
                return Err(EngineError::cancelled());
            }
            return Ok((
                DoneInfo { images: Vec::new(), texts: vec!["warm".into()], seed: req.seed, elapsed_ms: started.elapsed().as_millis() as u64 },
                Vec::new(),
            ));
        }
        if req.prompt.trim().is_empty() {
            return Err(EngineError::new(ErrorKind::BadRequest, "a prompt is needed"));
        }
        if task == "text" {
            return self.chat(req, blobs, progress, cancel, started);
        }
        if task == "layers" {
            return self.split(req, blobs, progress, preview, cancel, started);
        }
        if family == Family::Edit2511 {
            return self.edit2511(req, blobs, progress, preview, cancel, started);
        }
        // The pictures the model sees in order: image 1 is the picture being edited, then the
        // mask (a fill) or the references (an edit; a matte has none). The target takes image
        // 1's size, as ComfyUI's `TextEncodeQwenImage21` sizes its latent at `resolution 0`.
        // An expand is the exception: its one reference is the picture cut out of the canvas
        // and placed at its own position, and the target is the whole canvas.
        // A batch of edits (Upscale's tiles): `params.batch` lists a seed for each picture, the
        // picture and then each reference, and every one is edited alone with its seed.
        let batch: Option<Vec<u64>> = match req.params.get("batch") {
            None => None,
            Some(v) => match v.as_array().and_then(|a| a.iter().map(Value::as_u64).collect::<Option<Vec<u64>>>()) {
                Some(seeds) if task == "edit" && family == Family::QwenImage21 && seeds.len() == 1 + req.references.len() => Some(seeds),
                _ => {
                    return Err(EngineError::new(
                        ErrorKind::BadRequest,
                        format!("batch is a Qwen-Image-2.1 edit's seeds, one for the picture and one for each reference ({} given)", 1 + req.references.len()),
                    ));
                }
            },
        };
        let mut references: Vec<RefImage> = Vec::new();
        let mut canvas: Option<(usize, usize)> = None;
        match task {
            "edit" | "fill" | "expand" | "matte" => {
                let Some(image) = &req.image else { return Err(EngineError::new(ErrorKind::BadRequest, format!("{} needs the picture", req.model))) };
                let picture = reference_of(image, blobs, "picture")?;
                if req.model == TEMPLATE_OUTPAINT || req.model == TEMPLATE_OUTPAINT_TURBO {
                    // The grey canvas is image 1 as it is, the target its size.
                    references.push(picture);
                } else if task == "expand" {
                    canvas = Some((picture.width / 16 * 16, picture.height / 16 * 16));
                    references.push(placed_picture(&picture, &req.params)?);
                } else {
                    references.push(picture);
                }
                if task == "fill" {
                    let Some(mask) = &req.mask else { return Err(EngineError::new(ErrorKind::BadRequest, format!("{} needs the mask", req.model))) };
                    references.push(reference_of(mask, blobs, "mask")?);
                } else if task == "edit" {
                    if req.model == TEMPLATE_KREA2_EDIT && !req.references.is_empty() {
                        return Err(EngineError::new(ErrorKind::BadRequest, format!("{} edits the picture alone (no references)", req.model)));
                    }
                    for (i, r) in req.references.iter().enumerate() {
                        references.push(reference_of(r, blobs, &format!("reference {}", i + 2))?);
                    }
                }
                if batch.is_none() && references.len() > MAX_PICTURES {
                    return Err(EngineError::new(ErrorKind::BadRequest, format!("{} pictures; at most {MAX_PICTURES}", references.len())));
                }
            }
            // An image from references: the references alone, the target at the asked size.
            "image" if req.model == TEMPLATE_IMAGE_REF || req.model == TEMPLATE_KREA2_STYLE => {
                if req.references.is_empty() {
                    return Err(EngineError::new(ErrorKind::BadRequest, format!("{} needs at least one reference", req.model)));
                }
                for (i, r) in req.references.iter().enumerate() {
                    references.push(reference_of(r, blobs, &format!("reference {}", i + 1))?);
                }
                if references.len() > MAX_PICTURES - 1 {
                    return Err(EngineError::new(ErrorKind::BadRequest, format!("{} references; at most {}", references.len(), MAX_PICTURES - 1)));
                }
                canvas = Some((side(req.width), side(req.height)));
            }
            _ => {}
        }
        if req.model == TEMPLATE_KREA2_STYLE && (req.width == 0 || req.height == 0 || !(1..=2).contains(&references.len())) {
            return Err(EngineError::new(ErrorKind::BadRequest, "Krea style needs explicit width/height and one or two references"));
        }
        let default_steps = match family {
            Family::QwenImage21 => DEFAULT_STEPS,
            Family::Krea2 => KREA2_STEPS,
            Family::Layered => LAYERS_STEPS,
            Family::Edit2511 => 40,
            Family::Moss(_) if req.model == TEMPLATE_MOSS_TURBO || req.model == TEMPLATE_MOSS_EDIT_TURBO => MOSS_TURBO_STEPS,
            Family::Moss(_) if req.model == TEMPLATE_MOSS_EDIT => MOSS_EDIT_STEPS,
            Family::Moss(_) => MOSS_STEPS,
            // Klein's own (its checkpoint's 16).
            Family::Klein => 0,
        };
        let default_guidance = match family {
            Family::Moss(_) if req.model == TEMPLATE_MOSS_TURBO || req.model == TEMPLATE_MOSS_EDIT_TURBO => 1.0,
            Family::Moss(_) if req.model == TEMPLATE_MOSS_EDIT => MOSS_EDIT_GUIDANCE,
            Family::Moss(_) => MOSS_GUIDANCE,
            _ => 1.0,
        };
        let result: Result<Vec<Picture>, EngineError> = (|| {
            self.ensure_loaded(family, progress)?;
            if cancel.load(Ordering::SeqCst) {
                return Err(EngineError::cancelled());
            }
            let (width, height) = match canvas {
                Some(c) => c,
                None if references.is_empty() => (side(req.width), side(req.height)),
                None => (0, 0),
            };
            #[cfg(feature = "mlx")]
            if matches!(self.resident, Resident::Mlx(_)) {
                return self.run_mlx(req, references, (width, height), progress, cancel);
            }
            #[cfg(feature = "mlx")]
            if matches!(self.resident, Resident::Klein(_)) {
                return self.run_klein(req, references, (width, height), task, progress, cancel);
            }
            // A Krea 2 edit runs with its edit LoRA first (unless the request names it) and the
            // card's `ref_boost` unless the request gives one.
            let krea2_edit = req.model == TEMPLATE_KREA2_EDIT;
            let krea2_fill = req.model == TEMPLATE_KREA2_FILL;
            let params = if krea2_edit {
                with_edit_lora(&req.params)
            } else if req.model == TEMPLATE_OUTPAINT {
                with_lora_first(&req.params, QWEN21_OUTPAINT_LORA)
            } else if req.model == TEMPLATE_OUTPAINT_TURBO {
                with_lora_first(&with_lora_first(&req.params, QWEN21_TURBO_LORA), QWEN21_OUTPAINT_LORA)
            } else if krea2_fill {
                with_lora_first(&req.params, KREA2_ANYPAINT_LORA)
            } else if req.model == TEMPLATE_KREA2_STYLE {
                with_style_lora(&req.params)?
            } else if [TEMPLATE_MOSS_TURBO, TEMPLATE_MOSS_EDIT, TEMPLATE_MOSS_EDIT_TURBO].contains(&req.model.as_str()) {
                let missing = |what: &str| EngineError::new(ErrorKind::ModelMissing, format!("{what} is not in {}", self.models_root.join(LORA_DIR).display()));
                let mut params = req.params.clone();
                if req.model != TEMPLATE_MOSS_EDIT {
                    let turbo = self.template_lora(MOSS_TURBO_LORA).ok_or_else(|| missing(&format!("the MOSS turbo LoRA {MOSS_TURBO_LORA}")))?;
                    params = with_lora_first(&params, &turbo.to_string_lossy());
                }
                if req.model != TEMPLATE_MOSS_TURBO {
                    let edit = self.moss_edit_lora().ok_or_else(|| missing("the MOSS Edit LoRA (moss-image-edit/adapter_model.safetensors)"))?;
                    params = with_lora_first(&params, &edit.to_string_lossy());
                }
                params
            } else {
                req.params.clone()
            };
            let loras = self.stack_for(&params)?;
            let ref_boost = match req.params.get("ref_boost").and_then(Value::as_f64) {
                Some(b) if b.is_finite() && (0.0..=100.0).contains(&b) => b as f32,
                Some(b) => return Err(EngineError::new(ErrorKind::BadRequest, format!("ref_boost {b} is out of the range 0..100"))),
                None if krea2_edit => KREA2_REF_BOOST,
                None => 1.0,
            };
            let ask = Request {
                prompt: req.prompt.clone(),
                negative: req.negative.clone(),
                width,
                height,
                steps: if req.steps == 0 { default_steps } else { req.steps } as usize,
                seed: req.seed,
                guidance: if req.guidance > 0.0 { req.guidance } else { default_guidance },
                schedule: if req.model == TEMPLATE_OUTPAINT_TURBO { Schedule::ViggleTurbo } else { Schedule::Comfy },
                references,
                loras,
                ref_boost,
            };
            let fill_opts = if krea2_fill { Some(krea2_fill_options(&req.params)?) } else { None };
            let mut report = |f: f32, m: &str| progress(0.3 + 0.7 * f, m);
            let stop = || cancel.load(Ordering::SeqCst);
            let err = |e: anyhow::Error| EngineError::new(kind_of(&e), e.to_string());
            let mode = preview_mode(req.preview);
            self.ensure_tiny(mode, latent_space(family));
            let mut sink = |f: Frame| preview(engine_preview(f));
            let mut viewer = Previewer::new(mode, latent_space(family), self.tiny.as_ref(), &mut sink);
            let shown = (mode != PreviewMode::Off).then_some(&mut viewer as &mut dyn StepPreview);
            let out = match (&self.resident, &batch) {
                (Resident::Qwen(p), Some(seeds)) => {
                    let asks: Vec<Request> =
                        ask.references.iter().zip(seeds).map(|(r, &seed)| Request { references: vec![r.clone()], seed, ..ask.clone() }).collect();
                    p.generate_batch_previewed(&asks, &mut report, shown, &stop).map_err(err)
                }
                (Resident::Qwen(p), None) => p.generate_previewed(&ask, &mut report, shown, &stop).map(|p| vec![p]).map_err(err),
                (Resident::Krea2(p), _) if fill_opts.is_some() => {
                    let opts = fill_opts.clone().unwrap_or_default();
                    p.fill_previewed(&ask, &opts, &mut report, shown, &stop).map(|p| vec![p]).map_err(err)
                }
                (Resident::Krea2(p), _) if req.model == TEMPLATE_KREA2_STYLE => {
                    p.generate_style_previewed(&ask, &mut report, shown, &stop).map(|p| vec![p]).map_err(err)
                }
                (Resident::Krea2(p), _) => p.generate_previewed(&ask, &mut report, shown, &stop).map(|p| vec![p]).map_err(err),
                (Resident::Moss(p, _), _) => p.generate_previewed(&ask, &mut report, shown, &stop).map(|p| vec![p]).map_err(err),
                #[cfg(feature = "mlx")]
                (Resident::Mlx(_) | Resident::Klein(_), _) => Err(EngineError::new(ErrorKind::Internal, "MLX runs before the LoRA stack")),
                (Resident::Edit2511(_) | Resident::Layered(_) | Resident::None, _) => Err(EngineError::new(ErrorKind::Internal, "no pipeline")),
            };
            log_previews(mode, &viewer);
            out
        })();
        self.give_back();
        let pics = match result {
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
        let images = pics.iter().enumerate().map(|(i, p)| Blob { width: p.width, height: p.height, channels: 4, index: i as u32 }).collect();
        Ok((
            DoneInfo { images, texts: Vec::new(), seed: req.seed, elapsed_ms: started.elapsed().as_millis() as u64 },
            pics.into_iter().map(|p| p.rgba).collect(),
        ))
    }

    fn free(&mut self) {
        self.bind();
        self.resident = Resident::None;
        self.loras.clear();
        self.tiny = None;
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

/// The steps the CUDA route's MOSS templates default to (the editor sends them as given).
#[cfg(feature = "mlx")]
fn cuda_default_steps(template: &str) -> Option<u32> {
    match template {
        TEMPLATE_MOSS_TURBO | TEMPLATE_MOSS_EDIT_TURBO => Some(MOSS_TURBO_STEPS),
        TEMPLATE_MOSS_EDIT => Some(MOSS_EDIT_STEPS),
        TEMPLATE_MOSS | TEMPLATE_MOSS_BASE => Some(MOSS_STEPS),
        _ => None,
    }
}

/// The CFG the CUDA route's MOSS templates default to.
#[cfg(feature = "mlx")]
fn cuda_default_guidance(template: &str) -> Option<f32> {
    match template {
        TEMPLATE_MOSS_TURBO | TEMPLATE_MOSS_EDIT_TURBO => Some(1.0),
        TEMPLATE_MOSS_EDIT => Some(MOSS_EDIT_GUIDANCE),
        TEMPLATE_MOSS | TEMPLATE_MOSS_BASE => Some(MOSS_GUIDANCE),
        _ => None,
    }
}

#[cfg(feature = "mlx")]
impl DiffusionEngine {
    /// A Klein request: an image at `size` from the prompt alone, an edit with its pictures (the
    /// outpaint LoRA rides in the request's LoRAs for an expand), or a fill whose second picture
    /// is the mask. With pictures the size follows the first one (at most 768², as Klein sizes an
    /// edit) and `size` is (0, 0).
    fn run_klein(
        &mut self,
        req: &RunRequest,
        mut references: Vec<RefImage>,
        size: (usize, usize),
        task: &str,
        progress: &mut dyn FnMut(f32, &str),
        cancel: &AtomicBool,
    ) -> Result<Vec<Picture>, EngineError> {
        let dir = self.klein_dir().ok_or_else(|| {
            EngineError::new(ErrorKind::ModelMissing, format!("FLUX 3 Image Klein is not in {} ({})", self.models_root.display(), KLEIN_DIRS.join(" or ")))
        })?;
        let mut loras = Vec::new();
        for (file, strength) in lora_asks(&req.params)? {
            let path = self.lora_path(&file)?;
            if !path.is_file() {
                return Err(EngineError::new(
                    ErrorKind::ModelMissing,
                    format!("LoRA {file} is not installed (looked in {})", self.models_root.join(LORA_DIR).display()),
                ));
            }
            loras.push(gn_mlx::pipeline::LoraSpec { path, strength });
        }
        let pic = |r: RefImage| gn_mlx::pipeline::RefPicture { width: r.width, height: r.height, rgba: r.rgba };
        let mask = if task == "fill" {
            if references.len() != 2 {
                return Err(EngineError::new(ErrorKind::BadRequest, "a fill takes the picture and its mask"));
            }
            references.pop().map(pic)
        } else {
            None
        };
        let ask = gn_mlx::pipeline::Request {
            prompt: req.prompt.clone(),
            negative: req.negative.clone(),
            width: size.0,
            height: size.1,
            steps: req.steps as usize,
            guidance: if req.guidance > 0.0 { req.guidance } else { 1.0 },
            seed: req.seed,
            references: references.into_iter().map(pic).collect(),
            mask,
        };
        let Resident::Klein(k) = &mut self.resident else { return Err(EngineError::new(ErrorKind::Internal, "no Klein pipeline")) };
        let stop = || cancel.load(Ordering::SeqCst);
        let mut step = |i: usize, n: usize| progress(0.3 + 0.65 * i as f32 / n.max(1) as f32, if i == 0 { "Loading the models" } else { "Sampling" });
        let p = k.generate(&dir, &loras, &ask, &mut step, &stop).map_err(|e| EngineError::new(kind_of(&e), e.to_string()))?;
        let t = p.timings;
        eprintln!(
            "gn-diffusion: {} on MLX ({} steps, {}x{}, {} LoRAs): load {:.1} s, encode {:.1} s, steps {:.1} s, decode {:.1} s",
            req.model,
            ask.steps,
            p.width,
            p.height,
            loras.len(),
            t.load_s,
            t.encode_s,
            t.steps_s,
            t.decode_s
        );
        let n = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
        Ok(vec![Picture { width: n(p.width), height: n(p.height), rgba: p.rgba, timings: Default::default() }])
    }

    /// How MLX serves a MOSS template on this model root: the variant, the template's LoRAs
    /// (full paths, strength 1), its steps and CFG. Measured on a 16 GB M1 Air (gn-mlx,
    /// 2026-11-07): MOSS Turbo is MOSS-Image-SFT with akhaliq's Euler-native 4-step LoRA (1024²
    /// in 144 s against Kromoss + DMD2's 311 s), Kromoss + DMD2 when that LoRA is missing; MOSS
    /// Edit Turbo is the edit LoRA at 8 steps and CFG 4, because every CFG 1 stack (with the
    /// Kromoss DMD2, lodestones' SFT DMD or the 4-step LoRA) leaves the picture unedited.
    #[cfg(feature = "mlx")]
    fn mlx_recipe(&self, template: &str) -> Result<(MossVariant, Vec<PathBuf>, u32, f32), EngineError> {
        let has = |v: MossVariant| moss::find_files(&self.models_root, v).is_some();
        let missing = |what: String| EngineError::new(ErrorKind::ModelMissing, format!("{what} is not in {}", self.models_root.display()));
        let edit = || self.moss_edit_lora().ok_or_else(|| missing("the MOSS Edit LoRA (loras/moss-image-edit/adapter_model.safetensors)".into()));
        let sft = || {
            if has(MossVariant::Sft) {
                Ok(MossVariant::Sft)
            } else {
                Err(missing("MOSS-Image-SFT (diffusion_models/moss_image_sft_int8_convrot.safetensors)".into()))
            }
        };
        match template {
            TEMPLATE_MOSS_TURBO => {
                match (has(MossVariant::Sft), self.template_lora(MOSS_SFT_4STEP_LORA), has(MossVariant::Kromoss), self.template_lora(MOSS_TURBO_LORA)) {
                    (true, Some(four), _, _) => Ok((MossVariant::Sft, vec![four], 4, 1.0)),
                    (_, _, true, Some(dmd2)) => Ok((MossVariant::Kromoss, vec![dmd2], MOSS_TURBO_STEPS, 1.0)),
                    _ => Err(missing(format!("MOSS Turbo (MOSS-Image-SFT with loras/{MOSS_SFT_4STEP_LORA}, or Kromoss with loras/{MOSS_TURBO_LORA})"))),
                }
            }
            TEMPLATE_MOSS_EDIT_TURBO => Ok((sft()?, vec![edit()?], 8, MOSS_EDIT_GUIDANCE)),
            TEMPLATE_MOSS_EDIT => Ok((sft()?, vec![edit()?], MOSS_EDIT_STEPS, MOSS_EDIT_GUIDANCE)),
            TEMPLATE_MOSS => Ok((sft()?, Vec::new(), MOSS_STEPS, MOSS_GUIDANCE)),
            TEMPLATE_MOSS_BASE if has(MossVariant::Base) => Ok((MossVariant::Base, Vec::new(), MOSS_STEPS, MOSS_GUIDANCE)),
            t => Err(EngineError::new(ErrorKind::ModelMissing, format!("{t} is not served on MLX"))),
        }
    }

    /// A MOSS request on MLX: the template's LoRAs and then the request's, applied unmerged to
    /// the DiT the pipeline holds; no previews.
    #[cfg(feature = "mlx")]
    fn run_mlx(
        &mut self,
        req: &RunRequest,
        references: Vec<RefImage>,
        (width, height): (usize, usize),
        progress: &mut dyn FnMut(f32, &str),
        cancel: &AtomicBool,
    ) -> Result<Vec<Picture>, EngineError> {
        let (variant, template_loras, steps, guidance) = self.mlx_recipe(&req.model)?;
        let mut loras: Vec<gn_mlx::pipeline::LoraSpec> = template_loras.into_iter().map(|path| gn_mlx::pipeline::LoraSpec { path, strength: 1.0 }).collect();
        for (file, strength) in lora_asks(&req.params)? {
            let path = self.lora_path(&file)?;
            if !path.is_file() {
                return Err(EngineError::new(
                    ErrorKind::ModelMissing,
                    format!("LoRA {file} is not installed (looked in {})", self.models_root.join(LORA_DIR).display()),
                ));
            }
            if !loras.iter().any(|l| l.path == path) {
                loras.push(gn_mlx::pipeline::LoraSpec { path, strength });
            }
        }
        let ask = gn_mlx::pipeline::Request {
            prompt: req.prompt.clone(),
            negative: req.negative.clone(),
            width,
            height,
            // The editor sends the template's CUDA defaults (MOSS Turbo's 8 steps, Edit Turbo's
            // 8 steps at CFG 1): those mean "the template's", which on MLX is the recipe's.
            steps: if req.steps == 0 || cuda_default_steps(&req.model) == Some(req.steps) { steps } else { req.steps } as usize,
            guidance: if req.guidance <= 0.0 || cuda_default_guidance(&req.model) == Some(req.guidance) { guidance } else { req.guidance },
            seed: req.seed,
            references: references.into_iter().map(|r| gn_mlx::pipeline::RefPicture { width: r.width, height: r.height, rgba: r.rgba }).collect(),
            mask: None,
        };
        let Resident::Mlx(m) = &mut self.resident else { return Err(EngineError::new(ErrorKind::Internal, "no MLX pipeline")) };
        let stop = || cancel.load(Ordering::SeqCst);
        let mut step = |i: usize, n: usize| progress(0.3 + 0.65 * i as f32 / n.max(1) as f32, if i == 0 { "Loading the models" } else { "Sampling" });
        let p = m.generate(variant, &loras, &ask, &mut step, &stop).map_err(|e| EngineError::new(kind_of(&e), e.to_string()))?;
        let t = p.timings;
        eprintln!(
            "gn-diffusion: {} on MLX ({variant:?}, {} steps, CFG {}, {}x{}): encode {:.1} s, DiT load + references {:.1} s, steps {:.1} s, decode {:.1} s",
            req.model, ask.steps, ask.guidance, p.width, p.height, t.encode_s, t.load_s, t.steps_s, t.decode_s
        );
        let n = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
        Ok(vec![Picture { width: n(p.width), height: n(p.height), rgba: p.rgba, timings: Default::default() }])
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
    is_snapshot(&dir.join(MODEL_DIR))
        || krea2_pipeline::is_snapshot(&dir.join(KREA2_DIR))
        || qwen_image_rs::comfy_files::find_qwen21(dir).is_some()
        || qwen_image_rs::comfy_files::find_krea2(dir).is_some()
        || qwen_image_rs::layered_files::LayeredFiles::resolve(dir).is_ok()
        || qwen_image_rs::edit2511_files::Edit2511Files::resolve(dir).is_ok()
        || moss::find_files(dir, MossVariant::Sft).is_some()
        || moss::find_files(dir, MossVariant::Base).is_some()
        || moss::find_files(dir, MossVariant::Kromoss).is_some()
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
            preview: Default::default(),
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

    #[test]
    fn style_template_slot_and_availability() -> anyhow::Result<()> {
        assert_eq!(task_of(TEMPLATE_KREA2_STYLE), Some("image"));
        assert_eq!(family_of(TEMPLATE_KREA2_STYLE), Some(Family::Krea2));
        let mut params = BTreeMap::new();
        assert_eq!(with_style_lora(&params).and_then(|p| lora_asks(&p)), Ok(vec![(KREA2_STYLE_LORA.to_string(), 1.0)]));
        params.insert("models".into(), json!({"lora":"another.safetensors"}));
        params.insert("loras".into(), json!([{"file":"another.safetensors","strength":0.2},{"file":"extra.safetensors","strength":0.5}]));
        assert_eq!(with_style_lora(&params).and_then(|p| lora_asks(&p)), Ok(vec![("another.safetensors".into(), 1.0), ("extra.safetensors".into(), 0.5)]));
        let root = std::env::temp_dir().join(format!("gn-krea-style-{}", std::process::id()));
        let snapshot = root.join(KREA2_DIR);
        for dir in ["tokenizer", "text_encoder", "transformer", "vae"] {
            std::fs::create_dir_all(snapshot.join(dir))?;
        }
        std::fs::write(snapshot.join("tokenizer/tokenizer.json"), "{}")?;
        let mut engine = DiffusionEngine::new(root.clone(), false);
        engine.refused = None;
        let installed = engine.index("")["installed"].to_string();
        assert!(installed.contains(TEMPLATE_KREA2));
        assert!(!installed.contains(TEMPLATE_KREA2_STYLE));
        std::fs::create_dir_all(root.join(LORA_DIR))?;
        std::fs::write(root.join(LORA_DIR).join(KREA2_STYLE_LORA), [])?;
        assert!(engine.index("")["installed"].to_string().contains(TEMPLATE_KREA2_STYLE));
        // Only this test's explicitly created temporary directory is removed.
        std::fs::remove_file(root.join(LORA_DIR).join(KREA2_STYLE_LORA))?;
        std::fs::remove_file(snapshot.join("tokenizer/tokenizer.json"))?;
        for dir in ["tokenizer", "text_encoder", "transformer", "vae"] {
            std::fs::remove_dir(snapshot.join(dir))?;
        }
        std::fs::remove_dir(snapshot)?;
        std::fs::remove_dir(root.join(LORA_DIR))?;
        std::fs::remove_dir(root)?;
        Ok(())
    }

    /// A warm-up loads the template's family and generates nothing; without the files it says so.
    #[test]
    fn a_warmup_loads_and_generates_nothing() {
        let mut e = DiffusionEngine::new(PathBuf::from("Z:/nowhere/models"), false);
        e.refused = None;
        let mut w = req(TEMPLATE, "image");
        w.prompt = String::new();
        w.params.insert("warmup".into(), json!(true));
        let r = e.run(&w, &[], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::ModelMissing, .. })), "{r:?}");
    }

    /// A batch is a 2.1 edit's seeds, one per picture, checked before the models load.
    #[test]
    fn a_batch_names_a_seed_for_each_picture() {
        let mut e = DiffusionEngine::new(PathBuf::from("Z:/nowhere/models"), false);
        e.refused = None;
        let pic = || Blob { width: 2, height: 1, channels: 4, index: 0 };
        let mut b = req(TEMPLATE_EDIT, "edit");
        b.image = Some(pic());
        b.references = vec![Blob { index: 1, ..pic() }, Blob { index: 2, ..pic() }];
        let blobs = [vec![0; 8], vec![0; 8], vec![0; 8]];
        let run = |e: &mut DiffusionEngine, b: &RunRequest| e.run(b, &blobs, &mut |_, _| {}, &AtomicBool::new(false));
        b.params.insert("batch".into(), json!([1, 2]));
        assert!(matches!(&run(&mut e, &b), Err(EngineError { kind: ErrorKind::BadRequest, message }) if message.contains("3 given")));
        b.params.insert("batch".into(), json!([1, 2, "x"]));
        assert!(matches!(&run(&mut e, &b), Err(EngineError { kind: ErrorKind::BadRequest, .. })));
        b.params.insert("batch".into(), json!([1, 2, 3]));
        let r = run(&mut e, &b);
        assert!(matches!(r, Err(EngineError { kind: ErrorKind::ModelMissing, .. })), "{r:?}");
        let mut f = req(TEMPLATE_FILL, "fill");
        f.params.insert("batch".into(), json!([1]));
        assert!(matches!(&run(&mut e, &f), Err(EngineError { kind: ErrorKind::BadRequest, message }) if message.contains("batch")));
    }

    /// MOSS-Image's templates: text to image only, found by Comfy-Org's file names (a
    /// subfolder deep), a typed error naming the files when they are not there.
    #[test]
    fn moss_templates_find_their_files() -> anyhow::Result<()> {
        assert_eq!(task_of(TEMPLATE_MOSS), Some("image"));
        assert_eq!(family_of(TEMPLATE_MOSS_BASE), Some(Family::Moss(MossVariant::Base)));
        let root = std::env::temp_dir().join(format!("gn-moss-{}", std::process::id()));
        let mut e = DiffusionEngine::new(root.clone(), false);
        e.refused = None;
        let r = e.run(&req(TEMPLATE_MOSS, "image"), &[], &mut |_, _| {}, &AtomicBool::new(false));
        assert!(matches!(&r, Err(EngineError { kind: ErrorKind::ModelMissing, message }) if message.contains("MOSS-Image")), "{r:?}");
        let files = [
            "diffusion_models/moss/moss_image_sft_int8_convrot.safetensors",
            "text_encoders/qwen3.5_9b_int8_convrot.safetensors",
            "text_encoders/qwen35_tokenizer/tokenizer.json",
            "vae/flux2-vae_small.safetensors",
        ];
        for f in files {
            let path = root.join(f);
            std::fs::create_dir_all(path.parent().ok_or_else(|| anyhow::anyhow!("no parent"))?)?;
            std::fs::write(path, [])?;
        }
        let installed = e.index("")["installed"].to_string();
        assert!(installed.contains(TEMPLATE_MOSS) && !installed.contains(TEMPLATE_MOSS_BASE), "{installed}");
        assert!(has_models(&root));
        // Only this test's explicitly created temporary directory is removed.
        for f in files {
            std::fs::remove_file(root.join(f))?;
        }
        for d in ["diffusion_models/moss", "diffusion_models", "text_encoders/qwen35_tokenizer", "text_encoders", "vae", ""] {
            std::fs::remove_dir(root.join(d))?;
        }
        Ok(())
    }

    /// A Krea 2 edit runs with its edit LoRA in front of the request's own, once.
    #[test]
    fn a_krea2_edit_adds_its_lora_once() {
        let mut p = BTreeMap::new();
        let first = |p: &BTreeMap<String, Value>| p["loras"][0]["file"].as_str().map(str::to_string);
        assert_eq!(first(&with_edit_lora(&p)).as_deref(), Some(KREA2_EDIT_LORA));
        p.insert("loras".into(), json!([{"file": "krea2_anygles_rank32.safetensors", "strength": 0.8}]));
        let out = with_edit_lora(&p);
        assert_eq!(out["loras"].as_array().map(Vec::len), Some(2));
        assert_eq!(first(&out).as_deref(), Some(KREA2_EDIT_LORA));
        p.insert("loras".into(), json!([{"file": format!("C:/loras/{KREA2_EDIT_LORA}"), "strength": 0.6}]));
        assert_eq!(with_edit_lora(&p)["loras"].as_array().map(Vec::len), Some(1), "named already: kept at its own strength");
        assert_eq!(task_of(TEMPLATE_KREA2_EDIT), Some("edit"));
        assert_eq!(family_of(TEMPLATE_KREA2_EDIT), Some(Family::Krea2));
    }

    #[test]
    fn a_chat_names_krea2s_encoder_or_none() {
        let mut p = BTreeMap::new();
        assert!(!chat_on_krea2(&p).unwrap());
        p.insert("encoder".into(), json!(CHAT_ON_KREA2));
        assert!(chat_on_krea2(&p).unwrap());
        p.insert("encoder".into(), Value::Null);
        assert!(!chat_on_krea2(&p).unwrap());
        for bad in [json!("qwen"), json!(3), json!(["krea2"])] {
            p.insert("encoder".into(), bad);
            assert!(chat_on_krea2(&p).is_err());
        }
    }

    #[test]
    fn a_krea2_fill_runs_anypaint_and_reads_denoise_and_seam() {
        let mut p = BTreeMap::new();
        assert_eq!(with_lora_first(&p, KREA2_ANYPAINT_LORA)["loras"][0]["file"].as_str(), Some(KREA2_ANYPAINT_LORA));
        assert_eq!((task_of(TEMPLATE_KREA2_FILL), family_of(TEMPLATE_KREA2_FILL)), (Some("fill"), Some(Family::Krea2)));
        let o = krea2_fill_options(&p).unwrap();
        assert_eq!((o.denoise, o.seam_px), (1.0, 32));
        p.insert("denoise".into(), json!(0.7));
        p.insert("seam".into(), json!(24));
        let o = krea2_fill_options(&p).unwrap();
        assert_eq!((o.denoise, o.seam_px), (0.7, 24));
        p.insert("denoise".into(), json!(0.0));
        assert!(krea2_fill_options(&p).is_err());
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
        let params: BTreeMap<String, Value> =
            [("ref_x", 20u64), ("ref_y", 0), ("ref_w", 30), ("ref_h", 48)].into_iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        let p = placed_picture(&canvas, &params).unwrap();
        // 20..50 widened to 16..64: 48 wide, placed at x = 16.
        assert_eq!((p.width, p.height, p.offset), (48, 48, Some((16, 0))));
        assert_eq!((p.rgba[0], p.rgba[4 * 47]), (16, 63), "the first row runs from column 16 to 63");
        assert!(placed_picture(&canvas, &BTreeMap::new()).is_err(), "no place named");
        let off: BTreeMap<String, Value> =
            [("ref_x", 70u64), ("ref_y", 0), ("ref_w", 10), ("ref_h", 10)].into_iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        assert!(placed_picture(&canvas, &off).is_err(), "outside the canvas");
    }

    #[test]
    fn an_outpaint_runs_its_lora_first_once() {
        assert_eq!((task_of(TEMPLATE_OUTPAINT), family_of(TEMPLATE_OUTPAINT)), (Some("expand"), Some(Family::QwenImage21)));
        let mut p = BTreeMap::new();
        p.insert("loras".to_string(), json!([{"file": "style.safetensors", "strength": 0.5}]));
        let out = with_lora_first(&p, QWEN21_OUTPAINT_LORA);
        assert_eq!(out["loras"][0]["file"].as_str(), Some(QWEN21_OUTPAINT_LORA));
        assert_eq!(out["loras"][1]["file"].as_str(), Some("style.safetensors"));
        // Named by the request at its own strength: not added again.
        p.insert("loras".to_string(), json!([{"file": format!("loras/{QWEN21_OUTPAINT_LORA}"), "strength": 0.8}]));
        let out = with_lora_first(&p, QWEN21_OUTPAINT_LORA);
        assert_eq!(out["loras"].as_array().map(Vec::len), Some(1));
        assert_eq!(out["loras"][0]["strength"].as_f64(), Some(0.8));
        // The Turbo template stacks both, the Outpaint LoRA first.
        let both = with_lora_first(&with_lora_first(&BTreeMap::new(), QWEN21_TURBO_LORA), QWEN21_OUTPAINT_LORA);
        assert_eq!((both["loras"][0]["file"].as_str(), both["loras"][1]["file"].as_str()), (Some(QWEN21_OUTPAINT_LORA), Some(QWEN21_TURBO_LORA)));
        assert_eq!(task_of(TEMPLATE_OUTPAINT_TURBO), Some("expand"));
    }

    #[test]
    fn flags_and_sizes() {
        assert!(from_args(&["--runtime".into(), "a".into()]).is_none());
        let e = from_args(&["--runtime".into(), "a".into(), "--models".into(), "m".into(), "--text-quant".into()]).unwrap();
        assert!(e.quant_text);
        assert_eq!(e.snapshot_dir(), PathBuf::from("m").join(MODEL_DIR));
        assert_eq!(e.krea2_dir(), PathBuf::from("m").join(KREA2_DIR));
        assert_eq!((side(0), side(1024), side(1000), side(5)), (1024, 1024, 992, 16));
        assert_eq!(
            (task_of(TEMPLATE), task_of(TEMPLATE_EDIT), task_of(TEMPLATE_FILL), task_of("krea2-turbo/image")),
            (Some("image"), Some("edit"), Some("fill"), Some("image"))
        );
        assert_eq!((task_of(TEMPLATE_EXPAND), task_of(TEMPLATE_MATTE), task_of("nope/x")), (Some("expand"), Some("matte"), None));
        assert_eq!((family_of(TEMPLATE_KREA2), family_of(TEMPLATE_FILL), family_of("nope/x")), (Some(Family::Krea2), Some(Family::QwenImage21), None));
        assert_eq!(
            (task_of(TEMPLATE_CHAT_EDIT), task_of(TEMPLATE_CHAT_IMAGE), family_of(TEMPLATE_CHAT_EDIT)),
            (Some("text"), Some("text"), Some(Family::QwenImage21))
        );
        assert!(is_sticky_cuda_error(r#"DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was encountered")"#));
        assert!(
            !is_sticky_cuda_error("DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"),
            "out of memory is recoverable: the models are dropped instead"
        );
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

#[cfg(test)]
mod edit2511_tests {
    use super::*;
    #[test]
    fn graph_defaults_order_and_routing() {
        use qwen_image_rs::edit2511_files::ADAPTER_NAMES as A;
        let templates: Vec<_> = TEMPLATES.iter().filter(|(_, _, f)| *f == Family::Edit2511).collect();
        assert_eq!(templates.len(), 13);
        for (id, task, _) in templates {
            assert_eq!(task_of(id), Some(*task));
            assert!(edit2511_defaults(id).is_some());
        }
        assert_eq!(edit2511_defaults("qwen-edit-2511/angles"), Some((4, 1., vec![A[3], A[0]])));
        assert_eq!(edit2511_defaults("qwen-edit-2511/remove-lightning-8"), Some((8, 1., vec![A[1], A[2]])));
        assert_eq!(edit2511_defaults("qwen-edit-2511/edit"), Some((40, 4., vec![])));
    }
    #[test]
    fn edit_rgb_inputs_preserve_coverage_and_ignore_alpha() -> anyhow::Result<()> {
        for (channels, bytes, expected) in [
            (1, vec![7, 11], vec![7, 7, 7, 11, 11, 11]),
            (3, vec![1, 2, 3, 7, 11, 13], vec![1, 2, 3, 7, 11, 13]),
            (4, vec![1, 2, 3, 0, 7, 11, 13, 128], vec![1, 2, 3, 7, 11, 13]),
        ] {
            let mut req: RunRequest = serde_json::from_value(json!({"model":"qwen-edit-2511/image-ref", "task":"image", "prompt":"sky", "seed":7}))?;
            req.references.push(Blob { width: 2, height: 1, channels, index: 0 });
            let ask = edit2511_request(&req, &[bytes]).map_err(|e| anyhow::anyhow!(e.message))?;
            assert_eq!(ask.references[0].as_raw(), &expected);
            assert!(edit2511_request(&req, &[vec![]]).is_err());
        }
        Ok(())
    }

    #[test]
    fn graph_inputs_preserve_targets_and_prune_reference_slots() -> anyhow::Result<()> {
        let blob = Blob { width: 32, height: 48, channels: 3, index: 0 };
        let input = vec![(0..32 * 48 * 3).map(|n| (n % 251) as u8).collect::<Vec<_>>(), vec![128u8; 32 * 48 * 3]];
        let mut req: RunRequest = serde_json::from_value(json!({"model":"qwen-edit-2511/edit", "task":"edit", "prompt":"sky", "seed":7}))?;
        req.image = Some(blob.clone());
        req.references = vec![blob.clone(), blob.clone()];
        let ask = edit2511_request(&req, &input).map_err(|e| anyhow::anyhow!(e.message))?;
        assert_eq!(ask.references.len(), 3);
        assert_eq!((ask.steps, ask.cfg), (40, 4.));
        assert_eq!(ask.target.as_ref().map(|im| im.dimensions()), Some((32, 48)));
        req.model = "qwen-edit-2511/angles".into();
        req.references.clear();
        let ask = edit2511_request(&req, &input).map_err(|e| anyhow::anyhow!(e.message))?;
        assert_eq!(ask.target.as_ref(), ask.references.first());
        req.model = "qwen-edit-2511/fill-guided".into();
        req.task = "fill".into();
        req.mask = Some(Blob { index: 1, ..blob });
        let ask = edit2511_request(&req, &input).map_err(|e| anyhow::anyhow!(e.message))?;
        assert_eq!(ask.references.len(), 2);
        assert_eq!(ask.mask.as_ref().map(|im| im.get_pixel(0, 0).0), Some([128, 128, 128]));
        req.params.insert("sampler".into(), json!("heun"));
        assert!(edit2511_request(&req, &input).is_err());
        Ok(())
    }
}
