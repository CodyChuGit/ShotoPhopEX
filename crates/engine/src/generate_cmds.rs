//! `generate.*` commands: generative edits through a local backend (`photocraft-genai`).
//!
//! `generate.fill` is Generative Fill. The flattened composite around the selection and the
//! selection's coverage go to the native worker, and
//! the result comes back as a **new raster layer above the active one, masked to the selection**.
//! The original pixels never change. The command is a background job (`jobs::edit_job`): progress
//! comes from the server, Esc cancels and interrupts it, and the edit lands as one undo step.
//! `generate.health` and `generate.models` are queries for the UI, the CLI and agents.
//!
//! Preferences read here: `integrations.defaultEditModel`,
//! `integrations.generativeTimeoutSecs`, `integrations.allowResearchModels`,
//! `integrations.enhancePrompts` (the default of the fill, edit and image commands' `enhance`),
//! `integrations.previewWhileGenerating` (the pictures a run shows while it samples: kept on the
//! job with the place its result takes, `jobs::JobPreview`, for the canvas to draw).
//! The native worker manages model precision and residency.

use std::collections::BTreeMap;
use std::sync::Arc;

use photocraft_doc::{Layer, LayerId};
use photocraft_genai::template;
use photocraft_genai::{GenerativeBackend, Gray8, Health, Request, Rgba8, Task};
use photocraft_geom::Rect;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, layer_param};
use crate::jobs::JobCtx;
use crate::{EngineError, Result, Session};

// The commands' parts, one file each. Their items are re-exported here, so every path
// (`generate_cmds::quality`, `super::fill_region`) stays as it was.
#[path = "generate_params.rs"]
mod params;
pub use params::*;
#[path = "generate_routing.rs"]
mod routing;
pub use routing::*;
#[path = "generate_native.rs"]
mod native;
pub(crate) use native::*;
#[path = "generate_vision.rs"]
mod vision;
pub use vision::*;
#[path = "generate_pixel_ops.rs"]
mod pixel_ops;
pub(crate) use pixel_ops::*;
#[path = "generate_image.rs"]
mod image_gen;
pub use image_gen::*;
#[path = "generate_fill.rs"]
mod fill;
pub(crate) use fill::*;
#[path = "generate_removal.rs"]
mod removal;
pub use removal::*;
#[path = "generate_expand.rs"]
mod expand;
pub(crate) use expand::*;
#[path = "generate_edit.rs"]
mod edit;
pub use edit::*;
#[path = "generate_matte.rs"]
mod matte;
pub(crate) use matte::*;

pub const FILL: &str = "generate.fill";
pub const IMAGE: &str = "generate.image";
pub const HEALTH: &str = "generate.health";
pub const MODELS: &str = "generate.models";
pub const VARIATION: &str = "generate.variation";
/// Delete one variation of a run (the Variations strip's ×).
pub const DROP_VARIATION: &str = "generate.dropVariation";
/// Keep one variation of a run and delete the others (the Variations strip's Keep).
pub const KEEP_VARIATION: &str = "generate.keepVariation";
pub const FREE: &str = "generate.free";
pub const EXPAND: &str = "generate.expand";
/// Most results one `generate.fill` call makes (`variations`).
pub const MAX_VARIATIONS: u32 = 4;
/// The template value (and preference) that picks a fill template by what the server has.
pub const AUTO_TEMPLATE: &str = "auto";
/// `auto` tries these in order and takes the first whose model files the server lists: the
/// Lightning tier (8 steps, Apache-2.0 LoRA) when its LoRA is installed, else the 40-step base.
pub const AUTO_FILL_ORDER: &[&str] = &["qwen-edit-2511/fill-lightning-8", DEFAULT_FILL_TEMPLATE];
/// `auto` under Preferences › Quality `fast`: the 4-step tier first.
pub const FAST_FILL_ORDER: &[&str] = &["qwen-edit-2511/fill-lightning-4", "qwen-edit-2511/fill-lightning-8", DEFAULT_FILL_TEMPLATE];
/// `auto` under Quality `best`: the full-step base template only.
pub const BEST_FILL_ORDER: &[&str] = &[DEFAULT_FILL_TEMPLATE];
/// The Mac's fill (FLUX 3 Image Klein on MLX), picked by the Mac memory profiles.
pub const KLEIN_FILL_TEMPLATE: &str = "flux3-klein/fill";
pub const BEST_EXPAND_ORDER: &[&str] = &[DEFAULT_EXPAND_TEMPLATE];
/// The same for Generative Expand, whose templates show the model only the picture as a
/// reference while the padded canvas is the sampling latent.
pub const DEFAULT_EXPAND_TEMPLATE: &str = "qwen-edit-2511/expand";
pub const AUTO_EXPAND_ORDER: &[&str] = &["qwen-edit-2511/expand-lightning-8", DEFAULT_EXPAND_TEMPLATE];
/// A fill request is downscaled to this many pixels before it is sent (the editing models work
/// at about one megapixel, Photoshop's Generative Fill renders at most 1024 px on a side) and
/// the result is resampled back; the layer mask keeps the selection's full-resolution edge.
/// 1 MP (2026-10-21, `tune-run.ps1`): a small consistent gain in fine texture over 0.75 MP
/// for 15 % more time, the server still resident on the Lightning tier (the 0.75 MP cap came
/// from a 1 MP run that pushed the 40-step tier into offloading, 45 s instead of 18 s, in
/// 2026-10); 1.5 MP put a seam in a lawn. `maxPixels` on the fill commands overrides it.
const MAX_FILL_REQUEST_PIXELS: u64 = 1024 * 1024;
/// The pixels a P-H3-Image or MiniMax-H3-Image edit is sent at: the model edits at 4 MP (its maker's and AusBoss's
/// guide's working size, the native pipeline's own default), where the other editing models
/// work at one.
const H3_EDIT_REQUEST_PIXELS: u64 = 2048 * 2048;
/// The pixels an edit on `template` is downscaled to before it is sent. Under Quality Ultra
/// (`ultra`) the 16 MP model edits a large picture at its own size, up to 16 MP.
pub(crate) fn edit_request_pixels(template: &str, ultra: bool) -> u64 {
    match template {
        "minimax-h3-image/edit" if ultra => MAX_REQUEST_PIXELS,
        "p-h3-image/edit" | "minimax-h3-image/edit" => H3_EDIT_REQUEST_PIXELS,
        _ => MAX_FILL_REQUEST_PIXELS,
    }
}
/// Seeds stay below 2^53 so they survive a round trip through JSON numbers.
const MAX_SEED: u64 = 1 << 53;
pub const DEFAULT_FILL_TEMPLATE: &str = "qwen-edit-2511/fill";
/// A fill given reference images with no template named runs these (the plain fill templates
/// take none): the 8-step tier, or the 40-step base under Quality `best`.
pub const DEFAULT_FILL_REF_TEMPLATE: &str = "qwen-edit-2511/fill-ref-lightning-8";
pub const BEST_FILL_REF_TEMPLATE: &str = "qwen-edit-2511/fill-ref";
/// Most pixels `generate.fill`'s `grow` may widen the selection by.
pub const MAX_GROW: u32 = 512;
pub const DEFAULT_IMAGE_TEMPLATE: &str = "krea2-turbo/image";
/// Text-to-image sizes are rounded down to this grid (latent patches); the smallest side allowed.
const SIZE_STEP: u32 = 16;
const MIN_SIDE: u32 = 64;
const MAX_SIDE: u32 = 4096;
/// Largest request image sent to a backend; the selection plus its margin is refused above it.
const MAX_REQUEST_PIXELS: u64 = 16 * 1024 * 1024;
const MAX_PROMPT_CHARS: usize = 4000;

pub(crate) const INFO: &str = "generate.info";
pub(crate) const SIMILAR: &str = "generate.similar";
pub(crate) const SPLIT: &str = "generate.splitLayers";
/// Split into Layers asks for at most this many layers (the background and the subjects): a
/// café table of seven people, three cups, the cards and the table is twelve subjects already.
const MAX_SPLIT_LAYERS: u32 = 20;

/// The PSD additional-layer-info key a generative layer keeps its [`GenerativeInfo`] under
/// (JSON). An unmodelled block, so it survives PSD round trips and `Layer::duplicate`.
pub const GENERATIVE_BLOCK: [u8; 4] = *b"cpGn";

/// What a generative layer remembers about the run that made it: `generate.info` reads it,
/// `generate.similar` re-runs it with a new seed.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GenerativeInfo {
    /// `generate.fill`, `generate.expand`, `generate.edit` or `generate.image`.
    pub command: String,
    /// The prompt as typed (the template's wrapper is applied again on a re-run).
    pub prompt: String,
    pub negative: String,
    /// The template that ran (`auto` resolved).
    pub template: String,
    pub seed: u64,
    pub steps: u32,
    pub guidance: f32,
    pub edge: String,
    /// The request rectangle on the canvas: x, y, width, height.
    pub rect: [i32; 4],
    pub name: String,
    /// `generate.image` only: the size asked for and whether transparency was.
    pub width: u32,
    pub height: u32,
    pub transparent: bool,
    /// The prompt the model actually got when the vision-language model rewrote it (`enhance`);
    /// empty when it ran as typed.
    pub enhanced: String,
    /// The reference images the run had, as the `references` specs it was given
    /// (`{"layer"}`, `{"path"}`, `{"job", "index"}`); Generate Similar resolves them again.
    pub references: Vec<Value>,
    /// Fills: how many pixels the selection was widened by before the request (`grow`).
    pub grow: u32,
    /// Fills: the selection was flattened to grey in the request image (`erase`).
    pub erase: bool,
}

/// Remember `info` on `layer` (replacing an earlier one).
pub(crate) fn set_generative_info(layer: &mut Layer, info: &GenerativeInfo) {
    layer.psd_blocks.retain(|(k, _)| *k != GENERATIVE_BLOCK);
    if let Ok(bytes) = serde_json::to_vec(info) {
        layer.psd_blocks.push((GENERATIVE_BLOCK, Arc::new(bytes)));
    }
}

/// The run that made `layer`, if a generative command did.
pub fn generative_info(layer: &Layer) -> Option<GenerativeInfo> {
    layer.psd_blocks.iter().find(|(k, _)| *k == GENERATIVE_BLOCK).and_then(|(_, d)| serde_json::from_slice(d).ok())
}

pub(crate) const EDIT: &str = "generate.edit";
/// The picture seen from another camera (the Multiple Angles LoRA), as an edit.
pub const ANGLES: &str = "generate.angles";
/// Character Sheet: four views of the character on a layer, as a new document.
pub const CHARACTER_SHEET: &str = "generate.characterSheet";
/// The template `generate.angles` runs.
pub const ANGLES_TEMPLATE: &str = "qwen-edit-2511/angles";
/// `generate.angles` `rotation` (degrees around the subject, 0 = from the front) and the Multiple
/// Angles LoRA's word for it.
pub const ANGLE_ROTATIONS: [(u32, &str); 8] = [
    (0, "front view"),
    (45, "front-right quarter view"),
    (90, "right side view"),
    (135, "back-right quarter view"),
    (180, "back view"),
    (225, "back-left quarter view"),
    (270, "left side view"),
    (315, "front-left quarter view"),
];
/// `tilt` (the camera's height in degrees) and its words.
pub const ANGLE_TILTS: [(i32, &str); 4] = [(-30, "low-angle shot"), (0, "eye-level shot"), (30, "elevated shot"), (60, "high-angle shot")];
/// `zoom` (0 = wide, 10 = closest) and its words.
pub const ANGLE_ZOOMS: [(u32, &str); 3] = [(0, "wide shot"), (5, "medium shot"), (10, "close-up")];

/// Maps the backend's progress onto the job's bar: the `lo..hi` part of it, after the render step.
pub(crate) struct JobProgress<'a> {
    ctx: &'a JobCtx,
    lo: f32,
    hi: f32,
}

impl<'a> JobProgress<'a> {
    /// One backend run filling the bar from 5 % to 95 %.
    pub(crate) fn new(ctx: &'a JobCtx) -> Self {
        Self::span(ctx, 0.05, 0.95)
    }
    /// The part of the bar one of several backend runs occupies.
    pub(crate) fn span(ctx: &'a JobCtx, lo: f32, hi: f32) -> Self {
        Self { ctx, lo, hi }
    }
}

impl photocraft_genai::Progress for JobProgress<'_> {
    fn report(&self, fraction: f32, message: &str) {
        self.ctx.progress(self.lo + (self.hi - self.lo) * fraction.clamp(0.0, 1.0), message);
    }
    fn cancelled(&self) -> bool {
        self.ctx.cancelled()
    }
    /// The job keeps the latest picture for the canvas, over the place the work closure named
    /// with [`JobCtx::preview_at`].
    fn preview(&self, p: &photocraft_genai::Preview) {
        self.ctx.show_preview(p.image.width, p.image.height, p.image.data.clone(), p.step, p.steps, p.output);
    }
}

/// Are the native worker's folders set? The
/// classical tools go through the model only then, and fall back to their own algorithms
/// otherwise (a server that is set up but down is reported by the job instead).
pub fn generative_configured(s: &Session) -> bool {
    s.prefs().integrations.native_configured()
}

/// The variation commands' `layers` (1 to 16 ids, all in the active document) and 1-based
/// `index`.
fn variation_params(s: &Session, cmd: &str, p: &Value) -> Result<(Vec<LayerId>, usize)> {
    let bad_layers = || bad(cmd, "`layers` must be a list of 1 to 16 layer ids");
    let layers: Vec<LayerId> = p
        .get("layers")
        .and_then(Value::as_array)
        .ok_or_else(bad_layers)?
        .iter()
        .map(|v| v.as_u64().map(LayerId).ok_or_else(bad_layers))
        .collect::<Result<_>>()?;
    if layers.is_empty() || layers.len() > 16 {
        return Err(bad_layers());
    }
    let index = match opt_num(cmd, p, "index", 1.0, layers.len() as f64)? {
        Some(x) if x.fract() == 0.0 => x as usize,
        Some(_) => return Err(bad(cmd, "`index` must be a whole number (1 = the first variation)")),
        None => return Err(bad(cmd, "`index` is required (1 = the first variation)")),
    };
    let d = s.active().ok_or(EngineError::NoDocument)?;
    if let Some(missing) = layers.iter().find(|id| d.doc.layer(**id).is_none()) {
        return Err(EngineError::NoLayer(*missing));
    }
    if layers.iter().enumerate().any(|(i, id)| layers.iter().take(i).any(|o| o == id)) {
        return Err(bad(cmd, "`layers` lists a layer twice"));
    }
    Ok((layers, index))
}

/// `generate.variation`: show one of the layers a fill made as variations, hide the others.
fn run_variation(s: &mut Session, p: &Value) -> Result<Value> {
    let (layers, index) = variation_params(s, VARIATION, p)?;
    let shown = layers.get(index.saturating_sub(1)).copied();
    s.edit(&format!("Variation {index}"), |doc, active| {
        for (i, id) in layers.iter().enumerate() {
            doc.layer_mut(*id).ok_or(EngineError::NoLayer(*id))?.visible = i + 1 == index;
        }
        *active = shown;
        Ok(())
    })?;
    Ok(json!({"shown": shown.map(|l| l.0), "index": index, "count": layers.len()}))
}

/// `generate.dropVariation`: delete the index-th variation, as one undo step. When it was the
/// one on screen the next one (else the previous one) is shown and becomes the active layer.
fn run_drop_variation(s: &mut Session, p: &Value) -> Result<Value> {
    let (layers, index) = variation_params(s, DROP_VARIATION, p)?;
    let Some(dropped) = layers.get(index.saturating_sub(1)).copied() else { return Err(bad(DROP_VARIATION, "`index` is past the end of `layers`")) };
    let rest: Vec<LayerId> = layers.iter().copied().filter(|l| *l != dropped).collect();
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let was_shown = d.doc.layer(dropped).is_some_and(|l| l.visible);
    let others_hidden = rest.iter().all(|id| d.doc.layer(*id).is_some_and(|l| !l.visible));
    // The dropped one was on screen and none of the others is: show its neighbour.
    let next = if was_shown && others_hidden { rest.get(index.saturating_sub(1)).or(rest.last()).copied() } else { None };
    s.edit(&format!("Drop Variation {index}"), |doc, active| {
        doc.remove(dropped).ok_or(EngineError::NoLayer(dropped))?;
        if doc.layers.is_empty() {
            return Err(EngineError::Other("a document must keep at least one layer".into()));
        }
        if let Some(n) = next {
            doc.layer_mut(n).ok_or(EngineError::NoLayer(n))?.visible = true;
            *active = Some(n);
        } else if *active == Some(dropped) {
            *active = rest.first().copied().or_else(|| doc.top_layer());
        }
        Ok(())
    })?;
    let shown = rest.iter().copied().find(|id| s.active().and_then(|d| d.doc.layer(*id)).is_some_and(|l| l.visible));
    Ok(json!({"dropped": dropped.0, "layers": rest.iter().map(|l| l.0).collect::<Vec<_>>(), "shown": shown.map(|l| l.0)}))
}

/// `generate.keepVariation`: keep the index-th variation, shown and active, and delete the
/// others, as one undo step.
fn run_keep_variation(s: &mut Session, p: &Value) -> Result<Value> {
    let (layers, index) = variation_params(s, KEEP_VARIATION, p)?;
    let Some(kept) = layers.get(index.saturating_sub(1)).copied() else { return Err(bad(KEEP_VARIATION, "`index` is past the end of `layers`")) };
    let removed: Vec<LayerId> = layers.iter().copied().filter(|l| *l != kept).collect();
    s.edit("Keep Variation", |doc, active| {
        for id in &removed {
            doc.remove(*id).ok_or(EngineError::NoLayer(*id))?;
        }
        doc.layer_mut(kept).ok_or(EngineError::NoLayer(kept))?.visible = true;
        *active = Some(kept);
        Ok(())
    })?;
    Ok(json!({"kept": kept.0, "removed": removed.iter().map(|l| l.0).collect::<Vec<_>>()}))
}

fn split_enabled(s: &Session) -> std::result::Result<(), String> {
    web_unavailable()?;
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// `generate.splitLayers`: SAM 3.1 subjects, each on its own layer, over a background the
/// generative removal repaints ([`split_subjects`]). Qwen-Image-Layered split only 5 of 10
/// seeds of the owner's portrait whatever the sampler settings; this did 5 of 5 in a third of
/// the time, and the owner chose it alone (2026-11-02).
fn run_split(s: &mut Session, p: &Value) -> Result<Value> {
    s.active().ok_or(EngineError::NoDocument)?;
    // `simple` (the default): progressive removal, nearest first, until the detector finds
    // nothing; `classic`: the rule-checked peel (and `sweep`, `keepTries`, `complete`).
    let simple = match opt_str(SPLIT, p, "method", 40)?.map(str::trim) {
        None | Some("") | Some("simple") => true,
        Some("classic") => false,
        Some(other) => return Err(bad(SPLIT, format!("`method` must be simple or classic (got `{other}`)"))),
    };
    // No `layers`: the simple method takes as many as it finds (the dialog no longer asks;
    // `detail` decides), the classic one three.
    let layers = match opt_num(SPLIT, p, "layers", 2.0, f64::from(MAX_SPLIT_LAYERS))? {
        None if simple => MAX_SPLIT_LAYERS as usize,
        None => 3,
        Some(x) if x.fract() == 0.0 => x as usize,
        Some(_) => return Err(bad(SPLIT, format!("`layers` must be a whole number from 2 to {MAX_SPLIT_LAYERS} (the background and at least one subject)"))),
    };
    let prompt = opt_str(SPLIT, p, "prompt", MAX_PROMPT_CHARS)?.map(str::trim).filter(|t| !t.is_empty()).map(str::to_string);
    let seed = match p.get("seed") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_u64().ok_or_else(|| bad(SPLIT, "`seed` must be a whole number"))?),
    };
    let name = opt_str(SPLIT, p, "name", 200)?.map(str::to_string);
    let edges = split_subjects::Edges::parse(opt_str(SPLIT, p, "edges", 40)?).map_err(|e| bad(SPLIT, e))?;
    let complete = opt_bool(SPLIT, p, "complete", true)?;
    let sweep = opt_bool(SPLIT, p, "sweep", false)?;
    let keep_tries = opt_bool(SPLIT, p, "keepTries", false)?;
    let order = split_subjects::Order::parse(opt_str(SPLIT, p, "order", 40)?).map_err(|e| bad(SPLIT, e))?;
    let group = split_subjects::Group::parse(opt_str(SPLIT, p, "group", 40)?).map_err(|e| bad(SPLIT, e))?;
    let go = if simple { split_subjects::simple::run } else { split_subjects::run };
    go(
        s,
        split_subjects::SubjectsPlan {
            prompt,
            subjects: (layers - 1).min(split_subjects::MAX_SUBJECTS),
            seed,
            name,
            edges,
            complete,
            sweep,
            keep_tries,
            order,
            group,
            first_pass_only: opt_bool(SPLIT, p, "firstPassOnly", false)?,
            detail: split_subjects::Detail::parse(opt_str(SPLIT, p, "detail", 40)?).map_err(|e| bad(SPLIT, e))?,
        },
    )
}

/// Edit › Purge › Generative Models: the server unloads its models and frees their memory. A
/// server that has had several model families loaded can end up streaming weights on every run
/// (2 GB of VRAM free, fills three times slower); this resets it, at the price of one reload.
fn run_free(s: &mut Session, _p: &Value) -> Result<Value> {
    backend(s)?.free().map_err(gen_err)?;
    Ok(json!({"freed": true}))
}

/// `generate.health [{template}]`: the backend the preferences point at, or, with a template id,
/// the one that template's requests would go to (under `auto` the native worker may serve some).
fn run_health(s: &mut Session, p: &Value) -> Result<Value> {
    let chosen = match p.get("template").and_then(Value::as_str).filter(|t| !t.trim().is_empty()) {
        Some(t) => backend_for(s, t).map(|(b, _)| b),
        None => backend(s),
    };
    let h = match chosen {
        Ok(b) => b.health(),
        Err(e) => Health { backend: "native".into(), error: Some(e.to_string()), ..Health::default() },
    };
    serde_json::to_value(&h).map_err(|e| EngineError::Other(e.to_string()))
}

fn run_models(s: &mut Session, p: &Value) -> Result<Value> {
    let allow_research = s.prefs().integrations.allow_research_models;
    // An area and a count make every template report its time estimate for them (`eta`).
    let eta_for = match (opt_num(MODELS, p, "width", 1.0, 65_536.0)?, opt_num(MODELS, p, "height", 1.0, 65_536.0)?) {
        (Some(w), Some(h)) => Some((
            w as u32,
            h as u32,
            opt_num(MODELS, p, "count", 1.0, 64.0)?.map_or(1, |c| c as u32),
            opt_num(MODELS, p, "references", 0.0, 16.0)?.map_or(0, |c| c as u32),
        )),
        (None, None) => None,
        _ => return Err(bad(MODELS, "give both `width` and `height` (or neither)")),
    };
    // `probe: false` lists the templates without contacting the server (the UI's picker).
    let probe = match p.get("probe") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(bad(MODELS, "`probe` must be true or false")),
    };
    // `async: true` probes on a worker (the picker's installed badges): a background job whose
    // result is this command's usual answer.
    let q = auto_quality(s);
    if probe && opt_bool(MODELS, p, "async", false)? {
        let backend = backend(s)?;
        let picks = native21_picks(s);
        return crate::jobs::run(
            s,
            "Checking generative models",
            false,
            move |ctx| {
                ctx.progress(0.2, "Asking the server");
                let mut v = models_json(Some(backend.as_ref()), allow_research, q, eta_for);
                apply_picks(&mut v, &picks);
                Ok(v)
            },
            move |_, v| Ok(v),
        );
    }
    let backend = if probe { backend(s).ok() } else { None };
    let mut v = models_json(backend.as_deref(), allow_research, q, eta_for);
    apply_picks(&mut v, &native21_picks(s));
    Ok(v)
}

/// `generate.models`' `autoFill`, `autoExpand`, `autoEdit` and `autoRemove` where `auto` runs
/// Qwen-Image-2.1 on the native worker instead of the server's order ([`native21_auto`]).
fn native21_picks(s: &Session) -> Vec<(&'static str, &'static str)> {
    [("autoFill", FILL, false), ("autoExpand", EXPAND, false), ("autoEdit", EDIT, false), ("autoRemove", FILL, true)]
        .into_iter()
        .filter_map(|(key, command, removal)| auto_shown(s, command, removal).map(|t| (key, t)))
        .collect()
}

fn apply_picks(v: &mut Value, picks: &[(&str, &str)]) {
    for (key, t) in picks {
        v[*key] = json!(t);
    }
}

/// `generate.models`' answer: every template with its model slots, capabilities and estimate
/// and, with a `backend`, what the server has installed and what `auto` would pick under the
/// Quality preference `q`.
fn models_json(backend: Option<&dyn GenerativeBackend>, allow_research: bool, q: &str, eta_for: Option<(u32, u32, u32, u32)>) -> Value {
    let health = backend.map(|b| b.health());
    let online = health.as_ref().is_some_and(|h| h.ok);
    let mut folders: BTreeMap<String, Option<Vec<String>>> = BTreeMap::new();
    // The native worker's index names the templates it serves rather than model files (as in
    // `resolve_auto`): a served template has every slot installed.
    let served: Vec<String> = if online { backend.and_then(|b| b.model_files("").ok()).unwrap_or_default() } else { Vec::new() };
    let mut templates = Vec::new();
    for t in template::builtin() {
        let m = &t.meta;
        let serves = served.iter().any(|s| s == &m.id);
        let mut slots = Vec::new();
        for slot in &m.models {
            let files = if online {
                folders.entry(slot.folder.clone()).or_insert_with(|| backend.and_then(|b| b.model_files(&slot.folder).ok())).clone()
            } else {
                None
            };
            let installed = files.as_ref().map(|f| serves || f.iter().any(|x| x == &slot.default));
            let v = json!({"placeholder": slot.placeholder, "folder": slot.folder, "file": slot.default, "installed": installed});
            slots.push(v);
        }
        let eta = eta_for.and_then(|(w, h, c, r)| estimate_seconds(&m.id, w, h, c, r));
        templates.push(json!({
            "id": m.id, "name": m.name, "family": m.family, "task": m.task, "license": m.license, "licenseNote": m.license_note,
            // Never gated on the licence (the owner's own files); `license` says which it is.
            "allowed": true,
            "defaults": {"steps": m.defaults.steps, "guidance": m.defaults.guidance},
            "needsImage": m.needs_image, "needsMask": m.needs_mask, "models": slots,
            "creator": m.creator, "description": m.description, "badges": m.badges, "references": m.references,
            "sizes": m.sizes, "counts": m.counts, "estimatedSeconds": m.estimate(), "pages": m.pages, "eta": eta,
        }));
    }
    // What `auto` would pick right now (needs the server's model lists).
    let auto = |order, fallback| backend.filter(|_| online).map(|b| resolve_auto(b, order, fallback));
    // Remove Background, as its job routes `auto` (generate_matte.rs): the detector refined by
    // ViTMatte when both are served, else the matte model with the research opt-in, else the
    // detector.
    let auto_matte = backend.filter(|_| online).map(|b| {
        let seg = crate::select_ml_cmds::DEFAULT_SEGMENT_TEMPLATE;
        let vit = crate::select_ml_cmds::VITMATTE_TEMPLATE;
        if resolve_auto(b, &[vit], "") == vit && resolve_auto(b, &[seg], "") == seg {
            seg.to_string()
        } else if allow_research {
            resolve_auto(b, &[DEFAULT_MATTE_TEMPLATE], crate::select_ml_cmds::DEFAULT_SEGMENT_TEMPLATE)
        } else {
            crate::select_ml_cmds::DEFAULT_SEGMENT_TEMPLATE.to_string()
        }
    });
    // The GPU, and whether the NVFP4 policy's `auto` applies to it.
    let device = health.as_ref().map(|h| h.device.clone()).unwrap_or_default();
    json!({
        "server": health, "templates": templates,
        "device": device,
        "quality": q,
        "autoFill": auto(auto_order(FILL, q), DEFAULT_FILL_TEMPLATE),
        "autoExpand": auto(auto_order(EXPAND, q), DEFAULT_EXPAND_TEMPLATE),
        "autoEdit": auto(auto_order(EDIT, q), DEFAULT_EDIT_TEMPLATE),
        "autoRemove": auto(auto_order(FILL, q), DEFAULT_FILL_TEMPLATE),
        "autoMatte": auto_matte,
    })
}

/// `generate.info`: what a layer remembers about the run that made it.
fn run_info(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let layer = d.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    let job = s.generations.record_for_layer(d.doc.id, id);
    Ok(json!({"layer": id.0, "generative": generative_info(layer), "job": job}))
}

fn similar_enabled(s: &Session) -> std::result::Result<(), String> {
    web_unavailable()?;
    let d = s.active().ok_or("no document open")?;
    let layer = crate::active_layer_of(s)?;
    let _ = d;
    if generative_info(layer).is_none() {
        return Err("the active layer was not made by a generative command".into());
    }
    Ok(())
}

/// `generate.similar`: run what made the layer again with a new seed, in the same place (the
/// layer's mask is the area), as a new layer above it.
fn run_similar(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let layer = d.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    let info = generative_info(layer).ok_or_else(|| EngineError::Other("the layer was not made by a generative command".into()))?;
    let seed = parse_seed(SIMILAR, p)?;
    let variations = parse_variations(SIMILAR, p)?;
    if info.command == SPLIT {
        return Err(EngineError::Other("a split layer has no \"similar\": split the picture again instead".into()));
    }
    if info.command == IMAGE {
        // A generated image: the same request (its references resolved again), as a layer
        // above this one.
        let q = json!({
            "prompt": info.prompt, "negative": info.negative, "template": info.template, "seed": seed, "steps": info.steps,
            "guidance": info.guidance, "target": "layer", "width": info.width, "height": info.height, "name": info.name,
            "transparent": info.transparent, "references": info.references, "variations": variations,
        });
        s.edit("Select Layer", |_, active| {
            *active = Some(id);
            Ok(())
        })?;
        return run_image(s, &q);
    }
    template::find(&info.template).map_err(|e| bad(SIMILAR, e.to_string()))?;
    let canvas = d.doc.bounds();
    let rect = Rect::new(info.rect[0], info.rect[1], info.rect[0].saturating_add(info.rect[2]), info.rect[1].saturating_add(info.rect[3])).intersect(&canvas);
    if rect.is_empty() {
        return Err(EngineError::Other("the layer's area is no longer on the canvas".into()));
    }
    // An expand layer's area is regenerated as a fill (its canvas is already there); edits and
    // fills run as what they were.
    let q = quality(s);
    let (command, auto, auto_order, template_id) = match (info.command.as_str(), native21_auto(s, FILL, false)) {
        (c, _) if c == EDIT => (EDIT, false, auto_order(EDIT, auto_quality(s)), info.template.clone()),
        (c, _) if c == FILL => (FILL, false, auto_order(FILL, q), info.template.clone()),
        (_, Some(t)) => (FILL, false, auto_order(FILL, q), t.to_string()),
        _ => (FILL, true, auto_order(FILL, q), DEFAULT_FILL_TEMPLATE.to_string()),
    };
    let refs = crate::generate_refs::Refs { images: crate::generate_refs::resolve(s, &info.references)?, specs: info.references.clone() };
    // The engine that serves the template (the native worker under `auto` for 2.1 and SAM).
    let backend = backend(s)?;
    let job = FillJob {
        thinker: Arc::clone(&backend),
        backend,
        command,
        template_id,
        auto,
        auto_order,
        prompt: info.prompt.clone(),
        negative: info.negative.clone(),
        seed,
        steps: info.steps,
        guidance: info.guidance,
        name: info.name.clone(),
        models: Vec::new(),
        variations,
        edge: if info.edge == "hard" { Edge::Hard } else { Edge::Soft },
        // A layer whose prompt was rewritten is made again from the typed one, rewritten anew.
        enhance: !info.enhanced.is_empty(),
        refs,
        grow: info.grow,
        erase: info.erase,
        rewritten: None,
        removal: false,
        tone_most: 1.0,
        prefill: false,
        tuning: Tuning::default(),
        object: None,
    };
    let label = format!("Generate Similar: {}", short(&info.prompt));
    crate::jobs::edit_job(
        s,
        &label,
        move |doc, active, ctx| {
            let n = rect.width() as usize * rect.height() as usize;
            let coverage: Vec<f32> = match doc.layer(id).and_then(|l| l.mask.as_ref()) {
                Some(m) => {
                    let k = m.surface.channels().max(1);
                    m.surface.read_region(rect).chunks_exact(k).map(|c| c.first().copied().unwrap_or(0.0).clamp(0.0, 1.0)).collect()
                }
                None => vec![1.0; n],
            };
            // The result lands above the layer it is similar to.
            *active = Some(id);
            fill_region(doc, active, ctx, job, rect, coverage, None)
        },
        move |(made, w, h, request, template_id)| made_json(&made, w, h, request, &template_id),
    )
}

fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

/// `generate.enhancePrompt`: the prompt rewritten by the vision-language model, as a background
/// job (a second or two on the 5090 once the 4B encoder is loaded), for the task bar's Enhance
/// button, the dialogs and scripts. The picture the rewriter looks at is the composite at the
/// models' working size; for a fill, the selection's place is said in words.
/// The backend `generate.enhancePrompt` asks: under `auto` with the native folders set, the
/// worker when it serves the enhancer (its Qwen-Image-2.1 encoder answers as Qwen3-VL), so the
/// command needs no ComfyUI (the Rust rebuild's R0 inventory found it always went there);
/// otherwise the preferences' backend.
fn enhancer_backend(s: &Session) -> Result<Arc<dyn GenerativeBackend>> {
    // Its language model (MOSS's 9B encoder on a Mac) does not fit beside anything in 8 GB.
    if crate::memory_profile::active(s) == Some(crate::memory_profile::Profile::Mac8) {
        return Err(EngineError::Other("Prompt Enhance is off on the 8 GB Mac profile (its language model needs 16 GB)".into()));
    }
    #[cfg(not(target_arch = "wasm32"))]
    if s.prefs().integrations.native_configured()
        && let Ok(b) = native_backend(s)
        && b.model_files("").is_ok_and(|m| m.iter().any(|t| t == ENHANCE_EDIT_TEMPLATE))
    {
        return Ok(b);
    }
    backend(s)
}

fn run_enhance(s: &mut Session, p: &Value) -> Result<Value> {
    let prompt = opt_str(ENHANCE, p, "prompt", MAX_PROMPT_CHARS)?
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .ok_or_else(|| bad(ENHANCE, "`prompt` is required: the prompt to rewrite"))?
        .to_string();
    let kind = match opt_str(ENHANCE, p, "task", 20)? {
        None | Some("fill") => EnhanceKind::Fill,
        Some("edit") => EnhanceKind::Edit,
        Some("image") => EnhanceKind::Image,
        Some("photo") => EnhanceKind::ImagePhoto,
        Some(other) => return Err(bad(ENHANCE, format!("`task` must be \"fill\", \"edit\", \"image\" or \"photo\" (got `{other}`)"))),
    };
    let use_image = opt_bool(ENHANCE, p, "useImage", true)?;
    let seed = parse_seed(ENHANCE, p)?;
    let backend = enhancer_backend(s)?;
    let (image, place) = match (kind, use_image, s.active()) {
        (EnhanceKind::Image | EnhanceKind::ImagePhoto | EnhanceKind::ImageUltra, _, _) | (_, false, _) | (_, _, None) => (None, None),
        (_, true, Some(d)) => {
            let canvas = d.doc.bounds();
            if canvas.is_empty() || u64::from(canvas.width()) * u64::from(canvas.height()) > MAX_REQUEST_PIXELS {
                (None, None)
            } else {
                let composite = photocraft_compose::render(&d.doc, canvas).to_rgba8();
                let img = Rgba8::new(composite.width, composite.height, composite.pixels).map_err(gen_err)?;
                let (rw, rh) = request_size(canvas.width(), canvas.height());
                let img = if (rw, rh) == (canvas.width(), canvas.height()) { img } else { resize_rgba8(&img, rw, rh)? };
                let place = match (kind, d.doc.selection.as_ref()) {
                    (EnhanceKind::Fill, Some(sel)) => Some(place_in_words(sel.content_bounds().intersect(&canvas), canvas)),
                    _ => None,
                };
                (Some(img), place)
            }
        }
    };
    let typed = prompt.clone();
    crate::jobs::run(
        s,
        "Enhance Prompt",
        false,
        move |ctx| {
            ctx.progress(0.05, "Improving the prompt");
            let out = enhance_prompt(backend.as_ref(), kind, &prompt, image.as_ref(), place.as_deref(), seed, &JobProgress::span(ctx, 0.05, 0.95))
                .map_err(gen_err)?;
            ctx.check()?;
            Ok(out)
        },
        move |_, out| {
            let enhanced = out.is_some();
            Ok(json!({"prompt": out.unwrap_or(typed), "enhanced": enhanced}))
        },
    )
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: FILL,
            label: "Generative Fill…",
            menu: &[],
            shortcut: None,
            params: r#"{"prompt":text,"negative":text,"steps":0..250=0,"guidance":0..30=0,"margin":0..1=0.25,"grow":0..512=0,"edge":"soft|hard","variations":int=1,"seed":{u64?=random},"template":{id?=Preferences › Default Fill Template; "auto" = the fastest permissive tier whose files the server has},"model":{file?=Preferences},"enhance":{bool?=Preferences › Enhance prompts},"fillAsEdit":bool=false,"erase":{bool?=the template's},"sampler":{str?=euler},"scheduler":{str?=simple},"maxPixels":{int?=1048576},"denoise":{0.05..1?} (the 2511 fill templates: how much of the selection is redrawn, 1 replaces it; measured on the 40-step template, 0.9 keeps the old sheen and 0.8 little changes, the 8-step tier hardly moves below 1),"loras":{[{"file":name,"strength":-4..4=1}]?},"references":{[{"layer":id} | {"path":"…"} | {"job":id,"index":0}]?},"name":{str?}} → {"layer","layers":[id],"seed","seeds":[u64],"template","runId","width","height","requestWidth","requestHeight","ms","timings":[{"encodeMs","uploadMs","queueMs","runMs","downloadMs"}]} (steps and guidance 0 = the template's defaults; loras = LoRA files from the LoRA folder (the native model root's loras/), applied by the native engine as side paths on its resident model, up to 8; on the ComfyUI route each becomes a LoraLoaderModelOnly after the template's model loader, the file in that server's models/loras; fillAsEdit = when auto runs Qwen-Image-2.1 on the native worker, fill by instruction (measured weaker than the fill template with its rewriter: the namer can pick a neighbour) on its edit template: the rewriter names what the selection holds and the description becomes "Replace the <thing> with <prompt>" (an instruction such as "replace the cup with a vase" goes as it is; nothing named: the fill template); enhance = rewrite the prompt first with the vision-language model, the picture and the selection's place in view (generate.enhancePrompt), the layer remembering both; margin = context around the selection as a fraction of its larger side; grow = widen the selection by that many pixels first, so a bulkier replacement has room; edge soft = the result's layer mask fades across the band the model re-rendered around the selection (4 % of its size), dithered with grain so there is no visible line; hard = the mask is exactly the selection; references = up to two reference images: with no template named the run switches to the fill-from-reference template, which flattens the selection to grey in the request so the model paints what the references show there (erase; `erase: true` does the same on any fill, `false` keeps the pixels); variations 1..4 = results with consecutive seeds, each a layer, only the first visible; sampler and scheduler = the KSampler's, ComfyUI names; maxPixels = the request is downscaled to at most this many pixels (the model's working size); areas over 1 megapixel are sent downscaled and come back resampled; a background job: the result is a new layer above the active one; needs the native worker, see Preferences › AI Integrations)"#,
            enabled: fill_enabled,
            run: run_fill,
            journal: true,
        },
        CommandSpec {
            id: IMAGE,
            label: "Generate Image…",
            menu: &[],
            shortcut: None,
            params: r#"{"prompt":text,"negative":text,"target":"auto|layer|document","width":int=0,"height":int=0,"transparent":bool=false,"steps":0..250=0,"guidance":0..30=0,"refine":{{"scale":1..2,"denoise":0.05..1,"steps":1..20}?=none: krea2-turbo/image redrawn once more at scale times the size (at most 2560 a side)},"enhanceStyle":{"default"|"photo"?=photo under Preferences › Quality best (no references, not transparent), else default; photo = written like a real photo's caption},"sampler":{"euler"|"er_sde"?=Preferences › Quality: er_sde on krea2-turbo/image under best and balanced, else euler},"schedule":{"simple"|"krea-detail"|"beta"?=Preferences › Quality: beta (9 steps) on krea2-turbo/image under best and balanced},"seed":{u64?=random},"template":{id?=Preferences › Default Image Template},"model":{file?=Preferences},"enhance":{bool?=Preferences › Enhance prompts},"name":{str?},"variations":int=1,"references":[{"layer":id}|{"path":file}|{"job":id,"index":int=0}]?} → {"layer"|"document","layers":[id],"seed","seeds":[u64],"template","runId","width","height","ms"} (target auto = a layer over the whole canvas when a document is open, else a new document; enhance = expand the idea into a detailed paragraph first with the vision-language model (generate.enhancePrompt); width/height 0 = the document's size or 1024, otherwise 64..4096 rounded down to multiples of 16; transparent = ask for the subject alone on a transparent background and keep the alpha the model returns (templates whose model can, such as qwen-2.1/image; others refuse); steps and guidance 0 = the template's defaults; variations 1..4 = results with consecutive seeds, each a layer (in the one new document with target document), only the first visible; references = images that guide the result (a pixel layer of the active document, an image file, or a stored result of the Generations library), at most the template's references.max (generate.models); with references and no template named, a default template that takes none gives way to qwen-edit-2511/image-ref; a background job; needs the native worker, see Preferences › AI Integrations)"#,
            enabled: image_enabled,
            run: run_image,
            journal: true,
        },
        CommandSpec {
            id: EXPAND,
            label: "Generative Expand…",
            // Placed by the menu catalogue under Edit.
            menu: &[],
            shortcut: None,
            params: r#"{"prompt":text,"left":int=0,"top":int=0,"right":int=0,"bottom":int=0,"width":int=0,"height":int=0,"negative":text,"steps":0..250=0,"guidance":0..30=0,"edge":"soft|hard","variations":int=1,"margin":{0..1=0.25},"anchor":{str?=center},"seed":{u64?=random},"template":{id?="auto": the Lightning expand tier when its LoRA is installed, else the 40-step one},"model":{file?=Preferences},"name":{str?}} → {"layer","layers":[id],"seed","seeds":[u64],"template","canvas":[w,h],"offset":[x,y],"width","height","requestWidth","requestHeight","ms","timings"} (adds canvas (left/top/right/bottom in pixels, or a larger width/height placed by anchor: topLeft, top, topRight, left, center, right, bottomLeft, bottom, bottomRight) and has the model paint it, as one undo step: the picture moves by the left/top pads, the result is a new layer over the added area; edge soft = its mask fades across the re-rendered band just inside the old edge (8 % of the picture's longer side), dithered; hard = exactly the added area; an empty prompt continues the scene; otherwise as generate.fill; needs the native worker)"#,
            enabled: expand_enabled,
            run: run_expand,
            journal: true,
        },
        CommandSpec {
            id: EDIT,
            label: "Generative Edit…",
            // Placed by the menu catalogue under Edit, with the other generative items.
            menu: &[],
            shortcut: None,
            params: r#"{"prompt":text,"negative":text,"steps":0..250=0,"guidance":0..30=0,"edge":"soft|hard","variations":int=1,"seed":{u64?=random},"template":{id?="auto": the Lightning edit tier when its LoRA is installed, else the 40-step one},"model":{file?=Preferences},"enhance":{bool?=Preferences › Enhance prompts},"name":{str?},"wholePicture":bool=false,"swap":{str?=none},"repose":bool=false,"skeleton":bool=false,"references":[{"layer":id}|{"path":file}|{"job":id,"index":int=0}]?} → {"layer","layers":[id],"seed","seeds":[u64],"template","runId","width","height","requestWidth","requestHeight","ms","timings"} (edits the whole picture by instruction; enhance = rewrite the instruction first with the vision-language model, the picture in view, so it names what is there instead of "this" (generate.enhancePrompt): "make the sky stormy", "turn the boat blue", "remove the person"; a description such as "a stormy sky" is wrapped as "change this image so that it shows …"; the composite goes to the model and the result is a new layer above the active one; with a selection the layer is masked to it (edge soft = a dithered fade around it, hard = exactly) so only that part of the edit shows; variations 1..4 = results with consecutive seeds, each a layer, only the first visible; pictures over 0.75 megapixels are sent downscaled and come back resampled; references = extra images the instruction can name as image 2, image 3 (the 2511 templates take two, qwen-2.1/edit nine; see references.max in generate.models): a pixel layer, an image file or a stored generation; swap = put the head ("head", also "face") or the whole person ("body") from the first reference onto the person in the picture, with the BFS LoRAs for Qwen-Image-2.1 (MIT): runs qwen-2.1/edit unless a template is named, adds the LoRA (bfs_head_v1.1_qwen_2.1 or bfs_body_swap_v1.0_qwen_2.1, from the loras folder), sends the LoRA author's prompt with any typed text after it, Enhance off, and keeps the whole re-rendered picture (wholePicture: a subject mask left the old person's outline); repose = redraw the person in the first reference in the pose of the mannequin that is the open picture, with the VNCCS PoseStudio LoRA for Qwen-Image-2.1 (V3, else V2, else V1.1 from the loras folder): runs qwen-2.1/edit unless a template is named, sends the LoRA's prompt (plus a line telling it to leave out the skeleton when skeleton is true) with any typed text after it, Enhance off, the whole picture kept; a background job; needs the native worker)"#,
            enabled: edit_enabled,
            run: run_edit,
            journal: true,
        },
        CommandSpec {
            id: ANGLES,
            label: "Generative Angles…",
            // Placed by the menu catalogue under Edit; the app opens its orbit dialog.
            menu: &[],
            shortcut: None,
            params: r#"{"rotation":0|45|90|135|180|225|270|315=0,"tilt":-30|0|30|60=0,"zoom":0|5|10=5,"seed":{u64?=random},"variations":int=1,"lightning":8|4|0=8,"anglesStrength":0..2=1,"lightningStrength":{0..2?=0.75 on the 8-step, else 1},"steps":{0..250?},"guidance":{0..30?},"negative":text} → as generate.edit (the whole picture seen from another camera, as layers over the canvas like Generative Edit's: rotation = degrees around the subject from the front (45 front-right quarter view … 180 back view … 315 front-left quarter view), tilt = the camera's height (-30 low angle, 0 eye level, 30 elevated, 60 high angle), zoom = 0 wide, 5 medium, 10 close-up; runs qwen-edit-2511/angles, which needs fal's Multiple Angles LoRA (Apache-2.0) in the Models folder’s loras subfolder; a selection does not confine it; lightning = the speed LoRA under the Multiple Angles LoRA: 8 = the 8-step one (8 steps, CFG 1), 4 = the 4-step one (4 steps, CFG 1, faster, invents more on hidden sides), 0 = none (40 steps, CFG 4 with the negative); steps and guidance override those; anglesStrength and lightningStrength = the two LoRAs' strengths)"#,
            enabled: angles_enabled,
            run: run_angles,
            journal: true,
        },
        CommandSpec {
            id: CHARACTER_SHEET,
            label: "Character Sheet",
            // Placed by the menu catalogue under Edit, with the other generative items.
            menu: &[],
            shortcut: None,
            params: r#"{"details":text,"background":text,"layer":{id?=active},"seed":{u64?=random}} → as generate.image (a turnaround sheet of the character on the layer as a new document: four full-body views in one row, front, three-quarter, side profile and back, drawn by minimax-h3-image/image with the layer as Picture 1; details = what the picture does not show, named as things ("blue jeans and white sneakers"); background = the backdrop, plain white when empty; about 16 MP under Preferences › Quality ultra and best, 4 MP under balanced and fast; needs MiniMax H3 Image's files in the Models folder; a background job; needs the native worker)"#,
            enabled: edit_enabled,
            run: run_character_sheet,
            journal: true,
        },
        CommandSpec {
            id: REMOVE_BG,
            label: "Remove Background (Generative)…",
            // Placed by the menu catalogue under Edit, with the other generative items.
            menu: &[],
            shortcut: None,
            params: r#"{"prompt":text,"asSelection":bool=false,"sampleAllLayers":bool=false,"refine":bool=true,"mode":"replace|add|subtract|intersect","layer":{id?=active},"seed":{u64?=random},"steps":{0..250=0},"guidance":{0..30=0},"template":{id?="auto": sam3.1/segment refined by ViTMatte (hair strand by strand, about a second) when both are installed, else qwen-2.1/matte (a soft matte from Qwen-Image-2.1's alpha output; research licence, Preferences › AI Integrations › Allow Research-Only Models) when allowed and installed, else sam3.1/segment alone (the permissive detector)},"model":{file?}} → {"layer","selection","bounds":[x,y,w,h],"pixels","seed","template","runId","width","height","requestWidth","requestHeight","ms","timings"} (prompt = what to keep, e.g. "the lighthouse", empty = the model's own reading of the subject; the matte becomes the layer's mask (the Background becomes a normal layer; the pixels are never changed) or, with asSelection, the selection combined by mode; sampleAllLayers sends the composite instead of the layer alone; refine = soften the detector's hard mask with the classical edge refinement, as the Quick Action does; images over one megapixel are sent downscaled and the matte comes back resampled; a background job; needs the native worker; the classical Quick Action layer.removeBackground needs none)"#,
            enabled: remove_bg_enabled,
            run: run_remove_bg,
            journal: true,
        },
        CommandSpec {
            id: SIMILAR,
            label: "Generate Similar",
            // Placed by the menu catalogue under Edit, with the other generative items.
            menu: &[],
            shortcut: None,
            params: r#"{"layer":{id?=active},"seed":{u64?=random},"variations":int=1} → as the command that made the layer (runs it again with a new seed in the same place: the layer's mask is the area, the result is a new layer above it; a generated image is generated again as a layer; an expanded area is filled again with its prompt; needs the native worker)"#,
            enabled: similar_enabled,
            run: run_similar,
            journal: true,
        },
        CommandSpec {
            id: SPLIT,
            label: "Split into Layers (Generative)…",
            // Placed by the menu catalogue under Edit, with the other generative items.
            menu: &[],
            shortcut: None,
            params: r#"{"method":{str?=simple},"detail":"medium|rough|fine","layers":2..20=20,"prompt":text,"complete":bool=true,"sweep":bool=false,"keepTries":bool=false,"firstPassOnly":bool=false,"seed":{u64?=random},"name":{str?},"edges":{str?=auto},"order":{str?=auto},"group":{str?=auto}} → {"layers":[id],"count","subjects":[{"layer","name","term","bounds":[x,y,w,h],"pixels","completed"}],"incomplete":[name],"fills","generations","tries":[{"behind","fill","generations","verdict","at":[x,y,w,h]}],"leftBehind":[[x,y,w,h]],"refusedLayers":[id],"detector":{"requests","tiles","ms","fog","pictures"},"already":[[x,y,w,h]],"order","foundBy","template","seed","width","height","fillMs","ms","edges"} (SAM 3.1 finds what the picture is of (prompt = what to cut out, a list such as "person, cup, table" or "the dog"; a term "cup:2" asks for at most two; empty = the people, then animals, then the foreground things the vision-language model names) and each instance, up to layers - 1 of them (default: every one with the simple method, 2 with classic), becomes its own layer (detail, simple method: rough = only subjects of 1 % of the frame or more, a subject inside another staying in its layer; medium = 0.1 % or more, each part of a subject on its own layer above it; fine = everything found; smaller ones stay in the background, listed in tooSmall) ("Person 1" furthest back, "Cup"; "{name} k/n" with name) with the picture's pixels and an edge-refined alpha, above one background layer ("Background", or "Split background" when the document has a Background of its own); with complete (the default) the subjects are peeled off nearest first and the fill paints what each one hid, so every layer is whole and the background has nobody left in it; without it one fill repaints all the subjects at once and each layer holds its visible part; sweep = select the nearest subject and remove it, then find the rest again on the filled picture, until nothing is left (whoever stands behind is found whole where the fill completed them), instead of finding every subject first and peeling them in that order (a later find of a person's or an animal's kind just below a subject of that kind still on the list, within their width, is their part and goes into their layer); the layers stack in depth order (order: auto = who hides whom from the masks, then the lower bottom edge; size = largest lowest; found = as found); group: auto = instances that overlap by half become one layer, touching = any that touch, none; a subject whose hole the fill could not clear keeps the picture's own pixels and is listed in incomplete; when none could be cleared nothing is changed; tries = every fill tried, in order (behind = the subject's layer name, or all without complete; fill = removal, fill (the grey box), strips (a big hole's box in strips) or patch (the classical fill, the last try); generations = the model fills the try asked for (a strips try several, the patch none); verdict = accepted, or why it was rejected: the subject kept or painted back, someone new (however small: looked for tile by tile over what the fill touched; whoever was in the picture before any fill is nobody's work, however much the cut subject hid of them, a part of a subject still behind or their completed part joined to them is that subject, whoever is in a framed picture is a picture of a person, and a figure with no more detail than the fog around it is no figure; at = the bounds of the detection a refusal names), the hole left blank in whole or part, or a shape painted over it or over part of it; a thing painted where none was is the one refusal taken when every try is refused, "accepted as the last resort (…)"), fills = the tries' count, generations = the model fills' count, fillMs = the time of every try, refused ones included; leftBehind = the bounds of any person the detector still finds on the finished background, in the whole frame or tile by tile, apart from the subjects in incomplete and whoever was in the picture before (nobody is expected); keepTries = keep every refused fill as a hidden layer under the background, named for its subject, try and verdict (their ids in refusedLayers), so a refusal can be seen and not only read; detector = the segmentation requests the split made, how many showed the detector a part of the picture (tiles), their time, the detections dismissed as fog and those taken for pictures of people; already = the bounds of whoever was in the picture before any fill, apart from the subjects (looked for tile by tile too, unless the backdrop is plain); new layers above the active one, one undo step; a background job; needs the native worker with SAM 3.1 and a fill model)"#,
            enabled: split_enabled,
            run: run_split,
            journal: true,
        },
        CommandSpec {
            id: CLOWN_PASS,
            label: "Clown Pass (Generative)",
            // Placed by the menu catalogue under Edit, with the other generative items.
            menu: &[],
            shortcut: None,
            params: r#"{"prompt":text?,"name":{str?="Clown Pass"}} → {"group","layers":[id],"names":[str],"count","ms"} (SAM 3.1 picks out everything it can name (prompt = the kinds of thing to look for, comma-separated; empty = person, animal, vehicle, plant, furniture, food, bag, bottle, cup, sign, object) and each thing becomes its own layer with the picture's pixels under its hard mask, the rest a Background layer, in a new group above the active layer; nothing is painted: each pixel is on exactly one layer (the smallest thing covering it), so the layers together show the picture unchanged; each layer gets its own label colour; one undo step; a background job; needs the native worker with SAM 3.1)"#,
            enabled: clown_pass::enabled,
            run: clown_pass::run,
            journal: true,
        },
        CommandSpec {
            id: INFO,
            label: "Generative Layer Info",
            menu: &[],
            shortcut: None,
            params: r#"{"layer":{id?=active}} → {"layer","generative":{"command","prompt","negative","template","seed","steps","guidance","edge","rect":[x,y,w,h],"name","width","height","transparent","enhanced","references"}|null,"job":{u64?}} (what a layer remembers about the generative run that made it; prompt = as typed, enhanced = the rewritten prompt the model got when `enhance` was on, else empty; references = the reference specs the run had, which generate.similar resolves again; kept with the layer through PSD round trips; job = its row in the Generations library (generate.jobs) when this session made it)"#,
            enabled: doc_enabled,
            run: run_info,
            journal: false,
        },
        CommandSpec {
            id: ENHANCE,
            label: "Enhance Prompt",
            menu: &[],
            shortcut: None,
            params: r#"{"prompt":text,"task":"fill|edit|image","useImage":bool=true,"seed":{u64?=random}} → {"prompt","enhanced"} (rewrites a prompt the way the image models want it, with the native worker’s vision-language model: a fill or edit instruction becomes one sentence that leads with the verb, names what the picture shows instead of "this" or "it", says what the result looks like and what stays unchanged; an image idea becomes a paragraph of 60–120 words (subject, materials, pose, lighting, setting, style) keeping every subject, colour and medium given; useImage = the rewriter sees the composite (a fill is also told where the selection sits); enhanced false = the model said nothing usable and the prompt comes back as typed; a background job; the fill, edit and image commands take "enhance":true to do this on the way, and Preferences › AI Integrations › Enhance prompts makes it their default; needs the native worker’s chat model)"#,
            enabled: |_| web_unavailable(),
            run: run_enhance,
            journal: false,
        },
        CommandSpec {
            id: VARIATION,
            label: "Show Variation",
            menu: &[],
            shortcut: None,
            params: r#"{"layers":[id],"index":int=1} → {"shown","index","count"} (the layers a Generative Fill made as variations: shows the index-th (1-based) and hides the others, as one undo step)"#,
            enabled: doc_enabled,
            run: run_variation,
            journal: true,
        },
        CommandSpec {
            id: DROP_VARIATION,
            label: "Drop Variation",
            menu: &[],
            shortcut: None,
            params: r#"{"layers":[id],"index":int} → {"dropped","layers","shown"} (deletes the index-th (1-based) of a run's variation layers as one undo step; when it was the one on screen and the others are hidden, the next one (else the previous one) is shown and active; layers = the ones left; the Variations strip's ×)"#,
            enabled: doc_enabled,
            run: run_drop_variation,
            journal: true,
        },
        CommandSpec {
            id: KEEP_VARIATION,
            label: "Keep Variation",
            menu: &[],
            shortcut: None,
            params: r#"{"layers":[id],"index":int} → {"kept","removed"} (keeps the index-th (1-based) of a run's variation layers, shown and active, and deletes the others, as one undo step; the Variations strip's Keep)"#,
            enabled: doc_enabled,
            run: run_keep_variation,
            journal: true,
        },
        CommandSpec {
            id: FREE,
            label: "Generative Models",
            // Placed by the menu catalogue under Edit › Purge.
            menu: &[],
            shortcut: None,
            params: r#"{} → {"freed"} (asks the ComfyUI server to unload its models and free their VRAM and RAM; the next generation reloads what it needs. Use it when fills have become several times slower than usual: a server that has loaded more than one model family may be streaming weights on every run)"#,
            enabled: |_| web_unavailable(),
            run: run_free,
            journal: false,
        },
        CommandSpec {
            id: HEALTH,
            label: "Generative Server Status",
            menu: &[],
            shortcut: None,
            params: r#"{"template"?:string} → {"ok","backend","url","version","vramTotal","vramFree","queueRemaining","error"?} (with a template id: the backend that template's requests go to; under `auto` the native worker serves SAM 3.1 when its folders are set)"#,
            enabled: always,
            run: run_health,
            journal: false,
        },
        CommandSpec {
            id: MODELS,
            label: "List Generative Models",
            menu: &[],
            shortcut: None,
            params: r#"{"probe":bool=true,"width":int?,"height":int?,"count":int=1,"references":int=0} → {server,device,templates:[{id,name,family,task,license,allowed,defaults,models,creator,description,badges,references,sizes,counts,estimatedSeconds,pages,eta}],autoFill,autoExpand,autoEdit,autoRemove,autoMatte}; native worker capabilities; probe false lists metadata without starting or contacting a worker"#,
            enabled: always,
            run: run_models,
            journal: false,
        },
        CommandSpec {
            id: loras::LORAS,
            label: "List LoRAs",
            menu: &[],
            shortcut: None,
            params: r#"{"template"?:string} → {"folder","family","native","loras":[{"file","family"}]} (the LoRA files in Preferences › AI Integrations › Models folder's loras subfolder, each with the model family its tensor names are for (krea2, qwen-image-2.1, qwen-image-edit-2511, or null), only the template's family's when a template is given; native = the template would run on the native engine, the route that applies a request's loras. Reads the files' headers, never the server)"#,
            enabled: |_| web_unavailable(),
            run: loras::run_loras,
            journal: false,
        },
        CommandSpec {
            id: upscale::UPSCALE,
            label: "Generative Upscale",
            // Placed by the menu catalogue under Image.
            menu: &[],
            shortcut: None,
            params: r#"{"scale":1..4=2,"tileMegapixels"?:0.25..2,"prompt"?:string,"template"?:string,"seed"?:int,"steps"?:int,"guidance"?:number,"loras"?:[{"file","strength"}],"restoreStrength":0..1.5=0.5,"grid"?:[columns,rows],"name"?:string} → {"layer","tiles","grid","width","height","elapsedMs","template"} (the document enlarged by scale the plain way (bicubic), then redrawn tile by tile by the edit model with a restore prompt (AusBoss's wording unless prompt is given): by default the fewest equal tiles of at most ~1.15 MP each, never a strip of three or more, sharing at least a 128-px strip (grid [columns, rows], 1 to 16 each, names the tiling; tileMegapixels asks for square tiles of about that size instead), each moved back onto its source and matched to its colours, the strips faded together; the redraw is a new layer above. Auto: Qwen-Image-2.1's edit on the native engine when it serves it (with the Restore LoRA at restoreStrength when the LoRA folder has qwen-image-2.1-restore.safetensors and no loras are given), else the edit default)"#,
            enabled: edit_enabled,
            run: upscale::run_upscale,
            journal: true,
        },
    ]
}

#[path = "generate_loras.rs"]
pub(crate) mod loras;

#[path = "generate_repose.rs"]
pub mod repose;
#[path = "generate_swap.rs"]
pub mod swap;

/// `p` rewritten by its `swap` or its `repose` (not both), or None when it asks for neither.
pub(crate) fn edit_rewrite(s: &Session, p: &Value) -> Result<Option<Value>> {
    let swapping = p.get("swap").and_then(Value::as_str).is_some_and(|k| !k.trim().is_empty() && !k.eq_ignore_ascii_case("none"));
    if swapping && p.get("repose").and_then(Value::as_bool) == Some(true) {
        return Err(bad(EDIT, "`swap` and `repose` can't be asked together"));
    }
    match swap::apply(s, p)? {
        Some(q) => Ok(Some(q)),
        None => repose::apply(s, p),
    }
}

#[cfg(test)]
#[path = "generate_cmds_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "generate_image_tests.rs"]
mod image_tests;

#[path = "generate_upscale.rs"]
pub(crate) mod upscale;

#[cfg(test)]
#[path = "generate_template_tests.rs"]
mod template_tests;

#[path = "generate_matte_guide.rs"]
mod matte_guide;

#[cfg(test)]
#[path = "generate_matte_tests.rs"]
mod matte_tests;
#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "generate_native_tests.rs"]
mod native_tests;

#[cfg(test)]
#[path = "generate_edit_tests.rs"]
mod edit_tests;

#[cfg(test)]
#[path = "generate_similar_tests.rs"]
mod similar_tests;

#[path = "generate_clown_pass.rs"]
mod clown_pass;
pub(crate) use clown_pass::CLOWN_PASS;

#[cfg(test)]
#[path = "generate_clown_pass_tests.rs"]
mod clown_pass_tests;

#[path = "generate_split_subjects.rs"]
mod split_subjects;

#[cfg(test)]
#[path = "generate_split_tests.rs"]
mod split_tests;

#[cfg(test)]
#[path = "generate_enhance_tests.rs"]
mod enhance_tests;
