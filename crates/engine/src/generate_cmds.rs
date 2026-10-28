//! `generate.*` commands: generative edits through a local backend (`photocraft-genai`).
//!
//! `generate.fill` is Generative Fill. The flattened composite around the selection and the
//! selection's coverage go to the backend (a ComfyUI server, Preferences › AI Integrations), and
//! the result comes back as a **new raster layer above the active one, masked to the selection**.
//! The original pixels never change. The command is a background job (`jobs::edit_job`): progress
//! comes from the server, Esc cancels and interrupts it, and the edit lands as one undo step.
//! `generate.health` and `generate.models` are queries for the UI, the CLI and agents.
//!
//! Preferences read here: `integrations.comfyServer`, `integrations.defaultEditModel`,
//! `integrations.generativeTimeoutSecs`, `integrations.allowResearchModels`,
//! `integrations.enhancePrompts` (the default of the fill, edit and image commands' `enhance`),
//! `integrations.modelPrecision` (the backend's NVFP4 policy, see `backend`).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use photocraft_algo::selection::{self as sel, SelectionMode};
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Layer, LayerId, LayerMask, Size};
use photocraft_genai::template::{self, License, Template};
use photocraft_genai::{GenerativeBackend, Gray8, Health, Request, Rgba8, Task};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, layer_param};
use crate::jobs::JobCtx;
use crate::{EngineError, Result, Session};

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
/// A small request is upscaled until its longer side is this long: the 2511 templates sample at
/// the request's own size, and a handful of latent tokens cannot carry structure.
const MIN_FILL_REQUEST_SIDE: u32 = 512;
/// The sampler and scheduler the 2511 fill templates run unless a request says otherwise:
/// ComfyUI's official template's pair.
const DEFAULT_SAMPLER: &str = "euler";
const DEFAULT_SCHEDULER: &str = "simple";

/// What a fill request may tune beyond its template (`sampler`, `scheduler` and `maxPixels`
/// on `generate.fill` and the removal commands): the sampler and scheduler the 2511 fill
/// templates run, and the pixels a request is downscaled to ([`MAX_FILL_REQUEST_PIXELS`]
/// unless given). For measuring what the server's own settings are worth.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Tuning {
    pub sampler: Option<String>,
    pub scheduler: Option<String>,
    pub max_pixels: u64,
    /// LoRA adapters on the model for this request, `(file name, strength)`, in order
    /// (`loras` on the fill, edit, image and removal commands). The native engine applies them
    /// as side paths on its resident model (plan-native-engine §9: AnyPaint, AusBoss's
    /// Consistency, Outpaint, Restore and Outfit Swap, the BFS swaps); the ComfyUI route chains
    /// a `LoraLoaderModelOnly` per file after the template's model loader (`comfy::with_loras`),
    /// so the file must be in that server's `models/loras`.
    pub loras: Vec<(String, f32)>,
    /// A removal keeps what it removed: the object's pixels (the detector's silhouette, or
    /// the stroke's) as their own masked layer above the fill (`keepObject`), the two the same
    /// size, so the thing can be moved or something put behind it (AusBoss's Cut Out + Remove).
    pub keep_object: bool,
    /// A removal on a route that regenerates the whole request (every Qwen-Image-2.1
    /// template) also takes what changed with the object: the shadow and the reflection the
    /// model removed beside it, where they touch the hole (`shadow`, on by default; AusBoss's
    /// differential composite, [`change_band`]). A fill or an edit with a selection on such a
    /// route takes what the model drew past the selection the same way, within a smaller share
    /// (a swapped garment longer than the old one).
    pub shadow: bool,
    /// The context a removal's request takes around the grown hole, in the hole's longer side
    /// (`removeMargin`, 0.25 unless given): a shadow longer than that runs out of the request
    /// and stays (the dog's, 2026-10-27).
    pub remove_margin: f64,
    /// A named removal on a whole-picture route asks for the object "and its shadow"
    /// (`nameShadow`).
    pub name_shadow: bool,
    /// How much of a fill's selection the model redraws (`denoise`, 0.05 to 1; the templates
    /// whose sampler binds it, [`takes_denoise`]): 1 replaces it; on the 2511 40-step fill 0.9 keeps
    /// the old sheen, 0.8 changes little, the 8-step tier hardly moves below 1 (measured
    /// 2026-10-27). `None`: the template's own (1).
    pub denoise: Option<f64>,
}

/// Whether `template`'s sampler takes a `denoise` (the 2511 fill templates; Qwen-Image-2.1's
/// edit-style fill and the native engine redraw from noise and ignore it).
pub fn takes_denoise(template: &str) -> bool {
    template::find(template).is_ok_and(|t| t.meta.placeholders.contains_key("denoise"))
}

/// [`Tuning::remove_margin`] unless given.
const REMOVE_MARGIN: f64 = 0.25;
/// [`Tuning::name_shadow`] unless given.
const NAME_SHADOW: bool = false;

impl Default for Tuning {
    fn default() -> Self {
        Self { sampler: None, scheduler: None, max_pixels: MAX_FILL_REQUEST_PIXELS, loras: Vec::new(), keep_object: false, shadow: true, remove_margin: REMOVE_MARGIN, name_shadow: NAME_SHADOW, denoise: None }
    }
}

/// At most this many LoRAs on one request.
const MAX_LORAS: usize = 8;

impl Tuning {
    /// From a command's params; a sampler or scheduler is a plain ComfyUI name (letters,
    /// digits, underscores), `maxPixels` 65 536 to 16 M, `loras` a list of
    /// `{"file": "name.safetensors", "strength": 1.0}` (a file name, no folders; strength
    /// -4 to 4, 1 unless given).
    pub(crate) fn from_params(cmd: &str, p: &Value) -> Result<Tuning> {
        let name = |key: &str| -> Result<Option<String>> {
            match opt_str(cmd, p, key, 40)?.map(str::trim).filter(|s| !s.is_empty()) {
                None => Ok(None),
                Some(s) if s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') => Ok(Some(s.to_string())),
                Some(_) => Err(bad(cmd, format!("`{key}` must be a ComfyUI sampler or scheduler name (letters, digits and underscores)"))),
            }
        };
        let max_pixels = opt_num(cmd, p, "maxPixels", 65_536.0, MAX_REQUEST_PIXELS as f64)?.map_or(MAX_FILL_REQUEST_PIXELS, |v| v as u64);
        let flag = |key: &str, default: bool| -> Result<bool> {
            match p.get(key) {
                None | Some(Value::Null) => Ok(default),
                Some(Value::Bool(b)) => Ok(*b),
                Some(_) => Err(bad(cmd, format!("`{key}` must be true or false"))),
            }
        };
        let (keep_object, shadow, name_shadow) = (flag("keepObject", false)?, flag("shadow", true)?, flag("nameShadow", NAME_SHADOW)?);
        let remove_margin = opt_num(cmd, p, "removeMargin", 0.0, 1.0)?.unwrap_or(REMOVE_MARGIN);
        let denoise = opt_num(cmd, p, "denoise", 0.05, 1.0)?;
        Ok(Tuning { sampler: name("sampler")?, scheduler: name("scheduler")?, max_pixels, loras: loras_of(cmd, p)?, keep_object, shadow, remove_margin, name_shadow, denoise })
    }

    /// The `loras` request parameter for the worker, or nothing.
    pub(crate) fn loras_param(&self) -> Option<Value> {
        (!self.loras.is_empty()).then(|| Value::Array(self.loras.iter().map(|(f, s)| json!({"file": f, "strength": s})).collect()))
    }
}

/// `loras` from a command's params: `[{"file": …, "strength": …}]`, validated.
pub(crate) fn loras_of(cmd: &str, p: &Value) -> Result<Vec<(String, f32)>> {
    let Some(v) = p.get("loras").filter(|v| !v.is_null()) else { return Ok(Vec::new()) };
    let Some(items) = v.as_array() else { return Err(bad(cmd, "`loras` must be a list of {file, strength}")) };
    if items.len() > MAX_LORAS {
        return Err(bad(cmd, format!("at most {MAX_LORAS} LoRAs on one request")));
    }
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        let file = it.get("file").and_then(Value::as_str).map(str::trim).unwrap_or_default();
        if file.is_empty() || file.len() > 200 || file.contains(['/', '\\', ':']) || file.starts_with('.') {
            return Err(bad(cmd, "a LoRA is named by its file name in the LoRA folder (`file`), no folders"));
        }
        let strength = match it.get("strength") {
            None | Some(Value::Null) => 1.0,
            Some(s) => s.as_f64().filter(|v| v.is_finite() && (-4.0..=4.0).contains(v)).ok_or_else(|| bad(cmd, "a LoRA `strength` is a number from -4 to 4"))? as f32,
        };
        out.push((file.to_string(), strength));
    }
    Ok(out)
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

pub(crate) const REMOVE_BG: &str = "generate.removeBackground";
pub(crate) const DEFAULT_MATTE_TEMPLATE: &str = "qwen-2.1/matte";
/// ComfyUI's official background-removal instruction for Qwen-Image-2.1, sent when the prompt
/// does not name a subject (it opens with an imperative verb, so the template passes it through).
pub(crate) const MATTE_DEFAULT_PROMPT: &str = "Remove the background, and output a PNG image";
/// A matte request goes out at up to this many pixels: the model's own working size, and the
/// matte only needs to be resampled back, not the pixels.
const MAX_MATTE_REQUEST_PIXELS: u64 = 1024 * 1024;

pub(crate) const INFO: &str = "generate.info";
pub(crate) const SIMILAR: &str = "generate.similar";
pub(crate) const ENHANCE: &str = "generate.enhancePrompt";
const ENHANCE_EDIT_TEMPLATE: &str = "enhance/edit";
const ENHANCE_IMAGE_TEMPLATE: &str = "enhance/image";

/// The system turn before an edit or fill instruction. Short on purpose: the 4B model echoes
/// or rambles under a long system prompt. The rules are the ones that recur in the
/// Qwen-Image-Edit guides: lead with the verb, name things instead of "this"/"it", say what the
/// result looks like, say what stays; plus the case the first user session hit, a change of
/// medium on a doodle, where "keep the lines" would defeat the instruction.
const ENHANCE_EDIT_SYSTEM: &str = "You rewrite instructions for an image-editing AI. You can see the image.\n\
Rewrite the user's instruction as one clear sentence of at most 60 words that:\n\
- starts with the action verb (Change, Replace, Add, Remove, Turn, Make, Give),\n\
- names things by what they are in the image instead of \"this\", \"it\" or \"here\" (for example \"the hand-drawn smiley face\", \"the white lighthouse\", \"the woman in the red coat\"),\n\
- says concretely what the result should look like,\n\
- ends with what must stay unchanged, if anything should.\n\
When the instruction changes the style, medium or realism of the picture (\"make this realistic\", \"turn it into a painting\"), name the real-world subject the result shows and say that the drawing's strokes are replaced by it (a doodle of a face made realistic becomes \"a photograph of a real human face with the same expression and pose, replacing the black outline strokes with real skin, hair and facial features\"); never ask to keep the drawing's lines.\n\
Keep the user's intent. Add nothing they did not ask for. Keep any quoted text exactly.\n\
Reply with the sentence only: no explanation, no quotes, no markdown.";

/// Added to the system turn for a fill: where the selection is and what the instruction means
/// there.
const ENHANCE_FILL_NOTE: &str = "The user selected the {place} of the image; the instruction says what should be generated inside that selection so that it blends with its surroundings. Describe what appears there.";

/// The system turn's note for a removal, whose picture is two views side by side: the
/// surroundings on the left, a close-up of the painted area on the right ([`side_by_side`]).
/// The rewriter has no other view of what was painted: with the fill note and one crop it
/// named what lay under the stroke ("the window" for a stroke on a car, "the dog's head",
/// "the hand" for a ring) or the most salient thing in view ("the pendant" for a stroke on
/// the chain); with the painted area outlined in green it talked about the outline ("the
/// green circle", "the green-painted face"). Measured 2026-10-21 on the robustness set.
const ENHANCE_REMOVE_NOTE: &str = "You can see two views of the same photo side by side: on the right, a close-up of an object to remove (or of part of it); on the left, the same with its surroundings. Reply with exactly one sentence of the form \"Remove the <object>.\" The object is the whole thing, not a part of it (the whole car, not its window; the whole dog, not its head; the whole necklace, not its pendant; the whole cup, not the coffee in it), named by what it is and its colour in at most five words; never call it an area, region, part or selection. It is the object itself, not the person, animal or larger thing that wears, holds or carries it (the glasses, not the man wearing them; the mask, not the face; the ring, not the hand; the umbrella, not the person under it). Say nothing else: not what is behind it, not the two views, not other things in the picture.";

/// The system turn for judging two results of a removal side by side ([`judge_pair`]): the
/// edge-match score preferred a smeared skin-coloured blob over a man's eyes to the eyes
/// themselves (its colour and smoothness matched the cheeks), where a look at the two tells
/// them apart at once (measured 2026-10-21).
const ENHANCE_JUDGE_SYSTEM: &str = "You can see two versions of the same part of a photo side by side, left and right, after an attempt to remove something from it; the question names what was to be removed. A version where that thing is still there, whole or in part, loses. A version with a flaw where it was also loses: a smear, a blur, a blocky or pixelated patch, a flat panel, an area of flat red or other paint, a wrong texture, a duplicated line or object, or something that does not belong. Pick the version where the thing is gone and the scene looks like a natural, untouched photograph.\n\
Reply with exactly one sentence: \"The left one.\" or \"The right one.\" Nothing else.";

/// What the judge is asked, with the object's name when the rewriter found one ("left"
/// before "right", so a parroted question picks the left). Asked only which looked natural,
/// the judge kept a crow on its wire and a boat on its lake: both look natural (measured
/// 2026-10-22).
fn judge_question(object: Option<&str>) -> String {
    match object {
        Some(name) => format!("The {name} was to be removed. In which one is the {name} gone and the scene untouched, left or right?"),
        None => "Something was to be removed. In which one is it gone and the scene untouched, left or right?".to_string(),
    }
}

/// The system turn for asking whether the removed thing is still in one result
/// ([`still_there`]): a yes or no on one picture, asked of each result before the two are
/// judged side by side. The side-by-side judge kept transparent-framed glasses it was told
/// to remove, where the direct question on the one picture answers plainly (2026-10-22).
const ENHANCE_PRESENCE_SYSTEM: &str = "You can see one photo. The question names a thing. Say whether that thing, or any part of it, is visible in the photo: look carefully for thin, faint, transparent or partly hidden ones.\n\
Reply with exactly one word: Yes or No.";

/// The system turn for a fill from a reference image. The rewriter sees the picture (its
/// selection flattened to grey) and the reference side by side, as AusBoss's Krea 2 clothing
/// workflow shows its prompt writer the pair, and names what to take exactly as the reference
/// shows it; the template's own wrapper then says where it goes and what stays.
const ENHANCE_FILL_REF_SYSTEM: &str = "You can see two photos side by side. The left photo is being edited: its flat grey area will be repainted. The right photo is a reference. The user names what to take from the reference.\n\
Reply with one noun phrase of at most 50 words naming that thing and describing it exactly as the right photo shows it: for clothing its garment type, colour, material, knit or weave, pattern, collar, buttons, length and fit; for anything else its shape, colour, material and details. Describe it as it will look on the left photo's subject, in that pose. Keep the user's words where they are specific.\n\
Never mention grey, blank, masked, left, right, photo, image or reference. Reply with the phrase only: no sentence, no quotes, no markdown.";

/// The system turn before a text-to-image idea (the shape of the official Krea 2 template's
/// expander and of Qwen's rewriter, in a few lines).
const ENHANCE_IMAGE_SYSTEM: &str = "You are an expert prompt writer for a text-to-image model. Expand the user's idea into one paragraph of 60 to 120 words the model can follow: the subject and what it does first, then its materials, surfaces or clothing, pose and framing, lighting, the setting, and finally the style, palette and mood.\n\
Keep every subject, action, colour and spatial relation the user gave. Add no new objects, characters or animals. Keep any medium the user named (photo, painting, sketch, 3D render). Put any words to be rendered in the image in double quotes. If the idea is already detailed, polish it lightly and keep its wording.\n\
Reply with the paragraph only: no title, no bullets, no quotes around it, no markdown.";

/// Which rewriter an `enhance` runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EnhanceKind {
    /// A fill: the picture, the selection's place, and what to generate there.
    Fill,
    /// A fill from a reference: the picture (its selection flattened to grey) and the reference
    /// side by side, and what to take from the reference.
    FillRef,
    /// A removal: the surroundings and a close-up of the painted part side by side; name the
    /// whole object the part belongs to.
    Remove,
    /// Two results of a removal side by side: which looks like an untouched photograph.
    Judge,
    /// One result of a removal: is the named thing still in it.
    Presence,
    /// An edit: the picture and an instruction about it.
    Edit,
    /// Text to image: no picture.
    Image,
}

/// Where a region sits on a canvas, in words ("upper-left part", "centre", "right part").
pub(crate) fn place_in_words(region: Rect, canvas: Rect) -> String {
    let (cw, ch) = (f64::from(canvas.width().max(1)), f64::from(canvas.height().max(1)));
    let cx = (f64::from(region.x0 + region.x1) / 2.0 - f64::from(canvas.x0)) / cw;
    let cy = (f64::from(region.y0 + region.y1) / 2.0 - f64::from(canvas.y0)) / ch;
    let big = !canvas.is_empty() && f64::from(region.width()) * f64::from(region.height()) > 0.6 * cw * ch;
    if big {
        return "whole".to_string();
    }
    let v = if cy < 1.0 / 3.0 {
        "upper"
    } else if cy > 2.0 / 3.0 {
        "lower"
    } else {
        ""
    };
    let h = if cx < 1.0 / 3.0 {
        "left"
    } else if cx > 2.0 / 3.0 {
        "right"
    } else {
        ""
    };
    match (v, h) {
        ("", "") => "centre".to_string(),
        ("", h) => format!("{h} part"),
        (v, "") => format!("{v} part"),
        (v, h) => format!("{v}-{h} part"),
    }
}

/// The rewriter's answer, cleaned: the first non-empty text without the chat template's
/// leading "assistant" echo, surrounding quotes or markdown emphasis; `None` when the model
/// said nothing usable.
pub(crate) fn clean_enhanced(texts: &[String]) -> Option<String> {
    for t in texts {
        let mut s = t.trim();
        for prefix in ["assistant:", "Assistant:", "assistant", "Assistant", "**Final Prompt:**", "Final Prompt:", "Rewritten:", "Instruction:"] {
            if let Some(rest) = s.strip_prefix(prefix) {
                s = rest.trim();
            }
        }
        let s = s.trim_matches(|c: char| c == '"' || c == '*' || c == '`' || c == '\u{201c}' || c == '\u{201d}').trim();
        if s.split_whitespace().count() >= 3 {
            return Some(s.to_string());
        }
    }
    None
}

/// Rewrite `prompt` with the vision-language model: `Ok(Some)` the rewritten prompt, `Ok(None)`
/// when the model said nothing usable in two tries (consecutive seeds), `Err` when the server
/// could not run it (no encoder file, down). An edit or fill is rewritten with the picture in
/// view when one is given, blind otherwise. The rewriter goes straight to the backend: its
/// small encoder is not a model switch worth a purge, and it is the file Krea 2 loads anyway.
pub(crate) fn enhance_prompt(
    backend: &dyn GenerativeBackend,
    kind: EnhanceKind,
    prompt: &str,
    image: Option<&Rgba8>,
    place: Option<&str>,
    seed: u64,
    progress: &JobProgress,
) -> std::result::Result<Option<String>, photocraft_genai::Error> {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Ok(None);
    }
    let image = match kind {
        EnhanceKind::Image => None,
        _ => image.cloned(),
    };
    // The engine's instructions are the chat's system turn, the user's prompt its user turn.
    let edit_system = if image.is_some() { ENHANCE_EDIT_SYSTEM.to_string() } else { ENHANCE_EDIT_SYSTEM.replace(" You can see the image.", "") };
    let template = if image.is_some() { ENHANCE_EDIT_TEMPLATE } else { ENHANCE_IMAGE_TEMPLATE };
    let system = match kind {
        EnhanceKind::Image => ENHANCE_IMAGE_SYSTEM.to_string(),
        EnhanceKind::Edit => edit_system,
        EnhanceKind::Fill => format!("{edit_system}\n{}", ENHANCE_FILL_NOTE.replace("{place}", place.unwrap_or("selected part"))),
        EnhanceKind::Remove => format!("{edit_system}\n{ENHANCE_REMOVE_NOTE}"),
        EnhanceKind::Judge => ENHANCE_JUDGE_SYSTEM.to_string(),
        EnhanceKind::Presence => ENHANCE_PRESENCE_SYSTEM.to_string(),
        EnhanceKind::FillRef => ENHANCE_FILL_REF_SYSTEM.to_string(),
    };
    let mut params = BTreeMap::new();
    params.insert("system".to_string(), Value::String(system));
    for attempt in 0..2u64 {
        let req = Request {
            template: template.to_string(),
            prompt: prompt.to_string(),
            negative: String::new(),
            seed: seed.wrapping_add(attempt) % MAX_SEED,
            steps: 0,
            guidance: 0.0,
            image: image.clone(),
            mask: None,
            models: Vec::new(),
            size: None,
            params: params.clone(),
            references: Vec::new(),
        };
        let resp = backend.run(&req, progress)?;
        if let Some(s) = clean_enhanced(&resp.texts) {
            return Ok(Some(s));
        }
    }
    Ok(None)
}

pub(crate) const SPLIT: &str = "generate.splitLayers";
pub(crate) const DEFAULT_LAYERS_TEMPLATE: &str = "qwen-layered/split";
const MAX_SPLIT_LAYERS: u32 = 8;
/// The layered model's working size: 640 px on the longer side (what ComfyUI's official graph
/// scales the input to); the layers come back resampled to the canvas.
const LAYERED_MAX_SIDE: u32 = 640;
/// What the layered model is told when no description is given.
const SPLIT_DEFAULT_PROMPT: &str = "separate this image into layers: the background and each foreground subject on its own layer";

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

/// The Multiple Angles LoRA's prompt for a camera, `<sks> [azimuth] [elevation] [distance]`, or
/// None for a value it has no words for.
pub fn angle_prompt(rotation: u32, tilt: i32, zoom: u32) -> Option<String> {
    let r = ANGLE_ROTATIONS.iter().find(|(d, _)| *d == rotation)?.1;
    let t = ANGLE_TILTS.iter().find(|(d, _)| *d == tilt)?.1;
    let z = ANGLE_ZOOMS.iter().find(|(d, _)| *d == zoom)?.1;
    Some(format!("<sks> {r} {t} {z}"))
}

/// The template `generate.angles` runs: under `auto` with the research opt-in and the native
/// worker, the Qwen-Image-2.1 edit by instruction ([`angle_instruction`], about 6 s on the
/// worker), else 2511 with the Multiple Angles LoRA on ComfyUI.
pub fn angles_template(s: &Session) -> &'static str {
    native21_auto(s, EDIT, false).unwrap_or(ANGLES_TEMPLATE)
}

/// The plain camera instruction the Qwen-Image-2.1 edit gets for a camera, which has no
/// Multiple Angles LoRA: the camera named as moved ("Rotate the camera 90 degrees to the right
/// around the subject", "Raise the camera high above…"), the wording that made 2.1 orbit a
/// bench and raise or close in on people and a dog in the 2026-10-27 probe (`benchmarks.md`),
/// where naming the view ("a right side view") mostly turned the subject. None as above.
pub fn angle_instruction(rotation: u32, tilt: i32, zoom: u32) -> Option<String> {
    angle_prompt(rotation, tilt, zoom)?;
    let mut parts: Vec<String> = Vec::new();
    match rotation {
        0 => {}
        180 => parts.push("Move the camera behind the subject, so it is seen from the back.".into()),
        r if r < 180 => parts.push(format!("Rotate the camera {r} degrees to the right around the subject.")),
        r => parts.push(format!("Rotate the camera {} degrees to the left around the subject.", 360 - r)),
    }
    match tilt {
        -30 => parts.push("Lower the camera to a low angle, looking up at the subject.".into()),
        30 => parts.push("Raise the camera a little, looking slightly down at the subject.".into()),
        60 => parts.push("Raise the camera high above the subject, looking down at it.".into()),
        _ => {}
    }
    match zoom {
        0 => parts.push("Pull the camera back for a wide shot that shows more of the surroundings.".into()),
        10 => parts.push("Move the camera closer for a close-up of the subject.".into()),
        _ => {}
    }
    if parts.is_empty() {
        parts.push("Show the subject from the front at eye level.".into());
    }
    parts.push("Keep the subject, its identity, its clothing and the setting the same.".into());
    Some(parts.join(" "))
}
/// The Qwen-Image-2.1 template that `auto` runs for `command` (a removal when `removal`) instead
/// of the Qwen-Image-Edit-2511 order, when research-only models are allowed and the native
/// worker runs the family: the owner's choice of 2026-10-27, on the removal numbers of
/// `benchmarks.md` (2.1 as an edit 15 of 20 hard strokes clean at 6–28 s, 2511 16 at 33–58 s
/// through ComfyUI). A removal is an edit by instruction on the crop, as measured; a prompted
/// fill is the 2.1 fill (the mask in view); an expand the in-context 2.1 expand. Without the
/// opt-in or the worker, the 2511 orders stand (the permissive default).
pub(crate) fn native21_auto(s: &Session, command: &str, removal: bool) -> Option<&'static str> {
    let t = match command {
        FILL if removal => "qwen-2.1/edit",
        FILL => "qwen-2.1/fill",
        EXPAND => "qwen-2.1/expand",
        EDIT => "qwen-2.1/edit",
        _ => return None,
    };
    (s.prefs().integrations.allow_research_models && loras::runs_native(s, t)).then_some(t)
}

/// The crop a removal on Qwen-Image-2.1 goes out at when `maxPixels` is left at the default:
/// 0.75 MP, the size the 2.1 removal runs were measured at (1 MP is 2511's).
const NATIVE21_REMOVAL_PIXELS: u64 = 786_432;

pub(crate) const DEFAULT_EDIT_TEMPLATE: &str = "qwen-edit-2511/edit";
/// What `auto` tries for an edit, fastest permissive tier first.
pub(crate) const AUTO_EDIT_ORDER: &[&str] = &["qwen-edit-2511/edit-lightning-8", DEFAULT_EDIT_TEMPLATE];
pub(crate) const BEST_EDIT_ORDER: &[&str] = &[DEFAULT_EDIT_TEMPLATE];

/// Preferences › AI Integrations › Quality, normalised (`best` unless it says `balanced` or
/// `fast`).
pub fn quality(s: &Session) -> &'static str {
    match s.prefs().integrations.quality.trim() {
        "balanced" => "balanced",
        "fast" => "fast",
        _ => "best",
    }
}

/// The quality a removal (Content-Aware Fill, the Remove tool, Spot Healing through the
/// model) runs at: `best` becomes `balanced`. The 40-step tier was measured no better than
/// the 8-step one on removals and worse under the smoothed hole (a face erased to a blur,
/// a seam in a lawn; `benchmarks.md` §5i), and at two candidates a stroke it took a
/// Spot Healing dab from about 25 s to 100 s, which reads as the tool not working.
/// `fast` stays `fast`.
pub fn removal_quality(s: &Session) -> &'static str {
    match quality(s) {
        "fast" => "fast",
        _ => "balanced",
    }
}

/// The templates `auto` picks from for `command` under `quality` (the first one the server has
/// every file for runs): `best` = the full-step base template, `balanced` = the 8-step Lightning
/// tier then the base, `fast` = the 4-step tier where one exists.
pub fn auto_order(command: &str, quality: &str) -> &'static [&'static str] {
    match (command, quality) {
        (FILL, "best") => BEST_FILL_ORDER,
        (FILL, "fast") => FAST_FILL_ORDER,
        (FILL, _) => AUTO_FILL_ORDER,
        (EXPAND, "best") => BEST_EXPAND_ORDER,
        (EXPAND, _) => AUTO_EXPAND_ORDER,
        (EDIT, "best") => BEST_EDIT_ORDER,
        (EDIT, _) => AUTO_EDIT_ORDER,
        _ => &[DEFAULT_IMAGE_TEMPLATE],
    }
}

/// The model files (`folder/file`) the last run on each server loaded, for [`run_switching`].
/// Mirrored to a small file in the temp directory so the app, the CLI and MCP agree on what
/// a server holds across processes (each is otherwise a fresh process).
static LAST_MODELS: Mutex<BTreeMap<String, BTreeSet<String>>> = Mutex::new(BTreeMap::new());

/// Where [`LAST_MODELS`] is mirrored: `{server: [files]}` JSON.
fn last_models_path() -> std::path::PathBuf {
    std::env::temp_dir().join("photocraft-genai-last-models.json")
}

/// What the last run on `server` loaded, from this process's memory or the mirror file.
fn last_models(server: &str) -> Option<BTreeSet<String>> {
    if let Some(set) = LAST_MODELS.lock().unwrap_or_else(PoisonError::into_inner).get(server) {
        return Some(set.clone());
    }
    let text = std::fs::read_to_string(last_models_path()).ok()?;
    let all: BTreeMap<String, BTreeSet<String>> = serde_json::from_str(&text).ok()?;
    all.get(server).cloned()
}

/// Remember what `server` holds now, in memory and in the mirror file (best effort).
fn remember_models(server: &str, set: BTreeSet<String>) {
    let mut mem = LAST_MODELS.lock().unwrap_or_else(PoisonError::into_inner);
    mem.insert(server.to_string(), set);
    let path = last_models_path();
    let mut all: BTreeMap<String, BTreeSet<String>> = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    for (k, v) in mem.iter() {
        all.insert(k.clone(), v.clone());
    }
    // Test servers come and go on random ports: keep the file small.
    if all.len() > 32 {
        all.retain(|k, _| mem.contains_key(k));
    }
    if let Ok(text) = serde_json::to_string(&all) {
        let _ = std::fs::write(path, text);
    }
}

/// After a purge nothing is loaded: forget what `server` held (in memory and in the mirror).
fn forget_models(server: &str) {
    LAST_MODELS.lock().unwrap_or_else(PoisonError::into_inner).remove(server);
    let path = last_models_path();
    if let Some(mut all) = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<BTreeMap<String, BTreeSet<String>>>(&t).ok())
        && all.remove(server).is_some()
        && let Ok(text) = serde_json::to_string(&all)
    {
        let _ = std::fs::write(path, text);
    }
}

/// The server a session's generative requests go to: the key of [`LAST_MODELS`].
pub(crate) fn server_key(s: &Session) -> String {
    s.prefs().integrations.comfy_server.trim().to_string()
}

/// The model files (`folder/file`) a request loads: the template's slots with the request's
/// overrides.
fn model_set(req: &Request) -> BTreeSet<String> {
    let Ok(t) = template::find(&req.template) else { return BTreeSet::new() };
    let bound = t.model_bindings(&req.models).unwrap_or_default();
    t.meta.models.iter().map(|slot| format!("{}/{}", slot.folder, bound.get(&slot.placeholder).and_then(Value::as_str).unwrap_or(&slot.default))).collect()
}

/// `integrations.purgeModels`: when a run asks the server to unload what it holds first.
pub const PURGE_SWITCH: &str = "switch";
pub const PURGE_HEAVY: &str = "heavy";
pub const PURGE_NEVER: &str = "never";

/// Preferences › AI Integrations › Purge models, normalised (`switch` unless it says `heavy`
/// or `never`).
pub fn purge_mode(s: &Session) -> &'static str {
    match s.prefs().integrations.purge_models.trim() {
        "heavy" => PURGE_HEAVY,
        "never" => PURGE_NEVER,
        _ => PURGE_SWITCH,
    }
}

/// The diffusion models and checkpoints of a model set (`folder/file` entries): what a purge
/// under `heavy` is decided on.
fn heavy_files(set: &BTreeSet<String>) -> BTreeSet<String> {
    set.iter().filter(|f| f.starts_with("diffusion_models/") || f.starts_with("unet/") || f.starts_with("checkpoints/")).cloned().collect()
}

/// Run `req` on `backend`, first asking the server to unload its models when the request's
/// model files differ from what the previous run on `server` loaded (`purge`, see
/// [`purge_mode`]). ComfyUI keeps the earlier
/// models resident and, short of memory for the new set, loads it partially and streams the
/// rest on every step: the first run after a switch between the 2511 base and its Lightning
/// tier, or from 2511 to Qwen-Image-2.1, took 31–131 s on the 32 GB card against 13–20 s for
/// the runs after it. A purge costs one reload instead. The first run on a server never purges.
pub(crate) fn run_switching(
    backend: &dyn GenerativeBackend,
    server: &str,
    req: &Request,
    purge: &str,
    progress: &JobProgress,
) -> photocraft_genai::Result<photocraft_genai::Response> {
    // The files the backend will load (its precision policy may swap in NVFP4 ones).
    let prepared = backend.prepare(req);
    let req = &prepared;
    let set = model_set(req);
    // The native worker keeps its own residency (one diffusion family at a time, the detector
    // beside it): a purge there dropped the resident 2.1 pipeline whenever a removal's detector
    // ran between two fills, and the next fill loaded everything again (measured 2026-10-26).
    let switch = server != NATIVE_SERVER_KEY
        && !set.is_empty()
        && purge != PURGE_NEVER
        && match last_models(server) {
            // `heavy`: only a change of the diffusion model or checkpoint purges; a LoRA, an
            // encoder or a VAE swapped under the same model loads beside it.
            Some(prev) if purge == PURGE_HEAVY => heavy_files(&prev) != heavy_files(&set),
            Some(prev) => prev != set,
            // Nothing known about this server (no PhotoCraft process has run on it): something
            // else may have left it full of other models, and a card with less than a quarter
            // of its memory free would load the new set partially and stream the rest every
            // step.
            None => {
                let h = backend.health();
                matches!((h.vram_total, h.vram_free), (Some(total), Some(free)) if total > 0 && free.saturating_mul(4) < total)
            }
        };
    if switch {
        photocraft_genai::Progress::report(progress, 0.0, "Unloading the previous models");
        // A server that cannot purge just runs as before.
        let _ = backend.free();
    }
    let resp = backend.run(req, progress)?;
    if !set.is_empty() {
        remember_models(server, set);
    }
    Ok(resp)
}

pub(crate) fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

pub(crate) fn gen_err(e: photocraft_genai::Error) -> EngineError {
    match e {
        photocraft_genai::Error::Cancelled => EngineError::Cancelled,
        e => EngineError::Other(e.to_string()),
    }
}

/// `integrations.engine`, normalised: `auto`, `native` or `comfy` (anything else is `auto`).
pub fn engine_mode(s: &Session) -> &'static str {
    match s.prefs().integrations.engine.trim() {
        "native" => "native",
        "comfy" => "comfy",
        _ => "auto",
    }
}

/// The native engine's backend for this process: one worker (`photocraft --engine …`), started
/// on first use, shared by every command (plan-native-engine §3.4). A change of the runtime or
/// model folders (`integrations.runtimePath`, `integrations.modelsPath`) stops the worker so the
/// next call starts one with the new flags.
#[cfg(not(target_arch = "wasm32"))]
static NATIVE: std::sync::OnceLock<std::sync::Mutex<Option<Arc<photocraft_genai::native::NativeBackend>>>> = std::sync::OnceLock::new();

/// Tests: what `native` routes to instead (a fake engine served in-process).
#[cfg(all(test, not(target_arch = "wasm32")))]
static NATIVE_OVERRIDE: std::sync::OnceLock<std::sync::Mutex<Option<Arc<photocraft_genai::native::NativeBackend>>>> = std::sync::OnceLock::new();

/// SAM 3.1's precision on the worker's GPU: `f16` (1.9x the speed of f32; select by text on a
/// new picture 273–278 ms through the app against ONNX Runtime's 249–346 ms, every mask at an
/// IoU of at least 0.993 against ComfyUI's own f32 node: genai-native gn-segment F16.md), or
/// `f32` with `PHOTOCRAFT_SAM_PRECISION=f32` (equal to ComfyUI's within 3 pixels). Not yet a
/// preference: the owner decides whether it gets one.
#[cfg(not(target_arch = "wasm32"))]
fn sam_precision() -> &'static str {
    match std::env::var("PHOTOCRAFT_SAM_PRECISION").ok().as_deref().map(str::trim) {
        Some(p) if p.eq_ignore_ascii_case("f32") => "f32",
        _ => "f16",
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn native_backend(s: &Session) -> Result<Arc<dyn GenerativeBackend>> {
    #[cfg(test)]
    if let Some(b) = NATIVE_OVERRIDE.get_or_init(|| std::sync::Mutex::new(None)).lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        return Ok(Arc::clone(b) as Arc<dyn GenerativeBackend>);
    }
    let integrations = &s.prefs().integrations;
    let mut wanted = photocraft_genai::native::worker_args(&integrations.runtime_folders(), &integrations.models_root());
    if !wanted.is_empty() {
        wanted.extend(["--sam-precision".to_string(), sam_precision().to_string()]);
    }
    let slot = NATIVE.get_or_init(|| std::sync::Mutex::new(None));
    let mut g = slot.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(b) = g.as_ref() {
        // The flags after `--engine` are the preferences; a change restarts the worker. A
        // worker this process joined rather than started is kept as it is.
        if b.is_attached() || b.args().get(1..).unwrap_or(&[]) == wanted.as_slice() {
            return Ok(Arc::clone(b) as Arc<dyn GenerativeBackend>);
        }
        b.stop();
        *g = None;
    }
    // `PHOTOCRAFT_NATIVE_ENGINE=port:token` joins a worker already running (one started by
    // hand with `photocraft-cli --engine …` and `GN_ENGINE_TOKEN`), so a bench of many CLI
    // processes shares one resident set of models instead of loading them in every process.
    if let Ok(spec) = std::env::var("PHOTOCRAFT_NATIVE_ENGINE")
        && let Some((port, token)) = spec.split_once(':')
        && let Ok(port) = port.trim().parse::<u16>()
    {
        let b = Arc::new(photocraft_genai::native::NativeBackend::attached(port, token.trim()).map_err(gen_err)?);
        *g = Some(Arc::clone(&b));
        return Ok(b);
    }
    let dirs = photocraft_genai::native::runtime_dirs(&integrations.runtime_folders());
    let b = photocraft_genai::native::NativeBackend::this_executable(&wanted).map_err(gen_err)?.with_path_dirs(dirs);
    let b = Arc::new(b.with_log(crate::generate_engine_cmds::native_log_path()));
    *g = Some(Arc::clone(&b));
    Ok(b)
}

/// The worker this process holds now, if any (never starts one).
#[cfg(not(target_arch = "wasm32"))]
fn native_current() -> Option<Arc<photocraft_genai::native::NativeBackend>> {
    #[cfg(test)]
    if let Some(b) = NATIVE_OVERRIDE.get_or_init(|| std::sync::Mutex::new(None)).lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        return Some(Arc::clone(b));
    }
    NATIVE.get().and_then(|slot| slot.lock().unwrap_or_else(|e| e.into_inner()).clone())
}

/// `generate.engine.status`'s `native`: `off` under `integrations.engine = comfy`,
/// `unconfigured` without a runtime and a model folder, else the worker's own state
/// (`NativeBackend::status`: stopped / starting / running / ready / failed), without waiting
/// and without starting one.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn native_status(s: &Session) -> Value {
    let log = crate::generate_engine_cmds::native_log_path().to_string_lossy().into_owned();
    let mode = engine_mode(s);
    let state = |st: &str| json!({"state": st, "mode": mode, "log": log});
    if mode == "comfy" {
        return state("off");
    }
    if !s.prefs().integrations.native_configured() && native_current().is_none_or(|b| !b.is_attached()) {
        return state("unconfigured");
    }
    let st = native_current().map(|b| b.status()).unwrap_or_else(|| photocraft_genai::native::NativeStatus { state: "stopped".into(), ..Default::default() });
    let mut v = serde_json::to_value(st).unwrap_or_else(|_| json!({"state": "failed"}));
    v["mode"] = json!(mode);
    v["log"] = json!(log);
    v
}

/// Start the worker on a thread of its own (the call returns at once; the status says
/// `starting`, then `ready`).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn native_start(s: &Session) -> Result<()> {
    if !s.prefs().integrations.native_configured() {
        return Err(EngineError::Other("set the runtime and model folders in Preferences › AI Integrations first".into()));
    }
    let b = native_backend(s)?;
    std::thread::spawn(move || {
        let _ = b.health();
    });
    Ok(())
}

/// Stop the worker this process started (the next generative command starts a fresh one).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn native_stop() {
    if let Some(b) = native_current() {
        b.stop();
    }
}

/// Unload the worker's models, if it runs.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn native_purge() -> Result<()> {
    match native_current() {
        Some(b) => b.purge().map_err(gen_err),
        None => Ok(()),
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn set_native_backend(b: Arc<photocraft_genai::native::NativeBackend>) {
    let slot = NATIVE_OVERRIDE.get_or_init(|| std::sync::Mutex::new(None));
    *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(b);
}

/// The backend the preferences point at: the native worker under `integrations.engine =
/// native`, the ComfyUI server otherwise (`auto` is the ComfyUI route until a native model is
/// faster at parity; plan-native-engine §3.3). The web build has none yet.
pub(crate) fn backend(s: &Session) -> Result<Arc<dyn GenerativeBackend>> {
    #[cfg(not(target_arch = "wasm32"))]
    if engine_mode(s) == "native" {
        return native_backend(s);
    }
    let integrations = &s.prefs().integrations;
    let url = integrations.comfy_server.trim();
    if url.is_empty() {
        return Err(EngineError::Other("no generative server is configured (Preferences › AI Integrations › ComfyUI Server)".into()));
    }
    let timeout = Duration::from_secs(u64::from(integrations.generative_timeout_secs.max(5)));
    // Quality `best` keeps `auto` precision off the 4-bit files; an explicit `nvfp4` still wins.
    let precision = match photocraft_genai::Precision::from_name(&integrations.model_precision) {
        photocraft_genai::Precision::Auto if quality(s) == "best" => photocraft_genai::Precision::Default,
        p => p,
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        let b = photocraft_genai::comfy::ComfyBackend::new(url, timeout)
            .map_err(gen_err)?
            .with_precision(precision)
            .with_sidecar(crate::generate_engine_cmds::launch_from_prefs(s));
        Ok(Arc::new(b))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (timeout, precision);
        Err(EngineError::Other("generative features are not available in the web build yet".into()))
    }
}

/// The model-set key of the native worker's runs (`run_switching`): its own, so a native run
/// never counts as a model switch on the ComfyUI server.
pub(crate) const NATIVE_SERVER_KEY: &str = "native";

/// The backend one template's request goes to, with the key its model set is remembered under:
/// `native` and `comfy` as the preference says; under `auto`, the native worker when it serves
/// this template and the ComfyUI server otherwise (plan-native-engine §3.3: a model moves under
/// `auto` once the native route beats its row at parity, which SAM 3.1 did on 2026-10-19).
/// Every generative command asks here with its planned template (`generate.image`, `fill`,
/// `expand`, `edit`, the selection commands); a job whose `auto` template is resolved later
/// (`resolve_auto`) keeps the backend chosen for the default.
pub(crate) fn backend_for(s: &Session, template: &str) -> Result<(Arc<dyn GenerativeBackend>, String)> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if engine_mode(s) == "native" {
            return Ok((native_backend(s)?, NATIVE_SERVER_KEY.to_string()));
        }
        if engine_mode(s) == "auto"
            && let Some(b) = native_serving(s, template)
        {
            return Ok((b, NATIVE_SERVER_KEY.to_string()));
        }
    }
    let b = backend(s)?;
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(hint) = research_hint(s, template)
        && !b.health().ok
    {
        return Err(EngineError::Other(hint));
    }
    Ok((b, server_key(s)))
}

/// Under `auto` with research-only models off (the permissive default), a Qwen-Image-Edit-2511
/// template goes to ComfyUI even when the native worker is set up and runs Qwen-Image-2.1. When
/// no ComfyUI answers and none is started by the app, the request would fail with "cannot
/// reach the ComfyUI server" alone; this says both ways out instead (Rust rebuild R6: the
/// licence choice stays the user's). `None` when it does not apply.
#[cfg(not(target_arch = "wasm32"))]
fn research_hint(s: &Session, template: &str) -> Option<String> {
    let i = &s.prefs().integrations;
    let is_2511 = template::find(template).is_ok_and(|t| t.meta.family == "qwen-image-edit-2511");
    if engine_mode(s) != "auto" || !is_2511 || i.allow_research_models || !i.comfy_path.trim().is_empty() || !i.native_configured() {
        return None;
    }
    let serves_21 = native_backend(s).ok()?.model_files("").ok()?.iter().any(|m| m.starts_with("qwen-2.1/"));
    serves_21.then(|| {
        format!(
            "this runs Qwen-Image-Edit-2511 on a ComfyUI server, and none answers at {}. The native engine can run it with Qwen-Image-2.1 instead: turn on Allow Research-Only Models in Preferences › AI Integrations (its licence is for research and personal use, not commercial work). Or start ComfyUI, or name its folder there so the app starts it.",
            i.comfy_server.trim()
        )
    })
}

/// The backend for the vision-language questions a job asks (the rewriter, the judge, the
/// presence check). A job on the native worker asks the worker itself when it serves the
/// enhancer template (its Qwen-Image-2.1 text encoder answers as Qwen3-VL 8B, genai-native
/// `Pipeline::chat`): no second model on the card. Otherwise the backend serving the enhancer
/// template (the ComfyUI route); when no such route is configured, the job's own backend, which
/// then answers or declines as before.
pub(crate) fn thinker_for(s: &Session, own: &Arc<dyn GenerativeBackend>, server: &str) -> Arc<dyn GenerativeBackend> {
    // `PHOTOCRAFT_THINKER=comfy` keeps the questions on the ComfyUI route even for a native
    // job (a bench comparing the two rewriters; not a preference).
    let comfy_only = std::env::var("PHOTOCRAFT_THINKER").is_ok_and(|v| v.trim() == "comfy");
    if !comfy_only && server == NATIVE_SERVER_KEY && own.model_files("").is_ok_and(|m| m.iter().any(|t| t == ENHANCE_EDIT_TEMPLATE)) {
        return Arc::clone(own);
    }
    backend_for(s, ENHANCE_EDIT_TEMPLATE).map(|(b, _)| b).unwrap_or_else(|_| Arc::clone(own))
}

/// Whether two handles are the same backend (a job's questions answered by its own worker).
fn same_backend(a: &Arc<dyn GenerativeBackend>, b: &Arc<dyn GenerativeBackend>) -> bool {
    Arc::as_ptr(a) as *const () == Arc::as_ptr(b) as *const ()
}

/// On the native route a ComfyUI server answering the job's questions (the rewriter, the
/// judge) holds only the vision-language model: it leaves after each answer unless purges are
/// off, so the worker's resident models have the card. Measured 2026-10-26: with the 5 GB model
/// kept resident beside the worker's 2.1 set and the card's other tenants, a native stroke
/// paged for minutes; ComfyUI loads it again in two seconds for the next question. A worker
/// answering its own questions is never told to unload.
fn release_thinker(server: &str, purge: &str, thinker: &Arc<dyn GenerativeBackend>, own: &Arc<dyn GenerativeBackend>) {
    if server == NATIVE_SERVER_KEY && purge != "never" && !same_backend(thinker, own) {
        let _ = thinker.free();
    }
}

/// The model families `auto` sends to the native worker when it serves them: those whose native
/// route beat their ComfyUI row at parity (plan-native-engine §3.3, §12.1). SAM 3.1 since
/// 2026-10-19; Qwen-Image-2.1 text to image since 2026-10-20 (ComfyUI's noise, schedule and
/// prompt reproduced: the same seed gives the same picture, 40 dB on the fox, and the resident
/// worker is faster than the ComfyUI route per picture; dev log), its edit and fill templates
/// since the same day (the references through the native vision tower and VAE encoder, the
/// fox's winter edit at 40 dB); Krea 2 text to image since 2026-10-27 (ComfyUI's own fp8 files
/// on its fp8 recipe, exact attention, fused small ops: 19 of 28 pictures at the parity gate
/// against ComfyUI's route, whose fp8 and bf16 routes agree on 20, at 4.0 s a picture through
/// the app against 4.0–4.7). The worker serves a family only in a build with the
/// `native-diffusion` feature, and `native_serving` asks it template by template (Krea 2's
/// style reference stays on ComfyUI: the worker does not list it). Qwen-Image-Layered's split
/// since 2026-10-27: 1–3/255 and alpha IoU ≥ 0.994 per layer against ComfyUI's, 57 s against
/// 58 s at 640 px (genai-native `4fcbfc3`); Apache-2.0, so no research opt-in.
pub(crate) const NATIVE_AUTO_FAMILIES: [&str; 4] = ["sam3.1", "qwen-image-2.1", "krea2", "qwen-image-layered"];

/// Under `auto`: the native worker, when the preferences give it its folders, the template's
/// family is in [`NATIVE_AUTO_FAMILIES`] and the worker's index lists the template. A worker
/// that cannot start or does not serve the template is `None`, and the ComfyUI route stands.
#[cfg(not(target_arch = "wasm32"))]
fn native_serving(s: &Session, template: &str) -> Option<Arc<dyn GenerativeBackend>> {
    let i = &s.prefs().integrations;
    let configured = i.native_configured();
    if !configured || !template::find(template).is_ok_and(|t| NATIVE_AUTO_FAMILIES.contains(&t.meta.family.as_str())) {
        return None;
    }
    let b = native_backend(s).ok()?;
    b.model_files("").ok()?.iter().any(|m| m == template).then_some(b)
}

pub(crate) fn web_unavailable() -> std::result::Result<(), String> {
    if cfg!(target_arch = "wasm32") { Err("not available in the web build yet".into()) } else { Ok(()) }
}

fn fill_enabled(s: &Session) -> std::result::Result<(), String> {
    web_unavailable()?;
    let d = s.active().ok_or("no document open")?;
    if d.doc.selection.is_none() {
        return Err("make a selection first".into());
    }
    Ok(())
}

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
}

/// Everything `generate.fill` needs, validated before any pixel or network work.
struct FillPlan {
    template: Template,
    prompt: String,
    negative: String,
    seed: u64,
    steps: u32,
    guidance: f32,
    name: String,
    models: Vec<(String, String)>,
    /// The area sent to the backend: the selection's bounds grown by the margin, on the canvas.
    rect: Rect,
    /// How many results to make (consecutive seeds); only the first is visible.
    variations: u32,
    /// Pick the template inside the job from what the server has (`AUTO_FILL_ORDER`).
    auto: bool,
    /// How the result's layer mask meets the surroundings.
    edge: Edge,
    /// Rewrite the prompt with the vision-language model first (`enhance`, default from the
    /// `enhancePrompts` preference).
    enhance: bool,
    /// Reference images (`references`: edits, and fills from a reference).
    refs: crate::generate_refs::Refs,
    /// Widen the selection by this many pixels before the request (`grow`), so a bulkier
    /// replacement has room and the old object's edge is inside the repainted area.
    grow: u32,
    /// Flatten the selection to mid-grey in the request image (`erase`), so the model paints a
    /// hole instead of copying what was there.
    erase: bool,
    /// The sampler, scheduler and request size asked for ([`Tuning`]).
    tuning: Tuning,
}

/// How a generated layer's mask ends: softly, across the band the model re-rendered around the
/// selection, with per-pixel noise so the ramp reads as grain rather than a gradient or a line
/// (the default), or hard, exactly on the selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Edge {
    Soft,
    Hard,
}

fn parse_edge(cmd: &str, p: &Value) -> Result<Edge> {
    match opt_str(cmd, p, "edge", 10)? {
        None | Some("soft") => Ok(Edge::Soft),
        Some("hard") => Ok(Edge::Hard),
        Some(other) => Err(bad(cmd, format!("`edge` must be soft or hard (got `{other}`)"))),
    }
}

/// The layer mask of a soft edge: the feathered request mask, its ramp dithered with
/// per-pixel noise (zero inside and outside, strongest mid-ramp) so the transition has no
/// visible line and no banding. Deterministic for a seed, so variations differ.
fn dither_edge(mask: &[u8], seed: u64) -> Vec<f32> {
    mask.iter()
        .enumerate()
        .map(|(i, v)| {
            let m = f32::from(*v) / 255.0;
            if *v == 0 || *v == 255 {
                return m;
            }
            // splitmix64 of the pixel index and the seed → a uniform value in 0..1.
            let mut z = (i as u64).wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let n = (z >> 40) as f32 / (1u64 << 24) as f32;
            let amplitude = 0.6 * (1.0 - (2.0 * m - 1.0).abs());
            (m + (n - 0.5) * amplitude).clamp(0.0, 1.0)
        })
        .collect()
}

pub(crate) fn opt_str<'a>(cmd: &str, p: &'a Value, key: &str, max: usize) -> Result<Option<&'a str>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => {
            if s.chars().count() > max {
                return Err(bad(cmd, format!("`{key}` is longer than {max} characters")));
            }
            Ok(Some(s))
        }
        Some(_) => Err(bad(cmd, format!("`{key}` must be a string"))),
    }
}

/// An optional boolean parameter: absent or null = `default`.
pub(crate) fn opt_bool(cmd: &str, p: &Value, key: &str, default: bool) -> Result<bool> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(bad(cmd, format!("`{key}` must be true or false"))),
    }
}

pub(crate) fn opt_num(cmd: &str, p: &Value, key: &str, lo: f64, hi: f64) -> Result<Option<f64>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            let x = v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| bad(cmd, format!("`{key}` must be a number")))?;
            if !(lo..=hi).contains(&x) {
                return Err(bad(cmd, format!("`{key}` must be within {lo}..{hi} (got {x})")));
            }
            Ok(Some(x))
        }
    }
}

/// The first 40 characters of a prompt, for layer names and job labels.
pub(crate) fn short(prompt: &str) -> String {
    let mut s: String = prompt.chars().take(40).collect();
    if prompt.chars().count() > 40 {
        s.push('…');
    }
    s
}

/// A template id from a preference, or the built-in default when the preference is empty.
fn template_or(pref: &str, fallback: &str) -> String {
    let t = pref.trim();
    if t.is_empty() { fallback.to_string() } else { t.to_string() }
}

/// The template a fill with reference images runs when none is named: the 40-step base under
/// Quality `best`, the 8-step tier otherwise (both take the same model files as the plain fill).
pub fn fill_ref_template(quality: &str) -> &'static str {
    if quality == "best" { BEST_FILL_REF_TEMPLATE } else { DEFAULT_FILL_REF_TEMPLATE }
}

/// The request rectangle of a fill over `bounds`: the bounds grown by `margin` of their longer
/// side, on the canvas, on the 16-px grid the VAE and the patch grid want (so a template that
/// samples at the request's own size needs no padding and the result no resampling); refused
/// above the request cap.
pub(crate) fn fill_rect(bounds: Rect, margin: f64, canvas: Rect) -> Result<Rect> {
    let longer = i64::from(bounds.width()).max(i64::from(bounds.height()));
    let m = (margin * longer as f64).round().clamp(0.0, 65_536.0) as i32;
    let rect = Rect::new(bounds.x0.saturating_sub(m), bounds.y0.saturating_sub(m), bounds.x1.saturating_add(m), bounds.y1.saturating_add(m)).intersect(&canvas);
    if rect.is_empty() {
        return Err(EngineError::Other("the selection is empty".into()));
    }
    let rect = align_to_grid(rect, canvas);
    let pixels = u64::from(rect.width()) * u64::from(rect.height());
    if pixels > MAX_REQUEST_PIXELS {
        return Err(EngineError::Other(format!(
            "the selected area with its margin is {}×{} pixels; Generative Fill handles up to 16 megapixels at a time: select a smaller area or lower `margin`",
            rect.width(),
            rect.height()
        )));
    }
    Ok(rect)
}

/// What the generative Content-Aware Fill, Remove tool and Spot Healing ask the fill model,
/// with the picture's pixels kept under the noise mask. "Remove" opens the fill templates'
/// imperative wrapper, which adds "fit the scene, change nothing else". Measured on the
/// lighthouse's boat (2026-10-21, `caf-matrix-live.ps1`): this generic instruction removes the
/// object and continues the sea; naming the object does no better; the mask-guided template
/// matches it at twice the time; a hole flattened to grey is painted back as a grey panel by
/// 2511 (the `erase` that serves a swap from a reference defeats a removal).
pub const REMOVE_PROMPT: &str = "Remove the object in the selected area and fill its place with the surrounding scene continued seamlessly (its background, surfaces, textures, patterns and lines), as if nothing had ever been there";
/// The part of [`REMOVE_PROMPT`] the rewriter's name replaces ("the gold necklace").
const REMOVE_PROMPT_OBJECT: &str = "the object in the selected area";

/// The names the detector is asked for in turn: the name as given, then its last two words,
/// then its last word ("white ceramic coffee cup" → "coffee cup" → "cup"), without repeats.
pub(crate) fn name_candidates(name: &str) -> Vec<String> {
    let words: Vec<&str> = name.split_whitespace().collect();
    let mut out: Vec<String> = Vec::new();
    for n in [words.len(), 2, 1] {
        if n == 0 || n > words.len() {
            continue;
        }
        let c = words[words.len() - n..].join(" ");
        if !out.contains(&c) {
            out.push(c);
        }
    }
    out
}

/// The detector's candidates for a name that is a wearer and what it wears ("man wearing the
/// cap and mask" → "cap", "mask", "cap and mask", "man"), or `None` for a plain name. The
/// removal asks for each and keeps the one whose mask matches the stroke best.
pub(crate) fn wearer_candidates(name: &str) -> Option<Vec<String>> {
    let (wearer, worn) = [" wearing ", " holding ", " carrying "].iter().find_map(|m| name.split_once(m))?;
    let bare = |s: &str| -> String {
        let s = s.trim();
        ["the ", "a ", "an ", "his ", "her ", "their ", "its "].iter().find_map(|a| s.strip_prefix(a)).unwrap_or(s).trim().to_string()
    };
    let mut out: Vec<String> = worn.split(" and ").flat_map(|p| p.split(", ")).map(bare).filter(|p| !p.is_empty()).collect();
    for whole in [bare(worn), bare(wearer)] {
        if !whole.is_empty() && !out.contains(&whole) {
            out.push(whole);
        }
    }
    Some(out)
}

/// How well an object (`obj` over the canvas, `on` pixels) matches a stroke (`stroke` over
/// `sb`): intersection over union of the two.
fn stroke_overlap(stroke: &[f32], sb: Rect, obj: &[f32], on: u64, canvas: Rect) -> f64 {
    let cw = canvas.width().max(1) as usize;
    let bw = sb.width().max(1) as usize;
    let (mut inside, mut painted) = (0u64, 0u64);
    for (i, c) in stroke.iter().enumerate() {
        if *c < 0.5 {
            continue;
        }
        painted += 1;
        let (x, y) = (sb.x0 + (i % bw) as i32, sb.y0 + (i / bw) as i32);
        if canvas.contains(x, y) && obj.get((y - canvas.y0) as usize * cw + (x - canvas.x0) as usize).copied().unwrap_or(0.0) >= 0.5 {
            inside += 1;
        }
    }
    let union = (painted + on).saturating_sub(inside).max(1);
    inside as f64 / union as f64
}

/// `PHOTOCRAFT_DUMP_VIEWS=<folder>` (bench only): the picture a removal's rewriter is shown
/// is saved there as `views-<n>.png`, so `naming-probe.ps1` can ask about exactly that picture.
fn dump_views(seen: &Rgba8) {
    let Ok(dir) = std::env::var("PHOTOCRAFT_DUMP_VIEWS") else { return };
    let Ok(png) = photocraft_genai::png::encode_rgba8(seen) else { return };
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    let _ = std::fs::write(std::path::Path::new(dir.trim()).join(format!("views-{stamp}.png")), png);
}

/// Is a generative backend set up: a ComfyUI server, or the native worker's folders? The
/// classical tools go through the model only then, and fall back to their own algorithms
/// otherwise (a server that is set up but down is reported by the job instead).
pub fn generative_configured(s: &Session) -> bool {
    let i = &s.prefs().integrations;
    let native = i.native_configured();
    match i.engine.trim() {
        "native" => native,
        "comfy" => !i.comfy_server.trim().is_empty(),
        _ => native || !i.comfy_server.trim().is_empty(),
    }
}

/// `cov` (one value per pixel of `bounds`) laid over `rect`, zero elsewhere.
fn embed_coverage(bounds: Rect, cov: &[f32], rect: Rect) -> Vec<f32> {
    let (w, h) = (rect.width() as usize, rect.height() as usize);
    let bw = bounds.width().max(1) as usize;
    let mut out = vec![0.0f32; w * h];
    for (i, c) in cov.iter().enumerate() {
        if *c <= 0.0 {
            continue;
        }
        let (x, y) = (bounds.x0 + (i % bw) as i32, bounds.y0 + (i / bw) as i32);
        if rect.contains(x, y)
            && let Some(o) = out.get_mut((y - rect.y0) as usize * w + (x - rect.x0) as usize)
        {
            *o = *c;
        }
    }
    out
}

/// A removal's hole widened by this share of its larger side (at least [`REMOVE_GROW_MIN`], at
/// most [`REMOVE_GROW_MAX`] pixels). Unnamed ("the object in the selected area"), the fill
/// model redraws an object whose hole hugs it: the boat (145×37) grown 15 px came back at both
/// seeds, grown 30 at one of two, grown 45 and 60 was gone at both, and the necklace (a 6-px
/// chain) came back at one seed of two even grown 123. Named by the vision-language rewriter
/// ("Remove the red boat…", "Remove the gold necklace…") the boat went at 15 and 30 and the
/// necklace at 64 and 123, every seed, so the growth only has to take in the object's halo,
/// soft edge and the start of its reflection (a hole exactly the stroke left a smudge of the
/// boat's). `caf-grow-live.ps1`, `caf-shape-live.ps1`, `caf-grow-named-live.ps1`,
/// `necklace-*-live.ps1`, 2026-10-21.
const REMOVE_GROW_SHARE: f64 = 0.15;
const REMOVE_GROW_MIN: u32 = 16;
const REMOVE_GROW_MAX: u32 = 96;

/// A stroke removes the object the detector finds under it whole only when the stroke covers
/// at least this share of that object: a dab on a blemish (under 1 % of the face the detector
/// finds under it) must heal the blemish, not remove the face. A quarter was too strict: a
/// stroke along half a watch covers 22 % of it (measured 2026-10-21 on the robustness set).
const WHOLE_OBJECT_MIN_COVERED: f64 = 0.10;
/// …and never an object larger than this share of the canvas.
const WHOLE_OBJECT_MAX_CANVAS: f64 = 0.4;

/// How many seeds a removal runs unless told otherwise (`variations`): two, the one whose fill
/// matches its edge best shown ([`edge_mismatch`]). One seed in four or so filled a watch's
/// place with the table beside the wrist, or a crow's with blue on a grey sky (measured
/// 2026-10-21); a second seed at 6 s on the Lightning tier buys the choice.
pub const DEFAULT_REMOVAL_VARIATIONS: u32 = 2;
const MAX_REMOVAL_VARIATIONS: u32 = 4;

/// `variations` for a removal: [`DEFAULT_REMOVAL_VARIATIONS`] unless given, 1 to 4.
pub(crate) fn removal_variations(p: &Value) -> u32 {
    p.get("variations").and_then(Value::as_f64).filter(|v| v.is_finite()).map_or(DEFAULT_REMOVAL_VARIATIONS, |v| (v.round() as i64).clamp(1, i64::from(MAX_REMOVAL_VARIATIONS)) as u32)
}

/// A removal's request is at least this many pixels on a side (where the canvas allows), so
/// the model sees real context around a small dab and the request is not upscaled many times
/// over: a 40-px dab on the necklace's pendant made a 108-px request, sent at 512 (4.7×), and
/// came back as a lighter, smoother patch of fabric (measured 2026-10-21).
const REMOVE_MIN_REQUEST: i32 = 384;

/// The margin of the close-up the rewriter names a removal's object from, a share of the grown
/// stroke's longer side, by which model answers. ComfyUI's Qwen3-VL 4B names the whole object
/// from a loose close-up and names a part from a tight one (the dog's chain, the bicycle's
/// handlebar); the native worker's 8B names the most salient thing in a loose one, the wearer
/// ("the man wearing the cap and mask" for a stroke on the mask, three seeds of three) and the
/// worn thing from a tight one. Measured 2026-10-26 with `naming-probe.ps1` on the stock set's
/// hard scenes, three seeds, both strokes.
const REMOVE_CLOSE_MARGIN: f64 = 0.25;
const REMOVE_CLOSE_MARGIN_WORKER: f64 = 0.1;

/// The two views the rewriter names a removal's object from, for a stroke over `hole`: the
/// surroundings (at least [`REMOVE_MIN_REQUEST`] on a side) and the close-up (the hole grown as
/// the removal grows it, with the margin of the model that answers: `by_worker` when the native
/// worker's own 8B does).
pub(crate) fn naming_views(hole: Rect, canvas: Rect, by_worker: bool) -> Result<(Rect, Rect)> {
    let margin = if by_worker { REMOVE_CLOSE_MARGIN_WORKER } else { REMOVE_CLOSE_MARGIN };
    let close = fill_rect(hole.inflate(remove_grow(hole) as i32).intersect(&canvas), margin, canvas)?;
    Ok((at_least(close, REMOVE_MIN_REQUEST, canvas), close))
}

/// `rect` widened about its centre to at least `min_side` on each side, slid inside `canvas`
/// (clipped only where the canvas is smaller), on the request grid.
fn at_least(rect: Rect, min_side: i32, canvas: Rect) -> Rect {
    let span = |lo: i32, hi: i32, clo: i32, chi: i32| -> (i32, i32) {
        let len = hi - lo;
        if len >= min_side {
            return (lo, hi);
        }
        let want = min_side.min(chi - clo);
        let lo2 = (lo - (want - len) / 2).max(clo).min(chi - want);
        (lo2, lo2 + want)
    };
    let (x0, x1) = span(rect.x0, rect.x1, canvas.x0, canvas.x1);
    let (y0, y1) = span(rect.y0, rect.y1, canvas.y0, canvas.y1);
    align_to_grid(Rect::new(x0, y0, x1, y1).intersect(&canvas), canvas)
}

/// An object the detector found under the stroke's spots (its point prompt, no name) is
/// taken whole only when its box is at most this many times the stroke's in each direction:
/// the point prompt on the glasses returned the face, on a shirt's logo the shirt (measured
/// 2026-10-21); the object the rewriter names is found by name instead, with no such limit
/// (half a stroke on a necklace's chain names the whole necklace).
const POINT_OBJECT_MAX_SCALE: f64 = 1.5;

/// Whether an object's box `ob` is within [`POINT_OBJECT_MAX_SCALE`] of the stroke's `sb`.
fn fits_stroke(ob: Rect, sb: Rect) -> bool {
    f64::from(ob.width()) <= POINT_OBJECT_MAX_SCALE * f64::from(sb.width()) && f64::from(ob.height()) <= POINT_OBJECT_MAX_SCALE * f64::from(sb.height())
}

/// Whether a stroke (`stroke`, one value per pixel of `sb`) covers enough of an object (`obj`
/// over the canvas, `on` pixels of it) for a removal to take the object whole
/// ([`WHOLE_OBJECT_MIN_COVERED`], [`WHOLE_OBJECT_MAX_CANVAS`]).
fn stroke_takes(stroke: &[f32], sb: Rect, obj: &[f32], on: u64, canvas: Rect) -> bool {
    let cw = canvas.width().max(1) as usize;
    let bw = sb.width().max(1) as usize;
    let covered = stroke
        .iter()
        .enumerate()
        .filter(|(i, c)| {
            **c >= 0.5 && {
                let (x, y) = (sb.x0 + (*i % bw) as i32, sb.y0 + (*i / bw) as i32);
                canvas.contains(x, y) && obj.get((y - canvas.y0) as usize * cw + (x - canvas.x0) as usize).copied().unwrap_or(0.0) >= 0.5
            }
        })
        .count() as f64;
    let canvas_px = f64::from(canvas.width()) * f64::from(canvas.height());
    covered >= WHOLE_OBJECT_MIN_COVERED * on as f64 && (on as f64) <= WHOLE_OBJECT_MAX_CANVAS * canvas_px
}

/// `coverage` (over `area`) unioned with an object's coverage `obj` (over the canvas, its
/// bounds `ob`): the union over `area ∪ ob`, limited to the selection `sel` when there is one.
fn take_in(area: Rect, coverage: &[f32], obj: &[f32], ob: Rect, canvas: Rect, sel: Option<&photocraft_raster::Surface>) -> (Rect, Vec<f32>) {
    let a = area.union(&ob);
    let mut u = embed_coverage(area, coverage, a);
    let (cw, aw) = (canvas.width().max(1) as usize, a.width().max(1) as usize);
    for (i, v) in u.iter_mut().enumerate() {
        let (x, y) = (a.x0 + (i % aw) as i32, a.y0 + (i / aw) as i32);
        if canvas.contains(x, y) {
            *v = v.max(obj.get((y - canvas.y0) as usize * cw + (x - canvas.x0) as usize).copied().unwrap_or(0.0));
        }
    }
    if let Some(sel) = sel {
        let k = sel.channels().max(1);
        for (c, s) in u.iter_mut().zip(sel.read_region(a).chunks_exact(k)) {
            *c *= s.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
        }
    }
    (a, u)
}

/// The object a rewritten removal names: "Remove the gold necklace and replace it with…" →
/// "gold necklace"; `None` when the rewrite does not start by removing something, or names
/// it in more than six words (a sentence, not a name). What the detector is then asked for.
pub(crate) fn named_object(rewritten: &str) -> Option<String> {
    let t = rewritten.trim();
    let rest = ["Remove the ", "remove the ", "Remove ", "remove "].iter().find_map(|p| t.strip_prefix(p))?;
    let cut = [
        ", ", " and replace", " and fill", " from ", " with ", " so ", ".", ";", " replacing", " completely", " entirely", " to ", " in the ", " on the ", " perched", " sitting",
        " standing", " lying", " hanging", " taped", " mounted", " parked", " walking", " resting", " placed", " attached", " leaning",
    ]
    .iter()
    // A word after an article describes the next thing rather than ending the name: "the
    // black silhouette of the standing person" was cut to "black silhouette of the" (measured
    // 2026-10-26).
    .filter_map(|m| rest.match_indices(m).map(|(i, _)| i).find(|&i| !rest.get(..i).is_some_and(|b| [" the", " a", " an"].iter().any(|a| b.ends_with(a)))))
    .min()
    .unwrap_or(rest.len());
    let name = rest.get(..cut).unwrap_or("").trim();
    // "the painted-over area", "the selected region": the rewriter naming the selection
    // instead of what is in it (measured); the generic wording does better than that.
    let lower = name.to_lowercase();
    let not_a_thing = ["area", "region", "part", "selection", "selected", "painted", "marked", "outlined", "view", "section", "spot", "portion", "object", "item", "thing", "element", "border", "frame", "edge"]
        .iter()
        .any(|w| lower.split(|c: char| !c.is_alphanumeric()).any(|t| t == *w));
    (!name.is_empty() && !not_a_thing && name.split_whitespace().count() <= 6).then(|| name.to_string())
}

/// The fill's prompt for a removal the rewriter named: [`REMOVE_PROMPT`] with the name in it.
/// The rewriter is not asked what lies behind the object: its answers were wrong often
/// enough ("the beige wall" behind glasses on a face, "light blue sky" for a grey one, and
/// the model painted those) that the fill model's own reading of the surroundings, over two
/// seeds with the better edge match kept, does better (measured 2026-10-21).
pub(crate) fn removal_prompt(name: &str) -> String {
    REMOVE_PROMPT.replace(REMOVE_PROMPT_OBJECT, &format!("the {name}"))
}

/// How far a removal's hole of bounds `b` grows.
fn remove_grow(b: Rect) -> u32 {
    let side = f64::from(b.width().max(b.height()));
    ((side * REMOVE_GROW_SHARE).round() as u32).clamp(REMOVE_GROW_MIN, REMOVE_GROW_MAX)
}

/// Remove what `coverage` marks (one value per pixel of `bounds`) with the fill model, as
/// Generative Fill does with [`REMOVE_PROMPT`]: the hole widened by `grow` (or, when `grow`
/// is 0, by [`remove_grow`]), its pixels kept in the request (or flattened to grey when
/// `erase`, which 2511 paints back: measured), the result a masked layer named `label` above
/// the active one. With `object`, the detector is asked first for the object under the stroke
/// and the hole takes it in whole when the stroke covers enough of it (the fill model redraws
/// an object part of which stays in view: measured on the boat's bow). With `enhance`, the
/// vision-language rewriter names the object with the picture in view ("Remove the gold
/// necklace…"): what removes a thin necklace over skin, which the generic wording left or
/// redrew at half the seeds however far the hole was grown (measured; the rewriter failing
/// leaves the generic wording).
/// The classical Content-Aware Fill, Remove tool and Spot Healing run this when asked to be
/// generative and a backend is configured ([`generative_configured`]). With `within_selection`
/// a selection limits the hole, as it limits a brush.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generative_remove(
    s: &mut Session,
    label: &str,
    bounds: Rect,
    coverage: Vec<f32>,
    grow: u32,
    erase: bool,
    within_selection: bool,
    object: Option<crate::select_ml_cmds::ObjectUnderStroke>,
    enhance: bool,
    variations: u32,
    seed: Option<u64>,
    prefill: bool,
    tuning: Tuning,
) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let canvas = d.doc.bounds();
    let bounds = bounds.intersect(&canvas);
    let mut coverage = coverage;
    if within_selection && let Some(sel) = d.doc.selection.as_ref() {
        let k = sel.channels().max(1);
        for (c, s) in coverage.iter_mut().zip(sel.read_region(bounds).chunks_exact(k)) {
            *c *= s.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
        }
    }
    // A removal's hole is all or nothing wherever the brush touched: a soft brush's partial
    // coverage went into the noise mask as partial values, which told the model to keep that
    // much of the object's rim, and the boat stayed (measured 2026-10-21, `spot-generative-live.ps1`).
    for c in coverage.iter_mut() {
        *c = if *c > 0.0 { 1.0 } else { 0.0 };
    }
    let (hole, _) = crate::select_ml_cmds::coverage_bounds(&coverage, bounds);
    if hole.is_empty() {
        return Err(EngineError::Other("nothing to remove: the stroke covers no pixel that can change (check the selection)".into()));
    }
    let (default_template, default_model) = {
        let i = &s.prefs().integrations;
        (template_or(&i.default_fill_template, AUTO_TEMPLATE), i.default_edit_model.clone())
    };
    let mut auto = default_template == AUTO_TEMPLATE;
    let mut tuning = tuning;
    let base = match (auto, native21_auto(s, FILL, true)) {
        (true, Some(t)) => {
            auto = false;
            if tuning.max_pixels == MAX_FILL_REQUEST_PIXELS {
                tuning.max_pixels = NATIVE21_REMOVAL_PIXELS;
            }
            t.to_string()
        }
        (true, None) => DEFAULT_FILL_TEMPLATE.to_string(),
        (false, _) => default_template,
    };
    // An edit template as the fill template: the removal runs as a whole-crop instruction
    // ("Remove the glasses") with no noise mask, which is what an edit model was trained for,
    // and the layer mask confines the visible change to the hole.
    let as_edit = template::find(&base).map(|t| t.meta.task == Task::Edit).unwrap_or(false);
    let Common { template, prompt, negative, seed, steps, guidance, models, .. } =
        plan_common(s, FILL, &json!({"prompt": REMOVE_PROMPT, "seed": seed}), &base, if as_edit { Task::Edit } else { Task::Fill }, &default_model)?;
    let (backend, server) = backend_for(s, &template.meta.id)?;
    let purge = purge_mode(s);
    let job = FillJob {
        thinker: thinker_for(s, &backend, &server),
        backend,
        server,
        command: if as_edit { EDIT } else { FILL },
        template_id: template.meta.id.clone(),
        auto,
        auto_order: auto_order(FILL, removal_quality(s)),
        prompt,
        negative,
        seed,
        steps,
        guidance,
        name: label.to_string(),
        models,
        variations: variations.clamp(1, MAX_REMOVAL_VARIATIONS),
        edge: Edge::Soft,
        enhance,
        refs: Default::default(),
        grow,
        erase,
        rewritten: None,
        removal: true,
        prefill,
        tuning,
        object: None,
        purge,
    };
    let label_kept = label.to_string();
    crate::jobs::edit_job(
        s,
        label,
        move |doc, active, ctx| {
            let mut job = job;
            let stroke = (bounds, coverage.clone());
            let (mut area, mut coverage) = (bounds, coverage);
            let sel = if within_selection { doc.selection.as_ref() } else { None };
            // The object under the stroke, whole, when the stroke covers enough of it. The
            // rewriter names it first, with a close-up of the painted part and its surroundings
            // in view, and the detector is asked by that name (a point prompt on a thin object
            // returns only the brushed part, on a part of a bigger segment the segment: the
            // face for the glasses, the shirt for its logo; a text prompt the named thing).
            // The point prompt is the fallback when nothing is named or found, for an object
            // no bigger than the stroke's box and a half.
            let mut taken = false;
            if job.enhance {
                ctx.progress(0.0, "Naming the object");
                let (hole, _) = crate::select_ml_cmds::coverage_bounds(&coverage, area);
                let by_worker = job.server == NATIVE_SERVER_KEY && same_backend(&job.thinker, &job.backend);
                let (wide_rect, close_rect) = naming_views(hole, canvas, by_worker)?;
                let crop = |r: Rect| -> Result<Rgba8> {
                    let c = photocraft_compose::render(doc, r).to_rgba8();
                    Rgba8::new(c.width, c.height, c.pixels).map_err(gen_err)
                };
                let seen = side_by_side(&crop(wide_rect)?, &crop(close_rect)?)?;
                let (rw, rh) = request_size(seen.width, seen.height);
                let seen = if (rw, rh) == (seen.width, seen.height) { seen } else { resize_rgba8(&seen, rw, rh)? };
                let place = place_in_words(hole, wide_rect);
                // The rewriter's sentence gives the object's name and what lies behind it; the
                // fill's prompt is composed from them with the measured generic wording (the
                // rewriter's own closing clauses listed the scene's neighbours and the model
                // painted them: a rock for the boat, the table over a wrist).
                dump_views(&seen);
                if let Ok(Some(p)) =
                    enhance_prompt(job.thinker.as_ref(), EnhanceKind::Remove, &job.prompt, Some(&seen), Some(&place), job.seed, &JobProgress::span(ctx, 0.01, 0.04))
                    && let Some(name) = named_object(&p)
                {
                    job.rewritten = Some(removal_prompt(&name));
                    job.object = Some(name.clone());
                    if let Some(finder) = &object {
                        // "The man wearing the cap and mask": the wearer and each worn thing
                        // are asked for, and the one whose mask matches the stroke best is
                        // taken and named (the 8B names the wearer from a crop with a little
                        // face in it, measured 2026-10-27). Otherwise the name as given, then
                        // its last two words, then its last word: the detector found nothing
                        // called "white ceramic coffee cup" (measured).
                        let worn = wearer_candidates(&name);
                        let pick_best = worn.is_some();
                        let mut best: Option<(f64, String, Vec<f32>, Rect)> = None;
                        for candidate in worn.unwrap_or_else(|| name_candidates(&name)) {
                            ctx.progress(0.04, &format!("Finding the {candidate}"));
                            if let Some((obj, ob, on)) = finder.find_by_text(&candidate, doc, *active, ctx)?
                                && stroke_takes(&stroke.1, stroke.0, &obj, on, canvas)
                            {
                                let score = stroke_overlap(&stroke.1, stroke.0, &obj, on, canvas);
                                if best.as_ref().is_none_or(|b| score > b.0) {
                                    best = Some((score, candidate, obj, ob));
                                }
                                if !pick_best {
                                    break;
                                }
                            }
                        }
                        if let Some((_, candidate, obj, ob)) = best {
                            (area, coverage) = take_in(area, &coverage, &obj, ob, canvas, sel);
                            taken = true;
                            if pick_best {
                                job.rewritten = Some(removal_prompt(&candidate));
                                job.object = Some(candidate);
                            }
                        }
                    }
                }
                // `nameShadow` on a whole-picture route: the prompt takes the object's shadow
                // with it ("the golden retriever and its shadow"); the model removes both and
                // the change band shows the shadow's part.
                if job.tuning.name_shadow
                    && job.tuning.shadow
                    && template::find(&job.template_id).is_ok_and(|t| t.meta.family == "qwen-image-2.1")
                    && let Some(o) = job.object.as_ref()
                {
                    job.rewritten = Some(removal_prompt(&format!("{o} and its shadow")));
                }
                // Named or not, the fill's prompt is settled here.
                job.enhance = false;
                release_thinker(&job.server, job.purge, &job.thinker, &job.backend);
                ctx.check()?;
            }
            if !taken && let Some(finder) = &object {
                ctx.progress(0.05, "Finding the object");
                if let Some((obj, ob, on)) = finder.find(doc, *active, ctx)?
                    && fits_stroke(ob, stroke.0)
                    && stroke_takes(&stroke.1, stroke.0, &obj, on, canvas)
                {
                    (area, coverage) = take_in(area, &coverage, &obj, ob, canvas, sel);
                }
                ctx.check()?;
            }
            let (hole, _) = crate::select_ml_cmds::coverage_bounds(&coverage, area);
            if hole.is_empty() {
                return Err(EngineError::Other("nothing to remove: the stroke covers no pixel that can change (check the selection)".into()));
            }
            if job.grow == 0 {
                job.grow = remove_grow(hole);
            }
            let rect = fill_rect(hole.inflate(job.grow.min(MAX_GROW) as i32).intersect(&canvas), job.tuning.remove_margin, canvas)?;
            let rect = at_least(rect, REMOVE_MIN_REQUEST, canvas);
            // `keepObject`: the object's pixels and silhouette as they are now, before the
            // fill's layers go in; its layer lands above them, named after what was removed.
            let kept = if job.tuning.keep_object {
                let c = photocraft_compose::render(doc, hole).to_rgba8();
                Some((Rgba8::new(c.width, c.height, c.pixels).map_err(gen_err)?, embed_coverage(area, &coverage, hole), hole, job.object.clone()))
            } else {
                None
            };
            let coverage = embed_coverage(area, &coverage, rect);
            let filled = fill_region(doc, active, ctx, job, rect, coverage, None)?;
            let kept_layer = match kept {
                Some((pixels, cov, hole, object)) => {
                    let name = match object {
                        Some(o) => format!("{label_kept}: {o}"),
                        None => format!("{label_kept}: object"),
                    };
                    let mut layer = Layer::raster(name, doc.pixel_format());
                    write_rgba8(&mut layer, &pixels, hole)?;
                    let mut mask_surface = Surface::with_default(PixelFormat::GRAY8, &[0.0]);
                    mask_surface.write_region(hole, &cov);
                    layer.mask = Some(LayerMask { surface: mask_surface, enabled: true, linked: true, density: 1.0, feather: 0.0 });
                    let nid = doc.insert_above(*active, layer);
                    *active = Some(nid);
                    Some(nid)
                }
                None => None,
            };
            Ok((filled, kept_layer))
        },
        move |((made, w, h, request, template_id), kept)| {
            let mut v = made_json(&made, w, h, request, &template_id);
            if let (Some(id), Some(o)) = (kept, v.as_object_mut()) {
                o.insert("objectLayer".to_string(), json!(id.0));
            }
            v
        },
    )
}

fn plan_fill(s: &Session, p: &Value) -> Result<FillPlan> {
    let (default_template, default_model) = {
        let integrations = &s.prefs().integrations;
        (template_or(&integrations.default_fill_template, AUTO_TEMPLATE), integrations.default_edit_model.clone())
    };
    // `auto` (the preference's default) is resolved inside the job from the server's model
    // list; the plan validates the base template it falls back to.
    let requested = opt_str(FILL, p, "template", 200)?.map(str::to_string).unwrap_or(default_template);
    let mut auto = requested == AUTO_TEMPLATE;
    let mut q = p.clone();
    if auto && let Some(o) = q.as_object_mut() {
        o.remove("template");
    }
    // References with no template named (or `auto`): the reference template, whose grey hole
    // and wrapper are made for a swap; a named template is held to its own limit below.
    let ref_specs = crate::generate_refs::specs(FILL, p)?;
    let named = p.get("template").is_some_and(|v| !v.is_null()) && !auto;
    let mut base = if auto { DEFAULT_FILL_TEMPLATE.to_string() } else { requested.clone() };
    let enhance = opt_bool(FILL, p, "enhance", s.prefs().integrations.enhance_prompts)?;
    // `fillAsEdit`: Qwen-Image-2.1 fills by instruction ("Replace the denim jacket with a yellow
    // raincoat"), the selection's thing named by the rewriter. Off unless asked: the 8B named a
    // neighbour of the selection ("the dark car" for a denim jacket, "the black trash can" for
    // a bench) on 3 of 15 swaps, where the fill template with its own rewriter swapped all but
    // the watch (`benchmarks.md`, 2026-10-27).
    let mut as_edit = false;
    if auto && ref_specs.is_empty() && let Some(t) = native21_auto(s, FILL, false) {
        as_edit = opt_bool(FILL, p, "fillAsEdit", false)?;
        base = if as_edit { native21_auto(s, EDIT, false).unwrap_or(t).to_string() } else { t.to_string() };
        auto = false;
    }
    // A fill from references on the native worker: the 2.1 edit template, which takes nine
    // references beside the picture, instead of 2511's fill-ref on ComfyUI (the Rust rebuild,
    // `goals/rust-rebuild.md` R2): "Replace the <thing> with <prompt> as image 2 shows it".
    if auto && !ref_specs.is_empty() && native21_auto(s, FILL, false).is_some() && let Some(t) = native21_auto(s, EDIT, false) {
        as_edit = true;
        base = t.to_string();
        auto = false;
    }
    if !named && !ref_specs.is_empty() && template::find(&base).is_ok_and(|t| t.meta.references.max == 0) {
        base = fill_ref_template(quality(s)).to_string();
        auto = false;
        if let Some(o) = q.as_object_mut() {
            o.remove("template");
        }
    }
    let Common { template, prompt, negative, seed, steps, guidance, models, name } =
        plan_common(s, FILL, &q, &base, if as_edit { Task::Edit } else { Task::Fill }, &default_model)?;
    let margin = opt_num(FILL, p, "margin", 0.0, 1.0)?.unwrap_or(0.25);
    let edge = parse_edge(FILL, p)?;
    let variations = match opt_num(FILL, p, "variations", 1.0, f64::from(MAX_VARIATIONS))? {
        None => 1,
        Some(x) if x.fract() == 0.0 => x as u32,
        Some(_) => return Err(bad(FILL, format!("`variations` must be a whole number from 1 to {MAX_VARIATIONS}"))),
    };

    let d = s.active().ok_or(EngineError::NoDocument)?;
    let sel = d.doc.selection.as_ref().ok_or_else(|| EngineError::Other("make a selection first".into()))?;
    let canvas = d.doc.bounds();
    let bounds = sel.content_bounds().intersect(&canvas);
    if bounds.is_empty() {
        return Err(EngineError::Other("the selection is empty".into()));
    }
    let rect = fill_rect(bounds, margin, canvas)?;
    let name = name.unwrap_or_else(|| format!("Generative Fill: {}", short(&prompt)));
    // A plain fill's model sees the picture around the selection and takes no references; a
    // fill-from-reference template takes one or two.
    let refs = crate::generate_refs::plan(s, FILL, p, &template)?;
    let grow = match opt_num(FILL, p, "grow", 0.0, f64::from(MAX_GROW))? {
        None => 0,
        Some(x) if x.fract() == 0.0 => x as u32,
        Some(_) => return Err(bad(FILL, format!("`grow` must be a whole number of pixels, 0 to {MAX_GROW}"))),
    };
    let erase = opt_bool(FILL, p, "erase", template.meta.erase_selection)?;
    let tuning = Tuning::from_params(FILL, p)?;
    Ok(FillPlan { template, prompt, negative, seed, steps, guidance, name, models, rect, variations, auto, edge, enhance, refs, grow, erase, tuning })
}

/// The first of `candidates` whose model files the server lists (one `/models/<folder>` call per
/// folder), else `fallback`. Only permissive templates belong in `candidates`.
pub(crate) fn resolve_auto(backend: &dyn GenerativeBackend, candidates: &[&str], fallback: &str) -> String {
    let mut folders: BTreeMap<String, Option<Vec<String>>> = BTreeMap::new();
    // The native worker's index names the templates it serves rather than model files.
    let served: Vec<String> = backend.model_files("").unwrap_or_default();
    for id in candidates {
        let Ok(t) = template::find(id) else { continue };
        let installed = served.iter().any(|s| s == id)
            || t.meta.models.iter().all(|slot| {
                let files = folders.entry(slot.folder.clone()).or_insert_with(|| backend.model_files(&slot.folder).ok());
                files.as_ref().is_some_and(|f| f.iter().any(|x| x == &slot.default))
            });
        if installed {
            return (*id).to_string();
        }
    }
    fallback.to_string()
}

/// Grow `rect` inside `canvas` until its sides are multiples of [`SIZE_STEP`], extending right
/// and down first, then left and up. A canvas too small for a side leaves that side as it is
/// (the server crops to its grid and the result is resampled back).
pub(crate) fn align_to_grid(rect: Rect, canvas: Rect) -> Rect {
    let step = SIZE_STEP as i32;
    let grow = |lo: i32, hi: i32, clo: i32, chi: i32| -> (i32, i32) {
        let len = (hi - lo).max(1) as u32;
        let want = (len.div_ceil(SIZE_STEP) * SIZE_STEP) as i32;
        if want > chi - clo {
            return (lo, hi);
        }
        let hi2 = (lo + want).min(chi);
        let lo2 = (hi2 - want).max(clo);
        (lo2, lo2 + want)
    };
    let _ = step;
    let (x0, x1) = grow(rect.x0, rect.x1, canvas.x0, canvas.x1);
    let (y0, y1) = grow(rect.y0, rect.y1, canvas.y0, canvas.y1);
    Rect::new(x0, y0, x1, y1)
}

/// The size a `w`×`h` fill request is sent at: scaled down (aspect kept, multiples of 16) over
/// [`MAX_FILL_REQUEST_PIXELS`], scaled up when the longer side is under
/// [`MIN_FILL_REQUEST_SIDE`], otherwise unchanged.
pub(crate) fn request_size(w: u32, h: u32) -> (u32, u32) {
    request_size_within(w, h, MAX_FILL_REQUEST_PIXELS)
}

/// [`request_size`] with its own pixel cap (`maxPixels`).
pub(crate) fn request_size_within(w: u32, h: u32, max_pixels: u64) -> (u32, u32) {
    let (w, h) = fit_pixels(w, h, max_pixels);
    let longer = w.max(h);
    if longer == 0 || longer >= MIN_FILL_REQUEST_SIDE {
        return (w, h);
    }
    let scale = f64::from(MIN_FILL_REQUEST_SIDE) / f64::from(longer);
    let fit = |v: u32| (((f64::from(v) * scale).round() as u32) / SIZE_STEP * SIZE_STEP).max(MIN_SIDE);
    (fit(w), fit(h))
}

/// `w`×`h` shrunk (aspect kept, multiples of 16, at least 64 on a side) until it has at most
/// `max_pixels` pixels; unchanged when it already fits.
pub(crate) fn fit_pixels(w: u32, h: u32, max_pixels: u64) -> (u32, u32) {
    let pixels = u64::from(w) * u64::from(h);
    if pixels <= max_pixels || w == 0 || h == 0 {
        return (w, h);
    }
    let scale = (max_pixels as f64 / pixels as f64).sqrt();
    let fit = |v: u32| (((f64::from(v) * scale) as u32) / SIZE_STEP * SIZE_STEP).max(MIN_SIDE);
    (fit(w), fit(h))
}

/// Lanczos-3 kernel, widened by `scale` (> 1 when shrinking) so downsampling averages instead of
/// aliasing.
fn lanczos3(x: f32) -> f32 {
    let x = x.abs();
    if x < 1e-6 {
        1.0
    } else if x >= 3.0 {
        0.0
    } else {
        let px = std::f32::consts::PI * x;
        3.0 * px.sin() * (px / 3.0).sin() / (px * px)
    }
}

/// Triangle (tent) kernel: no negative lobes, for masks.
fn triangle(x: f32) -> f32 {
    (1.0 - x.abs()).max(0.0)
}

/// A resampling kernel and its radius in source pixels (at 1:1).
#[derive(Clone, Copy)]
struct Filter {
    kernel: fn(f32) -> f32,
    radius: f32,
}
const LANCZOS: Filter = Filter { kernel: lanczos3, radius: 3.0 };
const TENT: Filter = Filter { kernel: triangle, radius: 1.0 };

/// Per-output-index taps of a separable resampler: (first source index, normalised weights).
fn taps(src_len: usize, dst_len: usize, f: Filter) -> Vec<(usize, Vec<f32>)> {
    let scale = src_len as f32 / dst_len as f32;
    let widen = scale.max(1.0);
    let support = f.radius * widen;
    (0..dst_len)
        .map(|i| {
            let centre = (i as f32 + 0.5) * scale - 0.5;
            let lo = ((centre - support).floor().max(0.0)) as usize;
            let hi = ((centre + support).ceil() as usize).min(src_len.saturating_sub(1));
            let mut weights: Vec<f32> = (lo..=hi).map(|j| (f.kernel)((j as f32 - centre) / widen)).collect();
            let sum: f32 = weights.iter().sum();
            if sum.abs() > 1e-6 {
                for w in &mut weights {
                    *w /= sum;
                }
            } else if let Some(w) = weights.first_mut() {
                *w = 1.0;
            }
            (lo, weights)
        })
        .collect()
}

/// Separable resample of interleaved 8-bit samples (`ch` per pixel) from `sw`×`sh` to `dw`×`dh`.
fn resample_u8(src: &[u8], sw: usize, sh: usize, ch: usize, dw: usize, dh: usize, f: Filter) -> Vec<u8> {
    let xt = taps(sw, dw, f);
    let yt = taps(sh, dh, f);
    // Horizontal pass into f32 rows, then vertical.
    let mut mid = vec![0.0f32; dw * sh * ch];
    for y in 0..sh {
        let row = &src[y * sw * ch..((y + 1) * sw * ch).min(src.len())];
        for (x, (lo, ws)) in xt.iter().enumerate() {
            let out = &mut mid[(y * dw + x) * ch..(y * dw + x + 1) * ch];
            for (k, w) in ws.iter().enumerate() {
                let start = ((lo + k) * ch).min(row.len().saturating_sub(ch));
                for (o, p) in out.iter_mut().zip(row.iter().skip(start)) {
                    *o += w * f32::from(*p);
                }
            }
        }
    }
    let mut dst = vec![0u8; dw * dh * ch];
    for (y, (lo, ws)) in yt.iter().enumerate() {
        for x in 0..dw {
            for c in 0..ch {
                let mut acc = 0.0f32;
                for (k, w) in ws.iter().enumerate() {
                    acc += w * mid.get(((lo + k) * dw + x) * ch + c).copied().unwrap_or(0.0);
                }
                if let Some(d) = dst.get_mut((y * dw + x) * ch + c) {
                    *d = (acc + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    dst
}

/// Lanczos-3 resize (both ways), for requests over the size cap and for a backend that returns
/// a different size than it was given.
pub(crate) fn resize_rgba8(src: &Rgba8, w: u32, h: u32) -> Result<Rgba8> {
    if src.width == 0 || src.height == 0 || w == 0 || h == 0 {
        return Err(EngineError::Other("cannot resize an empty image".into()));
    }
    if (src.width, src.height) == (w, h) {
        return Ok(src.clone());
    }
    let out = resample_u8(&src.data, src.width as usize, src.height as usize, 4, w as usize, h as usize, LANCZOS);
    Rgba8::new(w, h, out).map_err(gen_err)
}

/// Soften a mask outward: full coverage at its edge falling to nothing about `2 × radius` pixels
/// beyond it, the inside untouched (two box blurs ≈ a triangle filter, the outer half of its
/// ramp doubled, then `max` with the original). The same shape ComfyUI's GrowMask + FeatherMask
/// make. As the latent noise mask it has the model re-render the band around the area and blend
/// it, with no step at the edge: a ramp that started at half coverage there (the first version)
/// left half of the tone difference as a line along the edge. As the layer mask (dithered, see
/// `dither_edge`) it fades the result into its surroundings across the same band.
pub(crate) fn feather_outward(mask: &Gray8, radius: usize) -> Result<Gray8> {
    let (w, h) = (mask.width as usize, mask.height as usize);
    if radius == 0 || w == 0 || h == 0 {
        return Ok(mask.clone());
    }
    let mut cur: Vec<f32> = mask.data.iter().map(|v| f32::from(*v)).collect();
    for _ in 0..2 {
        cur = box_blur(&cur, w, h, radius, true);
        cur = box_blur(&cur, w, h, radius, false);
    }
    let out: Vec<u8> = cur.iter().zip(&mask.data).map(|(b, o)| (*o).max((b * 2.0 + 0.5).clamp(0.0, 255.0) as u8)).collect();
    Gray8::new(mask.width, mask.height, out).map_err(gen_err)
}

/// One box-blur pass of radius `r` along rows (`horizontal`) or columns, edges clamped.
fn box_blur(src: &[f32], w: usize, h: usize, r: usize, horizontal: bool) -> Vec<f32> {
    let (len, lines) = if horizontal { (w, h) } else { (h, w) };
    let at = |line: usize, i: usize| -> f32 {
        let idx = if horizontal { line * w + i } else { i * w + line };
        src.get(idx).copied().unwrap_or(0.0)
    };
    let mut out = vec![0.0f32; src.len()];
    let window = (2 * r + 1) as f32;
    for line in 0..lines {
        // Running sum over a window clamped to the line's ends.
        let mut sum = 0.0f32;
        for i in 0..=r.min(len.saturating_sub(1)) {
            sum += at(line, i);
        }
        sum += at(line, 0) * r as f32;
        for i in 0..len {
            let idx = if horizontal { line * w + i } else { i * w + line };
            if let Some(o) = out.get_mut(idx) {
                *o = sum / window;
            }
            let leaving = at(line, i.saturating_sub(r));
            let entering = at(line, (i + r + 1).min(len.saturating_sub(1)));
            sum += entering - leaving;
        }
    }
    out
}

/// The outward feather of a fill's request mask: 4 % of the longer side, 6 to 48 pixels. The
/// model re-renders that band around the selection and the soft layer mask fades across it.
pub(crate) fn feather_radius(w: u32, h: u32) -> usize {
    ((f64::from(w.max(h)) * 0.04).round() as usize).clamp(6, 48)
}

/// The feather of an expand's request mask into the picture: 4 % of the longer side, 8 to 80
/// pixels (a band of twice that). The model re-renders that band of the original, so the new
/// area's tone is carried across the old edge instead of meeting it. Measured on the 1024-px
/// lighthouse: at 8 % the model took the licence to move the horizon inside the band (a ghosted
/// double horizon at one seed), at 2 % the old edge still read as a line; 4 % did neither.
pub(crate) fn outpaint_feather_radius(w: u32, h: u32) -> usize {
    ((f64::from(w.max(h)) * 0.04).round() as usize).clamp(8, 80)
}

/// Mask resize with a tent filter (no ringing, coverage stays within 0..=255).
pub(crate) fn resize_gray8(src: &Gray8, w: u32, h: u32) -> Result<Gray8> {
    if src.width == 0 || src.height == 0 || w == 0 || h == 0 {
        return Err(EngineError::Other("cannot resize an empty mask".into()));
    }
    if (src.width, src.height) == (w, h) {
        return Ok(src.clone());
    }
    let out = resample_u8(&src.data, src.width as usize, src.height as usize, 1, w as usize, h as usize, TENT);
    Gray8::new(w, h, out).map_err(gen_err)
}

/// Write RGBA8 pixels into a raster layer over `rect`, converting to the layer's own format
/// (gray, CMYK, 16-bit, float…); `src` must have `rect`'s size.
pub(crate) fn write_rgba8(layer: &mut Layer, src: &Rgba8, rect: Rect) -> Result<()> {
    if (src.width, src.height) != (rect.width(), rect.height()) {
        return Err(EngineError::Other("internal error: the image and its rectangle differ in size".into()));
    }
    let surf = crate::pixels_mut(layer)?;
    let fmt = surf.format();
    let n = fmt.channels();
    let mut px = vec![0.0f32; src.width as usize * src.height as usize * n];
    let mut tmp = [0.0f32; 8];
    for (i, s4) in src.data.as_chunks::<4>().0.iter().enumerate() {
        let rgba = [f32::from(s4[0]) / 255.0, f32::from(s4[1]) / 255.0, f32::from(s4[2]) / 255.0, f32::from(s4[3]) / 255.0];
        let used = photocraft_raster::from_rgba_into(&fmt, rgba, &mut tmp).min(n);
        if let (Some(dst), Some(src)) = (px.get_mut(i * n..i * n + used), tmp.get(..used)) {
            dst.copy_from_slice(src);
        }
    }
    surf.write_region(rect, &px);
    Ok(())
}

/// Everything `generate.image` needs, validated before any network work.
struct ImagePlan {
    template: Template,
    /// The prompt the model gets (wrapped for transparency when asked).
    prompt: String,
    /// The prompt as typed, remembered on the layer.
    typed: String,
    transparent: bool,
    /// The template's wording around a transparent request (`{prompt}` inside), empty otherwise;
    /// a rewritten prompt is wrapped in it the same way.
    wrap: String,
    /// Rewrite the prompt with the vision-language model first (`enhance`).
    enhance: bool,
    negative: String,
    seed: u64,
    steps: u32,
    guidance: f32,
    name: String,
    models: Vec<(String, String)>,
    /// The size asked of the model (rounded to the latent grid).
    width: u32,
    height: u32,
    /// `true`: a new document of the result's size; `false`: a new layer over the whole canvas.
    to_document: bool,
    /// How many results (consecutive seeds); only the first is visible.
    variations: u32,
    /// Reference images (`references`).
    refs: crate::generate_refs::Refs,
    /// LoRA adapters for the native engine (`loras`; see [`Tuning::loras`]).
    loras: Vec<(String, f32)>,
}

/// The template `generate.image` switches to when references come with the default template,
/// which takes none: permissive and runs on the edit model the fill already uses.
pub const DEFAULT_IMAGE_REF_TEMPLATE: &str = "qwen-edit-2511/image-ref";
/// The same on the native worker: Qwen-Image-2.1's text to image with the references in its
/// encoder's view (the Rust rebuild R2), when research-only models are allowed and the worker
/// runs the family.
pub const NATIVE_IMAGE_REF_TEMPLATE: &str = "qwen-2.1/image-ref";

/// The template an image from references runs when none that takes references is named.
pub fn image_ref_template(s: &Session) -> &'static str {
    if s.prefs().integrations.allow_research_models && loras::runs_native(s, NATIVE_IMAGE_REF_TEMPLATE) {
        NATIVE_IMAGE_REF_TEMPLATE
    } else {
        DEFAULT_IMAGE_REF_TEMPLATE
    }
}

/// Shared validation of the prompt, template, licence gate, seed, steps, guidance and model
/// override for both generative commands.
pub(crate) struct Common {
    pub(crate) template: Template,
    pub(crate) prompt: String,
    pub(crate) negative: String,
    pub(crate) seed: u64,
    pub(crate) steps: u32,
    pub(crate) guidance: f32,
    pub(crate) models: Vec<(String, String)>,
    pub(crate) name: Option<String>,
}

pub(crate) fn plan_common(s: &Session, cmd: &str, p: &Value, default_template: &str, task: Task, default_model: &str) -> Result<Common> {
    let prompt = opt_str(cmd, p, "prompt", MAX_PROMPT_CHARS)?
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .ok_or_else(|| bad(cmd, "`prompt` is required: describe what to generate"))?
        .to_string();
    let negative = opt_str(cmd, p, "negative", MAX_PROMPT_CHARS)?.unwrap_or("").to_string();
    let template_id = opt_str(cmd, p, "template", 100)?.unwrap_or(default_template);
    let template = template::find(template_id).map_err(|e| bad(cmd, e.to_string()))?;
    if template.meta.task != task {
        return Err(bad(cmd, format!("template `{template_id}` is a {:?} template, not {task:?}", template.meta.task)));
    }
    let integrations = &s.prefs().integrations;
    if template.meta.license == License::Research && !integrations.allow_research_models {
        return Err(EngineError::Other(format!(
            "`{}` uses a research-only model; turn on Allow Research-Only Models in Preferences › AI Integrations to use it",
            template.meta.id
        )));
    }
    let seed = parse_seed(cmd, p)?;
    // 0 (the generated dialog's default) means the template's own step count.
    let steps = opt_num(cmd, p, "steps", 0.0, 250.0)?.map_or(0, |x| x.round() as u32);
    let guidance = opt_num(cmd, p, "guidance", 0.0, 30.0)?.map_or(0.0, |x| x as f32);
    let name = opt_str(cmd, p, "name", 200)?.map(str::to_string);
    let mut models = Vec::new();
    let model = opt_str(cmd, p, "model", 200)?.map(str::to_string).or_else(|| Some(default_model.trim().to_string()).filter(|m| !m.is_empty()));
    if let Some(m) = model {
        let slot =
            template.meta.models.first().map(|s| s.placeholder.clone()).ok_or_else(|| bad(cmd, format!("template `{template_id}` has no model slot")))?;
        models.push((slot, m));
    }
    template.model_bindings(&models).map_err(|e| bad(cmd, e.to_string()))?;
    Ok(Common { template, prompt, negative, seed, steps, guidance, models, name })
}

/// `seed`: absent or null = random; else a whole number below 2^53.
pub(crate) fn parse_seed(cmd: &str, p: &Value) -> Result<u64> {
    match p.get("seed") {
        None | Some(Value::Null) => Ok(photocraft_genai::random_seed()),
        Some(v) => {
            let x = v
                .as_f64()
                .filter(|x| x.is_finite() && *x >= 0.0 && x.fract() == 0.0 && *x < 9_007_199_254_740_992.0)
                .ok_or_else(|| bad(cmd, "`seed` must be a non-negative integer below 2^53"))?;
            Ok(x as u64)
        }
    }
}

/// A side length rounded down to the latent grid, within the allowed range.
fn side(cmd: &str, p: &Value, key: &str, default: u32) -> Result<u32> {
    // 0 (the generated dialog's default) means "use the default size".
    let v = match opt_num(cmd, p, key, 0.0, f64::from(MAX_SIDE))?.map(|x| x.round() as u32) {
        None | Some(0) => default,
        Some(v) if v < MIN_SIDE => return Err(bad(cmd, format!("`{key}` must be 0 (default) or {MIN_SIDE}..{MAX_SIDE} (got {v})"))),
        Some(v) => v,
    };
    Ok((v / SIZE_STEP * SIZE_STEP).clamp(MIN_SIDE, MAX_SIDE))
}

fn plan_image(s: &Session, p: &Value) -> Result<ImagePlan> {
    let (default_template, default_model) = {
        let integrations = &s.prefs().integrations;
        (template_or(&integrations.default_image_template, DEFAULT_IMAGE_TEMPLATE), integrations.default_generate_model.clone())
    };
    // References with no template named: the default image template when it takes them, else
    // the reference template (its own model file, so no `model` override from the preference).
    let ref_specs = crate::generate_refs::specs(IMAGE, p)?;
    let named = p.get("template").is_some_and(|v| !v.is_null());
    let (default_template, default_model) = match template::find(&default_template) {
        Ok(t) if !named && !ref_specs.is_empty() && t.meta.references.max == 0 => (image_ref_template(s).to_string(), String::new()),
        _ => (default_template, default_model),
    };
    let Common { template, prompt, negative, seed, steps, guidance, models, name } = plan_common(s, IMAGE, p, &default_template, Task::Image, &default_model)?;
    crate::generate_refs::check_count(IMAGE, &template, ref_specs.len())?;
    let variations = parse_variations(IMAGE, p)?;
    let doc_size = s.active().map(|d| (d.doc.bounds().width(), d.doc.bounds().height()));
    let to_document = match opt_str(IMAGE, p, "target", 20)? {
        None | Some("auto") => doc_size.is_none(),
        Some("layer") => {
            if doc_size.is_none() {
                return Err(EngineError::Other("open or create a document first, or use \"target\": \"document\"".into()));
            }
            false
        }
        Some("document") => true,
        Some(other) => return Err(bad(IMAGE, format!("`target` must be \"auto\", \"layer\" or \"document\" (got `{other}`)"))),
    };
    let (dw, dh) = if to_document { (1024, 1024) } else { doc_size.unwrap_or((1024, 1024)) };
    let width = side(IMAGE, p, "width", dw)?;
    let height = side(IMAGE, p, "height", dh)?;
    let name = name.unwrap_or_else(|| format!("Generated: {}", short(&prompt)));
    // A transparent result: the template says how its model is asked for one (Qwen-Image-2.1
    // decodes RGBA); the layer keeps the alpha the PNG comes back with.
    let transparent = opt_bool(IMAGE, p, "transparent", false)?;
    let typed = prompt.clone();
    let mut wrap = String::new();
    let prompt = if transparent {
        let format = template.meta.prompt_format_transparent.trim();
        if format.is_empty() {
            return Err(bad(IMAGE, format!("template `{}` cannot output transparency; `qwen-2.1/image` can", template.meta.id)));
        }
        wrap = format.to_string();
        format.replace("{prompt}", &prompt)
    } else {
        prompt
    };
    let enhance = opt_bool(IMAGE, p, "enhance", s.prefs().integrations.enhance_prompts)?;
    let refs = crate::generate_refs::Refs { images: crate::generate_refs::resolve(s, &ref_specs)?, specs: ref_specs };
    let loras = loras_of(IMAGE, p)?;
    Ok(ImagePlan {
        loras,
        template,
        prompt,
        typed,
        transparent,
        wrap,
        enhance,
        negative,
        seed,
        steps,
        guidance,
        name,
        models,
        width,
        height,
        to_document,
        variations,
        refs,
    })
}

/// The prompt a `generate.image` job sends, and what the layer remembers as `enhanced`: the
/// typed idea expanded by the vision-language model when asked (then wrapped for transparency
/// as the typed one was), else the plan's prompt and an empty string.
fn image_prompt(backend: &dyn GenerativeBackend, ctx: &JobCtx, enhance: bool, typed: &str, planned: String, wrap: &str, seed: u64) -> (String, String) {
    if !enhance {
        return (planned, String::new());
    }
    ctx.progress(0.02, "Improving the prompt");
    match enhance_prompt(backend, EnhanceKind::Image, typed, None, None, seed, &JobProgress::span(ctx, 0.02, 0.05)) {
        Ok(Some(e)) => {
            let p = if wrap.is_empty() { e.clone() } else { wrap.replace("{prompt}", &e) };
            (p, e)
        }
        Ok(None) | Err(_) => (planned, String::new()),
    }
}

fn image_enabled(_: &Session) -> std::result::Result<(), String> {
    web_unavailable()
}

fn run_image(s: &mut Session, p: &Value) -> Result<Value> {
    let plan = plan_image(s, p)?;
    // Per template: under `auto` the native worker takes the families it serves at parity.
    let (backend, server) = backend_for(s, &plan.template.meta.id)?;
    let thinker = thinker_for(s, &backend, &server);
    let server_doc = server.clone();
    let purge = purge_mode(s);
    let ImagePlan {
        template,
        prompt,
        typed,
        transparent,
        wrap,
        enhance,
        negative,
        seed,
        steps,
        guidance,
        name,
        models,
        width,
        height,
        to_document,
        variations,
        refs,
        loras,
    } = plan;
    let label = name.clone();
    let tid = template.meta.id.clone();
    let params: BTreeMap<String, Value> = Tuning { loras, ..Tuning::default() }.loras_param().map(|l| BTreeMap::from([("loras".to_string(), l)])).unwrap_or_default();
    // What the result layer remembers (`generate.similar` re-runs it).
    let info = GenerativeInfo {
        command: IMAGE.to_string(),
        prompt: typed.clone(),
        negative: negative.clone(),
        template: tid.clone(),
        seed,
        steps,
        guidance,
        name: name.clone(),
        width,
        height,
        transparent,
        references: refs.specs,
        ..GenerativeInfo::default()
    };
    let mut req = Request {
        template: template.meta.id.clone(),
        prompt,
        negative,
        seed,
        steps,
        guidance,
        image: None,
        mask: None,
        models,
        size: Some((width, height)),
        params,
        references: refs.images,
    };
    let n = variations.max(1);
    // A layer name per result: "Generated: …" alone, or with "(2/3)".
    let numbered = move |base: &str, i: u32| if n > 1 { format!("{base} ({}/{n})", i + 1) } else { base.to_string() };
    if to_document {
        let doc_name = name.clone();
        return crate::jobs::run(
            s,
            &label,
            false,
            move |ctx| {
                let (prompt, enhanced) = image_prompt(thinker.as_ref(), ctx, enhance, &typed, req.prompt, &wrap, seed);
                req.prompt = prompt;
                ctx.check()?;
                let resps = run_variations(backend.as_ref(), &server_doc, &mut req, n, seed, purge, ctx)?;
                Ok((resps, enhanced))
            },
            move |s, (resps, enhanced): (Vec<photocraft_genai::Response>, String)| {
                // One document of the first result's size; every result a layer, the first visible.
                let first = resps.first().and_then(|r| r.images.first()).ok_or_else(|| EngineError::Other("the backend returned no image".into()))?;
                let (w, h) = (first.width, first.height);
                let mut doc = Document::new(doc_name, Size::new(w, h), ColorMode::Rgb, SampleType::U8);
                let mut ids = Vec::new();
                let mut seeds = Vec::new();
                for (i, resp) in resps.iter().enumerate() {
                    let Some(out) = resp.images.first() else { continue };
                    let out = if (out.width, out.height) == (w, h) { out.clone() } else { resize_rgba8(out, w, h)? };
                    let mut layer = Layer::raster(numbered("Generated", i as u32), doc.pixel_format());
                    write_rgba8(&mut layer, &out, doc.bounds())?;
                    set_generative_info(
                        &mut layer,
                        &GenerativeInfo { seed: resp.seed, rect: [0, 0, w as i32, h as i32], enhanced: enhanced.clone(), ..info.clone() },
                    );
                    layer.visible = i == 0;
                    ids.push(layer.id.0);
                    seeds.push(resp.seed);
                    doc.layers.push(layer);
                }
                let index = s.add_document(doc, None);
                let first = resps.first();
                Ok(json!({
                    "document": index, "layers": ids, "seed": first.map(|r| r.seed), "seeds": seeds, "template": tid,
                    "runId": first.map(|r| r.run_id.clone()), "width": w, "height": h, "ms": resps.iter().map(|r| r.elapsed_ms).sum::<u64>(),
                }))
            },
        );
    }
    crate::jobs::edit_job(
        s,
        &label,
        move |doc, active, ctx| {
            let (prompt, enhanced) = image_prompt(backend.as_ref(), ctx, enhance, &typed, req.prompt, &wrap, seed);
            req.prompt = prompt;
            ctx.check()?;
            let resps = run_variations(backend.as_ref(), &server, &mut req, n, seed, purge, ctx)?;
            let canvas = doc.bounds();
            let (w, h) = (canvas.width(), canvas.height());
            ctx.progress(0.96, "Placing");
            let mut made = Vec::new();
            for (i, resp) in resps.into_iter().enumerate() {
                let out = resp.images.into_iter().next().ok_or_else(|| EngineError::Other("the backend returned no image".into()))?;
                let out = if (out.width, out.height) == (w, h) { out } else { resize_rgba8(&out, w, h)? };
                let mut layer = Layer::raster(numbered(&name, i as u32), doc.pixel_format());
                write_rgba8(&mut layer, &out, canvas)?;
                set_generative_info(
                    &mut layer,
                    &GenerativeInfo { seed: resp.seed, rect: [canvas.x0, canvas.y0, w as i32, h as i32], enhanced: enhanced.clone(), ..info.clone() },
                );
                layer.visible = i == 0;
                let nid = doc.insert_above(*active, layer);
                *active = Some(nid);
                made.push((nid.0, resp.seed, resp.run_id, resp.elapsed_ms));
            }
            // The visible result is the active layer.
            *active = made.first().map(|m| LayerId(m.0));
            Ok((made, w, h))
        },
        move |(made, w, h)| {
            let first = made.first();
            json!({
                "layer": first.map(|m| m.0), "layers": made.iter().map(|m| m.0).collect::<Vec<_>>(), "seed": first.map(|m| m.1),
                "seeds": made.iter().map(|m| m.1).collect::<Vec<_>>(), "template": tid, "runId": first.map(|m| m.2.clone()),
                "width": w, "height": h, "ms": made.iter().map(|m| m.3).sum::<u64>(),
            })
        },
    )
}

/// `n` backend runs of `req` with consecutive seeds from `seed`, each its slice of the job's
/// progress.
fn run_variations(
    backend: &dyn GenerativeBackend,
    server: &str,
    req: &mut Request,
    n: u32,
    seed: u64,
    purge: &str,
    ctx: &JobCtx,
) -> Result<Vec<photocraft_genai::Response>> {
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        req.seed = seed.wrapping_add(u64::from(i)) % MAX_SEED;
        let (lo, hi) = (0.05 + 0.9 * i as f32 / n as f32, 0.05 + 0.9 * (i + 1) as f32 / n as f32);
        if n > 1 {
            ctx.progress(lo, &format!("Variation {}/{n}", i + 1));
        }
        out.push(run_switching(backend, server, req, purge, &JobProgress::span(ctx, lo, hi)).map_err(gen_err)?);
        ctx.check()?;
    }
    Ok(out)
}

/// What a fill or expand job needs once its plan is made and the backend is in hand.
struct FillJob {
    backend: Arc<dyn GenerativeBackend>,
    /// The backend that answers the vision-language questions (the rewriter, the judge, the
    /// presence check): the one serving the enhancer template, which is the ComfyUI route
    /// while the native worker runs the fill itself ([`thinker_for`]).
    thinker: Arc<dyn GenerativeBackend>,
    /// The server's key for [`run_switching`].
    server: String,
    /// The command the layers remember ([`GenerativeInfo::command`]).
    command: &'static str,
    template_id: String,
    auto: bool,
    /// The candidates `auto` picks from ([`AUTO_FILL_ORDER`] or [`AUTO_EXPAND_ORDER`]).
    auto_order: &'static [&'static str],
    prompt: String,
    negative: String,
    seed: u64,
    steps: u32,
    guidance: f32,
    name: String,
    models: Vec<(String, String)>,
    variations: u32,
    edge: Edge,
    /// Rewrite the prompt with the vision-language model first (`enhance`).
    enhance: bool,
    /// Reference images (edits, fills from a reference).
    refs: crate::generate_refs::Refs,
    /// Widen the selection by this many pixels first (fills).
    grow: u32,
    /// Flatten the selection to grey in the request image (fills).
    erase: bool,
    /// The prompt already rewritten by the vision-language model (a removal names its object
    /// before the request, so it can ask the detector for it by name); the fill takes it as
    /// the rewritten prompt instead of rewriting again.
    rewritten: Option<String>,
    /// A removal: the result's tone matched to the picture's at the hole's edge
    /// ([`tone_matched`]; a crow taken off a sky came back as a lighter patch of sky), and of
    /// several variations the one whose fill matches its edge best shown first
    /// ([`edge_mismatch`]; wood over a wrist, a wall over eyes, blue on a grey sky all lose
    /// to a seed that continued the edge). Measured 2026-10-21.
    removal: bool,
    /// Smooth the hole over in the picture the model sees ([`smooth_covered`]: a membrane
    /// from its edge, no structure), so an edit model that copies what it sees has nothing to
    /// copy; `prefill` on the removal commands, on by default: on the stock photographs it
    /// took the removals from 27 to 35 of 48 and cost nothing on the generated set
    /// (`benchmarks.md` §5i, 2026-10-21).
    prefill: bool,
    /// The sampler, scheduler and request size asked for ([`Tuning`]).
    tuning: Tuning,
    /// A removal: what the rewriter named ("gold necklace"), for the judge.
    object: Option<String>,
    /// When a run purges the server first ([`purge_mode`]).
    purge: &'static str,
}

/// The layers a job made: (id, seed, run id, ms, timings), the primary one first.
type Made = Vec<(LayerId, u64, String, u64, photocraft_genai::Timings)>;
/// A job's outcome: the layers, the rect's size, the request's size and the template used.
type Filled = (Made, u32, u32, (u32, u32), String);

/// Regenerate the pixels `coverage` marks (one value per pixel of `rect`) through the backend
/// Generative Expand's extras for [`fill_region`]: the part of the request that holds the
/// picture (the rest is empty canvas, painted mid-grey so the model sees "nothing here yet"
/// rather than black or transparent), and a wider blend: the result layer's mask is the soft
/// request mask, so the band of the picture the model re-rendered fades into the original
/// instead of meeting it at a hard line (the classic outpainting seam).
struct Outpaint {
    picture: Rect,
}

/// Regenerate the pixels `coverage` marks (one value per pixel of `rect`) through the backend
/// and add the results as layers above the active one, each masked to `coverage` (or to the
/// soft request mask when `outpaint` is set).
/// What the selection holds, named by the rewriter from the removal's two views (its
/// surroundings and a close-up, [`naming_views`]): "denim jacket". `None` when nothing usable
/// came back or the rewriter could not run.
fn name_selected(doc: &Document, job: &FillJob, area: Rect, coverage: &[f32], ctx: &JobCtx) -> Result<Option<String>> {
    ctx.progress(0.0, "Naming what is selected");
    let canvas = doc.bounds();
    let (hole, _) = crate::select_ml_cmds::coverage_bounds(coverage, area);
    if hole.is_empty() {
        return Ok(None);
    }
    let by_worker = job.server == NATIVE_SERVER_KEY && same_backend(&job.thinker, &job.backend);
    let (wide_rect, close_rect) = naming_views(hole, canvas, by_worker)?;
    let crop = |r: Rect| -> Result<Rgba8> {
        let c = photocraft_compose::render(doc, r).to_rgba8();
        Rgba8::new(c.width, c.height, c.pixels).map_err(gen_err)
    };
    let seen = side_by_side(&crop(wide_rect)?, &crop(close_rect)?)?;
    let (rw, rh) = request_size(seen.width, seen.height);
    let seen = if (rw, rh) == (seen.width, seen.height) { seen } else { resize_rgba8(&seen, rw, rh)? };
    let place = place_in_words(hole, wide_rect);
    let said = enhance_prompt(job.thinker.as_ref(), EnhanceKind::Remove, REMOVE_PROMPT, Some(&seen), Some(&place), job.seed, &JobProgress::span(ctx, 0.01, 0.04));
    Ok(said.ok().flatten().and_then(|p| named_object(&p)))
}

/// A fill's description as an edit instruction for the named thing: "Replace the denim jacket
/// with a yellow hooded raincoat, keeping everything else unchanged." (the swap benchmark's
/// wording, which the 2.1 and Krea 2 edit routes followed 27 times of 30).
pub(crate) fn swap_prompt(thing: &str, description: &str) -> String {
    let d = description.trim().trim_end_matches(['.', '!', ';', ',']).trim();
    format!("Replace the {thing} with {d}, keeping everything else unchanged.")
}

/// [`swap_prompt`] for a fill from references (image 2 is the first reference on the 2.1 edit
/// template): "Replace the denim jacket with the red sweater exactly as image 2 shows it,
/// keeping everything else unchanged."; with nothing named, the selected part of the picture.
pub(crate) fn swap_ref_prompt(thing: Option<&str>, description: &str) -> String {
    let d = description.trim().trim_end_matches(['.', '!', ';', ',']).trim();
    let what = thing.map_or_else(|| "the selected part of image 1".to_string(), |t| format!("the {t}"));
    format!("Replace {what} with {d} exactly as image 2 shows it, keeping everything else unchanged.")
}

/// The fill template a fill by instruction falls back to when nothing could be named.
fn native21_fill_fallback() -> &'static str {
    "qwen-2.1/fill"
}

fn fill_region(
    doc: &mut Document,
    active: &mut Option<LayerId>,
    ctx: &JobCtx,
    job: FillJob,
    rect: Rect,
    coverage: Vec<f32>,
    outpaint: Option<Outpaint>,
) -> Result<Filled> {
    ctx.progress(0.0, "Rendering");
    let (w, h) = (rect.width(), rect.height());
    let composite = photocraft_compose::render(doc, rect).to_rgba8();
    let mut image = Rgba8::new(composite.width, composite.height, composite.pixels).map_err(gen_err)?;
    if let Some(o) = &outpaint {
        let p = o.picture;
        prefill_outside(&mut image, Rect::new(p.x0 - rect.x0, p.y0 - rect.y0, p.x1 - rect.x0, p.y1 - rect.y0));
    }
    let FillJob {
        backend,
        thinker,
        server,
        command,
        template_id,
        auto,
        auto_order,
        prompt,
        negative,
        seed,
        steps,
        guidance,
        name,
        models,
        variations,
        edge,
        enhance,
        refs,
        grow,
        erase,
        rewritten,
        removal,
        prefill,
        tuning,
        object,
        purge,
    } = job;
    // `grow`: the selection widened, so the repainted area (and the result's mask) takes in a
    // band of the old object's surroundings; a bulkier replacement then has room. A removal
    // adapter trained on a painted highlight gets the selection nearly as it is instead
    // ([`HIGHLIGHT_GROW`]): a fat red hole came back red (measured 2026-10-22).
    let coverage_tight = (removal && outpaint.is_none()).then(|| dilate(&coverage, w as usize, h as usize, HIGHLIGHT_GROW));
    let coverage = if grow > 0 && outpaint.is_none() { dilate(&coverage, w as usize, h as usize, grow as usize) } else { coverage };
    // `erase`: the selection flattened to mid-grey in what the model sees, so it paints a hole
    // rather than copying what was there (a swap from a reference wants this).
    if erase && outpaint.is_none() {
        erase_covered(&mut image, &coverage);
    }
    let mask = Gray8::new(w, h, coverage.iter().map(|c| (c * 255.0 + 0.5) as u8).collect()).map_err(gen_err)?;
    // The model gets a softened mask so it re-renders a band around the area and blends it (much
    // wider for an expand, whose seam runs along a whole picture edge and whose sky or water tone
    // has to carry across it).
    let radius = if outpaint.is_some() { outpaint_feather_radius(w, h) } else { feather_radius(w, h) };
    let mask = feather_outward(&mask, radius)?;
    // Where the selection sits, in words, for the rewriter (a fill's instruction is about that
    // place).
    let place = (enhance && command != EDIT && outpaint.is_none()).then(|| {
        let (b, _) = crate::select_ml_cmds::coverage_bounds(&coverage, rect);
        place_in_words(b, rect)
    });
    // The layer mask: that same band, dithered, so the result fades into its surroundings with
    // no line; or exactly the selection when the caller asked for a hard edge.
    let layer_coverage: Vec<f32> = match edge {
        Edge::Soft => dither_edge(&mask.data, seed),
        Edge::Hard => coverage,
    };
    ctx.check()?;
    // Large areas go out at the models' working size and come back resampled; the layer mask
    // below keeps the selection's full-resolution edge either way.
    let (rw, rh) = request_size_within(w, h, tuning.max_pixels);
    let (image, mask) = if (rw, rh) == (w, h) { (image, mask) } else { (resize_rgba8(&image, rw, rh)?, resize_gray8(&mask, rw, rh)?) };
    let template_id = if auto { resolve_auto(backend.as_ref(), auto_order, &template_id) } else { template_id };
    // A removal's adapter variant: a template whose meta names a `highlight` (an adapter
    // trained to delete what is painted that colour) gets the selection nearly as it is,
    // painted in that colour, and the adapter's trigger sentence. The template named, when it
    // is one; else, under `auto` with two or more variations, the first installed one of
    // [`REMOVE_HIGHLIGHT_ORDER`] takes the first variation and the plain fill (the hole
    // smoothed over) the rest, so the judge picks between an adapter that removes cleanly on
    // most scenes and leaves its colour on some, and a fill that never does that.
    let n = variations.max(1);
    let adapter: Option<(String, Rgba8, Gray8)> = match (&coverage_tight, outpaint.is_none()) {
        (Some(tight), true) => {
            let named = template::find(&template_id).ok().filter(|t| t.meta.highlight.is_some()).map(|t| t.meta.id.clone());
            let id = named.or_else(|| (auto && n > 1).then(|| resolve_auto(backend.as_ref(), REMOVE_HIGHLIGHT_ORDER, "")).filter(|s| !s.is_empty()));
            match id.and_then(|id| template::find(&id).ok()).and_then(|t| t.meta.highlight.clone().map(|c| (t.meta.id.clone(), c))) {
                Some((id, colour)) => {
                    let g = Gray8::new(w, h, tight.iter().map(|c| (c * 255.0 + 0.5) as u8).collect()).map_err(gen_err)?;
                    let g = feather_outward(&g, radius)?;
                    let g = if (rw, rh) == (w, h) { g } else { resize_gray8(&g, rw, rh)? };
                    Some((id, highlight_covered(&image, &g, &colour), g))
                }
                None => None,
            }
        }
        _ => None,
    };
    // Every variation is the adapter's when the adapter template was named; else the first.
    let adapter_named = adapter.as_ref().is_some_and(|(id, _, _)| *id == template_id);
    let image_plain = image.clone();
    // A route that regenerates the whole picture (an edit by instruction; every Qwen-Image-2.1
    // template, whose "fill" is an edit with the mask in view) hands it back a few pixels off:
    // the result is realigned to the picture outside the mask (`generate_realign`).
    let qwen21 = template::find(&template_id).is_ok_and(|t| t.meta.family == "qwen-image-2.1");
    let realign = outpaint.is_none() && (command == EDIT || qwen21);
    // The smoothed hole helps 2511 (it copies what its reference shows, so a smooth
    // continuation gives it nothing to keep); Qwen-Image-2.1 hands the smoothing back as the
    // blur it is, and removes from the real picture as well as 2511 does from the smoothed one
    // (benchmarks.md 5i, 2026-10-26). So 2.1 sees the picture as it is.
    let image = if prefill && outpaint.is_none() && !adapter_named && !qwen21 { smooth_covered(&image, &mask) } else { image };
    // The prompt the model gets: as typed, or rewritten by the vision-language model with the
    // picture in view (an expand keeps its own wording).
    let typed = prompt.clone();
    let prompt = if let Some(r) = rewritten {
        r
    } else if enhance && outpaint.is_none() {
        ctx.progress(0.02, "Improving the prompt");
        // A fill from a reference shows the rewriter the picture (its hole grey) and the first
        // reference side by side, so it can name what to take as the reference shows it.
        let fill_ref = command == FILL && !refs.images.is_empty();
        let kind = if command == EDIT {
            EnhanceKind::Edit
        } else if fill_ref {
            EnhanceKind::FillRef
        } else {
            EnhanceKind::Fill
        };
        let seen = if fill_ref { refs.images.first().and_then(|r| side_by_side(&image, r).ok()) } else { None };
        let seen = seen.as_ref().unwrap_or(&image);
        // A rewriter that fails (no encoder on the server, nothing usable said) leaves the
        // prompt as typed; the run itself reports a server that is down.
        match enhance_prompt(thinker.as_ref(), kind, &prompt, Some(seen), place.as_deref(), seed, &JobProgress::span(ctx, 0.02, 0.05)) {
            Ok(Some(p)) => p,
            Ok(None) | Err(_) => prompt,
        }
    } else {
        prompt
    };
    ctx.check()?;
    release_thinker(&server, purge, &thinker, &backend);
    // An expand template crops the model's reference to the picture: its rectangle inside the
    // request, in the request's (possibly resampled) pixels.
    let mut params = BTreeMap::new();
    params.insert("sampler".to_string(), Value::String(tuning.sampler.clone().unwrap_or_else(|| DEFAULT_SAMPLER.to_string())));
    params.insert("scheduler".to_string(), Value::String(tuning.scheduler.clone().unwrap_or_else(|| DEFAULT_SCHEDULER.to_string())));
    if let Some(l) = tuning.loras_param() {
        params.insert("loras".to_string(), l);
    }
    if let Some(d) = tuning.denoise {
        params.insert("denoise".to_string(), json!(d));
    }
    if let Some(o) = &outpaint {
        let (sx, sy) = (f64::from(rw) / f64::from(w.max(1)), f64::from(rh) / f64::from(h.max(1)));
        let p = o.picture;
        let x0 = (f64::from(p.x0 - rect.x0) * sx).floor().clamp(0.0, f64::from(rw.saturating_sub(16)));
        let y0 = (f64::from(p.y0 - rect.y0) * sy).floor().clamp(0.0, f64::from(rh.saturating_sub(16)));
        let x1 = (f64::from(p.x1 - rect.x0) * sx).ceil().clamp(x0 + 16.0, f64::from(rw));
        let y1 = (f64::from(p.y1 - rect.y0) * sy).ceil().clamp(y0 + 16.0, f64::from(rh));
        params.insert("ref_x".to_string(), json!(x0 as u32));
        params.insert("ref_y".to_string(), json!(y0 as u32));
        params.insert("ref_w".to_string(), json!((x1 - x0) as u32));
        params.insert("ref_h".to_string(), json!((y1 - y0) as u32));
        // The canvas size, for templates that sample a latent of their own size (Qwen-Image-2.1
        // takes its target from the reference otherwise).
        params.insert("width".to_string(), json!(rw));
        params.insert("height".to_string(), json!(rh));
    }
    // An edit template has no noise mask: the picture goes alone and the selection only masks
    // the result layer.
    let wants_mask = |id: &str| template::find(id).map(|t| t.meta.needs_mask).unwrap_or(true);
    let base_wants_mask = wants_mask(&template_id);
    let mut req = Request {
        template: template_id.clone(),
        prompt: prompt.clone(),
        negative,
        seed,
        steps,
        guidance,
        image: Some(image.clone()),
        mask: base_wants_mask.then_some(mask.clone()),
        models,
        size: None,
        params,
        references: refs.images,
    };
    // One backend run per variation, consecutive seeds, each its own masked layer; the first
    // is visible, the alternates sit hidden above it (`generate.variation` switches).
    let mut made: Made = Vec::with_capacity(n as usize);
    let mut mismatches: Vec<f64> = Vec::with_capacity(n as usize);
    let mut judged: Vec<Rgba8> = Vec::with_capacity(n as usize);
    for i in 0..n {
        req.seed = seed.wrapping_add(u64::from(i)) % MAX_SEED;
        let (lo, hi) = (0.05 + 0.9 * i as f32 / n as f32, 0.05 + 0.9 * (i + 1) as f32 / n as f32);
        if n > 1 {
            ctx.progress(lo, &format!("Variation {}/{n}", i + 1));
        }
        // Which variant this run is: the adapter's (its template, its tight red hole, its
        // trigger sentence) or the plain fill's.
        let this_adapter = adapter.as_ref().filter(|_| adapter_named || i == 0);
        let (run_template, run_prompt, run_mask): (String, String, &Gray8) = match this_adapter {
            Some((id, img, m)) => {
                req.image = Some(img.clone());
                (id.clone(), HIGHLIGHT_REMOVE_PROMPT.to_string(), m)
            }
            None => {
                req.image = Some(image.clone());
                (template_id.clone(), prompt.clone(), &mask)
            }
        };
        req.template = run_template.clone();
        req.prompt = run_prompt.clone();
        req.mask = wants_mask(&run_template).then(|| run_mask.clone());
        let run_enhanced = if run_prompt == typed { String::new() } else { run_prompt.clone() };
        let resp = run_switching(backend.as_ref(), &server, &req, purge, &JobProgress::span(ctx, lo, hi)).map_err(gen_err)?;
        ctx.check()?;
        let out = resp.images.into_iter().next().ok_or_else(|| EngineError::Other("the backend returned no image".into()))?;
        let out = if realign && (out.width, out.height) == (image_plain.width, image_plain.height) {
            let d = crate::generate_realign::drift(&out, &image_plain, Some(run_mask), crate::generate_realign::max_shift(out.width, out.height));
            crate::generate_realign::shifted(&out, &image_plain, d)
        } else {
            out
        };
        // `shadow`: on a whole-picture route a removal also takes what changed with the object
        // beside the hole; that band joins the hole for the tone match (the membrane carries
        // the result across it, so its edge meets the picture) and the layer mask.
        // The same on a swap or an edit with a selection: what the model drew past the selection,
        // connected to it (a new coat longer than the old jacket), joins the layer mask, which
        // otherwise cut it off in a dark band at the selection's edge (measured 2026-10-27,
        // `benchmarks.md` swaps); held to a smaller share of the selection than a shadow.
        let whole_picture = qwen21 || command == EDIT;
        let (band_share, band_change) = if removal { (SHADOW_MAX_SHARE, SHADOW_CHANGE) } else { (EDIT_BAND_MAX_SHARE, EDIT_BAND_CHANGE) };
        let band = (whole_picture && tuning.shadow && outpaint.is_none() && this_adapter.is_none() && layer_coverage.iter().any(|c| *c < 1.0))
            .then(|| change_band(&out, &image_plain, run_mask, band_share, band_change))
            .flatten();
        let widened = band.as_ref().and_then(|b| Gray8::new(b.width, b.height, b.data.iter().zip(&run_mask.data).map(|(a, m)| (*a).max(*m)).collect()).ok());
        let run_mask = widened.as_ref().unwrap_or(run_mask);
        let band_coverage = match &band {
            Some(b) => {
                let b = if (b.width, b.height) == (w, h) { b.clone() } else { resize_gray8(b, w, h)? };
                Some(layer_coverage.iter().zip(&b.data).map(|(c, v)| c.max(f32::from(*v) / 255.0)).collect::<Vec<f32>>())
            }
            None => None,
        };
        // A removal's result is tone-matched to the picture at the hole's edge and judged.
        let (out, mismatch) = if removal && (out.width, out.height) == (image_plain.width, image_plain.height) {
            // The adapter leaving its red in the hole is a loss before any judging: the judge
            // once chose a red-tinted glass as the natural one (measured 2026-10-22).
            let painted = this_adapter.is_some() && paint_left(&out, run_mask);
            let matched = tone_matched(&out, &image_plain, run_mask);
            judged.push(crop_rgba8(&matched, judged_rect(run_mask, TONE_MATCH_INSIDE)));
            (matched, if painted { f64::INFINITY } else { edge_mismatch(&out, &image_plain, run_mask) })
        } else {
            (out, 0.0)
        };
        mismatches.push(mismatch);
        let out = if (out.width, out.height) == (w, h) { out } else { resize_rgba8(&out, w, h)? };
        // The result becomes a layer in the document's own format, masked to the coverage.
        let layer_name = if n > 1 { format!("{name} ({}/{n})", i + 1) } else { name.clone() };
        let mut layer = Layer::raster(layer_name, doc.pixel_format());
        write_rgba8(&mut layer, &out, rect)?;
        let mut mask_surface = Surface::with_default(PixelFormat::GRAY8, &[0.0]);
        mask_surface.write_region(rect, band_coverage.as_deref().unwrap_or(&layer_coverage));
        layer.mask = Some(LayerMask { surface: mask_surface, enabled: true, linked: true, density: 1.0, feather: 0.0 });
        set_generative_info(
            &mut layer,
            &GenerativeInfo {
                command: command.to_string(),
                prompt: typed.clone(),
                negative: req.negative.clone(),
                template: run_template.clone(),
                seed: req.seed,
                steps: req.steps,
                guidance: req.guidance,
                edge: match edge {
                    Edge::Soft => "soft".into(),
                    Edge::Hard => "hard".into(),
                },
                rect: [rect.x0, rect.y0, rect.width() as i32, rect.height() as i32],
                name: name.clone(),
                enhanced: run_enhanced,
                references: refs.specs.clone(),
                grow,
                erase,
                ..GenerativeInfo::default()
            },
        );
        layer.visible = i == 0;
        let nid = doc.insert_above(*active, layer);
        *active = Some(nid);
        made.push((nid, resp.seed, resp.run_id, resp.elapsed_ms, resp.timings));
    }
    ctx.progress(0.96, "Placing");
    // A removal shows the variation that looks most natural to the vision-language model,
    // each judged against the best so far ([`judge_pair`]); where it cannot say, the one whose
    // fill matches its edge best. The others stay as hidden variations, in their order.
    if removal && made.len() > 1 {
        let mut best = 0usize;
        for i in 1..made.len() {
            ctx.progress(0.96, "Judging");
            let (m_best, m_i) = (mismatches.get(best).copied().unwrap_or(0.0), mismatches.get(i).copied().unwrap_or(0.0));
            // A result the adapter painted is out; then, with [`PRESENCE_CHECK`], each result
            // is asked on its own whether the named object is still there, and one that says
            // no against one that says yes wins outright; then the two side by side.
            let verdict = if m_best.is_infinite() || m_i.is_infinite() {
                Some(m_best <= m_i)
            } else {
                let presence = match (object.as_deref().filter(|_| PRESENCE_CHECK), judged.get(best), judged.get(i)) {
                    (Some(name), Some(a), Some(b)) => {
                        let span = JobProgress::span(ctx, 0.96, 0.97);
                        match (still_there(thinker.as_ref(), a, name, seed.wrapping_add(200 + i as u64), &span), still_there(thinker.as_ref(), b, name, seed.wrapping_add(300 + i as u64), &span)) {
                            (Some(false), Some(true)) => Some(true),
                            (Some(true), Some(false)) => Some(false),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                presence.or_else(|| match (judged.get(best), judged.get(i)) {
                    (Some(a), Some(b)) => judge_pair(thinker.as_ref(), a, b, object.as_deref(), seed.wrapping_add(100 + i as u64), &JobProgress::span(ctx, 0.96, 0.97)),
                    _ => None,
                })
            };
            let left_wins = verdict.unwrap_or(m_best <= m_i);
            if !left_wins {
                best = i;
            }
        }
        release_thinker(&server, purge, &thinker, &backend);
        if best > 0 {
            let chosen = made.remove(best);
            made.insert(0, chosen);
            for (i, m) in made.iter().enumerate() {
                if let Some(l) = doc.layer_mut(m.0) {
                    l.visible = i == 0;
                }
            }
        }
    }
    // The visible result is the active layer.
    *active = made.first().map(|m| m.0);
    Ok((made, w, h, (rw, rh), template_id))
}

/// How far a result's fill is from its own edge: the distance between the mean colour of the
/// result inside the hole (mask at least [`TONE_MATCH_INSIDE`]) and the mean colour of the
/// picture in the band around it (mask above zero and below that), per channel, 0..1, plus
/// [`EDGE_TEXTURE_WEIGHT`] times the difference in texture (the mean absolute luminance
/// step between neighbours) between the two: a seed that drew a second wire on a plain sky
/// has the sky's colour but not its smoothness, and a plain fill on a checkered cloth has
/// the colour but not the texture. A removal wants its hole to look like its edge, so of
/// several seeds the lowest wins.
fn edge_mismatch(result: &Rgba8, picture: &Rgba8, mask: &Gray8) -> f64 {
    let (w, h) = (mask.width as usize, mask.height as usize);
    let n = w * h;
    if result.data.len() != n * 4 || picture.data.len() != n * 4 || mask.data.len() != n || w < 2 || h < 2 {
        return 0.0;
    }
    let luma = |d: &[u8], i: usize| (0.299 * f64::from(d[i * 4]) + 0.587 * f64::from(d[i * 4 + 1]) + 0.114 * f64::from(d[i * 4 + 2])) / 255.0;
    let (mut hole, mut band) = ([0.0f64; 3], [0.0f64; 3]);
    let (mut hn, mut bn) = (0usize, 0usize);
    let (mut hole_tex, mut band_tex) = (0.0f64, 0.0f64);
    let (mut htn, mut btn) = (0usize, 0usize);
    for (i, m) in mask.data.iter().enumerate() {
        let (x, y) = (i % w, i / w);
        if *m >= TONE_MATCH_INSIDE {
            for (acc, v) in hole.iter_mut().zip(&result.data[i * 4..i * 4 + 3]) {
                *acc += f64::from(*v);
            }
            hn += 1;
            if x + 1 < w && y + 1 < h {
                hole_tex += (luma(&result.data, i) - luma(&result.data, i + 1)).abs() + (luma(&result.data, i) - luma(&result.data, i + w)).abs();
                htn += 1;
            }
        } else if *m > 0 {
            for (acc, v) in band.iter_mut().zip(&picture.data[i * 4..i * 4 + 3]) {
                *acc += f64::from(*v);
            }
            bn += 1;
            if x + 1 < w && y + 1 < h {
                band_tex += (luma(&picture.data, i) - luma(&picture.data, i + 1)).abs() + (luma(&picture.data, i) - luma(&picture.data, i + w)).abs();
                btn += 1;
            }
        }
    }
    if hn == 0 || bn == 0 || htn == 0 || btn == 0 {
        return 0.0;
    }
    let colour = (0..3).map(|c| (hole[c] / hn as f64 - band[c] / bn as f64) / 255.0).map(|d| d * d).sum::<f64>().sqrt();
    let texture = (hole_tex / htn as f64 - band_tex / btn as f64).abs();
    colour + EDGE_TEXTURE_WEIGHT * texture
}

/// How much a texture difference weighs against a colour difference in [`edge_mismatch`]
/// (a luminance step is a few hundredths on a smooth surface, a few tenths on a pattern).
const EDGE_TEXTURE_WEIGHT: f64 = 2.0;

/// The JSON a fill or expand job answers with.
fn made_json(made: &Made, w: u32, h: u32, (rw, rh): (u32, u32), template_id: &str) -> Value {
    let first = made.first();
    json!({
        "layer": first.map(|m| m.0.0), "seed": first.map(|m| m.1), "runId": first.map(|m| m.2.clone()), "template": template_id,
        "layers": made.iter().map(|m| m.0.0).collect::<Vec<_>>(), "seeds": made.iter().map(|m| m.1).collect::<Vec<_>>(),
        "width": w, "height": h, "requestWidth": rw, "requestHeight": rh,
        "ms": made.iter().map(|m| m.3).sum::<u64>(),
        "timings": made.iter().map(|m| serde_json::to_value(m.4).unwrap_or(Value::Null)).collect::<Vec<_>>(),
    })
}

/// What empty canvas is painted with when there is no picture to continue: mid-grey.
const PREFILL: [u8; 4] = [128, 128, 128, 255];

/// Widen a coverage mask (one value per pixel, row-major) by `r` pixels: the maximum over a
/// square window, in two separable passes, so a selection's edge moves outward by `r`.
pub(crate) fn dilate(coverage: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 || w == 0 || h == 0 || coverage.len() != w * h {
        return coverage.to_vec();
    }
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let mut out = vec![0.0f32; src.len()];
        let (len, lines) = if horizontal { (w, h) } else { (h, w) };
        let at = |line: usize, i: usize| if horizontal { line * w + i } else { i * w + line };
        for line in 0..lines {
            for i in 0..len {
                let lo = i.saturating_sub(r);
                let hi = (i + r).min(len - 1);
                let mut m = 0.0f32;
                for j in lo..=hi {
                    m = m.max(src.get(at(line, j)).copied().unwrap_or(0.0));
                }
                if let Some(o) = out.get_mut(at(line, i)) {
                    *o = m;
                }
            }
        }
        out
    };
    pass(&pass(coverage, true), false)
}

/// `result` with its tone matched to `original`'s at the hole's edge: the hole is where
/// `mask` is at least [`TONE_MATCH_INSIDE`]; just outside it the model re-rendered the band
/// and the difference between the two pictures there is its tone shift, which a membrane
/// carries smoothly across the hole and takes out (Pérez's seamless cloning with the result
/// as the source). Outside the hole the original comes back (the layer mask fades there
/// anyway). Both pictures the same size; otherwise `result` as it was.
fn tone_matched(result: &Rgba8, original: &Rgba8, mask: &Gray8) -> Rgba8 {
    let (w, h) = (result.width as usize, result.height as usize);
    if (original.width, original.height) != (result.width, result.height) || (mask.width, mask.height) != (result.width, result.height) || w < 3 || h < 3 {
        return result.clone();
    }
    // A one-pixel unmasked margin keeps the problem Dirichlet (see `seamless_clone`).
    let hole: Vec<bool> = mask.data.iter().enumerate().map(|(i, m)| *m >= TONE_MATCH_INSIDE && i % w > 0 && i % w < w - 1 && i / w > 0 && i / w < h - 1).collect();
    if !hole.iter().any(|h| *h) {
        return result.clone();
    }
    let rgb = |img: &Rgba8| -> Vec<f32> { img.data.as_chunks::<4>().0.iter().flat_map(|p| [f32::from(p[0]) / 255.0, f32::from(p[1]) / 255.0, f32::from(p[2]) / 255.0]).collect() };
    let matched = photocraft_algo::poisson::seamless_clone(w, h, 3, &rgb(result), &rgb(original), &hole);
    let mut data = Vec::with_capacity(w * h * 4);
    for (px, src) in result.data.as_chunks::<4>().0.iter().zip(matched.as_chunks::<3>().0) {
        data.extend(src.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8));
        data.push(px[3]);
    }
    Rgba8::new(result.width, result.height, data).unwrap_or_else(|_| result.clone())
}

/// Mask values from here up count as the hole for [`tone_matched`] (the request's mask is
/// 255 inside the selection and feathered outward).
const TONE_MATCH_INSIDE: u8 = 200;

/// [`change_band`]: a pixel changed when the two pictures, each box-blurred twice at this
/// radius (texture the model redraws averages out, a shadow does not), differ by more than
/// [`SHADOW_CHANGE`] in some channel (0..1).
const SHADOW_BLUR: usize = 3;
pub(crate) const SHADOW_CHANGE: f32 = 0.08;
/// The change an edit's band counts: a swapped garment against what it replaced (a yellow coat
/// over black jeans), not the light re-rendering a whole-picture route gives the rest (the 2.1
/// edit's jeans, which ran past the reach and refused the band at the shadow's 0.08).
pub(crate) const EDIT_BAND_CHANGE: f32 = 0.25;
/// How far from the hole the band may reach, in the hole's longer side (at least 16 px).
const SHADOW_REACH: f64 = 0.75;
/// The band is taken only when it is at least this share of the hole (smaller is the model's
/// noise) and at most that (larger: the model shifted the whole picture's tone).
const SHADOW_MIN_SHARE: f64 = 0.03;
pub(crate) const SHADOW_MAX_SHARE: f64 = 2.0;
/// The most an edit's or a fill's band may add, in the selection's area: a garment longer than
/// the one it replaces, not a second subject.
pub(crate) const EDIT_BAND_MAX_SHARE: f64 = 0.75;

/// What changed with the object, outside the hole (`mask` at [`TONE_MATCH_INSIDE`] or more):
/// the pixels where `result` differs from `original` ([`SHADOW_CHANGE`] after a blur),
/// connected to the hole's feathered edge, within [`SHADOW_REACH`] of it, as a mask (255 on
/// the band, widened by 2 px), or `None` when nothing or too much changed. A route that
/// regenerates the whole request removes a cast shadow or a reflection with its object; the
/// hole's mask kept it from showing (AusBoss's differential composite, catalogue §2.2). A band
/// larger than `max_share` of the hole is refused; `threshold` is the change that counts
/// ([`SHADOW_CHANGE`] for a shadow, [`EDIT_BAND_CHANGE`] for an edit).
pub(crate) fn change_band(result: &Rgba8, original: &Rgba8, mask: &Gray8, max_share: f64, threshold: f32) -> Option<Gray8> {
    let (w, h) = (result.width as usize, result.height as usize);
    if (original.width, original.height) != (result.width, result.height) || (mask.width, mask.height) != (result.width, result.height) || w < 3 || h < 3 {
        return None;
    }
    let blurred = |img: &Rgba8, c: usize| -> Vec<f32> {
        let mut v: Vec<f32> = img.data.as_chunks::<4>().0.iter().map(|p| f32::from(p.get(c).copied().unwrap_or(0)) / 255.0).collect();
        for _ in 0..2 {
            v = box_blur(&v, w, h, SHADOW_BLUR, true);
            v = box_blur(&v, w, h, SHADOW_BLUR, false);
        }
        v
    };
    let mut diff = vec![0.0f32; w * h];
    for c in 0..3 {
        for ((d, a), b) in diff.iter_mut().zip(blurred(result, c)).zip(blurred(original, c)) {
            *d = d.max((a - b).abs());
        }
    }
    let hole: Vec<bool> = mask.data.iter().map(|m| *m >= TONE_MATCH_INSIDE).collect();
    let hole_px = hole.iter().filter(|h| **h).count();
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
    for (i, _) in hole.iter().enumerate().filter(|(_, h)| **h) {
        let (x, y) = (i % w, i / w);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
    }
    if hole_px == 0 {
        return None;
    }
    let reach = ((x1 - x0).max(y1 - y0) as f64 * SHADOW_REACH).max(16.0) as usize;
    let (lx0, ly0, lx1, ly1) = (x0.saturating_sub(reach), y0.saturating_sub(reach), (x1 + reach).min(w), (y1 + reach).min(h));
    let changed: Vec<f32> = diff
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let (x, y) = (i % w, i / w);
            let inside = hole.get(i).copied().unwrap_or(false);
            f32::from(u8::from(!inside && *d > threshold && x >= lx0 && x < lx1 && y >= ly0 && y < ly1))
        })
        .collect();
    // Specks out (an opening of one pixel), then the part connected to the hole's edge.
    let inverted: Vec<f32> = changed.iter().map(|c| 1.0 - c).collect();
    let eroded: Vec<f32> = dilate(&inverted, w, h, 1).iter().map(|c| 1.0 - c).collect();
    let opened = dilate(&eroded, w, h, 1);
    let open: Vec<bool> = opened.iter().zip(&changed).map(|(o, c)| *o >= 0.5 && *c >= 0.5).collect();
    let mut band = vec![false; w * h];
    let mut stack: Vec<usize> = (0..w * h).filter(|&i| open.get(i).copied().unwrap_or(false) && mask.data.get(i).copied().unwrap_or(0) > 0).collect();
    for &i in &stack {
        if let Some(b) = band.get_mut(i) {
            *b = true;
        }
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        let next = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
        for j in next.into_iter().flatten() {
            if open.get(j).copied().unwrap_or(false) && !band.get(j).copied().unwrap_or(true) {
                if let Some(b) = band.get_mut(j) {
                    *b = true;
                }
                stack.push(j);
            }
        }
    }
    // A band that runs into the request's edge (or the reach's) goes on outside it, where the
    // model was not asked: taking it would cut the shadow off in a straight line (the dog's,
    // measured 2026-10-27), so the shadow stays whole instead.
    let on_edge = |i: usize| {
        let (x, y) = (i % w, i / w);
        x == lx0 || y == ly0 || x + 1 == lx1 || y + 1 == ly1
    };
    if band.iter().enumerate().any(|(i, b)| *b && on_edge(i) && mask.data.get(i).copied().unwrap_or(0) == 0) {
        return None;
    }
    // Pixels the hole's own feather covers are the model's re-rendered rim, not a shadow.
    let taken = band.iter().zip(&mask.data).filter(|(b, m)| **b && **m == 0).count();
    if (taken as f64) < SHADOW_MIN_SHARE * hole_px as f64 || (taken as f64) > max_share * hole_px as f64 {
        return None;
    }
    let grown = dilate(&band.iter().map(|b| f32::from(u8::from(*b))).collect::<Vec<f32>>(), w, h, 2);
    Gray8::new(result.width, result.height, grown.iter().map(|g| if *g >= 0.5 { 255 } else { 0 }).collect()).ok()
}

/// `img` with the hole (`mask` at [`SMOOTH_INSIDE`] or more) smoothed over by a membrane from
/// its edge: the colours continue in, with no structure. An edit model copies what its
/// reference shows, so a hole with the object in it is redrawn and a flat grey one painted
/// back as a panel (both measured); a smooth continuation is neither. Unchanged when the
/// sizes differ or the hole is empty.
fn smooth_covered(img: &Rgba8, mask: &Gray8) -> Rgba8 {
    let (w, h) = (img.width as usize, img.height as usize);
    if (mask.width, mask.height) != (img.width, img.height) || w < 3 || h < 3 {
        return img.clone();
    }
    let hole: Vec<bool> = mask.data.iter().enumerate().map(|(i, m)| *m >= SMOOTH_INSIDE && i % w > 0 && i % w < w - 1 && i / w > 0 && i / w < h - 1).collect();
    if !hole.iter().any(|h| *h) {
        return img.clone();
    }
    let rgb: Vec<f32> = img.data.as_chunks::<4>().0.iter().flat_map(|p| [f32::from(p[0]) / 255.0, f32::from(p[1]) / 255.0, f32::from(p[2]) / 255.0]).collect();
    let filled = photocraft_algo::poisson::membrane_fill(w, h, 3, &rgb, &hole);
    let mut data = img.data.clone();
    for (i, in_hole) in hole.iter().enumerate() {
        if *in_hole && let (Some(d), Some(s)) = (data.get_mut(i * 4..i * 4 + 3), filled.get(i * 3..i * 3 + 3)) {
            for (dv, sv) in d.iter_mut().zip(s) {
                *dv = (sv.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
    Rgba8::new(img.width, img.height, data).unwrap_or_else(|_| img.clone())
}

/// Mask values from here up are the hole [`smooth_covered`] fills: the selection itself
/// (255), not the feathered band outside it.
const SMOOTH_INSIDE: u8 = 250;

/// What a removal adapter trained on a painted highlight is told (the Object Remover v2
/// card's sample prompts, every one of them this sentence).
const HIGHLIGHT_REMOVE_PROMPT: &str = "Remove the red highlighted object from the scene.";
/// The removal adapters `auto` tries, first installed one first, for a removal's first
/// variation (the plain fill takes the rest, and the judge picks).
const REMOVE_HIGHLIGHT_ORDER: &[&str] = &["qwen-edit-2511/remove-lightning-8"];
/// How far past the selection the adapter's painted hole reaches: a few pixels, as the
/// highlights it was trained on, not the removal's 15 % growth (a fat red hole came back red
/// on a third of the stock set; the selection is the object's silhouette already).
const HIGHLIGHT_GROW: usize = 6;

/// Whether a removal adapter left its paint in the hole: more than [`PAINT_LEFT_SHARE`] of
/// the pixels under `mask` (at [`SMOOTH_INSIDE`] or more) are saturated red in `result`.
fn paint_left(result: &Rgba8, mask: &Gray8) -> bool {
    if (mask.width, mask.height) != (result.width, result.height) {
        return false;
    }
    let (mut hole, mut red) = (0usize, 0usize);
    for (i, m) in mask.data.iter().enumerate() {
        if *m < SMOOTH_INSIDE {
            continue;
        }
        hole += 1;
        if let Some(px) = result.data.get(i * 4..i * 4 + 3)
            && px[0] >= 150
            && i32::from(px[0]) - i32::from(px[1].max(px[2])) >= 70
        {
            red += 1;
        }
    }
    hole > 0 && red as f64 / hole as f64 > PAINT_LEFT_SHARE
}

/// The share of the hole that may be saturated red before the adapter's result is out.
const PAINT_LEFT_SHARE: f64 = 0.05;

/// `img` with the hole (`mask` at [`SMOOTH_INSIDE`] or more) painted solid in `colour`
/// ("red", "green", "blue", or `#rrggbb`; anything else red): what a removal adapter trained
/// on a highlight expects to see.
fn highlight_covered(img: &Rgba8, mask: &Gray8, colour: &str) -> Rgba8 {
    let rgb = match colour.trim().to_ascii_lowercase().as_str() {
        "green" => [0, 255, 0],
        "blue" => [0, 0, 255],
        hex if hex.len() == 7 && hex.starts_with('#') => {
            let v = |i: usize| u8::from_str_radix(hex.get(i..i + 2).unwrap_or("ff"), 16).unwrap_or(255);
            [v(1), v(3), v(5)]
        }
        _ => [255, 0, 0],
    };
    if (mask.width, mask.height) != (img.width, img.height) {
        return img.clone();
    }
    let mut data = img.data.clone();
    for (i, m) in mask.data.iter().enumerate() {
        if *m >= SMOOTH_INSIDE && let Some(px) = data.get_mut(i * 4..i * 4 + 4) {
            px.copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
    }
    Rgba8::new(img.width, img.height, data).unwrap_or_else(|_| img.clone())
}

/// Paint the pixels `coverage` selects (half or more) mid-grey: the request image shows a hole
/// where the model is to paint, not the object it is to replace.
fn erase_covered(img: &mut Rgba8, coverage: &[f32]) {
    for (px, c) in img.data.as_chunks_mut::<4>().0.iter_mut().zip(coverage) {
        if *c >= 0.5 {
            px.copy_from_slice(&PREFILL);
        }
    }
}

/// Whether `left` looks more natural than `right` to the vision-language model
/// ([`ENHANCE_JUDGE_SYSTEM`]): `Some(true)` left, `Some(false)` right, `None` when it could
/// not be asked or did not answer (the caller falls back to [`edge_mismatch`]).
fn judge_pair(backend: &dyn GenerativeBackend, left: &Rgba8, right: &Rgba8, object: Option<&str>, seed: u64, progress: &JobProgress) -> Option<bool> {
    let seen = side_by_side(&judge_view(left), &judge_view(right)).ok()?;
    let (rw, rh) = request_size_within(seen.width, seen.height, JUDGE_PAIR_MAX_PIXELS);
    let seen = if (rw, rh) == (seen.width, seen.height) { seen } else { resize_rgba8(&seen, rw, rh).ok()? };
    let answer = enhance_prompt(backend, EnhanceKind::Judge, &judge_question(object), Some(&seen), None, seed, progress).ok()??.to_lowercase();
    match (answer.find("left"), answer.find("right")) {
        (Some(l), Some(r)) => Some(l < r),
        (Some(_), None) => Some(true),
        (None, Some(_)) => Some(false),
        (None, None) => None,
    }
}

/// `img` cut to `rect` (in its own pixels, clipped to it); `img` itself when the cut is empty.
fn crop_rgba8(img: &Rgba8, rect: Rect) -> Rgba8 {
    let r = rect.intersect(&Rect::new(0, 0, img.width as i32, img.height as i32));
    if r.is_empty() {
        return img.clone();
    }
    let (w, h) = (r.width() as usize, r.height() as usize);
    let stride = img.width as usize * 4;
    let mut data = Vec::with_capacity(w * h * 4);
    for y in r.y0..r.y1 {
        let start = y as usize * stride + r.x0 as usize * 4;
        data.extend_from_slice(img.data.get(start..start + w * 4).unwrap_or(&[]));
    }
    Rgba8::new(r.width(), r.height(), data).unwrap_or_else(|_| img.clone())
}

/// Whether each of a removal's results is asked on its own if the object is still there
/// ([`still_there`]) before the two are judged side by side. Off: the two extra loads of the
/// vision-language model per stroke, on top of the rewriter's and the judge's, had the GPU
/// driver reset twice in two runs (nvlddmkm event 153, the server killed mid-answer,
/// 2026-10-23), where the runs without it ran for hours; the enlarged judge view
/// ([`judge_view`]) is what let the judge see thin transparent frames, and stays.
const PRESENCE_CHECK: bool = false;

/// Whether the vision-language model still sees `object` in `result`'s judged crop
/// ([`ENHANCE_PRESENCE_SYSTEM`]): `Some(true)` yes, `Some(false)` no, `None` when it could
/// not be asked or did not answer.
fn still_there(backend: &dyn GenerativeBackend, result: &Rgba8, object: &str, seed: u64, progress: &JobProgress) -> Option<bool> {
    let seen = judge_view(result);
    let question = format!("Is there a {object}, or any part of one, in this photo?");
    let answer = enhance_prompt(backend, EnhanceKind::Presence, &question, Some(&seen), None, seed, progress).ok()??.to_lowercase();
    let first = answer.trim().split(|c: char| !c.is_alphanumeric()).next().unwrap_or("");
    match first {
        "yes" => Some(true),
        "no" => Some(false),
        _ => match (answer.find("yes"), answer.find("no")) {
            (Some(y), Some(n)) => Some(y < n),
            (Some(_), None) => Some(true),
            (None, Some(_)) => Some(false),
            (None, None) => None,
        },
    }
}

/// A judged crop enlarged until its shorter side is at least [`JUDGE_VIEW_MIN_SIDE`], its
/// longer side at most [`JUDGE_VIEW_MAX_SIDE`] either way: thin things in a small crop were
/// lost on the judge, and a pair of crops at 1024 px each (a 1 MP picture for the
/// vision-language model with the 20 GB fill model resident) is what ran when the GPU driver
/// reset four times in a night (2026-10-23); the pair stays under half a megapixel now.
fn judge_view(crop: &Rgba8) -> Rgba8 {
    let (w, h) = (crop.width, crop.height);
    let (shorter, longer) = (w.min(h).max(1), w.max(h).max(1));
    let up = f64::from(JUDGE_VIEW_MIN_SIDE) / f64::from(shorter);
    let cap = f64::from(JUDGE_VIEW_MAX_SIDE) / f64::from(longer);
    let scale = if longer > JUDGE_VIEW_MAX_SIDE { cap } else { up.min(cap).max(1.0) };
    if (scale - 1.0).abs() < 1e-6 {
        return crop.clone();
    }
    let (nw, nh) = (((f64::from(w) * scale).round() as u32).max(1), ((f64::from(h) * scale).round() as u32).max(1));
    resize_rgba8(crop, nw, nh).unwrap_or_else(|_| crop.clone())
}

const JUDGE_VIEW_MIN_SIDE: u32 = 320;
const JUDGE_VIEW_MAX_SIDE: u32 = 512;
/// The judge's side-by-side picture is sent at most this large (see [`judge_view`]).
const JUDGE_PAIR_MAX_PIXELS: u64 = 512 * 1024;

/// The box of the pixels of `mask` at or above `level`, widened by a quarter of its larger
/// side: what the judge of a removal looks at.
fn judged_rect(mask: &Gray8, level: u8) -> Rect {
    let w = mask.width as usize;
    let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for (i, m) in mask.data.iter().enumerate() {
        if *m >= level && w > 0 {
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
        }
    }
    if x1 <= x0 || y1 <= y0 {
        return Rect::new(0, 0, mask.width as i32, mask.height as i32);
    }
    let m = ((x1 - x0).max(y1 - y0) / 4).max(8);
    Rect::new(x0 - m, y0 - m, x1 + m, y1 + m).intersect(&Rect::new(0, 0, mask.width as i32, mask.height as i32))
}

/// The widest a [`side_by_side`]'s right half is made (a sliver of a reference scaled up to a
/// tall picture's height stays a bounded allocation; the request is shrunk to size after).
const MAX_SIDE_BY_SIDE_WIDTH: u32 = 8192;

/// `left` and `right` side by side, `right` scaled to `left`'s height with its aspect kept:
/// what the rewriter of a fill from a reference and of a removal looks at. (It was once held to
/// `left`'s width, which squeezed a wide close-up beside narrower surroundings: a boat's hull
/// under its 384-px surroundings at two thirds of its width.)
pub(crate) fn side_by_side(left: &Rgba8, right: &Rgba8) -> Result<Rgba8> {
    let h = left.height.max(1);
    let rw = ((f64::from(right.width) * f64::from(h) / f64::from(right.height.max(1))).round() as u32).clamp(1, MAX_SIDE_BY_SIDE_WIDTH);
    let right = if (right.width, right.height) == (rw, h) { right.clone() } else { resize_rgba8(right, rw, h)? };
    let w = left.width + rw;
    let mut data = Vec::with_capacity(w as usize * h as usize * 4);
    let (ls, rs) = (left.width as usize * 4, rw as usize * 4);
    for y in 0..h as usize {
        data.extend_from_slice(left.data.get(y * ls..(y + 1) * ls).unwrap_or(&[]));
        data.extend_from_slice(right.data.get(y * rs..(y + 1) * rs).unwrap_or(&[]));
    }
    Rgba8::new(w, h, data).map_err(gen_err)
}

/// Paint the pixels outside `inner` (in image coordinates) with the nearest pixel of `inner`, so
/// the canvas the VAE encodes continues the picture's edge colours instead of meeting a grey
/// wall. The model never sees this area as content (its reference is the cropped picture, see
/// `Outpaint`, and the noise mask is full over it), but the VAE's latents next to a wall of grey
/// carry its tone into the picture's edge band, which came back as a dark line along the old
/// edge. The replicated streaks are what the early padded-reference variant turned into
/// content; with the reference cropped they are only ever noised away. (Qwen-Image-2.1's
/// in-context expand template, whose reference is the picture placed inside the canvas, never
/// sees this area either; a mid-grey prefill tried with the whole canvas as the reference came
/// back as black canvas, 2026-10-20.)
fn prefill_outside(img: &mut Rgba8, inner: Rect) {
    let (w, h) = (img.width as i32, img.height as i32);
    let inner = inner.intersect(&Rect::new(0, 0, w, h));
    if inner.x0 == 0 && inner.y0 == 0 && inner.x1 == w && inner.y1 == h {
        return;
    }
    let empty = inner.x1 <= inner.x0 || inner.y1 <= inner.y0;
    let stride = img.width as usize * 4;
    for y in 0..h {
        let sy = if empty { 0 } else { y.clamp(inner.y0, inner.y1 - 1) };
        for x in 0..w {
            if x >= inner.x0 && x < inner.x1 && y >= inner.y0 && y < inner.y1 {
                continue;
            }
            let sx = if empty { 0 } else { x.clamp(inner.x0, inner.x1 - 1) };
            let src = sy as usize * stride + sx as usize * 4;
            let p = match img.data.get(src..src + 4) {
                Some(s) if !empty => [s[0], s[1], s[2], 255],
                _ => PREFILL,
            };
            let dst = y as usize * stride + x as usize * 4;
            if let Some(d) = img.data.get_mut(dst..dst + 4) {
                d.copy_from_slice(&p);
            }
        }
    }
}

fn run_fill(s: &mut Session, p: &Value) -> Result<Value> {
    let plan = plan_fill(s, p)?;
    let (backend, server) = backend_for(s, &plan.template.meta.id)?;
    let FillPlan { template, prompt, negative, seed, steps, guidance, name, models, rect, variations, auto, edge, enhance, refs, grow, erase, tuning } = plan;
    let label = name.clone();
    let purge = purge_mode(s);
    // A fill planned on an edit template (`fillAsEdit`) runs as an edit by instruction.
    let as_edit = template.meta.task == Task::Edit;
    let job = FillJob {
        thinker: thinker_for(s, &backend, &server),
        backend,
        server,
        command: if as_edit { EDIT } else { FILL },
        template_id: template.meta.id.clone(),
        auto,
        auto_order: auto_order(FILL, quality(s)),
        prompt,
        negative,
        seed,
        steps,
        guidance,
        name,
        models,
        variations,
        edge,
        enhance,
        refs,
        grow,
        erase,
        rewritten: None,
        removal: false,
        prefill: false,
        tuning,
        object: None,
        purge,
    };
    crate::jobs::edit_job(
        s,
        &label,
        move |doc, active, ctx| {
            let mut job = job;
            let sel = doc.selection.as_ref().ok_or_else(|| EngineError::Other("the selection is gone".into()))?;
            let k = sel.channels().max(1);
            let coverage: Vec<f32> = sel.read_region(rect).chunks_exact(k).map(|c| c.first().copied().unwrap_or(0.0).clamp(0.0, 1.0)).collect();
            // A fill by instruction: a description ("a glass of red wine") becomes "Replace the
            // <thing in the selection> with a glass of red wine." once the rewriter names the
            // thing (the removal's two views); an instruction is left to the edit rewriter.
            // Nothing named: the fill template after all.
            if job.command == EDIT && !template::is_imperative(&job.prompt) {
                let named = if job.enhance { name_selected(doc, &job, rect, &coverage, ctx)? } else { None };
                let with_refs = !job.refs.images.is_empty();
                match (named, with_refs) {
                    (Some(thing), false) => job.rewritten = Some(swap_prompt(&thing, &job.prompt)),
                    (Some(thing), true) => job.rewritten = Some(swap_ref_prompt(Some(&thing), &job.prompt)),
                    // From references nothing else takes them on the worker: the selection's
                    // place stands in for its name.
                    (None, true) => job.rewritten = Some(swap_ref_prompt(None, &job.prompt)),
                    (None, false) => {
                        job.command = FILL;
                        job.template_id = native21_fill_fallback().to_string();
                    }
                }
            }
            fill_region(doc, active, ctx, job, rect, coverage, None)
        },
        move |(made, w, h, request, template_id)| made_json(&made, w, h, request, &template_id),
    )
}

/// What `generate.expand` asks for: how much canvas to add on each side.
struct Pads {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
}

/// The prompt an expand uses when none is given (an instruction, so it takes the imperative
/// wrapper of the template).
const EXPAND_DEFAULT_PROMPT: &str = "extend the scene beyond its original edges, continuing it naturally";
const MAX_EXPAND_SIDE: u32 = 16_384;

fn plan_expand(s: &Session, p: &Value) -> Result<(FillPlan, Pads, (u32, u32))> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let (ow, oh) = (d.doc.size.width, d.doc.size.height);
    let side = |key: &str| -> Result<u32> {
        match opt_num(EXPAND, p, key, 0.0, f64::from(MAX_EXPAND_SIDE))? {
            None => Ok(0),
            Some(x) if x.fract() == 0.0 => Ok(x as u32),
            Some(_) => Err(bad(EXPAND, format!("`{key}` must be a whole number of pixels"))),
        }
    };
    let (width, height) = (side("width")?, side("height")?);
    let pads = if width > 0 || height > 0 {
        // A target size, the old picture placed by `anchor` (Canvas Size's names).
        let (nw, nh) = (if width > 0 { width } else { ow }, if height > 0 { height } else { oh });
        if nw < ow || nh < oh {
            return Err(bad(
                EXPAND,
                format!("`width`/`height` ({nw}×{nh}) must not be smaller than the canvas ({ow}×{oh}); Generative Expand only adds canvas"),
            ));
        }
        let anchor = opt_str(EXPAND, p, "anchor", 20)?.unwrap_or("center");
        let (ax, ay) = match anchor {
            "topLeft" => (0.0, 0.0),
            "top" => (0.5, 0.0),
            "topRight" => (1.0, 0.0),
            "left" => (0.0, 0.5),
            "center" => (0.5, 0.5),
            "right" => (1.0, 0.5),
            "bottomLeft" => (0.0, 1.0),
            "bottom" => (0.5, 1.0),
            "bottomRight" => (1.0, 1.0),
            other => {
                return Err(bad(
                    EXPAND,
                    format!("`anchor` must be one of topLeft, top, topRight, left, center, right, bottomLeft, bottom, bottomRight (got `{other}`)"),
                ));
            }
        };
        let (dw, dh) = (nw - ow, nh - oh);
        let left = (f64::from(dw) * ax).round() as u32;
        let top = (f64::from(dh) * ay).round() as u32;
        Pads { left, top, right: dw - left, bottom: dh - top }
    } else {
        Pads { left: side("left")?, top: side("top")?, right: side("right")?, bottom: side("bottom")? }
    };
    if pads.left + pads.right == 0 && pads.top + pads.bottom == 0 {
        return Err(bad(EXPAND, "nothing to add: give `left`/`top`/`right`/`bottom` in pixels, or a larger `width`/`height` with an `anchor`"));
    }
    let (nw, nh) = (ow.saturating_add(pads.left).saturating_add(pads.right), oh.saturating_add(pads.top).saturating_add(pads.bottom));
    if nw > MAX_EXPAND_SIDE || nh > MAX_EXPAND_SIDE {
        return Err(bad(EXPAND, format!("the expanded canvas would be {nw}×{nh}; at most {MAX_EXPAND_SIDE} px on a side")));
    }
    // The prompt is optional here: without one the model continues the scene.
    let mut q = p.clone();
    if let Some(o) = q.as_object_mut() {
        let blank = o.get("prompt").and_then(Value::as_str).is_none_or(|t| t.trim().is_empty());
        if blank {
            o.insert("prompt".into(), Value::String(EXPAND_DEFAULT_PROMPT.into()));
        }
        if o.get("name").is_none() {
            let shown = p.get("prompt").and_then(Value::as_str).map(str::trim).filter(|t| !t.is_empty());
            o.insert("name".into(), Value::String(shown.map_or("Generative Expand".to_string(), |t| format!("Generative Expand: {}", short(t)))));
        }
    }
    // Expand has its own templates (task `expand`); the fill preference does not apply, so
    // without a `template` param it is `auto`.
    let default_model = s.prefs().integrations.default_edit_model.clone();
    let mut requested = opt_str(EXPAND, &q, "template", 200)?.map(str::to_string).unwrap_or_else(|| AUTO_TEMPLATE.to_string());
    let mut auto = requested == AUTO_TEMPLATE;
    if auto && let Some(o) = q.as_object_mut() {
        o.remove("template");
    }
    if auto && let Some(t) = native21_auto(s, EXPAND, false) {
        requested = t.to_string();
        auto = false;
    }
    let Common { template, prompt, negative, seed, steps, guidance, models, name } =
        plan_common(s, EXPAND, &q, if auto { DEFAULT_EXPAND_TEMPLATE } else { &requested }, Task::Expand, &default_model)?;
    let variations = match opt_num(EXPAND, p, "variations", 1.0, f64::from(MAX_VARIATIONS))? {
        None => 1,
        Some(x) if x.fract() == 0.0 => x as u32,
        Some(_) => return Err(bad(EXPAND, format!("`variations` must be a whole number from 1 to {MAX_VARIATIONS}"))),
    };
    // How much of the picture goes along: the template's say (an in-context model wants all of
    // it, see `Meta::expand_margin`), else a quarter of the longer side.
    let margin = opt_num(EXPAND, p, "margin", 0.0, 1.0)?.unwrap_or_else(|| template.meta.expand_margin.map_or(0.25, |m| f64::from(m.clamp(0.0, 1.0))));
    let edge = parse_edge(EXPAND, p)?;
    // The request: the bounding box of the added canvas (an L or a frame when more than one
    // side grows) plus `margin` of the picture next to it, on the new canvas.
    let canvas = Rect::new(0, 0, nw as i32, nh as i32);
    let old = Rect::new(pads.left as i32, pads.top as i32, (pads.left + ow) as i32, (pads.top + oh) as i32);
    let longer = i64::from(nw.max(nh));
    let m = (margin * longer as f64).round().clamp(0.0, 65_536.0) as i32;
    let (l, t, r, b) = (pads.left > 0, pads.top > 0, pads.right > 0, pads.bottom > 0);
    let added = Rect::new(
        if l || t || b { 0 } else { old.x1.saturating_sub(m) },
        if t || l || r { 0 } else { old.y1.saturating_sub(m) },
        if r || t || b { canvas.x1 } else { old.x0.saturating_add(m) },
        if b || l || r { canvas.y1 } else { old.y0.saturating_add(m) },
    );
    let rect = align_to_grid(added.intersect(&canvas), canvas);
    let pixels = u64::from(rect.width()) * u64::from(rect.height());
    if pixels > MAX_REQUEST_PIXELS {
        return Err(EngineError::Other(format!(
            "the expanded area with its margin is {}×{} pixels; Generative Expand handles up to 16 megapixels at a time: add less canvas or lower `margin`",
            rect.width(),
            rect.height()
        )));
    }
    let name = name.unwrap_or_else(|| "Generative Expand".to_string());
    // An expand keeps its wording: its prompt continues the scene, the rewriter's rules
    // (name the subject, say what stays) are about a picture's content.
    Ok((
        FillPlan {
            template,
            prompt,
            negative,
            seed,
            steps,
            guidance,
            name,
            models,
            rect,
            variations,
            auto,
            edge,
            enhance: false,
            refs: Default::default(),
            grow: 0,
            erase: false,
            tuning: Tuning::default(),
        },
        pads,
        (nw, nh),
    ))
}

fn expand_enabled(s: &Session) -> std::result::Result<(), String> {
    web_unavailable()?;
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// `generate.expand`: grow the canvas and have the model paint the new area, as one undo step.
fn run_expand(s: &mut Session, p: &Value) -> Result<Value> {
    let (plan, pads, (nw, nh)) = plan_expand(s, p)?;
    let (backend, server) = backend_for(s, &plan.template.meta.id)?;
    let FillPlan { template, prompt, negative, seed, steps, guidance, name, models, rect, variations, auto, edge, enhance, refs, grow, erase, tuning } = plan;
    let label = name.clone();
    let purge = purge_mode(s);
    let job = FillJob {
        thinker: thinker_for(s, &backend, &server),
        backend,
        server,
        command: EXPAND,
        template_id: template.meta.id.clone(),
        auto,
        auto_order: auto_order(EXPAND, quality(s)),
        prompt,
        negative,
        seed,
        steps,
        guidance,
        name,
        models,
        variations,
        edge,
        enhance,
        refs,
        grow,
        erase,
        rewritten: None,
        removal: false,
        prefill: false,
        tuning,
        object: None,
        purge,
    };
    crate::jobs::edit_job(
        s,
        &label,
        move |doc, active, ctx| {
            // Canvas Size with a transparent extension: the picture moves by the left/top pads.
            let (ow, oh) = (doc.size.width, doc.size.height);
            crate::image_cmds::translate_doc(doc, EXPAND, pads.left as i32, pads.top as i32)?;
            doc.size = Size::new(nw, nh);
            crate::canvas_geom::refresh(doc, crate::canvas_geom::Refresh::Shapes);
            let old = Rect::new(pads.left as i32, pads.top as i32, (pads.left + ow) as i32, (pads.top + oh) as i32);
            // Everything in the request that is not the old picture is to be painted.
            let (w, h) = (rect.width() as usize, rect.height() as usize);
            let mut coverage = vec![1.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let (cx, cy) = (rect.x0 + x as i32, rect.y0 + y as i32);
                    if cx >= old.x0
                        && cx < old.x1
                        && cy >= old.y0
                        && cy < old.y1
                        && let Some(c) = coverage.get_mut(y * w + x)
                    {
                        *c = 0.0;
                    }
                }
            }
            let (made, rw, rh, request, template_id) = fill_region(doc, active, ctx, job, rect, coverage, Some(Outpaint { picture: old.intersect(&rect) }))?;
            Ok((made, rw, rh, request, template_id, old))
        },
        move |(made, w, h, request, template_id, old)| {
            let mut v = made_json(&made, w, h, request, &template_id);
            if let Some(o) = v.as_object_mut() {
                o.insert("canvas".into(), json!([nw, nh]));
                o.insert("offset".into(), json!([old.x0, old.y0]));
            }
            v
        },
    )
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

/// `variations`: 1..=[`MAX_VARIATIONS`] whole results with consecutive seeds.
fn parse_variations(cmd: &str, p: &Value) -> Result<u32> {
    match opt_num(cmd, p, "variations", 1.0, f64::from(MAX_VARIATIONS))? {
        None => Ok(1),
        Some(x) if x.fract() == 0.0 => Ok(x as u32),
        Some(_) => Err(bad(cmd, format!("`variations` must be a whole number from 1 to {MAX_VARIATIONS}"))),
    }
}

fn edit_enabled(s: &Session) -> std::result::Result<(), String> {
    web_unavailable()?;
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// `generate.edit`: the whole picture goes to an edit model with an instruction; the result is
/// a layer over the canvas, masked to the selection when there is one.
fn plan_edit(s: &Session, p: &Value) -> Result<FillPlan> {
    let default_model = s.prefs().integrations.default_edit_model.clone();
    let mut requested = opt_str(EDIT, p, "template", 200)?.map(str::to_string).unwrap_or_else(|| AUTO_TEMPLATE.to_string());
    let mut auto = requested == AUTO_TEMPLATE;
    let mut q = p.clone();
    if auto && let Some(o) = q.as_object_mut() {
        o.remove("template");
    }
    if auto && let Some(t) = native21_auto(s, EDIT, false) {
        requested = t.to_string();
        auto = false;
    }
    let Common { template, prompt, negative, seed, steps, guidance, models, name } =
        plan_common(s, EDIT, &q, if auto { DEFAULT_EDIT_TEMPLATE } else { &requested }, Task::Edit, &default_model)?;
    let edge = parse_edge(EDIT, p)?;
    let variations = parse_variations(EDIT, p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let rect = d.doc.bounds();
    if rect.is_empty() {
        return Err(EngineError::Other("the document is empty".into()));
    }
    if u64::from(rect.width()) * u64::from(rect.height()) > MAX_REQUEST_PIXELS {
        return Err(EngineError::Other(format!(
            "the picture is {}×{} pixels; Generative Edit handles up to 16 megapixels at a time",
            rect.width(),
            rect.height()
        )));
    }
    let name = name.unwrap_or_else(|| format!("Generative Edit: {}", short(&prompt)));
    let enhance = opt_bool(EDIT, p, "enhance", s.prefs().integrations.enhance_prompts)?;
    // References (image 2, image 3 …): checked against the planned template; `auto` only
    // picks among templates with the same slots.
    let refs = crate::generate_refs::plan(s, EDIT, p, &template)?;
    // An edit tunes nothing but its LoRAs (its templates bind no sampler; the size cap is its own).
    let tuning = Tuning { loras: loras_of(EDIT, p)?, ..Tuning::default() };
    Ok(FillPlan { template, prompt, negative, seed, steps, guidance, name, models, rect, variations, auto, edge, enhance, refs, grow: 0, erase: false, tuning })
}

fn run_edit(s: &mut Session, p: &Value) -> Result<Value> {
    let plan = plan_edit(s, p)?;
    let (backend, server) = backend_for(s, &plan.template.meta.id)?;
    let FillPlan { template, prompt, negative, seed, steps, guidance, name, models, rect, variations, auto, edge, enhance, refs, grow, erase, tuning } = plan;
    let label = name.clone();
    // `wholePicture` (Generative Angles): the selection does not confine the result.
    let whole = opt_bool(EDIT, p, "wholePicture", false)?;
    let purge = purge_mode(s);
    let job = FillJob {
        thinker: thinker_for(s, &backend, &server),
        backend,
        server,
        command: EDIT,
        template_id: template.meta.id.clone(),
        auto,
        auto_order: auto_order(EDIT, quality(s)),
        prompt,
        negative,
        seed,
        steps,
        guidance,
        name,
        models,
        variations,
        edge,
        enhance,
        refs,
        grow,
        erase,
        rewritten: None,
        removal: false,
        prefill: false,
        tuning,
        object: None,
        purge,
    };
    crate::jobs::edit_job(
        s,
        &label,
        move |doc, active, ctx| {
            // The whole canvas is re-rendered; a selection, when there is one, confines what
            // of it shows (the edit template takes no mask, see `fill_region`).
            let coverage: Vec<f32> = match doc.selection.as_ref().filter(|_| !whole) {
                Some(sel) => {
                    let k = sel.channels().max(1);
                    sel.read_region(rect).chunks_exact(k).map(|c| c.first().copied().unwrap_or(0.0).clamp(0.0, 1.0)).collect()
                }
                None => vec![1.0; rect.width() as usize * rect.height() as usize],
            };
            fill_region(doc, active, ctx, job, rect, coverage, None)
        },
        move |(made, w, h, request, template_id)| made_json(&made, w, h, request, &template_id),
    )
}

/// `generate.angles`: the whole picture from another camera position, as Generative Edit runs it
/// with the Multiple Angles template and the LoRA's words for `rotation`, `tilt` and `zoom`.
fn run_angles(s: &mut Session, p: &Value) -> Result<Value> {
    let whole = |key: &str, lo: f64, hi: f64, default: f64| -> Result<f64> {
        match opt_num(ANGLES, p, key, lo, hi)? {
            None => Ok(default),
            Some(x) if x.fract() == 0.0 => Ok(x),
            Some(_) => Err(bad(ANGLES, format!("`{key}` must be a whole number"))),
        }
    };
    let rotation = whole("rotation", 0.0, 315.0, 0.0)? as u32;
    let tilt = whole("tilt", -30.0, 60.0, 0.0)? as i32;
    let zoom = whole("zoom", 0.0, 10.0, 5.0)? as u32;
    let words = angle_prompt(rotation, tilt, zoom)
        .ok_or_else(|| bad(ANGLES, "`rotation` must be 0, 45, … 315; `tilt` -30, 0, 30 or 60; `zoom` 0 (wide), 5 (medium) or 10 (close-up)"))?;
    let camera = words.trim_start_matches("<sks> ").to_string();
    let template = angles_template(s);
    let mut q = json!({"prompt": words, "template": template, "wholePicture": true, "name": format!("Angles: {camera}")});
    if template != ANGLES_TEMPLATE {
        // The instruction is already the camera move; the rewriter would only blur it.
        q["prompt"] = json!(angle_instruction(rotation, tilt, zoom).unwrap_or_default());
        q["enhance"] = json!(false);
    }
    for key in ["seed", "variations"] {
        if let Some(v) = p.get(key) {
            q[key] = v.clone();
        }
    }
    run_edit(s, &q)
}

fn doc_enabled(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// Everything `generate.removeBackground` needs, validated before any network work.
struct MattePlan {
    common: Common,
    layer: LayerId,
    /// Load the matte as the selection (combined by `mode`) instead of masking the layer.
    as_selection: bool,
    mode: SelectionMode,
    /// Send the composite rather than the layer alone.
    all_layers: bool,
    /// What to keep, as typed (empty = the model's own reading of the subject).
    subject: String,
    /// `template: auto`: the job may still fall back from the matte model to the detector.
    auto: bool,
    /// The planned template is a detector (`Task::Segment`), not a matte model.
    segment: bool,
    /// Soften a detector's hard mask with the classical edge refinement (as the Quick Action
    /// does after Select Subject).
    refine: bool,
}

/// What the detector is asked for when Remove Background has no subject named.
const SUBJECT_PROMPT: &str = "the main subject";

fn plan_remove_bg(s: &Session, p: &Value) -> Result<MattePlan> {
    let layer = layer_param(s, p)?;
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    crate::cutout_cmds::check(doc, doc.layer(layer).ok_or(EngineError::NoLayer(layer))?).map_err(EngineError::Other)?;
    let subject = opt_str(REMOVE_BG, p, "prompt", MAX_PROMPT_CHARS)?.map(str::trim).unwrap_or("").to_string();
    // `auto`: the matte model (research licence) when the preference allows it, else the
    // permissive detector (SAM 3.1), whose mask is hard-edged but needs no opt-in.
    let research_ok = s.prefs().integrations.allow_research_models;
    let requested = opt_str(REMOVE_BG, p, "template", 200)?.unwrap_or(AUTO_TEMPLATE).to_string();
    let auto = requested == AUTO_TEMPLATE;
    let template_id: &str =
        if auto { if research_ok { DEFAULT_MATTE_TEMPLATE } else { crate::select_ml_cmds::DEFAULT_SEGMENT_TEMPLATE } } else { requested.as_str() };
    let task = match template::find(template_id).map(|t| t.meta.task) {
        Ok(Task::Matte) => Task::Matte,
        Ok(Task::Segment) => Task::Segment,
        Ok(other) => return Err(bad(REMOVE_BG, format!("template `{template_id}` is a {other:?} template, not a matte or a detector"))),
        Err(e) => return Err(bad(REMOVE_BG, e.to_string())),
    };
    let mut q = p.clone();
    if let Some(o) = q.as_object_mut() {
        // The prompt the job sends depends on the route; this one satisfies the plan.
        o.insert("prompt".into(), Value::String(if subject.is_empty() { MATTE_DEFAULT_PROMPT.into() } else { subject.clone() }));
        o.insert("template".into(), Value::String(template_id.to_string()));
    }
    let common = plan_common(s, REMOVE_BG, &q, template_id, task, "")?;
    let as_selection = opt_bool(REMOVE_BG, p, "asSelection", false)?;
    let mode = match opt_str(REMOVE_BG, p, "mode", 20)? {
        None | Some("replace") | Some("new") => SelectionMode::Replace,
        Some("add") => SelectionMode::Add,
        Some("subtract") => SelectionMode::Subtract,
        Some("intersect") => SelectionMode::Intersect,
        Some(other) => return Err(bad(REMOVE_BG, format!("`mode` must be replace, add, subtract or intersect (got `{other}`)"))),
    };
    let all_layers = opt_bool(REMOVE_BG, p, "sampleAllLayers", false)?;
    let refine = opt_bool(REMOVE_BG, p, "refine", true)?;
    Ok(MattePlan { common, layer, as_selection, mode, all_layers, subject, auto, segment: task == Task::Segment, refine })
}

fn remove_bg_enabled(s: &Session) -> std::result::Result<(), String> {
    web_unavailable()?;
    let d = s.active().ok_or("no document open")?;
    crate::cutout_cmds::check(&d.doc, crate::active_layer_of(s)?)
}

/// The size a matte request goes out at: within [`MAX_MATTE_REQUEST_PIXELS`], sides in
/// multiples of 32 (TextEncodeQwenImage21 rounds its references to that; sending them rounded
/// keeps the result the request's size).
pub(crate) fn matte_request_size(w: u32, h: u32) -> (u32, u32) {
    let (rw, rh) = fit_pixels(w, h, MAX_MATTE_REQUEST_PIXELS);
    ((rw / 32).max(1) * 32, (rh / 32).max(1) * 32)
}

/// `generate.removeBackground`: the model separates the subject and answers with an RGBA image;
/// its alpha becomes the layer's mask (or the selection). The pixels it came with are the
/// model's re-rendering and are never used: the layer keeps its own.
fn run_remove_bg(s: &mut Session, p: &Value) -> Result<Value> {
    let plan = plan_remove_bg(s, p)?;
    let (backend, server) = backend_for(s, &plan.common.template.meta.id)?;
    let MattePlan { common, layer: id, as_selection, mode, all_layers, subject, auto, segment, refine } = plan;
    let Common { template, negative, seed, steps, guidance, models, .. } = common;
    let purge = purge_mode(s);
    let template_id = template.meta.id.clone();
    let label = if as_selection { "Select Subject (Generative)" } else { "Remove Background (Generative)" };
    crate::jobs::edit_job(
        s,
        label,
        move |doc, _active, ctx| {
            ctx.progress(0.0, "Rendering");
            // The route: the matte model when it is wanted and the server has it, else the
            // detector (its masks are hard-edged, its licence permissive).
            let (template_id, segment, models) = if auto && !segment {
                let t = resolve_auto(backend.as_ref(), &[DEFAULT_MATTE_TEMPLATE], crate::select_ml_cmds::DEFAULT_SEGMENT_TEMPLATE);
                let seg = t != DEFAULT_MATTE_TEMPLATE;
                (t, seg, if seg { Vec::new() } else { models })
            } else {
                (template_id, segment, models)
            };
            let prompt = match (segment, subject.is_empty()) {
                (true, true) => SUBJECT_PROMPT.to_string(),
                (false, true) => MATTE_DEFAULT_PROMPT.to_string(),
                (_, false) => subject.clone(),
            };
            let area = doc.bounds();
            let (w, h) = (area.width(), area.height());
            let n = w as usize * h as usize;
            let layer = if all_layers { None } else { doc.layer(id).and_then(|l| l.surface()) };
            // The layer's own alpha: transparent pixels go to the model over mid-grey (its
            // loader drops alpha, and black holes would read as content) and stay out of the
            // matte afterwards.
            let mut own_alpha: Option<Vec<u8>> = None;
            let rgba8: Vec<u8> = match layer {
                Some(surf) => {
                    let mut px = vec![[0u8; 4]; n];
                    surf.read_rgba8_into(area, &mut px);
                    own_alpha = Some(px.iter().map(|p| p[3]).collect());
                    px.into_iter()
                        .flat_map(|p| {
                            let a = u16::from(p[3]);
                            let over = |c: u8| ((u16::from(c) * a + 128 * (255 - a)) / 255) as u8;
                            [over(p[0]), over(p[1]), over(p[2]), 255]
                        })
                        .collect()
                }
                None => photocraft_compose::render(doc, area).to_rgba8().pixels,
            };
            let image = Rgba8::new(w, h, rgba8).map_err(gen_err)?;
            let (rw, rh) = if segment { fit_pixels(w, h, MAX_SEGMENT_REQUEST_PIXELS) } else { matte_request_size(w, h) };
            let image = if (rw, rh) == (w, h) { image } else { resize_rgba8(&image, rw, rh)? };
            let mut params = BTreeMap::new();
            if segment {
                params.insert("threshold".to_string(), json!(0.5));
            }
            let req = Request {
                template: template_id.clone(),
                prompt,
                negative,
                seed,
                steps,
                guidance,
                image: Some(image),
                mask: None,
                models,
                size: None,
                params,
                references: Vec::new(),
            };
            let resp = match run_switching(backend.as_ref(), &server, &req, purge, &JobProgress::new(ctx)) {
                Ok(r) => r,
                Err(photocraft_genai::Error::NoOutput(_)) if segment => {
                    return Err(EngineError::Other("the detector found no subject: name it in the prompt".into()));
                }
                Err(e) => return Err(gen_err(e)),
            };
            ctx.check()?;
            ctx.progress(0.95, if as_selection { "Selecting" } else { "Masking" });
            // The matte model answers with one RGBA image whose alpha is the matte; the detector
            // with one mask image per instance (red = coverage), unioned.
            let mut cov: Vec<f32> = vec![0.0; n];
            let mut images = resp.images.into_iter().peekable();
            if images.peek().is_none() {
                return Err(EngineError::Other("the backend returned no image".into()));
            }
            for out in images {
                let channel = if segment { 0 } else { 3 };
                let plane = Gray8::new(out.width, out.height, out.data.as_chunks::<4>().0.iter().map(|p| p[channel]).collect()).map_err(gen_err)?;
                let plane = if (plane.width, plane.height) == (w, h) { plane } else { resize_gray8(&plane, w, h)? };
                for (c, v) in cov.iter_mut().zip(&plane.data) {
                    *c = c.max(f32::from(*v) / 255.0);
                }
                if !segment {
                    break;
                }
            }
            // A detector's mask is hard-edged: the classical edge refinement (a narrow smart
            // radius, a little smoothing and feather, as after Select Subject) gives hair and
            // soft edges partial coverage from the pixels themselves.
            if segment && refine {
                ctx.progress(0.9, "Refining edge");
                let (content, found) = crate::select_ml_cmds::coverage_bounds(&cov, area);
                if found > 0 {
                    let wv = w as usize;
                    let reader = |r: Rect| -> Vec<f32> {
                        let mut v = Vec::with_capacity(r.width() as usize * r.height() as usize);
                        for y in r.y0..r.y1 {
                            for x in r.x0..r.x1 {
                                let inside = x >= area.x0 && x < area.x1 && y >= area.y0 && y < area.y1;
                                v.push(if inside { cov.get((y - area.y0) as usize * wv + (x - area.x0) as usize).copied().unwrap_or(0.0) } else { 0.0 });
                            }
                        }
                        v
                    };
                    let refined = crate::smartselect_cmds::with_doc_sampler(doc, Some(id), all_layers, |smp, d| {
                        photocraft_algo::matting::refine_mask(smp, &reader, content, d.bounds(), &crate::cutout_cmds::REFINE)
                    });
                    if let Some(reg) = refined {
                        cov = photocraft_algo::matting::region_reader(&reg)(area);
                    }
                }
            }
            if let Some(own) = own_alpha {
                for (c, a) in cov.iter_mut().zip(own) {
                    *c *= f32::from(a) / 255.0;
                }
            }
            let (bounds, count) = crate::select_ml_cmds::coverage_bounds(&cov, area);
            if count == 0 {
                return Err(EngineError::Other("the model kept nothing: name the subject in the prompt".into()));
            }
            if as_selection {
                doc.selection = sel::combine(doc.selection.as_ref(), &cov, area, mode);
            } else {
                crate::extra_cmds::background_to_layer_for_mask(doc, id);
                let mut surface = Surface::with_default(PixelFormat::GRAY8, &[0.0]);
                surface.write_region(area, &cov);
                doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.mask = Some(LayerMask { surface, ..LayerMask::reveal_all() });
                doc.selection = None;
            }
            Ok((bounds, count, resp.seed, resp.run_id, resp.elapsed_ms, resp.timings, (rw, rh), (w, h), template_id))
        },
        move |(b, count, seed, run_id, ms, timings, (rw, rh), (w, h), tid)| {
            json!({
                "layer": id.0, "selection": as_selection, "bounds": [b.x0, b.y0, b.width(), b.height()], "pixels": count,
                "seed": seed, "template": tid, "runId": run_id, "width": w, "height": h, "requestWidth": rw, "requestHeight": rh,
                "ms": ms, "timings": [serde_json::to_value(timings).unwrap_or(Value::Null)],
            })
        },
    )
}

/// A detector request is sent at most this large (SAM 3.1 works at about 1 megapixel inside).
const MAX_SEGMENT_REQUEST_PIXELS: u64 = 2048 * 1024;

/// The size a split request goes out at: at most [`LAYERED_MAX_SIDE`] on the longer side, or,
/// with `max_pixels`, as large as fits that many pixels (up or down: 1 MP at 3:4 is 864×1152);
/// multiples of 16 (the layered latent's step), at least 64.
pub(crate) fn layered_request_size(w: u32, h: u32, max_pixels: Option<u64>) -> (u32, u32) {
    let longer = w.max(h).max(1);
    let scale = match max_pixels {
        Some(m) => (m as f64 / (f64::from(w.max(1)) * f64::from(h.max(1)))).sqrt(),
        None if longer > LAYERED_MAX_SIDE => f64::from(LAYERED_MAX_SIDE) / f64::from(longer),
        None => 1.0,
    };
    let fit = |v: u32| (((f64::from(v) * scale).round() as u32) / SIZE_STEP * SIZE_STEP).max(SIZE_STEP);
    (fit(w), fit(h))
}

/// Everything `generate.splitLayers` needs, validated before any network work.
struct SplitPlan {
    common: Common,
    layers: u32,
    all_layers: bool,
    /// `maxPixels`: the working size by pixel count instead of the 640 px side.
    max_pixels: Option<u64>,
}

fn plan_split(s: &Session, p: &Value) -> Result<SplitPlan> {
    s.active().ok_or(EngineError::NoDocument)?;
    let mut q = p.clone();
    if opt_str(SPLIT, p, "prompt", MAX_PROMPT_CHARS)?.map(str::trim).unwrap_or("").is_empty()
        && let Some(o) = q.as_object_mut()
    {
        o.insert("prompt".into(), Value::String(SPLIT_DEFAULT_PROMPT.into()));
    }
    let common = plan_common(s, SPLIT, &q, DEFAULT_LAYERS_TEMPLATE, Task::Layers, "")?;
    let layers = match opt_num(SPLIT, p, "layers", 1.0, f64::from(MAX_SPLIT_LAYERS))? {
        None => 3,
        Some(x) if x.fract() == 0.0 => x as u32,
        Some(_) => return Err(bad(SPLIT, format!("`layers` must be a whole number from 1 to {MAX_SPLIT_LAYERS}"))),
    };
    let all_layers = opt_bool(SPLIT, p, "sampleAllLayers", true)?;
    let max_pixels = opt_num(SPLIT, p, "maxPixels", 4096.0, 4_194_304.0)?.map(|m| m as u64);
    Ok(SplitPlan { common, layers, all_layers, max_pixels })
}

fn split_enabled(s: &Session) -> std::result::Result<(), String> {
    web_unavailable()?;
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// `generate.splitLayers`: the picture goes to a layered model that decomposes it into RGBA
/// layers (background first); each comes back as a new layer above the active one, with the
/// alpha the model gave it.
fn run_split(s: &mut Session, p: &Value) -> Result<Value> {
    let plan = plan_split(s, p)?;
    // The worker serves it when it has the files (Qwen-Image-Layered is Apache-2.0: no opt-in).
    let (backend, server) = backend_for(s, &plan.common.template.meta.id)?;
    let SplitPlan { common, layers, all_layers, max_pixels } = plan;
    let Common { template, prompt, negative, seed, steps, guidance, models, name } = common;
    let purge = purge_mode(s);
    let tid = template.meta.id.clone();
    let tid_out = tid.clone();
    let base = name.unwrap_or_else(|| "Split Layer".to_string());
    crate::jobs::edit_job(
        s,
        "Split into Layers",
        move |doc, active, ctx| {
            ctx.progress(0.0, "Rendering");
            let area = doc.bounds();
            let (w, h) = (area.width(), area.height());
            let n = w as usize * h as usize;
            let layer = if all_layers { None } else { active.and_then(|id| doc.layer(id)).and_then(|l| l.surface()) };
            let rgba8: Vec<u8> = match layer {
                Some(surf) => {
                    let mut px = vec![[0u8; 4]; n];
                    surf.read_rgba8_into(area, &mut px);
                    // Transparent pixels go to the model over mid-grey (its loader drops alpha).
                    px.into_iter()
                        .flat_map(|p| {
                            let a = u16::from(p[3]);
                            let over = |c: u8| ((u16::from(c) * a + 128 * (255 - a)) / 255) as u8;
                            [over(p[0]), over(p[1]), over(p[2]), 255]
                        })
                        .collect()
                }
                None => photocraft_compose::render(doc, area).to_rgba8().pixels,
            };
            let image = Rgba8::new(w, h, rgba8).map_err(gen_err)?;
            let (rw, rh) = layered_request_size(w, h, max_pixels);
            let image = if (rw, rh) == (w, h) { image } else { resize_rgba8(&image, rw, rh)? };
            let mut params = BTreeMap::new();
            params.insert("layers".to_string(), json!(layers));
            let req = Request {
                template: tid.clone(),
                prompt: prompt.clone(),
                negative,
                seed,
                steps,
                guidance,
                image: Some(image),
                mask: None,
                models,
                size: Some((rw, rh)),
                params,
                references: Vec::new(),
            };
            let resp = run_switching(backend.as_ref(), &server, &req, purge, &JobProgress::new(ctx)).map_err(gen_err)?;
            ctx.check()?;
            ctx.progress(0.95, "Placing");
            if resp.images.is_empty() {
                return Err(EngineError::Other("the backend returned no layers".into()));
            }
            let count = resp.images.len();
            let mut made: Vec<LayerId> = Vec::with_capacity(count);
            for (i, out) in resp.images.into_iter().enumerate() {
                let out = if (out.width, out.height) == (w, h) { out } else { resize_rgba8(&out, w, h)? };
                let layer_name = format!("{base} {}/{count}", i + 1);
                let mut layer = Layer::raster(layer_name.clone(), doc.pixel_format());
                write_rgba8(&mut layer, &out, area)?;
                set_generative_info(
                    &mut layer,
                    &GenerativeInfo {
                        command: SPLIT.to_string(),
                        prompt: prompt.clone(),
                        template: tid.clone(),
                        seed: resp.seed,
                        steps,
                        guidance,
                        rect: [area.x0, area.y0, w as i32, h as i32],
                        name: layer_name,
                        ..GenerativeInfo::default()
                    },
                );
                // Background first, each next layer above the last.
                let nid = doc.insert_above(*active, layer);
                *active = Some(nid);
                made.push(nid);
            }
            Ok((made, resp.seed, resp.run_id, resp.elapsed_ms, resp.timings, (rw, rh), (w, h)))
        },
        move |(made, seed, run_id, ms, timings, (rw, rh), (w, h))| {
            json!({
                "layers": made.iter().map(|l| l.0).collect::<Vec<_>>(), "count": made.len(), "seed": seed, "template": tid_out,
                "runId": run_id, "width": w, "height": h, "requestWidth": rw, "requestHeight": rh,
                "ms": ms, "timings": [serde_json::to_value(timings).unwrap_or(Value::Null)],
            })
        },
    )
}

/// Edit › Purge › Generative Models: the server unloads its models and frees their memory. A
/// server that has had several model families loaded can end up streaming weights on every run
/// (2 GB of VRAM free, fills three times slower); this resets it, at the price of one reload.
fn run_free(s: &mut Session, _p: &Value) -> Result<Value> {
    backend(s)?.free().map_err(gen_err)?;
    forget_models(&server_key(s));
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
        Err(e) => Health { backend: "comfyui".into(), url: s.prefs().integrations.comfy_server.clone(), error: Some(e.to_string()), ..Health::default() },
    };
    serde_json::to_value(&h).map_err(|e| EngineError::Other(e.to_string()))
}

/// The template `auto` most likely picks for `command` under `quality` (the first of its order:
/// what a server with every file runs), for estimates before the server has answered.
pub fn likely_auto(command: &str, quality: &str) -> &'static str {
    auto_order(command, quality).first().copied().unwrap_or(DEFAULT_FILL_TEMPLATE)
}

/// [`likely_auto`] in this session: the Qwen-Image-2.1 template where `auto` runs it on the
/// native worker ([`native21_auto`]), else the first of the server's order.
pub fn likely_auto_in(s: &Session, command: &str, quality: &str) -> &'static str {
    native21_auto(s, command, false).unwrap_or_else(|| likely_auto(command, quality))
}

/// Seconds a run of `template` for a `width × height` area with `count` variations and
/// `references` reference images is expected to take on a warm server (the template's
/// `estimatedSeconds`; a fill, expand or edit counts at the size it is sent at, see
/// [`request_size`]). `None` for an unknown template. The one
/// formula the Generate panel, `generate.models` and scripts share.
pub fn estimate_seconds(template: &str, width: u32, height: u32, count: u32, references: u32) -> Option<f32> {
    let t = template::find(template).ok()?;
    let (w, h) = match t.meta.task {
        Task::Fill | Task::Expand | Task::Edit => request_size(width, height),
        _ => (width, height),
    };
    Some(t.meta.estimate_seconds(w, h, count, references))
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
    let q = quality(s);
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
        .filter_map(|(key, command, removal)| native21_auto(s, command, removal).map(|t| (key, t)))
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
    let mut templates = Vec::new();
    for t in template::builtin() {
        let m = &t.meta;
        let mut slots = Vec::new();
        for slot in &m.models {
            let files = if online {
                folders.entry(slot.folder.clone()).or_insert_with(|| backend.and_then(|b| b.model_files(&slot.folder).ok())).clone()
            } else {
                None
            };
            let installed = files.as_ref().map(|f| f.iter().any(|x| x == &slot.default));
            let mut v = json!({"placeholder": slot.placeholder, "folder": slot.folder, "file": slot.default, "installed": installed});
            // The NVFP4 alternative, and whether the server has it (the policy picks it on a
            // Blackwell GPU, see `generate.models`'s `blackwell`).
            if let (Some(alt), Some(o)) = (&slot.nvfp4, v.as_object_mut()) {
                o.insert("nvfp4".into(), json!(alt));
                o.insert("nvfp4Installed".into(), json!(files.as_ref().map(|f| f.iter().any(|x| x == alt))));
            }
            slots.push(v);
        }
        let eta = eta_for.and_then(|(w, h, c, r)| estimate_seconds(&m.id, w, h, c, r));
        templates.push(json!({
            "id": m.id, "name": m.name, "family": m.family, "task": m.task, "license": m.license, "licenseNote": m.license_note,
            "allowed": m.license != License::Research || allow_research,
            "defaults": {"steps": m.defaults.steps, "guidance": m.defaults.guidance},
            "needsImage": m.needs_image, "needsMask": m.needs_mask, "models": slots,
            "creator": m.creator, "description": m.description, "badges": m.badges, "references": m.references,
            "sizes": m.sizes, "counts": m.counts, "estimatedSeconds": m.estimated_seconds, "pages": m.pages, "eta": eta,
        }));
    }
    // What `auto` would pick right now (needs the server's model lists).
    let auto = |order, fallback| backend.filter(|_| online).map(|b| resolve_auto(b, order, fallback));
    // Remove Background: the matte model only with the research opt-in, else the detector.
    let auto_matte = backend.filter(|_| online).map(|b| {
        if allow_research {
            resolve_auto(b, &[DEFAULT_MATTE_TEMPLATE], crate::select_ml_cmds::DEFAULT_SEGMENT_TEMPLATE)
        } else {
            crate::select_ml_cmds::DEFAULT_SEGMENT_TEMPLATE.to_string()
        }
    });
    // The GPU, and whether the NVFP4 policy's `auto` applies to it.
    let device = health.as_ref().map(|h| h.device.clone()).unwrap_or_default();
    json!({
        "server": health, "templates": templates,
        "device": device, "blackwell": photocraft_genai::is_blackwell(&device),
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
    let template = template::find(&info.template).map_err(|e| bad(SIMILAR, e.to_string()))?;
    if template.meta.license == License::Research && !s.prefs().integrations.allow_research_models {
        return Err(EngineError::Other(format!(
            "`{}` uses a research-only model; turn on Allow Research-Only Models in Preferences › AI Integrations to use it",
            template.meta.id
        )));
    }
    let canvas = d.doc.bounds();
    let rect = Rect::new(info.rect[0], info.rect[1], info.rect[0].saturating_add(info.rect[2]), info.rect[1].saturating_add(info.rect[3])).intersect(&canvas);
    if rect.is_empty() {
        return Err(EngineError::Other("the layer's area is no longer on the canvas".into()));
    }
    // An expand layer's area is regenerated as a fill (its canvas is already there); edits and
    // fills run as what they were.
    let q = quality(s);
    let (command, auto, auto_order, template_id) = match (info.command.as_str(), native21_auto(s, FILL, false)) {
        (c, _) if c == EDIT => (EDIT, false, auto_order(EDIT, q), info.template.clone()),
        (c, _) if c == FILL => (FILL, false, auto_order(FILL, q), info.template.clone()),
        (_, Some(t)) => (FILL, false, auto_order(FILL, q), t.to_string()),
        _ => (FILL, true, auto_order(FILL, q), DEFAULT_FILL_TEMPLATE.to_string()),
    };
    let refs = crate::generate_refs::Refs { images: crate::generate_refs::resolve(s, &info.references)?, specs: info.references.clone() };
    // The engine that serves the template (the native worker under `auto` for 2.1 and SAM).
    let (backend, server) = backend_for(s, &template_id)?;
    let purge = purge_mode(s);
    let job = FillJob {
        thinker: thinker_for(s, &backend, &server),
        backend,
        server,
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
        prefill: false,
        tuning: Tuning::default(),
        object: None,
        purge,
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
    #[cfg(not(target_arch = "wasm32"))]
    if engine_mode(s) == "auto"
        && s.prefs().integrations.native_configured()
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
        Some(other) => return Err(bad(ENHANCE, format!("`task` must be \"fill\", \"edit\" or \"image\" (got `{other}`)"))),
    };
    let use_image = opt_bool(ENHANCE, p, "useImage", true)?;
    let seed = parse_seed(ENHANCE, p)?;
    let backend = enhancer_backend(s)?;
    let (image, place) = match (kind, use_image, s.active()) {
        (EnhanceKind::Image, _, _) | (_, false, _) | (_, _, None) => (None, None),
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
            params: r#"{"prompt":text,"negative":text,"steps":0..250=0,"guidance":0..30=0,"margin":0..1=0.25,"grow":0..512=0,"edge":"soft|hard","variations":int=1,"seed":{u64?=random},"template":{id?=Preferences › Default Fill Template; "auto" = the fastest permissive tier whose files the server has},"model":{file?=Preferences},"enhance":{bool?=Preferences › Enhance prompts},"fillAsEdit":bool=false,"erase":{bool?=the template's},"sampler":{str?=euler},"scheduler":{str?=simple},"maxPixels":{int?=1048576},"denoise":{0.05..1?} (the 2511 fill templates: how much of the selection is redrawn, 1 replaces it; measured on the 40-step template, 0.9 keeps the old sheen and 0.8 little changes, the 8-step tier hardly moves below 1),"loras":{[{"file":name,"strength":-4..4=1}]?},"references":{[{"layer":id} | {"path":"…"} | {"job":id,"index":0}]?},"name":{str?}} → {"layer","layers":[id],"seed","seeds":[u64],"template","runId","width","height","requestWidth","requestHeight","ms","timings":[{"encodeMs","uploadMs","queueMs","runMs","downloadMs"}]} (steps and guidance 0 = the template's defaults; loras = LoRA files from the LoRA folder (the native model root's loras/), applied by the native engine as side paths on its resident model, up to 8; on the ComfyUI route each becomes a LoraLoaderModelOnly after the template's model loader, the file in that server's models/loras; fillAsEdit = when auto runs Qwen-Image-2.1 on the native worker, fill by instruction (measured weaker than the fill template with its rewriter: the namer can pick a neighbour) on its edit template: the rewriter names what the selection holds and the description becomes "Replace the <thing> with <prompt>" (an instruction such as "replace the cup with a vase" goes as it is; nothing named: the fill template); enhance = rewrite the prompt first with the vision-language model, the picture and the selection's place in view (generate.enhancePrompt), the layer remembering both; margin = context around the selection as a fraction of its larger side; grow = widen the selection by that many pixels first, so a bulkier replacement has room; edge soft = the result's layer mask fades across the band the model re-rendered around the selection (4 % of its size), dithered with grain so there is no visible line; hard = the mask is exactly the selection; references = up to two reference images: with no template named the run switches to the fill-from-reference template, which flattens the selection to grey in the request so the model paints what the references show there (erase; `erase: true` does the same on any fill, `false` keeps the pixels); variations 1..4 = results with consecutive seeds, each a layer, only the first visible; sampler and scheduler = the KSampler's, ComfyUI names; maxPixels = the request is downscaled to at most this many pixels (the model's working size); areas over 1 megapixel are sent downscaled and come back resampled; a background job: the result is a new layer above the active one; needs a ComfyUI server, see Preferences › AI Integrations)"#,
            enabled: fill_enabled,
            run: run_fill,
            journal: true,
        },
        CommandSpec {
            id: IMAGE,
            label: "Generate Image…",
            menu: &[],
            shortcut: None,
            params: r#"{"prompt":text,"negative":text,"target":"auto|layer|document","width":int=0,"height":int=0,"transparent":bool=false,"steps":0..250=0,"guidance":0..30=0,"seed":{u64?=random},"template":{id?=Preferences › Default Image Template},"model":{file?=Preferences},"enhance":{bool?=Preferences › Enhance prompts},"name":{str?},"variations":int=1,"references":[{"layer":id}|{"path":file}|{"job":id,"index":int=0}]?} → {"layer"|"document","layers":[id],"seed","seeds":[u64],"template","runId","width","height","ms"} (target auto = a layer over the whole canvas when a document is open, else a new document; enhance = expand the idea into a detailed paragraph first with the vision-language model (generate.enhancePrompt); width/height 0 = the document's size or 1024, otherwise 64..4096 rounded down to multiples of 16; transparent = ask for the subject alone on a transparent background and keep the alpha the model returns (templates whose model can, such as qwen-2.1/image; others refuse); steps and guidance 0 = the template's defaults; variations 1..4 = results with consecutive seeds, each a layer (in the one new document with target document), only the first visible; references = images that guide the result (a pixel layer of the active document, an image file, or a stored result of the Generations library), at most the template's references.max (generate.models); with references and no template named, a default template that takes none gives way to qwen-edit-2511/image-ref; a background job; needs a ComfyUI server, see Preferences › AI Integrations)"#,
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
            params: r#"{"prompt":text,"left":int=0,"top":int=0,"right":int=0,"bottom":int=0,"width":int=0,"height":int=0,"negative":text,"steps":0..250=0,"guidance":0..30=0,"edge":"soft|hard","variations":int=1,"margin":{0..1=0.25},"anchor":{str?=center},"seed":{u64?=random},"template":{id?="auto": the Lightning expand tier when its LoRA is installed, else the 40-step one},"model":{file?=Preferences},"name":{str?}} → {"layer","layers":[id],"seed","seeds":[u64],"template","canvas":[w,h],"offset":[x,y],"width","height","requestWidth","requestHeight","ms","timings"} (adds canvas (left/top/right/bottom in pixels, or a larger width/height placed by anchor: topLeft, top, topRight, left, center, right, bottomLeft, bottom, bottomRight) and has the model paint it, as one undo step: the picture moves by the left/top pads, the result is a new layer over the added area; edge soft = its mask fades across the re-rendered band just inside the old edge (8 % of the picture's longer side), dithered; hard = exactly the added area; an empty prompt continues the scene; otherwise as generate.fill; needs a ComfyUI server)"#,
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
            params: r#"{"prompt":text,"negative":text,"steps":0..250=0,"guidance":0..30=0,"edge":"soft|hard","variations":int=1,"seed":{u64?=random},"template":{id?="auto": the Lightning edit tier when its LoRA is installed, else the 40-step one},"model":{file?=Preferences},"enhance":{bool?=Preferences › Enhance prompts},"name":{str?},"wholePicture":bool=false,"references":[{"layer":id}|{"path":file}|{"job":id,"index":int=0}]?} → {"layer","layers":[id],"seed","seeds":[u64],"template","runId","width","height","requestWidth","requestHeight","ms","timings"} (edits the whole picture by instruction; enhance = rewrite the instruction first with the vision-language model, the picture in view, so it names what is there instead of "this" (generate.enhancePrompt): "make the sky stormy", "turn the boat blue", "remove the person"; a description such as "a stormy sky" is wrapped as "change this image so that it shows …"; the composite goes to the model and the result is a new layer above the active one; with a selection the layer is masked to it (edge soft = a dithered fade around it, hard = exactly) so only that part of the edit shows; variations 1..4 = results with consecutive seeds, each a layer, only the first visible; pictures over 0.75 megapixels are sent downscaled and come back resampled; references = extra images the instruction can name as image 2, image 3 (the 2511 templates take two, qwen-2.1/edit nine; see references.max in generate.models): a pixel layer, an image file or a stored generation; a background job; needs a ComfyUI server)"#,
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
            params: r#"{"rotation":0|45|90|135|180|225|270|315=0,"tilt":-30|0|30|60=0,"zoom":0|5|10=5,"seed":{u64?=random},"variations":int=1} → as generate.edit (the whole picture seen from another camera, as layers over the canvas like Generative Edit's: rotation = degrees around the subject from the front (45 front-right quarter view … 180 back view … 315 front-left quarter view), tilt = the camera's height (-30 low angle, 0 eye level, 30 elevated, 60 high angle), zoom = 0 wide, 5 medium, 10 close-up; runs qwen-edit-2511/angles, which needs fal's Multiple Angles LoRA (Apache-2.0) in ComfyUI/models/loras; a selection does not confine it)"#,
            enabled: edit_enabled,
            run: run_angles,
            journal: true,
        },
        CommandSpec {
            id: REMOVE_BG,
            label: "Remove Background (Generative)…",
            // Placed by the menu catalogue under Edit, with the other generative items.
            menu: &[],
            shortcut: None,
            params: r#"{"prompt":text,"asSelection":bool=false,"sampleAllLayers":bool=false,"refine":bool=true,"mode":"replace|add|subtract|intersect","layer":{id?=active},"seed":{u64?=random},"steps":{0..250=0},"guidance":{0..30=0},"template":{id?="auto": qwen-2.1/matte (a soft matte from Qwen-Image-2.1's alpha output; research licence, Preferences › AI Integrations › Allow Research-Only Models) when allowed and installed, else sam3.1/segment (the permissive detector)},"model":{file?}} → {"layer","selection","bounds":[x,y,w,h],"pixels","seed","template","runId","width","height","requestWidth","requestHeight","ms","timings"} (prompt = what to keep, e.g. "the lighthouse", empty = the model's own reading of the subject; the matte becomes the layer's mask (the Background becomes a normal layer; the pixels are never changed) or, with asSelection, the selection combined by mode; sampleAllLayers sends the composite instead of the layer alone; refine = soften the detector's hard mask with the classical edge refinement, as the Quick Action does; images over one megapixel are sent downscaled and the matte comes back resampled; a background job; needs a ComfyUI server; the classical Quick Action layer.removeBackground needs none)"#,
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
            params: r#"{"layer":{id?=active},"seed":{u64?=random},"variations":int=1} → as the command that made the layer (runs it again with a new seed in the same place: the layer's mask is the area, the result is a new layer above it; a generated image is generated again as a layer; an expanded area is filled again with its prompt; needs a ComfyUI server)"#,
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
            params: r#"{"layers":int=3,"prompt":text,"negative":text,"sampleAllLayers":bool=true,"maxPixels":4096..4194304?,"steps":0..250=0,"guidance":0..30=0,"seed":{u64?=random},"template":{id?=qwen-layered/split},"model":{file?},"name":{str?}} → {"layers":[id],"count","seed","template","runId","width","height","requestWidth","requestHeight","ms","timings"} (a layered model (Qwen-Image-Layered, Apache-2.0) decomposes the picture into `layers` RGBA layers, the background first, each a new layer above the active one with the alpha the model gave it; prompt = a description of the picture, empty = a generic instruction; sampleAllLayers false sends the active layer alone; the picture goes out at 640 px on its longer side, or at maxPixels pixels (1000000 = 864x1152 at 3:4), and the layers come back resampled; a background job; needs a ComfyUI server with the layered model's files)"#,
            enabled: split_enabled,
            run: run_split,
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
            params: r#"{"prompt":text,"task":"fill|edit|image","useImage":bool=true,"seed":{u64?=random}} → {"prompt","enhanced"} (rewrites a prompt the way the image models want it, with the Qwen3-VL 4B encoder the Krea 2 templates install run as a vision-language model (ComfyUI's TextGenerate): a fill or edit instruction becomes one sentence that leads with the verb, names what the picture shows instead of "this" or "it", says what the result looks like and what stays unchanged; an image idea becomes a paragraph of 60–120 words (subject, materials, pose, lighting, setting, style) keeping every subject, colour and medium given; useImage = the rewriter sees the composite (a fill is also told where the selection sits); enhanced false = the model said nothing usable and the prompt comes back as typed; a background job; the fill, edit and image commands take "enhance":true to do this on the way, and Preferences › AI Integrations › Enhance prompts makes it their default; needs a ComfyUI server with the Krea 2 text encoder)"#,
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
            params: r#"{"probe":bool=true,"width":{int?},"height":{int?},"count":int=1,"references":int=0} → {"server","device","blackwell","templates":[{"id","name","family","task","license","allowed","defaults","models":[{"placeholder","folder","file","installed","nvfp4"?,"nvfp4Installed"?}],"creator","description","badges","references":{"max","min","placeholders","nodes"},"sizes":{"aspects","min","max","step"},"counts":{"max","default"},"estimatedSeconds":{"base","perMegapixel","perReference"},"pages","eta"}],"autoFill","autoExpand","autoEdit","autoRemove","autoMatte"} (autoFill / autoExpand / autoEdit / autoRemove = the template `auto` runs for Generative Fill, Expand, Edit and the removals (Content-Aware Fill, Remove, Spot Healing): Qwen-Image-2.1 on the native worker when research-only models are allowed and the worker runs that family, else the server's Qwen-Image-Edit-2511 order; the capabilities drive the Generate panel: pages = where a template is offered (image, fill, edit, expand, matte, split, select, enhance), references = how many reference images it takes, sizes = an image template's aspects and side limits; with width and height every template reports eta = (base + perMegapixel × megapixels + perReference × extra references) × count seconds on a warm server, fills and edits at the size they are sent at (a reference-driven text to image's first reference is in base); probe false = list the templates without contacting the server: no server status, installed unknown; nvfp4 = the slot's NVFP4 alternative, which Preferences › AI Integrations › Model precision "auto" loads instead of file on a Blackwell GPU (blackwell true) when the server has it)"#,
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
            params: r#"{"scale":1..4=2,"tileMegapixels":0.25..2=1,"prompt"?:string,"template"?:string,"seed"?:int,"steps"?:int,"guidance"?:number,"loras"?:[{"file","strength"}],"restoreStrength":0..1.5=0.5,"name"?:string} → {"layer","tiles","width","height","elapsedMs","template"} (the document enlarged by scale the plain way (bicubic), then redrawn tile by tile by the edit model with a restore prompt (AusBoss's wording unless prompt is given): tiles of about tileMegapixels sharing a 128-px strip, each moved back onto its source and matched to its colours, the strips faded together; the redraw is a new layer above. Auto: Qwen-Image-2.1's edit on the native engine when it serves it (with the Restore LoRA at restoreStrength when the LoRA folder has qwen-image-2.1-restore.safetensors and no loras are given), else the edit default)"#,
            enabled: edit_enabled,
            run: upscale::run_upscale,
            journal: true,
        },
    ]
}

#[path = "generate_loras.rs"]
pub(crate) mod loras;

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

#[cfg(test)]
#[path = "generate_split_tests.rs"]
mod split_tests;

#[cfg(test)]
#[path = "generate_enhance_tests.rs"]
mod enhance_tests;

#[cfg(test)]
#[path = "generate_precision_tests.rs"]
mod precision_tests;
