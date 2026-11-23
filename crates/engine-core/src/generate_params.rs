//! Reading `generate.*` parameters: the shared option parsers, the sampling tuning a request
//! carries (steps, guidance, sampler, LoRAs) and the plan every generative command starts from.

use std::collections::BTreeMap;

use photocraft_genai::Task;
use photocraft_genai::template::{self, Template};
use serde_json::{Value, json};

use crate::{EngineError, Result, Session};

use super::{MAX_FILL_REQUEST_PIXELS, MAX_PROMPT_CHARS, MAX_REQUEST_PIXELS, MAX_SIDE, MAX_VARIATIONS, MIN_SIDE, NAME_SHADOW, REMOVE_MARGIN, SIZE_STEP};

/// A small request is upscaled until its longer side is this long: the 2511 templates sample at
/// the request's own size, and a handful of latent tokens cannot carry structure.
pub(super) const MIN_FILL_REQUEST_SIDE: u32 = 512;

/// The sampler and scheduler the 2511 fill templates run unless a request says otherwise:
/// ComfyUI's official template's pair.
pub(super) const DEFAULT_SAMPLER: &str = "euler";

pub(super) const DEFAULT_SCHEDULER: &str = "simple";

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
    /// as side paths on its resident model. Files live under the Models folder’s `loras/`.
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
    /// `integrations.fastDraft`: a 2511 template of 20 steps or more runs with the worker's
    /// first-block step cache ([`FAST_DRAFT_STEP_CACHE`]; Split into Layers has its own).
    pub fast_draft: bool,
    /// Worker params passed as they are (the Angles template's `lightning`, `angles_strength`
    /// and `lightning_strength`, see [`angles_extra`]).
    pub extra: BTreeMap<String, Value>,
}

/// The step-cache threshold Fast draft asks for: the 2511 40-step edit at 864x1152 39.4 to
/// 24.8 s, 32.5 dB against the full run, the snow slightly softer (2026-10-30).
pub(crate) const FAST_DRAFT_STEP_CACHE: f64 = 0.05;

/// Whether Fast draft applies to `template` at `steps`: a Qwen-Image-Edit-2511 template that is
/// not one of its few-step Lightning tiers.
pub(crate) fn fast_draft_applies(template: &str, steps: u32) -> bool {
    let steps = if steps == 0 { template::find(template).map_or(0, |t| t.meta.defaults.steps) } else { steps };
    template.starts_with("qwen-edit-2511/") && steps >= 20
}

/// Whether `template`'s sampler takes a `denoise` (the 2511 fill templates; Qwen-Image-2.1's
/// edit-style fill and the native engine redraw from noise and ignore it).
pub fn takes_denoise(template: &str) -> bool {
    template::find(template).is_ok_and(|t| t.meta.parameters.iter().any(|p| p == "denoise"))
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            sampler: None,
            scheduler: None,
            max_pixels: MAX_FILL_REQUEST_PIXELS,
            loras: Vec::new(),
            keep_object: false,
            shadow: true,
            remove_margin: REMOVE_MARGIN,
            name_shadow: NAME_SHADOW,
            denoise: None,
            fast_draft: false,
            extra: BTreeMap::new(),
        }
    }
}

/// At most this many LoRAs on one request.
pub(super) const MAX_LORAS: usize = 8;

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
                Some(_) => Err(bad(cmd, format!("`{key}` must be a sampler or scheduler name (letters, digits and underscores)"))),
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
        Ok(Tuning {
            sampler: name("sampler")?,
            scheduler: name("scheduler")?,
            max_pixels,
            loras: loras_of(cmd, p)?,
            keep_object,
            shadow,
            remove_margin,
            name_shadow,
            denoise,
            fast_draft: false,
            extra: BTreeMap::new(),
        })
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
            Some(s) => {
                s.as_f64().filter(|v| v.is_finite() && (-4.0..=4.0).contains(v)).ok_or_else(|| bad(cmd, "a LoRA `strength` is a number from -4 to 4"))? as f32
            }
        };
        out.push((file.to_string(), strength));
    }
    Ok(out)
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

/// How a generated layer's mask ends: softly, across the band the model re-rendered around the
/// selection, with per-pixel noise so the ramp reads as grain rather than a gradient or a line
/// (the default), or hard, exactly on the selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Edge {
    Soft,
    Hard,
}

impl Edge {
    /// The `edge` parameter's spelling.
    pub(super) fn name(self) -> &'static str {
        match self {
            Edge::Soft => "soft",
            Edge::Hard => "hard",
        }
    }
}

pub(super) fn parse_edge(cmd: &str, p: &Value) -> Result<Edge> {
    match opt_str(cmd, p, "edge", 10)? {
        None | Some("soft") => Ok(Edge::Soft),
        Some("hard") => Ok(Edge::Hard),
        Some(other) => Err(bad(cmd, format!("`edge` must be soft or hard (got `{other}`)"))),
    }
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

/// Templates that run with an empty prompt.
const PROMPTLESS_TEMPLATES: [&str; 1] = ["moss-image/restore"];

pub(crate) fn plan_common(_s: &Session, cmd: &str, p: &Value, default_template: &str, task: Task, default_model: &str) -> Result<Common> {
    let template_id = opt_str(cmd, p, "template", 100)?.unwrap_or(default_template);
    // A restore LoRA's run takes no prompt (its author's: both prompts empty).
    let prompt = match opt_str(cmd, p, "prompt", MAX_PROMPT_CHARS)?.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => t.to_string(),
        None if PROMPTLESS_TEMPLATES.contains(&template_id) => String::new(),
        None => return Err(bad(cmd, "`prompt` is required: describe what to generate")),
    };
    let negative = opt_str(cmd, p, "negative", MAX_PROMPT_CHARS)?.unwrap_or("").to_string();
    let template = template::find(template_id).map_err(|e| bad(cmd, e.to_string()))?;
    if template.meta.task != task {
        return Err(bad(cmd, format!("template `{template_id}` is a {:?} template, not {task:?}", template.meta.task)));
    }
    // Every model is the owner's own files: a research licence is shown in the picker, never a
    // reason to refuse a request (owner's call, 2026-11-01).
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
pub(super) fn side(cmd: &str, p: &Value, key: &str, default: u32) -> Result<u32> {
    // 0 (the generated dialog's default) means "use the default size".
    let v = match opt_num(cmd, p, key, 0.0, f64::from(MAX_SIDE))?.map(|x| x.round() as u32) {
        None | Some(0) => default,
        Some(v) if v < MIN_SIDE => return Err(bad(cmd, format!("`{key}` must be 0 (default) or {MIN_SIDE}..{MAX_SIDE} (got {v})"))),
        Some(v) => v,
    };
    Ok((v / SIZE_STEP * SIZE_STEP).clamp(MIN_SIDE, MAX_SIDE))
}

/// `variations`: 1..=[`MAX_VARIATIONS`] whole results with consecutive seeds.
pub(super) fn parse_variations(cmd: &str, p: &Value) -> Result<u32> {
    match opt_num(cmd, p, "variations", 1.0, f64::from(MAX_VARIATIONS))? {
        None => Ok(1),
        Some(x) if x.fract() == 0.0 => Ok(x as u32),
        Some(_) => Err(bad(cmd, format!("`variations` must be a whole number from 1 to {MAX_VARIATIONS}"))),
    }
}
