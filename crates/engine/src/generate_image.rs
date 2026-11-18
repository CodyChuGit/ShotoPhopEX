//! `generate.image`: text (and reference) to image, with its quality presets.

use std::collections::BTreeMap;
use std::sync::Arc;

use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{Document, Layer, LayerId, Size};
use photocraft_genai::template::{self, Template};
use photocraft_genai::{GenerativeBackend, Request, Task};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::jobs::JobCtx;
use crate::{EngineError, Result, Session};

use super::{
    AUTO_TEMPLATE, Common, EnhanceKind, GenerativeInfo, IMAGE, JobProgress, KLEIN_CHAT_PREFIX, KREA2_CHAT_ENCODER, MAX_SEED, MIN_SIDE, MINIMAX_CHAT_PREFIX,
    MOSS_CHAT_PREFIX, SIZE_STEP, Tuning, ULTRA_IMAGE_TEMPLATE, backend, bad, default_image_template, enhance_prompt, gen_err, loras, loras_of,
    names_other_medium, opt_bool, opt_num, opt_str, parse_variations, plan_common, quality, resize_rgba8, rgba8_at, set_generative_info, short, side,
    web_unavailable, write_rgba8,
};

/// Everything `generate.image` needs, validated before any network work.
pub(super) struct ImagePlan {
    pub(super) template: Template,
    /// The prompt the model gets (wrapped for transparency when asked).
    pub(super) prompt: String,
    /// The prompt as typed, remembered on the layer.
    pub(super) typed: String,
    pub(super) transparent: bool,
    /// The template's wording around a transparent request (`{prompt}` inside), empty otherwise;
    /// a rewritten prompt is wrapped in it the same way.
    pub(super) wrap: String,
    /// Rewrite the prompt with the vision-language model first (`enhance`), and how
    /// (`enhanceStyle`: "default" or "photo").
    pub(super) enhance: Option<EnhanceKind>,
    pub(super) negative: String,
    pub(super) seed: u64,
    pub(super) steps: u32,
    pub(super) guidance: f32,
    /// The native engine's schedule (`params.schedule`), `None` for the template's own.
    pub(super) schedule: Option<&'static str>,
    /// Krea 2 Turbo's second pass at a larger size (`params.refine`), `None` for none.
    pub(super) refine: Option<Value>,
    /// The native engine's sampler (`params.sampler`), `None` for Euler.
    pub(super) sampler: Option<&'static str>,
    pub(super) name: String,
    pub(super) models: Vec<(String, String)>,
    /// The size asked of the model (rounded to the latent grid).
    pub(super) width: u32,
    pub(super) height: u32,
    /// The size the result is brought down to under `supersample` (the asked size), `None`
    /// to keep what the model returns.
    pub(super) output: Option<(u32, u32)>,
    /// `true`: a new document of the result's size; `false`: a new layer over the whole canvas.
    pub(super) to_document: bool,
    /// How many results (consecutive seeds); only the first is visible.
    pub(super) variations: u32,
    /// Reference images (`references`).
    pub(super) refs: crate::generate_refs::Refs,
    /// LoRA adapters for the native engine (`loras`; see [`Tuning::loras`]).
    pub(super) loras: Vec<(String, f32)>,
}

/// The negative prompt Quality `best` gives Qwen-Image-2.1's text to image when none is typed:
/// what its CFG pushes away from (the community realism recipes' list, trimmed).
pub const BEST_IMAGE_NEGATIVE: &str = "blurry, low detail, deformed, oversaturated, plastic skin, watermark, text artifacts";

/// Steps and guidance a text-to-image run of `template` takes under Preferences › Quality when
/// the request leaves both at 0, or `None` for the template's own defaults. Only Qwen-Image-2.1
/// changes: under `best` 40 steps (Qwen's own recommendation) and true CFG 3 with
/// [`BEST_IMAGE_NEGATIVE`]. Measured in the app 2026-11-05 (seed 1234, four prompts, 1024²):
/// natural tones instead of the HDR look, a legible shop sign, 12.8 s against 4.7 s. More steps
/// alone (40 at CFG 1, Krea 2 at 12) looked the same as the defaults; `balanced` keeps the
/// guidance at 25 steps (to be confirmed in the app), `fast` the template's defaults; Krea 2 Turbo ignores guidance and MOSS already runs its full setting.
pub fn image_quality_tuning(template: &str, quality: &str) -> Option<(u32, f32)> {
    match (template, quality) {
        ("qwen-2.1/image", "best") => Some((40, 3.0)),
        // The middle ground: the guidance (what made the difference) at the template's 25 steps.
        ("qwen-2.1/image", "balanced") => Some((25, 3.0)),
        // Krea 2 Turbo's detail recipe ([`image_quality_schedule`]) takes one step more than
        // the template's 8: the Sick Ollie Creator Kit's 9.
        ("krea2-turbo/image", "best" | "balanced") => Some((KREA_DETAIL_STEPS, 1.0)),
        _ => None,
    }
}

/// The schedule the native engine runs Krea 2 Turbo's detail recipe under (`params.schedule`):
/// ComfyUI's `beta` scheduler over AuraFlow shift 1.25, the sampler settings of the Sick Ollie
/// Creator Kit Studio (civitai 2811445, v5; its Generation Core defaults). Against the
/// template's `simple` schedule at Krea's own shift it ends on sigmas 0.18 and 0.06 instead of
/// jumping from 0.31 to clean, so the last steps go to fine detail.
pub const KREA_DETAIL_SCHEDULE: &str = "krea-detail";

/// The kit's step count under that schedule.
pub const KREA_DETAIL_STEPS: u32 = 9;

/// The schedule a text-to-image run of `template` asks for under `quality` (`None`: the
/// template's own): Krea 2 Turbo under `best` and `balanced` runs [`KREA_DETAIL_SCHEDULE`].
/// Best runs the plain beta schedule over Krea's own shift with er_sde
/// ([`image_quality_sampler`]): the realism A/B of 2026-11-12 (6 prompts × 2 seeds in the app)
/// found er_sde gives finer skin, hair and crumb and less plastic look than euler at the same
/// time, with beta and simple hard to tell apart. Balanced too (round 5: crisper fur and wrinkles
/// than its euler krea-detail run, same 7.3 s); Fast stays on euler (no visible gain at 8 steps).
pub fn image_quality_schedule(template: &str, quality: &str) -> Option<&'static str> {
    match (template, quality) {
        ("krea2-turbo/image", "best" | "balanced") => Some("beta"),
        _ => None,
    }
}

/// The sampler a text-to-image run of `template` uses under `quality` (`None`: euler, the
/// template's): er_sde on Krea 2 Turbo under `best` and `balanced` (see [`image_quality_schedule`]).
pub fn image_quality_sampler(template: &str, quality: &str) -> Option<&'static str> {
    match (template, quality) {
        ("krea2-turbo/image", "best" | "balanced") => Some("er_sde"),
        _ => None,
    }
}

/// Pixels Krea 2 Turbo renders at under Quality `best` (1440², about 2 MP: the model is made for
/// 1-2 MP). Measured in the app 2026-11-05: visibly finer skin and texture at 100 % than 1024²,
/// about 10 s warm against 3 s.
pub const BEST_KREA_PIXELS: u64 = 1440 * 1440;

/// Pixels P-H3-Image renders at under Quality `best` (2912², 8 MP, its official target): H3 is
/// the model for large canvases such as character sheets (Cody, 2026-11-12), and came to the
/// app's 1024² default otherwise. 31.6 s on the 5090 at 28.8 GB peak (2048²: 14.6 s).
pub const BEST_H3_PIXELS: u64 = 2912 * 2912;

/// Pixels MiniMax-H3-Image renders at under Quality `best` and `ultra` (16 MP, what the model
/// is for), and under the faster qualities (4 MP, the area it was tuned on).
pub const ULTRA_PIXELS: u64 = 4096 * 4096;

pub const ULTRA_FAST_PIXELS: u64 = 2048 * 2048;

/// The same under Quality `balanced`: 1248², about 1.5 MP (to be confirmed in the app).
pub const BALANCED_KREA_PIXELS: u64 = 1248 * 1248;

/// The size a text-to-image run of `template` renders at under `quality` for an asked
/// `width × height`: Krea 2 Turbo under `best` (`balanced`) scales a smaller request up to
/// [`BEST_KREA_PIXELS`] ([`BALANCED_KREA_PIXELS`]) (same aspect, on the 16-px grid, at most 2048 a side); else the asked size.
pub fn image_quality_size(template: &str, quality: &str, width: u32, height: u32) -> (u32, u32) {
    let target = match (template, quality) {
        ("krea2-turbo/image", "best") => BEST_KREA_PIXELS,
        ("krea2-turbo/image", "balanced") => BALANCED_KREA_PIXELS,
        ("p-h3-image/image", "best") => BEST_H3_PIXELS,
        (ULTRA_IMAGE_TEMPLATE, "best") => ULTRA_PIXELS,
        (ULTRA_IMAGE_TEMPLATE, _) => ULTRA_FAST_PIXELS,
        _ => return (width, height),
    };
    let area = u64::from(width) * u64::from(height);
    if area == 0 || area >= target {
        return (width, height);
    }
    let k = (target as f64 / area as f64).sqrt();
    // H3 on its guide's grid (multiples of 32: 2912², 3360×2496, 3552×2368, 3872×2176).
    // MiniMax-H3-Image the same way, to 16 MP (9:16 is 3072×5440).
    let (most, step) = match template {
        "p-h3-image/image" => (4096, 32),
        ULTRA_IMAGE_TEMPLATE => (5472, 32),
        _ => (2048, SIZE_STEP),
    };
    let fit = |v: u32| (((f64::from(v) * k) as u32) / step * step).clamp(MIN_SIDE, most);
    (fit(width), fit(height))
}

/// How much longer a text-to-image run takes under [`image_quality_tuning`] than at the
/// template's defaults (true CFG runs the model twice a step), for the time estimate.
pub fn image_quality_cost(template: &str, quality: &str) -> f32 {
    match image_quality_tuning(template, quality) {
        Some((steps, guidance)) => {
            let base = template::find(template).map_or(25, |t| t.meta.defaults.steps.max(1));
            // 1.7, not 2: the fixed per-run costs (text encoder, decode) do not double; best measured 2.71x.
            steps as f32 / base as f32 * if guidance > 1.0 { 1.7 } else { 1.0 }
        }
        None => 1.0,
    }
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

pub(super) fn plan_image(s: &Session, p: &Value) -> Result<ImagePlan> {
    // `"template": "auto"` is the default image template, as on fill, edit and expand (the
    // 3090 benchmark's generate.image refused it, 2026-11-11).
    let without_auto;
    let p = if p.get("template").and_then(Value::as_str).is_some_and(|t| t.trim() == AUTO_TEMPLATE) {
        let mut q = p.clone();
        if let Some(o) = q.as_object_mut() {
            o.remove("template");
        }
        without_auto = q;
        &without_auto
    } else {
        p
    };
    let (default_template, default_model) = {
        let integrations = &s.prefs().integrations;
        (default_image_template(s), integrations.default_generate_model.clone())
    };
    // References with no template named: the default image template when it takes them, else
    // the reference template (its own model file, so no `model` override from the preference).
    let ref_specs = crate::generate_refs::specs(IMAGE, p)?;
    let named = p.get("template").is_some_and(|v| !v.is_null());
    let (default_template, default_model) = match template::find(&default_template) {
        Ok(t) if !named && !ref_specs.is_empty() && t.meta.references.max == 0 => (image_ref_template(s).to_string(), String::new()),
        _ => (default_template, default_model),
    };
    let Common { template, prompt, mut negative, seed, mut steps, mut guidance, models, name } =
        plan_common(s, IMAGE, p, &default_template, Task::Image, &default_model)?;
    // `schedule`: "simple" (the template's) or "krea-detail" (Krea 2 Turbo only); absent, the
    // Quality's.
    let asked_schedule = match opt_str(IMAGE, p, "schedule", 20)? {
        None => None,
        Some("simple") => Some(None),
        Some(KREA_DETAIL_SCHEDULE) if template.meta.id == "krea2-turbo/image" => Some(Some(KREA_DETAIL_SCHEDULE)),
        Some("beta") if template.meta.id == "krea2-turbo/image" => Some(Some("beta")),
        Some(other) => {
            return Err(bad(IMAGE, format!("`schedule` must be \"simple\" or, on krea2-turbo/image, \"{KREA_DETAIL_SCHEDULE}\" or \"beta\" (got `{other}`)")));
        }
    };
    let schedule = asked_schedule.unwrap_or_else(|| image_quality_schedule(&template.meta.id, quality(s)));
    // Preferences › Quality, when the request leaves the steps and guidance to the template
    // (and, for Krea 2, does not ask for its plain schedule).
    if steps == 0
        && guidance == 0.0
        && !(asked_schedule == Some(None) && template.meta.id == "krea2-turbo/image")
        && let Some((st, g)) = image_quality_tuning(&template.meta.id, quality(s))
    {
        (steps, guidance) = (st, g);
        if g > 1.0 && negative.trim().is_empty() {
            negative = BEST_IMAGE_NEGATIVE.to_string();
        }
    }
    if matches!(schedule, Some(KREA_DETAIL_SCHEDULE | "beta")) && steps == 0 {
        steps = KREA_DETAIL_STEPS;
    }
    // `sampler`: "euler" (the template's) or "er_sde" (Krea 2 Turbo only).
    let sampler = match opt_str(IMAGE, p, "sampler", 20)? {
        // Preferences › Quality, unless the request asks for Krea's plain schedule.
        None if asked_schedule == Some(None) => None,
        None => image_quality_sampler(&template.meta.id, quality(s)),
        Some("euler") => None,
        Some("er_sde") if template.meta.id == "krea2-turbo/image" => Some("er_sde"),
        Some(other) => return Err(bad(IMAGE, format!("`sampler` must be \"euler\" or, on krea2-turbo/image, \"er_sde\" (got `{other}`)"))),
    };
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
    // Quality `best` (`balanced`) renders Krea 2 at about 2 (1.5) MP: a new document keeps that size, a layer is
    // scaled to the canvas as any result of another size is.
    let (aw, ah) = (side(IMAGE, p, "width", dw)?, side(IMAGE, p, "height", dh)?);
    // `supersample`: render at that many times the asked size and bring the result down to it
    // (instead of Quality's own sizes); 1 renders exactly the asked size.
    let supersample = opt_num(IMAGE, p, "supersample", 1.0, MAX_SUPERSAMPLE)?;
    let (width, height, output) = match supersample {
        Some(k) => {
            let (w, h) = supersample_size(&template, aw, ah, k);
            (w, h, ((w, h) != (aw, ah)).then_some((aw, ah)))
        }
        None => {
            let (w, h) = image_quality_size(&template.meta.id, quality(s), aw, ah);
            (w, h, None)
        }
    };
    let refine = image_refine(p, &template.meta.id, width, height)?;
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
    // Under Quality Best the rewriter writes the prompt the way a real photo is captioned (the
    // biggest realism gain of the Best A/B, 2026-11-12); Enhance prompts off still means no
    // rewrite at all (Cody). An explicit `enhanceStyle` wins; transparent cut-outs and runs from
    // references keep the plain rewriter.
    let enhance = opt_bool(IMAGE, p, "enhance", s.prefs().integrations.enhance_prompts)?;
    let enhance = match enhance {
        false => None,
        true => Some(image_enhance_kind(IMAGE, s, &template.meta.id, &typed, transparent, !ref_specs.is_empty(), opt_str(IMAGE, p, "enhanceStyle", 20)?)?),
    };
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
        schedule,
        refine,
        sampler,
        name,
        models,
        width,
        height,
        output,
        to_document,
        variations,
        refs,
    })
}

/// Most a text-to-image run may supersample (`supersample`).
pub const MAX_SUPERSAMPLE: f64 = 4.0;

/// The size `template` renders an asked `width × height` at under `supersample` `k`: `k` times
/// each side, on the template's grid and within its longest side (the factor shrinks, same
/// aspect, when a side would pass it), never below the asked size.
pub fn supersample_size(template: &Template, width: u32, height: u32, k: f64) -> (u32, u32) {
    let sizes = &template.meta.sizes;
    let step = sizes.step.max(SIZE_STEP);
    let most = if sizes.max == 0 { 2048 } else { sizes.max };
    let long = f64::from(width.max(height).max(1));
    let k = k.min(f64::from(most) / long).max(1.0);
    let fit = |v: u32| ((f64::from(v) * k) as u32 / step * step).max(v.min(most)).max(MIN_SIDE);
    (fit(width), fit(height))
}

/// The longest side a Krea 2 refine pass redraws at.
pub const MAX_REFINE_SIDE: u32 = 2560;

/// `refine: {"scale": 1..2, "denoise": 0.05..1, "steps": 1..20}` on `generate.image` with
/// krea2-turbo/image: a second pass over the picture at `scale` times its size (on the 16-px
/// grid, at most [`MAX_REFINE_SIDE`] a side) as `params.refine` for the native engine.
pub(super) fn image_refine(p: &Value, template: &str, width: u32, height: u32) -> Result<Option<Value>> {
    let Some(r) = p.get("refine").filter(|v| !v.is_null() && **v != Value::Bool(false)) else { return Ok(None) };
    if template != "krea2-turbo/image" {
        return Err(bad(IMAGE, format!("`refine` is for krea2-turbo/image (not `{template}`)")));
    }
    let num = |k: &str, lo: f64, hi: f64| -> Result<f64> {
        r.get(k).and_then(Value::as_f64).filter(|x| (lo..=hi).contains(x)).ok_or_else(|| bad(IMAGE, format!("`refine.{k}` must be {lo}..{hi}")))
    };
    let (scale, denoise, steps) = (num("scale", 1.0, 2.0)?, num("denoise", 0.05, 1.0)?, num("steps", 1.0, 20.0)?.round() as u64);
    let long = f64::from(width.max(height));
    let k = scale.min(f64::from(MAX_REFINE_SIDE) / long).max(1.0);
    let fit = |v: u32| ((f64::from(v) * k) as u32 / SIZE_STEP * SIZE_STEP).max(SIZE_STEP);
    Ok(Some(serde_json::json!({"width": fit(width), "height": fit(height), "denoise": denoise, "steps": steps})))
}

/// Which rewriter a text-to-image idea for `template` gets: the 16 MP model its own long
/// caption (pictures from references too); under Quality Best a photo idea the photographic
/// rewriter (the biggest realism gain of the Best A/B, 2026-11-12); else the plain one. An
/// explicit `style` ("default" or "photo") wins; transparent cut-outs and runs from references
/// keep the plain rewriter. `generate.image` and the wand (`generate.enhancePrompt` with a
/// `template`) both ask this, so the prompt the wand shows is the one the run would write.
pub(crate) fn image_enhance_kind(
    command: &str,
    s: &Session,
    template: &str,
    typed: &str,
    transparent: bool,
    has_refs: bool,
    style: Option<&str>,
) -> Result<EnhanceKind> {
    let best_photo = quality(s) == "best" && !transparent && !has_refs;
    Ok(match style {
        None if template == ULTRA_IMAGE_TEMPLATE => EnhanceKind::ImageUltra,
        None if best_photo && !names_other_medium(typed) => EnhanceKind::ImagePhoto,
        None | Some("default") => EnhanceKind::Image,
        Some("photo") => EnhanceKind::ImagePhoto,
        Some(other) => return Err(bad(command, format!("`enhanceStyle` must be \"default\" or \"photo\" (got `{other}`)"))),
    })
}

/// The chat encoder a text-to-image rewrite for `template` runs on: on Krea 2, MOSS, Klein and
/// H3 the model's own, which the picture loads anyway (on MOSS the 8B swap made Turbo's 2 s
/// picture take 15 s, 2026-11-12); `None` for the default (Qwen-Image-2.1's).
pub(crate) fn own_chat_encoder(template: &str) -> Option<&str> {
    if template.starts_with("krea2-turbo/") {
        Some(KREA2_CHAT_ENCODER)
    } else if [MOSS_CHAT_PREFIX, KLEIN_CHAT_PREFIX, MINIMAX_CHAT_PREFIX].iter().any(|x| template.starts_with(x)) {
        Some(template)
    } else {
        None
    }
}

/// The prompt a `generate.image` job sends, and what the layer remembers as `enhanced`: the
/// typed idea expanded by the vision-language model when asked (then wrapped for transparency
/// as the typed one was), else the plan's prompt and an empty string.
#[allow(clippy::too_many_arguments)]
pub(super) fn image_prompt(
    backend: &dyn GenerativeBackend,
    ctx: &JobCtx,
    enhance: Option<EnhanceKind>,
    template: &str,
    typed: &str,
    planned: String,
    wrap: &str,
    seed: u64,
) -> (String, String) {
    let Some(kind) = enhance else { return (planned, String::new()) };
    ctx.progress(0.02, "Improving the prompt");
    let encoder = own_chat_encoder(template);
    match enhance_prompt(backend, kind, typed, None, encoder, seed, &JobProgress::span(ctx, 0.02, 0.05)) {
        Ok(Some(e)) => {
            let p = if wrap.is_empty() { e.clone() } else { wrap.replace("{prompt}", &e) };
            (p, e)
        }
        Ok(None) | Err(_) => (planned, String::new()),
    }
}

pub(super) fn image_enabled(_: &Session) -> std::result::Result<(), String> {
    web_unavailable()
}

pub(super) fn run_image(s: &mut Session, p: &Value) -> Result<Value> {
    let plan = plan_image(s, p)?;
    // Per template: under `auto` the native worker takes the families it serves at parity.
    let backend = backend(s)?;
    let thinker = Arc::clone(&backend);
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
        schedule,
        refine,
        sampler,
        name,
        models,
        width,
        height,
        output,
        to_document,
        variations,
        refs,
        loras,
    } = plan;
    let label = name.clone();
    let tid = template.meta.id.clone();
    let mut params: BTreeMap<String, Value> =
        Tuning { loras, ..Tuning::default() }.loras_param().map(|l| BTreeMap::from([("loras".to_string(), l)])).unwrap_or_default();
    if let Some(schedule) = schedule {
        params.insert("schedule".to_string(), Value::String(schedule.to_string()));
    }
    if let Some(refine) = refine {
        params.insert("refine".to_string(), refine);
    }
    if let Some(sampler) = sampler {
        params.insert("sampler".to_string(), Value::String(sampler.to_string()));
    }
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
    // The rewriter runs on the template's own encoder where it has one.
    let enhance_template = tid.clone();
    if to_document {
        let doc_name = name.clone();
        return crate::jobs::run(
            s,
            &label,
            false,
            move |ctx| {
                // No canvas yet: the pictures are the new document's (the Generations panel's row).
                ctx.preview_at(Rect::new(0, 0, width as i32, height as i32));
                let (prompt, enhanced) = image_prompt(thinker.as_ref(), ctx, enhance, &enhance_template, &typed, req.prompt, &wrap, seed);
                req.prompt = prompt;
                ctx.check()?;
                let resps = run_variations(backend.as_ref(), &mut req, n, seed, ctx)?;
                Ok((resps, enhanced))
            },
            move |s, (resps, enhanced): (Vec<photocraft_genai::Response>, String)| {
                // One document of the first result's size; every result a layer, the first visible.
                let first = resps.first().and_then(|r| r.images.first()).ok_or_else(|| EngineError::Other("the backend returned no image".into()))?;
                // Supersampled: the document has the asked size, the render brought down to it.
                let (w, h) = output.unwrap_or((first.width, first.height));
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
                    "renderWidth": width, "renderHeight": height,
                }))
            },
        );
    }
    crate::jobs::edit_job(
        s,
        &label,
        move |doc, active, ctx| {
            // The new layer covers the canvas.
            ctx.preview_at(doc.bounds());
            let (prompt, enhanced) = image_prompt(backend.as_ref(), ctx, enhance, &enhance_template, &typed, req.prompt, &wrap, seed);
            req.prompt = prompt;
            ctx.check()?;
            let resps = run_variations(backend.as_ref(), &mut req, n, seed, ctx)?;
            let canvas = doc.bounds();
            let (w, h) = (canvas.width(), canvas.height());
            ctx.progress(0.96, "Placing");
            let mut made = Vec::new();
            for (i, resp) in resps.into_iter().enumerate() {
                let out = resp.images.into_iter().next().ok_or_else(|| EngineError::Other("the backend returned no image".into()))?;
                let out = rgba8_at(out, w, h)?;
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
                "renderWidth": width, "renderHeight": height,
            })
        },
    )
}

/// `n` backend runs of `req` with consecutive seeds from `seed`, each its slice of the job's
/// progress.
pub(super) fn run_variations(backend: &dyn GenerativeBackend, req: &mut Request, n: u32, seed: u64, ctx: &JobCtx) -> Result<Vec<photocraft_genai::Response>> {
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        req.seed = seed.wrapping_add(u64::from(i)) % MAX_SEED;
        let (lo, hi) = (0.05 + 0.9 * i as f32 / n as f32, 0.05 + 0.9 * (i + 1) as f32 / n as f32);
        if n > 1 {
            ctx.progress(lo, &format!("Variation {}/{n}", i + 1));
        }
        out.push(backend.run(req, &JobProgress::span(ctx, lo, hi)).map_err(gen_err)?);
        ctx.check()?;
    }
    Ok(out)
}
