//! `generate.expand`: Generative Expand (outpainting), its pads, LoRA routes and stages.

use std::sync::Arc;

use photocraft_doc::{Document, LayerId, Size};
use photocraft_genai::{Rgba8, Task};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::{EngineError, Result, Session};

use super::{
    AUTO_TEMPLATE, Common, DEFAULT_EXPAND_TEMPLATE, ENHANCE_BACKDROP_REQUEST, ENHANCE_CUT_REQUEST, ENHANCE_SCENE_REQUEST, ENHANCE_SUBJECT_REQUEST, EXPAND,
    EXPAND_CAPTION, EXPAND_CAPTION_SEED, EnhanceKind, FillJob, FillPlan, JobProgress, KLEIN_FILL_TEMPLATE, MAX_REQUEST_PIXELS, MAX_VARIATIONS, Made, Outpaint,
    Tuning, align_to_grid, auto_order, backend, bad, caption_key, enhance_prompt, fill_region, gen_err, loras, made_json, matte_request_size, native21_auto,
    opt_num, opt_str, parse_edge, plan_common, rgba8_at, short, speed, web_unavailable,
};

/// What `generate.expand` asks for: how much canvas to add on each side.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Pads {
    pub(super) left: u32,
    pub(super) top: u32,
    pub(super) right: u32,
    pub(super) bottom: u32,
}

/// The prompt an expand uses when none is given and the background could not be described:
/// the background alone, no people. "Extend the scene beyond its original edges, continuing it
/// naturally" had both models paint a second copy of the person into the new area (Cody's
/// portrait, 2026-11-02).
pub(super) const EXPAND_DEFAULT_PROMPT: &str = "the same background continuing outward, empty, with no people or other subjects";

/// The prompt an expand with no prompt sends: `backdrop` (the vision-language model's
/// description of the picture's background, [`EnhanceKind::Backdrop`]) continuing in the
/// direction the canvas grows, with no people.
pub(super) fn expand_backdrop_prompt(backdrop: &str, pads: &Pads) -> String {
    let way = expand_way(pads);
    let b = strip_article(backdrop.trim().trim_end_matches(['.', '!', ';', ',']).trim());
    format!("the same {b} continuing {way}, empty, with no people or other subjects")
}

pub(super) const MAX_EXPAND_SIDE: u32 = 16_384;

/// An expand's plan.
pub(super) struct ExpandPlan {
    pub(super) fill: FillPlan,
    pub(super) pads: Pads,
    /// The grown canvas.
    pub(super) canvas: (u32, u32),
    /// The prompt was blank: the picture is described first.
    pub(super) describe: bool,
    /// How much of the picture the request takes in (see [`expand_rect`]).
    pub(super) margin: f64,
    /// Through the Outpaint LoRA ([`outpaint_lora`]): the prompt as typed (its `Scene:`
    /// line), empty when blank.
    pub(super) outpaint: Option<String>,
}

/// AusBoss's Outpaint LoRA for Qwen-Image-2.1 (https://huggingface.co/ausboss/Qwen-Image-2.1-
/// Outpaint-LoRA, Qwen Research License like the model; read 2026-11-03): trained to replace
/// flat #808080 grey around a picture with the scene's continuation, the padded canvas the edit's
/// reference at its own size, denoise 1, 25 steps at CFG 1. v2 for everyday extensions at
/// 1 to 2 MP (people and outfits), v1 for large ones from a small picture at about 1 MP.
pub(crate) const OUTPAINT_LORA: &str = "qwen-image-2.1-outpaint-v2.safetensors";

pub(crate) const OUTPAINT_LORA_WIDE: &str = "qwen-image-2.1-outpaint.safetensors";

/// Viggle's Turbo LoRA for Qwen-Image-2.1 (v0.3, six steps on its own sigmas): with it and the
/// v2 Outpaint LoRA in `loras/`, an expand under Quality `fast` runs the native worker's
/// [`OUTPAINT_TURBO_TEMPLATE`], which stacks both itself: 4 to 6 s warm against 6 to 10 s for
/// 25 steps with a blank prompt, but softer, with a fine grid in textured new areas (pines,
/// paving; Cody's PC, in the app, 2026-11-03), so the other qualities keep 25 steps.
pub(crate) const OUTPAINT_TURBO_LORA: &str = "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors";

pub(crate) const OUTPAINT_TURBO_TEMPLATE: &str = "qwen-2.1/outpaint-turbo";

/// The Mac's expand: AusBoss's Outpaint LoRA for FLUX 3 Image Klein on Klein's fill (the
/// worker's MLX route), the new area as the mask, so the picture is kept exactly and the model
/// reads it at half the side; the request at about 768². An M1 Air took 309 s at 768×512 as
/// an edit of the whole canvas, 187 s as this fill, the seam the same.
pub(crate) const KLEIN_OUTPAINT_LORA: &str = "flux-3-image-klein-outpaint.safetensors";

pub(super) const KLEIN_OUTPAINT_PIXELS: u64 = 589_824;

/// On CUDA, about 1 MP: the size AusBoss's Klein Outpaint workflow pads to.
pub(super) const KLEIN_CUDA_OUTPAINT_PIXELS: u64 = 1_048_576;

/// The 2.1 edit the plain Outpaint LoRA runs on.
pub(super) const OUTPAINT_EDIT_TEMPLATE: &str = "qwen-2.1/edit";

/// The picture's share of the grown canvas below which the v1 file goes first.
pub(super) const OUTPAINT_WIDE_SHARE: f64 = 0.3;

pub(super) const OUTPAINT_PIXELS: u64 = 1_572_864;

pub(super) const OUTPAINT_WIDE_PIXELS: u64 = 1_048_576;

/// The LoRA's trigger, its instruction, verbatim from the model card; a `Scene:` caption of
/// the whole picture follows it, as in 75 % of its training pairs.
pub(crate) const OUTPAINT_INSTRUCTION: &str =
    "Outpaint the image: replace the solid gray areas with a seamless continuation of the scene, keeping the existing picture unchanged.";

/// The Outpaint LoRA file an `auto` expand of an `old` picture to `canvas` takes, and the
/// request size it goes out at: when research-only models are allowed, the native engine runs
/// Qwen-Image-2.1's edit and the Models folder's `loras/` has the file. A backdrop-only
/// prompt on 2511 could not continue a person cut off at the picture's edge, and a prompt
/// that named her painted a second copy of her (Cody's portrait, 2026-11-03); the LoRA was
/// trained on exactly this.
/// How an expand goes through the Outpaint LoRA: the template, the LoRA the request names (none
/// when the template adds its own), and the request size.
pub(super) type OutpaintRoute = (&'static str, Option<&'static str>, u64);

pub(super) fn outpaint_lora(s: &Session, (ow, oh): (u32, u32), (nw, nh): (u32, u32)) -> Option<OutpaintRoute> {
    let root = s.prefs().integrations.models_root();
    if root.is_empty() || !loras::runs_native(s, "qwen-2.1/edit") {
        return None;
    }
    let has = |f: &str| std::path::Path::new(&root).join("loras").join(f).is_file();
    // An 8 to 16 GB card with FLUX 3 Image Klein: its fill with the Klein Outpaint LoRA, as on a
    // Mac (Qwen-Image-2.1's 14 GB DiT does not fit beside its encoder there).
    if crate::memory_profile::cuda_klein(s) && has(KLEIN_OUTPAINT_LORA) {
        return Some((KLEIN_FILL_TEMPLATE, Some(KLEIN_OUTPAINT_LORA), KLEIN_CUDA_OUTPAINT_PIXELS));
    }
    if s.prefs().integrations.allow_research_models {
        let share = (f64::from(ow) * f64::from(oh)) / (f64::from(nw) * f64::from(nh)).max(1.0);
        if let Some(route) = pick_outpaint_lora(has, share, speed(s) == "fast") {
            return Some(route);
        }
    }
    // A Mac: FLUX 3 Image Klein's fill with AusBoss's Klein Outpaint LoRA, at Klein's edit size.
    (cfg!(target_os = "macos") && has(KLEIN_OUTPAINT_LORA)).then_some((KLEIN_FILL_TEMPLATE, Some(KLEIN_OUTPAINT_LORA), KLEIN_OUTPAINT_PIXELS))
}

/// The route through the Outpaint LoRA files `has` finds: the six-step Turbo template when `fast`
/// and the Turbo LoRA and v2 are both there; else the 2.1 edit with the file for a picture that is
/// `share` of its grown canvas (v1 first below [`OUTPAINT_WIDE_SHARE`], else v2).
pub(super) fn pick_outpaint_lora(has: impl Fn(&str) -> bool, share: f64, fast: bool) -> Option<OutpaintRoute> {
    if fast && has(OUTPAINT_LORA) && has(OUTPAINT_TURBO_LORA) {
        return Some((OUTPAINT_TURBO_TEMPLATE, None, OUTPAINT_PIXELS));
    }
    let order = if share < OUTPAINT_WIDE_SHARE { [OUTPAINT_LORA_WIDE, OUTPAINT_LORA] } else { [OUTPAINT_LORA, OUTPAINT_LORA_WIDE] };
    order.into_iter().find(|f| has(f)).map(|f| (OUTPAINT_EDIT_TEMPLATE, Some(f), if f == OUTPAINT_LORA_WIDE { OUTPAINT_WIDE_PIXELS } else { OUTPAINT_PIXELS }))
}

/// The Outpaint LoRA's prompt: its instruction, then the scene.
pub(super) fn outpaint_prompt(scene: &str) -> String {
    let scene = scene.trim();
    if scene.is_empty() { OUTPAINT_INSTRUCTION.to_string() } else { format!("{OUTPAINT_INSTRUCTION} Scene: {scene}") }
}

pub(super) fn plan_expand(s: &Session, p: &Value) -> Result<ExpandPlan> {
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
    let blank = p.get("prompt").and_then(Value::as_str).is_none_or(|t| t.trim().is_empty());
    if let Some(o) = q.as_object_mut() {
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
    let lora = if auto { outpaint_lora(s, (ow, oh), (nw, nh)) } else { None };
    let Common { template, prompt, negative, seed, steps, guidance, models, name } = match lora {
        Some((t, ..)) => {
            let task = match t {
                OUTPAINT_EDIT_TEMPLATE => Task::Edit,
                KLEIN_FILL_TEMPLATE => Task::Fill,
                _ => Task::Expand,
            };
            plan_common(s, EXPAND, &q, t, task, &default_model)?
        }
        None => plan_common(s, EXPAND, &q, if auto { DEFAULT_EXPAND_TEMPLATE } else { &requested }, Task::Expand, &default_model)?,
    };
    if lora.is_some() {
        auto = false;
    }
    let variations = match opt_num(EXPAND, p, "variations", 1.0, f64::from(MAX_VARIATIONS))? {
        None => 1,
        Some(x) if x.fract() == 0.0 => x as u32,
        Some(_) => return Err(bad(EXPAND, format!("`variations` must be a whole number from 1 to {MAX_VARIATIONS}"))),
    };
    // How much of the picture goes along: the template's say (an in-context model wants all of
    // it, see `Meta::expand_margin`), else a quarter of the longer side.
    // The Outpaint LoRA sees the whole canvas.
    let margin = if lora.is_some() {
        1.0
    } else {
        opt_num(EXPAND, p, "margin", 0.0, 1.0)?.unwrap_or_else(|| template.meta.expand_margin.map_or(0.25, |m| f64::from(m.clamp(0.0, 1.0))))
    };
    let edge = parse_edge(EXPAND, p)?;
    let rect = expand_rect((ow, oh), &pads, margin);
    let pixels = u64::from(rect.width()) * u64::from(rect.height());
    if pixels > MAX_REQUEST_PIXELS {
        return Err(EngineError::Other(format!(
            "the expanded area with its margin is {}×{} pixels; Generative Expand handles up to 16 megapixels at a time: add less canvas or lower `margin`",
            rect.width(),
            rect.height()
        )));
    }
    let name = name.unwrap_or_else(|| "Generative Expand".to_string());
    let tuning = match lora {
        // Klein's Outpaint LoRA at 1.5, as AusBoss's Klein workflow runs it; the Qwen files at 1.
        Some((_, file, max_pixels)) => Tuning {
            loras: file.map(|f| (f.to_string(), if f == KLEIN_OUTPAINT_LORA { 1.5 } else { 1.0 })).into_iter().collect(),
            max_pixels,
            ..Tuning::default()
        },
        None => Tuning::default(),
    };
    let outpaint = lora.map(|_| if blank { String::new() } else { p.get("prompt").and_then(Value::as_str).unwrap_or_default().trim().to_string() });
    // An expand keeps its wording: its prompt continues the scene, the rewriter's rules
    // (name the subject, say what stays) are about a picture's content.
    Ok(ExpandPlan {
        fill: FillPlan {
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
            tuning,
        },
        pads,
        canvas: (nw, nh),
        describe: blank,
        margin,
        outpaint,
    })
}

/// An expand's request on its grown canvas: the bounding box of the added canvas (an L or a
/// frame when more than one side grows) plus `margin` of the picture next to it.
pub(super) fn expand_rect((ow, oh): (u32, u32), pads: &Pads, margin: f64) -> Rect {
    let (nw, nh) = (ow + pads.left + pads.right, oh + pads.top + pads.bottom);
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
    align_to_grid(added.intersect(&canvas), canvas)
}

/// The sides of a picture whose edge cuts its main subject off, from the vision-language
/// model's answer ("bottom, left"); "not cut off" or anything else is none.
pub(super) fn cut_sides(said: &str) -> Pads {
    let t = said.to_ascii_lowercase();
    let has = |w: &str| u32::from(t.split(|c: char| !c.is_ascii_alphabetic()).any(|x| x == w));
    Pads { left: has("left"), top: has("top"), right: has("right"), bottom: has("bottom") }
}

/// A blank expand's two stages when the subject runs off the picture: first every side but
/// those facing away from the cut (its body continues there), then those (background alone).
/// A side faces away from the cut when the picture does not cut the subject off there but does
/// on the opposite side: above a portrait cut at the chest. Told to continue the person on
/// every side at once, 2511 drew a second copy of her in the top band; told to paint background
/// alone, it cut her off at the old edge; bottom, left and right with her body first, then the
/// top as backdrop, continued her whole with no copy (Cody's portrait, in the app, 2026-11-03).
/// `None` when no side that grows cuts the subject off.
pub(super) fn expand_stages(pads: &Pads, cut: &Pads) -> Option<(Pads, Pads)> {
    let grows_cut =
        (pads.left > 0 && cut.left > 0) || (pads.top > 0 && cut.top > 0) || (pads.right > 0 && cut.right > 0) || (pads.bottom > 0 && cut.bottom > 0);
    if !grows_cut {
        return None;
    }
    let later = |p: u32, here: u32, opposite: u32| if here == 0 && opposite > 0 { p } else { 0 };
    let rest = Pads {
        left: later(pads.left, cut.left, cut.right),
        top: later(pads.top, cut.top, cut.bottom),
        right: later(pads.right, cut.right, cut.left),
        bottom: later(pads.bottom, cut.bottom, cut.top),
    };
    let first = Pads { left: pads.left - rest.left, top: pads.top - rest.top, right: pads.right - rest.right, bottom: pads.bottom - rest.bottom };
    Some((first, rest))
}

/// The prompt of an expand's first stage: `subject` (the vision-language model's name for it)
/// continuing past the cut edges in front of `backdrop`, with no one else.
pub(super) fn expand_subject_prompt(subject: &str, backdrop: Option<&str>, pads: &Pads) -> String {
    let way = expand_way(pads);
    let s = strip_article(subject.trim().trim_end_matches(['.', '!', ';', ',']).trim());
    let behind =
        backdrop.map(|b| format!(" in front of the same {}", strip_article(b.trim().trim_end_matches(['.', '!', ';', ',']).trim()))).unwrap_or_default();
    format!("the same {s} continuing {way} naturally past the edge, the body and clothing completed{behind}, with no additional people")
}

pub(super) fn strip_article(b: &str) -> &str {
    ["the ", "The ", "a ", "A ", "an ", "An "].iter().find_map(|a| b.strip_prefix(a)).unwrap_or(b)
}

/// Which way an expand grows, in words.
pub(super) fn expand_way(pads: &Pads) -> &'static str {
    match (pads.left > 0, pads.top > 0, pads.right > 0, pads.bottom > 0) {
        (false, true, false, false) => "upward",
        (false, false, false, true) => "downward",
        (true, false, false, false) => "to the left",
        (false, false, true, false) => "to the right",
        _ => "outward",
    }
}

/// The first stage's layers folded into each of the last stage's (its variations), so the
/// result is one layer per variation as after one stage; the first stage's layers go.
pub(super) fn fold_expand_stages(doc: &mut Document, first: &[LayerId], made: &Made) -> Result<()> {
    let shown = |doc: &Document, id: &LayerId| {
        doc.layer(*id).cloned().map(|mut l| {
            l.visible = true;
            l
        })
    };
    let Some(mut under) = first.first().and_then(|id| shown(doc, id)) else { return Ok(()) };
    for id in first.iter().skip(1) {
        if let Some(l) = shown(doc, id) {
            under = crate::pixels::merge_down(doc, &under, &l);
        }
    }
    for (id, ..) in made {
        let Some(top) = shown(doc, id) else { continue };
        let merged = crate::pixels::merge_down(doc, &under, &top);
        let l = doc.layer_mut(*id).ok_or(EngineError::NoLayer(*id))?;
        l.content = merged.content;
        l.mask = None;
    }
    for id in first {
        doc.remove(*id);
    }
    Ok(())
}

pub(super) fn expand_enabled(s: &Session) -> std::result::Result<(), String> {
    web_unavailable()?;
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// `generate.expand`: grow the canvas and have the model paint the new area, as one undo step.
pub(super) fn run_expand(s: &mut Session, p: &Value) -> Result<Value> {
    let ExpandPlan { fill: plan, pads, canvas: (nw, nh), describe, margin, outpaint } = plan_expand(s, p)?;
    let lora = outpaint.is_some();
    let backend = backend(s)?;
    let FillPlan { template, prompt, negative, seed, steps, guidance, name, models, rect: _, variations, auto, edge, enhance, refs, grow, erase, tuning } =
        plan;
    let label = name.clone();
    let template_id = template.meta.id.clone();
    let auto_order = auto_order(EXPAND, speed(s));
    let thinker = Arc::clone(&backend);
    // One job per stage: the first (when there are two) makes one picture, the last the
    // variations.
    let job = move |prompt: String, variations: u32| FillJob {
        thinker: Arc::clone(&backend),
        backend: Arc::clone(&backend),
        command: EXPAND,
        template_id: template_id.clone(),
        auto,
        auto_order,
        prompt,
        negative: negative.clone(),
        seed,
        steps,
        guidance,
        name: name.clone(),
        models: models.clone(),
        variations,
        edge,
        enhance,
        refs: refs.clone(),
        grow,
        erase,
        rewritten: None,
        removal: false,
        tone_most: 1.0,
        prefill: false,
        tuning: tuning.clone(),
        object: None,
    };
    crate::jobs::edit_job(
        s,
        &label,
        move |doc, active, ctx| {
            // Canvas Size with a transparent extension: the picture moves by the left/top pads.
            let (ow, oh) = (doc.size.width, doc.size.height);
            // No prompt: the vision-language model describes the background, and the model is
            // asked for that alone (else it paints the subject again into the new area). Where
            // the picture cuts its subject off, those sides go first, asked to continue it.
            let mut stages: Vec<(Pads, String)> = vec![(pads, prompt.clone())];
            if let Some(typed) = &outpaint {
                // Through the Outpaint LoRA: one run over every side, the scene (subject and
                // setting) described when no prompt was given.
                let scene = if typed.is_empty() {
                    let seen = photocraft_compose::render(doc, doc.bounds()).to_rgba8();
                    let seen = Rgba8::new(seen.width, seen.height, seen.pixels).map_err(gen_err)?;
                    let (rw, rh) = matte_request_size(seen.width, seen.height);
                    let seen = rgba8_at(seen, rw, rh)?;
                    let key = caption_key(&seen);
                    let cached = EXPAND_CAPTION.lock().unwrap_or_else(|e| e.into_inner()).as_ref().filter(|(k, _)| *k == key).map(|(_, c)| c.clone());
                    if let Some(caption) = cached {
                        caption
                    } else {
                        ctx.progress(0.0, "Describing the picture");
                        let caption = enhance_prompt(
                            thinker.as_ref(),
                            EnhanceKind::Scene,
                            ENHANCE_SCENE_REQUEST,
                            Some(&seen),
                            None,
                            EXPAND_CAPTION_SEED,
                            &JobProgress::span(ctx, 0.0, 0.03),
                        )
                        .ok()
                        .flatten()
                        .unwrap_or_default();
                        if !caption.is_empty() {
                            *EXPAND_CAPTION.lock().unwrap_or_else(|e| e.into_inner()) = Some((key, caption.clone()));
                        }
                        caption
                    }
                } else {
                    typed.clone()
                };
                stages = vec![(pads, outpaint_prompt(&scene))];
            } else if describe {
                let seen = photocraft_compose::render(doc, doc.bounds()).to_rgba8();
                let seen = Rgba8::new(seen.width, seen.height, seen.pixels).map_err(gen_err)?;
                let (rw, rh) = matte_request_size(seen.width, seen.height);
                let seen = rgba8_at(seen, rw, rh)?;
                let ask = |kind: EnhanceKind, request: &str, lo: f32, hi: f32| {
                    enhance_prompt(thinker.as_ref(), kind, request, Some(&seen), None, seed, &JobProgress::span(ctx, lo, hi)).ok().flatten()
                };
                ctx.progress(0.0, "Describing the picture");
                let backdrop = ask(EnhanceKind::Backdrop, ENHANCE_BACKDROP_REQUEST, 0.0, 0.02);
                let cut = ask(EnhanceKind::CutEdges, ENHANCE_CUT_REQUEST, 0.02, 0.03).map(|t| cut_sides(&t)).unwrap_or_default();
                let split = expand_stages(&pads, &cut);
                let backdrop_prompt = |p: &Pads| backdrop.as_deref().map_or_else(|| EXPAND_DEFAULT_PROMPT.to_string(), |b| expand_backdrop_prompt(b, p));
                let subject = if split.is_some() { ask(EnhanceKind::Subject, ENHANCE_SUBJECT_REQUEST, 0.03, 0.04) } else { None };
                stages = match (subject, split) {
                    (Some(subject), Some((first, rest))) => {
                        let mut v = vec![(first, expand_subject_prompt(&subject, backdrop.as_deref(), &first))];
                        if rest != Pads::default() {
                            v.push((rest, backdrop_prompt(&rest)));
                        }
                        v
                    }
                    _ => vec![(pads, backdrop_prompt(&pads))],
                };
            }
            let count = stages.len();
            let (mut shift_x, mut shift_y) = (0i32, 0i32);
            let mut first_layers: Vec<LayerId> = Vec::new();
            let mut first_ms = 0u64;
            let mut last = None;
            for (i, (stage, stage_prompt)) in stages.into_iter().enumerate() {
                let final_stage = i + 1 == count;
                let (cw, ch) = (doc.size.width, doc.size.height);
                let rect = expand_rect((cw, ch), &stage, margin);
                shift_x += stage.left as i32;
                shift_y += stage.top as i32;
                if count > 1 {
                    ctx.progress(0.0, if i == 0 { "Continuing the subject" } else { "Extending the background" });
                }
                // The document on screen keeps its size and origin until the result lands: the
                // pictures of the run go on the request area moved back by the pads so far.
                ctx.preview_at(rect.translate(-shift_x, -shift_y));
                crate::image_cmds::translate_doc(doc, EXPAND, stage.left as i32, stage.top as i32)?;
                doc.size = Size::new(cw + stage.left + stage.right, ch + stage.top + stage.bottom);
                crate::canvas_geom::refresh(doc, crate::canvas_geom::Refresh::Shapes);
                let old = Rect::new(stage.left as i32, stage.top as i32, (stage.left + cw) as i32, (stage.top + ch) as i32);
                // Everything in the request that is not the picture so far is to be painted.
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
                let filled = fill_region(
                    doc,
                    active,
                    ctx,
                    job(stage_prompt, if final_stage { variations } else { 1 }),
                    rect,
                    coverage,
                    Some(Outpaint { picture: old.intersect(&rect), lora }),
                )?;
                if final_stage {
                    last = Some(filled);
                } else {
                    first_ms += filled.0.iter().map(|m| m.3).sum::<u64>();
                    first_layers.extend(filled.0.iter().map(|m| m.0));
                }
            }
            let (mut made, rw, rh, request, template_id) = last.ok_or_else(|| EngineError::Other("internal error: an expand without a stage".into()))?;
            fold_expand_stages(doc, &first_layers, &made)?;
            for m in &mut made {
                m.3 += first_ms;
            }
            let old = Rect::new(pads.left as i32, pads.top as i32, (pads.left + ow) as i32, (pads.top + oh) as i32);
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
