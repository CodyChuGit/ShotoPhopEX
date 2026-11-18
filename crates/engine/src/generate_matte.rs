//! `generate.removeBackground`: a subject matte from the native worker, as a layer mask or a
//! selection.

use std::collections::BTreeMap;
use std::sync::Arc;

use photocraft_algo::selection::{self as sel, SelectionMode};
use photocraft_color::PixelFormat;
use photocraft_doc::{LayerId, LayerMask};
use photocraft_genai::{Gray8, Request, Rgba8, Task, template};
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::layer_param;
use crate::{EngineError, Result, Session};

use super::{
    AUTO_TEMPLATE, Common, ENHANCE_SUBJECT_REQUEST, EnhanceKind, JobProgress, MAX_PROMPT_CHARS, backend, bad, enhance_prompt, fit_pixels, gen_err, gray8_at,
    opt_bool, opt_str, plan_common, resize_rgba8, resolve_auto, web_unavailable,
};

pub(crate) const REMOVE_BG: &str = "generate.removeBackground";

pub(crate) const DEFAULT_MATTE_TEMPLATE: &str = "qwen-2.1/matte";

/// ComfyUI's official background-removal instruction for Qwen-Image-2.1, sent when the prompt
/// does not name a subject (it opens with an imperative verb, so the template passes it through).
pub(crate) const MATTE_DEFAULT_PROMPT: &str = "Remove the background, and output a PNG image";

/// A matte request goes out at up to this many pixels: the model's own working size, and the
/// matte only needs to be resampled back, not the pixels.
pub(super) const MAX_MATTE_REQUEST_PIXELS: u64 = 1024 * 1024;

/// Everything `generate.removeBackground` needs, validated before any network work.
pub(super) struct MattePlan {
    pub(super) common: Common,
    pub(super) layer: LayerId,
    /// Load the matte as the selection (combined by `mode`) instead of masking the layer.
    pub(super) as_selection: bool,
    pub(super) mode: SelectionMode,
    /// Send the composite rather than the layer alone.
    pub(super) all_layers: bool,
    /// What to keep, as typed (empty = the model's own reading of the subject).
    pub(super) subject: String,
    /// `template: auto`: the job may still fall back from the matte model to the detector.
    pub(super) auto: bool,
    /// The planned template is a detector (`Task::Segment`), not a matte model.
    pub(super) segment: bool,
    /// Soften a detector's hard mask with the classical edge refinement (as the Quick Action
    /// does after Select Subject).
    pub(super) refine: bool,
}

/// What the detector is asked for when Remove Background has no subject named.
pub(super) const SUBJECT_PROMPT: &str = "the main subject";

pub(super) fn plan_remove_bg(s: &Session, p: &Value) -> Result<MattePlan> {
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

pub(super) fn remove_bg_enabled(s: &Session) -> std::result::Result<(), String> {
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
pub(super) fn run_remove_bg(s: &mut Session, p: &Value) -> Result<Value> {
    let plan = plan_remove_bg(s, p)?;
    let backend = backend(s)?;
    let thinker = Arc::clone(&backend);
    let MattePlan { common, layer: id, as_selection, mode, all_layers, subject, auto, segment, refine } = plan;
    let Common { template, negative, seed, steps, guidance, models, .. } = common;
    let template_id = template.meta.id.clone();
    let label = if as_selection { "Select Subject (Generative)" } else { "Remove Background (Generative)" };
    // The matte model picks its own subject; the detector, when set up, says which thing it is
    // (see `matte_guide`). Not on the 8 GB Mac profile (SAM beside a model is over its budget).
    let mac8 = crate::memory_profile::active(s) == Some(crate::memory_profile::Profile::Mac8);
    let finder = if segment || mac8 { None } else { crate::select_ml_cmds::subject_finder(s, None).ok() };
    crate::jobs::edit_job(
        s,
        label,
        move |doc, _active, ctx| {
            ctx.progress(0.0, "Rendering");
            // The route: the matte model when it is wanted and the server has it, else the
            // detector (its masks are hard-edged, its licence permissive).
            // With ViTMatte served, the detector and ViTMatte (hair strand by strand, in about a
            // second) come before the matte model.
            let vit = auto
                && resolve_auto(backend.as_ref(), &[crate::select_ml_cmds::VITMATTE_TEMPLATE], "") == crate::select_ml_cmds::VITMATTE_TEMPLATE
                && resolve_auto(backend.as_ref(), &[crate::select_ml_cmds::DEFAULT_SEGMENT_TEMPLATE], "") == crate::select_ml_cmds::DEFAULT_SEGMENT_TEMPLATE;
            let (template_id, segment, models) = if vit {
                (crate::select_ml_cmds::DEFAULT_SEGMENT_TEMPLATE.to_string(), true, Vec::new())
            } else if auto && !segment {
                let t = resolve_auto(backend.as_ref(), &[DEFAULT_MATTE_TEMPLATE], crate::select_ml_cmds::DEFAULT_SEGMENT_TEMPLATE);
                let seg = t != DEFAULT_MATTE_TEMPLATE;
                (t, seg, if seg { Vec::new() } else { models })
            } else {
                (template_id, segment, models)
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
            let picture = if finder.is_some() { rgba8.clone() } else { Vec::new() };
            let image = Rgba8::new(w, h, rgba8).map_err(gen_err)?;
            let (rw, rh) = if segment { fit_pixels(w, h, MAX_SEGMENT_REQUEST_PIXELS) } else { matte_request_size(w, h) };
            let image = if (rw, rh) == (w, h) { image } else { resize_rgba8(&image, rw, rh)? };
            // No subject named: the vision-language model names it first. 2.1's matte with its
            // generic instruction kept nothing on 3 of 5 seeds through the app (2026-10-28) and
            // on 0 of 3 once "the man" was named; the generic instruction stays the fallback.
            let named = if !segment && subject.is_empty() {
                let said = enhance_prompt(
                    thinker.as_ref(),
                    EnhanceKind::Subject,
                    ENHANCE_SUBJECT_REQUEST,
                    Some(&image),
                    None,
                    seed,
                    &JobProgress::span(ctx, 0.0, 0.05),
                );
                said.ok().flatten().map(|n| n.trim_end_matches(['.', '!']).trim().to_string())
            } else {
                None
            };
            let guide_name = if subject.is_empty() { named.clone() } else { Some(subject.clone()) };
            let prompt = match (segment, subject.is_empty(), named) {
                (true, true, _) => SUBJECT_PROMPT.to_string(),
                (false, true, Some(n)) => n,
                (false, true, None) => MATTE_DEFAULT_PROMPT.to_string(),
                (_, false, _) => subject.clone(),
            };
            let mut params = BTreeMap::new();
            if segment {
                params.insert("threshold".to_string(), json!(0.5));
            }
            let mut req = Request {
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
            // No subject named and none found: the kinds of thing Select Subject tries next (a
            // kitchen has no "main subject" but has objects; the matte template, Windows' default,
            // never comes back empty). Still nothing: say where to name one.
            let fallbacks: &[&str] = if segment && subject.is_empty() { &crate::select_ml_cmds::SUBJECT_FALLBACKS[1..] } else { &[] };
            let mut next = fallbacks.iter();
            let resp = loop {
                match backend.as_ref().run(&req, &JobProgress::new(ctx)) {
                    Ok(r) => break r,
                    Err(photocraft_genai::Error::NoOutput(_)) if segment => match next.next() {
                        Some(p) => {
                            ctx.check()?;
                            req.prompt = (*p).to_string();
                        }
                        None => {
                            return Err(EngineError::Other(
                                "no subject found: type what to keep (e.g. \"the fridge\") in Remove Background's Subject box".into(),
                            ));
                        }
                    },
                    Err(e) => return Err(gen_err(e)),
                }
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
                let plane = gray8_at(plane, w, h)?;
                for (c, v) in cov.iter_mut().zip(&plane.data) {
                    *c = c.max(f32::from(*v) / 255.0);
                }
                if !segment {
                    break;
                }
            }
            if !segment {
                super::matte_guide::guide_and_clean(&mut cov, finder.as_ref(), &picture, area, guide_name.as_deref(), ctx)?;
            }
            // A detector's mask is hard-edged: the classical edge refinement (a narrow smart
            // radius, a little smoothing and feather, as after Select Subject) gives hair and
            // soft edges partial coverage from the pixels themselves.
            let mut matted = false;
            if segment && refine {
                ctx.progress(0.9, "Refining edge");
                let (bounds, found) = crate::select_ml_cmds::coverage_bounds(&cov, area);
                if found > 0 {
                    let picture = crate::select_ml_cmds::picture_rgba(doc, Some(id), all_layers);
                    if let Some(mut a) = crate::select_ml_cmds::vitmatte_alphas(backend.as_ref(), &picture, area, &[(cov.clone(), bounds, found)], None, ctx)? {
                        cov = a.swap_remove(0);
                        matted = true;
                    }
                }
                if !matted {
                    cov = crate::select_ml_cmds::refine_coverage(doc, Some(id), all_layers, cov, area);
                }
                crate::select_ml_cmds::despeckle_alpha(&mut cov, area, None);
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
            Ok((bounds, count, resp.seed, resp.run_id, resp.elapsed_ms, resp.timings, (rw, rh), (w, h), (template_id, matted)))
        },
        move |(b, count, seed, run_id, ms, timings, (rw, rh), (w, h), (tid, matted))| {
            json!({
                "layer": id.0, "selection": as_selection, "bounds": [b.x0, b.y0, b.width(), b.height()], "pixels": count, "vitmatte": matted,
                "seed": seed, "template": tid, "runId": run_id, "width": w, "height": h, "requestWidth": rw, "requestHeight": rh,
                "ms": ms, "timings": [serde_json::to_value(timings).unwrap_or(Value::Null)],
            })
        },
    )
}

/// A detector request is sent at most this large (SAM 3.1 works at about 1 megapixel inside).
pub(super) const MAX_SEGMENT_REQUEST_PIXELS: u64 = 2048 * 1024;
