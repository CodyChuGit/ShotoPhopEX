//! `generate.fill`: Generative Fill, from the selection's plan to the region the model paints
//! and the layer it lands in.

use std::collections::BTreeMap;
use std::sync::Arc;

use photocraft_color::PixelFormat;
use photocraft_doc::{Document, Layer, LayerId, LayerMask};
use photocraft_genai::template::{self, Template};
use photocraft_genai::{GenerativeBackend, Gray8, Request, Rgba8, Task};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::jobs::JobCtx;
use crate::{EngineError, Result, Session};

use super::{
    AUTO_TEMPLATE, Common, DEFAULT_FILL_TEMPLATE, DEFAULT_SAMPLER, DEFAULT_SCHEDULER, EDIT, EDIT_BAND_CHANGE, EDIT_BAND_MAX_SHARE, Edge, EnhanceKind,
    FAST_DRAFT_STEP_CACHE, FILL, GenerativeInfo, HIGHLIGHT_GROW, HIGHLIGHT_REMOVE_PROMPT, JobProgress, KLEIN_FILL_TEMPLATE, MAX_GROW, MAX_SEED, MAX_VARIATIONS,
    NATIVE21_FILL, PRESENCE_CHECK, REMOVE_PROMPT, SHADOW_CHANGE, SHADOW_MAX_SHARE, Tuning, auto_order, backend, bad, change_band, dilate, dither_edge,
    enhance_prompt, erase_covered, fast_draft_applies, feather_outward, feather_radius, fill_rect, fill_ref_template, gen_err, gray8_at, grey_outside,
    grey_where, judge_pair, krea2_fill, named_object, naming_picture, native_fill_order, native21_auto, native21_fill_fallback, opt_bool, opt_num, opt_str,
    outpaint_feather_radius, parse_edge, place_in_words, plan_common, prefill_outside, quality, realigned, removal_adapter, request_size_within, resize_gray8,
    resize_rgba8, resolve_auto, rgba8_at, set_generative_info, short, side_by_side, smooth_covered, still_there, template_or, tone_match_removal,
    web_unavailable, write_rgba8,
};

pub(super) fn fill_enabled(s: &Session) -> std::result::Result<(), String> {
    web_unavailable()?;
    let d = s.active().ok_or("no document open")?;
    if d.doc.selection.is_none() {
        return Err("make a selection first".into());
    }
    Ok(())
}

/// Everything `generate.fill` needs, validated before any pixel or network work.
pub(super) struct FillPlan {
    pub(super) template: Template,
    pub(super) prompt: String,
    pub(super) negative: String,
    pub(super) seed: u64,
    pub(super) steps: u32,
    pub(super) guidance: f32,
    pub(super) name: String,
    pub(super) models: Vec<(String, String)>,
    /// The area sent to the backend: the selection's bounds grown by the margin, on the canvas.
    pub(super) rect: Rect,
    /// How many results to make (consecutive seeds); only the first is visible.
    pub(super) variations: u32,
    /// Pick the template inside the job from what the server has (`AUTO_FILL_ORDER`).
    pub(super) auto: bool,
    /// How the result's layer mask meets the surroundings.
    pub(super) edge: Edge,
    /// Rewrite the prompt with the vision-language model first (`enhance`, default from the
    /// `enhancePrompts` preference).
    pub(super) enhance: bool,
    /// Reference images (`references`: edits, and fills from a reference).
    pub(super) refs: crate::generate_refs::Refs,
    /// Widen the selection by this many pixels before the request (`grow`), so a bulkier
    /// replacement has room and the old object's edge is inside the repainted area.
    pub(super) grow: u32,
    /// Flatten the selection to mid-grey in the request image (`erase`), so the model paints a
    /// hole instead of copying what was there.
    pub(super) erase: bool,
    /// The sampler, scheduler and request size asked for ([`Tuning`]).
    pub(super) tuning: Tuning,
}

pub(super) fn plan_fill(s: &Session, p: &Value) -> Result<FillPlan> {
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
    if auto
        && ref_specs.is_empty()
        && let Some(t) = native21_auto(s, FILL, false)
    {
        as_edit = opt_bool(FILL, p, "fillAsEdit", false)?;
        base = if as_edit { native21_auto(s, EDIT, false).unwrap_or(t).to_string() } else { t.to_string() };
        // A prompted fill still picks inside the job: Krea 2 AnyPaint when the worker serves it
        // and the prompt is rewritten or blank ([`native_fill_order`]), else this 2.1 fill.
        auto = !as_edit;
    }
    // A fill from references on the native worker: the 2.1 edit template, which takes nine
    // references beside the picture, instead of 2511's fill-ref on ComfyUI (the Rust rebuild,
    // `goals/rust-rebuild.md` R2): "Replace the <thing> with <prompt> as image 2 shows it".
    if auto
        && !ref_specs.is_empty()
        && native21_auto(s, FILL, false).is_some()
        && let Some(t) = native21_auto(s, EDIT, false)
    {
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
    let tuning = Tuning { fast_draft: s.prefs().integrations.fast_draft, ..Tuning::from_params(FILL, p)? };
    Ok(FillPlan { template, prompt, negative, seed, steps, guidance, name, models, rect, variations, auto, edge, enhance, refs, grow, erase, tuning })
}

/// What a fill or expand job needs once its plan is made and the backend is in hand.
#[derive(Clone)]
pub(super) struct FillJob {
    pub(super) backend: Arc<dyn GenerativeBackend>,
    /// The same native worker answers vision-language questions: rewriting, judging and
    /// checking whether an object is still present.
    pub(super) thinker: Arc<dyn GenerativeBackend>,
    /// The command the layers remember ([`GenerativeInfo::command`]).
    pub(super) command: &'static str,
    pub(super) template_id: String,
    pub(super) auto: bool,
    /// The candidates `auto` picks from ([`AUTO_FILL_ORDER`] or [`AUTO_EXPAND_ORDER`]).
    pub(super) auto_order: &'static [&'static str],
    pub(super) prompt: String,
    pub(super) negative: String,
    pub(super) seed: u64,
    pub(super) steps: u32,
    pub(super) guidance: f32,
    pub(super) name: String,
    pub(super) models: Vec<(String, String)>,
    pub(super) variations: u32,
    pub(super) edge: Edge,
    /// Rewrite the prompt with the vision-language model first (`enhance`).
    pub(super) enhance: bool,
    /// Reference images (edits, fills from a reference).
    pub(super) refs: crate::generate_refs::Refs,
    /// Widen the selection by this many pixels first (fills).
    pub(super) grow: u32,
    /// Flatten the selection to grey in the request image (fills).
    pub(super) erase: bool,
    /// The prompt already rewritten by the vision-language model (a removal names its object
    /// before the request, so it can ask the detector for it by name); the fill takes it as
    /// the rewritten prompt instead of rewriting again.
    pub(super) rewritten: Option<String>,
    /// A removal: the result's tone matched to the picture's at the hole's edge
    /// ([`tone_matched`]; a crow taken off a sky came back as a lighter patch of sky), and of
    /// several variations the one whose fill matches its edge best shown first
    /// ([`edge_mismatch`]; wood over a wrist, a wall over eyes, blue on a grey sky all lose
    /// to a seed that continued the edge). Measured 2026-10-21.
    pub(super) removal: bool,
    /// The most that tone match may move a channel, 0..1: the Remove tools match all the way
    /// (1.0; a removal wants its hole to look like its edge); a split's fills move at most
    /// [`TONE_MATCH_MAX`] levels, so its checks judge what the fill painted (the real fills
    /// come back a few levels off, well inside it).
    pub(super) tone_most: f32,
    /// Smooth the hole over in the picture the model sees ([`smooth_covered`]: a membrane
    /// from its edge, no structure), so an edit model that copies what it sees has nothing to
    /// copy; `prefill` on the removal commands, on by default: on the stock photographs it
    /// took the removals from 27 to 35 of 48 and cost nothing on the generated set
    /// (`benchmarks.md` §5i, 2026-10-21).
    pub(super) prefill: bool,
    /// The sampler, scheduler and request size asked for ([`Tuning`]).
    pub(super) tuning: Tuning,
    /// A removal: what the rewriter named ("gold necklace"), for the judge.
    pub(super) object: Option<String>,
}

/// The layers a job made: (id, seed, run id, ms, timings), the primary one first.
pub(super) type Made = Vec<(LayerId, u64, String, u64, photocraft_genai::Timings)>;

/// A job's outcome: the layers, the rect's size, the request's size and the template used.
pub(super) type Filled = (Made, u32, u32, (u32, u32), String);

/// Regenerate the pixels `coverage` marks (one value per pixel of `rect`) through the backend
/// Generative Expand's extras for [`fill_region`]: the part of the request that holds the
/// picture (the rest is empty canvas, painted mid-grey so the model sees "nothing here yet"
/// rather than black or transparent), and a wider blend: the result layer's mask is the soft
/// request mask, so the band of the picture the model re-rendered fades into the original
/// instead of meeting it at a hard line (the classic outpainting seam).
pub(super) struct Outpaint {
    pub(super) picture: Rect,
    /// The Outpaint LoRA's canvas: the new area flat grey (#808080, as it was trained) and the
    /// whole canvas the edit's reference, so no reference crop goes along.
    pub(super) lora: bool,
}

/// Regenerate the pixels `coverage` marks (one value per pixel of `rect`) through the backend
/// and add the results as layers above the active one, each masked to `coverage` (or to the
/// soft request mask when `outpaint` is set).
/// What the selection holds, named by the rewriter from the removal's two views (its
/// surroundings and a close-up, [`naming_views`]): "denim jacket". `None` when nothing usable
/// came back or the rewriter could not run.
pub(super) fn name_selected(doc: &Document, job: &FillJob, area: Rect, coverage: &[f32], ctx: &JobCtx) -> Result<Option<String>> {
    ctx.progress(0.0, "Naming what is selected");
    let (hole, _) = crate::select_ml_cmds::coverage_bounds(coverage, area);
    if hole.is_empty() {
        return Ok(None);
    }
    let (seen, place) = naming_picture(doc, hole, coverage, area)?;
    let said =
        enhance_prompt(job.thinker.as_ref(), EnhanceKind::Remove, REMOVE_PROMPT, Some(&seen), Some(&place), job.seed, &JobProgress::span(ctx, 0.01, 0.04));
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

/// The request's sampler, scheduler, LoRAs and denoise, and for an expand the picture's
/// rectangle inside the request (in the request's `(rw, rh)` pixels) and the canvas size.
pub(super) fn request_params(tuning: &Tuning, outpaint: Option<&Outpaint>, rect: Rect, (w, h): (u32, u32), (rw, rh): (u32, u32)) -> BTreeMap<String, Value> {
    let mut params = BTreeMap::new();
    params.insert("sampler".to_string(), Value::String(tuning.sampler.clone().unwrap_or_else(|| DEFAULT_SAMPLER.to_string())));
    params.insert("scheduler".to_string(), Value::String(tuning.scheduler.clone().unwrap_or_else(|| DEFAULT_SCHEDULER.to_string())));
    if let Some(l) = tuning.loras_param() {
        params.insert("loras".to_string(), l);
    }
    if let Some(d) = tuning.denoise {
        params.insert("denoise".to_string(), json!(d));
    }
    params.extend(tuning.extra.iter().map(|(k, v)| (k.clone(), v.clone())));
    if let Some(o) = outpaint.filter(|o| !o.lora) {
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
    params
}

/// A fill's or edit's prompt rewritten by the vision-language model with the picture in view
/// (`fill_region`'s `enhance`); the prompt as typed when the rewriter fails.
#[allow(clippy::too_many_arguments)]
pub(super) fn rewritten_fill_prompt(
    thinker: &dyn GenerativeBackend,
    command: &str,
    refs: &crate::generate_refs::Refs,
    image: &Rgba8,
    prompt: String,
    place: Option<&str>,
    scene: Option<&Gray8>,
    seed: u64,
    ctx: &JobCtx,
) -> String {
    ctx.progress(0.02, "Improving the prompt");
    // Krea 2 AnyPaint: the whole finished picture described, the area seen as flat grey.
    if let (Some(mask), true) = (scene, command == FILL && refs.images.is_empty()) {
        let grey = grey_where(image, mask);
        let words = format!("{} {}", super::FILL_SCENE_LEAD, prompt.trim());
        return match enhance_prompt(thinker, EnhanceKind::FillScene, &words, Some(&grey), None, seed, &JobProgress::span(ctx, 0.02, 0.05)) {
            Ok(Some(p)) => p,
            Ok(None) | Err(_) => prompt,
        };
    }
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
    let seen = if fill_ref { refs.images.first().and_then(|r| side_by_side(image, r).ok()) } else { None };
    let seen = seen.as_ref().unwrap_or(image);
    // A rewriter that fails (no encoder on the server, nothing usable said) leaves the
    // prompt as typed; the run itself reports a server that is down.
    match enhance_prompt(thinker, kind, &prompt, Some(seen), place, seed, &JobProgress::span(ctx, 0.02, 0.05)) {
        Ok(Some(p)) => p,
        Ok(None) | Err(_) => prompt,
    }
}

/// What a whole-picture route changed beside the hole, connected to it (`change_band`): the
/// request mask widened by it, and the layer coverage widened by it (at `(w, h)`); `None`s when
/// nothing changed there.
pub(super) fn change_band_coverage(
    out: &Rgba8,
    original: &Rgba8,
    run_mask: &Gray8,
    layer_coverage: &[f32],
    removal: bool,
    (w, h): (u32, u32),
) -> Result<(Option<Gray8>, Option<Vec<f32>>)> {
    let (band_share, band_change) = if removal { (SHADOW_MAX_SHARE, SHADOW_CHANGE) } else { (EDIT_BAND_MAX_SHARE, EDIT_BAND_CHANGE) };
    let Some(b) = change_band(out, original, run_mask, band_share, band_change) else { return Ok((None, None)) };
    let widened = Gray8::new(b.width, b.height, b.data.iter().zip(&run_mask.data).map(|(a, m)| (*a).max(*m)).collect()).ok();
    let b = gray8_at(b, w, h)?;
    let coverage = layer_coverage.iter().zip(&b.data).map(|(c, v)| c.max(f32::from(*v) / 255.0)).collect();
    Ok((widened, Some(coverage)))
}

/// The composite under `rect` that a fill sends, an expand's new area prefilled.
pub(super) fn source_picture(doc: &Document, ctx: &JobCtx, rect: Rect, outpaint: Option<&Outpaint>) -> Result<Rgba8> {
    ctx.progress(0.0, "Rendering");
    // The pictures of the run land on the area sent (an expand names its own place: the document
    // on screen has not grown yet).
    if outpaint.is_none() {
        ctx.preview_at(rect);
    }
    let composite = photocraft_compose::render(doc, rect).to_rgba8();
    let mut image = Rgba8::new(composite.width, composite.height, composite.pixels).map_err(gen_err)?;
    if let Some(o) = outpaint {
        let p = o.picture;
        let inner = Rect::new(p.x0 - rect.x0, p.y0 - rect.y0, p.x1 - rect.x0, p.y1 - rect.y0);
        if o.lora { grey_outside(&mut image, inner) } else { prefill_outside(&mut image, inner) }
    }
    Ok(image)
}

pub(super) fn fill_region(
    doc: &mut Document,
    active: &mut Option<LayerId>,
    ctx: &JobCtx,
    job: FillJob,
    rect: Rect,
    coverage: Vec<f32>,
    outpaint: Option<Outpaint>,
) -> Result<Filled> {
    let mut image = source_picture(doc, ctx, rect, outpaint.as_ref())?;
    let (w, h) = (rect.width(), rect.height());
    let FillJob {
        backend,
        thinker,
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
        tone_most,
        prefill,
        tuning,
        object,
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
    let adapter = match (&coverage_tight, outpaint.is_none()) {
        (Some(tight), true) => removal_adapter(backend.as_ref(), &template_id, auto, n, tight, &image, (w, h), (rw, rh), radius)?,
        _ => None,
    };
    // Every variation is the adapter's when the adapter template was named; else the first.
    let adapter_named = adapter.as_ref().is_some_and(|(id, _, _)| *id == template_id);
    let image_plain = image.clone();
    // A route that regenerates the whole picture (an edit by instruction; every Qwen-Image-2.1
    // template, whose "fill" is an edit with the mask in view) hands it back a few pixels off:
    // the result is realigned to the picture outside the mask (`generate_realign`).
    let qwen21 = template::find(&template_id).is_ok_and(|t| t.meta.family == "qwen-image-2.1");
    // The Outpaint LoRA's edit redraws the whole canvas too.
    let realign = (outpaint.is_none() && (command == EDIT || qwen21)) || outpaint.as_ref().is_some_and(|o| o.lora);
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
        let scene = krea2_fill(&template_id).then_some(&mask);
        rewritten_fill_prompt(thinker.as_ref(), command, &refs, &image, prompt, place.as_deref(), scene, seed, ctx)
    } else {
        prompt
    };
    ctx.check()?;
    // An expand template crops the model's reference to the picture: its rectangle inside the
    // request, in the request's (possibly resampled) pixels.
    let mut params = request_params(&tuning, outpaint.as_ref(), rect, (w, h), (rw, rh));
    if tuning.fast_draft && fast_draft_applies(&template_id, steps) {
        params.insert("step_cache".to_string(), json!(FAST_DRAFT_STEP_CACHE));
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
        let resp = backend.as_ref().run(&req, &JobProgress::span(ctx, lo, hi)).map_err(gen_err)?;
        ctx.check()?;
        let out = resp.images.into_iter().next().ok_or_else(|| EngineError::Other("the backend returned no image".into()))?;
        let out = if realign { realigned(out, &image_plain, run_mask) } else { out };
        // `shadow`: on a whole-picture route a removal also takes what changed with the object
        // beside the hole; that band joins the hole for the tone match (the membrane carries
        // the result across it, so its edge meets the picture) and the layer mask.
        // The same on a swap or an edit with a selection: what the model drew past the selection,
        // connected to it (a new coat longer than the old jacket), joins the layer mask, which
        // otherwise cut it off in a dark band at the selection's edge (measured 2026-10-27,
        // `benchmarks.md` swaps); held to a smaller share of the selection than a shadow.
        let take_band = (qwen21 || command == EDIT) && tuning.shadow && outpaint.is_none() && this_adapter.is_none() && layer_coverage.iter().any(|c| *c < 1.0);
        let (widened, band_coverage) =
            if take_band { change_band_coverage(&out, &image_plain, run_mask, &layer_coverage, removal, (w, h))? } else { (None, None) };
        let run_mask = widened.as_ref().unwrap_or(run_mask);
        // A removal's result is tone-matched to the picture at the hole's edge and judged.
        let (out, mismatch) =
            if removal { tone_match_removal(out, &image_plain, run_mask, this_adapter.is_some(), tone_most, &mut judged) } else { (out, 0.0) };
        mismatches.push(mismatch);
        let out = rgba8_at(out, w, h)?;
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
                edge: edge.name().into(),
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
                        match (
                            still_there(thinker.as_ref(), a, name, seed.wrapping_add(200 + i as u64), &span),
                            still_there(thinker.as_ref(), b, name, seed.wrapping_add(300 + i as u64), &span),
                        ) {
                            (Some(false), Some(true)) => Some(true),
                            (Some(true), Some(false)) => Some(false),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                presence.or_else(|| match (judged.get(best), judged.get(i)) {
                    (Some(a), Some(b)) => {
                        judge_pair(thinker.as_ref(), a, b, object.as_deref(), seed.wrapping_add(100 + i as u64), &JobProgress::span(ctx, 0.96, 0.97))
                    }
                    _ => None,
                })
            };
            let left_wins = verdict.unwrap_or(m_best <= m_i);
            if !left_wins {
                best = i;
            }
        }
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

/// The JSON a fill or expand job answers with.
pub(super) fn made_json(made: &Made, w: u32, h: u32, (rw, rh): (u32, u32), template_id: &str) -> Value {
    let first = made.first();
    json!({
        "layer": first.map(|m| m.0.0), "seed": first.map(|m| m.1), "runId": first.map(|m| m.2.clone()), "template": template_id,
        "layers": made.iter().map(|m| m.0.0).collect::<Vec<_>>(), "seeds": made.iter().map(|m| m.1).collect::<Vec<_>>(),
        "width": w, "height": h, "requestWidth": rw, "requestHeight": rh,
        "ms": made.iter().map(|m| m.3).sum::<u64>(),
        "timings": made.iter().map(|m| serde_json::to_value(m.4).unwrap_or(Value::Null)).collect::<Vec<_>>(),
    })
}

pub(super) fn run_fill(s: &mut Session, p: &Value) -> Result<Value> {
    let plan = plan_fill(s, p)?;
    let backend = backend(s)?;
    let FillPlan { template, prompt, negative, seed, steps, guidance, name, models, rect, variations, auto, edge, enhance, refs, grow, erase, tuning } = plan;
    let label = name.clone();
    // A fill planned on an edit template (`fillAsEdit`) runs as an edit by instruction.
    let as_edit = template.meta.task == Task::Edit;
    let job = FillJob {
        thinker: Arc::clone(&backend),
        backend,
        command: if as_edit { EDIT } else { FILL },
        template_id: template.meta.id.clone(),
        auto,
        auto_order: if template.meta.id == NATIVE21_FILL {
            native_fill_order(&prompt, enhance)
        } else if template.meta.id == KLEIN_FILL_TEMPLATE {
            &[KLEIN_FILL_TEMPLATE]
        } else {
            auto_order(FILL, quality(s))
        },
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
        tone_most: 1.0,
        prefill: false,
        tuning,
        object: None,
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
