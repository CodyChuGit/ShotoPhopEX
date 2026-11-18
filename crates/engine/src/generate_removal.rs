//! Generative Remove: naming what the selection holds, growing the hole, and the removal and
//! background jobs that paint it out.

use std::sync::Arc;

use photocraft_color::PixelFormat;
use photocraft_doc::{Layer, LayerMask};
use photocraft_genai::{GenerativeBackend, Gray8, Rgba8, Task, template};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::{EngineError, Result, Session};

use super::{
    AUTO_TEMPLATE, Common, DEFAULT_FILL_TEMPLATE, EDIT, Edge, EnhanceKind, FILL, FillJob, JobProgress, KLEIN_FILL_TEMPLATE, KLEIN_REMOVAL_PIXELS,
    MAX_FILL_REQUEST_PIXELS, MAX_GROW, NATIVE21_REMOVAL_PIXELS, TONE_MATCH_INSIDE, Tuning, align_to_grid, auto_order, backend, crop_rgba8, dim_outside,
    edge_mismatch, embed_coverage, enhance_prompt, feather_outward, fill_rect, fill_region, gen_err, highlight_covered, judged_rect, made_json, native21_auto,
    paint_left, place_in_words, plan_common, removal_quality, request_size, resize_gray8, resolve_auto, rgba8_at, side_by_side, speed, still_there,
    template_or, tone_matched, write_rgba8,
};

/// [`Tuning::remove_margin`] unless given.
pub(super) const REMOVE_MARGIN: f64 = 0.25;

/// [`Tuning::name_shadow`] unless given.
pub(super) const NAME_SHADOW: bool = false;

/// What the generative Content-Aware Fill, Remove tool and Spot Healing ask the fill model,
/// with the picture's pixels kept under the noise mask. "Remove" opens the fill templates'
/// imperative wrapper, which adds "fit the scene, change nothing else". Measured on the
/// lighthouse's boat (2026-10-21, `caf-matrix-live.ps1`): this generic instruction removes the
/// object and continues the sea; naming the object does no better; the mask-guided template
/// matches it at twice the time; a hole flattened to grey is painted back as a grey panel by
/// 2511 (the `erase` that serves a swap from a reference defeats a removal).
pub const REMOVE_PROMPT: &str = "Remove the object in the selected area and fill its place with the surrounding scene continued seamlessly (its background, surfaces, textures, patterns and lines), as if nothing had ever been there";

/// The part of [`REMOVE_PROMPT`] the rewriter's name replaces ("the gold necklace").
pub(super) const REMOVE_PROMPT_OBJECT: &str = "the object in the selected area";

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
pub(super) fn stroke_overlap(stroke: &[f32], sb: Rect, obj: &[f32], on: u64, canvas: Rect) -> f64 {
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
pub(super) fn dump_views(seen: &Rgba8) {
    let Ok(dir) = std::env::var("PHOTOCRAFT_DUMP_VIEWS") else { return };
    let Ok(png) = photocraft_genai::png::encode_rgba8(seen) else { return };
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    let _ = std::fs::write(std::path::Path::new(dir.trim()).join(format!("views-{stamp}.png")), png);
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
pub(super) const REMOVE_GROW_SHARE: f64 = 0.15;

pub(super) const REMOVE_GROW_MIN: u32 = 16;

pub(super) const REMOVE_GROW_MAX: u32 = 96;

/// A stroke removes the object the detector finds under it whole only when the stroke covers
/// at least this share of that object: a dab on a blemish (under 1 % of the face the detector
/// finds under it) must heal the blemish, not remove the face. A quarter was too strict: a
/// stroke along half a watch covers 22 % of it (measured 2026-10-21 on the robustness set).
pub(super) const WHOLE_OBJECT_MIN_COVERED: f64 = 0.10;

/// …and never an object larger than this share of the canvas.
pub(super) const WHOLE_OBJECT_MAX_CANVAS: f64 = 0.4;

/// How many seeds a removal runs unless told otherwise (`variations`): two, the one whose fill
/// matches its edge best shown ([`edge_mismatch`]). One seed in four or so filled a watch's
/// place with the table beside the wrist, or a crow's with blue on a grey sky (measured
/// 2026-10-21); a second seed at 6 s on the Lightning tier buys the choice.
pub const DEFAULT_REMOVAL_VARIATIONS: u32 = 2;

pub(super) const MAX_REMOVAL_VARIATIONS: u32 = 4;

/// `variations` for a removal: [`DEFAULT_REMOVAL_VARIATIONS`] unless given, 1 to 4.
pub(crate) fn removal_variations(p: &Value) -> u32 {
    p.get("variations")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .map_or(DEFAULT_REMOVAL_VARIATIONS, |v| (v.round() as i64).clamp(1, i64::from(MAX_REMOVAL_VARIATIONS)) as u32)
}

/// A removal's request is at least this many pixels on a side (where the canvas allows), so
/// the model sees real context around a small dab and the request is not upscaled many times
/// over: a 40-px dab on the necklace's pendant made a 108-px request, sent at 512 (4.7×), and
/// came back as a lighter, smoother patch of fabric (measured 2026-10-21).
pub(super) const REMOVE_MIN_REQUEST: i32 = 384;

/// The margin of the close-up the rewriter names a removal's object from, a share of the grown
/// stroke's longer side, by which model answers. ComfyUI's Qwen3-VL 4B names the whole object
/// from a loose close-up and names a part from a tight one (the dog's chain, the bicycle's
/// handlebar); the native worker's 8B names the most salient thing in a loose one, the wearer
/// ("the man wearing the cap and mask" for a stroke on the mask, three seeds of three) and the
/// worn thing from a tight one. Measured 2026-10-26 with `naming-probe.ps1` on the stock set's
/// hard scenes, three seeds, both strokes.
pub(super) const REMOVE_CLOSE_MARGIN: f64 = 0.25;

pub(super) const REMOVE_CLOSE_MARGIN_WORKER: f64 = 0.1;

/// The two views the rewriter names a removal's object from, for a stroke over `hole`: the
/// surroundings (at least [`REMOVE_MIN_REQUEST`] on a side) and the close-up (the hole grown as
/// the removal grows it, with the margin of the model that answers: `by_worker` when the native
/// worker's own 8B does).
pub(crate) fn naming_views(hole: Rect, canvas: Rect, by_worker: bool) -> Result<(Rect, Rect)> {
    let margin = if by_worker { REMOVE_CLOSE_MARGIN_WORKER } else { REMOVE_CLOSE_MARGIN };
    let close = fill_rect(hole.inflate(remove_grow(hole) as i32).intersect(&canvas), margin, canvas)?;
    Ok((at_least(close, REMOVE_MIN_REQUEST, canvas), close))
}

/// The picture the rewriter names a stroke's object from, at the request size: the
/// surroundings beside the close-up ([`naming_views`]), the close-up dimmed outside the stroke
/// (`coverage` over `area`), and where in the surroundings the hole is, in words.
pub(crate) fn naming_picture(doc: &photocraft_doc::Document, hole: Rect, coverage: &[f32], area: Rect) -> Result<(Rgba8, String)> {
    let by_worker = true;
    let (wide_rect, close_rect) = naming_views(hole, doc.bounds(), by_worker)?;
    let crop = |r: Rect| -> Result<Rgba8> {
        let c = photocraft_compose::render(doc, r).to_rgba8();
        Rgba8::new(c.width, c.height, c.pixels).map_err(gen_err)
    };
    let mut close = crop(close_rect)?;
    dim_outside(&mut close, close_rect, coverage, area);
    let seen = side_by_side(&crop(wide_rect)?, &close)?;
    let (rw, rh) = request_size(seen.width, seen.height);
    Ok((rgba8_at(seen, rw, rh)?, place_in_words(hole, wide_rect)))
}

/// `rect` widened about its centre to at least `min_side` on each side, slid inside `canvas`
/// (clipped only where the canvas is smaller), on the request grid.
pub(super) fn at_least(rect: Rect, min_side: i32, canvas: Rect) -> Rect {
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
pub(super) const POINT_OBJECT_MAX_SCALE: f64 = 1.5;

/// Whether an object's box `ob` is within [`POINT_OBJECT_MAX_SCALE`] of the stroke's `sb`.
pub(super) fn fits_stroke(ob: Rect, sb: Rect) -> bool {
    f64::from(ob.width()) <= POINT_OBJECT_MAX_SCALE * f64::from(sb.width()) && f64::from(ob.height()) <= POINT_OBJECT_MAX_SCALE * f64::from(sb.height())
}

/// Whether a stroke (`stroke`, one value per pixel of `sb`) covers enough of an object (`obj`
/// over the canvas, `on` pixels of it) for a removal to take the object whole
/// ([`WHOLE_OBJECT_MIN_COVERED`], [`WHOLE_OBJECT_MAX_CANVAS`]).
pub(super) fn stroke_takes(stroke: &[f32], sb: Rect, obj: &[f32], on: u64, canvas: Rect) -> bool {
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
pub(super) fn take_in(area: Rect, coverage: &[f32], obj: &[f32], ob: Rect, canvas: Rect, sel: Option<&photocraft_raster::Surface>) -> (Rect, Vec<f32>) {
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
        ", ",
        " and replace",
        " and fill",
        " from ",
        " with ",
        " so ",
        ".",
        ";",
        " replacing",
        " completely",
        " entirely",
        " to ",
        " in the ",
        " on the ",
        " perched",
        " sitting",
        " standing",
        " lying",
        " hanging",
        " taped",
        " mounted",
        " parked",
        " walking",
        " resting",
        " placed",
        " attached",
        " leaning",
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
    let not_a_thing = [
        "area",
        "region",
        "part",
        "selection",
        "selected",
        "painted",
        "marked",
        "outlined",
        "view",
        "section",
        "spot",
        "portion",
        "object",
        "item",
        "thing",
        "element",
        "border",
        "frame",
        "edge",
    ]
    .iter()
    .any(|w| lower.split(|c: char| !c.is_alphanumeric()).any(|t| t == *w));
    // Two things in the selection ("the man in a white shirt and the black car") name both: a
    // longer name is a sentence, but each part of one joined by "and" may be six words.
    let fits = name.split(" and ").all(|part| part.split_whitespace().count() <= 6) && name.split_whitespace().count() <= 13;
    (!name.is_empty() && !not_a_thing && fits).then(|| name.to_string())
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
pub(super) fn remove_grow(b: Rect) -> u32 {
    let side = f64::from(b.width().max(b.height()));
    ((side * REMOVE_GROW_SHARE).round() as u32).clamp(REMOVE_GROW_MIN, REMOVE_GROW_MAX)
}

/// What a generative removal's fill job asks for, apart from the session and the label.
#[derive(Clone)]
pub(super) struct RemovalOpts {
    /// How far the hole grows (0: [`remove_grow`] decides).
    pub grow: u32,
    /// Flatten the hole to grey in what the model sees.
    pub erase: bool,
    /// Let the rewriter name the object first.
    pub enhance: bool,
    /// How many results are judged.
    pub variations: u32,
    pub seed: Option<u64>,
    /// Pre-fill the hole before the model sees it.
    pub prefill: bool,
    pub tuning: Tuning,
}

/// Split into Layers' removal of one subject: the hole grown by [`remove_grow`], its pixels
/// kept and pre-filled, no rewriter.
pub(super) fn split_removal(variations: u32, seed: Option<u64>, tuning: Tuning) -> RemovalOpts {
    RemovalOpts { grow: 0, erase: false, enhance: false, variations, seed, prefill: true, tuning }
}

/// The fill job a generative removal runs ([`generative_remove`], Split into Layers'
/// subjects): the fill template the preferences give (under `auto` the native
/// Qwen-Image-2.1 removal route), the measured removal prompt, `variations` judged.
pub(super) fn removal_job(s: &Session, label: &str, opts: RemovalOpts) -> Result<FillJob> {
    removal_job_on(s, label, opts, None)
}

/// [`removal_job`] on `template` when one is given (the preferences and `auto` aside).
pub(super) fn removal_job_on(s: &Session, label: &str, opts: RemovalOpts, template: Option<&str>) -> Result<FillJob> {
    let RemovalOpts { grow, erase, enhance, variations, seed, prefill, tuning } = opts;
    let (default_template, default_model) = {
        let i = &s.prefs().integrations;
        (template.map_or_else(|| template_or(&i.default_fill_template, AUTO_TEMPLATE), str::to_string), i.default_edit_model.clone())
    };
    let mut auto = default_template == AUTO_TEMPLATE;
    let mut tuning = tuning;
    let base = match (auto, native21_auto(s, FILL, true)) {
        (true, Some(t)) => {
            auto = false;
            if tuning.max_pixels == MAX_FILL_REQUEST_PIXELS {
                tuning.max_pixels = if t == KLEIN_FILL_TEMPLATE { KLEIN_REMOVAL_PIXELS } else { NATIVE21_REMOVAL_PIXELS };
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
    let backend = backend(s)?;
    Ok(FillJob {
        thinker: Arc::clone(&backend),
        backend,
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
        tone_most: 1.0,
        prefill,
        tuning,
        object: None,
    })
}

/// What Split's background fill asks for once the removal has painted a large subject back:
/// the backdrop alone, the subject never named (a model told to remove "the person" from a
/// 12-megapixel portrait kept them in place on every route, measured in the app 2026-11-03).
pub(crate) const BACKGROUND_PROMPT: &str = "the empty background continued seamlessly: the same wall, floor and light as around it, with nobody there";

/// What Split's grey-box fill asks for behind a subject with others still behind it (the
/// peel): the scene continued as it would look with that one gone, nothing new.
pub(crate) const PEEL_PROMPT: &str = "the picture continued behind what stood in the grey area: the same scene, the people and things around it completed as they would look with it gone, the same wall, furniture and light, nothing new added";

/// Split's background fill for a large subject: a plain fill of [`BACKGROUND_PROMPT`] with the
/// hole flattened to grey in what the model sees (`erase`), on Qwen-Image-2.1's fill when it is
/// served, else the fill order. Seeing the subject, every fill route handed it back; seeing a
/// grey box, the 2.1 fill continued the curtain behind the owner's portrait (measured in the
/// app 2026-11-03).
pub(super) fn background_job(s: &Session, label: &str, seed: Option<u64>, tuning: Tuning, prompt: &str) -> Result<FillJob> {
    let default_model = s.prefs().integrations.default_edit_model.clone();
    let (base, auto) = match native21_auto(s, FILL, false) {
        Some(t) => (t.to_string(), false),
        None => (DEFAULT_FILL_TEMPLATE.to_string(), true),
    };
    let Common { template, prompt, negative, seed, steps, guidance, models, .. } =
        plan_common(s, FILL, &json!({"prompt": prompt, "seed": seed}), &base, Task::Fill, &default_model)?;
    let backend = backend(s)?;
    Ok(FillJob {
        thinker: Arc::clone(&backend),
        backend,
        command: FILL,
        template_id: template.meta.id.clone(),
        auto,
        auto_order: auto_order(FILL, speed(s)),
        prompt,
        negative,
        seed,
        steps,
        guidance,
        name: label.to_string(),
        models,
        variations: 1,
        edge: Edge::Soft,
        enhance: false,
        refs: Default::default(),
        grow: 0,
        erase: true,
        rewritten: None,
        removal: false,
        tone_most: 1.0,
        prefill: false,
        tuning,
        object: None,
    })
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
    // No Consistency LoRA here: on a pedestrian it turned the removal into a copy of the input
    // (mean change 0.9/255 inside the hole, measured in the app 2026-11-12); a request can still
    // name it in `loras`.
    let opts = RemovalOpts { grow, erase, enhance, variations, seed, prefill, tuning };
    let job = removal_job(s, label, opts.clone())?;
    // Qwen-Image-2.1 first; where the object is still there afterwards, the 2511 Remove LoRA
    // ([`REMOVE_RETRY_TEMPLATE`]): the owner's choice of 2026-11-16, when the 2.1 edit painted a
    // woman selected whole back in 4 of 4 runs and the 2511 LoRA removed her in 2 of 2.
    let retry = (job.template_id == "qwen-2.1/edit")
        .then(|| removal_job_on(s, label, RemovalOpts { enhance: false, ..opts }, Some(REMOVE_RETRY_TEMPLATE)).ok())
        .flatten();
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
                let (seen, place) = naming_picture(doc, hole, &coverage, area)?;
                // The rewriter's sentence gives the object's name and what lies behind it; the
                // fill's prompt is composed from them with the measured generic wording (the
                // rewriter's own closing clauses listed the scene's neighbours and the model
                // painted them: a rock for the boat, the table over a wrist).
                dump_views(&seen);
                let said = enhance_prompt(
                    job.thinker.as_ref(),
                    EnhanceKind::Remove,
                    &job.prompt,
                    Some(&seen),
                    Some(&place),
                    job.seed,
                    &JobProgress::span(ctx, 0.01, 0.04),
                );
                // What the rewriter said and what was taken from it: when nothing is named the
                // fill gets the generic "the object" and may take the wrong one (a pedestrian
                // kept and the car beside him removed, in the app 2026-11-13).
                eprintln!("remove: rewriter said {said:?}, named {:?}", said.as_ref().ok().and_then(|p| p.as_deref()).and_then(named_object));
                if let Ok(Some(p)) = said
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
            let (object, rewritten, thinker, check_seed, used_grow) =
                (job.object.clone(), job.rewritten.clone(), Arc::clone(&job.thinker), job.seed.wrapping_add(500), job.grow);
            let below = *active;
            let filled = fill_region(doc, active, ctx, job, rect, coverage.clone(), None)?;
            let filled = match (retry, object) {
                (Some(mut retry), Some(name)) if resolve_auto(retry.backend.as_ref(), &[retry.template_id.as_str()], "") == retry.template_id => {
                    ctx.progress(0.98, "Checking the result");
                    let c = photocraft_compose::render(doc, hole).to_rgba8();
                    let seen = Rgba8::new(c.width, c.height, c.pixels).map_err(gen_err)?;
                    let there = still_there(thinker.as_ref(), &seen, &name, check_seed, &JobProgress::span(ctx, 0.98, 0.99));
                    eprintln!("remove: after {}, is the {name} still there? {there:?}", filled.4);
                    if there == Some(true) {
                        ctx.check()?;
                        for m in &filled.0 {
                            doc.remove(m.0);
                        }
                        *active = below;
                        retry.object = Some(name);
                        retry.rewritten = rewritten;
                        retry.grow = used_grow;
                        fill_region(doc, active, ctx, retry, rect, coverage, None)?
                    } else {
                        filled
                    }
                }
                _ => filled,
            };
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

/// A removal's highlight adapter: its template id, the picture with the hole painted its colour,
/// and the tight hole's mask (see `fill_region`), or `None` when no adapter template applies.
#[allow(clippy::too_many_arguments)]
pub(super) fn removal_adapter(
    backend: &dyn GenerativeBackend,
    template_id: &str,
    auto: bool,
    n: u32,
    tight: &[f32],
    image: &Rgba8,
    (w, h): (u32, u32),
    (rw, rh): (u32, u32),
    radius: usize,
) -> Result<Option<(String, Rgba8, Gray8)>> {
    let named = template::find(template_id).ok().filter(|t| t.meta.highlight.is_some()).map(|t| t.meta.id.clone());
    let id = named.or_else(|| (auto && n > 1).then(|| resolve_auto(backend, REMOVE_HIGHLIGHT_ORDER, "")).filter(|s| !s.is_empty()));
    Ok(match id.and_then(|id| template::find(&id).ok()).and_then(|t| t.meta.highlight.clone().map(|c| (t.meta.id.clone(), c))) {
        Some((id, colour)) => {
            let g = Gray8::new(w, h, tight.iter().map(|c| (c * 255.0 + 0.5) as u8).collect()).map_err(gen_err)?;
            let g = feather_outward(&g, radius)?;
            let g = if (rw, rh) == (w, h) { g } else { resize_gray8(&g, rw, rh)? };
            Some((id, highlight_covered(image, &g, &colour), g))
        }
        None => None,
    })
}

/// A removal's result tone-matched to the picture at the hole's edge, and its edge mismatch for
/// the judge (its crop pushed to `judged`); as it came, mismatch 0, when it is not the
/// picture's size.
pub(super) fn tone_match_removal(out: Rgba8, picture: &Rgba8, mask: &Gray8, adapter: bool, most: f32, judged: &mut Vec<Rgba8>) -> (Rgba8, f64) {
    if (out.width, out.height) != (picture.width, picture.height) {
        return (out, 0.0);
    }
    // The adapter leaving its red in the hole is a loss before any judging: the judge
    // once chose a red-tinted glass as the natural one (measured 2026-10-22).
    let painted = adapter && paint_left(&out, mask);
    let matched = tone_matched(&out, picture, mask, most);
    judged.push(crop_rgba8(&matched, judged_rect(mask, TONE_MATCH_INSIDE)));
    let mismatch = if painted { f64::INFINITY } else { edge_mismatch(&out, picture, mask) };
    (matched, mismatch)
}

/// What a removal adapter trained on a painted highlight is told (the Object Remover v2
/// card's sample prompts, every one of them this sentence).
pub(super) const HIGHLIGHT_REMOVE_PROMPT: &str = "Remove the red highlighted object from the scene.";

/// The removal adapters `auto` tries, first installed one first, for a removal's first
/// variation (the plain fill takes the rest, and the judge picks).
pub(super) const REMOVE_HIGHLIGHT_ORDER: &[&str] = &["qwen-edit-2511/remove-lightning-8"];

/// What a removal on Qwen-Image-2.1 runs again on when the object is still there afterwards
/// ([`generative_remove`]); skipped where the worker does not serve it.
pub(super) const REMOVE_RETRY_TEMPLATE: &str = "qwen-edit-2511/remove-lightning-8";

/// How far past the selection the adapter's painted hole reaches: a few pixels, as the
/// highlights it was trained on, not the removal's 15 % growth (a fat red hole came back red
/// on a third of the stock set; the selection is the object's silhouette already).
pub(super) const HIGHLIGHT_GROW: usize = 6;
