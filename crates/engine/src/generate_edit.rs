//! `generate.edit`, `generate.angles` and `generate.characterSheet`: whole-picture edits by
//! instruction.

use std::collections::BTreeMap;
use std::sync::Arc;

use photocraft_color::PixelFormat;
use photocraft_doc::{Document, LayerId, LayerMask};
use photocraft_genai::Task;
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::layer_param;
use crate::jobs::JobCtx;
use crate::{EngineError, Result, Session};

use super::{
    ANGLE_ROTATIONS, ANGLE_TILTS, ANGLE_ZOOMS, ANGLES, ANGLES_2511_MIN_COMPUTE_CAP, ANGLES_TEMPLATE, ANGLES21_TEMPLATE, AUTO_TEMPLATE, CHARACTER_SHEET, Common,
    DEFAULT_EDIT_TEMPLATE, EDIT, FillJob, FillPlan, MAX_REQUEST_PIXELS, Tuning, ULTRA_IMAGE_TEMPLATE, auto_order, auto_quality, backend, bad,
    edit_request_pixels, edit_rewrite, fill_region, loras_of, made_json, native21_auto, opt_bool, opt_num, opt_str, parse_edge, parse_variations, plan_common,
    run_image, short, template_files_present, ultra, web_unavailable,
};

/// The Multiple Angles LoRA's words for a camera, `[azimuth] [elevation] [distance]`, or None
/// for a value it has no words for.
pub fn angle_words(rotation: u32, tilt: i32, zoom: u32) -> Option<String> {
    let r = ANGLE_ROTATIONS.iter().find(|(d, _)| *d == rotation)?.1;
    let t = ANGLE_TILTS.iter().find(|(d, _)| *d == tilt)?.1;
    let z = ANGLE_ZOOMS.iter().find(|(d, _)| *d == zoom)?.1;
    Some(format!("{r} {t} {z}"))
}

/// The camera move in plain words, one sentence per axis that moves (nothing for the front,
/// eye-level, medium camera). The lesson of the "Latent Layer Cameras" SD 1.5 set: each camera is
/// one named target in plain language. In the 2026-11-17 5090 test, the LoRA's tokens alone
/// reached a true side view at rotation 90 on 1 of 5 runs (the rest stopped at 10-45°); with
/// this sentence after them, 5 of 5, at the same speed, scene and detail.
pub fn angle_sentence(rotation: u32, tilt: i32, zoom: u32) -> String {
    let side = if rotation > 180 { "left" } else { "right" };
    let turn = match rotation {
        45 | 315 => format!("The camera has moved 45 degrees around to the subject's {side}: a three-quarter view."),
        90 | 270 => {
            format!("The camera has moved 90 degrees around to the subject's {side}: a true side profile, the subject seen exactly from its {side} side.")
        }
        135 | 225 => format!("The camera has moved 135 degrees around to the subject's {side}: we see the subject's back and {side} side."),
        180 => "The camera has moved 180 degrees around the subject: we see it from directly behind.".to_string(),
        _ => String::new(),
    };
    let height = match tilt {
        -30 => "The camera is lowered below the subject and looks up at it at 30 degrees.",
        30 => "The camera is raised above the subject and looks down at it at 30 degrees.",
        60 => "The camera is raised high above the subject and looks down at it at 60 degrees.",
        _ => "",
    };
    let distance = match zoom {
        0 => "The camera has moved back: the whole subject and its surroundings are in the frame.",
        10 => "The camera has moved close: the subject's main part fills the frame.",
        _ => "",
    };
    [turn.as_str(), height, distance].iter().filter(|w| !w.is_empty()).copied().collect::<Vec<_>>().join(" ")
}

/// The prompt for a camera: `<sks> [azimuth] [elevation] [distance]`, the LoRA's trained words,
/// then [`angle_sentence`]; None for a value the LoRA has no words for.
pub fn angle_prompt(rotation: u32, tilt: i32, zoom: u32) -> Option<String> {
    let words = angle_words(rotation, tilt, zoom)?;
    let sentence = angle_sentence(rotation, tilt, zoom);
    Some(if sentence.is_empty() { format!("<sks> {words}") } else { format!("<sks> {words} {sentence}") })
}

/// The template `generate.angles` runs: 2511 with the Multiple Angles LoRA, which was trained on
/// exactly these camera moves. Plain Qwen-Image-2.1 by instruction never turned the camera (0 of
/// 4 at 45°, 2026-11-03 bench), so it does not stand in when the LoRA is missing: the Angles
/// window says what to download instead. Bring-your-own: on a card below 2511's budget it still
/// runs, with the window's `may not fit` note.
///
/// On an NVIDIA card older than RTX 40 (no fp8 matrix cores) 2511's fp8 file does not run (a
/// 3090 crashed at 24 GB, 2026-11-17), so Qwen-Image-2.1 with its own Multiple Angles LoRA does
/// ([`ANGLES21_TEMPLATE`]; 21.7 GB, 24 s warm there). The owner's call, 2026-11-17.
pub fn angles_template(_s: &Session) -> &'static str {
    angles_template_for(photocraft_genai::card_compute_cap())
}

/// [`angles_template`] for a card of `compute_cap` (None: no NVIDIA card known, 2511 as ever).
pub fn angles_template_for(compute_cap: Option<u32>) -> &'static str {
    if !photocraft_genai::MLX && compute_cap.is_some_and(|c| c < ANGLES_2511_MIN_COMPUTE_CAP) { ANGLES21_TEMPLATE } else { ANGLES_TEMPLATE }
}

/// The 2.1 Multiple Angles LoRA's prompt for a camera, `<mva> {azimuth}, {elevation}[ close-up]`
/// (its card's vocabulary: 30° azimuth steps, eye-level to top-down, close-up or not). Our 45°
/// steps take its quarter views (60°, 120°…), the nearest it has with the same words; it has no
/// low angle, so a low camera is eye-level, and no wide shot apart from its default framing.
pub fn angle_prompt_21(rotation: u32, tilt: i32, zoom: u32) -> Option<String> {
    let azimuth = ANGLE_ROTATIONS.iter().find(|(d, _)| *d == rotation)?.1;
    let elevation = match tilt {
        -30 | 0 => "eye-level shot",
        30 => "elevated shot",
        60 => "high-angle shot",
        _ => return None,
    };
    let close = match zoom {
        0 | 5 => "",
        10 => " close-up",
        _ => return None,
    };
    Some(format!("<mva> {azimuth}, {elevation}{close}"))
}

/// The Mac build ([`photocraft_genai::MLX`]), for the UI.
pub fn mlx_build() -> bool {
    photocraft_genai::MLX
}

/// The tooltip of a tool the Mac build has no model for.
pub const NOT_ON_MAC: &str = "Not on Mac yet";

/// Generative Angles runs Qwen-Image-Edit with fal's LoRA, which has no MLX port.
pub(super) fn angles_enabled(s: &Session) -> std::result::Result<(), String> {
    if photocraft_genai::MLX {
        return Err(NOT_ON_MAC.into());
    }
    edit_enabled(s)
}

pub(super) fn edit_enabled(s: &Session) -> std::result::Result<(), String> {
    web_unavailable()?;
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// `generate.edit`: the whole picture goes to an edit model with an instruction; the result is
/// a layer over the canvas, masked to the selection when there is one.
pub(super) fn plan_edit(s: &Session, p: &Value) -> Result<FillPlan> {
    // `swap`: the BFS face or body swap (its template, LoRA and prompt), see `generate_swap.rs`;
    // `repose`: the PoseStudio repose, see `generate_repose.rs`.
    let swapped = edit_rewrite(s, p)?;
    let p = swapped.as_ref().unwrap_or(p);
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
    // An edit tunes nothing but its LoRAs (its templates bind no sampler; the size cap is its
    // template's, see `edit_request_pixels`).
    let tuning = Tuning {
        loras: loras_of(EDIT, p)?,
        fast_draft: s.prefs().integrations.fast_draft,
        max_pixels: edit_request_pixels(&template.meta.id, ultra(s)),
        extra: angles_extra(p)?,
        ..Tuning::default()
    };
    Ok(FillPlan { template, prompt, negative, seed, steps, guidance, name, models, rect, variations, auto, edge, enhance, refs, grow: 0, erase: false, tuning })
}

pub(super) fn run_edit(s: &mut Session, p: &Value) -> Result<Value> {
    let plan = plan_edit(s, p)?;
    let backend = backend(s)?;
    let FillPlan { template, prompt, negative, seed, steps, guidance, name, models, rect, variations, auto, edge, enhance, refs, grow, erase, tuning } = plan;
    let label = name.clone();
    // `wholePicture` (Generative Angles, and a swap, whose rewrite sets it): the selection does
    // not confine the result.
    let swapped = edit_rewrite(s, p)?;
    let whole = opt_bool(EDIT, swapped.as_ref().unwrap_or(p), "wholePicture", false)?;
    // An edit re-renders the whole picture; what shows of it is its subject (Select Subject on
    // the result), so a new thing (a Pikachu for the duck) sits in the untouched original. Not
    // for Angles, whose camera move is the whole picture; no detector, no subject mask.
    // Not on the 8 GB Mac profile either: the SAM pass took the edit from 5.3 to 10 GB at its peak
    // (6.7 GB with f16 SAM; measured 2026-11-08).
    let mac8 = crate::memory_profile::active(s) == Some(crate::memory_profile::Profile::Mac8);
    let finder = if whole || mac8 { None } else { crate::select_ml_cmds::subject_finder(s, None).ok() };
    let job = FillJob {
        thinker: Arc::clone(&backend),
        backend,
        command: EDIT,
        template_id: template.meta.id.clone(),
        auto,
        auto_order: auto_order(EDIT, auto_quality(s)),
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
            // The whole canvas is re-rendered; a selection, when there is one, confines what
            // of it shows (the edit template takes no mask, see `fill_region`).
            let selected = doc.selection.is_some() && !whole;
            let soft_edge = matches!(edge, super::params::Edge::Soft);
            let coverage: Vec<f32> = match doc.selection.as_ref().filter(|_| !whole) {
                Some(sel) => {
                    let k = sel.channels().max(1);
                    sel.read_region(rect).chunks_exact(k).map(|c| c.first().copied().unwrap_or(0.0).clamp(0.0, 1.0)).collect()
                }
                None => vec![1.0; rect.width() as usize * rect.height() as usize],
            };
            // The picture before the edit: what the edit changed shows along with the subject.
            let before: Option<Vec<u8>> = finder.as_ref().map(|_| photocraft_compose::render(doc, rect).to_rgba8().pixels);
            let filled = fill_region(doc, active, ctx, job, rect, coverage, None)?;
            if let Some(finder) = finder.as_ref() {
                for (id, ..) in &filled.0 {
                    mask_to_subject(doc, *id, rect, selected, before.as_deref(), soft_edge, finder, ctx)?;
                }
            }
            Ok(filled)
        },
        move |(made, w, h, request, template_id)| made_json(&made, w, h, request, &template_id),
    )
}

/// An edit result's layer mask: its subject (Select Subject's prompts on the result in turn, the
/// first that finds anything, its instances joined) and everything the edit visibly changed
/// ([`changed_area`] against `before`, the picture before the edit). With a selection that joins
/// the selection's mask (the selected area changes, and a new subject larger than it still shows
/// whole); without one it is the mask. A soft edge (the default) grows and fades the subject's
/// outline ([`soft_subject_edge`]); a hard one keeps Select Subject's. Nothing found: the mask
/// stays as it was. An edit that changed most of the picture (night, snow, a painting style, a
/// new background) shows whole.
///
/// The subject alone (2026-11-12 to 11-17) kept "make it night", "make it snow", "change the
/// background to a beach", "turn this into a watercolor", a sign's new text and "remove the bowl
/// of fruit" from showing at all on every edit model: only the person or the fridge changed
/// (Enhance round 5 on the 5090, 2026-11-17).
#[allow(clippy::too_many_arguments)]
pub(super) fn mask_to_subject(
    doc: &mut Document,
    id: LayerId,
    rect: Rect,
    selected: bool,
    before: Option<&[u8]>,
    soft_edge: bool,
    finder: &crate::select_ml_cmds::SubjectFinder,
    ctx: &JobCtx,
) -> Result<()> {
    let layer = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    let surf = layer.surface().ok_or(EngineError::NoLayer(id))?;
    let mut px = vec![[0u8; 4]; rect.width() as usize * rect.height() as usize];
    surf.read_rgba8_into(rect, &mut px);
    let picture: Vec<u8> = px.into_iter().flatten().collect();
    let (w, h) = (rect.width() as usize, rect.height() as usize);
    // With a selection, the selection says where the edit goes; what changed outside it stays hidden.
    let changed = before.filter(|b| !selected && b.len() == picture.len()).map(|b| changed_area(b, &picture, w, h));
    if changed.as_ref().is_some_and(|c| c.iter().sum::<f32>() > 0.5 * c.len() as f32) {
        // Most of the picture changed: the whole result shows, as the model made it.
        return Ok(());
    }
    let subject = subject_in(&picture, rect, finder, ctx)?.map(|s| if soft_edge { soft_subject_edge(&s, w, h) } else { s });
    let subject = match (subject, changed) {
        (Some(mut s), Some(c)) => {
            for (a, b) in s.iter_mut().zip(&c) {
                *a = a.max(*b);
            }
            s
        }
        (Some(s), None) => s,
        (None, Some(c)) => c,
        (None, None) => return Ok(()),
    };
    let layer = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
    let shown: Vec<f32> = match layer.mask.as_ref().filter(|_| selected) {
        Some(m) => m.surface.read_region(rect).iter().zip(&subject).map(|(a, b)| a.max(*b)).collect(),
        None => subject,
    };
    let mut surface = Surface::with_default(PixelFormat::GRAY8, &[0.0]);
    surface.write_region(rect, &shown);
    layer.mask = Some(LayerMask { surface, enabled: true, linked: true, density: 1.0, feather: 0.0 });
    Ok(())
}

/// Where an edit visibly changed the picture, 0..1 per pixel of a `w`×`h` RGBA8 pair: the
/// colour difference of 8-pixel block averages (so the model's small shifts and re-rendered
/// grain do not count), ramped from 0 at 0.07 to 1 at 0.15 of full scale on the most changed
/// channel, grown by one block and smoothed back to pixels.
pub(super) fn changed_area(before: &[u8], after: &[u8], w: usize, h: usize) -> Vec<f32> {
    const B: usize = 8;
    let (bw, bh) = (w.div_ceil(B), h.div_ceil(B));
    let mut blocks = vec![0.0f32; bw * bh];
    for by in 0..bh {
        for bx in 0..bw {
            let (mut a, mut z, mut n) = ([0.0f32; 3], [0.0f32; 3], 0.0f32);
            for y in by * B..((by + 1) * B).min(h) {
                for x in bx * B..((bx + 1) * B).min(w) {
                    let i = (y * w + x) * 4;
                    for c in 0..3 {
                        a[c] += f32::from(before[i + c]);
                        z[c] += f32::from(after[i + c]);
                    }
                    n += 1.0;
                }
            }
            let d = (0..3).map(|c| (a[c] - z[c]).abs() / (n * 255.0)).fold(0.0f32, f32::max);
            blocks[by * bw + bx] = ((d - 0.07) / 0.08).clamp(0.0, 1.0);
        }
    }
    // Grown by one block, so the edge of a changed thing is inside.
    let grown: Vec<f32> = (0..bh * bw)
        .map(|k| {
            let (bx, by) = ((k % bw) as isize, (k / bw) as isize);
            let mut m = 0.0f32;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (x, y) = (bx + dx, by + dy);
                    if x >= 0 && y >= 0 && (x as usize) < bw && (y as usize) < bh {
                        m = m.max(blocks[y as usize * bw + x as usize]);
                    }
                }
            }
            m
        })
        .collect();
    // Back to pixels, bilinear between block centres.
    let at = |x: isize, y: isize| grown[y.clamp(0, bh as isize - 1) as usize * bw + x.clamp(0, bw as isize - 1) as usize];
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        let fy = (y as f32 + 0.5) / B as f32 - 0.5;
        let (y0, ty) = (fy.floor() as isize, fy - fy.floor());
        for x in 0..w {
            let fx = (x as f32 + 0.5) / B as f32 - 0.5;
            let (x0, tx) = (fx.floor() as isize, fx - fx.floor());
            let top = at(x0, y0) * (1.0 - tx) + at(x0 + 1, y0) * tx;
            let bottom = at(x0, y0 + 1) * (1.0 - tx) + at(x0 + 1, y0 + 1) * tx;
            out[y * w + x] = top * (1.0 - ty) + bottom * ty;
        }
    }
    out
}

/// A subject's coverage grown [`subject_edge_radius`] pixels past Select Subject's edge, then
/// faded out over about twice that again (the inside untouched). Select Subject's edge sits on
/// the new subject; where the old one reached further (a bigger object, a different outline,
/// edge pixels the model repainted) a hard mask there showed a stair-stepped sliver of the old
/// one along the outline (tier bench, 2026-11-17: "make the fridge red" left a white rim on every
/// model and card). Outside the subject the edit's picture is the original's, so the band costs
/// nothing.
pub(super) fn soft_subject_edge(subject: &[f32], w: usize, h: usize) -> Vec<f32> {
    let r = subject_edge_radius(w, h);
    if w == 0 || h == 0 || subject.len() != w * h {
        return subject.to_vec();
    }
    let grown = super::pixel_ops::dilate(subject, w, h, r);
    let mut cur = grown.clone();
    for _ in 0..2 {
        cur = super::pixel_ops::box_blur(&cur, w, h, r, true);
        cur = super::pixel_ops::box_blur(&cur, w, h, r, false);
    }
    grown.iter().zip(&cur).map(|(g, b)| g.max((b * 2.0).min(1.0))).collect()
}

/// How far an edit's subject mask reaches past its edge: 0.5% of the longer side, 3 to 8 pixels.
pub(super) fn subject_edge_radius(w: usize, h: usize) -> usize {
    ((w.max(h) as f64 * 0.005).round() as usize).clamp(3, 8)
}

/// The subjects (at most [`SUBJECT_MASK_AT_MOST`]) Select Subject finds in `picture` (RGBA8 of
/// `rect`), joined into one coverage; None when nothing is found.
pub(super) fn subject_in(picture: &[u8], rect: Rect, finder: &crate::select_ml_cmds::SubjectFinder, ctx: &JobCtx) -> Result<Option<Vec<f32>>> {
    for prompt in crate::select_ml_cmds::SUBJECT_FALLBACKS.iter().take(2) {
        ctx.check()?;
        let found = finder.find_in(picture, rect, prompt, SUBJECT_MASK_AT_MOST, ctx)?;
        if found.is_empty() {
            continue;
        }
        let mut joined = vec![0.0f32; picture.len() / 4];
        for (cov, ..) in &found {
            for (j, c) in joined.iter_mut().zip(cov) {
                *j = j.max(*c);
            }
        }
        return Ok(Some(joined));
    }
    Ok(None)
}

/// How many subjects an edit result's mask may join.
pub(super) const SUBJECT_MASK_AT_MOST: usize = 3;

/// The Angles template's tuning from a request (`lightning` 4, 8 or 0 = none; `anglesStrength`
/// and `lightningStrength` 0 to 2), as the worker's params; empty when none is given.
pub(crate) fn angles_extra(p: &Value) -> Result<BTreeMap<String, Value>> {
    let mut out = BTreeMap::new();
    if let Some(v) = p.get("lightning").filter(|v| !v.is_null()) {
        match v.as_u64() {
            Some(n @ (0 | 4 | 8)) => {
                out.insert("lightning".to_string(), json!(n));
            }
            _ => return Err(bad(ANGLES, "`lightning` must be 4, 8 or 0 (none)")),
        }
    }
    for (key, wire) in [("anglesStrength", "angles_strength"), ("lightningStrength", "lightning_strength")] {
        if let Some(x) = opt_num(ANGLES, p, key, 0.0, 2.0)? {
            out.insert(wire.to_string(), json!(x));
        }
    }
    Ok(out)
}

/// Steps and CFG for the Angles template's `lightning` (8, the default → 8 at CFG 1, 4 → 4 at
/// CFG 1, 0 → the base model's 40 at CFG 4).
pub fn angles_sampling(lightning: u64) -> (u32, f64) {
    match lightning {
        4 => (4, 1.0),
        0 => (40, 4.0),
        _ => (8, 1.0),
    }
}

/// `generate.angles`: the whole picture from another camera position, as Generative Edit runs it
/// with the Multiple Angles template and the LoRA's words for `rotation`, `tilt` and `zoom`.
pub(super) fn run_angles(s: &mut Session, p: &Value) -> Result<Value> {
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
    let camera = angle_words(rotation, tilt, zoom).unwrap_or_default();
    let template = angles_template(s);
    // On 2.1 the LoRA's own words, its template's steps; no Lightning there.
    let on21 = template == ANGLES21_TEMPLATE;
    let words = if on21 { angle_prompt_21(rotation, tilt, zoom).unwrap_or(words) } else { words };
    let mut q = json!({"prompt": words, "template": template, "wholePicture": true, "name": format!("Angles: {camera}")});
    let keys: &[&str] = if on21 {
        &["seed", "variations", "negative", "steps", "guidance"]
    } else {
        &["seed", "variations", "negative", "lightning", "anglesStrength", "lightningStrength", "steps", "guidance"]
    };
    for key in keys {
        if let Some(v) = p.get(*key).filter(|v| !v.is_null()) {
            q[*key] = v.clone();
        }
    }
    // Each Lightning has its own steps and CFG unless the request names them.
    if !on21 {
        let (steps, guidance) = angles_sampling(p.get("lightning").and_then(Value::as_u64).unwrap_or(8));
        for (key, v) in [("steps", json!(steps)), ("guidance", json!(guidance))] {
            if q.get(key).is_none() {
                q[key] = v;
            }
        }
    }
    run_edit(s, &q)
}

/// `generate.characterSheet`: a turnaround sheet of the character on a layer, as a new
/// document: four full-body views in a row (front, three-quarter, profile, back), drawn by
/// MiniMax-H3-Image from the layer as Picture 1. 16 MP under Quality Ultra and Best, 4 MP
/// under the faster ones.
pub(super) fn run_character_sheet(s: &mut Session, p: &Value) -> Result<Value> {
    if !template_files_present(s, ULTRA_IMAGE_TEMPLATE) {
        return Err(EngineError::Other("Character Sheet needs MiniMax H3 Image: its files are not in the Models folder".into()));
    }
    let layer = layer_param(s, p)?;
    let words = |key: &str| -> Result<String> { Ok(opt_str(CHARACTER_SHEET, p, key, 500)?.unwrap_or("").trim().trim_end_matches('.').trim().to_string()) };
    let prompt = character_sheet_prompt(&words("details")?, &words("background")?);
    let mut q = json!({
        "prompt": prompt,
        "template": ULTRA_IMAGE_TEMPLATE,
        "references": [{"layer": layer.0}],
        "width": SHEET_SIZE.0,
        "height": SHEET_SIZE.1,
        "target": "document",
        "enhance": false,
        "name": "Character Sheet",
    });
    if let Some(v) = p.get("seed") {
        q["seed"] = v.clone();
    }
    run_image(s, &q)
}

/// The sheet's shape, four upright views in a row; the quality scales it to the model's size.
pub(super) const SHEET_SIZE: (u32, u32) = (2048, 1024);

/// The words of a character sheet: the layer is Picture 1; `details` names what the picture
/// does not show (the shoes of a waist-up portrait), `background` the backdrop (plain white
/// when empty).
pub(crate) fn character_sheet_prompt(details: &str, background: &str) -> String {
    let background = if background.is_empty() { "one plain white studio backdrop with soft even light" } else { background };
    let details =
        if details.is_empty() { String::new() } else { format!(" The character also has these details, in every view where they can be seen: {details}.") };
    format!(
        "A character turnaround sheet of the character in Picture 1: four full-body views of the same character side by side in one row, each the same width, clearly separated and never overlapping, on {background}. \
         The face, hair, skin, body, clothing and accessories are exactly as in Picture 1 and identical in all four views, in the same style and medium as Picture 1.{details} \
         From left to right: first, the character straight from the front, facing the camera. \
         Second, a three-quarter view from the character's left side, the face turned toward the left edge of the picture. \
         Third, a strict side profile at exactly 90 degrees, the nose pointing to the left edge of the picture, the far arm and far leg hidden behind the near ones. \
         Fourth, on the far right, the character straight from behind: the back of the head and the back are visible, the face is not. \
         The character holds the same relaxed standing pose in all four views, framed the same way in each, from the top of the head to the feet. Sharp focus and fine detail in the face, the hair and the clothing."
    )
}

pub(super) fn doc_enabled(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}
