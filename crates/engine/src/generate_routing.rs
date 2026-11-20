//! Which template runs: Preferences › Quality, the `auto` orders and the native worker's
//! Qwen-Image-2.1 and Klein routes, and the time estimates the UI shows.

use std::collections::BTreeMap;

use photocraft_genai::{GenerativeBackend, Task, template};

use crate::Session;

use super::{
    AUTO_EXPAND_ORDER, AUTO_FILL_ORDER, AUTO_TEMPLATE, BEST_EXPAND_ORDER, BEST_FILL_ORDER, BEST_FILL_REF_TEMPLATE, DEFAULT_FILL_REF_TEMPLATE,
    DEFAULT_FILL_TEMPLATE, DEFAULT_IMAGE_TEMPLATE, EDIT, EXPAND, FAST_FILL_ORDER, FILL, edit_request_pixels, loras, outpaint_lora, request_size,
    request_size_within,
};

/// The Multiple Angles LoRA file [`ANGLES_TEMPLATE`] loads (fal, Apache-2.0).
pub const ANGLES_LORA: &str = "qwen-image-edit-2511-multiple-angles-lora.safetensors";

/// The Qwen-Image-2.1 template that `auto` runs for `command` (a removal when `removal`) instead
/// of the Qwen-Image-Edit-2511 order, when research-only models are allowed and the native
/// worker runs the family: the owner's choice of 2026-10-27, on the removal numbers of
/// `benchmarks.md` (2.1 as an edit 15 of 20 hard strokes clean at 6–28 s, 2511 16 at 33–58 s
/// through ComfyUI). A removal is an edit by instruction on the crop, as measured; a prompted
/// fill is the 2.1 fill (the mask in view); an expand the in-context 2.1 expand. Without the
/// opt-in or the worker, the 2511 orders stand (the permissive default).
pub(crate) fn native21_auto(s: &Session, command: &str, removal: bool) -> Option<&'static str> {
    // An 8 to 16 GB card with FLUX 3 Image Klein installed edits with Klein.
    if command == EDIT && crate::memory_profile::cuda_klein(s) && loras::runs_native(s, crate::memory_profile::KLEIN_EDIT_TEMPLATE) {
        return Some(crate::memory_profile::KLEIN_EDIT_TEMPLATE);
    }
    // A small memory profile edits with MOSS Edit (permissive, so no research opt-in).
    if command == EDIT
        && let Some(t) = crate::memory_profile::edit_template(crate::memory_profile::active(s))
        && loras::runs_native(s, t)
    {
        return Some(t);
    }
    if command == FILL
        && let Some(t) = crate::memory_profile::fill_template(crate::memory_profile::active(s))
        && loras::runs_native(s, t)
    {
        return Some(t);
    }
    native21_only(s, command, removal)
}

/// [`native21_auto`] without the memory profile: the Qwen-Image-2.1 route alone.
pub(super) fn native21_only(s: &Session, command: &str, removal: bool) -> Option<&'static str> {
    let t = match command {
        // Remove: the Bbox Object Remover LoRA's route when the worker serves it (owner's
        // call 2026-11-19, after the 5090 A/B), else the plain 2.1 edit.
        FILL if removal && loras::runs_native(s, REMOVE21_TEMPLATE) => REMOVE21_TEMPLATE,
        FILL if removal => "qwen-2.1/edit",
        FILL => NATIVE21_FILL,
        // Expand stays on 2511's own tiers: its sampler keeps the picture's latent outside the
        // mask, while 2.1 samples the whole canvas from noise and drew the subject again over
        // the picture even with a backdrop-only prompt on four sides (2026-11-03).
        EDIT => "qwen-2.1/edit",
        _ => return None,
    };
    (s.prefs().integrations.allow_research_models && loras::runs_native(s, t)).then_some(t)
}

/// Generative Remove on Qwen-Image-2.1 with the Bbox Object Remover LoRA (a red box drawn
/// around the selection, see `highlight_covered`).
pub const REMOVE21_TEMPLATE: &str = "qwen-2.1/remove";

/// The 2.1 fill `auto` falls back to on the native worker.
pub(super) const NATIVE21_FILL: &str = "qwen-2.1/fill";

/// What `auto` runs for a prompted fill on the native worker with research-only models allowed:
/// Krea 2 with the AnyPaint LoRA when the worker serves it (its files and the LoRA there), else
/// the 2.1 fill. Measured in the app 2026-11-03 on three photos: AnyPaint swapped the blouse,
/// the bicycle and the pine cleanly in 1.1–3.2 s warm, the 2.1 fill in 10.6–13.3 s.
pub const NATIVE_FILL_ORDER: &[&str] = &["krea2-turbo/fill", NATIVE21_FILL];

/// What `auto` tries for a fill on the native worker, given its prompt and whether the prompt
/// writer runs. AnyPaint wants the whole finished picture described: given a short typed prompt
/// as it stands ("a blue vespa scooter", Enhance off) it cleared the area in 4 of 4 runs and
/// never drew the scooter, where the 2.1 fill drew it 4 of 4 at the same speed (2.07 s against
/// 1.99 s warm) and AnyPaint with Enhance on 4 of 4 (pre-release benchmark, 2026-11-03). So a
/// typed prompt taken verbatim goes to the 2.1 fill; Enhance on, or nothing typed, keeps AnyPaint.
pub(crate) fn native_fill_order(prompt: &str, enhance: bool) -> &'static [&'static str] {
    if enhance || prompt.trim().is_empty() { NATIVE_FILL_ORDER } else { &[NATIVE21_FILL] }
}

/// The crop a removal on Qwen-Image-2.1 goes out at when `maxPixels` is left at the default:
/// 0.75 MP, the size the 2.1 removal runs were measured at (1 MP is 2511's).
pub(super) const NATIVE21_REMOVAL_PIXELS: u64 = 786_432;

/// The same for Klein on a Mac: 512², where a fill takes about 200 s on a 16 GB M1.
pub(super) const KLEIN_REMOVAL_PIXELS: u64 = 262_144;

pub(crate) const DEFAULT_EDIT_TEMPLATE: &str = "qwen-edit-2511/edit";

/// What `auto` tries for an edit, fastest permissive tier first.
pub(crate) const AUTO_EDIT_ORDER: &[&str] = &["qwen-edit-2511/edit-lightning-8", DEFAULT_EDIT_TEMPLATE];

pub(crate) const BEST_EDIT_ORDER: &[&str] = &[DEFAULT_EDIT_TEMPLATE];

/// MiniMax-H3-Image, the 16 MP model Quality `ultra` runs where its files are installed.
pub const ULTRA_IMAGE_TEMPLATE: &str = "minimax-h3-image/image";

pub const ULTRA_EDIT_TEMPLATE: &str = "minimax-h3-image/edit";

/// Under `ultra`, an edit goes to MiniMax-H3-Image when the worker has it, else as under `best`.
pub(crate) const ULTRA_EDIT_ORDER: &[&str] = &[ULTRA_EDIT_TEMPLATE, DEFAULT_EDIT_TEMPLATE];

/// Preferences › AI Integrations › Quality, normalised (`best` unless it says `balanced` or
/// `fast`). `ultra` reads as `best` here: it is Best with the 16 MP model for new pictures and
/// edits ([`ultra`]).
pub fn quality(s: &Session) -> &'static str {
    match s.prefs().integrations.quality.trim() {
        "balanced" => "balanced",
        "fast" => "fast",
        _ => "best",
    }
}

/// Whether Quality is `ultra`: everything as under `best`, and where MiniMax-H3-Image's files
/// are installed, new pictures ([`default_image_template`]) and edits ([`auto_quality`]) run
/// on it at up to 16 MP.
pub fn ultra(s: &Session) -> bool {
    s.prefs().integrations.quality.trim() == "ultra"
}

/// Generate panel › Speed as the step tier [`auto_order`] and the template tuning read:
/// `fast`, `balanced`, or `best` (the full-step, no-turbo version; the preference says `full`).
/// Unset, it follows the tier the old Quality preference stood for (`ultra` = `best`).
pub fn speed(s: &Session) -> &'static str {
    match s.prefs().integrations.speed.trim() {
        "fast" => "fast",
        "balanced" => "balanced",
        "full" => "best",
        _ => quality(s),
    }
}

/// The tier [`auto_order`] is asked with: [`speed`]. Quality no longer swaps models (MiniMax H3
/// is chosen in the model picker like any other model).
pub fn auto_quality(s: &Session) -> &'static str {
    speed(s)
}

/// Whether every file of `template` is in the models folder (as the template names them, in
/// its folder or one subfolder below it, where the worker looks too).
pub fn template_files_present(s: &Session, template: &str) -> bool {
    let root = s.prefs().integrations.models_root();
    if root.is_empty() {
        return false;
    }
    let Ok(t) = template::find(template) else { return false };
    t.meta.models.iter().all(|m| {
        let dir = std::path::Path::new(&root).join(&m.folder);
        dir.join(&m.default).is_file()
            || std::fs::read_dir(&dir).is_ok_and(|rd| rd.filter_map(|e| e.ok()).any(|e| e.path().is_dir() && e.path().join(&m.default).is_file()))
    })
}

/// The quality a removal (Content-Aware Fill, the Remove tool, Spot Healing through the
/// model) runs at: `best` becomes `balanced`. The 40-step tier was measured no better than
/// the 8-step one on removals and worse under the smoothed hole (a face erased to a blur,
/// a seam in a lawn; `benchmarks.md` §5i), and at two candidates a stroke it took a
/// Spot Healing dab from about 25 s to 100 s, which reads as the tool not working.
/// `fast` stays `fast`.
pub fn removal_quality(s: &Session) -> &'static str {
    match speed(s) {
        "fast" => "fast",
        _ => "balanced",
    }
}

/// The templates `auto` picks from for `command` under `quality` (the first one the server has
/// every file for runs): `best` = the full-step base template, `balanced` = the 8-step Lightning
/// tier then the base, `fast` = the 4-step tier where one exists.
pub fn auto_order(command: &str, quality: &str) -> &'static [&'static str] {
    match (command, quality) {
        (EDIT, "ultra") => ULTRA_EDIT_ORDER,
        // the rest of `ultra` is `best`
        (c, "ultra") => auto_order(c, "best"),
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

/// A template id from a preference, or the built-in default when the preference is empty.
/// The image template a run with none named uses: Preferences › Default Image Template, where
/// `auto` (or nothing) is the memory profile's ([`crate::memory_profile::image_template`]).
pub fn default_image_template(s: &Session) -> String {
    match s.prefs().integrations.default_image_template.trim() {
        "" | AUTO_TEMPLATE => crate::memory_profile::image_template(crate::memory_profile::active(s)).to_string(),
        t => t.to_string(),
    }
}

pub(super) fn template_or(pref: &str, fallback: &str) -> String {
    let t = pref.trim();
    if t.is_empty() { fallback.to_string() } else { t.to_string() }
}

/// The template a fill with reference images runs when none is named: the 40-step base under
/// Quality `best`, the 8-step tier otherwise (both take the same model files as the plain fill).
pub fn fill_ref_template(quality: &str) -> &'static str {
    if quality == "best" { BEST_FILL_REF_TEMPLATE } else { DEFAULT_FILL_REF_TEMPLATE }
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

/// The fill template a fill by instruction falls back to when nothing could be named.
pub(super) fn native21_fill_fallback() -> &'static str {
    "qwen-2.1/fill"
}

/// Whether `template` is a Krea 2 fill (AnyPaint), whose prompt describes the finished picture.
pub(super) fn krea2_fill(template: &str) -> bool {
    template::find(template).is_ok_and(|t| t.meta.family == "krea2" && t.meta.task == Task::Fill)
}

/// The template `auto` most likely picks for `command` under `quality` (the first of its order:
/// what a server with every file runs), for estimates before the server has answered.
pub fn likely_auto(command: &str, quality: &str) -> &'static str {
    auto_order(command, quality).first().copied().unwrap_or(DEFAULT_FILL_TEMPLATE)
}

/// [`likely_auto`] in this session: the Qwen-Image-2.1 template where `auto` runs it on the
/// native worker ([`native21_auto`]), else the first of the server's order.
pub fn likely_auto_in(s: &Session, command: &str, quality: &str) -> &'static str {
    auto_shown(s, command, false).unwrap_or_else(|| likely_auto(command, quality))
}

/// What `auto` runs for `command` in this session where it is not the server's order: an expand
/// through the Outpaint LoRA when [`outpaint_lora`] takes it (the 2.1 edit, or the Turbo
/// template), else [`native21_auto`]. For the Auto line and `generate.models` only: a run plans
/// the Outpaint route itself from the canvas sizes.
pub(super) fn auto_shown(s: &Session, command: &str, removal: bool) -> Option<&'static str> {
    if command == EXPAND
        && let Some((t, ..)) = outpaint_lora(s, (1, 1), (1, 1))
    {
        return Some(t);
    }
    native21_auto(s, command, removal)
}

/// Seconds a run of `template` for a `width × height` area with `count` variations and
/// `references` reference images is expected to take on a warm server (the template's
/// `estimatedSeconds`; a fill, expand or edit counts at the size it is sent at, see
/// [`request_size`]). `None` for an unknown template. The one
/// formula the Generate panel, `generate.models` and scripts share.
pub fn estimate_seconds(template: &str, width: u32, height: u32, count: u32, references: u32) -> Option<f32> {
    let t = template::find(template).ok()?;
    let (w, h) = match t.meta.task {
        // an H3 edit goes at its own, larger size
        Task::Edit => request_size_within(width, height, edit_request_pixels(template, template == ULTRA_EDIT_TEMPLATE)),
        Task::Fill | Task::Expand => request_size(width, height),
        _ => (width, height),
    };
    Some(t.meta.estimate_seconds(w, h, count, references))
}
