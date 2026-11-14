//! `generate.edit` `repose`: redraw the person from a reference picture in the pose of a
//! mannequin, with MIUProject's VNCCS PoseStudio LoRA for Qwen-Image-2.1
//! (huggingface.co/MIUProject/VNCCS_PoseStudio_QI2.1; no licence stated, bring-your-own). The
//! open picture is image 1 (the posed mannequin, the guide), the first reference image 2 (who
//! to draw), the order and trigger the LoRA was trained on; the prompt is the one the Fisher Pose
//! node pack sends (github.com/noname19921992/ComfyUI-Any-Pose-Angle, MIT), with anything typed
//! added after it.
//!
//! Measured in the app on 2026-11-12/13 (5090, 1024², 25 steps, CFG 1): about 10 s warm. V2
//! reposed 8 of 12 runs where V1.1 handed the person back unchanged in 11 of 12; V3 took a step
//! in 3 of 6 walking runs to V2's 1. A skeleton drawn on the mannequin barely mattered and never
//! showed in the result; a floor shadow under the mannequin did, so guides go without one.

use serde_json::{Value, json};

use super::{EDIT, bad};
use crate::{EngineError, Result, Session};

/// The template a repose runs (the PoseStudio LoRA is for Qwen-Image-2.1).
pub const REPOSE_TEMPLATE: &str = "qwen-2.1/edit";

/// The PoseStudio LoRA files a repose looks for in the loras folder, best first.
pub const REPOSE_LORAS: [&str; 3] = ["VNCCS_QI2_PoseStudioV3.safetensors", "VNCCS_QI2_PoseStudioV2.safetensors", "VNCCS_QI2_PoseStudioV1.1.safetensors"];

/// The LoRA's trigger and the instruction after it (Fisher Pose's single-person prompt).
pub const REPOSE_PROMPT: &str =
    "Draw character from image2. Repose the person in image2 to match image1. including the head direction.\nPreserve their appearance and the scene.";

/// Added when the guide has a coloured skeleton drawn on the mannequin (`skeleton: true`).
pub const SKELETON_LINE: &str = "The colored skeleton lines and joint markers in image1 are pose guides only; omit them from the finished image.";

/// The first of [`REPOSE_LORAS`] in `loras` (a models folder's `loras`), or None.
pub fn installed_lora(loras: &std::path::Path) -> Option<&'static str> {
    REPOSE_LORAS.into_iter().find(|f| loras.join(f).is_file())
}

/// `p` rewritten for `repose: true`, or None when it asks for none: the template pinned to
/// [`REPOSE_TEMPLATE`] (unless one is named), the best installed PoseStudio LoRA at strength 1
/// (unless the request lists one), the LoRA's prompt (plus [`SKELETON_LINE`] for a skeleton
/// guide) with the typed one after it, Enhance off (a rewrite would lose the trigger), and the
/// whole picture kept: the result is a new picture, not a patch on the mannequin.
pub(crate) fn apply(s: &Session, p: &Value) -> Result<Option<Value>> {
    match p.get("repose") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => return Ok(None),
        Some(Value::Bool(true)) => {}
        Some(_) => return Err(bad(EDIT, "`repose` is true or false")),
    }
    if p.get("references").and_then(Value::as_array).is_none_or(|a| a.is_empty()) {
        return Err(bad(EDIT, "a repose needs the picture of who to draw, as `references` (image 2); the open picture is the posed mannequin"));
    }
    let mut q = p.clone();
    let o = q.as_object_mut().ok_or_else(|| bad(EDIT, "parameters must be an object"))?;
    let skeleton = match o.remove("skeleton") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => b,
        Some(_) => return Err(bad(EDIT, "`skeleton` is true or false")),
    };
    o.remove("repose");
    let mut list: Vec<Value> = o.get("loras").and_then(Value::as_array).cloned().unwrap_or_default();
    let listed = list.iter().any(|l| l.get("file").and_then(Value::as_str).is_some_and(|f| REPOSE_LORAS.contains(&f)));
    if !listed {
        let root = s.prefs().integrations.models_root();
        let loras = std::path::Path::new(&root).join("loras");
        match installed_lora(&loras) {
            Some(f) => list.insert(0, json!({"file": f, "strength": 1.0})),
            // No folder to look in (no Models folder set): name the best one and let the worker say.
            None if root.is_empty() || !loras.is_dir() => list.insert(0, json!({"file": REPOSE_LORAS[0], "strength": 1.0})),
            None => {
                return Err(EngineError::Other(format!(
                    "Repose needs the VNCCS PoseStudio LoRA ({}) in the Models folder's loras folder",
                    REPOSE_LORAS.join(", ")
                )));
            }
        }
    }
    o.insert("loras".into(), Value::Array(list));
    let named = o.get("template").and_then(Value::as_str).map(str::trim).filter(|t| !t.is_empty() && *t != super::AUTO_TEMPLATE);
    if named.is_none() {
        o.insert("template".into(), json!(REPOSE_TEMPLATE));
    }
    let typed = o.get("prompt").and_then(Value::as_str).map(str::trim).unwrap_or_default().to_string();
    let mut prompt = REPOSE_PROMPT.to_string();
    if skeleton {
        prompt = format!("{prompt}\n{SKELETON_LINE}");
    }
    if !typed.is_empty() {
        prompt = format!("{prompt}\n{typed}");
    }
    o.insert("prompt".into(), json!(prompt));
    o.insert("enhance".into(), json!(false));
    if o.get("wholePicture").is_none_or(Value::is_null) {
        o.insert("wholePicture".into(), json!(true));
    }
    if o.get("name").is_none_or(Value::is_null) {
        o.insert("name".into(), json!("Repose"));
    }
    Ok(Some(q))
}
