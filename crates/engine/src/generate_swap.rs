//! `generate.edit` `swap`: put the head (face and hair) or the whole person from a reference
//! picture onto the person in the open picture, with Alissonerdx's BFS (Best Face Swap) LoRAs
//! for Qwen-Image-2.1 (MIT, huggingface.co/Alissonerdx/BFS-Best-Face-Swap). The open picture is
//! image 1 (the body and scene that stay), the first reference image 2 (who is put in), the
//! order the BFS Qwen-Image-2.1 LoRAs were trained on; the prompt is the LoRA author's own, with
//! anything typed added after it. The author asks that it be used only with people who agreed.

use serde_json::{Value, json};

use super::{EDIT, bad};
use crate::{EngineError, Result, Session};

/// The template a swap runs (the BFS LoRAs are for Qwen-Image-2.1; nine reference slots).
pub const SWAP_TEMPLATE: &str = "qwen-2.1/edit";

/// The steps a swap runs when the request leaves `steps` at 0: in the app on 2026-11-11 (5090,
/// head swap 896x1152) 16 steps looked as sharp as the template's 25 at 5.1 s warm against
/// about 7 s, and the 8-step turbo LoRA (Pruna) came out softer at the same time.
pub const SWAP_STEPS: u64 = 16;

/// One kind of swap: its `swap` value, its LoRA file and its prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Swap {
    pub kind: &'static str,
    pub lora: &'static str,
    pub prompt: &'static str,
}

/// Head swap v1.1 (Civitai 2027766 version 3356102): the face, hair and head shape from image 2,
/// the gaze, head turn and expression from image 1, the author's prompt with
/// [`HEAD_FACIAL_HAIR`] after it.
pub const HEAD: Swap = Swap {
    kind: "head",
    lora: "bfs_head_v1.1_qwen_2.1.safetensors",
    prompt: "head_swap: start with <image1> as the base image, keeping its lighting, environment, and background. remove the head from <image1> completely and replace it with the head from <image2>, strictly preserving the hair, eye color, nose structure from <image2>. copy the direction of the eye, head rotation, micro expressions from <image1>, high quality, sharp details, 4k with the facial hair, or the clean-shaven chin, of <image2>",
};

/// Body swap v1.0 (version 3363725): the clothes, body shape and face from image 2 in image 1's
/// pose and scene (trained on full-body, front-facing references on a plain light background).
pub const BODY: Swap = Swap {
    kind: "body",
    lora: "bfs_body_swap_v1.0_qwen_2.1.safetensors",
    prompt: "body_swap: start with <image1> as the base image, keeping its lighting, environment, and background. replace the body from <image1> with the body from <image2>, strictly preserving the clothing, body shape and proportions from <image2>. strictly replicate the exact pose, arm positions, leg positions, hand gestures, head rotation, eye direction and micro expressions from <image1>",
};

/// What [`HEAD`] adds to the author's prompt: image 1's beard stayed on 2 of 5 seeds and partly
/// on a third without it, on none with it (2 partly), in the app 2026-11-13. The same words
/// placed among what the prompt preserves ("hair, facial hair, …") drew a beard on 4 of 5.
pub const HEAD_FACIAL_HAIR: &str = "with the facial hair, or the clean-shaven chin, of <image2>";

pub const SWAPS: [Swap; 2] = [HEAD, BODY];

/// The swap named by `kind` (`head`, also `face`; `body`).
pub fn find(kind: &str) -> Option<Swap> {
    match kind.trim().to_ascii_lowercase().as_str() {
        "head" | "face" => Some(HEAD),
        "body" => Some(BODY),
        _ => None,
    }
}

/// `p` rewritten for its `swap`, or None when it asks for none: the template pinned to
/// [`SWAP_TEMPLATE`] (unless one is named), the swap's LoRA added at strength 1 (unless the
/// request lists it), the author's prompt with the typed one after it, and Enhance off (a
/// rewritten prompt would lose the LoRA's trigger words), the whole picture kept (`wholePicture`),
/// and [`SWAP_STEPS`] unless steps are named.
pub(crate) fn apply(s: &Session, p: &Value) -> Result<Option<Value>> {
    let Some(kind) = p.get("swap").filter(|v| !v.is_null()) else { return Ok(None) };
    let kind = kind.as_str().ok_or_else(|| bad(EDIT, "`swap` is \"head\" or \"body\""))?;
    if kind.trim().is_empty() || kind.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    let swap = find(kind).ok_or_else(|| bad(EDIT, format!("unknown swap {kind:?}: \"head\" or \"body\"")))?;
    if p.get("references").and_then(Value::as_array).is_none_or(|a| a.is_empty()) {
        return Err(bad(EDIT, "a swap needs the picture of who to put in, as `references` (image 2)"));
    }
    let root = s.prefs().integrations.models_root();
    let loras = std::path::Path::new(&root).join("loras");
    if !root.is_empty() && loras.is_dir() && !loras.join(swap.lora).is_file() {
        return Err(EngineError::Other(format!(
            "{} swap needs the BFS LoRA {} in the Models folder's loras folder (Help › Get AI Models › Face & body swap)",
            if swap.kind == "head" { "Face" } else { "Body" },
            swap.lora
        )));
    }
    let mut q = p.clone();
    let o = q.as_object_mut().ok_or_else(|| bad(EDIT, "parameters must be an object"))?;
    o.remove("swap");
    let named = o.get("template").and_then(Value::as_str).map(str::trim).filter(|t| !t.is_empty() && *t != super::AUTO_TEMPLATE);
    if named.is_none() {
        o.insert("template".into(), json!(SWAP_TEMPLATE));
    }
    let typed = o.get("prompt").and_then(Value::as_str).map(str::trim).unwrap_or_default().to_string();
    let prompt = if typed.is_empty() { swap.prompt.to_string() } else { format!("{} {typed}", swap.prompt) };
    o.insert("prompt".into(), json!(prompt));
    o.insert("enhance".into(), json!(false));
    // The whole re-rendered picture is the result: masked to the new person (or to the old and
    // the new together) it left the old one's suit, then a hard outline and a darker patch where
    // he stood (measured in the app 2026-11-12); the BFS LoRAs keep the rest of image 1 as it was.
    if o.get("wholePicture").is_none_or(Value::is_null) {
        o.insert("wholePicture".into(), json!(true));
    }
    if o.get("steps").and_then(Value::as_u64).unwrap_or(0) == 0 {
        o.insert("steps".into(), json!(SWAP_STEPS));
    }
    let mut list: Vec<Value> = o.get("loras").and_then(Value::as_array).cloned().unwrap_or_default();
    if !list.iter().any(|l| l.get("file").and_then(Value::as_str) == Some(swap.lora)) {
        list.insert(0, json!({"file": swap.lora, "strength": 1.0}));
    }
    o.insert("loras".into(), Value::Array(list));
    if o.get("name").is_none_or(Value::is_null) {
        o.insert("name".into(), json!(if swap.kind == "head" { "Face Swap" } else { "Body Swap" }));
    }
    Ok(Some(q))
}
