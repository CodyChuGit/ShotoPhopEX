//! `integrations.engine = native` routes the generative commands to the native worker's channel
//! (plan-native-engine §3.3/§3.4): a fake engine served in-process stands in for the worker.

use std::sync::Arc;

use serde_json::json;

use crate::Session;
use crate::generate_cmds::{engine_mode, set_native_backend};

/// The native worker slot is process-wide: these tests take turns.
static WORKER_SLOT: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn the_engine_preference_routes_to_the_native_worker() {
    let _turn = WORKER_SLOT.lock().unwrap_or_else(|e| e.into_inner());
    let (port, token) = photocraft_genai::native::serve_fake(
        vec![
            "krea2-turbo/image".into(),
            "qwen-edit-2511/edit".into(),
            "sam3.1/segment".into(),
            "qwen-2.1/image".into(),
            "qwen-2.1/edit".into(),
            "qwen-2.1/fill".into(),
            "qwen-2.1/matte".into(),
        ],
        1,
    )
    .unwrap();
    set_native_backend(Arc::new(photocraft_genai::native::NativeBackend::attached(port, &token).unwrap()));
    let mut s = Session::new();
    assert_eq!(engine_mode(&s), "auto");
    assert!(s.execute("prefs.set", json!({"path": "integrations.engine", "value": "warp"})).is_err(), "only the listed engines");
    s.execute("prefs.set", json!({"path": "integrations.engine", "value": "native"})).unwrap();
    assert_eq!(engine_mode(&s), "native");
    // Health names the worker's engine.
    let h = s.execute("generate.health", json!({})).unwrap();
    assert_eq!((h["ok"].as_bool(), h["backend"].as_str()), (Some(true), Some("native/fake")), "{h}");
    // Generate Image through the channel: a new document holding the fake's picture.
    let r = s
        .execute("generate.image", json!({"prompt": "a fox", "template": "krea2-turbo/image", "width": 128, "height": 64, "target": "document", "seed": 7}))
        .unwrap();
    assert_eq!((r["width"].as_u64(), r["height"].as_u64(), r["template"].as_str()), (Some(128), Some(64), Some("krea2-turbo/image")), "{r}");
    let doc = &s.active().unwrap().doc;
    assert_eq!((doc.size.width, doc.size.height), (128, 64));
    // Qwen-Image-2.1 text to image goes through the channel too (a research-only model: opt in).
    s.execute("prefs.set", json!({"path": "integrations.allowResearchModels", "value": true})).unwrap();
    let r = s.execute("generate.image", json!({"prompt": "a fox", "template": "qwen-2.1/image", "width": 128, "height": 64, "target": "layer", "seed": 9})).unwrap();
    assert_eq!((r["width"].as_u64(), r["height"].as_u64(), r["template"].as_str()), (Some(128), Some(64), Some("qwen-2.1/image")), "{r}");
    // Generative Edit sends the picture across and gets one back the same size.
    let e = s.execute("generate.edit", json!({"prompt": "make it winter", "template": "qwen-edit-2511/edit", "seed": 3})).unwrap();
    assert_eq!((e["width"].as_u64(), e["height"].as_u64()), (Some(128), Some(64)), "{e}");
    // A template the worker lacks is a sentence, and the backend is still there.
    let err = s.execute("generate.edit", json!({"prompt": "x", "template": "qwen-edit-2511/edit-lightning-8"})).unwrap_err().to_string();
    assert!(err.contains("not installed"), "{err}");
    assert_eq!(s.execute("generate.health", json!({})).unwrap()["ok"], true);
    // Back to the ComfyUI route: health now reports that backend.
    s.execute("prefs.set", json!({"path": "integrations.engine", "value": "comfy"})).unwrap();
    assert_eq!(s.execute("generate.health", json!({})).unwrap()["backend"], "comfyui");
    // Under `auto`, a template the worker serves (SAM 3.1) goes native once the worker has its
    // folders, the rest stay on ComfyUI.
    s.execute("prefs.set", json!({"path": "integrations.engine", "value": "auto"})).unwrap();
    assert_eq!(s.execute("generate.health", json!({"template": "sam3.1/segment"})).unwrap()["backend"], "comfyui", "no folders: ComfyUI");
    s.execute("prefs.set", json!({"path": "integrations.runtimePath", "value": "runtime"})).unwrap();
    s.execute("prefs.set", json!({"path": "integrations.modelsPath", "value": "models"})).unwrap();
    assert_eq!(s.execute("generate.health", json!({})).unwrap()["backend"], "comfyui");
    assert_eq!(s.execute("generate.health", json!({"template": "sam3.1/segment"})).unwrap()["backend"], "native/fake");
    // Krea 2 text to image is in `NATIVE_AUTO_FAMILIES` too (2026-10-27); its style reference,
    // which this worker does not list, stays on ComfyUI.
    assert_eq!(s.execute("generate.health", json!({"template": "krea2-turbo/image"})).unwrap()["backend"], "native/fake");
    assert_eq!(s.execute("generate.health", json!({"template": "krea2-turbo/image-style"})).unwrap()["backend"], "comfyui");
    // Qwen-Image-2.1 text to image is in `NATIVE_AUTO_FAMILIES` and the worker lists it: native,
    // and `generate.image` itself goes there (no ComfyUI server is configured in this test).
    assert_eq!(s.execute("generate.health", json!({"template": "qwen-2.1/image"})).unwrap()["backend"], "native/fake");
    let r = s.execute("generate.image", json!({"prompt": "a fox", "template": "qwen-2.1/image", "width": 128, "height": 64, "target": "layer", "seed": 5})).unwrap();
    assert_eq!(r["template"].as_str(), Some("qwen-2.1/image"), "{r}");
    // Its edit and fill templates are served by the worker too: Generative Edit and Fill with
    // them go native under `auto` (the family is listed and the worker names the templates),
    // while a template the worker does not list stays on ComfyUI.
    assert_eq!(s.execute("generate.health", json!({"template": "qwen-2.1/edit"})).unwrap()["backend"], "native/fake");
    assert_eq!(s.execute("generate.health", json!({"template": "qwen-2.1/fill"})).unwrap()["backend"], "native/fake");
    assert_eq!(s.execute("generate.health", json!({"template": "qwen-2.1/matte"})).unwrap()["backend"], "native/fake");
    assert_eq!(s.execute("generate.health", json!({"template": "qwen-edit-2511/fill"})).unwrap()["backend"], "comfyui");
    let e = s.execute("generate.edit", json!({"prompt": "make it winter", "template": "qwen-2.1/edit", "seed": 3})).unwrap();
    assert_eq!((e["width"].as_u64(), e["height"].as_u64(), e["template"].as_str()), (Some(128), Some(64), Some("qwen-2.1/edit")), "{e}");
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 40, "height": 30})).unwrap();
    let f = s.execute("generate.fill", json!({"prompt": "a hat", "template": "qwen-2.1/fill", "seed": 4})).unwrap();
    assert_eq!(f["template"].as_str(), Some("qwen-2.1/fill"), "{f}");
    s.execute("select.deselect", json!({})).unwrap();
    // Remove Background under `auto` with research models allowed: the matte template, served
    // natively (`resolve_auto` sees the worker's template list), its alpha as the layer mask.
    let m = s.execute("generate.removeBackground", json!({"seed": 2})).unwrap();
    assert_eq!(m["template"].as_str(), Some("qwen-2.1/matte"), "{m}");
    assert!(m["pixels"].as_u64().unwrap_or(0) > 0 && !m["selection"].as_bool().unwrap_or(true), "{m}");
    let sel = s.execute("select.byText", json!({"prompt": "fox"})).unwrap();
    assert_eq!((sel["count"].as_u64(), sel["selected"].as_bool()), (Some(1), Some(true)), "{sel}");
    // The fake's coverage is a centred disc: the selection's bounds sit inside the 128×64 picture.
    let b = &sel["instances"][0]["bounds"];
    assert!(b[0].as_i64().unwrap_or(-1) >= 40 && b[2].as_i64().unwrap_or(0) <= 48, "{sel}");
    // A removal on the native 2.1 fill still asks the rewriter, which the worker does not
    // serve: the question goes to the ComfyUI route (`thinker_for`), the fill stays native.
    // Measured 2026-10-26: with both on the worker the rewriter's name was lost and the
    // detector fell back to the point prompt.
    let comfy = photocraft_genai::fake::FakeComfy::start().unwrap();
    s.edit_prefs(|p| {
        p.integrations.comfy_server = comfy.url.clone();
        p.integrations.default_fill_template = "qwen-2.1/fill".into();
    });
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 40, "height": 30})).unwrap();
    let r = s.execute("edit.contentAwareFill", json!({"generative": true, "variations": 1})).unwrap();
    assert_eq!(r["template"].as_str(), Some("qwen-2.1/fill"), "{r}");
    let st = comfy.state();
    assert_eq!(st.prompts.len(), 1, "the rewriter's ask alone reached ComfyUI: {}", st.prompts.len());
    assert_eq!(st.prompts[0].1["3"]["class_type"], "TextGenerate", "{}", st.prompts[0].1);
    drop(st);
    // LoRAs ride along as a request parameter for the native engine (validated here, applied
    // by the worker); the ComfyUI route chains a LoraLoaderModelOnly per file after the model
    // loader, and a malformed list is refused.
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 40, "height": 30})).unwrap();
    let f = s.execute("generate.fill", json!({"prompt": "a hat", "template": "qwen-2.1/fill", "seed": 4, "loras": [{"file": "qwen-image-2.1-consistency.safetensors", "strength": 0.5}]})).unwrap();
    assert_eq!(f["template"].as_str(), Some("qwen-2.1/fill"), "{f}");
    let f = s.execute("generate.fill", json!({"prompt": "a hat", "template": "qwen-edit-2511/fill", "seed": 4, "loras": [{"file": "x.safetensors"}]})).unwrap();
    assert_eq!(f["template"].as_str(), Some("qwen-edit-2511/fill"), "{f}");
    let st = comfy.state();
    let g = &st.prompts.last().unwrap().1;
    let lora = g.as_object().unwrap().values().find(|n| n["class_type"] == "LoraLoaderModelOnly" && n["inputs"]["lora_name"] == "x.safetensors");
    assert!(lora.is_some_and(|n| n["inputs"]["strength_model"] == 1.0), "{g}");
    drop(st);
    for bad in [json!("x"), json!([{"strength": 1}]), json!([{"file": "../x.safetensors"}]), json!([{"file": "x.safetensors", "strength": 9}])] {
        let e = s.execute("generate.fill", json!({"prompt": "a hat", "template": "qwen-2.1/fill", "loras": bad})).unwrap_err();
        assert!(matches!(e, crate::EngineError::BadParams { .. }), "{e}");
    }
    let t = crate::generate_cmds::Tuning::from_params("x", &json!({"loras": [{"file": "a.safetensors"}, {"file": "b.safetensors", "strength": -1}]})).unwrap();
    assert_eq!(t.loras_param(), Some(json!([{"file": "a.safetensors", "strength": 1.0}, {"file": "b.safetensors", "strength": -1.0}])));
}

/// A worker that serves the enhancer template (its 2.1 text encoder answering as Qwen3-VL)
/// answers a native job's questions itself: nothing reaches the ComfyUI server, and the
/// worker is not told to unload after answering.
#[test]
fn a_worker_that_serves_the_enhancer_answers_its_own_jobs_questions() {
    let _turn = WORKER_SLOT.lock().unwrap_or_else(|e| e.into_inner());
    let (port, token) =
        photocraft_genai::native::serve_fake(vec!["qwen-2.1/fill".into(), "enhance/edit".into(), "enhance/image".into(), "sam3.1/segment".into()], 1).unwrap();
    let worker = Arc::new(photocraft_genai::native::NativeBackend::attached(port, &token).unwrap());
    set_native_backend(Arc::clone(&worker));
    let comfy = photocraft_genai::fake::FakeComfy::start().unwrap();
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 128, "height": 64})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.edit_prefs(|p| {
        p.integrations.engine = "auto".into();
        p.integrations.allow_research_models = true;
        p.integrations.runtime_path = "runtime".into();
        p.integrations.models_path = "models".into();
        p.integrations.comfy_server = comfy.url.clone();
        p.integrations.default_fill_template = "qwen-2.1/fill".into();
    });
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 40, "height": 30})).unwrap();
    let r = s.execute("edit.contentAwareFill", json!({"generative": true, "variations": 2})).unwrap();
    assert_eq!(r["template"].as_str(), Some("qwen-2.1/fill"), "{r}");
    assert!(comfy.state().prompts.is_empty(), "the rewriter and the judge stayed on the worker");
    assert!(comfy.state().requests.iter().all(|q| q != "POST /free"), "nothing asked ComfyUI to unload");
    // The fill ran after the questions on a worker that was never freed: still healthy, still resident.
    assert_eq!(s.execute("generate.health", json!({"template": "qwen-2.1/fill"})).unwrap()["backend"], "native/fake");
}

/// `generate.engine.status` carries the native worker's state, asked without starting one;
/// Purge unloads its models and leaves it ready; Stop and Start name the engine.
#[test]
fn the_engine_status_shows_the_native_worker_and_purge_unloads_it() {
    let _turn = WORKER_SLOT.lock().unwrap_or_else(|e| e.into_inner());
    let (port, token) = photocraft_genai::native::serve_fake(vec!["krea2-turbo/image".into()], 1).unwrap();
    set_native_backend(Arc::new(photocraft_genai::native::NativeBackend::attached(port, &token).unwrap()));
    let mut s = Session::new();
    s.edit_prefs(|p| p.integrations.engine = "comfy".into());
    assert_eq!(s.execute("generate.engine.status", json!({})).unwrap()["native"]["state"], "off");
    s.edit_prefs(|p| {
        p.integrations.engine = "native".into();
        p.integrations.runtime_path = "runtime".into();
        p.integrations.models_path = "models".into();
    });
    let n = &s.execute("generate.engine.status", json!({})).unwrap()["native"];
    assert_eq!((n["state"].as_str(), n["engine"].as_str(), n["mode"].as_str()), (Some("ready"), Some("fake"), Some("native")), "{n}");
    assert!(n["log"].as_str().is_some_and(|l| l.ends_with("photocraft-native-engine.log")), "{n}");
    s.execute("generate.image", json!({"prompt": "a fox", "template": "krea2-turbo/image", "width": 64, "height": 64, "target": "document", "seed": 1})).unwrap();
    let n = s.execute("generate.engine.status", json!({})).unwrap()["native"].clone();
    assert_eq!(n["resident"], json!(["krea2-turbo/image"]), "{n}");
    let n = s.execute("generate.engine.purge", json!({"engine": "native"})).unwrap()["native"].clone();
    assert_eq!((n["state"].as_str(), n["resident"].clone()), (Some("ready"), json!([])), "{n}");
    assert!(s.execute("generate.engine.stop", json!({"engine": "warp"})).is_err(), "only the listed engines");
    // Stop lets go of the worker (this one was only joined, so it is not respawned).
    let st = s.execute("generate.engine.stop", json!({"engine": "native"})).unwrap();
    assert_eq!(st["native"]["state"], "stopped", "{st}");
}

/// Under `auto` with research-only models allowed and the native worker configured, fill,
/// expand, edit and the removals run Qwen-Image-2.1 on the worker, never the 2511 order (the
/// owner's choice, 2026-10-27); without the opt-in the 2511 order stands.
#[test]
fn auto_runs_qwen_image_2_1_on_the_worker_instead_of_2511() {
    let _turn = WORKER_SLOT.lock().unwrap_or_else(|e| e.into_inner());
    let served = vec![
        "qwen-2.1/fill".into(),
        "qwen-2.1/edit".into(),
        "qwen-2.1/expand".into(),
        "qwen-2.1/image-ref".into(),
        "qwen-layered/split".into(),
        "enhance/edit".into(),
        "enhance/image".into(),
        "sam3.1/segment".into(),
    ];
    let (port, token) = photocraft_genai::native::serve_fake(served, 1).unwrap();
    set_native_backend(Arc::new(photocraft_genai::native::NativeBackend::attached(port, &token).unwrap()));
    let comfy = photocraft_genai::fake::FakeComfy::start().unwrap();
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 128, "height": 64})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.edit_prefs(|p| {
        p.integrations.engine = "auto".into();
        p.integrations.runtime_path = "runtime".into();
        p.integrations.models_path = "models".into();
        p.integrations.comfy_server = comfy.url.clone();
    });
    let m = s.execute("generate.models", json!({"probe": false})).unwrap();
    assert!(m["autoEdit"].is_null() && m["autoRemove"].is_null(), "no opt-in: the server's order, unknown offline: {m}");
    s.edit_prefs(|p| p.integrations.allow_research_models = true);
    let m = s.execute("generate.models", json!({"probe": false})).unwrap();
    let picks = (m["autoFill"].as_str(), m["autoExpand"].as_str(), m["autoEdit"].as_str(), m["autoRemove"].as_str());
    assert_eq!(picks, (Some("qwen-2.1/fill"), Some("qwen-2.1/expand"), Some("qwen-2.1/edit"), Some("qwen-2.1/edit")), "{m}");
    let e = s.execute("generate.edit", json!({"prompt": "make it winter", "seed": 3, "enhance": false})).unwrap();
    assert_eq!(e["template"].as_str(), Some("qwen-2.1/edit"), "{e}");
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 40, "height": 30})).unwrap();
    let f = s.execute("generate.fill", json!({"prompt": "a red balloon", "seed": 4, "enhance": false})).unwrap();
    assert_eq!(f["template"].as_str(), Some("qwen-2.1/fill"), "{f}");
    let r = s.execute("edit.contentAwareFill", json!({"generative": true, "variations": 2})).unwrap();
    assert_eq!(r["template"].as_str(), Some("qwen-2.1/edit"), "{r}");
    // A fill by instruction (`fillAsEdit`): an instruction goes to the edit template as it is;
    // a description nothing names (the rewriter off) falls back to the fill template.
    let f = s.execute("generate.fill", json!({"prompt": "replace the box with a red balloon", "seed": 4, "enhance": false, "fillAsEdit": true})).unwrap();
    assert_eq!(f["template"].as_str(), Some("qwen-2.1/edit"), "{f}");
    let f = s.execute("generate.fill", json!({"prompt": "a red balloon", "seed": 4, "enhance": false, "fillAsEdit": true})).unwrap();
    assert_eq!(f["template"].as_str(), Some("qwen-2.1/fill"), "nothing named: {f}");
    // A fill from a reference: the 2.1 edit template on the worker (it takes nine), not 2511's
    // fill-ref on ComfyUI.
    let reference = s.active().unwrap().active_layer.unwrap();
    let f = s.execute("generate.fill", json!({"prompt": "the red sweater", "seed": 5, "enhance": false, "references": [{"layer": reference.0}]})).unwrap();
    assert_eq!(f["template"].as_str(), Some("qwen-2.1/edit"), "{f}");
    // An image from references: 2.1's image-ref template on the worker, not 2511's on ComfyUI.
    let i = s.execute("generate.image", json!({"prompt": "the sweater on a mannequin", "width": 128, "height": 128, "target": "document", "seed": 6, "enhance": false, "references": [{"layer": reference.0}]})).unwrap();
    assert_eq!(i["template"].as_str(), Some("qwen-2.1/image-ref"), "{i}");
    // Split into Layers on the worker that serves it (Qwen-Image-Layered, Apache-2.0).
    let l = s.execute("generate.splitLayers", json!({"layers": 2, "seed": 8})).unwrap();
    assert_eq!(l["template"].as_str(), Some("qwen-layered/split"), "{l}");
    // Generative Angles: the 2.1 edit by a camera instruction, not 2511 with the LoRA.
    let a = s.execute("generate.angles", json!({"rotation": 270, "tilt": 60, "zoom": 10, "seed": 7})).unwrap();
    assert_eq!(a["template"].as_str(), Some("qwen-2.1/edit"), "{a}");
    assert_eq!(
        crate::generate_cmds::angle_instruction(270, 60, 10).as_deref(),
        Some(
            "Rotate the camera 90 degrees to the left around the subject. Raise the camera high above the subject, looking down at it. \
             Move the camera closer for a close-up of the subject. Keep the subject, its identity, its clothing and the setting the same."
        )
    );
    assert_eq!(
        crate::generate_cmds::angle_instruction(180, 0, 5).as_deref(),
        Some("Move the camera behind the subject, so it is seen from the back. Keep the subject, its identity, its clothing and the setting the same.")
    );
    assert!(crate::generate_cmds::angle_instruction(10, 0, 5).is_none(), "the LoRA's cameras only");
    // The standalone rewriter asks the worker, which serves the enhancer, not ComfyUI.
    s.execute("generate.enhancePrompt", json!({"prompt": "a fox", "task": "image"})).unwrap();
    s.edit_prefs(|p| p.integrations.enhance_prompts = true);
    let m = s.execute("generate.models", json!({"probe": false})).unwrap();
    assert_eq!(m["autoFill"].as_str(), Some("qwen-2.1/fill"), "fills stay on the fill template unless fillAsEdit is asked: {m}");
    assert!(comfy.state().prompts.is_empty(), "nothing went to ComfyUI");
}