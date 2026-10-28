//! The NVFP4 policy (`integrations.modelPrecision`): slots with an NVFP4 alternative get it on a
//! Blackwell GPU when the server has the file, the purge sees the switch, other GPUs keep the
//! default files unless forced.

use photocraft_genai::fake::{FakeComfy, Options};
use photocraft_genai::is_blackwell;
use serde_json::json;

use super::*;

const FP8_TE: &str = "qwen_2.5_vl_7b_fp8_scaled.safetensors";
const NVFP4_TE: &str = "qwen_2.5_vl_7b_nvfp4.safetensors";
const BLACKWELL: &str = "NVIDIA GeForce RTX 5090 : cudaMallocAsync";

/// A 64×48 document with a layer and a 20×16 selection at (10, 8), talking to `url`.
fn session(url: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 10, "y": 8, "width": 20, "height": 16})).unwrap();
    s.edit_prefs(|p| {
        p.integrations.set_comfy_route(url.to_string());
        p.integrations.quality = "balanced".into();
    });
    s
}

fn clip_of(fake: &FakeComfy, index: usize) -> String {
    fake.state().prompts[index].1["2"]["inputs"]["clip_name"].as_str().unwrap_or_default().to_string()
}

fn frees(fake: &FakeComfy) -> usize {
    fake.state().requests.iter().filter(|r| r.as_str() == "POST /free").count()
}

#[test]
fn a_blackwell_server_gets_the_nvfp4_encoder_and_the_purge_sees_the_switch() {
    let fake = FakeComfy::start_with(Options { device: BLACKWELL.into(), ..Options::default() }).unwrap();
    let mut s = session(&fake.url);
    // The `switch` purge rule: an encoder swap is a different set (under the default `heavy`
    // only the diffusion model counts, see `the_server_is_purged_before_a_run_whose_model_files_differ_from_the_last`).
    s.edit_prefs(|p| p.integrations.purge_models = "switch".into());
    s.execute(FILL, json!({"prompt": "a kite", "seed": 1})).unwrap();
    assert_eq!(clip_of(&fake, 0), NVFP4_TE, "auto on a Blackwell card takes the NVFP4 encoder");
    // The picker learns about the alternative and whether it is installed.
    let m = s.execute(MODELS, json!({})).unwrap();
    let fill = m["templates"].as_array().unwrap().iter().find(|t| t["id"] == "qwen-edit-2511/fill").unwrap();
    let clip = fill["models"].as_array().unwrap().iter().find(|s| s["placeholder"] == "clip").unwrap();
    assert_eq!(clip["file"], FP8_TE, "the template's default stays the listed file");
    assert_eq!(clip["nvfp4"], NVFP4_TE);
    assert_eq!(clip["nvfp4Installed"], true);
    assert_eq!(m["device"], BLACKWELL);
    assert_eq!(m["blackwell"], true);
    // Back to the default files: a different set, so the server is purged once.
    assert_eq!(frees(&fake), 0);
    s.edit_prefs(|p| p.integrations.model_precision = "default".into());
    s.execute(FILL, json!({"prompt": "a kite", "seed": 2})).unwrap();
    assert_eq!(clip_of(&fake, 1), FP8_TE);
    assert_eq!(frees(&fake), 1, "the purge works on the files that really load");
    s.execute(FILL, json!({"prompt": "a kite", "seed": 3})).unwrap();
    assert_eq!(frees(&fake), 1);
    // A caller's own binding for a slot is never replaced (Krea 2's model slot through `model`).
    s.edit_prefs(|p| p.integrations.model_precision = "auto".into());
    s.execute("select.deselect", json!({})).unwrap();
    s.execute(IMAGE, json!({"prompt": "a fox", "seed": 1, "target": "layer", "model": "krea2_turbo_fp8_scaled.safetensors"})).unwrap();
    assert_eq!(fake.state().prompts[3].1["1"]["inputs"]["unet_name"], "krea2_turbo_fp8_scaled.safetensors");
    s.execute(IMAGE, json!({"prompt": "a fox", "seed": 2, "target": "layer"})).unwrap();
    assert_eq!(fake.state().prompts[4].1["1"]["inputs"]["unet_name"], "krea2_turbo_nvfp4.safetensors", "unbound: the policy picks");
}

/// The native worker manages what it keeps resident: a change of model set on its key never
/// asks it to unload (a removal's detector between two fills would otherwise drop the
/// resident diffusion pipeline), while the same change on a ComfyUI server purges.
#[test]
fn the_native_worker_is_never_purged_by_a_model_switch() {
    let fake = FakeComfy::start().unwrap();
    let backend = photocraft_genai::comfy::ComfyBackend::new(&fake.url, std::time::Duration::from_secs(10)).unwrap();
    let req = |unet: &str| Request {
        template: "krea2-turbo/image".into(),
        prompt: "a fox".into(),
        negative: String::new(),
        seed: 1,
        steps: 0,
        guidance: 0.0,
        image: None,
        mask: None,
        models: vec![("unet".into(), unet.into())],
        size: Some((64, 64)),
        params: Default::default(),
        references: Vec::new(),
    };
    let ctx = crate::jobs::JobCtx::new();
    let progress = JobProgress::span(&ctx, 0.0, 1.0);
    let key = NATIVE_SERVER_KEY;
    // A key of its own per run, so the other tests' runs on "native" do not count.
    forget_models(key);
    run_switching(&backend, key, &req("krea2_turbo_fp8_scaled.safetensors"), PURGE_SWITCH, &progress).unwrap();
    run_switching(&backend, key, &req("krea2_turbo_nvfp4.safetensors"), PURGE_SWITCH, &progress).unwrap();
    assert_eq!(frees(&fake), 0, "the native key never purges");
    let server = format!("{}-switch-test", fake.url);
    run_switching(&backend, &server, &req("krea2_turbo_fp8_scaled.safetensors"), PURGE_SWITCH, &progress).unwrap();
    run_switching(&backend, &server, &req("krea2_turbo_nvfp4.safetensors"), PURGE_SWITCH, &progress).unwrap();
    assert_eq!(frees(&fake), 1, "a ComfyUI server does");
}

#[test]
fn other_gpus_keep_the_default_files_unless_forced_and_a_missing_file_is_never_chosen() {
    let fake = FakeComfy::start().unwrap();
    let mut s = session(&fake.url);
    s.execute(FILL, json!({"prompt": "a kite", "seed": 1})).unwrap();
    assert_eq!(clip_of(&fake, 0), FP8_TE, "a \"Fake GPU\" is not Blackwell");
    let m = s.execute(MODELS, json!({})).unwrap();
    assert_eq!(m["blackwell"], false);
    s.edit_prefs(|p| p.integrations.model_precision = "nvfp4".into());
    s.execute(FILL, json!({"prompt": "a kite", "seed": 2})).unwrap();
    assert_eq!(clip_of(&fake, 1), NVFP4_TE, "forced");
    assert!(s.execute("prefs.set", json!({"path": "integrations.modelPrecision", "value": "fp3"})).is_err(), "only the listed names");
    s.execute("prefs.set", json!({"path": "integrations.modelPrecision", "value": "default"})).unwrap();
    assert_eq!(s.prefs().integrations.model_precision, "default");
    // A Blackwell server without the NVFP4 file keeps the default.
    let bare = FakeComfy::start_with(Options { device: BLACKWELL.into(), missing_files: vec![NVFP4_TE.into()], ..Options::default() }).unwrap();
    let mut s = session(&bare.url);
    s.execute(FILL, json!({"prompt": "a kite", "seed": 1})).unwrap();
    assert_eq!(clip_of(&bare, 0), FP8_TE);
    let m = s.execute(MODELS, json!({})).unwrap();
    let fill = m["templates"].as_array().unwrap().iter().find(|t| t["id"] == "qwen-edit-2511/fill").unwrap();
    let clip = fill["models"].as_array().unwrap().iter().find(|s| s["placeholder"] == "clip").unwrap();
    assert_eq!(clip["nvfp4Installed"], false);
}

#[test]
fn blackwell_names() {
    for yes in [
        "NVIDIA GeForce RTX 5090 : cudaMallocAsync",
        "NVIDIA GeForce RTX 5070 Ti",
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "NVIDIA B200",
        "NVIDIA GB10",
    ] {
        assert!(is_blackwell(yes), "{yes}");
    }
    for no in [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA RTX 5000 Ada Generation",
        "NVIDIA RTX 500 Ada Generation Laptop GPU",
        "NVIDIA A100",
        "Fake GPU",
        "",
    ] {
        assert!(!is_blackwell(no), "{no}");
    }
}
