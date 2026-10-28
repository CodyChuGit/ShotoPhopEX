//! Template selection: the research-licence gate and the default-template preferences, with the
//! Qwen-Image-2.1 templates as the research-only examples.

use photocraft_genai::fake::FakeComfy;
use serde_json::json;

use super::*;

fn session(url: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 16, "y": 16, "width": 32, "height": 32})).unwrap();
    s.edit_prefs(|p| {
        p.integrations.set_comfy_route(url.to_string());
        p.integrations.quality = "balanced".into();
    });
    s
}

#[test]
fn research_templates_need_the_opt_in() {
    let fake = FakeComfy::start().unwrap();
    let mut s = session(&fake.url);
    for (cmd, template) in [(FILL, "qwen-2.1/fill"), (IMAGE, "qwen-2.1/image")] {
        let e = s.execute(cmd, json!({"prompt": "x", "template": template})).unwrap_err();
        assert!(e.to_string().contains("research-only"), "{cmd}: {e}");
    }
    assert!(fake.state().requests.is_empty(), "refused before any request");
    // The listing says which templates are allowed right now.
    let m = s.execute(MODELS, json!({})).unwrap();
    for t in m["templates"].as_array().unwrap() {
        let research = t["license"] == "research";
        assert_eq!(t["allowed"], !research, "{}", t["id"]);
        assert!(!research || t["licenseNote"].as_str().unwrap().contains("Qwen Research License"), "{}", t["id"]);
    }

    s.edit_prefs(|p| p.integrations.allow_research_models = true);
    let m = s.execute(MODELS, json!({})).unwrap();
    assert!(m["templates"].as_array().unwrap().iter().all(|t| t["allowed"] == true));
    let r = s.execute(FILL, json!({"prompt": "a red bicycle", "template": "qwen-2.1/fill", "seed": 1})).unwrap();
    assert_eq!(r["template"], "qwen-2.1/fill");
    let st = fake.state();
    let g = &st.prompts[0].1;
    assert_eq!(g["6"]["class_type"], "TextEncodeQwenImage21");
    assert_eq!(g["6"]["inputs"]["images.image_1"], json!(["4", 0]), "the picture to edit (grouped input, dotted API key)");
    assert_eq!(g["6"]["inputs"]["images.image_2"], json!(["7", 0]), "the selection mask as a reference");
    let prompt = g["6"]["inputs"]["prompt"].as_str().unwrap();
    assert!(prompt.contains("a red bicycle") && prompt.contains("image 2"), "wrapped as a local-edit instruction: {prompt}");
    assert_eq!(g["4"]["inputs"]["image"], st.uploads[0].0, "the composite crop");
    assert_eq!(g["7"]["inputs"]["image"], st.uploads[1].0, "the selection mask");
    assert_eq!(g["10"]["inputs"]["latent_image"], json!(["6", 2]), "the encoder's empty latent sized to image 1");
    assert_eq!(g["5"]["class_type"], "QwenImage21Cache");
    assert_eq!(g["10"]["inputs"]["steps"], 25, "2.1 default");
    assert_eq!(g["10"]["inputs"]["cfg"], 1.0);
    assert_eq!(g["1"]["inputs"]["unet_name"], "qwen_image_2.1_int8_convrot.safetensors");
    assert_eq!(g["3"]["inputs"]["vae_name"], "qwen_image_2.1_vae_bf16.safetensors");
}

#[test]
fn the_default_template_preferences_pick_the_template() {
    let fake = FakeComfy::start().unwrap();
    let mut s = session(&fake.url);
    s.edit_prefs(|p| {
        p.integrations.allow_research_models = true;
        p.integrations.default_fill_template = "qwen-2.1/fill".into();
        p.integrations.default_image_template = "qwen-2.1/image".into();
    });
    let r = s.execute(FILL, json!({"prompt": "x"})).unwrap();
    assert_eq!(r["template"], "qwen-2.1/fill");
    let r = s.execute(IMAGE, json!({"prompt": "x"})).unwrap();
    assert_eq!(r["template"], "qwen-2.1/image");
    let st = fake.state();
    assert_eq!(st.prompts[1].1["4"]["class_type"], "QwenImage21Cache");
    assert_eq!(st.prompts[1].1["5"]["class_type"], "TextEncodeQwenImage21");
    assert_eq!(st.prompts[1].1["6"]["class_type"], "EmptyLatentImage");
    drop(st);

    // An explicit param still wins, and an empty preference means `auto` (the Lightning tier,
    // whose LoRA the fake lists).
    let r = s.execute(FILL, json!({"prompt": "x", "template": "qwen-edit-2511/fill"})).unwrap();
    assert_eq!(r["template"], "qwen-edit-2511/fill");
    s.edit_prefs(|p| p.integrations.default_fill_template = String::new());
    let r = s.execute(FILL, json!({"prompt": "x"})).unwrap();
    assert_eq!(r["template"], AUTO_FILL_ORDER[0]);

    // A preference naming an unknown or wrong-task template is a clear error, not a panic.
    s.edit_prefs(|p| p.integrations.default_fill_template = "nope/fill".into());
    let e = s.execute(FILL, json!({"prompt": "x"})).unwrap_err();
    assert!(matches!(e, EngineError::BadParams { .. }) && e.to_string().contains("nope/fill"), "{e}");
    s.edit_prefs(|p| p.integrations.default_fill_template = "krea2-turbo/image".into());
    assert!(matches!(s.execute(FILL, json!({"prompt": "x"})), Err(EngineError::BadParams { .. })));
    // The research gate applies to preferences too.
    s.edit_prefs(|p| {
        p.integrations.default_fill_template = "qwen-2.1/fill".into();
        p.integrations.allow_research_models = false;
    });
    assert!(s.execute(FILL, json!({"prompt": "x"})).unwrap_err().to_string().contains("research-only"));
}

#[test]
fn template_preferences_are_validated_when_set() {
    let mut p = crate::prefs::Preferences::default();
    assert!(p.set("integrations.defaultFillTemplate", json!("qwen-2.1/fill")).is_ok());
    assert!(p.set("integrations.defaultFillTemplate", json!("../x")).is_err());
    assert!(p.set("integrations.defaultImageTemplate", json!("Has Spaces")).is_err());
    assert!(p.set("integrations.comfyServer", json!("ftp://x")).is_err());
    assert!(p.set("integrations.comfyServer", json!("http://127.0.0.1:8188")).is_ok());
    assert!(p.set("integrations.comfyServer", json!("")).is_ok(), "empty disables");
    assert!(p.set("integrations.generativeTimeoutSecs", json!(1)).is_err());
    assert!(p.set("integrations.defaultEditModel", json!("a/b.safetensors")).is_err());
    assert_eq!(p.integrations.default_fill_template, "qwen-2.1/fill");
}
