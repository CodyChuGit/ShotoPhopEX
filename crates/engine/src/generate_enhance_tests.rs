//! The prompt enhancer: `generate.enhancePrompt` and the `enhance` option of the fill, edit and
//! image commands, against the fake server (whose `TextGenerate` graphs answer
//! `Enhanced: ` + the instruction, or `Options::enhanced_text`).

use photocraft_doc::LayerId;
use photocraft_genai::fake::{FakeComfy, Options};
use photocraft_geom::Rect;
use serde_json::json;

use super::*;

/// A 64×48 document with a layer and a 20×16 selection at (10, 8), talking to `url`.
fn session(url: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 10, "y": 8, "width": 20, "height": 16})).unwrap();
    s.edit_prefs(|p| p.integrations.set_comfy_route(url.to_string()));
    s
}

/// Does any string anywhere in the graph's node inputs contain `needle`?
fn graph_mentions(g: &Value, needle: &str) -> bool {
    fn walk(v: &Value, needle: &str) -> bool {
        match v {
            Value::String(s) => s.contains(needle),
            Value::Array(a) => a.iter().any(|v| walk(v, needle)),
            Value::Object(o) => o.values().any(|v| walk(v, needle)),
            _ => false,
        }
    }
    walk(g, needle)
}

fn is_text_graph(g: &Value) -> bool {
    g.as_object().is_some_and(|o| o.values().any(|n| n["class_type"] == "TextGenerate"))
}

/// The TextGenerate node's user turn (the prompt as typed) and system turn (the engine's rules).
fn text_prompt(g: &Value) -> String {
    text_input(g, "prompt")
}

fn system_prompt(g: &Value) -> String {
    text_input(g, "system_prompt")
}

fn text_input(g: &Value, key: &str) -> String {
    g.as_object().and_then(|o| o.values().find(|n| n["class_type"] == "TextGenerate")).and_then(|n| n["inputs"][key].as_str()).unwrap_or_default().to_string()
}

#[test]
fn enhance_prompt_rewrites_an_instruction_with_the_picture_in_view() {
    let fake = FakeComfy::start().unwrap();
    let mut s = session(&fake.url);
    assert!(s.is_enabled(ENHANCE), "needs no document, only a server");
    let r = s.execute(ENHANCE, json!({"prompt": " make this hyper realistic ", "task": "edit", "seed": 7})).unwrap();
    assert_eq!(r["prompt"], "Enhanced: make this hyper realistic");
    assert_eq!(r["enhanced"], true);
    {
        let st = fake.state();
        assert_eq!(st.prompts.len(), 1, "one try was enough");
        let g = &st.prompts[0].1;
        assert!(is_text_graph(g));
        assert_eq!(g["1"]["inputs"]["type"], "krea2", "the encoder is loaded as Krea 2 loads it");
        assert!(g.get("2").is_some_and(|n| n["class_type"] == "LoadImage"), "the picture goes along");
        assert_eq!(st.uploads.len(), 1, "the composite was uploaded");
        // The engine's rules are the chat's system turn, the user's words its user turn.
        assert_eq!(text_prompt(g), "make this hyper realistic");
        let sys = system_prompt(g);
        assert!(sys.starts_with("You rewrite instructions for an image-editing AI. You can see the image."), "{sys}");
        assert!(sys.contains("names things by what they are in the image") && sys.contains("changes the style, medium or realism"), "{sys}");
        assert!(!sys.contains("The user selected"), "an edit is about the whole picture");
        assert_eq!(g["3"]["inputs"]["sampling_mode.seed"], 7);
        assert_eq!(g["3"]["inputs"]["thinking"], true, "a plain assistant turn: the non-thinking 4B model answers nothing after an empty think block");
    }
    // A fill is told where the selection sits.
    let r = s.execute(ENHANCE, json!({"prompt": "a kite", "task": "fill"})).unwrap();
    assert_eq!(r["prompt"], "Enhanced: a kite");
    {
        let st = fake.state();
        let g = &st.prompts[1].1;
        // The 20×16 selection at (10, 8) on 64×48: its centre is in the left third, mid-height.
        let sys = system_prompt(g);
        assert!(sys.contains("The user selected the left part of the image"), "{sys}");
        assert_eq!(text_prompt(g), "a kite");
    }
    // Without the picture an instruction is rewritten blind through the text-only graph.
    s.execute(ENHANCE, json!({"prompt": "turn the boat blue", "task": "edit", "useImage": false})).unwrap();
    {
        let st = fake.state();
        let g = &st.prompts[2].1;
        assert!(is_text_graph(g) && g.get("2").is_none(), "no LoadImage: {g}");
        let sys = system_prompt(g);
        assert!(!sys.contains("You can see the image") && sys.starts_with("You rewrite instructions"), "{sys}");
        assert_eq!(text_prompt(g), "turn the boat blue");
    }
    // An image idea is expanded, never with a picture.
    s.execute(ENHANCE, json!({"prompt": "a fox in snow", "task": "image"})).unwrap();
    {
        let st = fake.state();
        let g = &st.prompts[3].1;
        assert!(g.get("2").is_none());
        assert!(system_prompt(g).starts_with("You are an expert prompt writer for a text-to-image model."));
        assert_eq!(text_prompt(g), "a fox in snow");
        assert_eq!(g["3"]["inputs"]["max_length"], 220, "a paragraph, not a sentence");
    }
    // Validation.
    for p in [json!({}), json!({"prompt": "  "}), json!({"prompt": "x", "task": "paint"}), json!({"prompt": "x", "useImage": "yes"})] {
        assert!(matches!(s.execute(ENHANCE, p.clone()), Err(EngineError::BadParams { .. })), "{p}");
    }
}

#[test]
fn the_edit_and_fill_commands_rewrite_on_the_way_and_the_layer_remembers_both() {
    let fake = FakeComfy::start().unwrap();
    let mut s = session(&fake.url);
    let r = s.execute(EDIT, json!({"prompt": "make this hyper realistic", "seed": 3, "enhance": true})).unwrap();
    let id = LayerId(r["layer"].as_u64().unwrap());
    {
        let st = fake.state();
        assert_eq!(st.prompts.len(), 2, "the rewriter, then the edit");
        assert!(is_text_graph(&st.prompts[0].1));
        let g = &st.prompts[1].1;
        assert!(!is_text_graph(g));
        let sent = g["6"]["inputs"]["prompt"].as_str().unwrap_or_default();
        assert!(sent.contains("Enhanced: make this hyper realistic"), "the model got the rewritten prompt: {sent}");
    }
    let info = s.execute(INFO, json!({"layer": id.0})).unwrap();
    assert_eq!(info["generative"]["prompt"], "make this hyper realistic", "as typed");
    assert_eq!(info["generative"]["enhanced"], "Enhanced: make this hyper realistic");
    let d = s.active().unwrap();
    assert_eq!(d.doc.layer(id).unwrap().name, "Generative Edit: make this hyper realistic", "named after the typed prompt");

    // Generate Similar on that layer rewrites anew (a new seed, a fresh sentence).
    let before = fake.state().prompts.len();
    let r = s.execute(SIMILAR, json!({"layer": id.0, "seed": 9})).unwrap();
    assert_eq!(fake.state().prompts.len(), before + 2);
    let info = s.execute(INFO, json!({"layer": r["layer"]})).unwrap();
    assert_eq!(info["generative"]["enhanced"], "Enhanced: make this hyper realistic");

    // The preference makes it the default of every fill; `enhance: false` opts a call out.
    s.edit_prefs(|p| p.integrations.enhance_prompts = true);
    let before = fake.state().prompts.len();
    let r = s.execute(FILL, json!({"prompt": "a kite", "seed": 1})).unwrap();
    {
        let st = fake.state();
        assert_eq!(st.prompts.len(), before + 2);
        let g = &st.prompts[before].1;
        assert!(is_text_graph(g) && g.get("2").is_some(), "the fill's rewriter sees the picture");
        assert!(system_prompt(g).contains("The user selected the"));
        assert_eq!(text_prompt(g), "a kite");
        assert!(graph_mentions(&st.prompts[before + 1].1, "Enhanced: a kite"));
    }
    let info = s.execute(INFO, json!({"layer": r["layer"]})).unwrap();
    assert_eq!(info["generative"]["enhanced"], "Enhanced: a kite");
    let before = fake.state().prompts.len();
    let r = s.execute(FILL, json!({"prompt": "a kite", "seed": 1, "enhance": false})).unwrap();
    {
        let st = fake.state();
        assert_eq!(st.prompts.len(), before + 1);
        assert!(!is_text_graph(&st.prompts[before].1));
    }
    let info = s.execute(INFO, json!({"layer": r["layer"]})).unwrap();
    assert_eq!(info["generative"]["enhanced"], "");
    // An expand keeps its wording even with the preference on.
    let before = fake.state().prompts.len();
    s.execute(EXPAND, json!({"prompt": "more sky", "right": 16, "seed": 1})).unwrap();
    let st = fake.state();
    assert_eq!(st.prompts.len(), before + 1);
    assert!(!is_text_graph(&st.prompts[before].1));
}

#[test]
fn an_image_idea_is_expanded_blind_and_wrapped_for_transparency_as_typed() {
    let fake = FakeComfy::start().unwrap();
    let mut s = session(&fake.url);
    s.execute("select.deselect", json!({})).unwrap();
    let r = s.execute(IMAGE, json!({"prompt": "a fox", "target": "layer", "seed": 5, "enhance": true})).unwrap();
    {
        let st = fake.state();
        assert_eq!(st.prompts.len(), 2);
        let g = &st.prompts[0].1;
        assert!(is_text_graph(g) && g.get("2").is_none(), "no picture for an idea");
        assert_eq!(text_prompt(g), "a fox");
        assert!(graph_mentions(&st.prompts[1].1, "Enhanced: a fox"));
    }
    let info = s.execute(INFO, json!({"layer": r["layer"]})).unwrap();
    assert_eq!(info["generative"]["prompt"], "a fox");
    assert_eq!(info["generative"]["enhanced"], "Enhanced: a fox");
    // Transparent: the rewritten idea goes inside the template's wording, as the typed one would.
    s.edit_prefs(|p| p.integrations.allow_research_models = true);
    let r =
        s.execute(IMAGE, json!({"prompt": "a fox", "target": "layer", "seed": 5, "enhance": true, "transparent": true, "template": "qwen-2.1/image"})).unwrap();
    {
        let st = fake.state();
        assert_eq!(st.prompts.len(), 4);
        assert!(graph_mentions(&st.prompts[3].1, "Enhanced: a fox, isolated on a transparent background, output a PNG image"), "{}", st.prompts[3].1);
    }
    let info = s.execute(INFO, json!({"layer": r["layer"]})).unwrap();
    assert_eq!(info["generative"]["enhanced"], "Enhanced: a fox");
    assert_eq!(info["generative"]["transparent"], true);
    // A new document from an idea remembers it too.
    let r = s.execute(IMAGE, json!({"prompt": "a barn", "target": "document", "seed": 5, "enhance": true})).unwrap();
    let info = s.execute(INFO, json!({})).unwrap();
    assert!(r["document"].as_u64().is_some());
    assert_eq!(info["generative"]["enhanced"], "Enhanced: a barn");
}

#[test]
fn a_rewriter_that_says_nothing_usable_leaves_the_prompt_as_typed() {
    // The model echoes the chat template's role word and nothing else, on both tries.
    let mute = FakeComfy::start_with(Options { enhanced_text: Some("assistant".into()), ..Options::default() }).unwrap();
    let mut s = session(&mute.url);
    let r = s.execute(ENHANCE, json!({"prompt": "a kite", "task": "fill"})).unwrap();
    assert_eq!(r["enhanced"], false);
    assert_eq!(r["prompt"], "a kite", "the typed prompt comes back");
    assert_eq!(mute.state().prompts.len(), 2, "two tries with consecutive seeds");
    let before = mute.state().prompts.len();
    let r = s.execute(EDIT, json!({"prompt": "make the sky stormy", "seed": 3, "enhance": true})).unwrap();
    {
        let st = mute.state();
        assert_eq!(st.prompts.len(), before + 3, "two tries, then the edit as typed");
        assert_eq!(st.prompts[before + 2].1["6"]["inputs"]["prompt"], "make the sky stormy. Keep everything else exactly as it is.");
    }
    let info = s.execute(INFO, json!({"layer": r["layer"]})).unwrap();
    assert_eq!(info["generative"]["enhanced"], "");
    // Quotes and markdown around a real answer are stripped; the role echo before it too.
    let dressed =
        FakeComfy::start_with(Options { enhanced_text: Some("Assistant: **\"Turn the sketch of a face into a photo.\"**".into()), ..Options::default() })
            .unwrap();
    let mut s = session(&dressed.url);
    let r = s.execute(ENHANCE, json!({"prompt": "x", "task": "edit"})).unwrap();
    assert_eq!(r["prompt"], "Turn the sketch of a face into a photo.");
}

#[test]
fn cleaning_and_the_place_in_words() {
    assert_eq!(
        clean_enhanced(&["assistant\nChange the red boat to blue, keeping the rest.".into()]).as_deref(),
        Some("Change the red boat to blue, keeping the rest.")
    );
    assert_eq!(clean_enhanced(&["\u{201c}Make the sky stormy.\u{201d}".into()]).as_deref(), Some("Make the sky stormy."));
    assert_eq!(
        clean_enhanced(&["".into(), "Assistant:".into(), "ok".into(), "Replace the cup with a vase.".into()]).as_deref(),
        Some("Replace the cup with a vase.")
    );
    assert_eq!(clean_enhanced(&["assistant".into()]), None);
    assert_eq!(clean_enhanced(&[]), None);
    let canvas = Rect::new(0, 0, 300, 300);
    assert_eq!(place_in_words(Rect::new(0, 0, 50, 50), canvas), "upper-left part");
    assert_eq!(place_in_words(Rect::new(250, 250, 300, 300), canvas), "lower-right part");
    assert_eq!(place_in_words(Rect::new(120, 120, 180, 180), canvas), "centre");
    assert_eq!(place_in_words(Rect::new(250, 120, 300, 180), canvas), "right part");
    assert_eq!(place_in_words(Rect::new(120, 0, 180, 40), canvas), "upper part");
    assert_eq!(place_in_words(Rect::new(10, 10, 290, 290), canvas), "whole");
    assert_eq!(place_in_words(Rect::new(10, 10, 50, 50), Rect::new(0, 0, 0, 0)), "lower-right part", "an empty canvas never divides by zero");
}
