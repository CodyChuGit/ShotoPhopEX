//! The prompt enhancer: `generate.enhancePrompt` and the `enhance` option of the fill, edit and
//! image commands, against the fake server (whose chat requests answer
//! `Enhanced: ` + the instruction, or `Options::enhanced_text`).

use photocraft_doc::LayerId;
use photocraft_genai::fake::{FakeNative, Options};
use photocraft_geom::Rect;
use serde_json::json;

use super::*;

/// A 64×48 document with a layer and a 20×16 selection at (10, 8), talking to `url`.
fn session(url: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 10, "y": 8, "width": 20, "height": 16})).unwrap();
    s.edit_prefs(|p| {
        p.integrations.models_path = url.to_string();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    s
}

/// Does any string anywhere in the native request contain `needle`?
fn request_mentions(g: &Value, needle: &str) -> bool {
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

fn is_text_request(g: &Value) -> bool {
    g["task"] == "text"
}
fn text_prompt(g: &Value) -> String {
    g["prompt"].as_str().unwrap_or_default().to_string()
}
fn system_prompt(g: &Value) -> String {
    g["params"]["system"].as_str().unwrap_or_default().to_string()
}

#[test]
fn enhance_prompt_rewrites_an_instruction_with_the_picture_in_view() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    assert!(s.is_enabled(ENHANCE), "needs no document, only a server");
    let r = s.execute(ENHANCE, json!({"prompt": " make this hyper realistic ", "task": "edit", "seed": 7})).unwrap();
    assert_eq!(r["prompt"], "Enhanced: make this hyper realistic");
    assert_eq!(r["enhanced"], true);
    {
        let st = fake.state();
        assert_eq!(st.runs.len(), 1, "one try was enough");
        let g = &st.runs[0].request;
        assert!(is_text_request(g));
        assert_eq!(st.pictures.len(), 1, "the composite was uploaded");
        // The engine's rules are the chat's system turn, the user's words its user turn.
        assert_eq!(text_prompt(g), "make this hyper realistic");
        let sys = system_prompt(g);
        assert!(sys.starts_with("You rewrite instructions for an image-editing AI. You can see the image."), "{sys}");
        assert!(sys.contains("name things by what they are in the image") && sys.contains("changes the style, medium or realism"), "{sys}");
        assert!(!sys.contains("The user selected"), "an edit is about the whole picture");
        assert_eq!(g["seed"], 7);
    }
    // A fill is told where the selection sits.
    let r = s.execute(ENHANCE, json!({"prompt": "a kite", "task": "fill"})).unwrap();
    assert_eq!(r["prompt"], "Enhanced: a kite");
    {
        let st = fake.state();
        let g = &st.runs[1].request;
        // The 20×16 selection at (10, 8) on 64×48: its centre is in the left third, mid-height.
        let sys = system_prompt(g);
        assert!(sys.contains("The user selected the left part of the image"), "{sys}");
        assert_eq!(text_prompt(g), "a kite");
    }
    // Without the picture an instruction is rewritten blind through the text-only graph.
    s.execute(ENHANCE, json!({"prompt": "turn the boat blue", "task": "edit", "useImage": false})).unwrap();
    {
        let st = fake.state();
        let g = &st.runs[2].request;
        let sys = system_prompt(g);
        assert!(!sys.contains("You can see the image") && sys.starts_with("You rewrite instructions"), "{sys}");
        assert_eq!(text_prompt(g), "turn the boat blue");
    }
    // An image idea is expanded, never with a picture.
    s.execute(ENHANCE, json!({"prompt": "a fox in snow", "task": "image"})).unwrap();
    {
        let st = fake.state();
        let g = &st.runs[3].request;
        assert!(g["image"].is_null());
        assert!(system_prompt(g).starts_with("You are an expert prompt writer for a text-to-image model."));
        assert_eq!(text_prompt(g), "a fox in snow");
    }
    // The photographic rewriter: written like a real photo's caption, no "8k" words.
    s.execute(ENHANCE, json!({"prompt": "a fox in snow", "task": "photo"})).unwrap();
    {
        let st = fake.state();
        let g = &st.runs.last().unwrap().request;
        let sys = system_prompt(g);
        assert!(g["image"].is_null());
        assert!(sys.starts_with("You are an expert prompt writer for a photorealistic") && sys.contains("Never write photorealistic"), "{sys}");
    }
    // generate.image with enhanceStyle "photo" rewrites with it first.
    s.execute(IMAGE, json!({"prompt": "a fox in snow", "template": "krea2-turbo/image", "target": "document", "enhance": true, "enhanceStyle": "photo"}))
        .unwrap();
    {
        let st = fake.state();
        assert!(st.runs.iter().any(|r| system_prompt(&r.request).starts_with("You are an expert prompt writer for a photorealistic")));
    }
    assert!(s.execute(IMAGE, json!({"prompt": "x", "template": "krea2-turbo/image", "target": "document", "enhance": true, "enhanceStyle": "film"})).is_err());
    // Validation.
    for p in [json!({}), json!({"prompt": "  "}), json!({"prompt": "x", "task": "paint"}), json!({"prompt": "x", "useImage": "yes"})] {
        assert!(matches!(s.execute(ENHANCE, p.clone()), Err(EngineError::BadParams { .. })), "{p}");
    }
}

#[test]
fn the_edit_and_fill_commands_rewrite_on_the_way_and_the_layer_remembers_both() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    let r = s.execute(EDIT, json!({"prompt": "make this hyper realistic", "seed": 3, "enhance": true})).unwrap();
    let id = LayerId(r["layer"].as_u64().unwrap());
    {
        let st = fake.state();
        // An edit's result is then searched for its subject ("segment" runs after it).
        assert_eq!(st.runs.iter().filter(|r| r.request["task"] != "segment").count(), 2, "the rewriter, then the edit");
        assert!(is_text_request(&st.runs[0].request));
        let g = &st.runs[1].request;
        assert!(!is_text_request(g));
        let sent = g["prompt"].as_str().unwrap_or_default();
        assert!(sent.contains("Enhanced: make this hyper realistic"), "the model got the rewritten prompt: {sent}");
    }
    let info = s.execute(INFO, json!({"layer": id.0})).unwrap();
    assert_eq!(info["generative"]["prompt"], "make this hyper realistic", "as typed");
    assert_eq!(info["generative"]["enhanced"], "Enhanced: make this hyper realistic");
    let d = s.active().unwrap();
    assert_eq!(d.doc.layer(id).unwrap().name, "Generative Edit: make this hyper realistic", "named after the typed prompt");

    // Generate Similar on that layer rewrites anew (a new seed, a fresh sentence).
    let before = fake.state().runs.len();
    let r = s.execute(SIMILAR, json!({"layer": id.0, "seed": 9})).unwrap();
    assert_eq!(fake.state().runs.len(), before + 2);
    let info = s.execute(INFO, json!({"layer": r["layer"]})).unwrap();
    assert_eq!(info["generative"]["enhanced"], "Enhanced: make this hyper realistic");

    // The preference makes it the default of every fill; `enhance: false` opts a call out.
    s.edit_prefs(|p| p.integrations.enhance_prompts = true);
    let before = fake.state().runs.len();
    let r = s.execute(FILL, json!({"prompt": "a kite", "seed": 1})).unwrap();
    {
        let st = fake.state();
        assert_eq!(st.runs.len(), before + 2);
        let g = &st.runs[before].request;
        assert!(is_text_request(g) && !g["image"].is_null(), "the fill's rewriter sees the picture");
        assert!(system_prompt(g).contains("The user selected the"));
        assert_eq!(text_prompt(g), "a kite");
        assert!(request_mentions(&st.runs[before + 1].request, "Enhanced: a kite"));
    }
    let info = s.execute(INFO, json!({"layer": r["layer"]})).unwrap();
    assert_eq!(info["generative"]["enhanced"], "Enhanced: a kite");
    let before = fake.state().runs.len();
    let r = s.execute(FILL, json!({"prompt": "a kite", "seed": 1, "enhance": false})).unwrap();
    {
        let st = fake.state();
        assert_eq!(st.runs.len(), before + 1);
        assert!(!is_text_request(&st.runs[before].request));
    }
    let info = s.execute(INFO, json!({"layer": r["layer"]})).unwrap();
    assert_eq!(info["generative"]["enhanced"], "");
    // An expand keeps its wording even with the preference on.
    let before = fake.state().runs.len();
    s.execute(EXPAND, json!({"prompt": "more sky", "right": 16, "seed": 1})).unwrap();
    let st = fake.state();
    assert_eq!(st.runs.len(), before + 1);
    assert!(!is_text_request(&st.runs[before].request));
}

#[test]
fn an_image_idea_is_expanded_blind_and_wrapped_for_transparency_as_typed() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    s.execute("select.deselect", json!({})).unwrap();
    let r = s.execute(IMAGE, json!({"prompt": "a fox", "target": "layer", "seed": 5, "enhance": true})).unwrap();
    {
        let st = fake.state();
        assert_eq!(st.runs.len(), 2);
        let g = &st.runs[0].request;
        assert!(is_text_request(g) && g["image"].is_null(), "no picture for an idea");
        assert_eq!(text_prompt(g), "a fox");
        assert!(request_mentions(&st.runs[1].request, "Enhanced: a fox"));
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
        assert_eq!(st.runs.len(), 4);
        assert!(request_mentions(&st.runs[3].request, "Enhanced: a fox, isolated on a transparent background, output a PNG image"), "{}", st.runs[3].request);
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
    let mute = FakeNative::start_with(Options { enhanced_text: Some("assistant".into()), ..Options::default() }).unwrap();
    let mut s = session(&mute.url);
    let r = s.execute(ENHANCE, json!({"prompt": "a kite", "task": "fill"})).unwrap();
    assert_eq!(r["enhanced"], false);
    assert_eq!(r["prompt"], "a kite", "the typed prompt comes back");
    assert_eq!(mute.state().runs.len(), 3, "three tries with consecutive seeds");
    let before = mute.state().runs.len();
    let r = s.execute(EDIT, json!({"prompt": "make the sky stormy", "seed": 3, "enhance": true})).unwrap();
    {
        let st = mute.state();
        assert_eq!(st.runs.iter().filter(|r| r.request["task"] != "segment").count(), before + 4, "three tries, then the edit as typed");
        assert_eq!(st.runs[before + 3].request["prompt"], "make the sky stormy. Keep everything else exactly as it is.");
    }
    let info = s.execute(INFO, json!({"layer": r["layer"]})).unwrap();
    assert_eq!(info["generative"]["enhanced"], "");
    // Quotes and markdown around a real answer are stripped; the role echo before it too.
    let dressed =
        FakeNative::start_with(Options { enhanced_text: Some("Assistant: **\"Turn the sketch of a face into a photo.\"**".into()), ..Options::default() })
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

#[test]
fn a_text_only_prompt_writer_rewrites_an_edit_from_its_words() {
    // A Mac's worker refuses a chat with a picture: the edit is asked again without it.
    let fake = FakeNative::start_with(Options { text_only: true, ..Default::default() }).unwrap();
    let mut s = session(&fake.url);
    let r = s.execute(ENHANCE, json!({"prompt": "make the duck blue", "task": "edit", "seed": 7})).unwrap();
    assert_eq!(r["prompt"], "Enhanced: make the duck blue");
    let st = fake.state();
    assert_eq!(st.runs.len(), 2, "refused with the picture, answered without");
    assert!(!system_prompt(&st.runs[1].request).contains("You can see the image."));
}

#[test]
fn quality_best_rewrites_photo_style_only_when_enhance_is_on() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    s.edit_prefs(|p| {
        p.integrations.quality = "best".into();
        p.integrations.enhance_prompts = false;
    });
    let photo = |fake: &FakeNative| {
        fake.state().runs.iter().filter(|r| system_prompt(&r.request).starts_with("You are an expert prompt writer for a photorealistic")).count()
    };
    let before = fake.state().runs.len();
    s.execute(IMAGE, json!({"prompt": "a fox in snow", "template": "krea2-turbo/image", "target": "document"})).unwrap();
    assert_eq!(fake.state().runs.len(), before + 1, "Enhance prompts off: the words go as typed, even on Best");
    s.edit_prefs(|p| p.integrations.enhance_prompts = true);
    s.execute(IMAGE, json!({"prompt": "a fox in snow", "template": "krea2-turbo/image", "target": "document"})).unwrap();
    assert_eq!(photo(&fake), 1, "Best + Enhance prompts: the photo rewriter");
    s.execute(IMAGE, json!({"prompt": "a fox in snow", "template": "krea2-turbo/image", "target": "document", "enhance": false})).unwrap();
    assert_eq!(photo(&fake), 1, "an explicit enhance: false opts out");
    s.execute(IMAGE, json!({"prompt": "a fox in snow", "template": "krea2-turbo/image", "target": "document", "enhanceStyle": "default"})).unwrap();
    assert_eq!(photo(&fake), 1, "an explicit style wins");
    s.execute(IMAGE, json!({"prompt": "an oil painting of a lighthouse", "template": "krea2-turbo/image", "target": "document"})).unwrap();
    assert_eq!(photo(&fake), 1, "a painting gets the plain rewriter");
    let last = fake.state().runs.iter().rev().find(|r| is_text_request(&r.request)).map(|r| system_prompt(&r.request)).unwrap();
    assert!(!last.contains("photorealistic text-to-image"), "{last}");
    s.edit_prefs(|p| p.integrations.quality = "balanced".into());
    s.execute(IMAGE, json!({"prompt": "a fox in snow", "template": "krea2-turbo/image", "target": "document"})).unwrap();
    assert_eq!(photo(&fake), 1, "Balanced uses the plain rewriter");
}

#[test]
fn other_media_are_recognised() {
    use crate::generate_cmds::names_other_medium;
    for p in ["an oil painting of a lighthouse", "Watercolour of a fox", "anime girl on a train", "a 3D render of a chair", "pencil sketch of hands"] {
        assert!(names_other_medium(p), "{p}");
    }
    for p in ["a woman in a cafe", "a ceramic coffee mug on a wooden desk", "street at night, shot on 35mm"] {
        assert!(!names_other_medium(p), "{p}");
    }
}

#[test]
fn a_text_to_image_rewrite_runs_on_the_pictures_own_encoder() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    let encoder_of = |s: &mut crate::Session, template: &str| {
        s.execute(IMAGE, json!({"prompt": "a fox in snow", "template": template, "target": "document", "enhance": true})).unwrap();
        let st = fake.state();
        st.runs.iter().rev().find(|r| is_text_request(&r.request)).map(|r| r.request["params"]["encoder"].clone()).unwrap()
    };
    assert_eq!(encoder_of(&mut s, "krea2-turbo/image"), "krea2");
    assert_eq!(encoder_of(&mut s, "moss-image/turbo"), "moss-image/turbo");
    assert_eq!(encoder_of(&mut s, "moss-image/image"), "moss-image/image");
    assert_eq!(encoder_of(&mut s, "flux3-klein/image"), "flux3-klein/image");
    assert_eq!(encoder_of(&mut s, "qwen-2.1/image"), Value::Null, "the default encoder");
    // the 16 MP model: its own writer, a longer caption planned for its resolution
    assert_eq!(encoder_of(&mut s, "minimax-h3-image/image"), "minimax-h3-image/image");
    let st = fake.state();
    let last = st.runs.iter().rev().find(|r| is_text_request(&r.request)).unwrap();
    assert!(system_prompt(&last.request).contains("150 to 300 words"), "{}", system_prompt(&last.request));
    assert_eq!(last.request["params"]["max_tokens"], 640);
}

#[test]
fn a_rewrite_is_checked_before_it_replaces_the_prompt() {
    use crate::generate_cmds::{EnhanceKind as K, Refusal, check_rewrite};
    let ok = |k, t: &str, s: &str| check_rewrite(k, t, s);
    // Good rewrites pass, tidied.
    assert_eq!(
        ok(K::Edit, "make the boat blue", "Change the white fishing boat to a deep blue, keeping the harbour and the sky unchanged.").as_deref(),
        Ok("Change the white fishing boat to a deep blue, keeping the harbour and the sky unchanged.")
    );
    assert_eq!(
        ok(K::Image, "a fox in snow", "Here is the enhanced prompt: A red fox stands in deep fresh snow, its fur frosted, soft overcast light.").as_deref(),
        Ok("A red fox stands in deep fresh snow, its fur frosted, soft overcast light.")
    );
    // A paragraph cut off by the token limit ends at its last full sentence.
    let cut = "A red fox stands in deep fresh snow at the edge of a pine forest, its thick winter fur dusted with flakes. Soft overcast light falls evenly across the scene. The fox looks toward the";
    assert_eq!(ok(K::Image, "a fox in snow", cut).unwrap(), cut[..cut.rfind('.').unwrap() + 1]);
    // Garbled answers: another script, broken words, one token over and over.
    for junk in [
        "的的的 Change 的 boat 蓝色 的的的 船 改变 颜色 的的的",
        "Change the boat boat boat boat boat blue",
        "xqzrtk blpfmn the qwrtzp boat vbnmqr blue sdfghk",
        "boat blue \u{FFFD}\u{FFFD} sea",
        "the blue boat the blue boat the blue boat the blue boat the blue boat in the sea",
    ] {
        assert_eq!(ok(K::Edit, "make the boat blue", junk), Err(Refusal::Garbled), "{junk}");
    }
    // A prompt typed in another script may be answered in it.
    assert!(ok(K::Image, "雪中的狐狸", "一只红狐狸站在雪地里，柔和的阴天光线。").is_ok());
    // Talk about the task.
    assert_eq!(ok(K::Edit, "make the boat blue", "I'm sorry, but I cannot edit images."), Err(Refusal::Chatter));
    assert_eq!(ok(K::Image, "a fox", "The user wants a fox. A red fox in a forest."), Err(Refusal::Chatter));
    // The prompt back as typed.
    assert_eq!(ok(K::Image, "a red fox in deep snow", "A red fox in deep snow."), Err(Refusal::Unchanged));
    // What the user asked for, dropped: the subject, a colour, a count, quoted words.
    assert_eq!(ok(K::Image, "a fox", "A wolf runs across a frozen lake under a pale winter sky."), Err(Refusal::LostIntent));
    assert_eq!(ok(K::Edit, "make the boat blue", "Change the white fishing boat to a deep red colour."), Err(Refusal::LostIntent));
    assert_eq!(ok(K::Image, "three cats on a sofa", "A cat sleeps on a soft velvet sofa in warm lamp light."), Err(Refusal::LostIntent));
    assert_eq!(ok(K::Image, "a sign that says \"OPEN 24/7\"", "A neon sign reading \"OPEN\" glows in a rainy window."), Err(Refusal::LostIntent));
    assert!(ok(K::Image, "a sign that says \"OPEN 24/7\"", "A red neon sign reading \"OPEN 24/7\" glows in a rainy diner window.").is_ok());
    // Close word forms and plurals count; filler and instruction verbs do not.
    assert!(ok(K::Edit, "make this hyper realistic", "Turn the doodle into a realistic photograph of a real human face with the same pose.").is_ok());
    assert!(ok(K::Image, "two dogs playing on the beaches", "Two golden retriever dogs play together on a sandy beach at sunset.").is_ok());
    // A reference fill describes the reference, not the typed words.
    assert!(ok(K::FillRef, "the jacket", "a cropped olive green bomber with ribbed cuffs and a brass zip").is_ok());
}

#[test]
fn the_wand_rewrites_an_idea_for_the_model_it_will_run_on() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    s.edit_prefs(|p| p.integrations.quality = "balanced".into());
    let last = |fake: &FakeNative| fake.state().runs.iter().rev().find(|r| is_text_request(&r.request)).map(|r| r.request.clone()).unwrap();
    s.execute(ENHANCE, json!({"prompt": "a fox in snow", "task": "image", "template": "krea2-turbo/image"})).unwrap();
    let g = last(&fake);
    assert_eq!(g["params"]["encoder"], "krea2", "Krea 2's own writer, no model swap");
    assert!(system_prompt(&g).starts_with("You are an expert prompt writer for a text-to-image model."));
    s.execute(ENHANCE, json!({"prompt": "a fox in snow", "task": "image", "template": "moss-image/turbo"})).unwrap();
    assert_eq!(last(&fake)["params"]["encoder"], "moss-image/turbo");
    s.execute(ENHANCE, json!({"prompt": "a fox in snow", "task": "image", "template": "qwen-2.1/image"})).unwrap();
    assert_eq!(last(&fake)["params"]["encoder"], Value::Null);
    // The 16 MP model: its long caption, as its run would write.
    s.execute(ENHANCE, json!({"prompt": "a fox in snow", "task": "image", "template": "minimax-h3-image/image"})).unwrap();
    assert!(system_prompt(&last(&fake)).contains("150 to 300 words"));
    // Quality Best: the photographic caption for a photo idea, the plain one for a painting.
    s.edit_prefs(|p| p.integrations.quality = "best".into());
    s.execute(ENHANCE, json!({"prompt": "a fox in snow", "task": "image", "template": "krea2-turbo/image"})).unwrap();
    assert!(system_prompt(&last(&fake)).starts_with("You are an expert prompt writer for a photorealistic"));
    s.execute(ENHANCE, json!({"prompt": "an oil painting of a fox", "task": "image", "template": "krea2-turbo/image"})).unwrap();
    assert!(!system_prompt(&last(&fake)).contains("photorealistic text-to-image"));
    // No template: as before, the plain rewriter on the default encoder.
    s.execute(ENHANCE, json!({"prompt": "a fox in snow", "task": "image"})).unwrap();
    let g = last(&fake);
    assert_eq!(g["params"]["encoder"], Value::Null);
    assert!(system_prompt(&g).starts_with("You are an expert prompt writer for a text-to-image model."));
}
