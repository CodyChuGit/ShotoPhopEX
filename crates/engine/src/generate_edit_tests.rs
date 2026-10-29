//! `generate.edit` (the whole picture by instruction, masked to the selection) and the automatic
//! purge before a run whose model files differ from the last run's, against the fake server.

use photocraft_doc::LayerId;
use photocraft_genai::fake::{FakeNative, Options};
use photocraft_geom::Rect;
use serde_json::json;

use super::*;

const FAKE_COLOR: [f32; 4] = [200.0 / 255.0, 40.0 / 255.0, 40.0 / 255.0, 1.0];

/// A 64×48 document with a textured layer and a 20×16 selection at (10, 8), talking to `url`.
fn session(url: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.edit("texture", |doc, active| {
        let r = doc.bounds();
        let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
        let n = surf.channels();
        let mut px = Vec::new();
        for y in 0..r.height() {
            for x in 0..r.width() {
                let v = ((x * 7 + y * 13) % 31) as f32 / 30.0;
                let mut tmp = [0.0f32; 8];
                let used = photocraft_raster::from_rgba_into(&surf.format(), [v, 1.0 - v, 0.5, 1.0], &mut tmp).min(n);
                px.extend_from_slice(&tmp[..used]);
            }
        }
        surf.write_region(r, &px);
        Ok(())
    })
    .unwrap();
    s.execute("select.rect", json!({"x": 10, "y": 8, "width": 20, "height": 16})).unwrap();
    s.edit_prefs(|p| {
        {
            p.integrations.models_path = url.to_string();
            p.integrations.runtime_path = "fake".into();
            p.integrations.allow_research_models = false;
        };
        p.integrations.quality = "balanced".into();
    });
    s
}

fn close(a: &[f32], b: [f32; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 2.5 / 255.0)
}

fn mask_at(s: &Session, id: LayerId, x: i32, y: i32) -> f32 {
    let d = s.active().unwrap();
    d.doc.layer(id).unwrap().mask.as_ref().unwrap().surface.read_region(Rect::new(x, y, x + 1, y + 1))[0]
}

#[test]
fn edit_re_renders_the_whole_canvas_into_a_layer_and_sends_no_mask() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    s.execute("select.deselect", json!({})).unwrap();
    let (layers, steps) = {
        let d = s.active().unwrap();
        (d.doc.layers.len(), d.history.past_len())
    };
    let r = s.execute(EDIT, json!({"prompt": "make the sky stormy", "seed": 3})).unwrap();
    assert_eq!(r["template"], AUTO_EDIT_ORDER[0], "auto takes the Lightning edit tier");
    assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(64), Some(48)), "the whole canvas");
    assert_eq!((r["requestWidth"].as_u64(), r["requestHeight"].as_u64()), (Some(512), Some(384)), "sent at the 512 px floor");
    let id = LayerId(r["layer"].as_u64().unwrap());
    let d = s.active().unwrap();
    assert_eq!(d.doc.layers.len(), layers + 1);
    assert_eq!(d.history.past_len(), steps + 1);
    assert_eq!(d.active_layer, Some(id));
    let l = d.doc.layer(id).unwrap();
    assert_eq!(l.name, "Generative Edit: make the sky stormy");
    assert!(close(&l.surface().unwrap().read_region(Rect::new(1, 1, 2, 2)), FAKE_COLOR), "the result covers the canvas");
    assert_eq!(mask_at(&s, id, 1, 1), 1.0, "no selection: everything shows");
    assert_eq!(mask_at(&s, id, 62, 46), 1.0);
    let st = fake.state();
    assert_eq!(st.pictures.len(), 1, "the picture alone, no mask upload");
    let g = &st.runs[0].request;
    assert!(g["mask"].is_null() && g["task"] == "edit", "an edit has no input mask");
    assert_eq!(g["params"]["models"]["lora"], "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors");
    assert_eq!(g["steps"], 8);
    assert_eq!(g["prompt"], "make the sky stormy. Keep everything else exactly as it is.");
    drop(st);
    // A description is turned into an instruction.
    s.execute(EDIT, json!({"prompt": "a stormy sky", "seed": 4})).unwrap();
    let g = &fake.state().runs[1].request;
    assert_eq!(g["prompt"], "Change this image so that it shows a stormy sky, keeping everything else exactly as it is.");
}

#[test]
fn a_selection_confines_what_of_the_edit_shows() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    let r = s.execute(EDIT, json!({"prompt": "turn the boat blue", "seed": 3})).unwrap();
    let id = LayerId(r["layer"].as_u64().unwrap());
    assert_eq!(mask_at(&s, id, 20, 16), 1.0, "inside the selection");
    assert_eq!(mask_at(&s, id, 62, 46), 0.0, "far outside");
    let band = mask_at(&s, id, 8, 16);
    assert!(band > 0.0 && band < 1.0, "a soft, dithered band around it: {band}");
    assert_eq!(fake.state().pictures.len(), 1, "still no mask upload: the selection is the layer mask only");
    let r = s.execute(EDIT, json!({"prompt": "turn the boat blue", "seed": 3, "edge": "hard"})).unwrap();
    let id = LayerId(r["layer"].as_u64().unwrap());
    assert_eq!(mask_at(&s, id, 8, 16), 0.0, "hard = exactly the selection");
    assert_eq!(mask_at(&s, id, 10, 8), 1.0);
}

#[test]
fn angles_sends_the_loras_words_over_the_whole_picture() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    // A selection is there, but a camera move is not confined by it.
    let r = s.execute(ANGLES, json!({"rotation": 135, "tilt": 60, "zoom": 5, "seed": 9, "variations": 2})).unwrap();
    assert_eq!(r["template"], ANGLES_TEMPLATE);
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().filter_map(|v| v.as_u64()).map(LayerId).collect();
    assert_eq!(ids.len(), 2);
    assert_eq!(mask_at(&s, ids[0], 62, 46), 1.0, "outside the selection too");
    let name = s.active().unwrap().doc.layer(ids[0]).unwrap().name.clone();
    assert!(name.starts_with("Angles: back-right quarter view high-angle shot medium shot"), "{name}");
    let st = fake.state();
    let g = &st.runs[0].request;
    assert_eq!(g["prompt"], "<sks> back-right quarter view high-angle shot medium shot", "the LoRA's words, no format");
    assert_eq!(g["params"]["models"]["lora"], "qwen-image-edit-2511-multiple-angles-lora.safetensors");
    assert_eq!(g["params"]["models"]["lightning"], "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors");
    assert_eq!((g["steps"].as_u64(), g["guidance"].as_f64()), (Some(4), Some(1.0)));
    drop(st);
    // The layer remembers an edit, so Generate Similar and Recreate run it again.
    let info = generative_info(s.active().unwrap().doc.layer(ids[0]).unwrap()).unwrap();
    assert_eq!((info.command.as_str(), info.template.as_str()), (EDIT, ANGLES_TEMPLATE));
    assert_eq!(angle_prompt(0, 0, 10).as_deref(), Some("<sks> front view eye-level shot close-up"));
    assert_eq!(angle_prompt(315, -30, 0).as_deref(), Some("<sks> front-left quarter view low-angle shot wide shot"));
    for p in [json!({"rotation": 30}), json!({"rotation": 360}), json!({"tilt": 45}), json!({"zoom": 7}), json!({"zoom": 2.5}), json!({"variations": 9})] {
        assert!(matches!(s.execute(ANGLES, p.clone()), Err(EngineError::BadParams { .. })), "{p}");
    }
    // Every camera has words: 8 × 4 × 3.
    let all =
        ANGLE_ROTATIONS.iter().flat_map(|(r, _)| ANGLE_TILTS.iter().flat_map(move |(t, _)| ANGLE_ZOOMS.iter().map(move |(z, _)| angle_prompt(*r, *t, *z))));
    assert_eq!(all.flatten().collect::<std::collections::BTreeSet<_>>().len(), 96);
}

#[test]
fn auto_falls_back_to_the_base_edit_template_and_params_are_validated() {
    let bare =
        FakeNative::start_with(Options { missing_files: vec!["Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors".into()], ..Options::default() })
            .unwrap();
    let mut s = session(&bare.url);
    let r = s.execute(EDIT, json!({"prompt": "x", "seed": 1})).unwrap();
    assert_eq!(r["template"], DEFAULT_EDIT_TEMPLATE);
    {
        // One guard, dropped before the next request (the fake's state lock is not reentrant).
        let st = bare.state();
        let g = &st.runs[0].request;
        assert_eq!(g["steps"], 40);
    }
    for p in [
        json!({}),
        json!({"prompt": "  "}),
        json!({"prompt": "x", "template": "qwen-edit-2511/fill"}),
        json!({"prompt": "x", "variations": 9}),
        json!({"prompt": "x", "edge": "blurry"}),
    ] {
        assert!(matches!(s.execute(EDIT, p.clone()), Err(EngineError::BadParams { .. })), "{p}");
    }
    let mut empty = Session::new();
    empty.edit_prefs(|p| {
        {
            p.integrations.models_path = bare.url.clone();
            p.integrations.runtime_path = "fake".into();
            p.integrations.allow_research_models = false;
        };
        p.integrations.quality = "balanced".into();
    });
    assert!(!empty.is_enabled(EDIT), "needs a document");
    let m = s.execute(MODELS, json!({})).unwrap();
    assert_eq!(m["autoEdit"], DEFAULT_EDIT_TEMPLATE, "what auto would pick without the LoRA");
}
