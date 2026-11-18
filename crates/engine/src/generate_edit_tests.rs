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
    // No selection: what shows is the result's subject and what the edit changed; the fake's
    // flat colour changed all of the picture, so all of the result shows, unmasked (a night or
    // snow edit, a new background, a painting style).
    for (x, y) in [(32, 24), (1, 1), (62, 46)] {
        assert_eq!(mask_at(&s, id, x, y), 1.0, "an edit that changed most of the picture shows whole, at ({x}, {y})");
    }
    let st = fake.state();
    let g = &st.runs[0].request;
    assert!(g["mask"].is_null() && g["task"] == "edit", "an edit has no input mask");
    assert_eq!(g["params"]["models"]["lora"], "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors");
    assert_eq!(g["steps"], 8);
    assert_eq!(g["prompt"], "make the sky stormy. Keep everything else exactly as it is.");
    drop(st);
    // A description is turned into an instruction.
    s.execute(EDIT, json!({"prompt": "a stormy sky", "seed": 4})).unwrap();
    let st = fake.state();
    let g = &st.runs.iter().filter(|r| r.request["task"] == "edit").nth(1).expect("the second edit").request;
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
    assert!(fake.state().runs[0].request["mask"].is_null(), "still no mask upload: the selection is the layer mask only");
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
    assert_eq!(
        g["prompt"],
        "<sks> back-right quarter view high-angle shot The camera has moved 135 degrees around to the subject's right: we see the subject's back and right side. The camera is raised high above the subject and looks down at it at 60 degrees.",
        "the LoRA's words, then the move in plain words"
    );
    assert_eq!(g["params"]["models"]["lora"], "qwen-image-edit-2511-multiple-angles-lora.safetensors");
    assert_eq!(g["params"]["models"]["lightning"], "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors");
    assert_eq!((g["steps"].as_u64(), g["guidance"].as_f64()), (Some(8), Some(1.0)), "the 8-step Lightning's");
    assert!(g["params"].get("lightning").is_none(), "the worker's own default (8 at 1)");
    drop(st);
    // The layer remembers an edit, so Generate Similar and Recreate run it again.
    let info = generative_info(s.active().unwrap().doc.layer(ids[0]).unwrap()).unwrap();
    assert_eq!((info.command.as_str(), info.template.as_str()), (EDIT, ANGLES_TEMPLATE));
    assert_eq!(angle_prompt(0, 0, 5).as_deref(), Some("<sks> front view eye-level shot"), "no move: no sentence, no distance");
    assert_eq!(
        angle_prompt(0, 0, 10).as_deref(),
        Some("<sks> front view eye-level shot close-up The camera has moved close: the subject's main part fills the frame.")
    );
    assert_eq!(
        angle_prompt(270, 0, 5).as_deref(),
        Some(
            "<sks> left side view eye-level shot The camera has moved 90 degrees around to the subject's left: a true side profile, the subject seen exactly from its left side."
        )
    );
    assert_eq!(
        angle_prompt(315, -30, 0).as_deref(),
        Some(
            "<sks> front-left quarter view low-angle shot wide shot The camera has moved a little, 45 degrees, to the subject's left: a three-quarter view with most of the front still visible. The camera is close to the ground, below the subject, and looks up at it: a low-angle view from 30 degrees below. The camera has moved back: the whole subject and its surroundings are in the frame."
        )
    );
    for p in [json!({"rotation": 30}), json!({"rotation": 360}), json!({"tilt": 45}), json!({"zoom": 7}), json!({"zoom": 2.5}), json!({"variations": 9})] {
        assert!(matches!(s.execute(ANGLES, p.clone()), Err(EngineError::BadParams { .. })), "{p}");
    }
    // Another Lightning brings its steps and CFG; the strengths go to the worker as they are.
    let before = fake.state().runs.len();
    s.execute(ANGLES, json!({"rotation": 45, "seed": 1, "lightning": 4})).unwrap();
    s.execute(ANGLES, json!({"rotation": 45, "seed": 1, "lightning": 0, "anglesStrength": 1.25, "lightningStrength": 0.5})).unwrap();
    s.execute(ANGLES, json!({"rotation": 45, "seed": 1, "lightning": 0, "steps": 30})).unwrap();
    let st = fake.state();
    let runs: Vec<&Value> = st.runs[before..].iter().map(|r| &r.request).collect();
    assert_eq!((runs[0]["steps"].as_u64(), runs[0]["guidance"].as_f64(), runs[0]["params"]["lightning"].as_u64()), (Some(4), Some(1.0), Some(4)));
    assert_eq!((runs[1]["steps"].as_u64(), runs[1]["guidance"].as_f64()), (Some(40), Some(4.0)));
    assert_eq!((runs[1]["params"]["angles_strength"].as_f64(), runs[1]["params"]["lightning_strength"].as_f64()), (Some(1.25), Some(0.5)));
    assert_eq!(runs[2]["steps"].as_u64(), Some(30), "steps given win");
    drop(st);
    for p in [json!({"lightning": 6}), json!({"anglesStrength": 3}), json!({"lightningStrength": -1})] {
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

/// `swap` pins the 2.1 edit, adds the BFS LoRA and the author's prompt, and needs a reference.
#[test]
fn a_swap_sends_the_bfs_lora_and_prompt() {
    let s = Session::new();
    let p = json!({"swap": "face", "prompt": "smiling", "references": [{"path": "who.png"}], "enhance": true});
    let q = swap::apply(&s, &p).unwrap().unwrap();
    assert_eq!(q["template"], swap::SWAP_TEMPLATE);
    assert_eq!(q["enhance"], false, "a rewritten prompt would lose the trigger words");
    assert_eq!(q["loras"][0]["file"], swap::HEAD.lora);
    assert_eq!(q["loras"][0]["strength"], 1.0);
    let prompt = q["prompt"].as_str().unwrap();
    assert!(prompt.starts_with("head_swap: start with <image1>") && prompt.ends_with(" smiling"), "{prompt}");
    assert_eq!(q["name"], "Face Swap");
    assert_eq!(q["steps"], swap::SWAP_STEPS);
    assert_eq!(q["wholePicture"], true, "a swap keeps the whole re-rendered picture");
    assert!(q.get("swap").is_none());

    // A named template stays, a LoRA already listed is not added twice.
    let p = json!({"swap": "body", "template": "qwen-2.1/edit", "references": [{"layer": 2}],
        "loras": [{"file": swap::BODY.lora, "strength": 0.8}], "steps": 25});
    let q = swap::apply(&s, &p).unwrap().unwrap();
    assert_eq!(q["loras"].as_array().unwrap().len(), 1);
    assert_eq!(q["loras"][0]["strength"], 0.8);
    assert!(q["prompt"].as_str().unwrap().starts_with("body_swap:"));
    assert_eq!(q["steps"], 25, "named steps stay");

    assert!(swap::apply(&s, &json!({"prompt": "x"})).unwrap().is_none(), "no swap, no change");
    assert!(swap::apply(&s, &json!({"swap": "none"})).unwrap().is_none());
    assert!(swap::apply(&s, &json!({"swap": "head"})).is_err(), "a swap needs image 2");
    assert!(swap::apply(&s, &json!({"swap": "hands", "references": [{"layer": 2}]})).is_err());
}

#[test]
fn the_head_swap_asks_for_the_reference_persons_facial_hair() {
    assert!(swap::HEAD.prompt.ends_with(swap::HEAD_FACIAL_HAIR));
}

#[test]
fn an_h3_edit_is_sent_at_four_megapixels() {
    assert_eq!(edit_request_pixels("p-h3-image/edit", false), 2048 * 2048);
    assert_eq!(edit_request_pixels("qwen-edit-2511/edit", true), 1024 * 1024);
    assert_eq!(request_size_within(4000, 3000, edit_request_pixels("p-h3-image/edit", false)), (2352, 1760));
    assert_eq!(request_size_within(1600, 1200, edit_request_pixels("p-h3-image/edit", false)), (1600, 1200), "a picture under 4 MP goes as it is");
    // Quality Ultra: the 16 MP model edits a large picture whole
    assert_eq!(edit_request_pixels("minimax-h3-image/edit", false), 2048 * 2048);
    assert_eq!(request_size_within(4000, 3000, edit_request_pixels("minimax-h3-image/edit", true)), (4000, 3000));
    assert_eq!(edit_request_pixels("p-h3-image/edit", true), 2048 * 2048);
}

/// `repose` pins the 2.1 edit, adds the best installed PoseStudio LoRA and its prompt, and needs
/// the person as a reference (the open picture is the mannequin).
#[test]
fn a_repose_sends_the_posestudio_lora_and_prompt() {
    let dir = std::env::temp_dir().join(format!("pex-repose-{}", std::process::id()));
    let loras = dir.join("loras");
    std::fs::create_dir_all(&loras).unwrap();
    let mut s = Session::new();
    s.edit_prefs(|p| p.integrations.models_path = dir.to_string_lossy().into_owned());

    let p = json!({"repose": true, "references": [{"path": "who.png"}]});
    let err = repose::apply(&s, &p).unwrap_err().to_string();
    assert!(err.contains("PoseStudio"), "says what to download: {err}");

    std::fs::write(loras.join(repose::REPOSE_LORAS[2]), b"x").unwrap();
    std::fs::write(loras.join(repose::REPOSE_LORAS[1]), b"x").unwrap();
    let p = json!({"repose": true, "skeleton": true, "prompt": "laughing", "references": [{"path": "who.png"}], "enhance": true});
    let q = repose::apply(&s, &p).unwrap().unwrap();
    assert_eq!(q["template"], repose::REPOSE_TEMPLATE);
    assert_eq!(q["loras"][0]["file"], repose::REPOSE_LORAS[1], "V2 over V1.1 when V3 is missing");
    assert_eq!(q["loras"][0]["strength"], 1.0);
    assert_eq!(q["enhance"], false);
    assert_eq!(q["wholePicture"], true);
    assert_eq!(q["name"], "Repose");
    let prompt = q["prompt"].as_str().unwrap();
    assert!(prompt.starts_with("Draw character from image2.") && prompt.contains(repose::SKELETON_LINE) && prompt.ends_with("\nlaughing"), "{prompt}");
    assert!(q.get("repose").is_none() && q.get("skeleton").is_none());

    std::fs::write(loras.join(repose::REPOSE_LORAS[0]), b"x").unwrap();
    let q = repose::apply(&s, &json!({"repose": true, "references": [{"layer": 2}]})).unwrap().unwrap();
    assert_eq!(q["loras"][0]["file"], repose::REPOSE_LORAS[0], "V3 first");
    assert!(!q["prompt"].as_str().unwrap().contains("skeleton"), "no skeleton line by default");

    // A listed PoseStudio LoRA stays as asked and is not added twice.
    let p = json!({"repose": true, "references": [{"layer": 2}], "loras": [{"file": repose::REPOSE_LORAS[2], "strength": 0.7}]});
    let q = repose::apply(&s, &p).unwrap().unwrap();
    assert_eq!(q["loras"].as_array().unwrap().len(), 1);
    assert_eq!(q["loras"][0]["strength"], 0.7);

    assert!(repose::apply(&s, &json!({"prompt": "x"})).unwrap().is_none(), "no repose, no change");
    assert!(repose::apply(&s, &json!({"repose": false})).unwrap().is_none());
    assert!(repose::apply(&s, &json!({"repose": true})).is_err(), "a repose needs image 2");
    assert!(edit_rewrite(&s, &json!({"repose": true, "swap": "head", "references": [{"layer": 2}]})).is_err(), "not both");
    assert_eq!(edit_rewrite(&s, &json!({"repose": true, "references": [{"layer": 2}]})).unwrap().unwrap()["name"], "Repose");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_edit_shows_what_it_changed_not_only_its_subject() {
    use super::edit::changed_area;
    let (w, h) = (64usize, 64usize);
    let before: Vec<u8> = (0..w * h).flat_map(|_| [120u8, 130, 140, 255]).collect();
    // Re-rendered grain everywhere (±6 levels), and a 16×16 square repainted red at (40, 8).
    let mut after = before.clone();
    for (k, p) in after.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let n = ((k * 7919) % 13) as i16 - 6;
        for c in &mut p[..3] {
            *c = (i16::from(*c) + n).clamp(0, 255) as u8;
        }
        let (x, y) = (k % w, k / w);
        if (40..56).contains(&x) && (8..24).contains(&y) {
            p[..3].copy_from_slice(&[220, 30, 30]);
        }
    }
    let m = changed_area(&before, &after, w, h);
    assert!(m[16 * w + 48] > 0.99, "inside the square: {}", m[16 * w + 48]);
    assert!(m[50 * w + 8] < 0.01, "far from it: {}", m[50 * w + 8]);
    let mean = m.iter().sum::<f32>() / m.len() as f32;
    assert!(mean < 0.35, "{mean}");
    // A night edit darkens everything: all of it changed.
    let night: Vec<u8> = before.as_chunks::<4>().0.iter().flat_map(|p| [p[0] / 3, p[1] / 3, p[2] / 2, 255]).collect();
    let m = changed_area(&before, &night, w, h);
    assert!(m.iter().all(|v| *v > 0.99));
}

#[test]
fn an_edit_s_subject_mask_reaches_past_its_edge_and_fades_out() {
    // A 20×20 square in a 64×64 result: full inside and a little beyond, then a ramp to nothing.
    let (w, h) = (64usize, 64usize);
    let mut subject = vec![0.0f32; w * h];
    for y in 22..42 {
        for x in 22..42 {
            subject[y * w + x] = 1.0;
        }
    }
    let soft = super::soft_subject_edge(&subject, w, h);
    let r = super::subject_edge_radius(w, h);
    assert_eq!(r, 3);
    let at = |x: usize| soft[32 * w + x];
    assert_eq!(at(32), 1.0, "the inside stays whole");
    assert_eq!(at(42 + r - 1), 1.0, "the old outline is covered");
    assert!(at(42 + r) > 0.0 && at(42 + r) < 1.0 && at(42 + r + 2) < at(42 + r), "then it fades");
    assert!(at(42 + 4 * r) < 1e-3, "and stops");
    assert_eq!(at(0), 0.0);
}

#[test]
fn angles_run_on_qwen_2_1_where_2511_s_fp8_file_cannot() {
    // RTX 30 (8.6) and older: 2.1 with its own Angles LoRA; RTX 40 (8.9), RTX 50 and an unknown
    // card: 2511 as ever.
    assert_eq!(angles_template_for(Some(86)), ANGLES21_TEMPLATE);
    assert_eq!(angles_template_for(Some(75)), ANGLES21_TEMPLATE);
    assert_eq!(angles_template_for(Some(89)), ANGLES_TEMPLATE);
    assert_eq!(angles_template_for(Some(120)), ANGLES_TEMPLATE);
    assert_eq!(angles_template_for(None), ANGLES_TEMPLATE);
    assert!(photocraft_genai::template::find(ANGLES21_TEMPLATE).is_ok(), "the 2.1 route has a template");
    // The 2.1 LoRA's words: quarter views for our 45° steps, no low angle, close-up or not.
    assert_eq!(angle_prompt_21(45, 0, 5).as_deref(), Some("<mva> front-right quarter view, eye-level shot"));
    assert_eq!(angle_prompt_21(90, 30, 10).as_deref(), Some("<mva> right side view, elevated shot close-up"));
    assert_eq!(angle_prompt_21(180, 60, 0).as_deref(), Some("<mva> back view, high-angle shot"));
    assert_eq!(angle_prompt_21(270, -30, 5).as_deref(), Some("<mva> left side view, eye-level shot"));
    assert_eq!(angle_prompt_21(30, 0, 5), None);
}
