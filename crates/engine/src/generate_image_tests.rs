use std::time::{Duration, Instant};

use photocraft_doc::LayerId;
use photocraft_genai::fake::FakeNative;
use photocraft_geom::Rect;
use serde_json::json;

use super::*;
use crate::jobs::{JobOutcome, Started};

const FAKE_COLOR: [f32; 4] = [200.0 / 255.0, 40.0 / 255.0, 40.0 / 255.0, 1.0];

fn close(a: &[f32], b: [f32; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 2.5 / 255.0)
}

/// A 64×48 document with one empty layer, talking to `url`.
fn session(url: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.edit_prefs(|p| {
        p.integrations.models_path = url.to_string();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    s
}

#[test]
fn image_lands_as_a_layer_covering_the_canvas() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    let (layers, steps) = {
        let d = s.active().unwrap();
        (d.doc.layers.len(), d.history.past_len())
    };
    let r = s.execute(IMAGE, json!({"prompt": "a lighthouse at dusk", "seed": 3})).unwrap();
    assert_eq!(r["template"], DEFAULT_IMAGE_TEMPLATE);
    assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(64), Some(48)));
    let id = LayerId(r["layer"].as_u64().unwrap());
    let d = s.active().unwrap();
    assert_eq!(d.doc.layers.len(), layers + 1);
    assert_eq!(d.active_layer, Some(id));
    assert_eq!(d.history.past_len(), steps + 1);
    let l = d.doc.layer(id).unwrap();
    assert_eq!(l.name, "Generated: a lighthouse at dusk");
    assert!(l.mask.is_none(), "no selection, no mask");
    let surf = l.surface().unwrap();
    assert_eq!(surf.content_bounds(), Rect::new(0, 0, 64, 48));
    assert!(close(&surf.read_region(Rect::new(30, 20, 31, 21)), FAKE_COLOR));

    let st = fake.state();
    assert!(st.pictures.is_empty(), "text to image uploads nothing");
    let g = &st.runs[0].request;
    assert_eq!(g["prompt"], "a lighthouse at dusk");
    assert_eq!(g["width"], 64);
    assert_eq!(g["height"], 64, "48 is below the 64-pixel minimum side; the result is resampled back to the canvas");
    assert_eq!(g["steps"], 8, "Krea 2 Turbo default");
    assert_eq!(g["guidance"], 1.0);
    assert_eq!(g["seed"], 3);
    assert_eq!(g["params"]["models"]["unet"], "krea2_turbo_fp8_scaled.safetensors");
    drop(st);

    assert!(s.undo());
    assert_eq!(s.active().unwrap().doc.layers.len(), layers);
}

#[test]
fn image_opens_a_new_document_when_none_is_open_or_when_asked() {
    let fake = FakeNative::start().unwrap();
    let mut s = Session::new();
    s.edit_prefs(|p| {
        p.integrations.models_path = fake.url.clone();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    assert!(s.is_enabled(IMAGE), "no document needed");
    let r = s.execute(IMAGE, json!({"prompt": "a poster", "width": 128, "height": 96})).unwrap();
    assert_eq!(r["document"], 0);
    assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(128), Some(96)));
    assert_eq!(s.documents().len(), 1);
    let d = s.active().unwrap();
    assert_eq!((d.doc.bounds().width(), d.doc.bounds().height()), (128, 96));
    assert_eq!(d.doc.layers.len(), 1);
    assert!(close(&d.doc.layers[0].surface().unwrap().read_region(Rect::new(100, 90, 101, 91)), FAKE_COLOR));
    assert_eq!(d.doc.name, "Generated: a poster");

    // With a document open, "document" still makes a new one and activates it.
    let r = s.execute(IMAGE, json!({"prompt": "another", "target": "document", "width": 256, "height": 256})).unwrap();
    assert_eq!(r["document"], 1);
    assert_eq!(s.documents().len(), 2);
    assert_eq!(s.active_index(), Some(1));
    assert_eq!(s.active().unwrap().doc.bounds().width(), 256);
}

#[test]
fn sizes_default_round_and_validate() {
    let fake = FakeNative::start().unwrap();
    let mut s = Session::new();
    s.edit_prefs(|p| {
        p.integrations.models_path = fake.url.clone();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    // 1000×700 rounds down to the 16-pixel grid.
    let r = s.execute(IMAGE, json!({"prompt": "x", "width": 1000, "height": 700})).unwrap();
    assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(992), Some(688)));
    assert_eq!(fake.state().runs[0].request["width"], 992);
    // Without a size and without a document: 1024².
    let r = s.execute(IMAGE, json!({"prompt": "x", "target": "document"})).unwrap();
    assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(1024), Some(1024)));
    for p in [
        json!({"prompt": "x", "width": 10}),
        json!({"prompt": "x", "height": 9000}),
        json!({"prompt": "x", "width": "big"}),
        json!({"prompt": "x", "target": "canvas"}),
    ] {
        assert!(matches!(s.execute(IMAGE, p.clone()), Err(EngineError::BadParams { .. })), "{p}");
    }
    // The generated dialog sends 0 for "default" sizes and steps, and "auto" for the target.
    let r = s.execute(IMAGE, json!({"prompt": "x", "width": 0, "height": 0, "target": "auto", "steps": 0, "guidance": 0})).unwrap();
    assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(1024), Some(1024)), "auto with a document open: a layer over its canvas");
    assert_eq!(fake.state().runs.last().unwrap().request["steps"], 8, "0 steps = the template default");
    let mut empty = Session::new();
    empty.edit_prefs(|p| {
        p.integrations.models_path = fake.url.clone();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    let e = empty.execute(IMAGE, json!({"prompt": "x", "target": "layer"})).unwrap_err();
    assert!(e.to_string().contains("open or create a document"), "{e}");
}

#[test]
fn image_runs_as_a_background_job() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    let layers = s.active().unwrap().doc.layers.len();
    let id = match s.start(IMAGE, json!({"prompt": "x"})).unwrap() {
        Started::Job(id) => id,
        Started::Done(v) => panic!("ran inline: {v}"),
    };
    let t = Instant::now();
    let e = loop {
        if let Some(e) = s.poll_jobs().into_iter().find(|e| e.id == id) {
            break e;
        }
        assert!(t.elapsed() < Duration::from_secs(30));
        std::thread::sleep(Duration::from_millis(5));
    };
    assert!(matches!(e.outcome, JobOutcome::Done(_)), "{e:?}");
    assert_eq!(s.active().unwrap().doc.layers.len(), layers + 1);

    // A new document from a background job too.
    let mut empty = Session::new();
    empty.edit_prefs(|p| {
        p.integrations.models_path = fake.url.clone();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    let id = match empty.start(IMAGE, json!({"prompt": "x", "width": 64, "height": 64})).unwrap() {
        Started::Job(id) => id,
        Started::Done(v) => panic!("ran inline: {v}"),
    };
    let v = empty.wait_job(id).unwrap();
    assert_eq!(v["document"], 0);
    assert_eq!(empty.documents().len(), 1);
}

#[test]
fn templates_are_checked_against_the_task() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    assert!(matches!(s.execute(IMAGE, json!({"prompt": "x", "template": "qwen-edit-2511/fill"})), Err(EngineError::BadParams { .. })));
    s.execute("select.rect", json!({"x": 1, "y": 1, "width": 8, "height": 8})).unwrap();
    assert!(matches!(s.execute(FILL, json!({"prompt": "x", "template": "krea2-turbo/image"})), Err(EngineError::BadParams { .. })));
    let m = s.execute(MODELS, json!({})).unwrap();
    let ids: Vec<&str> = m["templates"].as_array().unwrap().iter().map(|t| t["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"krea2-turbo/image") && ids.contains(&"qwen-edit-2511/fill"), "{ids:?}");
    let krea = m["templates"].as_array().unwrap().iter().find(|t| t["id"] == "krea2-turbo/image").unwrap();
    assert_eq!(krea["license"], "community");
    assert_eq!(krea["task"], "image");
    assert_eq!(krea["models"][0]["installed"], true);
}

#[test]
fn the_generate_model_preference_overrides_the_template() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    s.edit_prefs(|p| p.integrations.default_generate_model = "krea2_turbo_bf16.safetensors".into());
    s.execute(IMAGE, json!({"prompt": "x"})).unwrap();
    s.execute(IMAGE, json!({"prompt": "x", "model": "krea2_turbo_nvfp4.safetensors"})).unwrap();
    let st = fake.state();
    assert_eq!(st.runs[0].request["params"]["models"]["unet"], "krea2_turbo_bf16.safetensors");
    assert_eq!(st.runs[1].request["params"]["models"]["unet"], "krea2_turbo_nvfp4.safetensors");
}
