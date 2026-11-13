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
        // Sizes as asked: Quality best and balanced render Krea 2 larger (image_quality_size).
        p.integrations.quality = "fast".into();
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
        // Sizes as asked: Quality best and balanced render Krea 2 larger (image_quality_size).
        p.integrations.quality = "fast".into();
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
        // Sizes as asked: Quality best and balanced render Krea 2 larger (image_quality_size).
        p.integrations.quality = "fast".into();
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
        // Sizes as asked: Quality best and balanced render Krea 2 larger (image_quality_size).
        p.integrations.quality = "fast".into();
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
        // Sizes as asked: Quality best and balanced render Krea 2 larger (image_quality_size).
        p.integrations.quality = "fast".into();
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

/// Quality `best` runs Qwen-Image-2.1's text to image at 40 steps with true CFG 3 and the default
/// negative, `balanced` CFG 3 at 25 steps, `fast` the template's 25 at CFG 1; a typed negative or
/// explicit steps win.
#[test]
fn best_quality_tunes_qwen21_text_to_image() {
    assert_eq!(image_quality_tuning("qwen-2.1/image", "best"), Some((40, 3.0)));
    assert_eq!(image_quality_tuning("qwen-2.1/image", "balanced"), Some((25, 3.0)));
    assert_eq!(image_quality_tuning("qwen-2.1/image", "fast"), None);
    assert_eq!(image_quality_tuning("krea2-turbo/image", "fast"), None);
    let cost = image_quality_cost("qwen-2.1/image", "best");
    assert!((2.5..3.0).contains(&cost), "{cost}");
    assert_eq!(image_quality_cost("krea2-turbo/image", "best"), 9.0 / 8.0);
    assert_eq!(image_quality_cost("krea2-turbo/image", "fast"), 1.0);
}

/// Krea 2 Turbo under `best` and `balanced` runs the creator kit's detail recipe (9 steps,
/// `params.schedule = "krea-detail"`); `fast` keeps the template's 8 simple steps; a request's
/// `schedule` wins either way, and other templates refuse the Krea one.
#[test]
fn balanced_and_best_run_krea2_detail_schedule() {
    assert_eq!(image_quality_tuning("krea2-turbo/image", "best"), Some((KREA_DETAIL_STEPS, 1.0)));
    assert_eq!(image_quality_schedule("krea2-turbo/image", "balanced"), Some("beta"));
    assert_eq!(image_quality_schedule("krea2-turbo/image", "fast"), None);
    assert_eq!(image_quality_schedule("qwen-2.1/image", "best"), None);
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    let run = |s: &mut Session, extra: Value| {
        let mut p = json!({"prompt": "x", "template": "krea2-turbo/image", "target": "document", "width": 1024, "height": 1024});
        p.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        s.execute(IMAGE, p).unwrap();
        fake.state().runs.last().unwrap().request.clone()
    };
    s.edit_prefs(|p| p.integrations.quality = "best".into());
    let req = run(&mut s, json!({}));
    // Best: er_sde over beta at 9 steps (the realism A/B of 2026-11-12).
    assert_eq!((req["steps"].as_u64(), req["params"]["schedule"].as_str()), (Some(9), Some("beta")), "{req}");
    assert_eq!(req["params"]["sampler"], "er_sde", "{req}");
    // Asked for the plain schedule: the template's steps, sampler and schedule.
    let req = run(&mut s, json!({"schedule": "simple"}));
    assert_eq!((req["steps"].as_u64(), req["params"].get("schedule"), req["params"].get("sampler")), (Some(8), None, None), "{req}");
    let req = run(&mut s, json!({"sampler": "euler"}));
    assert_eq!((req["params"]["schedule"].as_str(), req["params"].get("sampler")), (Some("beta"), None), "{req}");
    s.edit_prefs(|p| p.integrations.quality = "fast".into());
    let req = run(&mut s, json!({}));
    assert_eq!((req["steps"].as_u64(), req["params"].get("schedule")), (Some(8), None), "{req}");
    let req = run(&mut s, json!({"schedule": "krea-detail"}));
    assert_eq!((req["steps"].as_u64(), req["params"]["schedule"].as_str()), (Some(9), Some("krea-detail")), "{req}");
    let err = s.execute(IMAGE, json!({"prompt": "x", "template": "qwen-2.1/image", "target": "document", "schedule": "krea-detail"}));
    assert!(err.is_err());
}

#[test]
fn best_quality_sends_qwen21_steps_guidance_and_negative() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    s.edit_prefs(|p| p.integrations.quality = "best".into());
    s.execute(IMAGE, json!({"prompt": "x", "template": "qwen-2.1/image", "target": "layer"})).unwrap();
    let req = fake.state().runs.last().unwrap().request.clone();
    assert_eq!((req["steps"].as_u64(), req["guidance"].as_f64()), (Some(40), Some(3.0)), "{req}");
    assert_eq!(req["negative"], BEST_IMAGE_NEGATIVE);
    // A typed negative is kept; explicit steps leave the template alone.
    s.execute(IMAGE, json!({"prompt": "x", "template": "qwen-2.1/image", "target": "layer", "negative": "fog"})).unwrap();
    assert_eq!(fake.state().runs.last().unwrap().request["negative"], "fog");
    s.execute(IMAGE, json!({"prompt": "x", "template": "qwen-2.1/image", "target": "layer", "steps": 25})).unwrap();
    assert_eq!(fake.state().runs.last().unwrap().request["guidance"].as_f64(), Some(1.0));
    s.edit_prefs(|p| p.integrations.quality = "fast".into());
    s.execute(IMAGE, json!({"prompt": "x", "template": "qwen-2.1/image", "target": "layer"})).unwrap();
    let req = fake.state().runs.last().unwrap().request.clone();
    assert_eq!((req["steps"].as_u64(), req["guidance"].as_f64()), (Some(25), Some(1.0)), "{req}");
}

#[test]
fn best_quality_renders_krea2_at_about_two_megapixels() {
    assert_eq!(image_quality_size("krea2-turbo/image", "best", 1024, 1024), (1440, 1440));
    assert_eq!(image_quality_size("krea2-turbo/image", "balanced", 1024, 1024), (1248, 1248));
    assert_eq!(image_quality_size("krea2-turbo/image", "fast", 1024, 1024), (1024, 1024));
    assert_eq!(image_quality_size("krea2-turbo/image", "best", 2048, 2048), (2048, 2048), "already larger: as asked");
    assert_eq!(image_quality_size("qwen-2.1/image", "best", 1024, 1024), (1024, 1024));
    let (w, h) = image_quality_size("krea2-turbo/image", "best", 1024, 576);
    assert!(w % 16 == 0 && h % 16 == 0 && w <= 2048 && (u64::from(w) * u64::from(h)).abs_diff(BEST_KREA_PIXELS) < 120_000, "{w}x{h}");
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    s.edit_prefs(|p| p.integrations.quality = "best".into());
    s.execute(IMAGE, json!({"prompt": "x", "template": "krea2-turbo/image", "target": "document", "width": 1024, "height": 1024})).unwrap();
    let req = fake.state().runs.last().unwrap().request.clone();
    assert_eq!((req["width"].as_u64(), req["height"].as_u64()), (Some(1440), Some(1440)), "{req}");
}

/// `refine` asks the native engine for Krea 2's second pass at `scale` times the size, on the
/// 16-px grid and at most 2560 a side; other templates refuse it.
#[test]
fn image_refine_sends_the_larger_size() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    s.edit_prefs(|p| p.integrations.quality = "best".into());
    let ask = |r: Value| json!({"prompt": "x", "template": "krea2-turbo/image", "target": "document", "width": 1024, "height": 1024, "refine": r});
    s.execute(IMAGE, ask(json!({"scale": 1.5, "denoise": 0.3, "steps": 4}))).unwrap();
    let req = fake.state().runs.last().unwrap().request.clone();
    assert_eq!(req["params"]["refine"], json!({"width": 2160, "height": 2160, "denoise": 0.3, "steps": 4}), "{req}");
    s.execute(IMAGE, ask(json!({"scale": 2.0, "denoise": 0.3, "steps": 4}))).unwrap();
    let req = fake.state().runs.last().unwrap().request.clone();
    assert_eq!(req["params"]["refine"]["width"], 2560, "capped: {req}");
    s.execute(IMAGE, ask(Value::Bool(false))).unwrap();
    assert!(fake.state().runs.last().unwrap().request["params"].get("refine").is_none());
    assert!(s.execute(IMAGE, ask(json!({"scale": 3.0, "denoise": 0.3, "steps": 4}))).is_err());
    let r = s.execute(IMAGE, json!({"prompt": "x", "template": "qwen-2.1/image", "target": "document", "refine": {"scale": 1.5, "denoise": 0.3, "steps": 4}}));
    assert!(r.is_err());
}

/// `sampler: "er_sde"` and `schedule: "beta"` reach the native engine on Krea 2 Turbo only.
#[test]
fn image_sampler_and_beta_schedule_pass_through() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    s.execute(IMAGE, json!({"prompt": "x", "template": "krea2-turbo/image", "target": "document", "sampler": "er_sde", "schedule": "beta", "steps": 9}))
        .unwrap();
    let req = fake.state().runs.last().unwrap().request.clone();
    assert_eq!(
        (req["params"]["sampler"].as_str(), req["params"]["schedule"].as_str(), req["steps"].as_u64()),
        (Some("er_sde"), Some("beta"), Some(9)),
        "{req}"
    );
    assert!(s.execute(IMAGE, json!({"prompt": "x", "template": "qwen-2.1/image", "target": "document", "sampler": "er_sde"})).is_err());
}

#[test]
fn best_runs_krea2_on_er_sde_and_beta_balanced_on_euler() {
    use crate::generate_cmds::{image_quality_sampler, image_quality_schedule};
    assert_eq!(image_quality_schedule("krea2-turbo/image", "best"), Some("beta"));
    assert_eq!(image_quality_sampler("krea2-turbo/image", "best"), Some("er_sde"));
    assert_eq!(image_quality_schedule("krea2-turbo/image", "balanced"), Some("beta"));
    assert_eq!(image_quality_sampler("krea2-turbo/image", "balanced"), Some("er_sde"));
    assert_eq!(image_quality_sampler("krea2-turbo/image", "fast"), None);
    assert_eq!(image_quality_sampler("qwen-2.1/image", "best"), None);
}

#[test]
fn best_renders_h3_at_eight_megapixels() {
    use crate::generate_cmds::image_quality_size;
    assert_eq!(image_quality_size("p-h3-image/image", "best", 1024, 1024), (2912, 2912));
    assert_eq!(image_quality_size("p-h3-image/image", "best", 1024, 512), (4096, 2048));
    assert_eq!(image_quality_size("p-h3-image/image", "balanced", 1024, 1024), (1024, 1024));
    assert_eq!(image_quality_size("p-h3-image/image", "best", 3200, 3200), (3200, 3200), "a larger ask is kept");
}
