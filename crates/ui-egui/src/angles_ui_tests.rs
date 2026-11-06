use egui::vec2;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use photocraft_genai::fake::{FakeNative, Options};
use serde_json::json;

use super::*;

/// A 96×64 document talking to `url` (inline runs).
fn app(url: &str) -> PhotocraftApp {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 96, "height": 64})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.edit_prefs(|p| {
        p.integrations.models_path = url.to_string();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    let mut app = PhotocraftApp::new(s, crate::Services::default());
    app.sync_views();
    app
}

#[test]
fn the_camera_snaps_to_the_loras_stops() {
    assert_eq!((snap_rotation(10.0), snap_rotation(30.0), snap_rotation(-40.0), snap_rotation(350.0)), (0, 45, 315, 0));
    assert_eq!((snap_tilt(-50.0), snap_tilt(10.0), snap_tilt(20.0), snap_tilt(99.0)), (-30, 0, 30, 60));
}

#[test]
fn the_menu_opens_the_window_and_ui_set_generates_into_the_variations_strip() {
    let fake = FakeNative::start().unwrap();
    let mut a = app(&fake.url);
    let ctx = egui::Context::default();
    crate::menus::invoke(&mut a, &ctx, "generate.angles", json!({})).unwrap();
    assert!(a.ui.angles.open);
    // Every value is checked before any is applied.
    for bad in [
        json!({"angles": {"rotation": 30}}),
        json!({"angles": {"tilt": 45}}),
        json!({"angles": {"zoom": 3}}),
        json!({"angles": {"spin": 1}}),
        json!({"angles": 5}),
    ] {
        assert!(set(&mut a, &bad).is_err(), "{bad}");
    }
    assert_eq!((a.ui.angles.rotation, a.ui.angles.tilt), (45, 0), "unchanged by the refused calls");
    set(&mut a, &json!({"angles": {"rotation": 180, "tilt": 30, "zoom": 10, "variations": 2, "generate": true}})).unwrap();
    let st = fake.state();
    assert_eq!(st.runs.len(), 2, "two variations");
    assert_eq!(st.runs[0].request["prompt"], "<sks> back view elevated shot close-up");
    drop(st);
    let runs = crate::variations_strip::bundles(&a);
    assert_eq!((runs.len(), runs[0].layers.len()), (1, 2), "the results are a run of the Variations strip");
    let v = inspect(&a);
    assert_eq!((v["prompt"].as_str(), v["open"].as_bool()), (Some("<sks> back view elevated shot close-up"), Some(true)));
    // No picture: no window.
    let mut empty = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    assert!(crate::menus::invoke(&mut empty, &ctx, "generate.angles", json!({})).is_err());
}

#[test]
fn the_window_draws_the_orbit_and_warns_when_the_lora_is_missing() {
    let fake =
        FakeNative::start_with(Options { missing_files: vec!["qwen-image-edit-2511-multiple-angles-lora.safetensors".into()], ..Options::default() }).unwrap();
    let mut a = app(&fake.url);
    let ctx = egui::Context::default();
    crate::menus::invoke(&mut a, &ctx, "generate.angles", json!({})).unwrap();
    let mut h = Harness::builder().with_size(vec2(700.0, 560.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            window(app, ui.ctx());
        },
        a,
    );
    crate::PhotocraftApp::setup_context(&h.ctx, Default::default());
    h.run_steps(4);
    assert!(h.query_by_label_contains("multiple-angles-lora.safetensors").is_some(), "the missing file is named");
    assert!(h.query_by_label_contains("Generate ·").is_some(), "the estimate is on the button");
    // A click on the orbit moves the camera to the nearest stop.
    let orbit = h.get_by_label("Camera").rect();
    let before = (h.state().ui.angles.rotation, h.state().ui.angles.tilt);
    let at = egui::pos2(orbit.left() + 8.0, orbit.center().y + 18.0);
    h.hover_at(at);
    h.run();
    h.drag_at(at);
    h.run();
    h.drop_at(at);
    h.run_steps(3);
    let after = (h.state().ui.angles.rotation, h.state().ui.angles.tilt);
    assert_ne!(after, before, "moved");
    assert_eq!(after.0 % 45, 0);
    assert!(ANGLE_TILTS.iter().any(|(t, _)| *t == after.1));
}
