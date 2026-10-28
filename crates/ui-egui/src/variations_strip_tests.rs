use std::time::{Duration, Instant};

use egui::vec2;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use photocraft_genai::fake::{FakeComfy, Options};
use serde_json::json;

use super::*;

/// A 256×192 document with a selection, talking to `url`; runs inline unless `background`.
fn app(url: &str) -> PhotocraftApp {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 256, "height": 192})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 40, "y": 30, "width": 120, "height": 90})).unwrap();
    s.edit_prefs(|p| p.integrations.set_comfy_route(url.to_string()));
    let mut app = PhotocraftApp::new(s, crate::Services::default());
    app.sync_views();
    app
}

/// Draw the strip over a 900×600 canvas.
fn harness(app: PhotocraftApp) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(900.0, 600.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            app.last_canvas_rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), vec2(900.0, 600.0));
            show(app, ui.ctx());
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, Default::default());
    h.run_steps(4);
    h
}

fn layers(h: &Harness<'_, PhotocraftApp>) -> usize {
    h.state().session.active().unwrap().doc.layers.len()
}

fn steps(h: &Harness<'_, PhotocraftApp>) -> usize {
    h.state().session.active().unwrap().history.past_len()
}

#[test]
fn three_variations_are_three_tiles_and_clicks_switch_drop_and_keep_in_one_step_each() {
    let fake = FakeComfy::start().unwrap();
    let mut a = app(&fake.url);
    a.run("generate.fill", json!({"prompt": "a kite", "variations": 3})).unwrap();
    let mut h = harness(a);
    let v = inspect(h.state(), &h.ctx);
    assert_eq!((v["place"].as_str(), v["visible"].as_bool()), (Some("right"), Some(true)), "{v}");
    let run = &v["runs"][0];
    assert_eq!((run["status"].as_str(), run["shown"].as_u64(), run["tiles"].as_array().map(Vec::len)), (Some("done"), Some(1), Some(3)), "{run}");
    let ids: Vec<u64> = run["layers"].as_array().unwrap().iter().filter_map(Value::as_u64).collect();
    // The column sits at the canvas's right edge.
    let first = &run["tiles"][0]["rect"];
    assert!(first[0].as_f64().unwrap() > 780.0 && first[2].as_f64() == Some(f64::from(TILE)), "{first}");
    // Click the second tile: it shows, the others hide, one undo step.
    let (n, s0) = (layers(&h), steps(&h));
    h.get_by_label("Variation 2 of 3").click();
    h.run_steps(3);
    let d = h.state().session.active().unwrap();
    let visible: Vec<bool> = ids.iter().map(|id| d.doc.layer(LayerId(*id)).unwrap().visible).collect();
    assert_eq!(visible, [false, true, false]);
    assert_eq!((steps(&h), bundles(h.state())[0].shown), (s0 + 1, Some(1)));
    // × on the third deletes that layer (one step); the shown one stays.
    h.get_all_by_label("Delete this variation").nth(2).unwrap().click();
    h.run_steps(3);
    assert_eq!((layers(&h), steps(&h)), (n - 1, s0 + 2));
    assert_eq!(bundles(h.state())[0].layers, [ids[0], ids[1]]);
    // ✓ on the shown one keeps it and deletes the rest.
    h.get_by_label("Keep this one").click();
    h.run_steps(3);
    assert_eq!((layers(&h), steps(&h)), (n - 2, s0 + 3));
    let b = bundles(h.state());
    assert_eq!((b[0].layers.clone(), b[0].shown), (vec![ids[1]], Some(0)));
    assert!(h.query_by_label("Keep this one").is_none(), "nothing left to choose between");
    // Undo brings the run's layers back into the strip.
    assert!(h.state_mut().session.undo());
    h.run_steps(2);
    assert_eq!(bundles(h.state())[0].layers.len(), 2);
}

#[test]
fn a_running_job_shows_placeholders_then_its_tiles_and_a_failed_one_offers_retry() {
    let fake = FakeComfy::start_with(Options { delay_ms: 300, ..Options::default() }).unwrap();
    let mut a = app(&fake.url);
    a.background_jobs = true;
    let r = a.run("generate.fill", json!({"prompt": "a kite", "variations": 2})).unwrap();
    assert_eq!(r["pending"], true, "{r}");
    let mut h = harness(a);
    let v = inspect(h.state(), &h.ctx);
    let run = &v["runs"][0];
    assert_eq!((run["status"].as_str(), run["pending"].as_array().map(Vec::len)), (Some("running"), Some(2)), "{run}");
    // The strip shows its percent and Cancel: no modal progress dialog over the canvas.
    let job = photocraft_engine::jobs::JobId(r["job"].as_u64().unwrap());
    assert!(shows_job(h.state(), &h.ctx, job));
    // The job ends: the placeholders become the two results.
    let t = Instant::now();
    while h.state().session.has_jobs() && t.elapsed() < Duration::from_secs(20) {
        let _ = h.state_mut().session.poll_jobs();
        h.run_steps(1);
        std::thread::sleep(Duration::from_millis(20));
    }
    h.run_steps(3);
    let v = inspect(h.state(), &h.ctx);
    assert_eq!((v["runs"][0]["status"].as_str(), v["runs"][0]["tiles"].as_array().map(Vec::len)), (Some("done"), Some(2)), "{v}");

    // A failure: a red tile with the reason, Retry (generate.recreate) and Dismiss.
    let bad = FakeComfy::start_with(Options { fail: Some("CUDA out of memory".into()), ..Options::default() }).unwrap();
    let mut a = app(&bad.url);
    assert!(a.run("generate.fill", json!({"prompt": "a kite"})).is_err());
    let mut h = harness(a);
    let v = inspect(h.state(), &h.ctx);
    let run = &v["runs"][0];
    assert_eq!(
        (run["status"].as_str(), run["failure"]["kind"].as_str(), run["retry"].as_array().map(Vec::len)),
        (Some("failed"), Some("outOfMemory"), Some(1)),
        "{run}"
    );
    let prompts = bad.state().prompts.len();
    h.get_by_label("Retry").click();
    h.run_steps(3);
    assert!(bad.state().prompts.len() > prompts, "Retry ran the request again");
    assert_eq!(bundles(h.state()).len(), 2, "the retry failed too: a second failed run");
    h.get_all_by_label("Dismiss").next().unwrap().click();
    h.run_steps(3);
    assert_eq!(bundles(h.state()).len(), 1, "dismissed");
}

#[test]
fn the_preference_moves_the_strip_to_the_bottom_or_hides_it() {
    let fake = FakeComfy::start().unwrap();
    let mut a = app(&fake.url);
    a.run("generate.fill", json!({"prompt": "a kite", "variations": 2})).unwrap();
    a.session.edit_prefs(|p| p.integrations.variations_strip = "bottom".into());
    let mut h = harness(a);
    let v = inspect(h.state(), &h.ctx);
    let tiles = v["runs"][0]["tiles"].as_array().unwrap().clone();
    assert_eq!(v["place"], "bottom");
    assert_eq!(tiles[0]["rect"][1], tiles[1]["rect"][1], "a row");
    assert!(tiles[0]["rect"][1].as_f64().unwrap() > 450.0, "along the bottom: {tiles:?}");
    h.state_mut().session.edit_prefs(|p| p.integrations.variations_strip = "off".into());
    h.run_steps(3);
    let v = inspect(h.state(), &h.ctx);
    assert_eq!((v["place"].as_str(), v["visible"].as_bool()), (Some("off"), Some(false)));
    assert!(h.query_by_label("Variation 1 of 2").is_none());
}

#[test]
fn in_the_whole_app_a_tile_click_reaches_only_the_strip() {
    let fake = FakeComfy::start().unwrap();
    let url = fake.url.clone();
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_max_steps(8).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut a = app(&url);
        a.run("generate.fill", json!({"prompt": "a kite", "variations": 3})).unwrap();
        a.ui.tool = crate::state::Tool::Brush;
        a
    });
    h.run_steps(6);
    let tool = h.state().ui.tool;
    let (n, s0) = (layers(&h), steps(&h));
    h.get_by_label("Variation 3 of 3").click();
    h.run_steps(4);
    assert_eq!(bundles(h.state())[0].shown, Some(2));
    assert_eq!(h.state().ui.tool, tool, "the click did not reach the tools");
    assert_eq!((layers(&h), steps(&h)), (n, s0 + 1), "one step, no brush dab");
}

#[test]
fn similar_adds_a_run_and_other_commands_stay_out() {
    let fake = FakeComfy::start().unwrap();
    let mut a = app(&fake.url);
    a.run("generate.fill", json!({"prompt": "a kite", "variations": 2})).unwrap();
    let mut h = harness(a);
    let prompts = fake.state().prompts.len();
    h.get_by_label("Generate Similar").click();
    h.run_steps(3);
    assert_eq!(fake.state().prompts.len(), prompts + 2, "as many as the run had");
    let b = bundles(h.state());
    assert_eq!((b.len(), b[0].command.as_str(), b[0].layers.len()), (2, "generate.fill", 2), "{b:?}");
    // A model-backed selection is in the library but is not a run of variations.
    let records = h.state().session.generations.records().len();
    let _ = h.state_mut().run("select.byText", json!({"prompt": "the kite"}));
    assert_eq!(h.state().session.generations.records().len(), records + 1, "recorded");
    h.run_steps(2);
    assert_eq!(bundles(h.state()).len(), 2, "not in the strip");
    // Five runs at most.
    for _ in 0..5 {
        h.state_mut().run("generate.fill", json!({"prompt": "a boat"})).unwrap();
    }
    h.run_steps(2);
    assert_eq!(bundles(h.state()).len(), MAX_BUNDLES);
}
