use egui::vec2;
use egui_kittest::Harness;
use photocraft_genai::fake::FakeNative;
use serde_json::json;

use super::*;
use crate::control::{ControlRequest, Outcome, handle};

fn temp_dir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("photocraft-generations-ui-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// A 64×48 document with a layer and a selection, talking to `url`, its library in `dir`.
fn app(url: &str, dir: &std::path::Path) -> PhotocraftApp {
    let mut s = photocraft_engine::Session::new();
    s.set_generations_dir(dir.to_path_buf());
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 10, "y": 8, "width": 20, "height": 16})).unwrap();
    s.edit_prefs(|p| {
        p.integrations.models_path = url.to_string();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    let mut app = PhotocraftApp::new(s, crate::Services::default());
    app.sync_views();
    app
}

fn control(app: &mut PhotocraftApp, ctx: &egui::Context, method: &str, params: Value) -> Value {
    let (req, _rx) = ControlRequest::new(method, params);
    match handle(app, ctx, &req) {
        Outcome::Done(v) => v,
        _ => panic!("{method}: the reply was deferred"),
    }
}

/// Draw the panel alone for a few frames.
fn draw(app: PhotocraftApp) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(300.0, 420.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            panel(app, ui);
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, Default::default());
    h.run_steps(4);
    h
}

#[test]
fn a_fill_shows_as_completed_with_its_thumbnail_and_opens_again_after_its_document_closed() {
    let fake = FakeNative::start().unwrap();
    let dir = temp_dir("completed");
    let mut app = app(&fake.url, &dir);
    let r = app.run("generate.fill", json!({"prompt": "a kite", "seed": 3})).unwrap();
    let layer = r["layer"].as_u64().unwrap();
    let mut h = draw(app);
    let v = inspect(h.state());
    assert_eq!(v["completed"], 1, "{v}");
    assert_eq!(v["thumbnails"], 1, "the thumbnail was decoded into a texture: {v}");
    let id = v["ids"][0].as_u64().unwrap();
    let ctx = h.ctx.clone();

    // Click while the document is open: its layer becomes active.
    let app = h.state_mut();
    app.run("layer.select", json!({"layer": app.session.active().unwrap().doc.layers[0].id.0})).unwrap();
    open(app, id);
    assert_eq!(app.session.active().unwrap().active_layer.map(|l| l.0), Some(layer));
    assert_eq!(app.ui.generations.last_viewed, Some(id));
    // Closed: the click opens the stored result as a new document of its size.
    while let Some(i) = app.session.active_index() {
        app.session.close(i);
    }
    open(app, id);
    let d = app.session.active().unwrap();
    assert_eq!((u64::from(d.doc.size.width), u64::from(d.doc.size.height)), (r["width"].as_u64().unwrap(), r["height"].as_u64().unwrap()));
    // Place as a layer into it.
    apply(app, &ctx, Action::Place(id, "layer"));
    assert_eq!(app.session.active().unwrap().doc.layers.len(), 2);
    h.run_steps(2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_mode_saves_the_picked_results_into_the_preferred_folder_under_its_naming() {
    use egui_kittest::kittest::Queryable;
    let fake = FakeNative::start().unwrap();
    let dir = temp_dir("select");
    let out = temp_dir("select-out");
    let mut a = app(&fake.url, &dir);
    for seed in [5, 6, 7] {
        a.run("generate.fill", json!({"prompt": "a kite", "seed": seed})).unwrap();
    }
    a.session.edit_prefs(|p| {
        p.integrations.save_results_dir = out.to_string_lossy().into_owned();
        p.integrations.save_results_naming = "kite_{seed}".into();
    });
    let mut h = draw(a);
    // Select mode: clicks pick instead of opening.
    h.get_by_label("Select").click();
    h.run_steps(2);
    assert_eq!(inspect(h.state())["selecting"], true);
    let ids: Vec<u64> = inspect(h.state())["ids"].as_array().unwrap().iter().filter_map(Value::as_u64).collect();
    let ctx = h.ctx.clone();
    apply(h.state_mut(), &ctx, Action::Toggle(ids[0]));
    apply(h.state_mut(), &ctx, Action::Toggle(ids[2]));
    apply(h.state_mut(), &ctx, Action::Toggle(ids[2]));
    apply(h.state_mut(), &ctx, Action::Toggle(ids[1]));
    h.run_steps(2);
    assert_eq!(inspect(h.state())["selected"], json!([ids[1], ids[0]]), "a set, toggled");
    assert!(h.query_by_label("2 selected").is_some());
    // Save selected: no dialog (the folder is set), the naming pattern, select mode ends.
    h.get_by_label("Save selected").click();
    h.run_steps(2);
    let mut names: Vec<String> = std::fs::read_dir(&out).unwrap().filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    assert_eq!(names, ["kite_6.png", "kite_7.png"], "the two newest");
    assert_eq!((inspect(h.state())["selecting"].as_bool(), h.state().ui.status.as_str()), (Some(false), "Saved 2 file(s)"));
    // Without a folder and without a save dialog (tests, web): it says so.
    h.state_mut().session.edit_prefs(|p| p.integrations.save_results_dir.clear());
    apply(h.state_mut(), &ctx, Action::Save(ids[0]));
    assert!(h.state().ui.status_error, "{}", h.state().ui.status);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn failed_runs_show_their_reason_and_dismiss_hides_them() {
    let dir = temp_dir("failed");
    let mut app = app("http://127.0.0.1:9", &dir);
    assert!(app.run("generate.fill", json!({"prompt": "a kite"})).is_err());
    let rec = app.session.generations.records().last().unwrap().clone();
    assert_eq!(rec.status, GenStatus::Failed);
    let kind = rec.failure.as_ref().unwrap().kind;
    assert_eq!(failure_label(kind), "Server unreachable");
    let mut h = draw(app);
    assert_eq!(inspect(h.state())["failed"], 1);
    let ctx = h.ctx.clone();
    let app = h.state_mut();
    apply(app, &ctx, Action::Dismiss(rec.id));
    assert_eq!(inspect(app)["failed"], 0);
    assert!(app.session.generations.get(rec.id).unwrap().dismissed, "hidden, not deleted");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_window_menu_toggles_the_panel_and_ui_set_switches_its_view() {
    let dir = temp_dir("menu");
    let mut app = app("http://127.0.0.1:9", &dir);
    let ctx = egui::Context::default();
    assert!(app.ui.panels.generations, "on in Essentials");
    assert!(crate::dock::Group::Generations.shown(&app.ui.panels));
    crate::menus::invoke(&mut app, &ctx, "window.panel.generations", Value::Null).unwrap();
    assert!(!app.ui.panels.generations);
    crate::menus::invoke(&mut app, &ctx, "window.toggle.generations", Value::Null).unwrap();
    assert!(app.ui.panels.generations);
    // A layout saved before the panel existed still shows it.
    let mut saved = serde_json::to_value(&app.ui.panels).unwrap();
    saved.as_object_mut().unwrap().remove("generations");
    let back: crate::state::Panels = serde_json::from_value(saved).unwrap();
    assert!(back.generations);

    assert_eq!(control(&mut app, &ctx, "ui.set", json!({"generationsView": "list"}))["ok"], true);
    assert_eq!(app.ui.generations.view, GenView::List);
    assert_eq!(control(&mut app, &ctx, "ui.set", json!({"generationsView": "tiles"}))["ok"], false);
    let v = control(&mut app, &ctx, "ui.inspect", json!({}));
    assert_eq!(v["result"]["generations"]["view"], "list");
    assert_eq!(v["result"]["panels"]["generations"], true);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_background_result_that_lands_while_the_panel_is_hidden_is_unread_until_it_is_drawn() {
    let fake = FakeNative::start_with(photocraft_genai::fake::Options { delay_ms: 100, ..Default::default() }).unwrap();
    let dir = temp_dir("unread");
    let mut app = app(&fake.url, &dir);
    app.background_jobs = true;
    app.frame = 50;
    let started = app.run("generate.fill", json!({"prompt": "a boat"})).unwrap();
    assert_eq!(started["pending"], true);
    assert_eq!(app.session.generations.running(), 1, "the rail badge counts it");
    let ctx = egui::Context::default();
    let t0 = std::time::Instant::now();
    while app.session.has_jobs() && t0.elapsed() < std::time::Duration::from_secs(20) {
        crate::jobs_ui::tick(&mut app, &ctx);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(!app.session.has_jobs());
    assert_eq!(app.ui.generations.unread.len(), 1);
    app.session.flush_generations();
    let mut h = draw(app);
    h.run_steps(2);
    assert_eq!(inspect(h.state())["unread"], 0, "drawn: read");
    assert_eq!(inspect(h.state())["completed"], 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_alerts_preference_picks_the_blip_for_each_ending() {
    let run = |url: &str, pref: &str, prompt: &str| -> Option<&'static str> {
        let dir = temp_dir(&format!("alert-{pref}-{}", prompt.len()));
        let mut app = app(url, &dir);
        app.background_jobs = true;
        app.session.edit_prefs(|p| p.integrations.alerts = pref.into());
        let _ = app.run("generate.fill", json!({"prompt": prompt}));
        let ctx = egui::Context::default();
        let t0 = std::time::Instant::now();
        while app.session.has_jobs() && t0.elapsed() < std::time::Duration::from_secs(20) {
            crate::jobs_ui::tick(&mut app, &ctx);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let _ = std::fs::remove_dir_all(&dir);
        app.ui.generations.last_alert
    };
    let ok = FakeNative::start().unwrap();
    let bad = FakeNative::start_with(photocraft_genai::fake::Options { fail: Some("CUDA out of memory".into()), ..Default::default() }).unwrap();
    assert_eq!(run(&ok.url, "off", "a boat"), None, "off by default");
    assert_eq!(run(&ok.url, "done", "a boat"), Some("done"));
    assert_eq!(run(&bad.url, "done", "a boat!"), None, "failures stay quiet under done");
    assert_eq!(run(&bad.url, "doneAndFailed", "a boat!!"), Some("failed"));
    assert_eq!(photocraft_engine::Session::new().prefs().integrations.alerts, "off");
}

#[test]
fn a_short_group_shows_the_compact_strip_with_every_kind_of_row() {
    let fake = FakeNative::start().unwrap();
    let dir = temp_dir("compact");
    let mut first = app(&fake.url, &dir);
    first.run("generate.fill", json!({"prompt": "a kite", "seed": 1})).unwrap();
    first.session.edit_prefs(|p| {
        p.integrations.models_path = "http://127.0.0.1:9".into();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    assert!(first.run("generate.fill", json!({"prompt": "a boat"})).is_err());
    // The group at its compact height: 80 pt minus the tab strip.
    let mut h = Harness::builder().with_size(vec2(300.0, 52.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                panel(app, ui);
            }
        },
        first,
    );
    PhotocraftApp::setup_context(&h.ctx, Default::default());
    h.run_steps(4);
    let v = inspect(h.state());
    assert_eq!(v["compact"], true, "{v}");
    assert_eq!((v["completed"].as_u64(), v["failed"].as_u64()), (Some(1), Some(1)));
    assert_eq!(v["thumbnails"], 1, "the result's thumbnail shows in the strip");
    // Tall enough: the full feed (the same library, read back by a new session).
    h.state_mut().session.flush_generations();
    let h = draw(app(&fake.url, &dir));
    let v = inspect(h.state());
    assert_eq!(v["compact"], false, "{v}");
    assert_eq!((v["completed"].as_u64(), v["failed"].as_u64()), (Some(1), Some(1)));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The dev log's numbers: panel frame time with 200 completed rows (thumbnails decoded) and the
/// index write time. `cargo test -p photocraft-ui-egui --lib bench_generations -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_generations_panel_with_200_rows() {
    let fake = FakeNative::start().unwrap();
    let dir = temp_dir("bench");
    let mut app = app(&fake.url, &dir);
    let t = std::time::Instant::now();
    for seed in 0..200 {
        app.run("generate.fill", json!({"prompt": format!("a kite number {seed}"), "seed": seed})).unwrap();
    }
    eprintln!("200 inline fills (fake server, stored results): {:.1} s", t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    app.session.flush_generations();
    eprintln!("index write, 200 rows: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    let bytes = std::fs::metadata(dir.join("index.json")).map(|m| m.len()).unwrap_or(0);
    eprintln!("index.json: {} KB", bytes / 1024);
    thread_local! {
        static PANEL_MS: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let mut h = Harness::builder().with_size(vec2(300.0, 700.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                let t = std::time::Instant::now();
                panel(app, ui);
                PANEL_MS.with(|v| v.borrow_mut().push(t.elapsed().as_secs_f64() * 1000.0));
            }
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, Default::default());
    for view in [GenView::Grid, GenView::List] {
        h.state_mut().ui.generations.view = view;
        h.run_steps(60);
        PANEL_MS.with(|v| v.borrow_mut().clear());
        let mut times = Vec::new();
        for _ in 0..30 {
            let t = std::time::Instant::now();
            h.step();
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(f64::total_cmp);
        let mut panel_ms = PANEL_MS.with(|v| v.borrow().clone());
        panel_ms.sort_by(f64::total_cmp);
        let mid = panel_ms.get(panel_ms.len() / 2).copied().unwrap_or(0.0);
        eprintln!(
            "{view:?}: panel() median {mid:.3} ms; whole harness frame median {:.2} ms, max {:.2} ms; {} thumbnails",
            times[15],
            times[29],
            inspect(h.state())["thumbnails"]
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn many_rows_draw_fast_and_only_a_few_thumbnails_decode_per_frame() {
    let fake = FakeNative::start().unwrap();
    let dir = temp_dir("many");
    let mut app = app(&fake.url, &dir);
    for seed in 0..10 {
        app.run("generate.fill", json!({"prompt": "a kite", "seed": seed})).unwrap();
    }
    let mut h = Harness::builder().with_size(vec2(300.0, 2000.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                panel(app, ui);
            }
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, Default::default());
    h.run_steps(2);
    let first = inspect(h.state())["thumbnails"].as_u64().unwrap();
    assert!(first <= DECODES_PER_FRAME as u64 * 2, "{first}");
    h.run_steps(4);
    assert_eq!(inspect(h.state())["thumbnails"], 10);
    // The list view draws the same rows.
    h.state_mut().ui.generations.view = GenView::List;
    let t = std::time::Instant::now();
    h.run_steps(5);
    assert!(t.elapsed() < std::time::Duration::from_secs(5));
    let _ = std::fs::remove_dir_all(&dir);
}
