use egui::vec2;
use egui_kittest::Harness;
use photocraft_doc::LayerId;
use photocraft_genai::fake::FakeNative;
use serde_json::json;

use super::*;
use crate::control::{ControlRequest, Outcome, handle};

/// A 64×48 document with one empty layer, talking to `url`; no background jobs (inline runs).
fn app(url: &str) -> PhotocraftApp {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
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

fn control(app: &mut PhotocraftApp, ctx: &egui::Context, method: &str, params: Value) -> Value {
    let (req, _rx) = ControlRequest::new(method, params);
    match handle(app, ctx, &req) {
        Outcome::Done(v) => v,
        _ => panic!("{method}: the reply was deferred"),
    }
}

#[test]
fn the_menu_opens_the_bar_and_generate_runs_the_fill_with_variations() {
    let fake = FakeNative::start().unwrap();
    let mut app = app(&fake.url);
    let ctx = egui::Context::default();
    menu_opens_the_bar(&mut app, &ctx, &fake);
    generate_runs_the_variations(&mut app, &ctx, &fake);
    ui_set_drives_the_bar(&mut app, &ctx, &fake);
    edit_mode_runs_generate_edit(&mut app, &ctx, &fake);
    research_templates_run_whatever_the_preference_says(&mut app, &ctx, &fake);
}

fn menu_opens_the_bar(app: &mut PhotocraftApp, ctx: &egui::Context, fake: &FakeNative) {
    // Without a selection the menu row is disabled, like the command itself.
    assert!(crate::menus::invoke(app, ctx, COMMAND, json!({})).is_err());
    assert!(!app.ui.generative_bar.open);
    app.run("select.rect", json!({"x": 10, "y": 8, "width": 20, "height": 16})).unwrap();
    let r = crate::menus::invoke(app, ctx, COMMAND, json!({})).unwrap();
    assert_eq!(r["generativeBar"], true);
    assert!(app.ui.generative_bar.open && app.ui.generative_bar.focus);
    assert!(app.ui.dialogs.is_empty(), "the bar replaces the generated dialog");
    let ids: Vec<&str> = app.ui.generative_bar.templates.iter().map(|t| t.id.as_str()).collect();
    assert!(ids.contains(&"qwen-edit-2511/fill") && ids.contains(&"qwen-2.1/fill"), "{ids:?}");
    assert!(app.ui.generative_bar.templates.iter().find(|t| t.id == "qwen-2.1/fill").unwrap().allowed, "a research licence never gates");
    assert!(fake.state().runs.is_empty(), "listing templates queues nothing");
    // The installed badges: the probe (inline here) found every file of the built-in templates.
    assert!(app.ui.generative_bar.templates.iter().all(|t| t.installed == Some(true)), "{:?}", app.ui.generative_bar.templates);
}

fn generate_runs_the_variations(app: &mut PhotocraftApp, _ctx: &egui::Context, fake: &FakeNative) {
    // An empty prompt does nothing; a prompt generates, inline here (no background jobs).
    assert!(generate(app).is_err());
    assert!(fake.state().runs.is_empty());
    app.ui.generate.prompt = "a kite".into();
    app.ui.generate.count = 2;
    let layers_before = app.session.active().unwrap().doc.layers.len();
    generate(app).unwrap();
    assert_eq!(fake.state().runs.len(), 2, "one backend run per variation");
    assert!(app.ui.generative_bar.job.is_none());
    // The run's variations are the Variations strip's newest run.
    let run = crate::variations_strip::bundles(app).first().cloned().unwrap();
    assert_eq!((run.layers.len(), run.shown), (2, Some(0)));
    let d = app.session.active().unwrap();
    assert_eq!(d.doc.layers.len(), layers_before + 2);
    assert_eq!(d.doc.layer(LayerId(run.layers[0])).unwrap().name, "Generative Fill: a kite (1/2)");

    // The strip shows the second variation through generate.variation (one undo step).
    let steps = app.session.active().unwrap().history.past_len();
    crate::variations_strip::choose(app, run.record, 1).unwrap();
    let d = app.session.active().unwrap();
    let (a, b) = (LayerId(run.layers[0]), LayerId(run.layers[1]));
    assert!(!d.doc.layer(a).unwrap().visible && d.doc.layer(b).unwrap().visible);
    assert_eq!(d.history.past_len(), steps + 1);
    assert_eq!(crate::variations_strip::bundles(app)[0].shown, Some(1));
    assert!(crate::variations_strip::choose(app, run.record, 7).is_ok(), "an index past the end shows the last one");
    assert_eq!(crate::variations_strip::bundles(app)[0].shown, Some(1));
}

fn ui_set_drives_the_bar(app: &mut PhotocraftApp, ctx: &egui::Context, _fake: &FakeNative) {
    // ui.inspect shows the bar; ui.set drives and validates it.
    let v = control(app, ctx, "ui.inspect", json!({}));
    assert_eq!(v["result"]["generativeBar"]["prompt"], "a kite");
    assert_eq!(v["result"]["variations"]["runs"][0]["layers"].as_array().unwrap().len(), 2);
    let bad = control(app, ctx, "ui.set", json!({"generativeVariations": 9}));
    assert_eq!(bad["ok"], false, "{bad}");
    let bad = control(app, ctx, "ui.set", json!({"generativePrompt": 5}));
    assert_eq!(bad["ok"], false, "{bad}");
    let ok = control(app, ctx, "ui.set", json!({"generativePrompt": "a boat", "generativeVariations": 3, "generativeTemplate": "qwen-2.1/fill"}));
    assert_eq!(ok["ok"], true, "{ok}");
    // The Generate panel's fields: the bar shares them.
    let g = &app.ui.generate;
    assert_eq!((g.prompt.as_str(), g.count, g.template.get("fill").map(String::as_str)), ("a boat", 3, Some("qwen-2.1/fill")));
    let v = control(app, ctx, "ui.inspect", json!({}));
    assert_eq!(v["result"]["generatePanel"]["prompt"], "a boat");
    assert_eq!(v["result"]["generativeBar"]["variations"], 3);
    assert_eq!(control(app, ctx, "ui.set", json!({"generativeBar": false}))["ok"], true);
    assert!(!app.ui.generative_bar.open);
    assert_eq!(control(app, ctx, "ui.set", json!({"generativeBar": true}))["ok"], true);
    assert!(app.ui.generative_bar.open);
}

fn edit_mode_runs_generate_edit(app: &mut PhotocraftApp, ctx: &egui::Context, fake: &FakeNative) {
    // Edit mode: the same bar runs generate.edit (no noise mask, the selection masks the
    // layer); the picker lists edit templates; a prompt that changes what is there is spotted.
    assert_eq!(control(app, ctx, "ui.set", json!({"generativeMode": "paint"}))["ok"], false);
    assert_eq!(
        control(
            app,
            ctx,
            "ui.set",
            json!({"generativeMode": "edit", "generativePrompt": "turn this drawing into a realistic portrait", "generativeVariations": 1, "generativeTemplate": ""})
        )["ok"],
        true
    );
    assert!(app.ui.generative_bar.templates.iter().any(|t| t.task == "edit" && t.id == "qwen-edit-2511/edit"));
    assert!(looks_like_an_edit("make this hyper realistic") && looks_like_an_edit("Turn it into a painting"));
    assert!(!looks_like_an_edit("a red boat") && !looks_like_an_edit("make a red boat"));
    let prompts_before = fake.state().runs.len();
    generate(app).unwrap();
    let st = fake.state();
    assert_eq!(st.runs.len(), prompts_before + 1);
    assert!(st.runs.last().unwrap().run.image.is_some());
    assert_eq!(st.runs.last().unwrap().run.task, "edit");
    let g = &st.runs.last().unwrap().request;
    assert_eq!(g["prompt"], "turn this drawing into a realistic portrait. Keep everything else exactly as it is.");
    drop(st);
    let d = app.session.active().unwrap();
    assert!(d.doc.layers.last().unwrap().name.starts_with("Generative Edit:"));
    // Enhance rewrites the prompt with the picture in view (inline here) and leaves it for the
    // user to read and run.
    let prompts_before = fake.state().runs.len();
    enhance(app).unwrap();
    assert_eq!(app.ui.generate.prompt, "Enhanced: turn this drawing into a realistic portrait");
    assert!(app.ui.generative_bar.enhance_job.is_none());
    let st = fake.state();
    assert_eq!(st.runs.len(), prompts_before + 1, "the rewriter ran, nothing was generated");
    assert!(st.runs.last().unwrap().run.image.is_some());
    drop(st);
    app.ui.generate.prompt.clear();
    assert!(enhance(app).is_err(), "nothing to enhance");
    // Back to the Fill state the rest of the test expects.
    assert_eq!(
        control(
            app,
            ctx,
            "ui.set",
            json!({"generativeMode": "fill", "generativePrompt": "a boat", "generativeVariations": 3, "generativeTemplate": "qwen-2.1/fill"})
        )["ok"],
        true
    );
}

fn research_templates_run_whatever_the_preference_says(app: &mut PhotocraftApp, _ctx: &egui::Context, _fake: &FakeNative) {
    // A research template runs with the research preference off: it only steers Auto.
    assert!(!app.session.prefs().integrations.allow_research_models);
    generate(app).unwrap();
    let run = crate::variations_strip::bundles(app).first().cloned().unwrap();
    assert_eq!(run.layers.len(), 3);
    assert_eq!(app.session.active().unwrap().doc.layer(LayerId(run.layers[1])).unwrap().name, "Generative Fill: a boat (2/3)");
}

#[test]
fn the_preference_hides_the_bar_and_the_menu_falls_back_to_the_dialog() {
    let fake = FakeNative::start().unwrap();
    let mut app = app(&fake.url);
    let ctx = egui::Context::default();
    app.session.edit_prefs(|p| p.integrations.show_generative_bar = "never".into());
    app.run("select.rect", json!({"x": 10, "y": 8, "width": 20, "height": 16})).unwrap();
    show(&mut app, &ctx);
    assert!(!app.ui.generative_bar.open, "no bar when the preference is off");
    let r = crate::menus::invoke(&mut app, &ctx, COMMAND, json!({})).unwrap();
    assert!(r["dialog"].is_u64(), "{r}");
    assert_eq!(app.ui.dialogs.len(), 1);
    // With params, automation runs the engine directly either way.
    app.ui.close_dialog(r["dialog"].as_u64().unwrap());
    app.session.edit_prefs(|p| p.integrations.show_generative_bar = "auto".into());
    let r = crate::menus::invoke(&mut app, &ctx, COMMAND, json!({"prompt": "x"})).unwrap();
    assert!(r["layer"].is_u64(), "{r}");
    assert!(!app.ui.generative_bar.open);
}

fn harness(mut app: PhotocraftApp) -> Harness<'static, PhotocraftApp> {
    app.sync_views();
    let mut h = Harness::builder().with_size(vec2(1000.0, 700.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
            show(app, &ctx);
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::default());
    h.run_steps(4);
    h
}

fn bar_rect(h: &Harness<'static, PhotocraftApp>) -> Option<egui::Rect> {
    h.ctx.memory(|m| m.area_rect(egui::Id::unique(AREA_ID)))
}

#[test]
fn the_bar_appears_under_a_new_selection_inside_the_canvas_and_goes_with_it() {
    let fake = FakeNative::start().unwrap();
    let app = app(&fake.url);
    let mut h = harness(app);
    assert!(!h.state().ui.generative_bar.open, "no selection, no bar");
    h.state_mut().run("select.rect", json!({"x": 10, "y": 8, "width": 20, "height": 16})).unwrap();
    h.run_steps(3);
    assert!(h.state().ui.generative_bar.open, "a selection opens the bar");
    assert!(!h.state().ui.generative_bar.templates.is_empty(), "the picker has its templates without a menu click");
    let bar = bar_rect(&h).expect("the bar is drawn");
    let app = h.state();
    let xf = crate::canvas::ViewXform::active(app).unwrap();
    let sel = xf.doc_rect(app.session.active().unwrap().doc.selection.as_ref().unwrap().content_bounds());
    let canvas = crate::rulers::content_rect(app, app.last_canvas_rect);
    assert!(bar.top() >= sel.bottom(), "below the selection: bar {bar:?}, selection {sel:?}");
    assert!((bar.center().x - sel.center().x).abs() < 2.0, "centred on it: bar {bar:?}, selection {sel:?}");
    assert!(canvas.contains_rect(bar), "inside the canvas: bar {bar:?}, canvas {canvas:?}");
    // Closed with ×, it stays closed for this selection, and a new selection reopens it.
    h.state_mut().ui.generative_bar.open = false;
    h.run_steps(2);
    assert!(!h.state().ui.generative_bar.open);
    h.state_mut().run("select.deselect", json!({})).unwrap();
    h.run_steps(2);
    assert!(!h.state().ui.generative_bar.open, "no selection, no bar");
    h.state_mut().run("select.all", json!({})).unwrap();
    h.run_steps(3);
    assert!(h.state().ui.generative_bar.open, "a new selection reopens it");
    let bar = bar_rect(&h).expect("drawn again");
    let app = h.state();
    let canvas = crate::rulers::content_rect(app, app.last_canvas_rect);
    assert!(canvas.contains_rect(bar), "a selection filling the view keeps the bar on the canvas: {bar:?} in {canvas:?}");
}

#[test]
fn placement_prefers_below_then_above_then_the_canvas_bottom() {
    let canvas = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(800.0, 600.0));
    let size = vec2(500.0, 40.0);
    let below = place(egui::Rect::from_min_max(egui::pos2(100.0, 100.0), egui::pos2(300.0, 200.0)), size, canvas);
    assert_eq!(below, egui::pos2(8.0, 212.0), "centred would start at -50: clamped to the left edge");
    let above = place(egui::Rect::from_min_max(egui::pos2(300.0, 400.0), egui::pos2(500.0, 580.0)), size, canvas);
    assert_eq!(above, egui::pos2(150.0, 348.0));
    let huge = place(egui::Rect::from_min_max(egui::pos2(-100.0, -100.0), egui::pos2(900.0, 700.0)), size, canvas);
    assert_eq!(huge, egui::pos2(150.0, 552.0));
}
