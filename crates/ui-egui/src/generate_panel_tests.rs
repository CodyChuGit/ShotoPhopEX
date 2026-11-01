use egui::vec2;
use egui_kittest::Harness;
use photocraft_genai::fake::FakeNative;
use serde_json::json;

use super::*;
use crate::control::{ControlRequest, Outcome, handle};

/// A 64×48 document with a textured layer and a small "Hat" layer, talking to `url`; no
/// background jobs (inline runs).
fn app(url: &str) -> (PhotocraftApp, u64, u64) {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("edit.fill", json!({"color": "#3366aa"})).unwrap();
    let base = s.active().unwrap().active_layer.unwrap().0;
    s.execute("layer.new.layer", json!({"name": "Hat"})).unwrap();
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 12})).unwrap();
    s.execute("edit.fill", json!({"color": "#cc2222"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    let hat = s.active().unwrap().active_layer.unwrap().0;
    s.execute("layer.select", json!({"layer": base})).unwrap();
    s.edit_prefs(|p| {
        p.integrations.models_path = url.to_string();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    let mut app = PhotocraftApp::new(s, crate::Services::default());
    app.sync_views();
    load_catalog(&mut app);
    (app, base, hat)
}

fn control(app: &mut PhotocraftApp, ctx: &egui::Context, method: &str, params: Value) -> Value {
    let (req, _rx) = ControlRequest::new(method, params);
    match handle(app, ctx, &req) {
        Outcome::Done(v) => v,
        _ => panic!("{method}: the reply was deferred"),
    }
}

#[test]
fn image_mode_with_layer_references_runs_the_reference_template_into_a_new_document() {
    let fake = FakeNative::start().unwrap();
    let (mut app, base, hat) = app(&fake.url);
    app.ui.generate.prompt = "a portrait wearing the hat from image 1".into();
    add_reference(&mut app, json!({"layer": hat}));
    add_reference(&mut app, json!({"layer": base}));
    let (cmd, q) = request(&app).unwrap();
    assert_eq!(cmd, "generate.image");
    assert_eq!(q["target"], "document");
    assert_eq!(q["references"], json!([{"layer": hat}, {"layer": base}]));
    assert!(q.get("template").is_none(), "Auto: the engine switches to the reference template");
    assert_eq!(effective_template(&app, "image"), photocraft_engine::generate_cmds::DEFAULT_IMAGE_REF_TEMPLATE);
    let docs = app.session.documents().len();
    let r = generate(&mut app).unwrap();
    assert_eq!(r["template"], "qwen-edit-2511/image-ref");
    assert_eq!(app.session.documents().len(), docs + 1, "a new document");
    let st = fake.state();
    assert_eq!(st.pictures.len(), 2, "the two references (no picture in image mode)");
    let g = &st.runs[0].request;
    assert_eq!((g["width"].as_u64(), g["height"].as_u64()), (Some(1024), Some(1024)), "1:1 at 1K");
}

#[test]
fn the_mode_decides_the_command_and_the_deck_follows_the_template() {
    let fake = FakeNative::start().unwrap();
    let (mut app, _, hat) = app(&fake.url);
    app.ui.generate.prompt = "make it night".into();
    app.ui.generate.mode = "fill".into();
    app.ui.generate.mode_chosen = true;
    assert!(request(&app).is_err(), "Fill needs a selection");
    assert_eq!(refs_limit(&app, "fill"), 2, "Auto: the fill-from-reference template's deck");
    app.run("select.rect", json!({"x": 4, "y": 4, "width": 20, "height": 20})).unwrap();
    assert_eq!(request(&app).unwrap().0, "generate.fill");
    // A reference in fill mode: sent along, and the panel expects the reference template.
    add_reference(&mut app, json!({"layer": hat}));
    let (cmd, q) = request(&app).unwrap();
    assert_eq!((cmd, q["references"].as_array().map(Vec::len)), ("generate.fill", Some(1)));
    let want = photocraft_engine::generate_cmds::fill_ref_template(photocraft_engine::generate_cmds::quality(&app.session));
    assert_eq!(effective_template(&app, "fill"), want);
    // Widen: the fill's `grow`, only when set.
    assert!(q.get("grow").is_none());
    app.ui.generate.grow = 6;
    assert_eq!(request(&app).unwrap().1["grow"], 6);
    let r = generate(&mut app).unwrap();
    assert_eq!(r["template"], want);
    assert_eq!(fake.state().pictures.len(), 3, "the picture, its mask and the reference");
    app.ui.generate.grow = 0;
    app.ui.generate.references.clear();
    app.ui.generate.mode = "edit".into();
    app.ui.generate.mode_chosen = true;
    assert_eq!(refs_limit(&app, "edit"), 2);
    add_reference(&mut app, json!({"layer": hat}));
    let (cmd, q) = request(&app).unwrap();
    assert_eq!((cmd, q["references"].as_array().map(Vec::len)), ("generate.edit", Some(1)));
    generate(&mut app).unwrap();
    assert_eq!(fake.state().pictures.len(), 5, "the picture and the reference, after the fill's three");
    // The limit: a third and fourth are refused with a status message.
    add_reference(&mut app, json!({"layer": 1}));
    add_reference(&mut app, json!({"layer": 2}));
    assert_eq!(app.ui.generate.references.len(), 2);
    assert!(app.ui.status.contains("at most 2"), "{}", app.ui.status);
    app.ui.generate.mode = "expand".into();
    app.ui.generate.mode_chosen = true;
    app.ui.generate.expand = [0, 0, 32, 0];
    let (cmd, q) = request(&app).unwrap();
    assert_eq!(cmd, "generate.expand");
    assert_eq!((q["right"].as_u64(), q.get("references")), (Some(32), None));
    app.ui.generate.expand = [0; 4];
    assert!(request(&app).is_err(), "nothing to expand");
    app.ui.generate.mode = "image".into();
    app.ui.generate.mode_chosen = true;
    app.ui.generate.into = "layer".into();
    app.ui.generate.references.clear();
    assert_eq!(request(&app).unwrap().1["target"], "layer");
}

#[test]
fn the_button_shows_the_formula_estimate() {
    let fake = FakeNative::start().unwrap();
    let (mut app, _, _) = app(&fake.url);
    app.ui.generate.mode = "image".into();
    app.ui.generate.mode_chosen = true;
    app.ui.generate.template.insert("image".into(), "krea2-turbo/image".into());
    app.ui.generate.aspect = "3:2".into();
    app.ui.generate.size = "1.5K".into();
    app.ui.generate.count = 2;
    assert_eq!(image_size(&app), (1536, 1024));
    let want = photocraft_engine::generate_cmds::estimate_seconds("krea2-turbo/image", 1536, 1024, 2, 0).unwrap();
    assert!((estimate(&app, "image").unwrap() - want).abs() < 1e-3);
    assert_eq!(eta_label(want), format!("~{:.0} s", want));
    assert_eq!(eta_label(150.0), "~3 min");
    assert_eq!(eta_label(0.0), "");
    // A custom size snaps to the template's grid and limits.
    app.ui.generate.aspect = "custom".into();
    app.ui.generate.width = 1000;
    app.ui.generate.height = 3000;
    assert_eq!(image_size(&app), (992, 2048));
    // Edit counts the picture at the size it is sent.
    app.ui.generate.mode = "edit".into();
    app.ui.generate.mode_chosen = true;
    let edit = estimate(&app, "edit").unwrap();
    // Auto follows the Quality preference (`best` by default: the full-step edit template).
    let auto = photocraft_engine::generate_cmds::likely_auto("generate.edit", photocraft_engine::generate_cmds::quality(&app.session));
    assert_eq!(auto, "qwen-edit-2511/edit");
    let e = photocraft_engine::generate_cmds::estimate_seconds(auto, 64, 48, 2, 0).unwrap();
    assert!((edit - e).abs() < 1e-3);
}

#[test]
fn ui_set_and_inspect_drive_the_panel_and_share_the_prompt_with_the_bar() {
    let fake = FakeNative::start().unwrap();
    let (mut app, _, hat) = app(&fake.url);
    let ctx = egui::Context::default();
    let ok = control(
        &mut app,
        &ctx,
        "ui.set",
        json!({"generatePanel": {"mode": "image", "prompt": "a cat", "aspect": "16:9", "size": "1K", "count": 3, "references": [{"layer": hat}], "into": "document"}}),
    );
    assert_eq!(ok["ok"], true, "{ok}");
    let v = control(&mut app, &ctx, "ui.inspect", json!({}));
    let p = &v["result"]["generatePanel"];
    assert_eq!(p["command"], "generate.image");
    assert_eq!(p["imageSize"], json!([1024, 576]));
    assert_eq!(p["params"]["variations"], 3);
    assert_eq!(p["effectiveTemplate"], "qwen-edit-2511/image-ref");
    assert_eq!(p["referencesMax"], 3);
    assert!(p["etaLabel"].as_str().is_some_and(|s| s.starts_with('~')), "{p}");
    // The bar's alias writes the same prompt.
    assert_eq!(control(&mut app, &ctx, "ui.set", json!({"generativePrompt": "a dog"}))["ok"], true);
    assert_eq!(control(&mut app, &ctx, "ui.inspect", json!({}))["result"]["generatePanel"]["prompt"], "a dog");
    // Validation happens before anything changes.
    for bad in [
        json!({"generatePanel": {"mode": "video"}}),
        json!({"generatePanel": {"count": 9}}),
        json!({"generatePanel": {"aspect": "wide"}}),
        json!({"generatePanel": {"template": "sam3.1/segment"}}),
        json!({"generatePanel": {"references": [{"pixels": 1}]}}),
        json!({"generatePanel": {"expand": [1, 2]}}),
        json!({"generatePanel": {"bogus": 1}}),
        json!({"generatePanel": 5}),
    ] {
        assert_eq!(control(&mut app, &ctx, "ui.set", bad.clone())["ok"], false, "{bad}");
    }
    assert_eq!(app.ui.generate.count, 3, "unchanged by the refused calls");
    // Switching the panel to Edit moves the bar too.
    assert_eq!(control(&mut app, &ctx, "ui.set", json!({"generatePanel": {"mode": "edit"}}))["ok"], true);
    assert_eq!(app.ui.generative_bar.mode, "edit");
}

/// Draw the panel alone at `height`.
fn harness(app: PhotocraftApp, height: f32) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(320.0, height)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                panel(app, ui);
            }
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, Default::default());
    h.run_steps(4);
    h
}

#[test]
fn enter_generates_only_when_the_preference_says_so_and_ctrl_enter_always() {
    let fake = FakeNative::start().unwrap();
    let (mut a, _, _) = app(&fake.url);
    a.ui.generate.mode = "image".into();
    a.ui.generate.mode_chosen = true;
    a.ui.generate.into = "layer".into();
    a.ui.generate.template.insert("image".into(), "krea2-turbo/image".into());
    a.ui.generate.prompt = "a lighthouse".into();
    a.ui.generate.focus = true;
    let mut h = harness(a, 420.0);
    assert!(!h.state().ui.generate.compact);
    h.key_press(egui::Key::Enter);
    h.run_steps(2);
    assert!(fake.state().runs.is_empty(), "Enter is a new line by default");
    h.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::Enter);
    h.run_steps(2);
    assert_eq!(fake.state().runs.len(), 1, "Ctrl+Enter generates");
    h.state_mut().session.edit_prefs(|p| p.integrations.enter_to_generate = true);
    h.state_mut().ui.generate.prompt = "a boat".into();
    h.run_steps(1);
    h.key_press(egui::Key::Enter);
    h.run_steps(2);
    assert_eq!(fake.state().runs.len(), 2, "Enter generates with the preference on");
}

#[test]
fn the_panel_opens_on_edit_over_an_open_picture_and_fill_over_a_selection() {
    let fake = FakeNative::start().unwrap();
    // A picture, no selection: Edit (an instruction changes the open picture).
    let (a, _, _) = app(&fake.url);
    let h = harness(a, 420.0);
    assert_eq!(h.state().ui.generate.mode, "edit");
    assert!(!h.state().ui.generate.mode_chosen, "a default, not a choice");
    // With a selection: Fill.
    let (mut a, _, _) = app(&fake.url);
    a.run("select.rect", json!({"x": 4, "y": 4, "width": 20, "height": 20})).unwrap();
    let h = harness(a, 420.0);
    assert_eq!(h.state().ui.generate.mode, "fill");
    // No document: Image.
    let mut s = photocraft_engine::Session::new();
    s.edit_prefs(|p| {
        p.integrations.models_path = fake.url.clone();
        p.integrations.runtime_path = "fake".into();
    });
    let h = harness(PhotocraftApp::new(s, crate::Services::default()), 420.0);
    assert_eq!(h.state().ui.generate.mode, "image");
}

#[test]
fn a_mode_the_person_picks_stays_whatever_the_canvas_holds() {
    use egui_kittest::kittest::Queryable;
    let fake = FakeNative::start().unwrap();
    let (a, _, _) = app(&fake.url);
    let mut h = harness(a, 420.0);
    assert_eq!(h.state().ui.generate.mode, "edit");
    h.get_by_label("Image").click();
    h.run_steps(2);
    assert_eq!(h.state().ui.generate.mode, "image");
    assert!(h.state().ui.generate.mode_chosen);
    // A selection later does not move it to Fill.
    h.state_mut().run("select.rect", json!({"x": 4, "y": 4, "width": 20, "height": 20})).unwrap();
    h.run_steps(2);
    assert_eq!(h.state().ui.generate.mode, "image");
}

#[test]
fn a_short_group_shows_one_row_and_drops_land_in_the_deck() {
    let fake = FakeNative::start().unwrap();
    let (a, _, _) = app(&fake.url);
    let h = harness(a, 60.0);
    assert!(h.state().ui.generate.compact, "one row");
    // The model picker stays reachable as the family glyph alone.
    {
        use egui_kittest::kittest::Queryable;
        assert!(h.query_by_label("Model: Auto").is_some(), "the glyph pill");
        let shown = crate::model_picker::inspect(&h.ctx);
        assert!(shown[0]["id"] == "generate-panel" && shown[0]["rect"][2].as_f64().is_some_and(|w| w < 60.0), "{shown}");
    }
    let rect = h.state().ui.generate.drop_rect.expect("the panel's rect");
    let target = h.state().drop_target(&egui::Context::default(), Some(rect.center()));
    assert_eq!(target, crate::file_open::DropTarget::References);
    // The bar steps aside while the panel is on screen (auto), and shows when it is hidden.
    let app = h.state();
    assert!(visible(app));
}

/// A safetensors file with these tensor names (one f32 each), for the LoRA list.
fn fake_lora(path: &std::path::Path, names: &[&str]) {
    let header: serde_json::Map<String, Value> =
        names.iter().enumerate().map(|(i, n)| (n.to_string(), json!({"dtype": "F32", "shape": [1], "data_offsets": [i * 4, i * 4 + 4]}))).collect();
    let h = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
    bytes.extend(h);
    bytes.extend(vec![0u8; names.len() * 4]);
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn the_lora_rows_list_the_templates_family_and_a_run_sends_the_ones_switched_on() {
    use egui_kittest::kittest::Queryable;
    let fake = FakeNative::start().unwrap();
    let (mut a, _, _) = app(&fake.url);
    let models = std::env::temp_dir().join(format!("pcx-loras-{}", std::process::id()));
    std::fs::create_dir_all(models.join("loras")).unwrap();
    fake_lora(
        &models.join("loras").join("restore.safetensors"),
        &["diffusion_model.transformer_blocks.0.attn.to_k.lora_A.weight", "diffusion_model.transformer_blocks.31.attn.to_k.lora_B.weight"],
    );
    fake_lora(&models.join("loras").join("anypaint.safetensors"), &["diffusion_model.blocks.0.attn.gate.lora_A.weight"]);
    a.session.edit_prefs(|p| {
        p.integrations.runtime_path = "C:/runtime".into();
        p.integrations.models_path = models.to_string_lossy().into_owned();
        p.integrations.allow_research_models = true;
    });
    a.ui.generate.mode = "edit".into();
    a.ui.generate.mode_chosen = true;
    a.ui.generate.template.insert("edit".into(), "qwen-2.1/edit".into());
    a.ui.generate.prompt = "restore the photo".into();
    let mut h = harness(a, 520.0);
    h.get_by_label("LoRAs").click();
    h.run_steps(2);
    assert!(h.query_by_label("restore").is_some(), "the 2.1 adapter is listed");
    assert!(h.query_by_label("anypaint").is_none(), "Krea 2's is not, for a 2.1 template");
    h.get_by_label("restore").click();
    h.run_steps(2);
    let v = inspect(h.state());
    assert_eq!(v["loras"]["files"], json!(["restore.safetensors"]));
    assert_eq!(v["params"]["loras"], json!([{"file": "restore.safetensors", "strength": 1.0}]), "{v}");
    let _ = std::fs::remove_dir_all(&models);
}

#[test]
fn denoise_goes_with_a_2511_fill_and_never_with_the_2_1_one() {
    let fake = FakeNative::start().unwrap();
    let (mut a, _, _) = app(&fake.url);
    a.session.edit_prefs(|p| p.integrations.allow_research_models = true);
    a.session.execute("select.rect", json!({"x": 8, "y": 8, "width": 24, "height": 16})).unwrap();
    a.ui.generate.mode = "fill".into();
    a.ui.generate.mode_chosen = true;
    a.ui.generate.prompt = "a red hat".into();
    a.ui.generate.template.insert("fill".into(), "qwen-edit-2511/fill-lightning-8".into());
    assert!(request(&a).unwrap().1.get("denoise").is_none(), "1 is the template's own");
    a.ui.generate.denoise = 0.6;
    assert_eq!(request(&a).unwrap().1["denoise"], json!(0.6));
    a.ui.generate.template.insert("fill".into(), "qwen-2.1/fill".into());
    assert!(request(&a).unwrap().1.get("denoise").is_none(), "the 2.1 fill redraws from noise");
    let mut h = harness(a, 520.0);
    h.run_steps(2);
    assert!(set(h.state_mut(), &json!({"generatePanel": {"denoise": 2.0}})).is_err(), "0.05 to 1");
}

/// The panel in every theme and mode at a dock's width, as PNGs in the folder that
/// `PHOTOCRAFT_GENERATE_SNAPSHOTS` names (pixel review, not a golden test; skipped without it).
#[test]
fn capture_generate_panel_snapshots() {
    let Some(directory) = std::env::var_os("PHOTOCRAFT_GENERATE_SNAPSHOTS").map(std::path::PathBuf::from) else { return };
    let _gpu = photocraft_testkit::gpu_lock();
    std::fs::create_dir_all(&directory).unwrap();
    let fake = FakeNative::start().unwrap();
    for kind in crate::theme::ThemeKind::ALL {
        for mode in MODES {
            let (mut a, base, _) = app(&fake.url);
            a.ui.generate.mode = mode.into();
            a.ui.generate.mode_chosen = true;
            a.ui.generate.prompt = "a lighthouse on a cliff at dusk, warm light, long exposure".into();
            if mode == "fill" {
                a.session.execute("select.rect", json!({"x": 8, "y": 8, "width": 24, "height": 16})).unwrap();
            }
            if mode == "image" {
                a.ui.generate.references.push(json!({"layer": base}));
            }
            let mut ready = false;
            let mut h = Harness::builder().with_size(vec2(320.0, 760.0)).wgpu().build_ui_state(
                move |ui, app: &mut PhotocraftApp| {
                    if !ready {
                        egui_extras::install_image_loaders(ui.ctx());
                        crate::theme::install_fonts(ui.ctx());
                        crate::theme::apply(ui.ctx(), kind);
                        ready = true;
                        return;
                    }
                    let t = Tokens::get(ui.ctx());
                    ui.painter().rect_filled(ui.max_rect(), 0.0, t.card);
                    egui::Frame::new().inner_margin(egui::Margin::same(8)).show(ui, |ui| panel(app, ui));
                },
                a,
            );
            h.run_steps(6);
            h.render().unwrap().save(directory.join(format!("generate-{}-{mode}.png", kind.id()))).unwrap();
        }
    }
}
