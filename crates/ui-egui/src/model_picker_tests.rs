use egui::vec2;
use egui_kittest::Harness;
use photocraft_genai::fake::{FakeNative, Options};
use serde_json::json;

use super::*;
use crate::PhotocraftApp;

fn item(id: &str, name: &str, family: &str) -> Item {
    Item { id: id.into(), name: name.into(), family: family.into(), allowed: true, ..Item::default() }
}

#[test]
fn names_seconds_and_bars_read_well() {
    assert_eq!(short_name("Generative Fill (Qwen-Image-Edit-2511, Lightning 8 steps)"), "Qwen-Image-Edit-2511, Lightning 8 steps");
    assert_eq!(short_name("SAM"), "SAM");
    let krea = [
        item("a", "Generate Image (Krea 2 Turbo)", "krea2"),
        item("b", "Generate Image with a Style Reference (Krea 2 Turbo)", "krea2"),
        item("c", "Generate Image (Qwen-Image-2.1)", "qwen-image-2.1"),
    ];
    assert_eq!(
        display_names(&krea),
        ["Generate Image (Krea 2 Turbo)", "Generate Image with a Style Reference (Krea 2 Turbo)", "Qwen-Image-2.1"],
        "colliding short names fall back to the full names"
    );
    // Seconds at the current Speed and Quality; a download mark (None) when files are missing.
    let mut m = item("x", "X", "krea2");
    assert_eq!(row_seconds(Some(&m)), Some(String::new()), "unknown: nothing");
    m.eta = Some(16.2);
    assert_eq!(row_seconds(Some(&m)).as_deref(), Some("~16 s"));
    m.installed = Some(false);
    assert_eq!(row_seconds(Some(&m)), None);
    // The bars.
    assert_eq!(speed_score(Some(5.0), Some(5.0), &[]), 5, "the quickest row");
    assert_eq!(speed_score(Some(25.0), Some(5.0), &[]), 1);
    assert_eq!(speed_score(None, None, &[Speed::Full]), 2, "unknown seconds: from the speeds offered");
    assert_eq!(speed_score(None, None, &[Speed::Fast, Speed::Balanced]), 4);
    assert_eq!(memory_score(Some(18.0), Some(22.0)), 5);
    assert_eq!(memory_score(Some(7.6), Some(30.0)), 2);
    assert_eq!(memory_score(None, Some(30.0)), 0, "unknown: no bar");
}

#[test]
fn the_pill_names_the_model_not_the_recipe() {
    let items =
        [item("krea2-turbo/image", "Generate Image (Krea 2 Turbo)", "krea2"), item("moss-image/image-base", "Generate Image (MOSS base)", "moss-image")];
    assert_eq!(label_for(&items, "", "image"), "Auto");
    assert_eq!(label_for(&items, "krea2-turbo/image", "image"), "Krea 2");
    assert_eq!(label_for(&items, "moss-image/image-base", "image"), "MOSS base", "a recipe no row shows: its own name");
    let select = [item("sam3.1/segment", "Select by Text (SAM 3.1)", "sam3.1"), item("sam3.1/segment-box", "Select by Box (SAM 3.1 box)", "sam3.1")];
    assert_eq!(label_for(&select, "sam3.1/segment-box", "select"), "SAM 3.1 box", "a tool's page: one row per template");
}

#[test]
fn the_panel_lists_the_pages_templates_with_estimates_and_the_probe_marks_missing_files() {
    // The fake leaves the Lightning 8 LoRA out: those templates are "not installed".
    let fake =
        FakeNative::start_with(Options { missing_files: vec!["Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors".into()], ..Options::default() })
            .unwrap();
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    s.execute("select.rect", json!({"x": 4, "y": 4, "width": 20, "height": 20})).unwrap();
    s.edit_prefs(|p| {
        p.integrations.models_path = fake.url.clone();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    let mut app = PhotocraftApp::new(s, crate::Services::default());
    crate::generate_panel::load_catalog(&mut app);
    // Loading the catalogue stays offline (the panel is on screen from startup).
    assert!(!app.session.has_jobs() && app.ui.generate.catalog.iter().all(|m| m.installed.is_none()));
    crate::generate_panel::probe_once(&mut app);
    assert!(app.ui.generate.probed);
    let items = crate::generate_panel::picker_items(&app, "fill", true);
    let ids: Vec<&str> = items.iter().map(|i| i.id.as_str()).collect();
    assert!(ids.contains(&"qwen-edit-2511/fill-lightning-8") && ids.contains(&"qwen-2.1/fill"), "{ids:?}");
    assert!(!ids.contains(&"krea2-turbo/image"), "another page's template");
    let l8 = items.iter().find(|i| i.id == "qwen-edit-2511/fill-lightning-8").unwrap();
    assert_eq!(l8.installed, Some(false), "probe_once (inline here) found the LoRA missing");
    assert!(l8.eta.is_some_and(|e| e > 0.0));
    let base = items.iter().find(|i| i.id == "qwen-edit-2511/fill").unwrap();
    assert_eq!(base.installed, Some(true));
    // A research licence is a badge, never a block, whatever the preference says.
    let research = items.iter().find(|i| i.id == "qwen-2.1/fill").unwrap();
    assert!(research.research && research.allowed);
    // The generated dialog of Generate Image carries the rows and draws the pill.
    let dialog = crate::filter_dialog::open(&mut app, "generate.image").unwrap();
    let fields = app.ui.dialogs.iter().find(|d| d.id == dialog).unwrap().fields.clone();
    let models = fields["__models"].as_array().unwrap();
    assert!(models.iter().any(|m| m["id"] == "krea2-turbo/image") && models.iter().all(|m| m["id"] != "qwen-edit-2511/fill"));
}

fn pill_items() -> Vec<Item> {
    let mut moss = item("moss-image/turbo", "Generate Image (MOSS Turbo)", "moss-image");
    moss.installed = Some(false);
    let mut krea = item("krea2-turbo/image", "Generate Image (Krea 2 Turbo)", "krea2");
    krea.eta = Some(9.0);
    vec![krea, moss]
}

#[test]
fn the_pill_chooses_a_model_asks_before_a_missing_one_and_goes_back_to_auto() {
    use egui_kittest::kittest::Queryable;
    let mut h = Harness::builder().with_size(vec2(520.0, 420.0)).build_ui_state(
        |ui, (current, changes): &mut (String, u32)| {
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            let opts = Opts { page: "image", speed: Speed::Balanced, auto: Some("Krea 2"), ..Opts::default() };
            if pill(ui, "test", &pill_items(), current, &opts, 200.0) {
                *changes += 1;
            }
        },
        (String::new(), 0),
    );
    PhotocraftApp::setup_context(&h.ctx, Default::default());
    h.run_steps(3);
    let shown = inspect(&h.ctx);
    assert_eq!((shown[0]["id"].as_str(), shown[0]["label"].as_str(), shown[0]["open"].as_bool()), (Some("test"), Some("Auto"), Some(false)), "{shown}");
    h.get_by_label("Model: Auto").click();
    h.run_steps(3);
    assert!(is_open(&h.ctx, "test") && inspect(&h.ctx)[0]["open"] == true, "the popover opened");
    h.get_by_role_and_label(egui::accesskit::Role::Button, "Krea 2").click();
    h.run_steps(3);
    assert_eq!(*h.state(), ("krea2-turbo/image".to_string(), 1));
    assert!(h.query_by_label("Model: Krea 2").is_some(), "the pill names the model");
    // MOSS's files are missing: choosing it asks the download dialog first.
    h.get_by_label("Model: Krea 2").click();
    h.run_steps(3);
    h.get_by_role_and_label(egui::accesskit::Role::Button, "MOSS").click();
    h.run_steps(3);
    assert_eq!(h.state().0, "krea2-turbo/image", "not chosen yet");
    let r = take_request(&h.ctx).expect("asked");
    assert_eq!((r.picker.as_str(), r.template.as_str(), r.model.key), ("test", "moss-image/turbo", "moss"));
    assert!(take_request(&h.ctx).is_none(), "asked once");
    confirm(&h.ctx, "test", "moss-image/turbo");
    h.run_steps(2);
    assert_eq!(*h.state(), ("moss-image/turbo".to_string(), 2), "the dialog confirmed it");
    // Auto again.
    h.get_by_label("Model: MOSS").click();
    h.run_steps(3);
    h.get_by_role_and_label(egui::accesskit::Role::Button, "Auto").click();
    h.run_steps(3);
    assert_eq!(*h.state(), (String::new(), 3));
    assert_eq!(auto_description(Some("Krea 2 Turbo")), "Uses: Krea 2 Turbo");
    assert_eq!(auto_description(None), "The fastest permissive model the server has");
}
