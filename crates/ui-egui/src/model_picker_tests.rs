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
fn families_group_into_submenus_from_eight_items_and_only_with_the_preference() {
    let mut items: Vec<Item> = (1..=4).map(|i| item(&format!("k{i}"), &format!("Krea {i}"), "krea2")).collect();
    items.extend((1..=3).map(|i| item(&format!("q{i}"), &format!("Qwen ({i} steps)"), "qwen-image-edit-2511")));
    assert_eq!(layout(&items, true).len(), 7, "seven items: no groups");
    items.push(item("s", "SAM", "sam3.1"));
    let grouped = layout(&items, true);
    assert_eq!(grouped.len(), 3, "{grouped:?}");
    assert!(matches!(&grouped[0], Entry::Family(name, v) if name == "Krea 2" && v.len() == 4));
    assert!(matches!(&grouped[2], Entry::Item(7)), "a family of one stays a row");
    assert_eq!(layout(&items, false).len(), 8, "the preference off: every row");
}

#[test]
fn newest_first_compares_numbers_as_numbers() {
    let mut v = vec!["Krea 1", "Krea 10", "Krea 2", "Krea 2 Turbo"];
    v.sort_by(|a, b| newest_first(a, b));
    assert_eq!(v, ["Krea 10", "Krea 2 Turbo", "Krea 2", "Krea 1"]);
    let items = vec![item("a", "Model 1.5", "krea2"), item("b", "Model 2", "krea2"), item("c", "Model 1.10", "krea2")];
    let order: Vec<usize> = layout(&items, false).into_iter().filter_map(|e| if let Entry::Item(i) = e { Some(i) } else { None }).collect();
    assert_eq!(order, [1, 2, 0], "2, then 1.10, then 1.5");
}

#[test]
fn research_models_are_marked_and_greyed_until_allowed_and_badges_read_well() {
    let mut r = item("qwen-2.1/fill", "Generative Fill (Qwen-Image-2.1)", "qwen-image-2.1");
    r.research = true;
    r.allowed = false;
    assert!(!selectable(&r));
    assert_eq!(badges(&r), ["research"]);
    r.allowed = true;
    assert!(selectable(&r), "Preferences › Allow research-only models");
    let mut m = item("x", "X", "krea2");
    m.eta = Some(16.2);
    m.installed = Some(false);
    assert_eq!(badges(&m), ["~16 s", "not installed"]);
    m.installed = None;
    m.eta = Some(150.0);
    assert_eq!(badges(&m), ["~3 min"]);
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
}

#[test]
fn every_shipped_family_has_a_glyph() {
    let mut s = photocraft_engine::Session::new();
    let v = s.execute("generate.models", json!({"probe": false})).unwrap();
    for t in v["templates"].as_array().unwrap() {
        let family = t["family"].as_str().unwrap();
        let style = family_style(family);
        assert!(style.color.is_some() && style.name != "Other", "{family} has no glyph");
    }
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
    let mut research = item("qwen-2.1/image", "Generate Image (Qwen-Image-2.1)", "qwen-image-2.1");
    research.research = true;
    research.allowed = false;
    vec![item("krea2-turbo/image", "Generate Image (Krea 2 Turbo)", "krea2"), research]
}

#[test]
fn the_pill_chooses_a_model_skips_greyed_rows_and_goes_back_to_auto() {
    use egui_kittest::kittest::Queryable;
    let mut h = Harness::builder().with_size(vec2(420.0, 300.0)).build_ui_state(
        |ui, (current, changes): &mut (String, u32)| {
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            if pill(ui, "test", &pill_items(), current, Some("Krea 2 Turbo"), true, 200.0) {
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
    assert!(h.query_by_label("Auto").is_some() && is_open(&h.ctx, "test"), "the popover opened");
    assert_eq!(inspect(&h.ctx)[0]["open"], true);
    h.get_by_label("Krea 2 Turbo").click();
    h.run_steps(3);
    assert_eq!(*h.state(), ("krea2-turbo/image".to_string(), 1));
    assert!(h.query_by_label("Model: Krea 2 Turbo").is_some(), "the pill names the choice");
    // The research row is listed but greyed: clicking it changes nothing.
    h.get_by_label("Model: Krea 2 Turbo").click();
    h.run_steps(3);
    h.get_by_label("Qwen-Image-2.1").click();
    h.run_steps(3);
    assert_eq!(h.state().0, "krea2-turbo/image");
    // Auto again.
    if h.query_by_label("Auto").is_none() {
        h.get_by_label("Model: Krea 2 Turbo").click();
        h.run_steps(3);
    }
    h.get_by_label("Auto").click();
    h.run_steps(3);
    assert_eq!(*h.state(), (String::new(), 2));
    assert_eq!(auto_description(Some("Krea 2 Turbo")), "Uses: Krea 2 Turbo");
    assert_eq!(auto_description(None), "The fastest permissive model the server has");
}
