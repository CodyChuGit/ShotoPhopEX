use egui::vec2;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::Document;
use photocraft_geom::Size;
use serde_json::json;

use super::*;
use crate::Services;

fn card(id: &str) -> Card {
    *CARDS.iter().find(|c| c.id == id).unwrap()
}

/// An app with no document whose open dialog "picks" `pick` (a 32×24 picture), or has no file
/// dialog at all.
fn app(pick: Option<&str>) -> PhotocraftApp {
    let answer = pick.map(|n| crate::FileDialogAnswer::Contents(n.to_string(), b"img".to_vec()));
    let services = Services {
        import: Some(Box::new(|name: &str, _bytes: &[u8], _opts: &photocraft_io::ImportOptions| {
            Ok((Document::new(name, Size::new(32, 24), ColorMode::Rgb, SampleType::U8), Vec::new()))
        })),
        file_dialog: pick.map(|_| crate::file_dialog::fake(vec![answer]).0),
        ..Services::default()
    };
    let mut s = photocraft_engine::Session::new();
    // No server: opening the Angles window must not reach a real ComfyUI from a test.
    s.edit_prefs(|p| p.integrations.models_path.clear());
    let mut app = PhotocraftApp::new(s, services);
    app.ui.panels.generate = false;
    app
}

#[test]
fn every_card_has_an_icon_and_a_place_to_go() {
    for c in CARDS {
        assert!(icons::exists(c.icon), "{}: {}", c.id, c.icon);
        match c.target {
            Target::Panel(m) => assert!(crate::generate_panel::MODES.contains(&m), "{m}"),
            Target::Command(id) => assert!(photocraft_engine::commands::find(id).is_some(), "{id}"),
            Target::Soon => assert_eq!(c.badge, Some("SOON")),
        }
        if c.badge == Some("SOON") {
            assert_eq!(c.target, Target::Soon, "{}", c.id);
        }
    }
    assert_eq!(CARDS.iter().filter(|c| c.create).count(), 1, "Create: Image");
}

#[test]
fn image_opens_the_panel_with_nothing_open_and_edit_cards_ask_for_a_file_first() {
    let ctx = egui::Context::default();
    // Create › Image: the panel in Image mode, a new document as the result, keyboard in the prompt.
    let mut a = app(None);
    activate(&mut a, &ctx, &card("image")).unwrap();
    assert!(a.ui.panels.generate);
    assert_eq!((a.ui.generate.mode.as_str(), a.ui.generate.into.as_str(), a.ui.generate.focus), ("image", "document", true));
    // Edit › Fill with nothing open and the dialog cancelled: it says so and waits for nothing.
    activate(&mut a, &ctx, &card("fill")).unwrap();
    assert_eq!((a.ui.hub.pending, a.ui.status.as_str()), (None, "Open a picture first"));
    assert_eq!(a.ui.generate.mode, "image", "unchanged");
    // A file picked: it opens (the dialog answers on a later frame) and Fill follows.
    let mut a = app(Some("boat.png"));
    activate(&mut a, &ctx, &card("fill")).unwrap();
    assert_eq!(a.ui.hub.pending, Some("fill"));
    tick(&mut a, &ctx);
    assert_eq!(a.ui.hub.pending, Some("fill"), "still waiting for the dialog");
    a.poll_file_dialog(&ctx, None);
    tick(&mut a, &ctx);
    assert_eq!(a.session.active().map(|d| d.doc.name.as_str()), Some("boat.png"));
    assert_eq!((a.ui.generate.mode.as_str(), a.ui.generative_bar.mode.as_str(), a.ui.hub.pending), ("fill", "fill", None));
    // With a picture open a command card opens its dialog.
    activate(&mut a, &ctx, &card("selectByText")).unwrap();
    assert_eq!(a.ui.dialogs.len(), 1, "Select by Text's dialog");
    // Angles opens its orbit window.
    activate(&mut a, &ctx, &card("angles")).unwrap();
    assert!(a.ui.angles.open);
}

#[test]
fn help_and_the_home_screen_show_the_cards() {
    let mut a = app(None);
    let ctx = egui::Context::default();
    crate::menus::invoke(&mut a, &ctx, "help.generativeFeatures", json!({})).unwrap();
    assert_eq!(inspect(&a, &ctx)["open"], true);
    let mut h = Harness::builder().with_size(vec2(1280.0, 900.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::canvas::document_area(app, ui);
            window(app, ui.ctx());
        },
        a,
    );
    PhotocraftApp::setup_context(&h.ctx, Default::default());
    h.run_steps(4);
    // The window and the Home screen each show every card.
    assert_eq!(h.get_all_by_label("Remove Background").count(), 2);
    let v = inspect(h.state(), &h.ctx);
    assert_eq!(v["cards"][4]["rects"].as_array().map(Vec::len), Some(2), "{v}");
    h.get_by_label("Close").click();
    h.run_steps(3);
    assert!(!h.state().ui.hub.open);
    assert_eq!(h.get_all_by_label("Split into Layers").count(), 1, "the Home screen's");
    // A Home card: Image opens the panel.
    h.get_by_label("Image").click();
    h.run_steps(3);
    assert_eq!((h.state().ui.panels.generate, h.state().ui.generate.mode.as_str()), (true, "image"));
}

#[test]
fn window_generate_puts_the_keyboard_in_the_prompt() {
    let mut a = app(None);
    let ctx = egui::Context::default();
    crate::menus::invoke(&mut a, &ctx, "window.toggle.generate", json!({})).unwrap();
    assert!(a.ui.panels.generate && a.ui.generate.focus);
}
