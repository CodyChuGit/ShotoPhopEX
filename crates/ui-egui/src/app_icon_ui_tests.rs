use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use serde_json::json;

use super::*;

fn app() -> PhotocraftApp {
    PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default())
}

/// A `w`×`h` opaque image of one colour, encoded as `format`.
fn image(w: u32, h: u32, rgb: [u8; 3], format: Format) -> Vec<u8> {
    let px: Vec<u8> = (0..w * h).flat_map(|_| [rgb[0], rgb[1], rgb[2], 255]).collect();
    let img = Image::from_u8(w, h, ChannelLayout::Rgba, px).unwrap();
    photocraft_codecs::encode(&img, format, &EncodeOptions::default()).unwrap()
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn custom_texture(ctx: &egui::Context) -> Option<TextureHandle> {
    ctx.data(|d| d.get_temp::<TextureHandle>(crate::brand::custom_id()))
}

#[test]
fn any_image_is_fitted_into_a_square_without_being_enlarged() {
    // Wide and small: kept at its size, centred on a transparent square.
    let px: Vec<u8> = (0..4 * 2).flat_map(|_| [255, 0, 0, 255]).collect();
    let (side, sq) = fit_square(4, 2, &px);
    assert_eq!(side, 4);
    let alpha = |x: usize, y: usize| sq[(y * 4 + x) * 4 + 3];
    assert_eq!((alpha(0, 0), alpha(0, 1), alpha(3, 2), alpha(0, 3)), (0, 255, 255, 0));
    assert_eq!(&sq[(4 + 1) * 4..(4 + 1) * 4 + 3], &[255, 0, 0]);
    // Large: shrunk to 256 px, the aspect kept.
    let png = to_icon_png(&image(600, 300, [0, 0, 255], Format::Png)).unwrap();
    let (w, h, px) = rgba(&png).unwrap();
    assert_eq!((w, h), (256, 256));
    assert_eq!(px[(128 * 256 + 128) * 4..(128 * 256 + 128) * 4 + 4], [0, 0, 255, 255]);
    assert_eq!(px[3], 0, "the band above the picture is transparent");
    // Not an image: a clear error, nothing added.
    assert!(to_icon_png(b"hello").unwrap_err().contains("not an image"));
}

#[test]
fn edit_app_icon_adds_uses_and_removes_icons_from_any_format() {
    let mut a = app();
    let ctx = egui::Context::default();
    let jpeg = image(40, 30, [10, 200, 10], Format::Jpeg);
    let r = crate::menus::invoke(&mut a, &ctx, COMMAND, json!({"add": {"name": "Leaf.jpg", "data": b64(&jpeg)}})).unwrap();
    assert_eq!(r["added"], "Leaf");
    assert_eq!(a.session.prefs().app_icon.active, "Leaf");
    assert!(custom_texture(&ctx).is_some(), "the title bar's mark shows it");
    // The same name again gets a number; the newest is in use.
    crate::menus::invoke(&mut a, &ctx, COMMAND, json!({"add": {"name": "Leaf.png", "data": b64(&image(8, 8, [1, 2, 3], Format::Png))}})).unwrap();
    assert_eq!(inspect(&a)["icons"], json!(["Leaf", "Leaf 2"]));
    assert_eq!(inspect(&a)["active"], "Leaf 2");
    // Back to the built-in icon in one step, and to a custom one again.
    crate::menus::invoke(&mut a, &ctx, COMMAND, json!({"use": ""})).unwrap();
    assert!(custom_texture(&ctx).is_none());
    crate::menus::invoke(&mut a, &ctx, COMMAND, json!({"use": "Leaf"})).unwrap();
    assert!(custom_texture(&ctx).is_some());
    // Removing the one in use brings the built-in one back.
    crate::menus::invoke(&mut a, &ctx, COMMAND, json!({"remove": "Leaf"})).unwrap();
    assert_eq!(inspect(&a)["active"], "");
    assert!(custom_texture(&ctx).is_none());
    // Mistakes are reported, not applied.
    assert!(crate::menus::invoke(&mut a, &ctx, COMMAND, json!({"use": "Nope"})).is_err());
    assert!(crate::menus::invoke(&mut a, &ctx, COMMAND, json!({"spin": 1})).is_err());
    assert!(crate::menus::invoke(&mut a, &ctx, COMMAND, json!({"add": {"data": "%%"}})).is_err());
    assert_eq!(inspect(&a)["icons"], json!(["Leaf 2"]));
    // The gallery is capped.
    for k in 0..MAX_ICONS {
        let r = crate::menus::invoke(&mut a, &ctx, COMMAND, json!({"add": {"name": format!("i{k}"), "data": b64(&image(4, 4, [9, 9, 9], Format::Png))}}));
        assert_eq!(r.is_ok(), k + 1 < MAX_ICONS, "icon {k}");
    }
    // No parameters: the window opens.
    crate::menus::invoke(&mut a, &ctx, COMMAND, json!({})).unwrap();
    assert!(a.ui.app_icon.open);
}

#[test]
fn icons_survive_a_preferences_round_trip() {
    let mut a = app();
    let ctx = egui::Context::default();
    crate::menus::invoke(&mut a, &ctx, COMMAND, json!({"add": {"name": "mine", "data": b64(&image(16, 16, [200, 0, 0], Format::Png))}})).unwrap();
    let text = serde_json::to_string(a.session.prefs()).unwrap();
    let back: photocraft_engine::prefs::Preferences = serde_json::from_str(&text).unwrap();
    assert_eq!(back.app_icon, a.session.prefs().app_icon);
    assert!(text.contains("\"appIcon\":{\"active\":\"mine\""));
}

fn fake_download(url: &str) -> Result<Vec<u8>, String> {
    match url {
        "https://example.test/fox.png" => {
            let px: Vec<u8> = (0..32 * 32).flat_map(|_| [255, 128, 0, 255]).collect();
            let img = Image::from_u8(32, 32, ChannelLayout::Rgba, px).map_err(|e| e.to_string())?;
            photocraft_codecs::encode(&img, Format::Png, &EncodeOptions::default()).map_err(|e| e.to_string())
        }
        _ => Err("404".into()),
    }
}

#[test]
fn a_downloaded_image_joins_the_gallery_and_one_click_switches_back() {
    let mut a = app();
    a.services.download = Some(fake_download);
    a.ui.app_icon.open = true;
    let mut h = Harness::builder().with_size(egui::vec2(800.0, 600.0)).build_ui_state(
        |ui, a: &mut PhotocraftApp| {
            // The first frame runs before the app's fonts are installed.
            if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                window(a, &ui.ctx().clone());
            }
        },
        a,
    );
    PhotocraftApp::setup_context(&h.ctx, Default::default());
    h.run_steps(2);
    h.get_by_role(egui::accesskit::Role::TextInput).click();
    h.run();
    h.get_by_role(egui::accesskit::Role::TextInput).type_text("https://example.test/fox.png");
    h.run();
    h.get_by_label("Download").click();
    for _ in 0..2000 {
        h.step();
        if h.state().ui.app_icon.pending.is_none() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(inspect(h.state())["icons"], json!(["fox"]));
    assert_eq!(inspect(h.state())["active"], "fox");
    assert!(h.state().ui.app_icon.url.is_empty());
    h.run();
    // One click on the built-in tile puts it back.
    h.get_by_label("Built-in").click();
    h.run();
    assert_eq!(inspect(h.state())["active"], "");
    // A failed download says why and adds nothing.
    h.state_mut().ui.app_icon.url = "https://example.test/missing.png".into();
    h.run();
    h.get_by_label("Download").click();
    for _ in 0..2000 {
        h.step();
        if h.state().ui.app_icon.pending.is_none() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(h.state().ui.app_icon.error.as_deref(), Some("404"));
    assert_eq!(inspect(h.state())["icons"], json!(["fox"]));
}
