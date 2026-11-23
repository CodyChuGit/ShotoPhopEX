//! The brand mark at the left of the title bar. This build paints the "Sp" tile
//! (`assets/app-icon/sp/`, see its README) in place of the "Px" tile in `assets/app-icon/`. The PNG
//! carries the icon's rounded corners; the framework `Picture` stores the decoded image once and
//! draws it at the title bar's icon size. A custom icon chosen in Edit › Preferences › App Icon…
//! (`app_icon_ui`) replaces it.

use craft::ui::display::Picture;
use egui::Color32;
use photocraft_codecs::{ChannelLayout, SampleType};

/// 128 px: sharp at 20 pt on a 3× display.
const ICON_PNG: &[u8] = include_bytes!("../../../../assets/app-icon/sp/sp-128.png");

/// The key the picture store keeps the decoded icon under.
const KEY: u64 = 0x7068_6f74_6f63_7266;

/// The icon's pixels (one transparent pixel if it couldn't be decoded, which a test rules out).
fn decode() -> egui::ColorImage {
    let rgba = photocraft_codecs::decode(ICON_PNG).ok().map(|img| img.convert(ChannelLayout::Rgba, SampleType::U8));
    match rgba {
        Some(img) if img.data().len() == img.width() as usize * img.height() as usize * 4 => {
            egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.data())
        }
        _ => egui::ColorImage::filled([1, 1], Color32::TRANSPARENT),
    }
}

/// The custom icon in use (`app_icon_ui`) and the picture-store key it is kept under, in the context's memory, so
/// each window (and each test) has its own.
type Custom = std::sync::Arc<(u64, egui::ColorImage)>;

fn custom_id() -> egui::Id {
    egui::Id::unique("shotophopex-custom-app-icon")
}

/// Shows `image` as the mark in place of the built-in icon; `None` restores the built-in one.
pub(crate) fn set_custom(ctx: &egui::Context, image: Option<egui::ColorImage>) {
    let keyed = image.map(|img| {
        use std::hash::{Hash as _, Hasher as _};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        img.size.hash(&mut h);
        img.as_raw().hash(&mut h);
        Custom::new((h.finish() | 1, img))
    });
    ctx.data_mut(|d| match keyed {
        Some(custom) => {
            d.insert_temp(custom_id(), custom);
        }
        None => {
            d.remove::<Custom>(custom_id());
        }
    });
}

/// The size of the custom icon the mark shows, or `None` for the built-in one.
#[cfg(test)]
pub(crate) fn custom_size(ctx: &egui::Context) -> Option<[usize; 2]> {
    ctx.data(|d| d.get_temp::<Custom>(custom_id())).map(|c| c.1.size)
}

/// Shows the brand mark as a decorative picture: the empty label keeps a screen reader from announcing it.
pub(crate) fn show_mark(ui: &mut egui::Ui) -> egui::Response {
    use craft::ui::contract::Show as _;
    match ui.ctx().data(|d| d.get_temp::<Custom>(custom_id())) {
        Some(custom) => {
            let key = custom.0;
            let image = move || custom.1.clone();
            Picture::new("", key, &image).show(ui)
        }
        None => Picture::new("", KEY, &decode).show(ui),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_icon_decodes_to_a_rounded_128_px_tile() {
        let img = decode();
        assert_eq!(img.size, [128, 128]);
        // Rounded corners: the corner pixel is transparent, the centre opaque.
        assert_eq!(img.pixels[0].a(), 0);
        assert_eq!(img.pixels[64 * 128 + 64].a(), 255);
    }
}
