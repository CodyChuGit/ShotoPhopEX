//! ShotoPhopEX: immediate-mode icon helpers for the generative panels, bars and dialogs, which are
//! still drawn with plain egui. Each draws a catalog icon through the framework's `Icon`, so the
//! glyph and its theme match the rest of the app.

use craft::behavior::IconId;
use craft::theme::ControlSize;
use craft::ui::contract::{Props as _, Show as _};
use craft::ui::display::Icon;
use craft::ui::primitives::{IconSpec, has_icon};
use craft::ui::style::StyleColor;
use egui::{Color32, Rect, Response, Sense, UiBuilder, Vec2};

use crate::theme::Tokens;

/// The registered icon `name` names: the 0.23 names our panels use, spelled as the catalog spells
/// them (underscores, and its own words for a few), or the app set's name as given.
fn catalog_name(name: &str) -> IconId {
    let aliased = match name {
        "x" => "close",
        "ellipsis" => "more",
        "triangle-alert" => "warning",
        other => other,
    };
    let id = IconId::from(std::borrow::Cow::Owned(aliased.replace('-', "_")));
    if has_icon(&id) { id } else { IconId::from(std::borrow::Cow::Owned(name.to_owned())) }
}

/// Whether icon `name` is registered, so [`paint`] draws a glyph for it.
pub fn exists(name: &str) -> bool {
    has_icon(&catalog_name(name))
}

/// Paints icon `name` centred in `rect`, about `size` points across, in `tint`.
pub fn paint(ui: &egui::Ui, rect: Rect, name: &str, size: f32, tint: Color32) {
    let side = if size < 16.0 { ControlSize::Sm } else { ControlSize::Md };
    let spec = IconSpec::svg(catalog_name(name)).color(StyleColor(tint));
    let id = ui.scope_id().with(("egui-paint-icon", name, rect.min.x.to_bits(), rect.min.y.to_bits()));
    let builder = UiBuilder::new().layer_id(ui.layer_id()).max_rect(rect).layout(egui::Layout::centered_and_justified(egui::Direction::TopDown));
    let mut child = egui::Ui::new(ui.ctx().clone(), id, builder);
    child.set_clip_rect(ui.clip_rect());
    let _ = Icon::new(spec).size(side).show(&mut child);
}

/// Selected and hover fill shared by square icon buttons. Returns the icon tint.
pub fn button_chrome(ui: &egui::Ui, rect: Rect, selected: bool, hovered: bool) -> Color32 {
    let t = Tokens::get(ui.ctx());
    if selected {
        ui.painter().rect_filled(rect, t.radius_sm, t.accent_soft);
        ui.painter().rect_stroke(rect, t.radius_sm, egui::Stroke::new(1.0, t.accent_border), egui::StrokeKind::Inside);
    } else if hovered {
        ui.painter().rect_filled(rect, t.radius_sm, t.hover);
    }
    if selected {
        t.accent_text
    } else if hovered {
        t.text
    } else {
        t.icon
    }
}

/// Square icon button: transparent until hovered; `selected` gets the accent treatment. `tooltip`
/// (translated) names it for screen readers and shows on hover.
pub fn button(ui: &mut egui::Ui, name: &str, box_size: f32, selected: bool, tooltip: &str) -> Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(box_size), Sense::click());
    let t = Tokens::get(ui.ctx());
    let tint = button_chrome(ui, rect, selected, resp.hovered());
    if resp.has_focus() {
        ui.painter().rect_stroke(rect, t.radius_sm, egui::Stroke::new(1.0, t.accent), egui::StrokeKind::Inside);
    }
    paint(ui, rect, name, (box_size * 0.52).round(), tint);
    let label = if tooltip.is_empty() { name.to_owned() } else { tl!(tooltip).to_owned() };
    resp.widget_info(|| egui::WidgetInfo::selected(egui::Role::Button, true, selected, label.clone()));
    if tooltip.is_empty() { resp } else { resp.on_hover_text(label) }
}
