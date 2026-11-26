//! Surfaces, section labels and hairlines.

use super::{Color32, Rect, Sense, Stroke, Tokens, Ui, theme, vec2};

/// A filled control surface; `raised` gives the bevelled themes their light top edge.
pub fn surface(ui: &Ui, rect: Rect, fill: Color32, raised: bool) {
    let t = Tokens::get(ui.ctx());
    let p = ui.painter();
    p.rect_filled(rect, t.radius_sm, fill);
    if t.bevel {
        let (hi, lo) = if raised { (Color32::WHITE, Color32::from_gray(64)) } else { (Color32::from_gray(64), Color32::WHITE) };
        p.line_segment([rect.left_bottom(), rect.left_top()], Stroke::new(1.0, hi));
        p.line_segment([rect.left_top(), rect.right_top()], Stroke::new(1.0, hi));
        p.line_segment([rect.right_top(), rect.right_bottom()], Stroke::new(1.0, lo));
        p.line_segment([rect.right_bottom(), rect.left_bottom()], Stroke::new(1.0, lo));
    }
}

/// Small caps section label.
pub fn section_label(ui: &mut Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    let text = tl!(text);
    let font = theme::medium(11.5);
    let galley = ui.painter().layout(text.to_owned(), font.clone(), t.text_faint, ui.available_width());
    // A glyph can start before its logical advance (the Urdu Stroke heading does). Reserve
    // that bearing in the layout instead of clipping the glyph or relaxing dialog bounds.
    let left = (galley.rect.left() - galley.mesh_bounds.left()).max(0.0).ceil() as i8;
    egui::Frame::NONE.inner_margin(egui::Margin { left, ..Default::default() }).show(ui, |ui| {
        ui.label(egui::RichText::new(text).font(font).color(t.text_faint));
    });
}

/// Hairline separator.
pub fn hairline(ui: &mut Ui) {
    let t = Tokens::get(ui.ctx());
    let (r, _) = ui.allocate_exact_size(vec2(ui.available_width(), 1.0), Sense::hover());
    ui.painter().line_segment([r.left_center(), r.right_center()], Stroke::new(1.0, t.separator));
}
