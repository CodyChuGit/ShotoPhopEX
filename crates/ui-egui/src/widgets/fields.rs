//! Dropdowns, numeric fields, elided labels and the transparency checkerboard.

use super::{Color32, Pos2, Rect, Response, Sense, Tokens, Ui, Vec2};

/// A dropdown of `options` (value and English label, translated here) about `width` points wide, the framework
/// `Select`. `name` names it for screen readers; `id` keeps it apart from another of the same name. Returns true when
/// the user picked another value.
pub fn dropdown<T: craft::behavior::Value + PartialEq + Clone + 'static>(
    ui: &mut Ui,
    id: &str,
    name: &str,
    current: &mut T,
    options: &[(T, &str)],
    width: f32,
) -> bool {
    use craft::ui::contract::{Bind, Entry, Props as _, Show as _, Sizing};
    let entries: Vec<Entry<T>> = options.iter().map(|(value, label)| Entry::new(value.clone(), tl!(label).to_owned())).collect();
    let picked = ui
        .push_id(id, |ui| {
            ui.set_max_width(width);
            craft::ui::pickers::Select::new(tl!(name).to_owned(), Bind::given(current.clone()), &entries).sizing(Sizing::Content).show(ui).value().cloned()
        })
        .inner;
    match picked {
        Some(value) if value != *current => {
            *current = value;
            true
        }
        _ => false,
    }
}

/// A numeric field for `value` in `range`, with `suffix` (a unit) after the number and `decimals`
/// places shown. Dragging scrubs; clicking types.
fn field(ui: &mut Ui, name: Option<&str>, value: &mut f32, range: std::ops::RangeInclusive<f32>, suffix: &str, width: f32, decimals: usize) -> Response {
    let suffix = if suffix.is_empty() { String::new() } else { format!(" {suffix}") };
    let step = if decimals == 0 { 1.0 } else { 10f64.powi(-(decimals as i32)) };
    let span = (*range.end() - *range.start()).abs().max(1.0) as f64;
    let speed = (span / 300.0).max(step);
    let drag = egui::DragValue::new(value).range(range).suffix(suffix).fixed_decimals(decimals).speed(speed);
    let resp = ui.add_sized(Vec2::new(width, super::buttons::button_height(ui.ctx())), drag);
    if let Some(name) = name {
        let name = tl!(name).to_owned();
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::SpinButton, true, name.clone()));
    }
    resp
}

/// A whole-number field (see `field`).
pub fn value_field(ui: &mut Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, suffix: &str, width: f32) -> Response {
    field(ui, None, value, range, suffix, width, 0)
}

/// A [`value_field`] that `name` names for screen readers.
pub fn value_field_named(ui: &mut Ui, name: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>, suffix: &str, width: f32) -> Response {
    field(ui, Some(name), value, range, suffix, width, 0)
}

/// A [`value_field`] showing `decimals` places.
pub fn value_field_prec(ui: &mut Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, suffix: &str, width: f32, decimals: u32) -> Response {
    field(ui, None, value, range, suffix, width, decimals as usize)
}

/// A single-line label no wider than `max_w`, cut with an ellipsis; the full text shows on hover.
pub fn elided_label(ui: &mut Ui, text: &str, max_w: f32) -> Response {
    let t = Tokens::get(ui.ctx());
    let font = egui::TextStyle::Body.resolve(ui.style());
    let mut job = egui::text::LayoutJob::simple_singleline(text.to_owned(), font, t.text_dim);
    job.wrap = egui::text::TextWrapping::truncate_at_width(max_w.max(1.0));
    let galley = ui.fonts_mut(|f| f.layout_job(job));
    let elided = galley.elided;
    let (rect, resp) = ui.allocate_exact_size(galley.size(), Sense::hover());
    ui.painter().galley(rect.min, galley, t.text_dim);
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Label, ui.is_enabled(), text));
    if elided { resp.on_hover_text(text) } else { resp }
}

/// Paints a transparency checkerboard of `cell`-point squares in `rect`.
pub fn checker(painter: &egui::Painter, rect: Rect, cell: f32) {
    painter.rect_filled(rect, 0.0, Color32::from_gray(250));
    let cols = (rect.width() / cell).ceil() as i32;
    let rows = (rect.height() / cell).ceil() as i32;
    for j in 0..rows {
        for i in 0..cols {
            if (i + j) % 2 == 1 {
                let r = Rect::from_min_size(Pos2::new(rect.min.x + i as f32 * cell, rect.min.y + j as f32 * cell), Vec2::splat(cell)).intersect(rect);
                painter.rect_filled(r, 0.0, Color32::from_gray(214));
            }
        }
    }
}

/// A thin progress bar in `rect`: filled to `frac`, or a sweeping segment while the amount is unknown.
pub fn progress_bar(ui: &egui::Ui, rect: Rect, frac: Option<f32>, t: &Tokens) {
    let p = ui.painter();
    let r = rect.height() / 2.0;
    p.rect_filled(rect, r, t.field);
    p.rect_stroke(rect, r, egui::Stroke::new(1.0, t.field_border), egui::StrokeKind::Inside);
    let fill = match frac {
        Some(f) => Rect::from_min_size(rect.min, Vec2::new((rect.width() * f.clamp(0.0, 1.0)).max(rect.height()), rect.height())),
        None => {
            let time = ui.input(|i| i.time) as f32;
            let w = rect.width() * 0.3;
            let x = rect.left() + ((time * 0.8).fract() * (rect.width() + w)) - w;
            Rect::from_min_max(Pos2::new(x.max(rect.left()), rect.top()), Pos2::new((x + w).min(rect.right()), rect.bottom()))
        }
    };
    if fill.width() > 0.0 {
        p.rect_filled(fill, r, t.accent);
    }
}
