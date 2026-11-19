//! The model picker (ShotoPhopEX): a pill naming the chosen model; its popover is a browser
//! with a preview. On the left, Auto and one short row per model (a two-letter glyph, the name,
//! and the seconds a run takes at the current Speed and Quality, or a download mark when its
//! files are missing); on the right, the hovered or chosen model's preview: a sample, what it
//! is good at, its tags, Speed, Detail and Memory bars, and Use (or Download…).
//!
//! The list shows no warnings. Choosing a model whose files are missing asks first: the
//! request goes to [`take_request`] and the download dialog (`model_download`) shows the size,
//! the licences and whether it may not fit, then confirms the choice with [`confirm`] or
//! returns to the picker.
//!
//! One component for the Generate panel, the task bar and the generated dialogs (`template`).
//! The data is plain [`Item`]s built by each caller from `generate.models`; rows and versions
//! come from [`crate::model_catalog`], so the picker itself decides nothing about the server.

use egui::{Align2, Color32, FontId, Rect, Sense, Stroke, Ui, pos2, vec2};

use crate::model_catalog::{self as catalog, Model, Row, Speed};
use crate::theme::Tokens;

/// One template the picker can offer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Item {
    pub id: String,
    /// The template's full name ("Generative Fill (Qwen-Image-Edit-2511, Lightning 8 steps)").
    pub name: String,
    pub family: String,
    pub description: String,
    /// A research-only licence (said in the download dialog, never in the list).
    pub research: bool,
    /// Selectable (research models only with Preferences › Allow research-only models).
    pub allowed: bool,
    /// The server has every file (None = unknown).
    pub installed: Option<bool>,
    /// Seconds for the current request, when known.
    pub eta: Option<f32>,
    /// Needs more memory than the memory profile has: said in the download dialog.
    pub may_not_fit: bool,
    /// Graphics memory the template needs, in GB, when known (the preview's Memory bar).
    pub need_gb: Option<f32>,
}

/// What the picker needs besides the items.
#[derive(Clone, Copy, Debug)]
pub struct Opts<'a> {
    /// The page the items come from (`image`, `fill`, `edit`, `expand`, or a tool's).
    pub page: &'a str,
    /// Generate › Speed: which version of a model a row runs (its seconds).
    pub speed: Speed,
    /// Reference pictures are attached.
    pub refs: bool,
    /// The short name of the model a run without a choice uses, for the Auto row.
    pub auto: Option<&'a str>,
    /// The memory profile's budget in GB (the Memory bar's scale), when known.
    pub budget_gb: Option<f32>,
}

impl Default for Opts<'_> {
    fn default() -> Self {
        Opts { page: "image", speed: Speed::Balanced, refs: false, auto: None, budget_gb: None }
    }
}

/// The part of a template name a row needs: what is inside its last parentheses
/// ("Qwen-Image-Edit-2511, Lightning 8 steps"), else the name.
pub fn short_name(name: &str) -> &str {
    match (name.rfind('('), name.rfind(')')) {
        (Some(a), Some(b)) if b > a + 1 => name[a + 1..b].trim(),
        _ => name,
    }
}

/// The names templates go by: the short name, or the full name where two items' short names
/// collide ("Generate Image (Krea 2 Turbo)" and "Generate Image with a Style Reference (Krea 2
/// Turbo)" both shorten to "Krea 2 Turbo").
pub fn display_names(items: &[Item]) -> Vec<String> {
    items
        .iter()
        .map(|it| {
            let s = short_name(&it.name);
            if items.iter().filter(|o| short_name(&o.name) == s).count() > 1 { it.name.clone() } else { s.to_string() }
        })
        .collect()
}

/// What the pill says for `current` (a template id; empty = Auto): the model's name on the
/// Generate pages, the row's label on a tool's, the template's short name when it is a recipe
/// no row shows (a script chose it).
pub fn label_for(items: &[Item], current: &str, page: &str) -> String {
    if current.is_empty() {
        return tl!("Auto").to_string();
    }
    let Some(it) = items.iter().find(|i| i.id == current) else { return current.to_string() };
    match catalog::rows(items, page, true).into_iter().find(|r| r.templates.iter().any(|&i| items[i].id == current)) {
        Some(r) => r.label,
        None => short_name(&it.name).to_string(),
    }
}

/// A row's glyph: a rounded square in the model's colour with its two letters.
pub fn glyph(ui: &Ui, rect: Rect, m: &Model) {
    let c = Color32::from_rgb((m.color >> 16) as u8, (m.color >> 8) as u8, m.color as u8);
    ui.painter().rect_filled(rect, 3.0, c);
    ui.painter().text(rect.center(), Align2::CENTER_CENTER, m.mark, FontId::proportional(rect.height() * 0.55), Color32::WHITE);
}

/// A model's colour.
fn color(m: &Model) -> Color32 {
    Color32::from_rgb((m.color >> 16) as u8, (m.color >> 8) as u8, m.color as u8)
}

/// A choice the picker asks the download dialog to confirm: the model's files are missing.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    /// The picker (`id_salt`) the choice goes back to.
    pub picker: String,
    /// The template the choice names.
    pub template: String,
    pub model: Model,
    /// Its label in the list.
    pub label: String,
    pub research: bool,
    pub may_not_fit: bool,
}

fn request_id() -> egui::Id {
    egui::Id::unique("model-picker-request")
}

fn confirmed_id(id_salt: &str) -> egui::Id {
    egui::Id::unique(("model-picker-confirmed", id_salt))
}

/// The pending download request, taken (the download dialog asks once per frame).
pub fn take_request(ctx: &egui::Context) -> Option<Request> {
    ctx.data_mut(|d| {
        let r = d.get_temp::<Request>(request_id());
        d.remove::<Request>(request_id());
        r
    })
}

/// The download dialog confirmed: picker `id_salt` takes `template` on its next frame.
pub fn confirm(ctx: &egui::Context, id_salt: &str, template: &str) {
    ctx.data_mut(|d| d.insert_temp(confirmed_id(id_salt), template.to_string()));
}

/// The pill and its popover. `current` is a template id, or empty for Auto. Under 60 pt wide
/// the pill is the glyph alone (one-row forms). Returns true when the choice changed.
pub fn pill(ui: &mut Ui, id_salt: &str, items: &[Item], current: &mut String, opts: &Opts<'_>, width: f32) -> bool {
    let t = Tokens::get(ui.ctx());
    let mut changed = false;
    // A choice the download dialog confirmed.
    if let Some(c) = ui.ctx().data_mut(|d| d.remove_temp::<String>(confirmed_id(id_salt)))
        && *current != c
    {
        *current = c;
        changed = true;
    }
    let rows = catalog::rows(items, opts.page, opts.refs);
    let chosen_row = rows.iter().position(|r| r.templates.iter().any(|&i| items[i].id == *current));
    let label = label_for(items, current, opts.page);
    let h = 24.0;
    let (rect, resp) = ui.allocate_exact_size(vec2(width, h), Sense::click());
    let frame = ui.ctx().cumulative_frame_nr();
    ui.ctx().data_mut(|d| d.get_temp_mut_or_default::<Drawn>(egui::Id::unique("model-pickers")).insert(id_salt.to_string(), (rect, frame, label.clone())));
    let p = ui.painter();
    p.rect_filled(rect, h / 2.0, if resp.hovered() { t.hover } else { t.field });
    p.rect_stroke(rect, h / 2.0, Stroke::new(1.0, t.field_border), egui::StrokeKind::Inside);
    let glyph_only = width < 60.0;
    let g = if glyph_only {
        Rect::from_center_size(rect.center(), vec2(16.0, 16.0))
    } else {
        Rect::from_min_size(pos2(rect.left() + 5.0, rect.center().y - 8.0), vec2(16.0, 16.0))
    };
    let chosen_model = items.iter().find(|i| i.id == *current).map(|i| catalog::model(catalog::model_key(&i.id, &i.family)));
    match &chosen_model {
        Some(m) => glyph(ui, g, m),
        None => crate::icons::paint(ui, g, "sparkles", 12.0, t.accent),
    }
    if !glyph_only {
        let text_w = (rect.width() - 40.0).max(10.0);
        let galley = ui.painter().layout(label.clone(), FontId::proportional(12.0), t.text, text_w);
        let row = galley.rows.first().map(|r| r.text()).unwrap_or_default().to_string();
        let shown = if galley.rows.len() > 1 { format!("{}…", row.trim_end()) } else { label.clone() };
        ui.painter().text(pos2(g.right() + 6.0, rect.center().y), Align2::LEFT_CENTER, shown, FontId::proportional(12.0), t.text);
        crate::icons::paint(ui, Rect::from_min_size(pos2(rect.right() - 18.0, rect.center().y - 6.0), vec2(12.0, 12.0)), "chevron-down", 10.0, t.text_dim);
    }
    let spoken = format!("{}: {label}", tl!("Model"));
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::ComboBox, ui.is_enabled(), &spoken));
    let tip = match items.iter().find(|i| i.id == *current) {
        Some(i) => i.name.clone(),
        None => auto_description(opts.auto),
    };
    let resp = resp.on_hover_text(tip);
    let hover_id = egui::Id::unique(("model-picker-hover", id_salt));
    let find_id = egui::Id::unique(("model-picker-find", id_salt));
    let shown = egui::Popup::menu(&resp).id(popup_id(id_salt)).gap(4.0).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
        let hovered: Option<Option<usize>> = ui.ctx().data(|d| d.get_temp(hover_id));
        let mut find: String = ui.ctx().data(|d| d.get_temp(find_id)).unwrap_or_default();
        let mut now_hovered: Option<Option<usize>> = None;
        let mut pick: Option<Pick> = None;
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing = vec2(0.0, 0.0);
            // The list.
            ui.vertical(|ui| {
                ui.set_width(LIST_W);
                if rows.len() > FIND_FROM {
                    let field = egui::TextEdit::singleline(&mut find).hint_text(tl!("Find a model")).desired_width(LIST_W - 8.0).id_salt((id_salt, "find"));
                    ui.add_space(2.0);
                    ui.add(field);
                    ui.add_space(4.0);
                }
                egui::ScrollArea::vertical().max_height(ROW_H * 10.0).auto_shrink([false, true]).id_salt((id_salt, "rows")).show(ui, |ui| {
                    let r = auto_row(ui, opts.auto, current.is_empty(), &t);
                    if r.hovered() {
                        now_hovered = Some(None);
                    }
                    if r.clicked() {
                        pick = Some(Pick::Auto);
                    }
                    let q = find.trim().to_lowercase();
                    for (n, row) in rows.iter().enumerate() {
                        if !q.is_empty() && !matches_find(row, &q) {
                            continue;
                        }
                        let it = catalog::resolve(items, row, opts.page, opts.speed, opts.refs);
                        let r = model_row(ui, row, it, chosen_row == Some(n), &t);
                        if r.hovered() {
                            now_hovered = Some(Some(n));
                        }
                        if r.clicked() {
                            pick = Some(Pick::Row(n));
                        }
                    }
                });
            });
            // The preview: the hovered row, else the chosen one.
            let (sep, _) = ui.allocate_exact_size(vec2(9.0, ui.min_rect().height().max(PREVIEW_H)), Sense::hover());
            ui.painter().vline(sep.center().x, sep.y_range(), Stroke::new(1.0, t.separator));
            ui.vertical(|ui| {
                ui.set_width(PREVIEW_W);
                // A steady height while the hover moves between rows.
                ui.set_min_height(PREVIEW_H);
                // None = Auto.
                let shown: Option<usize> = now_hovered.or(hovered).unwrap_or(chosen_row).filter(|n| *n < rows.len());
                if let Some(n) = shown {
                    if preview(ui, items, &rows, n, opts, chosen_row == Some(n), &t) {
                        pick = Some(Pick::Row(n));
                    }
                } else if preview_auto(ui, opts.auto, current.is_empty(), &t) {
                    pick = Some(Pick::Auto);
                }
            });
        });
        if let Some(h) = now_hovered {
            ui.ctx().data_mut(|d| d.insert_temp(hover_id, h));
        }
        ui.ctx().data_mut(|d| d.insert_temp(find_id, find));
        match pick {
            Some(Pick::Auto) => {
                if !current.is_empty() {
                    current.clear();
                    changed = true;
                }
                ui.close();
            }
            Some(Pick::Row(n)) => {
                if let Some(it) = rows.get(n).and_then(|row| catalog::resolve(items, row, opts.page, opts.speed, opts.refs)) {
                    if it.installed == Some(false) || !it.allowed {
                        let request = Request {
                            picker: id_salt.to_string(),
                            template: it.id.clone(),
                            model: rows[n].model,
                            label: rows[n].label.clone(),
                            research: it.research || !it.allowed,
                            may_not_fit: it.may_not_fit,
                        };
                        ui.ctx().data_mut(|d| d.insert_temp(request_id(), request));
                    } else if *current != it.id {
                        *current = it.id.clone();
                        changed = true;
                    }
                }
                ui.close();
            }
            None => {}
        }
    });
    // Over whatever holds the pill (the task bar is a foreground area too).
    if let Some(r) = shown {
        ui.ctx().move_to_top(r.response.layer_id);
    }
    if !is_open(ui.ctx(), id_salt) {
        ui.ctx().data_mut(|d| {
            d.remove_temp::<Option<usize>>(hover_id);
        });
    }
    changed
}

enum Pick {
    Auto,
    Row(usize),
}

const LIST_W: f32 = 176.0;
const PREVIEW_W: f32 = 192.0;
const PREVIEW_H: f32 = 270.0;
const ROW_H: f32 = 22.0;
/// The Find field shows from this many rows on.
const FIND_FROM: usize = 6;

fn matches_find(row: &Row, q: &str) -> bool {
    row.label.to_lowercase().contains(q)
        || row.model.name.to_lowercase().contains(q)
        || row.model.good.to_lowercase().contains(q)
        || row.model.tags.iter().any(|t| t.to_lowercase().contains(q))
}

/// The pickers drawn so far: id salt → (rect, frame, label).
type Drawn = std::collections::BTreeMap<String, (Rect, u64, String)>;

/// `ui.inspect.modelPickers`: the pickers drawn in the last two frames, with their rect (points),
/// label and whether their popover is open (scripts click them with `ui.click`).
pub fn inspect(ctx: &egui::Context) -> serde_json::Value {
    let now = ctx.cumulative_frame_nr();
    let drawn: Drawn = ctx.data(|d| d.get_temp(egui::Id::unique("model-pickers"))).unwrap_or_default();
    drawn
        .into_iter()
        .filter(|(_, (_, f, _))| now.saturating_sub(*f) <= 2)
        .map(|(id, (r, _, label))| serde_json::json!({"id": id, "rect": [r.left(), r.top(), r.width(), r.height()], "label": label, "open": is_open(ctx, &id)}))
        .collect()
}

fn popup_id(id_salt: &str) -> egui::Id {
    egui::Id::unique(("model-picker", id_salt))
}

/// Is the popover of the picker drawn with `id_salt` open? (Callers ask the server which
/// templates it has then.)
pub fn is_open(ctx: &egui::Context, id_salt: &str) -> bool {
    egui::Popup::is_id_open(ctx, popup_id(id_salt))
}

/// What the Auto row says: the model it resolves to, when known.
pub fn auto_description(auto: Option<&str>) -> String {
    match auto.filter(|a| !a.is_empty()) {
        Some(a) => crate::i18n::fmt(tl!("Uses: {name}"), &[("name", a)]),
        None => tl!("The fastest permissive model the server has").to_string(),
    }
}

/// What a row shows on its right: the seconds at the current Speed and Quality, or nothing
/// when unknown; `None` when the files are missing (the download mark instead).
pub fn row_seconds(it: Option<&Item>) -> Option<String> {
    match it {
        Some(i) if i.installed == Some(false) => None,
        Some(i) => Some(i.eta.map(crate::generate_panel::eta_label).unwrap_or_default()),
        None => Some(String::new()),
    }
}

fn row_frame(ui: &mut Ui, checked: bool, name: &str, t: &Tokens) -> (Rect, egui::Response) {
    let w = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(vec2(w, ROW_H), Sense::click());
    if checked {
        ui.painter().rect_filled(rect, 4.0, t.accent_soft);
    } else if resp.hovered() {
        ui.painter().rect_filled(rect, 4.0, t.hover);
    }
    resp.widget_info(|| egui::WidgetInfo::selected(egui::Role::Button, true, checked, name));
    (rect, resp)
}

fn auto_row(ui: &mut Ui, auto: Option<&str>, checked: bool, t: &Tokens) -> egui::Response {
    let name = tl!("Auto");
    let (rect, resp) = row_frame(ui, checked, name, t);
    let g = Rect::from_min_size(pos2(rect.left() + 4.0, rect.center().y - 7.0), vec2(14.0, 14.0));
    crate::icons::paint(ui, g, "sparkles", 12.0, t.accent);
    ui.painter().text(pos2(g.right() + 6.0, rect.center().y), Align2::LEFT_CENTER, name, FontId::proportional(12.0), t.text);
    if let Some(a) = auto.filter(|a| !a.is_empty()) {
        let galley = ui.painter().layout(a.to_string(), FontId::proportional(10.5), t.text_faint, 80.0);
        let row = galley.rows.first().map(|r| r.text()).unwrap_or_default().to_string();
        let shown = if galley.rows.len() > 1 { format!("{}…", row.trim_end()) } else { a.to_string() };
        ui.painter().text(pos2(rect.right() - 6.0, rect.center().y), Align2::RIGHT_CENTER, shown, FontId::proportional(10.5), t.text_faint);
    }
    resp.on_hover_text(auto_description(auto))
}

fn model_row(ui: &mut Ui, row: &Row, it: Option<&Item>, checked: bool, t: &Tokens) -> egui::Response {
    let (rect, resp) = row_frame(ui, checked, &row.label, t);
    let g = Rect::from_min_size(pos2(rect.left() + 4.0, rect.center().y - 7.0), vec2(14.0, 14.0));
    glyph(ui, g, &row.model);
    let right = match row_seconds(it) {
        Some(s) => {
            let galley = ui.painter().layout_no_wrap(s, FontId::monospace(10.5), t.text_dim);
            let w = galley.size().x;
            ui.painter().galley(pos2(rect.right() - 6.0 - w, rect.center().y - galley.size().y / 2.0), galley, t.text_dim);
            w
        }
        None => {
            let r = Rect::from_center_size(pos2(rect.right() - 12.0, rect.center().y), vec2(12.0, 12.0));
            crate::icons::paint(ui, r, "download", 11.0, t.text_dim);
            14.0
        }
    };
    let name_w = (rect.width() - 30.0 - right - 10.0).max(10.0);
    let galley = ui.painter().layout(row.label.clone(), FontId::proportional(12.0), t.text, name_w);
    let first = galley.rows.first().map(|r| r.text()).unwrap_or_default().to_string();
    let shown = if galley.rows.len() > 1 { format!("{}…", first.trim_end()) } else { row.label.clone() };
    ui.painter().text(pos2(g.right() + 6.0, rect.center().y), Align2::LEFT_CENTER, shown, FontId::proportional(12.0), t.text);
    match it {
        Some(i) if i.installed == Some(false) => resp.on_hover_text(tl!("Not downloaded yet")),
        _ => resp,
    }
}

/// A sample tile: the model's colour, its letters, until real sample pictures ship.
fn sample(ui: &mut Ui, m: Option<&Model>, t: &Tokens) {
    let w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(w, w * 10.0 / 16.0), Sense::hover());
    let base = m.map_or(t.accent, color);
    let p = ui.painter();
    p.rect_filled(rect, t.radius_sm, base.gamma_multiply(0.85));
    // A soft diagonal light.
    let mut mesh = egui::Mesh::default();
    let clear = Color32::TRANSPARENT;
    let light = Color32::from_white_alpha(46);
    mesh.colored_vertex(rect.left_top(), light);
    mesh.colored_vertex(rect.right_top(), clear);
    mesh.colored_vertex(rect.right_bottom(), Color32::from_black_alpha(60));
    mesh.colored_vertex(rect.left_bottom(), clear);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(0, 2, 3);
    p.add(mesh);
    match m {
        Some(m) => {
            p.text(rect.center(), Align2::CENTER_CENTER, m.mark, FontId::proportional(rect.height() * 0.36), Color32::from_white_alpha(230));
        }
        None => crate::icons::paint(ui, Rect::from_center_size(rect.center(), vec2(34.0, 34.0)), "sparkles", 30.0, Color32::WHITE),
    }
}

/// A 1 to 5 bar with its name and a value on the right.
fn bar(ui: &mut Ui, name: &str, filled: u8, value: &str, t: &Tokens) {
    let w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(w, 16.0), Sense::hover());
    let p = ui.painter();
    p.text(pos2(rect.left(), rect.center().y), Align2::LEFT_CENTER, name, FontId::proportional(10.5), t.text_dim);
    p.text(pos2(rect.right(), rect.center().y), Align2::RIGHT_CENTER, value, FontId::proportional(10.5), t.text_dim);
    let left = rect.left() + 48.0;
    let right = rect.right() - 44.0;
    let seg = ((right - left) - 4.0 * 3.0) / 5.0;
    for k in 0..5u8 {
        let x = left + f32::from(k) * (seg + 3.0);
        let r = Rect::from_min_size(pos2(x, rect.center().y - 2.5), vec2(seg, 5.0));
        p.rect_filled(r, 2.5, if k < filled { t.accent } else { t.field_border });
    }
}

/// The Speed bar: 5 for the quickest row on the page, fewer as a run takes longer; from the
/// speeds a model offers when the seconds are unknown.
pub fn speed_score(eta: Option<f32>, quickest: Option<f32>, speeds: &[Speed]) -> u8 {
    match (eta, quickest) {
        (Some(e), Some(q)) if e > 0.0 && q > 0.0 => (5.0 * q / e).round().clamp(1.0, 5.0) as u8,
        _ if speeds.contains(&Speed::Fast) => 4,
        _ if speeds.contains(&Speed::Balanced) => 3,
        _ => 2,
    }
}

/// The Memory bar: 1 for a small model, 5 for one that fills the budget (or 32 GB).
pub fn memory_score(need: Option<f32>, budget: Option<f32>) -> u8 {
    match need {
        Some(n) => (5.0 * n / budget.unwrap_or(32.0).max(1.0)).ceil().clamp(1.0, 5.0) as u8,
        None => 0,
    }
}

/// The preview of row `n`; true when Use (or Download…) was clicked.
fn preview(ui: &mut Ui, items: &[Item], rows: &[Row], n: usize, opts: &Opts<'_>, chosen: bool, t: &Tokens) -> bool {
    let row = &rows[n];
    let it = catalog::resolve(items, row, opts.page, opts.speed, opts.refs);
    ui.add_space(2.0);
    sample(ui, Some(&row.model), t);
    ui.add_space(6.0);
    ui.label(egui::RichText::new(&row.label).size(13.0).strong().color(t.text));
    let good = if row.model.good.is_empty() { it.map(|i| i.description.as_str()).unwrap_or_default() } else { row.model.good };
    if !good.is_empty() {
        ui.label(egui::RichText::new(good).size(11.0).color(t.text_dim));
    }
    if !row.model.tags.is_empty() {
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
            for tag in row.model.tags {
                let galley = ui.painter().layout_no_wrap(tag.to_string(), FontId::proportional(10.0), t.text_dim);
                let (r, _) = ui.allocate_exact_size(vec2(galley.size().x + 10.0, 16.0), Sense::hover());
                ui.painter().rect_stroke(r, 8.0, Stroke::new(1.0, t.field_border), egui::StrokeKind::Inside);
                ui.painter().galley(pos2(r.left() + 5.0, r.center().y - galley.size().y / 2.0), galley, t.text_dim);
            }
        });
    }
    ui.add_space(6.0);
    let quickest = rows
        .iter()
        .filter_map(|r| catalog::resolve(items, r, opts.page, opts.speed, opts.refs).and_then(|i| i.eta))
        .filter(|e| *e > 0.0)
        .fold(None, |a: Option<f32>, e| Some(a.map_or(e, |a| a.min(e))));
    let eta = it.and_then(|i| i.eta);
    let speeds = catalog::row_speeds(items, row, opts.page);
    let secs = eta.map(crate::generate_panel::eta_label).unwrap_or_default();
    bar(ui, tl!("Speed"), speed_score(eta, quickest, &speeds), &secs, t);
    bar(ui, tl!("Detail"), row.model.detail.clamp(1, 5), "", t);
    let need = it.and_then(|i| i.need_gb);
    let gb = need.map(|g| format!("{g:.0} GB")).unwrap_or_default();
    if need.is_some() {
        bar(ui, tl!("Memory"), memory_score(need, opts.budget_gb), &gb, t);
    }
    ui.add_space(8.0);
    let missing = it.is_some_and(|i| i.installed == Some(false) || !i.allowed);
    let label = if missing {
        tl!("Download…")
    } else if chosen {
        tl!("In use")
    } else {
        tl!("Use")
    };
    let w = ui.available_width();
    let clicked = ui.add_enabled_ui(it.is_some() && (missing || !chosen), |ui| crate::widgets::primary_button(ui, label, w)).inner.clicked();
    ui.add_space(2.0);
    clicked
}

/// The Auto preview; true when Use was clicked.
fn preview_auto(ui: &mut Ui, auto: Option<&str>, chosen: bool, t: &Tokens) -> bool {
    ui.add_space(2.0);
    sample(ui, None, t);
    ui.add_space(6.0);
    ui.label(egui::RichText::new(tl!("Auto")).size(13.0).strong().color(t.text));
    ui.label(egui::RichText::new(tl!("Picks the model for this computer, the Speed and what is attached")).size(11.0).color(t.text_dim));
    ui.add_space(4.0);
    ui.label(egui::RichText::new(auto_description(auto)).size(11.0).color(t.text_faint));
    ui.add_space(8.0);
    let w = ui.available_width();
    let label = if chosen { tl!("In use") } else { tl!("Use") };
    ui.add_enabled_ui(!chosen, |ui| crate::widgets::primary_button(ui, label, w)).inner.clicked()
}

#[cfg(test)]
#[path = "model_picker_tests.rs"]
mod tests;
