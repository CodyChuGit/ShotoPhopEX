//! Window › Generate's drawing: the compact row, the full form (mode switch, references deck,
//! composer, settings, LoRA rows, expand sides) and the footer with its Generate button, plus
//! the section, callout, disclosure and segmented widgets other panels share.

use egui::{Align2, FontId, Rect, RichText, Sense, Stroke, StrokeKind, pos2, vec2};
use serde_json::{Value, json};

use crate::theme::Tokens;
use crate::{PhotocraftApp, icons, widgets};

use super::{
    Action, MAX_PROMPT, MODES, POSE, SIZES, THUMB, auto_name, blocked, effective_template, estimate, eta_label, hint, image_size, info, mode_label,
    picker_items, probe_once, refresh_loras, refs_limit, reposing, request, running_job, swapping,
};

pub(super) fn generate_button(app: &PhotocraftApp, ui: &mut egui::Ui, action: &mut Option<Action>, width: f32) {
    if let Some(j) = running_job(app) {
        let progress = j.progress.clamp(0.0, 1.0);
        let label = format!("{} · {:.0}%", tl!("Cancel"), progress * 100.0);
        let r = widgets::secondary_button(ui, &label, width);
        if r.clicked() {
            *action = Some(Action::Cancel(j.id));
        }
        // The full form: a progress line under the button.
        if width > 0.0 {
            let t = Tokens::get(ui.ctx());
            let (bar, _) = ui.allocate_exact_size(vec2(r.rect.width(), 3.0), Sense::hover());
            ui.painter().rect_filled(bar, 1.5, t.field_border);
            let done = Rect::from_min_size(bar.min, vec2(bar.width() * progress, bar.height()));
            ui.painter().rect_filled(done, 1.5, t.accent);
        }
        return;
    }
    let mode = app.ui.generate.mode.clone();
    // A narrow one-line form keeps the prompt usable: the estimate moves to the tooltip.
    let narrow = width == 0.0 && ui.available_width() < 240.0;
    let eta = estimate(app, &mode).map(eta_label).filter(|s| !s.is_empty() && !narrow);
    let label = match eta {
        Some(e) => format!("{} · {e}", tl!("Generate")),
        None => tl!("Generate").to_string(),
    };
    let can = request(app).is_ok();
    let r = ui.add_enabled_ui(can, |ui| widgets::primary_button(ui, &label, width)).inner;
    if r.clicked() {
        *action = Some(Action::Generate);
    }
    if !can && let Err(why) = request(app) {
        r.on_disabled_hover_text(why);
    }
}

/// Enter (or Ctrl+Enter) in a focused prompt generates: Ctrl+Enter always, plain Enter when
/// `enter` (the one-line form always; the full form when the preference says so).
pub(super) fn submitted(ui: &egui::Ui, app: &PhotocraftApp, field: egui::Id, enter: bool) -> bool {
    let focused = ui.ctx().memory(|m| m.has_focus(field));
    focused
        && request(app).is_ok()
        && ui.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Enter) || (enter && i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)))
}

pub(super) fn compact_row(app: &mut PhotocraftApp, ui: &mut egui::Ui, _t: &Tokens, action: &mut Option<Action>) {
    let field_id = egui::Id::unique("generate-prompt-compact");
    if submitted(ui, app, field_id, true) {
        *action = Some(Action::Generate);
    }
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let mut mode = app.ui.generate.mode.clone();
        let options: Vec<(String, &str)> = MODES.iter().map(|m| (m.to_string(), mode_label(m))).collect();
        if widgets::dropdown(ui, "generate-mode-compact", tl!("Mode"), &mut mode, &options, 70.0) {
            *action = Some(Action::Mode(mode.clone()));
        }
        // The button at the right edge first; the prompt takes what is left.
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            generate_button(app, ui, action, 0.0);
            model_pill(app, ui, &mode, 30.0);
            let w = ui.available_width().max(40.0);
            let field = egui::TextEdit::singleline(&mut app.ui.generate.prompt).hint_text(hint(&mode)).desired_width(w).id(field_id);
            let resp = ui.add(field);
            if std::mem::take(&mut app.ui.generate.focus) {
                resp.request_focus();
            }
        });
    });
}

/// Height of the panel's controls (segmented controls, the model pill).
pub(crate) const ROW_H: f32 = 26.0;

/// The settings' label column.
pub(crate) const LABEL_W: f32 = 64.0;

/// Space between the panel's sections.
pub(crate) const SECTION_GAP: f32 = 12.0;

/// The full form, top to bottom: the mode switch, what blocks the mode (a callout), the
/// references, the prompt composer (its own toolbar: Enhance, the large editor, the counter), the
/// negative prompt behind a disclosure, the settings as labelled rows, LoRAs, and one full-width
/// Generate with the estimate (Cancel and a progress line while it runs).
pub(super) fn full(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &Tokens, action: &mut Option<Action>) {
    let mode = app.ui.generate.mode.clone();
    ui.spacing_mut().item_spacing.y = 6.0;
    // First run: no Models folder yet (nothing to migrate from an old ComfyUI install).
    if app.session.prefs().integrations.models_root().is_empty() {
        callout(ui, t, "folder-open", tl!("Download the AI models matched to this computer, or choose a folder that already holds them"), t.accent);
        let w = ui.available_width();
        if widgets::primary_button(ui, tl!("Get AI Models…"), w).clicked() {
            *action = Some(Action::GetModels);
        }
        if widgets::secondary_button(ui, tl!("Set Models Folder…"), w).clicked() {
            *action = Some(Action::OpenModelsFolder);
        }
        ui.add_space(SECTION_GAP);
    }
    mode_switch(ui, &mode, action);
    if let Some(why) = blocked(app, &mode) {
        callout(ui, t, "info", why, t.warning);
    }
    ui.add_space(SECTION_GAP);

    // References: their own section above the prompt, when the template takes them.
    let template = effective_template(app, &mode);
    let refs_max = refs_limit(app, &mode);
    let n_refs = app.ui.generate.references.len();
    let has_refs = matches!(mode.as_str(), "image" | "edit" | "fill") && n_refs > 0;
    if refs_max > 0 || has_refs {
        let color = if n_refs as u32 > refs_max { t.danger } else { t.text_faint };
        section(ui, t, tl!("References"), Some((format!("{n_refs} / {refs_max}"), color)));
        deck(app, ui, t, refs_max, action);
        ui.add_space(SECTION_GAP);
    }

    section(ui, t, tl!("Prompt"), None);
    composer(app, ui, t, &mode, action);
    if mode != "expand" {
        let open = app.ui.generate.show_negative;
        if disclosure(ui, t, tl!("Negative prompt"), open).clicked() {
            app.ui.generate.show_negative = !open;
        }
        if app.ui.generate.show_negative {
            ui.add(
                egui::TextEdit::singleline(&mut app.ui.generate.negative)
                    .hint_text(tl!("What to avoid…"))
                    .desired_width(f32::INFINITY)
                    .id_salt("generate-negative"),
            );
        }
    }
    ui.add_space(SECTION_GAP);

    section(ui, t, tl!("Settings"), None);
    settings(app, ui, t, &mode, &template);
    lora_rows(app, ui, &mode, &template);
}

/// The full form's footer, under the scrolling part so it never scrolls out of a short dock
/// (in Image mode the button sat below the fold, 2026-11-08): the one primary action, full
/// width; Clear all and the shortcut under it.
pub(super) fn footer(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &Tokens, action: &mut Option<Action>) {
    let w = ui.available_width();
    generate_button(app, ui, action, w);
    ui.horizontal(|ui| {
        if ui.link(RichText::new(tl!("Clear all")).small().color(t.text_dim)).clicked() {
            *action = Some(Action::ClearAll);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let key = if cfg!(target_os = "macos") { "Cmd+Enter" } else { "Ctrl+Enter" };
            let hint = if app.session.prefs().integrations.enter_to_generate { tl!("Enter to generate") } else { key };
            ui.label(RichText::new(hint).small().color(t.text_faint));
        });
    });
}

/// One option of a [`segmented`] control.
pub(crate) struct Seg<'a, T> {
    pub(super) value: T,
    pub(super) label: &'a str,
    pub(super) icon: Option<&'static str>,
    pub(super) tip: Option<&'a str>,
}

impl<'a, T> Seg<'a, T> {
    pub(crate) fn new(value: T, label: &'a str) -> Self {
        Self { value, label, icon: None, tip: None }
    }
    pub(crate) fn icon(mut self, icon: &'static str) -> Self {
        self.icon = Some(icon);
        self
    }
    pub(crate) fn tip(mut self, tip: &'a str) -> Self {
        self.tip = Some(tip);
        self
    }
}

/// A segmented control `width` wide (0: the available width): equal segments in one track, the
/// chosen one raised. A segment too narrow for its icon and label shows the icon alone (the label
/// moves to the tooltip). Returns the option clicked, when it is not the current one.
pub(crate) fn segmented<T: PartialEq + Clone>(ui: &mut egui::Ui, current: &T, options: &[Seg<'_, T>], width: f32) -> Option<T> {
    let t = Tokens::get(ui.ctx());
    let width = if width > 0.0 { width } else { ui.available_width() };
    let (track, track_resp) = ui.allocate_exact_size(vec2(width, ROW_H), Sense::hover());
    let r = t.radius_sm;
    ui.painter().rect_filled(track, r, t.field);
    ui.painter().rect_stroke(track, r, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
    let n = options.len().max(1) as f32;
    let seg_w = (track.width() - 4.0) / n;
    let enabled = ui.is_enabled();
    let mut picked = None;
    for (i, o) in options.iter().enumerate() {
        let rect = Rect::from_min_size(pos2(track.left() + 2.0 + seg_w * i as f32, track.top() + 2.0), vec2(seg_w, ROW_H - 4.0));
        let resp = ui.interact(rect, track_resp.id.with(i), Sense::click());
        let selected = o.value == *current;
        if selected {
            widgets::surface(ui, rect, t.hover, true);
            if !t.bevel {
                ui.painter().rect_stroke(rect, (r - 1.0).max(0.0), Stroke::new(1.0, t.field_border), StrokeKind::Inside);
            }
        } else if resp.hovered() && enabled {
            ui.painter().rect_filled(rect, (r - 1.0).max(0.0), t.hover.gamma_multiply(0.5));
        } else if i > 0 && options.get(i - 1).is_none_or(|p| p.value != *current) {
            // A divider between two unselected segments.
            let x = rect.left();
            ui.painter().line_segment([pos2(x, rect.top() + 5.0), pos2(x, rect.bottom() - 5.0)], Stroke::new(1.0, t.separator));
        }
        if resp.has_focus() {
            ui.painter().rect_stroke(rect, r, Stroke::new(2.0, t.accent), StrokeKind::Inside);
        }
        let color = if !enabled {
            t.text_faint
        } else if selected {
            t.text
        } else {
            t.text_dim
        };
        let icon_w = if o.icon.is_some() { 18.0 } else { 0.0 };
        // A text-only label too wide for its segment (four segments in a narrow dock) is set
        // smaller instead of being clipped at both ends.
        let lay = |size: f32| ui.painter().layout_no_wrap(o.label.to_string(), crate::theme::medium(size), color);
        let mut galley = lay(12.0);
        for size in [11.0, 10.0] {
            if o.icon.is_none() && galley.size().x + 4.0 > rect.width() {
                galley = lay(size);
            }
        }
        let fits = icon_w + galley.size().x + 8.0 <= rect.width();
        let show_label = fits || o.icon.is_none();
        let content_w = if show_label { icon_w + galley.size().x } else { 14.0 };
        let mut x = rect.center().x - content_w / 2.0;
        if let Some(icon) = o.icon {
            let ir = Rect::from_min_size(pos2(x, rect.center().y - 7.0), vec2(14.0, 14.0));
            icons::paint(ui, ir, icon, 14.0, color);
            x += icon_w;
        }
        if show_label {
            ui.painter().with_clip_rect(rect.shrink(2.0)).galley(pos2(x, rect.center().y - galley.size().y / 2.0), galley, color);
        }
        let label = o.label.to_string();
        resp.widget_info(|| egui::WidgetInfo::selected(egui::Role::Button, enabled, selected, &label));
        let resp = match (o.tip, show_label) {
            (Some(tip), true) => resp.on_hover_text(tip),
            (Some(tip), false) => resp.on_hover_text(format!("{}\n{tip}", o.label)),
            (None, false) => resp.on_hover_text(o.label),
            (None, true) => resp,
        };
        if resp.clicked() && enabled && !selected {
            picked = Some(o.value.clone());
        }
    }
    picked
}

/// Image · Fill · Edit · Expand, with the Generate hub's icons.
pub(super) fn mode_switch(ui: &mut egui::Ui, mode: &str, action: &mut Option<Action>) {
    let icon = |m: &str| match m {
        "fill" => "square-dashed",
        "edit" => "wand-sparkles",
        "expand" => "scaling",
        _ => "image",
    };
    let options: Vec<Seg<'_, String>> = MODES.iter().map(|m| Seg::new(m.to_string(), mode_label(m)).icon(icon(m))).collect();
    if let Some(m) = segmented(ui, &mode.to_string(), &options, 0.0) {
        *action = Some(Action::Mode(m));
    }
}

/// A section's title in small capitals, with an optional note at the right (a count).
pub(crate) fn section(ui: &mut egui::Ui, t: &Tokens, title: &str, right: Option<(String, egui::Color32)>) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(title.to_uppercase()).font(crate::theme::semibold(10.5)).color(t.text_faint));
        if let Some((note, color)) = right {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(note).small().color(color));
            });
        }
    });
}

/// A tinted note with an icon (what blocks the mode, the first-run Models folder).
pub(crate) fn callout(ui: &mut egui::Ui, t: &Tokens, icon: &str, text: &str, color: egui::Color32) {
    egui::Frame::new()
        .fill(color.gamma_multiply(0.12))
        .stroke(Stroke::new(1.0, color.gamma_multiply(0.45)))
        .corner_radius(t.radius_sm)
        .inner_margin(egui::Margin { left: 8, right: 8, top: 6, bottom: 6 })
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                let (r, _) = ui.allocate_exact_size(vec2(14.0, 16.0), Sense::hover());
                icons::paint(ui, r, icon, 14.0, color);
                ui.add(egui::Label::new(RichText::new(text).small().color(t.text)).wrap());
            });
        });
}

/// A chevron and a label that open and close what follows (the negative prompt).
pub(crate) fn disclosure(ui: &mut egui::Ui, t: &Tokens, label: &str, open: bool) -> egui::Response {
    let galley = ui.painter().layout_no_wrap(label.to_string(), FontId::proportional(12.0), t.text_dim);
    let (rect, resp) = ui.allocate_exact_size(vec2(18.0 + galley.size().x, 20.0), Sense::click());
    let color = if resp.hovered() { t.text } else { t.text_dim };
    let chevron = Rect::from_min_size(pos2(rect.left(), rect.center().y - 7.0), vec2(14.0, 14.0));
    icons::paint(ui, chevron, if open { "chevron-down" } else { "chevron-right" }, 12.0, color);
    ui.painter().galley_with_override_text_color(pos2(rect.left() + 18.0, rect.center().y - galley.size().y / 2.0), galley, color);
    if resp.has_focus() {
        ui.painter().rect_stroke(rect.expand(2.0), t.radius_sm, Stroke::new(2.0, t.accent), StrokeKind::Outside);
    }
    let enabled = ui.is_enabled();
    resp.widget_info(|| egui::WidgetInfo::selected(egui::Role::Button, enabled, open, label));
    resp
}

/// The prompt in one framed box (the accent frame while it has the keyboard) with its toolbar
/// inside: Enhance, the large editor, and the counter. Grows from three to eight rows.
pub(super) fn composer(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &Tokens, mode: &str, action: &mut Option<Action>) {
    let rows = app.ui.generate.prompt.lines().count().clamp(3, 8);
    let enter_generates = app.session.prefs().integrations.enter_to_generate;
    let field_id = egui::Id::unique("generate-prompt");
    let submit = submitted(ui, app, field_id, enter_generates);
    let focused = ui.ctx().memory(|m| m.has_focus(field_id));
    let border = if focused { t.accent } else { t.field_border };
    egui::Frame::new()
        .fill(t.field)
        .stroke(Stroke::new(1.0, border))
        .corner_radius(t.radius_sm)
        .inner_margin(egui::Margin { left: 8, right: 4, top: 6, bottom: 2 })
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            let field = egui::TextEdit::multiline(&mut app.ui.generate.prompt)
                .hint_text(hint(mode))
                .desired_rows(rows)
                .desired_width(f32::INFINITY)
                .char_limit(MAX_PROMPT * 2)
                .frame(egui::Frame::NONE)
                .id(field_id);
            let resp = ui.add(field);
            if std::mem::take(&mut app.ui.generate.focus) {
                resp.request_focus();
            }
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                let enhancing = app.ui.generate.enhance_job.is_some();
                let can = !enhancing && !app.ui.generate.prompt.trim().is_empty();
                let tip = if can || enhancing { tl!("Enhance prompt") } else { tl!("Type a prompt first") };
                if icons::button(ui, "wand-sparkles", 24.0, enhancing, tip).clicked() && can {
                    *action = Some(Action::Enhance);
                }
                if icons::button(ui, "maximize-2", 24.0, false, tl!("Open the prompt in a larger editor")).clicked() {
                    *action = Some(Action::OpenEditor(true));
                }
                if enhancing {
                    ui.label(RichText::new(tl!("Enhancing…")).small().color(t.accent));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(4.0);
                    let n = app.ui.generate.prompt.chars().count();
                    let color = if n > MAX_PROMPT { t.danger } else { t.text_faint };
                    ui.label(RichText::new(format!("{n} / {MAX_PROMPT}")).small().color(color));
                });
            });
        });
    if submit {
        *action = Some(Action::Generate);
    }
}

/// One settings row: the name in a fixed column (elided when a translation is longer), then
/// the control in the rest of the width.
pub(crate) fn setting_row(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.allocate_ui_with_layout(vec2(LABEL_W, ROW_H), egui::Layout::left_to_right(egui::Align::Center), |ui| {
            ui.set_min_size(vec2(LABEL_W, ROW_H));
            widgets::elided_label(ui, label, LABEL_W);
        });
        add(ui);
    });
}

/// The model picker of the mode (Auto, then the page's templates with their estimates); the
/// one-row form draws it as the family glyph alone. Its popover opening asks the server which
/// templates it has, once.
pub(super) fn model_pill(app: &mut PhotocraftApp, ui: &mut egui::Ui, mode: &str, width: f32) {
    let group = app.session.prefs().integrations.group_models_by_family;
    let mut current = app.ui.generate.template.get(mode).cloned().unwrap_or_default();
    let items = picker_items(app, mode, true);
    let auto = auto_name(app, mode);
    if crate::model_picker::pill(ui, "generate-panel", &items, &mut current, auto.as_deref(), group, width) {
        app.ui.generate.template.insert(mode.to_string(), current);
    }
    if crate::model_picker::is_open(ui.ctx(), "generate-panel") {
        probe_once(app);
    }
}

/// Preferences › AI Integrations › Quality beside the model: how far above the asked size a new
/// picture renders before it is brought down to it, on the models that gain from it (Basic 1x,
/// Better 1.25x, Best 1.5x, Ultra 2x and a light refine pass, offered on 24 GB cards or larger).
/// It never changes the model.
pub(super) fn quality_pick(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    use photocraft_engine::generate_cmds as g;
    let q = g::quality_level(&app.session).to_string();
    let mut options = Vec::with_capacity(4);
    if g::ultra_allowed(&app.session) {
        options.push(Seg::new("ultra".to_string(), tl!("Ultra")).tip(tl!("Renders at twice the size and brings it down: the most detail, the slowest")));
    }
    options.extend([
        Seg::new("best".to_string(), tl!("Best")).tip(tl!("Renders at 1.5 times the size and brings it down: finer skin, hair and text")),
        Seg::new("better".to_string(), tl!("Better")).tip(tl!("Renders at 1.25 times the size and brings it down")),
        Seg::new("basic".to_string(), tl!("Basic")).tip(tl!("Renders at the size asked: the fastest")),
    ]);
    if let Some(q) = segmented(ui, &q, &options, 0.0) {
        let _ = app.run("prefs.set", json!({"path": "integrations.qualityLevel", "value": q}));
    }
}

/// The settings rows: model and quality, then what the mode takes (aspect and size, the count,
/// where the picture goes, the fill's widen and denoise, the expand's sides).
pub(super) fn settings(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &Tokens, mode: &str, template: &str) {
    setting_row(ui, tl!("Model"), |ui| {
        let w = ui.available_width();
        model_pill(app, ui, mode, w);
    });
    setting_row(ui, tl!("Quality"), |ui| quality_pick(app, ui));
    let m = info(app, template).cloned().unwrap_or_default();
    if mode == "image" {
        setting_row(ui, tl!("Aspect"), |ui| {
            let mut aspect = app.ui.generate.aspect.clone();
            let aspects: Vec<(String, &str)> =
                m.aspects.iter().map(|a| (a.clone(), a.as_str())).chain(std::iter::once(("custom".to_string(), tl!("Custom")))).collect();
            let w = ui.available_width();
            if widgets::dropdown(ui, "generate-aspect", tl!("Aspect Ratio"), &mut aspect, &aspects, w) {
                app.ui.generate.aspect = aspect;
            }
        });
        setting_row(ui, tl!("Size"), |ui| {
            if app.ui.generate.aspect == "custom" {
                let (lo, hi) = (m.size_min.max(64) as f32, if m.size_max == 0 { 2048.0 } else { m.size_max as f32 });
                let fw = ((ui.available_width() - 20.0) / 2.0).clamp(52.0, 90.0);
                let mut w = app.ui.generate.width as f32;
                let mut h = app.ui.generate.height as f32;
                widgets::value_field(ui, &mut w, lo..=hi, "px", fw).on_hover_text(tl!("Width"));
                ui.label(RichText::new("×").color(t.text_dim));
                widgets::value_field(ui, &mut h, lo..=hi, "px", fw).on_hover_text(tl!("Height"));
                app.ui.generate.width = w.round() as u32;
                app.ui.generate.height = h.round() as u32;
            } else {
                let options: Vec<Seg<'_, String>> =
                    SIZES.iter().filter(|(_, v)| m.size_max == 0 || *v <= m.size_max).map(|(k, _)| Seg::new(k.to_string(), k)).collect();
                if let Some(s) = segmented(ui, &app.ui.generate.size, &options, 0.0) {
                    app.ui.generate.size = s;
                }
            }
        });
        // What it will make, on the template's grid, and what Quality renders it at first.
        let (w, h) = image_size(app);
        let tid = effective_template(app, "image");
        let (rw, rh) = photocraft_engine::generate_cmds::image_render_sizes(&app.session, &tid, w, h);
        let level = photocraft_engine::generate_cmds::quality_level(&app.session);
        let gains = photocraft_engine::generate_cmds::SUPERSAMPLE_TEMPLATES.contains(&tid.as_str());
        let hint = if (rw, rh) != (w, h) && gains && photocraft_engine::generate_cmds::ultra_refines(&tid, level) {
            format!("{w} × {h} px · {} {rw} × {rh} {}", tl!("rendered at"), tl!("and refined"))
        } else if (rw, rh) != (w, h) && gains {
            format!("{w} × {h} px · {} {rw} × {rh}", tl!("rendered at"))
        } else if (rw, rh) != (w, h) {
            format!("{rw} × {rh} px")
        } else if level != "basic" && !gains {
            format!("{w} × {h} px · {}", tl!("this model renders at the size asked"))
        } else {
            format!("{w} × {h} px")
        };
        setting_row(ui, "", |ui| {
            ui.label(RichText::new(hint).small().color(t.text_faint));
        });
    }
    if mode != "expand" {
        let max = m.counts_max.clamp(1, 4) as u8;
        let n = app.ui.generate.count.clamp(1, max);
        app.ui.generate.count = n;
        if max > 1 {
            setting_row(ui, tl!("Count"), |ui| {
                let options: Vec<Seg<'_, u8>> =
                    [(1u8, "1"), (2, "2"), (3, "3"), (4, "4")].into_iter().filter(|(c, _)| *c <= max).map(|(c, l)| Seg::new(c, l)).collect();
                if let Some(c) = segmented(ui, &n, &options, 0.0) {
                    app.ui.generate.count = c;
                }
            });
        }
    }
    if mode == "image" && app.session.active().is_some() {
        setting_row(ui, tl!("Output"), |ui| {
            let into = if app.ui.generate.into == "layer" { "layer".to_string() } else { "document".to_string() };
            let options = [
                Seg::new("document".to_string(), tl!("Document")).icon("file-plus").tip(tl!("A new document")),
                Seg::new("layer".to_string(), tl!("Layer")).icon("layers").tip(tl!("A new layer over the open picture")),
            ];
            if let Some(x) = segmented(ui, &into, &options, 0.0) {
                app.ui.generate.into = x;
            }
        });
    }
    if mode == "fill" {
        // Widen the selection first (the fill's `grow`): room for a bulkier replacement.
        setting_row(ui, tl!("Widen"), |ui| {
            let max = photocraft_engine::generate_cmds::MAX_GROW as f32;
            let mut g = app.ui.generate.grow as f32;
            widgets::value_field(ui, &mut g, 0.0..=max, "px", 72.0)
                .on_hover_text(tl!("Widen the selection by this many pixels before the fill, so a bulkier replacement has room"));
            app.ui.generate.grow = g.round().clamp(0.0, max) as u32;
        });
        // How much is redrawn, where the template's sampler takes it (the 2511 fills).
        if photocraft_engine::generate_cmds::takes_denoise(template) {
            setting_row(ui, tl!("Denoise"), |ui| {
                let mut d = app.ui.generate.denoise;
                widgets::value_field_prec(ui, &mut d, 0.05..=1.0, "", 72.0, 2)
                    .on_hover_text(tl!("How much of the selection is redrawn: 1 replaces it, about 0.9 keeps the old shape"));
                app.ui.generate.denoise = d.clamp(0.05, 1.0);
            });
        }
    }
    if mode == "edit" {
        // Swap: the head (face and hair) or the whole person from the first reference, onto the
        // person in the picture (the BFS LoRAs on Qwen-Image-2.1); or Pose: the person from the
        // first reference redrawn in the pose of the mannequin in the picture (PoseStudio).
        setting_row(ui, tl!("Person"), |ui| {
            let cur = if swapping(app) || reposing(app) { app.ui.generate.swap.clone() } else { String::new() };
            let options = [
                Seg::new(String::new(), tl!("Off")).tip(tl!("An ordinary edit by instruction")),
                Seg::new("head".to_string(), tl!("Face")).tip(tl!("Put the face and hair from the first reference onto the person in the picture")),
                Seg::new("body".to_string(), tl!("Body"))
                    .tip(tl!("Put the whole person from the first reference (clothes, build, face) into the picture, in the same pose")),
                Seg::new(POSE.to_string(), tl!("Pose")).tip(tl!("Draw the person from the first reference in the pose of the mannequin in the open picture")),
            ];
            if let Some(x) = segmented(ui, &cur, &options, 0.0) {
                app.ui.generate.swap = x;
                let template = effective_template(app, "edit");
                refresh_loras(app, &template);
            }
        });
        if swapping(app) && app.ui.generate.references.is_empty() {
            // Wrapped to the row's width: one long line widened the dock and pushed Generate off it.
            setting_row(ui, "", |ui| {
                let w = ui.available_width();
                ui.allocate_ui_with_layout(vec2(w, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
                    ui.set_max_width(w);
                    ui.add(egui::Label::new(RichText::new(tl!("Add a picture of who to swap in as a reference")).small().color(t.text_faint)).wrap());
                });
            });
        }
        if reposing(app) && app.ui.generate.references.is_empty() {
            setting_row(ui, "", |ui| {
                let w = ui.available_width();
                ui.allocate_ui_with_layout(vec2(w, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
                    ui.set_max_width(w);
                    ui.add(
                        egui::Label::new(
                            RichText::new(tl!("Add a picture of who to draw as a reference; the open picture is the pose")).small().color(t.text_faint),
                        )
                        .wrap(),
                    );
                });
            });
        }
    }
    if mode == "expand" {
        expand_sides(app, ui, t);
    }
}

/// A side of the expand, in pixels.
pub(super) fn side_field(ui: &mut egui::Ui, v: &mut u32, name: &str) {
    let mut f = *v as f32;
    widgets::value_field_named(ui, name, &mut f, 0.0..=4096.0, "px", 64.0).on_hover_text(name);
    *v = f.round() as u32;
}

/// The expand's four sides around a diagram of the result: the picture (grey) inside the grown
/// canvas (accent), to scale.
pub(super) fn expand_sides(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &Tokens) {
    let mut e = app.ui.generate.expand;
    ui.label(RichText::new(tl!("Pixels to add on each side")).small().color(t.text_faint));
    ui.vertical_centered(|ui| side_field(ui, &mut e[1], tl!("Top")));
    ui.horizontal(|ui| {
        let gap = ui.spacing().item_spacing.x;
        let box_w = (ui.available_width() - 2.0 * (64.0 + gap)).max(40.0);
        side_field(ui, &mut e[0], tl!("Left"));
        let (r, _) = ui.allocate_exact_size(vec2(box_w, 52.0), Sense::hover());
        let (dw, dh) = app.session.active().map_or((1.0, 1.0), |d| (d.doc.size.width.max(1) as f32, d.doc.size.height.max(1) as f32));
        let (nw, nh) = (dw + (e[0] + e[2]) as f32, dh + (e[1] + e[3]) as f32);
        let s = ((r.width() - 4.0) / nw).min((r.height() - 4.0) / nh).max(0.0);
        let outer = Rect::from_center_size(r.center(), vec2(nw * s, nh * s));
        let inner = Rect::from_min_size(outer.min + vec2(e[0] as f32 * s, e[1] as f32 * s), vec2(dw * s, dh * s));
        ui.painter().rect_filled(outer, 2.0, t.accent.gamma_multiply(0.22));
        ui.painter().rect_stroke(outer, 2.0, Stroke::new(1.0, t.accent), StrokeKind::Inside);
        ui.painter().rect_filled(inner, 1.0, t.text_faint);
        side_field(ui, &mut e[2], tl!("Right"));
    });
    ui.vertical_centered(|ui| side_field(ui, &mut e[3], tl!("Bottom")));
    app.ui.generate.expand = e;
}

/// The LoRA rows of the template's family (`generate.loras`), when the template runs on the
/// native engine: a switch and a strength per file, folded away under "LoRAs".
pub(super) fn lora_rows(app: &mut PhotocraftApp, ui: &mut egui::Ui, mode: &str, template: &str) {
    if !matches!(mode, "image" | "fill" | "edit") {
        return;
    }
    refresh_loras(app, template);
    let Some((_, true, files)) = app.ui.generate.lora_list.clone() else { return };
    if files.is_empty() {
        return;
    }
    let on = files.iter().filter(|f| app.ui.generate.loras.get(*f).is_some_and(|(o, _)| *o)).count();
    let title = if on > 0 { format!("{} ({on})", tl!("LoRAs")) } else { tl!("LoRAs").to_string() };
    egui::CollapsingHeader::new(RichText::new(title).small()).id_salt("generate-loras").show(ui, |ui| {
        ui.label(RichText::new(tl!("Adapters in the models folder's loras subfolder, applied by the native engine")).small().weak());
        for f in &files {
            let entry = app.ui.generate.loras.entry(f.clone()).or_insert((false, 1.0));
            ui.horizontal(|ui| {
                ui.checkbox(&mut entry.0, f.strip_suffix(".safetensors").unwrap_or(f));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    widgets::value_field_prec(ui, &mut entry.1, 0.0..=1.5, "", 52.0, 2).on_hover_text(tl!("Strength"));
                });
            });
        }
    });
}

/// The reference deck: thumbnails with × and context menus, then an Add tile with the + menu.
pub(super) fn deck(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &Tokens, max: u32, action: &mut Option<Action>) {
    let refs = app.ui.generate.references.clone();
    let body = |ui: &mut egui::Ui| {
        ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
        for (i, spec) in refs.iter().enumerate() {
            let (rect, resp) = ui.allocate_exact_size(vec2(THUMB, THUMB), Sense::click());
            widgets::checker(ui.painter(), rect, 4.0);
            match reference_texture(app, ui.ctx(), spec) {
                Some(tex) => {
                    ui.painter().image(tex, rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), egui::Color32::WHITE);
                }
                None => icons::paint(ui, rect, "image", 18.0, t.text_faint),
            }
            let over = i as u32 >= max;
            let stroke = if over {
                Stroke::new(1.5, t.danger)
            } else if resp.hovered() {
                Stroke::new(1.0, t.accent)
            } else {
                Stroke::new(1.0, t.separator)
            };
            ui.painter().rect_stroke(rect, t.radius_sm, stroke, StrokeKind::Inside);
            // The number the prompt can refer to ("image 2"), on a dark chip so it reads on any picture.
            let chip = Rect::from_min_size(rect.left_bottom() + vec2(2.0, -14.0), vec2(12.0, 12.0));
            ui.painter().rect_filled(chip, 3.0, egui::Color32::from_black_alpha(160));
            ui.painter().text(chip.center(), Align2::CENTER_CENTER, (i + 1).to_string(), FontId::proportional(9.0), egui::Color32::WHITE);
            if resp.hovered() {
                let x = Rect::from_min_size(pos2(rect.right() - 15.0, rect.top() + 2.0), vec2(13.0, 13.0));
                ui.painter().circle_filled(x.center(), 6.5, t.chrome.gamma_multiply(0.92));
                icons::paint(ui, x, "x", 9.0, t.text);
                if resp.clicked() && ui.input(|i| i.pointer.interact_pos()).is_some_and(|p| x.contains(p)) {
                    *action = Some(Action::RemoveRef(i));
                }
            }
            let tip = describe_reference(app, spec);
            let tip = if over { format!("{tip}\n{}", tl!("More than this model takes")) } else { tip };
            resp.on_hover_text(tip).context_menu(|ui| {
                if ui.button(tl!("Move left")).clicked() {
                    *action = Some(Action::MoveRef(i, -1));
                }
                if ui.button(tl!("Move right")).clicked() {
                    *action = Some(Action::MoveRef(i, 1));
                }
                if ui.button(tl!("Remove")).clicked() {
                    *action = Some(Action::RemoveRef(i));
                }
            });
        }
        if (refs.len() as u32) < max.max(1) {
            // The Add tile: a dashed square with a plus, the size of a thumbnail.
            let (rect, plus) = ui.allocate_exact_size(vec2(THUMB, THUMB), Sense::click());
            if plus.hovered() {
                ui.painter().rect_filled(rect, t.radius_sm, t.hover);
            }
            let color = if plus.hovered() { t.text } else { t.text_dim };
            dashed_rect(ui.painter(), rect.shrink(0.5), Stroke::new(1.0, if plus.hovered() { t.accent } else { t.field_border }));
            icons::paint(ui, rect, "plus", 16.0, color);
            let tip = tl!("Add a reference image");
            plus.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, tip));
            let plus = plus.on_hover_text(tip);
            egui::Popup::menu(&plus).show(|ui| add_menu(app, ui, action));
        }
        if refs.is_empty() {
            ui.allocate_ui_with_layout(vec2(ui.available_width().max(60.0), THUMB), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.add(
                    egui::Label::new(RichText::new(tl!("Optional: a layer, a file or a past generation to guide the result")).small().color(t.text_faint))
                        .wrap(),
                );
            });
        }
    };
    // Empty, the hint sits beside the Add tile; with thumbnails the row wraps.
    if refs.is_empty() {
        ui.horizontal(body);
    } else {
        ui.horizontal_wrapped(body);
    }
}

/// A dashed outline (the Add tile).
pub(super) fn dashed_rect(p: &egui::Painter, r: Rect, stroke: Stroke) {
    let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
    for w in pts.windows(2) {
        p.extend(egui::Shape::dashed_line(&[w[0], w[1]], stroke, 4.0, 3.0));
    }
}

pub(super) fn add_menu(app: &mut PhotocraftApp, ui: &mut egui::Ui, action: &mut Option<Action>) {
    // The document's pixel layers, active one first.
    if let Some(d) = app.session.active() {
        let active = d.active_layer;
        if let Some(l) = active.and_then(|a| d.doc.layer(a)).filter(|l| l.surface().is_some())
            && ui.button(crate::i18n::fmt(tl!("Active layer ({name})"), &[("name", &l.name)])).clicked()
        {
            *action = Some(Action::AddRef(json!({"layer": l.id.0})));
        }
        let layers: Vec<(u64, String)> = d.doc.layers.iter().rev().filter(|l| l.surface().is_some()).map(|l| (l.id.0, l.name.clone())).collect();
        if !layers.is_empty() {
            ui.menu_button(tl!("Layer"), |ui| {
                for (id, name) in layers {
                    if ui.button(name).clicked() {
                        *action = Some(Action::AddRef(json!({"layer": id})));
                    }
                }
            });
        }
    }
    if app.services.file_dialog.is_some() && ui.button(tl!("File…")).clicked() {
        *action = Some(Action::PickFile);
    }
    let recent: Vec<(u64, String)> = app
        .session
        .generations
        .records()
        .iter()
        .rev()
        .filter(|r| r.result.as_ref().is_some_and(|x| !x.files.is_empty()))
        .take(12)
        .map(|r| (r.id, if r.prompt.is_empty() { r.label.clone() } else { r.prompt.chars().take(40).collect() }))
        .collect();
    if !recent.is_empty() {
        ui.menu_button(tl!("From Generations"), |ui| {
            for (id, label) in recent {
                if ui.button(label).clicked() {
                    *action = Some(Action::AddRef(json!({"job": id, "index": 0})));
                }
            }
        });
    }
}

pub(super) fn describe_reference(app: &PhotocraftApp, spec: &Value) -> String {
    if let Some(id) = spec["layer"].as_u64() {
        let name = app
            .session
            .active()
            .and_then(|d| d.doc.layer(photocraft_doc::LayerId(id)))
            .map_or_else(|| tl!("a layer that is gone").to_string(), |l| l.name.clone());
        return crate::i18n::fmt(tl!("Layer: {name}"), &[("name", &name)]);
    }
    if let Some(p) = spec["path"].as_str() {
        return crate::i18n::fmt(tl!("File: {name}"), &[("name", &crate::file_open::display_name(p))]);
    }
    if let Some(id) = spec["job"].as_u64() {
        let label = app.session.generations.get(id).map_or_else(String::new, |r| r.prompt.clone());
        return crate::i18n::fmt(tl!("Generation: {name}"), &[("name", &label)]);
    }
    String::new()
}

/// A thumbnail for a reference: the layer's (from the layer cache), a generation's (from the
/// Generations panel's cache); files show an icon.
pub(super) fn reference_texture(app: &mut PhotocraftApp, ctx: &egui::Context, spec: &Value) -> Option<egui::TextureId> {
    if let Some(id) = spec["layer"].as_u64() {
        let doc = app.session.active()?.doc.clone();
        let layer = doc.layer(photocraft_doc::LayerId(id))?.clone();
        return Some(app.layer_thumb(ctx, &doc, &layer).0);
    }
    if let Some(id) = spec["job"].as_u64() {
        return crate::generations_panel::thumbnail(app, ctx, id).map(|t| t.id());
    }
    None
}

/// The large prompt editor (⤢).
pub(super) fn editor(app: &mut PhotocraftApp, ctx: &egui::Context, action: &mut Option<Action>) {
    if !app.ui.generate.editor {
        return;
    }
    let mode = app.ui.generate.mode.clone();
    egui::Modal::new(egui::Id::unique("generate-prompt-editor")).show(ctx, |ui| {
        ui.set_width(560.0);
        widgets::section_label(ui, mode_label(&mode));
        ui.add(egui::TextEdit::multiline(&mut app.ui.generate.prompt).hint_text(hint(&mode)).desired_rows(14).desired_width(f32::INFINITY));
        ui.add_space(8.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if widgets::primary_button(ui, tl!("Done"), 80.0).clicked() {
                *action = Some(Action::OpenEditor(false));
            }
        });
    });
}
