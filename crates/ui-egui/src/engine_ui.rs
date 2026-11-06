//! Native worker status, controls and resource details.
use egui::{Color32, Sense, vec2};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

/// Seconds between status polls (one quick local request each).
pub const POLL_EVERY: f64 = 2.0;

/// What the status bar knows: the last `generate.engine.status` and when it was asked.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EngineUi {
    pub status: Option<Value>,
    polled: f64,
}

/// The native worker state; unconfigured before the first poll.
pub fn state(app: &PhotocraftApp) -> String {
    native_state(app)
}
pub fn native_state(app: &PhotocraftApp) -> String {
    native(app).and_then(|v| v["state"].as_str()).unwrap_or("unconfigured").to_string()
}
fn native(app: &PhotocraftApp) -> Option<&Value> {
    app.ui.engine.status.as_ref().and_then(|v| v.get("native")).filter(|v| v.is_object())
}

/// The first line for a native worker state.
pub fn native_label(state: &str) -> String {
    match state {
        "unconfigured" => tl!("Native engine: not configured"),
        "starting" => tl!("Native engine: starting…"),
        "running" => tl!("Native engine: working"),
        "ready" => tl!("Native engine: ready"),
        "failed" => tl!("Native engine: failed"),
        _ => tl!("Native engine: stopped"),
    }
    .to_string()
}

/// The dot's colour for a state of either engine (`None` = hollow: nothing configured).
fn colour(state: &str, t: &Tokens) -> Option<Color32> {
    Some(match state {
        "unconfigured" => return None,
        "starting" => t.warning,
        "ready" | "running" => Color32::from_rgb(0x4c, 0xaf, 0x50),
        "failed" => t.danger,
        _ => t.text_faint,
    })
}

/// The native worker's lines for the tooltip and the engine page: its state, the card, the
/// memory, what is loaded, and the reason it gives (a refused card, a failed start). Empty while
/// the native engine is off.
pub fn native_lines(app: &PhotocraftApp) -> Vec<String> {
    let Some(n) = native(app) else { return Vec::new() };
    let st = native_state(app);
    let mut lines = vec![native_label(&st)];
    if st == "unconfigured" {
        lines.push(tl!("Set the runtime and model folders in Preferences › AI Integrations to use the native engine").to_string());
        return lines;
    }
    let device = n["device"].as_str().unwrap_or("").trim();
    if !device.is_empty() && device != "none" {
        lines.push(device.to_string());
    }
    match (n["vramFree"].as_u64(), n["vramTotal"].as_u64()) {
        (Some(free), Some(total)) => {
            let (free, total) = (crate::chrome_ui::fmt_bytes(free), crate::chrome_ui::fmt_bytes(total));
            lines.push(crate::i18n::fmt(tl!("{free} of {total} free"), &[("free", &free), ("total", &total)]));
        }
        (None, Some(total)) => lines.push(crate::chrome_ui::fmt_bytes(total)),
        _ => {}
    }
    let resident: Vec<&str> = n["resident"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    if !resident.is_empty() {
        lines.push(crate::i18n::fmt(tl!("Loaded: {models}"), &[("models", &resident.join(", "))]));
    }
    let detail = n["detail"].as_str().unwrap_or("").trim();
    if !detail.is_empty() {
        lines.push(detail.to_string());
    }
    lines
}

/// Poll the status every [`POLL_EVERY`] seconds (faster while starting, so the dot turns green
/// without a pointer move).
pub fn tick(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if cfg!(target_arch = "wasm32") {
        return;
    }
    let now = ctx.input(|i| i.time);
    let starting = native_state(app) == "starting";
    let every = if starting { 1.0 } else { POLL_EVERY };
    if app.ui.engine.status.is_some() && now - app.ui.engine.polled < every {
        return;
    }
    app.ui.engine.polled = now;
    // Straight to the engine: a poll must not touch the status line or the journal.
    app.ui.engine.status = app.session.execute("generate.engine.status", json!({})).ok();
    if native_state(app) == "starting" {
        ctx.request_repaint_after(std::time::Duration::from_secs(1));
    }
}

/// What a native button asks: `start`, `stop`, `purge` or `log`. Runs the command and keeps its
/// answer as the new status; errors become a notice.
pub fn native_action(app: &mut PhotocraftApp, action: &str) {
    let cmd = match action {
        "start" => "generate.engine.start",
        "stop" => "generate.engine.stop",
        "purge" => "generate.engine.purge",
        "log" => {
            let path = native(app).and_then(|v| v["log"].as_str()).map(str::to_string).unwrap_or_default();
            reveal(app, &path);
            return;
        }
        _ => return,
    };
    match app.run(cmd, json!({})) {
        Ok(v) => app.ui.engine.status = Some(v),
        Err(e) => crate::notices::error(app, e),
    }
}

/// The native section's buttons, in a menu or on the engine page; the chosen action.
pub fn native_buttons(ui: &mut egui::Ui, state: &str) -> Option<&'static str> {
    let running = matches!(state, "starting" | "running" | "ready" | "failed");
    let mut out = None;
    if ui.add_enabled(matches!(state, "stopped" | "failed"), egui::Button::new(tl!("Start engine"))).clicked() {
        out = Some("start");
    }
    if ui.add_enabled(running, egui::Button::new(tl!("Stop engine"))).clicked() {
        out = Some("stop");
    }
    if ui.add_enabled(matches!(state, "ready" | "failed"), egui::Button::new(tl!("Purge models"))).clicked() {
        out = Some("purge");
    }
    if ui.button(tl!("Show log")).clicked() {
        out = Some("log");
    }
    out
}

/// The dot and its menu, in a status bar.
pub fn dot(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    if cfg!(target_arch = "wasm32") {
        return;
    }
    let t = Tokens::get(ui.ctx());
    let nst = native_state(app);
    let (r, resp) = ui.allocate_exact_size(vec2(16.0, 16.0), Sense::click());
    if resp.hovered() {
        ui.painter().rect_filled(r, t.radius_sm, t.hover);
    }
    match colour(&nst, &t) {
        Some(c) => {
            ui.painter().circle_filled(r.center(), 4.0, c);
        }
        None => {
            ui.painter().circle_stroke(r.center(), 4.0, egui::Stroke::new(1.0, t.text_faint));
        }
    }
    let resp = resp.on_hover_text(native_lines(app).join("\n"));
    egui::Popup::menu(&resp).show(|ui| {
        ui.set_min_width(180.0);
        if let Some(a) = native_buttons(ui, &nst) {
            native_action(app, a);
            ui.close();
        }
    });
}

/// Reveal a file in the file manager (nothing for an empty path).
fn reveal(app: &mut PhotocraftApp, path: &str) {
    if path.is_empty() {
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        match photocraft_engine::layer_menu_cmds::reveal_command(path) {
            Ok((program, args)) => {
                if let Err(e) = std::process::Command::new(program).args(&args).spawn() {
                    crate::notices::error(app, format!("{program}: {e}"));
                }
            }
            Err(e) => crate::notices::error(app, e.to_string()),
        }
    }
    #[cfg(target_arch = "wasm32")]
    let _ = app;
}

/// End the worker owned by this process on quit.
pub fn on_quit(app: &mut PhotocraftApp) {
    let _ = app.run("generate.engine.stop", json!({}));
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_status_lines_include_resources() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.engine.status = Some(json!({"native":{"state":"ready","device":"Fake CPU","resident":["qwen"],"detail":"test detail"}}));
        assert_eq!(state(&app), "ready");
        let lines = native_lines(&app).join(" ");
        assert!(lines.contains("Fake CPU") && lines.contains("qwen") && lines.contains("test detail"));
        let t = Tokens::get(&egui::Context::default());
        assert!(colour("unconfigured", &t).is_none());
        assert_eq!(colour("failed", &t), Some(t.danger));
    }
}
