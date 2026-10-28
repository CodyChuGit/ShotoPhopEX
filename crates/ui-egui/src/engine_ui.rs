//! The generative engines' state in the status bar (plan-native-engine §5.4, §10): a dot coloured
//! by `generate.engine.status` (grey stopped, amber starting, green ready or working, blue
//! running outside the app, red failed, hollow when nothing is configured), its tooltip, and a
//! menu with a section per engine: the native worker (Start / Stop / Purge models / Show log)
//! and the app-managed ComfyUI (Start / Stop / Restart / Show log). The dot follows the engine
//! generative commands go to: the native worker under `native`, ComfyUI under `comfy`, and under
//! `auto` the native worker once it runs, ComfyUI before. The commands do the work; this module
//! polls the status every [`POLL_EVERY`] seconds and draws. Preferences › AI Integrations shows
//! the same lines and buttons ([`native_lines`], [`native_action`]).
//!
//! Start at launch and stop on quit follow Preferences › AI Integrations
//! (`generate_engine_cmds::start_on_launch` / `stop_on_quit`).

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

/// The ComfyUI engine's state name (`unconfigured|stopped|starting|ready|external|failed`),
/// `stopped` before the first poll.
pub fn state(app: &PhotocraftApp) -> String {
    app.ui.engine.status.as_ref().and_then(|v| v["state"].as_str()).unwrap_or("stopped").to_string()
}

/// The native worker's state name (`off|unconfigured|stopped|starting|running|ready|failed`),
/// `off` before the first poll.
pub fn native_state(app: &PhotocraftApp) -> String {
    native(app).and_then(|v| v["state"].as_str()).unwrap_or("off").to_string()
}

fn native(app: &PhotocraftApp) -> Option<&Value> {
    app.ui.engine.status.as_ref().map(|v| &v["native"]).filter(|v| v.is_object())
}

/// Whether the dot follows the native worker: under `native` always, under `auto` once it runs.
fn follows_native(app: &PhotocraftApp) -> bool {
    let st = native_state(app);
    let mode = native(app).and_then(|v| v["mode"].as_str()).unwrap_or("auto");
    match st.as_str() {
        "off" => false,
        "unconfigured" | "stopped" => mode == "native",
        _ => true,
    }
}

/// The tooltip's first line for a ComfyUI state.
pub fn label(state: &str) -> String {
    match state {
        "unconfigured" => tl!("Engine: not configured"),
        "starting" => tl!("Engine: starting…"),
        "ready" => tl!("Engine: ready"),
        "external" => tl!("Engine: running outside the app"),
        "failed" => tl!("Engine: failed"),
        _ => tl!("Engine: stopped"),
    }
    .to_string()
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
        "external" => t.accent,
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
    if st == "off" {
        return Vec::new();
    }
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
    let starting = state(app) == "starting" || native_state(app) == "starting";
    let every = if starting { 1.0 } else { POLL_EVERY };
    if app.ui.engine.status.is_some() && now - app.ui.engine.polled < every {
        return;
    }
    app.ui.engine.polled = now;
    // Straight to the engine: a poll must not touch the status line or the journal.
    app.ui.engine.status = app.session.execute("generate.engine.status", json!({})).ok();
    if state(app) == "starting" || native_state(app) == "starting" {
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
    match app.run(cmd, json!({"engine": "native"})) {
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
    let st = state(app);
    let nst = native_state(app);
    let shown = if follows_native(app) { nst.as_str() } else { st.as_str() };
    let detail = app.ui.engine.status.as_ref().and_then(|v| v["detail"].as_str()).unwrap_or("").to_string();
    let (r, resp) = ui.allocate_exact_size(vec2(16.0, 16.0), Sense::click());
    if resp.hovered() {
        ui.painter().rect_filled(r, t.radius_sm, t.hover);
    }
    match colour(shown, &t) {
        Some(c) => {
            ui.painter().circle_filled(r.center(), 4.0, c);
        }
        None => {
            ui.painter().circle_stroke(r.center(), 4.0, egui::Stroke::new(1.0, t.text_faint));
        }
    }
    let mut tip = native_lines(app).join("\n");
    if !tip.is_empty() {
        tip.push_str("\n\n");
    }
    tip.push_str(&label(&st));
    if st == "unconfigured" {
        tip.push('\n');
        tip.push_str(tl!("Set the ComfyUI folder in Preferences › AI Integrations to let the app start it"));
    } else if !detail.is_empty() {
        tip.push('\n');
        tip.push_str(&detail);
    }
    let resp = resp.on_hover_text(tip);
    egui::Popup::menu(&resp).show(|ui| {
        ui.set_min_width(180.0);
        if nst != "off" && nst != "unconfigured" {
            ui.label(egui::RichText::new(tl!("Native engine")).color(t.text_dim));
            if let Some(a) = native_buttons(ui, &nst) {
                native_action(app, a);
                ui.close();
            }
            ui.separator();
            ui.label(egui::RichText::new("ComfyUI").color(t.text_dim));
        }
        let configured = st != "unconfigured";
        let running = matches!(st.as_str(), "starting" | "ready");
        let mut cmd = None;
        if ui.add_enabled(configured && !running && st != "external", egui::Button::new(tl!("Start engine"))).clicked() {
            cmd = Some("generate.engine.start");
        }
        if ui.add_enabled(running, egui::Button::new(tl!("Stop engine"))).clicked() {
            cmd = Some("generate.engine.stop");
        }
        if ui.add_enabled(configured && st != "external", egui::Button::new(tl!("Restart engine"))).clicked() {
            cmd = Some("generate.engine.restart");
        }
        ui.separator();
        if ui.button(tl!("Show log")).clicked() {
            show_log(app);
            ui.close();
        }
        if let Some(c) = cmd {
            match app.run(c, json!({})) {
                Ok(v) => app.ui.engine.status = Some(v),
                Err(e) => crate::notices::error(app, e),
            }
            ui.close();
        }
    });
}

/// Reveal the ComfyUI engine's log in the file manager.
fn show_log(app: &mut PhotocraftApp) {
    let path = app.ui.engine.status.as_ref().and_then(|v| v["log"].as_str()).map(str::to_string).unwrap_or_default();
    reveal(app, &path);
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

/// At launch: start the engine when the preference says so.
pub fn on_launch(app: &mut PhotocraftApp) {
    if photocraft_engine::generate_engine_cmds::start_on_launch(&app.session) {
        app.ui.engine.status = app.run("generate.engine.start", json!({})).ok();
    }
}

/// At quit: stop the engine the app started when the preference says so (a server started by
/// hand is left alone by the command).
pub fn on_quit(app: &mut PhotocraftApp) {
    if photocraft_engine::generate_engine_cmds::stop_on_quit(&app.session) {
        let _ = app.run("generate.engine.stop", json!({}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_and_colours_cover_every_state() {
        let t = Tokens::get(&egui::Context::default());
        for s in ["unconfigured", "stopped", "starting", "ready", "external", "failed"] {
            assert!(label(s).starts_with("Engine"), "{s}");
        }
        for s in ["unconfigured", "stopped", "starting", "running", "ready", "failed"] {
            assert!(native_label(s).starts_with("Native engine"), "{s}");
        }
        assert!(colour("unconfigured", &t).is_none());
        assert_eq!(colour("failed", &t), Some(t.danger));
        assert_eq!(colour("ready", &t), Some(Color32::from_rgb(0x4c, 0xaf, 0x50)));
        assert_eq!(colour("running", &t), colour("ready", &t));
    }

    /// The poll stores `generate.engine.status` (unconfigured by default) and start-at-launch is
    /// a no-op without a folder.
    #[test]
    fn the_status_bar_polls_the_engine() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        on_launch(&mut app);
        assert!(app.ui.engine.status.is_none());
        tick(&mut app, &ctx);
        assert_eq!(state(&app), "unconfigured");
        assert_eq!(native_state(&app), "unconfigured");
        assert!(follows_native(&app), "native by default (2026-10-28): the worker's dot, even unconfigured");
        on_quit(&mut app);
    }

    /// The native lines name the card, the memory, what is loaded and the engine's reason; the
    /// dot follows the worker once it runs under `auto`.
    #[test]
    fn the_native_lines_say_what_the_worker_holds() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.engine.status = Some(json!({"state": "ready", "native": {
            "state": "ready", "mode": "auto", "device": "NVIDIA GeForce RTX 5090",
            "vramFree": 12u64 << 30, "vramTotal": 32u64 << 30, "resident": ["krea2-turbo"],
            "detail": "SAM 3.1: no runtime"
        }}));
        let lines = native_lines(&app);
        assert_eq!(lines[0], native_label("ready"));
        assert!(lines.iter().any(|l| l.contains("RTX 5090")), "{lines:?}");
        assert!(lines.iter().any(|l| l.contains("12.0G") && l.contains("32.0G")), "{lines:?}");
        assert!(lines.iter().any(|l| l.contains("krea2-turbo")), "{lines:?}");
        assert!(lines.iter().any(|l| l.contains("SAM 3.1")), "{lines:?}");
        assert!(follows_native(&app));
        app.ui.engine.status = Some(json!({"state": "ready", "native": {"state": "stopped", "mode": "auto"}}));
        assert!(!follows_native(&app), "auto with the worker not started yet: ComfyUI");
        app.ui.engine.status = Some(json!({"state": "ready", "native": {"state": "stopped", "mode": "native"}}));
        assert!(follows_native(&app));
        app.ui.engine.status = Some(json!({"state": "ready", "native": {"state": "off", "mode": "comfy"}}));
        assert!(native_lines(&app).is_empty() && !follows_native(&app));
    }
}
