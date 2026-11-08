//! Help › Get AI Models…: the memory profile's model pack (`models.packs`), one Download button
//! (`models.get`, a background job), its progress, and Cancel. Downloads resume after a cancel
//! or a crash; a Hugging Face token is asked for only when a file needs it and is never saved.
//! The licences of the files a run would fetch are listed with their links, and Download stays
//! off until they are accepted (`models.get` refuses without `acceptLicenses`).

use egui::RichText;
use photocraft_engine::jobs::JobId;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

pub const COMMAND: &str = "generate.getModels";

/// The window's state (`UiState::get_models`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GetModelsUi {
    pub open: bool,
    /// `models.packs`' answer, refreshed when the window opens and after a run.
    pub info: Option<Value>,
    /// Extras ticked.
    pub extras: Vec<String>,
    /// Typed for gated files; kept in memory only.
    pub token: String,
    /// The licences shown were accepted (reset each time the window opens).
    pub accepted: bool,
    pub job: Option<JobId>,
    pub error: Option<String>,
    pub done: Option<String>,
}

fn refresh(app: &mut PhotocraftApp) {
    match app.session.execute(photocraft_engine::model_packs::PACKS, json!({})) {
        Ok(v) => app.ui.get_models.info = Some(v),
        Err(e) => app.ui.get_models.error = Some(e.to_string()),
    }
}

/// Open the window (the menu item, the Generate panel's first-run note, the memory notice).
pub fn open(app: &mut PhotocraftApp) {
    app.ui.get_models.open = true;
    app.ui.get_models.error = None;
    app.ui.get_models.accepted = false;
    if app.ui.get_models.job.is_none() {
        app.ui.get_models.done = None;
    }
    refresh(app);
}

/// The generative commands that need models on this computer: from the UI, each opens this
/// window instead while there are none (see [`intercept`]).
const NEEDS_MODELS: &[&str] = &[
    "generate.fill",
    "generate.image",
    "generate.expand",
    "generate.edit",
    "generate.similar",
    "generate.removeBackground",
    "generate.splitLayers",
    "generate.clownPass",
    "generate.angles",
    "generate.upscale",
    "select.byText",
];

/// Whether this computer has no AI models yet: no models folder chosen or found (a fresh
/// install; Get AI Models sets the folder it downloads into). Any folder counts as having
/// models, so the window never nags someone who brought their own.
pub fn has_no_models(app: &PhotocraftApp) -> bool {
    app.session.prefs().integrations.models_root().is_empty()
}

/// A click on a generative tool (menu, task bar, toolbar, Generate panel) before any models are
/// downloaded opens Get AI Models instead of failing. Automation keeps the command's own error.
pub fn intercept(app: &mut PhotocraftApp, id: &str) -> bool {
    if app.automation_input || !NEEDS_MODELS.contains(&id) || !has_no_models(app) {
        return false;
    }
    open(app);
    true
}

pub fn menu(app: &mut PhotocraftApp, id: &str, _params: &Value) -> Option<Result<Value, String>> {
    (id == COMMAND).then(|| {
        open(app);
        Ok(json!({"getModels": true}))
    })
}

fn gb(b: u64) -> String {
    if b >= 1_000_000_000 {
        format!("{:.1} GB", b as f64 / 1e9)
    } else if b >= 1_000_000 {
        format!("{} MB", (b as f64 / 1e6).round())
    } else {
        format!("{} KB", (b as f64 / 1e3).ceil())
    }
}

fn start(app: &mut PhotocraftApp) {
    let g = &app.ui.get_models;
    let pack = g.info.as_ref().and_then(|i| i["pack"]["id"].as_str()).map(str::to_string);
    let mut p = json!({"pack": pack, "extras": g.extras, "acceptLicenses": g.accepted});
    if !g.token.trim().is_empty() {
        p["token"] = json!(g.token.trim());
    }
    app.ui.get_models.error = None;
    app.ui.get_models.done = None;
    match app.session.start(photocraft_engine::model_packs::GET, p) {
        Ok(photocraft_engine::jobs::Started::Job(id)) => app.ui.get_models.job = Some(id),
        Ok(photocraft_engine::jobs::Started::Done(_)) => {
            app.ui.get_models.done = Some(tl!("Everything is installed.").to_string());
            refresh(app);
        }
        Err(e) => app.ui.get_models.error = Some(e.to_string()),
    }
}

/// Follow the job: when it ends, show the outcome and refresh the list.
fn poll(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some(id) = app.ui.get_models.job else { return };
    // A job that has ended leaves the running list on the frame it ends (the status bar may
    // collect it first), so look in the recent ones too.
    let Some(info) = app.session.job(id).or_else(|| app.session.jobs_with_recent().into_iter().rev().find(|j| j.id == id)) else {
        app.ui.get_models.job = None;
        refresh(app);
        return;
    };
    match info.state {
        "running" => ctx.request_repaint_after(std::time::Duration::from_millis(250)),
        "done" => {
            app.ui.get_models.job = None;
            app.ui.get_models.token.clear();
            let n = info.result.as_ref().and_then(|r| r["downloaded"].as_array().map(Vec::len)).unwrap_or(0);
            let runtime = info.result.as_ref().is_some_and(|r| r["runtimePath"].is_string());
            app.ui.get_models.done = Some(match (n, runtime) {
                (0, true) => tl!("Done: NVIDIA's CUDA libraries installed.").to_string(),
                (n, true) => crate::i18n::fmt(tl!("Done: {n} files and NVIDIA's CUDA libraries downloaded and checked."), &[("n", &n.to_string())]),
                (n, false) => crate::i18n::fmt(tl!("Done: {n} files downloaded and checked."), &[("n", &n.to_string())]),
            });
            if let Err(e) = crate::prefs_ui::save_preferences(app) {
                log::warn!("couldn't save the Models folder: {e}");
            }
            refresh(app);
        }
        "cancelled" => {
            app.ui.get_models.job = None;
            app.ui.get_models.done = Some(tl!("Stopped. Download again to pick up where it left off.").to_string());
            refresh(app);
        }
        _ => {
            app.ui.get_models.job = None;
            app.ui.get_models.error = Some(info.error.unwrap_or_else(|| tl!("The download failed.").to_string()));
            refresh(app);
        }
    }
}

/// The licences of what a run would fetch: the missing files of the pack and the ticked extras,
/// and NVIDIA's libraries when they are needed.
fn licences_to_accept(info: &Value, extras: &[String]) -> Vec<Value> {
    let mut ids: Vec<&str> = info["pack"]["files"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(
            info["extras"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|e| extras.iter().any(|x| Some(x.as_str()) == e["id"].as_str()))
                .flat_map(|e| e["files"].as_array().into_iter().flatten()),
        )
        .filter(|f| f["installed"] == false)
        .filter_map(|f| f["license"].as_str())
        .collect();
    if info["runtime"]["needed"] == true
        && let Some(l) = info["runtime"]["license"].as_str()
    {
        ids.push(l);
    }
    info["licenses"].as_array().into_iter().flatten().filter(|l| l["id"].as_str().is_some_and(|id| ids.contains(&id))).cloned().collect()
}

fn licence_rows(ui: &mut egui::Ui, t: &Tokens, licences: &[Value]) {
    for l in licences {
        let commercial = match l["commercial"].as_str() {
            Some("yes") => tl!("commercial use allowed"),
            Some("conditional") => tl!("commercial use with conditions"),
            Some("no") => tl!("no commercial use"),
            _ => tl!("terms unclear"),
        };
        ui.horizontal_wrapped(|ui| {
            ui.hyperlink_to(l["name"].as_str().unwrap_or_default(), l["url"].as_str().unwrap_or_default());
            let color = if l["commercial"] == "yes" { t.text_dim } else { t.warning };
            ui.label(RichText::new(commercial).small().color(color));
        })
        .response
        .on_hover_text(l["summary"].as_str().unwrap_or_default());
    }
}

fn file_rows(ui: &mut egui::Ui, t: &Tokens, files: &[Value]) {
    for f in files {
        ui.horizontal(|ui| {
            let installed = f["installed"].as_bool().unwrap_or(false);
            let (mark, color) = if installed { ("✓", t.accent) } else { ("•", t.text_dim) };
            ui.label(RichText::new(mark).color(color));
            ui.label(f["name"].as_str().unwrap_or_default());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(gb(f["bytes"].as_u64().unwrap_or(0))).small().color(t.text_dim));
                if f["gated"].as_bool().unwrap_or(false) && !installed {
                    ui.label(RichText::new(tl!("needs sign-in")).small().color(t.warning));
                }
            });
        });
    }
}

pub fn window(app: &mut PhotocraftApp, ctx: &egui::Context) {
    poll(app, ctx);
    if !app.ui.get_models.open {
        return;
    }
    let t = Tokens::get(ctx);
    let Some(info) = app.ui.get_models.info.clone() else {
        app.ui.get_models.open = false;
        return;
    };
    let job = app.ui.get_models.job.and_then(|id| app.session.job(id));
    let running = job.as_ref().is_some_and(|j| j.state == "running");
    let mut open = true;
    let mut go = false;
    let mut cancel = false;
    let mut extras = app.ui.get_models.extras.clone();
    let mut token = app.ui.get_models.token.clone();
    let mut accepted = app.ui.get_models.accepted;
    let licences = licences_to_accept(&info, &extras);
    let pack = &info["pack"];
    let pack_files = pack["files"].as_array().cloned().unwrap_or_default();
    let any_gated = pack_files
        .iter()
        .chain(
            info["extras"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|e| extras.iter().any(|x| Some(x.as_str()) == e["id"].as_str()))
                .flat_map(|e| e["files"].as_array().into_iter().flatten()),
        )
        .any(|f| f["gated"] == true && f["installed"] == false);
    egui::Window::new(tl!("Get AI Models")).id(egui::Id::unique("get-models")).open(&mut open).resizable(false).default_width(420.0).show(ctx, |ui| {
        ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
        let profile = info["profileLabel"].as_str().map_or_else(|| tl!("not detected").to_string(), str::to_string);
        ui.label(
            RichText::new(crate::i18n::fmt(
                tl!("{pack} for this computer's memory profile ({profile})"),
                &[("pack", pack["name"].as_str().unwrap_or_default()), ("profile", &profile)],
            ))
            .strong(),
        );
        if let Some(d) = pack["description"].as_str().filter(|d| !d.is_empty()) {
            ui.label(RichText::new(d).small().color(t.text_dim));
        }
        file_rows(ui, &t, &pack_files);
        if let Some(list) = info["extras"].as_array().filter(|a| !a.is_empty()) {
            ui.add_space(4.0);
            ui.label(RichText::new(tl!("Extras")).color(t.text_dim));
            for e in list {
                let id = e["id"].as_str().unwrap_or_default().to_string();
                let mut on = extras.contains(&id);
                let missing = e["missingBytes"].as_u64().unwrap_or(0);
                let label = if missing == 0 {
                    format!("{} ✓", e["name"].as_str().unwrap_or_default())
                } else {
                    format!("{} ({})", e["name"].as_str().unwrap_or_default(), gb(missing))
                };
                ui.add_enabled_ui(!running, |ui| crate::widgets::checkbox(ui, &mut on, &label))
                    .inner
                    .on_hover_text(e["description"].as_str().unwrap_or_default());
                if on && !extras.contains(&id) {
                    extras.push(id);
                } else if !on {
                    extras.retain(|x| *x != id);
                }
            }
        }
        if info["runtime"]["needed"] == true {
            ui.label(
                RichText::new(crate::i18n::fmt(
                    tl!("Also NVIDIA's CUDA libraries ({size}), since no runtime folder is set."),
                    &[("size", &gb(info["runtime"]["bytes"].as_u64().unwrap_or(0)))],
                ))
                .small()
                .color(t.text_dim),
            );
        }
        if any_gated {
            ui.horizontal(|ui| {
                ui.label(tl!("Hugging Face token"));
                ui.add_enabled(!running, egui::TextEdit::singleline(&mut token).password(true).hint_text("hf_…").desired_width(200.0));
            });
            ui.label(
                RichText::new(tl!(
                    "Some files need a free Hugging Face account and the model's licence accepted on its page. The token is used once and not saved."
                ))
                .small()
                .color(t.text_dim),
            );
        }
        if !licences.is_empty() {
            ui.add_space(4.0);
            ui.label(RichText::new(tl!("Licences")).color(t.text_dim));
            licence_rows(ui, &t, &licences);
            ui.label(
                RichText::new(tl!(
                    "Model weights are not part of ShotoPhopEX and are not covered by its licence. Each file is downloaded from its publisher's page and is yours to use only under its own licence. This list is a summary, not legal advice."
                ))
                .small()
                .color(t.text_dim),
            );
            ui.add_enabled_ui(!running, |ui| crate::widgets::checkbox(ui, &mut accepted, tl!("I have read these licences and accept them for the files I download")));
        }
        let root = info["modelsRoot"].as_str().unwrap_or_default();
        ui.label(RichText::new(crate::i18n::fmt(tl!("Saved to {folder}"), &[("folder", root)])).small().color(t.text_dim));
        ui.separator();
        let extra_missing: u64 = info["extras"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|e| extras.iter().any(|x| Some(x.as_str()) == e["id"].as_str()))
            .filter_map(|e| e["missingBytes"].as_u64())
            .sum();
        let runtime_missing = if info["runtime"]["needed"] == true { info["runtime"]["bytes"].as_u64().unwrap_or(0) } else { 0 };
        let missing = pack["missingBytes"].as_u64().unwrap_or(0) + extra_missing + runtime_missing;
        if let Some(j) = job.as_ref().filter(|_| running) {
            ui.add(egui::ProgressBar::new(j.progress).show_percentage().desired_width(400.0));
            ui.label(RichText::new(&j.message).small().color(t.text_dim));
            if crate::widgets::secondary_button(ui, tl!("Cancel"), 90.0).clicked() {
                cancel = true;
            }
        } else if missing == 0 {
            ui.label(RichText::new(tl!("Everything for this profile is installed.")).color(t.accent));
        } else {
            let label = crate::i18n::fmt(tl!("Download {size}"), &[("size", &gb(missing))]);
            let ready = accepted || licences.is_empty();
            if ui.add_enabled_ui(ready, |ui| crate::widgets::primary_button(ui, &label, 140.0)).inner.clicked() && ready {
                go = true;
            }
        }
        if let Some(d) = &app.ui.get_models.done {
            ui.label(RichText::new(d).small().color(t.accent));
        }
        if let Some(e) = &app.ui.get_models.error {
            ui.label(RichText::new(e).small().color(t.warning));
        }
    });
    app.ui.get_models.extras = extras;
    app.ui.get_models.token = token;
    app.ui.get_models.accepted = accepted;
    // Closing the window leaves a running download going (the status bar shows it).
    app.ui.get_models.open = open;
    if cancel && let Some(id) = app.ui.get_models.job {
        app.session.cancel_job(id);
    }
    if go {
        start(app);
    }
}

/// `ui.inspect.getModels`.
pub fn inspect(app: &PhotocraftApp) -> Value {
    let g = &app.ui.get_models;
    json!({"open": g.open, "job": g.job.map(|j| j.0), "error": g.error, "done": g.done, "extras": g.extras, "accepted": g.accepted})
}
