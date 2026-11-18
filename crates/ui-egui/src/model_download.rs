//! The model picker's download dialog (ShotoPhopEX): choosing a model whose files are missing
//! asks here first. It names the model and what the download weighs, says whether it may not fit
//! this computer and what licences it comes under (the picker's list shows none of that), then
//! Download fetches its Get AI Models pack and makes it the choice, or Cancel returns to the
//! picture with the choice unchanged.
//!
//! A model with no pack (its files are placed by hand) can still be chosen; a research model
//! that is installed but not allowed asks to allow research-only models.

use egui::RichText;
use photocraft_engine::jobs::JobId;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::model_picker::{self, Request};
use crate::theme::Tokens;
use crate::widgets;

/// The dialog's state (`UiState::model_download`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelDownloadUi {
    /// The choice waiting for an answer.
    pub request: Option<Request>,
    /// A download this dialog started: when it ends the pickers ask the server again which
    /// templates it has.
    pub job: Option<JobId>,
    pub error: Option<String>,
}

/// The `models.packs` entry that brings `pack` here: the pack itself, or its 4-bit or 8-bit
/// version where only that runs on this system (FLUX 3 Klein on a Mac).
pub fn pack_entry<'a>(info: &'a Value, pack: &str) -> Option<&'a Value> {
    let all = || std::iter::once(&info["pack"]).chain(info["extras"].as_array().into_iter().flatten());
    [pack.to_string(), format!("{pack}-q4"), format!("{pack}-q8")].into_iter().find_map(|id| all().find(|p| p["id"].as_str() == Some(id.as_str())))
}

/// The licences of the files `entry` would fetch.
fn licences(info: &Value, entry: &Value) -> Vec<Value> {
    let ids: Vec<&str> = entry["files"].as_array().into_iter().flatten().filter(|f| f["installed"] == false).filter_map(|f| f["license"].as_str()).collect();
    info["licenses"].as_array().into_iter().flatten().filter(|l| l["id"].as_str().is_some_and(|id| ids.contains(&id))).cloned().collect()
}

fn gated(entry: &Value) -> bool {
    entry["files"].as_array().into_iter().flatten().any(|f| f["gated"] == true && f["installed"] == false)
}

/// What the dialog's main button does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Go {
    /// Fetch the pack, then choose the model.
    Download,
    /// The pack needs a Hugging Face sign-in: Get AI Models takes the token.
    GetModels,
    /// Turn on Allow research-only models and choose it.
    Allow,
    /// Nothing to fetch from here: choose it (its files go in the Models folder by hand).
    Choose,
}

fn start(app: &mut PhotocraftApp, ctx: &egui::Context, r: &Request, pack: &str) {
    let p = json!({"pack": pack, "acceptLicenses": true});
    app.ui.model_download.error = None;
    match app.session.start(photocraft_engine::model_packs::GET, p) {
        Ok(photocraft_engine::jobs::Started::Job(id)) => {
            app.ui.model_download.job = Some(id);
            // Get AI Models follows the job too: its progress, its outcome, the saved folder.
            app.ui.get_models.job = Some(id);
            model_picker::confirm(ctx, &r.picker, &r.template);
            app.ui.model_download.request = None;
        }
        Ok(photocraft_engine::jobs::Started::Done(_)) => {
            model_picker::confirm(ctx, &r.picker, &r.template);
            app.ui.model_download.request = None;
            app.ui.generate.probed = false;
        }
        Err(e) => app.ui.model_download.error = Some(e.to_string()),
    }
}

pub fn window(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if let Some(r) = model_picker::take_request(ctx) {
        app.ui.model_download.request = Some(r);
        app.ui.model_download.error = None;
        crate::get_models_ui::refresh(app);
    }
    // A download ended: the pickers ask again which templates the server has.
    if let Some(id) = app.ui.model_download.job
        && app.session.job(id).is_none_or(|j| j.state != "running")
    {
        app.ui.model_download.job = None;
        app.ui.generate.probed = false;
    }
    let Some(r) = app.ui.model_download.request.clone() else { return };
    let t = Tokens::get(ctx);
    let info = app.ui.get_models.info.clone().unwrap_or(Value::Null);
    let entry = r.model.pack.and_then(|p| pack_entry(&info, p)).cloned();
    let missing = entry.as_ref().and_then(|e| e["missingBytes"].as_u64()).unwrap_or(0);
    let go = match &entry {
        _ if r.research && missing == 0 && entry.is_some() => Go::Allow,
        Some(e) if missing > 0 && gated(e) => Go::GetModels,
        Some(_) if missing > 0 => Go::Download,
        _ if r.research && !app.session.prefs().integrations.allow_research_models => Go::Allow,
        _ => Go::Choose,
    };
    let running = app.ui.get_models.job.is_some_and(|j| app.session.job(j).is_some_and(|j| j.state == "running"));
    let mut answer = None;
    egui::Modal::new(egui::Id::unique("model-download")).show(ctx, |ui| {
        ui.set_max_width(340.0);
        ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
        let title = match go {
            Go::Download | Go::GetModels => crate::i18n::fmt(tl!("Download {name}?"), &[("name", &r.label)]),
            Go::Allow | Go::Choose => crate::i18n::fmt(tl!("Use {name}?"), &[("name", &r.label)]),
        };
        ui.label(RichText::new(title).strong().size(14.0));
        if !r.model.good.is_empty() {
            ui.label(RichText::new(r.model.good).small().color(t.text_dim));
        }
        match (&entry, go) {
            (Some(e), Go::Download | Go::GetModels) => {
                ui.label(crate::i18n::fmt(
                    tl!("{pack}: {size} to download"),
                    &[("pack", e["name"].as_str().unwrap_or_default()), ("size", &crate::get_models_ui::gb(missing))],
                ));
            }
            (None, Go::Choose) => {
                ui.label(
                    RichText::new(tl!("Get AI Models doesn't fetch this one yet: put its files in the Models folder (Preferences › AI Integrations)."))
                        .small()
                        .color(t.text_dim),
                );
            }
            _ => {}
        }
        // The warnings the list leaves out.
        if r.may_not_fit {
            ui.label(RichText::new(tl!("It may need more graphics memory than this computer has: it can run slowly or fail.")).color(t.warning));
        }
        if r.research {
            ui.label(RichText::new(tl!("Research licence: fine for personal and research use, not for commercial work.")).color(t.warning));
        }
        if go == Go::GetModels {
            ui.label(RichText::new(tl!("Its maker asks you to sign in to Hugging Face first: Get AI Models takes your token.")).color(t.warning));
        }
        if let Some(e) = &entry
            && matches!(go, Go::Download | Go::GetModels)
        {
            let ls = licences(&info, e);
            if !ls.is_empty() {
                ui.add_space(2.0);
                ui.label(RichText::new(tl!("Downloading accepts these licences:")).small().color(t.text_dim));
                crate::get_models_ui::licence_rows(ui, &t, &ls);
            }
        }
        if running && go == Go::Download {
            ui.label(RichText::new(tl!("Another download is running: this one starts when you press Download again after it ends.")).small().color(t.text_dim));
        }
        if let Some(e) = &app.ui.model_download.error {
            ui.colored_label(t.danger, e);
        }
        ui.add_space(8.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let label = match go {
                Go::Download => tl!("Download"),
                Go::GetModels => tl!("Open Get AI Models"),
                Go::Allow => tl!("Allow and use"),
                Go::Choose => tl!("Use anyway"),
            };
            answer = widgets::dialog_buttons(
                ui,
                &[
                    widgets::DialogButton::new(widgets::ButtonRole::Default, label, 96.0),
                    widgets::DialogButton::new(widgets::ButtonRole::Cancel, tl!("Cancel"), 80.0),
                ],
            );
        });
    });
    match answer {
        Some(widgets::ButtonRole::Default) => match go {
            Go::Download if running => {}
            Go::Download => {
                let pack = entry.as_ref().and_then(|e| e["id"].as_str()).unwrap_or_default().to_string();
                start(app, ctx, &r, &pack);
            }
            Go::GetModels => {
                crate::get_models_ui::open(app);
                if let Some(id) = entry.as_ref().and_then(|e| e["id"].as_str()) {
                    app.ui.get_models.extras = vec![id.to_string()];
                }
                model_picker::confirm(ctx, &r.picker, &r.template);
                app.ui.model_download.request = None;
            }
            Go::Allow => {
                let _ = app.run("prefs.set", json!({"path": "integrations.allowResearchModels", "value": true}));
                model_picker::confirm(ctx, &r.picker, &r.template);
                app.ui.model_download.request = None;
            }
            Go::Choose => {
                model_picker::confirm(ctx, &r.picker, &r.template);
                app.ui.model_download.request = None;
            }
        },
        Some(_) => app.ui.model_download.request = None,
        None => {}
    }
}

/// `ui.inspect.modelDownload`: the pending choice, if any.
pub fn inspect(app: &PhotocraftApp) -> Value {
    match &app.ui.model_download.request {
        Some(r) => json!({"open": true, "model": r.model.key, "template": r.template, "picker": r.picker, "research": r.research, "mayNotFit": r.may_not_fit}),
        None => json!({"open": false}),
    }
}

#[cfg(test)]
#[path = "model_download_tests.rs"]
mod tests;
