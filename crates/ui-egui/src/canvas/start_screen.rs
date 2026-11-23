//! The Home screen: the framework start page with PhotoCraft's recent files, links and drop hint, and the Home rule
//! the options bar's button, the rail's Home link and `window.home` share.

use craft::behavior::IconId;
use craft::ui::contract::Bind;
use craft::ui::runtime::env::Label;
use craft::ui::shell::{StartAction, StartLink, StartScreen, Template};

use super::PhotocraftApp;

/// Recent files listed on the Home screen.
const HOME_RECENT: usize = 11;

/// The recent cards keep one size, as many to a row as fit (six on a desktop window), instead of growing to fill it:
/// a factor on the card thumbnail.
const HOME_CARD_ZOOM: f32 = 1.65;

/// The Home screen in the document area. The recent list is empty on the web, which has no paths to reopen, and with
/// Show Home Recent Files off.
pub(super) fn start_screen(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let ctx = ui.ctx().clone();
    let show_recent = !cfg!(target_arch = "wasm32") && app.session.prefs().interface.show_home_recent_files;
    let recent = if show_recent { crate::home_recent_ui::recent_files(app, HOME_RECENT) } else { Vec::new() };
    let thumbs = crate::home_recent_ui::thumbnails(&ctx);
    // The source reads the app, so the page binds copies of the view and the sort and they are written back after.
    let (mut view, mut sort) = (app.ui.chrome.home_recent_view, app.ui.chrome.home_recent_sort);
    let links = links();
    let footer = footer_links();
    // ShotoPhopEX: the Generate hub's cards as the page's templates (not on the web, which has no generative commands).
    let hub = hub_templates();
    let actions = {
        let source = crate::home_recent_ui::source(app, &ctx);
        StartScreen::new(Bind::bound(&mut view), Bind::bound(&mut sort), &recent)
            .thumbnails(&thumbs, &source)
            .links(&links[..])
            .templates(&hub[..])
            .home_layout(Label::from(crate::native_menu::APP_NAME), &footer[..])
            .max_recent(HOME_RECENT)
            .card_zoom(HOME_CARD_ZOOM)
            .drop_hint(Label::from(tl!("Drop an image or PSD anywhere to open it.")))
            .show(ui)
            .actions
    };
    app.ui.chrome.home_recent_view = view;
    app.ui.chrome.home_recent_sort = sort;
    for action in actions {
        // The recent list's own actions and Open; New is the New Document dialog's, Show in Folder the platform's.
        if let StartAction::RevealRecent(path) = &action {
            reveal(app, path);
            continue;
        }
        if crate::home_recent_ui::act(app, &action) {
            continue;
        }
        if let StartAction::UseTemplate(id) = &action {
            use_hub_card(app, &ctx, id);
            continue;
        }
        if action == StartAction::New {
            crate::new_doc_ui::open(app);
        }
    }
}

/// Shows a recent file selected in the system file manager; a failure goes to the status line, as the other open
/// errors do. The start screen offers Show in Folder only where the platform can reveal a file.
fn reveal(app: &mut PhotocraftApp, path: &std::path::Path) {
    #[cfg(not(target_arch = "wasm32"))]
    if let Err(e) = craft::platform::open::reveal(path) {
        app.ui.status = e;
        app.ui.status_error = true;
    }
    #[cfg(target_arch = "wasm32")]
    let _ = (app, path);
}

/// The navigation of the Home rail: Home (the page shown, so selected), which leaves Home for the open documents.
fn links() -> [StartLink; 1] {
    [StartLink::command(tl!("Home"), Some(IconId::new("home")), "window.home").selected(true)]
}

/// The project links, at the bottom of the Home rail. ShotoPhopEX has no website or Discord of its own: its GitHub
/// repository and the upstream project. The upstream name is a placeholder, so branding leaves it as PhotoCraft.
fn footer_links() -> [StartLink; 2] {
    let based_on = crate::i18n::fmt(tl!("Based on {upstream}"), &[("upstream", photocraft_engine::build_info::UPSTREAM_NAME)]);
    [
        StartLink::url(tl!("GitHub"), Some(IconId::new("link_layers")), crate::links::GITHUB),
        StartLink::url(based_on, Some(IconId::new("link")), crate::links::UPSTREAM),
    ]
}

/// The Generate hub's cards as start-screen templates: the card id, its label and its sentence. None on the web.
fn hub_templates() -> Vec<Template> {
    if cfg!(target_arch = "wasm32") {
        return Vec::new();
    }
    crate::generate_hub::CARDS.iter().map(|c| Template::new(c.id, tl!(c.label), tl!(c.sentence))).collect()
}

/// Runs the hub card a template click chose; a refusal goes to the notices, as the hub window's do.
fn use_hub_card(app: &mut PhotocraftApp, ctx: &egui::Context, id: &str) {
    let Some(card) = crate::generate_hub::CARDS.iter().find(|c| c.id == id) else { return };
    if let Err(e) = crate::generate_hub::activate(app, ctx, card) {
        crate::notices::error(app, e);
    }
}

/// Shows Home over the open documents, or leaves it for them: the rule of the options bar's Home button, the Home
/// rail's Home link and `window.home`. With no document open there is nothing behind Home, so it does nothing.
pub(crate) fn toggle_home(app: &mut PhotocraftApp) {
    let documents = app.session.documents().len();
    if documents == 0 {
        return;
    }
    let auto = app.session.prefs().general.auto_show_home_screen;
    let up = app.ui.chrome.shows_home(documents, auto);
    app.ui.chrome.home = if up { None } else { Some(documents) };
}

#[cfg(test)]
#[path = "start_screen/tests.rs"]
mod tests;
