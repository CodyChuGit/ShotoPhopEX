//! The Retouch options bar's selection family: Quick Selection and Object Selection.

use craft::behavior::IconId;
use craft::ui::buttons::{Button, IconButton};
use craft::ui::choice::Segmented;
use craft::ui::contract::{Bind, Entry, Props as _, Show as _, Trigger, clicked};
use craft::ui::display::Divider;
use craft::ui::menu::Menu;
use craft::ui::runtime::access::PopupKind;
use craft::ui::text::{RangeAlert, TextField};

use super::heal::{check, choose};
use crate::PhotocraftApp;
use crate::option_cells::{hint, number};
use crate::state::ObjectFinderMode;

/// The Select Subject button both selection tools carry; `empty` is the message when nothing is found.
/// ShotoPhopEX: SAM 3.1 (or the High Quality matte) when a server answers, with the Fast / High
/// Quality choice beside it.
fn select_subject(ui: &mut egui::Ui, app: &mut PhotocraftApp, empty: &str) {
    let qualities = [Entry::new(false, tl!("Fast")), Entry::new(true, tl!("High Quality"))];
    choose(ui, tl!("Select Subject quality"), &mut app.ui.tool_options.subject_hq, &qualities, true, None);
    if clicked(&Button::new(tl!("Select Subject")).show(ui)) {
        let r = crate::retouch_ui::select_subject(app);
        crate::retouch_ui::report_smart_result(app, r, empty);
    }
}

/// Quick Selection's options.
pub(super) fn quick_selection(ui: &mut egui::Ui, app: &mut PhotocraftApp) -> Option<RangeAlert> {
    let o = &mut app.ui.tool_options;
    check(ui, tl!("Sample All Layers"), &mut o.sample_all.quick_selection, true, None);
    check(ui, tl!("Enhance Edge"), &mut o.enhance_edge, true, None);
    // ShotoPhopEX: with a generative server the stroke goes to SAM 3.1 (`quick_select_detector`).
    if crate::retouch_ui::server_configured(app) {
        hint(ui, tl!("Paint over an object to select it"));
    }
    hint(ui, crate::i18n::fmt(tl!("{key} to subtract"), &[("key", &crate::shortcuts::pretty("Alt"))]));
    Divider::vertical().show(ui);
    select_subject(ui, app, tl!("No subject found — try Object Selection or the Lasso"));
    None
}

/// A row of the Object Selection gear menu.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum GearOption {
    Finder,
    Subtract,
    Auto,
    Manual,
}

/// The Refresh button and the gear menu after the operation segments (#12857): Object Finder,
/// Object Subtract (stored only) and Object Finder Mode (Auto or Manual Refresh).
fn finder_buttons(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let refresh = IconButton::new(IconId::new("rotate_cw")).label(tl!("Refresh").to_string()).tooltip(tl!("Click to refresh object finder.")).show(ui);
    let label = tl!("Set additional options").to_string();
    let button = IconButton::new(IconId::new("settings")).label(label.clone()).tooltip(tl!("Set additional options")).popup(PopupKind::Menu).show(ui);
    let o = &app.ui.tool_options;
    let auto = o.object_finder_mode == ObjectFinderMode::Auto;
    let entries = [
        Entry::new(GearOption::Finder, tl!("Object Finder").to_string()).checked(o.object_finder),
        Entry::new(GearOption::Subtract, tl!("Object Subtract").to_string()).checked(o.object_subtract),
        Entry::new(GearOption::Auto, tl!("Auto Refresh").to_string()).checked(auto),
        Entry::new(GearOption::Manual, tl!("Manual Refresh").to_string()).checked(!auto),
    ];
    let picked = Menu::new(label).opened_by(&button, Trigger::Press).pick(ui, &entries).picked;
    let o = &mut app.ui.tool_options;
    match picked {
        Some(GearOption::Finder) => o.object_finder = !o.object_finder,
        Some(GearOption::Subtract) => o.object_subtract = !o.object_subtract,
        Some(GearOption::Auto) => o.object_finder_mode = ObjectFinderMode::Auto,
        Some(GearOption::Manual) => o.object_finder_mode = ObjectFinderMode::Manual,
        None => {}
    }
    if clicked(&refresh) {
        crate::object_finder::refresh(app);
    }
}

/// Object Selection's options: the selection operation (#12853), Refresh and the gear menu,
/// Sample All Layers and Hard Edge.
pub(super) fn object_selection(ui: &mut egui::Ui, app: &mut PhotocraftApp) -> Option<RangeAlert> {
    crate::panels::options_bar::selection::operation_segments(app, ui, true);
    Divider::vertical().show(ui);
    finder_buttons(app, ui);
    let o = &mut app.ui.tool_options;
    check(ui, tl!("Sample All Layers"), &mut o.sample_all.object_selection, true, None);
    check(ui, tl!("Hard Edge"), &mut o.object_hard_edge, true, None);
    if crate::retouch_ui::server_configured(app) {
        Divider::vertical().show(ui);
        detector_options(ui, app);
    }
    Divider::vertical().show(ui);
    select_subject(ui, app, tl!("No subject found — try a rectangle around the object"));
    None
}

/// ShotoPhopEX: Object Selection's detector controls, shown when a generative server is set up. A
/// click or a box selects through SAM 3.1; Refine adds spots to the last object (Alt excludes);
/// Parts takes the piece under the pointer; Next Match cycles what a click could mean; Widen,
/// Fill Holes and Soften are the inpainting workflows' mask stack; the text field selects by phrase.
fn detector_options(ui: &mut egui::Ui, app: &mut PhotocraftApp) {
    let o = &mut app.ui.tool_options;
    let modes = [Entry::new(false, tl!("New")), Entry::new(true, tl!("Refine"))];
    if let Some(on) = Segmented::new(tl!("Detector mode"), Bind::given(o.object_refine), &modes).show(ui).value().copied() {
        o.object_refine = on;
    }
    check(
        ui,
        tl!("Parts"),
        &mut o.object_parts,
        true,
        Some(tl!("Select a part of what is under the pointer (a piece of clothing) instead of the whole figure")),
    );
    let next = clicked(
        &Button::new(tl!("Next Match"))
            .tooltip(tl!("Swap the last click's object for the next thing it could mean (the whole, a part, a smaller part)"))
            .show(ui),
    );
    check(ui, tl!("Refine edge"), &mut o.refine_edge, true, Some(tl!("Soften the detector's hard edge from the pixels (hair, fur)")));
    let _ = number(tl!("Widen"), &mut o.object_widen, -64.0..=256.0)
        .unit("px")
        .whole()
        .tooltip(tl!("Widen the object's mask by this many pixels (negative shrinks it)"))
        .show(ui);
    check(ui, tl!("Fill Holes"), &mut o.object_fill_holes, true, Some(tl!("Fill the gaps the object's mask encloses")));
    let _ = number(tl!("Soften"), &mut o.object_soften, 0.0..=64.0).unit("px").whole().tooltip(tl!("Feather the mask's edge by this many pixels")).show(ui);
    let typed = TextField::new(tl!("Select by text"), Bind::bound(&mut o.select_text)).placeholder(tl!("Select by text…")).width_chars(16).submits().show(ui);
    let by_text = typed.submitted && !o.select_text.trim().is_empty();
    hint(ui, if o.object_refine { tl!("Click to include a spot, Alt-click to exclude one") } else { tl!("Click an object, or drag a box around it") });
    if next {
        crate::object_select::next_match(app);
    } else if by_text {
        let _ = crate::retouch_ui::select_by_text(app);
    }
}
