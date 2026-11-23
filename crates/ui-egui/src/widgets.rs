//! ShotoPhopEX: the plain-egui widgets the generative panels, bars and dialogs draw with. Upstream
//! moved its UI onto the craft UI framework in PhotoCraft 0.24; these keep the 0.23 look for our
//! panels until they are rebuilt on craft's components.

use egui::{Color32, Pos2, Rect, Response, Sense, Stroke, StrokeKind, Ui, Vec2, vec2};

use crate::theme::{self, Tokens};

mod buttons;
mod fields;
mod panel;

pub use buttons::{ButtonRole, DialogButton, checkbox, dialog_buttons, primary_button, secondary_button};
pub use fields::{checker, dropdown, elided_label, progress_bar, value_field, value_field_named, value_field_prec};
pub use panel::{hairline, section_label, surface};
