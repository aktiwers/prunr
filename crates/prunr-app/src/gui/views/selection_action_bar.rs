//! Selection actions — Delete / Copy / Cut / Invert / Clear, and Restore on a cut-out — rendered as
//! icon buttons in the toolbar's tool cluster while a selection exists.
//! Keyboard hints come from the shortcut table.

use egui::Ui;
use egui_material_icons::icons::{ICON_CLOSE, ICON_CONTENT_COPY, ICON_CONTENT_CUT, ICON_DELETE, ICON_FLIP, ICON_SETTINGS_BACKUP_RESTORE};

use super::chip::{icon_action_button, tooltip};
use super::shortcuts::Action;
use crate::gui::theme::{DESTRUCTIVE, TEXT_PRIMARY};

/// What the action bar wants the app to do. Returned to the caller
/// (PrunrApp) which holds the BatchManager + SystemBridge + history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SelectionAction {
    /// On a cut-out: bring the selected area back.
    Restore,
    Delete,
    Copy,
    Cut,
    Invert,
    Clear,
}

/// Render the actions; on a `cutout` Restore joins them. The caller lays
/// them out right-to-left, so the list is in reverse visual order: Delete
/// (and Restore) end up leftmost. Returns `Some(action)` when one was
/// clicked this frame.
pub(crate) fn render_selection_actions(ui: &mut Ui, cutout: bool) -> Option<SelectionAction> {
    let buttons = [
        (SelectionAction::Clear, ICON_CLOSE.codepoint, "Clear selection", Some(Action::Cancel), TEXT_PRIMARY),
        (SelectionAction::Invert, ICON_FLIP.codepoint, "Invert selection", Some(Action::Invert), TEXT_PRIMARY),
        (SelectionAction::Cut, ICON_CONTENT_CUT.codepoint, "Cut selection", Some(Action::Cut), TEXT_PRIMARY),
        (SelectionAction::Copy, ICON_CONTENT_COPY.codepoint, "Copy selection", Some(Action::Copy), TEXT_PRIMARY),
        (SelectionAction::Delete, ICON_DELETE.codepoint, "Delete selection", Some(Action::Delete), DESTRUCTIVE),
    ];
    let restore = (SelectionAction::Restore, ICON_SETTINGS_BACKUP_RESTORE.codepoint, "Restore selection", None, TEXT_PRIMARY);
    let mut chosen = None;
    for (action, icon, label, shortcut, color) in buttons.into_iter().chain(cutout.then_some(restore)) {
        if tooltip(icon_action_button(ui, icon, color), label, "", shortcut).clicked() {
            chosen = Some(action);
        }
    }
    chosen
}
