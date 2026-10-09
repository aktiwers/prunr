//! Selection actions — Delete / Copy / Cut / Invert / Clear — rendered as
//! icon buttons in the toolbar's tool cluster while a selection exists.
//! Keyboard hints come from the shortcut table.

use egui::Ui;
use egui_material_icons::icons::{ICON_CLOSE, ICON_CONTENT_COPY, ICON_CONTENT_CUT, ICON_DELETE, ICON_FLIP};

use super::chip::{icon_action_button, tooltip};
use super::shortcuts::Action;
use crate::gui::theme::{DESTRUCTIVE, TEXT_PRIMARY};

/// What the action bar wants the app to do. Returned to the caller
/// (PrunrApp) which holds the BatchManager + SystemBridge + history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SelectionAction {
    Delete,
    Copy,
    Cut,
    Invert,
    Clear,
}

/// Render the five actions. The caller lays them out right-to-left, so
/// the list is in reverse visual order: Delete ends up leftmost.
/// Returns `Some(action)` when one was clicked this frame.
pub(crate) fn render_selection_actions(ui: &mut Ui) -> Option<SelectionAction> {
    let buttons = [
        (SelectionAction::Clear, ICON_CLOSE.codepoint, "Clear selection", Action::Cancel, TEXT_PRIMARY),
        (SelectionAction::Invert, ICON_FLIP.codepoint, "Invert selection", Action::Invert, TEXT_PRIMARY),
        (SelectionAction::Cut, ICON_CONTENT_CUT.codepoint, "Cut selection", Action::Cut, TEXT_PRIMARY),
        (SelectionAction::Copy, ICON_CONTENT_COPY.codepoint, "Copy selection", Action::Copy, TEXT_PRIMARY),
        (SelectionAction::Delete, ICON_DELETE.codepoint, "Delete selection", Action::Delete, DESTRUCTIVE),
    ];
    let mut chosen = None;
    for (action, icon, label, shortcut, color) in buttons {
        if tooltip(icon_action_button(ui, icon, color), label, "", Some(shortcut)).clicked() {
            chosen = Some(action);
        }
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_action_variants_are_distinct() {
        let actions = [
            SelectionAction::Delete,
            SelectionAction::Copy,
            SelectionAction::Cut,
            SelectionAction::Invert,
            SelectionAction::Clear,
        ];
        for i in 0..actions.len() {
            for j in 0..actions.len() {
                if i != j {
                    assert_ne!(actions[i], actions[j]);
                }
            }
        }
    }
}
