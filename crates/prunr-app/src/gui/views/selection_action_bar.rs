//! Selection action bar — Delete / Copy / Cut / Invert / Clear.
//! Renders below the adjustments toolbar when a selection exists.
//! Keyboard hints come from the shortcut table.

use egui::{Color32, RichText, Stroke, Ui};
use egui_material_icons::icons::{ICON_CLOSE, ICON_CONTENT_COPY, ICON_CONTENT_CUT, ICON_DELETE, ICON_FLIP};

use super::chip::{tooltip, with_fill};
use super::shortcuts::Action;
use crate::gui::theme::{
    BG_SECONDARY, BUTTON_ROUNDING, CHIP_HEIGHT, DESTRUCTIVE, FONT_SIZE_BODY, SPACE_XS,
    STROKE_DEFAULT, TEXT_PRIMARY,
};

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

/// Render the selection action bar. Returns `Some(action)` when the user
/// clicked a button; `None` otherwise. The caller wires the returned action
/// into `apply_toolbar_change`.
///
/// Layout: `[ Delete ]  [ Copy ]  [ Cut ]  |  [ Invert ]  [ Clear ]`
/// Separator between Cut and Invert divides destructive-adjacent (left group)
/// from selection-modifying (right group) per 33-UI-SPEC.md.
pub(crate) fn render_selection_action_bar(ui: &mut Ui) -> Option<SelectionAction> {
    let mut chosen = None;

    ui.add_space(SPACE_XS);

    ui.horizontal(|ui| {
        if action_button(ui, ICON_DELETE.codepoint, "Delete", Action::Delete, Some(DESTRUCTIVE)).clicked() {
            chosen = Some(SelectionAction::Delete);
        }
        if action_button(ui, ICON_CONTENT_COPY.codepoint, "Copy", Action::Copy, None).clicked() {
            chosen = Some(SelectionAction::Copy);
        }
        if action_button(ui, ICON_CONTENT_CUT.codepoint, "Cut", Action::Cut, None).clicked() {
            chosen = Some(SelectionAction::Cut);
        }

        ui.add_space(SPACE_XS);
        let sep_rect = ui.available_rect_before_wrap();
        ui.painter().line_segment(
            [sep_rect.left_top(), sep_rect.left_bottom()],
            Stroke::new(STROKE_DEFAULT, Color32::from_rgb(0x50, 0x50, 0x50)),
        );
        ui.add_space(SPACE_XS);

        if action_button(ui, ICON_FLIP.codepoint, "Invert", Action::Invert, None).clicked() {
            chosen = Some(SelectionAction::Invert);
        }
        if action_button(ui, ICON_CLOSE.codepoint, "Clear", Action::Cancel, None).clicked() {
            chosen = Some(SelectionAction::Clear);
        }
    });

    chosen
}

/// Render a single action button; `label` doubles as the tooltip title.
/// Delete uses DESTRUCTIVE border; others use the standard stroke color.
fn action_button(
    ui: &mut Ui,
    icon: &str,
    label: &'static str,
    shortcut: Action,
    border_color: Option<Color32>,
) -> egui::Response {
    let border = border_color.unwrap_or(Color32::from_rgb(0x50, 0x50, 0x50));
    let stroke = Stroke::new(STROKE_DEFAULT, border);

    let saved_padding = ui.spacing().button_padding;
    ui.spacing_mut().button_padding = egui::vec2(8.0, 4.0);
    let resp = with_fill(ui, BG_SECONDARY, |ui| {
        ui.add(
            egui::Button::new(
                RichText::new(format!("{icon}  {label}")).color(TEXT_PRIMARY).size(FONT_SIZE_BODY),
            )
            .stroke(stroke)
            .corner_radius(BUTTON_ROUNDING)
            .min_size(egui::vec2(0.0, CHIP_HEIGHT)),
        )
    });
    ui.spacing_mut().button_padding = saved_padding;

    tooltip(resp, label, "", Some(shortcut))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies the SelectionAction variants are distinct and Copy-able.
    /// The render path is egui-headless — can't drive clicks without a real
    /// Context. This test pins the API surface so renames cause compile-time
    /// breakage rather than silent behavioral drift.
    #[test]
    fn selection_action_variants_are_distinct() {
        let actions = [
            SelectionAction::Delete,
            SelectionAction::Copy,
            SelectionAction::Cut,
            SelectionAction::Invert,
            SelectionAction::Clear,
        ];
        // All 5 are distinct (ne each other).
        for i in 0..actions.len() {
            for j in 0..actions.len() {
                if i != j {
                    assert_ne!(actions[i], actions[j],
                        "SelectionAction variants must be distinct");
                }
            }
        }
        // Copy is derivable (no panic).
        let _ = actions[0];
    }
}
