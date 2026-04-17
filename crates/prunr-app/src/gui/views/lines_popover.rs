//! Row 1 "Lines" control: a button that reveals a popover with two sub-pickers:
//! Output mode (Off / Edges only / Subject with outlines / Outline only) and
//! Model (DexiNed — only option today; future HED/PIDI will slot in here).
//!
//! Returns `true` when the line_mode changed, so the caller can invalidate
//! caches / textures / etc.
//!
//! View Component discipline: takes `&mut ItemSettings` — the minimal slice
//! it needs. Does NOT reach into PrunrApp.

use egui::{RichText, Ui};
use egui_material_icons::icons::*;

use crate::gui::item_settings::ItemSettings;
use crate::gui::theme;
use prunr_core::LineMode;

/// Height of the row 1 button.
const BTN_HEIGHT: f32 = 32.0;

/// Popover width.
const POPOVER_WIDTH: f32 = 300.0;

/// Human-friendly label for the currently-selected line mode.
fn mode_label(mode: LineMode) -> &'static str {
    match mode {
        LineMode::Off => "Off",
        LineMode::EdgesOnly => "Edges only",
        LineMode::SubjectOutline => "Outline only",
    }
}

/// Long-form label for the dropdown list.
fn mode_long_label(mode: LineMode) -> &'static str {
    match mode {
        LineMode::Off => "Off — no line extraction",
        LineMode::EdgesOnly => "Edges only — full image, skip BG removal",
        LineMode::SubjectOutline => "Outline only — BG removed, subject edges only",
    }
}

/// Render the row 1 Lines button + popover. Returns `true` if line_mode changed
/// (caller should invalidate edge cache and re-route pipeline).
#[allow(deprecated)]
pub fn render(ui: &mut Ui, settings: &mut ItemSettings) -> bool {
    let pop_id = egui::Id::new("lines_popover");
    let label = format!(
        "{}  Lines: {}",
        ICON_BRUSH.codepoint,
        mode_label(settings.line_mode)
    );
    let accent = settings.line_mode != LineMode::Off;

    let btn = egui::Button::new(
        RichText::new(label)
            .color(theme::TEXT_PRIMARY)
            .size(theme::FONT_SIZE_BODY),
    )
    .fill(theme::BG_SECONDARY)
    .stroke(if accent {
        egui::Stroke::new(1.0, theme::ACCENT)
    } else {
        egui::Stroke::new(1.0, egui::Color32::TRANSPARENT)
    })
    .corner_radius(theme::BUTTON_ROUNDING)
    .min_size(egui::vec2(0.0, BTN_HEIGHT));
    let resp = ui.add(btn).on_hover_text("Line extraction mode and model");

    if resp.clicked() {
        ui.memory_mut(|m| m.toggle_popup(pop_id));
    }

    let mut changed = false;
    egui::popup_below_widget(
        ui,
        pop_id,
        &resp,
        egui::PopupCloseBehavior::CloseOnClickOutside,
        |ui| {
            ui.set_min_width(POPOVER_WIDTH);

            // ── Output mode picker ──
            ui.label(
                RichText::new("Output")
                    .strong()
                    .color(theme::TEXT_PRIMARY),
            );
            ui.add_space(theme::SPACE_XS);
            for mode in [LineMode::Off, LineMode::EdgesOnly, LineMode::SubjectOutline] {
                let selected = settings.line_mode == mode;
                if ui
                    .selectable_label(selected, mode_long_label(mode))
                    .clicked()
                    && !selected
                {
                    settings.line_mode = mode;
                    changed = true;
                }
            }

            ui.add_space(theme::SPACE_SM);
            ui.separator();
            ui.add_space(theme::SPACE_SM);

            // ── Model picker ──
            // For now there's only DexiNed, but the structure is in place
            // so HED / PIDI / other edge models slot in cleanly later.
            ui.label(
                RichText::new("Model")
                    .strong()
                    .color(theme::TEXT_PRIMARY),
            );
            ui.add_space(theme::SPACE_XS);
            ui.label(
                RichText::new(format!("{}  DexiNed", ICON_NEUROLOGY.codepoint))
                    .color(theme::TEXT_SECONDARY)
                    .size(theme::FONT_SIZE_BODY),
            );
            ui.label(
                RichText::new("More models available in a future update.")
                    .color(theme::TEXT_HINT)
                    .size(theme::FONT_SIZE_MONO),
            );
        },
    );

    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_labels_cover_all_variants() {
        for mode in [LineMode::Off, LineMode::EdgesOnly, LineMode::SubjectOutline] {
            assert!(!mode_label(mode).is_empty());
            assert!(!mode_long_label(mode).is_empty());
        }
    }
}
