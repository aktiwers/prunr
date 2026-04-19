//! Row 3 "Sketch" control: a button that reveals a popover with two sub-pickers:
//! Output mode (Off / Full / Subject) and Model (DexiNed — only option today;
//! future HED/PIDI will slot in here).
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

/// Popover width.
const POPOVER_WIDTH: f32 = 300.0;

/// Human-friendly label for the currently-selected line mode.
fn mode_label(mode: LineMode) -> &'static str {
    match mode {
        LineMode::Off => "Off",
        LineMode::EdgesOnly => "Full",
        LineMode::SubjectOutline => "Subject",
    }
}

/// Short description for the dropdown list (shown beneath the title).
fn mode_description(mode: LineMode) -> &'static str {
    match mode {
        LineMode::Off => "no sketch extraction",
        LineMode::EdgesOnly => "sketch of the full image",
        LineMode::SubjectOutline => "sketch of the subject only, transparent background",
    }
}

/// Build a two-line selectable label: bold title on top, secondary-coloured
/// description underneath. Matches the visual hierarchy of other descriptive
/// controls in the app.
fn two_line_label(title: &str, description: &str) -> egui::text::LayoutJob {
    use egui::text::{LayoutJob, TextFormat};
    let mut job = LayoutJob::default();
    job.append(
        title,
        0.0,
        TextFormat {
            color: theme::TEXT_PRIMARY,
            font_id: egui::FontId::proportional(theme::FONT_SIZE_BODY),
            ..Default::default()
        },
    );
    job.append("\n", 0.0, TextFormat::default());
    job.append(
        description,
        0.0,
        TextFormat {
            color: theme::TEXT_PRIMARY,
            font_id: egui::FontId::proportional(theme::FONT_SIZE_MONO),
            ..Default::default()
        },
    );
    job
}

/// Render the row 3 Lines button + popover. Returns `true` if line_mode changed
/// (caller should invalidate edge cache and re-route pipeline).
///
/// `seg_model_name` is the display name of the currently-selected BG removal
/// model (e.g. "BiRefNet"). Shown in the Model section when the user hovers
/// over the Subject mode, since that mode runs seg → DexiNed and both models
/// are actually in play.
#[allow(deprecated)]
pub fn render(ui: &mut Ui, settings: &mut ItemSettings, seg_model_name: &str) -> bool {
    let pop_id = egui::Id::new("lines_popover");
    let label = format!(
        "{}  Sketch: {}",
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
        egui::Stroke::new(theme::STROKE_DEFAULT, theme::ACCENT)
    } else {
        egui::Stroke::new(theme::STROKE_DEFAULT, egui::Color32::TRANSPARENT)
    })
    .corner_radius(theme::BUTTON_ROUNDING)
    .min_size(egui::vec2(0.0, theme::CHIP_HEIGHT));
    let resp = ui.add(btn).on_hover_ui(|ui| {
        ui.label(RichText::new("Sketch").strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        ui.label(
            RichText::new(
                "Stage 1 of 4 in the lines pipeline. Picks what DexiNed sees: Off (skipped), Subject (seg model masks to subject-on-white first, then DexiNed), Full (DexiNed runs on the whole scene). The knobs to the right are no-ops when Sketch is Off.",
            )
            .color(theme::TEXT_PRIMARY)
            .size(theme::FONT_SIZE_MONO),
        );
    });

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
            // Track which mode the pointer is currently over so the Model
            // section below can preview the model set that mode would use.
            let mut hovered_mode: Option<LineMode> = None;
            for mode in [LineMode::Off, LineMode::SubjectOutline, LineMode::EdgesOnly] {
                let selected = settings.line_mode == mode;
                let label = two_line_label(mode_label(mode), mode_description(mode));
                let resp = ui.selectable_label(selected, label);
                if resp.hovered() {
                    hovered_mode = Some(mode);
                }
                if resp.clicked() {
                    if !selected {
                        settings.line_mode = mode;
                        changed = true;
                    }
                    // Close the popover on any selection (even re-clicking the
                    // current one) — the user's intent was to make a choice
                    // and move on, not to keep the menu open.
                    ui.memory_mut(|m| m.close_popup(pop_id));
                }
            }

            ui.add_space(theme::SPACE_SM);
            ui.separator();
            ui.add_space(theme::SPACE_SM);

            // ── Model picker ──
            // Preview the model set for the hovered row; fall back to the
            // currently-selected mode when nothing is hovered. Subject sketch
            // runs seg → DexiNed, so both models appear; other modes show just
            // DexiNed (or nothing for Off).
            ui.label(
                RichText::new("Model")
                    .strong()
                    .color(theme::TEXT_PRIMARY),
            );
            ui.add_space(theme::SPACE_XS);
            let effective_mode = hovered_mode.unwrap_or(settings.line_mode);
            let model_text = match effective_mode {
                LineMode::Off => format!("{}  (no model used)", ICON_NEUROLOGY.codepoint),
                LineMode::EdgesOnly => format!("{}  DexiNed", ICON_NEUROLOGY.codepoint),
                LineMode::SubjectOutline => format!(
                    "{}  {seg_model_name} + DexiNed",
                    ICON_NEUROLOGY.codepoint,
                ),
            };
            ui.label(
                RichText::new(model_text)
                    .color(theme::TEXT_PRIMARY)
                    .size(theme::FONT_SIZE_BODY),
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
            assert!(!mode_description(mode).is_empty());
        }
    }
}
