pub mod toolbar;
pub mod canvas;
pub mod statusbar;
pub mod shortcuts;
pub mod cli_help;
pub mod settings;
pub mod sidebar;
pub mod chip;
pub mod fmt;
pub mod lines_popover;
pub mod preset_dropdown;
pub mod adjustments_toolbar;
pub mod brush_chip;
pub(crate) mod magic_brush_chip;
pub mod eraser_chip;
pub mod brush_overlay;
pub mod pipeline_flow;
pub mod model_store;
pub mod runtime_prompt;
pub(crate) mod upscale_chip;
pub(crate) mod upscale_toolbar;
pub(crate) mod refinement_row;
pub(crate) mod progress_widget;
pub(crate) mod selection_overlay;
pub(crate) mod selection_action_bar;

use egui::RichText;
use egui_material_icons::icons::*;
use crate::gui::settings::SettingsModel;
use crate::gui::theme;

/// Bold section heading used in modals (settings, CLI help).
pub fn section_heading(ui: &mut egui::Ui, title: &str) {
    ui.add_space(theme::SPACE_XS);
    ui.label(
        RichText::new(title)
            .size(theme::FONT_SIZE_HEADING)
            .strong()
            .color(theme::TEXT_PRIMARY),
    );
    ui.add_space(theme::SPACE_SM);
}

/// Description text below a control. Mono font signals "supplemental".
/// Wraps to the available width so a long hint doesn't push its parent
/// container wider than the screen — egui's default Label is single-line
/// which forces parents to grow to the longest hint, which is what we
/// don't want inside narrow popovers (brush_chip) and fixed-width modals.
pub fn hint(ui: &mut egui::Ui, text: &str) {
    if text.is_empty() { return; }
    ui.add(
        egui::Label::new(
            RichText::new(text)
                .color(theme::TEXT_PRIMARY)
                .size(theme::FONT_SIZE_MONO),
        )
        .wrap(),
    );
}

/// Two-column key/value row for use inside `egui::Grid::new(..).num_columns(2)`.
/// Key uses monospace at FONT_SIZE_MONO; value uses sans at FONT_SIZE_BODY in
/// TEXT_PRIMARY. `key_color` lets callers pick TEXT_PRIMARY (CLI flags,
/// shortcut keys) vs TEXT_SECONDARY (hardware labels).
pub fn kv_row(ui: &mut egui::Ui, key: &str, value: &str, key_color: egui::Color32) {
    ui.label(
        RichText::new(key)
            .monospace()
            .size(theme::FONT_SIZE_MONO)
            .color(key_color),
    );
    ui.label(
        RichText::new(value)
            .size(theme::FONT_SIZE_BODY)
            .color(theme::TEXT_PRIMARY),
    );
    ui.end_row();
}

/// Format a byte count for human display. Used by the Model Store
/// (download progress, disk-usage footer) and stays here so future
/// callers don't reinvent it.
pub fn format_byte_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * MB;
    if bytes >= GB { format!("{:.2} GB", bytes as f64 / GB as f64) }
    else if bytes >= MB { format!("{:.0} MB", bytes as f64 / MB as f64) }
    else if bytes >= KB { format!("{} KB", bytes / KB) }
    else { format!("{bytes} B") }
}

/// Model display name (no icon).
pub fn model_name(model: SettingsModel) -> &'static str {
    model_info(model).1
}

/// Icon, display name and one-line blurb (strength · size) for a model.
pub fn model_info(model: SettingsModel) -> (&'static str, &'static str, &'static str) {
    match model {
        SettingsModel::Silueta => (ICON_SPRINT.codepoint, "Silueta", "fast \u{00b7} ~4 MB"),
        SettingsModel::U2net => (ICON_SMART_TOY.codepoint, "U2Net", "quality \u{00b7} ~170 MB"),
        SettingsModel::BiRefNetLite => (ICON_NEUROLOGY.codepoint, "BiRefNet", "detail \u{00b7} ~214 MB"),
        SettingsModel::None => (ICON_BLOCK.codepoint, "No model", "filters and lines only"),
        SettingsModel::Inpaint => (ICON_BRUSH.codepoint, "Eraser (LaMa)", "object removal \u{00b7} ~199 MB"),
        SettingsModel::BigInpaint => (ICON_BRUSH.codepoint, "Eraser (Big-LaMa)", "sharper fills \u{00b7} ~199 MB"),
        SettingsModel::MiganInpaint => (ICON_BRUSH.codepoint, "Eraser (MI-GAN)", "compact \u{00b7} ~26 MB"),
        SettingsModel::SdInpaint => (ICON_BRUSH.codepoint, "Eraser (SD 1.5)", "generative \u{00b7} ~2 GB"),
        SettingsModel::RealEsrganUpscale => (ICON_ARROW_UPWARD.codepoint, "Upscale (Real-ESRGAN)", "4\u{00d7} \u{00b7} ~64 MB"),
        SettingsModel::Nomos8kUpscale => (ICON_ARROW_UPWARD.codepoint, "Upscale (Nomos8k)", "4\u{00d7} photos \u{00b7} ~155 MB"),
        SettingsModel::FourXNmkdSiaxCxUpscale => (ICON_ARROW_UPWARD.codepoint, "Upscale (NMKD Siax-CX)", "4\u{00d7} clean photos \u{00b7} ~64 MB"),
        SettingsModel::FourXNmkdSuperscaleUpscale => (ICON_ARROW_UPWARD.codepoint, "Upscale (NMKD Superscale)", "4\u{00d7} restoration \u{00b7} ~64 MB"),
    }
}

#[cfg(test)]
mod format_byte_size_tests {
    use super::format_byte_size;
    #[test]
    fn formats_units_correctly() {
        assert_eq!(format_byte_size(0), "0 B");
        assert_eq!(format_byte_size(512), "512 B");
        assert_eq!(format_byte_size(2048), "2 KB");
        assert_eq!(format_byte_size(50 * 1024 * 1024), "50 MB");
        assert_eq!(format_byte_size(2 * 1024 * 1024 * 1024), "2.00 GB");
    }
}
