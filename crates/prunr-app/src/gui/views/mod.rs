pub mod toolbar;
pub mod canvas;
pub mod statusbar;
pub mod shortcuts;
pub mod cli_help;
pub mod settings;
pub mod sidebar;
pub mod chip;
pub mod lines_popover;
pub mod preset_dropdown;
pub mod adjustments_toolbar;
pub mod brush_chip;
pub mod brush_overlay;
pub mod pipeline_flow;
pub mod model_store;
pub mod runtime_prompt;

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

/// Description text below a control. Mono font signals "supplemental"; color
/// stays primary so modal body text reads comfortably. Empty strings render nothing.
pub fn hint(ui: &mut egui::Ui, text: &str) {
    if text.is_empty() { return; }
    ui.label(
        RichText::new(text)
            .color(theme::TEXT_PRIMARY)
            .size(theme::FONT_SIZE_MONO),
    );
}

/// Platform-aware modifier key name.
pub fn modifier_key() -> &'static str {
    if cfg!(target_os = "macos") { "Cmd" } else { "Ctrl" }
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

/// Model display name (no icon).
pub fn model_name(model: SettingsModel) -> &'static str {
    match model {
        SettingsModel::Silueta => "Silueta",
        SettingsModel::U2net => "U2Net",
        SettingsModel::BiRefNetLite => "BiRefNet",
        SettingsModel::None => "No model",
        SettingsModel::Inpaint => "Eraser (LaMa)",
        SettingsModel::BigInpaint => "Eraser (Big-LaMa)",
        SettingsModel::MiganInpaint => "Eraser (MI-GAN)",
        SettingsModel::SdInpaint => "Eraser (SD 1.5)",
    }
}

/// Model label with icon. `short` = selected text, `long` = dropdown row.
pub fn model_label(model: SettingsModel, short: bool) -> String {
    let (icon, name, speed, size) = match model {
        SettingsModel::Silueta => (ICON_SPRINT.codepoint, "Silueta", "fast", "~4 MB"),
        SettingsModel::U2net => (ICON_SMART_TOY.codepoint, "U2Net", "quality", "~170 MB"),
        SettingsModel::BiRefNetLite => (ICON_NEUROLOGY.codepoint, "BiRefNet", "detail", "~214 MB"),
        SettingsModel::None => (ICON_BLOCK.codepoint, "No model", "No background removal", "0 MB"),
        SettingsModel::Inpaint => (ICON_BRUSH.codepoint, "Eraser (LaMa)", "object removal", "~199 MB"),
        SettingsModel::BigInpaint => (ICON_BRUSH.codepoint, "Eraser (Big-LaMa)", "sharper fills", "~199 MB"),
        SettingsModel::MiganInpaint => (ICON_BRUSH.codepoint, "Eraser (MI-GAN)", "compact GAN", "~26 MB"),
        SettingsModel::SdInpaint => (ICON_BRUSH.codepoint, "Eraser (SD 1.5)", "generative", "~2 GB"),
    };
    if short {
        format!("{icon}  {name}")
    } else {
        format!("{icon}  {name}  \u{2022} {speed}  \u{2022} {size}")
    }
}
