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

/// Platform-aware modifier key name.
pub fn modifier_key() -> &'static str {
    if cfg!(target_os = "macos") { "Cmd" } else { "Ctrl" }
}

/// Model display name (no icon).
pub fn model_name(model: SettingsModel) -> &'static str {
    match model {
        SettingsModel::Silueta => "Silueta",
        SettingsModel::U2net => "U2Net",
        SettingsModel::BiRefNetLite => "BiRefNet",
    }
}

/// Model label with icon. `short` = selected text, `long` = dropdown row.
pub fn model_label(model: SettingsModel, short: bool) -> String {
    let (icon, name, speed, size) = match model {
        SettingsModel::Silueta => (ICON_SPRINT.codepoint, "Silueta", "fast", "~4 MB"),
        SettingsModel::U2net => (ICON_SMART_TOY.codepoint, "U2Net", "quality", "~170 MB"),
        SettingsModel::BiRefNetLite => (ICON_NEUROLOGY.codepoint, "BiRefNet", "detail", "~214 MB"),
    };
    if short {
        format!("{icon}  {name}")
    } else {
        format!("{icon}  {name}  \u{2022} {speed}  \u{2022} {size}")
    }
}
