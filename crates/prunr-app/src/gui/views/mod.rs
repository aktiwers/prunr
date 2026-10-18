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
pub mod tool_strip;
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
pub(crate) mod effects;
mod effect_chip;
pub(crate) mod faces_chip;
pub(crate) mod depth_chip;
pub(crate) mod get_model;
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
/// Ragged right: in a justified layout (every popup's) egui would justify
/// the lines and spread the words apart.
pub fn hint(ui: &mut egui::Ui, text: &str) {
    if text.is_empty() { return; }
    ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| ui.add(
        egui::Label::new(
            RichText::new(text)
                .color(theme::HINT)
                .size(theme::FONT_SIZE_CAPTION),
        )
        .wrap(),
    ));
}

/// Two-column key/value row for use inside `egui::Grid::new(..).num_columns(2)`.
/// Key uses monospace at FONT_SIZE_SECONDARY; value uses sans at FONT_SIZE_BODY in
/// TEXT_PRIMARY. `key_color` lets callers pick TEXT_PRIMARY (CLI flags,
/// shortcut keys) vs TEXT_SECONDARY (hardware labels).
pub fn kv_row(ui: &mut egui::Ui, key: &str, value: &str, key_color: egui::Color32) {
    ui.label(
        RichText::new(key)
            .monospace()
            .size(theme::FONT_SIZE_SECONDARY)
            .color(key_color),
    );
    ui.label(
        RichText::new(value)
            .size(theme::FONT_SIZE_BODY)
            .color(theme::TEXT_PRIMARY),
    );
    ui.end_row();
}

/// Float "is off its default" test shared by every chip and group.
pub(super) fn differs(a: f32, b: f32) -> bool {
    (a - b).abs() > f32::EPSILON
}

/// Which of `all` has `value`'s variant, whatever its parameters.
pub(super) fn variant_index<T>(all: &[T], value: &T) -> usize {
    all.iter().position(|o| std::mem::discriminant(o) == std::mem::discriminant(value)).unwrap_or(0)
}

/// Format a byte count for human display. Used by the Model Store
/// (download progress, disk-usage footer) and stays here so future
/// callers don't reinvent it.
pub fn format_byte_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * MB;
    if bytes >= GB { format!("{:.1} GB", bytes as f64 / GB as f64) }
    else if bytes >= MB { format!("{:.0} MB", bytes as f64 / MB as f64) }
    else if bytes >= KB { format!("{} KB", bytes / KB) }
    else { format!("{bytes} B") }
}

/// Model display name (no icon).
pub fn model_name(model: SettingsModel) -> &'static str {
    model_info(model).name
}

pub struct ModelInfo {
    pub icon: &'static str,
    pub name: &'static str,
    /// What the model is for, in a few words; a bundled model's size
    /// goes here too, since the registry has none for it.
    pub character: &'static str,
}

pub fn model_info(model: SettingsModel) -> ModelInfo {
    let (icon, name, character) = match model {
        SettingsModel::Silueta => (ICON_SPRINT.codepoint, "Silueta", "Fast \u{00b7} ~4 MB"),
        SettingsModel::U2net => (ICON_SMART_TOY.codepoint, "U2Net", "Quality \u{00b7} ~170 MB"),
        SettingsModel::BiRefNetLite => (ICON_NEUROLOGY.codepoint, "BiRefNet", "Detail \u{00b7} ~214 MB"),
        SettingsModel::None => (ICON_BLOCK.codepoint, "No model", "Effects, lines and background only"),
        SettingsModel::Inpaint => (ICON_BRUSH.codepoint, "Eraser (LaMa)", "Object removal"),
        SettingsModel::BigInpaint => (ICON_BRUSH.codepoint, "Eraser (Big-LaMa)", "Sharper fills"),
        SettingsModel::MiganInpaint => (ICON_BRUSH.codepoint, "Eraser (MI-GAN)", "Compact"),
        SettingsModel::SdInpaint => (ICON_BRUSH.codepoint, "Eraser (SD 1.5)", "Generative"),
        SettingsModel::RoremInpaint => (ICON_BRUSH.codepoint, "Eraser (RORem)", "Large objects"),
        SettingsModel::RealEsrganUpscale => (ICON_ARROW_UPWARD.codepoint, "Upscale (Real-ESRGAN)", "4\u{00d7}"),
        SettingsModel::Nomos8kUpscale => (ICON_ARROW_UPWARD.codepoint, "Upscale (Nomos8k)", "4\u{00d7} photos, the slowest"),
        SettingsModel::FourXNmkdSiaxCxUpscale => (ICON_ARROW_UPWARD.codepoint, "Upscale (NMKD Siax-CX)", "4\u{00d7} clean photos"),
        SettingsModel::FourXNmkdSuperscaleUpscale => (ICON_ARROW_UPWARD.codepoint, "Upscale (NMKD Superscale)", "4\u{00d7} restoration"),
        SettingsModel::Nomos2Upscale => (ICON_ARROW_UPWARD.codepoint, "Upscale (Nomos2)", "4\u{00d7} fast, natural texture"),
        SettingsModel::ColorizeDdcolor => (ICON_PALETTE.codepoint, "Colorize (DDColor)", "Black & white to colour"),
    };
    ModelInfo { icon, name, character }
}

/// A picker row's second line: what the model is for and, for a
/// download, its size from the registry.
pub fn model_blurb(model: SettingsModel) -> std::borrow::Cow<'static, str> {
    let info = model_info(model);
    match model.to_model_id().and_then(prunr_models::descriptor).and_then(|d| d.size_mb()) {
        Some(mb) => format!("{} \u{00b7} {}", info.character, get_model::size_label(mb)).into(),
        None => info.character.into(),
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
        assert_eq!(format_byte_size(2 * 1024 * 1024 * 1024), "2.0 GB");
    }
}
