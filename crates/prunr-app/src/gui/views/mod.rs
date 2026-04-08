pub mod toolbar;
pub mod canvas;
pub mod statusbar;
pub mod shortcuts;
pub mod cli_help;
pub mod settings;
pub mod animation;
pub mod sidebar;

use egui::RichText;
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
