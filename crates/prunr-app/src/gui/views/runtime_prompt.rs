//! First-launch prompt for hardware-recommended runtimes.

use egui::RichText;

use super::chip::{button, ButtonKind};
use crate::gui::theme;
use crate::runtime_install::RuntimeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimePromptAction {
    Install,
    NotNow,
    OpenSettings,
}

/// Returns `None` while the modal is open with no user response yet.
pub fn render_runtime_prompt(ctx: &egui::Context, rt: RuntimeId) -> Option<RuntimePromptAction> {
    let mut result: Option<RuntimePromptAction> = None;

    let backdrop_closed = theme::standard_modal_window(
        ctx, "runtime_prompt", "Faster inference is available",
        [theme::SETTINGS_DIALOG_WIDTH, 280.0],
        |ui| {
            ui.label(
                RichText::new("Faster inference is available")
                    .size(theme::FONT_SIZE_HEADING)
                    .strong()
                    .color(theme::TEXT_PRIMARY),
            );
            ui.add_space(theme::SPACE_SM);
            ui.label(
                RichText::new(rt.first_launch_prompt_body())
                .size(theme::FONT_SIZE_BODY)
                .color(theme::TEXT_PRIMARY),
            );
            ui.add_space(theme::SPACE_XS);
            ui.label(
                RichText::new("You can install it later from Settings → Hardware.")
                    .size(theme::FONT_SIZE_MONO)
                    .color(theme::TEXT_SECONDARY),
            );
            ui.add_space(theme::SPACE_MD);

            ui.horizontal(|ui| {
                if button(ui, ButtonKind::Secondary, "Not now").clicked() {
                    result = Some(RuntimePromptAction::NotNow);
                }
                if button(ui, ButtonKind::Secondary, "Open Settings").clicked() {
                    result = Some(RuntimePromptAction::OpenSettings);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if button(ui, ButtonKind::Primary, rt.install_button_label()).clicked() {
                        result = Some(RuntimePromptAction::Install);
                    }
                });
            });
        },
    );

    // Backdrop click = soft-dismiss = "Not now".
    if backdrop_closed && result.is_none() {
        result = Some(RuntimePromptAction::NotNow);
    }
    result
}
