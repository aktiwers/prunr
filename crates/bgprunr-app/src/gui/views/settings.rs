use egui::{Align2, RichText};

use crate::gui::app::BgPrunrApp;
use crate::gui::settings::SettingsModel;
use crate::gui::theme;

pub fn render(ctx: &egui::Context, app: &mut BgPrunrApp) {
    theme::draw_modal_backdrop(ctx, "settings_backdrop");

    egui::Window::new("Settings")
        .collapsible(false)
        .resizable(false)
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([theme::SETTINGS_DIALOG_WIDTH, theme::SETTINGS_DIALOG_HEIGHT])
        .frame(theme::overlay_frame())
        .show(ctx, |ui| {
            ui.vertical(|ui| {
                // Title row with close button
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Settings")
                            .size(theme::FONT_SIZE_HEADING)
                            .strong()
                            .color(theme::TEXT_PRIMARY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("✕").clicked() {
                            app.show_settings = false;
                        }
                    });
                });
                ui.separator();
                ui.add_space(theme::SPACE_SM);

                // Row 1 — Model selection
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Model")
                            .color(theme::TEXT_PRIMARY)
                            .size(theme::FONT_SIZE_BODY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let model_text = match app.settings.model {
                            SettingsModel::Silueta => "silueta (fast, ~4 MB)",
                            SettingsModel::U2net => "u2net (quality, ~170 MB)",
                        };
                        egui::ComboBox::from_id_salt("settings_model")
                            .selected_text(model_text)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut app.settings.model,
                                    SettingsModel::Silueta,
                                    "silueta (fast, ~4 MB)",
                                );
                                ui.selectable_value(
                                    &mut app.settings.model,
                                    SettingsModel::U2net,
                                    "u2net (quality, ~170 MB)",
                                );
                            });
                    });
                });
                ui.add_space(theme::SPACE_SM);

                // Row 2 — Auto-remove on import
                ui.horizontal(|ui| {
                    ui.checkbox(&mut app.settings.auto_remove_on_import, "");
                    ui.vertical(|ui| {
                        ui.label(
                            RichText::new("Auto-remove on import")
                                .color(theme::TEXT_PRIMARY)
                                .size(theme::FONT_SIZE_BODY),
                        );
                        ui.label(
                            RichText::new(
                                "Automatically process images when added to the queue",
                            )
                            .color(theme::TEXT_SECONDARY)
                            .size(theme::FONT_SIZE_MONO),
                        );
                    });
                });
                ui.add_space(theme::SPACE_SM);

                // Row 3 — Parallel jobs slider
                let max_jobs = num_cpus::get();
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Parallel jobs")
                            .color(theme::TEXT_PRIMARY)
                            .size(theme::FONT_SIZE_BODY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add(
                            egui::Slider::new(&mut app.settings.parallel_jobs, 1..=max_jobs)
                                .integer(),
                        );
                    });
                });
                ui.label(
                    RichText::new(format!(
                        "Number of images to process simultaneously (1\u{2013}{max_jobs})"
                    ))
                    .color(theme::TEXT_SECONDARY)
                    .size(theme::FONT_SIZE_MONO),
                );
                ui.add_space(theme::SPACE_SM);

                // Row 4 — Reveal animation toggle
                ui.horizontal(|ui| {
                    ui.checkbox(&mut app.settings.reveal_animation_enabled, "");
                    ui.vertical(|ui| {
                        ui.label(
                            RichText::new("Reveal animation")
                                .color(theme::TEXT_PRIMARY)
                                .size(theme::FONT_SIZE_BODY),
                        );
                        ui.label(
                            RichText::new(
                                "Play dissolve effect when background removal completes",
                            )
                            .color(theme::TEXT_SECONDARY)
                            .size(theme::FONT_SIZE_MONO),
                        );
                    });
                });
                ui.add_space(theme::SPACE_SM);

                // Row 5 — Inference backend (read-only)
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Inference backend")
                            .color(theme::TEXT_PRIMARY)
                            .size(theme::FONT_SIZE_BODY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(&app.settings.active_backend)
                                .monospace()
                                .size(theme::FONT_SIZE_MONO)
                                .color(theme::TEXT_SECONDARY),
                        );
                    });
                });
            });
        });
}
