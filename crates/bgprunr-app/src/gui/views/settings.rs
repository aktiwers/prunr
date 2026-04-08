use egui::{Align2, RichText};

use crate::gui::app::BgPrunrApp;
use crate::gui::settings::SettingsModel;
use crate::gui::theme;

pub fn render(ctx: &egui::Context, app: &mut BgPrunrApp) {
    let backdrop_clicked = theme::draw_modal_backdrop(ctx, "settings_backdrop");

    let mut open = true;
    egui::Window::new("Settings")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([theme::SETTINGS_DIALOG_WIDTH, theme::SETTINGS_DIALOG_HEIGHT])
        .frame(theme::overlay_frame())
        .show(ctx, |ui| {
            // Visible checkbox borders on dark overlay background
            {
                let vis = ui.visuals_mut();
                vis.widgets.inactive.bg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(0x60, 0x60, 0x60));
                vis.widgets.hovered.bg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(0x80, 0x80, 0x80));
            }

            ui.vertical(|ui| {
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
                ui.checkbox(
                    &mut app.settings.auto_remove_on_import,
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
                ui.add_space(theme::SPACE_SM);

                // Row 3 — Parallel jobs
                let max_jobs = num_cpus::get();
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Parallel jobs")
                            .color(theme::TEXT_PRIMARY)
                            .size(theme::FONT_SIZE_BODY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // + button
                        if ui.add_enabled(
                            app.settings.parallel_jobs < max_jobs,
                            egui::Button::new(RichText::new("+").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY))
                                .fill(theme::BG_SECONDARY).min_size(egui::vec2(28.0, 28.0)),
                        ).clicked() {
                            app.settings.parallel_jobs += 1;
                        }
                        // Value display
                        ui.label(
                            RichText::new(format!("{}", app.settings.parallel_jobs))
                                .color(theme::TEXT_PRIMARY)
                                .size(theme::FONT_SIZE_BODY)
                                .strong(),
                        );
                        // - button
                        if ui.add_enabled(
                            app.settings.parallel_jobs > 1,
                            egui::Button::new(RichText::new("\u{2212}").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY))
                                .fill(theme::BG_SECONDARY).min_size(egui::vec2(28.0, 28.0)),
                        ).clicked() {
                            app.settings.parallel_jobs -= 1;
                        }
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
                ui.checkbox(
                    &mut app.settings.reveal_animation_enabled,
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
    if !open || backdrop_clicked {
        app.show_settings = false;
    }
}
