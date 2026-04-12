use egui::{Align2, RichText};

use crate::gui::app::PrunrApp;
use crate::gui::settings::Settings;
use crate::gui::theme;

/// Slider row: label left, slider fills middle, value right.
fn slider_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    value_text: &str,
    logarithmic: bool,
    step: Option<f64>,
) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(label)
                .color(theme::TEXT_PRIMARY)
                .size(theme::FONT_SIZE_BODY),
        );
        let avail = ui.available_width() - 52.0;
        let mut slider = egui::Slider::new(value, range).show_value(false);
        if logarithmic { slider = slider.logarithmic(true); }
        if let Some(s) = step { slider = slider.step_by(s); }
        ui.add_sized([avail.max(100.0), 18.0], slider);
        ui.label(
            RichText::new(value_text)
                .monospace()
                .size(theme::FONT_SIZE_MONO)
                .color(theme::TEXT_PRIMARY),
        );
    });
}

/// Dimmed description text below a control.
fn hint(ui: &mut egui::Ui, text: &str) {
    ui.label(
        RichText::new(text)
            .color(theme::TEXT_HINT)
            .size(theme::FONT_SIZE_MONO),
    );
}

use super::section_heading;

pub fn render(ctx: &egui::Context, app: &mut PrunrApp) {
    theme::draw_modal_backdrop(ctx, "settings_backdrop");

    let mut open = true;
    let window_response = egui::Window::new("Settings")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([theme::SETTINGS_DIALOG_WIDTH, theme::SETTINGS_DIALOG_HEIGHT])
        .frame(theme::overlay_frame())
        .show(ctx, |ui| {
            {
                let vis = ui.visuals_mut();
                vis.widgets.inactive.bg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(0x60, 0x60, 0x60));
                vis.widgets.hovered.bg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(0x80, 0x80, 0x80));
                vis.widgets.inactive.bg_fill = egui::Color32::from_rgb(0x50, 0x50, 0x50);
                vis.widgets.inactive.fg_stroke =
                    egui::Stroke::new(1.0, theme::TEXT_PRIMARY);
                vis.widgets.hovered.bg_fill = egui::Color32::from_rgb(0x5a, 0x5a, 0x5a);
            }

            ui.vertical(|ui| {
                section_heading(ui, "General");

                let max_jobs = app.settings.max_jobs();
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Parallel jobs")
                            .color(theme::TEXT_PRIMARY)
                            .size(theme::FONT_SIZE_BODY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add_enabled(
                            app.settings.parallel_jobs < max_jobs,
                            egui::Button::new(RichText::new("+").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY))
                                .fill(theme::BG_SECONDARY).min_size(egui::vec2(28.0, 28.0)),
                        ).clicked() {
                            app.settings.parallel_jobs += 1;
                        }
                        ui.label(
                            RichText::new(format!("{}", app.settings.parallel_jobs))
                                .color(theme::TEXT_PRIMARY)
                                .size(theme::FONT_SIZE_BODY)
                                .strong(),
                        );
                        if ui.add_enabled(
                            app.settings.parallel_jobs > 1,
                            egui::Button::new(RichText::new("\u{2212}").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY))
                                .fill(theme::BG_SECONDARY).min_size(egui::vec2(28.0, 28.0)),
                        ).clicked() {
                            app.settings.parallel_jobs -= 1;
                        }
                    });
                });
                ui.add_space(-10.0);
                let jobs_hint = if app.settings.is_gpu() {
                    format!("Images processed at the same time (1\u{2013}{max_jobs}, GPU: 1\u{2013}2 is optimal)")
                } else {
                    format!("Images processed at the same time (1\u{2013}{max_jobs})")
                };
                hint(ui, &jobs_hint);
                ui.add_space(theme::SPACE_MD);

                ui.checkbox(
                    &mut app.settings.auto_remove_on_import,
                    RichText::new("Auto-remove on import")
                        .color(theme::TEXT_PRIMARY)
                        .size(theme::FONT_SIZE_BODY),
                );
                hint(ui, "Start removing background as soon as images are opened");
                ui.add_space(theme::SPACE_MD);

                ui.checkbox(
                    &mut app.settings.force_cpu,
                    RichText::new("Force CPU")
                        .color(theme::TEXT_PRIMARY)
                        .size(theme::FONT_SIZE_BODY),
                );
                hint(ui, "Use CPU even when GPU is available (slower but more compatible)");

                ui.add_space(theme::SPACE_MD);
                ui.separator();

                section_heading(ui, "Mask Tuning (Advanced)");

                let gamma_text = format!("{:.2}", app.settings.mask_gamma);
                slider_row(
                    ui, "Strength", &mut app.settings.mask_gamma,
                    0.2..=3.0, &gamma_text, true, None,
                );
                hint(ui, "Controls how much background is removed.");
                hint(ui, "Higher values cut deeper, lower values are");
                hint(ui, "more forgiving with edges and fine detail.");
                ui.add_space(theme::SPACE_MD);

                let threshold_text = format!("{:.2}", app.settings.mask_threshold);
                ui.horizontal(|ui| {
                    ui.checkbox(
                        &mut app.settings.mask_threshold_enabled,
                        RichText::new("Hard cutoff")
                            .color(theme::TEXT_PRIMARY)
                            .size(theme::FONT_SIZE_BODY),
                    );
                    if app.settings.mask_threshold_enabled {
                        let avail = ui.available_width() - 52.0;
                        ui.add_sized(
                            [avail.max(100.0), 18.0],
                            egui::Slider::new(&mut app.settings.mask_threshold, 0.01..=0.99)
                                .show_value(false),
                        );
                        ui.label(
                            RichText::new(&threshold_text)
                                .monospace()
                                .size(theme::FONT_SIZE_MONO)
                                .color(theme::TEXT_PRIMARY),
                        );
                    }
                });
                hint(ui, "Makes the result fully opaque or fully transparent");
                hint(ui, "\u{2014} no in-between. Lower values keep more of the");
                hint(ui, "subject, higher values remove more aggressively.");
                ui.add_space(theme::SPACE_MD);

                let edge_label = if app.settings.edge_shift > 0.5 {
                    format!("erode {:.0}px", app.settings.edge_shift)
                } else if app.settings.edge_shift < -0.5 {
                    format!("dilate {:.0}px", app.settings.edge_shift.abs())
                } else {
                    "off".to_string()
                };
                slider_row(
                    ui, "Edge refine", &mut app.settings.edge_shift,
                    -5.0..=5.0, &edge_label, false, Some(1.0),
                );
                hint(ui, "Adjusts the outline around your subject.");
                hint(ui, "Positive values trim away fringe pixels,");
                hint(ui, "negative values expand to keep more edge detail.");
                ui.add_space(theme::SPACE_MD);

                ui.checkbox(
                    &mut app.settings.refine_edges,
                    RichText::new("Refine edges")
                        .color(theme::TEXT_PRIMARY)
                        .size(theme::FONT_SIZE_BODY),
                );
                hint(ui, "Uses the original image colors to sharpen");
                hint(ui, "the mask around fine detail like hair and leaves.");

                ui.add_space(theme::SPACE_LG);
                ui.separator();
                ui.add_space(theme::SPACE_SM);

                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Backend")
                            .color(theme::TEXT_HINT)
                            .size(theme::FONT_SIZE_MONO),
                    );
                    ui.label(
                        RichText::new(&app.settings.active_backend)
                            .monospace()
                            .size(theme::FONT_SIZE_MONO)
                            .color(theme::TEXT_PRIMARY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button(
                            RichText::new("Reset to defaults")
                                .size(theme::FONT_SIZE_BODY),
                        ).clicked() {
                            let backend = app.settings.active_backend.clone();
                            app.settings = Settings::default();
                            app.settings.active_backend = backend;
                            app.settings.parallel_jobs = app.settings.default_jobs();
                        }
                    });
                });
            });
        });

    let now = ctx.input(|i| i.time);
    let debounce_passed = (now - app.settings_opened_at) > 0.15;
    let clicked_outside = debounce_passed && theme::clicked_outside_modal(window_response);

    if !open || clicked_outside {
        app.close_settings();
    }
}
