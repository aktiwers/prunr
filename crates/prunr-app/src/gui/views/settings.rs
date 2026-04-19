//! Settings modal. v2: app-wide config only — per-image knobs live on the
//! persistent adjustments toolbar (rows 2 + 3).
//!
//! Sections in this modal:
//!   • Performance — parallel jobs, force CPU, live preview toggle
//!   • Appearance  — dark checker, auto-hide adjustments
//!   • History     — history depth (shown when chain mode is on)
//!   • Processing  — chain mode
//!   • Default preset — applied to new images on import
//!
//! A Hotkeys tab for rebindable shortcuts will slot in next to General.

use egui::{Align2, RichText};

use crate::gui::app::PrunrApp;
use crate::gui::settings::Settings;
use crate::gui::theme;

use super::{hint, section_heading};

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

pub fn render(ctx: &egui::Context, app: &mut PrunrApp) {
    theme::draw_modal_backdrop(ctx, "settings_backdrop");

    let mut open = true;
    let window_response = egui::Window::new("Settings")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([theme::SETTINGS_DIALOG_WIDTH, 520.0])
        .frame(theme::overlay_frame())
        .show(ctx, |ui| {
            {
                let vis = ui.visuals_mut();
                vis.widgets.inactive.bg_stroke =
                    egui::Stroke::new(theme::STROKE_DEFAULT, egui::Color32::from_rgb(0x60, 0x60, 0x60));
                vis.widgets.hovered.bg_stroke =
                    egui::Stroke::new(theme::STROKE_DEFAULT, egui::Color32::from_rgb(0x80, 0x80, 0x80));
                vis.widgets.inactive.bg_fill = theme::WIDGET_INACTIVE_BG;
                vis.widgets.inactive.fg_stroke =
                    egui::Stroke::new(theme::STROKE_DEFAULT, theme::TEXT_PRIMARY);
                vis.widgets.hovered.bg_fill = theme::WIDGET_HOVER_BG;
            }

            // Header with Reset-all action on the right.
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("App settings")
                        .size(theme::FONT_SIZE_HEADING)
                        .strong()
                        .color(theme::TEXT_PRIMARY),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button(
                        RichText::new("Reset all")
                            .size(theme::FONT_SIZE_MONO)
                            .color(theme::TEXT_SECONDARY),
                    ).clicked() {
                        // Preserve presets + default pointer across a
                        // Reset of app-wide settings — those are per-user
                        // artifacts that don't belong to "app defaults."
                        let backend = app.settings.active_backend.clone();
                        let presets = std::mem::take(&mut app.settings.presets);
                        let default_preset = app.settings.default_preset.clone();
                        app.settings = Settings::default();
                        app.settings.active_backend = backend;
                        app.settings.parallel_jobs = app.settings.default_jobs();
                        app.settings.presets = presets;
                        app.settings.default_preset = default_preset;
                    }
                });
            });
            ui.separator();
            ui.add_space(theme::SPACE_SM);

            // ── Performance ──
            section_heading(ui, "Performance");
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
                            .fill(theme::BG_SECONDARY).min_size(egui::vec2(theme::CHIP_HEIGHT, theme::CHIP_HEIGHT)),
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
                            .fill(theme::BG_SECONDARY).min_size(egui::vec2(theme::CHIP_HEIGHT, theme::CHIP_HEIGHT)),
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

            let has_gpu = !prunr_core::OrtEngine::detect_active_provider().eq_ignore_ascii_case("CPU");
            if has_gpu {
                ui.checkbox(
                    &mut app.settings.force_cpu,
                    RichText::new("Force CPU")
                        .color(theme::TEXT_PRIMARY)
                        .size(theme::FONT_SIZE_BODY),
                );
                hint(ui, "Use CPU even when GPU is available (resets each launch).");
                ui.add_space(theme::SPACE_MD);
            }

            ui.checkbox(
                &mut app.settings.live_preview,
                RichText::new("Live preview")
                    .color(theme::TEXT_PRIMARY)
                    .size(theme::FONT_SIZE_BODY),
            );
            hint(ui, "Auto-rerun mask and edge tweaks as you adjust them.");
            ui.add_space(theme::SPACE_MD);

            ui.separator();
            ui.add_space(theme::SPACE_SM);

            // ── Appearance ──
            section_heading(ui, "Appearance");
            ui.checkbox(
                &mut app.settings.dark_checker,
                RichText::new("Dark checkerboard")
                    .color(theme::TEXT_PRIMARY)
                    .size(theme::FONT_SIZE_BODY),
            );
            hint(ui, "Use dark tones for the transparency pattern \u{2014} helps when viewing light results.");
            ui.add_space(theme::SPACE_MD);

            ui.checkbox(
                &mut app.settings.auto_hide_adjustments,
                RichText::new("Auto-hide adjustments toolbar")
                    .color(theme::TEXT_PRIMARY)
                    .size(theme::FONT_SIZE_BODY),
            );
            hint(ui, "Collapse the adjustments toolbar when the cursor leaves it. Toggle manually with Shift+H.");
            ui.add_space(theme::SPACE_MD);

            ui.separator();
            ui.add_space(theme::SPACE_SM);

            // ── Processing ──
            section_heading(ui, "Processing");
            ui.checkbox(
                &mut app.settings.auto_process_on_import,
                RichText::new("Auto-process on import")
                    .color(theme::TEXT_PRIMARY)
                    .size(theme::FONT_SIZE_BODY),
            );
            hint(ui, "When enabled, each image kicks off Process automatically on import. The full pipeline runs — BG removal or line extraction, whichever matches the current Line mode.");
            ui.add_space(theme::SPACE_MD);

            ui.checkbox(
                &mut app.settings.chain_mode,
                RichText::new("Chain mode")
                    .color(theme::TEXT_PRIMARY)
                    .size(theme::FONT_SIZE_BODY),
            );
            hint(ui, "Process the current result instead of the original \u{2014} stacks effects.");

            if app.settings.chain_mode {
                ui.add_space(theme::SPACE_SM);
                let mut depth_f32 = app.settings.history_depth as f32;
                let depth_text = format!("{}", app.settings.history_depth);
                slider_row(
                    ui, "History depth", &mut depth_f32,
                    1.0..=50.0, &depth_text, false, Some(1.0),
                );
                app.settings.history_depth = depth_f32 as usize;
                hint(ui, "Maximum undo steps per image. Higher = more memory.");
            }
            ui.add_space(theme::SPACE_MD);

            ui.separator();
            ui.add_space(theme::SPACE_SM);

            // ── Default preset for new images ──
            section_heading(ui, "Default preset");
            let preset_names = super::preset_dropdown::all_preset_names(&app.settings);
            let current = app.settings.default_preset.clone();
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("default_preset")
                    .selected_text(
                        RichText::new(&current)
                            .color(theme::TEXT_PRIMARY)
                            .size(theme::FONT_SIZE_BODY),
                    )
                    .show_ui(ui, |ui| {
                        for name in &preset_names {
                            let selected = app.settings.default_preset == *name;
                            if ui.selectable_label(selected, name).clicked() {
                                app.settings.default_preset = name.clone();
                            }
                        }
                    });
            });
            hint(ui, "New images inherit this preset. Reset-all-knobs on row 2 also restores this preset's values.");
            ui.add_space(theme::SPACE_MD);

            // Backend info at the bottom
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.label(
                    RichText::new(format!("Backend: {}", app.settings.active_backend))
                        .monospace()
                        .size(theme::FONT_SIZE_MONO)
                        .color(theme::TEXT_HINT),
                );
                ui.separator();
            });
        });

    let now = ctx.input(|i| i.time);
    let debounce_passed = (now - app.settings_opened_at) > 0.15;
    let close_via_backdrop = debounce_passed && theme::backdrop_clicked(ctx, &window_response);

    if !open || close_via_backdrop {
        app.close_settings(ctx);
    }
}
