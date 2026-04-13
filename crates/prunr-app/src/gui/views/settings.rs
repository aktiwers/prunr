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

/// Checkbox + color swatch + hex input on one row.
fn color_toggle_row(ui: &mut egui::Ui, enabled: &mut bool, label: &str, color: &mut [u8; 4]) {
    ui.horizontal(|ui| {
        ui.checkbox(
            enabled,
            RichText::new(label)
                .color(theme::TEXT_PRIMARY)
                .size(theme::FONT_SIZE_BODY),
        );
        if *enabled {
            let mut rgb = [
                color[0] as f32 / 255.0,
                color[1] as f32 / 255.0,
                color[2] as f32 / 255.0,
            ];
            if ui.color_edit_button_rgb(&mut rgb).changed() {
                color[0] = (rgb[0] * 255.0).round() as u8;
                color[1] = (rgb[1] * 255.0).round() as u8;
                color[2] = (rgb[2] * 255.0).round() as u8;
            }
            // Hex input (persisted while typing, applied on focus loss)
            let hex_id = ui.id().with("hex");
            let mut hex = ui.data_mut(|d| {
                d.get_temp::<String>(hex_id)
                    .unwrap_or_else(|| format!("{:02X}{:02X}{:02X}", color[0], color[1], color[2]))
            });
            let response = ui.add_sized(
                [62.0, 18.0],
                egui::TextEdit::singleline(&mut hex)
                    .font(egui::TextStyle::Monospace)
                    .char_limit(7),
            );
            if response.lost_focus() {
                let clean = hex.trim_start_matches('#');
                if clean.len() == 6 {
                    if let (Ok(r), Ok(g), Ok(b)) = (
                        u8::from_str_radix(&clean[0..2], 16),
                        u8::from_str_radix(&clean[2..4], 16),
                        u8::from_str_radix(&clean[4..6], 16),
                    ) {
                        color[0] = r; color[1] = g; color[2] = b;
                    }
                }
                ui.data_mut(|d| d.remove::<String>(hex_id));
            } else if response.has_focus() {
                ui.data_mut(|d| d.insert_temp(hex_id, hex));
            } else {
                ui.data_mut(|d| d.remove::<String>(hex_id));
            }
        }
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

            use crate::gui::settings::LineMode;

            // Tab state (persisted in egui temp data)
            let tab_id = egui::Id::new("settings_tab");
            let mut tab: usize = ui.data(|d| d.get_temp(tab_id).unwrap_or(0));

            // Tab bar + reset buttons on the same row
            ui.horizontal(|ui| {
                for (i, label) in ["General", "Lines", "Mask"].iter().enumerate() {
                    let selected = tab == i;
                    let text = RichText::new(*label)
                        .size(theme::FONT_SIZE_BODY)
                        .color(if selected { theme::TEXT_PRIMARY } else { theme::TEXT_SECONDARY });
                    let btn = egui::Button::new(text)
                        .fill(if selected { theme::BG_SECONDARY } else { egui::Color32::TRANSPARENT })
                        .corner_radius(4.0)
                        .min_size(egui::vec2(0.0, 28.0));
                    if ui.add(btn).clicked() {
                        tab = i;
                    }
                }
                // Push reset buttons to the right
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button(
                        RichText::new("Reset all")
                            .size(theme::FONT_SIZE_MONO)
                            .color(theme::TEXT_SECONDARY),
                    ).clicked() {
                        let backend = app.settings.active_backend.clone();
                        app.settings = Settings::default();
                        app.settings.active_backend = backend;
                        app.settings.parallel_jobs = app.settings.default_jobs();
                    }
                    let tab_name = ["General", "Lines", "Mask"][tab.min(2)];
                    let defaults = Settings::default();
                    if ui.small_button(
                        RichText::new(format!("Reset {tab_name}"))
                            .size(theme::FONT_SIZE_MONO)
                            .color(theme::TEXT_SECONDARY),
                    ).clicked() {
                        match tab {
                            0 => {
                                app.settings.parallel_jobs = app.settings.default_jobs();
                                app.settings.auto_remove_on_import = defaults.auto_remove_on_import;
                                app.settings.force_cpu = defaults.force_cpu;
                                app.settings.chain_mode = defaults.chain_mode;
                                app.settings.dark_checker = defaults.dark_checker;
                                app.settings.history_depth = defaults.history_depth;
                                app.settings.apply_bg_color = defaults.apply_bg_color;
                                app.settings.bg_color = defaults.bg_color;
                            }
                            1 => {
                                app.settings.line_mode = defaults.line_mode;
                                app.settings.line_strength = defaults.line_strength;
                                app.settings.solid_line_color = defaults.solid_line_color;
                                app.settings.line_color = defaults.line_color;
                            }
                            2 => {
                                app.settings.mask_gamma = defaults.mask_gamma;
                                app.settings.mask_threshold = defaults.mask_threshold;
                                app.settings.mask_threshold_enabled = defaults.mask_threshold_enabled;
                                app.settings.edge_shift = defaults.edge_shift;
                                app.settings.refine_edges = defaults.refine_edges;
                            }
                            _ => {}
                        }
                    }
                });
            });
            ui.separator();
            ui.add_space(theme::SPACE_SM);

            ui.data_mut(|d| d.insert_temp(tab_id, tab));

            match tab {
                // ── General ──
                0 => {
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
                        &mut app.settings.dark_checker,
                        RichText::new("Dark checkerboard")
                            .color(theme::TEXT_PRIMARY)
                            .size(theme::FONT_SIZE_BODY),
                    );
                    hint(ui, "Use dark tones for the transparency pattern — helps when viewing light results.");
                    ui.add_space(theme::SPACE_MD);

                    // Force CPU only makes sense when a GPU is actually in play.
                    // On CPU-only systems the checkbox is a no-op; the status bar
                    // already shows "Backend: CPU".
                    if app.settings.is_gpu() {
                        ui.checkbox(
                            &mut app.settings.force_cpu,
                            RichText::new("Force CPU")
                                .color(theme::TEXT_PRIMARY)
                                .size(theme::FONT_SIZE_BODY),
                        );
                        hint(ui, "Use CPU even when GPU is available (resets each launch)");
                        ui.add_space(theme::SPACE_MD);
                    }
                    ui.separator();
                    ui.add_space(theme::SPACE_SM);

                    section_heading(ui, "Background Color");
                    hint(ui, "Fill transparent areas with a solid color.");
                    ui.add_space(theme::SPACE_SM);
                    color_toggle_row(ui, &mut app.settings.apply_bg_color, "Apply background color", &mut app.settings.bg_color);

                    ui.add_space(theme::SPACE_MD);
                    ui.separator();
                    ui.add_space(theme::SPACE_SM);

                    section_heading(ui, "History");
                    ui.add_space(theme::SPACE_SM);
                    ui.checkbox(
                        &mut app.settings.chain_mode,
                        RichText::new("Chain mode")
                            .color(theme::TEXT_PRIMARY)
                            .size(theme::FONT_SIZE_BODY),
                    );
                    hint(ui, "Process the current result instead of the original.");
                    hint(ui, "Allows stacking effects: BG removal -> lines -> etc.");

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
                }

                // ── Lines ──
                1 => {
                    hint(ui, "Keep only edges and outlines \u{2014} great for logos,");
                    hint(ui, "graffiti, and illustrations. Uses DexiNed AI model.");
                    ui.add_space(theme::SPACE_SM);

                    let mode_label = match app.settings.line_mode {
                        LineMode::Off => "Off",
                        LineMode::LinesOnly => "Lines only",
                        LineMode::AfterBgRemoval => "After BG removal",
                    };
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("Extract lines")
                                .color(theme::TEXT_PRIMARY)
                                .size(theme::FONT_SIZE_BODY),
                        );
                        egui::ComboBox::from_id_salt("line_mode")
                            .selected_text(mode_label)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut app.settings.line_mode, LineMode::Off, "Off");
                                ui.selectable_value(&mut app.settings.line_mode, LineMode::LinesOnly, "Lines only");
                                ui.selectable_value(&mut app.settings.line_mode, LineMode::AfterBgRemoval, "After BG removal");
                            });
                    });
                    hint(ui, "Off = normal background removal.");
                    hint(ui, "Lines only = extract edges, skip BG removal.");
                    hint(ui, "After BG removal = remove BG, then extract lines.");

                    if app.settings.line_mode != LineMode::Off {
                        ui.add_space(theme::SPACE_MD);
                        let strength_text = format!("{:.2}", app.settings.line_strength);
                        slider_row(
                            ui, "Line strength", &mut app.settings.line_strength,
                            0.05..=1.0, &strength_text, false, None,
                        );
                        hint(ui, "How much detail to capture. Lower = bold outlines");
                        hint(ui, "only, higher = fine texture and subtle edges.");
                        ui.add_space(theme::SPACE_SM);
                        color_toggle_row(ui, &mut app.settings.solid_line_color, "Solid color lines", &mut app.settings.line_color);
                        hint(ui, "Paint all lines a single color instead of original.");
                    }
                }

                // ── Mask ──
                2 => {
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
                }

                _ => {}
            }

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
        app.close_settings();
    }
}
