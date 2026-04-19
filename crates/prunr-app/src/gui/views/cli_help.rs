use egui::{Align2, RichText};
use egui_material_icons::icons::*;

use crate::gui::theme;
use super::section_heading;

/// Returns true if the modal should close.
pub fn render(ctx: &egui::Context, toasts: &mut egui_notify::Toasts) -> bool {
    theme::draw_modal_backdrop(ctx, "cli_help_backdrop");

    let mut open = true;
    let window_response = egui::Window::new("CLI Reference")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([theme::SETTINGS_DIALOG_WIDTH, theme::SETTINGS_DIALOG_HEIGHT])
        .frame(theme::overlay_frame())
        .show(ctx, |ui| {
            {
                let vis = ui.visuals_mut();
                vis.widgets.inactive.bg_fill = theme::WIDGET_INACTIVE_BG;
                vis.widgets.inactive.fg_stroke =
                    egui::Stroke::new(theme::STROKE_DEFAULT, theme::TEXT_PRIMARY);
                vis.widgets.hovered.bg_fill = theme::WIDGET_HOVER_BG;
            }

            // Tab state
            let tab_id = egui::Id::new("cli_help_tab");
            let mut tab: usize = ui.data(|d| d.get_temp(tab_id).unwrap_or(0));

            ui.horizontal(|ui| {
                for (i, label) in ["Quick Start", "Lines", "Mask", "Advanced"].iter().enumerate() {
                    let selected = tab == i;
                    let text = RichText::new(*label)
                        .size(theme::FONT_SIZE_BODY)
                        .color(if selected { theme::TEXT_PRIMARY } else { theme::TEXT_SECONDARY });
                    let btn = egui::Button::new(text)
                        .fill(if selected { theme::BG_SECONDARY } else { egui::Color32::TRANSPARENT })
                        .corner_radius(theme::BUTTON_ROUNDING)
                        .min_size(egui::vec2(0.0, theme::CHIP_HEIGHT));
                    if ui.add(btn).clicked() {
                        tab = i;
                    }
                }
            });
            ui.separator();
            ui.add_space(theme::SPACE_SM);

            ui.data_mut(|d| d.insert_temp(tab_id, tab));

            match tab {
                // ── Quick Start ──
                0 => {
                    section_heading(ui, "Quick Start");
                    example_row(ui, toasts, "prunr photo.jpg",
                        "Remove background, save as photo_nobg.png");
                    example_row(ui, toasts, "prunr photo.jpg -o result.png",
                        "Choose where to save the result");
                    example_row(ui, toasts, "prunr *.jpg -o clean/",
                        "Process all JPEGs into a folder");

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Common Options");

                    egui::Grid::new("cli_opts_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_LG, theme::SPACE_SM])
                        .show(ui, |ui| {
                            opt_row(ui, "-o <path>", "Where to save (file or folder)");
                            opt_row(ui, "-m <model>", "silueta, u2net, or birefnet-lite");
                            opt_row(ui, "-j 4", "Process 4 images at once");
                            opt_row(ui, "-f", "Overwrite existing files");
                            opt_row(ui, "-q", "Silent mode (errors only)");
                            opt_row(ui, "--cpu", "Force CPU inference (skip GPU)");
                        });

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Examples");
                    example_row(ui, toasts, "prunr -m u2net portrait.jpg",
                        "Higher quality model");
                    example_row(ui, toasts, "prunr -j 8 -f *.jpg -o out/",
                        "Fast batch: 8 parallel, overwrite allowed");
                    example_row(ui, toasts, "prunr photo.jpg --bg-color ffffff",
                        "White background instead of transparent");
                }

                // ── Lines ──
                1 => {
                    section_heading(ui, "Line Extraction");
                    hint(ui, "Extract edges and outlines using DexiNed AI model.");
                    hint(ui, "Great for logos, graffiti, and illustrations.");

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Flags");

                    egui::Grid::new("cli_lines_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_LG, theme::SPACE_SM])
                        .show(ui, |ui| {
                            opt_row(ui, "--lines", "Extract lines only (skip BG removal)");
                            opt_row(ui, "--lines-after-bg", "Remove BG first, then extract lines");
                            opt_row(ui, "--line-strength <n>", "Detail level, 0.0\u{2013}1.0 (default 0.5)");
                            opt_row(ui, "--line-color <hex>", "Solid color for lines (e.g. 000000)");
                            opt_row(ui, "--line-scale <scale>", "Output scale: fine / balanced / bold / fused (default)");
                            opt_row(ui, "--bg-color <hex>", "Fill background with color (e.g. ffffff)");
                        });

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Examples");
                    example_row(ui, toasts, "prunr --lines logo.png",
                        "Extract outlines from a logo");
                    example_row(ui, toasts, "prunr --lines --line-color 000000 art.jpg",
                        "Black line art on transparent background");
                    example_row(ui, toasts, "prunr --lines-after-bg graffiti.jpg",
                        "Remove wall, then extract outlines");
                    example_row(ui, toasts, "prunr --lines --line-strength 0.8 photo.jpg",
                        "Fine detail lines (more edges)");
                    example_row(ui, toasts, "prunr --lines --line-color 333333 --bg-color eeeeee sketch.jpg",
                        "Dark gray lines on light gray background");
                    example_row(ui, toasts, "prunr --lines --line-scale bold sketch.jpg",
                        "Bold, abstracted outlines (DexiNed block5 output)");
                }

                // ── Mask ──
                2 => {
                    section_heading(ui, "Mask Tuning");
                    hint(ui, "Fine-tune how the background removal mask is generated.");
                    hint(ui, "These flags work with all models.");

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Flags");

                    egui::Grid::new("cli_mask_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_LG, theme::SPACE_SM])
                        .show(ui, |ui| {
                            opt_row(ui, "--gamma <n>",
                                "Removal strength (default 1.0, >1 aggressive)");
                            opt_row(ui, "--threshold <n>",
                                "Hard cutoff 0\u{2013}1 (off by default)");
                            opt_row(ui, "--edge-shift <n>",
                                "Trim (+) or expand (\u{2212}) edges, in px");
                            opt_row(ui, "--refine-edges",
                                "Guided filter for fine edge detail");
                        });

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Examples");
                    example_row(ui, toasts, "prunr --gamma 2.0 --threshold 0.5 photo.jpg",
                        "Aggressive removal with crisp edges");
                    example_row(ui, toasts, "prunr --gamma 0.5 --edge-shift -2 portrait.jpg",
                        "Gentle removal with expanded edges");
                    example_row(ui, toasts, "prunr --refine-edges photo.jpg",
                        "Sharpen mask around hair and leaves");
                    example_row(ui, toasts, "prunr --edge-shift 3 product.jpg",
                        "Trim 3px fringe around subject");
                }

                // ── Advanced ──
                3 => {
                    section_heading(ui, "Models");

                    egui::Grid::new("cli_models_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_LG, theme::SPACE_SM])
                        .show(ui, |ui| {
                            opt_row(ui, "silueta", "Fast, ~4 MB (default)");
                            opt_row(ui, "u2net", "Quality, ~170 MB");
                            opt_row(ui, "birefnet-lite", "Best detail, ~214 MB, 1024\u{00d7}1024");
                        });

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Large Image Handling");

                    egui::Grid::new("cli_large_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_LG, theme::SPACE_SM])
                        .show(ui, |ui| {
                            opt_row(ui, "--large-image downscale", "Auto-shrink to 4096px (default)");
                            opt_row(ui, "--large-image process", "Process at full resolution");
                        });

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Examples");
                    example_row(ui, toasts, "prunr -m birefnet-lite portrait.jpg",
                        "Best detail for hair, leaves, fine edges");
                    example_row(ui, toasts, "prunr --large-image process poster.png",
                        "Full-resolution processing for large images");
                    example_row(ui, toasts, "prunr -q photo.jpg -o output.png",
                        "Quiet mode for scripting");
                }

                _ => {}
            }

            // Footer
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.label(
                    RichText::new("Press F2 to close")
                        .size(theme::FONT_SIZE_MONO)
                        .color(theme::TEXT_HINT),
                );
            });
        });

    !open || theme::backdrop_clicked(ctx, &window_response)
}

use super::hint;

fn example_row(ui: &mut egui::Ui, toasts: &mut egui_notify::Toasts, cmd: &str, desc: &str) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(cmd)
                .monospace()
                .size(theme::FONT_SIZE_MONO)
                .color(theme::TEXT_PRIMARY),
        );
        let flash_id = egui::Id::new(("copy_flash", cmd));
        let flash = ui.ctx().animate_bool_with_time(flash_id, false, 0.4);
        let icon_color = if flash > 0.01 { egui::Color32::WHITE } else { theme::TEXT_HINT };
        let icon_text = if flash > 0.01 {
            ICON_CHECK.codepoint.to_string()
        } else {
            ICON_CONTENT_COPY.codepoint.to_string()
        };
        if ui.add(
            egui::Button::new(RichText::new(icon_text).size(12.0).color(icon_color))
                .frame(false)
                .min_size(egui::vec2(18.0, 18.0)),
        ).on_hover_text("Copy to clipboard").clicked() {
            ui.ctx().copy_text(cmd.to_string());
            ui.ctx().animate_bool_with_time(flash_id, true, 0.0);
            toasts.info("Copied to clipboard");
        }
    });
    ui.label(
        RichText::new(format!("  {desc}"))
            .size(theme::FONT_SIZE_BODY)
            .color(theme::TEXT_PRIMARY),
    );
    ui.add_space(theme::SPACE_XS);
}

fn opt_row(ui: &mut egui::Ui, flag: &str, desc: &str) {
    ui.label(
        RichText::new(flag)
            .monospace()
            .size(theme::FONT_SIZE_MONO)
            .color(theme::TEXT_PRIMARY),
    );
    ui.label(
        RichText::new(desc)
            .size(theme::FONT_SIZE_BODY)
            .color(theme::TEXT_PRIMARY),
    );
    ui.end_row();
}
