use egui::{Align2, RichText};
use egui_material_icons::icons::*;

use crate::gui::theme;

/// Returns true if the modal should close.
pub fn render(ctx: &egui::Context, toasts: &mut egui_notify::Toasts) -> bool {
    theme::draw_modal_backdrop(ctx, "cli_help_backdrop");

    let mut open = true;
    let window_response = egui::Window::new("CLI Reference")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([420.0, 800.0])
        .frame(theme::overlay_frame())
        .show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.add_space(theme::SPACE_SM);

                section_heading(ui, "Quick Start");
                example_row(ui, toasts, "prunr photo.jpg",
                    "Remove background and save as photo_nobg.png");
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
                    });

                ui.add_space(theme::SPACE_MD);
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
                section_heading(ui, "Examples");

                example_row(ui, toasts, "prunr -m birefnet-lite portrait.jpg",
                    "Best detail for hair, leaves, fine edges");
                example_row(ui, toasts, "prunr -m u2net portrait.jpg",
                    "Higher quality than default");
                example_row(ui, toasts, "prunr -j 8 -f *.jpg -o out/",
                    "Fast batch: 8 parallel, overwrite allowed");
                example_row(ui, toasts, "prunr --refine-edges photo.jpg",
                    "Sharpen mask edges using image colors");
                example_row(ui, toasts, "prunr --gamma 2.0 --threshold 0.5 photo.jpg",
                    "Aggressive removal with crisp edges");

                ui.add_space(theme::SPACE_MD);
                section_heading(ui, "Mask Tuning Flags");

                egui::Grid::new("cli_mask_grid")
                    .num_columns(2)
                    .spacing([theme::SPACE_LG, theme::SPACE_SM])
                    .show(ui, |ui| {
                        opt_row(ui, "--gamma <n>",
                            "Removal strength (default 1.0)");
                        opt_row(ui, "--threshold <n>",
                            "Hard cutoff, 0\u{2013}1 (off by default)");
                        opt_row(ui, "--edge-shift <n>",
                            "Trim (+) or expand (\u{2212}) edges, in px");
                        opt_row(ui, "--refine-edges",
                            "Guided filter for fine edge detail");
                        opt_row(ui, "--cpu",
                            "Force CPU inference (skip GPU)");
                    });

                ui.add_space(theme::SPACE_LG);
                ui.label(
                    RichText::new("Press F2 to close")
                        .size(theme::FONT_SIZE_MONO)
                        .color(theme::TEXT_HINT),
                );
            });
        });

    !open || theme::clicked_outside_modal(window_response)
}

use super::section_heading;

fn example_row(ui: &mut egui::Ui, toasts: &mut egui_notify::Toasts, cmd: &str, desc: &str) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(cmd)
                .monospace()
                .size(theme::FONT_SIZE_MONO)
                .color(theme::TEXT_PRIMARY),
        );
        // Flash animation: bright white on click, fades back to hint color
        let flash_id = egui::Id::new(("copy_flash", cmd));
        let flash = ui.ctx().animate_bool_with_time(flash_id, false, 0.4);
        let icon_color = if flash > 0.01 {
            egui::Color32::WHITE
        } else {
            theme::TEXT_HINT
        };
        let icon_text = if flash > 0.01 {
            ICON_CHECK.codepoint.to_string()
        } else {
            ICON_CONTENT_COPY.codepoint.to_string()
        };
        if ui.add(
            egui::Button::new(
                RichText::new(icon_text)
                    .size(12.0)
                    .color(icon_color),
            )
            .frame(false)
            .min_size(egui::vec2(18.0, 18.0)),
        ).on_hover_text("Copy to clipboard").clicked() {
            ui.ctx().copy_text(cmd.to_string());
            // Trigger flash: set to true then immediately back to false so it animates out
            ui.ctx().animate_bool_with_time(flash_id, true, 0.0);
            toasts.info("Copied to clipboard");
        }
    });
    ui.label(
        RichText::new(format!("  {desc}"))
            .size(theme::FONT_SIZE_BODY)
            .color(theme::TEXT_HINT),
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
            .color(theme::TEXT_HINT),
    );
    ui.end_row();
}
