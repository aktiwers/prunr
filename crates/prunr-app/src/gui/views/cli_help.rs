use egui::RichText;
use egui_material_icons::icons::*;

use crate::gui::theme;
use super::{kv_row, section_heading};

/// Room kept under the scrolling page for the "Press F2 to close" line.
const FOOTER_HEIGHT: f32 = 28.0;

#[derive(Clone, Copy)]
enum CliTab { QuickStart, Sketch, Mask, Eraser, Upscale, Advanced }

impl CliTab {
    const ALL: [Self; 6] = [Self::QuickStart, Self::Sketch, Self::Mask, Self::Eraser, Self::Upscale, Self::Advanced];

    fn label(self) -> &'static str {
        match self {
            Self::QuickStart => "Quick start",
            Self::Sketch => "Sketch",
            Self::Mask => "Mask",
            Self::Eraser => "Eraser",
            Self::Upscale => "Upscale",
            Self::Advanced => "Advanced",
        }
    }
}

/// Returns true if the modal should close.
pub(crate) fn render(ctx: &egui::Context, toasts: &mut crate::gui::toasts::Toasts) -> bool {
    theme::bare_modal_window(
        ctx, "cli_help", "Command-line reference",
        [theme::STANDARD_MODAL_WIDTH, theme::CLI_HELP_DIALOG_HEIGHT],
        |ui| super::chip::modal_page(ui, "Command-line reference", "", |ui| {
            theme::apply_modal_visuals(ui);

            let labels = CliTab::ALL.map(CliTab::label);
            let picked = super::chip::remembered_tab_strip(ui, egui::Id::new("cli_help_tab"), &labels);
            let tab = CliTab::ALL.get(picked).copied().unwrap_or(CliTab::QuickStart);
            ui.separator();
            ui.add_space(theme::SPACE_SM);

            // A short window: the page scrolls above its footer line.
            let page_height = (ui.available_height() - FOOTER_HEIGHT).max(0.0);
            egui::ScrollArea::vertical().max_height(page_height).auto_shrink([false, false]).show(ui, |ui| {
            match tab {
                CliTab::QuickStart => {
                    section_heading(ui, "Quick start");
                    example_row(ui, toasts, "prunr photo.jpg",
                        "Remove background, save as photo.prunr.png");
                    example_row(ui, toasts, "prunr photo.jpg -o result.png",
                        "Choose where to save the result");
                    example_row(ui, toasts, "prunr *.jpg -o clean/",
                        "Process all JPEGs into a folder");

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Common options");

                    egui::Grid::new("cli_opts_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_LG, theme::SPACE_SM])
                        .show(ui, |ui| {
                            opt_row(ui, "--output, -o <path>", "Where to save (file or folder)");
                            opt_row(ui, "--model, -m <model>", "silueta, u2net, or birefnet-lite");
                            opt_row(ui, "--jobs, -j 4", "Process 4 images at once");
                            opt_row(ui, "--force, -f", "Overwrite existing files");
                            opt_row(ui, "--quiet, -q", "Silent mode (errors only)");
                            opt_row(ui, "--cpu", "Run every model on the CPU");
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

                CliTab::Sketch => {
                    section_heading(ui, "Line extraction");
                    hint(ui, "Trace edges and outlines with the line-detection model.");
                    hint(ui, "Great for logos, graffiti, and illustrations.");

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Flags");

                    egui::Grid::new("cli_lines_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_LG, theme::SPACE_SM])
                        .show(ui, |ui| {
                            opt_row(ui, "--lines", "Extract lines only (skip background removal)");
                            opt_row(ui, "--lines-after-bg", "Remove the background first, then extract lines");
                            opt_row(ui, "--line-detail <n>", "Detail level, 0.0\u{2013}1.0 (default 0.5)");
                            opt_row(ui, "--line-width <px>", "Line width, 0\u{2013}20 px (default 0)");
                            opt_row(ui, "--line-composition <mode>", "With --lines-after-bg: lines-only / subject-filled / engraving / ghost");
                            opt_row(ui, "--line-color <hex>", "Solid color for lines (e.g. 000000)");
                            opt_row(ui, "--line-scale <scale>", "Line scale: fine / bold / fused (default)");
                            opt_row(ui, "--line-fade <0-100>", "Lines lighten with distance (needs the depth model)");
                            opt_row(ui, "--bg-color <hex>", "Fill background with color (e.g. ffffff)");
                            opt_row(ui, "--bg-image <path>", "Fill background with an image");
                            opt_row(ui, "--bg-image-fit <fit>", "cover (default) / contain / tile");
                            opt_row(ui, "--effect <name>", "Restyle the subject's colors: sepia, invert, pixelate, halftone, duotone\u{2026}");
                            opt_row(ui, "--outline <px[:hex]>", "Sticker outline around the subject, e.g. 8 or 8:000000");
                            opt_row(ui, "--shadow <kind>", "Shadow under the subject: soft or drop");
                            opt_row(ui, "--bg-effect <effect>", "Photo behind: blur, blur:N (1\u{2013}64), lens, lens:N (0\u{2013}100, needs the depth model), bw");
                        });

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Examples");
                    example_row(ui, toasts, "prunr --lines logo.png",
                        "Extract outlines from a logo");
                    example_row(ui, toasts, "prunr --lines --line-color 000000 art.jpg",
                        "Black line art on transparent background");
                    example_row(ui, toasts, "prunr --lines-after-bg graffiti.jpg",
                        "Remove wall, then extract outlines");
                    example_row(ui, toasts, "prunr --lines --line-detail 0.8 photo.jpg",
                        "Fine detail lines (more edges)");
                    example_row(ui, toasts, "prunr --lines --line-color 333333 --bg-color eeeeee sketch.jpg",
                        "Dark gray lines on light gray background");
                    example_row(ui, toasts, "prunr --lines --line-scale bold sketch.jpg",
                        "Bold, abstracted outlines");
                }

                CliTab::Mask => {
                    section_heading(ui, "Mask tuning");
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
                            opt_row(ui, "--contrast <n>",
                                "Edge contrast 0\u{2013}1, 1 is a hard edge (default 0)");
                            opt_row(ui, "--edge-shift <n>",
                                "Grow (+) or shrink (\u{2212}) the edge, in px");
                            opt_row(ui, "--refine-edges",
                                "Guided filter for fine edge detail");
                            opt_row(ui, "--edge-width <n>",
                                "How far refinement looks from the edge, in px (default 8)");
                            opt_row(ui, "--refine-guide <g>",
                                "What refinement follows: brightness (default) or color");
                            opt_row(ui, "--feather <n>",
                                "Soften the edge by n px; fractions allowed");
                        });

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Examples");
                    example_row(ui, toasts, "prunr --gamma 2.0 --contrast 1 photo.jpg",
                        "Aggressive removal with crisp edges");
                    example_row(ui, toasts, "prunr --gamma 0.5 --edge-shift 2 portrait.jpg",
                        "Gentle removal with expanded edges");
                    example_row(ui, toasts, "prunr --refine-edges photo.jpg",
                        "Sharpen mask around hair and leaves");
                    example_row(ui, toasts, "prunr --edge-shift -3 product.jpg",
                        "Trim a 3 px fringe around the subject");
                }

                CliTab::Eraser => {
                    section_heading(ui, "Object removal (eraser)");
                    hint(ui, "Paint over an unwanted object and let the eraser fill it in.");
                    hint(ui, "Pass a binary mask (white = remove here, black = keep).");

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Flags");

                    egui::Grid::new("cli_eraser_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_LG, theme::SPACE_SM])
                        .show(ui, |ui| {
                            opt_row(ui, "--inpaint", "Switch to Eraser mode");
                            opt_row(ui, "--mask <path>", "Binary mask, must match input dimensions");
                            opt_row(ui, "--fill-scale <scale>", "auto, full, half or quarter: large areas fill more convincingly smaller (default auto)");
                        });

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Examples");
                    example_row(ui, toasts, "prunr --inpaint photo.jpg --mask mask.png",
                        "Erase region defined by mask.png, save photo_erased.png");
                    example_row(ui, toasts, "prunr --inpaint photo.jpg --mask mask.png -o clean.png",
                        "Custom output path");
                }

                CliTab::Upscale => {
                    section_heading(ui, "Upscale");
                    hint(ui, "Enlarge a photo with an upscale model, refine it, and restore faces.");
                    hint(ui, "Models download in the app's Model Store.");

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Flags");

                    egui::Grid::new("cli_upscale_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_LG, theme::SPACE_SM])
                        .show(ui, |ui| {
                            opt_row(ui, "--upscale <model>", "realesrgan, nomos, siax or superscale: switch to Upscale mode");
                            opt_row(ui, "--scale <2|3|4>", "How much larger (default 4)");
                            opt_row(ui, "--denoise <0-1>", "Denoise before the model (default 0)");
                            opt_row(ui, "--sharpen <-1-1>", "Sharpen after the model; negative softens (default 0)");
                            opt_row(ui, "--ai-blend <0-1>", "1 keeps the model's output, 0 the plain enlargement (default 1)");
                            opt_row(ui, "--color-match", "Match the upscale's colors to the original");
                            opt_row(ui, "--restore-faces <model>", "gfpgan or restoreformer: redraw soft faces after the upscale");
                            opt_row(ui, "--face-strength <0-100>", "Up to 50 adds detail over the photo's shading (default 50)");
                        });

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Examples");
                    example_row(ui, toasts, "prunr --upscale realesrgan photo.jpg",
                        "Enlarge 4\u{00d7}, save photo_x4.png");
                    example_row(ui, toasts, "prunr --upscale realesrgan --scale 2 --restore-faces gfpgan portrait.jpg",
                        "Enlarge 2\u{00d7} and restore the faces");

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Colorize");
                    egui::Grid::new("cli_colorize_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_LG, theme::SPACE_SM])
                        .show(ui, |ui| {
                            opt_row(ui, "--colorize", "Color a black-and-white photo with DDColor");
                            opt_row(ui, "--color-strength <0-1.5>", "0 is black and white, 1 the model's colors (default 1)");
                        });
                    example_row(ui, toasts, "prunr --colorize old_photo.jpg",
                        "Color it, save old_photo_color.png");

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Depth");
                    egui::Grid::new("cli_depth_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_LG, theme::SPACE_SM])
                        .show(ui, |ui| {
                            opt_row(ui, "--depth-map", "Also save {stem}-depth.png: 16-bit gray, white = near");
                            opt_row(ui, "--depth-only", "With --depth-map: save only the depth maps");
                        });
                    example_row(ui, toasts, "prunr --depth-map --depth-only landscape.jpg",
                        "Save landscape-depth.png only");
                }

                CliTab::Advanced => {
                    section_heading(ui, "Models");

                    egui::Grid::new("cli_models_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_LG, theme::SPACE_SM])
                        .show(ui, |ui| {
                            opt_row(ui, "birefnet-lite", "Best detail, ~214 MB, 1024\u{00d7}1024 (default)");
                            opt_row(ui, "silueta", "Fast, ~4 MB");
                            opt_row(ui, "u2net", "Quality, ~170 MB");
                        });

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Large images");

                    egui::Grid::new("cli_large_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_LG, theme::SPACE_SM])
                        .show(ui, |ui| {
                            opt_row(ui, "--large-image process", "Keep photos over 8000 px at full size (default)");
                            opt_row(ui, "--large-image downscale", "Shrink photos over 8000 px to 4096 px first");
                        });

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Diagnostics and workflow");

                    egui::Grid::new("cli_diag_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_LG, theme::SPACE_SM])
                        .show(ui, |ui| {
                            opt_row(ui, "--open <path>", "Start the app with this image loaded");
                            opt_row(ui, "--doctor", "Print hardware, runtime, model and path facts, then exit");
                            opt_row(ui, "--clear-ep-cache", "Forget which accelerators failed which models (after a driver update)");
                            opt_row(ui, "--debug", "Verbose tracing (use when reporting bugs)");
                            opt_row(ui, "bench", "Time each model on the CPU and the GPU, check they agree, write prunr-bench.md");
                        });

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Examples");
                    example_row(ui, toasts, "prunr -m birefnet-lite portrait.jpg",
                        "Best detail for hair, leaves, fine edges");
                    example_row(ui, toasts, "prunr --large-image downscale poster.png",
                        "A faster, smaller result for a very large image");
                    example_row(ui, toasts, "prunr -q photo.jpg -o output.png",
                        "Quiet mode for scripting");
                    example_row(ui, toasts, "prunr --debug photo.jpg 2> prunr.log",
                        "Capture diagnostic log for bug reports");
                    example_row(ui, toasts, "prunr bench --models silueta,migan --runs 5",
                        "Check two models on the GPU against the CPU");
                }
            }
            });

            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.label(
                    RichText::new(format!("Press {} to close", super::shortcuts::keys(ui.ctx(), super::shortcuts::Action::CliHelp)))
                        .size(theme::FONT_SIZE_SECONDARY)
                        .color(theme::TEXT_HINT),
                );
            });
        }),
    )
}

use super::hint;

fn example_row(ui: &mut egui::Ui, toasts: &mut crate::gui::toasts::Toasts, cmd: &str, desc: &str) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(cmd)
                .monospace()
                .size(theme::FONT_SIZE_SECONDARY)
                .color(theme::TEXT_PRIMARY),
        );
        let flash_id = egui::Id::new(("copy_flash", cmd));
        let flash = ui.ctx().animate_bool_with_time(flash_id, false, 0.4);
        let icon_color = if flash > 0.01 { egui::Color32::WHITE } else { theme::TEXT_HINT };
        let icon_text = if flash > 0.01 {
            ICON_CHECK.codepoint
        } else {
            ICON_CONTENT_COPY.codepoint
        };
        let copy = ui.add(
            egui::Button::new(RichText::new(icon_text).size(theme::FONT_SIZE_SECONDARY).color(icon_color))
                .frame(false)
                .min_size(egui::vec2(18.0, 18.0)),
        );
        if super::chip::named(copy, "Copy to clipboard").on_hover_text("Copy to clipboard").clicked() {
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
    kv_row(ui, flag, desc, theme::TEXT_PRIMARY);
}
