//! F3 modal: the processing order of each mode, one tab per pipeline,
//! with a one-line explanation per step so the user can predict how a
//! knob change cascades. Step names match the toolbar rows.

use egui::RichText;
use egui_material_icons::icons::*;

use super::chip;
use crate::gui::theme;

const MODAL_WIDTH: f32 = 520.0;
const MODAL_HEIGHT: f32 = 600.0;

struct Stage {
    icon: &'static str,
    title: &'static str,
    tagline: &'static str,
}

struct Pipeline {
    tab: &'static str,
    input: &'static str,
    output: &'static str,
    stages: &'static [Stage],
}

const PIPELINES: &[Pipeline] = &[
    Pipeline {
        tab: "Mask",
        input: "Model output: a soft mask of the subject",
        output: "Final mask, applied as transparency",
        stages: &[
            Stage { icon: ICON_TONALITY.codepoint, title: "Gamma", tagline: "How hard the mask cuts. Every step below works on this result, so a higher gamma hands them a darker silhouette." },
            Stage { icon: ICON_BOLT.codepoint, title: "Hard threshold", tagline: "Optional. Snaps every pixel to kept or removed. The steps below then see hard edges: Refine can only clean stair-steps and Feather blurs a step." },
            Stage { icon: ICON_SWAP_HORIZ.codepoint, title: "Edge shift", tagline: "Erodes or dilates the mask boundary. Refine edges then snaps the shifted boundary to the photo's color edges." },
            Stage { icon: ICON_AUTO_FIX_HIGH.codepoint, title: "Refine edges", tagline: "Optional. Uses the photo's colors to snap the mask to real edges. A tighter input gives a tighter result." },
            Stage { icon: ICON_BLUR_LINEAR.codepoint, title: "Feather", tagline: "Softens the finished mask. Runs last on purpose: sharpen first, then soften." },
        ],
    },
    Pipeline {
        tab: "Lines",
        input: "The photo, or the cut-out subject when Sketch is Subject",
        output: "Lines, composed with the subject as chosen",
        stages: &[
            Stage { icon: ICON_DRAW.codepoint, title: "Sketch", tagline: "What the line detector sees: nothing, the subject only, or the whole photo." },
            Stage { icon: ICON_FILTER_B_AND_W.codepoint, title: "Pre-filter", tagline: "Grayscale, contrast or posterize applied before detection. Requires reprocessing." },
            Stage { icon: ICON_TUNE.codepoint, title: "Scale", tagline: "How zoomed-in the detector looks: fine texture, bold silhouettes, or every scale fused." },
            Stage { icon: ICON_TUNE.codepoint, title: "Line detail", tagline: "How much of the detected edge map becomes a line." },
            Stage { icon: ICON_LINE_WEIGHT.codepoint, title: "Line thickness", tagline: "Thickens the lines by whole pixels." },
            Stage { icon: ICON_LAYERS.codepoint, title: "Composition and color", tagline: "How the lines combine with the subject, and what color they take." },
        ],
    },
    Pipeline {
        tab: "Eraser",
        input: "The painted region",
        output: "The photo with the region filled",
        stages: &[
            Stage { icon: ICON_BRUSH.codepoint, title: "Paint region", tagline: "Everything under the brush is replaced." },
            Stage { icon: ICON_OPEN_IN_FULL.codepoint, title: "Expand region", tagline: "Grows or shrinks the region first; a little growth gives the fill more context." },
            Stage { icon: ICON_AUTO_AWESOME.codepoint, title: "Fill", tagline: "The eraser model invents what belongs there. SD models follow the prompt." },
            Stage { icon: ICON_BLUR_LINEAR.codepoint, title: "Edge blend", tagline: "Blends the fill into the photo across a band at the boundary." },
            Stage { icon: ICON_DEBLUR.codepoint, title: "Sharpen", tagline: "Sharpens the filled area, which comes out slightly soft." },
        ],
    },
    Pipeline {
        tab: "Upscale",
        input: "The photo, or the current result in chain mode",
        output: "The enlarged photo",
        stages: &[
            Stage { icon: ICON_BLUR_ON.codepoint, title: "Denoise", tagline: "Smooths grain first so it is not enlarged along with the detail. Requires reprocessing." },
            Stage { icon: ICON_BRIGHTNESS_6.codepoint, title: "Brightness lift", tagline: "Brightens shadows for the model, then restores the original brightness. Requires reprocessing." },
            Stage { icon: ICON_ARROW_UPWARD.codepoint, title: "Upscale", tagline: "The model enlarges the photo tile by tile at the chosen scale." },
            Stage { icon: ICON_PSYCHOLOGY.codepoint, title: "AI blend", tagline: "Mixes the plain enlargement back in to soften a plastic look." },
            Stage { icon: ICON_DEBLUR.codepoint, title: "Sharpen", tagline: "Negative softens, positive sharpens." },
            Stage { icon: ICON_PALETTE.codepoint, title: "Saturation and color match", tagline: "Adjusts color intensity, then matches the result's color balance to the photo." },
        ],
    },
];

/// Returns true if the modal should close.
pub fn render(ctx: &egui::Context) -> bool {
    theme::standard_modal_window(
        ctx, "pipeline_flow", "Pipelines",
        [MODAL_WIDTH, MODAL_HEIGHT],
        |ui| {
            let tab_id = egui::Id::new("pipeline_flow_tab");
            let mut tab: usize = ui.data(|d| d.get_temp(tab_id).unwrap_or(0));
            let labels = [PIPELINES[0].tab, PIPELINES[1].tab, PIPELINES[2].tab, PIPELINES[3].tab];
            chip::tab_strip(ui, &labels, &mut tab);
            ui.data_mut(|d| d.insert_temp(tab_id, tab));
            ui.separator();

            let pipeline = &PIPELINES[tab.min(PIPELINES.len() - 1)];
            ui.vertical(|ui| {
                ui.add_space(theme::SPACE_SM);
                endpoint_label(ui, pipeline.input);
                arrow(ui);
                for (i, stage) in pipeline.stages.iter().enumerate() {
                    stage_row(ui, i + 1, pipeline.stages.len(), stage);
                    arrow(ui);
                }
                endpoint_label(ui, pipeline.output);

                ui.add_space(theme::SPACE_MD);
                ui.separator();
                ui.add_space(theme::SPACE_SM);
                ui.label(
                    RichText::new(
                        "The order is fixed. Each step works on the result of the one above it, \
                         so a knob changes what every later step has to work with.",
                    )
                    .size(theme::FONT_SIZE_MONO)
                    .color(theme::TEXT_PRIMARY),
                );
            });
        },
    )
}

fn stage_row(ui: &mut egui::Ui, number: usize, total: usize, stage: &Stage) {
    ui.horizontal(|ui| {
        ui.add_space(theme::SPACE_SM);
        ui.label(
            RichText::new(stage.icon)
                .size(theme::ICON_SIZE_BUTTON)
                .color(theme::TEXT_PRIMARY),
        );
        ui.add_space(theme::SPACE_SM);
        ui.vertical(|ui| {
            ui.label(
                RichText::new(format!("Step {number} of {total}"))
                    .size(theme::FONT_SIZE_MONO)
                    .color(theme::TEXT_SECONDARY),
            );
            ui.label(
                RichText::new(stage.title)
                    .strong()
                    .size(theme::FONT_SIZE_BODY)
                    .color(theme::TEXT_PRIMARY),
            );
            ui.label(
                RichText::new(stage.tagline)
                    .size(theme::FONT_SIZE_MONO)
                    .color(theme::TEXT_PRIMARY),
            );
        });
    });
}

fn arrow(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.add_space(theme::SPACE_SM + theme::ICON_SIZE_BUTTON / 2.0 - 4.0);
        ui.label(
            RichText::new(ICON_ARROW_DOWNWARD.codepoint)
                .size(theme::FONT_SIZE_BODY)
                .color(theme::TEXT_SECONDARY),
        );
    });
}

fn endpoint_label(ui: &mut egui::Ui, text: &str) {
    ui.horizontal(|ui| {
        ui.add_space(theme::SPACE_SM);
        ui.label(
            RichText::new(text)
                .size(theme::FONT_SIZE_MONO)
                .color(theme::TEXT_SECONDARY),
        );
    });
}
