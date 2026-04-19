//! F3 modal: mask-pipeline reference diagram. Five fixed stages with a
//! one-line explanation each, so the user can predict how knob tweaks
//! cascade.

use egui::{Align2, RichText};
use egui_material_icons::icons::*;

use crate::gui::theme;

const MODAL_WIDTH: f32 = 480.0;
const MODAL_HEIGHT: f32 = 560.0;

struct Stage {
    number: u8,
    icon: &'static str,
    title: &'static str,
    tagline: &'static str,
}

const STAGES: &[Stage] = &[
    Stage {
        number: 1,
        icon: "γ",
        title: "Gamma",
        tagline: "How hard the mask cuts. Feeds every stage below — a higher gamma produces a more aggressive mask, so threshold / edge shift / refine / feather all operate on a darker silhouette.",
    },
    Stage {
        number: 2,
        icon: ICON_BOLT.codepoint,
        title: "Hard Threshold",
        tagline: "Optional. When on, collapses the mask to pure 0/1 — downstream stages lose the gradient. Refine can only clean up binary stairsteps; feather blurs a step function into gray.",
    },
    Stage {
        number: 3,
        icon: ICON_SWAP_HORIZ.codepoint,
        title: "Edge Shift",
        tagline: "Erodes (positive) or dilates (negative) the mask boundary. Refine Edges then snaps the shifted boundary to image color.",
    },
    Stage {
        number: 4,
        icon: ICON_AUTO_FIX_HIGH.codepoint,
        title: "Refine Edges",
        tagline: "Optional. Uses the original RGB to snap the mask to color transitions. Sees whatever threshold + edge shift produced — tighter input, tighter result.",
    },
    Stage {
        number: 5,
        icon: ICON_BLUR_LINEAR.codepoint,
        title: "Feather",
        tagline: "Final softening pass. Runs after refine on purpose: sharpen-then-soften beats soften-then-try-to-sharpen.",
    },
];

/// Returns true if the modal should close.
pub fn render(ctx: &egui::Context) -> bool {
    theme::draw_modal_backdrop(ctx, "pipeline_flow_backdrop");

    let mut open = true;
    let window_response = egui::Window::new("Mask Pipeline")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([MODAL_WIDTH, MODAL_HEIGHT])
        .frame(theme::overlay_frame())
        .show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.add_space(theme::SPACE_SM);
                endpoint_label(ui, "Raw ONNX tensor");
                arrow(ui);

                for (i, stage) in STAGES.iter().enumerate() {
                    stage_row(ui, stage);
                    if i < STAGES.len() - 1 {
                        arrow(ui);
                    }
                }

                arrow(ui);
                endpoint_label(ui, "Final mask → applied as alpha");

                ui.add_space(theme::SPACE_MD);
                ui.separator();
                ui.add_space(theme::SPACE_SM);
                ui.label(
                    RichText::new(
                        "Stage order is fixed. Each stage sees the output of the one above, \
                         so a knob's value changes what every downstream stage has to work with.",
                    )
                    .size(theme::FONT_SIZE_MONO)
                    .color(theme::TEXT_PRIMARY),
                );
            });
        });

    !open || theme::backdrop_clicked(ctx, &window_response)
}

fn stage_row(ui: &mut egui::Ui, stage: &Stage) {
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
                RichText::new(format!("Stage {} of {}", stage.number, STAGES.len()))
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
            RichText::new("↓")
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
