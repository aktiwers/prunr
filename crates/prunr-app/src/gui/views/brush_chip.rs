//! Brush settings chip — radius / hardness / mode + Clear strokes.
//!
//! Rendered next to the brush toggle in Row 2 when brush mode is on.
//! Wires through the existing chip primitives (`chip_button`,
//! `popup_for`, `slider_row_f32`).

use egui::Ui;

use crate::gui::brush_state::BrushState;
use prunr_core::brush::BrushMode;

use super::chip;

/// Returns `true` if the user clicked "Clear strokes" in the popover.
pub(super) fn render(ui: &mut Ui, brush_state: &mut BrushState) -> bool {
    let mode_label = match brush_state.settings().mode {
        BrushMode::Add => "Add",
        BrushMode::Subtract => "Sub",
    };
    let label = format!("{:.0}px {}", brush_state.settings().radius, mode_label);
    let resp = chip::chip_button(ui, "🖌", &label, /*accent=*/ true);
    let resp = chip::chip_tooltip(
        resp,
        "Brush settings",
        "Configure brush radius, edge hardness, and add/subtract mode. Click strokes to remove or restore subject regions on the result.",
    );

    let mut clear_requested = false;
    chip::popup_for(ui, ui.id().with("brush_chip_popover"), &resp, |ui| {
        let s = brush_state.settings_mut();

        chip::slider_row_f32(
            ui,
            "Radius",
            &mut s.radius,
            1.0..=200.0,
            /*logarithmic=*/ true,
            |v| format!("{v:.0} px"),
        );
        ui.add_space(4.0);

        chip::slider_row_f32(
            ui,
            "Hardness",
            &mut s.hardness,
            0.0..=1.0,
            false,
            |v| format!("{:.0}%", v * 100.0),
        );
        ui.add_space(6.0);

        ui.horizontal(|ui| {
            let add = ui.selectable_label(matches!(s.mode, BrushMode::Add), "Add");
            if add.clicked() {
                s.mode = BrushMode::Add;
            }
            let sub = ui.selectable_label(matches!(s.mode, BrushMode::Subtract), "Subtract");
            if sub.clicked() {
                s.mode = BrushMode::Subtract;
            }
        });

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(4.0);

        if ui.button("Clear strokes").on_hover_text("Discard all brush corrections on this image").clicked() {
            clear_requested = true;
        }
    });

    clear_requested
}
