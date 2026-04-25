//! Brush settings chip — radius / hardness / mode + Clear strokes
//! + a live preview of the brush stamp.
//!
//! Rendered next to the brush toggle in Row 2 when brush mode is on.

use egui::{Color32, Sense, Stroke, Ui};

use crate::gui::brush_state::{BrushSettings, BrushState};
use prunr_core::brush::{BrushMode, BrushShape};
use prunr_core::math::smoothstep;

use super::chip;

/// Width budget for the chip label, padded so 1- to 3-digit radii
/// don't reflow the popover anchor as the user drags the slider.
const LABEL_PAD_WIDTH: f32 = 88.0;

/// Preview area inside the popover. Brush radius clamps to fit so a
/// large brush still renders cleanly inside this box.
const PREVIEW_SIZE: f32 = 80.0;

/// Returns `true` if the user clicked "Clear strokes" in the popover.
pub(super) fn render(ui: &mut Ui, brush_state: &mut BrushState) -> bool {
    let label = chip_label(brush_state.settings());
    let resp = ui
        .scope(|ui| {
            ui.set_min_width(LABEL_PAD_WIDTH);
            chip::chip_button(ui, "🖌", &label, /*accent=*/ true)
        })
        .inner;
    let resp = chip::chip_tooltip(
        resp,
        "Brush settings",
        "Configure brush radius, edge hardness, and add/subtract mode. Click strokes to remove or restore subject regions on the result.",
    );

    let mut clear_requested = false;
    chip::popup_for(ui, ui.id().with("brush_chip_popover"), &resp, |ui| {
        let s = brush_state.settings_mut();

        // Two columns: controls on the left, preview on the right.
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.set_min_width(180.0);
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
                    // 1-decimal % gives ~0.5% drag granularity for fine
                    // edge-softness tuning.
                    |v| format!("{:.1}%", v * 100.0),
                );
                ui.add_space(4.0);
                chip::slider_row_f32(
                    ui,
                    "Strength",
                    &mut s.strength,
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
            });

            ui.add_space(8.0);
            draw_preview(ui, *s);
        });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("Shape:");
            for (shape, label) in [
                (BrushShape::Circle, "Circle"),
                (BrushShape::Square, "Square"),
                (BrushShape::Line, "Line"),
            ] {
                if ui.selectable_label(s.shape == shape, label).clicked() {
                    s.shape = shape;
                }
            }
        });

        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

        if ui
            .button("Clear strokes")
            .on_hover_text("Discard all brush corrections on this image")
            .clicked()
        {
            clear_requested = true;
        }
    });

    clear_requested
}

/// Right-padded label so 1-, 2-, 3-digit radius values render at the
/// same visual width and don't shift the popover anchor while the
/// user drags the slider. Combined with the chip's `min_width` scope,
/// the chip stays put even as the digit count changes.
fn chip_label(s: &BrushSettings) -> String {
    let mode = match s.mode {
        BrushMode::Add => "Add",
        BrushMode::Subtract => "Sub",
    };
    format!("{:>3} px {mode}", s.radius as u32)
}

/// Concentric-ring rendering of the brush stamp at current settings.
/// The brush radius is scaled to fit `PREVIEW_SIZE` regardless of the
/// user-chosen pixel value, so a 200 px brush and a 4 px brush both
/// render readably inside the popover.
fn draw_preview(ui: &mut Ui, settings: BrushSettings) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(PREVIEW_SIZE, PREVIEW_SIZE),
        Sense::hover(),
    );
    // Soft contrast frame so the brush silhouette reads against the
    // popover background regardless of the surface colour.
    ui.painter().rect_filled(
        rect,
        4.0,
        Color32::from_rgba_premultiplied(0, 0, 0, 100),
    );
    ui.painter().rect_stroke(
        rect,
        4.0,
        Stroke::new(1.0, Color32::from_rgba_premultiplied(120, 120, 120, 120)),
        egui::StrokeKind::Outside,
    );

    let center = rect.center();
    let max_r = (PREVIEW_SIZE / 2.0) - 4.0;
    // Map slider radius (1..=200) onto preview pixels with a soft cap.
    let r = (settings.radius / 200.0 * max_r).clamp(2.0, max_r);
    let inner = r * settings.hardness.clamp(0.0, 1.0);

    let (cr, cg, cb) = match settings.mode {
        BrushMode::Add => (140, 230, 170),
        BrushMode::Subtract => (230, 150, 150),
    };

    let solid = Color32::from_rgb(cr, cg, cb);
    let span = (r - inner).max(0.001);

    match settings.shape {
        BrushShape::Circle => {
            if inner >= 1.0 {
                ui.painter().circle_filled(center, inner, solid);
            }
            let steps = 14;
            for i in 0..steps {
                let t = (i as f32 + 0.5) / steps as f32;
                let dist = inner + span * t;
                let intensity = if span < 0.5 { 0.0 } else { smoothstep(1.0 - t) };
                let alpha = (intensity * 220.0) as u8;
                if alpha == 0 {
                    continue;
                }
                ui.painter().circle_stroke(
                    center,
                    dist,
                    Stroke::new(span / steps as f32 * 1.4, Color32::from_rgba_premultiplied(cr, cg, cb, alpha)),
                );
            }
        }
        BrushShape::Square => {
            if inner >= 1.0 {
                ui.painter().rect_filled(
                    egui::Rect::from_center_size(center, egui::vec2(inner * 2.0, inner * 2.0)),
                    0.0,
                    solid,
                );
            }
            let steps = 10;
            for i in 0..steps {
                let t = (i as f32 + 0.5) / steps as f32;
                let dist = inner + span * t;
                let intensity = if span < 0.5 { 0.0 } else { smoothstep(1.0 - t) };
                let alpha = (intensity * 220.0) as u8;
                if alpha == 0 {
                    continue;
                }
                ui.painter().rect_stroke(
                    egui::Rect::from_center_size(center, egui::vec2(dist * 2.0, dist * 2.0)),
                    0.0,
                    Stroke::new(span / steps as f32 * 1.4, Color32::from_rgba_premultiplied(cr, cg, cb, alpha)),
                    egui::StrokeKind::Outside,
                );
            }
        }
        BrushShape::Line => {
            // Diagonal segment with the brush radius as half-thickness.
            let half = max_r - 4.0;
            ui.painter().line_segment(
                [
                    egui::Pos2::new(center.x - half, center.y + half),
                    egui::Pos2::new(center.x + half, center.y - half),
                ],
                Stroke::new(r * 2.0, Color32::from_rgba_premultiplied(cr, cg, cb, 200)),
            );
        }
    }
}
