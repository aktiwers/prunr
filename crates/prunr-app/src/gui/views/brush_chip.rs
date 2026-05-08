//! Brush settings chip — radius / hardness / mode + Clear strokes
//! + a live preview of the brush stamp.
//!
//! Rendered next to the brush toggle in Row 2 when brush mode is on.

use egui::{Color32, Sense, Stroke, Ui};

use crate::gui::brush_state::BrushSettings;
use prunr_core::brush::{BrushMode, BrushShape};
use prunr_core::inpaint_sd::MASK_BLUR_MAX;

use super::chip;

/// Width budget for the chip label, padded so 1- to 3-digit radii
/// don't reflow the popover anchor as the user drags the slider.
const LABEL_PAD_WIDTH: f32 = 88.0;

/// Preview area inside the popover. Brush radius clamps to fit so a
/// large brush still renders cleanly inside this box.
const PREVIEW_SIZE: f32 = 80.0;

/// Maximum reduction the mask_blur preview applies to hardness in the
/// brush stamp render. At slider=MASK_BLUR_MAX, hardness is cut in half
/// — visualizes the soft-edge effect without literal Gaussian.
const MASK_BLUR_HARDNESS_REDUCTION_CAP: f32 = 0.5;

#[derive(Default, Clone, Copy)]
pub(super) struct BrushChipOutcome {
    pub clear_requested: bool,
    /// True on slider release / mode / shape click. Caller persists
    /// app-level brush settings on this signal.
    pub committed: bool,
}

pub(super) fn render(
    ui: &mut Ui,
    s: &mut BrushSettings,
    is_inpaint_mode: bool,
) -> BrushChipOutcome {
    let label = chip_label(s);
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

    let mut outcome = BrushChipOutcome::default();
    chip::popup_for(ui, ui.id().with("brush_chip_popover"), &resp, |ui| {

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.set_min_width(220.0);
                ui.set_max_width(280.0);
                let r = chip::slider_row_f32(
                    ui, "Radius", &mut s.radius, 1.0..=200.0, true,
                    |v| format!("{v:.0} px"),
                );
                outcome.committed |= r.commit;
                super::hint(ui, "Brush size in image pixels.");
                ui.add_space(4.0);
                let h = chip::slider_row_f32(
                    ui, "Hardness", &mut s.hardness, 0.0..=1.0, false,
                    // 1-decimal % gives ~0.5% drag granularity for fine
                    // edge-softness tuning.
                    |v| format!("{:.1}%", v * 100.0),
                );
                outcome.committed |= h.commit;
                super::hint(ui, "Edge falloff. 0% = soft smoothstep, 100% = hard disc.");
                ui.add_space(4.0);
                if !is_inpaint_mode {
                    // Strength is meaningful for the seg pipeline (multiplicative
                    // mask correction). For Eraser the mask is binarized at the
                    // LaMa boundary, so Strength has no effect — hide it.
                    let st = chip::slider_row_f32(
                        ui, "Strength", &mut s.strength, 0.0..=1.0, false,
                        |v| format!("{:.0}%", v * 100.0),
                    );
                    outcome.committed |= st.commit;
                    super::hint(ui, "How much each stroke shifts the mask. Lower = gentler corrections.");
                } else {
                    // Eraser-specific knobs. Live-update on release like
                    // every other slider so the user sees the diff.
                    ui.add_space(4.0);
                    let g = chip::slider_row_f32(
                        ui, "Mask grow", &mut s.inpaint_grow, -16.0..=16.0, false,
                        |v| format!("{v:+.0} px"),
                    );
                    outcome.committed |= g.commit;
                    super::hint(ui, "Expand (+) or shrink (−) the painted region in pixels before inpaint.");
                    ui.add_space(4.0);
                    let f = chip::slider_row_f32(
                        ui, "Edge softness", &mut s.inpaint_feather, 0.0..=32.0, false,
                        |v| format!("{v:.0} px"),
                    );
                    outcome.committed |= f.commit;
                    super::hint(ui, "Composite-time Gaussian blend at the boundary. Higher values give smoother transitions. Default 4 px matches A1111 / ComfyUI canonical behavior.");
                    ui.add_space(4.0);
                    // Sharpen displays as 0-100% on a 0-2 internal range.
                    let sh = chip::slider_row_f32(
                        ui, "Sharpen", &mut s.inpaint_sharpen, 0.0..=2.0, false,
                        |v| format!("{:.0}%", v * 50.0),
                    );
                    outcome.committed |= sh.commit;
                    super::hint(ui, "Unsharp-mask amount inside the inpainted region. Counters the model's slight blur.");
                }
                if !is_inpaint_mode {
                    // Inpaint has only one direction (paint = erase). Hide
                    // the toggle so the user doesn't see a knob with no
                    // effect; mode stays pinned to whatever it was.
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.selectable_label(matches!(s.mode, BrushMode::Add), "Add").clicked() {
                            s.mode = BrushMode::Add;
                            outcome.committed = true;
                        }
                        if ui.selectable_label(matches!(s.mode, BrushMode::Subtract), "Subtract").clicked() {
                            s.mode = BrushMode::Subtract;
                            outcome.committed = true;
                        }
                    });
                }
            });

            ui.add_space(8.0);
            ui.vertical(|ui| {
                draw_preview(ui, s);
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    for (shape, label) in [
                        (BrushShape::Circle, "Circle"),
                        (BrushShape::Square, "Square"),
                        (BrushShape::Line, "Line"),
                    ] {
                        if ui.selectable_label(s.shape == shape, label).clicked() {
                            s.shape = shape;
                            outcome.committed = true;
                        }
                    }
                });
            });
        });

        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

        if ui
            .button("Clear strokes")
            .on_hover_text("Discard all brush corrections on this image")
            .clicked()
        {
            outcome.clear_requested = true;
        }
    });

    outcome
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
fn draw_preview(ui: &mut Ui, settings: &BrushSettings) {
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
    // The preview's purpose is to communicate the brush pattern
    // (hardness curve, blur halo, shape gradient) — at small radii in
    // linear scale the brush is too tiny to read. Log scale with a 50%
    // floor: small radii fill ~54% of the preview, max fills 100%,
    // scale curves smoothly between. Absolute size feedback comes from
    // the slider value ("20 px"), the preview shows the pattern.
    // Formula: `(1 + 99×t)` maps t∈[0,1] onto [1, 100]; log10 maps
    // that onto [0, 2]; dividing by 2 renormalises to [0, 1]. The
    // 0.5 floor means slider=1 still fills 54% of the box.
    let r_norm = (1.0 + (settings.radius / 200.0) * 99.0).log10() / 2.0; // [0, 1]
    let r = (max_r * 0.5 + max_r * 0.5 * r_norm).clamp(max_r * 0.4, max_r);

    let (cr, cg, cb) = match settings.mode {
        BrushMode::Add => (140, 230, 170),
        BrushMode::Subtract => (230, 150, 150),
    };

    // Preview only — actual inference applies a fast_blur per the
    // mask_blur sigma; this visually hints at the effect by softening
    // the falloff edge proportional to mask_blur.
    let mask_blur_norm = (settings.sd_mask_blur / MASK_BLUR_MAX).clamp(0.0, 1.0);
    let effective_hardness = (settings.hardness
        * (1.0 - MASK_BLUR_HARDNESS_REDUCTION_CAP * mask_blur_norm))
        .clamp(0.0, 1.0);

    let color = Color32::from_rgb(cr, cg, cb);
    match settings.shape {
        BrushShape::Circle => {
            chip::paint_falloff_circle(ui.painter(), center, r, effective_hardness, color, 220, 14);
        }
        BrushShape::Square => {
            chip::paint_falloff_square(ui.painter(), center, r, effective_hardness, color, 220, 10);
        }
        BrushShape::Line => {
            let half = max_r - 4.0;
            ui.painter().line_segment(
                [
                    egui::Pos2::new(center.x - half, center.y + half),
                    egui::Pos2::new(center.x + half, center.y - half),
                ],
                Stroke::new(r * 2.0, color),
            );
        }
    }
}
