//! Brush settings chip — radius / hardness / mode + Reset brush
//! + a live preview of the brush stamp.
//!
//! Rendered next to the brush toggle while Paint Brush is on.

use egui::{Color32, Sense, Stroke, Ui};
use egui_material_icons::icons::ICON_BRUSH;

use crate::gui::brush_state::BrushSettings;
use prunr_core::brush::BrushShape;
use prunr_core::selection::BrushMode;

use super::chip;
use super::fmt;

/// Width budget for the chip label, padded so 1- to 3-digit radii
/// don't reflow the popover anchor as the user drags the slider.
const LABEL_PAD_WIDTH: f32 = 88.0;

/// Preview area inside the popover. Brush radius clamps to fit so a
/// large brush still renders cleanly inside this box.
const PREVIEW_SIZE: f32 = 80.0;

/// Maximum inpaint_feather value exposed in the UI, used to normalize
/// the hardness-reduction preview. At feather=INPAINT_FEATHER_MAX the
/// stamp's effective hardness is cut in half — visualizes the soft-edge
/// effect approximately. (Production uses an edge-preserving guided
/// filter, not a literal Gaussian — the preview just gives the user a
/// rough sense of how the slider value affects the boundary.)
const INPAINT_FEATHER_MAX: f32 = 32.0;
const FEATHER_HARDNESS_REDUCTION_CAP: f32 = 0.5;

#[derive(Default, Clone, Copy)]
pub(super) struct BrushChipOutcome {
    pub reset_brush_requested: bool,
    /// True on slider release / mode / shape click. Caller persists
    /// app-level brush settings on this signal.
    pub committed: bool,
    /// New value of `Settings::protect_selection` when its switch flipped.
    pub protect_selection: Option<bool>,
}

/// `protect` is `Some(current protect_selection)` for models whose strokes
/// can apply on their own (background removal); `None` hides the switch.
pub(super) fn render(
    ui: &mut Ui,
    s: &mut BrushSettings,
    is_inpaint_mode: bool,
    protect: Option<bool>,
) -> BrushChipOutcome {
    let label = chip_label(s);
    let resp = ui
        .scope(|ui| {
            ui.set_min_width(LABEL_PAD_WIDTH);
            chip::chip_button(ui, ICON_BRUSH.codepoint, &label, /*accent=*/ true)
        })
        .inner;
    let resp = chip::tooltip(
        resp,
        "Brush settings",
        "Size, hardness and whether strokes add to or subtract from the selection.",
        None,
    );

    let mut outcome = BrushChipOutcome::default();
    chip::popup_for(ui, ui.id().with("brush_chip_popover"), &resp, |ui| {
        if chip::popover_header(ui, "Brush", Some(("Reset size, hardness, expand, edge blend, sharpen and shape", false))) {
            outcome.reset_brush_requested = true;
        }
        outcome.committed |= render_cursor_section(ui, s);
        ui.add_space(4.0);
        if !is_inpaint_mode {
            // The eraser binarizes the region and paints in one direction,
            // so opacity and mode would have no effect there.
            let st = chip::slider_row_f32(ui, "Opacity", &mut s.strength, 0.0..=1.0, false, fmt::percent);
            outcome.committed |= st.commit;
            super::hint(ui, "How strongly each stroke changes the selection.");
            ui.add_space(6.0);
            let modes = [
                chip::Choice { value: BrushMode::Add, name: "Add", description: "Strokes add to the selection", enabled: true },
                chip::Choice { value: BrushMode::Subtract, name: "Subtract", description: "Strokes remove from the selection", enabled: true },
            ];
            outcome.committed |= chip::choice_row(ui, "Mode", &modes, &mut s.mode);
        } else {
            let g = chip::slider_row_f32(ui, "Expand region", &mut s.inpaint_grow, -16.0..=16.0, false, |v| fmt::signed_px(v, 0));
            outcome.committed |= g.commit;
            super::hint(ui, "Grow or shrink the painted region before the fill.");
            ui.add_space(4.0);
            let f = chip::slider_row_f32(ui, "Edge blend", &mut s.inpaint_feather, 0.0..=32.0, false, |v| fmt::px(v, 0));
            outcome.committed |= f.commit;
            super::hint(ui, "Width of the band where the fill blends into the photo.");
            ui.add_space(4.0);
            // Sharpen displays as 0-100% on a 0-2 internal range.
            let sh = chip::slider_row_f32(ui, "Sharpen", &mut s.inpaint_sharpen, 0.0..=2.0, false, |v| fmt::percent(v / 2.0));
            outcome.committed |= sh.commit;
            super::hint(ui, "Sharpen the filled area, which comes out slightly soft.");
        }

        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

        // Selection visualization knobs — shared with Magic Brush chip via
        // render_shared_selection_section.
        let sel_committed = render_shared_selection_section(ui, s);
        outcome.committed |= sel_committed;
        outcome.protect_selection = render_auto_apply_row(ui, protect);
    });

    outcome
}

/// The cursor block shared by both brushes: Size and Hardness on the
/// left, the live stamp preview and Shape on the right. Returns true if
/// anything committed this frame.
pub(super) fn render_cursor_section(ui: &mut Ui, s: &mut BrushSettings) -> bool {
    let mut committed = false;
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_min_width(220.0);
            ui.set_max_width(280.0);
            let r = chip::slider_row_f32(ui, "Size", &mut s.radius, 1.0..=200.0, true, |v| fmt::px(v, 0));
            committed |= r.commit;
            ui.add_space(4.0);
            // tenths give ~0.5% drag granularity for fine edge tuning
            let h = chip::slider_row_f32(ui, "Hardness", &mut s.hardness, 0.0..=1.0, false, fmt::percent_tenths);
            committed |= h.commit;
            super::hint(ui, "0% is a soft edge, 100% a hard one.");
        });
        ui.add_space(8.0);
        ui.vertical(|ui| {
            draw_preview(ui, s);
            ui.add_space(6.0);
            let shapes = [
                chip::Choice { value: BrushShape::Circle, name: "Circle", description: "", enabled: true },
                chip::Choice { value: BrushShape::Square, name: "Square", description: "", enabled: true },
                chip::Choice { value: BrushShape::Line, name: "Line", description: "", enabled: true },
            ];
            committed |= chip::choice_row(ui, "Shape", &shapes, &mut s.shape);
        });
    });
    committed
}

/// The "Auto-apply strokes" switch, shared by both brush popovers. It is
/// the inverse of `Settings::protect_selection`; returns the new
/// `protect_selection` when flipped so no caller has to negate it.
pub(super) fn render_auto_apply_row(ui: &mut egui::Ui, protect: Option<bool>) -> Option<bool> {
    let mut auto_apply = !protect?;
    ui.add_space(4.0);
    let flipped = chip::toggle_row(ui, "Auto-apply strokes", &mut auto_apply).changed;
    super::hint(ui, "Apply each stroke to the result at once. Off: paint freely, then click Process.");
    flipped.then_some(!auto_apply)
}

/// Selection visualization knobs shared between the Paint Brush chip and
/// the Magic Brush chip. Returns `true` if any slider committed this frame.
pub(super) fn render_shared_selection_section(ui: &mut egui::Ui, s: &mut BrushSettings) -> bool {
    let mut committed = false;
    ui.label(
        egui::RichText::new("Selection")
            .strong()
            .color(crate::gui::theme::TEXT_PRIMARY)
            .size(crate::gui::theme::FONT_SIZE_BODY),
    );
    ui.add_space(2.0);
    let fe = chip::slider_row_f32(
        ui, "Feather", &mut s.edge_feather, 0.0..=20.0, false,
        |v| fmt::off_or(v, 0.1, |v| fmt::px(v, 0)),
    );
    committed |= fe.commit;
    super::hint(ui, "Soften the selection edge.");
    ui.add_space(2.0);
    let ot = chip::slider_row_f32(
        ui, "Outline width", &mut s.outline_thickness, 0.0..=10.0, false,
        |v| fmt::px(v, 1),
    );
    committed |= ot.commit;
    ui.add_space(2.0);
    let oo = chip::slider_row_f32(
        ui, "Outline opacity", &mut s.outline_opacity, 0.0..=1.0, false,
        fmt::percent,
    );
    committed |= oo.commit;
    ui.add_space(2.0);
    let fo = chip::slider_row_f32(
        ui, "Fill opacity", &mut s.fill_opacity, 0.0..=1.0, false,
        fmt::percent,
    );
    committed |= fo.commit;
    committed
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
    // popover background regardless of the surface color.
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

    // Visually hint at the Edge softness setting by softening the
    // falloff edge proportional to inpaint_feather.
    let feather_norm = (settings.inpaint_feather / INPAINT_FEATHER_MAX).clamp(0.0, 1.0);
    let effective_hardness = (settings.hardness
        * (1.0 - FEATHER_HARDNESS_REDUCTION_CAP * feather_norm))
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
