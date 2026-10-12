//! The Paint Brush panel: cursor block, stroke knobs, selection look,
//! and a live preview of the brush stamp.

use egui::{Color32, Sense, Stroke, Ui};

use crate::gui::brush_state::{BrushSettings, BRUSH_RADIUS_RANGE};
use crate::gui::knob_catalog::{shows, Surface};
use crate::gui::settings::SettingsModel;
use prunr_core::brush::BrushShape;
use prunr_core::selection::BrushMode;

use super::chip;
use super::fmt;

/// Preview area inside the popover. Brush radius clamps to fit so a
/// large brush still renders cleanly inside this box.
const PREVIEW_SIZE: f32 = 80.0;

/// What a brush panel or the tool strip changed this frame.
#[derive(Default, Clone, Copy)]
pub(crate) struct BrushChipOutcome {
    pub reset_brush_requested: bool,
    /// A slider settled or a choice was made: the brush settings persist.
    pub committed: bool,
}

impl BrushChipOutcome {
    pub fn merge(&mut self, other: Self) {
        self.reset_brush_requested |= other.reset_brush_requested;
        self.committed |= other.committed;
    }
}

/// On a background-removal model a stroke edits the cut-out.
pub(super) const CUTOUT_MODES: [chip::Choice<BrushMode>; 2] = [
    chip::Choice { value: BrushMode::Add, name: "Restore", description: "Bring back what the model took away", enabled: true },
    chip::Choice { value: BrushMode::Subtract, name: "Erase", description: "Remove what the model left behind", enabled: true },
];

/// Elsewhere a stroke shapes the selection or the region to fill.
pub(super) const REGION_MODES: [chip::Choice<BrushMode>; 2] = [
    chip::Choice { value: BrushMode::Add, name: "Add", description: "Add to the selection", enabled: true },
    chip::Choice { value: BrushMode::Subtract, name: "Subtract", description: "Take away from the selection", enabled: true },
];

/// Restore / Erase on a cut-out; Add / Subtract on a selection or region.
pub(super) fn mode_choices(model: SettingsModel) -> &'static [chip::Choice<BrushMode>; 2] {
    if model.edits_cutout() { &CUTOUT_MODES } else { &REGION_MODES }
}

const BRUSH_SHAPES: [chip::Choice<BrushShape>; 3] = [
    chip::Choice { value: BrushShape::Circle, name: "Circle", description: "", enabled: true },
    chip::Choice { value: BrushShape::Square, name: "Square", description: "", enabled: true },
    chip::Choice { value: BrushShape::Line, name: "Line", description: "", enabled: true },
];

pub(super) fn flyout_body(ui: &mut Ui, s: &mut BrushSettings, model: SettingsModel) -> BrushChipOutcome {
    let mut outcome = BrushChipOutcome::default();
    if chip::popover_header(ui, "Brush", Some(("Reset the brush", false))) {
        outcome.reset_brush_requested = true;
    }
    let selection_strokes = shows(model, Surface::SelectionStrokes);
    outcome.committed |= render_cursor_section(ui, s, selection_strokes);
    ui.add_space(4.0);
    if selection_strokes {
        let st = chip::slider_row_f32(ui, "Opacity", &mut s.strength, 0.0..=1.0, false, fmt::percent);
        outcome.committed |= st.commit;
        super::hint(ui, "How strongly each stroke changes the selection.");
        ui.add_space(6.0);
        outcome.committed |= chip::choice_row(ui, "Mode", mode_choices(model), &mut s.mode);
    }
    if shows(model, Surface::FillRegionKnobs) {
        let g = expand_region_row(ui, s);
        outcome.committed |= g.commit;
        super::hint(ui, "Grow or shrink the painted region before the fill.");
        ui.add_space(4.0);
        let f = feather_row(ui, s);
        outcome.committed |= f.commit;
        super::hint(ui, "Fade the fill into the photo over a ring just outside the painted region.");
        ui.add_space(4.0);
        // Sharpen displays as 0-100% on a 0-2 internal range.
        let sh = chip::slider_row_f32(ui, "Sharpen", &mut s.inpaint_sharpen, 0.0..=2.0, false, |v| fmt::percent(v / 2.0));
        outcome.committed |= sh.commit;
        super::hint(ui, "Sharpen the filled area, which comes out slightly soft.");
    }

    if selection_strokes {
        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);
        outcome.committed |= render_shared_selection_section(ui, s);
    }
    outcome
}

/// The Paint Brush's cursor block: Size and, where strokes are graded,
/// Hardness on the left, the stamp preview and Shape on the right.
/// Returns true if anything committed this frame.
fn render_cursor_section(ui: &mut Ui, s: &mut BrushSettings, with_hardness: bool) -> bool {
    let mut committed = false;
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_min_width(220.0);
            ui.set_max_width(280.0);
            let r = chip::slider_row_f32(ui, "Size", &mut s.radius, BRUSH_RADIUS_RANGE, true, |v| fmt::px(v, 0));
            committed |= r.commit;
            if with_hardness {
                ui.add_space(4.0);
                // tenths give ~0.5% drag granularity for fine edge tuning
                let h = chip::slider_row_f32(ui, "Hardness", &mut s.hardness, 0.0..=1.0, false, fmt::percent_tenths);
                committed |= h.commit;
                super::hint(ui, "0% is a soft edge, 100% a hard one.");
            }
        });
        ui.add_space(8.0);
        ui.vertical(|ui| {
            draw_preview(ui, s, if with_hardness { s.hardness } else { 1.0 });
            ui.add_space(6.0);
            committed |= chip::choice_row(ui, "Shape", &BRUSH_SHAPES, &mut s.shape);
        });
    });
    committed
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
    committed
}

/// Concentric-ring rendering of the brush stamp at current settings.
/// The brush radius is scaled to fit `PREVIEW_SIZE` regardless of the
/// user-chosen pixel value, so a 200 px brush and a 4 px brush both
/// render readably inside the popover.
fn draw_preview(ui: &mut Ui, settings: &BrushSettings, hardness: f32) {
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

    let effective_hardness = hardness.clamp(0.0, 1.0);

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

/// The eraser's Expand region slider, in the flyout and the tool strip.
pub(super) fn expand_region_row(ui: &mut egui::Ui, s: &mut BrushSettings) -> chip::ChipChange {
    chip::slider_row_f32(ui, "Expand region", &mut s.inpaint_grow, -16.0..=16.0, false, |v| fmt::signed_px(v, 0))
}

/// The eraser's Feather slider, in the flyout and the tool strip.
pub(super) fn feather_row(ui: &mut egui::Ui, s: &mut BrushSettings) -> chip::ChipChange {
    chip::slider_row_f32(ui, "Feather", &mut s.inpaint_feather, 0.0..=32.0, false, |v| fmt::off_or(v, 0.5, |v| fmt::px(v, 0)))
}
