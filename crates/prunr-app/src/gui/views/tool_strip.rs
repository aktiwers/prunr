//! The strip over the top of the canvas while a brush tool is on: the
//! knobs touched on every stroke, and a button to the rest of them. It
//! replaces the row-2 settings chip, whose popover hid the cursor and
//! froze the canvas while it was open.

use std::ops::RangeInclusive;

use egui::{Align2, Id, Order, Rect, RichText, Ui};
use egui_material_icons::icons::ICON_MORE_HORIZ;

use crate::gui::app::PrunrApp;
use crate::gui::brush_state::{BrushSettings, BRUSH_RADIUS_RANGE};
use crate::gui::theme;
use crate::gui::views::shortcuts::Action;
use prunr_core::brush::BrushShape;
use prunr_core::selection::BrushMode;

use super::brush_chip::{self, BrushChipOutcome};
use super::{chip, fmt, magic_brush_chip};

const TOP_INSET: f32 = 12.0;
const SLIDER_WIDTH: f32 = 110.0;

#[derive(Clone, Copy)]
enum Tool {
    Paint,
    Magic,
}

pub(crate) fn render(ui: &mut Ui, app: &mut PrunrApp, canvas_rect: Rect) -> BrushChipOutcome {
    let tool = if app.magic_brush_state.is_active() {
        Tool::Magic
    } else if app.brush_state.is_enabled() {
        Tool::Paint
    } else {
        return BrushChipOutcome::default();
    };
    let is_inpaint = app.settings.model.is_inpaint();
    let protect = app.settings.model.uses_segmentation().then_some(app.settings.protect_selection);
    let encoder_pending = app.magic_brush_state.has_pending_encoder();
    let brush = &mut app.settings.brush;
    let (title, more) = match tool {
        Tool::Paint => ("Paint Brush", "More Paint Brush settings"),
        Tool::Magic => ("Magic Brush", "More Magic Brush settings"),
    };
    let mut outcome = BrushChipOutcome::default();
    egui::Area::new(Id::new("tool_strip"))
        .order(Order::Foreground)
        .pivot(Align2::CENTER_TOP)
        .fixed_pos(egui::pos2(canvas_rect.center().x, canvas_rect.top() + TOP_INSET))
        .show(ui.ctx(), |ui| {
            egui::Frame::new()
                .fill(theme::BG_SECONDARY)
                .stroke(egui::Stroke::new(theme::STROKE_DEFAULT, theme::ACCENT))
                .corner_radius(8.0)
                .inner_margin(egui::Margin::symmetric(12, 6))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = theme::SPACE_SM;
                        ui.spacing_mut().slider_width = SLIDER_WIDTH;
                        ui.label(RichText::new(title).strong().color(theme::TEXT_PRIMARY));
                        ui.separator();
                        outcome.committed |= match tool {
                            Tool::Paint => paint_knobs(ui, brush, is_inpaint),
                            Tool::Magic => magic_knobs(ui, brush),
                        };
                        ui.separator();
                        let dots = chip::tooltip(
                            chip::icon_square_button(ui, ICON_MORE_HORIZ.codepoint, theme::TEXT_PRIMARY, theme::BG_SECONDARY),
                            more,
                            "",
                            None,
                        );
                        chip::flyout_for(Id::new(("tool_strip_more", title)), &dots, |ui| match tool {
                            Tool::Paint => {
                                let o = brush_chip::flyout_body(ui, brush, is_inpaint, protect);
                                outcome.committed |= o.committed;
                                outcome.reset_brush_requested |= o.reset_brush_requested;
                                outcome.protect_selection = o.protect_selection;
                            }
                            Tool::Magic => {
                                let o = magic_brush_chip::flyout_body(ui, brush, encoder_pending, protect);
                                outcome.committed |= o.committed;
                                outcome.protect_selection = o.protect_selection;
                            }
                        });
                    });
                });
        });
    outcome
}

fn paint_knobs(ui: &mut Ui, s: &mut BrushSettings, is_inpaint: bool) -> bool {
    let mut committed = false;
    if !is_inpaint {
        committed |= segmented(ui, &[(BrushMode::Add, "Add"), (BrushMode::Subtract, "Subtract")], &mut s.mode);
    }
    committed |= size_knob(ui, &mut s.radius);
    committed |= knob(ui, "Hardness", &mut s.hardness, 0.0..=1.0, false, fmt::percent_tenths);
    if is_inpaint {
        committed |= knob(ui, "Expand region", &mut s.inpaint_grow, -16.0..=16.0, false, |v| fmt::signed_px(v, 0));
        committed |= knob(ui, "Edge blend", &mut s.inpaint_feather, 0.0..=32.0, false, |v| fmt::px(v, 0));
    } else {
        committed |= knob(ui, "Opacity", &mut s.strength, 0.0..=1.0, false, fmt::percent);
    }
    committed
}

fn magic_knobs(ui: &mut Ui, s: &mut BrushSettings) -> bool {
    let mut committed = size_knob(ui, &mut s.radius);
    committed |= knob(ui, "Confidence", &mut s.magic_confidence_threshold, 0.0..=1.0, false, fmt::percent);
    committed |= segmented(
        ui,
        &[(BrushShape::Circle, "Circle"), (BrushShape::Square, "Square"), (BrushShape::Line, "Line")],
        &mut s.shape,
    );
    committed
}

fn size_knob(ui: &mut Ui, radius: &mut f32) -> bool {
    let label = ui.label(RichText::new("Size").color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));
    let resp = ui
        .add(
            egui::Slider::new(radius, BRUSH_RADIUS_RANGE)
                .logarithmic(true)
                .show_value(true)
                .custom_formatter(|v, _| fmt::px(v as f32, 0)),
        )
        .labelled_by(label.id);
    let resp = chip::tooltip(resp, "Size", "Brush size in pixels.", Some(Action::BrushLarger));
    chip::slider_settled(&resp)
}

fn knob(ui: &mut Ui, label: &str, value: &mut f32, range: RangeInclusive<f32>, logarithmic: bool, format: impl Fn(f32) -> String) -> bool {
    let label = ui.label(RichText::new(label).color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));
    let resp = ui
        .add(
            egui::Slider::new(value, range)
                .logarithmic(logarithmic)
                .show_value(true)
                .custom_formatter(move |v, _| format(v as f32)),
        )
        .labelled_by(label.id);
    chip::slider_settled(&resp)
}

fn segmented<T: Copy + PartialEq>(ui: &mut Ui, options: &[(T, &'static str)], value: &mut T) -> bool {
    let mut changed = false;
    for (option, name) in options {
        if ui.selectable_label(*value == *option, *name).clicked() && *value != *option {
            *value = *option;
            changed = true;
        }
    }
    changed
}
