//! The strip over the top of the canvas while a brush tool is on: the
//! knobs touched on every stroke, and a button to the rest of them.

use egui::{Align2, Id, Order, Rect, RichText, Ui};
use egui_material_icons::icons::ICON_MORE_HORIZ;

use crate::gui::brush_state::{BrushSettings, BRUSH_RADIUS_RANGE};
use crate::gui::theme;

use super::brush_chip::{self, BrushChipOutcome, BRUSH_MODES, BRUSH_SHAPES};
use super::{chip, fmt, magic_brush_chip};

const TOP_INSET: f32 = 12.0;
const SLIDER_WIDTH: f32 = 110.0;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tool {
    Paint,
    Magic,
}

impl Tool {
    fn title(self) -> &'static str {
        match self {
            Tool::Paint => "Paint Brush",
            Tool::Magic => "Magic Brush",
        }
    }

    fn more(self) -> &'static str {
        match self {
            Tool::Paint => "More Paint Brush settings",
            Tool::Magic => "More Magic Brush settings",
        }
    }
}

/// What the panels need besides the brush settings.
#[derive(Clone, Copy)]
pub(crate) struct StripFacts {
    pub is_inpaint: bool,
    /// `Some(protect_selection)` for models whose strokes can apply on
    /// their own; `None` hides the Auto-apply switch.
    pub protect: Option<bool>,
    pub encoder_pending: bool,
    /// A selection is painted and Auto-apply is off: offer Apply.
    pub strokes_waiting: bool,
}

pub(crate) fn render(ui: &mut Ui, canvas_rect: Rect, tool: Tool, brush: &mut BrushSettings, facts: StripFacts) -> BrushChipOutcome {
    let mut change = BrushChipOutcome::default();
    egui::Area::new(Id::new("tool_strip"))
        .order(Order::Foreground)
        .pivot(Align2::CENTER_TOP)
        .fixed_pos(egui::pos2(canvas_rect.center().x, canvas_rect.top() + TOP_INSET))
        .show(ui.ctx(), |ui| {
            theme::strip_frame().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = theme::SPACE_SM;
                    ui.spacing_mut().slider_width = SLIDER_WIDTH;
                    ui.label(RichText::new(tool.title()).strong().color(theme::TEXT_PRIMARY));
                    ui.separator();
                    change.committed |= match tool {
                        Tool::Paint => paint_knobs(ui, brush, facts.is_inpaint),
                        Tool::Magic => magic_knobs(ui, brush),
                    };
                    if facts.strokes_waiting {
                        ui.separator();
                        let apply = chip::tooltip(
                            chip::button(ui, chip::ButtonKind::Primary, "Apply strokes"),
                            "Apply strokes",
                            "Run the model with the painted selection. Auto-apply in the panel does this on every stroke.",
                            None,
                        );
                        change.apply_requested |= apply.clicked();
                    }
                    ui.separator();
                    let dots = chip::tooltip(
                        chip::icon_square_button(ui, ICON_MORE_HORIZ.codepoint, theme::TEXT_PRIMARY, theme::BG_SECONDARY),
                        tool.more(),
                        "",
                        None,
                    );
                    chip::flyout_for(Id::new(("tool_strip_more", tool.title())), &dots, |ui| {
                        change.merge(panel(ui, tool, brush, facts));
                    });
                });
            });
        });
    change
}

fn panel(ui: &mut Ui, tool: Tool, brush: &mut BrushSettings, facts: StripFacts) -> BrushChipOutcome {
    match tool {
        Tool::Paint => brush_chip::flyout_body(ui, brush, facts.is_inpaint, facts.protect),
        Tool::Magic => magic_brush_chip::flyout_body(ui, brush, facts.encoder_pending, facts.protect),
    }
}

fn paint_knobs(ui: &mut Ui, s: &mut BrushSettings, is_inpaint: bool) -> bool {
    let mut committed = false;
    if !is_inpaint {
        committed |= chip::choice_buttons(ui, &BRUSH_MODES, &mut s.mode);
    }
    committed |= chip::slider_row_f32(ui, "Size", &mut s.radius, BRUSH_RADIUS_RANGE, true, |v| fmt::px(v, 0)).commit;
    committed |= chip::slider_row_f32(ui, "Hardness", &mut s.hardness, 0.0..=1.0, false, fmt::percent_tenths).commit;
    if is_inpaint {
        committed |= chip::slider_row_f32(ui, "Expand region", &mut s.inpaint_grow, -16.0..=16.0, false, |v| fmt::signed_px(v, 0)).commit;
        committed |= chip::slider_row_f32(ui, "Edge blend", &mut s.inpaint_feather, 0.0..=32.0, false, |v| fmt::px(v, 0)).commit;
    } else {
        committed |= chip::slider_row_f32(ui, "Opacity", &mut s.strength, 0.0..=1.0, false, fmt::percent).commit;
    }
    committed
}

fn magic_knobs(ui: &mut Ui, s: &mut BrushSettings) -> bool {
    let mut committed = chip::slider_row_f32(ui, "Size", &mut s.radius, BRUSH_RADIUS_RANGE, true, |v| fmt::px(v, 0)).commit;
    committed |= chip::slider_row_f32(ui, "Confidence", &mut s.magic_confidence_threshold, 0.0..=1.0, false, fmt::percent).commit;
    committed |= chip::choice_buttons(ui, &BRUSH_SHAPES, &mut s.shape);
    committed
}
