//! The strip over the top of the canvas while a brush tool is on: the
//! knobs touched on every stroke, and a button to the rest of them.

use egui::{Align2, Id, Order, Rect, RichText, Ui};
use egui_material_icons::icons::ICON_MORE_HORIZ;

use crate::gui::brush_state::{BrushSettings, BRUSH_RADIUS_RANGE};
use crate::gui::knob_catalog::{shows, Surface};
use crate::gui::settings::SettingsModel;
use crate::gui::theme;

use super::brush_chip::{self, BrushChipOutcome};
use super::{chip, fmt, magic_brush_chip};

const TOP_INSET: f32 = 12.0;
const SLIDER_WIDTH: f32 = 110.0;
const SLIDER_WIDTH_MIN: f32 = 56.0;
/// Everything in the strip but its three sliders, at the widest.
const FIXED_WIDTH: f32 = 760.0;

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
    pub model: SettingsModel,
    pub encoder_pending: bool,
    /// The shown fill can be generated again ("Try another").
    pub can_reroll: bool,
}

/// What the strip asks of the app this frame.
#[derive(Default)]
pub(crate) struct StripChange {
    pub brush: BrushChipOutcome,
    pub try_another: bool,
}

pub(crate) fn render(ui: &mut Ui, canvas_rect: Rect, tool: Tool, brush: &mut BrushSettings, facts: StripFacts) -> StripChange {
    let mut strip = StripChange::default();
    let change = &mut strip.brush;
    egui::Area::new(Id::new("tool_strip"))
        .order(Order::Foreground)
        .pivot(Align2::CENTER_TOP)
        .fixed_pos(egui::pos2(canvas_rect.center().x, canvas_rect.top() + TOP_INSET))
        .show(ui.ctx(), |ui| {
            let available = (canvas_rect.width() - 2.0 * TOP_INSET).max(0.0);
            ui.set_max_width(available);
            let mut dots = None;
            let strip_rect = theme::pill_frame().show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = theme::SPACE_SM;
                    ui.spacing_mut().slider_width = ((available - FIXED_WIDTH) / 3.0).clamp(SLIDER_WIDTH_MIN, SLIDER_WIDTH);
                    ui.label(RichText::new(tool.title()).strong().color(theme::TEXT_PRIMARY));
                    ui.separator();
                    change.committed |= match tool {
                        Tool::Paint => paint_knobs(ui, brush, facts.model),
                        Tool::Magic => magic_knobs(ui, brush, facts.model),
                    };
                    ui.separator();
                    dots = Some(chip::tooltip(
                        chip::icon_square_button(ui, ICON_MORE_HORIZ.codepoint, theme::TEXT_PRIMARY, theme::BG_SECONDARY),
                        tool.more(),
                        "",
                        None,
                    ));
                    if tool == Tool::Paint && facts.model == SettingsModel::SdInpaint {
                        ui.separator();
                        let reason = (!facts.can_reroll).then_some("Erase something first; this generates that fill again.");
                        let resp = chip::gated(ui, reason, |ui| {
                            chip::button(ui, chip::ButtonKind::Secondary, try_another_label())
                        });
                        let resp = chip::named(resp, "Try another");
                        strip.try_another = chip::tooltip(resp, "Try another", "Generate this fill again with a new seed. Undo brings the previous one back.", None).clicked();
                    }
                });
            }).response.rect;
            if let Some(dots) = dots {
                chip::flyout_under(Id::new(("tool_strip_more", tool.title())), &dots, strip_rect, |ui| {
                    change.merge(panel(ui, tool, brush, facts));
                });
            }
        });
    strip
}

/// The button's text: icon and words, built once.
fn try_another_label() -> &'static str {
    static LABEL: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    LABEL.get_or_init(|| format!("{}  Try another", egui_material_icons::icons::ICON_REFRESH.codepoint))
}

fn panel(ui: &mut Ui, tool: Tool, brush: &mut BrushSettings, facts: StripFacts) -> BrushChipOutcome {
    match tool {
        Tool::Paint => brush_chip::flyout_body(ui, brush, facts.model),
        Tool::Magic => magic_brush_chip::flyout_body(ui, brush, facts.encoder_pending, facts.model),
    }
}

fn paint_knobs(ui: &mut Ui, s: &mut BrushSettings, model: SettingsModel) -> bool {
    let mut committed = false;
    if shows(model, Surface::SelectionStrokes) {
        committed |= brush_chip::mode_buttons(ui, model, &mut s.mode);
    }
    committed |= chip::slider_row_f32(ui, "Size", &mut s.radius, BRUSH_RADIUS_RANGE, true, |v| fmt::px(v, 0)).commit;
    if shows(model, Surface::SelectionStrokes) {
        committed |= chip::slider_row_f32(ui, "Hardness", &mut s.hardness, 0.0..=1.0, false, fmt::percent_tenths).commit;
        committed |= chip::slider_row_f32(ui, "Opacity", &mut s.strength, 0.0..=1.0, false, fmt::percent).commit;
    }
    if shows(model, Surface::FillRegionKnobs) {
        committed |= brush_chip::expand_region_row(ui, s).commit;
        committed |= brush_chip::feather_row(ui, s).commit;
    }
    committed
}

fn magic_knobs(ui: &mut Ui, s: &mut BrushSettings, model: SettingsModel) -> bool {
    let mut committed = brush_chip::mode_buttons(ui, model, &mut s.mode);
    committed |= chip::slider_row_f32(ui, "Confidence", &mut s.magic_confidence_threshold, 0.0..=1.0, false, fmt::percent).commit;
    committed
}
