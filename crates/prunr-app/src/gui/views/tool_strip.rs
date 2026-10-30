//! The strip over the top of the canvas while a brush tool is on: every
//! setting of the brush, in two rows laid out as one grid, so names,
//! tracks and values line up down each column.

use egui::{Align2, Id, Order, Rect, RichText, Ui};

use crate::gui::brush_state::{BrushSettings, StrokeReach, Tool, BRUSH_RADIUS_RANGE, EXPAND_RANGE, FEATHER_RANGE, FILL_FEATHER_RANGE, UNIT_RANGE};
use crate::gui::knob_catalog::{shows, Surface};
use crate::gui::settings::SettingsModel;
use crate::gui::theme;

use super::brush_chip::{self, BrushChipOutcome};
use super::{chip, fmt, magic_brush_chip};

const TOP_INSET: f32 = 12.0;
const SLIDER_WIDTH: f32 = 110.0;
const SLIDER_WIDTH_MIN: f32 = 56.0;
const ROW_HEIGHT: f32 = theme::CHIP_HEIGHT;
const ROW_GAP: f32 = 6.0;

impl Tool {
    fn title(self) -> &'static str {
        match self {
            Tool::Paint => "Paint Brush",
            Tool::Magic => "Magic Brush",
        }
    }
}

/// What the strip needs besides the brush settings.
#[derive(Clone, Copy)]
pub(crate) struct StripFacts {
    pub model: SettingsModel,
    /// Why "Try another" can't run now (`None`: it can).
    pub reroll_refusal: Option<&'static str>,
}

/// What the strip asks of the app this frame.
#[derive(Default)]
pub(crate) struct StripChange {
    pub brush: BrushChipOutcome,
    pub try_another: bool,
}

/// One cell of the strip's grid.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Cell {
    Empty,
    Mode,
    Reach,
    Shape,
    Reset,
    Size,
    Hardness,
    Opacity,
    SelectionFeather,
    Expand,
    FillFeather,
    Pick,
    Reading,
}

const SIZE: &str = "Size";
const HARDNESS: &str = "Hardness";
const OPACITY: &str = "Opacity";
const FEATHER: &str = "Feather";
const EXPAND: &str = "Expand region";

impl Cell {
    /// The names a slider cell shows, which its column's names line up to.
    fn names(self) -> &'static [&'static str] {
        match self {
            Cell::Size => &[SIZE],
            Cell::Hardness => &[HARDNESS],
            Cell::Opacity => &[OPACITY],
            Cell::SelectionFeather | Cell::FillFeather => &[FEATHER],
            Cell::Expand => &[EXPAND],
            Cell::Reading => &[magic_brush_chip::CONFIDENCE, magic_brush_chip::RANGE],
            Cell::Empty | Cell::Mode | Cell::Reach | Cell::Shape | Cell::Reset | Cell::Pick => &[],
        }
    }

    fn is_slider(self) -> bool {
        !self.names().is_empty()
    }
}

/// A brush's grid: columns of a top and a bottom cell, then the end
/// column after a rule, and how it narrows when two rows do not fit
/// even without the title.
struct Grid {
    name: &'static str,
    columns: &'static [[Cell; 2]],
    end: [Cell; 2],
    narrow: Option<Narrow>,
}

/// A grid's three-row form: `columns` in place of its own, which drop
/// its last column, and `below` on a third row from the strip's left
/// edge.
struct Narrow {
    columns: &'static [[Cell; 2]],
    below: Cell,
}

const PAINT_SELECTING: Grid = Grid {
    name: "paint_selecting",
    columns: &[[Cell::Mode, Cell::Reach], [Cell::Size, Cell::Hardness], [Cell::Opacity, Cell::SelectionFeather]],
    end: [Cell::Shape, Cell::Reset],
    narrow: None,
};

const PAINT_ERASING: Grid = Grid {
    name: "paint_erasing",
    columns: &[[Cell::Size, Cell::Reach], [Cell::Expand, Cell::FillFeather]],
    end: [Cell::Shape, Cell::Reset],
    narrow: None,
};

const MAGIC_SELECTING: Grid = Grid {
    name: "magic_selecting",
    columns: &[[Cell::Mode, Cell::Reach], [Cell::Reading, Cell::SelectionFeather]],
    end: [Cell::Pick, Cell::Reset],
    narrow: None,
};

const MAGIC_ERASING: Grid = Grid {
    name: "magic_erasing",
    columns: &[[Cell::Mode, Cell::Reach], [Cell::Reading, Cell::Empty], [Cell::Expand, Cell::FillFeather]],
    end: [Cell::Pick, Cell::Reset],
    // Feather takes the room under Confidence; Expand region goes below.
    narrow: Some(Narrow { columns: &[[Cell::Mode, Cell::Reach], [Cell::Reading, Cell::FillFeather]], below: Cell::Expand }),
};

/// The brush's grid on `model`: strokes shape a selection, or an
/// eraser's area to fill.
fn grid(tool: Tool, model: SettingsModel) -> &'static Grid {
    match (tool, shows(model, Surface::SelectionStrokes)) {
        (Tool::Paint, true) => &PAINT_SELECTING,
        (Tool::Paint, false) => &PAINT_ERASING,
        (Tool::Magic, true) => &MAGIC_SELECTING,
        (Tool::Magic, false) => &MAGIC_ERASING,
    }
}

/// What the sliders adjust besides the strokes to come.
const REACHES: [chip::Choice<StrokeReach>; 3] = [
    chip::Choice { value: StrokeReach::Next, name: "Next", description: "The sliders change only the strokes you draw from now on", enabled: true },
    chip::Choice { value: StrokeReach::Last, name: "Last", description: "The sliders change your latest stroke too", enabled: true },
    chip::Choice { value: StrokeReach::All, name: "All", description: "The sliders change every stroke on this image since it last changed", enabled: true },
];

/// The strip's widths measured in its last two-row frame: everything
/// after the title at slider width `at`, the title with its rule, and
/// the column its three-row form drops (`spill`); and everything after
/// the title in its last three-row frame (0 before one).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Fit {
    core: f32,
    at: f32,
    title: f32,
    spill: f32,
    core3: f32,
}

/// How the strip lays out this frame.
struct Plan {
    title: bool,
    third_row: bool,
    slider: f32,
}

/// The sliders shorten first; then the title goes; then, where the grid
/// allows, a column drops to a third row, and the title comes back if
/// that leaves room for it.
fn plan(fit: Option<Fit>, room: f32, grid: &Grid) -> Plan {
    let Some(fit) = fit else { return Plan { title: true, third_row: false, slider: SLIDER_WIDTH } };
    let sliders = grid.columns.iter().filter(|c| c.iter().any(|cell| cell.is_slider())).count().max(1) as f32;
    let fixed = fit.core - sliders * fit.at;
    // Whole pixels: a fraction measures back a hair off and never settles.
    let slider_for = |title: f32| ((room - title - fixed) / sliders).floor().clamp(SLIDER_WIDTH_MIN, SLIDER_WIDTH);
    let narrowest = fixed + sliders * SLIDER_WIDTH_MIN;
    if narrowest + fit.title <= room {
        Plan { title: true, third_row: false, slider: slider_for(fit.title) }
    } else if narrowest <= room || grid.narrow.is_none() {
        Plan { title: false, third_row: false, slider: slider_for(0.0) }
    } else {
        let estimate = narrowest - (fit.spill + SLIDER_WIDTH_MIN - fit.at);
        let three_rows = if fit.core3 > 0.0 { fit.core3 } else { estimate };
        Plan { title: three_rows + fit.title <= room, third_row: true, slider: SLIDER_WIDTH_MIN }
    }
}

/// The brush settings a cell reads and changes, and what it asks of the app.
struct Knobs<'a> {
    tool: Tool,
    model: SettingsModel,
    brush: &'a mut BrushSettings,
    reach: &'a mut StrokeReach,
    change: BrushChipOutcome,
}

pub(crate) fn render(ui: &mut Ui, canvas_rect: Rect, tool: Tool, brush: &mut BrushSettings, reach: &mut StrokeReach, facts: StripFacts) -> StripChange {
    let action_before = brush.cutout_action;
    let grid = grid(tool, facts.model);
    let mut knobs = Knobs { tool, model: facts.model, brush, reach, change: BrushChipOutcome::default() };
    let mut try_another = false;
    egui::Area::new(Id::new("tool_strip"))
        .order(Order::Foreground)
        .pivot(Align2::CENTER_TOP)
        .fixed_pos(egui::pos2(canvas_rect.center().x, canvas_rect.top() + TOP_INSET))
        .show(ui.ctx(), |ui| {
            let available = (canvas_rect.width() - 2.0 * TOP_INSET).max(0.0);
            ui.set_max_width(available);
            let fit_id = fit_id(tool, facts.model);
            let fit = ui.ctx().data(|d| d.get_temp::<Fit>(fit_id));
            let room = available - theme::pill_frame().total_margin().sum().x;
            let plan = plan(fit, room, grid);
            let rows = if plan.third_row { 3.0 } else { 2.0 };
            let height = rows * ROW_HEIGHT + (rows - 1.0) * ROW_GAP;
            let (title, core, spill_width) = theme::pill_frame().show(ui, |ui| {
                ui.spacing_mut().item_spacing.x = theme::SPACE_SM;
                ui.spacing_mut().slider_width = plan.slider;
                ui.horizontal(|ui| {
                    let start = ui.cursor().left();
                    if plan.title {
                        spanning(ui, height, |ui| ui.label(RichText::new(tool.title()).strong().color(theme::TEXT_PRIMARY)));
                        rule(ui, height);
                    }
                    let core_left = ui.cursor().left();
                    let spill_width = ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = ROW_GAP;
                        let narrow = grid.narrow.as_ref().filter(|_| plan.third_row);
                        let lefts = ui.horizontal(|ui| {
                            // Each column's left edge, and where the next would start.
                            let columns = narrow.map_or(grid.columns, |n| n.columns);
                            let mut lefts = [0.0; 5];
                            for (i, column) in columns.iter().enumerate() {
                                lefts[i] = ui.cursor().left();
                                cells(ui, *column, &mut knobs);
                            }
                            lefts[columns.len()] = ui.cursor().left();
                            lefts
                        }).inner;
                        if let Some(narrow) = narrow {
                            ui.horizontal(|ui| {
                                let width = names_width(ui, [narrow.below, Cell::Empty]);
                                spanning(ui, ROW_HEIGHT, |ui| cell(ui, narrow.below, width, &mut knobs));
                            });
                        }
                        let last = grid.columns.len() - 1;
                        lefts[last + 1] - lefts[last]
                    }).inner;
                    rule(ui, height);
                    cells(ui, grid.end, &mut knobs);
                    if tool == Tool::Paint && facts.model.generates() {
                        rule(ui, height);
                        try_another = spanning(ui, height, |ui| try_another_button(ui, facts.reroll_refusal));
                    }
                    (core_left - start, ui.min_rect().right() - core_left, spill_width)
                }).inner
            }).inner;
            // A three-row frame keeps the two-row measure it was planned from.
            let title = if plan.title { title } else { fit.map_or(0.0, |f| f.title) };
            let measured = match (plan.third_row, fit) {
                (true, Some(fit)) => Fit { core3: core, title, ..fit },
                (true, None) => return,
                (false, _) => Fit { core, at: plan.slider, title, spill: spill_width, core3: fit.map_or(0.0, |f| f.core3) },
            };
            if fit != Some(measured) {
                ui.ctx().data_mut(|d| d.insert_temp(fit_id, measured));
            }
        });
    let mut change = knobs.change;
    change.action_changed = knobs.brush.cutout_action != action_before;
    StripChange { brush: change, try_another }
}

/// Where the strip keeps its measured widths between frames, per brush
/// and model.
pub(crate) fn fit_id(tool: Tool, model: SettingsModel) -> Id {
    Id::new(("tool_strip_fit", tool.title(), grid(tool, model).name))
}

/// `add` centred across the strip's rows.
fn spanning<R>(ui: &mut Ui, height: f32, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.allocate_ui_with_layout(egui::vec2(ui.available_width(), height), egui::Layout::left_to_right(egui::Align::Center), |ui| {
        ui.set_min_height(height);
        add(ui)
    })
    .inner
}

/// A vertical rule down the strip's rows.
fn rule(ui: &mut Ui, height: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(1.0, height), egui::Sense::hover());
    ui.painter().vline(rect.center().x, rect.y_range(), ui.visuals().widgets.noninteractive.bg_stroke);
}

/// One column: a top cell over a bottom one, slider names in line.
fn cells(ui: &mut Ui, column: [Cell; 2], knobs: &mut Knobs) {
    let name_width = names_width(ui, column);
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = ROW_GAP;
        for c in column {
            spanning(ui, ROW_HEIGHT, |ui| cell(ui, c, name_width, knobs));
        }
    });
}

fn names_width(ui: &Ui, column: [Cell; 2]) -> f32 {
    chip::names_width(ui, ("strip", column), || column.iter().flat_map(|c| c.names().iter().copied()).collect())
}

fn cell(ui: &mut Ui, cell: Cell, name_width: f32, knobs: &mut Knobs) {
    let s = &mut *knobs.brush;
    let change = &mut knobs.change;
    let committed = match cell {
        Cell::Empty => false,
        Cell::Mode => brush_chip::mode_buttons(ui, knobs.model, s),
        Cell::Reach => {
            chip::choice_buttons(ui, &REACHES, knobs.reach);
            false
        }
        Cell::Shape => brush_chip::shape_buttons(ui, s),
        Cell::Reset => {
            let resp = chip::reset_button(ui, "Reset", true);
            if chip::tooltip(resp, "Reset the brush", "Put this brush's settings back as your default preset has them.", None).clicked() {
                change.reset = Some(knobs.tool);
            }
            false
        }
        Cell::Size => chip::slider_cell_f32(ui, SIZE, name_width, &mut s.radius, BRUSH_RADIUS_RANGE, true, |v| fmt::px(v, 0)).commit,
        Cell::Hardness => chip::slider_cell_f32(ui, HARDNESS, name_width, &mut s.hardness, UNIT_RANGE, false, fmt::percent_tenths).commit,
        Cell::Opacity => chip::slider_cell_f32(ui, OPACITY, name_width, &mut s.strength, UNIT_RANGE, false, fmt::percent).commit,
        Cell::SelectionFeather => {
            chip::slider_cell_f32(ui, FEATHER, name_width, &mut s.edge_feather, FEATHER_RANGE, false, |v| fmt::off_or(v, 0.1, |v| fmt::px(v, 0))).commit
        }
        // An eraser stroke keeps its region and band in whole pixels.
        Cell::Expand => {
            let change = chip::slider_cell_f32(ui, EXPAND, name_width, &mut s.inpaint_grow, EXPAND_RANGE, false, |v| fmt::signed_px(v, 0));
            s.inpaint_grow = s.inpaint_grow.round();
            change.commit
        }
        Cell::FillFeather => {
            let change = chip::slider_cell_f32(ui, FEATHER, name_width, &mut s.inpaint_feather, FILL_FEATHER_RANGE, false, |v| fmt::off_or(v, 0.5, |v| fmt::px(v, 0)));
            s.inpaint_feather = s.inpaint_feather.round();
            change.commit
        }
        Cell::Pick => magic_brush_chip::pick(ui, s),
        Cell::Reading => magic_brush_chip::reading(ui, s, name_width),
    };
    change.committed |= committed;
}

fn try_another_button(ui: &mut Ui, refusal: Option<&'static str>) -> bool {
    let resp = chip::gated(ui, refusal, |ui| chip::button(ui, chip::ButtonKind::Secondary, try_another_label()));
    let resp = chip::named(resp, "Try another");
    chip::tooltip(resp, "Try another", "Generate this fill again with a new seed. Undo brings the previous one back.", None).clicked()
}

/// The button's text: icon and words, built once.
fn try_another_label() -> &'static str {
    static LABEL: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    LABEL.get_or_init(|| format!("{}  Try another", egui_material_icons::icons::ICON_REFRESH.codepoint))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::brush_state::StrokeDrawn;
    use crate::gui::settings::Settings;

    /// How a test moves the slider in `cell` (with Magic Brush reading by
    /// `depth`); `None` for a cell that holds no slider.
    fn slider(cell: Cell, depth: bool) -> Option<fn(&mut BrushSettings)> {
        match cell {
            Cell::Size => Some(|s| s.radius += 4.0),
            Cell::Hardness => Some(|s| s.hardness = if s.hardness > 0.5 { 0.2 } else { 0.8 }),
            Cell::Opacity => Some(|s| s.strength = if s.strength > 0.5 { 0.3 } else { 0.9 }),
            Cell::SelectionFeather => Some(|s| s.edge_feather += 3.0),
            Cell::Expand => Some(|s| s.inpaint_grow += 3.0),
            Cell::FillFeather => Some(|s| s.inpaint_feather += 3.0),
            Cell::Reading if depth => Some(|s| s.magic_depth_range += 0.05),
            Cell::Reading => Some(|s| s.magic_confidence_threshold = if s.magic_confidence_threshold > 0.5 { 0.2 } else { 0.8 }),
            Cell::Empty | Cell::Mode | Cell::Reach | Cell::Shape | Cell::Reset | Cell::Pick => None,
        }
    }

    /// A stroke `tool` draws under `settings` (a depth pick when `depth`).
    fn stroke(tool: Tool, depth: bool, settings: &Settings) -> StrokeDrawn {
        use prunr_core::selection::BrushMode;
        let (edge, mode) = (settings.stroke_edge(), BrushMode::Add);
        match (tool, depth) {
            (Tool::Paint, _) => crate::gui::tests::fixtures::painted_with_edge(prunr_core::brush::BrushShape::Circle, settings.brush.radius, &[(8.0, 8.0, 2.0)], settings.brush.stamp(), edge),
            (Tool::Magic, true) => StrokeDrawn::Depth {
                depth: std::sync::Arc::new(prunr_core::depth::DepthMap::from_raw(4, 4, vec![0.5; 16])),
                around: 0.5, mode, range: settings.brush.magic_depth_range, edge,
            },
            (Tool::Magic, false) => {
                let m = prunr_core::sam::SAM_MASK_RESOLUTION as usize;
                let output = prunr_core::sam::SamDecoderOutput { masks: vec![0.0; 3 * m * m], iou_predictions: [0.9, 0.1, 0.1] }.best_only();
                StrokeDrawn::Magic { output: std::sync::Arc::new(output), mode, reading: settings.brush.magic_reading(), edge }
            }
        }
    }

    /// Reset puts back every slider on the strip it sits on, on every
    /// model; Opacity carries over a reset.
    #[test]
    fn reset_puts_back_every_slider_on_its_strip() {
        for model in SettingsModel::ALL.into_iter().filter(|&m| shows(m, Surface::Brushes)) {
            for tool in [Tool::Paint, Tool::Magic] {
                let g = grid(tool, model);
                let narrow = g.narrow.iter().flat_map(|n| n.columns.iter().flatten().chain(std::iter::once(&n.below)));
                for &cell in g.columns.iter().flatten().chain(&g.end).chain(narrow).filter(|&&c| c != Cell::Opacity) {
                    for depth in [false, true] {
                        let Some(slide) = slider(cell, depth) else { continue };
                        let shipped = BrushSettings { magic_depth: depth, ..Default::default() };
                        let mut brush = shipped.clone();
                        slide(&mut brush);
                        brush.reset_tool_from(tool, &shipped);
                        assert_eq!(brush, shipped, "{model:?}, {}, {:?}{}", tool.title(), cell.names(), if depth { " (depth)" } else { "" });
                    }
                }
            }
        }
    }

    /// Every slider on every brush strip, on every model, moves a value
    /// its strokes keep, so Next, Last and All decide which strokes it
    /// reaches: a slider the strokes do not keep would change all of them
    /// at once, whatever the switch says.
    #[test]
    fn every_strip_slider_is_a_value_its_strokes_keep() {
        for model in SettingsModel::ALL.into_iter().filter(|&m| shows(m, Surface::Brushes)) {
            for tool in [Tool::Paint, Tool::Magic] {
                let g = grid(tool, model);
                let narrow = g.narrow.iter().flat_map(|n| n.columns.iter().flatten().chain(std::iter::once(&n.below)));
                for &cell in g.columns.iter().flatten().chain(&g.end).chain(narrow) {
                    for depth in [false, true] {
                        let Some(slide) = slider(cell, depth) else { continue };
                        let mut settings = Settings { model, ..Default::default() };
                        settings.brush.magic_depth = depth;
                        let drawn = stroke(tool, depth, &settings);
                        let before = settings.stroke_values();
                        slide(&mut settings.brush);
                        let after = settings.stroke_values();
                        let case = format!("{model:?}, {}, {:?}{}", tool.title(), cell.names(), if depth { " (depth)" } else { "" });
                        assert_ne!(before, after, "{case}: the slider moves a stroke value");
                        let keeps = |redrawn: StrokeDrawn| redrawn.values() != drawn.values();
                        assert!(keeps(drawn.at(drawn.values().moved(before, after))), "{case}: Last redraws the stroke with it");
                        assert!(keeps(drawn.at(drawn.values().adjusted(before, after))), "{case}: All moves the stroke by it");
                    }
                }
            }
        }
    }
}
