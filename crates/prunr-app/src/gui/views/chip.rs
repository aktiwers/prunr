//! Compact "chip" widgets for the adjustments toolbar.
//!
//! A chip renders as a small pill-shaped button: `[icon  value]`. Clicking
//! it opens a popover containing the full slider / color picker + a reset-
//! to-default button. Non-default values get an accent outline so the user
//! can see at a glance which knobs have been tuned.
//!
//! Chips take `&mut` to their underlying value only, never `&mut PrunrApp`,
//! so they stay reusable and testable.

use egui::{Color32, RichText, Response, Ui};
use egui::ecolor::Hsva;
use egui::TextureId;
use egui_material_icons::icons::*;

use std::borrow::Cow;

use super::shortcuts::{self, Action};
use crate::gui::knob_catalog::StaticKnob;
use crate::gui::theme;
use crate::gui::views::effects::paint_mesh;

const CHIP_PADDING_X: f32 = 8.0;

/// Return value of every chip function. Callers aggregate these into a
/// `ToolbarChange` and use them to decide whether a live-preview dispatch
/// should debounce or fire immediately.
///
/// - `changed` — the underlying value was edited this frame.
/// - `commit` — the change is "settled" (slider released, checkbox toggled,
///   color picked) and a pending preview should flush now instead of waiting
///   for debounce. Sliders mid-drag set `changed=true, commit=false` so a
///   flurry of drag events debounces into a single rerun.
#[derive(Default, Debug, Clone, Copy)]
pub struct ChipChange {
    pub changed: bool,
    pub commit: bool,
}

/// Returns true when a slider interaction has "settled" — drag released or
/// value changed without an active drag (keyboard, click-jump). Used to
/// flip the `commit` flag so live preview flushes instead of debouncing.
pub(super) fn slider_settled(resp: &egui::Response) -> bool {
    resp.drag_stopped() || (resp.changed() && !resp.dragged())
}

/// Run `body` with `base` as the resting fill of every button inside,
/// brightening it on hover and again while pressed. A fixed `.fill()`
/// on a button would freeze all three states, which reads as a dead
/// control; this is the one place the three shades are defined.
pub(super) fn with_fill<R>(ui: &mut Ui, base: Color32, body: impl FnOnce(&mut Ui) -> R) -> R {
    ui.scope(|ui| {
        let w = &mut ui.visuals_mut().widgets;
        w.inactive.weak_bg_fill = base;
        w.hovered.weak_bg_fill = base.lerp_to_gamma(Color32::WHITE, 0.08);
        w.active.weak_bg_fill = base.lerp_to_gamma(Color32::WHITE, 0.18);
        w.open.weak_bg_fill = base.lerp_to_gamma(Color32::WHITE, 0.08);
        body(ui)
    })
    .inner
}

/// `accent` draws an accent border (the non-default value indicator).
pub(super) fn chip_button(ui: &mut Ui, icon: &str, value: &str, accent: bool) -> Response {
    let text = if compact(ui) { icon.to_owned() } else { format!("{icon}  {value}") };
    chip_face(ui, text, accent)
}

/// A chip that opens a menu: the face ends in a chevron.
pub(super) fn menu_chip_button(ui: &mut Ui, icon: &str, value: &str) -> Response {
    let chevron = ICON_EXPAND_MORE.codepoint;
    let text = if compact(ui) { format!("{icon} {chevron}") } else { format!("{icon}  {value}  {chevron}") };
    chip_face(ui, text, false)
}

const COMPACT: &str = "chip_compact";

/// A toolbar row whose chips drop their words one at a time, from the
/// right, until the row fits its room. Each chip's width with and
/// without its words is measured as it lays out, and the next frame
/// decides from them.
pub(super) struct TrimmedRow {
    id: egui::Id,
    last: Vec<ChipWidth>,
    now: Vec<ChipWidth>,
    /// The chips showing their icon only, by key.
    trimmed: Vec<&'static str>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ChipWidth {
    key: &'static str,
    /// With its words; 0 until laid out so.
    words: f32,
    /// Its icon only; 0 until laid out so.
    icon: f32,
}

/// An icon chip's width before one has been laid out.
const ICON_CHIP_GUESS: f32 = theme::CHIP_HEIGHT + 4.0;

impl TrimmedRow {
    /// The row `id`, given `room` for its chips.
    pub(super) fn begin(ui: &Ui, id: egui::Id, room: f32) -> Self {
        let last: Vec<ChipWidth> = ui.data(|d| d.get_temp(id)).unwrap_or_default();
        let n = trimmed_count(&last, ui.spacing().item_spacing.x, room);
        let trimmed = last[last.len() - n..].iter().map(|c| c.key).collect();
        Self { id, last, now: Vec::new(), trimmed }
    }

    /// One chip of the row, `key` naming it from frame to frame.
    pub(super) fn chip<R>(&mut self, ui: &mut Ui, key: &'static str, body: impl FnOnce(&mut Ui) -> R) -> R {
        let iconic = self.trimmed.contains(&key);
        let r = ui.scope(|ui| compact_chips(ui, iconic, body));
        let width = r.response.rect.width();
        let was = self.last.iter().find(|c| c.key == key).copied().unwrap_or(ChipWidth { key, words: 0.0, icon: 0.0 });
        self.now.push(if iconic { ChipWidth { icon: width, ..was } } else { ChipWidth { words: width, ..was } });
        r.inner
    }

    /// Keep this frame's widths for the next; another frame follows when
    /// they changed, so the row settles at once.
    pub(super) fn finish(self, ui: &Ui) {
        let settled = self.now.len() == self.last.len()
            && self.now.iter().zip(&self.last).all(|(a, b)| a.key == b.key && (a.words - b.words).abs() <= 0.5 && (a.icon - b.icon).abs() <= 0.5);
        if !settled {
            ui.data_mut(|d| d.insert_temp(self.id, self.now));
            ui.ctx().request_repaint();
        }
    }
}

/// How many chips, from the right, show their icon only so the row fits
/// `room`: the fewest that do, or all of them.
fn trimmed_count(chips: &[ChipWidth], spacing: f32, room: f32) -> usize {
    let gaps = spacing * chips.len().saturating_sub(1) as f32;
    let icon = |c: &ChipWidth| if c.icon > 0.0 { c.icon } else { ICON_CHIP_GUESS };
    (0..=chips.len())
        .find(|&n| {
            let (kept, trimmed) = chips.split_at(chips.len() - n);
            kept.iter().map(|c| c.words).sum::<f32>() + trimmed.iter().map(icon).sum::<f32>() + gaps <= room
        })
        .unwrap_or(chips.len())
}

/// Run `body` with its chips showing their icon only, when `on`: a
/// toolbar row too narrow for their words. Tooltips and names stay.
pub(super) fn compact_chips<R>(ui: &mut Ui, on: bool, body: impl FnOnce(&mut Ui) -> R) -> R {
    with_flag(ui, COMPACT, on, body)
}

fn compact(ui: &Ui) -> bool {
    flag(ui, COMPACT)
}

/// Run `body` with the ambient flag `key` set to `on`, restoring it after.
fn with_flag<R>(ui: &mut Ui, key: &str, on: bool, body: impl FnOnce(&mut Ui) -> R) -> R {
    let id = egui::Id::new(key);
    let outer = flag(ui, key);
    ui.data_mut(|d| d.insert_temp(id, on));
    let out = body(ui);
    ui.data_mut(|d| d.insert_temp(id, outer));
    out
}

fn flag(ui: &Ui, key: &str) -> bool {
    ui.data(|d| d.get_temp::<bool>(egui::Id::new(key)).unwrap_or(false))
}

fn chip_face(ui: &mut Ui, text: String, accent: bool) -> Response {
    let stroke = if accent {
        egui::Stroke::new(theme::STROKE_DEFAULT, theme::ACCENT_BRIGHT)
    } else {
        egui::Stroke::new(theme::STROKE_DEFAULT, Color32::TRANSPARENT)
    };
    let btn = egui::Button::new(
        RichText::new(text.as_str()).color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY),
    )
    .stroke(stroke)
    .corner_radius(theme::BUTTON_ROUNDING)
    .min_size(egui::vec2(0.0, theme::CHIP_HEIGHT));
    let resp = with_fill(ui, theme::BG_SECONDARY, |ui| {
        let saved_padding = ui.spacing().button_padding;
        ui.spacing_mut().button_padding = egui::vec2(CHIP_PADDING_X, 4.0);
        let resp = ui.add(btn);
        ui.spacing_mut().button_padding = saved_padding;
        resp
    });
    crate::gui::automation::record::clip(ui.ctx(), &text, resp.rect, crate::gui::automation::record::room(ui));
    resp
}

/// A soft brush stamp's look: the solid core (its radius and colour, when
/// it is big enough to draw) and the falloff rings around it as
/// `(distance from centre, stroke)`, following the same smoothstep curve
/// as `paint_circle`.
fn falloff(
    outer: f32,
    hardness: f32,
    color: egui::Color32,
    base_alpha: u8,
    steps: u32,
) -> (Option<(f32, egui::Color32)>, impl Iterator<Item = (f32, egui::Stroke)>) {
    use prunr_core::math::smoothstep;
    let inner = outer * hardness.clamp(0.0, 1.0);
    let core = (inner >= 0.5)
        .then(|| (inner, egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), base_alpha)));
    let span = (outer - inner).max(0.001);
    let rings = (0..steps).filter_map(move |i| {
        let t = (i as f32 + 0.5) / steps as f32;
        let dist = inner + span * t;
        let intensity = if span < 0.5 { 0.0 } else { smoothstep(1.0 - t) };
        let a = (base_alpha as f32 * intensity) as u8;
        if a == 0 { return None; }
        let stroke_color = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), a);
        Some((dist, egui::Stroke::new(span / steps as f32 * 1.4, stroke_color)))
    });
    (core, rings)
}


/// A soft round brush stamp: a solid core and fading rings out to
/// `outer`, as `hardness` says.
pub(super) fn falloff_circle_shapes(
    center: egui::Pos2,
    outer: f32,
    hardness: f32,
    color: egui::Color32,
    base_alpha: u8,
    steps: u32,
) -> impl Iterator<Item = egui::Shape> {
    let (core, rings) = falloff(outer, hardness, color, base_alpha, steps);
    core.map(|(inner, solid)| egui::Shape::circle_filled(center, inner, solid)).into_iter()
        .chain(rings.into_iter().map(move |(dist, stroke)| egui::Shape::circle_stroke(center, dist, stroke)))
}

/// The square stamp of `falloff_circle_shapes`: an axis-aligned rect
/// fading toward the corners (chebyshev).
pub(super) fn falloff_square_shapes(
    center: egui::Pos2,
    outer: f32,
    hardness: f32,
    color: egui::Color32,
    base_alpha: u8,
    steps: u32,
) -> impl Iterator<Item = egui::Shape> {
    let square = move |half: f32| egui::Rect::from_center_size(center, egui::vec2(half * 2.0, half * 2.0));
    let (core, rings) = falloff(outer, hardness, color, base_alpha, steps);
    core.map(|(inner, solid)| egui::Shape::rect_filled(square(inner), 0.0, solid)).into_iter()
        .chain(rings.into_iter().map(move |(dist, stroke)| egui::Shape::rect_stroke(square(dist), 0.0, stroke, egui::StrokeKind::Outside)))
}

/// Square icon-only button with an active/inactive visual state.
///
/// `icon` is `&str` rather than `char` even though every caller passes a
/// single Unicode code point — `egui::RichText::new` doesn't accept
/// `char` directly, so a `char` parameter would force a single-char
/// `String` allocation per call.
pub(super) fn icon_toggle_button(ui: &mut Ui, icon: &str, active: bool) -> Response {
    if active {
        icon_square_button(ui, icon, theme::TEXT_PRIMARY, theme::ACCENT)
    } else {
        icon_square_button(ui, icon, theme::TEXT_SECONDARY, theme::BG_SECONDARY)
    }
}

/// Momentary icon button (Reset, Help, Delete). Same size as
/// `icon_toggle_button` but never filled with the accent — the fill is
/// the toggle's "on" state. `color` tints the glyph (DESTRUCTIVE for Delete).
pub(super) fn icon_action_button(ui: &mut Ui, icon: &str, color: Color32) -> Response {
    icon_square_button(ui, icon, color, theme::BG_SECONDARY)
}

pub(super) fn icon_square_button(ui: &mut Ui, icon: &str, color: Color32, fill: Color32) -> Response {
    with_fill(ui, fill, |ui| {
        ui.add(
            egui::Button::new(RichText::new(icon).color(color).size(theme::ICON_SIZE_SMALL))
                .corner_radius(theme::BUTTON_ROUNDING)
                .min_size(egui::vec2(theme::CHIP_HEIGHT, theme::CHIP_HEIGHT)),
        )
    })
}

/// The one hover tooltip for every control: strong title, one-sentence
/// body, and the keys when the control has a shortcut. Pass an empty
/// body for icon buttons whose title says it all.
pub(super) fn tooltip(resp: Response, title: &str, body: &str, shortcut: Option<Action>) -> Response {
    crate::gui::automation::record::tooltip(&resp.ctx, title, body, shortcut);
    named(resp, title).on_hover_ui(|ui| {
        ui.label(RichText::new(title).strong().color(theme::TEXT_PRIMARY));
        if !body.is_empty() {
            ui.add_space(theme::SPACE_XS);
            ui.label(
                RichText::new(body)
                    .color(theme::TEXT_PRIMARY)
                    .size(theme::FONT_SIZE_SECONDARY),
            );
        }
        if let Some(action) = shortcut {
            ui.add_space(theme::SPACE_XS);
            ui.label(
                RichText::new(&*shortcuts::keys(ui.ctx(), action))
                    .color(theme::TEXT_SECONDARY)
                    .size(theme::FONT_SIZE_SECONDARY),
            );
        }
    })
}

/// A destructive button that needs two clicks: the first arms it (it then
/// reads `armed_text`), the second returns true. Arming is held under
/// `scope`, so arming another `key` disarms this one, and the caller
/// clears `scope` when its surface closes.
pub(super) fn confirm_button(ui: &mut Ui, scope: egui::Id, key: &str, idle: RichText, armed_text: RichText, name: &str) -> bool {
    let armed = ui.data(|d| d.get_temp::<String>(scope)).as_deref() == Some(key);
    let clicked = named(ui.small_button(if armed { armed_text } else { idle }), name)
        .on_hover_text(if armed { "Click again to confirm" } else { name })
        .clicked();
    if clicked && !armed {
        ui.data_mut(|d| d.insert_temp(scope, key.to_string()));
    }
    if clicked && armed {
        ui.data_mut(|d| d.remove::<String>(scope));
    }
    clicked && armed
}

/// Names a control for the accessibility tree, which is what the test
/// harness and the control socket find widgets by. Icon-only buttons
/// have no readable name of their own; the tooltip title is it. Free
/// when the tree is off.
pub(super) fn named(resp: Response, name: &str) -> Response {
    resp.ctx.accesskit_node_builder(resp.id, |node| node.set_label(name));
    resp
}

/// `named` for a name built per widget (from a file or model name):
/// built only while the accessibility tree is being read.
pub(super) fn named_with(resp: Response, name: impl FnOnce() -> String) -> Response {
    resp.ctx.accesskit_node_builder(resp.id, |node| node.set_label(name()));
    resp
}

/// The one Reset control, in popovers and panels alike: a ghost pill
/// with the restart icon, greyed while there is nothing to reset.
pub(super) fn reset_button(ui: &mut Ui, label: &str, enabled: bool) -> Response {
    ui.add_enabled_ui(enabled, |ui| button(ui, ButtonKind::Ghost, &format!("{}  {label}", ICON_RESTART_ALT.codepoint))).inner
}

/// Wire a popup to a chip button. Handles toggle-on-click.
/// Uses the legacy `popup_below_widget` API; the deprecation is isolated
/// to this helper.
#[allow(deprecated)]
pub(super) fn popup_for(
    ui: &mut Ui,
    id: egui::Id,
    resp: &Response,
    body: impl FnOnce(&mut Ui),
) {
    if resp.clicked() {
        ui.memory_mut(|m| m.toggle_popup(id));
    }
    outline_when_open(resp, id);
    egui::popup_below_widget(
        ui,
        id,
        resp,
        egui::PopupCloseBehavior::CloseOnClickOutside,
        |ui| {
            ui.set_min_width(theme::POPOVER_WIDTH);
            body(ui);
        },
    );
}

/// A popover for knobs that change the image live. It stays open while
/// the user works on the canvas: its chip, Escape or another chip closes
/// it, never a click elsewhere. While one of its sliders is held it fades
/// so the image under it stays readable.
pub(super) fn flyout_for(id: egui::Id, resp: &Response, body: impl FnOnce(&mut Ui)) {
    show_flyout(id, resp, below_chip(resp, id), |ui| {
        // The frame is drawn inside so it fades with the contents.
        egui::Frame::popup(ui.style()).show(ui, |ui| {
            ui.set_min_width(theme::POPOVER_WIDTH);
            body(ui);
        });
    });
}

/// A flyout popup under the chip `resp`, falling back to its other sides.
fn below_chip(resp: &Response, id: egui::Id) -> egui::Popup<'static> {
    egui::Popup::from_toggle_button_response(resp)
        .id(id)
        .align(egui::RectAlign::BOTTOM_START)
        // A chip near the right edge (the strip's dots) opens leftwards.
        .align_alternatives(&[egui::RectAlign::BOTTOM_END, egui::RectAlign::TOP_START, egui::RectAlign::TOP_END])
}

/// What every flyout shares: its chip's outline while open, clicks
/// elsewhere ignored, registered as a flyout, and faded while one of its
/// sliders is held. `body` draws its own frame so the frame fades too.
fn show_flyout(id: egui::Id, chip: &Response, popup: egui::Popup<'_>, body: impl FnOnce(&mut Ui)) {
    outline_when_open(chip, id);
    popup
        .close_behavior(egui::PopupCloseBehavior::IgnoreClicks)
        .frame(egui::Frame::NONE)
        .show(|ui| {
            theme::register_flyout(ui.ctx(), id);
            if scrubbing_in(ui) {
                ui.set_opacity(theme::SCRUB_OPACITY);
            }
            body(ui);
        });
}


/// The accent outline on the chip whose popup is open, so the eye finds
/// its way back from the panel to the control that opened it.
fn outline_when_open(resp: &Response, id: egui::Id) {
    if egui::Popup::is_id_open(&resp.ctx, id) {
        resp.ctx.layer_painter(resp.layer_id).rect_stroke(
            resp.rect,
            theme::BUTTON_ROUNDING,
            egui::Stroke::new(theme::STROKE_DEFAULT, theme::ACCENT_BRIGHT),
            egui::StrokeKind::Outside,
        );
    }
}

/// A drag that began on this layer is still held.
fn scrubbing_in(ui: &Ui) -> bool {
    let ctx = ui.ctx();
    if ctx.dragged_id().is_none() {
        return false;
    }
    // Read the input first, then ask for the layer: `layer_id_at` takes
    // the context lock again, and a nested read deadlocks against a
    // writer waiting on another thread.
    let origin = ctx.input(|i| i.pointer.primary_down().then_some(i.pointer.press_origin()).flatten());
    origin.is_some_and(|o| ctx.layer_id_at(o) == Some(ui.layer_id()))
}

/// A percent stored as `u8`, shown and dragged as 0–100 %.
pub fn slider_row_pct(ui: &mut Ui, label: &str, pct: &mut u8, format: impl Fn(f32) -> String) -> ChipChange {
    let mut v = f32::from(*pct) / 100.0;
    let change = slider_row_f32(ui, label, &mut v, 0.0..=1.0, false, format);
    if change.changed {
        *pct = (v * 100.0).round() as u8;
    }
    change
}

/// Float slider row with the log scale and formatter float knobs need.
pub fn slider_row_f32(
    ui: &mut Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    logarithmic: bool,
    format: impl Fn(f32) -> String,
) -> ChipChange {
    slider_f32(ui, label, None, value, range, logarithmic, format)
}

/// `slider_row_f32` with its name in a column `name_width` wide, so the
/// sliders stacked in one column of a strip line up.
pub(super) fn slider_cell_f32(
    ui: &mut Ui,
    label: &str,
    name_width: f32,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    logarithmic: bool,
    format: impl Fn(f32) -> String,
) -> ChipChange {
    slider_f32(ui, label, Some(name_width), value, range, logarithmic, format)
}

/// The width of the widest of `names` as a strip names its sliders.
/// Kept per `key`, so the names are laid out once, not every frame.
pub(super) fn names_width(ui: &Ui, key: impl std::hash::Hash, names: impl FnOnce() -> Vec<&'static str>) -> f32 {
    cached_width(ui, egui::Id::new(("names_width", key)), || {
        let font = egui::FontId::proportional(theme::FONT_SIZE_SECONDARY);
        names().iter()
            .map(|n| ui.painter().layout_no_wrap((*n).to_owned(), font.clone(), Color32::WHITE).size().x)
            .fold(0.0, f32::max)
            .ceil()
    })
}

/// A width measured once under `id` and kept: text is laid out only on
/// the first frame that asks.
fn cached_width(ui: &Ui, id: egui::Id, measure: impl FnOnce() -> f32) -> f32 {
    if let Some(width) = ui.data(|d| d.get_temp::<f32>(id)) {
        return width;
    }
    let width = measure();
    ui.data_mut(|d| d.insert_temp(id, width));
    width
}

fn slider_f32(
    ui: &mut Ui,
    label: &str,
    name_width: Option<f32>,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    logarithmic: bool,
    format: impl Fn(f32) -> String,
) -> ChipChange {
    let (lo, hi) = (*range.start() as f64, *range.end() as f64);
    let value_w = value_box_width(ui, (label, lo.to_bits(), hi.to_bits(), logarithmic), || {
        sample_values(lo, hi, logarithmic).map(|v| format(v as f32)).collect()
    });
    let (change, resp) = labelled_slider(
        ui,
        label,
        name_width,
        egui::Slider::new(value, range)
            .custom_formatter(move |v, _| format(v as f32))
            .logarithmic(logarithmic),
        value_w,
    );
    paint_handle(ui, &resp, slider_fraction(*value as f64, lo, hi, logarithmic));
    change
}

/// A slider and its value box with no label beside it, for a row that
/// names the knob itself.
pub(super) fn bare_slider_f32(ui: &mut Ui, name: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>, format: impl Fn(f32) -> String) -> ChipChange {
    let (lo, hi) = (*range.start() as f64, *range.end() as f64);
    let value_w = value_box_width(ui, (name, lo.to_bits(), hi.to_bits(), false), || {
        sample_values(lo, hi, false).map(|v| format(v as f32)).collect()
    });
    let resp = chip_named_slider(ui, name, egui::Slider::new(value, range).custom_formatter(move |v, _| format(v as f32)), value_w);
    paint_handle(ui, &resp, slider_fraction(*value as f64, lo, hi, false));
    ChipChange { changed: resp.changed(), commit: slider_settled(&resp) }
}

/// Label above a full-width slider whose box shows the value — the one
/// slider layout for popovers and Settings.
pub fn slider_row<T: egui::emath::Numeric>(
    ui: &mut Ui,
    label: &str,
    value: &mut T,
    range: std::ops::RangeInclusive<T>,
) -> ChipChange {
    let (lo, hi) = (range.start().to_f64(), range.end().to_f64());
    // Every caller's range is whole numbers, whose widest text is an end.
    let value_w = value_box_width(ui, (label, lo.to_bits(), hi.to_bits(), false), || vec![format!("{lo}"), format!("{hi}")]);
    let (change, resp) = labelled_slider(ui, label, None, egui::Slider::new(value, range), value_w);
    paint_handle(ui, &resp, slider_fraction(value.to_f64(), lo, hi, false));
    change
}

/// A slider named `label`: in a panel row, or with its name beside it,
/// in a column `name_width` wide when given.
fn labelled_slider(ui: &mut Ui, label: &str, name_width: Option<f32>, slider: egui::Slider<'_>, value_w: f32) -> (ChipChange, Response) {
    if in_panel_rows(ui) {
        return panel_slider(ui, label, slider, value_w);
    }
    let room = crate::gui::automation::record::room(ui);
    let text = label;
    let name = egui::Label::new(RichText::new(text).color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_SECONDARY));
    let label = match name_width {
        Some(width) => ui.allocate_ui_with_layout(egui::vec2(width, ui.spacing().interact_size.y), egui::Layout::left_to_right(egui::Align::Center), |ui| {
            ui.set_min_width(width);
            ui.add(name)
        }).inner,
        None => ui.add(name),
    };
    crate::gui::automation::record::clip(ui.ctx(), text, label.rect, room);
    let resp = add_slider(ui, slider, value_w).labelled_by(label.id);
    (ChipChange { changed: resp.changed(), commit: slider_settled(&resp) }, resp)
}

/// Values across `lo..=hi` whose texts stand for every width a knob's
/// value can take: the ends, zero, and points between (spaced evenly in
/// log10 on a logarithmic knob).
fn sample_values(lo: f64, hi: f64, logarithmic: bool) -> impl Iterator<Item = f64> {
    const STEPS: u32 = 16;
    let between = (0..=STEPS).map(move |i| {
        let t = f64::from(i) / f64::from(STEPS);
        if logarithmic && lo > 0.0 { lo * (hi / lo).powf(t) } else { lo + (hi - lo) * t }
    });
    between.chain((lo < 0.0 && hi > 0.0).then_some(0.0))
}

/// A value box as wide as the widest of `texts`, so the box and every
/// control after it in its row hold still while the value changes. Kept
/// per knob (`key`), so the texts are formatted once, not every frame.
fn value_box_width(ui: &Ui, key: impl std::hash::Hash, texts: impl FnOnce() -> Vec<String>) -> f32 {
    cached_width(ui, egui::Id::new(("value_box_width", key)), || {
        let font = egui::TextStyle::Button.resolve(ui.style());
        let widest = texts().into_iter()
            .map(|t| ui.painter().layout_no_wrap(t, font.clone(), Color32::WHITE).size().x)
            .fold(0.0, f32::max);
        (widest + 2.0 * ui.spacing().button_padding.x).max(ui.spacing().interact_size.x).ceil()
    })
}

/// The key under which `panel_rows` marks a panel whose sliders lay out
/// as panel rows.
const PANEL_ROWS: &str = "chip_panel_rows";
const PANEL_NAME_WIDTH: f32 = 116.0;
/// A panel row's height: the segmented control's, so every row's name
/// centres on its control.
const PANEL_ROW_HEIGHT: f32 = 30.0;
const PANEL_VALUE_WIDTH: f32 = 74.0;

/// Run `body` with its sliders laid out as panel rows: the
/// name in a fixed column, the slider filling the row, the value at its end.
pub(super) fn panel_rows<R>(ui: &mut Ui, body: impl FnOnce(&mut Ui) -> R) -> R {
    let width = ui.spacing().slider_width;
    let out = with_flag(ui, PANEL_ROWS, true, body);
    ui.spacing_mut().slider_width = width;
    out
}

fn in_panel_rows(ui: &Ui) -> bool {
    flag(ui, PANEL_ROWS)
}

/// A panel row's name, in its fixed column.
pub(super) fn panel_name(ui: &mut Ui, label: &str) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(PANEL_NAME_WIDTH, PANEL_ROW_HEIGHT), egui::Sense::hover());
    let name = ui.painter().layout_no_wrap(label.to_owned(), egui::FontId::proportional(theme::FONT_SIZE_BODY), theme::TEXT_PRIMARY);
    let at = egui::pos2(rect.left(), rect.center().y - name.size().y / 2.0);
    crate::gui::automation::record::clip(ui.ctx(), label, egui::Rect::from_min_size(at, name.size()), rect);
    ui.painter().galley(at, name, theme::TEXT_PRIMARY);
}

fn panel_slider(ui: &mut Ui, label: &str, slider: egui::Slider<'_>, value_w: f32) -> (ChipChange, Response) {
    // On the caller's ui too: the handle is painted from its slider width.
    ui.spacing_mut().slider_width = (ui.available_width() - PANEL_NAME_WIDTH - PANEL_VALUE_WIDTH).max(60.0);
    ui.horizontal(|ui| {
        panel_name(ui, label);
        let resp = chip_named_slider(ui, label, slider, value_w);
        (ChipChange { changed: resp.changed(), commit: slider_settled(&resp) }, resp)
    })
    .inner
}

/// A slider with its value box, named `label` for the tree.
fn chip_named_slider(ui: &mut Ui, label: &str, slider: egui::Slider<'_>, value_w: f32) -> Response {
    named(add_slider(ui, slider, value_w), label)
}

/// The value box's padding above and below its number.
const SLIDER_VALUE_PAD_Y: f32 = 1.0;

/// A slider and its value box. The box keeps egui's slim vertical
/// padding: taller than the slider, it would make the response (the
/// union of both) taller than the rail, and the white handle, painted
/// at the response's centre, would slide off egui's own handle and
/// leave its outline showing as a ring.
fn add_slider(ui: &mut Ui, slider: egui::Slider<'_>, value_w: f32) -> Response {
    ui.scope(|ui| {
        ui.spacing_mut().button_padding.y = SLIDER_VALUE_PAD_Y;
        // egui sizes the value box to its text, at least this wide.
        ui.spacing_mut().interact_size.x = value_w;
        ui.add(slider.show_value(true))
    })
    .inner
}

const HANDLE: Color32 = Color32::from_rgb(0xf4, 0xf4, 0xf4);

/// Where a slider's handle sits, 0–1, by egui's own mapping (log10
/// between the bounds for a logarithmic one, 1e-6 standing in for 0).
fn slider_fraction(value: f64, lo: f64, hi: f64, logarithmic: bool) -> f32 {
    if hi <= lo {
        return 0.5;
    }
    let v = value.clamp(lo, hi);
    let t = if logarithmic && lo >= 0.0 {
        let floor = |x: f64| x.max(1e-6).log10();
        (floor(v) - floor(lo)) / (floor(hi) - floor(lo))
    } else {
        (v - lo) / (hi - lo)
    };
    t.clamp(0.0, 1.0) as f32
}

/// The white handle, painted over egui's (which shares the
/// rail's colour) at the same place and size: the slider is the row's
/// first `slider_width`, its handle radius a 2.5th of its thickness.
fn paint_handle(ui: &Ui, resp: &Response, fraction: f32) {
    let thickness = ui.text_style_height(&egui::TextStyle::Body).max(ui.spacing().interact_size.y);
    let width = ui.spacing().slider_width;
    let radius = thickness / 2.5;
    let x = egui::lerp(resp.rect.left() + radius..=resp.rect.left() + width - radius, fraction);
    let center = egui::pos2(x, resp.rect.center().y);
    let radius = radius + ui.style().interact(resp).expansion;
    let fill = if ui.is_enabled() { HANDLE } else { HANDLE.gamma_multiply(0.5) };
    ui.painter().circle_filled(center + egui::vec2(0.0, 1.0), radius + 1.0, Color32::from_black_alpha(90));
    ui.painter().circle_filled(center, radius, fill);
}

/// A list row's fill: the purple wash when picked, a faint one on hover.
pub(super) fn row_wash(resp: &Response, selected: bool) -> Color32 {
    if selected {
        theme::ROW_PICKED
    } else if resp.hovered() {
        theme::ROW_HOVER
    } else {
        Color32::TRANSPARENT
    }
}

const ROW_CHECK_SIZE: f32 = 16.0;

/// The picked row's check, `inset` from its right edge.
fn paint_row_check(painter: &egui::Painter, rect: egui::Rect, inset: f32) {
    painter.text(egui::pos2(rect.right() - inset, rect.center().y), egui::Align2::RIGHT_CENTER, ICON_CHECK.codepoint,
        egui::FontId::proportional(ROW_CHECK_SIZE), theme::ACCENT_RING);
}

/// A row's accessible name: its title, and its description on a line under.
fn row_name(title: &str, description: &str) -> String {
    if description.is_empty() { title.to_owned() } else { format!("{title}\n{description}") }
}

/// A choice list row's geometry: the icon circle's inset and radius, the
/// gap to the text, and the room kept right of the text for the check.
const LIST_ICON_INSET: f32 = 12.0;
const LIST_ICON_RADIUS: f32 = 15.0;
const LIST_TEXT_GAP: f32 = 12.0;
const LIST_CHECK_INSET: f32 = 14.0;
const LIST_CHECK_ROOM: f32 = LIST_CHECK_INSET + ROW_CHECK_SIZE + LIST_TEXT_GAP;

/// The width a `choice_list_row` needs to show `title` and `description`
/// whole, for a list that sizes itself to its longest row.
pub(super) fn choice_list_row_width(ui: &Ui, title: &str, description: &str) -> f32 {
    let width = |text: &str, size: f32| ui.painter().layout_no_wrap(text.to_owned(), egui::FontId::proportional(size), theme::TEXT_PRIMARY).size().x;
    let text = width(title, theme::FONT_SIZE_BODY).max(width(description, theme::FONT_SIZE_SECONDARY));
    LIST_ICON_INSET + 2.0 * LIST_ICON_RADIUS + LIST_TEXT_GAP + text + LIST_CHECK_ROOM
}

/// A choice-list row: an icon in a dim circle, the title over
/// a one-line description, and when picked a purple wash and a check.
/// Reads as a button named title, newline, description.
pub(super) fn choice_list_row(ui: &mut Ui, icon: &str, title: &str, description: &str, selected: bool, icon_fill: Color32) -> Response {
    const HEIGHT: f32 = 46.0;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), HEIGHT), egui::Sense::click());
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, ui.is_enabled(), selected, row_name(title, description)));
    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        painter.rect_filled(rect, theme::ROW_ROUNDING, row_wash(&resp, selected));
        let icon_center = egui::pos2(rect.left() + LIST_ICON_INSET + LIST_ICON_RADIUS, rect.center().y);
        painter.circle_filled(icon_center, LIST_ICON_RADIUS, icon_fill);
        painter.text(icon_center, egui::Align2::CENTER_CENTER, icon, egui::FontId::proportional(15.0), theme::GLOW);
        let text_x = icon_center.x + LIST_ICON_RADIUS + LIST_TEXT_GAP;
        let dim = if ui.is_enabled() { 1.0 } else { 0.5 };
        painter.text(egui::pos2(text_x, rect.center().y - 1.0), egui::Align2::LEFT_BOTTOM, title,
            egui::FontId::proportional(theme::FONT_SIZE_BODY), theme::TEXT_PRIMARY.gamma_multiply(dim));
        painter.text(egui::pos2(text_x, rect.center().y + 2.0), egui::Align2::LEFT_TOP, description,
            egui::FontId::proportional(theme::FONT_SIZE_SECONDARY), theme::TEXT_SECONDARY.gamma_multiply(dim));
        if selected {
            paint_row_check(painter, rect, LIST_CHECK_INSET);
        }
    }
    resp
}

/// A choice list's dim icon circle; a purple one marks an action.
pub(super) const LIST_ICON_FILL: Color32 = theme::FIELD_FILL;
pub(super) const LIST_ACTION_FILL: Color32 = Color32::from_rgba_premultiplied(39, 20, 52, 64);

/// A selectable option with a title and a one-line description, for
/// every picker popover. Returns the row's response.
pub(super) fn picker_row(ui: &mut Ui, selected: bool, title: &str, description: &str) -> Response {
    const PAD: egui::Vec2 = egui::vec2(12.0, 6.0);
    const CHECK_ROOM: f32 = 28.0;
    let dim = if ui.is_enabled() { 1.0 } else { 0.5 };
    let title_galley = ui.painter().layout_no_wrap(title.to_owned(),
        egui::FontId::proportional(theme::FONT_SIZE_BODY), theme::TEXT_PRIMARY.gamma_multiply(dim));
    let desc_galley = (!description.is_empty()).then(|| ui.painter().layout_no_wrap(description.to_owned(),
        egui::FontId::proportional(theme::FONT_SIZE_SECONDARY), theme::TEXT_SECONDARY.gamma_multiply(dim)));
    let text_h = title_galley.size().y + desc_galley.as_ref().map_or(0.0, |g| g.size().y + 2.0);
    let text_w = title_galley.size().x.max(desc_galley.as_ref().map_or(0.0, |g| g.size().x));
    let size = egui::vec2(
        ui.available_width().max(text_w + 2.0 * PAD.x + CHECK_ROOM),
        (text_h + 2.0 * PAD.y).max(theme::CHIP_HEIGHT),
    );
    let room = crate::gui::automation::record::room(ui);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    crate::gui::automation::record::clip(ui.ctx(), title, rect, room);
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, ui.is_enabled(), selected, row_name(title, description)));
    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        painter.rect_filled(rect, theme::ROW_ROUNDING, row_wash(&resp, selected));
        let top = rect.center().y - text_h / 2.0;
        let desc_top = top + title_galley.size().y + 2.0;
        painter.galley(egui::pos2(rect.left() + PAD.x, top), title_galley, theme::TEXT_PRIMARY);
        if let Some(g) = desc_galley {
            painter.galley(egui::pos2(rect.left() + PAD.x, desc_top), g, theme::TEXT_SECONDARY);
        }
        if selected {
            paint_row_check(painter, rect, PAD.x);
        }
    }
    resp
}

/// A bare modal's page: the margin, the header row, then `body`. True on
/// the frame its close was clicked.
pub(super) fn modal_page(ui: &mut Ui, title: &str, subtitle: &str, body: impl FnOnce(&mut Ui)) -> bool {
    modal_page_closable(ui, title, subtitle, true, body)
}

/// `modal_page` whose close is greyed while not `closable`.
pub(super) fn modal_page_closable(ui: &mut Ui, title: &str, subtitle: &str, closable: bool, body: impl FnOnce(&mut Ui)) -> bool {
    modal_frame(ui, |ui| {
        let close = modal_header(ui, title, subtitle, closable);
        ui.add_space(theme::SPACE_MD);
        body(ui);
        close
    })
}

/// A bare modal's page margin and full width, around `body`.
pub(super) fn modal_frame<R>(ui: &mut Ui, body: impl FnOnce(&mut Ui) -> R) -> R {
    egui::Frame::NONE.inner_margin(theme::MODAL_PAGE_MARGIN).show(ui, |ui| {
        ui.set_width(ui.available_width());
        body(ui)
    }).inner
}

/// A progress bar the width of the row, `height` tall, `fraction` full.
pub(super) fn progress_bar(ui: &mut Ui, fraction: f32, height: f32) {
    let (bar, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), height), egui::Sense::hover());
    ui.painter().rect_filled(bar, 2.0, theme::RAIL);
    ui.painter().rect_filled(egui::Rect::from_min_size(bar.min, egui::vec2(bar.width() * fraction.clamp(0.0, 1.0), height)), 2.0, theme::ACCENT_BRIGHT);
}

/// A modal's header row: its title, a line under it, and the round close
/// button at the right. True on the frame close was clicked.
pub(super) fn modal_header(ui: &mut Ui, title: &str, subtitle: &str, closable: bool) -> bool {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.label(RichText::new(title).size(theme::FONT_SIZE_MODAL_TITLE).color(theme::TEXT_PRIMARY));
            if !subtitle.is_empty() {
                ui.label(RichText::new(subtitle).size(theme::FONT_SIZE_SECONDARY).color(theme::TEXT_SECONDARY));
            }
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
            let x = ui.add_enabled_ui(closable, |ui| icon_square_button(ui, ICON_CLOSE.codepoint, theme::TEXT_SECONDARY, theme::FIELD_FILL)).inner;
            named(x, &format!("Close {title}")).clicked()
        })
        .inner
    })
    .inner
}

/// A menu item on one line: its name, and its shortcut right-aligned.
pub(super) fn menu_row(ui: &mut Ui, title: &str, shortcut: &str) -> Response {
    const PAD: f32 = 12.0;
    let title_galley = ui.painter().layout_no_wrap(title.to_owned(), egui::FontId::proportional(theme::FONT_SIZE_BODY), theme::TEXT_PRIMARY);
    let key_galley = ui.painter().layout_no_wrap(shortcut.to_owned(), egui::FontId::proportional(theme::FONT_SIZE_SECONDARY), theme::TEXT_SECONDARY);
    let width = ui.available_width().max(title_galley.size().x + key_galley.size().x + 4.0 * PAD);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, theme::CHIP_HEIGHT), egui::Sense::click());
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), title));
    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        painter.rect_filled(rect, theme::ROW_ROUNDING, row_wash(&resp, false));
        let y = rect.center().y;
        painter.galley(egui::pos2(rect.left() + PAD, y - title_galley.size().y / 2.0), title_galley, theme::TEXT_PRIMARY);
        painter.galley(egui::pos2(rect.right() - PAD - key_galley.size().x, y - key_galley.size().y / 2.0), key_galley, theme::TEXT_SECONDARY);
    }
    resp
}

/// The track of a segmented control: a dim pill, its segments 2 px apart.
fn segmented<R>(ui: &mut Ui, body: impl FnOnce(&mut Ui) -> R) -> R {
    egui::Frame::NONE.fill(theme::FIELD_FILL).corner_radius(theme::PILL_RADIUS).inner_margin(3).show(ui, |ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        body(ui)
    }).inner
}

/// Horizontal tab strip: the active tab is filled, the rest are flat.
pub(super) fn tab_strip(ui: &mut Ui, labels: &[&str], selected: &mut usize) {
    segmented(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().button_padding = egui::vec2(12.0, 2.0);
            for (i, label) in labels.iter().enumerate() {
                let active = *selected == i;
                let text = RichText::new(*label)
                    .size(theme::FONT_SIZE_BODY)
                    .color(if active { Color32::WHITE } else { theme::TEXT_SECONDARY });
                let btn = egui::Button::new(text)
                    .stroke(egui::Stroke::NONE)
                    .corner_radius(theme::BUTTON_ROUNDING)
                    .min_size(egui::vec2(0.0, theme::CHIP_HEIGHT - 6.0));
                let fill = if active { theme::ACCENT_BRIGHT } else { Color32::TRANSPARENT };
                if with_fill(ui, fill, |ui| ui.add(btn)).clicked() {
                    *selected = i;
                }
            }
        });
    });
}

/// A `tab_strip` whose pick is kept under `id` from frame to frame.
/// Returns the picked tab's index.
pub(super) fn remembered_tab_strip(ui: &mut Ui, id: egui::Id, labels: &[&str]) -> usize {
    let mut tab = ui.data(|d| d.get_temp(id).unwrap_or(0));
    tab_strip(ui, labels, &mut tab);
    ui.data_mut(|d| d.insert_temp(id, tab));
    tab
}

/// Visual weight of a text button. Process is the only `Primary` on the
/// toolbar; modals use `Primary` for their one confirming action,
/// `Secondary` for the rest and `Destructive` for anything that deletes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ButtonKind {
    Primary,
    Secondary,
    Destructive,
    /// Text only, for a quiet action beside a louder one.
    Ghost,
}

/// Themed text button at chip height. Row 1 builds its own taller
/// buttons; everything else goes through here.
pub(super) fn button(ui: &mut Ui, kind: ButtonKind, text: &str) -> Response {
    let (fill, color) = match kind {
        ButtonKind::Primary => (theme::ACCENT_BRIGHT, Color32::WHITE),
        ButtonKind::Secondary => (Color32::from_white_alpha(20), theme::TEXT_PRIMARY),
        ButtonKind::Destructive => (theme::DESTRUCTIVE, Color32::WHITE),
        ButtonKind::Ghost => (Color32::TRANSPARENT, theme::TEXT_SECONDARY),
    };
    with_fill(ui, fill, |ui| {
        ui.add(
            egui::Button::new(RichText::new(text).color(color).size(theme::FONT_SIZE_BODY))
                .stroke(egui::Stroke::NONE)
                .corner_radius(theme::BUTTON_ROUNDING)
                .min_size(egui::vec2(0.0, theme::CHIP_HEIGHT)),
        )
    })
}

/// One pipeline stage on the toolbar. The face reads "{label} · {summary}"
/// and carries the accent outline while any knob inside is off its
/// default; the popover opens with a title row and a Reset for the
/// whole group, then `body` lists the knobs in processing order.
#[derive(Clone, Copy)]
pub(super) struct GroupChip<'a> {
    pub id_salt: &'a str,
    pub icon: &'a str,
    pub label: &'a str,
    pub summary: &'a str,
    pub tooltip: &'a str,
    pub tuned: bool,
    pub width: f32,
    /// Its knobs change the image live, so it is a flyout the canvas
    /// stays live under; otherwise a popover that closes on click outside.
    pub live: bool,
}

impl GroupChip<'_> {
    /// The chip's face: its label, and the summary after a dot.
    fn face(&self) -> Cow<'_, str> {
        if self.summary.is_empty() {
            Cow::Borrowed(self.label)
        } else {
            Cow::Owned(format!("{} \u{00b7} {}", self.label, self.summary))
        }
    }
}

fn group_chip_id(id_salt: &str) -> egui::Id {
    egui::Id::new(("group_chip", id_salt))
}

/// Render a group chip. `body` receives `reset == true` on the frame the
/// group's Reset was clicked, before any of its rows render. Returns the
/// body's value while the popover is open.
pub(super) fn group_chip<R>(
    ui: &mut Ui,
    g: GroupChip<'_>,
    body: impl FnOnce(&mut Ui, bool) -> R,
) -> Option<R> {
    let resp = tooltip(chip_button(ui, g.icon, &g.face(), g.tuned), g.label, g.tooltip, None);
    let pop_id = group_chip_id(g.id_salt);
    let mut out = None;
    let contents = |ui: &mut Ui| {
        // Exactly its width: the header's right-aligned Reset would
        // otherwise stretch a flyout to the popup's default width.
        ui.set_width(g.width);
        ui.spacing_mut().slider_width = g.width - PANEL_VALUE_WIDTH;
        let reset = popover_header(ui, g.label, Some(("Reset every setting in this group", !g.tuned)));
        out = Some(body(ui, reset));
    };
    if g.live {
        flyout_for(pop_id, &resp, contents);
    } else {
        popup_for(ui, pop_id, &resp, contents);
    }
    out
}

/// Which part of a quick-strip chip `body` is drawing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Part {
    /// The strip under the chip: the knobs people reach for.
    Quick,
    /// The full panel the strip's dots open; `reset` is true on the frame
    /// its Reset was clicked, before any row renders.
    Full { reset: bool },
}

/// A group chip whose flyout is a quick strip that expands: a glass pill with the main knobs and dots, and under
/// it, while the dots are on, the full panel. One `body` draws both parts
/// so both may edit the same settings. The canvas stays live under it and
/// it fades while one of its sliders is held, like any flyout.
pub(super) fn strip_chip(ui: &mut Ui, g: GroupChip<'_>, more: &str, header: PanelHeader<'_>, body: impl FnMut(&mut Ui, Part)) {
    strip_chip_faced(ui, g, &g.face(), more, header, body);
}

/// A `strip_chip` whose chip shows `face` instead of its label: for a
/// chip whose name is too long to sit in the toolbar.
pub(super) fn strip_chip_faced(ui: &mut Ui, g: GroupChip<'_>, face: &str, more: &str, header: PanelHeader<'_>, mut body: impl FnMut(&mut Ui, Part)) {
    let resp = tooltip(chip_button(ui, g.icon, face, g.tuned), g.label, g.tooltip, None);
    let pop_id = group_chip_id(g.id_salt);
    let expanded_id = pop_id.with("expanded");
    // The strip's width in its normal layout, measured while in it: past
    // the window's room it lays out compact. Measured only while normal,
    // so the choice does not flip back once compact fits.
    let fit_id = pop_id.with("fit");
    show_flyout(pop_id, &resp, below_chip(&resp, pop_id), |ui| {
            let mut expanded = ui.data(|d| d.get_temp::<bool>(expanded_id).unwrap_or(false));
            let room = ui.ctx().content_rect().width() - 2.0 * STRIP_EDGE;
            let compact = ui.data(|d| d.get_temp::<f32>(fit_id)).is_some_and(|w| w > room);
            let strip = theme::pill_frame().show(ui, |ui| {
                ui.horizontal(|ui| {
                    // The row's centre line is set by its first height: the
                    // title and knobs centred on 18 px while the '...' grew
                    // it to 28.
                    ui.set_min_height(theme::CHIP_HEIGHT);
                    ui.spacing_mut().item_spacing.x = if compact { STRIP_SPACING_COMPACT } else { STRIP_SPACING };
                    ui.spacing_mut().slider_width = if compact { STRIP_SLIDER_WIDTH_COMPACT } else { STRIP_SLIDER_WIDTH };
                    ui.label(RichText::new(g.label).strong().color(theme::TEXT_PRIMARY));
                    body(ui, Part::Quick);
                    let fill = if expanded { theme::ACCENT_BRIGHT.gamma_multiply(0.35) } else { theme::FIELD_FILL };
                    let dots = tooltip(icon_square_button(ui, ICON_MORE_HORIZ.codepoint, theme::TEXT_PRIMARY, fill), more, "", None);
                    if dots.clicked() {
                        expanded = !expanded;
                    }
                    // The panel stays open across closing the strip, so the
                    // tree says whether it is, and a script opens it only once.
                    dots.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, true, expanded, more));
                });
            });
            if !compact {
                let width = strip.response.rect.width();
                ui.data_mut(|d| d.insert_temp(fit_id, width));
            }
            if expanded {
                ui.add_space(10.0);
                let frame = theme::panel_frame();
                // As wide as the strip above it, exactly: a right-aligned
                // row takes all it is offered, so a minimum would only grow.
                let width = (strip.response.rect.width() - frame.total_margin().sum().x).max(g.width);
                frame.show(ui, |ui| {
                    ui.set_width(width);
                    let reset = panel_header(ui, g.label, header, !g.tuned);
                    panel_rows(ui, |ui| body(ui, Part::Full { reset }));
                });
            }
            ui.data_mut(|d| d.insert_temp(expanded_id, expanded));
        });
}

/// What a full panel's title row says beside its title, and what its
/// reset is called: "Reset", or what the reset goes back to.
#[derive(Clone, Copy)]
pub(super) struct PanelHeader<'a> {
    pub note: &'a str,
    pub reset: &'a str,
}

impl PanelHeader<'_> {
    pub const RESET: PanelHeader<'static> = PanelHeader { note: "", reset: "Reset" };
}

/// A full panel's title row: the title, the note, and the reset at the
/// end, greyed while the group sits at its default. True on the frame
/// the reset was clicked.
fn panel_header(ui: &mut Ui, title: &str, header: PanelHeader<'_>, at_default: bool) -> bool {
    title_row(ui, title, |ui| {
        let clicked = reset_button(ui, header.reset, !at_default).clicked();
        if !header.note.is_empty() {
            ui.label(RichText::new(header.note).color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_SECONDARY));
        }
        clicked
    })
}

/// A title on the left, `end` laid out from the right edge, and the gap
/// under the row. Returns what `end` returns.
fn title_row(ui: &mut Ui, title: &str, end: impl FnOnce(&mut Ui) -> bool) -> bool {
    let clicked = ui
        .horizontal(|ui| {
            let room = crate::gui::automation::record::room(ui);
            let label = ui.label(RichText::new(title).strong().color(theme::TEXT_PRIMARY));
            crate::gui::automation::record::clip(ui.ctx(), title, label.rect, room);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), end).inner
        })
        .inner;
    ui.add_space(theme::SPACE_XS);
    clicked
}

/// A colour knob as a panel row: its name and a dot in the colour, which
/// opens the picker in place under the row. True when the colour changed.
pub(super) fn color_row(ui: &mut Ui, label: &str, rgb: &mut [u8; 3]) -> bool {
    let open_id = ui.id().with(("color_row", label));
    let mut open = ui.data(|d| d.get_temp::<bool>(open_id).unwrap_or(false));
    let [r, g, b] = *rgb;
    let dot = ui
        .horizontal(|ui| {
            if in_panel_rows(ui) {
                panel_name(ui, label);
            } else {
                ui.label(RichText::new(label).color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_SECONDARY));
            }
            color_dot(ui, label, Color32::from_rgb(r, g, b), open)
        })
        .inner;
    if dot.clicked() {
        open = !open;
    }
    let mut changed = false;
    if open {
        ui.horizontal(|ui| {
            if in_panel_rows(ui) {
                ui.add_space(PANEL_NAME_WIDTH + ui.spacing().item_spacing.x);
            }
            ui.vertical(|ui| changed = rgb_picker(ui, rgb));
        });
    }
    ui.data_mut(|d| d.insert_temp(open_id, open));
    changed
}

/// A picture tile: `paint` fills a rounded swatch of `size`, the name
/// sits under it when `caption`, and the picked one is ringed in accent.
/// Reads as a selectable item named `name`.
pub(super) fn swatch_tile(ui: &mut Ui, name: &str, selected: bool, size: egui::Vec2, caption: bool, paint: impl FnOnce(&egui::Painter, egui::Rect)) -> Response {
    const CAPTION_HEIGHT: f32 = 18.0;
    const TILE_RADIUS: f32 = 10.0;
    let tile = egui::vec2(size.x + 8.0, size.y + if caption { CAPTION_HEIGHT + 6.0 } else { 0.0 });
    let (rect, resp) = ui.allocate_exact_size(tile, egui::Sense::click());
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, ui.is_enabled(), selected, name));
    if ui.is_rect_visible(rect) {
        let swatch = egui::Rect::from_center_size(egui::pos2(rect.center().x, rect.top() + size.y / 2.0), size);
        paint(ui.painter(), swatch);
        let ring = if selected {
            Some(theme::ACCENT_RING)
        } else if resp.hovered() {
            Some(Color32::from_white_alpha(70))
        } else {
            None
        };
        if let Some(ring) = ring {
            ui.painter().rect_stroke(swatch.expand(1.0), TILE_RADIUS + 1.0, egui::Stroke::new(2.0, ring), egui::StrokeKind::Outside);
        }
        if caption {
            ui.painter().text(
                egui::pos2(rect.center().x, swatch.bottom() + 6.0),
                egui::Align2::CENTER_TOP,
                name,
                egui::FontId::proportional(11.5),
                if selected { theme::TEXT_PRIMARY } else { theme::TEXT_SECONDARY },
            );
        }
    }
    resp
}

const STRIP_SLIDER_WIDTH: f32 = 72.0;
/// A strip's sliders and gaps when it would not fit the window otherwise.
const STRIP_SLIDER_WIDTH_COMPACT: f32 = 52.0;
const STRIP_SPACING: f32 = 16.0;
const STRIP_SPACING_COMPACT: f32 = 8.0;
/// The room a strip keeps from each window edge.
const STRIP_EDGE: f32 = 16.0;

/// One numbered step of a full panel: the number in its badge (dim while
/// the step is off) and the step's rows beside it.
pub(super) fn step<R>(ui: &mut Ui, number: u8, active: bool, body: impl FnOnce(&mut Ui) -> R) -> R {
    ui.horizontal_top(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(24.0, 24.0), egui::Sense::hover());
        let (fill, text) = if active {
            (theme::ACCENT_BRIGHT.gamma_multiply(0.2), theme::GLOW)
        } else {
            (theme::FIELD_FILL, theme::TEXT_SECONDARY)
        };
        ui.painter().circle_filled(rect.center(), 12.0, fill);
        ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, STEP_NUMBERS[number as usize % STEP_NUMBERS.len()],
            egui::FontId::proportional(11.5), text);
        ui.add_space(theme::SPACE_SM);
        ui.vertical(|ui| body(ui)).inner
    })
    .inner
}

const STEP_NUMBERS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

/// The first row of every popover: the control's title on the left and,
/// when the popover has a default, Reset on the right. `reset` carries
/// the tooltip and whether the popover already sits at its default, in
/// which case Reset is greyed out. Returns true on the frame Reset was
/// clicked.
pub(super) fn popover_header(ui: &mut Ui, title: &str, reset: Option<(&str, bool)>) -> bool {
    title_row(ui, title, |ui| {
        reset.is_some_and(|(tip, at_default)| reset_button(ui, "Reset", !at_default).on_hover_text(tip).clicked())
    })
}

/// "n tuned", or nothing at the default, for a group chip face.
pub(super) fn tuned_summary(tuned: usize) -> Cow<'static, str> {
    match tuned {
        0 => Cow::Borrowed(""),
        n => Cow::Owned(format!("{n} tuned")),
    }
}

/// One knob of a group, described once: whether it is off its default,
/// how to put it back, and which catalog knob dispatches it (`None` for
/// knobs the recipe diff picks up on its own).
pub(super) struct Field<T> {
    pub knob: Option<StaticKnob>,
    pub differs: fn(&T, &T) -> bool,
    pub reset: fn(&mut T, &T),
}

pub(super) fn tuned_count<T>(fields: &[Field<T>], cur: &T, default: &T) -> usize {
    fields.iter().filter(|f| (f.differs)(cur, default)).count()
}

pub(super) fn reset_fields<T>(fields: &[Field<T>], cur: &mut T, default: &T) {
    for f in fields {
        (f.reset)(cur, default);
    }
}

/// Render `body` disabled, with `reason` on hover, when there is a reason.
pub(super) fn gated<R>(ui: &mut Ui, reason: Option<&str>, body: impl FnOnce(&mut Ui) -> R) -> R {
    let r = ui.add_enabled_ui(reason.is_none(), |ui| {
        if reason.is_some() {
            ui.multiply_opacity(GATED_FADE);
        }
        body(ui)
    });
    if let Some(reason) = reason {
        explain_on_click(ui, &r.response, reason);
        r.response.on_disabled_hover_text(reason);
    }
    r.inner
}

/// How much further a disabled control fades than egui's own disabled
/// look: a tuned chip's accent ring otherwise reads as clickable.
const GATED_FADE: f32 = 0.5;

/// How long a click on a disabled control shows why.
const WHY_SECS: f64 = 4.0;

/// A click on a disabled control shows why at once, for `WHY_SECS`: the
/// hover tooltip waits for the pointer to rest, which on a TV it rarely
/// does.
fn explain_on_click(ui: &Ui, resp: &Response, reason: &str) {
    let until_id = resp.id.with("why_until");
    let now = ui.input(|i| i.time);
    if resp.contains_pointer() && ui.input(|i| i.pointer.primary_clicked()) {
        ui.data_mut(|d| d.insert_temp(until_id, now + WHY_SECS));
    }
    let until = ui.data(|d| d.get_temp::<f64>(until_id)).unwrap_or(0.0);
    if now < until {
        resp.show_tooltip_text(reason);
        ui.ctx().request_repaint_after(std::time::Duration::from_secs_f64(until - now));
    }
}

/// A panel's section heading: a small capitals caption, and
/// an optional note beside it.
pub(super) fn panel_section(ui: &mut Ui, caption: &str, note: Option<&str>) {
    ui.add_space(theme::SPACE_SM);
    ui.horizontal(|ui| {
        ui.label(RichText::new(caption).size(theme::FONT_SIZE_CAPTION).color(theme::TEXT_SECONDARY));
        if let Some(note) = note {
            ui.label(RichText::new(note).size(theme::FONT_SIZE_CAPTION).color(theme::HINT));
        }
    });
}

/// Secondary caption above a row or a list inside a popover.
pub(super) fn section_label(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_SECONDARY));
}

/// One option of a `choice_row`.
pub(super) struct Choice<T> {
    pub value: T,
    pub name: &'static str,
    pub description: &'static str,
    pub enabled: bool,
}

/// Label above a wrapped row of selectable options, for enums with a
/// handful of values. Each option explains itself on hover. Returns true
/// when the value changed.
pub(super) fn choice_row<T: Copy + PartialEq>(
    ui: &mut Ui,
    label: &str,
    options: &[Choice<T>],
    value: &mut T,
) -> bool {
    if in_panel_rows(ui) {
        return ui.horizontal(|ui| {
            panel_name(ui, label);
            choice_buttons(ui, options, value)
        })
        .inner;
    }
    section_label(ui, label);
    ui.horizontal_wrapped(|ui| {
        // Break the row between options, never inside a label.
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        choice_buttons(ui, options, value)
    })
    .inner
}

/// The option buttons of a `choice_row`, inline, as a segmented
/// control: one dim pill holding the options, the picked one purple.
pub(super) fn choice_buttons<T: Copy + PartialEq>(ui: &mut Ui, options: &[Choice<T>], value: &mut T) -> bool {
    choice_buttons_picked(ui, options, value, theme::ACCENT_BRIGHT)
}

/// `choice_buttons` with the picked option in `picked` (Erase's warm red).
pub(super) fn choice_buttons_picked<T: Copy + PartialEq>(ui: &mut Ui, options: &[Choice<T>], value: &mut T, picked: Color32) -> bool {
    choice_pills(ui, options, None, value, picked)
}

/// An icon a choice button draws in its square `rect`, in `color`.
pub(super) type IconPaint = fn(&egui::Painter, egui::Rect, Color32);

/// `choice_buttons` drawing each option as its icon from `icons`: its
/// name is its tooltip's title and its name in the tree.
pub(super) fn icon_choice_buttons<T: Copy + PartialEq>(ui: &mut Ui, options: &[Choice<T>], icons: &[IconPaint], value: &mut T) -> bool {
    choice_pills(ui, options, Some(icons), value, theme::ACCENT_BRIGHT)
}

/// An icon that turns `on` on and off, drawn as a one-pill segmented
/// control like `icon_choice_buttons`. Returns true when it changed.
pub(super) fn icon_toggle(ui: &mut Ui, name: &str, description: &str, icon: IconPaint, on: &mut bool) -> bool {
    segmented(ui, |ui| {
        ui.horizontal(|ui| {
            pill_style(ui, theme::ACCENT_BRIGHT);
            let clicked = icon_pill(ui, *on, true, icon, name, description).clicked();
            if clicked {
                *on = !*on;
            }
            clicked
        })
        .inner
    })
}

/// The look of a segmented control's pills, the picked one in `picked`.
fn pill_style(ui: &mut Ui, picked: Color32) {
    ui.spacing_mut().button_padding = egui::vec2(11.0, 5.0);
    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
    let v = ui.visuals_mut();
    v.selection.stroke.color = Color32::WHITE;
    v.selection.bg_fill = picked;
    for w in [&mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active] {
        w.corner_radius = egui::CornerRadius::same(theme::PILL_RADIUS);
        w.bg_stroke = egui::Stroke::NONE;
    }
    v.widgets.inactive.fg_stroke.color = theme::TEXT_SECONDARY;
}

/// A pill drawing `icon`, named `name` in its tooltip and the tree.
fn icon_pill(ui: &mut Ui, selected: bool, enabled: bool, icon: IconPaint, name: &str, description: &str) -> Response {
    // The pill holds an em space as wide as the icon it draws.
    let text = RichText::new("\u{2003}").size(ICON_CHOICE_SIZE);
    let resp = ui.add_enabled_ui(enabled, |ui| ui.selectable_label(selected, text)).inner;
    let color = ui.style().interact_selectable(&resp, selected).text_color();
    icon(ui.painter(), egui::Rect::from_center_size(resp.rect.center(), egui::Vec2::splat(ICON_CHOICE_SIZE)), color);
    tooltip(resp, name, description, None)
}

fn choice_pills<T: Copy + PartialEq>(ui: &mut Ui, options: &[Choice<T>], icons: Option<&[IconPaint]>, value: &mut T, picked: Color32) -> bool {
    segmented(ui, |ui| {
            ui.horizontal(|ui| {
                pill_style(ui, picked);
                let mut changed = false;
                for (i, opt) in options.iter().enumerate() {
                    let selected = *value == opt.value;
                    let resp = match icons.and_then(|icons| icons.get(i).copied()) {
                        Some(icon) => icon_pill(ui, selected, opt.enabled, icon, opt.name, opt.description),
                        None => {
                            let text = RichText::new(opt.name).size(theme::FONT_SIZE_SECONDARY);
                            let resp = ui.add_enabled_ui(opt.enabled, |ui| ui.selectable_label(selected, text)).inner;
                            if opt.description.is_empty() { resp } else { resp.on_hover_text(opt.description) }
                        }
                    };
                    if resp.clicked() && !selected {
                        *value = opt.value;
                        changed = true;
                    }
                }
                changed
            })
            .inner
        })
}

/// An icon choice's glyph size: the shapes read at the size of the
/// strip's text beside them.
const ICON_CHOICE_SIZE: f32 = 16.0;

/// A dropdown (a rounded field with a chevron) for a choice too
/// long for a segmented row; in panel rows its name sits in the name
/// column. The list opens inline under the field: egui keeps one popup
/// open at a time, so a popup list would close the flyout holding it.
/// Returns true when the value changed.
pub(super) fn dropdown<T: Copy + PartialEq>(ui: &mut Ui, label: &str, options: &[Choice<T>], value: &mut T) -> bool {
    let current = options.iter().find(|o| o.value == *value).map_or("", |o| o.name);
    let open_id = ui.id().with(("dropdown", label));
    let mut open = ui.data(|d| d.get_temp::<bool>(open_id).unwrap_or(false));
    let panel = in_panel_rows(ui);
    let field = |ui: &mut Ui, open: bool| {
        let width = (ui.available_width() - if panel { PANEL_VALUE_WIDTH } else { 0.0 }).max(120.0);
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, PANEL_ROW_HEIGHT), egui::Sense::click());
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::ComboBox, ui.is_enabled(), label));
        let fill = if resp.hovered() || open { theme::FIELD_FILL_HOVER } else { theme::FIELD_FILL };
        let painter = ui.painter();
        painter.rect_filled(rect, theme::INPUT_RADIUS, fill);
        let text = if ui.is_enabled() { theme::TEXT_PRIMARY } else { theme::TEXT_SECONDARY };
        painter.text(rect.left_center() + egui::vec2(12.0, 0.0), egui::Align2::LEFT_CENTER, current, egui::FontId::proportional(theme::FONT_SIZE_BODY), text);
        let chevron = if open { ICON_EXPAND_LESS.codepoint } else { ICON_EXPAND_MORE.codepoint };
        painter.text(rect.right_center() - egui::vec2(10.0, 0.0), egui::Align2::RIGHT_CENTER, chevron, egui::FontId::proportional(16.0), theme::TEXT_SECONDARY);
        resp
    };
    let resp = if panel {
        ui.horizontal(|ui| {
            panel_name(ui, label);
            field(ui, open)
        })
        .inner
    } else {
        section_label(ui, label);
        field(ui, open)
    };
    if resp.clicked() {
        open = !open;
    }
    let mut changed = false;
    if open {
        ui.horizontal(|ui| {
            if panel {
                ui.add_space(PANEL_NAME_WIDTH + ui.spacing().item_spacing.x);
            }
            ui.vertical(|ui| {
                for opt in options {
                    let selected = *value == opt.value;
                    let resp = ui.add_enabled_ui(opt.enabled, |ui| picker_row(ui, selected, opt.name, opt.description)).inner;
                    if resp.clicked() {
                        open = false;
                        if !selected {
                            *value = opt.value;
                            changed = true;
                        }
                    }
                }
            });
        });
    }
    ui.data_mut(|d| d.insert_temp(open_id, open));
    changed
}

/// A named colour among the swatch dots.
pub(super) struct Swatch {
    pub name: &'static str,
    pub rgb: [u8; 3],
}

/// Colour dots in a row, the one matching `current` ringed
/// white. Returns the clicked dot's colour.
pub(super) fn swatch_dots(ui: &mut Ui, swatches: &[Swatch], current: Option<[u8; 3]>) -> Option<[u8; 3]> {
    let mut picked = None;
    for sw in swatches {
        let [r, g, b] = sw.rgb;
        if color_dot(ui, sw.name, Color32::from_rgb(r, g, b), current == Some(sw.rgb)).clicked() {
            picked = Some(sw.rgb);
        }
    }
    picked
}

/// One 24 px swatch dot, ringed white while `on`; reads as a radio
/// button named `name` to the tree.
pub(super) fn color_dot(ui: &mut Ui, name: &str, fill: Color32, on: bool) -> Response {
    painted_dot(ui, name, on, |painter, rect| {
        painter.circle_filled(rect.center(), rect.width() / 2.0, fill);
    })
}

/// A swatch dot whose inside `paint` draws (a checkerboard, a picture).
pub(super) fn painted_dot(ui: &mut Ui, name: &str, on: bool, paint: impl FnOnce(&egui::Painter, egui::Rect)) -> Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(24.0, 24.0), egui::Sense::click());
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::RadioButton, ui.is_enabled(), on, name));
    if ui.is_rect_visible(rect) {
        let ring = if on { Color32::WHITE } else if resp.hovered() { Color32::from_white_alpha(90) } else { Color32::from_white_alpha(38) };
        paint(ui.painter(), egui::Rect::from_center_size(rect.center(), egui::vec2(22.0, 22.0)));
        ui.painter().circle_stroke(rect.center(), 11.0, egui::Stroke::new(2.0, ring));
    }
    resp.on_hover_text(name)
}

/// The "custom colour" dot: a hue ring around the colour itself.
pub(super) fn custom_color_dot(ui: &mut Ui, rgb: [u8; 3], on: bool) -> Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(24.0, 24.0), egui::Sense::click());
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::RadioButton, ui.is_enabled(), on, "Custom color"));
    if ui.is_rect_visible(rect) {
        let c = rect.center();
        const SEGMENTS: usize = 24;
        static HUES: std::sync::OnceLock<[Color32; SEGMENTS]> = std::sync::OnceLock::new();
        let hues = HUES.get_or_init(|| std::array::from_fn(|i| Hsva::new(i as f32 / SEGMENTS as f32, 0.8, 1.0, 1.0).into()));
        for (i, &color) in hues.iter().enumerate() {
            let a0 = i as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
            let a1 = (i + 1) as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
            let p = |a: f32| c + 11.0 * egui::vec2(a.cos(), a.sin());
            ui.painter().line_segment([p(a0), p(a1)], egui::Stroke::new(2.5, color));
        }
        let [r, g, b] = rgb;
        ui.painter().circle_filled(c, 7.5, Color32::from_rgb(r, g, b));
        if on {
            ui.painter().circle_stroke(c, 12.5, egui::Stroke::new(1.0, Color32::WHITE));
        }
    }
    resp.on_hover_text("Custom color")
}

/// On/off row inside a popover: the label, and the switch at the
/// row's end.
pub(super) fn toggle_row(ui: &mut Ui, label: &str, value: &mut bool) -> ChipChange {
    let changed = ui.horizontal(|ui| {
        if in_panel_rows(ui) {
            panel_name(ui, label);
        } else {
            ui.label(RichText::new(label).color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| switch(ui, label, value)).inner
    }).inner.changed();
    ChipChange { changed, commit: changed }
}

/// A switch with its label beside it, for a quick strip.
pub(super) fn inline_switch(ui: &mut Ui, label: &str, value: &mut bool) -> ChipChange {
    ui.label(RichText::new(label).color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_SECONDARY));
    let changed = switch(ui, label, value).changed();
    ChipChange { changed, commit: changed }
}

/// A switch: a 34×20 pill, purple when on, with a white knob
/// that slides across. Reads as a checkbox named `name` to the tree.
pub(super) fn switch(ui: &mut Ui, name: &str, on: &mut bool) -> Response {
    let (rect, mut resp) = ui.allocate_exact_size(egui::vec2(34.0, 20.0), egui::Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Checkbox, ui.is_enabled(), *on, name));
    if ui.is_rect_visible(rect) {
        let t = ui.ctx().animate_bool_responsive(resp.id, *on);
        let track = theme::RAIL.lerp_to_gamma(theme::ACCENT_BRIGHT, t);
        let track = if resp.hovered() { track.lerp_to_gamma(Color32::WHITE, 0.06) } else { track };
        let painter = ui.painter();
        painter.rect_filled(rect, egui::CornerRadius::same(theme::PILL_RADIUS), if ui.is_enabled() { track } else { track.gamma_multiply(0.5) });
        let knob_x = egui::lerp(rect.left() + 10.0..=rect.right() - 10.0, t);
        painter.circle_filled(egui::pos2(knob_x, rect.center().y), 8.0, Color32::WHITE);
    }
    resp
}

/// The hue wheel unrolled, with the hues within `tolerance` degrees of
/// `hue` bright and the rest dimmed: which colors a color splash keeps.
pub(super) fn kept_hues(ui: &mut Ui, hue: u16, tolerance: u16) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.spacing().slider_width, 10.0), egui::Sense::hover());
    let kept = |u: f32| {
        let d = (u * 360.0 - hue as f32).rem_euclid(360.0);
        d.min(360.0 - d) <= tolerance as f32
    };
    paint_mesh(ui.painter(), rect, TextureId::default(), |u, _, _| {
        let color: Color32 = Hsva::new(u, 1.0, 1.0, 1.0).into();
        if kept(u) { color } else { color.gamma_multiply(0.2) }
    });
}

const RGB_PICKER_WIDTH: f32 = 240.0;

/// Inline RGB picker for a `[u8; 3]` inside a parent popover: a
/// saturation/brightness square, a hue strip and Red / Green / Blue
/// fields, each named. Inline because a picker in its own popup reads as
/// a click outside the parent popover and closes it. Returns `true` when
/// the user changed the color.
pub(super) fn rgb_picker(ui: &mut Ui, rgb: &mut [u8; 3]) -> bool {
    let width = RGB_PICKER_WIDTH;
    let (square, square_resp) = ui.allocate_exact_size(egui::vec2(width, width * 0.6), egui::Sense::drag());
    // Hue and saturation are kept across frames: a grey or black RGB holds
    // neither, so recomputing them from RGB would snap the hue to red.
    let memo = square_resp.id.with("hsv");
    let remembered = ui.data(|d| d.get_temp::<([u8; 3], Hsva)>(memo));
    let mut hsv = remembered
        .filter(|(seen, _)| seen == rgb)
        .map_or_else(|| Hsva::from_srgb(*rgb), |(_, hsv)| hsv);
    let before = hsv;
    if let Some(p) = square_resp.interact_pointer_pos() {
        hsv.s = ((p.x - square.left()) / square.width()).clamp(0.0, 1.0);
        hsv.v = 1.0 - ((p.y - square.top()) / square.height()).clamp(0.0, 1.0);
    }
    paint_mesh(ui.painter(), square, TextureId::default(), |u, v, _| Hsva::new(hsv.h, u, 1.0 - v, 1.0).into());
    let marker = egui::pos2(square.left() + hsv.s * square.width(), square.top() + (1.0 - hsv.v) * square.height());
    ui.painter().circle_stroke(marker, 4.0, egui::Stroke::new(1.5, Color32::WHITE));
    named(square_resp, "Saturation and brightness");

    let (strip, strip_resp) = ui.allocate_exact_size(egui::vec2(width, 12.0), egui::Sense::drag());
    if let Some(p) = strip_resp.interact_pointer_pos() {
        hsv.h = ((p.x - strip.left()) / strip.width()).clamp(0.0, 1.0);
    }
    paint_mesh(ui.painter(), strip, TextureId::default(), |u, _, _| Hsva::new(u, 1.0, 1.0, 1.0).into());
    let x = strip.left() + hsv.h * strip.width();
    ui.painter().vline(x, strip.y_range(), egui::Stroke::new(2.0, Color32::WHITE));
    named(strip_resp, "Hue");

    let mut channels = if hsv == before { *rgb } else { hsv.to_srgb() };
    let mut typed = false;
    ui.horizontal(|ui| {
        for (value, (name, prefix)) in channels.iter_mut().zip([("Red", "R "), ("Green", "G "), ("Blue", "B ")]) {
            let resp = ui.add(egui::DragValue::new(value).speed(0.5).prefix(prefix));
            typed |= named(resp, name).changed();
        }
    });
    if typed {
        hsv = Hsva::from_srgb(channels);
    }
    if remembered != Some((channels, hsv)) {
        ui.data_mut(|d| d.insert_temp(memo, (channels, hsv)));
    }
    let changed = channels != *rgb;
    *rgb = channels;
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tight row drops words from its right end, only as many chips as
    /// it must, and all of them when nothing else fits.
    #[test]
    fn a_tight_row_drops_words_from_the_right() {
        let chip = |key, words, icon| ChipWidth { key, words, icon };
        let row = [chip("model", 100.0, 40.0), chip("mask", 110.0, 30.0), chip("sketch", 120.0, 30.0), chip("effect", 110.0, 30.0)];
        assert_eq!(trimmed_count(&row, 8.0, 500.0), 0, "room for every word");
        assert_eq!(trimmed_count(&row, 8.0, 400.0), 1, "the last chip only");
        assert_eq!(trimmed_count(&row, 8.0, 300.0), 2);
        assert_eq!(trimmed_count(&row, 8.0, 100.0), 4, "nothing fits: all icons");
        let unlaid = [chip("model", 100.0, 0.0), chip("mask", 110.0, 0.0)];
        assert_eq!(trimmed_count(&unlaid, 8.0, 100.0 + ICON_CHIP_GUESS + 8.0), 1, "an icon not laid out yet counts as a guess");
    }

    /// With the app's pill padding the value box would stand taller than
    /// the slider and drag the painted handle off the rail.
    #[test]
    fn a_slider_with_its_value_box_is_no_taller_than_its_rail() {
        let ctx = egui::Context::default();
        let _ = ctx.run_ui(Default::default(), |ui| {
            ui.spacing_mut().button_padding = egui::vec2(12.0, 4.0);
            let thickness = ui.text_style_height(&egui::TextStyle::Body).max(ui.spacing().interact_size.y);
            let mut v = 0.5f32;
            let resp = ui.horizontal(|ui| add_slider(ui, egui::Slider::new(&mut v, 0.0..=1.0), 40.0)).inner;
            assert!(resp.rect.height() <= thickness + 0.5, "{} > {thickness}", resp.rect.height());
        });
    }

    /// A knob's value box keeps its width whatever it shows, so the
    /// controls after it on a strip hold still while a slider moves.
    #[test]
    fn a_value_box_holds_its_width_while_the_value_changes() {
        let ctx = egui::Context::default();
        let row_width = |draw: &mut dyn FnMut(&mut Ui)| {
            let mut w = 0.0;
            let _ = ctx.run_ui(Default::default(), |ui| w = ui.horizontal(|ui| draw(ui)).response.rect.width());
            w
        };
        let px = |v: f32| format!("{v:.0} px");
        let strip = |v: f32| row_width(&mut |ui| { bare_slider_f32(ui, "Feather", &mut v.clone(), 0.0..=20.0, px); });
        assert_eq!(strip(1.0), strip(20.0), "a strip slider");
        let degrees = |v: i32| row_width(&mut |ui| { slider_row(ui, "Degrees", &mut v.clone(), -180..=180); });
        assert_eq!(degrees(5), degrees(-180), "a whole-number slider");
        let signed = |v: f32| row_width(&mut |ui| { slider_row_f32(ui, "Shift edge", &mut v.clone(), -10.0..=10.0, false, |v| format!("{v:+.0} px")); });
        assert_eq!(signed(0.0), signed(-10.0), "a signed slider");
    }

    /// The painted handle sits where egui's mapping puts it: linear, and
    /// log10 for a logarithmic slider (Gamma 1.0 on 0.01-10 is two thirds).
    #[test]
    fn the_handle_follows_egui_s_mapping() {
        assert_eq!(super::slider_fraction(5.0, 0.0, 10.0, false), 0.5);
        assert!((super::slider_fraction(1.0, 0.01, 10.0, true) - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(super::slider_fraction(-3.0, 0.0, 10.0, false), 0.0);
        assert_eq!(super::slider_fraction(1.0, 1.0, 1.0, false), 0.5);
    }
}
