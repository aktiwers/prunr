//! Compact "chip" widgets for the adjustments toolbar.
//!
//! A chip renders as a small pill-shaped button: `[icon  value]`. Clicking
//! it opens a popover containing the full slider / color picker + a reset-
//! to-default button. Non-default values get an accent outline so the user
//! can see at a glance which knobs have been tuned.
//!
//! View Component discipline: chips take `&mut` to their underlying value
//! only (e.g. `&mut f32`) — never `&mut PrunrApp`. This keeps them reusable
//! and testable, and prevents PrunrApp from growing into a God Object.
//!
//! All chip functions return `bool` indicating whether the value changed,
//! so callers can invalidate textures or kick off live preview.

use egui::{Color32, RichText, Response, Ui};
use egui::ecolor::Hsva;
use egui::TextureId;
use egui_material_icons::icons::*;

use std::borrow::Cow;

use super::shortcuts::{self, Action};
use crate::gui::knob_catalog::StaticKnob;
use crate::gui::theme;
use crate::gui::views::effects::paint_mesh;

/// Horizontal padding inside a chip.
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

/// Shared chip-button renderer. Returns the response for popup wiring.
/// `accent` = true draws an accent border (non-default value indicator).
pub(super) fn chip_button(ui: &mut Ui, icon: &str, value: &str, accent: bool) -> Response {
    let stroke = if accent {
        egui::Stroke::new(theme::STROKE_DEFAULT, theme::ACCENT_BRIGHT)
    } else {
        egui::Stroke::new(theme::STROKE_DEFAULT, Color32::TRANSPARENT)
    };
    let text = format!("{icon}  {value}");
    let btn = egui::Button::new(
        RichText::new(text).color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY),
    )
    .stroke(stroke)
    .corner_radius(theme::BUTTON_ROUNDING)
    .min_size(egui::vec2(0.0, theme::CHIP_HEIGHT));
    with_fill(ui, theme::BG_SECONDARY, |ui| {
        let saved_padding = ui.spacing().button_padding;
        ui.spacing_mut().button_padding = egui::vec2(CHIP_PADDING_X, 4.0);
        let resp = ui.add(btn);
        ui.spacing_mut().button_padding = saved_padding;
        resp
    })
}

/// Solid inner + concentric soft-falloff strokes following the same
/// smoothstep curve as `paint_circle`. Shared by the canvas trail and
/// the popover preview so the on-screen brush always matches what
/// `paint_circle` will write into the mask.
pub(super) fn paint_falloff_circle(
    painter: &egui::Painter,
    center: egui::Pos2,
    outer: f32,
    hardness: f32,
    color: egui::Color32,
    base_alpha: u8,
    steps: u32,
) {
    use prunr_core::math::smoothstep;
    let inner = outer * hardness.clamp(0.0, 1.0);
    if inner >= 0.5 {
        let solid = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), base_alpha);
        painter.circle_filled(center, inner, solid);
    }
    let span = (outer - inner).max(0.001);
    for i in 0..steps {
        let t = (i as f32 + 0.5) / steps as f32;
        let dist = inner + span * t;
        let intensity = if span < 0.5 { 0.0 } else { smoothstep(1.0 - t) };
        let a = (base_alpha as f32 * intensity) as u8;
        if a == 0 { continue; }
        let stroke_color = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), a);
        let stroke = egui::Stroke::new(span / steps as f32 * 1.4, stroke_color);
        painter.circle_stroke(center, dist, stroke);
    }
}

/// Square variant of `paint_falloff_circle` — axis-aligned rect with
/// the same smoothstep falloff toward the corners (chebyshev).
pub(super) fn paint_falloff_square(
    painter: &egui::Painter,
    center: egui::Pos2,
    outer: f32,
    hardness: f32,
    color: egui::Color32,
    base_alpha: u8,
    steps: u32,
) {
    use prunr_core::math::smoothstep;
    let inner = outer * hardness.clamp(0.0, 1.0);
    if inner >= 0.5 {
        let solid = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), base_alpha);
        painter.rect_filled(
            egui::Rect::from_center_size(center, egui::vec2(inner * 2.0, inner * 2.0)),
            0.0,
            solid,
        );
    }
    let span = (outer - inner).max(0.001);
    for i in 0..steps {
        let t = (i as f32 + 0.5) / steps as f32;
        let dist = inner + span * t;
        let intensity = if span < 0.5 { 0.0 } else { smoothstep(1.0 - t) };
        let a = (base_alpha as f32 * intensity) as u8;
        if a == 0 { continue; }
        let stroke_color = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), a);
        let stroke = egui::Stroke::new(span / steps as f32 * 1.4, stroke_color);
        painter.rect_stroke(
            egui::Rect::from_center_size(center, egui::vec2(dist * 2.0, dist * 2.0)),
            0.0,
            stroke,
            egui::StrokeKind::Outside,
        );
    }
}

/// Square icon-only button with an active/inactive visual state. Used by
/// the toolbar's Reset / Brush toggles and any future single-icon pill.
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
    named(resp, title).on_hover_ui(|ui| {
        ui.label(RichText::new(title).strong().color(theme::TEXT_PRIMARY));
        if !body.is_empty() {
            ui.add_space(theme::SPACE_XS);
            ui.label(
                RichText::new(body)
                    .color(theme::TEXT_PRIMARY)
                    .size(theme::FONT_SIZE_MONO),
            );
        }
        if let Some(action) = shortcut {
            ui.add_space(theme::SPACE_XS);
            ui.label(
                RichText::new(&*shortcuts::keys(ui.ctx(), action))
                    .color(theme::TEXT_SECONDARY)
                    .size(theme::FONT_SIZE_MONO),
            );
        }
    })
}

/// Names a control for the accessibility tree, which is what the test
/// harness and the control socket find widgets by. Icon-only buttons
/// have no readable name of their own; the tooltip title is it. Free
/// when the tree is off.
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

/// Render the standard reset-to-default button at the bottom of a popover.
pub(super) fn reset_button(ui: &mut Ui, tooltip: &str) -> bool {
    ui.small_button(
        RichText::new(format!("{}  Reset", ICON_RESTART_ALT.codepoint))
            .color(theme::TEXT_SECONDARY)
            .size(theme::FONT_SIZE_MONO),
    )
    .on_hover_text(tooltip)
    .clicked()
}

/// Wire a popup to a chip button. Handles toggle-on-click.
/// Uses the legacy `popup_below_widget` API; egui's newer `Popup::` builder
/// is a future cleanup. Deprecation is isolated to this helper.
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

/// Opacity of a flyout while one of its sliders is held.
const GHOST_OPACITY: f32 = 0.35;

/// A popover for knobs that change the image live. It stays open while
/// the user works on the canvas: its chip, Escape or another chip closes
/// it, never a click elsewhere. While one of its sliders is held it fades
/// so the image under it stays readable.
pub(super) fn flyout_for(id: egui::Id, resp: &Response, body: impl FnOnce(&mut Ui)) {
    outline_when_open(resp, id);
    egui::Popup::from_toggle_button_response(resp)
        .id(id)
        .close_behavior(egui::PopupCloseBehavior::IgnoreClicks)
        .align(egui::RectAlign::BOTTOM_START)
        // A chip near the right edge (the strip's dots) opens leftwards.
        .align_alternatives(&[egui::RectAlign::BOTTOM_END, egui::RectAlign::TOP_START, egui::RectAlign::TOP_END])
        .frame(egui::Frame::NONE)
        .show(|ui| {
            theme::register_flyout(ui.ctx(), id);
            if scrubbing_in(ui) {
                ui.set_opacity(GHOST_OPACITY);
            }
            // The frame is drawn inside so it fades with the contents.
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_min_width(theme::POPOVER_WIDTH);
                body(ui);
            });
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

/// Float slider row with the log scale and formatter float knobs need.
pub fn slider_row_f32(
    ui: &mut Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    logarithmic: bool,
    format: impl Fn(f32) -> String,
) -> ChipChange {
    let (lo, hi) = (*range.start() as f64, *range.end() as f64);
    let (change, resp) = labelled_slider(
        ui,
        label,
        egui::Slider::new(value, range)
            .custom_formatter(move |v, _| format(v as f32))
            .logarithmic(logarithmic),
    );
    paint_handle(ui, &resp, slider_fraction(*value as f64, lo, hi, logarithmic));
    change
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
    let (change, resp) = labelled_slider(ui, label, egui::Slider::new(value, range));
    paint_handle(ui, &resp, slider_fraction(value.to_f64(), lo, hi, false));
    change
}

fn labelled_slider(ui: &mut Ui, label: &str, slider: egui::Slider<'_>) -> (ChipChange, Response) {
    if ui.data(|d| d.get_temp::<bool>(egui::Id::new(PANEL_ROWS)).unwrap_or(false)) {
        return panel_slider(ui, label, slider);
    }
    let label = ui.label(RichText::new(label).color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));
    let resp = ui.add(slider.show_value(true)).labelled_by(label.id);
    (ChipChange { changed: resp.changed(), commit: slider_settled(&resp) }, resp)
}

/// The key under which `panel_rows` marks a panel whose sliders lay out
/// as the board's rows.
const PANEL_ROWS: &str = "chip_panel_rows";
/// The name column of a panel row (the board's 116 px).
const PANEL_NAME_WIDTH: f32 = 116.0;
/// Room for a panel row's value box.
const PANEL_VALUE_WIDTH: f32 = 74.0;

/// Run `body` with its sliders laid out as the board's panel rows: the
/// name in a fixed column, the slider filling the row, the value at its end.
pub(super) fn panel_rows<R>(ui: &mut Ui, body: impl FnOnce(&mut Ui) -> R) -> R {
    let key = egui::Id::new(PANEL_ROWS);
    let outer = ui.data(|d| d.get_temp::<bool>(key).unwrap_or(false));
    let width = ui.spacing().slider_width;
    ui.data_mut(|d| d.insert_temp(key, true));
    let out = body(ui);
    ui.data_mut(|d| d.insert_temp(key, outer));
    ui.spacing_mut().slider_width = width;
    out
}

fn panel_slider(ui: &mut Ui, label: &str, slider: egui::Slider<'_>) -> (ChipChange, Response) {
    // On the caller's ui too: the handle is painted from its slider width.
    ui.spacing_mut().slider_width = (ui.available_width() - PANEL_NAME_WIDTH - PANEL_VALUE_WIDTH).max(60.0);
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(PANEL_NAME_WIDTH, ui.spacing().interact_size.y), egui::Sense::hover());
        let name = ui.painter().layout_no_wrap(label.to_owned(), egui::FontId::proportional(13.0), theme::TEXT_PRIMARY);
        ui.painter().galley(egui::pos2(rect.left(), rect.center().y - name.size().y / 2.0), name, theme::TEXT_PRIMARY);
        let resp = chip_named_slider(ui, label, slider);
        (ChipChange { changed: resp.changed(), commit: slider_settled(&resp) }, resp)
    })
    .inner
}

/// A slider with its value box, named `label` for the tree.
fn chip_named_slider(ui: &mut Ui, label: &str, slider: egui::Slider<'_>) -> Response {
    named(ui.add(slider.show_value(true)), label)
}

/// The handle's colour on the board: near-white over the rail.
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

/// The board's white handle, painted over egui's (which shares the
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

/// A selectable option with a title and a one-line description, for
/// every picker popover. Returns the row's response.
pub(super) fn picker_row(ui: &mut Ui, selected: bool, title: &str, description: &str) -> Response {
    use egui::text::{LayoutJob, TextFormat};
    let mut job = LayoutJob::default();
    job.append(
        title,
        0.0,
        TextFormat {
            color: theme::TEXT_PRIMARY,
            font_id: egui::FontId::proportional(theme::FONT_SIZE_BODY),
            ..Default::default()
        },
    );
    if !description.is_empty() {
        job.append("\n", 0.0, TextFormat::default());
        job.append(
            description,
            0.0,
            TextFormat {
                color: theme::TEXT_SECONDARY,
                font_id: egui::FontId::proportional(theme::FONT_SIZE_MONO),
                ..Default::default()
            },
        );
    }
    ui.selectable_label(selected, job)
}

/// Horizontal tab strip: the active tab is filled, the rest are flat.
/// Used by Settings, the CLI reference and the Model Store filters.
pub(super) fn tab_strip(ui: &mut Ui, labels: &[&str], selected: &mut usize) {
    ui.horizontal(|ui| {
        for (i, label) in labels.iter().enumerate() {
            let active = *selected == i;
            let text = RichText::new(*label)
                .size(theme::FONT_SIZE_BODY)
                .color(if active { theme::TEXT_PRIMARY } else { theme::TEXT_SECONDARY });
            let btn = egui::Button::new(text)
                .corner_radius(theme::BUTTON_ROUNDING)
                .min_size(egui::vec2(0.0, theme::CHIP_HEIGHT));
            let fill = if active { theme::BG_SECONDARY } else { Color32::TRANSPARENT };
            if with_fill(ui, fill, |ui| ui.add(btn)).clicked() {
                *selected = i;
            }
        }
    });
}

/// Visual weight of a text button. Process is the only `Primary` on the
/// toolbar; modals use `Primary` for their one confirming action,
/// `Secondary` for the rest and `Destructive` for anything that deletes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ButtonKind {
    Primary,
    Secondary,
    Destructive,
}

/// Themed text button at chip height. Row 1 builds its own taller
/// buttons; everything else goes through here.
pub(super) fn button(ui: &mut Ui, kind: ButtonKind, text: &str) -> Response {
    let (fill, color) = match kind {
        ButtonKind::Primary => (theme::ACCENT, Color32::WHITE),
        ButtonKind::Secondary => (theme::BG_SECONDARY, theme::TEXT_PRIMARY),
        ButtonKind::Destructive => (theme::DESTRUCTIVE, Color32::WHITE),
    };
    with_fill(ui, fill, |ui| {
        ui.add(
            egui::Button::new(RichText::new(text).color(color).size(theme::FONT_SIZE_BODY))
                .corner_radius(theme::BUTTON_ROUNDING)
                .min_size(egui::vec2(0.0, theme::CHIP_HEIGHT)),
        )
    })
}

/// One pipeline stage on the toolbar. The face reads "{label} · {summary}"
/// and carries the accent outline while any knob inside is off its
/// default; the popover opens with a title row and a Reset for the
/// whole group, then `body` lists the knobs in processing order.
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

/// Render a group chip. `body` receives `reset == true` on the frame the
/// group's Reset was clicked, before any of its rows render. Returns the
/// body's value while the popover is open.
pub(super) fn group_chip<R>(
    ui: &mut Ui,
    g: GroupChip<'_>,
    body: impl FnOnce(&mut Ui, bool) -> R,
) -> Option<R> {
    let face: Cow<'_, str> = if g.summary.is_empty() {
        Cow::Borrowed(g.label)
    } else {
        Cow::Owned(format!("{} \u{00b7} {}", g.label, g.summary))
    };
    let resp = tooltip(chip_button(ui, g.icon, &face, g.tuned), g.label, g.tooltip, None);
    let pop_id = egui::Id::new(("group_chip", g.id_salt));
    let mut out = None;
    let contents = |ui: &mut Ui| {
        ui.set_min_width(g.width);
        let reset = popover_header(ui, g.label, Some(("Reset every knob in this group", !g.tuned)));
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

/// A group chip whose flyout is a quick strip that expands (surfaces
/// mockup, board 3): a glass pill with the main knobs and dots, and under
/// it, while the dots are on, the full panel. One `body` draws both parts
/// so both may edit the same settings. The canvas stays live under it and
/// it fades while one of its sliders is held, like any flyout.
pub(super) fn strip_chip(ui: &mut Ui, g: GroupChip<'_>, more: &str, mut body: impl FnMut(&mut Ui, Part)) {
    let face: Cow<'_, str> = if g.summary.is_empty() {
        Cow::Borrowed(g.label)
    } else {
        Cow::Owned(format!("{} \u{00b7} {}", g.label, g.summary))
    };
    let resp = tooltip(chip_button(ui, g.icon, &face, g.tuned), g.label, g.tooltip, None);
    let pop_id = egui::Id::new(("group_chip", g.id_salt));
    let expanded_id = pop_id.with("expanded");
    outline_when_open(&resp, pop_id);
    egui::Popup::from_toggle_button_response(&resp)
        .id(pop_id)
        .close_behavior(egui::PopupCloseBehavior::IgnoreClicks)
        .align(egui::RectAlign::BOTTOM_START)
        .align_alternatives(&[egui::RectAlign::BOTTOM_END, egui::RectAlign::TOP_START, egui::RectAlign::TOP_END])
        .frame(egui::Frame::NONE)
        .show(|ui| {
            theme::register_flyout(ui.ctx(), pop_id);
            if scrubbing_in(ui) {
                ui.set_opacity(GHOST_OPACITY);
            }
            let mut expanded = ui.data(|d| d.get_temp::<bool>(expanded_id).unwrap_or(false));
            theme::pill_frame().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 16.0;
                    ui.spacing_mut().slider_width = STRIP_SLIDER_WIDTH;
                    ui.label(RichText::new(g.label).strong().color(theme::TEXT_PRIMARY));
                    body(ui, Part::Quick);
                    let fill = if expanded { theme::ACCENT_BRIGHT.gamma_multiply(0.35) } else { Color32::from_white_alpha(15) };
                    let dots = tooltip(icon_square_button(ui, ICON_MORE_HORIZ.codepoint, theme::TEXT_PRIMARY, fill), more, "", None);
                    if dots.clicked() {
                        expanded = !expanded;
                    }
                });
            });
            if expanded {
                ui.add_space(10.0);
                theme::panel_frame().show(ui, |ui| {
                    ui.set_min_width(g.width);
                    let reset = popover_header(ui, g.label, Some(("Reset every knob in this group", !g.tuned)));
                    panel_rows(ui, |ui| body(ui, Part::Full { reset }));
                });
            }
            ui.data_mut(|d| d.insert_temp(expanded_id, expanded));
        });
}

/// A quick strip's slider width (the board's 72 px).
const STRIP_SLIDER_WIDTH: f32 = 72.0;

/// One numbered step of a full panel: the number in its badge (dim while
/// the step is off) and the step's rows beside it.
pub(super) fn step<R>(ui: &mut Ui, number: u8, active: bool, body: impl FnOnce(&mut Ui) -> R) -> R {
    ui.horizontal_top(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(24.0, 24.0), egui::Sense::hover());
        let (fill, text) = if active {
            (theme::ACCENT_BRIGHT.gamma_multiply(0.2), theme::GLOW)
        } else {
            (Color32::from_white_alpha(15), theme::TEXT_SECONDARY)
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
    let clicked = ui
        .horizontal(|ui| {
            ui.label(RichText::new(title).strong().color(theme::TEXT_PRIMARY));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                reset.is_some_and(|(tip, at_default)| {
                    ui.add_enabled_ui(!at_default, |ui| reset_button(ui, tip)).inner
                })
            })
            .inner
        })
        .inner;
    ui.add_space(theme::SPACE_XS);
    clicked
}

/// "default" or "n tuned", for a group chip face.
pub(super) fn tuned_summary(tuned: usize) -> Cow<'static, str> {
    match tuned {
        0 => Cow::Borrowed("default"),
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
    let r = ui.add_enabled_ui(reason.is_none(), body);
    if let Some(reason) = reason {
        r.response.on_disabled_hover_text(reason);
    }
    r.inner
}

/// Secondary caption above a row or a list inside a popover.
pub(super) fn section_label(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));
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
    section_label(ui, label);
    ui.horizontal_wrapped(|ui| {
        // Break the row between options, never inside a label.
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        choice_buttons(ui, options, value)
    })
    .inner
}

/// The option buttons of a `choice_row`, inline: for a strip or a row
/// that carries its own label.
pub(super) fn choice_buttons<T: Copy + PartialEq>(ui: &mut Ui, options: &[Choice<T>], value: &mut T) -> bool {
    let mut changed = false;
    for opt in options {
        let selected = *value == opt.value;
        let resp = ui
            .add_enabled_ui(opt.enabled, |ui| ui.selectable_label(selected, opt.name))
            .inner;
        let resp = if opt.description.is_empty() { resp } else { resp.on_hover_text(opt.description) };
        if resp.clicked() && !selected {
            *value = opt.value;
            changed = true;
        }
    }
    changed
}

/// On/off row inside a popover: the label, and the board's switch at the
/// row's end.
pub(super) fn toggle_row(ui: &mut Ui, label: &str, value: &mut bool) -> ChipChange {
    let changed = ui.horizontal(|ui| {
        ui.label(RichText::new(label).color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| switch(ui, label, value)).inner
    }).inner.changed();
    ChipChange { changed, commit: changed }
}

/// A switch with its label beside it, for a quick strip.
pub(super) fn inline_switch(ui: &mut Ui, label: &str, value: &mut bool) -> ChipChange {
    ui.label(RichText::new(label).color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));
    let changed = switch(ui, label, value).changed();
    ChipChange { changed, commit: changed }
}

/// The board's switch: a 34×20 pill, purple when on, with a white knob
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

/// Inline RGB picker for a `[u8; 3]` inside a parent popover: a
/// saturation/brightness square, a hue strip and Red / Green / Blue
/// fields, each named. Inline because a picker in its own popup reads as
/// a click outside the parent popover and closes it. Returns `true` when
/// the user changed the color.
pub(super) fn rgb_picker(ui: &mut Ui, rgb: &mut [u8; 3]) -> bool {
    let width = ui.spacing().slider_width;
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

    /// The painted handle sits where egui's mapping puts it: linear, and
    /// log10 for a logarithmic slider (Gamma 1.0 on 0.01-10 is two thirds).
    #[test]
    fn the_handle_follows_egui_s_mapping() {
        assert_eq!(super::slider_fraction(5.0, 0.0, 10.0, false), 0.5);
        assert!((super::slider_fraction(1.0, 0.01, 10.0, true) - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(super::slider_fraction(-3.0, 0.0, 10.0, false), 0.0);
        assert_eq!(super::slider_fraction(1.0, 1.0, 1.0, false), 0.5);
    }

    // The chip widgets require an egui::Ui to render, so unit tests here
    // only check the non-render helpers. Visual integration tests belong
    // in the adjustments_toolbar smoke test suite.
    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn popover_width_is_sane() {
        assert!(crate::gui::theme::POPOVER_WIDTH >= 200.0);
        assert!(crate::gui::theme::POPOVER_WIDTH <= 400.0);
    }
}
