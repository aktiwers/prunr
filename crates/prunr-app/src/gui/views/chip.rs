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
use egui::widgets::color_picker::{color_picker_color32, Alpha};
use egui_material_icons::icons::*;

use std::borrow::Cow;

use super::shortcuts::{self, Action};
use crate::gui::knob_catalog::StaticKnob;
use crate::gui::theme;

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
fn slider_settled(resp: &egui::Response) -> bool {
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
        egui::Stroke::new(theme::STROKE_DEFAULT, theme::ACCENT)
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

fn icon_square_button(ui: &mut Ui, icon: &str, color: Color32, fill: Color32) -> Response {
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
    resp.on_hover_ui(|ui| {
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
                RichText::new(shortcuts::keys(action))
                    .color(theme::TEXT_SECONDARY)
                    .size(theme::FONT_SIZE_MONO),
            );
        }
    })
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
    egui::popup_below_widget(
        ui,
        id,
        resp,
        egui::PopupCloseBehavior::CloseOnClickOutside,
        |ui| {
            // Match `selectable_label` selection highlight to the app accent
            // — egui's default blue clashed in chip popovers (fill-style
            // list, channel-swap variants, etc.).
            ui.visuals_mut().selection.bg_fill = theme::ACCENT;
            ui.set_min_width(theme::POPOVER_WIDTH);
            body(ui);
        },
    );
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
    labelled_slider(
        ui,
        label,
        egui::Slider::new(value, range)
            .custom_formatter(move |v, _| format(v as f32))
            .logarithmic(logarithmic),
    )
}

/// Label above a full-width slider whose box shows the value — the one
/// slider layout for popovers and Settings.
pub fn slider_row<T: egui::emath::Numeric>(
    ui: &mut Ui,
    label: &str,
    value: &mut T,
    range: std::ops::RangeInclusive<T>,
) -> ChipChange {
    labelled_slider(ui, label, egui::Slider::new(value, range))
}

fn labelled_slider(ui: &mut Ui, label: &str, slider: egui::Slider<'_>) -> ChipChange {
    ui.label(RichText::new(label).color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));
    let resp = ui.add(slider.show_value(true));
    ChipChange { changed: resp.changed(), commit: slider_settled(&resp) }
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
    popup_for(ui, pop_id, &resp, |ui| {
        ui.set_min_width(g.width);
        let reset = popover_header(ui, g.label, Some(("Reset every knob in this group", !g.tuned)));
        out = Some(body(ui, reset));
    });
    out
}

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
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
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
    });
    changed
}

/// On/off row inside a popover.
pub(super) fn toggle_row(ui: &mut Ui, label: &str, value: &mut bool) -> ChipChange {
    let changed = ui.checkbox(value, label).changed();
    ChipChange { changed, commit: changed }
}

/// Inline RGB picker primitive used wherever a `[u8; 3]` needs to be
/// edited inside a parent popover. Uses egui's `color_picker_color32`
/// (the inline hue ring + sliders) rather than `color_edit_button_srgb`
/// (which opens its own popup — the parent popover's `CloseOnClickOutside`
/// treats that as "outside" and dismisses on first click).
/// Returns `true` when the user changed the color.
pub(super) fn rgb_picker(ui: &mut Ui, rgb: &mut [u8; 3]) -> bool {
    let mut c = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
    if color_picker_color32(ui, &mut c, Alpha::Opaque) {
        *rgb = [c.r(), c.g(), c.b()];
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
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
