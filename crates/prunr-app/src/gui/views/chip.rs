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

use crate::gui::theme;

/// Height of a chip button on the toolbar row.
pub const CHIP_HEIGHT: f32 = 28.0;

/// Horizontal padding inside a chip.
const CHIP_PADDING_X: f32 = 8.0;

/// Popover width — wide enough for slider + value + reset.
const POPOVER_WIDTH: f32 = 260.0;

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

/// Shared chip-button renderer. Returns the response for popup wiring.
/// `accent` = true draws an accent border (non-default value indicator).
fn chip_button(ui: &mut Ui, icon: &str, value: &str, accent: bool) -> Response {
    let fill = theme::BG_SECONDARY;
    let stroke = if accent {
        egui::Stroke::new(1.0, theme::ACCENT)
    } else {
        egui::Stroke::new(1.0, Color32::TRANSPARENT)
    };
    let text = format!("{icon}  {value}");
    let btn = egui::Button::new(
        RichText::new(text).color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY),
    )
    .fill(fill)
    .stroke(stroke)
    .corner_radius(theme::BUTTON_ROUNDING)
    .min_size(egui::vec2(0.0, CHIP_HEIGHT));
    let saved_padding = ui.spacing().button_padding;
    ui.spacing_mut().button_padding = egui::vec2(CHIP_PADDING_X, 4.0);
    let resp = ui.add(btn);
    ui.spacing_mut().button_padding = saved_padding;
    resp
}

/// Render the standard reset-to-default button at the bottom of a popover.
fn reset_button(ui: &mut Ui, tooltip: &str) -> bool {
    ui.small_button(
        RichText::new(format!("{}  Reset", ICON_RESTART_ALT.codepoint))
            .color(theme::TEXT_SECONDARY)
            .size(theme::FONT_SIZE_MONO),
    )
    .on_hover_text(tooltip)
    .clicked()
}

use super::hint;

/// Wire a popup to a chip button. Handles toggle-on-click.
/// Uses the legacy `popup_below_widget` API; egui's newer `Popup::` builder
/// is a future cleanup. Deprecation is isolated to this helper.
#[allow(deprecated)]
fn popup_for(
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
            ui.set_min_width(POPOVER_WIDTH);
            body(ui);
        },
    );
}

/// Continuous f32 chip (gamma, edge_shift, line_strength, ...).
/// `format` converts the value to display text (e.g. "1.20", "2px erode").
/// `tooltip` shows on hover over the chip button and inside the popover.
pub fn chip_f32(
    ui: &mut Ui,
    id_salt: &str,
    icon: &str,
    label: &str,
    tooltip: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    default_value: f32,
    logarithmic: bool,
    format: impl Fn(f32) -> String,
) -> ChipChange {
    let pop_id = egui::Id::new(("chip_f32", id_salt));
    let accent = (*value - default_value).abs() > f32::EPSILON;
    let display = format(*value);
    let resp = chip_button(ui, icon, &display, accent).on_hover_text(tooltip);

    let mut out = ChipChange::default();
    popup_for(ui, pop_id, &resp, |ui| {
        ui.label(RichText::new(label).strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        let slider = ui.add(
            egui::Slider::new(value, range)
                .show_value(true)
                .fixed_decimals(2)
                .logarithmic(logarithmic),
        );
        if slider.changed() { out.changed = true; }
        if slider_settled(&slider) { out.commit = true; }
        hint(ui, tooltip);
        ui.add_space(theme::SPACE_XS);
        if reset_button(ui, "Reset to factory default") {
            *value = default_value;
            out.changed = true;
            out.commit = true;
        }
    });
    out
}

/// Integer chip. Used for pixel counts where fractional values don't make sense.
pub fn chip_u32(
    ui: &mut Ui,
    id_salt: &str,
    icon: &str,
    label: &str,
    tooltip: &str,
    value: &mut u32,
    range: std::ops::RangeInclusive<u32>,
    default_value: u32,
    format: impl Fn(u32) -> String,
) -> ChipChange {
    let pop_id = egui::Id::new(("chip_u32", id_salt));
    let accent = *value != default_value;
    let display = format(*value);
    let resp = chip_button(ui, icon, &display, accent).on_hover_text(tooltip);

    let mut out = ChipChange::default();
    popup_for(ui, pop_id, &resp, |ui| {
        ui.label(RichText::new(label).strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        let slider = ui.add(egui::Slider::new(value, range).show_value(true));
        if slider.changed() { out.changed = true; }
        if slider_settled(&slider) { out.commit = true; }
        hint(ui, tooltip);
        ui.add_space(theme::SPACE_XS);
        if reset_button(ui, "Reset to factory default") {
            *value = default_value;
            out.changed = true;
            out.commit = true;
        }
    });
    out
}

/// Optional f32 chip (threshold). Toggle enables/disables; slider sets the value.
pub fn chip_option_f32(
    ui: &mut Ui,
    id_salt: &str,
    icon: &str,
    label: &str,
    tooltip: &str,
    value: &mut Option<f32>,
    range: std::ops::RangeInclusive<f32>,
    default_when_enabled: f32,
    off_label: &str,
    format: impl Fn(f32) -> String,
) -> ChipChange {
    let pop_id = egui::Id::new(("chip_option_f32", id_salt));
    let accent = value.is_some();
    let display = match value {
        Some(v) => format(*v),
        None => off_label.to_string(),
    };
    let resp = chip_button(ui, icon, &display, accent).on_hover_text(tooltip);

    let mut out = ChipChange::default();
    popup_for(ui, pop_id, &resp, |ui| {
        ui.label(RichText::new(label).strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        let mut enabled = value.is_some();
        if ui.checkbox(&mut enabled, "Enabled").changed() {
            out.changed = true;
            out.commit = true; // toggle settles immediately
            *value = if enabled { Some(default_when_enabled) } else { None };
        }
        if let Some(v) = value.as_mut() {
            ui.add_space(theme::SPACE_XS);
            let slider = ui.add(
                egui::Slider::new(v, range)
                    .show_value(true)
                    .fixed_decimals(3),
            );
            if slider.changed() { out.changed = true; }
            if slider_settled(&slider) { out.commit = true; }
        }
        hint(ui, tooltip);
        ui.add_space(theme::SPACE_XS);
        if reset_button(ui, "Disable") {
            *value = None;
            out.changed = true;
            out.commit = true;
        }
    });
    out
}

/// Bool chip (refine_edges). Simple on/off toggle; popover shows the toggle + label.
pub fn chip_bool(
    ui: &mut Ui,
    id_salt: &str,
    icon: &str,
    label: &str,
    tooltip: &str,
    value: &mut bool,
) -> ChipChange {
    let pop_id = egui::Id::new(("chip_bool", id_salt));
    let display = if *value { "On" } else { "Off" };
    let resp = chip_button(ui, icon, display, *value).on_hover_text(tooltip);

    let mut out = ChipChange::default();
    popup_for(ui, pop_id, &resp, |ui| {
        ui.label(RichText::new(label).strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        if ui.checkbox(value, label).changed() {
            out.changed = true;
            out.commit = true; // toggle always commits
        }
        hint(ui, tooltip);
    });
    out
}

/// Optional RGBA chip (bg color). Toggle enables; inline color picker sets the value.
/// Displays "None" when disabled, a swatch preview when enabled.
///
/// Uses `color_picker_color32` (inline picker) instead of `color_edit_button`
/// (button that opens a nested popup). egui's nested popup close behavior
/// mis-handles the chip's `CloseOnClickOutside` when the color picker popup
/// opens outside the chip popover's rect — click on the color palette was
/// seen as "outside" and closed the chip popover. Inline picker avoids the
/// whole class of bug.
pub fn chip_option_rgba(
    ui: &mut Ui,
    id_salt: &str,
    icon: &str,
    label: &str,
    tooltip: &str,
    value: &mut Option<[u8; 4]>,
    default_when_enabled: [u8; 4],
) -> ChipChange {
    let pop_id = egui::Id::new(("chip_option_rgba", id_salt));
    let accent = value.is_some();
    let display = match value {
        Some(_) => "Set".to_string(),
        None => "None".to_string(),
    };
    let resp = chip_button(ui, icon, &display, accent).on_hover_text(tooltip);

    let mut out = ChipChange::default();
    popup_for(ui, pop_id, &resp, |ui| {
        ui.label(RichText::new(label).strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        let mut enabled = value.is_some();
        if ui.checkbox(&mut enabled, "Enabled").changed() {
            out.changed = true;
            out.commit = true;
            *value = if enabled { Some(default_when_enabled) } else { None };
        }
        if let Some(rgba) = value.as_mut() {
            ui.add_space(theme::SPACE_XS);
            let mut c = Color32::from_rgba_unmultiplied(rgba[0], rgba[1], rgba[2], rgba[3]);
            if color_picker_color32(ui, &mut c, Alpha::OnlyBlend) {
                let [r, g, b, a] = c.to_srgba_unmultiplied();
                *rgba = [r, g, b, a];
                out.changed = true;
                out.commit = true;
            }
        }
        hint(ui, tooltip);
        ui.add_space(theme::SPACE_XS);
        if reset_button(ui, "Clear background color") {
            *value = None;
            out.changed = true;
            out.commit = true;
        }
    });
    out
}

/// Optional RGB chip (solid_line_color). Same pattern as chip_option_rgba but 3-channel.
pub fn chip_option_rgb(
    ui: &mut Ui,
    id_salt: &str,
    icon: &str,
    label: &str,
    tooltip: &str,
    value: &mut Option<[u8; 3]>,
    default_when_enabled: [u8; 3],
) -> ChipChange {
    let pop_id = egui::Id::new(("chip_option_rgb", id_salt));
    let accent = value.is_some();
    let display = match value {
        Some(_) => "Set".to_string(),
        None => "Original".to_string(),
    };
    let resp = chip_button(ui, icon, &display, accent).on_hover_text(tooltip);

    let mut out = ChipChange::default();
    popup_for(ui, pop_id, &resp, |ui| {
        ui.label(RichText::new(label).strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        let mut enabled = value.is_some();
        if ui.checkbox(&mut enabled, "Override").changed() {
            out.changed = true;
            out.commit = true;
            *value = if enabled { Some(default_when_enabled) } else { None };
        }
        if let Some(rgb) = value.as_mut() {
            ui.add_space(theme::SPACE_XS);
            let mut c = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
            if color_picker_color32(ui, &mut c, Alpha::Opaque) {
                *rgb = [c.r(), c.g(), c.b()];
                out.changed = true;
                out.commit = true;
            }
        }
        hint(ui, tooltip);
        ui.add_space(theme::SPACE_XS);
        if reset_button(ui, "Use original line colors") {
            *value = None;
            out.changed = true;
            out.commit = true;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    // The chip widgets require an egui::Ui to render, so unit tests here
    // only check the non-render helpers. Visual integration tests belong
    // in the adjustments_toolbar smoke test suite.
    #[test]
    fn popover_width_is_sane() {
        assert!(super::POPOVER_WIDTH >= 200.0);
        assert!(super::POPOVER_WIDTH <= 400.0);
    }
}
