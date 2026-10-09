//! The Refine group for the upscale row: six knobs, before- and
//! after-upscale, in processing order.
//!
//! Mutations land on `item_settings` directly; the recipe diff at the
//! next dispatch decides whether to reprocess or update live.

use egui::Ui;
use egui_material_icons::icons::ICON_TUNE;

use super::chip::{self, Field, GroupChip};
use super::{differs, fmt, hint};
use crate::gui::item_settings::ItemSettings;
use crate::gui::theme;

/// The six refinement knobs, in processing order. The recipe diff picks
/// up every change, so none needs a catalog knob.
const REFINE_FIELDS: &[Field<ItemSettings>] = &[
    Field { knob: None, differs: |s, d| differs(s.pre_denoise, d.pre_denoise), reset: |s, d| s.pre_denoise = d.pre_denoise },
    Field { knob: None, differs: |s, d| differs(s.brightness_lift, d.brightness_lift), reset: |s, d| s.brightness_lift = d.brightness_lift },
    Field { knob: None, differs: |s, d| differs(s.sharpen, d.sharpen), reset: |s, d| s.sharpen = d.sharpen },
    Field { knob: None, differs: |s, d| differs(s.ai_blend, d.ai_blend), reset: |s, d| s.ai_blend = d.ai_blend },
    Field { knob: None, differs: |s, d| differs(s.saturation, d.saturation), reset: |s, d| s.saturation = d.saturation },
    Field { knob: None, differs: |s, d| s.color_match != d.color_match, reset: |s, d| s.color_match = d.color_match },
];

pub(crate) fn render_refine_group(ui: &mut Ui, s: &mut ItemSettings) {
    let d = ItemSettings::default();
    let tuned = chip::tuned_count(REFINE_FIELDS, s, &d);
    let summary = chip::tuned_summary(tuned);
    let group = GroupChip {
        id_salt: "refine",
        icon: ICON_TUNE.codepoint,
        label: "Refine",
        summary: &summary,
        tooltip: "Clean the photo before upscaling and tune the result after.",
        tuned: tuned > 0,
        width: theme::POPOVER_WIDTH,
    };
    chip::group_chip(ui, group, |ui, reset| {
        if reset {
            chip::reset_fields(REFINE_FIELDS, s, &d);
        }
        hint(ui, "Before upscaling. Requires reprocessing.");
        chip::slider_row_f32(ui, "Denoise", &mut s.pre_denoise, 0.0..=1.0, false, |v| fmt::off_or(v, f32::EPSILON, |v| fmt::plain(v, 2)));
        hint(ui, "Smooth grain and speckles first, so the upscaler does not sharpen them.");
        ui.add_space(theme::SPACE_XS);
        chip::slider_row_f32(ui, "Brightness lift", &mut s.brightness_lift, -2.0..=2.0, false, fmt::ev);
        hint(ui, "Brighten shadows so more detail is seen, then restore the original brightness.");
        ui.add_space(theme::SPACE_SM);
        ui.separator();
        hint(ui, "After upscaling.");
        chip::slider_row_f32(ui, "Sharpen", &mut s.sharpen, -1.0..=1.0, false, |v| fmt::off_or(v, f32::EPSILON, |v| fmt::signed_plain(v, 2)));
        hint(ui, "Negative softens, positive sharpens.");
        ui.add_space(theme::SPACE_XS);
        chip::slider_row_f32(ui, "AI blend", &mut s.ai_blend, 0.0..=1.0, false, fmt::percent);
        hint(ui, "Below 100% mixes the plain enlargement back in to soften a plastic look.");
        ui.add_space(theme::SPACE_XS);
        chip::slider_row_f32(ui, "Saturation", &mut s.saturation, -1.0..=1.0, false, |v| fmt::off_or(v, f32::EPSILON, |v| fmt::signed_plain(v, 2)));
        ui.add_space(theme::SPACE_XS);
        chip::toggle_row(ui, "Color match", &mut s.color_match);
        hint(ui, "Match the result's color balance to the original photo.");
    });
}
