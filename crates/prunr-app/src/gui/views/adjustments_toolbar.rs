//! Rows 2 + 3 of the persistent toolbar. Row 2 holds the mask / composite
//! adjustments (gamma, threshold, edge shift, refine edges, bg color).
//! Row 3 holds the line-specific knobs (line strength, solid line color)
//! and is only visible when `line_mode != Off`.
//!
//! View Component discipline: `render` takes `&mut ItemSettings` + a `&AppSettings`
//! reference for defaults / live-preview flag lookups. Never `&mut PrunrApp`.
//!
//! Returns a `ToolbarChange` summarizing WHAT changed so the caller can
//! invalidate the right textures and schedule live-preview reruns in Phase 3.

use egui::{RichText, Ui};
use egui_material_icons::icons::*;

use crate::gui::item_settings::ItemSettings;
use crate::gui::settings::{Settings, SettingsModel};
use crate::gui::theme;
use crate::gui::views::{chip, preset_dropdown};
use prunr_core::LineMode;

use super::model_label;

/// Summary of what a toolbar render cycle changed.
///
/// - `mask` / `edge` / `bg` — which tier was affected. Live preview dispatches
///   to Tier 2 (mask/edge) or Tier 3 (bg) accordingly.
/// - `commit` — the edit has settled (slider released, checkbox toggled,
///   color picked). When true the caller should FLUSH any pending live
///   preview instead of debouncing further.
/// - `model_changed` — the segmentation model was swapped via the row-2
///   dropdown. Caller invalidates both tensor caches (old caches were run
///   with a different model) and should kick off a fresh Tier 1 Process.
/// - `preset_applied` — a preset replaced the current item's settings.
///   Invalidates both caches since any/all fields may have changed.
#[derive(Default, Debug, Clone, Copy)]
pub struct ToolbarChange {
    pub mask: bool,
    pub edge: bool,
    pub bg: bool,
    pub commit: bool,
    pub model_changed: bool,
    pub preset_applied: bool,
}

impl ToolbarChange {
    /// Whether the change invalidates the cached tensors (mask + edge).
    /// Caller clears them + the user should re-Process for a correct result.
    pub fn needs_cache_invalidation(&self) -> bool {
        self.model_changed || self.preset_applied
    }
}

/// Factory default values for per-chip reset + "Reset all" button.
///
/// Does NOT depend on `AppSettings.item_defaults` — that field is the template
/// for NEW imports (possibly from v1 migration). Reset means "go back to
/// known-good factory defaults." Users who want their preferred template back
/// should use the Preset dropdown.
struct Defaults {
    template: ItemSettings,
    /// "Pick this when user toggles enabled" fallback for Option chips that
    /// have no factory value (threshold/bg/line_color are None by default but
    /// the color/slider inside the popover needs a starting value).
    threshold_value: f32,
    bg_value: [u8; 4],
    solid_line_color_value: [u8; 3],
}

impl Defaults {
    fn new() -> Self {
        Self {
            template: ItemSettings::default(),
            threshold_value: 0.5,
            bg_value: [255, 255, 255, 255],
            solid_line_color_value: [0, 0, 0],
        }
    }
}

/// Render rows 2 + 3. Returns a `ToolbarChange` summarizing what was edited.
/// Takes `&mut Settings` because Row 2 hosts the model dropdown (app-global)
/// and the preset dropdown (reads/writes the preset map).
pub fn render(
    ui: &mut Ui,
    item_settings: &mut ItemSettings,
    app_settings: &mut Settings,
    processing: bool,
) -> ToolbarChange {
    let mut change = ToolbarChange::default();
    let defaults = Defaults::new();

    ui.spacing_mut().item_spacing.x = theme::SPACE_SM;

    // Aggregate helper: lift a chip's ChipChange into the aggregate ToolbarChange
    // by the tier it affects. Keeps per-chip call sites a single if-block.
    let aggregate = |ch: chip::ChipChange, tier: Tier, acc: &mut ToolbarChange| {
        if ch.changed {
            match tier {
                Tier::Mask => acc.mask = true,
                Tier::Edge => acc.edge = true,
                Tier::Bg   => acc.bg   = true,
            }
        }
        if ch.commit { acc.commit = true; }
    };

    // ── Row 2: model (leftmost) + mask + composite + preset + reset (right) ──
    ui.horizontal(|ui| {
        // Model dropdown at the very left of row 2.
        render_model_dropdown(ui, app_settings, processing, &mut change);

        // Mask knobs are irrelevant when segmentation is skipped (EdgesOnly mode).
        let mask_active = item_settings.line_mode != LineMode::EdgesOnly;
        ui.add_enabled_ui(mask_active, |ui| {
            aggregate(chip::chip_f32(
                ui, "gamma", "γ", "Gamma",
                "How hard the mask cuts. >1 removes more aggressively, <1 is gentler on fine edges.",
                &mut item_settings.gamma,
                0.2..=3.0, defaults.template.gamma,
                |v| format!("{v:.2}"),
            ), Tier::Mask, &mut change);

            aggregate(chip::chip_option_f32(
                ui, "threshold",
                &ICON_BOLT.codepoint.to_string(), "Hard threshold",
                "Snap the mask to fully opaque or fully transparent at this cutoff. Soft = smooth alpha, on = crisp silhouette.",
                &mut item_settings.threshold,
                0.01..=0.99, defaults.threshold_value, "Soft",
                |v| format!("{v:.2}"),
            ), Tier::Mask, &mut change);

            aggregate(chip::chip_f32(
                ui, "edge_shift",
                &ICON_SWAP_HORIZ.codepoint.to_string(), "Edge shift",
                "Shrink or grow the mask outline. Positive = erode (trim fringe pixels), negative = dilate (keep more edge detail).",
                &mut item_settings.edge_shift,
                -5.0..=5.0, defaults.template.edge_shift,
                |v| {
                    if v > 0.5 { format!("erode {v:.0}px") }
                    else if v < -0.5 { format!("dilate {:.0}px", v.abs()) }
                    else { "0px".to_string() }
                },
            ), Tier::Mask, &mut change);

            aggregate(chip::chip_bool(
                ui, "refine_edges",
                &ICON_AUTO_FIX_HIGH.codepoint.to_string(), "Refine edges",
                "Use the original image's colors to sharpen the mask around fine detail like hair or leaves. Runs a guided filter — slower but higher quality.",
                &mut item_settings.refine_edges,
            ), Tier::Mask, &mut change);
        });

        // Divider between mask and composite groups.
        ui.separator();

        // Background color — a composite concern, always visible (works in all modes).
        aggregate(chip::chip_option_rgba(
            ui, "bg",
            &ICON_PALETTE.codepoint.to_string(), "Background color",
            "Fill transparent areas with a solid color. Applied at display time, does not change the saved PNG's transparency.",
            &mut item_settings.bg,
            defaults.bg_value,
        ), Tier::Bg, &mut change);

        // Right-aligned cluster: reset, preset. Right-to-left layout fills
        // from the right edge so items stack: [..free space..] [preset] [↺].
        ui.with_layout(
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                // Reset-all-knobs button visible directly (per user feedback —
                // was previously hidden in the kebab overflow menu).
                let reset_btn = egui::Button::new(
                    RichText::new(ICON_RESTART_ALT.codepoint)
                        .color(theme::TEXT_SECONDARY)
                        .size(18.0),
                )
                .fill(theme::BG_SECONDARY)
                .corner_radius(theme::BUTTON_ROUNDING)
                .min_size(egui::vec2(chip::CHIP_HEIGHT, chip::CHIP_HEIGHT));
                if ui.add(reset_btn)
                    .on_hover_text("Reset all knobs on this image to factory defaults")
                    .clicked()
                {
                    *item_settings = defaults.template;
                    // Act like "everything changed" — live preview dispatches
                    // a fresh Tier 2 and the user sees the reset take effect.
                    change.mask = true;
                    change.edge = true;
                    change.bg = true;
                    change.commit = true;
                }

                // Preset dropdown — applies/saves presets for the current image.
                let preset_applied = preset_dropdown::render(ui, app_settings, item_settings);
                if preset_applied {
                    change.preset_applied = true;
                    change.commit = true;
                    change.mask = true;
                    change.edge = true;
                    change.bg = true;
                }
            },
        );
    });

    // ── Row 3: line-specific knobs (conditional) ──
    if item_settings.line_mode != LineMode::Off {
        ui.add_space(theme::SPACE_XS);
        ui.horizontal(|ui| {
            aggregate(chip::chip_f32(
                ui, "line_strength",
                &ICON_TUNE.codepoint.to_string(), "Line strength",
                "How much edge detail to capture. Lower = bold outlines only, higher = fine texture and subtle edges.",
                &mut item_settings.line_strength,
                0.05..=1.0, defaults.template.line_strength,
                |v| format!("{v:.2}"),
            ), Tier::Edge, &mut change);

            aggregate(chip::chip_option_rgb(
                ui, "solid_line_color",
                &ICON_BRUSH.codepoint.to_string(), "Solid line color",
                "Paint every edge the same color instead of keeping the original RGB underneath.",
                &mut item_settings.solid_line_color,
                defaults.solid_line_color_value,
            ), Tier::Edge, &mut change);
        });
    }

    change
}

/// Which tier a chip's change lifts into on the aggregate ToolbarChange.
#[derive(Copy, Clone)]
enum Tier { Mask, Edge, Bg }

/// Row 2 leftmost: model dropdown. Edits `app_settings.model` directly and
/// sets `change.model_changed` + `commit` when the selection flips so caller
/// can invalidate tensor caches and fire a fresh Tier 1.
fn render_model_dropdown(
    ui: &mut Ui,
    app_settings: &mut Settings,
    processing: bool,
    change: &mut ToolbarChange,
) {
    let prev_model = app_settings.model;
    ui.add_enabled_ui(!processing, |ui| {
        // Match the combobox visuals used by row 1's other dropdowns.
        let vis = ui.visuals_mut();
        vis.widgets.inactive.weak_bg_fill = theme::BG_SECONDARY;
        vis.widgets.inactive.fg_stroke.color = theme::TEXT_PRIMARY;
        vis.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(0x30, 0x2e, 0x32);
        vis.widgets.hovered.fg_stroke.color = theme::TEXT_PRIMARY;
        vis.widgets.open.weak_bg_fill = theme::BG_SECONDARY;
        vis.widgets.open.fg_stroke.color = theme::TEXT_PRIMARY;
        vis.widgets.active.fg_stroke.color = theme::TEXT_PRIMARY;
        vis.widgets.noninteractive.fg_stroke.color = theme::TEXT_SECONDARY;

        ui.spacing_mut().interact_size.y = chip::CHIP_HEIGHT;
        egui::ComboBox::from_id_salt("adjustments_model")
            .selected_text(
                RichText::new(model_label(app_settings.model, true))
                    .color(theme::TEXT_PRIMARY),
            )
            .show_ui(ui, |ui| {
                for variant in SettingsModel::ALL {
                    ui.selectable_value(
                        &mut app_settings.model,
                        variant,
                        RichText::new(model_label(variant, false))
                            .color(theme::TEXT_PRIMARY),
                    );
                }
            })
            .response
            .on_hover_text("Segmentation model — affects quality, speed, and download size");
    });

    if app_settings.model != prev_model {
        // Clamp parallel jobs to the new model's safe maximum. Keeping this
        // here (vs the caller) because it's a correctness invariant on
        // app_settings — we mustn't leave an invalid parallel_jobs value.
        // Disk persistence, toasts, and cache invalidation all live on the
        // caller via `change.model_changed`.
        let max = app_settings.max_jobs();
        if app_settings.parallel_jobs > max {
            app_settings.parallel_jobs = max;
        }
        change.model_changed = true;
        change.commit = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn needs_cache_invalidation_fires_on_model_or_preset() {
        let mut c = ToolbarChange::default();
        assert!(!c.needs_cache_invalidation());
        c.model_changed = true;
        assert!(c.needs_cache_invalidation());
        let mut c = ToolbarChange::default();
        c.preset_applied = true;
        assert!(c.needs_cache_invalidation());
    }
}
