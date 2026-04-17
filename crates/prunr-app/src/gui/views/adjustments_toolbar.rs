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
use crate::gui::theme;
use crate::gui::views::chip;
use prunr_core::LineMode;

/// Summary of what a toolbar render cycle changed. Phase 2 callers use `any`
/// to trigger texture rebuilds; Phase 3 will branch on `mask`, `edge`, `bg`
/// to dispatch the correct Tier 2 path.
#[derive(Default, Debug, Clone, Copy)]
pub struct ToolbarChange {
    pub mask: bool,
    pub edge: bool,
    pub bg: bool,
}

impl ToolbarChange {
    pub fn any(&self) -> bool {
        self.mask || self.edge || self.bg
    }
}

/// Factory default values for per-chip reset + "Reset all" kebab action.
///
/// Intentionally does NOT depend on `AppSettings.item_defaults` — that field is
/// the template for NEW imports (possibly populated from v1 migration with a
/// user's old preferences). Reset buttons should mean "go back to known-good
/// factory defaults," which is a separate concern. Users who want to restore
/// their preferred template should use the Preset dropdown.
struct Defaults {
    template: ItemSettings,
    /// Scalar values used as "pick this when user toggles enabled" for Option chips.
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

    fn gamma(&self) -> f32 { self.template.gamma }
    fn edge_shift(&self) -> f32 { self.template.edge_shift }
    fn line_strength(&self) -> f32 { self.template.line_strength }
}

/// Render rows 2 + 3. Returns a `ToolbarChange` summarizing what was edited.
pub fn render(
    ui: &mut Ui,
    item_settings: &mut ItemSettings,
) -> ToolbarChange {
    let mut change = ToolbarChange::default();
    let defaults = Defaults::new();

    ui.spacing_mut().item_spacing.x = theme::SPACE_SM;

    // ── Row 2: mask + composite ──
    ui.horizontal(|ui| {
        // Mask knobs are irrelevant when segmentation is skipped (EdgesOnly mode).
        let mask_active = item_settings.line_mode != LineMode::EdgesOnly;
        ui.add_enabled_ui(mask_active, |ui| {
            if chip::chip_f32(
                ui,
                "gamma",
                "γ",
                "Gamma",
                &mut item_settings.gamma,
                0.2..=3.0,
                defaults.gamma(),
                |v| format!("{v:.2}"),
            ) {
                change.mask = true;
            }
            if chip::chip_option_f32(
                ui,
                "threshold",
                &ICON_BOLT.codepoint.to_string(),
                "Hard threshold",
                &mut item_settings.threshold,
                0.01..=0.99,
                defaults.threshold_value,
                "Soft",
                |v| format!("{v:.2}"),
            ) {
                change.mask = true;
            }
            if chip::chip_f32(
                ui,
                "edge_shift",
                &ICON_SWAP_HORIZ.codepoint.to_string(),
                "Edge shift",
                &mut item_settings.edge_shift,
                -5.0..=5.0,
                defaults.edge_shift(),
                |v| {
                    if v > 0.5 {
                        format!("erode {v:.0}px")
                    } else if v < -0.5 {
                        format!("dilate {:.0}px", v.abs())
                    } else {
                        "0px".to_string()
                    }
                },
            ) {
                change.mask = true;
            }
            if chip::chip_bool(
                ui,
                "refine_edges",
                &ICON_AUTO_FIX_HIGH.codepoint.to_string(),
                "Refine edges",
                &mut item_settings.refine_edges,
                Some("Uses the original image colors to sharpen the mask around fine detail."),
            ) {
                change.mask = true;
            }
        });

        // Divider between mask and composite groups.
        ui.separator();

        // Background color — a composite concern, always visible (works in all modes).
        if chip::chip_option_rgba(
            ui,
            "bg",
            &ICON_PALETTE.codepoint.to_string(),
            "Background color",
            &mut item_settings.bg,
            defaults.bg_value,
        ) {
            change.bg = true;
        }

        // Overflow / kebab menu at the end of row 2.
        ui.with_layout(
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                kebab_menu(ui, item_settings, &defaults);
            },
        );
    });

    // ── Row 3: line-specific knobs (conditional) ──
    if item_settings.line_mode != LineMode::Off {
        ui.add_space(theme::SPACE_XS);
        ui.horizontal(|ui| {
            if chip::chip_f32(
                ui,
                "line_strength",
                &ICON_TUNE.codepoint.to_string(),
                "Line strength",
                &mut item_settings.line_strength,
                0.05..=1.0,
                defaults.line_strength(),
                |v| format!("{v:.2}"),
            ) {
                change.edge = true;
            }
            if chip::chip_option_rgb(
                ui,
                "solid_line_color",
                &ICON_BRUSH.codepoint.to_string(),
                "Solid line color",
                &mut item_settings.solid_line_color,
                defaults.solid_line_color_value,
            ) {
                change.edge = true;
            }
        });
    }

    change
}

/// Row 2 overflow kebab. Houses rarely-used actions that don't deserve a chip slot.
#[allow(deprecated)]
fn kebab_menu(ui: &mut Ui, item_settings: &mut ItemSettings, defaults: &Defaults) {
    let pop_id = egui::Id::new("adjustments_kebab");
    let btn = egui::Button::new(
        RichText::new(ICON_MORE_VERT.codepoint)
            .color(theme::TEXT_SECONDARY)
            .size(18.0),
    )
    .fill(egui::Color32::TRANSPARENT)
    .min_size(egui::vec2(chip::CHIP_HEIGHT, chip::CHIP_HEIGHT));
    let resp = ui.add(btn).on_hover_text("More…");
    if resp.clicked() {
        ui.memory_mut(|m| m.toggle_popup(pop_id));
    }
    egui::popup_below_widget(
        ui,
        pop_id,
        &resp,
        egui::PopupCloseBehavior::CloseOnClickOutside,
        |ui| {
            ui.set_min_width(200.0);
            if ui.button("Reset all knobs on this image").clicked() {
                // Restore to the app-wide defaults template exactly — preserves
                // the user's default_preset / item_defaults Option<> settings
                // instead of forcing everything to None.
                *item_settings = defaults.template;
                ui.memory_mut(|m| m.close_popup(pop_id));
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toolbar_change_any_detects_any_field() {
        assert!(!ToolbarChange::default().any());
        let mut c = ToolbarChange::default();
        c.mask = true;
        assert!(c.any());
        let mut c = ToolbarChange::default();
        c.edge = true;
        assert!(c.any());
        let mut c = ToolbarChange::default();
        c.bg = true;
        assert!(c.any());
    }
}
