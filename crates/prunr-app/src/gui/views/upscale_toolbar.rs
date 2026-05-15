//! Upscale toolbar row.
//!
//! Rendered when `app_settings.model.is_upscale()` is true, replacing
//! the segmentation chip row. Layout: model dropdown · scale chip ·
//! right-aligned reset/preset cluster. Tile-progress is shown by the
//! unified banner / modal overlay on the canvas
//! (`gui::views::progress_widget`), not inline.
//!
//! Row 3 (line knobs) is entirely absent in upscale mode — the caller
//! in `adjustments_toolbar::render` gates it on `!upscale_mode`.

use egui::Ui;

use super::adjustments_toolbar::{ToolbarChange, render_model_dropdown, render_reset_preset_cluster};
use super::upscale_chip::render_scale_chip;
use crate::gui::item_settings::ItemSettings;
use crate::gui::settings::Settings;

/// Render the upscale toolbar row.
///
/// `source_dims` is `(w, h)` of the active item's input image —
/// `result_rgba` dimensions when chain mode is on and a result exists,
/// otherwise the raw source image dimensions. Resolved by the caller.
///
/// `is_processing` disables the model dropdown while a batch is in
/// flight.
pub(crate) fn render_upscale_row(
    ui: &mut Ui,
    app_settings: &mut Settings,
    item_settings: &mut ItemSettings,
    applied_preset: &mut String,
    source_dims: (u32, u32),
    is_processing: bool,
    change: &mut ToolbarChange,
) {
    ui.horizontal(|ui| {
        render_model_dropdown(ui, app_settings, is_processing, false, change);

        let factor = item_settings.output_scale.factor();
        let out_w = source_dims.0.saturating_mul(factor);
        let out_h = source_dims.1.saturating_mul(factor);
        render_scale_chip(ui, &mut item_settings.output_scale, (out_w, out_h));

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            render_reset_preset_cluster(ui, app_settings, item_settings, applied_preset, change);
        });
    });
}
