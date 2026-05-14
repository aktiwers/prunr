//! Upscale toolbar row.
//!
//! Rendered when `app_settings.model.is_upscale()` is true, replacing
//! the segmentation chip row. Layout: model dropdown · scale chip ·
//! inline progress bar (only during dispatch) · right-aligned
//! reset/preset cluster.
//!
//! Row 3 (line knobs) is entirely absent in upscale mode — the caller
//! in `adjustments_toolbar::render` gates it on `!upscale_mode`.

use egui::Ui;

use super::adjustments_toolbar::{ToolbarChange, render_model_dropdown, render_reset_preset_cluster};
use super::upscale_chip::render_scale_chip;
use crate::gui::item_settings::ItemSettings;
use crate::gui::processor::Processor;
use crate::gui::settings::Settings;
use crate::gui::theme;

/// Pixel reserve at the right edge of the progress bar — leaves room for
/// the reset button + preset dropdown in the right-aligned cluster.
const PROGRESS_BAR_RIGHT_RESERVE_PX: f32 = 80.0;

/// Render the upscale toolbar row.
///
/// `source_dims` is `(w, h)` of the active item's input image —
/// `result_rgba` dimensions when chain mode is on and a result exists,
/// otherwise the raw source image dimensions. Resolved by the caller.
///
/// `is_processing` disables the model dropdown while a batch is in
/// flight. `processor` is the source for the live tile-progress poll.
// args mirror the ToolbarChange protocol one-for-one — packing into a
// struct would force borrow-splitting at every caller (the existing
// adjustments_toolbar::render already passes the same separate borrows).
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_upscale_row(
    ui: &mut Ui,
    app_settings: &mut Settings,
    item_settings: &mut ItemSettings,
    applied_preset: &mut String,
    source_dims: (u32, u32),
    is_processing: bool,
    processor: &Processor,
    change: &mut ToolbarChange,
) {
    ui.horizontal(|ui| {
        render_model_dropdown(ui, app_settings, is_processing, false, change);

        let out_w = source_dims.0.saturating_mul(item_settings.upscale_scale);
        let out_h = source_dims.1.saturating_mul(item_settings.upscale_scale);
        if render_scale_chip(ui, &mut item_settings.upscale_scale, (out_w, out_h)) {
            change.upscale_scale_changed = true;
        }

        if let Some((done, total)) = processor.upscale_tile_progress() {
            let avail = (ui.available_width() - PROGRESS_BAR_RIGHT_RESERVE_PX).max(0.0);
            ui.add(
                egui::ProgressBar::new(done as f32 / total.max(1) as f32)
                    .fill(theme::PROGRESS_FILL)
                    .desired_width(avail)
                    .desired_height(theme::PROGRESS_BAR_HEIGHT),
            );
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            render_reset_preset_cluster(ui, app_settings, item_settings, applied_preset, change);
        });
    });
}
