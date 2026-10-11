//! Upscale chips for the toolbar: the Scale picker and the Refine group.
//! The Model picker and the preset cluster are rendered by the caller.

use egui::Ui;

use super::refinement_row::render_refine_group;
use super::upscale_chip::render_output_scale_chip;
use crate::gui::item_settings::ItemSettings;
use crate::gui::settings::Settings;

/// `source_dims` is `(w, h)` of the image the next upscale starts from,
/// resolved by the caller.
pub(crate) fn render_upscale_chips(
    ui: &mut Ui,
    app_settings: &Settings,
    item_settings: &mut ItemSettings,
    source_dims: (u32, u32),
) {
    let factor = item_settings.output_scale.factor();
    let out_w = source_dims.0.saturating_mul(factor);
    let out_h = source_dims.1.saturating_mul(factor);
    if app_settings.model.to_model_id().is_none() {
        return;
    }
    render_output_scale_chip(ui, &mut item_settings.output_scale, (out_w, out_h));
    render_refine_group(ui, item_settings);
}
