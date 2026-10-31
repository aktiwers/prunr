//! Which settings a model's preset entry holds: those its toolbar and
//! brush strips show (`knob_catalog::shows`), by their names in a preset
//! file. Prunr's preset lists exactly these for every model; a saved
//! preset keeps those of them that differ.

use super::knob_catalog::{shows, Surface};
use super::settings::SettingsModel;

/// The image settings a surface's controls set.
fn item_fields(surface: Surface) -> &'static [&'static str] {
    match surface {
        Surface::Mask => &["gamma", "contrast", "edge_shift", "refine_edges", "guided_radius", "guided_epsilon", "refine_guide", "feather"],
        Surface::Lines => &[
            "line_mode", "line_strength", "solid_line_color", "edge_thickness", "edge_scale",
            "compose_mode", "line_style", "input_transform", "line_fade_pct",
        ],
        Surface::Effect => &["fill_style"],
        Surface::Decoration => &["decoration"],
        Surface::Background => &["bg", "bg_effect", "bg_image_hash", "bg_image_fit"],
        Surface::Upscale => &["output_scale", "pre_denoise", "sharpen", "ai_blend", "color_match"],
        Surface::Colorize => &["color_strength_pct"],
        Surface::Faces => &["faces"],
        Surface::Depth => &["depth"],
        Surface::SdChips | Surface::RoremChips | Surface::FillChips | Surface::Brushes | Surface::SelectionStrokes | Surface::FillRegionKnobs => &[],
    }
}

/// The brush settings a surface's controls set. The Stable Diffusion
/// eraser's prompt, quality, steps, guidance, seed and decoder sit in the
/// entry's `sd` part instead.
fn brush_fields(surface: Surface) -> &'static [&'static str] {
    match surface {
        Surface::Brushes => &["radius", "shape", "mode", "magic_confidence_threshold", "magic_remove_specks", "magic_depth", "magic_depth_range"],
        Surface::SelectionStrokes => &["hardness", "strength", "cutout_action", "edge_feather"],
        Surface::FillRegionKnobs => &["inpaint_grow", "inpaint_feather"],
        Surface::FillChips => &["inpaint_fill_scale", "inpaint_sharpen", "inpaint_edge_color"],
        Surface::SdChips => &["inpaint_fill_scale", "inpaint_sharpen", "inpaint_edge_color", "sd_soft_edge", "sd_sampler", "sd_strength"],
        Surface::RoremChips => &["inpaint_fill_scale", "inpaint_sharpen", "inpaint_edge_color", "sd_soft_edge", "rorem_steps", "sd_seed"],
        Surface::Mask | Surface::Lines | Surface::Effect | Surface::Decoration | Surface::Background
        | Surface::Upscale | Surface::Colorize | Surface::Faces | Surface::Depth => &[],
    }
}

/// The settings `model`'s preset entry holds.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Fields {
    pub item: Vec<&'static str>,
    pub brush: Vec<&'static str>,
    /// The entry carries an `sd` part: the Stable Diffusion eraser's.
    pub sd: bool,
}

pub(crate) fn of(model: SettingsModel) -> Fields {
    let mut item: Vec<&'static str> = Vec::new();
    let mut brush: Vec<&'static str> = Vec::new();
    for surface in Surface::ALL.into_iter().filter(|&s| shows(model, s)) {
        for &field in item_fields(surface) {
            if !item.contains(&field) {
                item.push(field);
            }
        }
        for &field in brush_fields(surface) {
            if !brush.contains(&field) {
                brush.push(field);
            }
        }
    }
    Fields { item, brush, sd: shows(model, Surface::SdChips) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::brush_state::BrushSettings;
    use crate::gui::item_settings::ItemSettings;

    /// The brush settings the Stable Diffusion eraser keeps in its entry's
    /// `sd` part instead.
    const SD_PART: &[&str] = &["sd_prompt", "sd_negative_prompt", "sd_guidance_scale", "sd_scheduler", "sd_steps", "sd_seed", "sd_use_taesd"];

    fn keys<T: serde::Serialize>(value: &T) -> Vec<String> {
        let mut keys: Vec<String> = serde_json::to_value(value).unwrap().as_object().unwrap().keys().cloned().collect();
        keys.sort();
        keys
    }

    fn sorted<'a>(fields: impl IntoIterator<Item = &'a str>) -> Vec<String> {
        let mut fields: Vec<String> = fields.into_iter().map(str::to_string).collect();
        fields.sort();
        fields.dedup();
        fields
    }

    /// Every image and brush setting belongs to a surface, so a new one
    /// lands in some model's preset entry and in Prunr's file.
    #[test]
    fn every_setting_belongs_to_a_surface() {
        let item = sorted(Surface::ALL.into_iter().flat_map(item_fields).copied());
        assert_eq!(item, keys(&ItemSettings::default()), "an image setting no surface sets");
        let brush = sorted(Surface::ALL.into_iter().flat_map(brush_fields).copied().chain(SD_PART.iter().copied()));
        assert_eq!(brush, keys(&BrushSettings::default()), "a brush setting no surface sets");
    }

    #[test]
    fn a_model_holds_the_settings_it_shows() {
        let silueta = of(SettingsModel::Silueta);
        assert!(silueta.item.contains(&"gamma") && silueta.item.contains(&"fill_style") && !silueta.item.contains(&"sharpen"));
        assert!(silueta.brush.contains(&"cutout_action") && !silueta.brush.contains(&"inpaint_grow") && !silueta.sd);
        let nomos2 = of(SettingsModel::Nomos2Upscale);
        assert!(nomos2.item.contains(&"sharpen") && nomos2.item.contains(&"faces") && nomos2.brush.is_empty());
        assert!(of(SettingsModel::SdInpaint).sd && !of(SettingsModel::RoremInpaint).sd, "RORem takes no prompt or quality");
        assert!(of(SettingsModel::RoremInpaint).brush.contains(&"rorem_steps"));
        assert!(!of(SettingsModel::None).item.contains(&"gamma") && of(SettingsModel::None).item.contains(&"depth"));
    }
}
