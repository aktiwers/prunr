//! Which settings each model's preset entry holds, by their names in a
//! preset file. A setting lives with the model that makes its result: a
//! model the user picks keeps the settings of its own panels
//! (`knob_catalog::shows`), and a helper model that runs inside them (the
//! face restorers, the depth model, Magic Brush's, the line model) keeps
//! its own wherever it runs. Prunr's preset lists exactly these for every
//! model; a saved preset keeps those of them that differ.
//!
//! A name with a dot is a setting inside a group (`faces.hair`): the
//! picked model chooses a face restorer, the restorer keeps its strength.

use prunr_models::ModelId;

use super::item_settings::FaceModel;
use super::knob_catalog::{shows, Surface};
use super::settings::SettingsModel;

/// The models that run inside the picked one and keep their own settings.
pub(crate) const HELPERS: [ModelId; 5] = [
    ModelId::Teed,
    ModelId::Sam21HieraSmall,
    ModelId::FaceGfpgan,
    ModelId::FaceRestoreFormerPp,
    ModelId::DepthAnythingV2Small,
];

/// The image settings a surface's controls set that belong to the picked
/// model.
fn item_fields(surface: Surface) -> &'static [&'static str] {
    match surface {
        Surface::Mask => &["gamma", "contrast", "edge_shift", "refine_edges", "guided_radius", "guided_epsilon", "refine_guide", "feather"],
        Surface::Lines => &["line_mode", "compose_mode"],
        Surface::Effect => &["fill_style"],
        Surface::Decoration => &["decoration"],
        Surface::Background => &["bg", "bg_effect", "bg_image_hash", "bg_image_fit"],
        Surface::Upscale => &["output_scale", "pre_denoise", "sharpen", "ai_blend", "color_match"],
        Surface::Colorize => &["color_strength_pct"],
        Surface::Faces => &["faces.on", "faces.model"],
        Surface::Depth | Surface::SdChips | Surface::RoremChips | Surface::FillChips | Surface::Brushes | Surface::SelectionStrokes
        | Surface::FillRegionKnobs => &[],
    }
}

/// The brush settings a surface's controls set that belong to the picked
/// model. The Stable Diffusion eraser's prompt, quality, steps, guidance,
/// seed and decoder sit in the entry's `sd` part instead.
fn brush_fields(surface: Surface) -> &'static [&'static str] {
    match surface {
        Surface::Brushes => &["radius", "shape", "mode"],
        Surface::SelectionStrokes => &["hardness", "strength", "cutout_action", "edge_feather"],
        Surface::FillRegionKnobs => &["inpaint_grow", "inpaint_feather"],
        Surface::FillChips => &["inpaint_fill_scale", "inpaint_sharpen", "inpaint_edge_color"],
        Surface::SdChips => &["inpaint_fill_scale", "inpaint_sharpen", "inpaint_edge_color", "sd_soft_edge", "sd_sampler", "sd_strength"],
        Surface::RoremChips => &["inpaint_fill_scale", "inpaint_sharpen", "inpaint_edge_color", "sd_soft_edge", "rorem_steps", "sd_seed"],
        Surface::Mask | Surface::Lines | Surface::Effect | Surface::Decoration | Surface::Background
        | Surface::Upscale | Surface::Colorize | Surface::Faces | Surface::Depth => &[],
    }
}

/// The settings a model's preset entry holds.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Fields {
    pub item: Vec<&'static str>,
    pub brush: Vec<&'static str>,
    /// The entry carries an `sd` part: the Stable Diffusion eraser's.
    pub sd: bool,
}

/// The settings of the picked model's own panels.
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

/// A helper model's own settings, wherever it runs.
pub(crate) fn of_helper(helper: ModelId) -> Fields {
    let (item, brush): (&[&'static str], &[&'static str]) = match helper {
        ModelId::Teed => (&["line_strength", "solid_line_color", "edge_thickness", "edge_scale", "line_style", "input_transform", "line_fade_pct"], &[]),
        ModelId::Sam21HieraSmall => (&[], &["magic_confidence_threshold", "magic_remove_specks", "magic_depth", "magic_depth_range"]),
        ModelId::FaceGfpgan | ModelId::FaceRestoreFormerPp => (&["faces.strength_pct", "faces.largest_only", "faces.hair"], &[]),
        ModelId::DepthAnythingV2Small => (&["depth"], &[]),
        _ => (&[], &[]),
    };
    Fields { item: item.to_vec(), brush: brush.to_vec(), sd: false }
}

/// The helper models that run inside `model`, with `face` the face
/// restorer it has chosen: the line model with Sketch, the face restorer
/// with Faces, the depth model with Depth, Magic Brush's with brushes.
pub(crate) fn helpers_of(model: SettingsModel, face: FaceModel) -> Vec<ModelId> {
    [
        (Surface::Lines, ModelId::Teed),
        (Surface::Faces, face.id()),
        (Surface::Depth, ModelId::DepthAnythingV2Small),
        (Surface::Brushes, ModelId::Sam21HieraSmall),
    ]
    .into_iter()
    .filter_map(|(surface, helper)| shows(model, surface).then_some(helper))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::brush_state::BrushSettings;
    use crate::gui::item_settings::ItemSettings;

    /// The brush settings the Stable Diffusion eraser keeps in its entry's
    /// `sd` part instead.
    const SD_PART: &[&str] = &["sd_prompt", "sd_negative_prompt", "sd_guidance_scale", "sd_scheduler", "sd_steps", "sd_seed", "sd_use_taesd"];

    /// A struct's settings by name, a group split into its own (`faces`).
    fn names<T: serde::Serialize>(value: &T, split: &[&str]) -> Vec<String> {
        let mut names = Vec::new();
        for (key, value) in serde_json::to_value(value).unwrap().as_object().unwrap() {
            match value.as_object().filter(|_| split.contains(&key.as_str())) {
                Some(group) => names.extend(group.keys().map(|sub| format!("{key}.{sub}"))),
                None => names.push(key.clone()),
            }
        }
        names.sort();
        names
    }

    fn sorted<'a>(fields: impl IntoIterator<Item = &'a str>) -> Vec<String> {
        let mut fields: Vec<String> = fields.into_iter().map(str::to_string).collect();
        fields.sort();
        fields.dedup();
        fields
    }

    /// Every image and brush setting belongs to a picked model's panel or
    /// to a helper model, so a new one lands in some entry and in Prunr's
    /// file.
    #[test]
    fn every_setting_belongs_to_a_model() {
        let helper_item = HELPERS.into_iter().flat_map(|h| of_helper(h).item);
        let item = sorted(Surface::ALL.into_iter().flat_map(item_fields).copied().chain(helper_item));
        assert_eq!(item, names(&ItemSettings::default(), &["faces"]), "an image setting no model keeps");
        let helper_brush = HELPERS.into_iter().flat_map(|h| of_helper(h).brush);
        let brush = sorted(Surface::ALL.into_iter().flat_map(brush_fields).copied().chain(helper_brush).chain(SD_PART.iter().copied()));
        assert_eq!(brush, names(&BrushSettings::default(), &[]), "a brush setting no model keeps");
    }

    #[test]
    fn a_model_holds_the_settings_of_its_own_panels() {
        let silueta = of(SettingsModel::Silueta);
        assert!(silueta.item.contains(&"gamma") && silueta.item.contains(&"line_mode") && !silueta.item.contains(&"line_strength"));
        assert!(silueta.item.contains(&"faces.model") && !silueta.item.contains(&"faces.strength_pct"), "the restorer keeps its strength");
        assert!(silueta.brush.contains(&"cutout_action") && !silueta.brush.contains(&"magic_confidence_threshold"));
        let nomos2 = of(SettingsModel::Nomos2Upscale);
        assert!(nomos2.item.contains(&"sharpen") && nomos2.brush.is_empty());
        assert!(of(SettingsModel::SdInpaint).sd && !of(SettingsModel::RoremInpaint).sd, "RORem takes no prompt or quality");
        assert!(!of(SettingsModel::None).item.contains(&"depth"), "depth is the depth model's");
    }

    #[test]
    fn a_helper_runs_where_its_panel_shows() {
        use ModelId::*;
        assert_eq!(helpers_of(SettingsModel::BiRefNetLite, FaceModel::Gfpgan), [Teed, FaceGfpgan, Sam21HieraSmall]);
        assert_eq!(helpers_of(SettingsModel::None, FaceModel::RestoreFormerPp), [Teed, FaceRestoreFormerPp, DepthAnythingV2Small, Sam21HieraSmall]);
        assert_eq!(helpers_of(SettingsModel::Nomos2Upscale, FaceModel::Gfpgan), [FaceGfpgan]);
        assert_eq!(helpers_of(SettingsModel::Inpaint, FaceModel::Gfpgan), [Sam21HieraSmall]);
        assert!(helpers_of(SettingsModel::ColorizeDdcolor, FaceModel::Gfpgan).is_empty());
    }
}
