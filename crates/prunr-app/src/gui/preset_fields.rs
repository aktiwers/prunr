//! Which settings each model's preset entry holds, by their names in a
//! preset file. A setting lives with the model that makes its result: a
//! model the user picks keeps the settings of the panels it shows, and a
//! helper model that runs inside them (the
//! face restorers, the depth model, Magic Brush's, the line model) keeps
//! its own wherever it runs. Prunr's preset lists exactly these for every
//! model; a saved preset keeps those of them that differ.
//!
//! A name with a dot is a setting inside a group (`faces.hair`): the
//! picked model chooses a face restorer, the restorer keeps its strength.

use prunr_models::ModelId;

use super::item_settings::FaceModel;
use super::knob_catalog::{shows, Surface};
use super::presets::{model_id_key, preset_key};
use super::settings::SettingsModel;

/// A model that runs inside the picked one and keeps its own settings.
pub(crate) struct Helper {
    pub id: ModelId,
    /// How the preset list names it.
    pub name: &'static str,
    /// The panel it runs from.
    pub surface: Surface,
    pub item: &'static [&'static str],
    pub brush: &'static [&'static str],
}

const RESTORER: &[&str] = &["faces.strength_pct", "faces.largest_only", "faces.hair"];

/// Every helper model, in the order the preset list names them.
pub(crate) const HELPERS: [Helper; 5] = [
    Helper {
        id: ModelId::Teed,
        name: "Lines",
        surface: Surface::Lines,
        item: &["line_strength", "solid_line_color", "edge_thickness", "edge_scale", "line_style", "input_transform", "line_fade_pct"],
        brush: &[],
    },
    Helper { id: ModelId::FaceGfpgan, name: "Faces (GFPGAN)", surface: Surface::Faces, item: RESTORER, brush: &[] },
    Helper { id: ModelId::FaceRestoreFormerPp, name: "Faces (RestoreFormer++)", surface: Surface::Faces, item: RESTORER, brush: &[] },
    Helper { id: ModelId::DepthAnythingV2Small, name: "Depth", surface: Surface::Depth, item: &["depth"], brush: &[] },
    Helper {
        id: ModelId::Sam21HieraSmall,
        name: "Magic Brush",
        surface: Surface::Brushes,
        item: &[],
        brush: &["magic_confidence_threshold", "magic_remove_specks", "magic_depth", "magic_depth_range"],
    },
];

impl Helper {
    pub(crate) fn fields(&self) -> Fields {
        Fields { item: self.item.to_vec(), brush: self.brush.to_vec(), sd: false }
    }
}

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

/// The brush settings the Stable Diffusion eraser keeps in its entry's
/// `sd` part.
const SD_PART: &[&str] = &["sd_prompt", "sd_negative_prompt", "sd_guidance_scale", "sd_scheduler", "sd_steps", "sd_seed", "sd_use_taesd"];

/// Whether a surface's brush settings tune an eraser's fill rather than
/// the brush. Every surface is named, so a new one is decided here.
fn tunes_the_fill(surface: Surface) -> bool {
    match surface {
        Surface::FillChips | Surface::SdChips | Surface::RoremChips => true,
        Surface::Brushes | Surface::SelectionStrokes | Surface::FillRegionKnobs | Surface::Mask | Surface::Lines | Surface::Effect
        | Surface::Decoration | Surface::Background | Surface::Upscale | Surface::Colorize | Surface::Faces | Surface::Depth => false,
    }
}

/// The brush settings that tune `model`'s fill, its SD part's included.
/// They follow the eraser on a switch even when changed: a value tuned
/// for one eraser lowers another's fill (Sharpen crunches a generated
/// one).
pub(crate) fn fill_tuning_of(model: SettingsModel) -> Vec<&'static str> {
    let sd = if shows(model, Surface::SdChips) { SD_PART } else { &[] };
    Surface::ALL.into_iter().filter(|&s| tunes_the_fill(s) && shows(model, s)).flat_map(brush_fields).chain(sd).copied().collect()
}

/// The settings a model's preset entry holds.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Fields {
    pub item: Vec<&'static str>,
    pub brush: Vec<&'static str>,
    /// The entry carries an `sd` part: the Stable Diffusion eraser's.
    pub sd: bool,
}

impl Fields {
    /// The brush settings the entry sets, those of its `sd` part included.
    pub(crate) fn brush_and_sd(&self) -> Vec<&'static str> {
        let sd = if self.sd { SD_PART } else { &[] };
        self.brush.iter().chain(sd).copied().collect()
    }
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

/// The helper models whose panel `model` shows, each face restorer among
/// them.
pub(crate) fn may_run(model: SettingsModel) -> impl Iterator<Item = &'static Helper> {
    HELPERS.iter().filter(move |h| shows(model, h.surface))
}

/// The helper models that run inside `model`, with `face` the face
/// restorer it has chosen.
pub(crate) fn helpers_of(model: SettingsModel, face: FaceModel) -> impl Iterator<Item = &'static Helper> {
    may_run(model).filter(move |h| h.surface != Surface::Faces || h.id == face.id())
}

/// The entries `model` reads, by key, with `face` the restorer it has
/// chosen: its own, then those of the helper models that run inside it.
pub(crate) fn entries_of(model: SettingsModel, face: FaceModel) -> Vec<(String, Fields)> {
    std::iter::once((preset_key(model), of(model)))
        .chain(helpers_of(model, face).map(|helper| (model_id_key(helper.id), helper.fields())))
        .collect()
}

/// `entries_of` for the brush: a face restorer has no brush settings, so
/// which one runs does not matter.
pub(crate) fn brush_entries_of(model: SettingsModel) -> Vec<(String, Fields)> {
    entries_of(model, FaceModel::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::brush_state::BrushSettings;
    use crate::gui::item_settings::ItemSettings;

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
        let helper_item = HELPERS.iter().flat_map(|h| h.item.iter().copied());
        let item = sorted(Surface::ALL.into_iter().flat_map(item_fields).copied().chain(helper_item));
        assert_eq!(item, names(&ItemSettings::default(), &["faces"]), "an image setting no model keeps");
        let helper_brush = HELPERS.iter().flat_map(|h| h.brush.iter().copied());
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
        let ids = |model, face| helpers_of(model, face).map(|h| h.id).collect::<Vec<_>>();
        assert_eq!(ids(SettingsModel::BiRefNetLite, FaceModel::Gfpgan), [Teed, FaceGfpgan, Sam21HieraSmall]);
        assert_eq!(ids(SettingsModel::None, FaceModel::RestoreFormerPp), [Teed, FaceRestoreFormerPp, DepthAnythingV2Small, Sam21HieraSmall]);
        assert_eq!(ids(SettingsModel::Nomos2Upscale, FaceModel::Gfpgan), [FaceGfpgan]);
        assert_eq!(ids(SettingsModel::Inpaint, FaceModel::Gfpgan), [Sam21HieraSmall]);
        assert!(ids(SettingsModel::ColorizeDdcolor, FaceModel::Gfpgan).is_empty());
    }
}
