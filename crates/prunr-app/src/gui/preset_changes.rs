//! The settings changed since a preset was applied, each under the model
//! it was changed on: a model switch keeps them and takes the rest from
//! the preset, and Save writes each model's under its own entry.

use std::collections::BTreeMap;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::brush_state::BrushSettings;
use super::item_settings::ItemSettings;
use super::preset_fields::Fields;
use super::presets::differing;

/// The image settings or the brush: the part of an entry it fills.
pub(crate) trait Part: Serialize + DeserializeOwned + Clone {
    fn fields(entry: &Fields) -> Vec<&'static str>;
}

impl Part for ItemSettings {
    fn fields(entry: &Fields) -> Vec<&'static str> {
        entry.item.clone()
    }
}

impl Part for BrushSettings {
    fn fields(entry: &Fields) -> Vec<&'static str> {
        entry.brush_and_sd()
    }
}

/// What changed in `T` since a preset set it: the changes made on models
/// since left are noted under their entry keys, the rest is what differs
/// from `from`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Changes<T> {
    /// The settings as the preset, or the last model switch, left them.
    from: T,
    /// Per entry key, the settings changed while that model was in use.
    on: BTreeMap<String, Map<String, Value>>,
}

impl<T: Part> Changes<T> {
    /// Nothing changed yet from `from`.
    pub(crate) fn new(from: T) -> Self {
        Self { from, on: BTreeMap::new() }
    }

    /// Note what `now` changed under `entries`, the entries of the model in
    /// use: each entry's own settings that differ from `from`.
    pub(crate) fn note(&mut self, now: &T, entries: &[(String, Fields)]) {
        for (key, entry) in entries {
            let changed = differing(now, &self.from, &T::fields(entry));
            if !changed.is_empty() {
                self.on.entry(key.clone()).or_default().extend(changed);
            }
        }
    }

    /// `note`, on a copy: what a save writes for the model in use.
    pub(crate) fn noted(&self, now: &T, entries: &[(String, Fields)]) -> Self {
        let mut noted = self.clone();
        noted.note(now, entries);
        noted
    }

    /// Start comparing from `now`: after a switch has filled in the new
    /// model's settings.
    pub(crate) fn rebase(&mut self, now: &T) {
        self.from = now.clone();
    }

    /// Whether `name` was changed on any model; call `note` first for the
    /// one in use.
    pub(crate) fn touched(&self, name: &str) -> bool {
        self.on.values().any(|changed| changed.contains_key(name))
    }

    /// The settings changed on `key`'s model, by name.
    pub(crate) fn on(&self, key: &str) -> Option<&Map<String, Value>> {
        self.on.get(key)
    }

    /// The entry keys with changes.
    pub(crate) fn keys(&self) -> impl Iterator<Item = &str> {
        self.on.keys().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::item_settings::FaceModel;
    use crate::gui::preset_fields::entries_of;
    use crate::gui::settings::SettingsModel;

    #[test]
    fn a_change_is_noted_under_the_model_that_owns_it() {
        let entries = entries_of(SettingsModel::BiRefNetLite, FaceModel::Gfpgan);
        let mut changes = Changes::new(ItemSettings::default());
        let mut now = ItemSettings { gamma: 1.4, edge_thickness: 3, ..ItemSettings::default() };
        now.faces.strength_pct = 40;
        changes.note(&now, &entries);
        assert_eq!(changes.on("BiRefNetLite").map(|c| c.keys().map(String::as_str).collect::<Vec<_>>()), Some(vec!["gamma"]));
        assert!(changes.on("Teed").is_some_and(|c| c.contains_key("edge_thickness")), "the line model's");
        assert!(changes.on("FaceGfpgan").is_some_and(|c| c.contains_key("faces.strength_pct")), "the restorer's");
        assert!(changes.touched("gamma") && !changes.touched("contrast"));
    }

    #[test]
    fn a_setting_set_back_is_no_change() {
        let entries = entries_of(SettingsModel::Silueta, FaceModel::Gfpgan);
        let mut changes = Changes::new(ItemSettings::default());
        changes.note(&ItemSettings::default(), &entries);
        assert_eq!(changes.keys().count(), 0);
    }

    #[test]
    fn a_later_change_on_the_same_model_replaces_the_noted_one() {
        let entries = entries_of(SettingsModel::Silueta, FaceModel::Gfpgan);
        let mut now = ItemSettings { gamma: 1.4, ..ItemSettings::default() };
        let mut changes = Changes::new(ItemSettings::default());
        changes.note(&now, &entries);
        changes.rebase(&now);
        now.gamma = 0.8;
        changes.note(&now, &entries);
        assert_eq!(changes.on("Silueta").and_then(|c| c.get("gamma")), Some(&serde_json::json!(0.8)));
    }

    #[test]
    fn the_brush_notes_its_sd_part_under_the_sd_eraser() {
        let entries = entries_of(SettingsModel::SdInpaint, FaceModel::Gfpgan);
        let mut brush = BrushSettings::default();
        brush.sd_prompt = "a meadow".into();
        brush.magic_remove_specks = !brush.magic_remove_specks;
        let changes = Changes::new(BrushSettings::default()).noted(&brush, &entries);
        assert!(changes.on("SdV15InpaintFp16").is_some_and(|c| c.contains_key("sd_prompt")));
        assert!(changes.on("Sam21HieraSmall").is_some_and(|c| c.contains_key("magic_remove_specks")), "Magic Brush's own");
    }

    /// The brush's changes are kept in the settings file, so a restart
    /// does not count them as untouched.
    #[test]
    fn brush_changes_round_trip_through_json() {
        let entries = entries_of(SettingsModel::Inpaint, FaceModel::Gfpgan);
        let brush = BrushSettings { inpaint_sharpen: 0.25, ..BrushSettings::default() };
        let changes = Changes::new(BrushSettings::default()).noted(&brush, &entries);
        let back: Changes<BrushSettings> = serde_json::from_str(&serde_json::to_string(&changes).unwrap()).unwrap();
        assert_eq!(back, changes);
    }
}
