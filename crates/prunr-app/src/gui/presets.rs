//! Presets: per model, the settings a preset sets. Prunr's own preset
//! (`presets/Prunr.json`, built in) holds every model's defaults; a saved
//! preset keeps, per model, only the settings that differ from Prunr's,
//! so the rest follow Prunr's and improve with it. A model a preset has no
//! entry for takes Prunr's settings. v1 files migrate in at load.
//!
//! Old files MUST keep loading: every struct reads missing fields from
//! default and ignores unknown ones (the tripwire tests). `models` is keyed
//! by name, not `ModelId`, so a file from a newer Prunr with models this
//! one doesn't know still loads and keeps them.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use prunr_models::ModelId;

use super::brush_state::{BrushSettings, SdScheduler};
use super::item_settings::ItemSettings;
use super::settings::SettingsModel;

pub(crate) const PRESET_FORMAT_VERSION: u32 = 2;

/// A model's name in a preset's `models`, and in
/// `Settings.accepted_licenses`.
pub(crate) fn model_id_key(id: ModelId) -> String {
    format!("{id:?}")
}

/// The `models` key of "No model", which has no `ModelId`.
pub(crate) const NO_MODEL_KEY: &str = "NoModel";

/// The `models` key `model`'s entry is filed under.
pub(crate) fn preset_key(model: SettingsModel) -> String {
    model.to_model_id().map_or_else(|| NO_MODEL_KEY.to_string(), model_id_key)
}

/// The model filed under `key`; `None` for one this Prunr does not know.
fn model_of_key(key: &str) -> Option<SettingsModel> {
    SettingsModel::ALL.into_iter().find(|&m| preset_key(m) == key)
}

/// The settings of the `sd` part, by name.
const SD_FIELDS: &[&str] = &["prompt", "negative_prompt", "use_taesd", "seed", "active_scheduler", "schedulers"];

/// One model's settings as a file lists them, by name: only those the
/// preset sets.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Entry {
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub item_settings: Map<String, Value>,
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub brush: Map<String, Value>,
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub sd: Map<String, Value>,
}

impl Entry {
    fn is_empty(&self) -> bool {
        self.item_settings.is_empty() && self.brush.is_empty() && self.sd.is_empty()
    }
}

/// A preset as its file keeps it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct PresetFile {
    pub format_version: u32,
    pub models: HashMap<String, Entry>,
}

impl Default for PresetFile {
    fn default() -> Self {
        Self { format_version: PRESET_FORMAT_VERSION, models: HashMap::new() }
    }
}

impl PresetFile {
    /// A preset of image settings alone, filed under the default model: a
    /// v1 file, which predates models in presets.
    pub(crate) fn of_item_settings(item_settings: ItemSettings) -> Self {
        let model = super::settings::DEFAULT_MODEL;
        Self::default().with_settings(model, &item_settings, &prunr_model(model).brush)
    }

    /// The file with `model`'s entry set from the current settings, `item`
    /// and `brush`: those of its settings that differ from Prunr's. The SD
    /// part keeps the bundles of the qualities not in use. A model with
    /// nothing apart from Prunr's has no entry.
    pub(crate) fn with_settings(mut self, model: SettingsModel, item: &ItemSettings, brush: &BrushSettings) -> Self {
        let key = preset_key(model);
        let mut sd = prunr_model(model).sd.as_ref().map(|prunr| {
            let kept = self.models.get(&key).map_or(Map::new(), |entry| entry.sd.clone());
            overlay(prunr.clone(), &kept)
        });
        if let Some(sd) = &mut sd {
            take_sd_from_brush(sd, brush);
        }
        let entry = differences(model, item, brush, sd.as_ref());
        if entry.is_empty() {
            self.models.remove(&key);
        } else {
            self.models.insert(key, entry);
        }
        self.format_version = PRESET_FORMAT_VERSION;
        self
    }

    /// The file as it is kept: each model's entry trimmed to its settings
    /// that differ from Prunr's (a file an earlier Prunr saved whole, or
    /// one written by hand); a model this Prunr does not know is kept as it
    /// is, for the Prunr that does.
    pub(crate) fn trimmed(mut self) -> Self {
        self.models.retain(|key, entry| match model_of_key(key) {
            Some(model) => {
                let view = ModelView::of(model, entry);
                *entry = differences(model, &view.item_settings, &view.brush, view.sd.as_ref());
                !entry.is_empty()
            }
            None => true,
        });
        self
    }
}

/// The settings of `model`'s that differ from Prunr's, by name.
fn differences(model: SettingsModel, item: &ItemSettings, brush: &BrushSettings, sd: Option<&SdPreset>) -> Entry {
    let fields = super::preset_fields::of(model);
    let prunr = prunr_model(model);
    Entry {
        item_settings: differing(item, &prunr.item_settings, &fields.item),
        brush: differing(brush, &prunr.brush, &fields.brush),
        sd: sd.zip(prunr.sd.as_ref()).map_or(Map::new(), |(sd, base)| differing(sd, base, SD_FIELDS)),
    }
}

/// `value`'s `fields` that differ from `base`'s. A field written only
/// when set (an `Option`) is written as null when `value` leaves it out.
fn differing<T: Serialize>(value: &T, base: &T, fields: &[&str]) -> Map<String, Value> {
    let (value, base) = (serde_json::to_value(value).unwrap_or_default(), serde_json::to_value(base).unwrap_or_default());
    fields
        .iter()
        .filter_map(|&field| {
            let own = value.get(field).cloned().unwrap_or(Value::Null);
            (own != base.get(field).cloned().unwrap_or(Value::Null)).then(|| (field.to_string(), own))
        })
        .collect()
}

/// `base` with `over`'s settings laid on it, by name. A preset whose
/// settings do not read leaves `base` as it is.
fn overlay<T: Serialize + serde::de::DeserializeOwned>(base: T, over: &Map<String, Value>) -> T {
    if over.is_empty() {
        return base;
    }
    let mut value = serde_json::to_value(&base).unwrap_or_default();
    if let Some(fields) = value.as_object_mut() {
        fields.extend(over.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    serde_json::from_value(value).unwrap_or_else(|err| {
        tracing::warn!(%err, "preset settings that do not read were left out");
        base
    })
}

/// One model's settings, resolved: its brush before a quality's steps and
/// guidance are laid on it, and the Stable Diffusion eraser's SD part.
#[derive(Debug, Clone, PartialEq)]
struct ModelView {
    item_settings: ItemSettings,
    brush: BrushSettings,
    sd: Option<SdPreset>,
}

impl ModelView {
    /// `entry`'s settings over Prunr's for `model`.
    fn of(model: SettingsModel, entry: &Entry) -> Self {
        let prunr = prunr_model(model);
        Self {
            item_settings: overlay(prunr.item_settings, &entry.item_settings),
            brush: overlay(prunr.brush.clone(), &entry.brush),
            sd: prunr.sd.clone().map(|sd| overlay(sd, &entry.sd)),
        }
    }

    /// `scheduler`: `None` uses the SD part's own quality; `Some`
    /// overrides it, so a quality picked in the session doesn't rewrite
    /// the preset.
    fn resolve(&self, scheduler: Option<SdScheduler>) -> ResolvedView {
        let mut brush = self.brush.clone();
        if let Some(sd) = &self.sd {
            apply_sd_to_brush(&mut brush, sd, scheduler);
        }
        ResolvedView { item_settings: self.item_settings, brush }
    }
}

#[derive(Debug, Deserialize)]
struct PrunrFile {
    models: HashMap<String, Entry>,
}

/// Prunr's settings for every model, in `SettingsModel::ALL` order: its
/// built-in file over the code's own defaults, read once.
fn prunr_models() -> &'static [ModelView] {
    static MODELS: std::sync::OnceLock<Vec<ModelView>> = std::sync::OnceLock::new();
    MODELS.get_or_init(|| {
        // A test reads the file and resolves every model from it.
        let file: PrunrFile = serde_json::from_str(include_str!("../../presets/Prunr.json")).expect("Prunr.json reads");
        SettingsModel::ALL
            .into_iter()
            .map(|model| {
                let entry = file.models.get(&preset_key(model)).cloned().unwrap_or_default();
                ModelView {
                    item_settings: overlay(ItemSettings::default(), &entry.item_settings),
                    brush: overlay(BrushSettings::default(), &entry.brush),
                    sd: (!entry.sd.is_empty()).then(|| overlay(SdPreset::default(), &entry.sd)),
                }
            })
            .collect()
    })
}

fn model_index(model: SettingsModel) -> usize {
    // Every model is in ALL: `SettingsModel::ALL` lists them all.
    SettingsModel::ALL.iter().position(|&m| m == model).unwrap_or(0)
}

fn prunr_model(model: SettingsModel) -> &'static ModelView {
    &prunr_models()[model_index(model)]
}

/// Prunr's settings for `model`, resolved for `scheduler` (`None`: its
/// own quality).
pub(crate) fn prunr_view(model: SettingsModel, scheduler: Option<SdScheduler>) -> ResolvedView {
    prunr_model(model).resolve(scheduler)
}

/// Prunr's image settings for `model`, without cloning its brush.
pub(crate) fn prunr_item_settings(model: SettingsModel) -> ItemSettings {
    prunr_model(model).item_settings
}

/// A preset as the app holds it: its file, and each model's settings
/// resolved once, as the toolbar reads them every frame.
#[derive(Debug, Clone, Deserialize)]
#[serde(from = "PresetFile")]
pub(crate) struct Preset {
    file: PresetFile,
    /// Per model in `SettingsModel::ALL` order; `None` without an entry.
    views: Vec<Option<ModelView>>,
}

impl From<PresetFile> for Preset {
    fn from(file: PresetFile) -> Self {
        let views = SettingsModel::ALL
            .into_iter()
            .map(|model| file.models.get(&preset_key(model)).map(|entry| ModelView::of(model, entry)))
            .collect();
        Self { file, views }
    }
}

impl Preset {
    pub(crate) fn file(&self) -> &PresetFile {
        &self.file
    }

    /// The preset's settings for `model`, Prunr's where it has none.
    pub(crate) fn resolve(&self, model: SettingsModel, scheduler: Option<SdScheduler>) -> ResolvedView {
        self.views[model_index(model)].as_ref().map_or_else(|| prunr_view(model, scheduler), |view| view.resolve(scheduler))
    }

    /// `resolve`'s image settings, without cloning a brush.
    pub(crate) fn item_settings(&self, model: SettingsModel) -> ItemSettings {
        self.views[model_index(model)].as_ref().map_or_else(|| prunr_item_settings(model), |view| view.item_settings)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SdPreset {
    #[serde(deserialize_with = "super::brush_state::own_prompt")]
    pub prompt: String,
    pub negative_prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_taesd: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub active_scheduler: SdScheduler,
    pub schedulers: HashMap<SdScheduler, SdSchedulerBundle>,
}

impl Default for SdPreset {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: super::brush_state::default_sd_negative_prompt(),
            use_taesd: None,
            seed: None,
            active_scheduler: SdScheduler::default(),
            schedulers: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SdSchedulerBundle {
    pub steps: u32,
    pub guidance_scale: f32,
}

impl Default for SdSchedulerBundle {
    fn default() -> Self {
        Self::default_for(SdScheduler::default())
    }
}

impl SdSchedulerBundle {
    /// A quality's steps and guidance when an SD part has no bundle for it.
    pub fn default_for(scheduler: SdScheduler) -> Self {
        match scheduler {
            SdScheduler::Lcm => Self { steps: 8, guidance_scale: 1.5 },
            SdScheduler::DpmPlusPlus2MKarras => Self { steps: 25, guidance_scale: 4.0 },
        }
    }
}

/// A preset's settings for a model and quality: the values the GUI
/// applies.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedView {
    pub item_settings: ItemSettings,
    pub brush: BrushSettings,
}

fn apply_sd_to_brush(brush: &mut BrushSettings, sd: &SdPreset, scheduler: Option<SdScheduler>) {
    let chosen = scheduler.unwrap_or(sd.active_scheduler);
    let bundle = sd.schedulers.get(&chosen).copied().unwrap_or_else(|| SdSchedulerBundle::default_for(chosen));
    brush.sd_scheduler = chosen;
    brush.sd_steps = bundle.steps;
    brush.sd_guidance_scale = bundle.guidance_scale;
    brush.sd_prompt = sd.prompt.clone();
    brush.sd_negative_prompt = sd.negative_prompt.clone();
    brush.sd_seed = sd.seed;
    brush.sd_use_taesd = sd.use_taesd;
}

/// `sd` with the brush's prompt, seed, decoder and quality, and that
/// quality's steps and guidance; the other qualities' bundles stay.
fn take_sd_from_brush(sd: &mut SdPreset, brush: &BrushSettings) {
    sd.prompt = brush.sd_prompt.clone();
    sd.negative_prompt = brush.sd_negative_prompt.clone();
    sd.use_taesd = brush.sd_use_taesd;
    sd.seed = brush.sd_seed;
    sd.active_scheduler = brush.sd_scheduler;
    sd.schedulers.insert(brush.sd_scheduler, SdSchedulerBundle { steps: brush.sd_steps, guidance_scale: brush.sd_guidance_scale });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prunr's file has an entry for every model, listing exactly the
    /// settings that model shows, each one readable: a new model or a new
    /// setting has to be given its default there.
    #[test]
    fn prunr_holds_every_models_settings() {
        fn keys(map: &Map<String, Value>) -> Vec<&str> {
            let mut keys: Vec<&str> = map.keys().map(String::as_str).collect();
            keys.sort_unstable();
            keys
        }
        fn sorted(mut fields: Vec<&'static str>) -> Vec<&'static str> {
            fields.sort_unstable();
            fields
        }
        fn reads<T: Serialize + serde::de::DeserializeOwned>(base: T, over: &Map<String, Value>) -> Result<T, serde_json::Error> {
            let mut value = serde_json::to_value(base)?;
            value.as_object_mut().expect("a struct").extend(over.clone());
            serde_json::from_value(value)
        }
        let file: PrunrFile = serde_json::from_str(include_str!("../../presets/Prunr.json")).expect("Prunr.json reads");
        for model in SettingsModel::ALL {
            let entry = file.models.get(&preset_key(model)).unwrap_or_else(|| panic!("{model:?} has no entry in Prunr.json"));
            let fields = super::super::preset_fields::of(model);
            assert_eq!(keys(&entry.item_settings), sorted(fields.item), "{model:?} image settings");
            assert_eq!(keys(&entry.brush), sorted(fields.brush), "{model:?} brush settings");
            assert_eq!(!entry.sd.is_empty(), fields.sd, "{model:?} SD part");
            reads(ItemSettings::default(), &entry.item_settings).unwrap_or_else(|e| panic!("{model:?} image settings: {e}"));
            reads(BrushSettings::default(), &entry.brush).unwrap_or_else(|e| panic!("{model:?} brush settings: {e}"));
            reads(SdPreset::default(), &entry.sd).unwrap_or_else(|e| panic!("{model:?} SD part: {e}"));
        }
        assert_eq!(file.models.len(), SettingsModel::ALL.len(), "an entry for a model the app does not offer");
    }

    /// Moving the defaults into Prunr's file changed nothing: every model
    /// resolves as the code's defaults did (one set of image settings,
    /// Sharpen 0 on the generative erasers, the factory quality bundles).
    #[test]
    fn moving_the_defaults_into_prunr_changed_nothing() {
        for model in SettingsModel::ALL {
            let generated = model.to_model_id().is_some_and(|id| id.is_sd_family());
            let schedulers: &[Option<SdScheduler>] = if model == SettingsModel::SdInpaint {
                &[None, Some(SdScheduler::Lcm), Some(SdScheduler::DpmPlusPlus2MKarras)]
            } else {
                &[None]
            };
            for &scheduler in schedulers {
                let mut brush = BrushSettings { inpaint_sharpen: if generated { 0.0 } else { 0.6 }, ..BrushSettings::default() };
                if model == SettingsModel::SdInpaint {
                    apply_sd_to_brush(&mut brush, &SdPreset::default(), scheduler);
                }
                let view = prunr_view(model, scheduler);
                assert_eq!(view.item_settings, ItemSettings::default(), "{model:?}");
                assert_eq!(view.brush, brush, "{model:?} {scheduler:?}");
            }
        }
    }

    fn file(json: &str) -> PresetFile {
        serde_json::from_str(json).expect("a preset file")
    }

    /// A saved preset keeps, per model, only what differs from Prunr's:
    /// the rest follows Prunr's, and a model with nothing changed has no
    /// entry.
    #[test]
    fn a_preset_keeps_only_what_differs_from_prunr() {
        let model = SettingsModel::BiRefNetLite;
        let item = ItemSettings { gamma: 1.5, line_mode: prunr_core::LineMode::SubjectOutline, ..prunr_item_settings(model) };
        let brush = BrushSettings { radius: 40.0, ..prunr_view(model, None).brush };
        let saved = serde_json::to_value(PresetFile::default().with_settings(model, &item, &brush)).unwrap();
        assert_eq!(
            saved["models"],
            serde_json::json!({ "BiRefNetLite": { "item_settings": { "gamma": 1.5, "line_mode": "SubjectOutline" }, "brush": { "radius": 40.0 } } }),
        );
        let unchanged = PresetFile::default().with_settings(model, &prunr_item_settings(model), &prunr_view(model, None).brush);
        assert!(unchanged.models.is_empty(), "nothing apart from Prunr's");
    }

    /// A model the preset has no entry for takes Prunr's settings, never
    /// another model's; a setting an entry leaves out takes Prunr's.
    #[test]
    fn what_a_preset_does_not_set_is_prunrs() {
        let preset = Preset::from(file(r#"{ "format_version": 2, "models": { "BiRefNetLite": { "item_settings": { "gamma": 2.5 } } } }"#));
        let birefnet = preset.resolve(SettingsModel::BiRefNetLite, None);
        assert_eq!(birefnet.item_settings, ItemSettings { gamma: 2.5, ..prunr_item_settings(SettingsModel::BiRefNetLite) });
        for other in [SettingsModel::U2net, SettingsModel::Silueta, SettingsModel::None, SettingsModel::Nomos2Upscale] {
            assert_eq!(preset.resolve(other, None), prunr_view(other, None), "{other:?}");
        }
    }

    /// No model files under its own key.
    #[test]
    fn no_model_has_its_own_entry() {
        let item = ItemSettings { line_mode: prunr_core::LineMode::EdgesOnly, ..prunr_item_settings(SettingsModel::None) };
        let saved = PresetFile::default().with_settings(SettingsModel::None, &item, &prunr_view(SettingsModel::None, None).brush);
        assert!(saved.models.contains_key(NO_MODEL_KEY));
        assert_eq!(Preset::from(saved).item_settings(SettingsModel::None).line_mode, prunr_core::LineMode::EdgesOnly);
    }

    /// A file an earlier Prunr saved whole loads as what differs from
    /// Prunr's, with the settings the model does not show dropped; a model
    /// this Prunr does not know is kept as it is.
    #[test]
    fn an_old_whole_file_is_trimmed_to_what_differs() {
        let whole = serde_json::to_value(ItemSettings { gamma: 1.5, sharpen: 0.7, ..ItemSettings::default() }).unwrap();
        let old = file(&serde_json::json!({
            "format_version": 2,
            "models": {
                "Silueta": { "item_settings": whole, "brush": serde_json::to_value(BrushSettings::default()).unwrap() },
                "FutureModel": { "item_settings": { "gamma": 9.0 } },
            }
        }).to_string());
        let trimmed = old.trimmed();
        assert_eq!(trimmed.models["Silueta"], Entry { item_settings: serde_json::from_str(r#"{ "gamma": 1.5 }"#).unwrap(), ..Entry::default() });
        assert_eq!(trimmed.models["FutureModel"].item_settings["gamma"], 9.0, "kept for a Prunr that knows it");
    }

    /// Saving into a preset sets the model in use and keeps the others'
    /// entries, and the SD part keeps the bundles of the other qualities.
    #[test]
    fn saving_into_a_preset_keeps_the_rest() {
        let existing = file(r#"{ "format_version": 2, "models": {
            "LaMaFp32": { "brush": { "inpaint_grow": 6.0 } },
            "SdV15InpaintFp16": { "sd": { "prompt": "old", "schedulers": { "Lcm": { "steps": 4, "guidance_scale": 1.0 } } } }
        } }"#);
        let sd = SettingsModel::SdInpaint;
        let brush = BrushSettings {
            sd_prompt: "new".into(),
            sd_scheduler: SdScheduler::DpmPlusPlus2MKarras,
            sd_steps: 30,
            sd_guidance_scale: 5.0,
            ..prunr_view(sd, None).brush
        };
        let saved = Preset::from(existing.clone().with_settings(sd, &prunr_item_settings(sd), &brush));
        assert_eq!(saved.file().models["LaMaFp32"], existing.models["LaMaFp32"], "another model's entry stays");
        let quality = saved.resolve(sd, Some(SdScheduler::DpmPlusPlus2MKarras)).brush;
        assert_eq!((quality.sd_prompt.as_str(), quality.sd_steps, quality.sd_guidance_scale), ("new", 30, 5.0));
        let fast = saved.resolve(sd, Some(SdScheduler::Lcm)).brush;
        assert_eq!((fast.sd_steps, fast.sd_guidance_scale), (4, 1.0), "the other quality's bundle stays");
        assert_eq!(saved.resolve(sd, None).brush.sd_scheduler, SdScheduler::DpmPlusPlus2MKarras, "saved with the quality in use");
    }

    /// A Stable Diffusion part resolves its quality's bundle, the
    /// factory's for a quality it has none for, and Prunr's when the entry
    /// has no SD part.
    #[test]
    fn the_sd_part_resolves_per_quality() {
        let sd = SettingsModel::SdInpaint;
        let preset = Preset::from(file(r#"{ "format_version": 2, "models": { "SdV15InpaintFp16": {
            "sd": { "active_scheduler": "DpmPlusPlus2MKarras", "schedulers": { "DpmPlusPlus2MKarras": { "steps": 20, "guidance_scale": 7.5 } } }
        } } }"#));
        let own = preset.resolve(sd, None).brush;
        assert_eq!((own.sd_scheduler, own.sd_steps, own.sd_guidance_scale), (SdScheduler::DpmPlusPlus2MKarras, 20, 7.5));
        let fast = preset.resolve(sd, Some(SdScheduler::Lcm)).brush;
        assert_eq!((fast.sd_steps, fast.sd_guidance_scale), (8, 1.5), "the factory bundle for the quality it lacks");
        let brush_only = Preset::from(file(r#"{ "format_version": 2, "models": { "SdV15InpaintFp16": { "brush": { "radius": 50.0 } } } }"#));
        assert_eq!(brush_only.resolve(sd, None).brush, BrushSettings { radius: 50.0, ..prunr_view(sd, None).brush });
    }

    /// RORem takes no prompt or quality: its seed and steps are brush
    /// settings, and it never gets an SD part.
    #[test]
    fn rorem_keeps_its_own_settings_and_no_sd_part() {
        let rorem = SettingsModel::RoremInpaint;
        let brush = BrushSettings { rorem_steps: 6, sd_seed: Some(7), sd_prompt: "ignored".into(), ..prunr_view(rorem, None).brush };
        let saved = PresetFile::default().with_settings(rorem, &prunr_item_settings(rorem), &brush);
        let entry = &saved.models["RoremFp16"];
        assert!(entry.sd.is_empty() && !entry.brush.contains_key("sd_prompt"));
        assert_eq!((entry.brush["rorem_steps"].as_u64(), entry.brush["sd_seed"].as_u64()), (Some(6), Some(7)));
    }

    /// The generative erasers start with Sharpen off (they draw their own
    /// detail); the others with it on.
    #[test]
    fn sd_erasers_start_with_sharpen_off() {
        for model in [SettingsModel::SdInpaint, SettingsModel::RoremInpaint] {
            assert_eq!(prunr_view(model, None).brush.inpaint_sharpen, 0.0, "{model:?}");
        }
        assert_eq!(prunr_view(SettingsModel::Inpaint, None).brush.inpaint_sharpen, 0.6);
    }

    #[test]
    fn sd_scheduler_bundle_default_for_each_quality() {
        assert_eq!(SdSchedulerBundle::default_for(SdScheduler::Lcm), SdSchedulerBundle { steps: 8, guidance_scale: 1.5 });
        assert_eq!(SdSchedulerBundle::default_for(SdScheduler::DpmPlusPlus2MKarras), SdSchedulerBundle { steps: 25, guidance_scale: 4.0 });
    }

    /// Old and newer files load: missing fields read as default, unknown
    /// ones are ignored, in every struct a preset file holds.
    #[test]
    fn a_preset_file_loads_with_fields_missing_or_unknown() {
        assert_eq!(file("{}"), PresetFile::default());
        assert_eq!(file(r#"{ "format_version": 2, "models": {}, "extra": "ignored" }"#), PresetFile::default());
        let entry: Entry = serde_json::from_str(r#"{ "made_up_field": 42 }"#).unwrap();
        assert_eq!(entry, Entry::default());
        let sd: SdPreset = serde_json::from_str(r#"{ "future_field": true }"#).unwrap();
        assert_eq!(sd, SdPreset::default());
        let bundle: SdSchedulerBundle = serde_json::from_str(r#"{ "tomorrows_field": 1.5 }"#).unwrap();
        assert_eq!(bundle, SdSchedulerBundle::default());
    }

    /// A preset saved before the upscale settings existed loads them as
    /// Prunr's.
    #[test]
    fn an_old_preset_without_upscale_keys_loads_prunrs() {
        let old = Preset::from(file(r#"{ "format_version": 2, "models": { "Nomos8kSchatL": { "item_settings": { "gamma": 1.5 } } } }"#).trimmed());
        assert_eq!(old.item_settings(SettingsModel::Nomos8kUpscale), prunr_item_settings(SettingsModel::Nomos8kUpscale), "gamma is not an upscaler setting");
    }

    /// A saved preset round-trips through its file: key by model name, the
    /// SD part's empty fields left out.
    #[test]
    fn a_saved_preset_round_trips_through_json() {
        let sd = SettingsModel::SdInpaint;
        let brush = BrushSettings { sd_seed: Some(3), ..prunr_view(sd, None).brush };
        let saved = PresetFile::default().with_settings(sd, &prunr_item_settings(sd), &brush);
        let json = serde_json::to_string(&saved).unwrap();
        assert!(json.contains("\"SdV15InpaintFp16\"") && !json.contains("use_taesd"), "{json}");
        assert_eq!(file(&json), saved);
    }
}
