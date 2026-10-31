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
use super::preset_fields::{helpers_of, HELPERS};
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

/// The helper model filed under `key`.
fn helper_of_key(key: &str) -> Option<ModelId> {
    HELPERS.into_iter().find(|&h| model_id_key(h) == key)
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
        Self::default().with_settings(model, &item_settings, &prunr_view(model, None).brush)
    }

    /// The file with `model`'s entry, and those of the helper models that
    /// run inside it, set from the current settings, `item` and `brush`:
    /// those of their settings that differ from Prunr's. The SD part keeps
    /// the bundles of the qualities not in use. A model with nothing apart
    /// from Prunr's has no entry.
    pub(crate) fn with_settings(mut self, model: SettingsModel, item: &ItemSettings, brush: &BrushSettings) -> Self {
        let key = preset_key(model);
        let mut sd = prunr_view_of(model).sd.as_ref().map(|prunr| {
            let kept = self.models.get(&key).map_or(Map::new(), |entry| entry.sd.clone());
            overlay(prunr.clone(), &kept)
        });
        if let Some(sd) = &mut sd {
            take_sd_from_brush(sd, brush);
        }
        self.set(key, own_differences(model, item, brush, sd.as_ref()));
        for helper in helpers_of(model, item.faces.model) {
            self.set(model_id_key(helper), helper_differences(helper, item, brush));
        }
        self.format_version = PRESET_FORMAT_VERSION;
        self
    }

    fn set(&mut self, key: String, entry: Entry) {
        if entry.is_empty() {
            self.models.remove(&key);
        } else {
            self.models.insert(key, entry);
        }
    }

    /// The file as it is kept: each model's entry trimmed to its settings
    /// that differ from Prunr's (a file an earlier Prunr saved whole, or
    /// one written by hand). A picked model's entry from before helper
    /// models had their own hands its line, face, depth and Magic Brush
    /// settings to them, unless they have an entry already. A model this
    /// Prunr does not know is kept as it is, for the Prunr that does.
    pub(crate) fn trimmed(self) -> Self {
        let mut kept = Self { format_version: self.format_version, models: HashMap::new() };
        for (key, entry) in &self.models {
            if let Some(helper) = helper_of_key(key) {
                let (item, brush) = helper_values(helper, entry);
                kept.set(key.clone(), helper_differences(helper, &item, &brush));
            } else if model_of_key(key).is_none() {
                kept.models.insert(key.clone(), entry.clone());
            }
        }
        for model in SettingsModel::ALL {
            let Some(entry) = self.models.get(&preset_key(model)) else { continue };
            let prunr = prunr_view_of(model);
            let item = overlay(prunr.item_settings, &entry.item_settings);
            let brush = overlay(prunr.brush.clone(), &entry.brush);
            let sd = prunr.sd.clone().map(|sd| overlay(sd, &entry.sd));
            kept.set(preset_key(model), own_differences(model, &item, &brush, sd.as_ref()));
            for helper in helpers_of(model, item.faces.model) {
                let key = model_id_key(helper);
                if !kept.models.contains_key(&key) {
                    kept.set(key, helper_differences(helper, &item, &brush));
                }
            }
        }
        kept
    }
}

/// The picked model's own settings that differ from Prunr's, by name.
fn own_differences(model: SettingsModel, item: &ItemSettings, brush: &BrushSettings, sd: Option<&SdPreset>) -> Entry {
    let fields = super::preset_fields::of(model);
    let prunr = prunr_view_of(model);
    Entry {
        item_settings: differing(item, &prunr.item_settings, &fields.item),
        brush: differing(brush, &prunr.brush, &fields.brush),
        sd: sd.zip(prunr.sd.as_ref()).map_or(Map::new(), |(sd, base)| differing(sd, base, SD_FIELDS)),
    }
}

/// A helper model's own settings that differ from Prunr's, by name.
fn helper_differences(helper: ModelId, item: &ItemSettings, brush: &BrushSettings) -> Entry {
    let fields = super::preset_fields::of_helper(helper);
    let (prunr_item, prunr_brush) = prunr_helper(helper);
    Entry {
        item_settings: differing(item, &prunr_item, &fields.item),
        brush: differing(brush, &prunr_brush, &fields.brush),
        sd: Map::new(),
    }
}

/// A setting by its name in a file: `faces.hair` inside `faces`.
fn field<'a>(value: &'a Value, name: &str) -> Option<&'a Value> {
    name.split('.').try_fold(value, |value, part| value.get(part))
}

fn set_field(value: &mut Value, name: &str, to: Value) {
    let mut at = value;
    let mut parts = name.split('.').peekable();
    while let Some(part) = parts.next() {
        let Some(group) = at.as_object_mut() else { return };
        if parts.peek().is_none() {
            group.insert(part.to_string(), to);
            return;
        }
        at = group.entry(part).or_insert_with(|| Value::Object(Map::new()));
    }
}

/// `value`'s `fields` that differ from `base`'s. A field written only
/// when set (an `Option`) is written as null when `value` leaves it out.
fn differing<T: Serialize>(value: &T, base: &T, fields: &[&str]) -> Map<String, Value> {
    let (value, base) = (serde_json::to_value(value).unwrap_or_default(), serde_json::to_value(base).unwrap_or_default());
    fields
        .iter()
        .filter_map(|&name| {
            let own = field(&value, name).cloned().unwrap_or(Value::Null);
            (own != field(&base, name).cloned().unwrap_or(Value::Null)).then(|| (name.to_string(), tidy(own)))
        })
        .collect()
}

/// `value` with each number written as people type it: settings are
/// `f32`, which JSON widens (0.8 becomes 0.800000011920929); the shortest
/// form that reads back as the same `f32` goes in the file instead.
fn tidy(value: Value) -> Value {
    match value {
        Value::Number(n) if n.is_f64() => n
            .as_f64()
            .and_then(|x| (x as f32).to_string().parse::<f64>().ok())
            .and_then(serde_json::Number::from_f64)
            .map_or(Value::Number(n), Value::Number),
        Value::Array(items) => Value::Array(items.into_iter().map(tidy).collect()),
        Value::Object(fields) => Value::Object(fields.into_iter().map(|(k, v)| (k, tidy(v))).collect()),
        other => other,
    }
}

/// `base` with `over`'s settings laid on it, by name. A preset whose
/// settings do not read leaves `base` as it is.
fn overlay<T: Serialize + serde::de::DeserializeOwned>(base: T, over: &Map<String, Value>) -> T {
    if over.is_empty() {
        return base;
    }
    let mut value = serde_json::to_value(&base).unwrap_or_default();
    for (name, to) in over {
        set_field(&mut value, name, to.clone());
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

/// `model`'s settings with `entries` (a preset's) laid over Prunr's: its
/// own entry, then that of each helper model that runs inside it, by the
/// face restorer it has chosen.
fn view_of(model: SettingsModel, entries: &HashMap<String, Entry>) -> ModelView {
    let prunr = prunr_entries();
    let layers = |key: &str| [prunr.get(key), entries.get(key)].into_iter().flatten().collect::<Vec<_>>();
    let own = layers(&preset_key(model));
    let mut item = own.iter().fold(ItemSettings::default(), |item, entry| overlay(item, &entry.item_settings));
    let mut brush = own.iter().fold(BrushSettings::default(), |brush, entry| overlay(brush, &entry.brush));
    let sd = own.first().is_some_and(|entry| !entry.sd.is_empty()).then(|| own.iter().fold(SdPreset::default(), |sd, entry| overlay(sd, &entry.sd)));
    for helper in helpers_of(model, item.faces.model) {
        for entry in layers(&model_id_key(helper)) {
            item = overlay(item, &entry.item_settings);
            brush = overlay(brush, &entry.brush);
        }
    }
    ModelView { item_settings: item, brush, sd }
}

#[derive(Debug, Deserialize)]
struct PrunrFile {
    models: HashMap<String, Entry>,
}

/// Prunr's own preset as its built-in file lists it.
fn prunr_entries() -> &'static HashMap<String, Entry> {
    static FILE: std::sync::OnceLock<HashMap<String, Entry>> = std::sync::OnceLock::new();
    // A test reads the file and resolves every model from it.
    FILE.get_or_init(|| serde_json::from_str::<PrunrFile>(include_str!("../../presets/Prunr.json")).expect("Prunr.json reads").models)
}

/// Prunr's settings for every picked model, in `SettingsModel::ALL`
/// order, resolved once.
fn prunr_models() -> &'static [ModelView] {
    static MODELS: std::sync::OnceLock<Vec<ModelView>> = std::sync::OnceLock::new();
    MODELS.get_or_init(|| SettingsModel::ALL.into_iter().map(|model| view_of(model, &HashMap::new())).collect())
}

fn model_index(model: SettingsModel) -> usize {
    // Every model is in ALL: `SettingsModel::ALL` lists them all.
    SettingsModel::ALL.iter().position(|&m| m == model).unwrap_or(0)
}

fn prunr_view_of(model: SettingsModel) -> &'static ModelView {
    &prunr_models()[model_index(model)]
}

/// Prunr's settings for a helper model, over the code's defaults.
fn prunr_helper(helper: ModelId) -> (ItemSettings, BrushSettings) {
    helper_values(helper, &Entry::default())
}

/// A helper model's settings from `entry` over Prunr's.
fn helper_values(helper: ModelId, entry: &Entry) -> (ItemSettings, BrushSettings) {
    let empty = Entry::default();
    let prunr = prunr_entries().get(&model_id_key(helper)).unwrap_or(&empty);
    let item = overlay(overlay(ItemSettings::default(), &prunr.item_settings), &entry.item_settings);
    let brush = overlay(overlay(BrushSettings::default(), &prunr.brush), &entry.brush);
    (item, brush)
}

/// Prunr's settings for `model`, resolved for `scheduler` (`None`: its
/// own quality).
pub(crate) fn prunr_view(model: SettingsModel, scheduler: Option<SdScheduler>) -> ResolvedView {
    prunr_view_of(model).resolve(scheduler)
}

/// Prunr's image settings for `model`, without cloning its brush.
pub(crate) fn prunr_item_settings(model: SettingsModel) -> ItemSettings {
    prunr_view_of(model).item_settings
}

/// A preset as the app holds it: its file, and each picked model's
/// settings resolved once, as the toolbar reads them every frame.
#[derive(Debug, Clone, Deserialize)]
#[serde(from = "PresetFile")]
pub(crate) struct Preset {
    file: PresetFile,
    /// Per model in `SettingsModel::ALL` order.
    views: Vec<ModelView>,
}

impl From<PresetFile> for Preset {
    fn from(file: PresetFile) -> Self {
        let views = SettingsModel::ALL.into_iter().map(|model| view_of(model, &file.models)).collect();
        Self { file, views }
    }
}

impl Preset {
    pub(crate) fn file(&self) -> &PresetFile {
        &self.file
    }

    /// The preset's settings for `model`, Prunr's where it has none.
    pub(crate) fn resolve(&self, model: SettingsModel, scheduler: Option<SdScheduler>) -> ResolvedView {
        self.views[model_index(model)].resolve(scheduler)
    }

    /// `resolve`'s image settings, without cloning a brush.
    pub(crate) fn item_settings(&self, model: SettingsModel) -> ItemSettings {
        self.views[model_index(model)].item_settings
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

    /// Prunr's file has an entry for every picked model and every helper
    /// model, listing exactly the settings that model keeps, each one
    /// readable: a new model or a new setting has to be given its default
    /// there.
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
            for (name, to) in over {
                set_field(&mut value, name, to.clone());
            }
            serde_json::from_value(value)
        }
        let file: PrunrFile = serde_json::from_str(include_str!("../../presets/Prunr.json")).expect("Prunr.json reads");
        let picked = SettingsModel::ALL.into_iter().map(|m| (preset_key(m), super::super::preset_fields::of(m)));
        let helpers = HELPERS.into_iter().map(|h| (model_id_key(h), super::super::preset_fields::of_helper(h)));
        for (key, fields) in picked.chain(helpers) {
            let entry = file.models.get(&key).unwrap_or_else(|| panic!("{key} has no entry in Prunr.json"));
            assert_eq!(keys(&entry.item_settings), sorted(fields.item), "{key} image settings");
            assert_eq!(keys(&entry.brush), sorted(fields.brush), "{key} brush settings");
            assert_eq!(!entry.sd.is_empty(), fields.sd, "{key} SD part");
            reads(ItemSettings::default(), &entry.item_settings).unwrap_or_else(|e| panic!("{key} image settings: {e}"));
            reads(BrushSettings::default(), &entry.brush).unwrap_or_else(|e| panic!("{key} brush settings: {e}"));
            reads(SdPreset::default(), &entry.sd).unwrap_or_else(|e| panic!("{key} SD part: {e}"));
        }
        assert_eq!(file.models.len(), SettingsModel::ALL.len() + HELPERS.len(), "an entry for a model the app does not run");
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

    /// A helper model keeps its settings in its own entry, and they apply
    /// wherever it runs; each face restorer has its own.
    #[test]
    fn a_helper_keeps_its_settings_wherever_it_runs() {
        use super::super::item_settings::FaceModel;
        let birefnet = SettingsModel::BiRefNetLite;
        let mut item = prunr_item_settings(birefnet);
        item.line_strength = 0.75;
        item.faces.strength_pct = 70;
        let saved = PresetFile::default().with_settings(birefnet, &item, &prunr_view(birefnet, None).brush);
        assert!(!saved.models.contains_key("BiRefNetLite"), "nothing of BiRefNet's own changed");
        assert_eq!(saved.models["Teed"].item_settings["line_strength"], 0.75);
        assert_eq!(saved.models["FaceGfpgan"].item_settings["faces.strength_pct"], 70);
        let preset = Preset::from(saved);
        assert_eq!(preset.item_settings(SettingsModel::None).line_strength, 0.75, "the line model runs on No model too");
        assert_eq!(preset.item_settings(SettingsModel::Nomos2Upscale).faces.strength_pct, 70, "GFPGAN after an upscaler");
        let restoreformer = Preset::from(file(r#"{ "format_version": 2, "models": { "Nomos8kSchatL": { "item_settings": { "faces.model": "RestoreFormerPp" } } } }"#));
        let faces = restoreformer.item_settings(SettingsModel::Nomos8kUpscale).faces;
        assert_eq!((faces.model, faces.strength_pct), (FaceModel::RestoreFormerPp, 50), "RestoreFormer++'s own strength, not GFPGAN's");
    }

    /// A file from before helper models had entries keeps its line and face
    /// settings: they move from the picked model's entry to the helpers'.
    #[test]
    fn an_old_entry_hands_its_helper_settings_over() {
        let old = file(r#"{ "format_version": 2, "models": { "Silueta": { "item_settings": {
            "gamma": 1.5, "line_strength": 0.8, "faces": { "on": true, "model": "Gfpgan", "strength_pct": 70, "largest_only": false, "hair": false }
        } } } }"#).trimmed();
        assert_eq!(old.models["Silueta"].item_settings, serde_json::from_str::<Map<String, Value>>(r#"{ "gamma": 1.5, "faces.on": true }"#).unwrap());
        assert_eq!(old.models["Teed"].item_settings["line_strength"], 0.8);
        assert_eq!(old.models["FaceGfpgan"].item_settings["faces.strength_pct"], 70);
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

    /// A saved number is written as typed, not as JSON widens an `f32`.
    #[test]
    fn a_saved_number_is_written_as_typed() {
        let model = SettingsModel::Silueta;
        let item = ItemSettings { gamma: 0.8, feather: 2.3, ..prunr_item_settings(model) };
        let saved = PresetFile::default().with_settings(model, &item, &prunr_view(model, None).brush);
        let json = serde_json::to_string(&saved.models["Silueta"]).unwrap();
        assert!(json.contains("\"gamma\":0.8") && json.contains("\"feather\":2.3"), "{json}");
        assert_eq!(Preset::from(saved).item_settings(model).gamma, 0.8, "reads back as the same value");
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
