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
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use prunr_models::ModelId;

use super::brush_state::{BrushSettings, SdScheduler};
use super::item_settings::{FaceModel, FaceSettings, ItemSettings};
use super::preset_changes::{Changes, Part};
use super::preset_fields::{brush_entries_of, entries_of, helpers_of, may_run, Helper, HELPERS};
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

/// Whose settings an entry holds.
#[derive(Clone, Copy)]
enum Owner {
    Picked(SettingsModel),
    Helper(&'static Helper),
    /// A model this Prunr does not know: a newer Prunr's.
    Unknown,
}

/// Every key this Prunr knows and its owner: the picked models in their
/// menu order, then the helpers.
fn known_keys() -> &'static [(String, Owner)] {
    static KEYS: OnceLock<Vec<(String, Owner)>> = OnceLock::new();
    KEYS.get_or_init(|| {
        let picked = SettingsModel::ALL.into_iter().map(|model| (preset_key(model), Owner::Picked(model)));
        let helpers = HELPERS.iter().map(|helper| (model_id_key(helper.id), Owner::Helper(helper)));
        picked.chain(helpers).collect()
    })
}

fn owner(key: &str) -> Owner {
    known_keys().iter().find(|(known, _)| known == key).map_or(Owner::Unknown, |&(_, owner)| owner)
}

/// Where `key`'s model sits in a list of models: the picked models in
/// their menu order, then the helpers, then models this Prunr does not know.
fn key_order(key: &str) -> usize {
    known_keys().iter().position(|(known, _)| known == key).unwrap_or(usize::MAX)
}

/// `key`'s model as a person reads it.
pub(crate) fn entry_name(key: &str) -> &str {
    match owner(key) {
        Owner::Picked(model) => super::views::model_name(model),
        Owner::Helper(helper) => helper.name,
        Owner::Unknown => key,
    }
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
    pub(crate) fn is_empty(&self) -> bool {
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
    /// A preset of image settings alone (a v1 file, which predates models
    /// in presets): they belong to every model that shows them.
    pub(crate) fn of_item_settings(item_settings: ItemSettings) -> Self {
        SettingsModel::ALL.into_iter().fold(Self::default(), |file, model| {
            file.with_settings(model, &item_settings, prunr_model(model).brush(None))
        })
    }

    /// The file with `model`'s entry, and those of the helper models that
    /// run inside it, set from the current settings, `item` and `brush`:
    /// those of their settings that differ from Prunr's. The SD part keeps
    /// the bundles of the qualities not in use. A model with nothing apart
    /// from Prunr's has no entry.
    pub(crate) fn with_settings(self, model: SettingsModel, item: &ItemSettings, brush: &BrushSettings) -> Self {
        let entries: Vec<_> = entries_of(model, item.faces.model)
            .into_iter()
            .map(|(key, _)| {
                let entry = entry_of(&key, &self.models, *item, brush.clone());
                (key, entry)
            })
            .collect();
        self.with_entries(entries)
    }

    /// The file with `entries` set, each replacing its model's; an empty
    /// one removes it.
    pub(crate) fn with_entries(mut self, entries: impl IntoIterator<Item = (String, Entry)>) -> Self {
        for (key, entry) in entries {
            self.set(key, entry);
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
            match owner(key) {
                Owner::Helper(helper) => {
                    let (item, brush) = helper_values(helper, Some(entry));
                    kept.set(key.clone(), helper_differences(helper, &item, &brush));
                }
                Owner::Unknown => {
                    kept.models.insert(key.clone(), entry.clone());
                }
                Owner::Picked(_) => {}
            }
        }
        for model in SettingsModel::ALL {
            let Some(entry) = self.models.get(&preset_key(model)) else { continue };
            let prunr = prunr_model(model);
            // Over Prunr's settings for the face restorer the entry picks,
            // so an untouched restorer setting reads as Prunr's.
            let face = overlay(prunr.item_settings, &entry.item_settings).faces.model;
            let item = overlay(view_of(model, &HashMap::new(), Some(face)).item_settings, &entry.item_settings);
            let brush = overlay(prunr.brush.clone(), &entry.brush);
            let sd = prunr.sd.clone().map(|sd| overlay(sd, &entry.sd));
            kept.set(preset_key(model), own_differences(model, &item, &brush, sd.as_ref()));
            for helper in helpers_of(model, item.faces.model) {
                let key = model_id_key(helper.id);
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
    let prunr = prunr_model(model);
    Entry {
        item_settings: differing(item, &prunr.item_settings, &fields.item),
        brush: differing(brush, &prunr.brush, &fields.brush),
        sd: sd.zip(prunr.sd.as_ref()).map_or(Map::new(), |(sd, base)| differing(sd, base, SD_FIELDS)),
    }
}

/// A helper model's own settings that differ from Prunr's, by name.
fn helper_differences(helper: &Helper, item: &ItemSettings, brush: &BrushSettings) -> Entry {
    let (prunr_item, prunr_brush) = helper_values(helper, None);
    Entry {
        item_settings: differing(item, &prunr_item, helper.item),
        brush: differing(brush, &prunr_brush, helper.brush),
        sd: Map::new(),
    }
}

/// `key`'s entry for the settings `item` and `brush`: those that differ
/// from Prunr's, the SD part's other qualities kept from `entries`. A
/// model this Prunr does not know keeps its entry in `entries`.
fn entry_of(key: &str, entries: &HashMap<String, Entry>, item: ItemSettings, brush: BrushSettings) -> Entry {
    match owner(key) {
        Owner::Picked(model) => {
            let sd = own_sd(model, entries).map(|mut sd| {
                take_sd_from_brush(&mut sd, &brush);
                sd
            });
            own_differences(model, &item, &brush, sd.as_ref())
        }
        Owner::Helper(helper) => helper_differences(helper, &item, &brush),
        Owner::Unknown => entries.get(key).cloned().unwrap_or_default(),
    }
}

/// `key`'s settings as `entries` set them over Prunr's.
fn values_of(key: &str, entries: &HashMap<String, Entry>) -> (ItemSettings, BrushSettings) {
    match owner(key) {
        Owner::Picked(model) => {
            let view = view_of(model, entries, None);
            (view.item_settings, view.brush(None).clone())
        }
        Owner::Helper(helper) => helper_values(helper, entries.get(key)),
        Owner::Unknown => Default::default(),
    }
}

/// An image's settings as a save reads them, and what was changed since
/// its preset was applied.
pub(crate) struct Shown<'a> {
    pub model: SettingsModel,
    pub item: &'a ItemSettings,
    pub brush: &'a BrushSettings,
    pub item_changes: &'a Changes<ItemSettings>,
    pub brush_changes: &'a Changes<BrushSettings>,
}

/// The entries a save of the shown image writes, by key, in menu order:
/// those of the model in use and its helpers as the image shows them,
/// then every other model's that `started` (the preset the image started
/// from) has an entry for or that the user changed, as `started` sets
/// them with those changes. Such an entry may be empty: that model is
/// Prunr's again.
pub(crate) fn draft(started: &Preset, shown: &Shown) -> Vec<(String, Entry)> {
    let entries = &started.file.models;
    let in_use = entries_of(shown.model, shown.item.faces.model);
    let item_changes = shown.item_changes.noted(shown.item, &in_use);
    let brush_changes = shown.brush_changes.noted(shown.brush, &in_use);
    let changed = |key: &str| item_changes.on(key).is_some() || brush_changes.on(key).is_some();
    let mut others: Vec<&str> = entries.keys().map(String::as_str).chain(item_changes.keys()).chain(brush_changes.keys())
        .filter(|key| !in_use.iter().any(|(used, _)| used == key))
        .collect();
    others.sort_by_key(|key| (key_order(key), *key));
    others.dedup();
    let used = in_use.iter().map(|(key, _)| {
        let entry = entry_of(key, entries, *shown.item, shown.brush.clone());
        (key.clone(), entry)
    });
    let unchanged = Map::new();
    let rest = others.into_iter().map(|key| {
        let (item, brush) = values_of(key, entries);
        let item = overlay(item, item_changes.on(key).unwrap_or(&unchanged));
        let brush = overlay(brush, brush_changes.on(key).unwrap_or(&unchanged));
        (key.to_string(), entry_of(key, entries, item, brush))
    });
    used.filter(|(key, entry)| !entry.is_empty() || entries.contains_key(key) || changed(key)).chain(rest).collect()
}

/// The brush after a switch to `model`: each setting `changes` has not
/// touched takes `preset`'s value for `model`, and the eraser's fill
/// tuning is its own: what was changed on it, else `preset`'s.
pub(crate) fn take_untouched_brush(preset: &Preset, model: SettingsModel, brush: BrushSettings, changes: &Changes<BrushSettings>) -> BrushSettings {
    let tuning = super::preset_fields::fill_tuning_of(model);
    let own = changes.on(&preset_key(model));
    let scheduler = own.and_then(|own| own.get("sd_scheduler")).and_then(|quality| serde_json::from_value(quality.clone()).ok());
    let target = serde_json::to_value(preset.brush(model, scheduler)).unwrap_or_default();
    let names: Vec<&str> = brush_entries_of(model).iter().flat_map(|(_, fields)| BrushSettings::fields(fields)).collect();
    let mut values = pick(&target, &names);
    values.retain(|name, _| tuning.contains(&name.as_str()) || !changes.touched(name));
    for (name, value) in own.into_iter().flatten().filter(|(name, _)| tuning.contains(&name.as_str())) {
        values.insert(name.clone(), value.clone());
    }
    let mut brush = overlay(brush, &values);
    brush.fit_sd_to_quality();
    brush
}

/// `to` with each of `values` that `changes` has not touched.
pub(crate) fn take<T: Part>(to: T, values: &Map<String, Value>, changes: &Changes<T>) -> T {
    let untouched: Map<String, Value> = values.iter().filter(|(name, _)| !changes.touched(name)).map(|(name, value)| (name.clone(), value.clone())).collect();
    overlay(to, &untouched)
}

/// `value`'s settings `names`, by name; one it leaves out (an unset
/// `Option`) as null.
fn pick(value: &Value, names: &[&str]) -> Map<String, Value> {
    names.iter().map(|&name| (name.to_string(), field(value, name).cloned().unwrap_or(Value::Null))).collect()
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
pub(crate) fn differing<T: Serialize>(value: &T, base: &T, fields: &[&str]) -> Map<String, Value> {
    differing_values(&serde_json::to_value(value).unwrap_or_default(), &serde_json::to_value(base).unwrap_or_default(), fields)
}

/// `differing` on settings already turned into values.
pub(crate) fn differing_values(value: &Value, base: &Value, fields: &[&str]) -> Map<String, Value> {
    fields
        .iter()
        .filter_map(|&name| {
            let own = field(value, name).cloned().unwrap_or(Value::Null);
            (own != field(base, name).cloned().unwrap_or(Value::Null)).then(|| (name.to_string(), tidy(own)))
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

/// `base` with `over`'s settings laid on it, by name. A setting that does
/// not read (a newer Prunr's value, a typo) is left out, not the others.
fn overlay<T: Serialize + serde::de::DeserializeOwned>(base: T, over: &Map<String, Value>) -> T {
    if over.is_empty() {
        return base;
    }
    let start = serde_json::to_value(&base).unwrap_or_default();
    let mut value = start.clone();
    for (name, to) in over {
        set_field(&mut value, name, to.clone());
    }
    if let Ok(laid) = serde_json::from_value(value) {
        return laid;
    }
    let mut value = start;
    for (name, to) in over {
        let mut tried = value.clone();
        set_field(&mut tried, name, to.clone());
        match serde_json::from_value::<T>(tried.clone()) {
            Ok(_) => value = tried,
            Err(err) => tracing::warn!(setting = %name, %err, "a preset setting that does not read was left out"),
        }
    }
    serde_json::from_value(value).unwrap_or(base)
}

/// One model's settings, resolved.
#[derive(Debug, Clone, PartialEq)]
struct ModelView {
    item_settings: ItemSettings,
    /// Its brush before a quality's steps and guidance are laid on it.
    brush: BrushSettings,
    /// The Stable Diffusion eraser's SD part.
    sd: Option<SdPreset>,
    /// With an SD part, the brush for each quality in `SdScheduler::ALL`
    /// order, as the eraser's panels read it every frame.
    by_quality: Vec<BrushSettings>,
}

impl ModelView {
    fn new(item_settings: ItemSettings, brush: BrushSettings, sd: Option<SdPreset>) -> Self {
        let by_quality = sd.as_ref().map_or_else(Vec::new, |sd| {
            SdScheduler::ALL.iter().map(|&quality| {
                let mut brush = brush.clone();
                apply_sd_to_brush(&mut brush, sd, Some(quality));
                brush
            }).collect()
        });
        Self { item_settings, brush, sd, by_quality }
    }

    /// The brush for `scheduler`; `None` uses the SD part's own quality.
    /// A quality picked in the session does not rewrite the preset.
    fn brush(&self, scheduler: Option<SdScheduler>) -> &BrushSettings {
        let Some(sd) = &self.sd else { return &self.brush };
        let quality = scheduler.unwrap_or(sd.active_scheduler);
        SdScheduler::ALL.iter().position(|&q| q == quality).and_then(|at| self.by_quality.get(at)).unwrap_or(&self.brush)
    }

    fn resolve(&self, scheduler: Option<SdScheduler>) -> ResolvedView {
        ResolvedView { item_settings: self.item_settings, brush: self.brush(scheduler).clone() }
    }
}

/// `key`'s entries: Prunr's, then `entries`' (a preset's).
fn layers<'a>(key: &str, entries: &'a HashMap<String, Entry>) -> impl Iterator<Item = &'a Entry> {
    [prunr_entries().get(key), entries.get(key)].into_iter().flatten()
}

/// `model`'s SD part with `entries` laid over Prunr's; `None` for a model
/// without one.
fn own_sd(model: SettingsModel, entries: &HashMap<String, Entry>) -> Option<SdPreset> {
    let own: Vec<&Entry> = layers(&preset_key(model), entries).collect();
    own.first().is_some_and(|entry| !entry.sd.is_empty()).then(|| own.iter().fold(SdPreset::default(), |sd, entry| overlay(sd, &entry.sd)))
}

/// `model`'s settings with `entries` laid over Prunr's: its own entry,
/// then that of each helper model that runs inside it with face restorer
/// `face`, or the one its entry chooses for `None`.
fn view_of(model: SettingsModel, entries: &HashMap<String, Entry>, face: Option<FaceModel>) -> ModelView {
    let key = preset_key(model);
    let mut item = layers(&key, entries).fold(ItemSettings::default(), |item, entry| overlay(item, &entry.item_settings));
    let mut brush = layers(&key, entries).fold(BrushSettings::default(), |brush, entry| overlay(brush, &entry.brush));
    for helper in helpers_of(model, face.unwrap_or(item.faces.model)) {
        for entry in layers(&model_id_key(helper.id), entries) {
            item = overlay(item, &entry.item_settings);
            brush = overlay(brush, &entry.brush);
        }
    }
    ModelView::new(item, brush, own_sd(model, entries))
}

/// Prunr's own preset's built-in file.
pub(crate) fn prunr_file_json() -> &'static str {
    include_str!("../../presets/Prunr.json")
}

/// Prunr's own preset as its built-in file lists it.
fn prunr_entries() -> &'static HashMap<String, Entry> {
    static FILE: OnceLock<HashMap<String, Entry>> = OnceLock::new();
    // A test reads the file and resolves every model from it.
    FILE.get_or_init(|| serde_json::from_str::<PresetFile>(prunr_file_json()).expect("Prunr.json reads").models)
}

/// Prunr's settings for every picked model, in `SettingsModel::ALL`
/// order, resolved once.
fn prunr_views() -> &'static [ModelView] {
    static VIEWS: OnceLock<Vec<ModelView>> = OnceLock::new();
    VIEWS.get_or_init(|| SettingsModel::ALL.into_iter().map(|model| view_of(model, &HashMap::new(), None)).collect())
}

fn model_index(model: SettingsModel) -> usize {
    SettingsModel::ALL.iter().position(|&m| m == model).unwrap_or(0)
}

fn prunr_model(model: SettingsModel) -> &'static ModelView {
    &prunr_views()[model_index(model)]
}

/// A helper model's settings from `entry` over Prunr's.
fn helper_values(helper: &Helper, entry: Option<&Entry>) -> (ItemSettings, BrushSettings) {
    let key = model_id_key(helper.id);
    [prunr_entries().get(&key), entry].into_iter().flatten().fold((ItemSettings::default(), BrushSettings::default()), |(item, brush), entry| {
        (overlay(item, &entry.item_settings), overlay(brush, &entry.brush))
    })
}

/// Prunr's settings for `model`, resolved for `scheduler` (`None`: its
/// own quality).
#[cfg(test)]
pub(crate) fn prunr_view(model: SettingsModel, scheduler: Option<SdScheduler>) -> ResolvedView {
    prunr_model(model).resolve(scheduler)
}

/// Prunr's image settings for `model`.
#[cfg(test)]
pub(crate) fn prunr_item_settings(model: SettingsModel) -> ItemSettings {
    prunr_model(model).item_settings
}

/// The settings a command-line run starts from for `model`: preset
/// `name`'s (one in the presets folder, or a preset file), Prunr's for
/// `None` or "Prunr"; with `faces`, that face restorer runs with its own
/// settings from the preset. Why not, when the preset cannot be read.
pub fn command_line_settings(name: Option<&str>, model: SettingsModel, faces: Option<FaceModel>) -> Result<(ItemSettings, BrushSettings), String> {
    let found = name.filter(|name| !super::presets_fs::is_prunr(name)).map(super::presets_fs::find_preset).transpose()?;
    let preset = found.as_ref().unwrap_or_else(|| Preset::prunr());
    let view = preset.resolve(model, None);
    let mut item = view.item_settings;
    if let Some(face) = faces {
        item.faces = FaceSettings { on: true, ..preset.faces(model, face) };
    }
    Ok((item, view.brush))
}

/// Preset `name` in `presets`; Prunr's for "Prunr" or one not there.
pub(crate) fn named<'a>(presets: &'a HashMap<String, Preset>, name: &str) -> &'a Preset {
    (!super::presets_fs::is_prunr(name)).then(|| presets.get(name)).flatten().unwrap_or_else(|| Preset::prunr())
}

/// A preset as the app holds it: its file, and each picked model's
/// settings resolved once, as the toolbar reads them every frame.
#[derive(Debug, Clone, Deserialize)]
#[serde(from = "PresetFile")]
pub(crate) struct Preset {
    /// Shared with the copies of it, so a read of the folder can tell a
    /// preset it was asked over from one saved since.
    file: Arc<PresetFile>,
    /// Per model in `SettingsModel::ALL` order.
    views: Vec<ModelView>,
    /// Per helper model in `HELPERS` order, its image settings.
    helpers: Vec<ItemSettings>,
    /// The models it has settings for, as the preset list names them.
    covers: String,
}

impl From<PresetFile> for Preset {
    fn from(file: PresetFile) -> Self {
        // A model whose settings the file leaves all to Prunr shares Prunr's.
        let views = SettingsModel::ALL.into_iter().zip(prunr_views()).map(|(model, prunr)| {
            let mut keys = std::iter::once(preset_key(model)).chain(may_run(model).map(|helper| model_id_key(helper.id)));
            if keys.any(|key| file.models.contains_key(&key)) { view_of(model, &file.models, None) } else { prunr.clone() }
        }).collect();
        let helpers = HELPERS.iter().map(|helper| helper_values(helper, file.models.get(&model_id_key(helper.id))).0).collect();
        let mut keys: Vec<&str> = file.models.keys().map(String::as_str).collect();
        keys.sort_by_key(|key| (key_order(key), *key));
        let covers = keys.into_iter().map(entry_name).collect::<Vec<_>>().join(" \u{b7} ");
        Self { file: Arc::new(file), views, helpers, covers }
    }
}

impl Preset {
    /// Prunr's own: its file is empty, so every model reads Prunr's.
    pub(crate) fn prunr() -> &'static Preset {
        static PRUNR: OnceLock<Preset> = OnceLock::new();
        PRUNR.get_or_init(|| Preset::from(PresetFile::default()))
    }

    pub(crate) fn file(&self) -> &PresetFile {
        &self.file
    }

    pub(crate) fn shared_file(&self) -> Arc<PresetFile> {
        Arc::clone(&self.file)
    }

    /// The models it has settings for: "BiRefNet · Lines".
    pub(crate) fn covers(&self) -> &str {
        &self.covers
    }

    fn view(&self, model: SettingsModel) -> &ModelView {
        &self.views[model_index(model)]
    }

    /// The preset's settings for `model`, Prunr's where it has none.
    pub(crate) fn resolve(&self, model: SettingsModel, scheduler: Option<SdScheduler>) -> ResolvedView {
        self.view(model).resolve(scheduler)
    }

    pub(crate) fn item_settings(&self, model: SettingsModel) -> ItemSettings {
        self.view(model).item_settings
    }

    /// `resolve`'s brush, without a clone.
    pub(crate) fn brush(&self, model: SettingsModel, scheduler: Option<SdScheduler>) -> &BrushSettings {
        self.view(model).brush(scheduler)
    }

    /// Helper model `id`'s image settings, wherever it runs.
    fn helper(&self, id: ModelId) -> ItemSettings {
        HELPERS.iter().position(|helper| helper.id == id).and_then(|at| self.helpers.get(at)).copied().unwrap_or_default()
    }

    /// The Faces settings for `model` with restorer `face`: whether faces
    /// are on from the model's entry, the rest the restorer's.
    pub(crate) fn faces(&self, model: SettingsModel, face: FaceModel) -> FaceSettings {
        FaceSettings { on: self.item_settings(model).faces.on, model: face, ..self.helper(face.id()).faces }
    }

    /// The face restorer a switch to `model` leaves an image on: the one
    /// it chose, or this preset's.
    pub(crate) fn face_after(&self, model: SettingsModel, item: &ItemSettings, changes: &Changes<ItemSettings>) -> FaceModel {
        if changes.touched("faces.model") { item.faces.model } else { self.item_settings(model).faces.model }
    }

    /// What a switch to `model` gives an image with settings `item` and
    /// `changes`: `switch_values` for the face restorer it is left on.
    pub(crate) fn values_after_switch(&self, model: SettingsModel, item: &ItemSettings, changes: &Changes<ItemSettings>) -> Map<String, Value> {
        self.switch_values(model, self.face_after(model, item, changes))
    }

    /// What a switch to `model` gives each image setting it may take, by
    /// name: the model's own, and those of the helpers it runs with face
    /// restorer `face`. The same for every image on this preset.
    pub(crate) fn switch_values(&self, model: SettingsModel, face: FaceModel) -> Map<String, Value> {
        let own = serde_json::to_value(self.item_settings(model)).unwrap_or_default();
        let mut values = pick(&own, &super::preset_fields::of(model).item);
        for helper in helpers_of(model, face) {
            let theirs = serde_json::to_value(self.helper(helper.id)).unwrap_or_default();
            values.extend(pick(&theirs, helper.item));
        }
        values
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
        let file: PresetFile = serde_json::from_str(prunr_file_json()).expect("Prunr.json reads");
        let picked = SettingsModel::ALL.into_iter().map(|m| (preset_key(m), super::super::preset_fields::of(m)));
        let helpers = HELPERS.iter().map(|h| (model_id_key(h.id), h.fields()));
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

    /// A switch keeps what was changed and takes the rest of the new
    /// model's settings from the preset, Prunr's where it has none.
    #[test]
    fn a_switch_keeps_the_changes_and_takes_the_rest_from_the_preset() {
        use super::super::preset_fields::entries_of;
        let preset = Preset::from(file(r#"{ "format_version": 2, "models": {
            "BiRefNetLite": { "item_settings": { "gamma": 1.2, "feather": 2.0 } },
            "Silueta": { "item_settings": { "gamma": 0.8 } }
        } }"#));
        let (birefnet, silueta) = (SettingsModel::BiRefNetLite, SettingsModel::Silueta);
        let mut item = preset.item_settings(birefnet);
        let mut changes = Changes::new(item);
        item.contrast = 1.3;
        changes.note(&item, &entries_of(birefnet, item.faces.model));
        let item = take(item, &preset.values_after_switch(silueta, &item, &changes), &changes);
        assert_eq!(item.gamma, 0.8, "Silueta's own from the preset");
        assert_eq!(item.feather, prunr_item_settings(silueta).feather, "Prunr's where the preset has none for Silueta");
        assert_eq!(item.contrast, 1.3, "the change stays");
    }

    /// Picking the other face restorer takes its own settings from the
    /// preset; what was changed stays.
    #[test]
    fn another_face_restorer_takes_its_own_settings() {
        use super::super::preset_fields::entries_of;
        let preset = Preset::from(file(r#"{ "format_version": 2, "models": {
            "FaceRestoreFormerPp": { "item_settings": { "faces.strength_pct": 35 } }
        } }"#));
        let model = SettingsModel::BiRefNetLite;
        let mut item = preset.item_settings(model);
        let mut changes = Changes::new(item);
        let hair = !item.faces.hair;
        item.faces.hair = hair;
        item.faces.model = FaceModel::RestoreFormerPp;
        changes.note(&item, &entries_of(model, FaceModel::Gfpgan));
        let item = take(item, &preset.values_after_switch(model, &item, &changes), &changes);
        assert_eq!((item.faces.model, item.faces.strength_pct), (FaceModel::RestoreFormerPp, 35));
        assert_eq!(item.faces.hair, hair, "a change stays");
    }

    fn shown_save(started: Option<&Preset>, model: SettingsModel, item: &ItemSettings, item_changes: &Changes<ItemSettings>) -> Vec<(String, Entry)> {
        let brush = prunr_view(model, None).brush;
        let brush_changes = Changes::new(brush.clone());
        draft(started.unwrap_or_else(|| Preset::prunr()), &Shown { model, item, brush: &brush, item_changes, brush_changes: &brush_changes })
    }

    /// A save writes the model in use as the image shows it, every model
    /// left before with the changes made on it, and the started preset's
    /// other models as it has them.
    #[test]
    fn a_save_writes_each_change_under_its_own_model() {
        use super::super::preset_fields::entries_of;
        let (birefnet, silueta, nomos2) = (SettingsModel::BiRefNetLite, SettingsModel::Silueta, SettingsModel::Nomos2Upscale);
        let sharper = ItemSettings { sharpen: 0.4, ..prunr_item_settings(nomos2) };
        let started = Preset::from(PresetFile::default().with_settings(nomos2, &sharper, &prunr_view(nomos2, None).brush));
        let mut item = started.item_settings(birefnet);
        let mut changes = Changes::new(item);
        item.gamma = 1.4;
        changes.note(&item, &entries_of(birefnet, item.faces.model));
        let mut item = take(item, &started.values_after_switch(silueta, &item, &changes), &changes);
        changes.rebase(&item);
        item.contrast = 1.3;

        let saved = shown_save(Some(&started), silueta, &item, &changes);
        let keys: Vec<&str> = saved.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(keys, [preset_key(silueta), preset_key(birefnet), preset_key(nomos2)], "the model in use first, then menu order");
        let own = |at: usize| saved[at].1.item_settings.keys().map(String::as_str).collect::<Vec<_>>();
        assert_eq!(own(0), ["contrast", "gamma"], "Silueta as shown");
        assert_eq!(own(1), ["gamma"], "BiRefNet with what was changed on it");
        assert_eq!(saved[2].1, started.file().models[&preset_key(nomos2)], "the started preset's other model");
    }

    /// A model the started preset has an entry for, set back to Prunr's
    /// settings, is listed with an empty entry: saving it takes the entry
    /// out.
    #[test]
    fn a_model_set_back_to_prunrs_leaves_the_preset() {
        let birefnet = SettingsModel::BiRefNetLite;
        let started = Preset::from(file(r#"{ "format_version": 2, "models": { "BiRefNetLite": { "item_settings": { "gamma": 1.2 } } } }"#));
        let item = prunr_item_settings(birefnet);
        let saved = shown_save(Some(&started), birefnet, &item, &Changes::new(started.item_settings(birefnet)));
        assert_eq!(saved, [(preset_key(birefnet), Entry::default())]);
        assert!(started.file().clone().with_entries(saved).models.is_empty());
    }

    /// The SD eraser's prompt changed before a switch is saved in its
    /// own entry's SD part.
    #[test]
    fn the_sd_erasers_changes_are_saved_after_a_switch() {
        use super::super::preset_fields::entries_of;
        let (sd, lama) = (SettingsModel::SdInpaint, SettingsModel::Inpaint);
        let mut brush = prunr_view(sd, None).brush;
        let mut brush_changes = Changes::new(brush.clone());
        brush.sd_prompt = "a meadow".into();
        brush_changes.note(&brush, &entries_of(sd, FaceModel::default()));
        let brush = take_untouched_brush(Preset::prunr(), lama, brush, &brush_changes);
        let item = prunr_item_settings(lama);
        let item_changes = Changes::new(item);
        let saved = draft(Preset::prunr(), &Shown { model: lama, item: &item, brush: &brush, item_changes: &item_changes, brush_changes: &brush_changes });
        let sd_entry = saved.iter().find(|(key, _)| *key == preset_key(sd)).map(|(_, entry)| entry);
        assert_eq!(sd_entry.map(|e| &e.sd["prompt"]), Some(&serde_json::json!("a meadow")));
    }

    /// The preset list names a preset's models in menu order.
    #[test]
    fn a_preset_names_its_models_in_menu_order() {
        let preset = Preset::from(file(r#"{ "format_version": 2, "models": {
            "Teed": { "item_settings": { "line_strength": 0.5 } },
            "BiRefNetLite": { "item_settings": { "gamma": 1.2 } },
            "Silueta": { "item_settings": { "gamma": 0.8 } }
        } }"#));
        assert_eq!(preset.covers(), "Silueta \u{b7} BiRefNet \u{b7} Lines");
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

    /// Every SD part setting is one a save compares: a new one in
    /// `SdPreset` has to join `SD_FIELDS`, or a save drops it.
    #[test]
    fn every_sd_setting_is_compared_on_save() {
        let full = SdPreset { use_taesd: Some(true), seed: Some(1), ..SdPreset::default() };
        let mut names: Vec<String> = serde_json::to_value(full).unwrap().as_object().unwrap().keys().cloned().collect();
        names.sort();
        let mut fields = SD_FIELDS.to_vec();
        fields.sort_unstable();
        assert_eq!(names, fields);
    }

    /// A v1 preset's settings belong to every model that shows them, as
    /// they applied before presets had models.
    #[test]
    fn a_v1_preset_applies_on_every_model_that_shows_its_settings() {
        let v1 = ItemSettings { gamma: 1.3, line_strength: 0.8, ..ItemSettings::default() };
        let preset = Preset::from(PresetFile::of_item_settings(v1));
        for model in [SettingsModel::Silueta, SettingsModel::U2net, SettingsModel::BiRefNetLite] {
            assert_eq!(preset.item_settings(model).gamma, 1.3, "{model:?}");
        }
        assert_eq!(preset.item_settings(SettingsModel::None).line_strength, 0.8, "the line model's, wherever it runs");
        assert_eq!(preset.item_settings(SettingsModel::Nomos2Upscale), prunr_item_settings(SettingsModel::Nomos2Upscale), "an upscaler shows none of them");
    }

    /// A setting that does not read leaves the others of its entry.
    #[test]
    fn a_setting_that_does_not_read_leaves_the_others() {
        let preset = Preset::from(file(r#"{ "format_version": 2, "models": { "Silueta": { "item_settings": {
            "gamma": 1.4, "fill_style": { "not_an_effect_yet": {} }, "feather": 2.0
        } } } }"#));
        let item = preset.item_settings(SettingsModel::Silueta);
        assert_eq!((item.gamma, item.feather), (1.4, 2.0));
        assert_eq!(item.effect, prunr_item_settings(SettingsModel::Silueta).effect);
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
