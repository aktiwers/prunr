//! One JSON file per preset in the config dir's `presets/`, named by the
//! sanitized preset name, so a user shares one by sending the file. JSON
//! because shared files must stay human-readable and hand-editable.
//!
//! Old preset files MUST keep loading as `ItemSettings` grows:
//! - every field reads from default (`#[serde(default)]` on the struct);
//! - a renamed field keeps `#[serde(alias = "old_name")]`;
//! - a type change that breaks existing JSON (e.g. `u8` → enum) bumps the
//!   format version with a migration in `parse_preset_value`.
//!
//! The tests below are the tripwires.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;

use super::item_settings::ItemSettings;
use super::presets::{Preset, PresetFile, PRESET_FORMAT_VERSION};
use super::settings::PRUNR_PRESET;

const PRESETS_SUBDIR: &str = "prunr/presets";

pub(crate) const PRESET_EXT: &str = "json";

/// Created if needed; `None` without a platform config dir.
pub(crate) fn presets_dir() -> Option<PathBuf> {
    let base = super::settings::config_root()?;
    let dir = base.join(PRESETS_SUBDIR);
    let _ = std::fs::create_dir_all(&dir);
    Some(dir)
}

/// Whether `name` is Prunr's own preset, in any case.
pub(crate) fn is_prunr(name: &str) -> bool {
    name.trim().eq_ignore_ascii_case(PRUNR_PRESET)
}

/// The file preset `name` is kept in.
pub(crate) fn file_name(name: &str) -> String {
    format!("{name}.{PRESET_EXT}")
}

/// `None` for the reserved "Prunr" or without a config dir.
pub(crate) fn preset_path(name: &str) -> Option<PathBuf> {
    if is_prunr(name) { return None; }
    let dir = presets_dir()?;
    Some(dir.join(file_name(&sanitize_filename(name))))
}

/// Preset names in the order the list shows them, case ignored.
pub(crate) fn sorted_names(names: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut names: Vec<String> = names.into_iter().collect();
    names.sort_by_key(|name| name.to_lowercase());
    names
}

/// A name Windows keeps for a device, whatever follows a dot: CON, PRN,
/// AUX, NUL, COM0-9, LPT0-9.
fn is_device_name(name: &str) -> bool {
    let name = name.to_ascii_uppercase();
    matches!(name.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (name.len() == 4 && (name.starts_with("COM") || name.starts_with("LPT")) && name.as_bytes()[3].is_ascii_digit())
}

/// The name a typed one saves under, as its file keeps it; `None` for a
/// blank one.
pub(crate) fn preset_name(typed: &str) -> Option<String> {
    let typed = typed.trim();
    (!typed.is_empty()).then(|| sanitize_filename(typed))
}

/// Strips path separators, `..`, control chars, the characters Windows
/// refuses in a file name and leading dots (no hidden files), and marks a
/// device name; an empty result becomes `preset`. Every platform does
/// all of it: presets are shared between computers.
pub fn sanitize_filename(name: &str) -> String {
    let mut out: String = name
        .trim()
        .chars()
        .filter(|c| !c.is_control() && !matches!(c, '/' | '\\' | '<' | '>' | ':' | '"' | '|' | '?' | '*'))
        .collect();
    if out.starts_with('.') {
        out = out.trim_start_matches('.').to_string();
    }
    if out.contains("..") {
        out = out.replace("..", "_");
    }
    if out.is_empty() {
        return "preset".to_string();
    }
    let device = out.split('.').next().unwrap_or_default();
    if is_device_name(device) {
        out.insert(device.len(), '_');
    }
    out
}

/// A v1 file (raw `ItemSettings`, no model) wraps under the default
/// model; its brush and SD values are Prunr's. The file on disk stays v1
/// until the user re-saves it.
fn migrate_v1_to_v2(value: &serde_json::Value) -> Option<PresetFile> {
    // `#[serde(default)]` would read any JSON as a fully-default
    // ItemSettings, masking a corrupt or unrelated file.
    if !names_a_setting(value) {
        return None;
    }
    let item_settings: ItemSettings = serde_json::from_value(value.clone()).ok()?;
    Some(PresetFile::of_item_settings(item_settings))
}

/// The showcase presets built into Prunr, read-only like Prunr's own.
const SHOWCASES: &[(&str, &str)] = &[
    ("Catalog", include_str!("../../presets/showcases/Catalog.json")),
    ("Cinematic", include_str!("../../presets/showcases/Cinematic.json")),
    ("Color Pop", include_str!("../../presets/showcases/Color Pop.json")),
    ("Comic", include_str!("../../presets/showcases/Comic.json")),
    ("Duotone Poster", include_str!("../../presets/showcases/Duotone Poster.json")),
    ("Engraving", include_str!("../../presets/showcases/Engraving.json")),
    ("Golden Bokeh", include_str!("../../presets/showcases/Golden Bokeh.json")),
    ("Halftone Pop", include_str!("../../presets/showcases/Halftone Pop.json")),
    ("Halftone Print", include_str!("../../presets/showcases/Halftone Print.json")),
    ("Ink Stamp", include_str!("../../presets/showcases/Ink Stamp.json")),
    ("Lens Portrait", include_str!("../../presets/showcases/Lens Portrait.json")),
    ("Midnight Duotone", include_str!("../../presets/showcases/Midnight Duotone.json")),
    ("Miniature", include_str!("../../presets/showcases/Miniature.json")),
    ("Misty Morning", include_str!("../../presets/showcases/Misty Morning.json")),
    ("Neon Glow", include_str!("../../presets/showcases/Neon Glow.json")),
    ("Night Vision", include_str!("../../presets/showcases/Night Vision.json")),
    ("Noir", include_str!("../../presets/showcases/Noir.json")),
    ("Old Photo", include_str!("../../presets/showcases/Old Photo.json")),
    ("Pencil Sketch", include_str!("../../presets/showcases/Pencil Sketch.json")),
    ("Pixel Art", include_str!("../../presets/showcases/Pixel Art.json")),
    ("Pop Sticker", include_str!("../../presets/showcases/Pop Sticker.json")),
    ("Soft Focus", include_str!("../../presets/showcases/Soft Focus.json")),
    ("Sticker", include_str!("../../presets/showcases/Sticker.json")),
    ("Studio", include_str!("../../presets/showcases/Studio.json")),
    ("Sunset Lines", include_str!("../../presets/showcases/Sunset Lines.json")),
    ("Teal and Orange", include_str!("../../presets/showcases/Teal and Orange.json")),
    ("Thumbnail", include_str!("../../presets/showcases/Thumbnail.json")),
    ("Warhol", include_str!("../../presets/showcases/Warhol.json")),
    ("X-Ray", include_str!("../../presets/showcases/X-Ray.json")),
];

/// The showcases as presets, read once.
fn showcases() -> &'static [(&'static str, Preset)] {
    static READ: std::sync::OnceLock<Vec<(&str, Preset)>> = std::sync::OnceLock::new();
    READ.get_or_init(|| {
        SHOWCASES
            .iter()
            .filter_map(|&(name, json)| parse_preset_bytes(json.as_bytes()).ok().map(|file| (name, Preset::from(file).built_in())))
            .collect()
    })
}

/// The presets in the presets folder, and the showcases not `hidden` and
/// not named by a file there (a file of the user's wins); the names a
/// newer Prunr saved are left out. A malformed file is skipped with a
/// warning, so one broken preset doesn't hide the rest.
pub(crate) fn load_all(hidden: &BTreeSet<String>) -> (HashMap<String, Preset>, HashSet<String>) {
    let (mut presets, newer) = presets_dir().map(|dir| load_files(&dir)).unwrap_or_default();
    add_showcases(&mut presets, hidden, &newer);
    (presets, newer)
}

fn add_showcases(presets: &mut HashMap<String, Preset>, hidden: &BTreeSet<String>, newer: &HashSet<String>) {
    for (name, showcase) in showcases() {
        let taken = hidden.iter().chain(presets.keys()).any(|other| other.eq_ignore_ascii_case(name)) || newer.contains(&name.to_lowercase());
        if !taken {
            presets.insert(name.to_string(), showcase.clone());
        }
    }
}

fn load_files(dir: &Path) -> (HashMap<String, Preset>, HashSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return Default::default() };

    let paths: Vec<PathBuf> = entries
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            (p.extension().and_then(|s| s.to_str()) == Some(PRESET_EXT))
                .then_some(p)
        })
        .collect();

    let read: Vec<(String, Result<Preset, Skipped>)> = paths
        .par_iter()
        .filter_map(|path| {
            let name = path.file_stem()?.to_str()?.to_string();
            (!is_prunr(&name)).then(|| (name, read_preset(path).map(Preset::from)))
        })
        .collect();
    let mut presets = HashMap::new();
    let mut newer = HashSet::new();
    for (name, preset) in read {
        match preset {
            Ok(preset) => {
                presets.insert(name, preset);
            }
            Err(Skipped::Newer) => {
                newer.insert(name.to_lowercase());
            }
            Err(Skipped::Unreadable) => {}
        }
    }
    (presets, newer)
}

/// Preset `name` from the presets folder, its case ignored, or the
/// preset file at that path; why not, when it cannot be read.
pub(crate) fn find_preset(name: &str) -> Result<Preset, String> {
    let path = Path::new(name);
    if path.is_file() {
        return read_preset(path).map(Preset::from).map_err(|skipped| match skipped {
            Skipped::Newer => format!("{name} was saved by a newer Prunr"),
            Skipped::Unreadable => format!("{name} is not a Prunr preset"),
        });
    }
    let (mut presets, newer) = load_all(&BTreeSet::new());
    let found = presets.keys().find(|key| key.eq_ignore_ascii_case(name)).cloned();
    if let Some(preset) = found.and_then(|key| presets.remove(&key)) {
        return Ok(preset);
    }
    if newer.contains(&name.to_lowercase()) {
        return Err(format!("the preset \"{name}\" was saved by a newer Prunr"));
    }
    let names: Vec<String> = std::iter::once(PRUNR_PRESET.to_string()).chain(sorted_names(presets.into_keys())).collect();
    Err(format!("no preset named \"{name}\" (presets: {})", names.join(", ")))
}

/// Why a preset file was left out.
#[derive(Debug, PartialEq)]
enum Skipped {
    /// A newer Prunr saved it, in a format this one cannot read.
    Newer,
    Unreadable,
}

/// Why a preset cannot be saved under `name`, when it cannot: "Prunr" and
/// the built-in presets in `presets` are never written over, and a name
/// in `newer` (a newer Prunr's preset) is kept for it.
pub(crate) fn name_refusal(name: &str, newer: &HashSet<String>, presets: &HashMap<String, Preset>) -> Option<&'static str> {
    if is_prunr(name) {
        return Some(RESERVED_NAME);
    }
    if presets.iter().any(|(other, preset)| preset.is_built_in() && other.eq_ignore_ascii_case(name.trim())) {
        return Some("A preset built into Prunr has this name — pick another.");
    }
    newer.contains(&name.trim().to_lowercase()).then_some("A newer version of Prunr saved a preset with this name — pick another.")
}

const RESERVED_NAME: &str = "\"Prunr\" is reserved — pick another name.";

/// Parse a `serde_json::Value` into a `PresetFile`, trimmed to what
/// differs from Prunr's. v2 (has `format_version: 2`) deserializes
/// directly; one with no version is attempted as a v1 `ItemSettings`
/// migration. A later version is a newer Prunr's, never read as v1.
fn parse_preset_value(mut value: serde_json::Value) -> Result<PresetFile, Skipped> {
    super::item_settings::migrate_old_item_settings(&mut value);
    match value.get("format_version").and_then(|v| v.as_u64()) {
        Some(v) if v > u64::from(PRESET_FORMAT_VERSION) => Err(Skipped::Newer),
        Some(2) => serde_json::from_value::<PresetFile>(value).map(PresetFile::trimmed).map_err(|_| Skipped::Unreadable),
        _ => migrate_v1_to_v2(&value).ok_or(Skipped::Unreadable),
    }
}

/// A preset file's bytes as a preset, or why not.
fn parse_preset_bytes(data: &[u8]) -> Result<PresetFile, Skipped> {
    serde_json::from_slice(data).map_err(|_| Skipped::Unreadable).and_then(parse_preset_value)
}

/// The preset, or why it was left out: unreadable, malformed, matching
/// neither format, or a newer Prunr's.
fn read_preset(path: &std::path::Path) -> Result<PresetFile, Skipped> {
    let data = std::fs::read(path).map_err(|e| {
        tracing::warn!(?path, %e, "skipping unreadable preset");
        Skipped::Unreadable
    })?;
    parse_preset_bytes(&data).inspect_err(|skipped| match skipped {
        Skipped::Newer => tracing::warn!(?path, "skipping a preset a newer Prunr saved; it is kept as it is"),
        Skipped::Unreadable => tracing::warn!(?path, "skipping a file that is no preset"),
    })
}

/// A preset file to import: picked or dropped by path, or dropped as its
/// bytes.
pub(crate) enum ImportSource {
    Path(PathBuf),
    Bytes { name: String, bytes: Arc<[u8]> },
}

/// How the import of one file went.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Imported {
    /// Added under this name.
    As(String),
    /// A newer Prunr's, added under this name: listed once Prunr is
    /// updated.
    Newer(String),
    /// This file is not a Prunr preset, or could not be read.
    NotAPreset(String),
    /// This file could not be copied into the presets folder.
    NotCopied(String),
}

/// Copy each preset in `sources` into the presets folder as it is, under
/// its file's name or, when that is taken, the next free one
/// ("Portrait 2"), so no preset is replaced.
pub(crate) fn import(sources: Vec<ImportSource>) -> Vec<Imported> {
    let Some(dir) = presets_dir() else { return Vec::new() };
    import_into(&dir, sources)
}

fn import_into(dir: &Path, sources: Vec<ImportSource>) -> Vec<Imported> {
    sources.into_iter().map(|source| import_one(dir, source)).collect()
}

fn import_one(dir: &Path, source: ImportSource) -> Imported {
    let (file, bytes) = match source {
        ImportSource::Path(path) => (path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string(), std::fs::read(&path).ok()),
        ImportSource::Bytes { name, bytes } => (name, Some(bytes.to_vec())),
    };
    let Some(bytes) = bytes else { return Imported::NotAPreset(file) };
    let newer = match parse_preset_bytes(&bytes) {
        Ok(_) => false,
        Err(Skipped::Newer) => true,
        Err(Skipped::Unreadable) => return Imported::NotAPreset(file),
    };
    let stem = Path::new(&file).file_stem().and_then(|s| s.to_str()).unwrap_or_default();
    let name = free_name(dir, stem);
    match crate::fs_util::write_bytes_replacing(&dir.join(file_name(&name)), &bytes) {
        Ok(()) if newer => Imported::Newer(name),
        Ok(()) => Imported::As(name),
        Err(err) => {
            tracing::warn!(%err, file, "preset not imported");
            Imported::NotCopied(file)
        }
    }
}

/// Whether `value` reads as a preset rather than any JSON: it has a
/// format version, or names an image setting.
fn names_a_setting(value: &serde_json::Value) -> bool {
    let Some(map) = value.as_object() else { return false };
    let settings = serde_json::to_value(ItemSettings::default()).unwrap_or_default();
    map.contains_key("format_version") || settings.as_object().is_some_and(|known| map.keys().any(|key| known.contains_key(key)))
}

/// `stem` as a preset name no file in `dir` has yet, and not "Prunr".
fn free_name(dir: &Path, stem: &str) -> String {
    let base = preset_name(stem).unwrap_or_else(|| "Imported".to_string());
    let taken = |name: &str| is_prunr(name) || dir.join(file_name(name)).exists();
    let mut name = base.clone();
    let mut n = 1;
    while taken(&name) {
        n += 1;
        name = format!("{base} {n}");
    }
    name
}

/// Overwrites a file of the same name; "Prunr" is refused.
pub(crate) fn save(name: &str, values: &PresetFile) -> std::io::Result<()> {
    write_preset(&writable_path(name)?, values)
}

/// `preset_path` for any name but the reserved "Prunr".
fn writable_path(name: &str) -> std::io::Result<PathBuf> {
    if is_prunr(name) {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, RESERVED_NAME));
    }
    preset_path(name).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "config dir unavailable"))
}

fn write_preset(path: &std::path::Path, file: &PresetFile) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    crate::fs_util::write_bytes_replacing(path, json.as_bytes())
}

/// Write a preset off the UI thread, after every config write asked for
/// before it. "Prunr" is never written.
pub(crate) fn queue_save(name: &str, file: &PresetFile) {
    if let Ok(path) = writable_path(name) {
        queue_write(path, file);
    }
}

/// Write `file` at `path` off the UI thread, in order with the other
/// config writes.
pub(crate) fn queue_write(path: PathBuf, file: &PresetFile) {
    match serde_json::to_string_pretty(file) {
        Ok(json) => super::disk_queue::write(path, json.into_bytes()),
        Err(err) => tracing::error!(?path, %err, "preset not written"),
    }
}

/// Remove a preset's file off the UI thread.
pub(crate) fn queue_delete(name: &str) {
    if let Some(path) = preset_path(name) {
        super::disk_queue::remove(path);
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    /// A preset name becomes a file name inside the presets folder:
    /// separators, control characters, leading dots and an embedded ".."
    /// cannot reach outside it, and an empty name falls back.
    #[test]
    fn a_preset_name_stays_inside_the_presets_folder() {
        for (name, file) in [
            ("foo/bar", "foobar"),
            ("foo\\bar", "foobar"),
            // Slash stripped to "..etc", then the leading dots.
            ("../etc", "etc"),
            ("foo..bar", "foo_bar"),
            (".hidden", "hidden"),
            ("good\x00bad\x01name", "goodbadname"),
            ("  Portrait  ", "Portrait"),
            ("", "preset"),
            ("   ", "preset"),
        ] {
            assert_eq!(sanitize_filename(name), file, "{name:?}");
        }
    }

    /// A name Windows cannot save stays savable there, and on the
    /// computer a preset is shared to: its forbidden characters go, and a
    /// device name (CON, NUL, COM1...) gets a mark.
    #[test]
    fn a_preset_name_is_a_file_name_windows_can_save() {
        for (name, file) in [
            ("Sky: blue?", "Sky blue"),
            ("a<b>c|d\"e*f", "abcdef"),
            ("CON", "CON_"),
            ("nul.portrait", "nul_.portrait"),
            ("com1", "com1_"),
            ("LPT9", "LPT9_"),
            ("Console", "Console"),
            ("COM10", "COM10"),
        ] {
            assert_eq!(sanitize_filename(name), file, "{name:?}");
        }
    }

    /// A typed name is the preset's name as its file keeps it, so the
    /// dialog checks what will be saved: "a:b" overwrites "ab".
    #[test]
    fn a_typed_name_is_the_name_its_file_keeps() {
        assert_eq!(preset_name("  Sky: blue? "), Some("Sky blue".to_string()));
        assert_eq!(preset_name("CON"), Some("CON_".to_string()));
        assert_eq!(preset_name("Portrait"), Some("Portrait".to_string()));
        assert_eq!(preset_name("   "), None);
    }

    /// A preset a newer Prunr saved is left out, not read as defaults, and
    /// its name cannot be saved over, so that version keeps it.
    #[test]
    fn a_newer_preset_is_left_out_and_never_written_over() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("Future.json");
        std::fs::write(&path, r#"{ "format_version": 3, "models": {}, "something_new": true }"#).unwrap();
        assert_eq!(read_preset(&path).unwrap_err(), Skipped::Newer);
        let newer: HashSet<String> = ["future".to_string()].into();
        assert!(name_refusal("Future", &newer, &HashMap::new()).is_some(), "its name is refused, any case");
        assert!(name_refusal(" future ", &newer, &HashMap::new()).is_some());
        assert_eq!(name_refusal("Other", &newer, &HashMap::new()), None);
        assert!(name_refusal("prunr", &HashSet::new(), &HashMap::new()).is_some(), "Prunr stays reserved");
    }

    #[test]
    fn prunr_has_no_file_to_delete() {
        // Would be disastrous to delete anything when a user toggles the
        // Prunr row, so it names no file.
        assert!(preset_path(PRUNR_PRESET).is_none());
    }

    // ── Forward-compat tripwires ──────────────────────────────────────
    //
    // These tests fail loudly if `ItemSettings` loses its forward-compat
    // properties. If you added a field and one of these broke, the fix is
    // almost always `#[serde(default)]` on the struct (already present) or
    // on the new field. See the module docstring.

    #[test]
    fn loads_preset_with_unknown_fields() {
        // Only unknown keys: every real field defaults (old presets load)
        // and the unknown ones are ignored (a newer build's presets load).
        let json = r#"{
            "definitely_not_a_real_field": 42,
            "another_future_field": "hello"
        }"#;
        let parsed: ItemSettings = serde_json::from_str(json)
            .expect("ItemSettings must default every missing field and ignore unknown ones — no #[serde(deny_unknown_fields)]");
        assert_eq!(parsed, ItemSettings::default());
    }

    /// Every file in `presets/showcases` is built in, reads as a preset,
    /// and holds only what differs from Prunr's settings.
    #[test]
    fn every_showcase_file_is_built_in_and_holds_only_its_differences() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("presets/showcases");
        let mut files: Vec<String> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path().file_stem().unwrap().to_str().unwrap().to_string()).collect();
        files.sort_by_key(|name| name.to_lowercase());
        let listed: Vec<&str> = SHOWCASES.iter().map(|&(name, _)| name).collect();
        assert_eq!(files, listed, "SHOWCASES lists every file once, by name");
        assert_eq!(showcases().len(), SHOWCASES.len(), "every showcase reads");
        for &(name, json) in SHOWCASES {
            assert!(!is_prunr(name));
            let file: PresetFile = serde_json::from_str(json).unwrap();
            assert_eq!(file.clone().trimmed(), file, "{name} sets only what differs from Prunr's");
        }
    }

    /// The list shows a showcase unless it was hidden or a file of the
    /// user's has its name; only an unshadowed one is built in.
    #[test]
    fn a_showcase_shows_unless_hidden_or_the_user_has_its_name() {
        let (name, _) = showcases()[0];
        let shown = |hidden: &BTreeSet<String>, mut presets: HashMap<String, Preset>| {
            add_showcases(&mut presets, hidden, &HashSet::new());
            presets.get(name).map(Preset::is_built_in)
        };
        assert_eq!(shown(&BTreeSet::new(), HashMap::new()), Some(true));
        assert_eq!(shown(&BTreeSet::from([name.to_lowercase()]), HashMap::new()), None, "hidden, in any case");
        assert_eq!(shown(&BTreeSet::new(), HashMap::from([(name.to_string(), PresetFile::default().into())])), Some(false), "the user's file wins");
    }

    /// A shown showcase is never written over; once hidden, or shadowed
    /// by a file of the user's, its name is free.
    #[test]
    fn a_shown_showcases_name_is_refused_for_a_save() {
        let (name, preset) = &showcases()[0];
        let shown = HashMap::from([(name.to_string(), preset.clone())]);
        assert!(name_refusal(&name.to_uppercase(), &HashSet::new(), &shown).is_some());
        assert_eq!(name_refusal(name, &HashSet::new(), &HashMap::new()), None);
        let users = HashMap::from([(name.to_string(), PresetFile::default().into())]);
        assert_eq!(name_refusal(name, &HashSet::new(), &users), None);
    }

    // ── v1 → v2 migration ────────────────────────────────────────────
    //
    // v1 preset files are raw `ItemSettings` JSON (no `format_version`
    // key). They predate per-model presets, so every v1 file wraps under
    // the default model, keeping what differs from Prunr's.

    fn default_entry(file: &PresetFile) -> Option<&super::super::presets::Entry> {
        file.models.get(&super::super::presets::preset_key(super::super::settings::DEFAULT_MODEL))
    }

    /// A v1 file keeps what differs from Prunr's, under every model that
    /// shows it.
    #[test]
    fn migrate_v1_keeps_what_differs_under_every_model() {
        let migrated = migrate_v1_to_v2(&serde_json::json!({ "gamma": 2.0, "edge_thickness": 5, "definitely_not_a_real_field": 42 })).expect("migrates");
        assert_eq!(migrated.format_version, 2);
        let entry = default_entry(&migrated).expect("default model entry");
        assert_eq!(entry.item_settings["gamma"].as_f64(), Some(2.0));
        assert_eq!(entry.item_settings.len(), 1, "only what differs");
        assert!(entry.brush.is_empty() && entry.sd.is_empty());
        assert_eq!(migrated.models["Teed"].item_settings["edge_thickness"].as_u64(), Some(5), "the line width is the line model's");
        let prunrs = migrate_v1_to_v2(&serde_json::json!({ "gamma": 1.0 })).expect("migrates");
        assert!(prunrs.models.is_empty(), "nothing apart from Prunr's");
    }

    #[test]
    fn migrate_v1_with_garbage_json_returns_none() {
        // JSON that names no image setting is no preset: load_all skips it,
        // and an import refuses it.
        assert!(migrate_v1_to_v2(&serde_json::json!("not an object")).is_none());
        assert!(migrate_v1_to_v2(&serde_json::json!([1, 2, 3])).is_none());
        assert!(migrate_v1_to_v2(&serde_json::json!({})).is_none());
        assert!(migrate_v1_to_v2(&serde_json::json!({ "name": "web", "version": "1.0" })).is_none());
    }

    /// Once saved as v2, a migrated file loads back as it was.
    #[test]
    fn v1_to_v2_round_trip_is_stable() {
        let migrated = migrate_v1_to_v2(&serde_json::json!({ "gamma": 1.5, "feather": 3.5 })).expect("migrates");
        let saved: serde_json::Value = serde_json::from_str(&serde_json::to_string_pretty(&migrated).unwrap()).unwrap();
        assert_eq!(saved.get("format_version").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(parse_preset_value(saved).expect("loads as v2"), migrated);
    }

    // ── Saving into a preset ──────────────────────────────────────────
    //
    // Saving sets the model in use and keeps every other model's entry as
    // it was. Tests use `save_at` so they don't write to the user's
    // presets folder.

    /// A save as the app makes it after applying the preset at `path` and
    /// setting `item` and `brush` on `model`: the draft's rows, all ticked,
    /// over the preset as loaded.
    fn save_at(path: &std::path::Path, model: super::super::settings::SettingsModel, item: &ItemSettings, brush: &super::super::brush_state::BrushSettings) -> std::io::Result<()> {
        use super::super::preset_changes::Changes;
        use super::super::presets::{draft, Shown};
        let existing = read_preset(path).unwrap_or_default();
        let started = Preset::from(existing.clone());
        let (item_changes, brush_changes) = (Changes::new(started.item_settings(model)), Changes::new(started.brush(model, None).clone()));
        let shown = Shown { model, item, brush, item_changes: &item_changes, brush_changes: &brush_changes };
        write_preset(path, &existing.with_entries(draft(&started, &shown)))
    }

    #[test]
    fn saving_keeps_the_other_models_entries() {
        use super::super::presets::{prunr_item_settings, prunr_view, Preset};
        use super::super::settings::SettingsModel;
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("Foo.json");
        let silueta = ItemSettings { gamma: 1.5, ..prunr_item_settings(SettingsModel::Silueta) };
        save_at(&path, SettingsModel::Silueta, &silueta, &prunr_view(SettingsModel::Silueta, None).brush).unwrap();
        let lama_brush = super::super::brush_state::BrushSettings { inpaint_grow: 6.0, ..prunr_view(SettingsModel::Inpaint, None).brush };
        save_at(&path, SettingsModel::Inpaint, &prunr_item_settings(SettingsModel::Inpaint), &lama_brush).unwrap();
        let reloaded = Preset::from(read_preset(&path).expect("reloads"));
        assert_eq!(reloaded.item_settings(SettingsModel::Silueta), silueta, "the earlier model's entry stays");
        assert_eq!(reloaded.resolve(SettingsModel::Inpaint, None).brush, lama_brush);
    }

    /// Saving onto a v1 file keeps its settings under the default model
    /// beside the new entry.
    #[test]
    fn saving_onto_a_v1_file_keeps_it() {
        use super::super::presets::{prunr_item_settings, prunr_view, Preset};
        use super::super::settings::{SettingsModel, DEFAULT_MODEL};
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("LegacyV1.json");
        std::fs::write(&path, serde_json::json!({ "gamma": 1.25, "edge_thickness": 7 }).to_string()).unwrap();
        let nomos = ItemSettings { sharpen: 0.5, ..prunr_item_settings(SettingsModel::Nomos2Upscale) };
        save_at(&path, SettingsModel::Nomos2Upscale, &nomos, &prunr_view(SettingsModel::Nomos2Upscale, None).brush).unwrap();
        let reloaded = Preset::from(read_preset(&path).expect("reloads"));
        let v1 = reloaded.item_settings(DEFAULT_MODEL);
        assert_eq!((v1.gamma, v1.edge_thickness), (1.25, 7));
        assert_eq!(reloaded.item_settings(SettingsModel::Nomos2Upscale).sharpen, 0.5);
    }

    #[test]
    fn a_queued_save_rejects_the_prunr_name() {
        for name in ["Prunr", "prunr"] {
            let err = writable_path(name).expect_err("Prunr is reserved");
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        }
    }

    /// Every Stable Diffusion setting a save keeps comes back on load: a
    /// new one that does not read from default, or is dropped on save,
    /// breaks the round trip.
    #[test]
    fn every_sd_setting_round_trips_through_a_save() {
        use super::super::brush_state::{BrushSettings, SdScheduler};
        use super::super::presets::{prunr_item_settings, prunr_view, Preset};
        use super::super::settings::SettingsModel;
        let sd = SettingsModel::SdInpaint;
        let brush = BrushSettings {
            radius: 73.0,
            sd_prompt: "a-distinctive-prompt".into(),
            sd_negative_prompt: "a-distinctive-neg-prompt".into(),
            sd_steps: 17,
            sd_guidance_scale: 5.25,
            sd_scheduler: SdScheduler::DpmPlusPlus2MKarras,
            sd_seed: Some(13579),
            sd_use_taesd: Some(true),
            sd_strength: 0.75,
            inpaint_grow: 5.0,
            ..prunr_view(sd, None).brush
        };
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("Dynamic.json");
        save_at(&path, sd, &prunr_item_settings(sd), &brush).unwrap();
        let reloaded = Preset::from(read_preset(&path).expect("reloads"));
        assert_eq!(reloaded.resolve(sd, None).brush, brush);
    }

    /// An import copies each preset in under its file's name, the next
    /// free one when that is taken or "Prunr", keeps a newer Prunr's for
    /// it, and leaves out what is not a preset.
    #[test]
    fn an_import_adds_each_preset_under_a_free_name() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let preset: &[u8] = br#"{ "format_version": 2, "models": { "Silueta": { "item_settings": { "gamma": 1.5 } } } }"#;
        std::fs::write(dir.path().join("Portrait.json"), preset).unwrap();
        let file = |name: &str, bytes: &[u8]| ImportSource::Bytes { name: name.into(), bytes: Arc::from(bytes) };
        let done = import_into(dir.path(), vec![
            file("Portrait.json", preset),
            file("Prunr.json", preset),
            file("old.json", br#"{ "gamma": 1.2 }"#),
            file("package.json", br#"{ "name": "web", "version": "1.0" }"#),
            file("broken.json", b"{ not json"),
            file("Future.json", br#"{ "format_version": 99 }"#),
        ]);
        assert_eq!(done, [
            Imported::As("Portrait 2".into()),
            Imported::As("Prunr 2".into()),
            Imported::As("old".into()),
            Imported::NotAPreset("package.json".into()),
            Imported::NotAPreset("broken.json".into()),
            Imported::Newer("Future".into()),
        ]);
        assert_eq!(std::fs::read(dir.path().join("Portrait 2.json")).unwrap(), preset, "copied as it is");
        assert_eq!(std::fs::read(dir.path().join("Portrait.json")).unwrap(), preset, "the one there stays");
    }
}
