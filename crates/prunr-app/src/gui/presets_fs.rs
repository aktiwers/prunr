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

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use rayon::prelude::*;

use super::item_settings::ItemSettings;
use super::presets::{
    model_id_key, ModelPreset, PresetFile, PRESET_FORMAT_VERSION,
};
use super::settings::PRUNR_PRESET;
use prunr_core::{ComposeMode, SubjectEffect, LineMode, LineStyle};

const PRESETS_SUBDIR: &str = "prunr/presets";

const PRESET_EXT: &str = "json";

/// Present once the built-ins were seeded, so a deleted built-in stays
/// deleted. Bump the suffix when shipping a fresh batch of built-ins.
const SEED_MARKER: &str = ".builtins_seeded_v2";

/// Created if needed; `None` without a platform config dir.
pub(crate) fn presets_dir() -> Option<PathBuf> {
    let base = super::settings::config_root()?;
    let dir = base.join(PRESETS_SUBDIR);
    let _ = std::fs::create_dir_all(&dir);
    Some(dir)
}

/// `None` for the reserved "Prunr" or without a config dir.
pub(crate) fn preset_path(name: &str) -> Option<PathBuf> {
    if name.eq_ignore_ascii_case(PRUNR_PRESET) { return None; }
    let dir = presets_dir()?;
    Some(dir.join(format!("{}.{PRESET_EXT}", sanitize_filename(name))))
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

/// A v1 file (raw `ItemSettings`, no model) wraps under
/// `DEFAULT_PRESET_MODEL`; brush and SD values come from the factory floor
/// at apply time. The file on disk stays v1 until the user re-saves it.
fn migrate_v1_to_v2(value: &serde_json::Value) -> Option<PresetFile> {
    // `#[serde(default)]` would read an array or unrelated value as a
    // fully-default ItemSettings, masking a corrupt file.
    if !value.is_object() {
        return None;
    }
    let item_settings: ItemSettings = serde_json::from_value(value.clone()).ok()?;
    Some(PresetFile::of_item_settings(item_settings))
}

/// A malformed file is skipped with a warning, so one broken preset
/// doesn't hide the rest.
pub(crate) fn load_all() -> (HashMap<String, PresetFile>, HashSet<String>) {
    let Some(dir) = presets_dir() else { return Default::default() };
    let Ok(entries) = std::fs::read_dir(&dir) else { return Default::default() };

    let paths: Vec<PathBuf> = entries
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            (p.extension().and_then(|s| s.to_str()) == Some(PRESET_EXT))
                .then_some(p)
        })
        .collect();

    let read: Vec<(String, Result<PresetFile, Skipped>)> = paths
        .par_iter()
        .filter_map(|path| {
            let name = path.file_stem()?.to_str()?.to_string();
            (!name.eq_ignore_ascii_case(PRUNR_PRESET)).then(|| (name, read_preset(path)))
        })
        .collect();
    let mut presets = HashMap::new();
    let mut newer = HashSet::new();
    for (name, file) in read {
        match file {
            Ok(file) => {
                presets.insert(name, file);
            }
            Err(Skipped::Newer) => {
                newer.insert(name.to_lowercase());
            }
            Err(Skipped::Unreadable) => {}
        }
    }
    (presets, newer)
}

/// Why a preset file was left out.
#[derive(Debug, PartialEq)]
enum Skipped {
    /// A newer Prunr saved it, in a format this one cannot read.
    Newer,
    Unreadable,
}

/// Why a preset cannot be saved under `name`, when it cannot: "Prunr" is
/// reserved, and a name in `newer` (a newer Prunr's preset) is kept for it.
pub(crate) fn name_refusal(name: &str, newer: &HashSet<String>) -> Option<&'static str> {
    let name = name.trim();
    if name.eq_ignore_ascii_case(PRUNR_PRESET) {
        return Some(RESERVED_NAME);
    }
    newer.contains(&name.to_lowercase()).then_some("A newer version of Prunr saved a preset with this name — pick another.")
}

const RESERVED_NAME: &str = "\"Prunr\" is reserved — pick another name.";

/// Parse a `serde_json::Value` into a `PresetFile`. v2 (has
/// `format_version: 2`) deserializes directly; one with no version is
/// attempted as a v1 `ItemSettings` migration. A later version is a
/// newer Prunr's, never read as v1.
fn parse_preset_value(mut value: serde_json::Value) -> Result<PresetFile, Skipped> {
    super::item_settings::migrate_old_item_settings(&mut value);
    match value.get("format_version").and_then(|v| v.as_u64()) {
        Some(v) if v > u64::from(PRESET_FORMAT_VERSION) => Err(Skipped::Newer),
        Some(2) => serde_json::from_value(value).map_err(|_| Skipped::Unreadable),
        _ => migrate_v1_to_v2(&value).ok_or(Skipped::Unreadable),
    }
}

/// The preset, or why it was left out: unreadable, malformed, matching
/// neither format, or a newer Prunr's.
fn read_preset(path: &std::path::Path) -> Result<PresetFile, Skipped> {
    let data = std::fs::read(path).map_err(|e| {
        tracing::warn!(?path, %e, "skipping unreadable preset");
        Skipped::Unreadable
    })?;
    let value: serde_json::Value = serde_json::from_slice(&data).map_err(|e| {
        tracing::warn!(?path, %e, "skipping malformed-JSON preset");
        Skipped::Unreadable
    })?;
    parse_preset_value(value).inspect_err(|skipped| match skipped {
        Skipped::Newer => tracing::warn!(?path, "skipping a preset a newer Prunr saved; it is kept as it is"),
        Skipped::Unreadable => tracing::warn!(?path, "skipping unrecognised preset shape (not v2 and not v1 ItemSettings)"),
    })
}

/// Overwrites a file of the same name; "Prunr" is refused.
pub(crate) fn save(name: &str, values: &PresetFile) -> std::io::Result<()> {
    write_preset(&writable_path(name)?, values)
}

/// `preset_path` for any name but the reserved "Prunr".
fn writable_path(name: &str) -> std::io::Result<PathBuf> {
    if name.trim().eq_ignore_ascii_case(PRUNR_PRESET) {
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
    let Ok(path) = writable_path(name) else { return };
    match serde_json::to_string_pretty(file) {
        Ok(json) => super::disk_queue::write(path, json.into_bytes()),
        Err(err) => tracing::error!(preset = %name, %err, "preset not saved"),
    }
}

/// Remove a preset's file off the UI thread.
pub(crate) fn queue_delete(name: &str) {
    if let Some(path) = preset_path(name) {
        super::disk_queue::remove(path);
    }
}

/// `model_preset` merged into `file`, the preset as it stands: entries for
/// other models stay verbatim. Scheduler bundles within `model_id`'s entry
/// come straight from `model_preset.sd.schedulers`: the caller carries
/// over any prior bundles it wants to keep.
pub(crate) fn merged(mut file: PresetFile, model_id: prunr_models::ModelId, model_preset: ModelPreset) -> PresetFile {
    file.format_version = PRESET_FORMAT_VERSION;
    file.models.insert(model_id_key(model_id), model_preset);
    file
}

/// No-op once the seed marker exists.
pub fn seed_builtins_once() {
    let Some(dir) = presets_dir() else { return };
    let marker = dir.join(SEED_MARKER);
    if marker.exists() { return; }
    for (name, item_settings) in builtin_presets() {
        // Never overwrite: a re-seed after a mid-seed crash, or a new
        // batch reusing a name, must keep the user's file.
        let Some(path) = preset_path(name) else { continue };
        if path.exists() { continue; }
        let _ = save(name, &PresetFile::of_item_settings(item_settings));
    }
    let _ = std::fs::write(&marker, b"");
}

fn builtin_presets() -> Vec<(&'static str, ItemSettings)> {
    let base = ItemSettings::default();
    vec![
        ("Comic", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::LinesOnly,
            line_strength: 0.7,
            edge_thickness: 2,
            solid_line_color: Some([0, 0, 0]),
            effect: SubjectEffect::Posterize { levels: 4 },
            gamma: 1.3,
            ..base
        }),
        ("Pencil Sketch", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::LinesOnly,
            line_strength: 0.8,
            edge_thickness: 0,
            solid_line_color: None,
            effect: SubjectEffect::Desaturate,
            gamma: 1.1,
            ..base
        }),
        ("Neon Glow", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::Ghost,
            line_strength: 0.6,
            edge_thickness: 3,
            line_style: LineStyle::Rainbow { cycles: 2 },
            effect: SubjectEffect::Saturate { percent: 220 },
            ..base
        }),
        ("Sepia", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::LinesOnly,
            line_strength: 0.6,
            edge_thickness: 1,
            solid_line_color: Some([70, 45, 20]),
            effect: SubjectEffect::Sepia,
            ..base
        }),
        ("Duotone Poster", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::SubjectFilled,
            line_strength: 0.6,
            edge_thickness: 2,
            solid_line_color: Some([10, 10, 40]),
            effect: SubjectEffect::Duotone { dark: [20, 20, 60], light: [240, 220, 180] },
            ..base
        }),
        ("X-Ray", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::Ghost,
            line_strength: 0.8,
            edge_thickness: 1,
            solid_line_color: Some([200, 230, 255]),
            effect: SubjectEffect::Invert,
            bg: Some([0, 0, 30, 255]),
            ..base
        }),
        ("Pop Art", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::SubjectFilled,
            line_strength: 0.65,
            edge_thickness: 3,
            solid_line_color: Some([0, 0, 0]),
            effect: SubjectEffect::Posterize { levels: 3 },
            gamma: 1.4,
            ..base
        }),
        ("Ghost", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::Ghost,
            line_strength: 0.55,
            edge_thickness: 1,
            solid_line_color: Some([230, 230, 235]),
            effect: SubjectEffect::Desaturate,
            ..base
        }),
        ("Sunset Lines", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::LinesOnly,
            line_strength: 0.7,
            edge_thickness: 2,
            line_style: LineStyle::GradientY {
                top: [255, 180, 40],
                bottom: [120, 20, 90],
            },
            ..base
        }),
        ("Pixel Art", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::SubjectFilled,
            line_strength: 0.5,
            edge_thickness: 2,
            solid_line_color: Some([0, 0, 0]),
            effect: SubjectEffect::Pixelate { block_size: 10 },
            ..base
        }),
    ]
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
        assert!(name_refusal("Future", &newer).is_some(), "its name is refused, any case");
        assert!(name_refusal(" future ", &newer).is_some());
        assert_eq!(name_refusal("Other", &newer), None);
        assert!(name_refusal("prunr", &HashSet::new()).is_some(), "Prunr stays reserved");
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

    #[test]
    fn builtins_have_unique_names_and_serialize() {
        let presets = super::builtin_presets();
        assert!(presets.len() >= 5, "ship at least 5 curated looks; got {}", presets.len());
        let mut seen = std::collections::HashSet::new();
        for (name, settings) in &presets {
            assert!(seen.insert(*name), "duplicate builtin preset name: {name}");
            // Reserved name would be silently rejected by `save`.
            assert!(
                !name.eq_ignore_ascii_case(PRUNR_PRESET),
                "built-in preset cannot use reserved name '{PRUNR_PRESET}'",
            );
            // Each preset must round-trip through JSON — otherwise seed_builtins_once
            // would silently skip it.
            let json = serde_json::to_string(settings).expect("preset serializes");
            let _: ItemSettings = serde_json::from_str(&json).expect("preset deserializes");
        }
    }

    // ── v1 → v2 migration ────────────────────────────────────────────
    //
    // v1 preset files are raw `ItemSettings` JSON (no `format_version`
    // key). They predate per-model bundling, so they carry no model
    // identity — every v1 file wraps under the workspace default
    // (`Settings::default().model`, currently `BiRefNetLite`).

    #[test]
    fn migrate_v1_with_item_settings_fields_wraps_into_default_model_entry() {
        let v1_json = serde_json::json!({
            "gamma": 2.0,
            "edge_thickness": 5,
        });
        let migrated = migrate_v1_to_v2(&v1_json).expect("must migrate");
        assert_eq!(migrated.format_version, 2);
        let default_key = model_id_key(prunr_models::ModelId::BiRefNetLite);
        let entry = migrated.models.get(&default_key).expect("default model entry");
        assert_eq!(entry.item_settings.gamma, 2.0);
        assert_eq!(entry.item_settings.edge_thickness, 5);
        assert_eq!(entry.brush, super::super::brush_state::BrushSettings::default());
        assert!(entry.sd.is_none());
    }

    #[test]
    fn migrate_v1_empty_object_wraps_default_item_settings() {
        // An empty JSON object is a valid v1 file (every ItemSettings field
        // serde-defaults). Migration must succeed and yield factory defaults.
        let v1_json = serde_json::json!({});
        let migrated = migrate_v1_to_v2(&v1_json).expect("must migrate empty");
        assert_eq!(migrated.format_version, 2);
        let default_key = model_id_key(prunr_models::ModelId::BiRefNetLite);
        let entry = migrated.models.get(&default_key).expect("default model entry");
        assert_eq!(entry.item_settings, ItemSettings::default());
    }

    #[test]
    fn migrate_v1_with_unknown_fields_succeeds() {
        // Unknown forward-compat fields must not break v1 migration.
        let v1_json = serde_json::json!({
            "gamma": 1.5,
            "definitely_not_a_real_field": 42,
        });
        let migrated = migrate_v1_to_v2(&v1_json).expect("must migrate");
        let default_key = model_id_key(prunr_models::ModelId::BiRefNetLite);
        let entry = migrated.models.get(&default_key).expect("default model entry");
        assert_eq!(entry.item_settings.gamma, 1.5);
    }

    #[test]
    fn migrate_v1_with_garbage_json_returns_none() {
        // Anything that doesn't deserialize as ItemSettings → None →
        // load_all silently skips. (Strings, arrays, scalars at top level
        // can't be ItemSettings.)
        let v1_json = serde_json::json!("not an object");
        assert!(migrate_v1_to_v2(&v1_json).is_none());
        let v1_json = serde_json::json!([1, 2, 3]);
        assert!(migrate_v1_to_v2(&v1_json).is_none());
    }

    #[test]
    fn v1_to_v2_round_trip_via_serde_is_stable() {
        // Seed a v1-shaped JSON, migrate to v2 in-memory, serialize as if
        // saving, parse back via the load_all branch logic, assert the
        // second-load PresetFile equals the migrated one. Proves migration
        // is idempotent — once persisted as v2, it stays v2 verbatim.
        let v1_json = serde_json::json!({
            "gamma": 1.7,
            "feather": 3.5,
        });
        let migrated = migrate_v1_to_v2(&v1_json).expect("migrate");
        let serialized = serde_json::to_string_pretty(&migrated).expect("serialize");
        let value: serde_json::Value = serde_json::from_str(&serialized).expect("reparse");
        // Second load must take the v2 branch, not re-migrate.
        assert_eq!(value.get("format_version").and_then(|v| v.as_u64()), Some(2));
        let reloaded: PresetFile = serde_json::from_value(value).expect("v2 deserialize");
        assert_eq!(reloaded, migrated);
    }

    // ── Merge-save semantics ──────────────────────────────────────────
    //
    // Overwriting a preset for the active model touches ONLY that model's
    // entry, written as the caller built it; other models round-trip
    // bytewise. Tests use `overwrite_at` so they don't write to the
    // user's real presets dir.

    /// An overwrite as the app makes it: the preset as loaded, merged,
    /// written whole.
    fn overwrite_at(path: &std::path::Path, model_id: prunr_models::ModelId, mp: ModelPreset) -> std::io::Result<()> {
        let existing = read_preset(path).unwrap_or_default();
        write_preset(path, &merged(existing, model_id, mp))
    }

    #[test]
    fn overwrite_preserves_other_model_entries() {
        use super::super::brush_state::BrushSettings;
        use super::super::presets::SdPreset;
        use prunr_models::ModelId;

        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("Foo.json");

        // Seed: Silueta entry with custom gamma + an SD entry alongside.
        let silueta_mp = ModelPreset {
            item_settings: ItemSettings { gamma: 1.5, ..ItemSettings::default() },
            brush: BrushSettings::default(),
            sd: None,
        };
        let mut seed = PresetFile::default();
        seed.models.insert(model_id_key(ModelId::Silueta), silueta_mp.clone());
        seed.models.insert(
            model_id_key(ModelId::SdV15InpaintFp16),
            ModelPreset { sd: Some(SdPreset::default()), ..ModelPreset::default() },
        );
        std::fs::write(&path, serde_json::to_string_pretty(&seed).unwrap()).unwrap();

        // Save-merge a fresh SD entry. Silueta must survive untouched.
        let new_sd_mp = ModelPreset {
            item_settings: ItemSettings { gamma: 2.5, ..ItemSettings::default() },
            brush: BrushSettings::default(),
            sd: Some(SdPreset { prompt: "rewritten".into(), ..SdPreset::default() }),
        };
        overwrite_at(&path, ModelId::SdV15InpaintFp16, new_sd_mp.clone()).unwrap();

        let reloaded = read_preset(&path).expect("reload");
        let silueta_back = reloaded
            .models
            .get(&model_id_key(ModelId::Silueta))
            .expect("Silueta entry preserved");
        assert_eq!(*silueta_back, silueta_mp, "Silueta entry must be byte-equal");
        let sd_back = reloaded
            .models
            .get(&model_id_key(ModelId::SdV15InpaintFp16))
            .expect("SD entry exists");
        assert_eq!(*sd_back, new_sd_mp, "SD entry must equal the merged-in preset");
    }

    #[test]
    fn overwrite_creates_file_when_none_exists() {
        use prunr_models::ModelId;

        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("Fresh.json");
        assert!(!path.exists());

        let mp = ModelPreset {
            item_settings: ItemSettings { gamma: 1.9, ..ItemSettings::default() },
            ..ModelPreset::default()
        };
        overwrite_at(&path, ModelId::Silueta, mp.clone()).unwrap();

        let reloaded = read_preset(&path).expect("reload");
        assert_eq!(reloaded.format_version, PRESET_FORMAT_VERSION);
        assert_eq!(reloaded.models.len(), 1);
        assert_eq!(
            reloaded.models.get(&model_id_key(ModelId::Silueta)),
            Some(&mp),
        );
    }

    #[test]
    fn overwrite_handles_v1_file_via_load_path() {
        use prunr_models::ModelId;

        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("LegacyV1.json");

        // v1 file: raw ItemSettings JSON, no format_version.
        let v1_json = serde_json::json!({
            "gamma": 1.4,
            "edge_thickness": 7,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&v1_json).unwrap()).unwrap();

        // Save-merge a different model on top. v1 → v2 migration runs
        // before merge so the v1 entry survives.
        let lama_mp = ModelPreset {
            item_settings: ItemSettings { gamma: 2.0, ..ItemSettings::default() },
            ..ModelPreset::default()
        };
        overwrite_at(&path, ModelId::LaMaFp32, lama_mp.clone()).unwrap();

        let reloaded = read_preset(&path).expect("reload");
        assert_eq!(reloaded.format_version, PRESET_FORMAT_VERSION);
        // v1 wraps under the workspace default model (BiRefNetLite).
        let v1_back = reloaded
            .models
            .get(&model_id_key(ModelId::BiRefNetLite))
            .expect("v1 entry migrated under workspace default");
        assert!((v1_back.item_settings.gamma - 1.4).abs() < f32::EPSILON);
        assert_eq!(v1_back.item_settings.edge_thickness, 7);
        // New entry merged in alongside.
        assert_eq!(
            reloaded.models.get(&model_id_key(ModelId::LaMaFp32)),
            Some(&lama_mp),
        );
    }

    #[test]
    fn a_queued_save_rejects_the_prunr_name() {
        for name in ["Prunr", "prunr"] {
            let err = writable_path(name).expect_err("Prunr is reserved");
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        }
    }

    #[test]
    fn merge_save_dynamic_field_capture() {
        use super::super::brush_state::{BrushSettings, SdScheduler};
        use super::super::presets::{fuse_brush_for_apply, sd_preset_for_save};
        use prunr_models::ModelId;

        // Build a BrushSettings with every SD-tuning field mutated to a
        // distinctive value. If a future field is added that doesn't
        // serde-default — or that gets stripped at split/save — the
        // round-trip equality breaks.
        let mut base = BrushSettings::default();
        base.radius = 73.0;
        base.hardness = 0.42;
        base.sd_prompt = "a-distinctive-prompt".into();
        base.sd_negative_prompt = "a-distinctive-neg-prompt".into();
        base.sd_steps = 17;
        base.sd_guidance_scale = 5.25;
        base.sd_scheduler = SdScheduler::DpmPlusPlus2MKarras;
        base.sd_seed = Some(13579);
        base.sd_use_taesd = Some(true);

        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("Dynamic.json");

        let mp = ModelPreset {
            item_settings: ItemSettings::default(),
            brush: base.clone(),
            sd: sd_preset_for_save(&base, ModelId::SdV15InpaintFp16),
        };
        overwrite_at(&path, ModelId::SdV15InpaintFp16, mp).unwrap();

        let reloaded = read_preset(&path).expect("reload");
        let mp_back = reloaded
            .models
            .get(&model_id_key(ModelId::SdV15InpaintFp16))
            .expect("SD entry");
        let fused = fuse_brush_for_apply(mp_back, Some(base.sd_scheduler));
        assert_eq!(fused, base, "every BrushSettings field must round-trip");
    }
}
