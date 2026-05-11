//! One-file-per-preset store for sharing presets between users.
//!
//! Presets live as individual JSON files in `~/.config/prunr/presets/`
//! (or the platform-equivalent config dir). Filename = preset name + ".json"
//! (with path-unsafe characters sanitized). To share a preset, a user sends
//! the `.json` file; the receiver drops it into the folder and the preset
//! appears in their dropdown on the next launch. JSON is the format because
//! the sharing workflow needs human-readable, hand-editable files — binary
//! formats would be faster to parse but break that.
//!
//! ## Forward-compatibility contract (read before editing `ItemSettings`)
//!
//! Old preset files MUST keep loading when we add new fields. The rule:
//!
//! - Every field in `ItemSettings` must have `#[serde(default)]` OR a
//!   `Default` impl that covers the whole struct (we use the latter via
//!   `#[serde(default)]` on the struct itself — see `item_settings.rs`).
//! - Never rename a field without also adding a `#[serde(alias = "old_name")]`.
//! - Never change a field's type in a way that breaks existing JSON
//!   (e.g. `u8` → enum). If you must, bump a format version and add a
//!   migration path in `load_all`.
//!
//! The `loads_old_preset_with_unknown_fields` and
//! `loads_empty_preset_as_defaults` tests at the bottom of this file are
//! the tripwires — if you add a field without `#[serde(default)]`, they
//! will fail and tell you what's missing.

use std::collections::HashMap;
use std::path::PathBuf;

use rayon::prelude::*;

use super::item_settings::ItemSettings;
use super::presets::{
    model_id_key, ModelPreset, PresetFile, PRESET_FORMAT_VERSION,
};
use super::settings::{PRUNR_PRESET, Settings};
use prunr_core::{ComposeMode, FillStyle, LineMode, LineStyle};

/// Name of the presets subdirectory under the app config dir.
const PRESETS_SUBDIR: &str = "prunr/presets";

/// File extension for preset files.
const PRESET_EXT: &str = "json";

/// Marker file written after the first successful seed of the curated
/// built-in presets. Presence = "don't seed again on this install." Lets
/// users delete any built-in they don't want without having it re-appear
/// on next launch. Bump the suffix (`_v2`, `_v3`, ...) when shipping a
/// fresh batch of curated presets — the old marker won't gate the new
/// batch.
const SEED_MARKER: &str = ".builtins_seeded_v2";

/// Resolve the presets directory, creating it if needed. Returns `None` if
/// the platform config dir can't be resolved (very rare).
pub(crate) fn presets_dir() -> Option<PathBuf> {
    let base = dirs::config_dir()?;
    let dir = base.join(PRESETS_SUBDIR);
    let _ = std::fs::create_dir_all(&dir);
    Some(dir)
}

/// Resolve the on-disk path for a preset name, rejecting "Prunr" (reserved)
/// and returning `None` when the config dir is unavailable.
pub(crate) fn preset_path(name: &str) -> Option<PathBuf> {
    if name.eq_ignore_ascii_case(PRUNR_PRESET) { return None; }
    let dir = presets_dir()?;
    Some(dir.join(format!("{}.{PRESET_EXT}", sanitize_filename(name))))
}

/// Sanitize a preset name into a filesystem-safe stem.
///
/// Strips path separators, parent-traversal (`..`), control chars, and the
/// leading-dot convention (no hidden files). Trims whitespace. Empty
/// results map to `preset` so a malformed name still yields a valid path.
pub fn sanitize_filename(name: &str) -> String {
    let mut out: String = name
        .trim()
        .chars()
        .filter(|c| !c.is_control() && !matches!(c, '/' | '\\' | '\0'))
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
    out
}

/// Wrap a v1 preset file (raw `ItemSettings` shape, no `format_version`
/// key) into a v2 `PresetFile`. v1 files predate per-model bundling, so
/// they carry no model identity — every v1 file wraps under the
/// workspace default (`Settings::default().model`, currently
/// `BiRefNetLite`). The resolver pulls Prunr-floor brush/sd values at
/// apply-time, so missing brush/sd is not data loss.
///
/// Returns `None` when the JSON does not parse as `ItemSettings` — caller
/// logs and skips. Successful migrations are read-only on disk; the v2
/// shape only persists once the user explicitly re-saves the preset.
fn migrate_v1_to_v2(value: &serde_json::Value) -> Option<PresetFile> {
    // Reject non-objects up front — `#[serde(default)]` on ItemSettings
    // makes empty arrays / unrelated maps deserialize to a fully-default
    // ItemSettings, which would silently mask a corrupt file.
    if !value.is_object() {
        return None;
    }
    let item_settings: ItemSettings = serde_json::from_value(value.clone()).ok()?;
    // Settings::default().model is BiRefNetLite, whose to_model_id() is
    // Some(ModelId::BiRefNetLite) — invariant is local (settings.rs Default
    // impl + SettingsModel::to_model_id match arms).
    let wrap_key = Settings::default()
        .model
        .to_model_id()
        .expect("Settings::default().model always has a model_id");
    let mut models = HashMap::new();
    models.insert(model_id_key(wrap_key), ModelPreset {
        item_settings,
        brush: Default::default(),
        sd: None,
    });
    Some(PresetFile { format_version: PRESET_FORMAT_VERSION, models })
}

/// Load every preset file in the directory into a map. Malformed files
/// are skipped (with a `warn!` for the user's log) so 9 of 10 valid
/// presets still load when one is broken.
///
/// v1 files (raw `ItemSettings` shape, no `format_version` key) are
/// auto-wrapped into a v2 `PresetFile` envelope on load; the on-disk
/// file is untouched until the user explicitly re-saves it. v2 files
/// deserialize directly.
///
/// Called once at startup from `Settings::load` (before the first frame) and
/// once per preset delete to refresh the in-memory map. Not on the render path.
pub(crate) fn load_all() -> HashMap<String, PresetFile> {
    let Some(dir) = presets_dir() else { return HashMap::new() };
    let Ok(entries) = std::fs::read_dir(&dir) else { return HashMap::new() };

    let paths: Vec<PathBuf> = entries
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            (p.extension().and_then(|s| s.to_str()) == Some(PRESET_EXT))
                .then_some(p)
        })
        .collect();

    paths
        .par_iter()
        .filter_map(|path| {
            let name = path.file_stem()?.to_str()?.to_string();
            if name.eq_ignore_ascii_case(PRUNR_PRESET) { return None; }
            let file = load_from_path(path)?;
            Some((name, file))
        })
        .collect()
}

/// Parse a `serde_json::Value` into a `PresetFile`. v2 (has
/// `format_version: 2`) deserializes directly; anything else is
/// attempted as a v1 `ItemSettings` migration. Returns `None` when
/// neither path succeeds; the caller decides whether to log + skip.
fn parse_preset_value(value: serde_json::Value) -> Option<PresetFile> {
    if value.get("format_version").and_then(|v| v.as_u64()) == Some(2) {
        serde_json::from_value(value).ok()
    } else {
        migrate_v1_to_v2(&value)
    }
}

/// Load a single preset file from disk. Returns `None` when the file
/// is unreadable, malformed, or doesn't match either format. Used by
/// `load_all`'s parallel scan AND by callers that want to refresh a
/// single entry without rescanning the whole directory.
pub(crate) fn load_from_path(path: &std::path::Path) -> Option<PresetFile> {
    let data = std::fs::read(path)
        .map_err(|e| tracing::warn!(?path, %e, "skipping unreadable preset")).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&data)
        .map_err(|e| tracing::warn!(?path, %e, "skipping malformed-JSON preset")).ok()?;
    parse_preset_value(value).or_else(|| {
        tracing::warn!(?path, "skipping unrecognised preset shape (not v2 and not v1 ItemSettings)");
        None
    })
}

/// Write a preset to disk. Overwrites any existing file of the same name.
/// "Prunr" is rejected — that name is synthetic and cannot be persisted.
pub(crate) fn save(name: &str, values: &PresetFile) -> std::io::Result<()> {
    let path = preset_path(name).ok_or_else(|| {
        if name.eq_ignore_ascii_case(PRUNR_PRESET) {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "\"Prunr\" is a reserved preset name",
            )
        } else {
            std::io::Error::new(std::io::ErrorKind::NotFound, "config dir unavailable")
        }
    })?;
    let json = serde_json::to_string_pretty(values)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

fn merge_into(
    file: &mut PresetFile,
    model_id: prunr_models::ModelId,
    mp: ModelPreset,
) {
    file.format_version = PRESET_FORMAT_VERSION;
    file.models.insert(model_id_key(model_id), mp);
}

/// Read an existing preset file (v1 auto-migrated to v2) into a
/// `PresetFile`, or `PresetFile::default()` when the path is absent.
fn load_existing_for_merge(path: &std::path::Path) -> std::io::Result<PresetFile> {
    match std::fs::read(path) {
        Ok(data) => {
            let value: serde_json::Value = serde_json::from_slice(&data)
                .unwrap_or(serde_json::Value::Null);
            if value.is_null() {
                Ok(PresetFile::default())
            } else {
                Ok(parse_preset_value(value).unwrap_or_default())
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(PresetFile::default()),
        Err(e) => Err(e),
    }
}

/// Merge-aware save. Loads the existing file (v1 auto-migrated to v2),
/// updates the entry for `model_id` via `merge_into`, writes the
/// merged result. Preserves entries for other model_ids verbatim.
/// Scheduler bundles within `model_id`'s entry come straight from
/// `model_preset.sd.schedulers` — the caller is responsible for
/// carrying over any prior bundles it wants to retain.
///
/// If no existing file → behaves like `save` with a single-entry
/// PresetFile. The "save current as new preset" path skips this
/// function — it has nothing to merge against.
pub(crate) fn save_merged(
    name: &str,
    model_id: prunr_models::ModelId,
    model_preset: ModelPreset,
) -> std::io::Result<()> {
    let path = preset_path(name).ok_or_else(|| {
        if name.eq_ignore_ascii_case(PRUNR_PRESET) {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "\"Prunr\" is a reserved preset name",
            )
        } else {
            std::io::Error::new(std::io::ErrorKind::NotFound, "config dir unavailable")
        }
    })?;
    let mut file = load_existing_for_merge(&path)?;
    merge_into(&mut file, model_id, model_preset);
    let json = serde_json::to_string_pretty(&file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&path, json)
}

#[cfg(test)]
fn save_merged_to_path(
    path: &std::path::Path,
    model_id: prunr_models::ModelId,
    model_preset: ModelPreset,
) -> std::io::Result<()> {
    let mut file = load_existing_for_merge(path)?;
    merge_into(&mut file, model_id, model_preset);
    let json = serde_json::to_string_pretty(&file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(path, json)
}

/// Remove a preset's file. Silent no-op if the file doesn't exist — caller
/// already drops the in-memory entry, and we don't want to surface "already
/// deleted" as an error.
pub fn delete(name: &str) -> std::io::Result<()> {
    let Some(path) = preset_path(name) else { return Ok(()); };
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// Write the curated built-in presets to disk on first run. No-op when the
/// seed marker already exists — users can delete any built-in and the
/// deletion sticks.
pub fn seed_builtins_once() {
    let Some(dir) = presets_dir() else { return };
    let marker = dir.join(SEED_MARKER);
    if marker.exists() { return; }
    let wrap_key = Settings::default()
        .model
        .to_model_id()
        .expect("Settings::default().model always has a model_id");
    let wrap_key = model_id_key(wrap_key);
    for (name, item_settings) in builtin_presets() {
        // Skip if a preset file with this name already exists. Protects
        // user customisations against two edge cases:
        //   1. A mid-seed crash (some presets saved, marker not yet written)
        //      → next launch re-seeds, but won't overwrite files saved in
        //      the previous partial run.
        //   2. A future `_v3` seed that reuses a name the user edited under
        //      `_v2` — keeps their version, skips the rename.
        let Some(path) = preset_path(name) else { continue };
        if path.exists() { continue; }
        let mut models = HashMap::new();
        models.insert(wrap_key.clone(), ModelPreset {
            item_settings,
            brush: Default::default(),
            sd: None,
        });
        let file = PresetFile { format_version: PRESET_FORMAT_VERSION, models };
        let _ = save(name, &file);
    }
    let _ = std::fs::write(&marker, b"");
}

/// Factory list of curated named presets. Each covers a visually-distinct
/// look and showcases a combination of the compose / line / fill enums
/// that a user would otherwise need several chip clicks to discover.
fn builtin_presets() -> Vec<(&'static str, ItemSettings)> {
    let base = ItemSettings::default();
    vec![
        ("Comic", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::LinesOnly,
            line_strength: 0.7,
            edge_thickness: 2,
            solid_line_color: Some([0, 0, 0]),
            fill_style: FillStyle::Posterize { levels: 4 },
            gamma: 1.3,
            ..base
        }),
        ("Pencil Sketch", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::LinesOnly,
            line_strength: 0.8,
            edge_thickness: 0,
            solid_line_color: None,
            fill_style: FillStyle::Desaturate,
            gamma: 1.1,
            ..base
        }),
        ("Neon Glow", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::Ghost,
            line_strength: 0.6,
            edge_thickness: 3,
            line_style: LineStyle::Rainbow { cycles: 2 },
            fill_style: FillStyle::Saturate { percent: 220 },
            ..base
        }),
        ("Sepia", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::LinesOnly,
            line_strength: 0.6,
            edge_thickness: 1,
            solid_line_color: Some([70, 45, 20]),
            fill_style: FillStyle::Sepia,
            ..base
        }),
        ("Duotone Poster", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::SubjectFilled,
            line_strength: 0.6,
            edge_thickness: 2,
            solid_line_color: Some([10, 10, 40]),
            fill_style: FillStyle::Duotone { dark: [20, 20, 60], light: [240, 220, 180] },
            ..base
        }),
        ("X-Ray", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::Ghost,
            line_strength: 0.8,
            edge_thickness: 1,
            solid_line_color: Some([200, 230, 255]),
            fill_style: FillStyle::Invert,
            bg: Some([0, 0, 30, 255]),
            ..base
        }),
        ("Pop Art", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::SubjectFilled,
            line_strength: 0.65,
            edge_thickness: 3,
            solid_line_color: Some([0, 0, 0]),
            fill_style: FillStyle::Posterize { levels: 3 },
            gamma: 1.4,
            ..base
        }),
        ("Ghost", ItemSettings {
            line_mode: LineMode::SubjectOutline,
            compose_mode: ComposeMode::Ghost,
            line_strength: 0.55,
            edge_thickness: 1,
            solid_line_color: Some([230, 230, 235]),
            fill_style: FillStyle::Desaturate,
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
            fill_style: FillStyle::Pixelate { block_size: 10 },
            ..base
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_strips_path_separators() {
        assert_eq!(sanitize_filename("foo/bar"), "foobar");
        assert_eq!(sanitize_filename("foo\\bar"), "foobar");
    }

    #[test]
    fn sanitize_rejects_parent_traversal() {
        // "../etc": slash stripped → "..etc" → leading dots stripped → "etc".
        // Output is confined to presets dir (no slashes possible anymore).
        assert_eq!(sanitize_filename("../etc"), "etc");
        // "foo..bar": no leading dot, so embedded ".." replaced with "_".
        assert_eq!(sanitize_filename("foo..bar"), "foo_bar");
    }

    #[test]
    fn sanitize_strips_leading_dot() {
        assert_eq!(sanitize_filename(".hidden"), "hidden");
    }

    #[test]
    fn sanitize_strips_control_chars() {
        let name = "good\x00bad\x01name";
        assert_eq!(sanitize_filename(name), "goodbadname");
    }

    #[test]
    fn sanitize_trims_whitespace() {
        assert_eq!(sanitize_filename("  Portrait  "), "Portrait");
    }

    #[test]
    fn sanitize_empty_falls_back() {
        assert_eq!(sanitize_filename(""), "preset");
        assert_eq!(sanitize_filename("   "), "preset");
    }

    #[test]
    fn save_rejects_prunr_name() {
        let result = save("Prunr", &PresetFile::default());
        assert!(result.is_err());
        let result = save("prunr", &PresetFile::default());
        assert!(result.is_err());
    }

    #[test]
    fn delete_prunr_is_noop() {
        // Would be disastrous to delete anything when a user toggles the
        // Prunr row, so this must be a safe no-op.
        assert!(delete(PRUNR_PRESET).is_ok());
    }

    // ── Forward-compat tripwires ──────────────────────────────────────
    //
    // These tests fail loudly if `ItemSettings` loses its forward-compat
    // properties. If you added a field and one of these broke, the fix is
    // almost always `#[serde(default)]` on the struct (already present) or
    // on the new field. See the module docstring.

    #[test]
    fn loads_empty_preset_as_defaults() {
        // An empty JSON object must deserialize to full defaults — proves
        // every field has a default value and old presets (pre-dating new
        // fields) still load.
        let parsed: ItemSettings = serde_json::from_str("{}")
            .expect("ItemSettings must deserialize from `{}` — add #[serde(default)] to any new field");
        assert_eq!(parsed, ItemSettings::default());
    }

    #[test]
    fn loads_preset_with_unknown_fields() {
        // A preset file carrying fields we don't know about (e.g. a
        // forward-compat field added by a newer build) must still load.
        let json = r#"{
            "definitely_not_a_real_field": 42,
            "another_future_field": "hello"
        }"#;
        let parsed: ItemSettings = serde_json::from_str(json)
            .expect("ItemSettings must ignore unknown fields — do NOT add #[serde(deny_unknown_fields)]");
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

    #[test]
    fn preset_roundtrips_through_json() {
        // A preset must survive serialize → deserialize unchanged. If this
        // breaks, a field serializes to something it can't parse back.
        let original = ItemSettings::default();
        let json = serde_json::to_string(&original).expect("must serialize");
        let parsed: ItemSettings = serde_json::from_str(&json).expect("must deserialize");
        assert_eq!(parsed, original);
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
    fn v2_preset_file_round_trips_through_save_load_pure() {
        // Pure round-trip (no disk): build a v2 PresetFile, serialize via
        // the same code save() uses, deserialize via the same code load_all
        // uses, assert byte-stable shape. Pins the on-disk format contract
        // without touching the user's real config dir.
        let mut original = PresetFile::default();
        original.models.insert(
            model_id_key(prunr_models::ModelId::SdV15InpaintFp16),
            ModelPreset::default(),
        );
        let json = serde_json::to_string_pretty(&original).expect("serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse");
        // load_all branches on format_version == 2 → direct deserialize.
        assert_eq!(value.get("format_version").and_then(|v| v.as_u64()), Some(2));
        let restored: PresetFile = serde_json::from_value(value).expect("deserialize");
        assert_eq!(restored, original);
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

    #[test]
    fn save_writes_v2_envelope_to_disk() {
        // End-to-end: save a v2 PresetFile to a tmpdir-backed path and
        // assert the on-disk JSON has `format_version: 2` and `models`.
        // Bypasses presets_dir() (which goes to the real user config) by
        // writing directly via the same logic save() uses internally.
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("Test.json");
        let mut original = PresetFile::default();
        original.models.insert(
            model_id_key(prunr_models::ModelId::BiRefNetLite),
            ModelPreset::default(),
        );
        let json = serde_json::to_string_pretty(&original).expect("serialize");
        std::fs::write(&path, json).expect("write");
        let read_back = std::fs::read_to_string(&path).expect("read");
        let value: serde_json::Value = serde_json::from_str(&read_back).expect("parse");
        assert_eq!(value.get("format_version").and_then(|v| v.as_u64()), Some(2));
        assert!(value.get("models").is_some());
    }

    // ── Merge-save semantics ──────────────────────────────────────────
    //
    // Overwriting a preset on (active_model, active_scheduler) must
    // touch ONLY that model's entry and ONLY that scheduler's bundle.
    // Other models and other scheduler bundles round-trip bytewise.
    // Tests use `save_merged_to_path` so they don't write to the
    // user's real presets dir.

    #[test]
    fn save_merged_preserves_other_model_entries() {
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
        save_merged_to_path(&path, ModelId::SdV15InpaintFp16, new_sd_mp.clone()).unwrap();

        let reloaded = load_from_path(&path).expect("reload");
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
    fn save_merged_preserves_other_scheduler_bundles_within_sd() {
        use super::super::brush_state::SdScheduler;
        use super::super::presets::{SdPreset, SdSchedulerBundle};
        use prunr_models::ModelId;

        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("Foo.json");

        let lcm_bundle = SdSchedulerBundle {
            steps: 8, guidance_scale: 1.5, use_karras_sigmas: false, strength: 1.0,
        };
        let ddim_bundle_old = SdSchedulerBundle {
            steps: 20, guidance_scale: 7.5, use_karras_sigmas: false, strength: 1.0,
        };
        let mut schedulers_seed = HashMap::new();
        schedulers_seed.insert(SdScheduler::Lcm, lcm_bundle);
        schedulers_seed.insert(SdScheduler::Ddim, ddim_bundle_old);
        let seed_sd = SdPreset {
            prompt: "old".into(),
            negative_prompt: "old-neg".into(),
            use_taesd: Some(false),
            seed: Some(11),
            active_scheduler: SdScheduler::Lcm,
            schedulers: schedulers_seed,
        };
        let seed_mp = ModelPreset {
            sd: Some(seed_sd),
            ..ModelPreset::default()
        };
        let mut seed_file = PresetFile::default();
        seed_file.models.insert(model_id_key(ModelId::SdV15InpaintFp16), seed_mp);
        std::fs::write(&path, serde_json::to_string_pretty(&seed_file).unwrap()).unwrap();

        // Save-merge: caller only touched DDIM. LCM must survive.
        let ddim_bundle_new = SdSchedulerBundle {
            steps: 30, guidance_scale: 8.0, use_karras_sigmas: true, strength: 0.85,
        };
        let mut new_schedulers = HashMap::new();
        new_schedulers.insert(SdScheduler::Ddim, ddim_bundle_new);
        let new_sd = SdPreset {
            prompt: "new".into(),
            negative_prompt: "new-neg".into(),
            use_taesd: Some(true),
            seed: Some(99),
            active_scheduler: SdScheduler::Ddim,
            schedulers: new_schedulers,
        };
        let new_mp = ModelPreset { sd: Some(new_sd), ..ModelPreset::default() };
        save_merged_to_path(&path, ModelId::SdV15InpaintFp16, new_mp).unwrap();

        let reloaded = load_from_path(&path).expect("reload");
        let mp_back = reloaded
            .models
            .get(&model_id_key(ModelId::SdV15InpaintFp16))
            .expect("SD entry");
        let sd_back = mp_back.sd.as_ref().expect("SD bundle");

        // `save_merged` merges by ModelPreset, not by scheduler — the
        // caller decides which scheduler bundles to carry over inside
        // the SdPreset they pass. This test pins the round-trip: the
        // scheduler-bundle map is written exactly as the caller built
        // it, with no silent rewrite. (Cross-MODEL preservation is
        // pinned by `save_merged_preserves_other_model_entries`.)
        assert_eq!(
            sd_back.schedulers.get(&SdScheduler::Ddim).copied(),
            Some(ddim_bundle_new),
            "Ddim bundle equals the freshly-merged values",
        );
        assert_eq!(sd_back.active_scheduler, SdScheduler::Ddim);
        assert_eq!(sd_back.prompt, "new");
        assert_eq!(sd_back.negative_prompt, "new-neg");
        assert_eq!(sd_back.use_taesd, Some(true));
        assert_eq!(sd_back.seed, Some(99));
    }

    #[test]
    fn save_merged_creates_file_when_none_exists() {
        use prunr_models::ModelId;

        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("Fresh.json");
        assert!(!path.exists());

        let mp = ModelPreset {
            item_settings: ItemSettings { gamma: 1.9, ..ItemSettings::default() },
            ..ModelPreset::default()
        };
        save_merged_to_path(&path, ModelId::Silueta, mp.clone()).unwrap();

        let reloaded = load_from_path(&path).expect("reload");
        assert_eq!(reloaded.format_version, PRESET_FORMAT_VERSION);
        assert_eq!(reloaded.models.len(), 1);
        assert_eq!(
            reloaded.models.get(&model_id_key(ModelId::Silueta)),
            Some(&mp),
        );
    }

    #[test]
    fn save_merged_handles_v1_file_via_load_path() {
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
        save_merged_to_path(&path, ModelId::LaMaFp32, lama_mp.clone()).unwrap();

        let reloaded = load_from_path(&path).expect("reload");
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
    fn save_merged_rejects_prunr_name() {
        use prunr_models::ModelId;
        let result = save_merged("Prunr", ModelId::Silueta, ModelPreset::default());
        let err = result.expect_err("Prunr is reserved");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        // Case-insensitive — same shape as `save`.
        let result = save_merged("prunr", ModelId::Silueta, ModelPreset::default());
        let err = result.expect_err("prunr is reserved");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn merge_save_dynamic_field_capture() {
        use super::super::brush_state::{BrushSettings, SdScheduler};
        use super::super::presets::split_brush_for_save;
        use prunr_models::ModelId;

        // Build a BrushSettings with every SD-tuning field mutated to a
        // distinctive value. If a future field is added that doesn't
        // serde-default — or that gets stripped at split/save — the
        // round-trip equality breaks.
        let base = BrushSettings {
            radius: 73.0,
            hardness: 0.42,
            sd_prompt: "a-distinctive-prompt".into(),
            sd_negative_prompt: "a-distinctive-neg-prompt".into(),
            sd_steps: 17,
            sd_guidance_scale: 5.25,
            sd_scheduler: SdScheduler::Ddim,
            sd_use_karras_sigmas: true,
            sd_strength: 0.66,
            sd_seed: Some(13579),
            sd_use_taesd: Some(true),
            ..Default::default()
        };

        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("Dynamic.json");

        let (brush_split, sd) = split_brush_for_save(&base, ModelId::SdV15InpaintFp16);
        let mp = ModelPreset {
            item_settings: ItemSettings::default(),
            brush: brush_split,
            sd,
        };
        save_merged_to_path(&path, ModelId::SdV15InpaintFp16, mp).unwrap();

        let reloaded = load_from_path(&path).expect("reload");
        let mp_back = reloaded
            .models
            .get(&model_id_key(ModelId::SdV15InpaintFp16))
            .expect("SD entry");

        let mut fused = mp_back.brush.clone();
        if let Some(sd) = mp_back.sd.as_ref() {
            // Manual fuse for test: copy SD fields back to brush
            fused.sd_scheduler = sd.active_scheduler;
            fused.sd_prompt = sd.prompt.clone();
            fused.sd_negative_prompt = sd.negative_prompt.clone();
            fused.sd_use_taesd = sd.use_taesd;
            fused.sd_seed = sd.seed;
            if let Some(bundle) = sd.schedulers.get(&sd.active_scheduler) {
                fused.sd_steps = bundle.steps;
                fused.sd_guidance_scale = bundle.guidance_scale;
                fused.sd_use_karras_sigmas = bundle.use_karras_sigmas;
                fused.sd_strength = bundle.strength;
            }
        }
        assert_eq!(fused, base, "every BrushSettings field must round-trip");
    }
}
