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
use super::settings::PRUNR_PRESET;

/// Name of the presets subdirectory under the app config dir.
const PRESETS_SUBDIR: &str = "prunr/presets";

/// File extension for preset files.
const PRESET_EXT: &str = "json";

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
fn preset_path(name: &str) -> Option<PathBuf> {
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

/// Load every preset file in the directory into a map. Invalid files are
/// silently skipped — we'd rather load 9 of 10 valid presets than reject
/// the whole batch because one file is malformed.
pub fn load_all() -> HashMap<String, ItemSettings> {
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
            let data = std::fs::read(path).ok()?;
            let values: ItemSettings = serde_json::from_slice(&data).ok()?;
            Some((name, values))
        })
        .collect()
}

/// Write a preset to disk. Overwrites any existing file of the same name.
/// "Prunr" is rejected — that name is synthetic and cannot be persisted.
pub fn save(name: &str, values: &ItemSettings) -> std::io::Result<()> {
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
        let result = save("Prunr", &ItemSettings::default());
        assert!(result.is_err());
        let result = save("prunr", &ItemSettings::default());
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
    fn preset_roundtrips_through_json() {
        // A preset must survive serialize → deserialize unchanged. If this
        // breaks, a field serializes to something it can't parse back.
        let original = ItemSettings::default();
        let json = serde_json::to_string(&original).expect("must serialize");
        let parsed: ItemSettings = serde_json::from_str(&json).expect("must deserialize");
        assert_eq!(parsed, original);
    }
}
