//! Persistent EP × model compatibility cache.
//!
//! Phase 19's `is_available()` filter answers "did the loaded ORT
//! compile this EP in?" but it doesn't know the EP can't actually run
//! a specific model — e.g. OpenVINO rejects Silueta's ONNX graph
//! because of cycles, but `OpenVINO::is_available()`
//! still returns true.
//!
//! Without a cache, every cold-start re-tries the doomed (EP, model)
//! pair and pays the failed-load tax (~5s for OpenVINO+Silueta on
//! this machine). One-shot: record failures persistently, skip them
//! on subsequent runs.
//!
//! Cache location: `<data>/prunr/ep_compat.json`.
//! Versioned by `CARGO_PKG_VERSION` — when the app updates, the cache
//! invalidates because the loaded ORT may have new EP capabilities. A
//! downloaded model's entry also names its files' checksums, so a new
//! download of it is tried again.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use crate::engine::EpKind;

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Serialize, Deserialize, Default)]
struct CacheFile {
    version: String,
    failures: HashMap<String, String>,
}

fn cache_path() -> Option<PathBuf> {
    prunr_models::data_dir().map(|d| d.join("ep_compat.json"))
}

/// `"{ep}::{model}"`, plus `"@{tag}"` for a downloaded model so a failure
/// sticks to the files that failed (`prunr_models::content_tag`). The EP
/// part keeps the historic `Display` strings ("OpenVINO", "CUDA",
/// "CoreML", "DirectML").
fn key(ep: EpKind, model: prunr_models::ModelId) -> String {
    key_with(ep, model, prunr_models::content_tag(model))
}

fn key_with(ep: EpKind, model: prunr_models::ModelId, tag: Option<u64>) -> String {
    match tag {
        Some(tag) => format!("{ep}::{model:?}@{tag:016x}"),
        None => format!("{ep}::{model:?}"),
    }
}

fn cache() -> &'static Mutex<HashMap<String, String>> {
    static CACHE: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(load_from_disk()))
}

fn load_from_disk() -> HashMap<String, String> {
    let Some(path) = cache_path() else { return HashMap::new() };
    let Ok(data) = std::fs::read_to_string(&path) else { return HashMap::new() };
    let Ok(file) = serde_json::from_str::<CacheFile>(&data) else { return HashMap::new() };
    if file.version != APP_VERSION { return HashMap::new(); }
    file.failures
}

fn save_to_disk(map: &HashMap<String, String>) {
    let Some(path) = cache_path() else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = CacheFile {
        version: APP_VERSION.to_string(),
        failures: map.clone(),
    };
    if let Ok(json) = serde_json::to_string_pretty(&file) {
        let _ = std::fs::write(&path, json);
    }
}

/// True when this (EP, model) combo is on the persistent skip list.
/// Cheap: in-memory hashmap lookup behind a Mutex.
pub(crate) fn is_known_failure(ep: EpKind, model: prunr_models::ModelId) -> bool {
    let map = cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    map.contains_key(&key(ep, model))
}

/// Idempotent — re-recording the same combo is a no-op (no extra disk
/// writes). Best-effort persistence; a failed write is logged but not
/// fatal — next session will re-discover the failure.
pub(crate) fn record_failure(ep: EpKind, model: prunr_models::ModelId, error: &str) {
    let k = key(ep, model);
    let mut map = cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if map.contains_key(&k) { return; }
    tracing::info!(%ep, ?model, %error, "recording EP failure to skip cache");
    map.insert(k, error.to_string());
    save_to_disk(&map);
}

/// Wipe the cache — used by `prunr --clear-ep-cache` and the future
/// Settings → Hardware "Reset" button. Returns the number of entries
/// removed for the caller's confirmation message.
pub fn clear() -> usize {
    let mut map = cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let n = map.len();
    map.clear();
    save_to_disk(&map);
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    // JSON-key shape is the on-disk contract — existing user
    // `ep_compat.json` files keyed by these strings must still match
    // post-refactor. Pin both EP-as-string and model-as-Debug fragments.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn key_format_is_stable_openvino() {
        assert_eq!(
            key(EpKind::OpenVino, prunr_models::ModelId::Silueta),
            "OpenVINO::Silueta",
        );
        assert!(key(EpKind::Cuda, prunr_models::ModelId::SdV15InpaintFp16)
            .starts_with("CUDA::SdV15InpaintFp16@"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn key_format_is_stable_coreml() {
        assert_eq!(
            key(EpKind::CoreMl, prunr_models::ModelId::Silueta),
            "CoreML::Silueta",
        );
    }

    #[cfg(windows)]
    #[test]
    fn key_format_is_stable_directml() {
        assert!(key(EpKind::DirectMl, prunr_models::ModelId::SdV15InpaintFp16)
            .starts_with("DirectML::SdV15InpaintFp16@"));
    }

    /// A failure belongs to the files that failed: a fixed or re-exported
    /// download of the same model is a different key, so it gets tried
    /// again instead of staying on the CPU for good.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_new_download_of_a_model_is_a_new_key() {
        let lcm = prunr_models::ModelId::SdV15LcmInpaintFp16;
        let old = key_with(EpKind::OpenVino, lcm, Some(1));
        assert_ne!(old, key_with(EpKind::OpenVino, lcm, Some(2)));
        assert!(old.starts_with("OpenVINO::SdV15LcmInpaintFp16@"));
        assert_eq!(key_with(EpKind::OpenVino, prunr_models::ModelId::Silueta, None), "OpenVINO::Silueta",
            "built-in models keep the historic key; the app version scopes them");
        assert_eq!(key(EpKind::OpenVino, lcm), key_with(EpKind::OpenVino, lcm, prunr_models::content_tag(lcm)));
    }

    #[test]
    fn cache_file_json_shape_is_stable() {
        // Pre-existing user cache JSON — must still parse and round-trip
        // unchanged so an upgrade doesn't wipe `ep_compat.json` entries.
        let stale = CacheFile {
            version: "0.0.0-stale".to_string(),
            failures: HashMap::from([(
                "OpenVINO::Silueta".to_string(),
                "graph cycles".to_string(),
            )]),
        };
        let json = serde_json::to_string(&stale).unwrap();
        assert!(json.contains("\"OpenVINO::Silueta\""));
        assert!(json.contains("\"version\":\"0.0.0-stale\""));
        let parsed: CacheFile = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.version, "0.0.0-stale");
        assert_ne!(parsed.version, APP_VERSION);
        assert_eq!(
            parsed.failures.get("OpenVINO::Silueta").map(String::as_str),
            Some("graph cycles"),
        );
    }
}
