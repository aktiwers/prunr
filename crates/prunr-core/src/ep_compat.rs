//! Persistent EP × model compatibility cache.
//!
//! `is_available()` only says the EP is compiled in, not that it can run
//! a given model (OpenVINO rejects Silueta's graph for cycles), so failed
//! pairs are recorded and skipped instead of paying the failed load
//! (~5 s for OpenVINO+Silueta) on every cold start.
//!
//! Cache location: `<data>/prunr/ep_compat.json`.
//! Scoped to what a failure says something about: the app version, the
//! loaded runtime build and the graphics drivers ([`set_drivers`]). Any
//! of them changing starts the list afresh. A downloaded model's entry
//! also names its files' checksums, so a new download of it is tried
//! again. A failure the device had, not the model (out of memory, a
//! device removed or hung), is never recorded.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::engine::{EpKind, Weights};

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

static DRIVERS: OnceLock<String> = OnceLock::new();

/// Names the graphics drivers this process runs on, so a failure
/// recorded under one driver is tried again under the next. Call it
/// before the first session; later calls are ignored.
pub fn set_drivers(identity: String) {
    let _ = DRIVERS.set(identity);
}

/// What a recorded failure holds for: this app version, runtime build
/// and drivers.
fn scope() -> String {
    scope_of(crate::ort_runtime::runtime_tag(), DRIVERS.get().map(String::as_str))
}

fn scope_of(runtime: Option<&str>, drivers: Option<&str>) -> String {
    format!("{APP_VERSION} ort {} drivers {}", runtime.unwrap_or("unknown"), drivers.unwrap_or(""))
}

/// Error text of a failure the device had rather than the model: memory
/// ran out, the GPU was removed, hung or reset (DXGI 887A0005-7, as
/// DirectML reports them), or CoreML found the compile it was saving
/// already saved by another session building at the same time. The next
/// session may well succeed.
pub(crate) fn is_transient(error: &str) -> bool {
    const MARKS: &[&str] = &[
        "out of memory", "outofmemory", "8007000e", "bad allocation", "bad_alloc", "failed to allocate",
        "887a0005", "887a0006", "887a0007", "device removed", "device_removed", "device hung", "device_hung",
        "has been suspended", "will not respond to more commands",
        "error copying compiled model to cache path",
    ];
    let error = error.to_ascii_lowercase();
    MARKS.iter().any(|m| error.contains(m))
}

#[derive(Serialize, Deserialize, Default)]
struct CacheFile {
    // The on-disk key predates the scope growing past the app version.
    #[serde(rename = "version")]
    scope: String,
    failures: HashMap<String, String>,
}

/// What the file holds for one scope.
#[derive(Debug, Default, Clone, PartialEq)]
struct Entries {
    failures: HashMap<String, String>,
}

fn cache_path() -> Option<PathBuf> {
    #[cfg(test)]
    {
        Some(std::env::temp_dir().join(format!("prunr-test-ep-compat-{}.json", std::process::id())))
    }
    #[cfg(not(test))]
    {
        prunr_models::data_dir().map(|d| d.join("ep_compat.json"))
    }
}

/// `"{ep}::{model}"`, then `"/fp16"` or `"/int8"` for a lighter copy of
/// the weights, so a failure of one copy leaves the others to try, and
/// `"@{tag}"` for a downloaded model so a failure sticks to the files that
/// failed (`prunr_models::content_tag`). The model's own weights keep the
/// historic key, so entries written before weights were told apart still
/// read, as failures of those. The EP part keeps the historic `Display`
/// strings ("OpenVINO", "CUDA", "CoreML", "DirectML"), plus `"+gpu"` for
/// an OpenVINO session on the Intel GPU, so a GPU failure does not bar
/// OpenVINO on the CPU once the switch is off.
fn key(ep: EpKind, model: prunr_models::ModelId, weights: Weights) -> String {
    key_with(ep, model, weights, prunr_models::content_tag(model), on_intel_gpu(ep, model))
}

/// Whether `model`'s session on `ep` asks for the Intel GPU: OpenVINO
/// engines with the switch on. Fills stay on the CPU, and SD parts carry
/// their own device.
fn on_intel_gpu(ep: EpKind, model: prunr_models::ModelId) -> bool {
    #[cfg(not(target_os = "macos"))]
    {
        ep == EpKind::OpenVino
            && crate::engine::openvino_on_gpu()
            && prunr_models::descriptor(model).is_some_and(|d| d.category != prunr_models::ModelCategory::Inpaint)
    }
    #[cfg(target_os = "macos")]
    {
        let _ = (ep, model);
        false
    }
}

fn key_with(ep: EpKind, model: prunr_models::ModelId, weights: Weights, tag: Option<u64>, intel_gpu: bool) -> String {
    let copy = match weights {
        Weights::Fp32 => "",
        Weights::Fp16 => "/fp16",
        Weights::Int8 => "/int8",
    };
    let device = if intel_gpu { "+gpu" } else { "" };
    match tag {
        Some(tag) => format!("{ep}{device}::{model:?}{copy}@{tag:016x}"),
        None => format!("{ep}{device}::{model:?}{copy}"),
    }
}

/// The entries as last read or written, and the file's stamp then.
#[derive(Default)]
struct Store {
    entries: Entries,
    stamp: Option<(SystemTime, u64)>,
}

fn cache() -> &'static Mutex<Store> {
    static CACHE: OnceLock<Mutex<Store>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut store = Store::default();
        if let Some(path) = cache_path() {
            store.stamp = stamp(&path);
            store.entries = load_from(&path, &scope()).unwrap_or_default();
        }
        Mutex::new(store)
    })
}

fn stamp(path: &Path) -> Option<(SystemTime, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

/// The entries in the file at `path` recorded under `scope`; empty for
/// another scope, `None` when the file cannot be read.
fn load_from(path: &Path, scope: &str) -> Option<Entries> {
    let file = serde_json::from_str::<CacheFile>(&std::fs::read_to_string(path).ok()?).ok()?;
    Some(if file.scope == scope { Entries { failures: file.failures } } else { Entries::default() })
}

/// Written whole and renamed into place, so another process reading it
/// never sees half a file.
fn save_to(path: &Path, scope: &str, entries: &Entries) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = CacheFile { scope: scope.to_string(), failures: entries.failures.clone() };
    let Ok(json) = serde_json::to_string_pretty(&file) else { return };
    let draft = path.with_extension(format!("json.{}", std::process::id()));
    if let Err(e) = std::fs::write(&draft, json).and_then(|()| std::fs::rename(&draft, path)) {
        tracing::warn!(%e, path = %path.display(), "could not save the EP failure list");
        let _ = std::fs::remove_file(&draft);
    }
}

fn save(store: &mut Store) {
    let Some(path) = cache_path() else { return };
    save_to(&path, &scope(), &store.entries);
    store.stamp = stamp(&path);
}

/// Take in what another process wrote since this one last read the
/// file: a worker sees the list Settings reset, and does not write the
/// failures it held back over it. Session builds call it, never a frame.
pub(crate) fn refresh() {
    let Some(path) = cache_path() else { return };
    let mut store = cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    store.refresh_from(&path, &scope());
}

impl Store {
    fn refresh_from(&mut self, path: &Path, scope: &str) {
        let now = stamp(path);
        if now.is_none() || now == self.stamp {
            return;
        }
        if let Some(entries) = load_from(path, scope) {
            self.entries = entries;
            self.stamp = now;
        }
    }
}

/// True when these weights of the model are on the EP's persistent skip
/// list.
/// Read the store from disk now, if not yet read.
pub(crate) fn warm() {
    let _ = cache();
}

pub(crate) fn is_known_failure(ep: EpKind, model: prunr_models::ModelId, weights: Weights) -> bool {
    failure(ep, model, weights).is_some()
}

/// Why these weights of the model are on the EP's skip list.
pub(crate) fn failure(ep: EpKind, model: prunr_models::ModelId, weights: Weights) -> Option<String> {
    let store = cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    store.entries.failures.get(&key(ep, model, weights)).cloned()
}

/// Idempotent. A transient failure is logged and not recorded. A failed
/// write is logged, not fatal: the next session re-discovers the failure.
pub(crate) fn record_failure(ep: EpKind, model: prunr_models::ModelId, weights: Weights, error: &str) {
    if is_transient(error) {
        tracing::info!(%ep, ?model, ?weights, %error, "EP failure looks transient; not recorded");
        return;
    }
    let k = key(ep, model, weights);
    let mut store = cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if store.entries.failures.contains_key(&k) { return; }
    tracing::info!(%ep, ?model, ?weights, %error, "recording EP failure to skip cache");
    store.entries.failures.insert(k, error.to_string());
    save(&mut store);
}

/// Wipe the cache, so each failed pair is tried again. Returns the number
/// of failures removed.
pub fn clear() -> usize {
    let mut store = cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let n = store.entries.failures.len();
    store.entries = Entries::default();
    save(&mut store);
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
            key_with(EpKind::OpenVino, prunr_models::ModelId::Silueta, Weights::Fp32, None, false),
            "OpenVINO::Silueta",
        );
        assert!(key(EpKind::Cuda, prunr_models::ModelId::SdV15InpaintFp16, Weights::Fp32)
            .starts_with("CUDA::SdV15InpaintFp16@"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn key_format_is_stable_coreml() {
        assert_eq!(
            key(EpKind::CoreMl, prunr_models::ModelId::Silueta, Weights::Fp32),
            "CoreML::Silueta",
        );
    }

    #[cfg(windows)]
    #[test]
    fn key_format_is_stable_directml() {
        assert!(key(EpKind::DirectMl, prunr_models::ModelId::SdV15InpaintFp16, Weights::Fp32)
            .starts_with("DirectML::SdV15InpaintFp16@"));
    }

    /// A failure belongs to the files that failed: a fixed or re-exported
    /// download of the same model is a different key, so it gets tried
    /// again instead of staying on the CPU for good.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_new_download_of_a_model_is_a_new_key() {
        let lcm = prunr_models::ModelId::SdV15LcmInpaintFp16;
        let old = key_with(EpKind::OpenVino, lcm, Weights::Fp32, Some(1), false);
        assert_ne!(old, key_with(EpKind::OpenVino, lcm, Weights::Fp32, Some(2), false));
        assert!(old.starts_with("OpenVINO::SdV15LcmInpaintFp16@"));
        assert_eq!(key_with(EpKind::OpenVino, prunr_models::ModelId::Silueta, Weights::Fp32, None, false), "OpenVINO::Silueta",
            "built-in models keep the historic key; the app version scopes them");
    }

    /// A failure on the Intel GPU is its own key: turning the switch off
    /// tries OpenVINO on the CPU again. Fills never ask for the GPU.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn an_intel_gpu_failure_has_its_own_key() {
        let silueta = prunr_models::ModelId::Silueta;
        assert_eq!(key_with(EpKind::OpenVino, silueta, Weights::Fp32, None, true), "OpenVINO+gpu::Silueta");
        assert_ne!(
            key_with(EpKind::OpenVino, silueta, Weights::Fp32, None, true),
            key_with(EpKind::OpenVino, silueta, Weights::Fp32, None, false),
        );
        if crate::engine::openvino_on_gpu() {
            assert!(on_intel_gpu(EpKind::OpenVino, silueta));
        }
        assert!(!on_intel_gpu(EpKind::OpenVino, prunr_models::ModelId::LaMaFp32), "a fill stays on the CPU");
        assert!(!on_intel_gpu(EpKind::Cuda, silueta));
    }

    /// Each copy of the weights fails on its own key; the model's own
    /// weights keep the key every earlier entry was written under.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn each_copy_of_the_weights_has_its_own_key() {
        let u2net = prunr_models::ModelId::U2net;
        let keys = Weights::ALL.map(|w| key_with(EpKind::OpenVino, u2net, w, Some(0x5ca0), false));
        assert_eq!(keys, [
            "OpenVINO::U2net@0000000000005ca0",
            "OpenVINO::U2net/fp16@0000000000005ca0",
            "OpenVINO::U2net/int8@0000000000005ca0",
        ]);
    }

    /// Out of memory or a lost device says nothing about the model; each
    /// way ORT, CUDA and DirectML word them is let through, and a real
    /// incompatibility is not.
    #[test]
    fn a_device_failure_is_transient_and_a_model_failure_is_not() {
        for transient in [
            "ORT session creation failed (CUDA): CUDA failure 2: out of memory",
            "ORT session creation failed (DirectML): ... 887A0005 The GPU device instance has been suspended.",
            "DirectML test run failed: ... 887A0006 The GPU will not respond to more commands",
            "ORT session creation failed (DirectML): HRESULT 0x8007000E",
            "Failed to allocate memory for requested buffer of size 1073741824",
            "bad allocation",
            "ORT session creation failed (CoreML): Error copying compiled model to cache path: /x/size-1-3-256-256/1/0_dynamic_mlprogram/model/compiled_model.mlmodelc, reason: \u{201c}model_9C5D.mlmodelc\u{201d} couldn\u{2019}t be moved to \u{201c}model\u{201d} because an item with the same name already exists.",
        ] {
            assert!(is_transient(transient), "{transient}");
        }
        for lasting in [
            "ORT session creation failed (OpenVINO): Graph contains cycles",
            "tensor(float16) is invalid",
            "DirectML test run failed: output is NaN",
        ] {
            assert!(!is_transient(lasting), "{lasting}");
        }
    }

    /// A failure holds only for the app version, runtime build and
    /// drivers it happened under: a driver update or another runtime
    /// tries the provider again.
    #[test]
    fn a_failure_is_forgotten_under_other_drivers_or_runtime() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ep_compat.json");
        let then = scope_of(Some("1.24.1-aaaa"), Some("10de:2684:32.0.15.6094"));
        let entries = Entries { failures: HashMap::from([("DirectML::U2net".to_string(), "broke".to_string())]) };
        save_to(&path, &then, &entries);
        assert_eq!(load_from(&path, &then), Some(entries));
        assert_eq!(load_from(&path, &scope_of(Some("1.24.1-aaaa"), Some("10de:2684:32.0.15.7000"))), Some(Entries::default()));
        assert_eq!(load_from(&path, &scope_of(Some("1.24.1-bbbb"), Some("10de:2684:32.0.15.6094"))), Some(Entries::default()));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1, "the save leaves no draft");
    }

    /// The list Settings resets in the GUI process reaches a worker that
    /// had read it earlier, before the worker builds its next session.
    #[test]
    fn a_reset_by_another_process_is_seen_on_refresh() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ep_compat.json");
        let entries = Entries { failures: HashMap::from([("DirectML::U2net".to_string(), "broke".to_string())]) };
        save_to(&path, "scope", &entries);
        let mut worker = Store { entries, stamp: stamp(&path) };
        worker.refresh_from(&path, "scope");
        assert_eq!(worker.entries.failures.len(), 1, "an unchanged file keeps the list");
        std::thread::sleep(std::time::Duration::from_millis(20));
        save_to(&path, "scope", &Entries::default());
        worker.refresh_from(&path, "scope");
        assert!(worker.entries.failures.is_empty(), "the other process's reset is taken in");
    }

    #[test]
    fn a_transient_failure_is_not_recorded() {
        let model = prunr_models::ModelId::RealEsrganX2Plus;
        #[cfg(not(target_os = "macos"))]
        let ep = EpKind::OpenVino;
        #[cfg(target_os = "macos")]
        let ep = EpKind::CoreMl;
        record_failure(ep, model, Weights::Fp16, "CUDA failure 2: out of memory");
        assert!(!is_known_failure(ep, model, Weights::Fp16));
    }

    /// A file written while checks against the CPU were kept beside the
    /// failures still reads, its failures kept.
    #[test]
    fn a_file_with_passed_checks_reads() {
        let parsed: CacheFile = serde_json::from_str(r#"{"version":"v","failures":{"CUDA::U2net":"broke"},"passed":["CUDA::Silueta"]}"#).unwrap();
        assert_eq!(parsed.failures.len(), 1);
    }

    #[test]
    fn cache_file_json_shape_is_stable() {
        // Pre-existing user cache JSON — must still parse and round-trip
        // unchanged so an upgrade doesn't wipe `ep_compat.json` entries.
        let stale = CacheFile {
            scope: "0.0.0-stale".to_string(),
            failures: HashMap::from([(
                "OpenVINO::Silueta".to_string(),
                "graph cycles".to_string(),
            )]),
        };
        let json = serde_json::to_string(&stale).unwrap();
        assert!(json.contains("\"OpenVINO::Silueta\""));
        assert!(json.contains("\"version\":\"0.0.0-stale\""));
        let parsed: CacheFile = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.scope, "0.0.0-stale");
        assert_ne!(parsed.scope, APP_VERSION);
        assert_eq!(
            parsed.failures.get("OpenVINO::Silueta").map(String::as_str),
            Some("graph cycles"),
        );
    }
}
