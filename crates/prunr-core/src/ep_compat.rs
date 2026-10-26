//! What this machine learned about each model on each accelerator: the
//! loads that failed, and where Settings › Hardware › Test this computer
//! found each model runs best.
//!
//! `is_available()` only says the EP is compiled in, not that it can run
//! a given model (OpenVINO rejects Silueta's graph for cycles), so failed
//! pairs are recorded and skipped instead of paying the failed load
//! (~5 s for OpenVINO+Silueta) on every cold start. A test's places stand
//! in for the registry's (`engine::places_here`).
//!
//! Cache location: `<data>/prunr/ep_compat.json`.
//! Scoped to what a failure says something about: the loaded runtime
//! build, the graphics drivers ([`set_drivers`]) and how sessions are set
//! up (`SESSION_SETUP`). Any changing starts the list afresh; another app
//! update does not, so what failed is not tried again after every update.
//! Each entry names the model's files (`prunr_models::files_id`), so a new
//! copy of a model is tried again. A failure the device had, not the
//! model (out of memory, a device removed or hung), is never recorded.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::engine::{EpKind, Weights};

static DRIVERS: OnceLock<String> = OnceLock::new();

/// Names the graphics drivers this process runs on, so a failure
/// recorded under one driver is tried again under the next. Call it
/// before the first session; later calls are ignored.
pub fn set_drivers(identity: String) {
    let _ = DRIVERS.set(identity);
}

/// Bumped by a release that changes how sessions are set up (provider
/// options, cache handling), so the failures it may have fixed are tried
/// again; an app update that leaves it alone keeps the list.
const SESSION_SETUP: u32 = 1;

/// What a recorded failure holds for: this way of setting sessions up,
/// runtime build and drivers.
fn scope() -> String {
    scope_of(crate::ort_runtime::runtime_tag(), DRIVERS.get().map(String::as_str))
}

fn scope_of(runtime: Option<&str>, drivers: Option<&str>) -> String {
    format!("setup {SESSION_SETUP} ort {} drivers {}", runtime.unwrap_or("unknown"), drivers.unwrap_or(""))
}

/// Error text of a failure the device had rather than the model: memory
/// ran out, the GPU was removed, hung or reset (DXGI 887A0005-7, as
/// DirectML reports them), or CoreML found the compile it was saving
/// already saved by another session building at the same time. The next
/// session may well succeed.
pub fn is_transient(error: &str) -> bool {
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
    // The on-disk key is named for what the scope once was: the app version.
    #[serde(rename = "version")]
    scope: String,
    #[serde(flatten)]
    learned: Learned,
}

/// Each failing pair's key and error.
type Failures = HashMap<String, String>;

/// What the file holds for one scope.
#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq)]
struct Learned {
    failures: Failures,
    /// Each place a test measured, by `measured_key`.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    measured: HashMap<String, Measured>,
    /// When the last test finished, in seconds since 1970.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tested: Option<u64>,
}

/// Where a test found a model runs on one accelerator, as the registry's
/// `Place` says it, and the measurement behind it.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Measured {
    pub kind: PlaceKind,
    pub text: String,
}

/// `Place::There`, or the `Why` of `Place::Cpu`, by name.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlaceKind {
    There,
    Fails,
    Differs,
    Slower,
    NoGain,
}

/// The reason a place measured here gives, wherever a `Place` is shown.
pub const MEASURED_HERE: &str = "measured on this computer (Settings › Hardware › Test this computer)";

impl PlaceKind {
    pub fn place(self) -> prunr_models::Place {
        use prunr_models::{Place, Why};
        match self {
            Self::There => Place::There(MEASURED_HERE),
            Self::Fails => Place::Cpu(Why::Fails(MEASURED_HERE)),
            Self::Differs => Place::Cpu(Why::Differs(MEASURED_HERE)),
            Self::Slower => Place::Cpu(Why::Slower(MEASURED_HERE)),
            Self::NoGain => Place::Cpu(Why::NoGain(MEASURED_HERE)),
        }
    }
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
/// `"@{files}"` so a failure sticks to the files that failed. The EP part
/// keeps the historic `Display` strings ("OpenVINO", "CUDA", "CoreML",
/// "DirectML"), plus `"+gpu"` for an OpenVINO session on the Intel GPU, so
/// a GPU failure does not bar OpenVINO on the CPU once the switch is off.
fn key(ep: EpKind, model: prunr_models::ModelId, weights: Weights) -> String {
    key_with(ep, model, weights, &prunr_models::files_id(model), on_intel_gpu(ep, model))
}

/// Whether `model`'s session on `ep` asks for the Intel GPU: OpenVINO
/// engines with the switch on. Fills stay on the CPU, and SD parts carry
/// their own device.
fn on_intel_gpu(ep: EpKind, model: prunr_models::ModelId) -> bool {
    #[cfg(not(target_os = "macos"))]
    {
        ep == EpKind::OpenVino && crate::engine::on_intel_gpu(model)
    }
    #[cfg(target_os = "macos")]
    {
        let _ = (ep, model);
        false
    }
}

fn key_with(ep: EpKind, model: prunr_models::ModelId, weights: Weights, files: &str, intel_gpu: bool) -> String {
    let copy = match weights {
        Weights::Fp32 => "",
        Weights::Fp16 => "/fp16",
        Weights::Int8 => "/int8",
    };
    let device = if intel_gpu { "+gpu" } else { "" };
    format!("{ep}{device}::{model:?}{copy}@{files}")
}

/// `"{column}::{model}@{files}"`: a place belongs to the files measured,
/// as a failure does.
fn measured_key(column: prunr_models::Column, model: prunr_models::ModelId) -> String {
    format!("{}::{model:?}@{}", column.field(), prunr_models::files_id(model))
}

/// What was learned as last read or written, and the file's stamp then.
#[derive(Default)]
struct Store {
    learned: Learned,
    stamp: Option<(SystemTime, u64)>,
}

fn cache() -> &'static Mutex<Store> {
    static CACHE: OnceLock<Mutex<Store>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut store = Store::default();
        if let Some(path) = cache_path() {
            store.stamp = stamp(&path);
            store.learned = load_from(&path, &scope()).unwrap_or_default();
        }
        Mutex::new(store)
    })
}

fn stamp(path: &Path) -> Option<(SystemTime, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

/// What the file at `path` holds under `scope`; nothing for another
/// scope, `None` when the file cannot be read.
fn load_from(path: &Path, scope: &str) -> Option<Learned> {
    let file = serde_json::from_str::<CacheFile>(&std::fs::read_to_string(path).ok()?).ok()?;
    Some(if file.scope == scope { file.learned } else { Learned::default() })
}

/// Written whole and renamed into place, so another process reading it
/// never sees half a file.
fn save_to(path: &Path, scope: &str, learned: &Learned) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = CacheFile { scope: scope.to_string(), learned: learned.clone() };
    let Ok(json) = serde_json::to_string_pretty(&file) else { return };
    let draft = path.with_extension(format!("json.{}", std::process::id()));
    if let Err(e) = std::fs::write(&draft, json).and_then(|()| std::fs::rename(&draft, path)) {
        tracing::warn!(%e, path = %path.display(), "could not save the EP failure list");
        let _ = std::fs::remove_file(&draft);
    }
}

fn save(store: &mut Store) {
    let Some(path) = cache_path() else { return };
    save_to(&path, &scope(), &store.learned);
    store.stamp = stamp(&path);
}

/// Change what is learned on top of what any process wrote last, and save
/// it: the test's process and the app's both write the file.
fn change<R>(edit: impl FnOnce(&mut Learned) -> R) -> R {
    let mut store = cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(path) = cache_path() {
        store.refresh_from(&path, &scope());
    }
    let out = edit(&mut store.learned);
    save(&mut store);
    out
}

/// Take in what another process wrote since this one last read the
/// file: a worker sees the list Settings reset, and does not write the
/// failures it held back over it; the app sees a test's places. Session
/// builds call it, never a frame.
pub fn refresh() {
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
        if let Some(learned) = load_from(path, scope) {
            self.learned = learned;
            self.stamp = now;
        }
    }
}

/// Read the store from disk now, if not yet read.
pub(crate) fn warm() {
    let _ = cache();
}

/// True when these weights of the model are on the EP's persistent skip
/// list.
pub(crate) fn is_known_failure(ep: EpKind, model: prunr_models::ModelId, weights: Weights) -> bool {
    // Before the lock: the key asks where the model runs, which reads the
    // places measured here under the same lock.
    let key = key(ep, model, weights);
    let store = cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    store.learned.failures.contains_key(&key)
}

/// Idempotent. A transient failure is logged and not recorded. A failed
/// write is logged, not fatal: the next session re-discovers the failure.
pub(crate) fn record_failure(ep: EpKind, model: prunr_models::ModelId, weights: Weights, error: &str) {
    if is_transient(error) {
        tracing::info!(%ep, ?model, ?weights, %error, "EP failure looks transient; not recorded");
        return;
    }
    let k = key(ep, model, weights);
    {
        let store = cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if store.learned.failures.contains_key(&k) { return; }
    }
    tracing::info!(%ep, ?model, ?weights, %error, "recording EP failure to skip cache");
    change(|learned| learned.failures.insert(k, error.to_string()));
}

/// Wipe the failures, so each failed pair is tried again. Returns the
/// number of failures removed.
pub fn clear() -> usize {
    change(|learned| std::mem::take(&mut learned.failures).len())
}

/// Where a test measured `model` runs in `column`, for its files on disk
/// now.
pub fn measured(model: prunr_models::ModelId, column: prunr_models::Column) -> Option<Measured> {
    let store = cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    store.learned.measured.get(&measured_key(column, model)).cloned()
}

/// Every place a test measured for the models' files on disk now, and
/// when the last test finished (seconds since 1970).
pub fn measured_here() -> (Option<u64>, Vec<(prunr_models::ModelId, prunr_models::Column, Measured)>) {
    let store = cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let places = prunr_models::REGISTRY.iter()
        .flat_map(|d| prunr_models::Column::ALL.map(|column| (d.id, column)))
        .filter_map(|(model, column)| store.learned.measured.get(&measured_key(column, model)).map(|m| (model, column, m.clone())))
        .collect();
    (store.learned.tested, places)
}

pub fn set_measured(model: prunr_models::ModelId, column: prunr_models::Column, measured: Measured) {
    change(|learned| learned.measured.insert(measured_key(column, model), measured));
}

/// A test finished at `when` (seconds since 1970).
pub fn set_tested(when: u64) {
    change(|learned| learned.tested = Some(when));
}

/// Forget every place a test measured: the registry's apply again.
/// Returns how many there were.
pub fn clear_measured() -> usize {
    change(|learned| {
        learned.tested = None;
        std::mem::take(&mut learned.measured).len()
    })
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
            key_with(EpKind::OpenVino, prunr_models::ModelId::Silueta, Weights::Fp32, "v1.0.0", false),
            "OpenVINO::Silueta@v1.0.0",
        );
        assert!(key(EpKind::Cuda, prunr_models::ModelId::SdV15InpaintFp16, Weights::Fp32)
            .starts_with("CUDA::SdV15InpaintFp16@"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn key_format_is_stable_coreml() {
        assert_eq!(
            key(EpKind::CoreMl, prunr_models::ModelId::Silueta, Weights::Fp32),
            format!("CoreML::Silueta@app{}", prunr_models::BUILT_IN_VERSION),
        );
    }

    #[cfg(windows)]
    #[test]
    fn key_format_is_stable_directml() {
        assert!(key(EpKind::DirectMl, prunr_models::ModelId::SdV15InpaintFp16, Weights::Fp32)
            .starts_with("DirectML::SdV15InpaintFp16@"));
    }

    /// A failure belongs to the files that failed: a fixed or re-exported
    /// download, or a built-in model of another version, is a different
    /// key, so it gets tried again instead of staying on the CPU for good.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_new_copy_of_a_model_is_a_new_key() {
        let lcm = prunr_models::ModelId::SdV15LcmInpaintFp16;
        let old = key_with(EpKind::OpenVino, lcm, Weights::Fp32, "0000000000000001", false);
        assert_ne!(old, key_with(EpKind::OpenVino, lcm, Weights::Fp32, "0000000000000002", false));
        assert!(key(EpKind::OpenVino, lcm, Weights::Fp32).starts_with("OpenVINO::SdV15LcmInpaintFp16@"));
        let silueta = prunr_models::ModelId::Silueta;
        assert_eq!(key(EpKind::OpenVino, silueta, Weights::Fp32), format!("OpenVINO::Silueta@app{}", prunr_models::BUILT_IN_VERSION),
            "a built-in model by the app it ships with");
    }

    /// A failure on the Intel GPU is its own key: turning the switch off
    /// tries OpenVINO on the CPU again. Fills never ask for the GPU.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn an_intel_gpu_failure_has_its_own_key() {
        let silueta = prunr_models::ModelId::Silueta;
        assert_eq!(key_with(EpKind::OpenVino, silueta, Weights::Fp32, "v1", true), "OpenVINO+gpu::Silueta@v1");
        assert_ne!(
            key_with(EpKind::OpenVino, silueta, Weights::Fp32, "v1", true),
            key_with(EpKind::OpenVino, silueta, Weights::Fp32, "v1", false),
        );
        if crate::engine::openvino_on_gpu() {
            assert!(on_intel_gpu(EpKind::OpenVino, prunr_models::ModelId::RealEsrganX4Plus), "faster on the graphics");
            assert!(!on_intel_gpu(EpKind::OpenVino, silueta), "kept off the graphics");
        }
        assert!(!on_intel_gpu(EpKind::OpenVino, prunr_models::ModelId::LaMaFp32), "a fill stays on the CPU");
        assert!(!on_intel_gpu(EpKind::Cuda, silueta));
    }

    /// Each copy of the weights fails on its own key.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn each_copy_of_the_weights_has_its_own_key() {
        let u2net = prunr_models::ModelId::U2net;
        let keys = Weights::ALL.map(|w| key_with(EpKind::OpenVino, u2net, w, "0000000000005ca0", false));
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

    /// A failure holds only for the runtime build and drivers it happened
    /// under: a driver update or another runtime tries the provider
    /// again, an app update alone does not.
    #[test]
    fn a_failure_is_forgotten_under_other_drivers_or_runtime() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ep_compat.json");
        let then = scope_of(Some("1.24.1-aaaa"), Some("10de:2684:32.0.15.6094"));
        assert_eq!(then, format!("setup {SESSION_SETUP} ort 1.24.1-aaaa drivers 10de:2684:32.0.15.6094"), "no app version in it: an update keeps the list");
        let learned = Learned {
            failures: Failures::from([("DirectML::U2net".to_string(), "broke".to_string())]),
            measured: HashMap::from([("directml::Silueta@app1".to_string(), Measured { kind: PlaceKind::Slower, text: "0.8x".into() })]),
            tested: Some(1),
        };
        save_to(&path, &then, &learned);
        assert_eq!(load_from(&path, &then), Some(learned));
        assert_eq!(load_from(&path, &scope_of(Some("1.24.1-aaaa"), Some("10de:2684:32.0.15.7000"))), Some(Learned::default()));
        assert_eq!(load_from(&path, &scope_of(Some("1.24.1-bbbb"), Some("10de:2684:32.0.15.6094"))), Some(Learned::default()));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1, "the save leaves no draft");
    }

    /// The list Settings resets in the GUI process reaches a worker that
    /// had read it earlier, before the worker builds its next session.
    #[test]
    fn a_reset_by_another_process_is_seen_on_refresh() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ep_compat.json");
        let learned = Learned { failures: Failures::from([("DirectML::U2net".to_string(), "broke".to_string())]), ..Learned::default() };
        save_to(&path, "scope", &learned);
        let mut worker = Store { learned, stamp: stamp(&path) };
        worker.refresh_from(&path, "scope");
        assert_eq!(worker.learned.failures.len(), 1, "an unchanged file keeps the list");
        std::thread::sleep(std::time::Duration::from_millis(20));
        save_to(&path, "scope", &Learned::default());
        worker.refresh_from(&path, "scope");
        assert!(worker.learned.failures.is_empty(), "the other process's reset is taken in");
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
        assert_eq!(parsed.learned.failures.len(), 1);
        assert!(parsed.learned.measured.is_empty(), "a file from before tests has none");
    }

    #[test]
    fn cache_file_json_shape_is_stable() {
        // Pre-existing user cache JSON — must still parse and round-trip
        // unchanged so an upgrade doesn't wipe `ep_compat.json` entries.
        let stale = CacheFile {
            scope: "0.0.0-stale".to_string(),
            learned: Learned {
                failures: HashMap::from([(
                    "OpenVINO::Silueta".to_string(),
                    "graph cycles".to_string(),
                )]),
                ..Learned::default()
            },
        };
        let json = serde_json::to_string(&stale).unwrap();
        assert_eq!(json, r#"{"version":"0.0.0-stale","failures":{"OpenVINO::Silueta":"graph cycles"}}"#, "nothing new in a file without tests");
        let parsed: CacheFile = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.scope, "0.0.0-stale");
        assert_eq!(
            parsed.learned.failures.get("OpenVINO::Silueta").map(String::as_str),
            Some("graph cycles"),
        );
    }
}
