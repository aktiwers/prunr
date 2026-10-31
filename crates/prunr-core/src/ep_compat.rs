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
/// runtime build, drivers and the add-ons registered beside the runtime.
fn scope() -> String {
    scope_of(crate::ort_runtime::runtime_tag(), DRIVERS.get().map(String::as_str), crate::ort_runtime::addons_tag().as_deref())
}

fn scope_of(runtime: Option<&str>, drivers: Option<&str>, addons: Option<&str>) -> String {
    let values = [&SESSION_SETUP.to_string(), runtime.unwrap_or("unknown"), drivers.unwrap_or(""), addons.unwrap_or("")];
    let parts: Vec<String> = ScopePart::ALL.iter().zip(values).map(|(part, value)| format!("{} {value}", part.label())).collect();
    parts.join(" ")
}

/// A part of what a recorded failure or measurement holds for, in the
/// order a scope is written; each opens with its label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScopePart {
    /// How Prunr sets sessions up (`SESSION_SETUP`).
    Setup,
    /// The runtime build.
    Ort,
    /// The graphics drivers.
    Drivers,
    /// The provider add-ons registered beside the runtime.
    Addons,
}

impl ScopePart {
    const ALL: [Self; 4] = [Self::Setup, Self::Ort, Self::Drivers, Self::Addons];

    fn label(self) -> &'static str {
        match self {
            Self::Setup => "setup",
            Self::Ort => "ort",
            Self::Drivers => "drivers",
            Self::Addons => "addons",
        }
    }
}

/// Each part's value in `scope`; a part a scope was written without (one
/// from before it existed) reads empty.
fn parts_of(scope: &str) -> [&str; ScopePart::ALL.len()] {
    let start = |i: usize| -> Option<(usize, usize)> {
        let label = ScopePart::ALL[i].label();
        if i == 0 {
            return scope.starts_with(&format!("{label} ")).then_some((0, label.len() + 1));
        }
        scope.find(&format!(" {label} ")).map(|at| (at, label.len() + 2))
    };
    let starts: Vec<Option<(usize, usize)>> = (0..ScopePart::ALL.len()).map(start).collect();
    std::array::from_fn(|i| {
        let Some((at, marker)) = starts[i] else { return "" };
        let end = starts[i + 1..].iter().flatten().map(|&(next, _)| next).find(|&next| next >= at + marker).unwrap_or(scope.len());
        scope[at + marker..end].trim()
    })
}

/// The parts that differ between scopes `old` and `new`.
fn changed_parts(old: &str, new: &str) -> Vec<ScopePart> {
    let (old, new) = (parts_of(old), parts_of(new));
    ScopePart::ALL
        .into_iter()
        .zip(old.into_iter().zip(new))
        .filter(|&(part, (a, b))| match part {
            ScopePart::Addons => addons_changed(a, b),
            _ => a != b,
        })
        .map(|(part, _)| part)
        .collect()
}

/// Whether an add-on in both lists of tags ("cuda-…, openvino-…") is
/// another version in `new`. One installed or removed is no change:
/// nothing measured ran on it before, and what ran on it holds again if
/// it comes back.
fn addons_changed(old: &str, new: &str) -> bool {
    fn tags(list: &str) -> Vec<(&str, &str)> {
        list.split(',').filter_map(|tag| tag.trim().split_once('-')).collect()
    }
    let new = tags(new);
    tags(old).into_iter().any(|(addon, version)| new.iter().any(|&(other, again)| other == addon && again != version))
}

/// Error text of a failure the device had rather than the model: memory
/// ran out, the GPU was removed, hung or reset (DXGI 887A0005-7, as
/// DirectML reports them), or CoreML found the compile it was saving
/// already saved by another session building at the same time. The next
/// session may well succeed.
pub fn is_transient(error: &str) -> bool {
    is_out_of_memory(error) || is_device_lost(error) || error.to_ascii_lowercase().contains("error copying compiled model to cache path")
}

/// Error text of a GPU that was removed, hung or reset.
pub(crate) fn is_device_lost(error: &str) -> bool {
    const MARKS: &[&str] = &[
        "887a0005", "887a0006", "887a0007", "device removed", "device_removed", "device hung", "device_hung",
        "has been suspended", "will not respond to more commands",
    ];
    let lower = error.to_ascii_lowercase();
    MARKS.iter().any(|m| lower.contains(m))
}

/// Error text of a run or session that ran out of memory, the card's or
/// the system's: a smaller piece of work, or the processor, may fit.
/// `is_out_of_memory`, of the card's memory rather than the system's: a
/// failed allocation in the host's C++ runtime names no card.
pub(crate) fn is_card_out_of_memory(error: &str) -> bool {
    const HOST: &[&str] = &["bad_alloc", "bad allocation"];
    let lower = error.to_ascii_lowercase();
    is_out_of_memory(error) && !HOST.iter().any(|m| lower.contains(m))
}

pub(crate) fn is_out_of_memory(error: &str) -> bool {
    const MARKS: &[&str] = &["out of memory", "outofmemory", "8007000e", "bad allocation", "bad_alloc", "failed to allocate"];
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
    /// What changed since this computer was tested, when that change
    /// dropped what a test measured; cleared by the next test of the
    /// whole computer, or Forget.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    changed: Vec<ScopePart>,
    /// The app said what `changed` holds; a part not yet in it is said
    /// when it changes.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    change_told: bool,
}

impl Learned {
    /// A test, or Forget, leaves nothing changed since the test to say.
    fn forget_change(&mut self) {
        self.changed.clear();
        self.change_told = false;
    }
}

/// Where a test found a model runs on one accelerator, as the registry's
/// `Place` says it, and the measurement behind it.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Measured {
    pub kind: PlaceKind,
    /// The reason, or what the times leave out; empty when nothing.
    pub text: String,
    /// Milliseconds a run took there, when it ran to the end and was timed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ms: Option<f32>,
    /// Milliseconds the same run took on the CPU, when the test timed it:
    /// kept for a model that fails there too, so the CPU's time is known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_ms: Option<f32>,
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

impl Measured {
    pub fn new(kind: PlaceKind, text: impl Into<String>) -> Self {
        Self { kind, text: text.into(), ms: None, cpu_ms: None }
    }

    /// A place found by timing it against the CPU: `ms` there, `cpu_ms`
    /// on the CPU.
    pub fn timed(kind: PlaceKind, ms: f32, cpu_ms: f32, text: impl Into<String>) -> Self {
        Self { kind, text: text.into(), ms: Some(ms), cpu_ms: Some(cpu_ms) }
    }

    /// With the times the test took, where it took them.
    pub fn with_times(self, ms: Option<f32>, cpu_ms: Option<f32>) -> Self {
        Self { ms, cpu_ms, ..self }
    }

    /// As the registry's `Place` says it. A faster place needs its times,
    /// which are what set it against another accelerator's.
    pub fn place(&self) -> Option<prunr_models::Place> {
        use prunr_models::{Place, Speed, Why};
        Some(match self.kind {
            PlaceKind::There => {
                Place::There(Speed::on(self.ms?, self.cpu_ms?, MEASURED_HERE))
            }
            PlaceKind::Fails => Place::Cpu(Why::Fails(MEASURED_HERE)),
            PlaceKind::Differs => Place::Cpu(Why::Differs(MEASURED_HERE)),
            PlaceKind::Slower => Place::Cpu(Why::Slower(MEASURED_HERE)),
            PlaceKind::NoGain => Place::Cpu(Why::NoGain(MEASURED_HERE)),
        })
    }
}

impl std::fmt::Display for Measured {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let times = self.ms.zip(self.cpu_ms).map(|(ms, cpu_ms)| prunr_models::times_text(ms, cpu_ms));
        match (times, self.text.is_empty()) {
            (Some(times), true) => f.write_str(&times),
            (Some(times), false) => write!(f, "{times}, {}", self.text),
            (None, _) => f.write_str(&self.text),
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
/// "DirectML"), plus `"+gpu"` for an OpenVINO session on the Intel GPU
/// (`engine::asks_intel_gpu`), so a GPU failure does not bar OpenVINO on
/// the CPU once the switch is off, and Intel's driver after it.
fn key(ep: EpKind, model: prunr_models::ModelId, weights: Weights, intel_gpu: bool) -> String {
    key_with(ep, model, weights, &prunr_models::files_id(model), intel_gpu.then(intel_graphics_driver))
}

/// `graphics`: the session ran on the Intel graphics, under this driver.
fn key_with(ep: EpKind, model: prunr_models::ModelId, weights: Weights, files: &str, graphics: Option<&str>) -> String {
    let copy = match weights {
        Weights::Fp32 => "",
        Weights::Fp16 => "/fp16",
        Weights::Int8 => "/int8",
    };
    let device = graphics.map_or(String::new(), |driver| format!("+gpu{}", under(driver)));
    format!("{ep}{device}::{model:?}{copy}@{files}")
}

/// `"{column}::{model}@{files}"`: a place belongs to the files measured,
/// as a failure does; on the Intel graphics, to Intel's driver too.
fn measured_key(column: prunr_models::Column, model: prunr_models::ModelId, files: &str) -> String {
    let driver = if column == prunr_models::Column::OpenVinoGpu { intel_graphics_driver() } else { "" };
    measured_key_with(column, model, files, driver)
}

fn measured_key_with(column: prunr_models::Column, model: prunr_models::ModelId, files: &str, driver: &str) -> String {
    format!("{}{}::{model:?}@{files}", column.field(), under(driver))
}

/// `"~{driver}"`, or nothing for a driver not read.
fn under(driver: &str) -> String {
    if driver.is_empty() { String::new() } else { format!("~{driver}") }
}

/// Intel's GPU driver for OpenCL, which OpenVINO runs the Intel graphics
/// on, as a tag; empty where none is read, and off Linux. A record of
/// OpenVINO on the Intel graphics holds for it alone, so its update tries
/// those again and leaves every other engine's records. Not in the scope
/// with the other drivers: on a computer with an NVIDIA card as well, it
/// would drop CUDA's too.
fn intel_graphics_driver() -> &'static str {
    static DRIVER: OnceLock<String> = OnceLock::new();
    DRIVER.get_or_init(|| {
        #[cfg(target_os = "linux")]
        {
            intel_opencl_driver(Path::new("/etc/OpenCL/vendors")).unwrap_or_default()
        }
        #[cfg(not(target_os = "linux"))]
        {
            String::new()
        }
    })
}

/// Each Intel driver `vendors` lists (`/etc/OpenCL/vendors`), by the size
/// and time of the library it names, which its update changes; `None`
/// without one. Not the kernel's Intel module, which every kernel update
/// moves.
#[cfg(any(target_os = "linux", test))]
fn intel_opencl_driver(vendors: &Path) -> Option<String> {
    let stamps: Vec<(u64, u64)> = std::fs::read_dir(vendors)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "icd") && path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("intel")))
        .filter_map(|driver| crate::ort_runtime::file_stamp(Path::new(std::fs::read_to_string(driver).ok()?.trim())))
        .collect();
    let (len, modified) = stamps.iter().fold((0, 0), |(len, modified), &(l, m)| (len + l, modified.max(m)));
    (!stamps.is_empty()).then(|| crate::ort_runtime::tag_for(Some("intel"), len, modified))
}

/// What was learned as last read or written, and the file's stamp then.
#[derive(Default)]
struct Store {
    learned: Learned,
    stamp: Option<(SystemTime, u64)>,
}

/// The store, locked. Nothing here reaches into the engine while it holds
/// it: the engine asks for the measured places under the same lock.
fn locked() -> std::sync::MutexGuard<'static, Store> {
    cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn cache() -> &'static Mutex<Store> {
    static CACHE: OnceLock<Mutex<Store>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut store = Store::default();
        if let Some(path) = cache_path() {
            store.stamp = stamp(&path);
            store.learned = store.load(&path, &scope()).unwrap_or_default();
        }
        Mutex::new(store)
    })
}

fn stamp(path: &Path) -> Option<(SystemTime, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

/// What the file at `path` holds under `scope`, and whether it was
/// written under another setup and so dropped. A dropped file that held
/// a test's measurements leaves what changed since (`Learned::changed`).
/// A scope differing only in how it is written (a part added since) is
/// the same setup. `None` when the file cannot be read.
fn load_from(path: &Path, scope: &str) -> Option<(Learned, bool)> {
    let file = serde_json::from_str::<CacheFile>(&std::fs::read_to_string(path).ok()?).ok()?;
    let changed = changed_parts(&file.scope, scope);
    if changed.is_empty() {
        return Some((file.learned, false));
    }
    let old = file.learned;
    let was_tested = old.tested.is_some() || !old.measured.is_empty() || !old.changed.is_empty();
    let (mut kept, mut told) = (old.changed, old.change_told);
    if was_tested {
        for part in changed {
            if !kept.contains(&part) {
                kept.push(part);
                told = false;
            }
        }
    }
    tracing::info!(from = %file.scope, to = %scope, "the setup changed since the EP list was written; it starts afresh");
    Some((Learned { changed: kept, change_told: told, ..Learned::default() }, true))
}

/// Whether a change since this computer was tested is yet to be said;
/// `true` once, whichever process found it.
pub fn take_untold_change() -> bool {
    let untold = |learned: &Learned| !learned.changed.is_empty() && !learned.change_told;
    if !untold(&locked().learned) {
        return false;
    }
    change(|learned| {
        let now = untold(learned);
        learned.change_told = true;
        now
    })
}

/// What changed since this computer was tested; empty when nothing did
/// or it was tested since.
pub fn changed_since_test() -> Vec<ScopePart> {
    locked().learned.changed.clone()
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

/// Change what is learned on top of what any process wrote last, and save
/// it: the test's process and the app's both write the file. The write
/// happens outside the store's lock, which the engine takes every frame
/// while a batch runs; this process's changes are written one at a time.
fn change<R>(edit: impl FnOnce(&mut Learned) -> R) -> R {
    static WRITING: Mutex<()> = Mutex::new(());
    let _writing = WRITING.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let path = cache_path();
    let scope = scope();
    let (out, learned) = {
        let mut store = locked();
        if let Some(path) = &path {
            store.refresh_from(path, || scope.clone());
        }
        let out = edit(&mut store.learned);
        (out, store.learned.clone())
    };
    if let Some(path) = path {
        save_to(&path, &scope, &learned);
        locked().stamp = stamp(&path);
    }
    out
}

/// Take in what another process wrote since this one last read the
/// file: a worker sees the list Settings reset, and does not write the
/// failures it held back over it; the app sees a test's places. Session
/// builds call it, never a frame.
pub fn refresh() {
    let Some(path) = cache_path() else { return };
    locked().refresh_from(&path, scope);
}

impl Store {
    /// `scope` is asked for only when the file changed since it was read.
    fn refresh_from(&mut self, path: &Path, scope: impl FnOnce() -> String) {
        let now = stamp(path);
        if now.is_none() || now == self.stamp {
            return;
        }
        if let Some(learned) = self.load(path, &scope()) {
            self.learned = learned;
        }
    }

    /// `load_from`, and a drop written back at once under this scope, so
    /// what changed is said once and kept until the next test. The stamp
    /// is the one before the read: a write by another process while it
    /// reads is read at the next refresh, not taken as read.
    fn load(&mut self, path: &Path, scope: &str) -> Option<Learned> {
        let read_at = stamp(path);
        let (learned, dropped) = load_from(path, scope)?;
        self.stamp = if dropped {
            save_to(path, scope, &learned);
            stamp(path)
        } else {
            read_at
        };
        Some(learned)
    }
}

/// Read the store from disk now, if not yet read.
pub(crate) fn warm() {
    let _ = cache();
}

/// True when these weights of the model are on the EP's persistent skip
/// list; `intel_gpu`: the session asks for the Intel GPU.
pub(crate) fn is_known_failure(ep: EpKind, model: prunr_models::ModelId, weights: Weights, intel_gpu: bool) -> bool {
    let key = key(ep, model, weights, intel_gpu);
    locked().learned.failures.contains_key(&key)
}

/// Idempotent. A transient failure is logged and not recorded. A failed
/// write is logged, not fatal: the next session re-discovers the failure.
pub(crate) fn record_failure(ep: EpKind, model: prunr_models::ModelId, weights: Weights, intel_gpu: bool, error: &str) {
    if is_transient(error) {
        tracing::info!(%ep, ?model, ?weights, %error, "EP failure looks transient; not recorded");
        return;
    }
    let k = key(ep, model, weights, intel_gpu);
    if locked().learned.failures.contains_key(&k) {
        return;
    }
    tracing::info!(%ep, ?model, ?weights, %error, "recording EP failure to skip cache");
    change(|learned| learned.failures.insert(k, error.to_string()));
}

/// Wipe the failures, so each failed pair is tried again. Returns the
/// number of failures removed.
pub fn clear() -> usize {
    change(|learned| std::mem::take(&mut learned.failures).len())
}

/// Where a test measured `model` runs, per column in `Column::ALL` order,
/// for its files on disk now. The engine asks every frame while a batch
/// runs: one lock, and no key built when nothing was measured.
pub fn measured_for(model: prunr_models::ModelId) -> [Option<prunr_models::Place>; prunr_models::Column::ALL.len()] {
    in_each_column(model, Measured::place)
}

/// What a test measured for `model` in each column, times included, for
/// its files on disk now.
pub fn measurements(model: prunr_models::ModelId) -> [Option<Measured>; prunr_models::Column::ALL.len()] {
    in_each_column(model, |measured| Some(measured.clone()))
}

fn in_each_column<T>(model: prunr_models::ModelId, read: impl Fn(&Measured) -> Option<T>) -> [Option<T>; prunr_models::Column::ALL.len()] {
    let store = locked();
    if store.learned.measured.is_empty() {
        return std::array::from_fn(|_| None);
    }
    let files = prunr_models::files_id(model);
    prunr_models::Column::ALL.map(|column| store.learned.measured.get(&measured_key(column, model, &files)).and_then(&read))
}

/// When the last test finished, in seconds since 1970.
pub fn tested() -> Option<u64> {
    locked().learned.tested
}

pub fn set_measured(model: prunr_models::ModelId, column: prunr_models::Column, measured: Measured) {
    let key = measured_key(column, model, &prunr_models::files_id(model));
    change(|learned| learned.measured.insert(key, measured));
}

/// A test finished now.
pub fn set_tested() {
    let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    change(|learned| {
        learned.tested = Some(now);
        learned.forget_change();
    });
}

/// Forget every place a test measured: the registry's apply again.
/// Returns how many there were.
pub fn clear_measured() -> usize {
    change(|learned| {
        learned.tested = None;
        learned.forget_change();
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
            key_with(EpKind::OpenVino, prunr_models::ModelId::Silueta, Weights::Fp32, "v1.0.0", None),
            "OpenVINO::Silueta@v1.0.0",
        );
        assert!(key(EpKind::Cuda, prunr_models::ModelId::SdV15InpaintFp16, Weights::Fp32, false)
            .starts_with("CUDA::SdV15InpaintFp16@"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn key_format_is_stable_coreml() {
        assert_eq!(
            key(EpKind::CoreMl, prunr_models::ModelId::Silueta, Weights::Fp32, false),
            format!("CoreML::Silueta@app{}", prunr_models::BUILT_IN_VERSION),
        );
    }

    #[cfg(windows)]
    #[test]
    fn key_format_is_stable_directml() {
        assert!(key(EpKind::DirectMl, prunr_models::ModelId::SdV15InpaintFp16, Weights::Fp32, false)
            .starts_with("DirectML::SdV15InpaintFp16@"));
    }

    /// A failure belongs to the files that failed: a fixed or re-exported
    /// download, or a built-in model of another version, is a different
    /// key, so it gets tried again instead of staying on the CPU for good.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_new_copy_of_a_model_is_a_new_key() {
        let lcm = prunr_models::ModelId::SdV15LcmInpaintFp16;
        let old = key_with(EpKind::OpenVino, lcm, Weights::Fp32, "0000000000000001", None);
        assert_ne!(old, key_with(EpKind::OpenVino, lcm, Weights::Fp32, "0000000000000002", None));
        assert!(key(EpKind::OpenVino, lcm, Weights::Fp32, false).starts_with("OpenVINO::SdV15LcmInpaintFp16@"));
        let silueta = prunr_models::ModelId::Silueta;
        assert_eq!(key(EpKind::OpenVino, silueta, Weights::Fp32, false), format!("OpenVINO::Silueta@app{}", prunr_models::BUILT_IN_VERSION),
            "a built-in model by the app it ships with");
    }

    /// A failure on the Intel GPU is its own key: turning the switch off
    /// tries OpenVINO on the CPU again. Fills never ask for the GPU.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn an_intel_gpu_failure_has_its_own_key() {
        let silueta = prunr_models::ModelId::Silueta;
        assert_eq!(key_with(EpKind::OpenVino, silueta, Weights::Fp32, "v1", Some("")), "OpenVINO+gpu::Silueta@v1");
        assert_eq!(key_with(EpKind::OpenVino, silueta, Weights::Fp32, "v1", Some("intel-1a2b3c4d")), "OpenVINO+gpu~intel-1a2b3c4d::Silueta@v1");
        assert_ne!(
            key_with(EpKind::OpenVino, silueta, Weights::Fp32, "v1", Some("")),
            key_with(EpKind::OpenVino, silueta, Weights::Fp32, "v1", None),
        );
        if crate::engine::openvino_on_gpu() {
            assert!(crate::engine::asks_intel_gpu(EpKind::OpenVino, prunr_models::ModelId::RealEsrganX4Plus), "faster on the graphics");
            assert!(!crate::engine::asks_intel_gpu(EpKind::OpenVino, silueta), "kept off the graphics");
        }
        assert!(!crate::engine::asks_intel_gpu(EpKind::OpenVino, prunr_models::ModelId::LaMaFp32), "a fill stays on the CPU");
        assert!(!crate::engine::asks_intel_gpu(EpKind::Cuda, silueta));
    }

    /// A new Intel graphics driver gives the records of OpenVINO on the
    /// Intel graphics new keys, so they are tried again, and leaves every
    /// other engine's.
    #[test]
    fn an_intel_driver_update_moves_only_the_intel_graphics_records() {
        use prunr_models::{Column, ModelId};
        let key = |column, driver| measured_key_with(column, ModelId::Teed, "v1", driver);
        assert_ne!(key(Column::OpenVinoGpu, "intel-1"), key(Column::OpenVinoGpu, "intel-2"));
        assert_eq!(measured_key(Column::Cuda, ModelId::Teed, "v1"), "cuda::Teed@v1", "no driver in another engine's");
        assert_eq!(measured_key(Column::OpenVino, ModelId::Teed, "v1"), "openvino::Teed@v1");
    }

    /// Intel's OpenCL driver is known by each Intel driver the vendors
    /// folder lists, by its library's size and time: an update of it is a
    /// new tag; another vendor's driver, or one whose library is gone, is
    /// none.
    #[test]
    fn intels_opencl_driver_is_known_by_its_library() {
        let vendors = tempfile::tempdir().unwrap();
        let libraries = tempfile::tempdir().unwrap();
        let driver = |icd: &str, library: &str, bytes: usize| {
            let path = libraries.path().join(library);
            std::fs::write(&path, vec![0u8; bytes]).unwrap();
            std::fs::File::options().write(true).open(&path).unwrap().set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000)).unwrap();
            std::fs::write(vendors.path().join(icd), format!("{}\n", path.display())).unwrap();
        };
        assert_eq!(intel_opencl_driver(vendors.path()), None, "none listed");
        driver("pocl.icd", "libpocl.so.2", 7);
        assert_eq!(intel_opencl_driver(vendors.path()), None, "another vendor's");
        driver("intel.icd", "libigdrcl.so", 31);
        driver("intel_legacy1.icd", "libigdrcl_legacy1.so", 9);
        let both = intel_opencl_driver(vendors.path()).expect("two Intel drivers");
        assert!(both.starts_with("intel-"), "{both}");
        driver("intel.icd", "libigdrcl.so", 32);
        let updated = intel_opencl_driver(vendors.path()).expect("an update");
        assert_ne!(updated, both, "an update is a new tag");
        std::fs::remove_file(libraries.path().join("libigdrcl_legacy1.so")).unwrap();
        assert_ne!(intel_opencl_driver(vendors.path()), Some(updated), "a library gone");
        assert_eq!(intel_opencl_driver(&vendors.path().join("missing")), None);
    }

    /// Each copy of the weights fails on its own key.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn each_copy_of_the_weights_has_its_own_key() {
        let u2net = prunr_models::ModelId::U2net;
        let keys = Weights::ALL.map(|w| key_with(EpKind::OpenVino, u2net, w, "0000000000005ca0", None));
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

    /// Running out of memory, as CUDA, ORT's pool and DirectML word it,
    /// is told apart from a lost device, which a smaller tile would not
    /// help.
    #[test]
    fn out_of_memory_is_told_apart_from_a_lost_device() {
        for oom in [
            "Non-zero status code returned while running Gather node. Status Message: CUDA error: out of memory",
            "bfc_arena.cc:360 Failed to allocate memory for requested buffer of size 268435456",
            "ORT session creation failed (DirectML): HRESULT 0x8007000E",
            "Exception during initialization: bad allocation",
        ] {
            assert!(is_out_of_memory(oom) && is_transient(oom), "{oom}");
        }
        for other in ["887A0005 The GPU device instance has been suspended.", "Graph contains cycles"] {
            assert!(!is_out_of_memory(other), "{other}");
        }
        assert!(is_card_out_of_memory("CUDA error: out of memory") && is_card_out_of_memory("HRESULT 0x8007000E"));
        assert!(!is_card_out_of_memory("Exception during initialization: bad allocation"), "the system's memory, not the card's");
    }

    /// A failure holds only for the runtime build and drivers it happened
    /// under: a driver update or another runtime tries the provider
    /// again, an app update alone does not.
    #[test]
    fn a_failure_is_forgotten_under_other_drivers_or_runtime() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ep_compat.json");
        let then = scope_of(Some("1.24.1-aaaa"), Some("10de:2684:32.0.15.6094"), None);
        assert_eq!(then, format!("setup {SESSION_SETUP} ort 1.24.1-aaaa drivers 10de:2684:32.0.15.6094 addons "), "no app version in it: an update keeps the list");
        let learned = Learned {
            failures: Failures::from([("DirectML::U2net".to_string(), "broke".to_string())]),
            measured: HashMap::from([("directml::Silueta@app1".to_string(), Measured::new(PlaceKind::Slower, "0.8x"))]),
            tested: Some(1),
            changed: Vec::new(),
            change_told: false,
        };
        save_to(&path, &then, &learned);
        let afresh = |changed: &[ScopePart]| Some((Learned { changed: changed.to_vec(), ..Learned::default() }, true));
        assert_eq!(load_from(&path, &then), Some((learned.clone(), false)));
        assert_eq!(load_from(&path, &scope_of(Some("1.24.1-aaaa"), Some("10de:2684:32.0.15.7000"), None)), afresh(&[ScopePart::Drivers]));
        assert_eq!(load_from(&path, &scope_of(Some("1.24.1-bbbb"), Some("10de:2684:32.0.15.6094"), None)), afresh(&[ScopePart::Ort]));
        let with_addon = scope_of(Some("1.24.1-aaaa"), Some("10de:2684:32.0.15.6094"), Some("cuda-0.2.0-1234"));
        assert_eq!(load_from(&path, &with_addon), Some((learned.clone(), false)), "an add-on installed keeps what was measured");
        save_to(&path, &with_addon, &learned);
        assert_eq!(load_from(&path, &then), Some((learned.clone(), false)), "and removed too");
        assert_eq!(load_from(&path, &scope_of(Some("1.24.1-aaaa"), Some("10de:2684:32.0.15.6094"), Some("cuda-0.3.0-5678"))), afresh(&[ScopePart::Addons]), "another version of it is a change");
        let both = scope_of(Some("1.24.1-aaaa"), Some("10de:2684:32.0.15.6094"), Some("cuda-0.2.0-1234, openvino-1.8.1-9abc"));
        assert_eq!(load_from(&path, &both), Some((learned.clone(), false)), "a second add-on beside it keeps what was measured");
        save_to(&path, &both, &learned);
        assert_eq!(load_from(&path, &with_addon), Some((learned.clone(), false)), "and so does removing one of two");
        let newer = scope_of(Some("1.24.1-aaaa"), Some("10de:2684:32.0.15.6094"), Some("cuda-0.2.0-1234, openvino-1.8.2-def0"));
        assert_eq!(load_from(&path, &newer), afresh(&[ScopePart::Addons]), "another version of either is a change");
        save_to(&path, &then, &learned);
        // A list written before the add-ons part existed is the same setup.
        save_to(&path, "setup 1 ort 1.24.1-aaaa drivers 10de:2684:32.0.15.6094", &learned);
        assert_eq!(load_from(&path, &then), Some((learned, false)));
        // What was said stays said; a part not yet said is said.
        let said = Learned { changed: vec![ScopePart::Drivers], change_told: true, ..Learned::default() };
        save_to(&path, &then, &said);
        let again = scope_of(Some("1.24.1-aaaa"), Some("10de:2684:32.0.15.9999"), None);
        assert_eq!(load_from(&path, &again).map(|(l, _)| (l.changed, l.change_told)), Some((vec![ScopePart::Drivers], true)));
        let more = scope_of(Some("1.24.1-cccc"), Some("10de:2684:32.0.15.9999"), None);
        assert_eq!(load_from(&path, &more).map(|(l, _)| (l.changed, l.change_told)), Some((vec![ScopePart::Drivers, ScopePart::Ort], false)));
        // An untested computer has nothing to say changed.
        save_to(&path, &then, &Learned { failures: Failures::from([("DirectML::U2net".to_string(), "broke".to_string())]), ..Learned::default() });
        assert_eq!(load_from(&path, &scope_of(Some("1.24.1-bbbb"), None, None)), afresh(&[]));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1, "the save leaves no draft");
    }

    /// What changed is kept by its scope label, as files already on disk
    /// hold it.
    #[test]
    fn what_changed_is_kept_by_its_label() {
        for part in ScopePart::ALL {
            assert_eq!(serde_json::to_string(&part).unwrap(), format!("\"{}\"", part.label()));
        }
        let learned: Learned = serde_json::from_str(r#"{"failures":{},"changed":["drivers","addons"]}"#).unwrap();
        assert_eq!(learned.changed, [ScopePart::Drivers, ScopePart::Addons]);
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
        worker.refresh_from(&path, || "scope".to_string());
        assert_eq!(worker.learned.failures.len(), 1, "an unchanged file keeps the list");
        std::thread::sleep(std::time::Duration::from_millis(20));
        save_to(&path, "scope", &Learned::default());
        worker.refresh_from(&path, || "scope".to_string());
        assert!(worker.learned.failures.is_empty(), "the other process's reset is taken in");
    }

    /// A place written for a model's files reads back in its column, and
    /// only there, a faster one with its times; a test's finish is stamped.
    #[test]
    fn a_measured_place_reads_back_in_its_column() {
        use prunr_models::{Column, ModelId, Place, Speed, Why};
        let model = ModelId::Teed;
        set_measured(model, Column::MiGraphX, Measured::new(PlaceKind::Slower, "0.8x"));
        set_measured(model, Column::Cuda, Measured::timed(PlaceKind::There, 12.0, 150.0, ""));
        set_tested();
        let found = measured_for(model);
        assert_eq!(found[Column::MiGraphX.index()], Some(Place::Cpu(Why::Slower(MEASURED_HERE))));
        assert_eq!(found[Column::Cuda.index()], Some(Place::There(Speed::on(12.0, 150.0, MEASURED_HERE))));
        assert_eq!(found.iter().flatten().count(), 2, "{found:?}");
        assert!(tested().is_some());
        assert_eq!(measured_for(ModelId::U2net), [None; Column::ALL.len()], "another model has none");
    }

    /// A faster place without its times has nothing to set against
    /// another accelerator's, so it reads as unmeasured; shown, the times
    /// come first.
    #[test]
    fn a_faster_place_needs_its_times() {
        assert_eq!(Measured::new(PlaceKind::There, "").place(), None);
        let timed = Measured::timed(PlaceKind::There, 300.0, 400.0, "within 10%");
        assert_eq!(timed.to_string(), "1.33x: 300 ms against the CPU's 400 ms, within 10%");
        assert_eq!(Measured::new(PlaceKind::Fails, "no kernel").to_string(), "no kernel");
    }

    #[test]
    fn a_transient_failure_is_not_recorded() {
        let model = prunr_models::ModelId::RealEsrganX2Plus;
        #[cfg(not(target_os = "macos"))]
        let ep = EpKind::OpenVino;
        #[cfg(target_os = "macos")]
        let ep = EpKind::CoreMl;
        record_failure(ep, model, Weights::Fp16, false, "CUDA failure 2: out of memory");
        assert!(!is_known_failure(ep, model, Weights::Fp16, false));
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
