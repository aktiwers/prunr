//! ONNX Runtime dylib resolution and the one process-wide `ort` init.
//!
//! Resolution order, first hit wins:
//!   1. `ORT_DYLIB_PATH` env var (escape hatch + dev override). Set but
//!      not a file is an error, never a silent fall-through.
//!   2. The runtime [`PREFERRED_FILE`] names: a Runtime Store install
//!      (`<data>/prunr/runtimes/<ep>-<ver>-<rid>/`) or, as
//!      [`PREFER_BUILT_IN`], the one beside the executable (a macOS
//!      `.app` keeps it in `Contents/Frameworks/`)
//!   3. Without a preference: on Windows the built-in runtime (the
//!      DirectML build) before the store's first install by name;
//!      elsewhere the store's first install before the built-in one
//!
//! Every session in the workspace starts from [`session_builder`], which
//! runs [`ensure_initialized`] first. Under `load-dynamic`, building a
//! session before the runtime is loaded does not fail: ort re-enters its
//! own loader lock while formatting the "library not found" error and
//! the thread parks forever. Routing every builder through here turns
//! that hang into an error.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ort::session::builder::SessionBuilder;
use ort::session::Session;

use crate::types::CoreError;

pub const DYLIB_NAME: &str = if cfg!(windows) {
    "onnxruntime.dll"
} else if cfg!(target_os = "macos") {
    "libonnxruntime.dylib"
} else {
    "libonnxruntime.so"
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DylibSource {
    EnvVar,
    RuntimeStore,
    Bundled,
}

impl std::fmt::Display for DylibSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::EnvVar => "ORT_DYLIB_PATH",
            Self::RuntimeStore => "runtime-store",
            Self::Bundled => "bundled",
        })
    }
}

static INIT: OnceLock<Result<DylibSource, String>> = OnceLock::new();

/// For a test that builds a session: `Err(why)` to skip it where no
/// runtime is installed (the macOS runners). `PRUNR_REQUIRE_ORT=1`, set
/// where CI installs one, turns a missing runtime into a failure, so a
/// broken install cannot pass as a skip.
#[cfg(any(test, feature = "test-utils"))]
pub fn runtime_for_tests() -> Result<(), String> {
    let Err(why) = ensure_initialized() else { return Ok(()) };
    if crate::env::flag("PRUNR_REQUIRE_ORT") == Some(true) {
        panic!("PRUNR_REQUIRE_ORT=1 but no runtime: {why}");
    }
    Err(why)
}

/// Load the runtime once per process. Every later call returns the
/// first outcome, so a missing runtime surfaces as the same error at
/// every session site instead of a hang.
pub fn ensure_initialized() -> Result<DylibSource, String> {
    INIT.get_or_init(init).clone()
}

/// Where this process's runtime came from, once it has loaded; never
/// loads it.
pub fn loaded_source() -> Option<DylibSource> {
    INIT.get()?.as_ref().ok().copied()
}

/// Names the loaded runtime build ("1.24.1-3fa2c901"), once it has
/// loaded: its version and a fingerprint of the library file, which
/// every runtime upgrade changes.
pub fn runtime_tag() -> Option<&'static str> {
    static TAG: OnceLock<Option<String>> = OnceLock::new();
    TAG.get_or_init(|| {
        ensure_initialized().ok()?;
        let meta = std::fs::metadata(LOADED.get()?).ok()?;
        let modified = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
        Some(tag_for(runtime_version(), meta.len(), modified))
    }).as_deref()
}

/// The loaded runtime's version ("1.24.1"), as the library reports it.
/// Its build info names it only on some builds: PyPI's say
/// `git-branch=HEAD`.
pub fn runtime_version() -> Option<&'static str> {
    ensure_initialized().ok()?;
    VERSION.get()?.as_deref()
}

static VERSION: OnceLock<Option<String>> = OnceLock::new();

fn version_of(path: &Path) -> Option<String> {
    // SAFETY: `path` is the runtime already loaded, so this open finds it
    // loaded; `OrtGetApiBase` returns a static table whose version string
    // is static too, as ort reads it.
    unsafe {
        let lib = libloading::Library::new(path).ok()?;
        let entry = lib.get::<unsafe extern "C" fn() -> *const ort::sys::OrtApiBase>(b"OrtGetApiBase").ok()?;
        let base = entry();
        if base.is_null() {
            return None;
        }
        let version = ((*base).GetVersionString)();
        (!version.is_null()).then(|| std::ffi::CStr::from_ptr(version).to_string_lossy().into_owned())
    }
}

/// The library loaded by `init`.
static LOADED: OnceLock<PathBuf> = OnceLock::new();

/// The runtime library this process loaded, once it has.
pub fn loaded_dylib() -> Option<&'static Path> {
    LOADED.get().map(PathBuf::as_path)
}

pub(crate) fn tag_for(version: Option<&str>, len: u64, modified: u64) -> String {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in len.to_le_bytes().into_iter().chain(modified.to_le_bytes()) {
        hash = (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193);
    }
    format!("{}-{hash:08x}", version.unwrap_or("build"))
}

/// The process's ORT environment, which every session is built from. ort
/// allows one per process, so it lives as long as the process.
static ENV: OnceLock<ort::environment::Environment> = OnceLock::new();

/// A session builder on the loaded runtime.
pub fn session_builder() -> Result<SessionBuilder, CoreError> {
    ensure_initialized().map_err(CoreError::Inference)?;
    let env = ENV.get().ok_or_else(|| CoreError::Inference("ORT environment missing after load".into()))?;
    #[allow(clippy::disallowed_methods)] // the sanctioned wrapper: the runtime is loaded two lines above
    // No spin-waiting between ops: idle ORT threads spinning steal the cores
    // from rayon's tile blending and from a second session running at once.
    // Measured (4-core i7): LaMa 5.2 s -> 3.5 s alone,
    // LaMa + face restore 7.5 s -> 4.9 s, a Silueta cut-out 4.15 s -> 3.6 s;
    // an OpenVINO session is unchanged.
    let builder = Session::builder(env)
        .map_err(|e| CoreError::Inference(format!("ORT builder init failed: {e}")))?
        .with_intra_op_spinning(false)
        .map_err(|e| CoreError::Inference(format!("ORT builder init failed: {e}")))?;
    without_optimizers(builder, &[])
}

/// `builder` with `extra` of ORT's optimizers turned off, and constant
/// folding where it corrupts weights. ORT keeps one list and a second
/// call replaces the first, so every disabled optimizer goes through here.
pub(crate) fn without_optimizers(builder: SessionBuilder, extra: &[&str]) -> Result<SessionBuilder, CoreError> {
    match disabled_optimizers(runtime_version(), extra) {
        Some(list) => builder
            .with_disabled_optimizers(list)
            .map_err(|e| CoreError::Inference(format!("ORT disable optimizers failed: {e}"))),
        None => Ok(builder),
    }
}

fn disabled_optimizers(version: Option<&str>, extra: &[&str]) -> Option<String> {
    let folding = folding_is_broken(version).then_some("ConstantFolding");
    let list: Vec<&str> = folding.into_iter().chain(extra.iter().copied()).collect();
    (!list.is_empty()).then(|| list.join(";"))
}

/// From ORT 1.24.1 constant folding turns an aliasing op (a Reshape of a
/// weight) into a constant that points at memory freed later in the same
/// pass (microsoft/onnxruntime#29788, fixed by microsoft/onnxruntime#29789
/// in 1.29.0; 1.28.3 still lacks it): SAM 2.1's encoder came out as values
/// near 1e28 in some processes on a MacBook Air. 1.24 to 1.28, and a
/// runtime whose version cannot be read, count as affected. Off, the
/// cut-out, upscale, colorize and depth models ran as fast on the CPU and
/// on CUDA; SAM's encoder ~30% slower on the CPU.
fn folding_is_broken(version: Option<&str>) -> bool {
    let Some(version) = version else { return true };
    let mut parts = version.split('.').map(|p| p.parse::<u32>());
    match (parts.next(), parts.next()) {
        (Some(Ok(1)), Some(Ok(minor))) => (24..29).contains(&minor),
        _ => true,
    }
}

/// Libraries loaded from beside the runtime before the runtime itself.
/// The DirectML build of `onnxruntime.dll` delay-loads `DirectML.dll`.
/// ORT's hook looks beside itself first, but when that fails Windows
/// searches by name and finds System32's older copy, which rejects the
/// feature level ORT asks for. A copy already loaded is what every later
/// load by name resolves to.
const PRELOAD_BESIDE_RUNTIME: &[&str] = if cfg!(windows) { &["DirectML.dll"] } else { &[] };

fn init() -> Result<DylibSource, String> {
    let candidates = candidates()?;
    // After a failed load ort's library cell reads as loaded and holds
    // nothing, so ort is called once: each candidate is opened and checked
    // here, and ort gets the first that passes.
    let ((path, source), failures) = load_first(&candidates, probe_runtime)?;
    load_runtime(path)?;
    if !failures.is_empty() {
        tracing::warn!(failures = %failures.join("; "), %source, "runtime fallback");
        let _ = FELL_BACK.set(match source {
            DylibSource::Bundled => "The chosen runtime would not load, so Prunr uses the built-in one. Install it again in Settings.",
            _ => "The chosen runtime would not load, so Prunr uses another installed one. Install it again in Settings.",
        });
    }
    let version = version_of(path);
    tracing::info!(path = %path.display(), %source, ?version, "ORT runtime loaded");
    let _ = VERSION.set(version);
    let _ = LOADED.set(path.clone());
    Ok(*source)
}

/// Every runtime to try, in order. `ORT_DYLIB_PATH` names the only one:
/// an explicit runtime has no stand-in.
fn candidates() -> Result<Vec<(PathBuf, DylibSource)>, String> {
    let candidates = match env_override() {
        Some(path) if !path.is_file() => return Err(format!("ORT_DYLIB_PATH={} is not a file", path.display())),
        Some(path) => vec![(path, DylibSource::EnvVar)],
        None => runtime_candidates(prunr_models::runtime_store_dir().as_deref(), bundled_dylib(), BUILT_IN_HAS_GPU),
    };
    if candidates.is_empty() {
        return Err(not_found());
    }
    Ok(candidates)
}

/// Whether `path` opens as an ONNX Runtime: the library loads and has its
/// entry point. The one that passes stays open, so ort's open of it finds
/// it loaded.
fn probe_runtime(path: &Path) -> Result<(), String> {
    // SAFETY: opening a library runs its initialisers; the candidates are
    // the runtimes the app ships or installed, the ones ort would load.
    let lib = unsafe { libloading::Library::new(path) }.map_err(|e| format!("{}: {e}", path.display()))?;
    // SAFETY: the symbol is looked up, never called.
    let entry = unsafe { lib.get::<unsafe extern "C" fn() -> *const std::ffi::c_void>(b"OrtGetApiBase") };
    entry.map_err(|e| format!("{}: {e}", path.display()))?;
    std::mem::forget(lib);
    Ok(())
}

fn load_runtime(path: &Path) -> Result<(), String> {
    for name in PRELOAD_BESIDE_RUNTIME {
        preload_beside(path, name);
    }
    #[allow(clippy::disallowed_methods)] // the sanctioned init: guarded by the OnceLock in ensure_initialized
    let env = ort::init_from(path)
        .map_err(|e| format!("ort::init_from({}): {e}", path.display()))?
        // Off, though `ort` turns it on: since 1.29 ONNX Runtime on Linux and
        // macOS uploads telemetry through Microsoft's 1DS client, whose
        // thread aborted Prunr at exit (a mutex already torn down: 2 of 3
        // bench runs on a MacBook Air, ORT 1.30). Prunr reports nothing.
        .with_telemetry(false)
        .build()
        .map_err(|e| format!("ORT environment on {}: {e}", path.display()))?;
    let _ = ENV.set(env);
    Ok(())
}

/// The first of `candidates` that `load` takes, with why each before it
/// failed; every failure joined when none does.
fn load_first(
    candidates: &[(PathBuf, DylibSource)],
    mut load: impl FnMut(&Path) -> Result<(), String>,
) -> Result<(&(PathBuf, DylibSource), Vec<String>), String> {
    let mut failures = Vec::new();
    for candidate in candidates {
        match load(&candidate.0) {
            Ok(()) => return Ok((candidate, failures)),
            Err(why) => {
                tracing::warn!(path = %candidate.0.display(), %why, "runtime did not load");
                failures.push(why);
            }
        }
    }
    Err(failures.join("; "))
}

static FELL_BACK: OnceLock<&'static str> = OnceLock::new();

/// Why the runtime loaded is not the chosen one, when it fell back.
pub fn fallback_note() -> Option<&'static str> {
    FELL_BACK.get().copied()
}

/// A runtime without `name` beside it (the CPU build) needs none.
fn preload_beside(runtime: &Path, name: &str) {
    let lib = runtime.with_file_name(name);
    if !lib.is_file() {
        return;
    }
    match ort::util::preload_dylib(lib.as_path()) {
        Ok(()) => tracing::info!(path = %lib.display(), "preloaded beside the runtime"),
        Err(e) => tracing::warn!(path = %lib.display(), %e, "preload beside the runtime failed"),
    }
}

pub fn resolve_dylib_path() -> Result<(PathBuf, DylibSource), String> {
    candidates()?.into_iter().next().ok_or_else(not_found)
}

fn not_found() -> String {
    format!(
        "ONNX Runtime not found. Looked for `{DYLIB_NAME}` in (in order): \
         ORT_DYLIB_PATH env, user Runtime Store, bundled location next to executable. \
         Set ORT_DYLIB_PATH or run `cargo xtask install-runtime onnxruntime <version>`."
    )
}

fn env_override() -> Option<PathBuf> {
    std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from)
}

/// File in the runtime store naming the install every process loads.
/// One process loads one runtime, and no wheel carries another's
/// provider, so with two installed the choice is explicit; a file is
/// what the worker subprocess and the CLI see as well as the GUI.
pub const PREFERRED_FILE: &str = "preferred";

/// The preference for the runtime beside the executable over any store
/// install. Install dirs are `<short>-<ver>-<rid>`, so it names none.
pub const PREFER_BUILT_IN: &str = "built-in";

/// The built-in runtime runs models on a GPU: on Windows it is the
/// DirectML build, which also runs them on the CPU without a DirectX 12
/// GPU. On Linux it is CPU-only, so an installed accelerator runtime
/// wins when no preference says otherwise. The macOS one carries CoreML,
/// and no runtime the store offers has a macOS wheel to compete with it.
pub const BUILT_IN_HAS_GPU: bool = cfg!(windows);

/// Whether a runtime sits beside the executable (release builds).
pub fn has_built_in() -> bool {
    bundled_dylib().is_some()
}

/// The runtime every process loads first, when one is set: an install
/// subdir, or [`PREFER_BUILT_IN`].
pub fn preferred_runtime() -> Option<String> {
    read_preferred(&prunr_models::runtime_store_dir()?)
}

/// Make `subdir` (an install, or [`PREFER_BUILT_IN`]) the runtime every
/// process loads from its next start.
pub fn set_preferred_runtime(subdir: &str) -> Result<(), String> {
    let root = prunr_models::runtime_store_dir().ok_or("could not resolve user data dir")?;
    write_preferred(&root, subdir)
}

/// Drop the preference when it names `subdir`, which is going away.
pub fn forget_preferred_runtime(subdir: &str) {
    let Some(root) = prunr_models::runtime_store_dir().filter(|root| read_preferred(root).as_deref() == Some(subdir)) else {
        return;
    };
    if let Err(e) = std::fs::remove_file(root.join(PREFERRED_FILE)) {
        tracing::warn!(%e, subdir, "could not clear the preferred runtime");
    }
}

/// Present when the user lets OpenVINO run models on the Intel GPU
/// (Settings › Hardware). A file, like [`PREFERRED_FILE`], so the worker
/// subprocess and the CLI follow it too.
pub const INTEL_GPU_FILE: &str = "intel-gpu";

/// Whether the user lets OpenVINO use the Intel GPU, read from disk now.
pub fn intel_gpu_chosen() -> bool {
    intel_gpu_file().is_some_and(|file| file.is_file())
}

/// The file whose presence lets OpenVINO use the Intel GPU, from each
/// process's next start.
pub fn intel_gpu_file() -> Option<PathBuf> {
    prunr_models::runtime_store_dir().map(|root| root.join(INTEL_GPU_FILE))
}

/// Where an installed runtime keeps the NVIDIA libraries the app
/// downloaded for it (CUDA 12, cuDNN 9), beside its own library.
pub const NVIDIA_LIBS_DIR: &str = "nvidia";

/// The NVIDIA libraries the CUDA provider links, in the order each one's
/// own links are met. The rest (cuDNN's engines, NVRTC) load later, from
/// beside these: NVIDIA builds them with `$ORIGIN` as their search path.
const NVIDIA_LINKED: [&str; 6] = [
    "libcudart.so.12",
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcurand.so.10",
    "libcufft.so.11",
    "libcudnn.so.9",
];

/// Load the NVIDIA libraries beside the loaded runtime, when there are
/// any, so the CUDA provider finds them: it links them by name and has no
/// search path of its own, and the loader reuses a library already loaded
/// under that name. Once per process; the libraries stay loaded.
pub fn preload_nvidia_libraries() {
    static PRELOADED: OnceLock<()> = OnceLock::new();
    PRELOADED.get_or_init(|| {
        let Some(dir) = loaded_dylib().and_then(Path::parent).map(|d| d.join(NVIDIA_LIBS_DIR)) else { return };
        if !dir.is_dir() {
            return;
        }
        let started = std::time::Instant::now();
        for name in NVIDIA_LINKED {
            if let Err(e) = load_global(&dir.join(name)) {
                tracing::warn!(name, %e, "NVIDIA library did not load");
            }
        }
        tracing::info!(dir = %dir.display(), elapsed_ms = started.elapsed().as_millis() as u64, "NVIDIA libraries loaded");
    });
}

/// Load `path` for the whole process and keep it loaded.
fn load_global(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_LAZY};
        // SAFETY: NVIDIA's libraries run no initializers that depend on
        // the caller; they stay loaded for the process (forgotten handle).
        let lib = unsafe { Library::open(Some(path), RTLD_LAZY | RTLD_GLOBAL) }.map_err(|e| e.to_string())?;
        std::mem::forget(lib);
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Err("NVIDIA libraries are preloaded on Linux only".to_string())
    }
}

fn read_preferred(root: &Path) -> Option<String> {
    let text = std::fs::read_to_string(root.join(PREFERRED_FILE)).ok()?;
    let name = text.trim();
    (!name.is_empty()).then(|| name.to_string())
}

fn write_preferred(root: &Path, subdir: &str) -> Result<(), String> {
    std::fs::create_dir_all(root).map_err(|e| format!("create {}: {e}", root.display()))?;
    let path = root.join(PREFERRED_FILE);
    std::fs::write(&path, subdir).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Every runtime there is to load, in the order to try them: the
/// preferred one, then the built-in one first when `built_in_first`, else
/// the store's installs by name, then the rest.
fn runtime_candidates(store_root: Option<&Path>, built_in: Option<PathBuf>, built_in_first: bool) -> Vec<(PathBuf, DylibSource)> {
    let installs: Vec<(String, PathBuf)> = store_root
        .map(store_entries_in)
        .unwrap_or_default()
        .into_iter()
        .filter(|(_, p)| p.is_file())
        .collect();
    let built_in = built_in.map(|p| (p, DylibSource::Bundled));
    let preferred = store_root.and_then(read_preferred).and_then(|want| {
        if want == PREFER_BUILT_IN {
            built_in.clone()
        } else {
            installs.iter().find(|(name, _)| *name == want).map(|(_, p)| (p.clone(), DylibSource::RuntimeStore))
        }
    });
    let installs = installs.into_iter().map(|(_, p)| (p, DylibSource::RuntimeStore));
    let (first, then) = if built_in_first { (built_in, None) } else { (None, built_in) };
    let mut candidates: Vec<(PathBuf, DylibSource)> = Vec::new();
    for candidate in preferred.into_iter().chain(first).chain(installs).chain(then) {
        if !candidates.iter().any(|(p, _)| *p == candidate.0) {
            candidates.push(candidate);
        }
    }
    candidates
}

fn store_entries() -> Vec<(String, PathBuf)> {
    prunr_models::runtime_store_dir().map(|root| store_entries_in(&root)).unwrap_or_default()
}

/// Every runtime-store entry as `(dir name, path the dylib would have)`,
/// sorted by name so logs reproduce.
fn store_entries_in(root: &Path) -> Vec<(String, PathBuf)> {
    let Ok(read) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut entries: Vec<(String, PathBuf)> = read
        .filter_map(Result::ok)
        .filter(|e| e.file_type().ok().is_some_and(|ft| ft.is_dir()))
        // A hidden folder is an install still being extracted.
        .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path().join(DYLIB_NAME)))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

/// Runtime resolution state as a fresh init would see it.
#[derive(Debug)]
pub struct Diagnostics {
    pub env_path: Option<PathBuf>,
    pub store_root: Option<PathBuf>,
    pub store_entries: Vec<(String, bool)>,
    pub preferred: Option<String>,
    pub bundled: Option<(PathBuf, bool)>,
    pub resolved: Option<(PathBuf, DylibSource)>,
    /// Each library preloaded from beside the resolved runtime, and
    /// whether it is there.
    pub beside_runtime: Vec<(PathBuf, bool)>,
}

pub fn diagnose() -> Diagnostics {
    let resolved = resolve_dylib_path().ok();
    let beside_runtime = resolved.as_ref().map_or_else(Vec::new, |(runtime, _)| {
        PRELOAD_BESIDE_RUNTIME.iter()
            .map(|name| { let lib = runtime.with_file_name(name); let exists = lib.is_file(); (lib, exists) })
            .collect()
    });
    Diagnostics {
        env_path: env_override(),
        store_root: prunr_models::runtime_store_dir(),
        store_entries: store_entries().into_iter().map(|(name, p)| { let exists = p.is_file(); (name, exists) }).collect(),
        preferred: preferred_runtime(),
        bundled: bundled_dylib().map(|p| (p, true))
            .or_else(|| bundled_paths().into_iter().next().map(|p| (p, false))),
        resolved,
        beside_runtime,
    }
}

/// True iff at least one ORT source other than `excluding` exists, so a
/// removal never leaves the app with no runtime to load (a dev build has
/// no bundled one).
pub fn has_fallback_excluding(excluding: &Path) -> bool {
    env_override().is_some_and(|p| p.is_file() && p != excluding)
        || store_entries().iter().any(|(_, p)| p.is_file() && p != excluding && !p.starts_with(excluding))
        || bundled_dylib().is_some()
}

fn bundled_paths() -> Vec<PathBuf> {
    let Some(exe) = std::env::current_exe().ok() else { return Vec::new() };
    // macOS reports the path the binary was launched by; a Homebrew
    // `bin/prunr` symlink must still find the runtime beside the real
    // binary, as dyld's `@executable_path` would.
    let exe = if cfg!(target_os = "macos") { exe.canonicalize().unwrap_or(exe) } else { exe };
    exe.parent().map_or_else(Vec::new, |dir| bundled_candidates(dir, cfg!(target_os = "macos")))
}

/// Where a packaged build keeps the runtime: release CI's `runtime/`
/// beside the binary, the binary's own dir (packagers that flatten
/// `runtime/`, the macOS tar.gz), and a macOS `.app`'s
/// `Contents/Frameworks/`.
fn bundled_candidates(exe_dir: &Path, macos: bool) -> Vec<PathBuf> {
    let mut paths = vec![exe_dir.join("runtime").join(DYLIB_NAME), exe_dir.join(DYLIB_NAME)];
    if macos {
        paths.extend(exe_dir.parent().map(|contents| contents.join("Frameworks").join(DYLIB_NAME)));
    }
    paths
}

fn bundled_dylib() -> Option<PathBuf> {
    bundled_paths().into_iter().find(|p| p.is_file())
}

#[cfg(test)]
mod tests {

    /// The runtime starts with telemetry off: `ort` turns it on, and the
    /// upload client it starts in ONNX Runtime 1.29+ aborted Prunr at exit.
    #[test]
    fn the_runtime_starts_with_telemetry_off() {
        let source = include_str!("ort_runtime.rs");
        let init = source.find("ort::init_from(path)").expect("the one runtime init");
        let built = init + source[init..].find(".build()").expect("built");
        assert!(source[init..built].contains(".with_telemetry(false)"), "telemetry must be switched off where the environment is built");
    }
    use super::*;

    /// Folding stays off where ORT's aliasing bug lives (1.24 to 1.28, or
    /// no readable version); on for the rest.
    #[test]
    fn constant_folding_is_off_where_it_corrupts_weights() {
        assert!(folding_is_broken(Some("1.24.1")));
        assert!(folding_is_broken(Some("1.24.4")));
        assert!(folding_is_broken(None));
        assert!(folding_is_broken(Some("1.27.1")));
        assert!(!folding_is_broken(Some("1.23.2")));
        assert!(folding_is_broken(Some("1.28.3")));
        assert!(!folding_is_broken(Some("1.29.0")));
        assert!(folding_is_broken(Some("odd")), "an unreadable version counts as affected");
    }

    #[test]
    fn the_loaded_runtime_reports_its_version() {
        if ensure_initialized().is_err() {
            return;
        }
        let version = runtime_version().expect("a loaded runtime reports its version");
        assert!(version.split('.').take(2).all(|part| part.parse::<u32>().is_ok()), "{version}");
    }

    #[test]
    fn a_session_s_own_disabled_optimizers_keep_folding_off() {
        assert_eq!(disabled_optimizers(Some("1.24.1"), &["MatMulAddFusion"]).as_deref(), Some("ConstantFolding;MatMulAddFusion"));
        assert_eq!(disabled_optimizers(Some("1.24.1"), &[]).as_deref(), Some("ConstantFolding"));
        assert_eq!(disabled_optimizers(Some("1.29.0"), &["MatMulAddFusion"]).as_deref(), Some("MatMulAddFusion"));
        assert_eq!(disabled_optimizers(Some("1.29.0"), &[]), None);
    }

    #[test]
    fn bundled_candidates_reach_the_app_bundle_frameworks_on_macos() {
        let exe_dir = Path::new("/Applications/Prunr.app/Contents/MacOS");
        let frameworks = Path::new("/Applications/Prunr.app/Contents/Frameworks").join(DYLIB_NAME);
        let expected_common = [exe_dir.join("runtime").join(DYLIB_NAME), exe_dir.join(DYLIB_NAME)];

        let mac = bundled_candidates(exe_dir, true);
        assert_eq!(mac[..2], expected_common);
        assert_eq!(mac.get(2), Some(&frameworks));
        assert_eq!(mac.len(), 3);

        assert_eq!(bundled_candidates(exe_dir, false), expected_common);
    }

    fn install(root: &Path, subdir: &str) {
        let dir = root.join(subdir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(DYLIB_NAME), b"").unwrap();
    }

    fn picked(root: &Path, built_in: Option<&Path>, built_in_first: bool) -> Option<PathBuf> {
        runtime_candidates(Some(root), built_in.map(Path::to_path_buf), built_in_first).into_iter().next().map(|(p, _)| p)
    }

    #[test]
    fn the_first_install_by_name_loads_without_a_preference() {
        let root = tempfile::tempdir().unwrap();
        install(root.path(), "openvino-1.24.1-windows-x64");
        install(root.path(), "directml-1.24.1-windows-x64");
        assert_eq!(picked(root.path(), None, false), Some(root.path().join("directml-1.24.1-windows-x64").join(DYLIB_NAME)));
    }

    #[test]
    fn the_preferred_install_loads_even_when_another_sorts_first() {
        let root = tempfile::tempdir().unwrap();
        install(root.path(), "directml-1.24.1-windows-x64");
        install(root.path(), "openvino-1.24.1-windows-x64");
        write_preferred(root.path(), "openvino-1.24.1-windows-x64").unwrap();
        assert_eq!(picked(root.path(), None, false), Some(root.path().join("openvino-1.24.1-windows-x64").join(DYLIB_NAME)));
    }

    #[test]
    fn a_preference_for_a_missing_or_broken_install_falls_back_to_name_order() {
        let root = tempfile::tempdir().unwrap();
        install(root.path(), "openvino-1.24.1-windows-x64");
        write_preferred(root.path(), "directml-1.24.1-windows-x64").unwrap();
        let fallback = Some(root.path().join("openvino-1.24.1-windows-x64").join(DYLIB_NAME));
        assert_eq!(picked(root.path(), None, false), fallback);
        std::fs::create_dir_all(root.path().join("directml-1.24.1-windows-x64")).unwrap();
        assert_eq!(picked(root.path(), None, false), fallback, "an install without its library is skipped");
    }

    /// Windows ships DirectML built in: an OpenVINO install loads only
    /// when it is the preferred one, and the built-in choice beats it.
    #[test]
    fn the_built_in_gpu_runtime_wins_unless_a_store_install_is_preferred() {
        let root = tempfile::tempdir().unwrap();
        let exe_dir = tempfile::tempdir().unwrap();
        let built_in = exe_dir.path().join(DYLIB_NAME);
        install(root.path(), "openvino-1.24.1-windows-x64");
        let openvino = Some(root.path().join("openvino-1.24.1-windows-x64").join(DYLIB_NAME));
        assert_eq!(picked(root.path(), Some(&built_in), true), Some(built_in.clone()), "no preference: built in");
        assert_eq!(picked(root.path(), Some(&built_in), false), openvino, "a CPU-only built-in runtime yields");
        write_preferred(root.path(), "openvino-1.24.1-windows-x64").unwrap();
        assert_eq!(picked(root.path(), Some(&built_in), true), openvino, "the store install the user picked");
        write_preferred(root.path(), PREFER_BUILT_IN).unwrap();
        assert_eq!(picked(root.path(), Some(&built_in), false), Some(built_in.clone()), "built in, as picked");
        assert_eq!(picked(root.path(), None, false), openvino, "built in picked but absent (a dev build)");
        assert_eq!(runtime_candidates(Some(root.path()), Some(built_in), true).first().map(|(_, s)| s.to_string()), Some("bundled".to_string()));
    }

    #[test]
    fn the_preference_file_is_not_an_install_and_reads_back_trimmed() {
        let root = tempfile::tempdir().unwrap();
        assert_eq!(picked(root.path(), None, false), None);
        std::fs::write(root.path().join(PREFERRED_FILE), "openvino-1.24.1-linux-x64\n").unwrap();
        assert_eq!(store_entries_in(root.path()), Vec::new());
        assert_eq!(read_preferred(root.path()).as_deref(), Some("openvino-1.24.1-linux-x64"));
        std::fs::write(root.path().join(PREFERRED_FILE), "  ").unwrap();
        assert_eq!(read_preferred(root.path()), None);
    }

    /// A runtime that will not load (a cut-off install) gives way to the
    /// next one to try, and the reasons are kept; none loading is an error
    /// naming every failure.
    #[test]
    fn a_runtime_that_will_not_load_gives_way_to_the_next() {
        let store = PathBuf::from("/store/openvino/lib");
        let built_in = PathBuf::from("/app/lib");
        let candidates = vec![(store.clone(), DylibSource::RuntimeStore), (built_in.clone(), DylibSource::Bundled)];
        let ((path, source), failures) = load_first(&candidates, |p| if p == store { Err("truncated".into()) } else { Ok(()) }).unwrap();
        assert_eq!((path, *source), (&built_in, DylibSource::Bundled));
        assert_eq!(failures, vec!["truncated".to_string()]);
        let none = load_first(&candidates, |p| Err(format!("bad {}", p.display()))).err();
        assert_eq!(none.as_deref(), Some("bad /store/openvino/lib; bad /app/lib"));
    }

    /// Every runtime is a candidate, the preferred one first and none twice;
    /// a hidden folder (an install being extracted) is none.
    #[test]
    fn every_runtime_is_tried_once_the_preferred_first() {
        let root = tempfile::tempdir().unwrap();
        let exe_dir = tempfile::tempdir().unwrap();
        let built_in = exe_dir.path().join(DYLIB_NAME);
        install(root.path(), "openvino-1.24.1-linux-x64");
        install(root.path(), "a-runtime-1.24.1-linux-x64");
        install(root.path(), ".openvino-1.24.1-linux-x64.staging");
        write_preferred(root.path(), "openvino-1.24.1-linux-x64").unwrap();
        let paths: Vec<PathBuf> = runtime_candidates(Some(root.path()), Some(built_in.clone()), false).into_iter().map(|(p, _)| p).collect();
        assert_eq!(paths, vec![
            root.path().join("openvino-1.24.1-linux-x64").join(DYLIB_NAME),
            root.path().join("a-runtime-1.24.1-linux-x64").join(DYLIB_NAME),
            built_in,
        ]);
    }

    /// A file that is no runtime fails the check before ort ever sees
    /// it; the runtime this process loaded passes.
    #[test]
    fn only_a_real_runtime_passes_the_probe() {
        let dir = tempfile::tempdir().unwrap();
        let garbage = dir.path().join(DYLIB_NAME);
        std::fs::write(&garbage, [0x42u8; 4096]).unwrap();
        assert!(probe_runtime(&garbage).is_err());
        assert!(probe_runtime(&dir.path().join("missing").join(DYLIB_NAME)).is_err());
        if let Err(why) = runtime_for_tests() {
            eprintln!("SKIP: {why}");
            return;
        }
        let loaded = LOADED.get().expect("the runtime loaded");
        assert_eq!(probe_runtime(loaded), Ok(()));
    }
}
