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
/// loaded: its version where the build info carries one (Microsoft's
/// own builds say `git-branch=HEAD`), and a fingerprint of the library
/// file, which every runtime upgrade changes.
pub fn runtime_tag() -> Option<&'static str> {
    static TAG: OnceLock<Option<String>> = OnceLock::new();
    TAG.get_or_init(|| {
        ensure_initialized().ok()?;
        let meta = std::fs::metadata(LOADED.get()?).ok()?;
        let modified = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
        Some(tag_for(runtime_version_from_info(ort::info()).as_deref(), meta.len(), modified))
    }).as_deref()
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

/// The version in ORT's build info, whose branch is `rel-<version>`.
pub fn runtime_version_from_info(info: &str) -> Option<String> {
    let branch = info.split("git-branch=").nth(1)?.split(',').next()?.trim();
    branch.strip_prefix("rel-").map(str::to_string)
}

/// A session builder on the loaded runtime.
pub fn session_builder() -> Result<SessionBuilder, CoreError> {
    ensure_initialized().map_err(CoreError::Inference)?;
    #[allow(clippy::disallowed_methods)] // the sanctioned wrapper: the runtime is loaded one line above
    Session::builder().map_err(|e| CoreError::Inference(format!("ORT builder init failed: {e}")))
}

/// Libraries loaded from beside the runtime before the runtime itself.
/// The DirectML build of `onnxruntime.dll` delay-loads `DirectML.dll`.
/// ORT's hook looks beside itself first, but when that fails Windows
/// searches by name and finds System32's older copy, which rejects the
/// feature level ORT asks for. A copy already loaded is what every later
/// load by name resolves to.
const PRELOAD_BESIDE_RUNTIME: &[&str] = if cfg!(windows) { &["DirectML.dll"] } else { &[] };

fn init() -> Result<DylibSource, String> {
    let (path, source) = resolve_dylib_path()?;
    for name in PRELOAD_BESIDE_RUNTIME {
        preload_beside(&path, name);
    }
    #[allow(clippy::disallowed_methods)] // the sanctioned init: guarded by the OnceLock in ensure_initialized
    let env = ort::init_from(&path)
        .map_err(|e| format!("ort::init_from({}): {e}", path.display()))?;
    // false means an env was already committed; ORT is initialised either way.
    let _ = env.commit();
    tracing::info!(path = %path.display(), %source, "ORT runtime loaded");
    let _ = LOADED.set(path);
    Ok(source)
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
    if let Some(path) = env_override() {
        if !path.is_file() {
            return Err(format!("ORT_DYLIB_PATH={} is not a file", path.display()));
        }
        return Ok((path, DylibSource::EnvVar));
    }
    if let Some(found) = pick_runtime(prunr_models::runtime_store_dir().as_deref(), bundled_dylib(), BUILT_IN_HAS_GPU) {
        return Ok(found);
    }
    Err(format!(
        "ONNX Runtime not found. Looked for `{DYLIB_NAME}` in (in order): \
         ORT_DYLIB_PATH env, user Runtime Store, bundled location next to executable. \
         Set ORT_DYLIB_PATH or run `cargo xtask install-runtime onnxruntime <version>`."
    ))
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

/// The runtime to load from the store under `store_root` and the
/// `built_in` one beside the executable: the preferred one when it is
/// there, else the built-in one first when `built_in_first`, else the
/// store's first install by name.
fn pick_runtime(store_root: Option<&Path>, built_in: Option<PathBuf>, built_in_first: bool) -> Option<(PathBuf, DylibSource)> {
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
    let first_install = installs.into_iter().next().map(|(_, p)| (p, DylibSource::RuntimeStore));
    preferred.or(if built_in_first { built_in.or(first_install) } else { first_install.or(built_in) })
}

fn store_entries() -> Vec<(String, PathBuf)> {
    prunr_models::runtime_store_dir().map(|root| store_entries_in(&root)).unwrap_or_default()
}

/// Every runtime-store entry as `(dir name, path the dylib would have)`,
/// sorted by name so logs reproduce; meaningful EP selection happens at
/// the per-session ladder, not at this layer.
fn store_entries_in(root: &Path) -> Vec<(String, PathBuf)> {
    let Ok(read) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut entries: Vec<(String, PathBuf)> = read
        .filter_map(Result::ok)
        .filter(|e| e.file_type().ok().is_some_and(|ft| ft.is_dir()))
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path().join(DYLIB_NAME)))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

/// Structured snapshot of runtime resolution state for `prunr doctor`.
/// Each field reflects what the resolver would see on a fresh init.
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

/// True iff at least one ORT source other than `excluding` exists. Used
/// by Settings → Hardware → Uninstall to refuse a removal that would
/// leave the app unable to load any runtime — this is the dev-build
/// case where there's no bundled fallback next to the executable, and
/// the user's runtime-store entry is their only ORT.
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
    use super::*;

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
        pick_runtime(Some(root), built_in.map(Path::to_path_buf), built_in_first).map(|(p, _)| p)
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
        assert_eq!(pick_runtime(Some(root.path()), Some(built_in), true).map(|(_, s)| s.to_string()), Some("bundled".to_string()));
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
}
