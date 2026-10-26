//! Per-EP compiled-model cache.
//!
//! Every EP that supports caching its compiled IR (OpenVINO model
//! cache, CoreML mlmodelc cache, ORT graph optimization for CPU /
//! CUDA) gets a directory under `<cache root>/<ep>/<key>/` (see
//! [`cache_root`]), where `<key>` is
//! `<model_stable_name>-<model_version>-<files>-ort<runtime>-v<format>`,
//! then `-<cpu vector ISA>` on the CPU (see [`cache_key`]). Bumping any
//! component invalidates the cache automatically.
//!
//! EP names are `OrtEngine::active_provider()` strings (`"OpenVINO"`,
//! `"CPU"`, …) rather than `EpKind`, because the CPU provider has no `EpKind`.

use std::path::{Path, PathBuf};
use prunr_models::ModelId;
use crate::engine::Weights;

/// Cache format version. Bump when the cache layout changes OR when
/// the `ort` crate version bumps — cached optimized graphs and EP
/// blobs are not portable across ORT versions, and we don't put the
/// ORT version in the key directly because `ort` doesn't expose it
/// at compile time. Old directories are never read; GC reclaims them.
/// 2: graphs written before constant folding was turned off on ORT 1.24
/// may hold the constants its folding corrupted.
const CACHE_FORMAT_VERSION: u32 = 2;

/// Recursion bound for `walk_dir_size`, so a symlink loop terminates;
/// the cache is two levels deep.
const MAX_WALK_DEPTH: u32 = 16;

/// Root cache directory: `<data>/prunr/ep_cache/`, on macOS
/// `~/Library/Caches/prunr/ep_cache/`. Returns `None` when the OS gives
/// no such dir; callers fall back to no-cache rather than erroring out.
pub fn cache_root() -> Option<PathBuf> {
    #[cfg(test)]
    if let Some(root) = OWN_ROOT.with_borrow(Clone::clone).or_else(tests::temp_root) {
        return Some(root);
    }
    root_in(prunr_models::data_dir(), dirs::cache_dir(), cfg!(target_os = "macos"))
}

#[cfg(test)]
thread_local! {
    static OWN_ROOT: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

/// The calling test's own cache root until dropped: a test that breaks a
/// model's cached files on purpose would otherwise race other tests
/// building that model in the same process.
#[cfg(test)]
pub(crate) struct OwnCacheRoot;

#[cfg(test)]
impl OwnCacheRoot {
    pub(crate) fn new(name: &str) -> Self {
        OWN_ROOT.set(tests::temp_root().and_then(|run| Some(run.parent()?.join(name).join("ep_cache"))));
        Self
    }
}

#[cfg(test)]
impl Drop for OwnCacheRoot {
    fn drop(&mut self) {
        OWN_ROOT.set(None);
    }
}

/// macOS keeps the cache out of `Application Support`: CoreML opens
/// its compiled model through a file URL that a space in the path can
/// turn into nil (microsoft/onnxruntime#33306).
fn root_in(prunr_data_dir: Option<PathBuf>, os_cache_dir: Option<PathBuf>, macos: bool) -> Option<PathBuf> {
    if macos {
        os_cache_dir.map(|d| d.join("prunr").join("ep_cache"))
    } else {
        prunr_data_dir.map(|d| d.join("ep_cache"))
    }
}

/// Where macOS kept the cache before it moved to `Caches`: nothing
/// reads it any more.
fn retired_root_in(prunr_data_dir: Option<PathBuf>, macos: bool) -> Option<PathBuf> {
    prunr_data_dir.filter(|_| macos).map(|d| d.join("ep_cache"))
}

/// Remove the cache macOS kept under `Application Support` before it
/// moved (up to a few GB of compiled models). Returns the bytes freed;
/// nothing to do once it is gone. Run off the UI thread.
pub fn remove_retired_root() -> u64 {
    let Some(old) = retired_root_in(prunr_models::data_dir(), cfg!(target_os = "macos")) else { return 0 };
    if !old.exists() {
        return 0;
    }
    let bytes = walk_dir_size(&old, 0).unwrap_or(0);
    match std::fs::remove_dir_all(&old) {
        Ok(()) => {
            tracing::info!(dir = %old.display(), bytes, "removed the retired EP cache");
            bytes
        }
        Err(e) => {
            tracing::warn!(%e, dir = %old.display(), "failed to remove the retired EP cache");
            0
        }
    }
}

/// Pure path builder — no filesystem side effects; `cache_dir_for`
/// creates the directory.
pub fn cache_path_for(id: ModelId, ep_name: &str) -> Option<PathBuf> {
    let root = cache_root()?;
    let key = cache_key(id, ep_name, crate::ort_runtime::runtime_tag())?;
    Some(root.join(ep_name.to_ascii_lowercase()).join(key))
}

/// `<model>-<version>-<files>-ort<runtime tag>-v<format>`, then on the CPU
/// `-<vector extension>`: an optimized graph is tied to the model files it
/// was built from, the ORT build that wrote it and, on the CPU, the vector
/// width its layout was chosen for. A download's files are told apart by
/// their checksums (a re-export may keep its version string); a built-in
/// model changes only with the app.
fn cache_key(id: ModelId, ep_name: &str, runtime: Option<&str>) -> Option<String> {
    let descriptor = prunr_models::descriptor(id)?;
    let files = prunr_models::files_id(id);
    let mut key = format!(
        "{}-{}-{files}-ort{}-v{}",
        id.stable_name(), descriptor.version, runtime.unwrap_or("unknown"), CACHE_FORMAT_VERSION,
    );
    if ep_name.eq_ignore_ascii_case(crate::engine::CPU_NAME) {
        key.push('-');
        key.push_str(cpu_vector_isa());
    }
    Some(key)
}

/// The CPU a layout-optimized graph was built for: which convolutions its
/// NCHWc nodes take depends on the block of channels the kernels work in
/// (16 floats with AVX-512, 8 on other x86 CPUs), so a cache carried to
/// another machine (a synced profile, a moved VM) is not loaded there.
fn cpu_vector_isa() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        return "avx512";
    }
    std::env::consts::ARCH
}

/// Per-`(model, ep)` cache directory, **created** if missing. Use
/// from session-build sites that are about to write into the cache.
/// Returns `None` when `data_dir()` is unavailable, the model id is
/// unknown, or `create_dir_all` fails.
pub fn cache_dir_for(id: ModelId, ep_name: &str) -> Option<PathBuf> {
    let dir = cache_path_for(id, ep_name)?;
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(?id, ep = %ep_name, %e, "failed to create EP cache dir");
        return None;
    }
    Some(dir)
}

/// Pure path to the ORT-graph-optimized model file for `(id, ep_name)`
/// built from `weights`: each copy of the weights optimizes to a graph
/// of its own. No filesystem side effects: writers call `cache_dir_for`
/// first so the parent directory exists.
pub fn optimized_model_path(id: ModelId, ep_name: &str, weights: Weights) -> Option<PathBuf> {
    cache_path_for(id, ep_name).map(|d| d.join(optimized_file_name(weights)))
}

fn optimized_file_name(weights: Weights) -> &'static str {
    match weights {
        Weights::Fp32 => "optimized-fp32.onnx",
        Weights::Fp16 => "optimized-fp16.onnx",
        Weights::Int8 => "optimized-int8.onnx",
    }
}

/// The optimized graph every copy of the weights once shared: it holds
/// whichever copy built first, so no session can trust it.
const SHARED_GRAPH_FILE: &str = "optimized.onnx";

/// An optimized graph a session is writing for `path`. ORT writes it in
/// place while the session builds, so it goes to a file of its own and
/// is renamed onto `path` once the session built: another session (a
/// second worker, a parallel test) loading the cache meanwhile never
/// reads half a graph, and a build that dies leaves no broken cache.
/// Dropped unpublished, the draft is removed.
pub struct GraphDraft {
    draft: PathBuf,
    path: PathBuf,
    /// The disk space the graph will take, held until it is written.
    _room: Option<Reserved>,
}

/// A draft left by a build that died is older than any build takes.
const ABANDONED_DRAFT: std::time::Duration = std::time::Duration::from_secs(3600);

impl GraphDraft {
    /// A draft for `path`, named for this process and build; drafts an
    /// earlier build abandoned beside it are removed.
    pub fn for_path(path: &Path) -> Self {
        static BUILDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let build = BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let name = path.file_name().map_or_else(Default::default, |n| n.to_string_lossy().into_owned());
        let draft = path.with_file_name(format!("{name}.{}-{build}.draft", std::process::id()));
        if let Some(dir) = path.parent() {
            remove_abandoned_drafts(dir, &name);
        }
        Self { draft, path: path.to_path_buf(), _room: None }
    }

    /// Where ORT writes the graph.
    pub fn draft(&self) -> &Path {
        &self.draft
    }

    /// The session built: its graph becomes the cache, marked as made
    /// on this machine.
    pub fn publish(self) {
        if std::fs::metadata(&self.draft).is_ok_and(|m| m.len() == 0) {
            return;
        }
        match std::fs::rename(&self.draft, &self.path) {
            Ok(()) => {
                let _ = std::fs::write(graph_mark(&self.path, MADE_HERE), b"");
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!(cache = %self.path.display(), %e, "optimized graph not cached"),
        }
    }
}

impl Drop for GraphDraft {
    fn drop(&mut self) {
        // After `publish` the draft is gone and this finds nothing.
        let _ = std::fs::remove_file(&self.draft);
    }
}

/// Marks beside a saved graph: this machine wrote it, or one it wrote
/// failed to load back and it is no longer read. In the graph's folder,
/// so a new runtime or a cleared cache starts afresh.
const MADE_HERE: &str = "made-here";
const NOT_READ: &str = "not-read";

fn graph_mark(path: &Path, mark: &str) -> PathBuf {
    let mut name = path.file_name().map(std::ffi::OsStr::to_os_string).unwrap_or_default();
    name.push(format!(".{mark}"));
    path.with_file_name(name)
}

/// A saved graph at `path` failed to load back (it is deleted by then).
/// One copied from another machine is simply built again here; one this
/// machine wrote will fail again, so it is no longer read.
pub(crate) fn note_graph_failed(path: &Path) {
    if !graph_mark(path, MADE_HERE).exists() {
        return;
    }
    match std::fs::write(graph_mark(path, NOT_READ), b"") {
        Ok(()) => tracing::warn!(graph = %path.display(), "a graph saved here failed to load back; it is no longer read"),
        Err(e) => tracing::warn!(graph = %path.display(), %e, "could not mark the failed graph"),
    }
}

pub(crate) fn graph_not_read(path: &Path) -> bool {
    graph_mark(path, NOT_READ).exists()
}

/// A cache miss: `builder` writes its optimized graph (about
/// `graph_bytes`) to a draft of `path`, published once the session
/// builds. Where the draft cannot be written, or ORT refuses the path,
/// the build goes on uncached.
pub(crate) fn attach_draft(
    builder: ort::session::builder::SessionBuilder,
    path: &Path,
    graph_bytes: u64,
    model_id: ModelId,
    ep_name: &str,
    part: Option<&str>,
) -> (ort::session::builder::SessionBuilder, Option<GraphDraft>) {
    if !take_draft_slot() {
        tracing::debug!(?model_id, ep = %ep_name, part, "built again without its optimized graph");
        return (builder, None);
    }
    let mut draft = GraphDraft::for_path(path);
    match writable(draft.draft(), graph_bytes) {
        Some(room) => draft._room = Some(room),
        None => {
            tracing::warn!(?model_id, ep = %ep_name, part, path = %path.display(), "optimized graph cannot be written here; built uncached");
            return (builder, None);
        }
    }
    match builder.with_optimized_model_path(draft.draft()) {
        Ok(builder) => {
            tracing::debug!(?model_id, ep = %ep_name, part, path = %path.display(), "cache miss (writing optimized graph)");
            (builder, Some(draft))
        }
        Err(e) => {
            tracing::warn!(?model_id, ep = %ep_name, part, %e, "with_optimized_model_path failed; cache disabled for this build");
            (e.recover(), None)
        }
    }
}

/// The draft can be made and the graph fits, twice over, beside the
/// graphs other builds are writing: ORT fails a whole session it cannot
/// write the graph for (a read-only or missing folder, a full disk), and
/// the cache must never cost the model. The room stays reserved while the
/// returned guard lives.
fn writable(draft: &Path, graph_bytes: u64) -> Option<Reserved> {
    std::fs::File::create(draft).ok()?;
    IN_FLIGHT.reserve(crate::sysmem::free_disk_bytes(draft), graph_bytes)
}

/// Disk space the optimized graphs being written will take: the parts of
/// one model build in parallel, and each finding room for itself alone
/// they filled the disk together.
struct InFlight(std::sync::atomic::AtomicU64);

static IN_FLIGHT: InFlight = InFlight::new();

impl InFlight {
    const fn new() -> Self {
        Self(std::sync::atomic::AtomicU64::new(0))
    }

    /// Room for `bytes` more on a disk with `free` (twice over, with what
    /// is in flight; unknown free space is trusted), until the guard drops.
    fn reserve(&'static self, free: Option<u64>, bytes: u64) -> Option<Reserved> {
        use std::sync::atomic::Ordering::SeqCst;
        let others = self.0.fetch_add(bytes, SeqCst);
        if free.is_none_or(|free| free / 2 >= others + bytes) {
            Some(Reserved { by: self, bytes })
        } else {
            self.0.fetch_sub(bytes, SeqCst);
            None
        }
    }
}

struct Reserved {
    by: &'static InFlight,
    bytes: u64,
}

impl Drop for Reserved {
    fn drop(&mut self) {
        self.by.0.fetch_sub(self.bytes, std::sync::atomic::Ordering::SeqCst);
    }
}

thread_local! {
    /// Drafts this thread attached, and whether it may attach them now.
    static DRAFTS: std::cell::Cell<(u32, bool)> = const { std::cell::Cell::new((0, true)) };
}

/// A draft may be written for the session this thread builds; counted,
/// so a failed build can tell it was writing one.
pub(crate) fn take_draft_slot() -> bool {
    DRAFTS.with(|d| {
        let (attached, allowed) = d.get();
        if allowed {
            d.set((attached + 1, allowed));
        }
        allowed
    })
}

/// Drafts this thread has attached so far.
pub(crate) fn drafts_attached() -> u32 {
    DRAFTS.with(|d| d.get().0)
}

/// `build` with no optimized graph written: the retry of a session that
/// failed while writing one.
pub(crate) fn without_drafts<T>(build: impl FnOnce() -> T) -> T {
    let before = DRAFTS.with(|d| d.replace((d.get().0, false)));
    let built = build();
    DRAFTS.with(|d| d.set((d.get().0, before.1)));
    built
}

fn remove_abandoned_drafts(dir: &Path, name: &str) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let prefix = format!("{name}.");
    for entry in entries.flatten() {
        let file = entry.file_name();
        let Some(file) = file.to_str() else { continue };
        let abandoned = entry.metadata().ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > ABANDONED_DRAFT);
        if file.starts_with(&prefix) && file.ends_with(".draft") && abandoned {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Per-part variant of `cache_path_for` — isolates each bundle part so
/// optimized blobs from one part can't collide with another's in shared
/// EP cache directories.
pub fn cache_path_for_part(id: ModelId, ep_name: &str, part: &str) -> Option<PathBuf> {
    cache_path_for(id, ep_name).map(|p| p.join(part))
}

/// Per-part variant of `cache_dir_for` — creates the part-scoped dir.
pub fn cache_dir_for_part(id: ModelId, ep_name: &str, part: &str) -> Option<PathBuf> {
    let dir = cache_path_for_part(id, ep_name, part)?;
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(?id, ep = %ep_name, part, %e, "failed to create per-part EP cache dir");
        return None;
    }
    Some(dir)
}

/// Mark the part `used` of `(id, ep_name)` as used now, and remove the
/// least recently used of its siblings named `prefix…` beyond `keep`.
/// Best-effort: a process still using a removed one compiles it again.
pub fn keep_newest_parts(id: ModelId, ep_name: &str, used: &str, prefix: &str, keep: usize) {
    let Some(dir) = cache_path_for(id, ep_name) else { return };
    if let Err(e) = std::fs::File::open(dir.join(used)).and_then(|f| f.set_modified(std::time::SystemTime::now())) {
        tracing::debug!(?id, ep = %ep_name, used, %e, "could not mark a cached part as used");
    }
    for stale in least_recent_parts(&dir, prefix, keep) {
        match std::fs::remove_dir_all(&stale) {
            Ok(()) => tracing::info!(?id, ep = %ep_name, part = %stale.display(), "removed a cached part not used lately"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!(?id, ep = %ep_name, part = %stale.display(), %e, "could not remove a cached part"),
        }
    }
}

/// The folders in `dir` named `prefix…` beyond the `keep` most recently
/// modified.
fn least_recent_parts(dir: &Path, prefix: &str, keep: usize) -> Vec<PathBuf> {
    let mut parts: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(dir).into_iter().flatten().flatten()
        .filter(|entry| entry.file_name().to_str().is_some_and(|name| name.starts_with(prefix)) && entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.path())))
        .collect();
    parts.sort_by(|a, b| b.0.cmp(&a.0));
    parts.into_iter().skip(keep).map(|(_, path)| path).collect()
}

/// Per-part path to the ORT-graph-optimized file. No filesystem side
/// effects — same pure/creating split as `cache_path_for`.
pub fn optimized_model_path_for_part(id: ModelId, ep_name: &str, part: &str) -> Option<PathBuf> {
    cache_path_for_part(id, ep_name, part).map(|d| d.join("optimized.onnx"))
}

/// On session-build entry, sweep stale cache entries for the same
/// `(model, ep)` pair. A stale entry shares this model's `stable_name`
/// but has a different `version` or `CACHE_FORMAT_VERSION` — i.e. an
/// older prunr install or model bump left it behind. Best-effort —
/// any I/O error logs and falls back to leaving entries alone.
///
/// Safe across processes: same-version ones agree on the current key
/// and the `remove_dir_all` loser sees `NotFound`; an older instance
/// still running just gets a cache miss and rebuilds cold.
pub fn gc_stale_for_model(id: ModelId, ep_name: &str) {
    let Some(current_path) = cache_path_for(id, ep_name) else { return };
    remove_shared_graph(&current_path);
    let Some(ep_dir) = current_path.parent().map(Path::to_path_buf) else { return };
    let Some(current_name) = current_path.file_name().map(|n| n.to_owned()) else { return };
    let our_prefix = format!("{}-", id.stable_name());

    let entries = match std::fs::read_dir(&ep_dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => {
            tracing::warn!(?id, ep = %ep_name, %e, "gc: read_dir failed");
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name() else { continue };
        if name == current_name { continue; }
        let Some(name_str) = name.to_str() else { continue };
        // `<stable>-` prefix is unambiguous given the registry convention
        // that `stable_name` uses `_` internally, never `-`.
        if !name_str.starts_with(&our_prefix) { continue; }
        if current_name.to_str().is_some_and(|current| another_cpus_entry(current, name_str, ep_name.eq_ignore_ascii_case(crate::engine::CPU_NAME))) {
            continue;
        }
        match std::fs::remove_dir_all(&path) {
            Ok(()) => tracing::info!(
                ?id, ep = %ep_name, dir = %path.display(),
                "gc: removed stale cache entry",
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!(
                ?id, ep = %ep_name, dir = %path.display(), %e,
                "gc: failed to remove stale cache entry",
            ),
        }
    }
}

/// `other` is the entry another CPU keeps under the same key as `current`:
/// on the CPU the key ends in the vector width (`cache_key`), and the rest
/// matches.
fn another_cpus_entry(current: &str, other: &str, cpu: bool) -> bool {
    fn base(key: &str) -> Option<&str> {
        key.rsplit_once('-').map(|(base, _)| base)
    }
    cpu && base(current).is_some() && base(current) == base(other)
}

fn remove_shared_graph(entry: &Path) {
    let path = entry.join(SHARED_GRAPH_FILE);
    match std::fs::remove_file(&path) {
        Ok(()) => tracing::info!(cache = %path.display(), "gc: removed the optimized graph all weights shared"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => tracing::warn!(cache = %path.display(), %e, "gc: failed to remove the shared optimized graph"),
    }
}

/// Wipe every compiled-model cache entry for `id` across **all** EP
/// subdirs, once its weights are gone. Returns total bytes reclaimed
/// (best-effort; based on a pre-removal walk).
pub fn clear_for_model(id: ModelId) -> u64 {
    let Some(root) = cache_root() else { return 0 };
    let our_prefix = format!("{}-", id.stable_name());

    let mut bytes_freed = 0u64;
    let ep_entries = match std::fs::read_dir(&root) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return 0,
        Err(e) => {
            tracing::warn!(?id, %e, "clear_for_model: read_dir(root) failed");
            return 0;
        }
    };
    for ep_entry in ep_entries.flatten() {
        let ep_dir = ep_entry.path();
        let Ok(model_entries) = std::fs::read_dir(&ep_dir) else { continue };
        for entry in model_entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
            if !name.starts_with(&our_prefix) { continue; }
            let size = walk_dir_size(&path, 0).unwrap_or(0);
            match std::fs::remove_dir_all(&path) {
                Ok(()) => {
                    bytes_freed += size;
                    tracing::info!(
                        ?id, dir = %path.display(), bytes = size,
                        "cleared compiled-model cache for uninstalled model",
                    );
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => tracing::warn!(
                    ?id, dir = %path.display(), %e,
                    "clear_for_model: remove_dir_all failed",
                ),
            }
        }
    }
    bytes_freed
}

/// Wipe the entire EP cache root. Returns the number of bytes
/// reclaimed (best-effort; based on a pre-removal directory walk).
/// Run on a background thread when called from GUI handlers — the
/// walk + remove can take hundreds of ms on multi-GB caches over
/// slow disks.
pub fn clear_all() -> u64 {
    let Some(root) = cache_root() else { return 0 };
    if !root.exists() { return 0 }
    let bytes = walk_dir_size(&root, 0).unwrap_or(0);
    if let Err(e) = std::fs::remove_dir_all(&root) {
        tracing::warn!(%e, dir = %root.display(), "failed to clear EP cache");
        return 0;
    }
    bytes
}

/// Directory size in bytes, skipping entries past `MAX_WALK_DEPTH`.
fn walk_dir_size(dir: &Path, depth: u32) -> std::io::Result<u64> {
    if depth > MAX_WALK_DEPTH { return Ok(0); }
    let mut total = 0u64;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_dir() {
            total += walk_dir_size(&entry.path(), depth + 1)?;
        } else {
            total += meta.len();
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Windows opens a folder as a file only with backup semantics.
    #[cfg(unix)]
    #[test]
    fn only_the_least_recently_used_sizes_beyond_the_kept_go() {
        let dir = std::env::temp_dir().join(format!("prunr-least-recent-parts-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let now = std::time::SystemTime::now();
        for (name, age_s) in [("size-a", 30), ("size-b", 10), ("size-c", 20), ("size-d", 40), ("head", 50)] {
            std::fs::create_dir_all(dir.join(name)).unwrap();
            let at = now - std::time::Duration::from_secs(age_s);
            std::fs::File::open(dir.join(name)).unwrap().set_modified(at).unwrap();
        }
        std::fs::write(dir.join("size-file"), b"").unwrap();
        let mut stale = least_recent_parts(&dir, "size-", 2);
        stale.sort();
        assert_eq!(stale, vec![dir.join("size-a"), dir.join("size-d")]);
        assert!(least_recent_parts(&dir, "size-", 4).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A draft is attached only where ORT can write the graph: ORT fails
    /// the whole session otherwise, and the cache must not cost the model.
    #[cfg(unix)]
    #[test]
    fn a_draft_is_attached_only_where_the_graph_can_be_written() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        assert!(writable(&dir.path().join("graph.draft"), 1024).is_some(), "a folder with room");
        assert!(writable(&dir.path().join("missing/graph.draft"), 1024).is_none(), "a folder that is not there");
        assert!(writable(&dir.path().join("graph.draft"), u64::MAX / 4).is_none(), "a graph the disk cannot hold");
        let locked = dir.path().join("locked");
        std::fs::create_dir(&locked).unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o555)).unwrap();
        assert!(writable(&locked.join("graph.draft"), 1024).is_none(), "a read-only folder");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    /// The directory-writing tests share one temp root (never the user's
    /// cache), so they still must not interleave with each other.
    static DIR_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// One directory per test process, in place of the real cache root
    /// for every `cache_root()` call made under `cfg(test)`. A static is
    /// never dropped, so a `TempDir` here leaked each run's optimized
    /// graphs (up to ~450 MB) into the system temp dir, which on Linux is
    /// often RAM. It lives beside the test binary instead, and each run
    /// removes the directories of test processes that have ended.
    pub(super) fn temp_root() -> Option<PathBuf> {
        static ROOT: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
        ROOT.get_or_init(|| {
            let base = std::env::current_exe().ok()?.parent()?.parent()?.join("test-ep-cache");
            remove_ended_runs(&base);
            let own = base.join(std::process::id().to_string());
            std::fs::create_dir_all(&own).ok()?;
            Some(own.join("ep_cache"))
        }).clone()
    }

    fn remove_ended_runs(base: &Path) {
        let Ok(entries) = std::fs::read_dir(base) else { return };
        for entry in entries.flatten() {
            let ended = entry.file_name().to_str().and_then(|pid| pid.parse::<u32>().ok())
                .is_some_and(|pid| !process_runs(pid));
            if ended {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }

    /// Whether `pid` is a running process. Only Linux can tell cheaply;
    /// elsewhere every directory is kept (CI machines are thrown away).
    fn process_runs(pid: u32) -> bool {
        if cfg!(target_os = "linux") {
            Path::new("/proc").join(pid.to_string()).exists()
        } else {
            true
        }
    }

    /// A draft becomes the cache only when published; one dropped (a
    /// failed build) leaves nothing, and a reader never sees it.
    #[test]
    fn a_graph_is_cached_only_once_its_session_built() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("optimized.onnx");
        let failed = GraphDraft::for_path(&path);
        std::fs::write(failed.draft(), b"half a gra").unwrap();
        assert!(!path.exists(), "a reader sees no cache while it is written");
        drop(failed);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0, "a failed build leaves nothing");
        let built = GraphDraft::for_path(&path);
        let other = GraphDraft::for_path(&path);
        assert_ne!(built.draft(), other.draft(), "two builds write apart");
        std::fs::write(built.draft(), b"graph").unwrap();
        built.publish();
        assert_eq!(std::fs::read(&path).unwrap(), b"graph");
        drop(other);
        assert!(graph_mark(&path, MADE_HERE).exists(), "marked as made here");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2, "only the cache and its mark stay");
    }

    /// A CPU entry that differs from this machine's only by its vector
    /// width is another machine's live cache (a synced profile, a shared
    /// home): the clean-up keeps it, and still removes an older runtime's.
    #[test]
    fn the_cleanup_keeps_another_cpus_entry() {
        let ours = "silueta-1.0.0-app0.4.8-ortbuild-ce7b65b0-v1-avx512";
        assert!(another_cpus_entry(ours, "silueta-1.0.0-app0.4.8-ortbuild-ce7b65b0-v1-x86_64", true));
        assert!(!another_cpus_entry(ours, "silueta-1.0.0-app0.4.8-ortbuild-aaaaaaaa-v1-x86_64", true), "another runtime's");
        assert!(!another_cpus_entry("silueta-1.0.0-app0.4.8-ortbuild-ce7b65b0-v1", "silueta-1.0.0-app0.4.7-ortbuild-ce7b65b0-v1", false), "a GPU entry has no CPU part");
    }

    /// Graphs written side by side (the parts of one model build at once)
    /// each count the space the others in flight will take, so they cannot
    /// all find room alone and overrun the disk together; a draft done
    /// gives its share back.
    #[test]
    fn graphs_written_at_once_share_the_free_space() {
        static IN_FLIGHT: InFlight = InFlight::new();
        const GB: u64 = 1 << 30;
        let free = Some(10 * GB);
        let first = IN_FLIGHT.reserve(free, 3 * GB).expect("3 GB of 10 fits twice over");
        assert!(IN_FLIGHT.reserve(free, 3 * GB).is_none(), "with the first's 3 GB it no longer does");
        drop(first);
        assert!(IN_FLIGHT.reserve(free, 3 * GB).is_some(), "the first's share is back");
        assert!(IN_FLIGHT.reserve(None, 100 * GB).is_some(), "an unknown disk is trusted");
    }

    /// `writable` creates the draft before ORT runs, so a session that
    /// built without writing its graph leaves an empty draft: that must
    /// not become a cache the next launch reads as a hit and fails on.
    #[test]
    fn a_draft_the_session_never_wrote_is_not_cached() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("optimized.onnx");
        let draft = GraphDraft::for_path(&path);
        assert!(writable(draft.draft(), 1).is_some());
        draft.publish();
        assert!(!path.exists(), "an empty draft is not a cache");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0, "nor is it left behind");
    }

    /// Only macOS has a retired cache, and it is never the live one.
    #[test]
    fn the_retired_cache_is_macos_application_support_only() {
        let data = PathBuf::from("/Users/a/Library/Application Support/prunr");
        let caches = PathBuf::from("/Users/a/Library/Caches");
        assert_eq!(retired_root_in(Some(data.clone()), true), Some(data.join("ep_cache")));
        assert_ne!(retired_root_in(Some(data.clone()), true), root_in(Some(data.clone()), Some(caches.clone()), true));
        assert_eq!(retired_root_in(Some(data.clone()), false), None, "elsewhere that is the live cache");
        assert_eq!(root_in(Some(data.clone()), Some(caches), false), Some(data.join("ep_cache")));
    }

    /// A graph ORT optimized is tied to that ORT build: one made by an
    /// older runtime loaded into a newer one ran the upscalers 33% slower
    /// until rebuilt. Each runtime gets its own cache entry.
    #[test]
    fn the_cache_key_names_the_runtime_that_built_it() {
        use crate::ort_runtime::tag_for;
        let old = tag_for(Some("1.24.1"), 30_000_000, 1_700_000_000);
        let same_name_new_file = tag_for(None, 31_000_000, 1_760_000_000);
        let other_build = tag_for(None, 30_000_000, 1_760_000_000);
        assert_ne!(cache_key(ModelId::Silueta, "CPU", Some(&old)), cache_key(ModelId::Silueta, "CPU", Some(&same_name_new_file)));
        assert_ne!(same_name_new_file, other_build, "a build whose info names no version still differs");
        assert!(old.starts_with("1.24.1-"), "{old}");
    }

    /// The key names what the graph was built from: a download's files (a
    /// re-export under the same version string gets an entry of its own),
    /// the app for a built-in model, and on the CPU the vector extension
    /// the graph's layout was chosen for.
    #[test]
    fn the_cache_key_names_the_files_and_the_cpu_it_was_built_for() {
        let tag = prunr_models::content_tag(ModelId::LaMaFp32).expect("a download");
        assert!(cache_key(ModelId::LaMaFp32, "OpenVINO", None).unwrap().contains(&format!("-{tag:016x}-")));
        let built_in = cache_key(ModelId::Silueta, "OpenVINO", None).unwrap();
        assert!(built_in.contains(&format!("-app{}-", prunr_models::BUILT_IN_VERSION)), "{built_in}");
        let isa = cpu_vector_isa();
        assert!(cache_key(ModelId::Silueta, "CPU", None).unwrap().ends_with(&format!("-{isa}")));
        assert!(!built_in.ends_with(&format!("-{isa}")), "a GPU graph is not laid out for the CPU");
    }

    #[test]
    fn the_macos_cache_path_has_no_space_for_coreml_to_trip_on() {
        let home = Path::new("/Users/ada");
        let data = Some(home.join("Library/Application Support/prunr"));
        let caches = Some(home.join("Library/Caches"));
        let mac = root_in(data, caches, true).unwrap();
        assert!(!mac.to_string_lossy().contains(' '), "{mac:?}");

        let linux = root_in(Some("/home/ada/.local/share/prunr".into()), Some("/home/ada/.cache".into()), false);
        assert_eq!(linux, Some(PathBuf::from("/home/ada/.local/share/prunr/ep_cache")));
    }

    #[test]
    fn tests_never_touch_the_real_cache() {
        let root = cache_root().unwrap();
        let real = root_in(prunr_models::data_dir(), dirs::cache_dir(), cfg!(target_os = "macos"));
        assert_ne!(Some(&root), real.as_ref());
        assert!(root.ends_with(PathBuf::from(std::process::id().to_string()).join("ep_cache")), "one per test run: {root:?}");
    }

    /// Cross-EP collision would corrupt a load if e.g. CPU-optimized
    /// IR were handed to OpenVINO. Same `model_id`, different EP →
    /// different paths.
    #[test]
    fn cache_path_per_ep_is_distinct() {
        let Some(_) = cache_root() else { return };
        let openvino = cache_path_for(ModelId::Silueta, "OpenVINO");
        let cpu = cache_path_for(ModelId::Silueta, "CPU");
        let coreml = cache_path_for(ModelId::Silueta, "CoreML");
        assert!(openvino.is_some() && cpu.is_some() && coreml.is_some());
        assert_ne!(openvino, cpu);
        assert_ne!(openvino, coreml);
        assert_ne!(cpu, coreml);
    }

    /// EP-name casing must not split the cache (e.g. "openvino" and
    /// "OpenVINO" landing in different subdirs would silently miss the
    /// warm cache on lookups). Lowercased uniformly.
    #[test]
    fn cache_path_normalises_ep_name_case() {
        let Some(_) = cache_root() else { return };
        let lower = cache_path_for(ModelId::Silueta, "openvino").unwrap();
        let upper = cache_path_for(ModelId::Silueta, "OPENVINO").unwrap();
        let mixed = cache_path_for(ModelId::Silueta, "OpenVINO").unwrap();
        assert_eq!(lower, upper);
        assert_eq!(lower, mixed);
    }

    /// Bumping `CACHE_FORMAT_VERSION` or the descriptor's `version`
    /// must change the cache key so a stale-format cache doesn't get
    /// loaded against a newer ORT or layout. Also: stable_name (not
    /// Debug) means a future variant rename doesn't silently
    /// invalidate every user's cache.
    #[test]
    fn cache_key_includes_stable_name_and_versions() {
        let Some(dir) = cache_path_for(ModelId::Silueta, "CPU") else { return };
        let key = dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
        assert!(key.contains(ModelId::Silueta.stable_name()),
            "cache key must contain stable_name; got {key:?}");
        let descriptor = prunr_models::descriptor(ModelId::Silueta).unwrap();
        assert!(key.contains(descriptor.version),
            "cache key must contain model version {}; got {key:?}", descriptor.version);
        assert!(key.contains(&format!("v{CACHE_FORMAT_VERSION}")),
            "cache key must contain CACHE_FORMAT_VERSION; got {key:?}");
    }

    /// Stale-entry GC must remove sibling dirs from older versions /
    /// older `CACHE_FORMAT_VERSION`s, leave the current entry alone,
    /// and never touch other models' caches.
    ///
    /// Uses `LaMaFp32` (not used by other tests in this module) to avoid
    /// races under `cargo test`'s default parallel runner.
    #[test]
    fn gc_stale_for_model_removes_siblings_only() {
        let _dir = DIR_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(current) = cache_dir_for(ModelId::LaMaFp32, "CPU") else { return };
        let Some(ep_root) = cache_root().map(|r| r.join("cpu")) else { return };
        let stale = ep_root.join(format!("{}-0.0.1-v0", ModelId::LaMaFp32.stable_name()));
        std::fs::create_dir_all(&stale).unwrap();
        std::fs::write(stale.join("dummy"), b"x").unwrap();
        let other_dir = ep_root.join(format!("{}-1.0.0-v1", ModelId::Teed.stable_name()));
        std::fs::create_dir_all(&other_dir).unwrap();
        std::fs::write(other_dir.join("dummy"), b"x").unwrap();

        gc_stale_for_model(ModelId::LaMaFp32, "CPU");

        assert!(current.is_dir(), "current cache must be preserved");
        assert!(!stale.exists(), "stale sibling must be removed");
        assert!(other_dir.is_dir(), "other model's cache must be untouched");
        let _ = std::fs::remove_dir_all(&current);
        let _ = std::fs::remove_dir_all(&other_dir);
    }

    /// `clear_for_model` must wipe all version-keyed cache dirs for the
    /// uninstalled model across every EP subdir, leave other models'
    /// caches alone, and report the byte size it reclaimed.
    ///
    /// Uses `BigLaMa` (not used elsewhere in this module) to avoid
    /// races with parallel test runs.
    #[test]
    fn clear_for_model_removes_across_all_eps() {
        let _dir = DIR_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(root) = cache_root() else { return };
        let stable = ModelId::BigLaMa.stable_name();
        let cpu_dir = root.join("cpu").join(format!("{stable}-1.0.0-v1"));
        let openvino_dir = root.join("openvino").join(format!("{stable}-1.0.0-v1"));
        let other = root.join("cpu").join(format!("{}-1.0.0-v1", ModelId::U2net.stable_name()));
        for d in [&cpu_dir, &openvino_dir, &other] {
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join("dummy"), b"abc").unwrap();
        }

        let bytes = clear_for_model(ModelId::BigLaMa);

        assert!(bytes >= 6, "should report bytes from both EPs (got {bytes})");
        assert!(!cpu_dir.exists());
        assert!(!openvino_dir.exists());
        assert!(other.exists(), "other model's cache must be untouched");
        let _ = std::fs::remove_dir_all(&other);
    }

    /// FP32, FP16 and INT8 weights optimize to different graphs: a hit
    /// on one copy's path must never hand a session another copy's graph.
    #[test]
    fn each_copy_of_the_weights_caches_its_own_graph() {
        let Some(_) = cache_root() else { return };
        let paths = Weights::ALL
            .map(|w| optimized_model_path(ModelId::Silueta, "CPU", w).unwrap());
        assert_ne!(paths[0], paths[1]);
        assert_ne!(paths[0], paths[2]);
        assert_ne!(paths[1], paths[2]);
        assert!(paths.iter().all(|p| p.parent() == cache_path_for(ModelId::Silueta, "CPU").as_deref()));
        assert!(paths.iter().all(|p| p.file_name().unwrap() != SHARED_GRAPH_FILE));
    }

    /// The graph all copies shared may hold any of them, so the sweep at
    /// session build drops it and leaves each copy's own graph.
    #[test]
    fn gc_removes_the_graph_every_copy_shared() {
        let _dir = DIR_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(dir) = cache_dir_for(ModelId::Silueta, "CPU") else { return };
        let shared = dir.join(SHARED_GRAPH_FILE);
        let own = optimized_model_path(ModelId::Silueta, "CPU", Weights::Int8).unwrap();
        std::fs::write(&shared, b"int8 graph").unwrap();
        std::fs::write(&own, b"int8 graph").unwrap();
        gc_stale_for_model(ModelId::Silueta, "CPU");
        assert!(!shared.exists(), "the shared graph is dropped");
        assert!(own.exists(), "each copy's own graph stays");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
