//! IPC protocol types for parent ↔ child subprocess communication.
//!
//! All types are serde-serializable and sent as length-prefixed bincode
//! frames over stdin/stdout.

use prunr_core::{ModelKind, MaskSettings, EdgeSettings, LineMode};
use prunr_core::inpaint_sd::SdInpaintRequest;
use prunr_models::ModelId;
use serde::{Serialize, Deserialize};

/// Paths cross in the OS's own form: a file name need not be UTF-8,
/// which serde's `Path` refuses.
mod wire_path {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::path::{Path, PathBuf};

    pub fn serialize<S: Serializer>(path: &Path, s: S) -> Result<S::Ok, S::Error> {
        path.as_os_str().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<PathBuf, D::Error> {
        std::ffi::OsString::deserialize(d).map(PathBuf::from)
    }
}

/// `ImageError.error` value emitted when a per-item `CancelItem` trips at
/// dispatch. The parent matches on this exact string to revert the item to
/// `Pending` rather than flag it as `Error`; any drift silently breaks the
/// round-trip.
pub const CANCELLED_ERR_MSG: &str = "Cancelled";

/// An image error that reports a cancel: the dispatch-time
/// `CANCELLED_ERR_MSG`, or core's own text when the cancel stopped a run
/// part-way.
pub fn is_cancel_error(error: &str) -> bool {
    error == CANCELLED_ERR_MSG || error == prunr_core::CoreError::Cancelled.to_string()
}

/// `InpaintBridgeResult::Error.error` value when the bridge's watchdog kills the
/// subprocess under system memory pressure; it gets its own toast so the user
/// knows the abort was self-defence, not a stroke-quality failure.
pub const MEMORY_PRESSURE_ABORT_MSG: &str = "memory-pressure-abort";

/// The version a worker must be to talk with this app: a worker of
/// another (an update replaced the binary while the app ran) would not
/// read its messages.
pub const WORKER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The exit code of a worker started by another version of the app.
pub const WORKER_OF_ANOTHER_VERSION: i32 = 75;

/// The longer side of the preview a run shows while it works: about a
/// canvas at most, so a preview costs a few MB whatever the photo's
/// size. Runs in the app and in the worker ask for the same.
pub const TILE_PREVIEW_EDGE: u32 = 1536;

/// Which line scales a worker runs the detector at for each image.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InferredScales {
    /// The ones the lines draw before the result; then, for the image on
    /// screen only, the rest, so the app's later scale switch there needs
    /// no new run. Fine is most of the detector's work, so a result that
    /// does not draw it waits for none of it.
    #[default]
    DrawnFirst,
    /// Only the ones the lines draw: the CLI never switches scale.
    Drawn,
}

impl InferredScales {
    /// The scales a run makes before its result: the drawn ones, with
    /// those their detector runs make anyway.
    pub fn of(self, edge: &EdgeSettings) -> &'static [prunr_core::EdgeScale] {
        match self {
            Self::DrawnFirst => prunr_core::scales_from_same_runs(edge.drawn_scales()),
            Self::Drawn => edge.drawn_scales(),
        }
    }

    /// The scales a run of `item_id` leaves for after its result: the
    /// rest when it is the image `on_screen`, none for an image in the
    /// background.
    pub fn spare(self, edge: &EdgeSettings, item_id: u64, on_screen: Option<u64>) -> Vec<prunr_core::EdgeScale> {
        if self == Self::Drawn || on_screen != Some(item_id) {
            return Vec::new();
        }
        let made = self.of(edge);
        prunr_core::EdgeScale::ALL.into_iter().filter(|s| !made.contains(s)).collect()
    }
}

/// Parent → Child commands (sent over child's stdin).
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub enum SubprocessCommand {
    /// Initialize the worker: load model, create engine pool. With
    /// `inpaint_only` no pool is built and the seg/edge fields are ignored, so
    /// the SD-inpaint worker doesn't waste GB on a seg engine.
    Init {
        model: ModelKind,
        jobs: usize,
        mask: MaskSettings,
        cpu_only: bool,
        line_mode: LineMode,
        edge: EdgeSettings,
        /// IPC temp directory (set by parent so child uses the same PID-namespaced dir)
        #[serde(with = "wire_path")]
        ipc_dir: std::path::PathBuf,
        #[serde(default)]
        inpaint_only: bool,
        /// Fade with distance, 0-100: the worker measures each photo's
        /// depth and fades its lines. The app sends 0 and fades the
        /// lines it shows from its own depth map.
        #[serde(default)]
        line_fade_pct: u8,
        #[serde(default)]
        edge_scales: InferredScales,
    },
    /// Process a single image. `image_path` points to a temp file with
    /// the raw image bytes (avoids piping large payloads through stdin).
    ProcessImage {
        item_id: u64,
        #[serde(with = "wire_path")]
        image_path: std::path::PathBuf,
        /// For a run that starts from an earlier result: path to temp file with its RGBA + dimensions.
        chain_input: Option<ChainInput>,
    },
    /// Tier 2: Re-run postprocess from a cached tensor (skip inference).
    /// The parent sends the raw tensor + original image via temp files.
    RePostProcess {
        item_id: u64,
        /// Path to temp file with raw f32 tensor data.
        #[serde(with = "wire_path")]
        tensor_path: std::path::PathBuf,
        tensor_height: u32,
        tensor_width: u32,
        model: ModelKind,
        #[serde(with = "wire_path")]
        original_image_path: std::path::PathBuf,
        mask: MaskSettings,
    },
    /// AddEdgeInference tier: seg tensor is cached, run only TEED on the
    /// masked image and compose. Used for Off → SubjectOutline transitions so
    /// enabling the outline doesn't re-run seg inference.
    AddEdgeInference {
        item_id: u64,
        #[serde(with = "wire_path")]
        image_path: std::path::PathBuf,
        /// Path to temp file with the cached raw f32 seg tensor. It
        /// is NOT deleted by the worker: it comes back as `tensor_cache` on
        /// `ImageDone` for the parent to read-and-delete, saving a copy.
        #[serde(with = "wire_path")]
        seg_tensor_path: std::path::PathBuf,
        seg_tensor_height: u32,
        seg_tensor_width: u32,
        /// Model that produced the seg tensor (must match Init's model).
        model: ModelKind,
        /// Per-item mask settings (may differ from Init's mask).
        mask: MaskSettings,
    },
    /// Inpaint a region with `model_id` and the full post-process (color
    /// match, seam-guided blend, sharpen), returning the finished RGBA. Runs
    /// its own session, independent of the seg/edge engine pool.
    Inpaint(InpaintJob),
    /// Cancel one item by id — worker emits `ImageError { error:
    /// CANCELLED_ERR_MSG }` at the next dispatch check.
    CancelItem { item_id: u64 },
    /// New mask, Sketch mode and line settings for a worker already
    /// running: every later `ProcessImage` and `AddEdgeInference` reads
    /// them, and the worker builds the engines the mode needs that it
    /// does not have yet, so a change never costs a respawn.
    Configure {
        mask: MaskSettings,
        line_mode: LineMode,
        edge: EdgeSettings,
        /// The image whose run reports a first look at the subject before
        /// its own; `None` lets the first-look model go.
        #[serde(default)]
        first_look: Option<u64>,
        /// The image on screen: its Sketch runs make the line scales they
        /// do not draw after their result.
        #[serde(default)]
        on_screen: Option<u64>,
    },
    Shutdown,
}

/// One inpaint stroke, as the app's bridge queues it and the worker runs it.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct InpaintJob {
    pub item_id: u64,
    /// The stroke on this item; names its files and comes back on its
    /// Done or Error, since strokes on one item can run at once.
    pub generation: u64,
    pub model_id: ModelId,
    /// The photo as `ipc::write_image` writes it.
    #[serde(with = "wire_path")]
    pub image_path: std::path::PathBuf,
    /// Single-channel mask, the same way: 255 = inpaint here, 0 = keep.
    #[serde(with = "wire_path")]
    pub mask_path: std::path::PathBuf,
    /// SD-specific knobs. `None` for LaMa / Big-LaMa / MI-GAN.
    pub sd_req: Option<SdInpaintRequest>,
    /// Seam-blend feather override (px). 0 ⇒ pipeline default.
    #[serde(default)]
    pub feather_px: f32,
    /// Unsharp-mask strength applied inside the painted region.
    /// 0 ⇒ skip sharpen.
    #[serde(default)]
    pub sharpen: f32,
    #[serde(default)]
    pub edge_colour: prunr_core::inpaint_blend::EdgeColour,
}

/// The image a run starts from when it is an earlier result: raw RGBA in a temp file.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct ChainInput {
    #[serde(with = "wire_path")]
    pub path: std::path::PathBuf,
    pub width: u32,
    pub height: u32,
}

/// Sends a pipeline's progress reports as `SubprocessEvent::Report` for
/// `item_id`, on the same channel as its result, so a report never
/// arrives after the result. It keeps the last step, and when it came,
/// for the run's pulse.
struct IpcProgress {
    item_id: u64,
    tx: std::sync::mpsc::Sender<SubprocessEvent>,
    step: std::sync::Mutex<Option<(prunr_core::Step, std::time::Instant)>>,
}

impl IpcProgress {
    fn send(&self, update: prunr_core::ProgressUpdate) -> bool {
        self.tx.send(SubprocessEvent::Report { item_id: self.item_id, update }).is_ok()
    }
}

impl prunr_core::ProgressSink for IpcProgress {
    fn report(&self, update: prunr_core::ProgressUpdate) {
        if let prunr_core::ProgressUpdate::Step(step) = update {
            *self.step.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some((step, std::time::Instant::now()));
        }
        self.send(update);
    }
}

/// How often a worker says it is still at work, well inside the parent's
/// hang watchdog.
pub const PULSE_EVERY: std::time::Duration = std::time::Duration::from_secs(10);

/// A run this long since its last step of its own is taken for hung all
/// the same, so a deadlock inside one is still caught.
const PULSE_LIMIT: std::time::Duration = std::time::Duration::from_secs(600);

/// A progress handle whose reports go to the parent as events, and the
/// run's pulse: while it is held the run's last step goes out again every
/// `PULSE_EVERY`, so one long model run (a minute on a slow CPU) is not
/// taken for a hung worker.
pub fn ipc_progress(item_id: u64, tx: std::sync::mpsc::Sender<SubprocessEvent>) -> (prunr_core::Progress, Pulse) {
    pulsing_progress(item_id, tx, PULSE_EVERY, PULSE_LIMIT)
}

fn pulsing_progress(
    item_id: u64,
    tx: std::sync::mpsc::Sender<SubprocessEvent>,
    every: std::time::Duration,
    limit: std::time::Duration,
) -> (prunr_core::Progress, Pulse) {
    let sink = std::sync::Arc::new(IpcProgress { item_id, tx, step: std::sync::Mutex::new(None) });
    let beating = std::sync::Arc::clone(&sink);
    let pulse = Pulse::start(every, move || {
        let step = *beating.step.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        step.is_none_or(|(s, at)| at.elapsed() < limit && beating.send(prunr_core::ProgressUpdate::Step(s)))
    });
    (prunr_core::Progress::new(sink), pulse)
}

/// Calls `beat` now and every `every` after, on a thread of its own,
/// until dropped or until `beat` returns false. Dropping it waits out a
/// beat under way, so nothing it sends lands after.
pub struct Pulse {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    beating: Option<std::thread::JoinHandle<()>>,
}

impl Pulse {
    pub fn start(every: std::time::Duration, beat: impl Fn() -> bool + Send + 'static) -> Self {
        use std::sync::atomic::{AtomicBool, Ordering};
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let stopped = std::sync::Arc::clone(&stop);
        let beating = std::thread::spawn(move || {
            while !stopped.load(Ordering::Acquire) && beat() {
                std::thread::park_timeout(every);
            }
        });
        Self { stop, beating: Some(beating) }
    }
}

impl Drop for Pulse {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(beating) = self.beating.take() {
            beating.thread().unpark();
            let _ = beating.join();
        }
    }
}

/// Send a run's answer as the last thing it sends: its pulse stops first,
/// so no beat repeats the finished image's step after it.
pub fn answer(tx: &std::sync::mpsc::Sender<SubprocessEvent>, pulse: Pulse, event: SubprocessEvent) {
    drop(pulse);
    let _ = tx.send(event);
}

/// A raw f32 segmentation tensor a worker left in a file, `height` × `width`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TensorCacheFile {
    #[serde(with = "wire_path")]
    pub path: std::path::PathBuf,
    pub height: u32,
    pub width: u32,
}

/// The line detector's planes (f32 probabilities, pre-threshold) a
/// worker wrote for an image: `scales`' planes one after another in that
/// order, each at its own size, `edge_scale_dims` of the base `height` ×
/// `width`. A run may write only some scales.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EdgeCacheFile {
    #[serde(with = "wire_path")]
    pub path: std::path::PathBuf,
    pub height: u32,
    pub width: u32,
    pub scales: Vec<prunr_core::EdgeScale>,
}

/// What became of a run's spare line scales.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SpareOutcome {
    Made(EdgeCacheFile),
    /// Not made on purpose: the image left the screen, a cancel, a
    /// shutdown. Nothing should run in their place.
    Skipped,
    /// The detector or the file write failed.
    Failed,
}

/// Child → Parent events (sent over child's stdout).
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub enum SubprocessEvent {
    /// Engine pool created successfully.
    Ready {
        active_provider: String,
    },
    /// Any progress report from a pipeline (`prunr_core::Progress`): a
    /// new step or count needs no new variant.
    Report {
        item_id: u64,
        update: prunr_core::ProgressUpdate,
    },
    /// Image processed successfully.
    /// Result RGBA is written to `result_path` temp file (not piped).
    ImageDone {
        item_id: u64,
        #[serde(with = "wire_path")]
        result_path: std::path::PathBuf,
        width: u32,
        height: u32,
        active_provider: String,
        /// The raw segmentation tensor (Tier 1 output). Only present for
        /// full pipeline runs, not RePostProcess.
        tensor_cache: Option<TensorCacheFile>,
        /// The line detector's planes, for Tier 2 edge reruns. Only
        /// present when the run used TEED (EdgesOnly or SubjectOutline).
        #[serde(default)]
        edge_cache: Option<EdgeCacheFile>,
        /// A `SpareEdgeScales` with this token follows: the run left the
        /// scales it does not draw for after its result.
        #[serde(default)]
        spare_scales: Option<u64>,
    },
    /// The line scales a run left for after its `ImageDone`, under the
    /// token that event named.
    SpareEdgeScales {
        item_id: u64,
        token: u64,
        outcome: SpareOutcome,
    },
    /// Image processing failed (non-fatal).
    ImageError {
        item_id: u64,
        error: String,
    },
    /// Inpaint completed — raw RGBA, `width` × `height`, lives at
    /// `rgba_path`. Same read-and-delete contract as `ImageDone.result_path`.
    InpaintDone {
        item_id: u64,
        generation: u64,
        #[serde(with = "wire_path")]
        rgba_path: std::path::PathBuf,
        width: u32,
        height: u32,
        /// How large the region was filled: 1 is full size.
        fill_scale: f32,
    },
    /// Inpaint dispatch failed (model missing, ORT error, IO error).
    InpaintError {
        item_id: u64,
        generation: u64,
        error: String,
    },
    /// Current RSS of the subprocess (sent after each image).
    RssUpdate {
        rss_bytes: u64,
    },
    /// All work complete (or cancelled).
    Finished,
    /// Fatal initialization error (engine creation failed).
    InitError {
        error: String,
    },
}

/// PID-namespaced IPC temp dir, RAM-backed (/dev/shm) on Linux when available.
/// Its one-time init sweeps this PID's stale files and dead-PID sibling dirs,
/// so writers never race a later cleanup and /dev/shm doesn't fill up.
pub fn ipc_temp_dir() -> &'static std::path::Path {
    use std::sync::OnceLock;
    static DIR: OnceLock<std::path::PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let pid = std::process::id();
        let dir = init_temp_dir_for_pid(pid);
        if let Some(parent) = dir.parent() {
            crate::fs_util::sweep_dead_pid_dirs(parent, "prunr-ipc-", pid);
        }
        dir
    }).as_path()
}

/// `ipc_temp_dir()`'s init without the dead-PID sweep, so tests can drive
/// it with synthetic PIDs in isolation.
fn init_temp_dir_for_pid(pid: u32) -> std::path::PathBuf {
    let dir = resolve_temp_dir_path(pid);
    let _ = std::fs::create_dir_all(&dir);
    crate::fs_util::sweep_dir_files(&dir);
    dir
}

fn resolve_temp_dir_path(pid: u32) -> std::path::PathBuf {
    #[cfg(target_os = "linux")]
    if std::path::Path::new("/dev/shm").is_dir() && shm_is_roomy() {
        return std::path::PathBuf::from(format!("/dev/shm/prunr-ipc-{pid}"));
    }
    std::env::temp_dir().join(format!("prunr-ipc-{pid}"))
}

/// A `/dev/shm` this small is a container's (Docker gives 64 MB): one
/// 24 MP result alone is 96 MB.
#[cfg(target_os = "linux")]
const SHM_MIN_BYTES: u64 = 1 << 30;

#[cfg(target_os = "linux")]
fn shm_is_roomy() -> bool {
    let mounts = std::fs::read_to_string("/proc/mounts").unwrap_or_default();
    tmpfs_size(&mounts, "/dev/shm").is_none_or(|bytes| bytes >= SHM_MIN_BYTES)
}

/// The fixed size of the tmpfs mounted at `at`, from `/proc/mounts`;
/// `None` when it has none (a share of RAM, the default half).
#[cfg(any(target_os = "linux", test))]
fn tmpfs_size(mounts: &str, at: &str) -> Option<u64> {
    let options = mounts.lines().find_map(|line| {
        let mut fields = line.split_whitespace().skip(1);
        (fields.next() == Some(at) && fields.next() == Some("tmpfs")).then(|| fields.next()).flatten()
    })?;
    let size = options.split(',').find_map(|o| o.strip_prefix("size="))?;
    let (digits, shift) = match size.as_bytes().last()? {
        b'k' | b'K' => (&size[..size.len() - 1], 10),
        b'm' | b'M' => (&size[..size.len() - 1], 20),
        b'g' | b'G' => (&size[..size.len() - 1], 30),
        _ => (size, 0),
    };
    digits.parse::<u64>().ok().map(|n| n << shift)
}

/// Remove every file directly inside `dir`, restricted to files whose name
/// starts with one of the supplied prefixes.
fn sweep_dir_with_prefix(dir: &std::path::Path, prefixes: &[&str]) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        // `file_name` must be bound so `to_str()`'s &str doesn't dangle.
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else { continue };
        if prefixes.iter().any(|p| name.starts_with(p)) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Every IPC temp-file kind: its file-name prefix and extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpcKind {
    /// Parent → seg worker: encoded source bytes.
    Input,
    /// Parent → seg worker: chained RGBA from a prior tier.
    Chain,
    /// Parent → seg worker: cached segmentation tensor for the
    /// AddEdgeInference path (skip re-running the seg model).
    Seg,
    /// Parent → seg worker: cached segmentation tensor for the
    /// RePostProcess path. Distinct from `Seg` so the two cache flavours
    /// stay separable in temp-dir scans and cleanup.
    Tensor,
    /// Parent → seg worker: original encoded image for a re-postprocess.
    Orig,
    /// Seg worker → parent: composited RGBA result.
    Result,
    /// Seg worker → parent: TEED multi-scale edge tensor cache.
    Edge,
    /// Seg worker → parent: the line scales a run made after its result,
    /// named by the pass's token rather than the image, so no other
    /// file of the image (or a later run's) ever shares its path.
    SpareEdge,
    /// CLI parent: downscaled image temp before subprocess dispatch.
    CliDs,
    /// Parent → inpaint worker: the raw source for a stroke (generation-suffixed).
    InpaintImg,
    /// Parent → inpaint worker: the raw mask for a stroke (generation-suffixed).
    InpaintMask,
    /// Inpaint worker → parent: composited raw RGBA result.
    InpaintOut,
}

impl IpcKind {
    /// Filename prefix (everything before the id). Used by sweep / cleanup.
    pub const fn prefix(self) -> &'static str {
        match self {
            IpcKind::Input => "input_",
            IpcKind::Chain => "chain_",
            IpcKind::Seg => "seg_",
            IpcKind::Tensor => "tensor_",
            IpcKind::Orig => "orig_",
            IpcKind::Result => "result_",
            IpcKind::Edge => "edge_",
            IpcKind::SpareEdge => "spare_edge_",
            IpcKind::CliDs => "cli_ds_",
            IpcKind::InpaintImg => "inpaint-img-",
            IpcKind::InpaintMask => "inpaint-mask-",
            IpcKind::InpaintOut => "inpaint-out-",
        }
    }

    /// Filename extension (without leading dot).
    pub const fn ext(self) -> &'static str {
        match self {
            IpcKind::Input | IpcKind::Orig | IpcKind::CliDs => "img",
            IpcKind::Chain | IpcKind::Seg | IpcKind::Tensor | IpcKind::Result | IpcKind::Edge | IpcKind::SpareEdge => "raw",
            IpcKind::InpaintImg | IpcKind::InpaintMask | IpcKind::InpaintOut => "png",
        }
    }

    /// Build a path for an item-keyed file: `<ipc>/<prefix><id>.<ext>`.
    pub fn path_for(self, ipc_dir: &std::path::Path, id: u64) -> std::path::PathBuf {
        ipc_dir.join(format!("{}{id}.{}", self.prefix(), self.ext()))
    }

    /// Build a path for a generation-suffixed inpaint file:
    /// `<ipc>/<prefix><id>-<generation>.<ext>`. Only valid for the three
    /// inpaint kinds: strokes on one item can run at once, so each stroke's
    /// files carry its generation.
    pub fn path_for_generation(self, ipc_dir: &std::path::Path, id: u64, generation: u64) -> std::path::PathBuf {
        // A full assert, not a debug one: a wrong variant would produce
        // e.g. `result_5-0.raw` in release, a name no reader expects.
        assert!(
            matches!(self, IpcKind::InpaintImg | IpcKind::InpaintMask | IpcKind::InpaintOut),
            "path_for_generation only valid for the inpaint kinds; got {self:?}"
        );
        ipc_dir.join(format!("{}{id}-{generation}.{}", self.prefix(), self.ext()))
    }
}

/// File-name prefixes the seg worker writes or consumes-and-deletes
/// during a dispatch. Crash-recovery cleanup wipes these because the
/// worker may have died mid-write or with partial input.
///
/// `CliDs` is excluded on purpose: the CLI re-sends those on every retry,
/// and wiping them once surfaced as bogus "Insufficient memory" errors.
pub const SEG_PIPELINE_PREFIXES: &[&str] = &[
    IpcKind::Chain.prefix(),
    IpcKind::Edge.prefix(),
    IpcKind::Input.prefix(),
    IpcKind::Orig.prefix(),
    IpcKind::Result.prefix(),
    IpcKind::Seg.prefix(),
    IpcKind::Tensor.prefix(),
    IpcKind::SpareEdge.prefix(),
];

/// File-name prefixes the CLI parent owns *across* worker crashes: its
/// downscaled temps, re-sent on every retry, swept once the batch ends.
pub const CLI_PERSISTENT_PREFIXES: &[&str] = &[
    IpcKind::CliDs.prefix(),
];

/// Crash-recovery cleanup for the seg / CLI subprocess, scoped to its own
/// file names so it doesn't race a sibling writing the shared dir: a
/// sibling inpaint subprocess's `inpaint-*` files and the CLI's retry
/// inputs are left alone.
pub fn cleanup_seg_pipeline_temps() {
    sweep_dir_with_prefix(ipc_temp_dir(), SEG_PIPELINE_PREFIXES);
}

/// End-of-batch cleanup for CLI-owned downscaled temps. Call once the
/// run_batch loop has fully exited so the worker has stopped reading
/// from `cli_ds_*` paths. Idempotent.
pub fn cleanup_cli_persistent_temps() {
    sweep_dir_with_prefix(ipc_temp_dir(), CLI_PERSISTENT_PREFIXES);
}

#[cfg(test)]
mod tests {
    //! Bincode round-trip tests for every `SubprocessCommand` and
    //! `SubprocessEvent` variant. A single generic helper covers both enums
    //! via `PartialEq` — any drift between encoded and decoded payload fails
    //! the assertion with a real diff, not a stringified debug comparison.
    use super::*;
    use bincode::config::standard;
    use serde::de::DeserializeOwned;
    use std::fmt::Debug;
    use std::path::PathBuf;

    #[test]
    fn a_cancel_reads_as_a_cancel_whoever_reported_it() {
        assert!(is_cancel_error(CANCELLED_ERR_MSG));
        assert!(is_cancel_error(&prunr_core::CoreError::Cancelled.message()));
        assert!(!is_cancel_error(&prunr_core::CoreError::Inference("out of memory".into()).message()));
    }

    fn roundtrip<T>(value: &T)
    where
        T: serde::Serialize + DeserializeOwned + PartialEq + Debug,
    {
        let bytes = bincode::serde::encode_to_vec(value, standard()).unwrap();
        let (decoded, _): (T, _) =
            bincode::serde::decode_from_slice(&bytes, standard()).unwrap();
        assert_eq!(value, &decoded);
    }

    /// Each command's index in `COMMANDS`. No wildcard: a new command
    /// does not compile until it is named here, and its index fails
    /// `every_message_round_trips` until it has a sample.
    fn command_index(command: &SubprocessCommand) -> usize {
        match command {
            SubprocessCommand::Init { .. } => 0,
            SubprocessCommand::ProcessImage { .. } => 1,
            SubprocessCommand::RePostProcess { .. } => 2,
            SubprocessCommand::AddEdgeInference { .. } => 3,
            SubprocessCommand::Inpaint(_) => 4,
            SubprocessCommand::CancelItem { .. } => 5,
            SubprocessCommand::Configure { .. } => 6,
            SubprocessCommand::Shutdown => 7,
        }
    }
    const COMMANDS: usize = 8;

    /// Each event's index in `EVENTS`, as for commands.
    fn event_index(event: &SubprocessEvent) -> usize {
        match event {
            SubprocessEvent::Ready { .. } => 0,
            SubprocessEvent::Report { .. } => 1,
            SubprocessEvent::ImageDone { .. } => 2,
            SubprocessEvent::SpareEdgeScales { .. } => 3,
            SubprocessEvent::ImageError { .. } => 4,
            SubprocessEvent::InpaintDone { .. } => 5,
            SubprocessEvent::InpaintError { .. } => 6,
            SubprocessEvent::RssUpdate { .. } => 7,
            SubprocessEvent::Finished => 8,
            SubprocessEvent::InitError { .. } => 9,
        }
    }
    const EVENTS: usize = 10;

    /// A path that is not Unicode where the OS allows one, so a path
    /// field that crosses the way serde's `Path` does fails its sample.
    fn path() -> PathBuf {
        #[cfg(unix)]
        {
            <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(b"/tmp/prunr-sampl\xe9").into()
        }
        #[cfg(windows)]
        {
            <std::ffi::OsString as std::os::windows::ffi::OsStringExt>::from_wide(&[0x43, 0xD800]).into()
        }
        #[cfg(not(any(unix, windows)))]
        {
            PathBuf::from("/tmp/prunr-sample")
        }
    }

    fn command_samples() -> Vec<SubprocessCommand> {
        vec![
            SubprocessCommand::Init {
                model: ModelKind::Silueta,
                jobs: 1,
                mask: MaskSettings::default(),
                cpu_only: false,
                line_mode: LineMode::Off,
                edge: EdgeSettings::default(),
                ipc_dir: path(),
                inpaint_only: false,
                line_fade_pct: 0,
                edge_scales: InferredScales::Drawn,
            },
            SubprocessCommand::ProcessImage { item_id: 1, image_path: path(), chain_input: Some(ChainInput { path: path(), width: 2, height: 2 }) },
            SubprocessCommand::RePostProcess {
                item_id: 1,
                tensor_path: path(),
                tensor_height: 2,
                tensor_width: 2,
                model: ModelKind::Silueta,
                original_image_path: path(),
                mask: MaskSettings::default(),
            },
            SubprocessCommand::AddEdgeInference {
                item_id: 1,
                image_path: path(),
                seg_tensor_path: path(),
                seg_tensor_height: 2,
                seg_tensor_width: 2,
                model: ModelKind::Silueta,
                mask: MaskSettings::default(),
            },
            SubprocessCommand::Inpaint(InpaintJob {
                item_id: 1,
                generation: 1,
                model_id: prunr_models::ModelId::LaMaFp32,
                image_path: path(),
                mask_path: path(),
                sd_req: None,
                feather_px: 0.0,
                sharpen: 0.0,
                edge_colour: prunr_core::inpaint_blend::EdgeColour::AlongEdge,
            }),
            SubprocessCommand::CancelItem { item_id: 1 },
            SubprocessCommand::Configure { mask: MaskSettings::default(), line_mode: LineMode::Off, edge: EdgeSettings::default(), first_look: None, on_screen: None },
            SubprocessCommand::Shutdown,
        ]
    }

    fn event_samples() -> Vec<SubprocessEvent> {
        vec![
            SubprocessEvent::Ready { active_provider: "CPU".into() },
            SubprocessEvent::Report { item_id: 1, update: prunr_core::ProgressUpdate::Step(prunr_core::Step::Denoising) },
            SubprocessEvent::ImageDone {
                item_id: 1,
                result_path: path(),
                width: 2,
                height: 2,
                active_provider: "CPU".into(),
                tensor_cache: Some(TensorCacheFile { path: path(), height: 2, width: 2 }),
                edge_cache: Some(EdgeCacheFile { path: path(), height: 2, width: 2, scales: vec![prunr_core::EdgeScale::Bold] }),
                spare_scales: Some(1),
            },
            SubprocessEvent::SpareEdgeScales {
                item_id: 1,
                token: 1,
                outcome: SpareOutcome::Made(EdgeCacheFile { path: path(), height: 2, width: 2, scales: vec![prunr_core::EdgeScale::Fine] }),
            },
            SubprocessEvent::ImageError { item_id: 1, error: "failed".into() },
            SubprocessEvent::InpaintDone { item_id: 1, generation: 1, rgba_path: path(), width: 2, height: 2, fill_scale: 1.0 },
            SubprocessEvent::InpaintError { item_id: 1, generation: 1, error: "failed".into() },
            SubprocessEvent::RssUpdate { rss_bytes: 1 },
            SubprocessEvent::Finished,
            SubprocessEvent::InitError { error: "failed".into() },
        ]
    }

    /// Every command and event the worker and the app trade has a sample
    /// that survives the wire.
    #[test]
    fn every_message_round_trips() {
        let mut commands = [false; COMMANDS];
        for command in command_samples() {
            roundtrip(&command);
            commands[command_index(&command)] = true;
        }
        let mut events = [false; EVENTS];
        for event in event_samples() {
            roundtrip(&event);
            events[event_index(&event)] = true;
        }
        assert!(commands.iter().all(|&c| c), "a command without a sample: {commands:?}");
        assert!(events.iter().all(|&e| e), "an event without a sample: {events:?}");
    }

    #[test]
    fn command_init_roundtrip() {
        roundtrip(&SubprocessCommand::Init {
            model: ModelKind::Silueta,
            jobs: 2,
            mask: MaskSettings::default(),
            cpu_only: false,
            line_mode: LineMode::Off,
            edge: EdgeSettings { line_strength: 0.5, solid_line_color: Some([255, 0, 0]), edge_thickness: 0, edge_scale: prunr_core::EdgeScale::Bold, compose_mode: prunr_core::ComposeMode::default(), line_style: prunr_core::LineStyle::default(), input_transform: prunr_core::InputTransform::default() },
            ipc_dir: PathBuf::from("/tmp/prunr-ipc-test"),
            inpaint_only: false,
            line_fade_pct: 40,
            edge_scales: InferredScales::DrawnFirst,
        });
        roundtrip(&SubprocessCommand::Init {
            model: ModelKind::Silueta,
            jobs: 1,
            mask: MaskSettings::default(),
            cpu_only: false,
            line_mode: LineMode::Off,
            edge: EdgeSettings::default(),
            ipc_dir: PathBuf::from("/tmp/prunr-ipc-test"),
            inpaint_only: true,
            line_fade_pct: 0,
            edge_scales: InferredScales::Drawn,
        });
    }

    #[test]
    fn command_process_image_with_and_without_chain_roundtrip() {
        roundtrip(&SubprocessCommand::ProcessImage {
            item_id: 7,
            image_path: PathBuf::from("/tmp/img.png"),
            chain_input: Some(ChainInput {
                path: PathBuf::from("/tmp/chain.bin"),
                width: 1920,
                height: 1080,
            }),
        });
        roundtrip(&SubprocessCommand::ProcessImage {
            item_id: 8,
            image_path: PathBuf::from("/tmp/img2.png"),
            chain_input: None,
        });
    }

    #[test]
    fn command_configure_roundtrip() {
        roundtrip(&SubprocessCommand::Configure {
            mask: MaskSettings { gamma: 1.4, ..MaskSettings::default() },
            line_mode: LineMode::SubjectOutline,
            edge: EdgeSettings { line_strength: 0.8, solid_line_color: None, edge_thickness: 2, edge_scale: prunr_core::EdgeScale::Fine, compose_mode: prunr_core::ComposeMode::default(), line_style: prunr_core::LineStyle::default(), input_transform: prunr_core::InputTransform::default() },
            first_look: Some(12),
            on_screen: Some(12),
        });
        roundtrip(&SubprocessCommand::Configure {
            mask: MaskSettings::default(),
            line_mode: LineMode::Off,
            edge: EdgeSettings::default(),
            first_look: None,
            on_screen: Some(3),
        });
    }

    #[test]
    fn event_image_done_with_and_without_edge_cache_roundtrip() {
        roundtrip(&SubprocessEvent::ImageDone {
            item_id: 11,
            result_path: PathBuf::from("/tmp/result.bin"),
            width: 1000,
            height: 1000,
            active_provider: "CPU".to_string(),
            tensor_cache: Some(TensorCacheFile { path: PathBuf::from("/tmp/tensor.bin"), height: 320, width: 320 }),
            edge_cache: None,
            spare_scales: None,
        });
        roundtrip(&SubprocessEvent::ImageDone {
            item_id: 12,
            result_path: PathBuf::from("/tmp/result2.bin"),
            width: 500,
            height: 500,
            active_provider: "CoreML".to_string(),
            tensor_cache: None,
            edge_cache: Some(EdgeCacheFile {
                path: PathBuf::from("/tmp/edge.bin"),
                height: 480,
                width: 640,
                scales: vec![prunr_core::EdgeScale::Bold, prunr_core::EdgeScale::Fused],
            }),
            spare_scales: Some(u64::MAX - 1),
        });
    }

    #[test]
    fn event_spare_edge_scales_roundtrip() {
        roundtrip(&SubprocessEvent::SpareEdgeScales { item_id: 5, token: 77, outcome: SpareOutcome::Skipped });
        roundtrip(&SubprocessEvent::SpareEdgeScales { item_id: 5, token: 78, outcome: SpareOutcome::Failed });
        roundtrip(&SubprocessEvent::SpareEdgeScales {
            item_id: 6,
            token: (1 << 32) | 3,
            outcome: SpareOutcome::Made(EdgeCacheFile {
                path: PathBuf::from("/tmp/spare_edge_6.raw"),
                height: 480,
                width: 640,
                scales: vec![prunr_core::EdgeScale::Fine],
            }),
        });
    }

    #[test]
    fn command_inpaint_sd_roundtrip() {
        roundtrip(&SubprocessCommand::Inpaint(InpaintJob {
            item_id: 12,
            generation: 3,
            model_id: prunr_models::ModelId::SdV15InpaintFp16,
            image_path: std::path::PathBuf::from("/tmp/prunr-inpaint-img-12"),
            mask_path: std::path::PathBuf::from("/tmp/prunr-inpaint-mask-12"),
            sd_req: Some(prunr_core::inpaint_sd::SdInpaintRequest {
                prompt: "remove subject".into(),
                negative_prompt: "blurry".into(),
                num_inference_steps: 20,
                guidance_scale: 7.5,
                seed: Some(42),
                use_taesd: false,
                scheduler: prunr_core::inpaint_sd::SchedulerKind::DpmPp2MKarras,
                tuning: prunr_core::inpaint_sd::SdTuning { keep_loaded: true, ov_device: Some("GPU".into()), tall_crop: true, margin_mb: 512, cpu_only: true, mem: prunr_core::inpaint_sd::SdMemPlan { memory_pattern: Some(false), device_initializers: true, intra_threads: 4 } },
                fill_scale: prunr_core::inpaint::FillScale::Quarter,
                soft_edge_px: 22.0,
            }),
            feather_px: 4.5,
            sharpen: 0.6,
            edge_colour: prunr_core::inpaint_blend::EdgeColour::Off,
        }));
    }

    #[test]
    fn event_inpaint_done_and_error_roundtrip() {
        roundtrip(&SubprocessEvent::InpaintDone {
            item_id: 11,
            generation: 3,
            rgba_path: std::path::PathBuf::from("/tmp/prunr-inpaint-out-11"),
            width: 1920,
            height: 1080,
            fill_scale: 0.5,
        });
        roundtrip(&SubprocessEvent::InpaintError {
            item_id: 11,
            generation: 3,
            error: "lama session failed: tensor shape mismatch".into(),
        });
    }

    #[test]
    fn event_report_roundtrips_every_update() {
        use prunr_core::{ProgressUpdate, Step, TileRect, Unit};
        for update in [
            ProgressUpdate::Step(Step::Denoising),
            ProgressUpdate::Outer { done: 1, total: 2, unit: Unit::Crop },
            ProgressUpdate::Inner { done: 12, total: 20, unit: Unit::Step },
            ProgressUpdate::Tiles(vec![TileRect { x: 0.0, y: 0.5, w: 0.25, h: 0.5 }]),
            ProgressUpdate::Tile { index: 0, done: true },
            ProgressUpdate::Subject(prunr_core::SubjectHint { width: 2, height: 1, coverage: vec![0, 255] }),
            ProgressUpdate::Faces(vec![prunr_core::face::FacePlace::of(
                &prunr_core::face::Face { score: 0.9, bbox: [0.0; 4], landmarks: [[30.0, 40.0], [60.0, 40.0], [45.0, 55.0], [33.0, 70.0], [57.0, 70.0]] },
                [96, 128],
            )]),
            ProgressUpdate::TilePreview(prunr_core::TilePreview { x: 1, y: 0, width: 1, height: 1, full_width: 2, full_height: 1, rgba: vec![1, 2, 3, 4] }),
        ] {
            roundtrip(&SubprocessEvent::Report { item_id: 9, update });
        }
    }

    /// While a run's pulse is held its last step goes out again and
    /// again, so one long model run is not taken for a hung worker: none
    /// before a first step, and the channel closes once both are gone.
    #[test]
    fn a_run_pulses_its_last_step() {
        use std::time::Duration;
        let wait = Duration::from_secs(10);
        let (tx, rx) = std::sync::mpsc::channel();
        let (progress, pulse) = super::pulsing_progress(4, tx, Duration::from_millis(20), Duration::from_secs(600));
        progress.step(prunr_core::Step::FindingSubject);
        let said = || SubprocessEvent::Report { item_id: 4, update: prunr_core::ProgressUpdate::Step(prunr_core::Step::FindingSubject) };
        for _ in 0..3 {
            assert_eq!(rx.recv_timeout(wait), Ok(said()));
        }
        drop((pulse, progress));
        let end = loop {
            if let Err(e) = rx.recv_timeout(wait) {
                break e;
            }
        };
        assert_eq!(end, std::sync::mpsc::RecvTimeoutError::Disconnected);
    }

    /// A run's answer is the last thing it sends: its pulse stops first,
    /// so no beat repeats the finished image's step after it (the spare
    /// edge scales can keep the job going for minutes).
    #[test]
    fn nothing_follows_a_runs_answer() {
        use std::time::Duration;
        for _ in 0..20 {
            let (tx, rx) = std::sync::mpsc::channel();
            let (progress, pulse) = super::pulsing_progress(4, tx.clone(), Duration::from_millis(1), Duration::from_secs(600));
            progress.step(prunr_core::Step::FindingSubject);
            std::thread::sleep(Duration::from_millis(5));
            super::answer(&tx, pulse, SubprocessEvent::ImageError { item_id: 4, error: "done".into() });
            std::thread::sleep(Duration::from_millis(5));
            drop((progress, tx));
            let events: Vec<_> = rx.iter().collect();
            assert!(matches!(events.last(), Some(SubprocessEvent::ImageError { .. })), "{events:?}");
        }
    }

    /// The limit counts from the run's last step of its own: a run that
    /// keeps reporting keeps its pulse.
    #[test]
    fn a_new_step_renews_the_pulse() {
        use std::time::Duration;
        let wait = Duration::from_secs(10);
        let (tx, rx) = std::sync::mpsc::channel();
        let limit = Duration::from_millis(500);
        let (progress, _pulse) = super::pulsing_progress(4, tx, Duration::from_millis(10), limit);
        progress.step(prunr_core::Step::FindingSubject);
        let started = std::time::Instant::now();
        while started.elapsed() < limit * 2 {
            progress.step(prunr_core::Step::RefiningEdges);
            assert!(rx.recv_timeout(wait).is_ok());
        }
        let repeat = SubprocessEvent::Report { item_id: 4, update: prunr_core::ProgressUpdate::Step(prunr_core::Step::RefiningEdges) };
        while rx.try_recv().is_ok() {}
        assert_eq!(rx.recv_timeout(wait), Ok(repeat), "still pulsing past the limit from the start");
    }

    /// A pulse ends at its limit even while held: a run silent that long
    /// is taken for hung after all, so a deadlock is still caught.
    #[test]
    fn a_pulse_ends_at_its_limit() {
        use std::time::Duration;
        let (tx, rx) = std::sync::mpsc::channel();
        let (progress, _pulse) = super::pulsing_progress(4, tx, Duration::from_millis(10), Duration::from_millis(100));
        progress.step(prunr_core::Step::FindingSubject);
        drop(progress);
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while rx.recv_timeout(Duration::from_secs(10)).is_ok() {
            assert!(std::time::Instant::now() < deadline, "the pulse went on past its limit");
        }
    }

    #[test]
    fn ipc_progress_sends_reports_for_its_item() {
        let (tx, rx) = std::sync::mpsc::channel();
        let (p, _pulse) = ipc_progress(4, tx);
        p.step(prunr_core::Step::FindingSubject);
        assert_eq!(rx.try_recv().unwrap(), SubprocessEvent::Report {
            item_id: 4, update: prunr_core::ProgressUpdate::Step(prunr_core::Step::FindingSubject),
        });
    }

    // ---------- ipc_temp_dir lifecycle ----------

    /// "test"-prefixed pseudo-PID base; `wrapping_add(real pid)` ensures the
    /// path never collides with a real `ipc_temp_dir()` in this same binary.
    const TEST_PID_BASE: u32 = 0x_7E57_0000;

    fn poison(dir: &std::path::Path, files: &[&str]) {
        std::fs::create_dir_all(dir).unwrap();
        for f in files {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
    }

    fn assert_gone(dir: &std::path::Path, files: &[&str]) {
        for f in files {
            assert!(!dir.join(f).exists(), "expected {f} swept");
        }
    }

    /// A tmpfs's size reads from its mount options in any unit; a mount
    /// with none, or a share of RAM, has no fixed size.
    #[test]
    fn a_tmpfs_size_reads_from_its_mount_options() {
        let mounts = |opts: &str| format!("proc /proc proc rw 0 0\nshm /dev/shm tmpfs rw,nosuid,{opts} 0 0\n");
        assert_eq!(super::tmpfs_size(&mounts("size=65536k"), "/dev/shm"), Some(64 << 20), "Docker's");
        assert_eq!(super::tmpfs_size(&mounts("size=2g,mode=1777"), "/dev/shm"), Some(2 << 30));
        assert_eq!(super::tmpfs_size(&mounts("size=512m"), "/dev/shm"), Some(512 << 20));
        assert_eq!(super::tmpfs_size(&mounts("size=1000"), "/dev/shm"), Some(1000));
        assert_eq!(super::tmpfs_size(&mounts("inode64"), "/dev/shm"), None, "a desktop's: half of RAM");
        assert_eq!(super::tmpfs_size(&mounts("size=50%"), "/dev/shm"), None);
        assert_eq!(super::tmpfs_size(&mounts("size=64m"), "/run"), None, "another mount");
    }

    #[test]
    fn init_temp_dir_for_pid_sweeps_stale_leftovers() {
        let pid = TEST_PID_BASE.wrapping_add(std::process::id());
        let dir = super::resolve_temp_dir_path(pid);
        let stale = ["stale-cli_ds_0.img", "stale-inpaint-out.png"];
        poison(&dir, &stale);

        let returned = super::init_temp_dir_for_pid(pid);

        assert_eq!(returned, dir);
        assert_gone(&dir, &stale);
        assert!(dir.is_dir());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Pins "init unconditionally sweeps; OnceLock above it gates once-only."
    /// If init ever becomes idempotent on its own, this assertion updates
    /// consciously — callers must not rely on that.
    #[test]
    fn second_init_call_sweeps_post_init_writes() {
        let pid = TEST_PID_BASE.wrapping_add(std::process::id().wrapping_add(1));
        let dir = super::resolve_temp_dir_path(pid);
        let _ = std::fs::remove_dir_all(&dir);

        super::init_temp_dir_for_pid(pid);
        std::fs::write(dir.join("cli_ds_0.img"), b"payload").unwrap();
        super::init_temp_dir_for_pid(pid);
        assert_gone(&dir, &["cli_ds_0.img"]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A seg-side crash cleanup must not touch a sibling inpaint
    /// subprocess's `inpaint-*` files. Seeds both kinds, runs the
    /// prefix-scoped sweep, asserts seg-prefixed gone and
    /// inpaint-prefixed survive.
    #[test]
    fn sweep_dir_with_prefix_leaves_other_owners_alone() {
        let scratch = tempfile::tempdir().unwrap();
        let dir = scratch.path();

        let seg = ["chain_42.raw", "edge_5.bin", "spare_edge_5.raw", "input_3.img",
                   "orig_3.img", "result_3.raw", "seg_3.png", "tensor_3.raw"];
        let cli = ["cli_ds_0.img"];
        let inpaint = ["inpaint-img-7-0.png", "inpaint-mask-7-0.png", "inpaint-out-7.png"];
        poison(dir, &seg);
        poison(dir, &cli);
        poison(dir, &inpaint);

        super::sweep_dir_with_prefix(dir, SEG_PIPELINE_PREFIXES);

        assert_gone(dir, &seg);
        for f in &inpaint {
            assert!(dir.join(f).exists(), "inpaint file {f} must survive a seg-scoped cleanup");
        }
        for f in &cli {
            assert!(dir.join(f).exists(),
                "CLI-persistent file {f} must survive a seg-scoped cleanup");
        }
    }

    /// Every temp-file kind has an owner that removes it after a crash:
    /// the seg pipeline's sweep, the CLI's end-of-batch sweep, or the
    /// eraser bridge's per-stroke removal. A new kind fails to compile here
    /// until it is given one.
    #[test]
    fn every_ipc_kind_has_an_owner_that_cleans_it() {
        use IpcKind::*;
        let owner = |kind: IpcKind| match kind {
            Input | Chain | Seg | Tensor | Orig | Result | Edge | SpareEdge => Some(SEG_PIPELINE_PREFIXES),
            CliDs => Some(CLI_PERSISTENT_PREFIXES),
            InpaintImg | InpaintMask | InpaintOut => None,
        };
        let kinds = [Input, Chain, Seg, Tensor, Orig, Result, Edge, SpareEdge, CliDs, InpaintImg, InpaintMask, InpaintOut];
        for kind in kinds {
            if let Some(prefixes) = owner(kind) {
                assert!(prefixes.contains(&kind.prefix()), "{kind:?} is swept by its owner's prefixes");
            }
        }
    }

    /// `cleanup_seg_pipeline_temps` must NOT delete `cli_ds_*` — those
    /// are the CLI parent's pre-staged retry inputs. Wiping them mid-
    /// batch on a worker crash made the CLI re-send paths to files it
    /// had just deleted, which surfaced as "Insufficient memory"
    /// errors whose actual cause was self-inflicted.
    #[test]
    fn seg_pipeline_cleanup_preserves_cli_persistent_temps() {
        let dir = std::env::temp_dir().join(format!(
            "prunr-test-cli-ds-survive-{}", std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Seed one cli_ds file (CLI-owned) and one seg file (worker-owned).
        std::fs::write(dir.join("cli_ds_42.img"), b"retry-input").unwrap();
        std::fs::write(dir.join("seg_42.raw"), b"worker-output").unwrap();

        super::sweep_dir_with_prefix(&dir, SEG_PIPELINE_PREFIXES);

        assert!(dir.join("cli_ds_42.img").exists(),
            "cli_ds_*.img must survive crash-recovery cleanup");
        assert!(!dir.join("seg_42.raw").exists(),
            "seg_*.raw must be wiped by crash-recovery cleanup");

        // CLI-driven end-of-batch cleanup wipes the cli_ds_ file.
        super::sweep_dir_with_prefix(&dir, CLI_PERSISTENT_PREFIXES);
        assert!(!dir.join("cli_ds_42.img").exists(),
            "cli_ds_*.img must be wiped by CLI end-of-batch cleanup");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sweep_dead_pid_dirs_removes_dead_pids_keeps_self_and_unrelated() {
        // Stage three siblings under a fresh parent dir: own pid (keep),
        // a guaranteed-dead pid (remove), and a non-prunr name (keep).
        let parent = tempfile::tempdir().unwrap();
        let self_pid = std::process::id();
        let self_dir = parent.path().join(format!("prunr-ipc-{self_pid}"));
        let dead_pid = crate::fs_util::a_free_pid();
        let dead_dir = parent.path().join(format!("prunr-ipc-{dead_pid}"));
        let foreign = parent.path().join("not-a-prunr-dir");
        for d in [&self_dir, &dead_dir, &foreign] {
            std::fs::create_dir_all(d).unwrap();
        }
        crate::fs_util::sweep_dead_pid_dirs(self_dir.parent().unwrap(), "prunr-ipc-", self_pid);
        assert!(self_dir.exists(), "own pid dir must survive");
        assert!(foreign.exists(), "non-prunr names must be ignored");
        assert!(!dead_dir.exists(), "dead-pid dir should be removed");
    }

    /// A session open for days keeps its folder: liveness is the process,
    /// not the folder's age (another instance once took a live session's
    /// files after 24 h untouched).
    #[test]
    fn a_live_sessions_folder_is_kept_however_old() {
        let parent = tempfile::tempdir().unwrap();
        #[cfg(unix)]
        let waits = std::process::Command::new("sleep").arg("30").spawn();
        #[cfg(windows)]
        let waits = std::process::Command::new("ping").args(["-n", "30", "127.0.0.1"]).stdout(std::process::Stdio::null()).spawn();
        let mut other = waits.unwrap();
        let live_dir = parent.path().join(format!("prunr-ipc-{}", other.id()));
        std::fs::create_dir_all(&live_dir).unwrap();
        let mut open = std::fs::File::options();
        open.read(true);
        // A folder opens on Windows only with FILE_FLAG_BACKUP_SEMANTICS.
        #[cfg(windows)]
        std::os::windows::fs::OpenOptionsExt::custom_flags(open.write(true), 0x0200_0000);
        let days_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(3 * 86_400);
        open.open(&live_dir).unwrap().set_modified(days_ago).unwrap();
        crate::fs_util::sweep_dead_pid_dirs(parent.path(), "prunr-ipc-", std::process::id());
        let kept = live_dir.exists();
        let _ = other.kill();
        let _ = other.wait();
        assert!(kept, "a live session's folder survives the sweep");
    }


    /// The app's runs make the drawn scales (and what their detector run
    /// makes anyway) before the result, the rest after it only for the
    /// image on screen; the CLI runs only the ones its lines draw.
    #[test]
    fn a_run_makes_the_drawn_scales_first_and_the_rest_only_on_screen() {
        use prunr_core::EdgeScale::{self, Bold, Fine, Fused};
        assert_eq!(InferredScales::default(), InferredScales::DrawnFirst);
        let (shown, background) = (Some(7), Some(8));
        for (edge_scale, first, rest) in [
            (Fused, &[Bold, Fused][..], &[Fine][..]),
            (Bold, &[Bold, Fused][..], &[Fine][..]),
            (Fine, &[Fine][..], &[Bold, Fused][..]),
        ] {
            let edge = EdgeSettings { edge_scale, ..EdgeSettings::default() };
            assert_eq!(InferredScales::DrawnFirst.of(&edge), first, "{edge_scale:?}");
            assert_eq!(InferredScales::DrawnFirst.spare(&edge, 7, shown), rest, "{edge_scale:?} on screen");
            assert!(InferredScales::DrawnFirst.spare(&edge, 7, background).is_empty(), "{edge_scale:?} in the background");
            assert!(InferredScales::DrawnFirst.spare(&edge, 7, None).is_empty(), "{edge_scale:?} with nothing on screen");
            assert_eq!(InferredScales::Drawn.of(&edge), &[edge_scale]);
            assert!(InferredScales::Drawn.spare(&edge, 7, shown).is_empty(), "the CLI makes no spare scales");
        }
        let dual = EdgeSettings { line_style: prunr_core::LineStyle::DualScale { fine_color: [0; 3], bold_color: [0; 3] }, ..EdgeSettings::default() };
        assert_eq!(InferredScales::DrawnFirst.of(&dual), &EdgeScale::ALL);
        assert!(InferredScales::DrawnFirst.spare(&dual, 7, shown).is_empty(), "Dual scale draws every run");
    }

}
