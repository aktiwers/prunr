//! IPC protocol types for parent ↔ child subprocess communication.
//!
//! All types are serde-serializable and sent as length-prefixed bincode
//! frames over stdin/stdout.

use prunr_core::{ModelKind, MaskSettings, EdgeSettings, LineMode};
use prunr_core::inpaint_sd::SdInpaintRequest;
use prunr_models::ModelId;
use serde::{Serialize, Deserialize};

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

/// `InpaintBridgeResult::Error.error` value the inpaint bridge emits
/// when its watchdog kills the subprocess to prevent a system-wide
/// memory-pressure stall. Routed to a distinct user-facing toast
/// ("Erase aborted — system memory low") so the user knows the abort
/// was self-defence, not a stroke-quality failure.
pub const MEMORY_PRESSURE_ABORT_MSG: &str = "memory-pressure-abort";

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
    /// Initialize the worker: load model, create engine pool.
    /// When `inpaint_only` is true the worker SKIPS engine pool creation
    /// and only handles `Inpaint` commands — the seg/edge fields
    /// (`model`, `mask`, `line_mode`, `edge`) are ignored. Used by the
    /// dedicated SD-inpaint subprocess so it doesn't waste GB on a seg
    /// engine that never runs.
    Init {
        model: ModelKind,
        jobs: usize,
        mask: MaskSettings,
        cpu_only: bool,
        line_mode: LineMode,
        edge: EdgeSettings,
        /// IPC temp directory (set by parent so child uses the same PID-namespaced dir)
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
        /// Path to temp file containing the source image bytes.
        image_path: std::path::PathBuf,
        /// For a run that starts from an earlier result: path to temp file with its RGBA + dimensions.
        chain_input: Option<ChainInput>,
    },
    /// Tier 2: Re-run postprocess from a cached tensor (skip inference).
    /// The parent sends the raw tensor + original image via temp files.
    RePostProcess {
        item_id: u64,
        /// Path to temp file with raw f32 tensor data.
        tensor_path: std::path::PathBuf,
        tensor_height: u32,
        tensor_width: u32,
        model: ModelKind,
        /// Path to temp file with original image bytes.
        original_image_path: std::path::PathBuf,
        /// Updated mask settings for re-postprocessing.
        mask: MaskSettings,
    },
    /// AddEdgeInference tier: seg tensor is cached, run only TEED on the
    /// masked image and compose. Used for Off → SubjectOutline transitions so
    /// enabling the outline doesn't re-run seg inference.
    AddEdgeInference {
        item_id: u64,
        /// Path to temp file with original image bytes.
        image_path: std::path::PathBuf,
        /// Path to temp file with the cached raw f32 seg tensor. The worker
        /// does NOT delete this file — it hands the path back as
        /// `tensor_cache_path` on `ImageDone`, and the parent's reader takes
        /// ownership (read-and-delete). This preserves the cache without an
        /// extra copy round-trip.
        seg_tensor_path: std::path::PathBuf,
        seg_tensor_height: u32,
        seg_tensor_width: u32,
        /// Model that produced the seg tensor (must match Init's model).
        model: ModelKind,
        /// Per-item mask settings (may differ from Init's mask).
        mask: MaskSettings,
    },
    /// Inpaint a region of an image. Worker dispatches to the inpaint
    /// model named by `model_id` (LaMa / Big-LaMa / MI-GAN / SD 1.5 /
    /// SD 1.5 LCM), runs the full post-process pipeline (color match +
    /// seam-guided blend + sharpen) and returns the finished RGBA.
    /// For SD models, `sd_req` carries prompt + steps + guidance;
    /// `None` for non-SD models. Independent of the seg/edge engine
    /// pool — each inpaint model runs its own session inside the worker.
    Inpaint {
        item_id: u64,
        /// Which inpaint model to run.
        model_id: ModelId,
        /// The photo as `ipc::write_image` writes it.
        image_path: std::path::PathBuf,
        /// Single-channel mask, the same way: 255 = inpaint here, 0 = keep.
        mask_path: std::path::PathBuf,
        /// SD-specific knobs. `None` for LaMa / Big-LaMa / MI-GAN.
        sd_req: Option<SdInpaintRequest>,
        /// Seam-blend feather override (px). 0 ⇒ pipeline default.
        #[serde(default)]
        feather_px: f32,
        /// Unsharp-mask strength applied inside the painted region.
        /// 0 ⇒ skip sharpen.
        #[serde(default)]
        sharpen: f32,
    },
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
    /// Shut down gracefully.
    Shutdown,
}

/// The image a run starts from when it is an earlier result: raw RGBA in a temp file.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct ChainInput {
    pub path: std::path::PathBuf,
    pub width: u32,
    pub height: u32,
}

/// Sends a pipeline's progress reports as `SubprocessEvent::Report` for
/// `item_id`, on the same channel as its result, so a report never
/// arrives after the result.
struct IpcProgress {
    item_id: u64,
    tx: std::sync::mpsc::Sender<SubprocessEvent>,
}

impl prunr_core::ProgressSink for IpcProgress {
    fn report(&self, update: prunr_core::ProgressUpdate) {
        let _ = self.tx.send(SubprocessEvent::Report { item_id: self.item_id, update });
    }
}

/// A progress handle whose reports go to the parent as events.
pub fn ipc_progress(item_id: u64, tx: std::sync::mpsc::Sender<SubprocessEvent>) -> prunr_core::Progress {
    prunr_core::Progress::new(std::sync::Arc::new(IpcProgress { item_id, tx }))
}

/// The line detector's planes (f32 probabilities, pre-threshold) a
/// worker wrote for an image: `scales`' planes one after another in that
/// order, each at its own size, `edge_scale_dims` of the base `height` ×
/// `width`. A run may write only some scales.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EdgeCacheFile {
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
        result_path: std::path::PathBuf,
        width: u32,
        height: u32,
        active_provider: String,
        /// Path to cached raw segmentation tensor (Tier 1 output).
        /// Only present for full pipeline runs, not RePostProcess.
        tensor_cache_path: Option<std::path::PathBuf>,
        tensor_cache_height: Option<u32>,
        tensor_cache_width: Option<u32>,
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
        rgba_path: std::path::PathBuf,
        width: u32,
        height: u32,
        /// How large the region was filled: 1 is full size.
        fill_scale: f32,
    },
    /// Inpaint dispatch failed (model missing, ORT error, IO error).
    InpaintError {
        item_id: u64,
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
/// The OnceLock initializer creates the dir, sweeps any stale files left by
/// a previous prunr process with the same PID, AND sweeps dead-PID sibling
/// dirs from past parent crashes — so writers (CLI downscale, inpaint
/// bridge, manager) never race a later cleanup, and `/dev/shm` doesn't
/// fill over weeks of OOM-killed parents.
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

/// Body of `ipc_temp_dir()`'s OnceLock init, exposed for unit tests
/// (the global is process-wide and order-dependent across tests). Tests
/// pass synthetic PIDs and rely on this NOT sweeping sibling dirs — the
/// dead-PID sweep lives in `ipc_temp_dir`'s OnceLock body so tests can
/// drive `init_temp_dir_for_pid` in isolation.
fn init_temp_dir_for_pid(pid: u32) -> std::path::PathBuf {
    let dir = resolve_temp_dir_path(pid);
    let _ = std::fs::create_dir_all(&dir);
    crate::fs_util::sweep_dir_files(&dir);
    dir
}

fn resolve_temp_dir_path(pid: u32) -> std::path::PathBuf {
    #[cfg(target_os = "linux")]
    if std::path::Path::new("/dev/shm").is_dir() {
        return std::path::PathBuf::from(format!("/dev/shm/prunr-ipc-{pid}"));
    }
    std::env::temp_dir().join(format!("prunr-ipc-{pid}"))
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
    /// RePostProcess (Tier 2) path. Distinct from `Seg` despite a
    /// similar payload — different filename keeps the two cache
    /// flavours separable in temp dir scans + cleanup.
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
    /// `<ipc>/<prefix><id>-<generation>.<ext>`. Only valid for `InpaintImg` /
    /// `InpaintMask` (the inpaint worker rewrites these per stroke and
    /// the generation suffix prevents stale-file races).
    pub fn path_for_generation(self, ipc_dir: &std::path::Path, id: u64, generation: u64) -> std::path::PathBuf {
        // A full assert, not a debug one: a wrong variant would produce
        // e.g. `result_5-0.raw` in release, a name no reader expects.
        assert!(
            matches!(self, IpcKind::InpaintImg | IpcKind::InpaintMask),
            "path_for_generation only valid for InpaintImg / InpaintMask; got {self:?}"
        );
        ipc_dir.join(format!("{}{id}-{generation}.{}", self.prefix(), self.ext()))
    }
}

/// File-name prefixes the seg worker writes or consumes-and-deletes
/// during a dispatch. Crash-recovery cleanup wipes these because the
/// worker may have died mid-write or with partial input.
///
/// `CliDs` is intentionally excluded: those `cli_ds_*` files are
/// pre-staged by the CLI parent before the run_batch loop starts and
/// re-sent on every retry from `valid_paths[idx]`. Including the
/// prefix here once silently wiped the retry inputs the CLI was about
/// to feed the re-spawned worker, surfacing as "Insufficient memory"
/// errors whose real cause was self-inflicted.
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

/// File-name prefixes owned by the inpaint subprocess.
pub const INPAINT_PREFIXES: &[&str] = &[
    IpcKind::InpaintImg.prefix(),
    IpcKind::InpaintMask.prefix(),
    IpcKind::InpaintOut.prefix(),
];

/// File-name prefixes the CLI parent owns *across* worker crashes.
/// `cli_ds_*` are the downscaled temps the CLI writes once before the
/// run_batch loop and re-sends on every retry from `valid_paths[idx]`.
/// Crash-recovery cleanup (`cleanup_seg_pipeline_temps`) deliberately
/// skips these; the CLI sweeps them itself once the batch loop has
/// fully exited.
pub const CLI_PERSISTENT_PREFIXES: &[&str] = &[
    IpcKind::CliDs.prefix(),
];

/// Cleanup scoped to a single subprocess's owned filenames. Use from
/// crash-recovery paths so the wipe doesn't race a sibling subprocess
/// writing the shared dir.
pub fn cleanup_ipc_temp_for_prefix(prefixes: &[&str]) {
    sweep_dir_with_prefix(ipc_temp_dir(), prefixes);
}

/// Crash-recovery cleanup for the seg / CLI subprocess. Leaves a sibling
/// inpaint subprocess's `inpaint-*` files alone, and intentionally
/// preserves `cli_ds_*` (the CLI's pre-staged retry inputs — see
/// `CLI_PERSISTENT_PREFIXES`).
pub fn cleanup_seg_pipeline_temps() {
    cleanup_ipc_temp_for_prefix(SEG_PIPELINE_PREFIXES);
}

/// End-of-batch cleanup for CLI-owned downscaled temps. Call once the
/// run_batch loop has fully exited so the worker has stopped reading
/// from `cli_ds_*` paths. Idempotent.
pub fn cleanup_cli_persistent_temps() {
    cleanup_ipc_temp_for_prefix(CLI_PERSISTENT_PREFIXES);
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
            SubprocessCommand::Inpaint { .. } => 4,
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

    fn command_samples() -> Vec<SubprocessCommand> {
        let path = || PathBuf::from("/tmp/prunr-sample");
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
            SubprocessCommand::ProcessImage { item_id: 1, image_path: path(), chain_input: None },
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
            SubprocessCommand::Inpaint {
                item_id: 1,
                model_id: prunr_models::ModelId::LaMaFp32,
                image_path: path(),
                mask_path: path(),
                sd_req: None,
                feather_px: 0.0,
                sharpen: 0.0,
            },
            SubprocessCommand::CancelItem { item_id: 1 },
            SubprocessCommand::Configure { mask: MaskSettings::default(), line_mode: LineMode::Off, edge: EdgeSettings::default(), first_look: None, on_screen: None },
            SubprocessCommand::Shutdown,
        ]
    }

    fn event_samples() -> Vec<SubprocessEvent> {
        let path = || PathBuf::from("/tmp/prunr-sample");
        vec![
            SubprocessEvent::Ready { active_provider: "CPU".into() },
            SubprocessEvent::Report { item_id: 1, update: prunr_core::ProgressUpdate::Step(prunr_core::Step::Denoising) },
            SubprocessEvent::ImageDone {
                item_id: 1,
                result_path: path(),
                width: 2,
                height: 2,
                active_provider: "CPU".into(),
                tensor_cache_path: None,
                tensor_cache_height: None,
                tensor_cache_width: None,
                edge_cache: None,
                spare_scales: None,
            },
            SubprocessEvent::SpareEdgeScales { item_id: 1, token: 1, outcome: SpareOutcome::Skipped },
            SubprocessEvent::ImageError { item_id: 1, error: "failed".into() },
            SubprocessEvent::InpaintDone { item_id: 1, rgba_path: path(), width: 2, height: 2, fill_scale: 1.0 },
            SubprocessEvent::InpaintError { item_id: 1, error: "failed".into() },
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
    fn command_cancel_item_roundtrip() {
        roundtrip(&SubprocessCommand::CancelItem { item_id: 42 });
        roundtrip(&SubprocessCommand::CancelItem { item_id: 0 });
        roundtrip(&SubprocessCommand::CancelItem { item_id: u64::MAX });
    }

    #[test]
    fn event_image_done_with_and_without_edge_cache_roundtrip() {
        roundtrip(&SubprocessEvent::ImageDone {
            item_id: 11,
            result_path: PathBuf::from("/tmp/result.bin"),
            width: 1000,
            height: 1000,
            active_provider: "CPU".to_string(),
            tensor_cache_path: Some(PathBuf::from("/tmp/tensor.bin")),
            tensor_cache_height: Some(320),
            tensor_cache_width: Some(320),
            edge_cache: None,
            spare_scales: None,
        });
        roundtrip(&SubprocessEvent::ImageDone {
            item_id: 12,
            result_path: PathBuf::from("/tmp/result2.bin"),
            width: 500,
            height: 500,
            active_provider: "CoreML".to_string(),
            tensor_cache_path: None,
            tensor_cache_height: None,
            tensor_cache_width: None,
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
    fn command_inpaint_lama_roundtrip() {
        roundtrip(&SubprocessCommand::Inpaint {
            item_id: 11,
            model_id: prunr_models::ModelId::LaMaFp32,
            image_path: std::path::PathBuf::from("/tmp/prunr-inpaint-img-11"),
            mask_path: std::path::PathBuf::from("/tmp/prunr-inpaint-mask-11"),
            sd_req: None,
            feather_px: 0.0,
            sharpen: 0.0,
        });
    }

    #[test]
    fn command_inpaint_sd_roundtrip() {
        roundtrip(&SubprocessCommand::Inpaint {
            item_id: 12,
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
            }),
            feather_px: 4.5,
            sharpen: 0.6,
        });
    }

    #[test]
    fn event_inpaint_done_and_error_roundtrip() {
        roundtrip(&SubprocessEvent::InpaintDone {
            item_id: 11,
            rgba_path: std::path::PathBuf::from("/tmp/prunr-inpaint-out-11"),
            width: 1920,
            height: 1080,
            fill_scale: 0.5,
        });
        roundtrip(&SubprocessEvent::InpaintError {
            item_id: 11,
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

    #[test]
    fn ipc_progress_sends_reports_for_its_item() {
        let (tx, rx) = std::sync::mpsc::channel();
        let p = ipc_progress(4, tx);
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

    /// Symmetric direction: an inpaint-scoped cleanup leaves seg files alone.
    /// No production caller targets only inpaint today, but the prefix table
    /// already declares them — pinning the contract avoids accidental
    /// asymmetry (e.g. a future inpaint-only cancel path that wipes the
    /// seg pipeline).
    #[test]
    fn sweep_dir_with_prefix_inpaint_scope_is_symmetric() {
        let scratch = tempfile::tempdir().unwrap();
        let dir = scratch.path();
        let seg = ["chain_1.raw", "result_1.raw"];
        let inpaint = ["inpaint-img-1-0.png", "inpaint-out-1.png"];
        poison(dir, &seg);
        poison(dir, &inpaint);

        super::sweep_dir_with_prefix(dir, INPAINT_PREFIXES);

        for f in &seg {
            assert!(dir.join(f).exists(), "seg file {f} must survive an inpaint-scoped cleanup");
        }
        assert_gone(dir, &inpaint);
    }

    /// Files matching no supplied prefix are left in place — the sweep is
    /// "remove only what I own", not "remove everything except what I name."
    #[test]
    fn sweep_dir_with_prefix_ignores_unmatched_files() {
        let scratch = tempfile::tempdir().unwrap();
        let dir = scratch.path();
        poison(dir, &["unrelated.txt", "seg_1.png"]);

        super::sweep_dir_with_prefix(dir, SEG_PIPELINE_PREFIXES);

        assert_gone(dir, &["seg_1.png"]);
        assert!(dir.join("unrelated.txt").exists());
    }

    /// Inventory of IPC filename kinds; keep in sync with new IPC writers
    /// (and with `SEG_PIPELINE_PREFIXES` / `INPAINT_PREFIXES`). Pinning the
    /// inventory in one place documents the naming contract; it does not
    /// catch drift on its own — a new writer that lands without updating
    /// this list AND the prefix tables would still slip through.
    #[test]
    fn every_ipc_filename_in_tree_matches_a_known_prefix() {
        let production_filenames = [
            "chain_42.raw", "cli_ds_0.img", "edge_5.bin", "spare_edge_5.raw", "input_3.img",
            "orig_3.img", "result_3.raw", "seg_3.png", "tensor_3.raw",
            "inpaint-img-7-0.png", "inpaint-mask-7-0.png", "inpaint-out-7.png",
        ];
        for name in &production_filenames {
            assert!(
                SEG_PIPELINE_PREFIXES.iter()
                    .chain(INPAINT_PREFIXES.iter())
                    .chain(CLI_PERSISTENT_PREFIXES.iter())
                    .any(|p| name.starts_with(p)),
                "filename {name} matches no prefix in SEG_PIPELINE_PREFIXES + INPAINT_PREFIXES + CLI_PERSISTENT_PREFIXES"
            );
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
        // On non-Linux the dead-pid removal is gated out — only the
        // self/foreign preservation arms run there.
        let parent = tempfile::tempdir().unwrap();
        let self_pid = std::process::id();
        let self_dir = parent.path().join(format!("prunr-ipc-{self_pid}"));
        let dead_pid = pick_definitely_dead_pid();
        let dead_dir = parent.path().join(format!("prunr-ipc-{dead_pid}"));
        let foreign = parent.path().join("not-a-prunr-dir");
        for d in [&self_dir, &dead_dir, &foreign] {
            std::fs::create_dir_all(d).unwrap();
        }
        crate::fs_util::sweep_dead_pid_dirs(self_dir.parent().unwrap(), "prunr-ipc-", self_pid);
        assert!(self_dir.exists(), "own pid dir must survive");
        assert!(foreign.exists(), "non-prunr names must be ignored");
        // On non-Linux the 24h fallback won't kill a fresh dir; only assert
        // the dead-pid removal where it's deterministic.
        #[cfg(target_os = "linux")]
        assert!(!dead_dir.exists(), "dead-pid dir should be removed");
    }

    #[cfg(target_os = "linux")]
    fn pick_definitely_dead_pid() -> u32 {
        // Iterate down from a high pid. The first `/proc/{pid}` miss
        // wins. Theoretical race: kernel could re-allocate the picked
        // pid before `sweep_dead_pid_dirs` runs, leaving the dir
        // intact. In practice the window is microseconds and Linux
        // doesn't reuse pids that fast under normal load. Worst case
        // is a flaky test, not a correctness bug.
        for pid in (1_000_000..2_000_000).rev() {
            if !std::path::Path::new(&format!("/proc/{pid}")).exists() {
                return pid;
            }
        }
        unreachable!("could not find a dead pid in [1M, 2M)")
    }

    #[cfg(not(target_os = "linux"))]
    fn pick_definitely_dead_pid() -> u32 {
        // Liveness check on non-Linux falls back to mtime > 24h, which a
        // freshly-created dir won't satisfy. Test still exercises the
        // happy path of self/foreign preservation; pid value is unused.
        u32::MAX
    }

    // ── serde(default) field roundtrip tests ────────────────────────────────
    //
    // `#[serde(default)]` fields — `inpaint_only`, `feather_px`, `sharpen`,
    // `edge_cache` — are pinned here to document their expected defaults
    // and catch accidental value drift. Bincode is a positional format; it
    // does NOT support "missing trailing field → default" across a wire
    // boundary (it returns `UnexpectedEnd` on a short payload). These tests
    // therefore verify that the defaults serialise and deserialise correctly
    // in a full roundtrip, not that a legacy binary can omit the field.
    //
    // The real forward-compat contract is: parent and child are always
    // launched from the same binary, so no cross-version decode is
    // expected in practice.

    /// `Init.inpaint_only` defaults to `false`. A worker that sees `false`
    /// creates the full engine pool; `true` skips it. Roundtrip both values
    /// so a stray `serde(default = "true_fn")` regression fails loudly.
    #[test]
    fn command_init_inpaint_only_field_roundtrips_both_values() {
        roundtrip(&SubprocessCommand::Init {
            model: ModelKind::Silueta,
            jobs: 1,
            mask: MaskSettings::default(),
            cpu_only: false,
            line_mode: LineMode::Off,
            edge: EdgeSettings::default(),
            ipc_dir: PathBuf::from("/tmp/test"),
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
            ipc_dir: PathBuf::from("/tmp/test"),
            inpaint_only: true,
            line_fade_pct: 0,
            edge_scales: InferredScales::Drawn,
        });
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

    /// `Inpaint.feather_px` and `Inpaint.sharpen` default to 0.0 (no
    /// post-process tuning). Roundtrip zero and non-zero values to catch
    /// default drift.
    #[test]
    fn command_inpaint_feather_and_sharpen_roundtrip_zero_and_nonzero() {
        roundtrip(&SubprocessCommand::Inpaint {
            item_id: 5,
            model_id: prunr_models::ModelId::LaMaFp32,
            image_path: PathBuf::from("/tmp/img.png"),
            mask_path: PathBuf::from("/tmp/mask.png"),
            sd_req: None,
            feather_px: 0.0,
            sharpen: 0.0,
        });
        roundtrip(&SubprocessCommand::Inpaint {
            item_id: 6,
            model_id: prunr_models::ModelId::LaMaFp32,
            image_path: PathBuf::from("/tmp/img2.png"),
            mask_path: PathBuf::from("/tmp/mask2.png"),
            sd_req: None,
            feather_px: 8.5,
            sharpen: 0.4,
        });
    }
}
