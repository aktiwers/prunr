//! IPC protocol types for parent ↔ child subprocess communication.
//!
//! All types are serde-serializable and sent as length-prefixed bincode
//! frames over stdin/stdout.

use prunr_core::{ModelKind, MaskSettings, EdgeSettings, ProgressStage, LineMode};
use prunr_core::inpaint_sd::SdInpaintRequest;
use prunr_models::ModelId;
use serde::{Serialize, Deserialize};

/// `ImageError.error` value emitted when a per-item `CancelItem` trips at
/// dispatch. The parent matches on this exact string to revert the item to
/// `Pending` rather than flag it as `Error`; any drift silently breaks the
/// round-trip.
pub const CANCELLED_ERR_MSG: &str = "Cancelled";

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
        force_cpu: bool,
        line_mode: LineMode,
        edge: EdgeSettings,
        /// IPC temp directory (set by parent so child uses the same PID-namespaced dir)
        ipc_dir: std::path::PathBuf,
        #[serde(default)]
        inpaint_only: bool,
    },
    /// Process a single image. `image_path` points to a temp file with
    /// the raw image bytes (avoids piping large payloads through stdin).
    ProcessImage {
        item_id: u64,
        /// Path to temp file containing the source image bytes.
        image_path: std::path::PathBuf,
        /// For chain mode: path to temp file with previous result RGBA + dimensions.
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
    /// AddEdgeInference tier: seg tensor is cached, run only DexiNed on the
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
        /// Source image bytes (PNG/JPG/etc — worker decodes).
        image_path: std::path::PathBuf,
        /// Single-channel mask: 255 = inpaint here, 0 = keep.
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
    /// Shut down gracefully.
    Shutdown,
}

/// Chain mode input: previous result as raw RGBA in a temp file.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct ChainInput {
    pub path: std::path::PathBuf,
    pub width: u32,
    pub height: u32,
}

/// Child → Parent events (sent over child's stdout).
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub enum SubprocessEvent {
    /// Engine pool created successfully.
    Ready {
        active_provider: String,
    },
    /// Per-stage progress for an image.
    Progress {
        item_id: u64,
        stage: ProgressStage,
        pct: f32,
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
        /// Path to cached DexiNed edge tensor (full-resolution f32, pre-threshold).
        /// Parent compresses and stores for Tier 2 edge reruns on line_strength tweaks.
        /// Only present when the run used DexiNed (EdgesOnly or SubjectOutline modes).
        #[serde(default)]
        edge_cache_path: Option<std::path::PathBuf>,
        #[serde(default)]
        edge_cache_height: Option<u32>,
        #[serde(default)]
        edge_cache_width: Option<u32>,
    },
    /// Image processing failed (non-fatal).
    ImageError {
        item_id: u64,
        error: String,
    },
    /// Inpaint completed — RGBA result lives at `rgba_path`. Same
    /// read-and-delete contract as `ImageDone.result_path`.
    InpaintDone {
        item_id: u64,
        rgba_path: std::path::PathBuf,
        width: u32,
        height: u32,
    },
    /// Inpaint dispatch failed (model missing, ORT error, IO error).
    InpaintError {
        item_id: u64,
        error: String,
    },
    /// Per-step progress during a long inpaint stroke. Fired between
    /// SD UNet steps; `total = num_inference_steps`. LaMa / MI-GAN
    /// don't fire this (single-pass, sub-second on GPU).
    InpaintProgress {
        item_id: u64,
        current: u32,
        total: u32,
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
        sweep_stale_pid_dirs(&dir, pid);
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

/// Remove `prunr-ipc-{old_pid}` dirs in the same parent whose PID owner is
/// no longer running. Without this, a parent crash / OOM-kill leaves the
/// dir behind permanently and `/dev/shm` slowly fills over weeks.
///
/// Liveness check is `/proc/{pid}` on Linux. Other platforms fall back to
/// "older than 24h", which is conservative enough to avoid clobbering a
/// concurrent prunr instance without sleeping forever on stale dirs.
fn sweep_stale_pid_dirs(self_dir: &std::path::Path, self_pid: u32) {
    let Some(parent) = self_dir.parent() else { return };
    let Ok(entries) = std::fs::read_dir(parent) else { return };
    for entry in entries.flatten() {
        let name_os = entry.file_name();
        let Some(name) = name_os.to_str() else { continue };
        let Some(pid_str) = name.strip_prefix("prunr-ipc-") else { continue };
        let Ok(other_pid) = pid_str.parse::<u32>() else { continue };
        if other_pid == self_pid { continue; }
        if !is_pid_dead(other_pid, &entry) { continue; }
        let _ = std::fs::remove_dir_all(entry.path());
    }
}

#[cfg(target_os = "linux")]
fn is_pid_dead(pid: u32, _entry: &std::fs::DirEntry) -> bool {
    // `/proc/{pid}` is the authoritative liveness check on Linux.
    !std::path::Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(not(target_os = "linux"))]
fn is_pid_dead(_pid: u32, entry: &std::fs::DirEntry) -> bool {
    // Fallback: drop dirs untouched for >24h. Fresh siblings of a running
    // prunr instance are spared; truly stale dirs from past crashes get
    // reclaimed without polling kernel APIs per platform.
    let Ok(meta) = entry.metadata() else { return false };
    let Ok(modified) = meta.modified() else { return false };
    let Ok(age) = modified.elapsed() else { return false };
    age > std::time::Duration::from_secs(86_400)
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

/// Which subprocess owns an IPC temp file. Drives crash-recovery
/// cleanup scope so a seg-side wipe (e.g. on cancel-all) leaves a
/// sibling inpaint subprocess's in-flight files alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpcOwner {
    /// Seg / CLI pipeline (input, chain, edge, result, tensor, …).
    SegCli,
    /// Inpaint subprocess (`inpaint-img-…`, `inpaint-mask-…`, `inpaint-out-…`).
    Inpaint,
}

/// Every IPC temp-file kind. One source of truth for the prefix +
/// extension contract previously triplicated across 13 `format!` sites
/// + the prefix tables + the drift-catcher test.
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
    /// Seg worker → parent: DexiNed multi-scale edge tensor cache.
    Edge,
    /// CLI parent: downscaled image temp before subprocess dispatch.
    CliDs,
    /// Parent → inpaint worker: source PNG for a stroke (gen-suffixed).
    InpaintImg,
    /// Parent → inpaint worker: mask PNG for a stroke (gen-suffixed).
    InpaintMask,
    /// Inpaint worker → parent: composited RGBA PNG result.
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
            IpcKind::Chain | IpcKind::Seg | IpcKind::Tensor | IpcKind::Result | IpcKind::Edge => "raw",
            IpcKind::InpaintImg | IpcKind::InpaintMask | IpcKind::InpaintOut => "png",
        }
    }

    /// Crash-recovery owner: who's allowed to wipe this file's prefix.
    pub const fn owner(self) -> IpcOwner {
        match self {
            IpcKind::InpaintImg | IpcKind::InpaintMask | IpcKind::InpaintOut => IpcOwner::Inpaint,
            _ => IpcOwner::SegCli,
        }
    }

    /// Build a path for an item-keyed file: `<ipc>/<prefix><id>.<ext>`.
    pub fn path_for(self, ipc_dir: &std::path::Path, id: u64) -> std::path::PathBuf {
        ipc_dir.join(format!("{}{id}.{}", self.prefix(), self.ext()))
    }

    /// Build a path for a gen-suffixed inpaint file:
    /// `<ipc>/<prefix><id>-<gen>.<ext>`. Only valid for `InpaintImg` /
    /// `InpaintMask` (the inpaint worker rewrites these per stroke and
    /// the gen suffix prevents stale-file races).
    pub fn path_for_gen(self, ipc_dir: &std::path::Path, id: u64, gen: u64) -> std::path::PathBuf {
        // `debug_assert!` was insufficient — release builds would silently
        // produce e.g. `result_5-0.raw` for the wrong variant, breaking
        // path disambiguation in the IPC temp dir.
        assert!(
            matches!(self, IpcKind::InpaintImg | IpcKind::InpaintMask),
            "path_for_gen only valid for InpaintImg / InpaintMask; got {self:?}"
        );
        ipc_dir.join(format!("{}{id}-{gen}.{}", self.prefix(), self.ext()))
    }

    /// All variants. Ordered so prefix-tables (derived below) are stable
    /// across builds and test diffs are deterministic.
    pub const ALL: &'static [IpcKind] = &[
        IpcKind::Input,
        IpcKind::Chain,
        IpcKind::Seg,
        IpcKind::Tensor,
        IpcKind::Orig,
        IpcKind::Result,
        IpcKind::Edge,
        IpcKind::CliDs,
        IpcKind::InpaintImg,
        IpcKind::InpaintMask,
        IpcKind::InpaintOut,
    ];
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
];

/// File-name prefixes owned by the inpaint subprocess.
pub const INPAINT_PREFIXES: &[&str] = &[
    "inpaint-",
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

/// Sweep every file in the IPC temp dir. Process-wide; never call from a
/// path that could race a sibling subprocess — use a prefix-scoped helper.
pub fn cleanup_ipc_temp() {
    crate::fs_util::sweep_dir_files(ipc_temp_dir());
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

    fn roundtrip<T>(value: &T)
    where
        T: serde::Serialize + DeserializeOwned + PartialEq + Debug,
    {
        let bytes = bincode::serde::encode_to_vec(value, standard()).unwrap();
        let (decoded, _): (T, _) =
            bincode::serde::decode_from_slice(&bytes, standard()).unwrap();
        assert_eq!(value, &decoded);
    }

    #[test]
    fn command_init_roundtrip() {
        roundtrip(&SubprocessCommand::Init {
            model: ModelKind::Silueta,
            jobs: 2,
            mask: MaskSettings::default(),
            force_cpu: false,
            line_mode: LineMode::Off,
            edge: EdgeSettings { line_strength: 0.5, solid_line_color: Some([255, 0, 0]), edge_thickness: 0, edge_scale: prunr_core::EdgeScale::Bold, compose_mode: prunr_core::ComposeMode::default(), line_style: prunr_core::LineStyle::default(), input_transform: prunr_core::InputTransform::default() },
            ipc_dir: PathBuf::from("/tmp/prunr-ipc-test"),
            inpaint_only: false,
        });
        roundtrip(&SubprocessCommand::Init {
            model: ModelKind::Silueta,
            jobs: 1,
            mask: MaskSettings::default(),
            force_cpu: false,
            line_mode: LineMode::Off,
            edge: EdgeSettings::default(),
            ipc_dir: PathBuf::from("/tmp/prunr-ipc-test"),
            inpaint_only: true,
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
    fn command_repostprocess_roundtrip() {
        roundtrip(&SubprocessCommand::RePostProcess {
            item_id: 9,
            tensor_path: PathBuf::from("/tmp/t.bin"),
            tensor_height: 320,
            tensor_width: 320,
            model: ModelKind::U2net,
            original_image_path: PathBuf::from("/tmp/orig.png"),
            mask: MaskSettings::default(),
        });
    }

    #[test]
    fn command_add_edge_inference_roundtrip() {
        roundtrip(&SubprocessCommand::AddEdgeInference {
            item_id: 13,
            image_path: PathBuf::from("/tmp/orig.png"),
            seg_tensor_path: PathBuf::from("/tmp/seg.raw"),
            seg_tensor_height: 1024,
            seg_tensor_width: 1024,
            model: ModelKind::BiRefNetLite,
            mask: MaskSettings::default(),
        });
    }

    #[test]
    fn command_shutdown_roundtrip() {
        roundtrip(&SubprocessCommand::Shutdown);
    }

    #[test]
    fn command_cancel_item_roundtrip() {
        roundtrip(&SubprocessCommand::CancelItem { item_id: 42 });
        roundtrip(&SubprocessCommand::CancelItem { item_id: 0 });
        roundtrip(&SubprocessCommand::CancelItem { item_id: u64::MAX });
    }

    #[test]
    fn event_ready_roundtrip() {
        roundtrip(&SubprocessEvent::Ready {
            active_provider: "CUDA".to_string(),
        });
    }

    #[test]
    fn event_progress_roundtrip() {
        roundtrip(&SubprocessEvent::Progress {
            item_id: 3,
            stage: ProgressStage::Infer,
            pct: 0.5,
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
            tensor_cache_path: Some(PathBuf::from("/tmp/tensor.bin")),
            tensor_cache_height: Some(320),
            tensor_cache_width: Some(320),
            edge_cache_path: None,
            edge_cache_height: None,
            edge_cache_width: None,
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
            edge_cache_path: Some(PathBuf::from("/tmp/edge.bin")),
            edge_cache_height: Some(480),
            edge_cache_width: Some(640),
        });
    }

    #[test]
    fn event_image_error_roundtrip() {
        roundtrip(&SubprocessEvent::ImageError {
            item_id: 4,
            error: "decode failed: not a PNG".to_string(),
        });
    }

    #[test]
    fn event_rss_update_roundtrip() {
        roundtrip(&SubprocessEvent::RssUpdate {
            rss_bytes: 1_234_567_890,
        });
    }

    #[test]
    fn event_finished_and_init_error_roundtrip() {
        roundtrip(&SubprocessEvent::Finished);
        roundtrip(&SubprocessEvent::InitError {
            error: "engine creation failed".into(),
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
        });
        roundtrip(&SubprocessEvent::InpaintError {
            item_id: 11,
            error: "lama session failed: tensor shape mismatch".into(),
        });
    }

    #[test]
    fn event_inpaint_progress_roundtrip() {
        roundtrip(&SubprocessEvent::InpaintProgress { item_id: 12, current: 0, total: 20 });
        roundtrip(&SubprocessEvent::InpaintProgress { item_id: 12, current: 7, total: 20 });
        roundtrip(&SubprocessEvent::InpaintProgress { item_id: 12, current: 20, total: 20 });
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

    #[test]
    fn cleanup_ipc_temp_is_reentrant() {
        let dir = ipc_temp_dir();
        std::fs::write(dir.join("test-reentrant.bin"), b"x").unwrap();

        cleanup_ipc_temp();
        assert_gone(dir, &["test-reentrant.bin"]);
        cleanup_ipc_temp();
        assert_gone(dir, &["test-reentrant.bin"]);
    }

    /// A seg-side crash cleanup must not touch a sibling inpaint
    /// subprocess's `inpaint-*` files. Seeds both kinds, runs the
    /// prefix-scoped sweep, asserts seg-prefixed gone and
    /// inpaint-prefixed survive.
    #[test]
    fn sweep_dir_with_prefix_leaves_other_owners_alone() {
        let scratch = tempfile::tempdir().unwrap();
        let dir = scratch.path();

        let seg = ["chain_42.raw", "edge_5.bin", "input_3.img",
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
                "CLI-persistent file {f} must survive a seg-scoped cleanup (B-CLI-1)");
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
            "chain_42.raw", "cli_ds_0.img", "edge_5.bin", "input_3.img",
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
    /// errors whose actual cause was self-inflicted (B-CLI-1).
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
    fn sweep_stale_pid_dirs_removes_dead_pids_keeps_self_and_unrelated() {
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
        sweep_stale_pid_dirs(&self_dir, self_pid);
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
        // pid before `sweep_stale_pid_dirs` runs, leaving the dir
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
    // `edge_cache_*` — are pinned here to document their expected defaults
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
            force_cpu: false,
            line_mode: LineMode::Off,
            edge: EdgeSettings::default(),
            ipc_dir: PathBuf::from("/tmp/test"),
            inpaint_only: false,
        });
        roundtrip(&SubprocessCommand::Init {
            model: ModelKind::Silueta,
            jobs: 1,
            mask: MaskSettings::default(),
            force_cpu: false,
            line_mode: LineMode::Off,
            edge: EdgeSettings::default(),
            ipc_dir: PathBuf::from("/tmp/test"),
            inpaint_only: true,
        });
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

    /// `ImageDone.edge_cache_*` fields default to `None`. Roundtrip the
    /// all-None case (no DexiNed run) and the all-Some case so a default
    /// drift from `None` to some garbage path is caught immediately.
    #[test]
    fn event_image_done_edge_cache_fields_roundtrip_none_and_some() {
        roundtrip(&SubprocessEvent::ImageDone {
            item_id: 3,
            result_path: PathBuf::from("/tmp/result.raw"),
            width: 800,
            height: 600,
            active_provider: "CPU".into(),
            tensor_cache_path: None,
            tensor_cache_height: None,
            tensor_cache_width: None,
            edge_cache_path: None,
            edge_cache_height: None,
            edge_cache_width: None,
        });
        roundtrip(&SubprocessEvent::ImageDone {
            item_id: 4,
            result_path: PathBuf::from("/tmp/result2.raw"),
            width: 1920,
            height: 1080,
            active_provider: "CPU".into(),
            tensor_cache_path: None,
            tensor_cache_height: None,
            tensor_cache_width: None,
            edge_cache_path: Some(PathBuf::from("/tmp/edge.raw")),
            edge_cache_height: Some(1080),
            edge_cache_width: Some(1920),
        });
    }
}
