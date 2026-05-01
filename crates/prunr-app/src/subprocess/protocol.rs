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
/// The OnceLock initializer creates the dir AND sweeps any stale files left
/// by a previous prunr process with the same PID, so writers (CLI downscale,
/// inpaint bridge, manager) never race a later cleanup.
pub fn ipc_temp_dir() -> &'static std::path::Path {
    use std::sync::OnceLock;
    static DIR: OnceLock<std::path::PathBuf> = OnceLock::new();
    DIR.get_or_init(|| init_temp_dir_for_pid(std::process::id())).as_path()
}

/// Body of `ipc_temp_dir()`'s OnceLock init, exposed for unit tests
/// (the global is process-wide and order-dependent across tests).
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

/// File-name prefixes owned by the seg / CLI subprocess pipeline.
pub const SEG_PIPELINE_PREFIXES: &[&str] = &[
    "chain_",
    "cli_ds_",
    "edge_",
    "input_",
    "orig_",
    "result_",
    "seg_",
    "tensor_",
];

/// File-name prefixes owned by the inpaint subprocess.
pub const INPAINT_PREFIXES: &[&str] = &[
    "inpaint-",
];

/// Cleanup scoped to a single subprocess's owned filenames. Use from
/// crash-recovery paths so the wipe doesn't race a sibling subprocess
/// writing the shared dir.
pub fn cleanup_ipc_temp_for_prefix(prefixes: &[&str]) {
    sweep_dir_with_prefix(ipc_temp_dir(), prefixes);
}

/// Crash-recovery cleanup for the seg / CLI subprocess. Leaves a sibling
/// inpaint subprocess's `inpaint-*` files alone.
pub fn cleanup_seg_pipeline_temps() {
    cleanup_ipc_temp_for_prefix(SEG_PIPELINE_PREFIXES);
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

    // ---------- ipc_temp_dir lifecycle (B2 / Phase 21-02) ----------

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

    /// B3 / Phase 21-03: a seg-side crash cleanup must not touch a sibling
    /// inpaint subprocess's `inpaint-*` files. Seeds both kinds, runs the
    /// prefix-scoped sweep, asserts seg-prefixed gone and inpaint-prefixed
    /// survive.
    #[test]
    fn sweep_dir_with_prefix_leaves_other_owners_alone() {
        let scratch = tempfile::tempdir().unwrap();
        let dir = scratch.path();

        let seg = ["chain_42.raw", "cli_ds_0.img", "edge_5.bin", "input_3.img",
                   "orig_3.img", "result_3.raw", "seg_3.png", "tensor_3.raw"];
        let inpaint = ["inpaint-img-7-0.png", "inpaint-mask-7-0.png", "inpaint-out-7.png"];
        poison(dir, &seg);
        poison(dir, &inpaint);

        super::sweep_dir_with_prefix(dir, SEG_PIPELINE_PREFIXES);

        assert_gone(dir, &seg);
        for f in &inpaint {
            assert!(dir.join(f).exists(), "inpaint file {f} must survive a seg-scoped cleanup");
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
                    .any(|p| name.starts_with(p)),
                "filename {name} matches no prefix in SEG_PIPELINE_PREFIXES + INPAINT_PREFIXES"
            );
        }
    }
}
