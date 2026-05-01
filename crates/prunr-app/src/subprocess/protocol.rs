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
    /// Cancel: stop after current image, send Finished.
    Cancel,
    /// Cancel one item by id — worker emits `ImageError { error:
    /// CANCELLED_ERR_MSG }` at the next dispatch check. Other in-flight
    /// jobs keep running, unlike `Cancel` which stops the whole batch.
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

/// Return the preferred directory for IPC temp files.
///
/// PID-namespaced to prevent collisions between multiple app instances.
/// Uses the **parent** PID so both parent and child agree on the same directory.
/// Prefers RAM-backed tmpfs (/dev/shm on Linux), falls back to system temp dir.
///
/// **Init-time sweep:** the OnceLock initializer creates the directory AND
/// sweeps any stale files left by a previous prunr process with the same PID
/// (rare but possible after a crash + OS pid reuse). This means any caller —
/// CLI downscale path writing `cli_ds_*.img`, inpaint bridge writing
/// `inpaint-img-*.png`, etc. — can write to a path derived from
/// `ipc_temp_dir()` without racing a later cleanup. The sweep happens on
/// first access; subsequent calls return the cached path with no sweep.
///
/// Earlier design had the sweep inside `SubprocessManager::spawn_inner`, which
/// raced any caller that wrote temp files BEFORE spawning the subprocess
/// (CLI's `--large-image=downscale` path; the inpaint bridge's lazy
/// first-dispatch — see `5454d46`'s once-gate fix and B2 in the review).
pub fn ipc_temp_dir() -> std::path::PathBuf {
    use std::sync::OnceLock;
    static DIR: OnceLock<std::path::PathBuf> = OnceLock::new();
    DIR.get_or_init(|| init_temp_dir_for_pid(std::process::id())).clone()
}

/// Resolve the IPC temp dir path for `pid`, create it, and sweep any stale
/// files. Extracted from `ipc_temp_dir()`'s OnceLock initializer so the
/// init-time-sweep contract can be tested in isolation (without depending on
/// the global OnceLock state, which is process-wide and order-dependent
/// across tests in the same binary).
fn init_temp_dir_for_pid(pid: u32) -> std::path::PathBuf {
    let dir = resolve_temp_dir_path(pid);
    let _ = std::fs::create_dir_all(&dir);
    sweep_dir(&dir);
    dir
}

fn resolve_temp_dir_path(pid: u32) -> std::path::PathBuf {
    #[cfg(target_os = "linux")]
    {
        let dir = std::path::PathBuf::from(format!("/dev/shm/prunr-ipc-{pid}"));
        // Use /dev/shm if it exists (it does on every Linux distro we
        // support); the initializer below will create the per-pid subdir.
        if std::path::Path::new("/dev/shm").is_dir() {
            return dir;
        }
    }
    std::env::temp_dir().join(format!("prunr-ipc-{pid}"))
}

/// Remove every file directly inside `dir` (non-recursive). Misses on
/// IO errors (best-effort); a stale orphan won't break the next run because
/// the IPC filenames are item-id-keyed and writers re-create them.
fn sweep_dir(dir: &std::path::Path) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Explicit re-entrant cleanup of the current process's IPC temp dir. Used
/// for crash recovery after a subprocess hard-kill (see
/// `worker::cancel_subprocess` and `cli.rs` post-crash retry path) — these
/// callers know they want a sweep regardless of the once-only init.
pub fn cleanup_ipc_temp() {
    sweep_dir(&ipc_temp_dir());
}

#[cfg(test)]
mod tests {
    //! Bincode round-trip tests for every `SubprocessCommand` and
    //! `SubprocessEvent` variant. A single generic helper covers both enums
    //! via `PartialEq` — any drift between encoded and decoded payload fails
    //! the assertion with a real diff, not a stringified debug comparison.
    //!
    //! Plus boundary tests for `ipc_temp_dir`'s init-time sweep contract
    //! (Phase 21-02 / B2): the sweep must run on first access, must NOT run
    //! again on subsequent accesses, and the public `cleanup_ipc_temp` must
    //! be safely re-entrant for crash-recovery callers.
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
    fn command_cancel_and_shutdown_roundtrip() {
        roundtrip(&SubprocessCommand::Cancel);
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

    /// Build a unique sandbox dir for a test (avoids touching the real
    /// `/dev/shm/prunr-ipc-<pid>` path which is OnceLock-cached and would
    /// be polluted by prior tests in the same binary).
    fn unique_sandbox(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        // pid + nanos gives uniqueness across parallel test threads in the
        // same process (cargo runs lib tests on a thread pool); the Debug
        // formatting of ThreadId is stable enough for a sandbox label.
        let tid = format!("{:?}", std::thread::current().id());
        std::env::temp_dir().join(format!(
            "prunr-test-{}-{}-{}-{}",
            label,
            std::process::id(),
            nanos,
            tid.replace(['(', ')', ' '], ""),
        ))
    }

    #[test]
    fn sweep_dir_removes_files_non_recursive() {
        let dir = unique_sandbox("sweep-dir");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.bin"), b"a").unwrap();
        std::fs::write(dir.join("b.png"), b"b").unwrap();
        std::fs::create_dir_all(dir.join("subdir")).unwrap();
        std::fs::write(dir.join("subdir").join("nested.bin"), b"c").unwrap();

        super::sweep_dir(&dir);

        // Files at the top level: gone.
        assert!(!dir.join("a.bin").exists(), "sweep_dir must remove top-level files");
        assert!(!dir.join("b.png").exists(), "sweep_dir must remove top-level files");
        // Nested subdir: untouched (sweep is non-recursive by design — IPC
        // doesn't create subdirs in the temp area, and recursing would risk
        // wiping anything a future caller mistakenly drops there).
        assert!(dir.join("subdir").is_dir(), "sweep_dir must NOT recurse");
        assert!(
            dir.join("subdir").join("nested.bin").exists(),
            "sweep_dir must NOT recurse"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Phase 21-02 / B2 contract: the init function (the body of
    /// `ipc_temp_dir()`'s OnceLock initializer) must sweep stale files when
    /// it runs. Pre-poisons the dir for a test-only PID, calls the init
    /// helper, asserts stale files are gone — proving that any caller who
    /// then writes via the path (CLI downscale, inpaint bridge,
    /// SubprocessManager) lands in a clean dir.
    #[test]
    fn init_temp_dir_for_pid_sweeps_stale_leftovers() {
        // Use a unique PID-like number so we never collide with a real
        // ipc_temp_dir() in this same test binary's process.
        let test_pid: u32 = 0xDEAD_BEAFu32.wrapping_add(std::process::id());
        let dir = super::resolve_temp_dir_path(test_pid);

        // Pre-poison: simulate "previous prunr instance with same PID
        // crashed and left these files behind".
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("stale-cli_ds_0.img"), b"orphan").unwrap();
        std::fs::write(dir.join("stale-inpaint-out.png"), b"orphan").unwrap();
        assert!(dir.join("stale-cli_ds_0.img").exists());
        assert!(dir.join("stale-inpaint-out.png").exists());

        // Init: should sweep.
        let returned = super::init_temp_dir_for_pid(test_pid);
        assert_eq!(returned, dir);

        // Stale files gone.
        assert!(
            !dir.join("stale-cli_ds_0.img").exists(),
            "init_temp_dir_for_pid must sweep stale files"
        );
        assert!(
            !dir.join("stale-inpaint-out.png").exists(),
            "init_temp_dir_for_pid must sweep stale files"
        );
        // Dir itself still exists (init created it before sweeping).
        assert!(dir.is_dir());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Files written AFTER init must survive any subsequent `init` re-call
    /// for the same PID. This is the exact race shape B2 fixed: a caller
    /// writes a file, then something later (formerly: spawn) wipes the
    /// dir. Init is now idempotent-by-construction (sweep only happens on
    /// the first call inside the OnceLock); this test verifies the
    /// underlying `init_temp_dir_for_pid` is safe to call again — though
    /// in practice the OnceLock guarantees it never is.
    #[test]
    fn writes_after_init_survive_a_second_init_with_no_pre_existing_files() {
        let test_pid: u32 = 0xCAFE_BABEu32.wrapping_add(std::process::id());
        let dir = super::resolve_temp_dir_path(test_pid);
        let _ = std::fs::remove_dir_all(&dir);

        // First init on a fresh path: dir created, no files to sweep.
        super::init_temp_dir_for_pid(test_pid);
        // Caller writes a file (the CLI / inpaint bridge pattern).
        let marker = dir.join("cli_ds_0.img");
        std::fs::write(&marker, b"important payload").unwrap();
        assert!(marker.exists());

        // Second init: the underlying helper DOES sweep (it doesn't know
        // any better — the OnceLock above it is what enforces "once").
        // This test pins that contract: writers must rely on the OnceLock
        // gate, not on init being idempotent. If we ever change init to
        // be idempotent (skip sweep on subsequent calls), this assertion
        // updates and the contract changes consciously.
        super::init_temp_dir_for_pid(test_pid);
        assert!(
            !marker.exists(),
            "init_temp_dir_for_pid sweeps unconditionally — \
             OnceLock above it gates the once-only contract; \
             callers must never invoke init_temp_dir_for_pid twice for the same pid"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Public `cleanup_ipc_temp` must be safely re-entrant for crash-recovery
    /// callers (`worker::cancel_subprocess`, `cli.rs` post-crash retry path).
    /// Re-entrancy here means: calling it multiple times must not panic and
    /// must always leave the dir empty (sweeping each time is fine; the
    /// OnceLock-gated init only runs once but cleanup_ipc_temp uses sweep_dir
    /// directly, not init).
    #[test]
    fn cleanup_ipc_temp_is_reentrant() {
        // Trigger the global ipc_temp_dir() init (idempotent in this test
        // binary's process — only happens once across all tests).
        let dir = ipc_temp_dir();

        // Drop a marker so we have something to sweep.
        std::fs::write(dir.join("test-reentrant.bin"), b"x").unwrap();
        assert!(dir.join("test-reentrant.bin").exists());

        cleanup_ipc_temp();
        assert!(!dir.join("test-reentrant.bin").exists());

        // Second call: must not panic, dir stays empty.
        cleanup_ipc_temp();
        assert!(!dir.join("test-reentrant.bin").exists());
    }
}
