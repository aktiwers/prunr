//! IPC protocol types for parent ↔ child subprocess communication.
//!
//! All types are serde-serializable and sent as length-prefixed bincode
//! frames over stdin/stdout.

use prunr_core::{ModelKind, MaskSettings, ProgressStage};
use serde::{Serialize, Deserialize};

use crate::gui::settings::LineMode;

/// Parent → Child commands (sent over child's stdin).
#[derive(Serialize, Deserialize, Debug)]
pub enum SubprocessCommand {
    /// Initialize the worker: load model, create engine pool.
    Init {
        model: ModelKind,
        jobs: usize,
        mask: MaskSettings,
        force_cpu: bool,
        line_mode: LineMode,
        line_strength: f32,
        solid_line_color: Option<[u8; 3]>,
        /// IPC temp directory (set by parent so child uses the same PID-namespaced dir)
        ipc_dir: std::path::PathBuf,
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
    /// Cancel: stop after current image, send Finished.
    Cancel,
    /// Shut down gracefully.
    Shutdown,
}

/// Chain mode input: previous result as raw RGBA in a temp file.
#[derive(Serialize, Deserialize, Debug)]
pub struct ChainInput {
    pub path: std::path::PathBuf,
    pub width: u32,
    pub height: u32,
}

/// Child → Parent events (sent over child's stdout).
#[derive(Serialize, Deserialize, Debug)]
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
        /// Path to cached raw tensor (Tier 1 output). Parent stores for future Tier 2 runs.
        /// Only present for full pipeline runs, not RePostProcess.
        tensor_cache_path: Option<std::path::PathBuf>,
        tensor_cache_height: Option<u32>,
        tensor_cache_width: Option<u32>,
    },
    /// Image processing failed (non-fatal).
    ImageError {
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

/// Return the preferred directory for IPC temp files.
/// PID-namespaced to prevent collisions between multiple app instances.
/// Uses the **parent** PID so both parent and child agree on the same directory.
/// Prefers RAM-backed tmpfs (/dev/shm on Linux), falls back to system temp dir.
pub fn ipc_temp_dir() -> std::path::PathBuf {
    use std::sync::OnceLock;
    static DIR: OnceLock<std::path::PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let pid = std::process::id();
        #[cfg(target_os = "linux")]
        {
            let dir = std::path::PathBuf::from(format!("/dev/shm/prunr-ipc-{pid}"));
            if std::fs::create_dir_all(&dir).is_ok() {
                return dir;
            }
        }
        let dir = std::env::temp_dir().join(format!("prunr-ipc-{pid}"));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }).clone()
}

/// Clean up all IPC temp files.
pub fn cleanup_ipc_temp() {
    let dir = ipc_temp_dir();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}
