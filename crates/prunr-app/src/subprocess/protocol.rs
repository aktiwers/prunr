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
        bg_color: Option<[u8; 3]>,
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
/// Prefers RAM-backed tmpfs (/dev/shm on Linux) when available,
/// falls back to system temp dir.
pub fn ipc_temp_dir() -> std::path::PathBuf {
    // Linux: /dev/shm is RAM-backed tmpfs — no disk I/O.
    // Try to create subdirectory directly; fall back to system temp on failure.
    #[cfg(target_os = "linux")]
    {
        let dir = std::path::PathBuf::from("/dev/shm/prunr-ipc");
        if std::fs::create_dir_all(&dir).is_ok() {
            return dir;
        }
    }
    let dir = std::env::temp_dir().join("prunr-ipc");
    let _ = std::fs::create_dir_all(&dir);
    dir
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
