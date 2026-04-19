//! IPC protocol types for parent ↔ child subprocess communication.
//!
//! All types are serde-serializable and sent as length-prefixed bincode
//! frames over stdin/stdout.

use prunr_core::{ModelKind, MaskSettings, EdgeSettings, ProgressStage, LineMode};
use serde::{Serialize, Deserialize};

/// Parent → Child commands (sent over child's stdin).
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub enum SubprocessCommand {
    /// Initialize the worker: load model, create engine pool.
    Init {
        model: ModelKind,
        jobs: usize,
        mask: MaskSettings,
        force_cpu: bool,
        line_mode: LineMode,
        edge: EdgeSettings,
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
    /// Cancel: stop after current image, send Finished.
    Cancel,
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
}
