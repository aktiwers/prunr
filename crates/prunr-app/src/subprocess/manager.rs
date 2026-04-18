//! Parent-side subprocess manager.
//!
//! Spawns `prunr --worker`, manages IPC over stdin/stdout, tracks in-flight
//! items, and monitors RSS for admission throttling.

use std::collections::HashSet;
use std::io::{BufReader, BufWriter};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;

use prunr_core::{ModelKind, MaskSettings, EdgeSettings};
use crate::gui::settings::LineMode;
use super::protocol::*;
use super::ipc::{write_message, read_message};

/// Manages a subprocess worker: spawning, IPC, in-flight tracking, RSS throttling.
pub struct SubprocessManager {
    child: Child,
    stdin_writer: BufWriter<std::process::ChildStdin>,
    /// Channel from the reader thread (reads child stdout → events).
    event_rx: mpsc::Receiver<ReaderEvent>,
    /// Items currently being processed by this subprocess.
    in_flight: HashSet<u64>,
    /// RSS limit: pause admission when child RSS exceeds this.
    rss_limit: u64,
    /// Resume threshold (hysteresis): resume when RSS drops below this.
    rss_resume: u64,
    /// Whether admission is paused due to high RSS.
    rss_paused: bool,
    /// Last known RSS from child.
    last_rss: u64,
}

/// Events from the reader thread.
enum ReaderEvent {
    /// Successfully read a subprocess event.
    Event(SubprocessEvent),
    /// Reader thread exited (child stdout closed or error).
    Disconnected,
}

impl SubprocessManager {
    /// Spawn a `prunr --worker` subprocess and send the Init command.
    /// Blocks until the child sends `Ready` or `InitError`.
    pub fn spawn(
        model: ModelKind,
        jobs: usize,
        mask: MaskSettings,
        force_cpu: bool,
        line_mode: LineMode,
        edge: EdgeSettings,
    ) -> Result<(Self, String), String> {
        // Clean up stale IPC temp files from previous workers (crash recovery)
        super::protocol::cleanup_ipc_temp();

        let exe = std::env::current_exe()
            .map_err(|e| format!("Failed to get current exe: {e}"))?;

        let mut child = Command::new(&exe)
            .arg("--worker")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // child errors visible in parent console
            .spawn()
            .map_err(|e| format!("Failed to spawn worker: {e}"))?;

        let child_stdin = child.stdin.take()
            .ok_or("Failed to capture worker stdin")?;
        let child_stdout = child.stdout.take()
            .ok_or("Failed to capture worker stdout")?;

        let mut stdin_writer = BufWriter::new(child_stdin);

        // Send Init command
        write_message(&mut stdin_writer, &SubprocessCommand::Init {
            model, jobs, mask, force_cpu, line_mode, edge,
            ipc_dir: ipc_temp_dir(),
        }).map_err(|e| format!("Failed to send Init: {e}"))?;

        // Spawn reader thread for non-blocking stdout consumption
        let (event_tx, event_rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("subprocess-reader".into())
            .spawn(move || {
                let mut reader = BufReader::new(child_stdout);
                loop {
                    match read_message::<_, SubprocessEvent>(&mut reader) {
                        Ok(Some(evt)) => {
                            if event_tx.send(ReaderEvent::Event(evt)).is_err() {
                                break; // parent dropped receiver
                            }
                        }
                        Ok(None) => {
                            // Child stdout closed (clean exit or crash)
                            let _ = event_tx.send(ReaderEvent::Disconnected);
                            break;
                        }
                        Err(_) => {
                            let _ = event_tx.send(ReaderEvent::Disconnected);
                            break;
                        }
                    }
                }
            })
            .map_err(|e| format!("Failed to spawn reader thread: {e}"))?;

        // Wait for Ready or InitError. 300s timeout covers macOS CoreML
        // first-run compilation which can take 2-5 minutes on slow systems.
        let active_provider = match event_rx.recv_timeout(std::time::Duration::from_secs(300)) {
            Ok(ReaderEvent::Event(SubprocessEvent::Ready { active_provider })) => {
                active_provider
            }
            Ok(ReaderEvent::Event(SubprocessEvent::InitError { error })) => {
                let _ = child.kill();
                return Err(format!("Worker init failed: {error}"));
            }
            Ok(ReaderEvent::Disconnected) => {
                let _ = child.kill();
                return Err("Worker exited during initialization".to_string());
            }
            Ok(ReaderEvent::Event(other)) => {
                let _ = child.kill();
                return Err(format!("Unexpected event during init: {other:?}"));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = child.kill();
                return Err("Worker init timed out (300s)".to_string());
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = child.kill();
                return Err("Worker reader thread died during init".to_string());
            }
        };

        // Calculate RSS limits from available system RAM
        let available = {
            use sysinfo::System;
            let mut sys = System::new();
            sys.refresh_memory();
            sys.available_memory()
        };
        let rss_limit = (available as f64 * 0.80) as u64;
        let rss_resume = (rss_limit as f64 * 0.70) as u64;

        Ok((Self {
            child,
            stdin_writer,
            event_rx,
            in_flight: HashSet::new(),
            rss_limit,
            rss_resume,
            rss_paused: false,
            last_rss: 0,
        }, active_provider))
    }

    /// Send an image for processing. Writes bytes to a temp file first.
    pub fn send_image(
        &mut self,
        item_id: u64,
        image_bytes: &[u8],
        chain_input: Option<(&image::RgbaImage, u32, u32)>,
    ) -> Result<(), String> {
        // Write image bytes to temp file
        let image_path = ipc_temp_dir().join(format!("input_{item_id}.img"));
        std::fs::write(&image_path, image_bytes)
            .map_err(|e| format!("Failed to write input temp file: {e}"))?;

        // Write chain input to temp file if present
        let chain = chain_input.map(|(rgba, w, h)| {
            let path = ipc_temp_dir().join(format!("chain_{item_id}.raw"));
            let _ = std::fs::write(&path, rgba.as_raw());
            ChainInput { path, width: w, height: h }
        });

        write_message(&mut self.stdin_writer, &SubprocessCommand::ProcessImage {
            item_id,
            image_path,
            chain_input: chain,
        }).map_err(|e| format!("Failed to send ProcessImage: {e}"))?;

        self.in_flight.insert(item_id);
        Ok(())
    }

    /// Send an image for processing directly from a file path (no copy).
    /// The subprocess reads the file; caller is responsible for cleanup.
    pub fn send_image_path(
        &mut self,
        item_id: u64,
        image_path: std::path::PathBuf,
    ) -> Result<(), String> {
        write_message(&mut self.stdin_writer, &SubprocessCommand::ProcessImage {
            item_id,
            image_path,
            chain_input: None,
        }).map_err(|e| format!("Failed to send ProcessImage: {e}"))?;
        self.in_flight.insert(item_id);
        Ok(())
    }

    /// Send a Tier 2 re-postprocess command (skip inference, reuse cached tensor).
    pub fn send_repostprocess(
        &mut self,
        item_id: u64,
        tensor_data: &[f32],
        tensor_height: u32,
        tensor_width: u32,
        model: prunr_core::ModelKind,
        original_image_bytes: &[u8],
        mask: prunr_core::MaskSettings,
    ) -> Result<(), String> {
        // Write tensor as raw f32 LE bytes to temp file
        let tensor_path = ipc_temp_dir().join(format!("tensor_{item_id}.raw"));
        let tensor_bytes = super::ipc::f32s_to_le_bytes(tensor_data);
        std::fs::write(&tensor_path, &tensor_bytes)
            .map_err(|e| format!("Failed to write tensor temp file: {e}"))?;

        // Write original image bytes to temp file
        let orig_path = ipc_temp_dir().join(format!("orig_{item_id}.img"));
        std::fs::write(&orig_path, original_image_bytes)
            .map_err(|e| format!("Failed to write original temp file: {e}"))?;

        write_message(&mut self.stdin_writer, &SubprocessCommand::RePostProcess {
            item_id,
            tensor_path,
            tensor_height,
            tensor_width,
            model,
            original_image_path: orig_path,
            mask,
        }).map_err(|e| format!("Failed to send RePostProcess: {e}"))?;

        self.in_flight.insert(item_id);
        Ok(())
    }

    /// Send cancel signal to the child.
    pub fn send_cancel(&mut self) -> Result<(), String> {
        write_message(&mut self.stdin_writer, &SubprocessCommand::Cancel)
            .map_err(|e| format!("Failed to send Cancel: {e}"))
    }

    /// Send shutdown signal to the child.
    pub fn send_shutdown(&mut self) -> Result<(), String> {
        write_message(&mut self.stdin_writer, &SubprocessCommand::Shutdown)
            .map_err(|e| format!("Failed to send Shutdown: {e}"))
    }

    /// Non-blocking poll for events from the subprocess.
    pub fn poll_events(&mut self) -> Vec<SubprocessEvent> {
        let mut events = Vec::new();
        loop {
            match self.event_rx.try_recv() {
                Ok(ReaderEvent::Event(evt)) => {
                    // Update internal state based on event type
                    match &evt {
                        SubprocessEvent::ImageDone { item_id, .. }
                        | SubprocessEvent::ImageError { item_id, .. } => {
                            self.in_flight.remove(item_id);
                        }
                        SubprocessEvent::RssUpdate { rss_bytes } => {
                            self.last_rss = *rss_bytes;
                            if *rss_bytes > self.rss_limit {
                                self.rss_paused = true;
                            } else if *rss_bytes < self.rss_resume {
                                self.rss_paused = false;
                            }
                        }
                        _ => {}
                    }
                    events.push(evt);
                }
                Ok(ReaderEvent::Disconnected) => {
                    // Child exited — mark all in-flight as failed
                    // (caller handles retry logic)
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
        events
    }

    /// Check if the subprocess is still alive.
    pub fn is_alive(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(Some(_)) => false, // exited
            Ok(None) => true,     // still running
            Err(_) => false,      // error checking — assume dead
        }
    }

    /// Whether admission should be paused due to high child RSS.
    pub fn should_pause_admission(&self) -> bool {
        self.rss_paused
    }

    /// Get the set of in-flight item IDs (for re-queuing on crash).
    pub fn in_flight_items(&self) -> &HashSet<u64> {
        &self.in_flight
    }

    /// Describe why the subprocess died (for user-facing messages).
    pub fn crash_reason(&mut self) -> String {
        match self.child.try_wait() {
            Ok(Some(status)) => {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    if let Some(signal) = status.signal() {
                        return match signal {
                            9 => "Process killed by OS (out of memory)".to_string(),
                            11 => "Process crashed (segmentation fault)".to_string(),
                            _ => format!("Process killed by signal {signal}"),
                        };
                    }
                }
                if let Some(code) = status.code() {
                    format!("Process exited with code {code}")
                } else {
                    "Process terminated unexpectedly".to_string()
                }
            }
            _ => "Worker process stopped responding".to_string(),
        }
    }

    /// Kill the subprocess forcefully.
    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait(); // reap zombie
    }
}

impl Drop for SubprocessManager {
    fn drop(&mut self) {
        // Try graceful shutdown first, then force kill
        let _ = write_message(&mut self.stdin_writer, &SubprocessCommand::Shutdown);
        // Give child 1 second to exit
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                _ if std::time::Instant::now() > deadline => {
                    self.kill();
                    break;
                }
                _ => std::thread::sleep(std::time::Duration::from_millis(50)),
            }
        }
    }
}
