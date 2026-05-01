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
    #[tracing::instrument(
        skip_all,
        fields(model = ?model, jobs, force_cpu, line_mode = ?line_mode),
    )]
    pub fn spawn(
        model: ModelKind,
        jobs: usize,
        mask: MaskSettings,
        force_cpu: bool,
        line_mode: LineMode,
        edge: EdgeSettings,
    ) -> Result<(Self, String), String> {
        Self::spawn_inner(model, jobs, mask, force_cpu, line_mode, edge, false)
    }

    /// Spawn a SD-inpaint-only subprocess. The worker skips engine
    /// creation when `inpaint_only` is set, so the seg fields on
    /// `Init` are dead weight at this stage — we pass placeholders.
    /// Used by `Processor::dispatch_inpaint` for SD models so an OOM
    /// during the bundle build kills the subprocess, not the GUI.
    pub fn spawn_inpaint_only() -> Result<(Self, String), String> {
        Self::spawn_inner(
            ModelKind::Silueta, 1,
            MaskSettings::default(), false,
            LineMode::Off, EdgeSettings::default(),
            true,
        )
    }

    fn spawn_inner(
        model: ModelKind,
        jobs: usize,
        mask: MaskSettings,
        force_cpu: bool,
        line_mode: LineMode,
        edge: EdgeSettings,
        inpaint_only: bool,
    ) -> Result<(Self, String), String> {
        // Stale-file cleanup is now owned by `ipc_temp_dir()`'s init-time
        // sweep (see `protocol::ipc_temp_dir` doc). Spawning a subprocess
        // must NOT touch the temp dir — callers like the CLI downscale path
        // and the inpaint bridge's lazy first-dispatch write input files
        // BEFORE spawn, and a spawn-time wipe would race those writes (B2).
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
            inpaint_only,
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

    /// Send an AddEdgeInference command (seg cached, run DexiNed on masked).
    /// Writes the seg tensor + original image bytes to temp files. The worker
    /// reads the seg tensor without deleting it and hands the path back as
    /// `tensor_cache_path` on `ImageDone`, so the parent's reader takes
    /// ownership — no extra copy round-trip.
    pub fn send_add_edge_inference(
        &mut self,
        item_id: u64,
        tensor_data: &[f32],
        tensor_height: u32,
        tensor_width: u32,
        model: prunr_core::ModelKind,
        original_image_bytes: &[u8],
        mask: prunr_core::MaskSettings,
    ) -> Result<(), String> {
        let seg_tensor_path = ipc_temp_dir().join(format!("seg_{item_id}.raw"));
        std::fs::write(&seg_tensor_path, super::ipc::f32s_as_le_bytes(tensor_data))
            .map_err(|e| format!("Failed to write seg tensor temp file: {e}"))?;
        let image_path = ipc_temp_dir().join(format!("input_{item_id}.img"));
        std::fs::write(&image_path, original_image_bytes)
            .map_err(|e| format!("Failed to write input temp file: {e}"))?;

        write_message(&mut self.stdin_writer, &SubprocessCommand::AddEdgeInference {
            item_id,
            image_path,
            seg_tensor_path,
            seg_tensor_height: tensor_height,
            seg_tensor_width: tensor_width,
            model,
            mask,
        }).map_err(|e| format!("Failed to send AddEdgeInference: {e}"))?;

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
        std::fs::write(&tensor_path, super::ipc::f32s_as_le_bytes(tensor_data))
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

    /// Send an `Inpaint` command. Caller writes image + mask to PNG
    /// files (worker decodes); this method just forwards the paths +
    /// model_id + sd_req + post-process tuning. Returns immediately —
    /// completion comes back via `InpaintDone` / `InpaintError` events
    /// through `poll_events`.
    pub fn send_inpaint(
        &mut self,
        item_id: u64,
        model_id: prunr_models::ModelId,
        image_path: std::path::PathBuf,
        mask_path: std::path::PathBuf,
        sd_req: Option<prunr_core::inpaint_sd::SdInpaintRequest>,
        feather_px: f32,
        sharpen: f32,
    ) -> Result<(), String> {
        write_message(&mut self.stdin_writer, &SubprocessCommand::Inpaint {
            item_id, model_id, image_path, mask_path, sd_req, feather_px, sharpen,
        }).map_err(|e| format!("Failed to send Inpaint: {e}"))?;
        self.in_flight.insert(item_id);
        Ok(())
    }

    /// Send cancel signal to the child.
    pub fn send_cancel(&mut self) -> Result<(), String> {
        write_message(&mut self.stdin_writer, &SubprocessCommand::Cancel)
            .map_err(|e| format!("Failed to send Cancel: {e}"))
    }

    /// Cancel one item by id — worker drops it at the next dispatch check
    /// and emits `ImageError { error: "Cancelled" }`.
    pub fn send_cancel_item(&mut self, item_id: u64) -> Result<(), String> {
        write_message(&mut self.stdin_writer, &SubprocessCommand::CancelItem { item_id })
            .map_err(|e| format!("Failed to send CancelItem: {e}"))
    }

    /// Send shutdown signal to the child.
    pub fn send_shutdown(&mut self) -> Result<(), String> {
        write_message(&mut self.stdin_writer, &SubprocessCommand::Shutdown)
            .map_err(|e| format!("Failed to send Shutdown: {e}"))
    }

    /// Send Shutdown and wait up to `timeout` for the child to exit.
    /// Force-kills if unresponsive. Returns true iff graceful exit succeeded.
    ///
    /// Use a long timeout (e.g. 5s) for end-of-batch cleanup — the child may
    /// be flushing model caches. Use a short timeout (e.g. 1s) for Drop paths
    /// that must not stall the UI thread.
    pub fn shutdown_with_timeout(&mut self, timeout: std::time::Duration) -> bool {
        // Ignore send errors — child may already be dead, we just need to wait/kill.
        let _ = write_message(&mut self.stdin_writer, &SubprocessCommand::Shutdown);
        let deadline = std::time::Instant::now() + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => return true,
                _ if std::time::Instant::now() > deadline => {
                    self.kill();
                    return false;
                }
                _ => std::thread::sleep(std::time::Duration::from_millis(50)),
            }
        }
    }

    /// Non-blocking poll for events from the subprocess.
    pub fn poll_events(&mut self) -> Vec<SubprocessEvent> {
        let mut events = Vec::new();
        while let Some(evt) = self.next_event_with_state(None) {
            events.push(evt);
        }
        events
    }

    /// Block up to `timeout` for the first event, then drain anything
    /// else already enqueued. Wakes immediately on event arrival.
    pub fn poll_events_blocking(&mut self, timeout: std::time::Duration) -> Vec<SubprocessEvent> {
        let mut events = Vec::new();
        if let Some(evt) = self.next_event_with_state(Some(timeout)) {
            events.push(evt);
            while let Some(evt) = self.next_event_with_state(None) {
                events.push(evt);
            }
        }
        events
    }

    /// Pull at most one event from the channel, applying state updates
    /// (in-flight removal, RSS pause/resume). `None` timeout = `try_recv`,
    /// `Some(t)` = `recv_timeout`. Disconnected and recv-error both
    /// collapse to `None`; caller checks `is_alive`.
    fn next_event_with_state(&mut self, timeout: Option<std::time::Duration>) -> Option<SubprocessEvent> {
        let raw = match timeout {
            None => self.event_rx.try_recv().ok(),
            Some(t) => self.event_rx.recv_timeout(t).ok(),
        };
        let evt = match raw? {
            ReaderEvent::Event(e) => e,
            ReaderEvent::Disconnected => return None,
        };
        apply_event_state(
            &mut self.in_flight,
            &mut self.last_rss,
            &mut self.rss_paused,
            self.rss_limit,
            self.rss_resume,
            &evt,
        );
        Some(evt)
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
        // 1s is aggressive but necessary — Drop runs on the UI thread during
        // app shutdown / panic unwinding and must not stall.
        let _ = self.shutdown_with_timeout(std::time::Duration::from_secs(1));
    }
}

/// Apply a single subprocess event to the manager's bookkeeping state.
///
/// **Variant-exhaustive contract** (Phase 21-01 boundary test pins this):
/// every Done/Error variant must decrement `in_flight`; mid-flight Progress
/// variants and one-off Ready/Finished/InitError/RssUpdate must NOT.
///
/// This is split out of `next_event_with_state` so the contract can be
/// tested without spawning a real child process. The B1 regression
/// (InpaintDone/InpaintError leaking from the `_ => {}` arm) shipped because
/// the bookkeeping was inlined and untested per-variant.
fn apply_event_state(
    in_flight: &mut HashSet<u64>,
    last_rss: &mut u64,
    rss_paused: &mut bool,
    rss_limit: u64,
    rss_resume: u64,
    evt: &SubprocessEvent,
) {
    match evt {
        // Terminal events: the item is no longer being worked on. Every
        // Done/Error variant must land here. `InpaintProgress` is NOT
        // terminal — it's a mid-stroke tick during a long SD inpaint.
        SubprocessEvent::ImageDone { item_id, .. }
        | SubprocessEvent::ImageError { item_id, .. }
        | SubprocessEvent::InpaintDone { item_id, .. }
        | SubprocessEvent::InpaintError { item_id, .. } => {
            in_flight.remove(item_id);
        }
        // RSS hysteresis: pause when above limit, resume only when below
        // the (lower) resume threshold. Sticky in the band between.
        SubprocessEvent::RssUpdate { rss_bytes } => {
            *last_rss = *rss_bytes;
            if *rss_bytes > rss_limit {
                *rss_paused = true;
            } else if *rss_bytes < rss_resume {
                *rss_paused = false;
            }
        }
        // Non-terminal / one-off variants — no state change here.
        SubprocessEvent::Ready { .. }
        | SubprocessEvent::Progress { .. }
        | SubprocessEvent::InpaintProgress { .. }
        | SubprocessEvent::Finished
        | SubprocessEvent::InitError { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prunr_core::ProgressStage;
    use std::path::PathBuf;

    fn fresh_state() -> (HashSet<u64>, u64, bool) {
        let in_flight: HashSet<u64> = (0..10).collect();
        (in_flight, 0u64, false)
    }

    fn step(state: (&mut HashSet<u64>, &mut u64, &mut bool), evt: &SubprocessEvent) {
        apply_event_state(state.0, state.1, state.2, 1_000_000_000, 800_000_000, evt);
    }

    /// Phase 21-01 / B1 fix: every terminal Done/Error variant must
    /// decrement `in_flight`. Caught the regression where adding
    /// `InpaintDone`/`InpaintError` IPC variants in the SD subprocess
    /// refactor left them falling through `_ => {}` — the SD bundle
    /// stayed resident forever after the first stroke because the idle
    /// eviction gate keyed off `in_flight_items().is_empty()`.
    #[test]
    fn apply_event_state_decrements_on_every_terminal_variant() {
        let (mut in_flight, mut last_rss, mut rss_paused) = fresh_state();

        let terminal = [
            SubprocessEvent::ImageDone {
                item_id: 1,
                result_path: PathBuf::from("/tmp/x"),
                width: 100,
                height: 100,
                active_provider: "CPU".into(),
                tensor_cache_path: None,
                tensor_cache_height: None,
                tensor_cache_width: None,
                edge_cache_path: None,
                edge_cache_height: None,
                edge_cache_width: None,
            },
            SubprocessEvent::ImageError {
                item_id: 2,
                error: "decode failed".into(),
            },
            SubprocessEvent::InpaintDone {
                item_id: 3,
                rgba_path: PathBuf::from("/tmp/y"),
                width: 100,
                height: 100,
            },
            SubprocessEvent::InpaintError {
                item_id: 4,
                error: "lama session failed".into(),
            },
        ];

        for e in &terminal {
            step((&mut in_flight, &mut last_rss, &mut rss_paused), e);
        }

        assert!(!in_flight.contains(&1), "ImageDone must decrement in_flight");
        assert!(!in_flight.contains(&2), "ImageError must decrement in_flight");
        assert!(!in_flight.contains(&3), "InpaintDone must decrement in_flight (B1 regression)");
        assert!(!in_flight.contains(&4), "InpaintError must decrement in_flight (B1 regression)");
        // Items not referenced by any event are still in flight
        for id in 5..10 {
            assert!(in_flight.contains(&id));
        }
    }

    /// Mid-flight Progress variants signal "still working" — they MUST NOT
    /// decrement `in_flight`, otherwise the idle-eviction gate would fire
    /// while a long SD stroke is still computing.
    #[test]
    fn apply_event_state_does_not_decrement_on_progress_variants() {
        let (mut in_flight, mut last_rss, mut rss_paused) = fresh_state();

        step(
            (&mut in_flight, &mut last_rss, &mut rss_paused),
            &SubprocessEvent::Progress {
                item_id: 1,
                stage: ProgressStage::Infer,
                pct: 0.5,
            },
        );
        step(
            (&mut in_flight, &mut last_rss, &mut rss_paused),
            &SubprocessEvent::InpaintProgress {
                item_id: 2,
                current: 5,
                total: 20,
            },
        );

        assert!(in_flight.contains(&1), "Progress is mid-flight — in_flight must NOT change");
        assert!(in_flight.contains(&2), "InpaintProgress is mid-flight — in_flight must NOT change");
    }

    /// One-off lifecycle variants don't carry an `item_id` (or shouldn't
    /// affect bookkeeping when they do). They MUST NOT touch `in_flight`.
    #[test]
    fn apply_event_state_ignores_lifecycle_variants() {
        let (mut in_flight, mut last_rss, mut rss_paused) = fresh_state();
        let original = in_flight.clone();

        for evt in [
            SubprocessEvent::Ready {
                active_provider: "CPU".into(),
            },
            SubprocessEvent::Finished,
            SubprocessEvent::InitError {
                error: "no session".into(),
            },
        ] {
            step((&mut in_flight, &mut last_rss, &mut rss_paused), &evt);
        }
        assert_eq!(in_flight, original);
    }

    /// RSS hysteresis: pause when ABOVE limit, resume only when BELOW resume.
    /// Sticky in the band between resume and limit (the contract that
    /// prevented oscillation under steady ~limit pressure).
    #[test]
    fn apply_event_state_rss_hysteresis() {
        let (mut in_flight, mut last_rss, mut rss_paused) = fresh_state();

        // Above limit → pause
        step(
            (&mut in_flight, &mut last_rss, &mut rss_paused),
            &SubprocessEvent::RssUpdate {
                rss_bytes: 1_500_000_000,
            },
        );
        assert!(rss_paused);
        assert_eq!(last_rss, 1_500_000_000);

        // In hysteresis band (between resume and limit) → still paused
        step(
            (&mut in_flight, &mut last_rss, &mut rss_paused),
            &SubprocessEvent::RssUpdate {
                rss_bytes: 900_000_000,
            },
        );
        assert!(rss_paused, "RSS in hysteresis band must remain paused");

        // Below resume → unpause
        step(
            (&mut in_flight, &mut last_rss, &mut rss_paused),
            &SubprocessEvent::RssUpdate {
                rss_bytes: 700_000_000,
            },
        );
        assert!(!rss_paused);
        assert_eq!(last_rss, 700_000_000);
    }
}
