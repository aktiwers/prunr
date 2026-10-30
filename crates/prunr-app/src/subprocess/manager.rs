//! Parent-side subprocess manager.
//!
//! Spawns `prunr --worker`, manages IPC over stdin/stdout, tracks in-flight
//! items, and monitors RSS for admission throttling.

use std::collections::HashSet;
use std::io::{BufReader, BufWriter, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use prunr_core::{ModelKind, MaskSettings, EdgeSettings};
use crate::gui::settings::LineMode;
use super::protocol::*;
use super::ipc::{write_message, read_message};

/// Borrowed `(data, height, width)` triple so subprocess `send_*` helpers
/// don't carry three parallel args.
pub struct TensorView<'a> {
    pub data: &'a [f32],
    pub height: u32,
    pub width: u32,
}

/// The photo a re-run finishes: its file's bytes, or the pixels already
/// decoded.
pub enum Photo {
    Encoded(std::sync::Arc<Vec<u8>>),
    Decoded(std::sync::Arc<image::RgbaImage>),
}

/// Manages a subprocess worker: spawning, IPC, in-flight tracking, RSS throttling.
pub struct SubprocessManager {
    child: Child,
    stdin_writer: BufWriter<std::process::ChildStdin>,
    /// Channel from the reader thread (reads child stdout → events).
    event_rx: mpsc::Receiver<ReaderEvent>,
    /// When the reader thread was seen to end, its last event delivered.
    reader_gone_at: Option<Instant>,
    in_flight: HashSet<u64>,
    /// RSS limit: pause admission when child RSS exceeds this.
    rss_limit: u64,
    /// Resume threshold (hysteresis): resume when RSS drops below this.
    rss_resume: u64,
    rss_paused: bool,
    /// Counter into `RSS_LIMIT_REFRESH_EVERY`: re-query system memory
    /// every Nth `RssUpdate` so a sibling subprocess loading a 5 GB
    /// SD bundle after us doesn't leave our pause threshold stale.
    rss_update_count: u32,
}

/// Children emit one `RssUpdate` per finished job, so system memory is
/// re-queried every 16th job (a few ms each).
const RSS_LIMIT_REFRESH_EVERY: u32 = 16;

/// A child closes its output a moment before it can be waited on, so its
/// exit is awaited this long before a live one with no reader is killed.
const READER_GONE_GRACE: Duration = Duration::from_secs(1);

enum ReaderEvent {
    Event(SubprocessEvent),
    /// Reader thread exited (child stdout closed or error).
    Disconnected,
}

/// The lines a worker draws: the mode, their settings, how much they
/// fade with distance (0-100), and the scales it runs the detector at.
#[derive(Debug, Clone, Copy)]
pub struct WorkerLines {
    pub mode: LineMode,
    pub edge: EdgeSettings,
    pub fade_pct: u8,
    pub scales: InferredScales,
}

impl WorkerLines {
    /// The app's lines: it fades the lines it shows from its own depth
    /// map, and keeps every scale of the image on screen so a scale
    /// switch there needs no new run.
    pub fn app(mode: LineMode, edge: EdgeSettings) -> Self {
        Self { mode, edge, fade_pct: 0, scales: InferredScales::DrawnFirst }
    }
}

/// Why a worker died. What the user is told depends on it; `Display` is
/// the detail for the log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Crash {
    /// The system killed it (SIGKILL), which on a desktop means memory.
    OutOfMemory,
    /// It is alive but can no longer be heard: it sent nothing for too
    /// long, or what it sent could not be read.
    Unresponsive,
    /// A crash or exit, described.
    Other(String),
}

impl Crash {
    /// How a process that ended with `status` stopped.
    fn of(status: std::process::ExitStatus) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            if let Some(signal) = status.signal() {
                return match signal {
                    9 => Crash::OutOfMemory,
                    11 => Crash::Other("crashed (segmentation fault)".into()),
                    _ => Crash::Other(format!("killed by signal {signal}")),
                };
            }
        }
        Crash::Other(status.code().map_or_else(|| "terminated unexpectedly".into(), |code| format!("exited with code {code}")))
    }
}

impl std::fmt::Display for Crash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfMemory => f.write_str("killed by the system (out of memory)"),
            Self::Unresponsive => f.write_str("stopped responding"),
            Self::Other(what) => f.write_str(what),
        }
    }
}

impl SubprocessManager {
    /// Spawn a `prunr --worker` subprocess and send the Init command.
    /// Blocks until the child sends `Ready` or `InitError`.
    #[tracing::instrument(
        skip_all,
        fields(model = ?model, jobs, cpu_only, line_mode = ?lines.mode),
    )]
    pub fn spawn(
        model: ModelKind,
        jobs: usize,
        mask: MaskSettings,
        cpu_only: bool,
        lines: WorkerLines,
    ) -> Result<(Self, String), String> {
        Self::spawn_inner(model, jobs, mask, cpu_only, lines, false)
    }

    /// Spawn a SD-inpaint-only subprocess, so an OOM during the bundle build
    /// kills it rather than the GUI. The seg fields on `Init` are placeholders.
    pub fn spawn_inpaint_only() -> Result<(Self, String), String> {
        Self::spawn_inner(
            ModelKind::Silueta, 1,
            MaskSettings::default(), false,
            WorkerLines::app(LineMode::Off, EdgeSettings::default()),
            true,
        )
    }

    fn spawn_inner(
        model: ModelKind,
        jobs: usize,
        mask: MaskSettings,
        cpu_only: bool,
        lines: WorkerLines,
        inpaint_only: bool,
    ) -> Result<(Self, String), String> {
        let WorkerLines { mode: line_mode, edge, fade_pct: line_fade_pct, scales: edge_scales } = lines;
        // Spawn must NOT touch the temp dir: the CLI and the inpaint bridge write
        // inputs before spawning. `ipc_temp_dir()`'s init-time sweep owns cleanup.
        let exe = worker_exe()?;

        let mut command = Command::new(&exe);
        prunr_core::ort_runtime::same_runtime_for(&mut command);
        let mut child = command
            .arg("--worker")
            .args(["--worker-of", WORKER_VERSION])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // child errors visible in parent console
            .spawn()
            .map_err(|e| format!("Failed to spawn worker: {e}"))?;
        let (mut stdin_writer, event_rx) = attach(&mut child)?;

        let child = send_init(child, &mut stdin_writer, &SubprocessCommand::Init {
            model, jobs, mask, cpu_only, line_mode, edge,
            ipc_dir: ipc_temp_dir().to_path_buf(),
            inpaint_only,
            line_fade_pct,
            edge_scales,
        })?;

        // Seg/edge workers build engine pools at Init (a CoreML first-run compile
        // alone can take 2–5 min), so 300 s; inpaint-only builds lazily, so 30 s
        // bounds a pre-Ready hang such as a wedged OpenVINO probe.
        let init_timeout = std::time::Duration::from_secs(if inpaint_only { 30 } else { 300 });
        let active_provider = match event_rx.recv_timeout(init_timeout) {
            Ok(ReaderEvent::Event(SubprocessEvent::Ready { active_provider })) => {
                active_provider
            }
            Ok(ReaderEvent::Event(SubprocessEvent::InitError { error })) => {
                stop(child);
                return Err(format!("Worker init failed: {error}"));
            }
            Ok(ReaderEvent::Disconnected) => return Err(init_ended(child, "Worker exited during initialization")),
            Ok(ReaderEvent::Event(other)) => {
                stop(child);
                return Err(format!("Unexpected event during init: {other:?}"));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                stop(child);
                return Err(format!(
                    "Worker init timed out ({:?}). \
                     If this happened on first SD eraser use, the \
                     OpenVINO availability probe may be wedged — check \
                     the worker log for 'detect_active_provider' tracing.",
                    init_timeout,
                ));
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                stop(child);
                return Err("Worker reader thread died during init".to_string());
            }
        };

        Ok((Self::ready(child, stdin_writer, event_rx), active_provider))
    }

    fn ready(child: Child, stdin_writer: BufWriter<std::process::ChildStdin>, event_rx: mpsc::Receiver<ReaderEvent>) -> Self {
        let (rss_limit, rss_resume) = current_rss_thresholds();
        Self {
            child,
            stdin_writer,
            event_rx,
            reader_gone_at: None,
            in_flight: HashSet::new(),
            rss_limit,
            rss_resume,
            rss_paused: false,
            rss_update_count: 0,
        }
    }

    /// A manager around any piped child, taken as ready: a stand-in worker
    /// a test can let die.
    #[cfg(test)]
    pub(crate) fn for_test(mut child: Child) -> Self {
        let (stdin_writer, event_rx) = attach(&mut child).expect("a piped child");
        Self::ready(child, stdin_writer, event_rx)
    }

    /// Send an image for processing. Writes bytes to a temp file first.
    pub fn send_image(
        &mut self,
        item_id: u64,
        image_bytes: &[u8],
        chain_input: Option<(&image::RgbaImage, u32, u32)>,
    ) -> Result<(), String> {
        let image_path = super::protocol::IpcKind::Input.path_for(ipc_temp_dir(), item_id);
        std::fs::write(&image_path, image_bytes)
            .map_err(|e| format!("Failed to write input temp file: {e}"))?;

        // Propagate write errors: a missing or partial chain file made the worker
        // silently process the original image instead.
        let chain = chain_input
            .map(|(rgba, w, h)| -> Result<ChainInput, String> {
                let path = super::protocol::IpcKind::Chain.path_for(ipc_temp_dir(), item_id);
                std::fs::write(&path, rgba.as_raw())
                    .map_err(|e| format!("Failed to write chain temp file: {e}"))?;
                Ok(ChainInput { path, width: w, height: h })
            })
            .transpose()?;
        self.send_process_image(item_id, image_path, chain)
    }

    /// Send an image for processing directly from a file path (no copy).
    /// The subprocess reads the file; caller is responsible for cleanup.
    pub fn send_image_path(&mut self, item_id: u64, image_path: std::path::PathBuf) -> Result<(), String> {
        self.send_process_image(item_id, image_path, None)
    }

    fn send_process_image(&mut self, item_id: u64, image_path: std::path::PathBuf, chain_input: Option<ChainInput>) -> Result<(), String> {
        self.send_cmd(&SubprocessCommand::ProcessImage { item_id, image_path, chain_input })
            .map_err(|e| format!("Failed to send ProcessImage: {e}"))?;
        self.in_flight.insert(item_id);
        Ok(())
    }

    /// Write a command and flush. Parent commands must be visible to the child
    /// immediately — a non-flushed BufWriter would stall blocking `read_message`
    /// on the child side until the buffer fills.
    fn send_cmd(&mut self, cmd: &SubprocessCommand) -> std::io::Result<()> {
        write_message(&mut self.stdin_writer, cmd)?;
        self.stdin_writer.flush()
    }

    /// Write tensor + image to temp files, then dispatch the caller-built command.
    fn send_with_tensor_and_image(
        &mut self,
        item_id: u64,
        tensor: &TensorView<'_>,
        tensor_kind: super::protocol::IpcKind,
        photo: &Photo,
        image_kind: super::protocol::IpcKind,
        build_cmd: impl FnOnce(std::path::PathBuf, PhotoFile) -> SubprocessCommand,
    ) -> Result<(), String> {
        let tensor_path = tensor_kind.path_for(ipc_temp_dir(), item_id);
        std::fs::write(&tensor_path, super::ipc::f32s_as_le_bytes(tensor.data))
            .map_err(|e| format!("Failed to write tensor temp file: {e}"))?;
        let image_path = image_kind.path_for(ipc_temp_dir(), item_id);
        let original = match photo {
            Photo::Encoded(bytes) => std::fs::write(&image_path, bytes.as_slice()).map(|()| PhotoFile::Encoded(image_path)),
            Photo::Decoded(rgba) => super::ipc::write_image(&image_path, rgba).map(|()| PhotoFile::Decoded(image_path)),
        }
        .map_err(|e| format!("Failed to write image temp file: {e}"))?;

        self.send_cmd(&build_cmd(tensor_path, original))
            .map_err(|e| format!("Failed to send command: {e}"))?;

        self.in_flight.insert(item_id);
        Ok(())
    }

    /// Send an AddEdgeInference command (seg cached, run TEED on masked).
    /// Writes the seg tensor + original image bytes to temp files. The worker
    /// reads the seg tensor without deleting it and hands the path back as
    /// `tensor_cache` on `ImageDone`, so the parent's reader takes
    /// ownership — no extra copy round-trip.
    pub fn send_add_edge_inference(
        &mut self,
        item_id: u64,
        tensor: &TensorView<'_>,
        model: prunr_core::ModelKind,
        photo: &Photo,
        mask: prunr_core::MaskSettings,
    ) -> Result<(), String> {
        self.send_with_tensor_and_image(
            item_id,
            tensor,
            super::protocol::IpcKind::Seg,
            photo,
            super::protocol::IpcKind::Input,
            |seg_tensor_path, original| SubprocessCommand::AddEdgeInference {
                item_id,
                original,
                seg_tensor_path,
                seg_tensor_height: tensor.height,
                seg_tensor_width: tensor.width,
                model,
                mask,
            },
        )
    }

    /// Send a Tier 2 re-postprocess command (skip inference, reuse cached tensor).
    pub fn send_repostprocess(
        &mut self,
        item_id: u64,
        tensor: &TensorView<'_>,
        model: prunr_core::ModelKind,
        photo: &Photo,
        mask: prunr_core::MaskSettings,
    ) -> Result<(), String> {
        self.send_with_tensor_and_image(
            item_id,
            tensor,
            super::protocol::IpcKind::Tensor,
            photo,
            super::protocol::IpcKind::Orig,
            |tensor_path, original| SubprocessCommand::RePostProcess {
                item_id,
                tensor_path,
                tensor_height: tensor.height,
                tensor_width: tensor.width,
                model,
                original,
                mask,
            },
        )
    }

    /// Send an `Inpaint` command. Caller writes image + mask to PNG
    /// files (worker decodes); this method just forwards the job.
    /// Returns immediately — completion comes back via `InpaintDone` /
    /// `InpaintError` events through `poll_events`.
    pub fn send_inpaint(&mut self, job: InpaintJob) -> Result<(), String> {
        let item_id = job.item_id;
        self.send_cmd(&SubprocessCommand::Inpaint(job)).map_err(|e| format!("Failed to send Inpaint: {e}"))?;
        self.in_flight.insert(item_id);
        Ok(())
    }

    /// New mask, Sketch mode and line settings for the running worker,
    /// the image that shows a first look (`None`: no first look, and the
    /// worker lets its first-look model go) and the image on screen.
    pub fn send_configure(&mut self, mask: MaskSettings, line_mode: LineMode, edge: EdgeSettings, first_look: Option<u64>, on_screen: Option<u64>) -> Result<(), String> {
        self.send_cmd(&SubprocessCommand::Configure { mask, line_mode, edge, first_look, on_screen })
            .map_err(|e| format!("Failed to send Configure: {e}"))
    }

    /// Cancel one item by id — worker drops it at the next dispatch check
    /// and emits `ImageError` with `CANCELLED_ERR_MSG`.
    pub fn send_cancel_item(&mut self, item_id: u64) -> Result<(), String> {
        self.send_cmd(&SubprocessCommand::CancelItem { item_id })
            .map_err(|e| format!("Failed to send CancelItem: {e}"))
    }

    /// Send Shutdown and wait up to `timeout` for the child to exit.
    /// Force-kills if unresponsive. Returns true iff graceful exit succeeded.
    ///
    /// Use a long timeout (e.g. 5s) for end-of-batch cleanup — the child may
    /// be flushing model caches. Use a short timeout (e.g. 1s) for Drop paths
    /// that must not stall the UI thread.
    pub fn shutdown_with_timeout(&mut self, timeout: std::time::Duration) -> bool {
        // Ignore send errors — child may already be dead, we just need to wait/kill.
        let _ = self.send_cmd(&SubprocessCommand::Shutdown);
        let start = std::time::Instant::now();
        let deadline = start + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => return true,
                _ if std::time::Instant::now() > deadline => {
                    let elapsed_ms = start.elapsed().as_millis();
                    tracing::warn!(
                        elapsed_ms,
                        "subprocess shutdown timed out — force-killing child \
                         (CoreML first-run compile or model flush still in progress)"
                    );
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
    /// `Some(t)` = `recv_timeout`. A reader that has ended is noted for
    /// `died`.
    fn next_event_with_state(&mut self, timeout: Option<std::time::Duration>) -> Option<SubprocessEvent> {
        let raw = match timeout {
            None => self.event_rx.try_recv().map_err(|e| e == mpsc::TryRecvError::Disconnected),
            Some(t) => self.event_rx.recv_timeout(t).map_err(|e| e == mpsc::RecvTimeoutError::Disconnected),
        };
        let evt = match raw {
            Ok(ReaderEvent::Event(e)) => e,
            Ok(ReaderEvent::Disconnected) | Err(true) => {
                self.reader_gone_at.get_or_insert_with(Instant::now);
                return None;
            }
            Err(false) => return None,
        };
        if matches!(evt, SubprocessEvent::RssUpdate { .. }) {
            self.rss_update_count = self.rss_update_count.wrapping_add(1);
            if self.rss_update_count.is_multiple_of(RSS_LIMIT_REFRESH_EVERY) {
                let (limit, resume) = current_rss_thresholds();
                self.rss_limit = limit;
                self.rss_resume = resume;
            }
        }
        apply_event_state(
            &mut self.in_flight,
            &mut self.rss_paused,
            self.rss_limit,
            self.rss_resume,
            &evt,
        );
        Some(evt)
    }

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

    /// Why the subprocess died, once it has. One still running whose output
    /// can no longer be read (a frame too large, garbage) is as good as
    /// dead: it is killed and reported unresponsive.
    pub fn died(&mut self) -> Option<Crash> {
        if !self.is_alive() {
            return Some(self.crash());
        }
        let gone = self.reader_gone_at?;
        (gone.elapsed() >= READER_GONE_GRACE).then(|| {
            tracing::warn!("worker output unreadable; killed");
            self.kill();
            Crash::Unresponsive
        })
    }

    fn crash(&mut self) -> Crash {
        match self.child.try_wait() {
            Ok(Some(status)) => Crash::of(status),
            _ => Crash::Unresponsive,
        }
    }

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

/// The binary to run a worker from: this process's own. On Linux an update
/// can replace the file while the app runs (the path then reads
/// "… (deleted)"); /proc/self/exe still reaches the binary running.
pub(crate) fn worker_exe() -> Result<std::path::PathBuf, String> {
    #[cfg(target_os = "linux")]
    {
        let own = std::path::PathBuf::from("/proc/self/exe");
        if own.exists() {
            return Ok(own);
        }
    }
    std::env::current_exe().map_err(|e| format!("Failed to get current exe: {e}"))
}

/// Why a worker that closed its output during Init stopped, once it has:
/// `otherwise` unless it was one of another version.
fn init_ended(mut child: Child, otherwise: &str) -> String {
    let deadline = Instant::now() + Duration::from_secs(2);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            _ => break None,
        }
    };
    if status.is_none() {
        stop(child);
    }
    if status.and_then(|s| s.code()) == Some(WORKER_OF_ANOTHER_VERSION) {
        "Prunr was updated while it was open. Restart it to keep processing.".to_string()
    } else {
        otherwise.to_string()
    }
}

/// Send `init` to a worker just spawned, flushed at once: its blocking
/// `read_message` waits on it. One that already stopped (of another
/// version) closed the pipe, so its exit says why, not the write.
fn send_init(child: Child, stdin: &mut BufWriter<std::process::ChildStdin>, init: &SubprocessCommand) -> Result<Child, String> {
    match write_message(stdin, init).and_then(|()| stdin.flush()) {
        Ok(()) => Ok(child),
        Err(e) => Err(init_ended(child, &format!("Failed to send Init: {e}"))),
    }
}

/// Kill a worker and reap it, so no zombie stays until the app exits.
fn stop(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Take `child`'s pipes: its stdin to write commands to, and its stdout
/// read on a thread into events, ending with `Disconnected`.
fn attach(child: &mut Child) -> Result<(BufWriter<std::process::ChildStdin>, mpsc::Receiver<ReaderEvent>), String> {
    let child_stdin = child.stdin.take().ok_or("Failed to capture worker stdin")?;
    let child_stdout = child.stdout.take().ok_or("Failed to capture worker stdout")?;
    let (event_tx, event_rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("subprocess-reader".into())
        .spawn(move || {
            let mut reader = BufReader::new(child_stdout);
            // A closed stdout (clean exit or crash) or garbage ends it.
            while let Ok(Some(evt)) = read_message::<_, SubprocessEvent>(&mut reader) {
                if event_tx.send(ReaderEvent::Event(evt)).is_err() {
                    return; // parent dropped receiver
                }
            }
            let _ = event_tx.send(ReaderEvent::Disconnected);
        })
        .map_err(|e| format!("Failed to spawn reader thread: {e}"))?;
    Ok((BufWriter::new(child_stdin), event_rx))
}

/// (rss_limit, rss_resume) from the system's current available memory.
fn current_rss_thresholds() -> (u64, u64) {
    let available = crate::hardware::available_ram_bytes_now();
    let limit = (available as f64 * 0.80) as u64;
    let resume = (limit as f64 * 0.70) as u64;
    (limit, resume)
}

fn apply_event_state(
    in_flight: &mut HashSet<u64>,
    rss_paused: &mut bool,
    rss_limit: u64,
    rss_resume: u64,
    evt: &SubprocessEvent,
) {
    match evt {
        // Terminal events: the item is no longer being worked on. Every
        // Done/Error variant must land here; progress reports are not.
        SubprocessEvent::ImageDone { item_id, .. }
        | SubprocessEvent::ImageError { item_id, .. }
        | SubprocessEvent::InpaintDone { item_id, .. }
        | SubprocessEvent::InpaintError { item_id, .. } => {
            in_flight.remove(item_id);
        }
        // RSS hysteresis: pause when above limit, resume only when below
        // the (lower) resume threshold. Sticky in the band between.
        SubprocessEvent::RssUpdate { rss_bytes } => {
            if *rss_bytes > rss_limit {
                *rss_paused = true;
            } else if *rss_bytes < rss_resume {
                *rss_paused = false;
            }
        }
        // Non-terminal / one-off variants — no state change here.
        SubprocessEvent::Ready { .. }
        | SubprocessEvent::Report { .. }
        | SubprocessEvent::SpareEdgeScales { .. }
        | SubprocessEvent::Finished
        | SubprocessEvent::InitError { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A worker that stopped as soon as it began says why: one of another
    /// version (the app was updated while open) asks for a restart.
    #[cfg(unix)]
    #[test]
    fn a_worker_that_stops_at_init_says_why() {
        let exit = |code: i32| Command::new("sh").args(["-c", &format!("exit {code}")]).spawn().unwrap();
        assert!(init_ended(exit(WORKER_OF_ANOTHER_VERSION), "").starts_with("Prunr was updated while it was open"));
        assert_eq!(init_ended(exit(3), "Worker exited during initialization"), "Worker exited during initialization");
    }

    /// One that stopped before its Init was even written closed the pipe:
    /// the write fails, and the reason is still its exit.
    #[cfg(unix)]
    #[test]
    fn a_worker_gone_before_its_init_says_why() {
        let mut child = Command::new("sh").args(["-c", &format!("exit {WORKER_OF_ANOTHER_VERSION}")])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        let (mut stdin, _events) = attach(&mut child).unwrap();
        while child.try_wait().unwrap().is_none() {
            std::thread::sleep(Duration::from_millis(5));
        }
        // A test spawning beside this one holds a copy of the pipe from its
        // fork to its exec; once a write finds the pipe closed, it stays so.
        while std::io::Write::write(stdin.get_mut(), &[0]).is_ok() {
            std::thread::sleep(Duration::from_millis(5));
        }
        let why = send_init(child, &mut stdin, &SubprocessCommand::Shutdown).expect_err("the pipe is closed");
        assert!(why.starts_with("Prunr was updated while it was open"), "{why}");
    }

    /// The worker runs from this process's own binary, which on Linux an
    /// update may have replaced on disk.
    #[test]
    fn the_worker_runs_from_this_processs_own_binary() {
        let exe = worker_exe().unwrap();
        #[cfg(target_os = "linux")]
        assert_eq!(exe, PathBuf::from("/proc/self/exe"));
        assert!(exe.exists());
    }

    fn fresh_state() -> (HashSet<u64>, bool) {
        let in_flight: HashSet<u64> = (0..10).collect();
        (in_flight, false)
    }

    fn step(state: (&mut HashSet<u64>, &mut bool), evt: &SubprocessEvent) {
        apply_event_state(state.0, state.1, 1_000_000_000, 800_000_000, evt);
    }

    /// Every terminal Done/Error variant must decrement `in_flight`. A
    /// variant that falls through `_ => {}` leaves the SD bundle resident
    /// forever, because the idle eviction gate keys off
    /// `in_flight_items().is_empty()`.
    #[test]
    fn apply_event_state_decrements_on_every_terminal_variant() {
        let (mut in_flight, mut rss_paused) = fresh_state();

        let terminal = [
            SubprocessEvent::ImageDone {
                item_id: 1,
                result_path: PathBuf::from("/tmp/x"),
                width: 100,
                height: 100,
                active_provider: "CPU".into(),
                tensor_cache: None,
                edge_cache: None,
                spare_scales: None,
            },
            SubprocessEvent::ImageError {
                item_id: 2,
                error: "decode failed".into(),
            },
            SubprocessEvent::InpaintDone {
                item_id: 3,
                generation: 1,
                rgba_path: PathBuf::from("/tmp/y"),
                width: 100,
                height: 100,
                fill_scale: 1.0,
            },
            SubprocessEvent::InpaintError {
                item_id: 4,
                generation: 1,
                error: "lama session failed".into(),
            },
        ];

        for e in &terminal {
            step((&mut in_flight, &mut rss_paused), e);
        }

        assert!(!in_flight.contains(&1), "ImageDone must decrement in_flight");
        assert!(!in_flight.contains(&2), "ImageError must decrement in_flight");
        assert!(!in_flight.contains(&3), "InpaintDone must decrement in_flight");
        assert!(!in_flight.contains(&4), "InpaintError must decrement in_flight");
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
        let (mut in_flight, mut rss_paused) = fresh_state();

        step(
            (&mut in_flight, &mut rss_paused),
            &SubprocessEvent::Report { item_id: 1, update: prunr_core::ProgressUpdate::Step(prunr_core::Step::FindingSubject) },
        );

        assert!(in_flight.contains(&1), "Progress is mid-flight — in_flight must NOT change");
    }

    /// One-off lifecycle variants don't carry an `item_id` (or shouldn't
    /// affect bookkeeping when they do). They MUST NOT touch `in_flight`.
    #[test]
    fn apply_event_state_ignores_lifecycle_variants() {
        let (mut in_flight, mut rss_paused) = fresh_state();
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
            step((&mut in_flight, &mut rss_paused), &evt);
        }
        assert_eq!(in_flight, original);
    }

    /// RSS hysteresis: pause when ABOVE limit, resume only when BELOW resume.
    /// Sticky in the band between resume and limit (the contract that
    /// prevented oscillation under steady ~limit pressure).
    #[test]
    fn apply_event_state_rss_hysteresis() {
        let (mut in_flight, mut rss_paused) = fresh_state();

        // Above limit → pause
        step(
            (&mut in_flight, &mut rss_paused),
            &SubprocessEvent::RssUpdate {
                rss_bytes: 1_500_000_000,
            },
        );
        assert!(rss_paused);

        // In hysteresis band (between resume and limit) → still paused
        step(
            (&mut in_flight, &mut rss_paused),
            &SubprocessEvent::RssUpdate {
                rss_bytes: 900_000_000,
            },
        );
        assert!(rss_paused, "RSS in hysteresis band must remain paused");

        // Below resume → unpause
        step(
            (&mut in_flight, &mut rss_paused),
            &SubprocessEvent::RssUpdate {
                rss_bytes: 700_000_000,
            },
        );
        assert!(!rss_paused);
    }
}
