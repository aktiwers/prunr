//! Coordinator for on-demand model downloads.
//!
//! Concurrency cap: 1 active download at a time. Additional `start_download`
//! calls queue in submission order — keeps bandwidth + disk pressure
//! predictable, and a fresh stroke from the user can't stack 5 simultaneous
//! 800 MB downloads.
//!
//! HTTP + SHA + atomic-write logic lives in `download_to_file` so the
//! state machine here stays testable without a network.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use prunr_models::{descriptor, on_demand_dir, ModelId, ModelPart, ModelSource};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadState {
    Idle,
    Queued,
    InProgress { bytes_so_far: u64, total_bytes: u64 },
    Verifying,
    Failed { error: String, retryable: bool },
    Done,
}

#[derive(Debug)]
pub enum DownloadEvent {
    Progress { id: ModelId, bytes_so_far: u64, total: u64 },
    Verifying { id: ModelId },
    Complete { id: ModelId },
    Failed { id: ModelId, error: String, retryable: bool },
}

pub(crate) struct DownloadManager {
    states: HashMap<ModelId, DownloadState>,
    queue: VecDeque<ModelId>,
    cancel_flags: HashMap<ModelId, Arc<AtomicBool>>,
    progress_tx: mpsc::Sender<DownloadEvent>,
    progress_rx: mpsc::Receiver<DownloadEvent>,
    active: Option<ModelId>,
}

impl DownloadManager {
    pub(crate) fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            states: HashMap::new(),
            queue: VecDeque::new(),
            cancel_flags: HashMap::new(),
            progress_tx: tx,
            progress_rx: rx,
            active: None,
        }
    }

    /// Returns `Idle` for ids that have never been touched.
    pub(crate) fn state(&self, id: ModelId) -> DownloadState {
        self.states.get(&id).cloned().unwrap_or(DownloadState::Idle)
    }

    /// No-op when the model is already installed / in progress / queued.
    pub(crate) fn start_download(&mut self, id: ModelId) {
        if prunr_models::is_available(id) {
            self.states.insert(id, DownloadState::Done);
            return;
        }
        match self.state(id) {
            DownloadState::InProgress { .. }
            | DownloadState::Queued
            | DownloadState::Verifying => return,
            _ => {}
        }
        if self.active.is_some() {
            self.states.insert(id, DownloadState::Queued);
            self.queue.push_back(id);
        } else {
            self.kick_off(id);
        }
    }

    /// The download thread polls the flag every chunk and tears down the
    /// partial file on next poll.
    pub(crate) fn cancel_download(&mut self, id: ModelId) {
        if let Some(flag) = self.cancel_flags.get(&id) {
            flag.store(true, Ordering::Release);
        }
        self.queue.retain(|q| *q != id);
        if self.active != Some(id) {
            self.states.insert(id, DownloadState::Idle);
        }
    }

    /// Returns drained events so the UI can fan them out to toasts /
    /// repaints — the manager itself owns no UI state.
    pub(crate) fn pump(&mut self) -> Vec<DownloadEvent> {
        let mut out = Vec::new();
        while let Ok(event) = self.progress_rx.try_recv() {
            self.apply_event(&event);
            out.push(event);
        }
        // If nothing is active and there's a queued model, kick the next.
        if self.active.is_none() {
            if let Some(next) = self.queue.pop_front() {
                self.kick_off(next);
            }
        }
        out
    }

    fn apply_event(&mut self, event: &DownloadEvent) {
        match *event {
            DownloadEvent::Progress { id, bytes_so_far, total } => {
                self.states.insert(id, DownloadState::InProgress { bytes_so_far, total_bytes: total });
            }
            DownloadEvent::Verifying { id } => {
                self.states.insert(id, DownloadState::Verifying);
            }
            DownloadEvent::Complete { id } => {
                self.states.insert(id, DownloadState::Done);
                self.cancel_flags.remove(&id);
                if self.active == Some(id) {
                    self.active = None;
                }
            }
            DownloadEvent::Failed { id, ref error, retryable } => {
                self.states.insert(id, DownloadState::Failed { error: error.clone(), retryable });
                self.cancel_flags.remove(&id);
                if self.active == Some(id) {
                    self.active = None;
                }
            }
        }
    }

    fn kick_off(&mut self, id: ModelId) {
        let Some(desc) = descriptor(id) else {
            self.send_failure(id, format!("Unknown model id: {id:?}"), false);
            return;
        };
        match desc.source {
            ModelSource::Bundled => {
                self.send_failure(id, format!("{id:?} is bundled — no download path"), false);
            }
            ModelSource::OnDemand { url, sha256, filename, size_mb, .. } => {
                self.kick_off_single(id, url, sha256, filename, size_mb);
            }
            ModelSource::MultiPartOnDemand { subdir, parts, .. } => {
                self.kick_off_multi(id, subdir, parts);
            }
        }
    }

    fn kick_off_single(
        &mut self,
        id: ModelId,
        url: &'static str,
        sha256: &'static str,
        filename: &'static str,
        size_mb: u32,
    ) {
        let Some(dir) = on_demand_dir() else {
            self.send_failure(id, "Could not resolve user data directory".into(), false);
            return;
        };
        if let Err(e) = std::fs::create_dir_all(&dir) {
            self.send_failure(id, format!("Could not create models directory: {e}"), true);
            return;
        }
        let dest = dir.join(filename);
        let total = (size_mb as u64) * 1024 * 1024;
        let cancel = self.begin_active(id, total);
        let tx = self.progress_tx.clone();
        std::thread::spawn(move || {
            let on_progress = {
                let tx = tx.clone();
                move |bytes_so_far: u64, total: u64| {
                    let _ = tx.send(DownloadEvent::Progress { id, bytes_so_far, total });
                }
            };
            let on_verifying = {
                let tx = tx.clone();
                move || {
                    let _ = tx.send(DownloadEvent::Verifying { id });
                }
            };
            match download_to_file(url, &dest, sha256, cancel, &on_progress, &on_verifying) {
                Ok(()) => { let _ = tx.send(DownloadEvent::Complete { id }); }
                Err(e) => {
                    let _ = tx.send(DownloadEvent::Failed {
                        id, error: e.message, retryable: e.retryable,
                    });
                }
            }
        });
    }

    /// Multi-part dispatch: download each part sequentially under
    /// `<on_demand_dir>/<subdir>/`, aggregating progress so the UI's
    /// `InProgress { bytes_so_far, total_bytes }` tracks the bundle as a
    /// whole. Already-renamed parts are skipped on retry — gives the
    /// user effective resume across cancels without HTTP Range support.
    fn kick_off_multi(
        &mut self,
        id: ModelId,
        subdir: &'static str,
        parts: &'static [ModelPart],
    ) {
        let Some(root) = on_demand_dir() else {
            self.send_failure(id, "Could not resolve user data directory".into(), false);
            return;
        };
        let bundle_dir = root.join(subdir);
        if let Err(e) = std::fs::create_dir_all(&bundle_dir) {
            self.send_failure(id, format!("Could not create models directory: {e}"), true);
            return;
        }
        let total: u64 = parts.iter().map(|p| p.size_bytes).sum();
        let cancel = self.begin_active(id, total);
        let tx = self.progress_tx.clone();
        std::thread::spawn(move || {
            let mut bytes_done: u64 = 0;
            for part in parts {
                let dest = bundle_dir.join(part.filename);
                if dest.is_file() {
                    bytes_done += part.size_bytes;
                    let _ = tx.send(DownloadEvent::Progress { id, bytes_so_far: bytes_done, total });
                    continue;
                }
                let on_progress = {
                    let tx = tx.clone();
                    let so_far = bytes_done;
                    move |part_so_far: u64, _part_total: u64| {
                        let _ = tx.send(DownloadEvent::Progress {
                            id, bytes_so_far: so_far + part_so_far, total,
                        });
                    }
                };
                let on_verifying = {
                    let tx = tx.clone();
                    move || { let _ = tx.send(DownloadEvent::Verifying { id }); }
                };
                match download_to_file(part.url, &dest, part.sha256, cancel.clone(), &on_progress, &on_verifying) {
                    Ok(()) => { bytes_done += part.size_bytes; }
                    Err(e) => {
                        let _ = tx.send(DownloadEvent::Failed {
                            id,
                            error: format!("{}: {}", part.key, e.message),
                            retryable: e.retryable,
                        });
                        return;
                    }
                }
            }
            let _ = tx.send(DownloadEvent::Complete { id });
        });
    }

    /// Common state-machine prelude for any active download (single or
    /// multi-part): seed `InProgress`, register the cancel flag, mark
    /// `id` as the currently active download. Returns the cancel flag
    /// for the worker thread.
    fn begin_active(&mut self, id: ModelId, total_bytes: u64) -> Arc<AtomicBool> {
        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel_flags.insert(id, cancel.clone());
        self.active = Some(id);
        self.states.insert(id, DownloadState::InProgress { bytes_so_far: 0, total_bytes });
        cancel
    }

    fn send_failure(&mut self, id: ModelId, error: String, retryable: bool) {
        self.states.insert(id, DownloadState::Failed { error: error.clone(), retryable });
        let _ = self.progress_tx.send(DownloadEvent::Failed { id, error, retryable });
    }
}

#[derive(Debug)]
struct DownloadError {
    message: String,
    retryable: bool,
}

impl DownloadError {
    fn fatal(msg: impl Into<String>) -> Self {
        Self { message: msg.into(), retryable: false }
    }
    fn transient(msg: impl Into<String>) -> Self {
        Self { message: msg.into(), retryable: true }
    }
}

/// Download `url` to `dest`, verifying SHA256 against `expected_sha`.
/// Writes to a `<dest>.partial` sidecar; renames to `dest` only after
/// verification passes. Wraps `download_attempt` in a retry loop so
/// transient network errors (timeout, 5xx) get up to 3 tries with
/// exponential backoff. Fatal errors (404, SHA mismatch, disk full,
/// user cancel) fail fast without retry.
fn download_to_file(
    url: &str,
    dest: &Path,
    expected_sha: &str,
    cancel: Arc<AtomicBool>,
    on_progress: &dyn Fn(u64, u64),
    on_verifying: &dyn Fn(),
) -> Result<(), DownloadError> {
    retry_with_backoff(&cancel, 3, 500, || {
        download_attempt(url, dest, expected_sha, &cancel, on_progress, on_verifying)
    })
}

/// Run `attempt` up to `max_attempts` times. Returns immediately on
/// non-retryable errors. Between attempts, sleeps for an exponentially
/// growing duration starting at `base_ms` (250 → 500 → 1000 → 2000…),
/// polling `cancel` every 50 ms so a user-cancel during backoff is
/// honoured promptly.
fn retry_with_backoff<F>(
    cancel: &Arc<AtomicBool>,
    max_attempts: u32,
    base_ms: u64,
    mut attempt: F,
) -> Result<(), DownloadError>
where
    F: FnMut() -> Result<(), DownloadError>,
{
    let mut tries: u32 = 0;
    loop {
        match attempt() {
            Ok(()) => return Ok(()),
            Err(e) if !e.retryable => return Err(e),
            Err(e) if tries + 1 >= max_attempts => return Err(e),
            Err(e) => {
                tries += 1;
                let delay = std::time::Duration::from_millis(base_ms.saturating_mul(1 << tries.min(6)));
                tracing::warn!(tries, error = %e.message, ?delay, "transient download error — retrying");
                let deadline = std::time::Instant::now() + delay;
                while std::time::Instant::now() < deadline {
                    if cancel.load(Ordering::Acquire) {
                        return Err(DownloadError::fatal("Cancelled by user"));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }
        }
    }
}

fn download_attempt(
    url: &str,
    dest: &Path,
    expected_sha: &str,
    cancel: &Arc<AtomicBool>,
    on_progress: &dyn Fn(u64, u64),
    on_verifying: &dyn Fn(),
) -> Result<(), DownloadError> {
    let partial = partial_path(dest);

    // Tear down any leftover partial from a previous aborted attempt so
    // the writer starts fresh. (Resuming via Range requests would be
    // nicer but it's a v2 polish.)
    let _ = std::fs::remove_file(&partial);

    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("prunr/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| DownloadError::fatal(format!("HTTP client init failed: {e}")))?;

    let response = client.get(url).send()
        .map_err(|e| DownloadError::transient(format!("Network error: {e}")))?;

    let status = response.status();
    if !status.is_success() {
        return Err(if status.as_u16() >= 500 {
            DownloadError::transient(format!("Server returned {status}"))
        } else {
            DownloadError::fatal(format!("Download failed: {status}"))
        });
    }

    let total = response.content_length().unwrap_or(0);
    let mut response = response;
    let mut file = std::fs::File::create(&partial)
        .map_err(|e| DownloadError::fatal(format!("Could not create {}: {e}", partial.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut written: u64 = 0;
    loop {
        if cancel.load(Ordering::Acquire) {
            let _ = std::fs::remove_file(&partial);
            return Err(DownloadError::fatal("Cancelled by user"));
        }
        let n = std::io::Read::read(&mut response, &mut buf)
            .map_err(|e| DownloadError::transient(format!("Read error: {e}")))?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buf[..n])
            .map_err(|e| DownloadError::fatal(format!("Write error: {e}")))?;
        hasher.update(&buf[..n]);
        written += n as u64;
        on_progress(written, total);
    }
    std::io::Write::flush(&mut file)
        .map_err(|e| DownloadError::fatal(format!("Flush error: {e}")))?;
    drop(file);

    on_verifying();
    let actual_sha = hex::encode(hasher.finalize());
    if !actual_sha.eq_ignore_ascii_case(expected_sha) {
        let _ = std::fs::remove_file(&partial);
        return Err(DownloadError::fatal(format!(
            "Checksum mismatch — expected {expected_sha}, got {actual_sha}",
        )));
    }

    std::fs::rename(&partial, dest)
        .map_err(|e| DownloadError::fatal(format!("Could not finalize file: {e}")))?;
    Ok(())
}

fn partial_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".partial");
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> DownloadManager {
        DownloadManager::new()
    }

    #[test]
    fn idle_state_for_untouched_id() {
        let dm = fixture();
        assert_eq!(dm.state(ModelId::U2net), DownloadState::Idle);
        assert_eq!(dm.state(ModelId::LaMaFp32), DownloadState::Idle);
    }

    #[test]
    fn start_for_bundled_model_marks_done_immediately() {
        // Silueta is Bundled — no actual network call should happen.
        let mut dm = fixture();
        dm.start_download(ModelId::Silueta);
        assert_eq!(dm.state(ModelId::Silueta), DownloadState::Done);
        assert!(dm.active().is_none());
    }

    #[test]
    fn cancel_unstarted_id_is_safe_no_op() {
        let mut dm = fixture();
        dm.cancel_download(ModelId::U2net); // never started
        assert_eq!(dm.state(ModelId::U2net), DownloadState::Idle);
    }

    #[test]
    fn apply_event_progress_updates_bytes_so_far() {
        let mut dm = fixture();
        dm.apply_event(&DownloadEvent::Progress {
            id: ModelId::U2net, bytes_so_far: 1024, total: 4096,
        });
        match dm.state(ModelId::U2net) {
            DownloadState::InProgress { bytes_so_far, total_bytes } => {
                assert_eq!(bytes_so_far, 1024);
                assert_eq!(total_bytes, 4096);
            }
            other => panic!("expected InProgress, got {other:?}"),
        }
    }

    #[test]
    fn apply_event_complete_clears_active() {
        let mut dm = fixture();
        dm.active = Some(ModelId::U2net);
        dm.apply_event(&DownloadEvent::Complete { id: ModelId::U2net });
        assert_eq!(dm.state(ModelId::U2net), DownloadState::Done);
        assert!(dm.active.is_none());
    }

    #[test]
    fn apply_event_failed_records_retryable_flag() {
        let mut dm = fixture();
        dm.apply_event(&DownloadEvent::Failed {
            id: ModelId::U2net, error: "timeout".into(), retryable: true,
        });
        match dm.state(ModelId::U2net) {
            DownloadState::Failed { retryable, .. } => assert!(retryable),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn partial_path_appends_partial_suffix() {
        let p = partial_path(Path::new("/tmp/big_lama.onnx"));
        assert_eq!(p, PathBuf::from("/tmp/big_lama.onnx.partial"));
    }

    // ── Retry policy ──────────────────────────────────────────────────────

    fn never_cancel() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    #[test]
    fn retry_succeeds_after_two_transient_failures() {
        let attempts = std::sync::atomic::AtomicU32::new(0);
        let cancel = never_cancel();
        let result = retry_with_backoff(&cancel, 3, 1, || {
            let n = attempts.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                Err(DownloadError::transient("simulated timeout"))
            } else {
                Ok(())
            }
        });
        assert!(result.is_ok());
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn retry_fails_fast_on_fatal_error() {
        let attempts = std::sync::atomic::AtomicU32::new(0);
        let cancel = never_cancel();
        let result = retry_with_backoff(&cancel, 3, 1, || {
            attempts.fetch_add(1, Ordering::SeqCst);
            Err(DownloadError::fatal("404"))
        });
        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1, "fatal error must not retry");
    }

    #[test]
    fn retry_exhausts_max_attempts() {
        let attempts = std::sync::atomic::AtomicU32::new(0);
        let cancel = never_cancel();
        let result = retry_with_backoff(&cancel, 3, 1, || {
            attempts.fetch_add(1, Ordering::SeqCst);
            Err(DownloadError::transient("network"))
        });
        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn retry_aborts_during_backoff_on_cancel() {
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_for_thread = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(30));
            cancel_for_thread.store(true, Ordering::Release);
        });
        let attempts = std::sync::atomic::AtomicU32::new(0);
        let started = std::time::Instant::now();
        let result = retry_with_backoff(&cancel, 3, 200, || {
            attempts.fetch_add(1, Ordering::SeqCst);
            Err(DownloadError::transient("network"))
        });
        let elapsed = started.elapsed();
        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert!(elapsed < std::time::Duration::from_millis(200),
            "cancel during backoff must abort promptly, took {elapsed:?}");
    }

    // ── Multi-part dispatch ──────────────────────────────────────────────

    /// Aggregated bytes_so_far accumulates across parts: emitting the
    /// same Progress event sequence the UI expects works whether the
    /// download is single-file or multi-part.
    #[test]
    fn multi_part_progress_aggregation_via_apply_event() {
        let mut dm = fixture();
        let id = ModelId::SdV15InpaintFp16;
        // Simulate part-1 finishing (e.g. 1 GB) toward a 2 GB total.
        dm.apply_event(&DownloadEvent::Progress { id, bytes_so_far: 1_073_741_824, total: 2_147_483_648 });
        match dm.state(id) {
            DownloadState::InProgress { bytes_so_far, total_bytes } => {
                assert_eq!(bytes_so_far, 1_073_741_824);
                assert_eq!(total_bytes, 2_147_483_648);
            }
            other => panic!("expected InProgress, got {other:?}"),
        }
        // Bundle complete.
        dm.apply_event(&DownloadEvent::Complete { id });
        assert_eq!(dm.state(id), DownloadState::Done);
    }

    // ── Queue behaviour ───────────────────────────────────────────────────

    #[test]
    fn cancel_dequeues_a_pending_id() {
        let mut dm = fixture();
        dm.active = Some(ModelId::U2net);
        dm.queue.push_back(ModelId::LaMaFp32);
        dm.states.insert(ModelId::LaMaFp32, DownloadState::Queued);

        dm.cancel_download(ModelId::LaMaFp32);

        assert!(dm.queue.is_empty());
        assert_eq!(dm.state(ModelId::LaMaFp32), DownloadState::Idle);
        assert_eq!(dm.active(), Some(ModelId::U2net),
            "cancelling a queued id must not clear the active download");
    }
}
