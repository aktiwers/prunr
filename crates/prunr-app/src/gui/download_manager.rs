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

use prunr_runtime_install::DlError;
use prunr_models::{descriptor, on_demand_dir, ModelId, ModelPart, ModelSource, OnDemandVariant};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadState {
    Idle,
    Queued,
    InProgress { bytes_so_far: u64, total_bytes: u64 },
    Verifying,
    Failed { error: String },
    Done,
}

#[derive(Debug)]
pub enum DownloadEvent {
    Progress { id: ModelId, bytes_so_far: u64, total: u64 },
    Verifying { id: ModelId },
    /// It is installed; `note` says what the person should know (a
    /// companion file that did not come).
    Complete { id: ModelId, note: Option<String> },
    Failed { id: ModelId, error: String },
    Cancelled { id: ModelId },
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

    #[cfg(test)]
    pub(crate) fn active(&self) -> Option<ModelId> {
        self.active
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
            DownloadEvent::Complete { id, .. } => {
                self.states.insert(id, DownloadState::Done);
                self.end_active(id);
            }
            DownloadEvent::Failed { id, ref error } => {
                tracing::warn!(?id, %error, "model download failed");
                self.states.insert(id, DownloadState::Failed { error: error.clone() });
                self.end_active(id);
            }
            DownloadEvent::Cancelled { id } => {
                tracing::info!(?id, "model download cancelled");
                self.states.insert(id, DownloadState::Idle);
                self.end_active(id);
            }
        }
    }

    fn kick_off(&mut self, id: ModelId) {
        let Some(desc) = descriptor(id) else {
            self.send_failure(id, format!("Unknown model id: {id:?}"));
            return;
        };
        match desc.source {
            ModelSource::Bundled { .. } | ModelSource::MultiPartBundled { .. } => {
                self.send_failure(id, format!("{id:?} is bundled — no download path"));
            }
            ModelSource::OnDemand { url, sha256, filename, size_mb, fp16, .. } => {
                self.kick_off_single(id, url, sha256, filename, size_mb, fp16);
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
        fp16: Option<OnDemandVariant>,
    ) {
        let Some(dir) = self.prepare_dir(id, None) else { return };
        let dest = dir.join(filename);
        let fp16_dest = fp16.as_ref().map(|v| dir.join(v.filename));
        // Combined byte budget for progress: main file + optional fp16
        // companion. The UI bar tracks both as one logical download so
        // the user sees a single progress trajectory.
        let main_bytes = (size_mb as u64) * 1024 * 1024;
        let fp16_bytes = fp16.as_ref().map(|v| (v.size_mb as u64) * 1024 * 1024).unwrap_or(0);
        let total = main_bytes + fp16_bytes;
        let cancel = self.begin_active(id, total);
        let tx = self.progress_tx.clone();
        std::thread::spawn(move || {
            if let Some(why) = no_room(&dir, total) {
                let _ = tx.send(failure_event(id, why, &cancel));
                return;
            }
            let on_progress_main = {
                let tx = tx.clone();
                move |bytes_so_far: u64, _part_total: u64| {
                    let _ = tx.send(DownloadEvent::Progress { id, bytes_so_far, total });
                }
            };
            let on_verifying = {
                let tx = tx.clone();
                move || {
                    let _ = tx.send(DownloadEvent::Verifying { id });
                }
            };
            if let Err(e) = download_to_file(url, &dest, sha256, cancel.clone(), &on_progress_main, &on_verifying) {
                let _ = tx.send(failure_event(id, e.message, &cancel));
                return;
            }

            let mut note = None;
            if let (Some(variant), Some(dest_fp16)) = (fp16.as_ref(), fp16_dest.as_ref()) {
                let on_progress_fp16 = {
                    let tx = tx.clone();
                    move |part_so_far: u64, _part_total: u64| {
                        let _ = tx.send(DownloadEvent::Progress {
                            id, bytes_so_far: main_bytes + part_so_far, total,
                        });
                    }
                };
                if let Err(e) = download_to_file(variant.url, dest_fp16, variant.sha256, cancel, &on_progress_fp16, &on_verifying) {
                    // fp16 companion failed: keep the main fp32 install
                    // and surface a non-fatal warning. Without the fp16
                    // file on disk the engine loads the fp32 weights.
                    tracing::warn!(
                        ?id,
                        error = %e.message,
                        "fp16 companion download failed — fp32 install succeeded; engine will use fp32"
                    );
                    note = Some("Its faster half-precision file did not download, so it runs at full precision. Download it again to get it.".to_string());
                }
            }

            let _ = tx.send(DownloadEvent::Complete { id, note });
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
        let Some(bundle_dir) = self.prepare_dir(id, Some(subdir)) else { return };
        let total: u64 = parts.iter().map(|p| p.size_bytes).sum();
        let cancel = self.begin_active(id, total);
        let tx = self.progress_tx.clone();
        std::thread::spawn(move || {
            let to_fetch: u64 = parts.iter().filter(|p| !bundle_dir.join(p.filename).is_file()).map(|p| p.size_bytes).sum();
            if let Some(why) = no_room(&bundle_dir, to_fetch) {
                let _ = tx.send(failure_event(id, why, &cancel));
                return;
            }
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
                        let _ = tx.send(failure_event(id, format!("{}: {}", part.key, e.message), &cancel));
                        return;
                    }
                }
            }
            let _ = tx.send(DownloadEvent::Complete { id, note: None });
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

    fn end_active(&mut self, id: ModelId) {
        self.cancel_flags.remove(&id);
        if self.active == Some(id) {
            self.active = None;
        }
    }

    /// The on-demand models directory (or its `sub` folder), created.
    /// `None` after reporting the failure for `id`.
    fn prepare_dir(&mut self, id: ModelId, sub: Option<&str>) -> Option<PathBuf> {
        let Some(root) = on_demand_dir() else {
            self.send_failure(id, "Could not resolve user data directory".into());
            return None;
        };
        let dir = sub.map_or_else(|| root.clone(), |s| root.join(s));
        if let Err(e) = std::fs::create_dir_all(&dir) {
            self.send_failure(id, format!("Could not create models directory: {e}"));
            return None;
        }
        Some(dir)
    }

    fn send_failure(&mut self, id: ModelId, error: String) {
        self.states.insert(id, DownloadState::Failed { error: error.clone() });
        let _ = self.progress_tx.send(DownloadEvent::Failed { id, error });
    }
}

/// How a download that stopped early ends: cancelled by the user, or failed.
fn failure_event(id: ModelId, error: String, cancel: &AtomicBool) -> DownloadEvent {
    if cancel.load(Ordering::Acquire) {
        DownloadEvent::Cancelled { id }
    } else {
        DownloadEvent::Failed { id, error }
    }
}

/// Download `url` to `dest`, verifying SHA256 against `expected_sha`.
/// Writes to a `<dest>.partial` sidecar; renames to `dest` only after
/// verification passes. Wraps `download_attempt` in the shared retry
/// loop (`prunr_runtime_install::retry_with_backoff`) so transient
/// network errors (timeout, 5xx) get up to 3 tries with exponential
/// backoff. Fatal errors (404, SHA mismatch, disk full, user cancel)
/// fail fast without retry.
fn download_to_file(
    url: &str,
    dest: &Path,
    expected_sha: &str,
    cancel: Arc<AtomicBool>,
    on_progress: &dyn Fn(u64, u64),
    on_verifying: &dyn Fn(),
) -> Result<(), DlError> {
    use prunr_runtime_install::{retry_with_backoff, DOWNLOAD_ATTEMPTS, DOWNLOAD_BACKOFF_BASE_MS};
    retry_with_backoff(&cancel, DOWNLOAD_ATTEMPTS, DOWNLOAD_BACKOFF_BASE_MS, |_attempt| {
        download_attempt(url, dest, expected_sha, &cancel, on_progress, on_verifying)
    })
}

fn download_attempt(
    url: &str,
    dest: &Path,
    expected_sha: &str,
    cancel: &Arc<AtomicBool>,
    on_progress: &dyn Fn(u64, u64),
    on_verifying: &dyn Fn(),
) -> Result<(), DlError> {
    let partial = partial_path(dest);

    // Tear down any leftover partial from a previous aborted attempt so
    // the writer starts fresh.
    let _ = std::fs::remove_file(&partial);

    let client = prunr_runtime_install::http_client(crate::runtime_install::USER_AGENT)
        .map_err(|e| DlError::fatal(format!("Could not start the download: {e}")))?;

    let response = client.get(url).send()
        .map_err(|e| DlError::transient(format!("Network error: {e}")))?;

    let status = response.status();
    if !status.is_success() {
        let msg = if status.is_server_error() { format!("The server answered {status}; try again later") } else { format!("The download was refused ({status})") };
        return Err(DlError::for_status(status, msg));
    }

    let total = response.content_length().unwrap_or(0);
    finish_download(&partial, dest, fetch_into(response, &partial, total, expected_sha, cancel, on_progress, on_verifying))
}

/// A fetched file moves into place; one that failed (a full disk, a
/// dropped connection, a cancel, a bad checksum) leaves no partial, which
/// on a full disk could be gigabytes keeping it full.
fn finish_download(partial: &Path, dest: &Path, fetched: Result<(), DlError>) -> Result<(), DlError> {
    let result = fetched.and_then(|()| std::fs::rename(partial, dest).map_err(|e| DlError::fatal(format!("Could not finalize file: {e}"))));
    if result.is_err() {
        let _ = std::fs::remove_file(partial);
    }
    result
}

/// Stream `response` into `partial`, hashing as it goes, and check the hash.
fn fetch_into(
    mut response: impl std::io::Read,
    partial: &Path,
    total: u64,
    expected_sha: &str,
    cancel: &Arc<AtomicBool>,
    on_progress: &dyn Fn(u64, u64),
    on_verifying: &dyn Fn(),
) -> Result<(), DlError> {
    let mut file = std::fs::File::create(partial)
        .map_err(|e| DlError::fatal(format!("Could not create {}: {e}", partial.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut written: u64 = 0;
    loop {
        if cancel.load(Ordering::Acquire) {
            return Err(<DlError as prunr_runtime_install::Retryable>::cancelled());
        }
        let n = std::io::Read::read(&mut response, &mut buf)
            .map_err(|e| DlError::transient(format!("Read error: {e}")))?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buf[..n])
            .map_err(|e| DlError::fatal(format!("Write error: {e}")))?;
        hasher.update(&buf[..n]);
        written += n as u64;
        on_progress(written, total);
    }
    std::io::Write::flush(&mut file)
        .map_err(|e| DlError::fatal(format!("Flush error: {e}")))?;
    drop(file);

    on_verifying();
    let actual_sha = hex::encode(hasher.finalize());
    if !actual_sha.eq_ignore_ascii_case(expected_sha) {
        return Err(DlError::fatal(format!(
            "Checksum mismatch — expected {expected_sha}, got {actual_sha}",
        )));
    }

    Ok(())
}

/// Why `needed` bytes should not be downloaded into `dir`.
fn no_room(dir: &Path, needed: u64) -> Option<String> {
    crate::hardware::free_space_at(dir).and_then(|free| space_refusal(needed, free))
}

/// Why a download of `needed` bytes should not start with `free` bytes
/// free: it would fill the disk. Room is left over for the rest of the
/// system.
fn space_refusal(needed: u64, free: u64) -> Option<String> {
    const SPARE: u64 = 500 * 1024 * 1024;
    (free < needed.saturating_add(SPARE)).then(|| {
        format!(
            "Not enough disk space: it needs {}, and {} is free.",
            crate::gui::views::format_byte_size(needed),
            crate::gui::views::format_byte_size(free),
        )
    })
}

fn partial_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".partial");
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A download that would fill the disk does not start, and says how
    /// much it needs; one with room does.
    #[test]
    fn a_download_needs_room_to_start() {
        const GB: u64 = 1024 * 1024 * 1024;
        let why = space_refusal(5 * GB, GB).expect("refused");
        assert!(why.starts_with("Not enough disk space"), "{why}");
        assert_eq!(space_refusal(GB, 5 * GB), None);
        assert!(space_refusal(GB, GB).is_some(), "room is left for the system");
        assert!(crate::hardware::free_space_at(&std::env::temp_dir()).is_some_and(|free| free > 0));
    }

    /// A download that failed part-way leaves no partial file; one that
    /// finished moves into place.
    #[test]
    fn a_failed_download_leaves_no_partial_file() {
        let dir = tempfile::tempdir().unwrap();
        let (partial, dest) = (dir.path().join("model.onnx.partial"), dir.path().join("model.onnx"));
        std::fs::write(&partial, b"half").unwrap();
        assert!(finish_download(&partial, &dest, Err(DlError::fatal("Write error: no space"))).is_err());
        assert!(!partial.exists() && !dest.exists());
        std::fs::write(&partial, b"whole").unwrap();
        finish_download(&partial, &dest, Ok(())).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"whole");
    }

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
        dm.apply_event(&DownloadEvent::Complete { id: ModelId::U2net, note: None });
        assert_eq!(dm.state(ModelId::U2net), DownloadState::Done);
        assert!(dm.active.is_none());
    }

    #[test]
    fn apply_event_failed_records_the_error() {
        let mut dm = fixture();
        dm.active = Some(ModelId::U2net);
        dm.apply_event(&DownloadEvent::Failed { id: ModelId::U2net, error: "timeout".into() });
        assert_eq!(dm.state(ModelId::U2net), DownloadState::Failed { error: "timeout".into() });
        assert!(dm.active().is_none());
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
        let result: Result<(), DlError> = prunr_runtime_install::retry_with_backoff(&cancel, 3, 1, |_attempt| {
            let n = attempts.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                Err(DlError::transient("simulated timeout"))
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
        let result: Result<(), DlError> = prunr_runtime_install::retry_with_backoff(&cancel, 3, 1, |_attempt| {
            attempts.fetch_add(1, Ordering::SeqCst);
            Err(DlError::fatal("404"))
        });
        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1, "fatal error must not retry");
    }

    #[test]
    fn retry_exhausts_max_attempts() {
        let attempts = std::sync::atomic::AtomicU32::new(0);
        let cancel = never_cancel();
        let result: Result<(), DlError> = prunr_runtime_install::retry_with_backoff(&cancel, 3, 1, |_attempt| {
            attempts.fetch_add(1, Ordering::SeqCst);
            Err(DlError::transient("network"))
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
        let result: Result<(), DlError> = prunr_runtime_install::retry_with_backoff(&cancel, 3, 200, |_attempt| {
            attempts.fetch_add(1, Ordering::SeqCst);
            Err(DlError::transient("network"))
        });
        let elapsed = started.elapsed();
        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        // An un-aborted run sleeps 200 ms + 400 ms before its third attempt;
        // anything under that proves the abort, with room for CI scheduling
        // jitter (a 231 ms run was observed on a loaded macOS runner).
        assert!(elapsed < std::time::Duration::from_millis(500),
            "cancel during backoff must abort before the schedule completes, took {elapsed:?}");
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
        dm.apply_event(&DownloadEvent::Complete { id, note: None });
        assert_eq!(dm.state(id), DownloadState::Done);
    }

    // ── Queue behaviour ───────────────────────────────────────────────────

    /// A cancelled download goes back to "Download", not to a failure
    /// whose Retry was disabled until restart.
    #[test]
    fn a_cancelled_download_returns_to_idle_and_frees_the_slot() {
        let mut dm = fixture();
        dm.active = Some(ModelId::LaMaFp32);
        dm.states.insert(ModelId::LaMaFp32, DownloadState::InProgress { bytes_so_far: 1, total_bytes: 9 });
        dm.apply_event(&DownloadEvent::Cancelled { id: ModelId::LaMaFp32 });
        assert_eq!(dm.state(ModelId::LaMaFp32), DownloadState::Idle);
        assert!(dm.active().is_none());
    }

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
