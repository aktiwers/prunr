//! Coordinator for on-demand model downloads.
//!
//! One active download at a time; more queue in submission order, so a
//! burst of strokes can't stack five 800 MB downloads. HTTP, SHA and the
//! atomic write live in `download_to_file`, so the state machine here is
//! testable without a network. A download onto an accelerator no one
//! measured the model on ends with a check of its result there
//! (`bench::here::check`), one at a time while the app is idle.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, OnceLock};

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
    progress_tx: Waking,
    progress_rx: mpsc::Receiver<DownloadEvent>,
    active: Option<ModelId>,
    /// Downloads waiting for their check, and the one being checked.
    checks: VecDeque<ModelId>,
    checking: Option<(ModelId, crate::gui::machine_test::MachineTest)>,
}

/// The download threads' end of the channel: each event wakes the window
/// too. The checksum pass sends nothing for seconds, the window then goes
/// idle, and the event that ends a download would sit unread.
#[derive(Clone)]
struct Waking {
    tx: mpsc::Sender<DownloadEvent>,
    repaint: Arc<OnceLock<egui::Context>>,
}

impl Waking {
    fn send(&self, event: DownloadEvent) -> Result<(), mpsc::SendError<DownloadEvent>> {
        let sent = self.tx.send(event);
        self.wake();
        sent
    }

    fn wake(&self) {
        if let Some(ctx) = self.repaint.get() {
            ctx.request_repaint();
        }
    }
}

impl DownloadManager {
    pub(crate) fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            states: HashMap::new(),
            queue: VecDeque::new(),
            cancel_flags: HashMap::new(),
            progress_tx: Waking { tx, repaint: Arc::default() },
            progress_rx: rx,
            active: None,
            checks: VecDeque::new(),
            checking: None,
        }
    }

    /// A check waits, or runs.
    pub(crate) fn has_checks(&self) -> bool {
        self.checking.is_some() || !self.checks.is_empty()
    }

    pub(crate) fn checking(&self) -> bool {
        self.checking.is_some()
    }

    /// Check `id`'s result once the app is idle.
    pub(crate) fn queue_check(&mut self, id: ModelId) {
        let checking = self.checking.as_ref().is_some_and(|(m, _)| *m == id);
        if !checking && !self.checks.contains(&id) {
            self.checks.push_back(id);
        }
    }

    /// Each frame: follow the check under way, or start the next while the
    /// app is `idle`. The model whose check just ended, and how.
    pub(crate) fn pump_checks(&mut self, idle: bool) -> Option<(ModelId, crate::gui::machine_test::Ended)> {
        if let Some((id, check)) = &mut self.checking {
            let (id, ended) = (*id, check.pump()?);
            self.checking = None;
            return Some((id, ended));
        }
        if !idle {
            return None;
        }
        let id = self.checks.pop_front()?;
        let waking = self.progress_tx.clone();
        match crate::gui::machine_test::MachineTest::check(id, move || waking.wake()) {
            Ok(check) => self.checking = Some((id, check)),
            Err(e) => tracing::warn!(?id, %e, "a download's check did not start"),
        }
        None
    }

    /// The window the download threads wake with each event.
    pub(crate) fn wake(&self, ctx: &egui::Context) {
        let _ = self.progress_tx.repaint.set(ctx.clone());
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

    /// Drained events for the UI to fan out; the manager owns no UI state.
    pub(crate) fn pump(&mut self) -> Vec<DownloadEvent> {
        let mut out = Vec::new();
        while let Ok(event) = self.progress_rx.try_recv() {
            self.apply_event(&event);
            out.push(event);
        }
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
        // Main file + optional fp16 companion count as one download, one bar.
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

    /// Download each part in turn under `<on_demand_dir>/<subdir>/`, progress
    /// counted over the bundle. Parts already renamed are skipped on retry:
    /// resume across cancels without HTTP Range.
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

    /// Seed `InProgress`, register the cancel flag and mark `id` active; the
    /// flag goes to the worker thread.
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

/// Download `url` to `dest` via a `.partial` sidecar, renamed only once its
/// SHA256 matches. Transient network errors (timeout, 5xx) get up to 3
/// tries with backoff; fatal ones (404, SHA mismatch, disk full, cancel)
/// fail fast.
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
    sweep_stale_partials(dest);

    let client = prunr_runtime_install::http_client(crate::runtime_install::USER_AGENT)
        .map_err(|e| DlError::fatal(format!("Could not start the download: {}", prunr_runtime_install::with_causes(&e))))?;

    let response = client.get(url).send()
        .map_err(|e| DlError::transient(format!("Network error: {}", prunr_runtime_install::with_causes(&e))))?;

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
            .map_err(|e| DlError::transient(format!("Read error: {}", prunr_runtime_install::with_causes(&e))))?;
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
    prunr_core::sysmem::free_disk_bytes(dir).and_then(|free| space_refusal(needed, free))
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

/// This process's own partial of `dest`: two instances fetching one
/// model would otherwise write into one file, and the one finishing first
/// would keep whatever the other made of it.
fn partial_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(format!(".partial.{}", std::process::id()));
    PathBuf::from(s)
}

/// Remove the partials of `dest` an earlier run left (quit mid-download
/// keeps gigabytes), sparing one a running instance is writing now: the
/// suffix names the process writing it, and one without (an older
/// version's) is no live download's.
fn sweep_stale_partials(dest: &Path) {
    let (Some(dir), Some(name)) = (dest.parent(), dest.file_name()) else { return };
    let mut prefix = name.to_owned();
    prefix.push(".partial");
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let file = entry.file_name();
        let Some(suffix) = file.as_encoded_bytes().strip_prefix(prefix.as_encoded_bytes()) else { continue };
        let abandoned = match suffix {
            [] => true,
            [b'.', pid @ ..] => std::str::from_utf8(pid).ok().and_then(|p| p.parse::<u32>().ok())
                .is_some_and(|pid| !crate::fs_util::pid_runs(pid)),
            _ => false,
        };
        if abandoned {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A download that cannot connect says why (the connection refused,
    /// a certificate not trusted), not only that the request failed.
    #[test]
    fn a_failed_download_names_its_cause() {
        let dir = tempfile::tempdir().unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let failed = download_attempt("http://127.0.0.1:1/model.onnx", &dir.path().join("model.onnx"), "00", &cancel, &|_, _| {}, &|| {});
        let why = failed.expect_err("nothing listens on port 1").message.to_lowercase();
        assert!(why.contains("refused"), "{why}");
    }

    /// A download that would fill the disk does not start, and says how
    /// much it needs; one with room does.
    #[test]
    fn a_download_needs_room_to_start() {
        const GB: u64 = 1024 * 1024 * 1024;
        let why = space_refusal(5 * GB, GB).expect("refused");
        assert!(why.starts_with("Not enough disk space"), "{why}");
        assert_eq!(space_refusal(GB, 5 * GB), None);
        assert!(space_refusal(GB, GB).is_some(), "room is left for the system");
        assert!(prunr_core::sysmem::free_disk_bytes(&std::env::temp_dir()).is_some_and(|free| free > 0));
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

    /// A download runs on its own thread while the window may sit idle
    /// (the checksum pass sends nothing for seconds): each event it sends
    /// wakes the window, or a finished download sat unread and the queued
    /// ones never started until the pointer moved.
    #[test]
    fn every_download_event_wakes_the_window() {
        let ctx = egui::Context::default();
        for _ in 0..5 {
            let _ = ctx.run_ui(Default::default(), |_| {});
        }
        assert!(!ctx.has_requested_repaint(), "an idle window");
        let mut dm = DownloadManager::new();
        dm.wake(&ctx);
        dm.active = Some(ModelId::LaMaFp32);
        dm.states.insert(ModelId::BigLaMa, DownloadState::Queued);
        dm.queue.push_back(ModelId::BigLaMa);
        let tx = dm.progress_tx.clone();
        std::thread::spawn(move || { let _ = tx.send(DownloadEvent::Cancelled { id: ModelId::LaMaFp32 }); }).join().unwrap();
        assert!(ctx.has_requested_repaint(), "the window is woken");
        dm.pump();
        assert_ne!(dm.state(ModelId::BigLaMa), DownloadState::Queued, "and the next one starts");
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

    /// Each process downloads into a partial of its own: two instances
    /// fetching one model wrote into one file, and the one finishing first
    /// kept whatever the other had made of it.
    #[test]
    fn each_process_downloads_into_its_own_partial() {
        let p = partial_path(Path::new("/tmp/big_lama.onnx"));
        assert_eq!(p, PathBuf::from(format!("/tmp/big_lama.onnx.partial.{}", std::process::id())));
    }

    /// A partial an earlier run left (quit mid-download: gigabytes kept)
    /// goes when the model is fetched again, however fresh: a retry right
    /// after a quit must not be refused for the space it holds. One a
    /// running instance writes stays however old (a laptop asleep for
    /// hours), and so do other models' files.
    #[test]
    fn a_partial_left_by_an_earlier_run_is_swept() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("big_lama.onnx");
        let old_time = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 3600);
        let file = |name: String, old: bool| {
            let path = dir.path().join(name);
            std::fs::write(&path, b"half").unwrap();
            if old {
                std::fs::File::options().write(true).open(&path).unwrap().set_modified(old_time).unwrap();
            }
            path
        };
        let dead = crate::fs_util::a_free_pid();
        let live = std::process::id();
        let left = [file("big_lama.onnx.partial".into(), true), file(format!("big_lama.onnx.partial.{dead}"), false)];
        let kept = [file(format!("big_lama.onnx.partial.{live}"), true), file(format!("migan.onnx.partial.{dead}"), true)];
        sweep_stale_partials(&dest);
        assert!(left.iter().all(|p| !p.exists()), "an earlier run's partials go");
        assert!(kept.iter().all(|p| p.exists()), "a live download and other models' files stay");
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
