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
use prunr_models::{descriptor, on_demand_dir, Column, ModelId, ModelPart, ModelSource, OnDemandVariant};

use crate::download::download_to_file;

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
    /// Downloads waiting for their check, and the one being checked, each
    /// with the column it runs in: the check's own finding can move the
    /// model to another accelerator.
    checks: VecDeque<(ModelId, Column)>,
    checking: Option<(ModelId, Column, crate::gui::machine_test::MachineTest)>,
}

/// The download threads' end of the channel: each event wakes the window
/// too. The checksum pass sends nothing for seconds, the window then goes
/// idle, and the event that ends a download would sit unread. Progress
/// comes with every network read, so it wakes nothing: the window draws
/// it at its own pace while the Model Store shows it.
#[derive(Clone)]
struct Waking {
    tx: mpsc::Sender<DownloadEvent>,
    repaint: Arc<OnceLock<egui::Context>>,
}

/// What a download's file reports as its checksum is checked: only the
/// last file of a download says Verifying. A download in parts (Nomos2's
/// thirty) or with a half-precision companion otherwise flips its card
/// between the bar and "Verifying" at every file.
fn verifying(tx: &Waking, id: ModelId, last: bool) -> impl Fn() {
    let tx = tx.clone();
    move || {
        if last {
            let _ = tx.send(DownloadEvent::Verifying { id });
        }
    }
}

impl Waking {
    fn send(&self, event: DownloadEvent) -> Result<(), mpsc::SendError<DownloadEvent>> {
        let wakes = !matches!(event, DownloadEvent::Progress { .. });
        let sent = self.tx.send(event);
        if wakes {
            self.wake();
        }
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

    /// Check `id`'s result in `column` once the app is idle.
    pub(crate) fn queue_check(&mut self, id: ModelId, column: Column) {
        let checking = self.checking.as_ref().is_some_and(|(m, ..)| *m == id);
        if !checking && !self.checks.iter().any(|(m, _)| *m == id) {
            self.checks.push_back((id, column));
        }
    }

    /// Each frame: follow the check under way, or start the next while the
    /// app is `idle`. The model whose check just ended, and how.
    pub(crate) fn pump_checks(&mut self, idle: bool) -> Option<(ModelId, Column, crate::gui::machine_test::Ended)> {
        if let Some((id, column, check)) = &mut self.checking {
            let (id, column, ended) = (*id, *column, check.pump()?);
            self.checking = None;
            return Some((id, column, ended));
        }
        if !idle {
            return None;
        }
        let (id, column) = self.checks.pop_front()?;
        let waking = self.progress_tx.clone();
        match crate::gui::machine_test::MachineTest::check(id, move || waking.wake()) {
            Ok(check) => self.checking = Some((id, column, check)),
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

    pub(crate) fn active(&self) -> Option<ModelId> {
        self.active
    }

    /// Downloads waiting for the one under way.
    pub(crate) fn queued(&self) -> usize {
        self.queue.len()
    }

    /// `id` taken as downloading, with no thread behind it.
    #[cfg(test)]
    pub(crate) fn begin_for_test(&mut self, id: ModelId) {
        self.active = Some(id);
        self.states.insert(id, DownloadState::InProgress { bytes_so_far: 0, total_bytes: 1 });
    }

    /// What a download thread sends events with.
    #[cfg(test)]
    pub(crate) fn sender_for_test(&self) -> impl Fn(DownloadEvent) + Send + 'static {
        let tx = self.progress_tx.clone();
        move |event| drop(tx.send(event))
    }

    /// No-op when the model is already installed with every file it
    /// downloads, or in progress / queued.
    pub(crate) fn start_download(&mut self, id: ModelId) {
        if prunr_models::is_available(id) && !prunr_models::lacks_fp16_companion(id) {
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

    /// Drained events for the UI to act on; the manager owns no UI state.
    /// Progress, one per network read, is taken in here and not returned.
    pub(crate) fn pump(&mut self) -> Vec<DownloadEvent> {
        let mut out = Vec::new();
        while let Ok(event) = self.progress_rx.try_recv() {
            self.apply_event(&event);
            if !matches!(event, DownloadEvent::Progress { .. }) {
                out.push(event);
            }
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
        let own = Fetch { url, sha256, dest: dir.join(filename), bytes: u64::from(size_mb) * 1024 * 1024 };
        let companion = fp16.map(|v| Fetch { url: v.url, sha256: v.sha256, dest: dir.join(v.filename), bytes: v.size_bytes() });
        let single = Single::new(own, companion);
        // Main file + optional fp16 companion count as one download, one bar.
        let total = single.total();
        let cancel = self.begin_active(id, total);
        let tx = self.progress_tx.clone();
        std::thread::spawn(move || {
            if let Some(why) = no_room(&dir, total) {
                let _ = tx.send(failure_event(id, why, &cancel));
                return;
            }
            let _ = tx.send(single.run(id, &tx, &cancel, download_to_file));
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
            for (index, part) in parts.iter().enumerate() {
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
                match download_to_file(part.url, &dest, part.sha256, cancel.clone(), &on_progress, &verifying(&tx, id, index + 1 == parts.len())) {
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

/// One file of a download: from `url` to `dest`, checked against `sha256`.
struct Fetch {
    url: &'static str,
    sha256: &'static str,
    dest: PathBuf,
    bytes: u64,
}

/// A single-file download: the model's own file, then its half-precision
/// companion where it has one.
struct Single {
    /// `None` when it is in place: only the companion did not come.
    own: Option<Fetch>,
    companion: Option<Fetch>,
}

impl Single {
    fn new(own: Fetch, companion: Option<Fetch>) -> Self {
        Self { own: Some(own).filter(|own| !own.dest.is_file()), companion }
    }

    fn total(&self) -> u64 {
        self.own.iter().chain(&self.companion).map(|f| f.bytes).sum()
    }

    /// Fetch each file in turn with `fetch` (`download_to_file`, or a
    /// test's); how the download ends.
    fn run(
        &self,
        id: ModelId,
        tx: &Waking,
        cancel: &Arc<AtomicBool>,
        fetch: impl Fn(&str, &Path, &str, Arc<AtomicBool>, &dyn Fn(u64, u64), &dyn Fn()) -> Result<(), DlError>,
    ) -> DownloadEvent {
        let total = self.total();
        let mut done = 0;
        if let Some(own) = &self.own {
            let on_progress = |bytes_so_far: u64, _part_total: u64| {
                let _ = tx.send(DownloadEvent::Progress { id, bytes_so_far, total });
            };
            if let Err(e) = fetch(own.url, &own.dest, own.sha256, Arc::clone(cancel), &on_progress, &verifying(tx, id, self.companion.is_none())) {
                return failure_event(id, e.message, cancel);
            }
            done = own.bytes;
        }
        let Some(companion) = &self.companion else { return DownloadEvent::Complete { id, note: None } };
        let on_progress = |part_so_far: u64, _part_total: u64| {
            let _ = tx.send(DownloadEvent::Progress { id, bytes_so_far: done + part_so_far, total });
        };
        match fetch(companion.url, &companion.dest, companion.sha256, Arc::clone(cancel), &on_progress, &verifying(tx, id, true)) {
            Ok(()) => DownloadEvent::Complete { id, note: None },
            // Fetched alone, or cancelled: it ends as that, the model's own
            // file staying installed.
            Err(e) if self.own.is_none() || cancel.load(Ordering::Acquire) => failure_event(id, e.message, cancel),
            Err(e) => {
                // fp16 companion failed: keep the main fp32 install
                // and surface a non-fatal warning. Without the fp16
                // file on disk the engine loads the fp32 weights.
                tracing::warn!(
                    ?id,
                    error = %e.message,
                    "fp16 companion download failed — fp32 install succeeded; engine will use fp32"
                );
                DownloadEvent::Complete { id, note: Some(COMPANION_FAILED.to_string()) }
            }
        }
    }
}

const COMPANION_FAILED: &str = "Its faster half-precision file did not download, so it runs at full precision. Download it again to get it.";

/// How a download that stopped early ends: cancelled by the user, or failed.
fn failure_event(id: ModelId, error: String, cancel: &AtomicBool) -> DownloadEvent {
    if cancel.load(Ordering::Acquire) {
        DownloadEvent::Cancelled { id }
    } else {
        DownloadEvent::Failed { id, error }
    }
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

/// Remove every partial an earlier run left in the model store: one from a
/// quit mid-download stayed for good when that model was never fetched
/// again.
pub(crate) fn cleanup_stale() {
    if let Some(store) = prunr_models::on_demand_dir() {
        sweep_store_partials(&store);
    }
}

/// `cleanup_stale` in `store` and each bundle's folder in it.
fn sweep_store_partials(store: &Path) {
    for dir in crate::fs_util::dir_and_subfolders(store) {
        crate::fs_util::sweep_left_by_ended_runs(&dir, crate::download::PARTIAL, |_| true);
    }
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
        assert!(prunr_core::sysmem::free_disk_bytes(&std::env::temp_dir()).is_some_and(|free| free > 0));
    }

    fn fixture() -> DownloadManager {
        DownloadManager::new()
    }

    /// A download runs on its own thread while the window may sit idle
    /// (the checksum pass sends nothing for seconds): each event it sends
    /// but progress wakes the window, or a finished download sat unread
    /// and the queued ones never started until the pointer moved.
    #[test]
    fn a_download_s_end_wakes_the_window() {
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

    /// Progress comes with every network read, hundreds a second: it
    /// wakes nothing and is taken in without being handed on, so a
    /// download does not draw the window per read.
    #[test]
    fn progress_does_not_wake_the_window() {
        let ctx = egui::Context::default();
        for _ in 0..5 {
            let _ = ctx.run_ui(Default::default(), |_| {});
        }
        let mut dm = DownloadManager::new();
        dm.wake(&ctx);
        let tx = dm.progress_tx.clone();
        std::thread::spawn(move || {
            for bytes_so_far in 1..=100 {
                let _ = tx.send(DownloadEvent::Progress { id: ModelId::LaMaFp32, bytes_so_far, total: 100 });
            }
        })
        .join()
        .unwrap();
        assert!(!ctx.has_requested_repaint(), "no read woke the window");
        assert!(dm.pump().is_empty(), "nothing for the window to act on");
        assert_eq!(dm.state(ModelId::LaMaFp32), DownloadState::InProgress { bytes_so_far: 100, total_bytes: 100 });
    }

    /// Only a download's last file says Verifying: the card of one in
    /// parts kept flipping between its bar and "Verifying" at each part.
    #[test]
    fn only_the_last_file_says_verifying() {
        let mut dm = DownloadManager::new();
        dm.begin_for_test(ModelId::FourXNomos2RealPlksr);
        for last in [false, false, true] {
            super::verifying(&dm.progress_tx, ModelId::FourXNomos2RealPlksr, last)();
            dm.pump();
            let verifying = dm.state(ModelId::FourXNomos2RealPlksr) == DownloadState::Verifying;
            assert_eq!(verifying, last, "a part that is not the last says nothing");
        }
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

    /// At start every model's partial an earlier run left goes, a bundle's
    /// too, even for a model never fetched again; a running instance's
    /// download and the installed models stay.
    #[test]
    fn the_start_sweeps_every_abandoned_partial_in_the_store() {
        let store = tempfile::tempdir().unwrap();
        let bundle = store.path().join("sd15-inpaint-fp16-1.0.0");
        std::fs::create_dir_all(&bundle).unwrap();
        let file = |path: PathBuf| {
            std::fs::write(&path, b"x").unwrap();
            path
        };
        let dead = crate::fs_util::a_free_pid();
        let live = std::process::id();
        let left = [
            file(store.path().join(format!("big_lama.onnx.partial.{dead}"))),
            file(store.path().join("migan.onnx.partial")),
            file(bundle.join(format!("unet.onnx.partial.{dead}"))),
        ];
        let kept = [
            file(store.path().join(format!("u2net.onnx.partial.{live}"))),
            file(store.path().join("u2net-1.0.0.onnx")),
            file(bundle.join("unet.onnx")),
            file(bundle.join(format!("vae.onnx.partial.{live}"))),
        ];
        sweep_store_partials(store.path());
        assert!(left.iter().all(|p| !p.exists()), "an earlier run's partials go");
        assert!(kept.iter().all(|p| p.exists()), "a running download and the installed models stay");
    }

    /// A model folder linked from elsewhere (a developer's export) is not
    /// swept: what lies there is not the store's.
    #[cfg(unix)]
    #[test]
    fn the_start_sweep_does_not_follow_a_linked_folder() {
        let store = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let theirs = elsewhere.path().join(format!("unet.onnx.partial.{}", crate::fs_util::a_free_pid()));
        std::fs::write(&theirs, b"x").unwrap();
        std::os::unix::fs::symlink(elsewhere.path(), store.path().join("rorem-4s-fp16-1.0.0")).unwrap();
        sweep_store_partials(store.path());
        assert!(theirs.exists());
        assert!(store.path().join("rorem-4s-fp16-1.0.0").exists(), "the link itself stays");
    }

    // ── Single-file downloads ─────────────────────────────────────────────

    fn file_in(dir: &Path, name: &'static str, mb: u64) -> Fetch {
        Fetch { url: name, sha256: "", dest: dir.join(name), bytes: mb * 1024 * 1024 }
    }

    /// Run `single` against a network that answers each url with
    /// `answer`: the urls it asked for, and how it ended.
    fn run_offline(single: &Single, cancel: &Arc<AtomicBool>, answer: impl Fn(&str, &AtomicBool) -> Result<(), DlError>) -> (Vec<String>, DownloadEvent) {
        let dm = fixture();
        let asked = std::cell::RefCell::new(Vec::new());
        let ended = single.run(ModelId::RealEsrganX4Plus, &dm.progress_tx, cancel, |url, _, _, cancel, _, _| {
            asked.borrow_mut().push(url.to_string());
            answer(url, &cancel)
        });
        (asked.into_inner(), ended)
    }

    /// A model whose own file is in place (its half-precision file failed
    /// or was cancelled) fetches only the half-precision file, its bar
    /// counting that file alone.
    #[test]
    fn with_its_own_file_in_place_only_the_half_precision_file_is_fetched() {
        let dir = tempfile::tempdir().unwrap();
        let own = file_in(dir.path(), "x4.onnx", 64);
        std::fs::write(&own.dest, b"fp32").unwrap();
        let single = Single::new(own, Some(file_in(dir.path(), "x4_fp16.onnx", 33)));
        assert_eq!(single.total(), 33 * 1024 * 1024);
        let (asked, ended) = run_offline(&single, &Arc::default(), |_, _| Ok(()));
        assert_eq!(asked, ["x4_fp16.onnx"]);
        assert!(matches!(ended, DownloadEvent::Complete { note: None, .. }), "{ended:?}");
    }

    /// A cancel while the half-precision file comes ends as a cancel, not
    /// as "ready" with a note that the file failed; the model's own file,
    /// fetched first, stays.
    #[test]
    fn a_cancel_during_the_half_precision_file_ends_as_a_cancel() {
        let dir = tempfile::tempdir().unwrap();
        let single = Single::new(file_in(dir.path(), "x4.onnx", 64), Some(file_in(dir.path(), "x4_fp16.onnx", 33)));
        let cancel = Arc::new(AtomicBool::new(false));
        let (asked, ended) = run_offline(&single, &cancel, |url, cancel| {
            if url == "x4_fp16.onnx" {
                cancel.store(true, Ordering::Release);
                return Err(<DlError as prunr_runtime_install::Retryable>::cancelled());
            }
            Ok(())
        });
        assert_eq!(asked, ["x4.onnx", "x4_fp16.onnx"]);
        assert!(matches!(ended, DownloadEvent::Cancelled { .. }), "{ended:?}");
    }

    /// A half-precision file that fails within a whole download leaves the
    /// model ready and says to download it again; fetched alone, its
    /// failure is a failed download, which the card offers to try again,
    /// and the next try brings it.
    #[test]
    fn a_failed_half_precision_file_can_be_tried_again() {
        let dir = tempfile::tempdir().unwrap();
        let (own, companion) = (|| file_in(dir.path(), "x4.onnx", 64), || Some(file_in(dir.path(), "x4_fp16.onnx", 33)));
        let refused = |url: &str, _: &AtomicBool| if url == "x4_fp16.onnx" { Err(DlError::fatal("The download was refused (404 Not Found)")) } else { Ok(()) };
        let (_, ended) = run_offline(&Single::new(own(), companion()), &Arc::default(), refused);
        assert!(matches!(&ended, DownloadEvent::Complete { note: Some(note), .. } if note == COMPANION_FAILED), "{ended:?}");

        std::fs::write(own().dest, b"fp32").unwrap();
        let alone = Single::new(own(), companion());
        let (_, ended) = run_offline(&alone, &Arc::default(), refused);
        let mut dm = fixture();
        dm.begin_for_test(ModelId::RealEsrganX4Plus);
        dm.apply_event(&ended);
        assert!(matches!(dm.state(ModelId::RealEsrganX4Plus), DownloadState::Failed { .. }), "{ended:?}");
        let (asked, ended) = run_offline(&alone, &Arc::default(), |_, _| Ok(()));
        assert_eq!(asked, ["x4_fp16.onnx"], "the try again fetches it alone");
        assert!(matches!(ended, DownloadEvent::Complete { note: None, .. }), "{ended:?}");
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
