//! Batch state, lifecycle, memory governance, and texture coordination.
//!
//! Owns:
//! - The `Vec<BatchItem>` and selection state
//! - The next-id counter for new items
//! - The `BackgroundIO` channels (thumbnail, decode, save-done, tex-prep,
//!   file-load)
//!
//! Does NOT own:
//! - Inference dispatch (worker channels, admission, live-preview) — `Processor`
//! - Undo — each item's own `timeline`
//! - Drag-out lifecycle — `DragExportState`

use std::collections::HashSet;
use std::sync::Arc;

use super::background_io::BackgroundIO;
use super::item::{BatchItem, BatchStatus, ImageSource, RunTier};
use super::memory::{AdmissionController, ImageMemCost, RESIDENT_HISTORY_STEPS};
use super::state::AppState;
use super::worker::{CachedTensorWork, WorkItem};

/// Per-status counts across the batch.
#[derive(Default, Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct StatusCounts {
    pub done: usize,
    pub processing: usize,
    pub errored: usize,
}

impl StatusCounts {
    pub(crate) fn batch_total(&self) -> usize {
        self.done + self.processing + self.errored
    }
}

/// Derived progress summary for the status bar.
#[derive(Default, Clone, Debug)]
pub(crate) struct StatusReport {
    pub stage: String,
    pub pct: f32,
}

/// Which shape the Process button takes for the current selection.
// `Process*` prefix matches the user-visible button labels and the
// dispatcher arms in `process_items` — renaming would split that pairing.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProcessButtonLabel {
    /// No checkboxes set — click targets the currently-viewed item only.
    ProcessViewed,
    /// N checkboxes set, 1 ≤ N < total (or 1 ≤ N and total == 1).
    ProcessSelected(usize),
    /// All N ≥ 2 items checked — click targets the whole batch.
    ProcessAll(usize),
}

/// Combined budget for compressed segmentation (`cached_tensor`) + edge
/// (`cached_edge_tensors`) caches across all batch items, in bytes.
const TENSOR_BUDGET: usize = 512 * 1024 * 1024;

/// Maximum thumbnail edge length in pixels.
const THUMB_MAX_PX: u32 = 160;

pub(crate) struct BatchManager {
    pub(crate) items: Vec<BatchItem>,
    pub(crate) selected_index: usize,
    pub(crate) next_id: u64,
    pub(crate) bg_io: BackgroundIO,
    last_history_sweep: std::time::Instant,
    /// A move of undo states to disk is running.
    demoting: Arc<std::sync::atomic::AtomicBool>,
    /// Selection edits made off the UI thread, in order per image.
    pub(crate) selection_queue: super::selection_queue::SelectionQueue,
}

const HISTORY_SWEEP_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);

/// What landing a background picture did, for the app's part: the saved
/// path map and the message.
pub(crate) enum BgLanded {
    Picked { hash: u64, path: std::path::PathBuf },
    Filled,
    Failed { picked: bool, error: String },
    Gone,
}

impl BatchManager {
    /// Compress every stored undo state nothing else holds, off-thread,
    /// behind the decode slots so a burst can't fan out N zstd encoders.
    /// Doing it inline cost ~0.1-0.6 s per switch, undo or redo.
    pub(crate) fn compress_history_off_thread(&mut self) {
        // A few times a second is plenty; every frame walked every step.
        let now = std::time::Instant::now();
        if now.saturating_duration_since(self.last_history_sweep) < HISTORY_SWEEP_INTERVAL {
            return;
        }
        self.last_history_sweep = now;
        let jobs: std::collections::VecDeque<super::timeline::CompressJob> = self.items.iter()
            .flat_map(|item| {
                let images = item.timeline.stored_images().filter_map(|c| c.compress_job());
                images.chain(item.timeline.stored_selections().filter_map(|c| c.compress_job()))
            })
            .collect();
        if jobs.is_empty() {
            return;
        }
        // A few workers share the sweep's jobs: after a batch lands there
        // can be a hundred, and a thread each cost the frame milliseconds.
        let workers = jobs.len().min(std::thread::available_parallelism().map_or(2, |n| n.get()));
        let jobs = Arc::new(std::sync::Mutex::new(jobs));
        for _ in 0..workers {
            let (jobs, slots) = (Arc::clone(&jobs), Arc::clone(&self.bg_io.decode_slots));
            std::thread::spawn(move || loop {
                let Some(job) = jobs.lock().unwrap_or_else(std::sync::PoisonError::into_inner).pop_front() else { break };
                let _slot = slots.acquire();
                job();
            });
        }
    }

    /// Under memory pressure: move compressed undo states to disk, on a
    /// worker. One move runs at a time; while it does, another ask is
    /// skipped, and the next landing under pressure asks again.
    pub(crate) fn demote_history_to_disk(&self) {
        use std::sync::atomic::Ordering;
        if self.demoting.swap(true, Ordering::AcqRel) {
            return;
        }
        let cells: Vec<super::timeline::StoredImage> = self.items.iter()
            .flat_map(|item| item.timeline.stored_images().cloned())
            .collect();
        let demoting = Arc::clone(&self.demoting);
        std::thread::spawn(move || {
            let moved = cells.iter().filter(|cell| cell.demote_to_disk()).count();
            tracing::info!(moved, "undo states moved to disk");
            demoting.store(false, Ordering::Release);
        });
    }

    pub(crate) fn new() -> Self {
        Self {
            items: Vec::new(),
            selected_index: 0,
            next_id: 0,
            bg_io: BackgroundIO::new(),
            last_history_sweep: std::time::Instant::now(),
            demoting: Arc::default(),
            selection_queue: super::selection_queue::SelectionQueue::new(),
        }
    }

    pub(crate) fn selected_item(&self) -> Option<&BatchItem> {
        self.items.get(self.selected_index)
    }

    /// `None` is common: a worker event can arrive after its item was removed.
    pub(crate) fn find_by_id(&self, id: u64) -> Option<&BatchItem> {
        self.items.iter().find(|b| b.id == id)
    }

    pub(crate) fn index_of(&self, id: u64) -> Option<usize> {
        self.items.iter().position(|b| b.id == id)
    }

    pub(crate) fn find_by_id_mut(&mut self, id: u64) -> Option<&mut BatchItem> {
        self.items.iter_mut().find(|b| b.id == id)
    }

    /// Derives app-level state from the selected item's status.
    pub(crate) fn app_state(&self) -> AppState {
        match self.selected_item() {
            None => AppState::Empty,
            Some(item) => match item.status() {
                BatchStatus::Done => AppState::Done,
                BatchStatus::Processing => AppState::Processing,
                _ => AppState::Loaded,
            },
        }
    }

    pub(crate) fn is_selected(&self, id: u64) -> bool {
        self.selected_item().is_some_and(|b| b.id == id)
    }

    /// A cut is queued or under way on some image.
    pub(crate) fn any_awaiting_cut(&self) -> bool {
        self.items.iter().any(BatchItem::awaits_cut)
    }

    /// The ticked images whose cut is queued or under way: what Cancel
    /// Selected stops, and what its label counts.
    pub(crate) fn ticked_awaiting_cut(&self) -> impl Iterator<Item = u64> + '_ {
        self.items.iter().filter(|i| i.ticked && i.awaits_cut()).map(|i| i.id)
    }

    pub(crate) fn ticked_ids(&self) -> Vec<u64> {
        self.items.iter().filter(|i| i.ticked).map(|i| i.id).collect()
    }


    pub(crate) fn set_all_ticked(&mut self, ticked: bool) {
        for item in &mut self.items {
            item.ticked = ticked;
        }
    }

    /// Selected index clamped to the batch size; `None` when the batch is empty.
    pub(crate) fn selected_idx_clamped(&self) -> Option<usize> {
        (!self.items.is_empty()).then(|| self.selected_index.min(self.items.len() - 1))
    }

    /// Clamp `selected_index` to the batch size (0 on an empty batch).
    pub(crate) fn clamp_selected_index(&mut self) {
        self.selected_index = self.selected_index.min(self.items.len().saturating_sub(1));
    }

    /// `true` when the selection changed, so the caller resets the canvas.
    pub(crate) fn select_item(&mut self, idx: usize) -> bool {
        debug_assert!(idx < self.items.len(), "select_item: idx {idx} out of bounds (len={})", self.items.len());
        if self.selected_index == idx {
            return false;
        }
        self.selected_index = idx;
        true
    }

    /// Move the item at `from` to `to` (`items.len()` appends), keeping the
    /// same item selected.
    pub(crate) fn reorder(&mut self, from: usize, to: usize) {
        if from == to || from >= self.items.len() {
            return;
        }
        let item = self.items.remove(from);
        let dst = if from < to { to - 1 } else { to };
        self.items.insert(dst, item);
        if self.selected_index == from {
            self.selected_index = dst;
        } else if from < self.selected_index && self.selected_index <= to {
            self.selected_index -= 1;
        } else if to <= self.selected_index && self.selected_index < from {
            self.selected_index += 1;
        }
    }

    /// Derive the Process button's label shape from current selection state.
    /// Single-item batches where the one item is checked render as
    /// `ProcessSelected(1)` (not `ProcessAll(1)`) — "Process All [1]" reads
    /// as a glitch.
    pub(crate) fn process_button_label(&self) -> ProcessButtonLabel {
        let total = self.items.len();
        let ticked = self.ticked_count();
        match (ticked, total) {
            (0, _) => ProcessButtonLabel::ProcessViewed,
            (n, t) if n == t && t >= 2 => ProcessButtonLabel::ProcessAll(t),
            (n, _) => ProcessButtonLabel::ProcessSelected(n),
        }
    }

    /// Item ids the Process button should dispatch: the checkbox-checked set
    /// if any are checked, else the currently-viewed item. Empty when the
    /// batch is empty.
    pub(crate) fn items_to_process(&self) -> Vec<u64> {
        self.process_targets().map(|i| i.id).collect()
    }

    /// The images Process runs on, as `items_to_process` picks them: the
    /// Process gate reads the same set, so a stale index cannot grey the
    /// button while the run would go.
    pub(crate) fn process_targets(&self) -> impl Iterator<Item = &BatchItem> {
        self.targets(self.selected_idx_clamped()).map(|(_, i)| i)
    }

    /// Any item in the action target set satisfies `f`, without
    /// allocating (toolbar Undo/Redo enablement).
    pub(crate) fn any_target_can<F: FnMut(&BatchItem) -> bool>(&self, mut f: F) -> bool {
        self.targets(self.viewed_unclamped()).any(|(_, item)| f(item))
    }

    pub(crate) fn targeted_indices(&self) -> Vec<usize> {
        self.targets(self.viewed_unclamped()).map(|(i, _)| i).collect()
    }

    /// The action target set, with indices: the ticked items when any are
    /// ticked, else the one at `viewed`. Callers resolve `viewed`
    /// themselves: Process clamps a stale index, Undo/Redo and the
    /// targeted indices skip it.
    fn targets(&self, viewed: Option<usize>) -> impl Iterator<Item = (usize, &BatchItem)> {
        let ticked = self.any_ticked();
        self.items.iter().enumerate()
            .filter(move |&(i, item)| if ticked { item.ticked } else { Some(i) == viewed })
    }

    /// `selected_index` when it names an item.
    fn viewed_unclamped(&self) -> Option<usize> {
        self.selected_item().map(|_| self.selected_index)
    }

    /// Remove the item at `idx`; its undo files go with its timeline.
    /// Returns `false` when `idx` is out of bounds. Caller is responsible
    /// for clamping `selected_index` and any downstream sync.
    /// Abandon every cut, under way or waiting for a slot, after a
    /// Cancel-All: each image shows its own state again. Late arrivals
    /// from the worker are dropped by the job check in `on_batch_item_done`.
    pub(crate) fn abandon_cuts(&mut self) {
        for item in &mut self.items {
            if item.awaits_cut() {
                item.abandon_job();
            }
        }
    }

    /// Results that landed on images not on screen: once their step holds
    /// them, the full-size copy and any texture go (the timeline keeps the
    /// image and compresses it); they come back when the image is shown.
    /// A batch of 4x upscales otherwise held ~0.8 GB per 12 MP image.
    pub(crate) fn park_off_screen_results(&mut self) {
        for item in &mut self.items {
            item.park_off_screen();
        }
    }

    /// Mark the images `marks` name as saved in those states.
    pub(crate) fn mark_saved(&mut self, marks: &[crate::gui::item::SaveMark]) {
        for &mark in marks {
            if let Some(item) = self.find_by_id_mut(mark.id) {
                item.mark_saved(mark);
            }
        }
    }

    /// Why `format` cannot hold the images Save writes (the ticked ones,
    /// else the one shown), if it cannot: the largest of them is past its
    /// side limit.
    pub(crate) fn save_refusal(&self, format: prunr_core::ExportFormat) -> Option<String> {
        let any_ticked = self.any_ticked();
        let shown = self.selected_item().filter(|_| !any_ticked);
        let targets = self.items.iter().filter(|i| any_ticked && i.ticked).chain(shown);
        let (w, h) = targets.map(BatchItem::shown_dims).max_by_key(|&(w, h)| w.max(h))?;
        format.refusal(w, h)
    }

    /// Images whose work closing or removing them would lose.
    pub(crate) fn unsaved_count(&self) -> usize {
        self.items.iter().filter(|item| item.has_unsaved_work()).count()
    }

    pub(crate) fn ticked_count(&self) -> usize {
        self.items.iter().filter(|i| i.ticked).count()
    }

    pub(crate) fn any_ticked(&self) -> bool {
        self.items.iter().any(|i| i.ticked)
    }

    /// An empty batch is not all ticked.
    pub(crate) fn all_ticked(&self) -> bool {
        !self.items.is_empty() && self.items.iter().all(|i| i.ticked)
    }

    /// The per-status counts, in one pass.
    pub(crate) fn status_counts(&self) -> StatusCounts {
        let mut c = StatusCounts::default();
        for item in &self.items {
            match item.status() {
                BatchStatus::Done => c.done += 1,
                BatchStatus::Processing => c.processing += 1,
                BatchStatus::Error(_) => c.errored += 1,
                _ => {}
            }
        }
        c
    }

    pub(crate) fn progress(&self) -> StatusReport {
        let counts = self.status_counts();
        let total = counts.batch_total();
        let stage = if self.any_awaiting_cut() {
            format!("Processing {}/{total}", counts.done)
        } else {
            "Finishing up".to_string()
        };
        StatusReport {
            stage,
            pct: counts.done as f32 / total.max(1) as f32,
        }
    }

    /// First error message across errored items, for a batch-failure toast.
    pub(crate) fn first_error_message(&self) -> Option<&str> {
        self.items.iter().find_map(BatchItem::error)
    }

    /// Decode source bytes on a background thread into `bg_io.decode_rx`. A
    /// failure travels as `Err` so the item stops waiting for a decode.
    pub(crate) fn request_decode_bytes(&self, item_id: u64, bytes: Arc<Vec<u8>>) {
        self.spawn_decode(item_id, move || Ok(bytes));
    }

    /// Pre-decode from an ImageSource. The file is read on the decode
    /// thread, not the UI thread; a read that fails lands as a failed
    /// decode, so the image stops waiting for one.
    pub(crate) fn request_decode_source(&self, item_id: u64, source: &ImageSource) {
        let source = source.clone();
        self.spawn_decode(item_id, move || source.load_bytes().map_err(super::item::source_unreadable));
    }

    fn spawn_decode(&self, item_id: u64, bytes: impl FnOnce() -> Result<Arc<Vec<u8>>, String> + Send + 'static) {
        let tx = self.bg_io.decode_tx.clone();
        // A slot past `available_parallelism()` waits, so a 50-image
        // Process All doesn't hold 50 × ~80 MB transient peaks.
        self.bg_io.decode_slots.spawn(move || {
            // The `DynamicImage` (~50 MB at 4 K RGBA) and the encoded
            // bytes both drop before the channel send, so concurrent
            // decodes don't pile up DynamicImage + RGBA per thread.
            // Every decode answers, a panic included: the image waits on
            // it until then.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                bytes().and_then(|bytes| match prunr_core::load_image_from_bytes(&bytes) {
                    Ok(img) => Ok(Arc::new(img.into_rgba8())),
                    Err(e) => Err(format!("Could not read the image: {e}")),
                })
            })).unwrap_or_else(|_| {
                tracing::error!(item_id, "image decode panicked");
                Err("Could not read the image.".to_string())
            });
            let _ = tx.send((item_id, result));
        });
    }

    /// A background picture finished loading. A picked one replaces the
    /// background as one undo step; a preset's fills in the picture its
    /// settings name, if they still do; a preset's that failed is dropped.
    pub(crate) fn land_bg_image(&mut self, load: super::background_io::BgImageLoad) -> BgLanded {
        let super::background_io::BgImageLoad { item_id, path, picked, result } = load;
        let Some(item) = self.find_by_id_mut(item_id) else { return BgLanded::Gone };
        match result {
            Ok((img, hash)) if picked => {
                item.settings.bg = None;
                item.settings.bg_effect = prunr_core::BgEffect::None;
                item.set_bg_image_hashed(img, Some(path.clone()), hash);
                item.commit(super::timeline::StepKind::Adjust);
                BgLanded::Picked { hash, path }
            }
            Ok((img, hash)) => {
                if item.settings.bg_image_hash.map(|h| h.get()) == Some(hash) {
                    item.set_bg_image_hashed(img, Some(path), hash);
                }
                BgLanded::Filled
            }
            Err(error) => {
                if !picked {
                    item.clear_bg_image();
                }
                BgLanded::Failed { picked, error }
            }
        }
    }

    /// Load and hash a background picture off the UI thread; it lands on
    /// `bg_io.bg_image_rx`.
    pub(crate) fn request_bg_image(&self, item_id: u64, path: std::path::PathBuf, picked: bool) {
        let tx = self.bg_io.bg_image_tx.clone();
        self.bg_io.decode_slots.spawn(move || {
            let result = prunr_core::load_image_from_path(&path)
                .map(|img| {
                    let hash = super::item::bg_image_content_hash(&img);
                    (img, hash)
                })
                .map_err(|e| e.to_string());
            let _ = tx.send(super::background_io::BgImageLoad { item_id, path, picked, result });
        });
    }

    /// Filter-only Process off the UI thread; the result lands on
    /// `bg_io.filter_only_rx`.
    pub(crate) fn request_filter_only(
        &self,
        item_id: u64,
        run: u64,
        source: &ImageSource,
        base: Option<super::worker::RunImage>,
        effect: prunr_core::SubjectEffect,
    ) {
        let tx = self.bg_io.filter_only_tx.clone();
        let source = source.clone();
        self.bg_io.decode_slots.spawn(move || {
            let load = || source.load_bytes().map_err(super::item::source_unreadable);
            // Drop bytes / DynamicImage / RGBA before send so concurrent
            // threads don't pile per-image peaks.
            let mut chain_start = None;
            let result: Result<Arc<image::RgbaImage>, String> = (|| {
                let (start, built) = base.map_or(Ok((None, None)), |run| run.resolve(item_id, load))?;
                chain_start = built;
                let mut rgba = match start {
                    // No effect: the pass is its start, without a copy.
                    Some(start) if effect == prunr_core::SubjectEffect::None => return Ok(start),
                    Some(start) => Arc::unwrap_or_clone(start),
                    None => {
                        let bytes = load()?;
                        let original = prunr_core::load_image_from_bytes(&bytes)
                            .map_err(|_| "Decode failed".to_string())?;
                        drop(bytes);
                        original.into_rgba8()
                    }
                };
                prunr_core::apply_subject_effect(&mut rgba, effect);
                Ok(Arc::new(rgba))
            })();
            let _ = tx.send((item_id, run, result, chain_start));
        });
    }

    /// Build a thumbnail on a background thread, from `result_rgba` when it
    /// is `Some`, else from the source. Results are stored transparent, so
    /// the thumb gets its own background composite to match the canvas.
    pub(crate) fn request_thumbnail(
        &self,
        item_id: u64,
        source: &ImageSource,
        result: Option<super::item::Pixels>,
    ) {
        let tx = self.bg_io.thumb_tx.clone();
        if let Some(result) = result {
            self.bg_io.decode_slots.spawn(move || {
                // A parked result unpacks here, off the UI thread.
                let image = result.load().map(|rgba| {
                    let (w, h) = fit_dimensions(rgba.width(), rgba.height(), THUMB_MAX_PX, THUMB_MAX_PX);
                    let thumb = prunr_core::formats::resize_rgba(&rgba, w, h, prunr_core::formats::ResizeFilter::Bilinear);
                    (thumb.width(), thumb.height(), thumb.into_raw())
                });
                let _ = tx.send(super::background_io::ThumbMsg { item_id, image });
            });
        } else {
            let source = source.clone();
            self.bg_io.decode_slots.spawn(move || {
                // Source bytes, DynamicImage and full RGBA drop before the send: on a
                // 50-image batch they were N × ~50 MB co-resident.
                let thumb = (|| -> Option<image::RgbaImage> {
                    let bytes = source.load_bytes().ok()?;
                    let img = prunr_core::load_image_from_bytes(&bytes).ok()?;
                    drop(bytes);
                    let (w, h) = fit_dimensions(img.width(), img.height(), THUMB_MAX_PX, THUMB_MAX_PX);
                    Some(prunr_core::formats::resize_to_rgba(&img, w, h, prunr_core::formats::ResizeFilter::Bilinear))
                })();
                let image = thumb.map(|t| (t.width(), t.height(), t.into_raw()));
                let _ = tx.send(super::background_io::ThumbMsg { item_id, image });
            });
        }
    }

    /// Evict tensor caches, oldest-loaded first, until under budget. Both
    /// caches go together: one without the other is useless.
    pub(crate) fn enforce_tensor_budget(&mut self) {
        // Budget only what eviction can free — selection planes and
        // embeddings would otherwise force tensor evictions they never
        // relieve.
        let total: usize = self.items.iter().map(BatchItem::evictable_tensor_bytes).sum();
        if total <= TENSOR_BUDGET { return; }
        let selected_id = self.selected_item().map(|b| b.id);
        let mut remaining = total;
        for item in &mut self.items {
            if remaining <= TENSOR_BUDGET { break; }
            // Preserve the selected item's tensors (most likely to be reused).
            if Some(item.id) == selected_id { continue; }
            remaining -= item.evictable_tensor_bytes();
            item.drop_tensor_caches();
        }
    }

    /// Keep the selected item's selection texture current: request a
    /// rebuild whenever the mask or the baked style differs from what the
    /// texture (or the build in flight) was made from. One path for every
    /// producer — commits, undo/redo, image switch, knob changes.
    pub(crate) fn ensure_selection_texture(
        &mut self,
        style: super::background_io::SelectionStyle,
        ctx: &egui::Context,
    ) {
        let Some(idx) = self.selected_idx_clamped() else { return };
        let item = &mut self.items[idx];
        // No selection left: nothing will replace the old texture, which
        // the overlay would keep drawing.
        let Some(hash) = item.selection_hash else {
            item.selection_texture = None;
            item.selection_tex_pending = None;
            return;
        };
        let key = (hash, style);
        if item.current_selection_texture(style).is_some() || item.selection_tex_pending == Some(key) {
            return;
        }
        let Some(mask) = item.selection_mask.clone() else { return };
        item.selection_tex_pending = Some(key);
        let (item_id, source) = (item.id, item.source_rgba.clone());
        // A patch applies only to a texture of the same style, and only
        // without feather: feathering reshapes the plane beyond the stroke.
        let base = item.selection_texture.as_ref()
            .filter(|t| t.key.1 == style && style.edge_feather_px == 0)
            .map(|t| (t.key.0, Arc::clone(&t.shown)));
        self.bg_io.request_selection_visualization(item_id, mask, key, source, base, ctx.clone());
    }

    /// The items a drag from row `idx` carries: every ticked one when that
    /// row is ticked, else that row. Like Save, a drag writes what each
    /// image shows, a run under way or not.
    pub(crate) fn drag_targets(&self, idx: usize) -> Vec<u64> {
        match self.items.get(idx) {
            Some(row) if row.ticked => self.items.iter().filter(|b| b.ticked).map(|b| b.id).collect(),
            Some(row) => vec![row.id],
            None => Vec::new(),
        }
    }

    /// Evict all tensor caches except the selected item (called under memory pressure).
    pub(crate) fn evict_all_tensors(&mut self) {
        let selected_id = self.selected_item().map(|b| b.id);
        for item in &mut self.items {
            if Some(item.id) != selected_id {
                item.drop_tensor_caches();
            }
        }
    }

    /// Commit a new selection mask to the named item, registering it for
    /// undo (`BatchItem::commit_selection_mask`). Does NOT fire any model
    /// dispatch; `apply_selection_to_active_model` owns that decision.
    ///
    /// Returns `false` if the item was not found or the mask is
    /// byte-identical to the current selection (nothing mutated).
    #[cfg(test)]
    pub(crate) fn commit_selection(
        &mut self,
        item_id: u64,
        mask: prunr_core::selection::MaskArtifact,
        kind: super::timeline::StepKind,
    ) -> bool {
        self.find_by_id_mut(item_id)
            .is_some_and(|item| item.commit_selection_mask(Arc::new(mask), kind))
    }

    /// Drop the selection on the named item (Esc, Clear button) as an
    /// undo step. Returns true if there was one.
    pub(crate) fn clear_selection(&mut self, item_id: u64) -> bool {
        let Some(item) = self.find_by_id_mut(item_id) else { return false };
        let was_present = item.selection_mask.is_some();
        item.invalidate_selection();
        if was_present {
            item.commit(super::timeline::StepKind::Selection);
        }
        was_present
    }


    /// Give the images in `ids` the settings of the image at `from`, with
    /// its background picture: a run applies what is shown to the batch.
    pub fn share_settings(&mut self, from: usize, ids: &HashSet<u64>) {
        let Some(source) = self.items.get(from) else { return };
        let (settings, bg_image) = (source.settings, source.bg_image.clone());
        for item in self.in_set(ids) {
            item.adopt_settings(settings, bg_image.clone());
        }
    }

    /// Drop every image's Magic Brush embedding (16 MB each).
    pub fn clear_magic_embeddings(&mut self) {
        for item in &mut self.items {
            super::reclaim::later(item.magic_brush_embedding.take());
        }
    }

    /// Without a background-removal model there is no subject to outline,
    /// so Subject lines draw over the whole photo, the nearest mode that
    /// still has lines.
    pub fn fit_line_modes_to(&mut self, model: super::settings::SettingsModel) {
        if model.uses_segmentation() {
            return;
        }
        for item in self.items.iter_mut().filter(|i| i.settings.line_mode == prunr_core::LineMode::SubjectOutline) {
            item.settings.line_mode = prunr_core::LineMode::EdgesOnly;
        }
    }
}

/// The images a Process run sends, by how much of the pipeline each
/// needs, and how many were already up to date (for the toast).
#[derive(Default)]
pub(crate) struct ClassifiedTiers {
    pub(crate) full: HashSet<u64>,
    pub(crate) mask: HashSet<u64>,
    pub(crate) add_edge: HashSet<u64>,
    pub(crate) up_to_date_count: usize,
}

impl ClassifiedTiers {
    pub(crate) fn process_count(&self) -> usize {
        self.full.len() + self.mask.len() + self.add_edge.len()
    }

    pub(crate) fn all_process_ids(&self) -> HashSet<u64> {
        self.full.iter()
            .chain(self.mask.iter())
            .chain(self.add_edge.iter())
            .copied().collect()
    }
}

impl BatchManager {
    pub(crate) fn in_set<'a>(&'a mut self, ids: &'a HashSet<u64>) -> impl Iterator<Item = &'a mut BatchItem> + 'a {
        self.items.iter_mut().filter(move |i| ids.contains(&i.id))
    }

    /// Whether any image has a decode, thumbnail or texture on its way.
    pub(crate) fn has_pending_work(&self) -> bool {
        self.items.iter().any(BatchItem::has_pending_work) || self.selection_queue.any_pending()
    }

    /// Sort the images in `ids` by the run a `step` by `cutter` needs
    /// (`BatchItem::run_tier`). One that chains onto another model's
    /// cut-out is a full run, whatever its recipe says.
    pub(crate) fn classify(&mut self, ids: &HashSet<u64>, model: prunr_core::ModelKind, step: Option<super::timeline::StepKind>, cutter: super::item::Cutter) -> ClassifiedTiers {
        let mut tiers = ClassifiedTiers::default();
        for item in self.in_set(ids) {
            if item.chains(step, cutter) {
                tiers.full.insert(item.id);
                continue;
            }
            match item.run_tier(model) {
                RunTier::UpToDate => tiers.up_to_date_count += 1,
                RunTier::Full => { tiers.full.insert(item.id); }
                RunTier::Mask => { tiers.mask.insert(item.id); }
                RunTier::AddEdge => { tiers.add_edge.insert(item.id); }
            }
        }
        tiers
    }

    /// The work for the images in `ids` that run again from their cached
    /// tensor.
    pub(crate) fn cached_tensor_work(&mut self, ids: &HashSet<u64>) -> Vec<CachedTensorWork> {
        self.in_set(ids)
            .filter_map(|item| {
                let work = CachedTensorWork {
                    item_id: item.id,
                    tensor: item.cached_tensor.clone()?,
                    source: item.source.clone(),
                    mask: item.settings.mask_settings(),
                };
                item.mark_running();
                Some(work)
            })
            .collect()
    }

    /// What a full run on each image in `ids` is expected to hold in RAM.
    pub(crate) fn full_run_costs(&self, ids: &HashSet<u64>) -> Vec<ImageMemCost> {
        self.items.iter()
            .filter(|i| ids.contains(&i.id))
            .map(|i| AdmissionController::estimate_cost(i.id, i.dimensions, i.source_len, RESIDENT_HISTORY_STEPS))
            .collect()
    }

    /// The work for full runs on the images in `ids`, in batch order.
    pub(crate) fn full_run_work(&mut self, ids: &HashSet<u64>) -> Vec<WorkItem> {
        self.in_set(ids).map(|item| {
            item.mark_running();
            item.full_run_work()
        }).collect()
    }

}

/// Compute dimensions that fit within `max_w` × `max_h` preserving aspect ratio.
fn fit_dimensions(src_w: u32, src_h: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    let scale = (max_w as f32 / src_w as f32).min(max_h as f32 / src_h as f32).min(1.0);
    ((src_w as f32 * scale).round().max(1.0) as u32,
     (src_h as f32 * scale).round().max(1.0) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::item_settings::ItemSettings;

    /// A format refuses Save when an image Save writes is past its largest
    /// side: the ticked images, else the one shown.
    #[test]
    fn a_format_refuses_save_for_an_image_past_its_largest_side() {
        use prunr_core::ExportFormat::{Png, Webp};
        let mut mgr = BatchManager::new();
        mgr.items.push(item_with_cache(1, 0));
        mgr.items.push(item_with_cache(2, 0));
        mgr.items[0].dimensions = (16_385, 900);
        mgr.items[1].dimensions = (4_000, 3_000);
        mgr.select_item(0);
        let why = mgr.save_refusal(Webp).expect("the shown image is too wide for WebP");
        assert!(why.contains("16385"), "{why}");
        assert_eq!(mgr.save_refusal(Png), None);
        mgr.items[1].ticked = true;
        assert_eq!(mgr.save_refusal(Webp), None, "only the ticked image is saved");
        mgr.items[0].ticked = true;
        assert!(mgr.save_refusal(Webp).is_some());
    }
    use crate::gui::worker::{CompressedTensor, RawSegTensor};
    use prunr_core::ModelKind;
    use std::time::Duration;

    fn fixture() -> BatchManager {
        BatchManager::new()
    }

    /// Recording a step (a switch, an undo, a new result) must cost the UI
    /// thread nothing: a stored state is compressed off-thread once nothing
    /// else holds it, and reads back unchanged. The step an undo returns to
    /// is held unpacked instead.
    #[test]
    fn undo_states_are_compressed_off_the_ui_thread() {
        let mut batch = fixture();
        let mut item = item_with_cache(1, 0);
        item.on_screen = true;
        let rgba = Arc::new(image::RgbaImage::from_pixel(64, 64, image::Rgba([9, 8, 7, 255])));
        item.result_rgba = Some(Arc::clone(&rgba));
        crate::gui::tests::fixtures::put_in(&mut item, BatchStatus::Done);
        item.commit(crate::gui::timeline::StepKind::Process);
        item.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(64, 64, image::Rgba([1, 1, 1, 255]))));
        item.commit(crate::gui::timeline::StepKind::Edit);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(64, 64)));
        item.commit(crate::gui::timeline::StepKind::Edit);
        batch.items.push(item);
        drop(rgba);

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let first_compressed = |b: &BatchManager| b.items[0].timeline.stored_images().next().is_some_and(|c| c.is_compressed());
        while !first_compressed(&batch) && std::time::Instant::now() < deadline {
            batch.compress_history_off_thread();
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(first_compressed(&batch), "an older result, held by nothing else, is compressed");
        let warm = |b: &BatchManager| b.items[0].warm_steps.lock().unwrap_or_else(std::sync::PoisonError::into_inner)[crate::gui::item::BACK].as_ref().is_some_and(|w| w.image.is_some());
        while !warm(&batch) && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let previous_compressed = batch.items[0].timeline.stored_images().nth(1).is_some_and(|c| c.is_compressed());
        assert!(warm(&batch) && !previous_compressed, "the step undo returns to stays unpacked");
        let item = &mut batch.items[0];
        let _ = item.undo_step();
        let _ = item.undo_step();
        assert_eq!(item.shown_result().expect("shown").get_pixel(5, 5), &image::Rgba([9, 8, 7, 255]));
    }

    /// Undo onto a packed step unpacks nothing on the UI thread: the step
    /// stays packed and shows through its handle until the result
    /// texture's build unpacks it, and the item holds what that build
    /// unpacked only while the step still shows it.
    #[test]
    fn undo_onto_a_packed_step_leaves_it_for_the_texture_build() {
        let mut batch = fixture();
        let mut item = item_with_cache(1, 0);
        item.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(64, 64, image::Rgba([9, 8, 7, 255]))));
        crate::gui::tests::fixtures::put_in(&mut item, BatchStatus::Done);
        item.commit(crate::gui::timeline::StepKind::Process);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(64, 64)));
        item.commit(crate::gui::timeline::StepKind::Edit);
        batch.items.push(item);

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let first_packed = |b: &BatchManager| b.items[0].timeline.stored_images().next().is_some_and(|c| c.is_compressed());
        while !first_packed(&batch) && std::time::Instant::now() < deadline {
            batch.compress_history_off_thread();
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(first_packed(&batch));
        let item = &mut batch.items[0];
        item.on_screen = true;
        let _ = item.undo_step();
        assert!(item.result_rgba.is_none() && first_packed(&batch), "undo unpacked nothing");
        let item = &mut batch.items[0];
        let pixels = item.result_pixels().expect("the step's result");
        assert!(matches!(pixels, crate::gui::item::Pixels::Parked(_)));
        let unpacked = pixels.read().expect("unpacks");
        item.hold_unpacked_result(Arc::new(image::RgbaImage::new(64, 64)));
        assert!(item.result_rgba.is_none(), "pixels of another step are not held");
        item.hold_unpacked_result(Arc::clone(&unpacked));
        assert!(item.result_rgba.as_ref().is_some_and(|r| Arc::ptr_eq(r, &unpacked)));
        assert_eq!(unpacked.get_pixel(5, 5), &image::Rgba([9, 8, 7, 255]));
    }

    /// A packed step that cannot be read ends its texture request instead
    /// of leaving the image busy, and is not asked for again while it shows.
    #[test]
    fn an_unreadable_step_ends_its_texture_request() {
        let mut item = item_with_cache(1, 0);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(4, 4)));
        crate::gui::tests::fixtures::put_in(&mut item, BatchStatus::Done);
        item.commit(crate::gui::timeline::StepKind::Process);
        item.result_tex_pending = true;
        let target = item.request_result_texture(false);
        item.texture_unreadable(target);
        assert!(!item.result_tex_pending && !item.has_pending_work());
        assert!(item.result_unreadable(), "not asked for again");
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(4, 4)));
        item.commit(crate::gui::timeline::StepKind::Edit);
        assert!(!item.result_unreadable(), "another step is asked for");
    }

    /// The step undo returns to keeps its selection and edits unpacked too,
    /// so restoring them reads the held planes.
    #[test]
    fn the_step_undo_returns_to_keeps_its_planes_unpacked() {
        let mut item = item_with_cache(1, 0);
        item.on_screen = true;
        let plane = |v: i8| Arc::new(prunr_core::selection::MaskArtifact::from_cells(4, 4, vec![v; 16]));
        item.selection_mask = Some(plane(1));
        item.edits = Some(plane(-1));
        item.commit(crate::gui::timeline::StepKind::Stroke);
        item.selection_mask = None;
        item.edits = None;
        item.commit(crate::gui::timeline::StepKind::Stroke);
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let held = |i: &BatchItem| i.warm_steps.lock().unwrap_or_else(std::sync::PoisonError::into_inner)[crate::gui::item::BACK]
            .as_ref().map(|w| w.planes.clone());
        while held(&item).is_none_or(|p| p.iter().any(Option::is_none)) && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let [selection, edits] = held(&item).expect("warmed");
        let back = item.timeline.previous().expect("previous");
        assert!(selection.is_some_and(|s| back.selection.as_ref().is_some_and(|b| b.holds(&s))));
        assert!(edits.is_some_and(|e| back.edits.as_ref().is_some_and(|b| b.holds(&e))));
    }

    /// Under memory pressure the undo states move to disk on a worker,
    /// not on the frame that asked.
    #[test]
    fn undo_states_move_to_disk_off_the_ui_thread() {
        let mut batch = fixture();
        let mut item = item_with_cache(1, 0);
        item.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(16, 16, image::Rgba([4, 5, 6, 255]))));
        item.commit(crate::gui::timeline::StepKind::Process);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(16, 16)));
        item.commit(crate::gui::timeline::StepKind::Edit);
        batch.items.push(item);
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let first = |b: &BatchManager| b.items[0].timeline.stored_images().next().cloned().expect("a stored step");
        while !first(&batch).is_compressed() && std::time::Instant::now() < deadline {
            batch.compress_history_off_thread();
            std::thread::sleep(Duration::from_millis(5));
        }
        batch.demote_history_to_disk();
        while !first(&batch).is_on_disk() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(first(&batch).is_on_disk(), "moved by the worker");
        assert_eq!(first(&batch).load().expect("reads back").get_pixel(1, 1), &image::Rgba([4, 5, 6, 255]));
    }

    /// A BatchItem with `tensor_floats` f32s of cached tensor (0 = none).
    fn item_with_cache(id: u64, tensor_floats: usize) -> BatchItem {
        let mut item = BatchItem::new(
            id,
            format!("item_{id}.png"),
            ImageSource::Bytes(Arc::new(Vec::new())),
            (10, 10),
            ItemSettings::default(),
            String::new(),
        );
        if tensor_floats > 0 {
            // Real CompressedTensor: zstd-compress f32 data via from_raw.
            let data: Vec<f32> = vec![0.5; tensor_floats];
            let cache = RawSegTensor { data, height: 10, width: 10, model: ModelKind::Silueta };
            item.set_cached_tensor(CompressedTensor::from_raw(cache));
        }
        item
    }

    /// Every image in the batch, not just the one on screen: an image left
    /// on Subject would otherwise run a cut-out under No model.
    #[test]
    fn without_a_cutout_model_every_subject_line_mode_becomes_whole_photo() {
        use crate::gui::settings::SettingsModel;
        use prunr_core::LineMode;
        let mut batch = fixture();
        for (id, mode) in [(1, LineMode::SubjectOutline), (2, LineMode::Off), (3, LineMode::SubjectOutline)] {
            let mut item = item_with_cache(id, 0);
            item.settings.line_mode = mode;
            batch.items.push(item);
        }
        batch.fit_line_modes_to(SettingsModel::Silueta);
        assert_eq!(batch.items[0].settings.line_mode, LineMode::SubjectOutline, "a cut-out model keeps Subject");
        batch.fit_line_modes_to(SettingsModel::None);
        let modes: Vec<_> = batch.items.iter().map(|i| i.settings.line_mode).collect();
        assert_eq!(modes, [LineMode::EdgesOnly, LineMode::Off, LineMode::EdgesOnly]);
    }

    /// A run gives every image the shown image's settings, and with them
    /// its background image: the bytes, not just the settings' reference
    /// to them, or the others would show and save no background.
    #[test]
    fn a_run_shares_the_shown_images_background_picture() {
        let mut batch = fixture();
        for id in 1..=3 {
            batch.items.push(item_with_cache(id, 0));
        }
        batch.items[0].set_bg_image(image::DynamicImage::new_rgba8(4, 4), None);
        batch.items[0].settings.gamma = 1.7;
        batch.share_settings(0, &std::collections::HashSet::from([1, 2]));
        let shared = &batch.items[1];
        assert_eq!(shared.settings.gamma, 1.7);
        assert!(shared.bg_image.is_some(), "the picture came along");
        assert_eq!(shared.bg_image.as_ref().map(|b| b.hash), shared.settings.bg_image_hash.map(|h| h.get()));
        assert_eq!(batch.items[2].settings.gamma, ItemSettings::default().gamma, "an image outside the run is untouched");
        assert!(batch.items[2].bg_image.is_none());
    }

    // ── selected_item / is_selected / selected_idx_clamped ──────────────

    /// The selection getters agree on an empty batch, an index in range
    /// and a stale one left past the end by a removal: the accessor must
    /// not panic, and the clamped index falls back to the last item.
    #[test]
    fn selection_getters_on_empty_in_range_and_stale_index() {
        let empty = fixture();
        assert!(empty.selected_item().is_none());
        assert!(!empty.is_selected(0));
        assert_eq!(empty.selected_idx_clamped(), None);

        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        bm.items.push(item_with_cache(42, 0));
        bm.selected_index = 1;
        assert_eq!(bm.selected_item().map(|i| i.id), Some(42));
        assert!(bm.is_selected(42) && !bm.is_selected(1));
        assert_eq!(bm.selected_idx_clamped(), Some(1));

        bm.selected_index = 99;
        assert!(bm.selected_item().is_none());
        assert!(!bm.is_selected(42));
        assert_eq!(bm.selected_idx_clamped(), Some(1));
    }
    // ── status_counts ───────────────────────────────────────────────────

    /// A batch of items put in each status, in order.
    fn batch_of(statuses: Vec<BatchStatus>) -> BatchManager {
        let mut bm = fixture();
        for (id, status) in (1..).zip(statuses) {
            let mut item = item_with_cache(id, 0);
            crate::gui::tests::fixtures::put_in(&mut item, status);
            bm.items.push(item);
        }
        bm
    }

    #[test]
    fn status_counts_totals_by_status_excluding_pending() {
        use BatchStatus::*;
        let bm = batch_of(vec![Done, Done, Processing, Error("oops".into()), Pending]);
        let c = bm.status_counts();
        assert_eq!(c.done, 2);
        assert_eq!(c.processing, 1);
        assert_eq!(c.errored, 1);
        assert_eq!(c.batch_total(), 4, "Pending items are excluded from batch_total");
    }

    #[test]
    fn an_empty_batch_counts_nothing_and_divides_by_nothing() {
        let bm = fixture();
        let c = bm.status_counts();
        assert_eq!(c, StatusCounts::default());
        assert_eq!(c.batch_total(), 0);
        let report = bm.progress();
        assert_eq!(report.stage, "Finishing up");
        assert_eq!(report.pct, 0.0);
    }

    #[test]
    fn progress_stage_says_processing_while_in_flight() {
        let bm = batch_of(vec![BatchStatus::Done, BatchStatus::Processing, BatchStatus::Pending]);
        let report = bm.progress();
        assert_eq!(report.stage, "Processing 1/2");
        // pct = done(1) / batch_total(2) = 0.5; pending excluded from total.
        assert!((report.pct - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn progress_stage_says_finishing_when_no_in_flight() {
        let report = batch_of(vec![BatchStatus::Done]).progress();
        assert_eq!(report.stage, "Finishing up");
        assert!((report.pct - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn first_error_message_returns_first_errored_item() {
        use BatchStatus::*;
        let bm = batch_of(vec![Done, Error("first problem".into()), Error("second problem".into())]);
        assert_eq!(bm.first_error_message(), Some("first problem"));
        assert_eq!(batch_of(vec![Done]).first_error_message(), None);
    }

    // ── enforce_tensor_budget / evict_all_tensors ───────────────────────

    #[test]
    fn enforce_tensor_budget_no_op_below_budget() {
        let mut bm = fixture();
        // Tiny tensors well under 512 MB.
        bm.items.push(item_with_cache(1, 100));
        bm.items.push(item_with_cache(2, 100));
        let before_1 = bm.items[0].cached_tensor.is_some();
        let before_2 = bm.items[1].cached_tensor.is_some();
        bm.enforce_tensor_budget();
        assert_eq!(bm.items[0].cached_tensor.is_some(), before_1);
        assert_eq!(bm.items[1].cached_tensor.is_some(), before_2);
    }

    #[test]
    fn evict_all_tensors_clears_all_except_selected() {
        let mut bm = fixture();
        let mut item = item_with_cache(1, 100);
        item.edge_planes.store_placeholder();
        bm.items.push(item);
        bm.items.push(item_with_cache(2, 100));
        bm.items.push(item_with_cache(3, 100));
        bm.selected_index = 1; // item id=2
        assert!(bm.items.iter().all(|i| i.cached_tensor.is_some()));

        bm.evict_all_tensors();

        assert!(bm.items[0].cached_tensor.is_none(), "non-selected evicted");
        assert!(bm.items[0].edge_planes.is_empty() && bm.items[0].cached_edge_tensors.is_none(), "its edge cache too");
        assert!(bm.items[1].cached_tensor.is_some(), "selected preserved");
        assert!(bm.items[2].cached_tensor.is_none(), "non-selected evicted");
    }
    // ── request_decode_bytes / request_thumbnail (thread-spawning) ──────

    /// 1x1 PNG byte sequence — minimum valid PNG.
    fn one_pixel_png() -> Vec<u8> {
        let img = image::RgbaImage::from_pixel(1, 1, image::Rgba([200, 100, 50, 255]));
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    /// A file that can't be read reports a failed decode: before, nothing
    /// came back and the image waited for a decode forever.
    #[test]
    fn a_source_that_cannot_be_read_lands_as_a_failed_decode() {
        let bm = fixture();
        bm.request_decode_source(5, &ImageSource::file("/nonexistent/prunr-test/missing.jpg".into()));
        let (id, result) = bm.bg_io.decode_rx.recv_timeout(Duration::from_secs(5)).expect("an answer comes back");
        assert_eq!(id, 5);
        assert!(result.is_err());
    }

    #[test]
    fn request_decode_bytes_emits_decoded_rgba_to_decode_rx() {
        let bm = fixture();
        let png = Arc::new(one_pixel_png());
        bm.request_decode_bytes(42, png);

        let (id, result) = bm.bg_io.decode_rx.recv_timeout(Duration::from_secs(2))
            .expect("decode_tx must produce a result within 2s");
        assert_eq!(id, 42);
        let rgba = result.expect("valid PNG must decode");
        assert_eq!(rgba.dimensions(), (1, 1));
        assert_eq!(rgba.get_pixel(0, 0).0, [200, 100, 50, 255]);
    }

    #[test]
    fn request_decode_source_for_bytes_variant_routes_through() {
        let bm = fixture();
        let source = ImageSource::Bytes(Arc::new(one_pixel_png()));
        bm.request_decode_source(99, &source);

        let (id, result) = bm.bg_io.decode_rx.recv_timeout(Duration::from_secs(2))
            .expect("decode_tx must produce a result within 2s");
        assert_eq!(id, 99);
        assert!(result.is_ok());
    }

    /// A background picture loads off the UI thread with its content hash.
    #[test]
    fn a_background_picture_loads_with_its_hash() {
        let bm = fixture();
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("bg.png");
        std::fs::write(&path, one_pixel_png()).unwrap();
        bm.request_bg_image(5, path.clone(), true);
        let load = bm.bg_io.bg_image_rx.recv_timeout(Duration::from_secs(2)).expect("lands within 2 s");
        assert_eq!((load.item_id, load.picked, load.path), (5, true, path));
        let (img, hash) = load.result.expect("decodes");
        assert_eq!(hash, crate::gui::item::bg_image_content_hash(&img));
    }

    /// A malformed image must surface a recognisable error via `Err` so
    /// the UI clears `decode_pending` and flags the item instead of
    /// leaving it stuck "still loading."
    #[test]
    fn request_decode_bytes_emits_err_on_malformed_input() {
        let bm = fixture();
        let garbage = Arc::new(b"not an image".to_vec());
        bm.request_decode_bytes(7, garbage);

        let (id, result) = bm.bg_io.decode_rx.recv_timeout(Duration::from_secs(2))
            .expect("decode_tx must produce a result within 2s");
        assert_eq!(id, 7);
        let err = result.expect_err("garbage bytes must produce Err");
        assert!(err.starts_with("Could not read the image"), "{err}");
    }

    #[test]
    fn request_thumbnail_with_result_rgba_emits_to_thumb_rx() {
        let bm = fixture();
        let result = Arc::new(image::RgbaImage::from_pixel(200, 200, image::Rgba([10, 20, 30, 255])));
        let source = ImageSource::Bytes(Arc::new(Vec::new())); // unused when result_rgba is Some
        bm.request_thumbnail(7, &source, Some(crate::gui::item::Pixels::Resident(result)));

        let msg = bm.bg_io.thumb_rx.recv_timeout(Duration::from_secs(2))
            .expect("thumb_tx must produce a result within 2s");
        assert_eq!(msg.item_id, 7);
        let (w, h, pixels) = msg.image.expect("a thumbnail");
        // 200x200 fits within 160x160 → scaled to 160x160.
        assert_eq!((w, h), (160, 160));
        assert_eq!(pixels.len(), (w * h * 4) as usize);
    }

    #[test]
    fn request_thumbnail_without_result_decodes_source() {
        let bm = fixture();
        let source = ImageSource::Bytes(Arc::new(one_pixel_png()));
        bm.request_thumbnail(8, &source, None);

        let msg = bm.bg_io.thumb_rx.recv_timeout(Duration::from_secs(2))
            .expect("thumb_tx must produce a result within 2s");
        assert_eq!(msg.item_id, 8);
        let (w, h, pixels) = msg.image.expect("a thumbnail");
        // 1x1 fits trivially → stays 1x1.
        assert_eq!((w, h), (1, 1));
        assert_eq!(pixels.len(), 4);
    }

    /// A thumbnail that cannot be made still answers, so its request ends:
    /// an image that cannot be decoded left the "Preparing" pill up.
    #[test]
    fn a_thumbnail_that_cannot_be_made_still_answers() {
        let bm = fixture();
        bm.request_thumbnail(9, &ImageSource::Bytes(Arc::new(b"not an image".to_vec())), None);
        let answer = bm.bg_io.thumb_rx.recv_timeout(Duration::from_secs(2)).expect("no answer: the request never ends");
        assert_eq!(answer.item_id, 9);
        assert!(answer.image.is_none());
    }

    // ── fit_dimensions (private helper) ─────────────────────────────────

    /// Scales down preserving aspect, never upscales, never reaches zero.
    #[test]
    fn fit_dimensions_shrinks_only_and_keeps_a_pixel() {
        assert_eq!(fit_dimensions(4000, 2000, 1000, 1000), (1000, 500));
        assert_eq!(fit_dimensions(10, 10, 1000, 1000), (10, 10));
        let (w, h) = fit_dimensions(1000, 1, 50, 50);
        assert!(w >= 1 && h >= 1);
    }

    // ── process_button_label ─────────────────────────────────────────────

    fn checked(id: u64) -> BatchItem {
        let mut i = item_with_cache(id, 0);
        i.ticked = true;
        i
    }

    /// "Process All [1]" reads as a glitch: one item ticked of one stays
    /// Selected; All needs at least two.
    #[test]
    fn process_button_label_follows_the_ticks() {
        use ProcessButtonLabel::*;
        let rows: [(&[bool], ProcessButtonLabel); 6] = [
            (&[], ProcessViewed),
            (&[false, false, false], ProcessViewed),
            (&[true, false, false], ProcessSelected(1)),
            (&[true, true, false], ProcessSelected(2)),
            (&[true, true, true], ProcessAll(3)),
            (&[true], ProcessSelected(1)),
        ];
        for (ticks, want) in rows {
            let mut bm = fixture();
            for (id, &t) in (1..).zip(ticks) {
                bm.items.push(if t { checked(id) } else { item_with_cache(id, 0) });
            }
            assert_eq!(bm.process_button_label(), want, "{ticks:?}");
        }
    }

    // ── items_to_process ─────────────────────────────────────────────────

    /// The three target-set readers agree on ticked items and on the item
    /// on screen; a stale index clamps for Process only, the other two
    /// target nothing.
    #[test]
    fn every_target_reader_picks_the_same_items() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(10, 0));
        bm.items.push(item_with_cache(20, 0));
        bm.items.push(item_with_cache(30, 0));
        bm.selected_index = 1;
        assert_eq!(bm.items_to_process(), vec![20]);
        assert_eq!(bm.targeted_indices(), vec![1]);
        assert!(bm.any_target_can(|i| i.id == 20));
        assert!(!bm.any_target_can(|i| i.id != 20));

        bm.items[0].ticked = true;
        bm.items[2].ticked = true;
        assert_eq!(bm.items_to_process(), vec![10, 30]);
        assert_eq!(bm.targeted_indices(), vec![0, 2]);
        assert!(bm.any_target_can(|i| i.id == 30));
        assert!(!bm.any_target_can(|i| i.id == 20));

        bm.set_all_ticked(false);
        bm.selected_index = 99;
        assert_eq!(bm.items_to_process(), vec![30]);
        assert!(bm.targeted_indices().is_empty());
        assert!(!bm.any_target_can(|_| true));

        let empty = fixture();
        assert!(empty.items_to_process().is_empty());
        assert!(empty.targeted_indices().is_empty());
        assert!(!empty.any_target_can(|_| true));
    }

    #[test]
    fn items_to_process_all_checked_returns_all() {
        let mut bm = fixture();
        bm.items.push(checked(1));
        bm.items.push(checked(2));
        bm.items.push(checked(3));
        let got = bm.items_to_process();
        assert_eq!(got.len(), 3);
    }
    /// Filter-only Process must be off the UI thread — decodes + applies
    /// the subject effect on a background thread; UI learns via `filter_only_rx`.
    #[test]
    fn request_filter_only_emits_processed_rgba_to_filter_only_rx() {
        let bm = fixture();
        let source = ImageSource::Bytes(Arc::new(one_pixel_png()));
        bm.request_filter_only(7, 0, &source, None, prunr_core::SubjectEffect::default());

        let (id, _, result, _) = bm.bg_io.filter_only_rx.recv_timeout(Duration::from_secs(2))
            .expect("filter_only_tx must produce a result within 2s");
        assert_eq!(id, 7);
        let rgba = result.expect("default SubjectEffect on a 1x1 source must succeed");
        assert_eq!(rgba.dimensions(), (1, 1));
    }

    /// Load failures must travel the same channel — the UI's drain maps
    /// `Err` to `BatchStatus::Error`. Without this the drain branch would
    /// be unreachable and bad sources would silently stall in `Processing`.
    #[test]
    fn request_filter_only_emits_load_failure_via_err_variant() {
        let bm = fixture();
        let source = ImageSource::file(std::path::PathBuf::from(
            "/nonexistent/prunr-test-missing.png"
        ));
        bm.request_filter_only(11, 0, &source, None, prunr_core::SubjectEffect::default());

        let (id, _, result, _) = bm.bg_io.filter_only_rx.recv_timeout(Duration::from_secs(2))
            .expect("filter_only_tx must produce a result within 2s");
        assert_eq!(id, 11);
        assert!(result.is_err(), "missing file must surface as Err, not stall");
    }

    // ── select_item ─────────────────────────────────────────────────────

    #[test]
    fn select_item_reports_whether_the_index_changed() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        bm.items.push(item_with_cache(2, 0));
        assert!(!bm.select_item(0));
        assert!(bm.select_item(1));
        assert_eq!(bm.selected_index, 1);
    }

    // ── reorder ─────────────────────────────────────────────────────────

    /// The selected item follows its new position; a no-op move or an
    /// out-of-bounds source leaves the order alone.
    #[test]
    fn reorder_moves_the_item_and_the_selection_follows() {
        let rows: [(usize, usize, usize, [u64; 4], usize); 4] = [
            (1, 0, 3, [20, 30, 10, 40], 0),
            (2, 3, 1, [10, 40, 20, 30], 3),
            (1, 1, 1, [10, 20, 30, 40], 1),
            (1, 5, 0, [10, 20, 30, 40], 1),
        ];
        for (selected, from, to, order, now_selected) in rows {
            let mut bm = fixture();
            for id in [10, 20, 30, 40] { bm.items.push(item_with_cache(id, 0)); }
            bm.selected_index = selected;
            bm.reorder(from, to);
            assert_eq!(bm.items.iter().map(|i| i.id).collect::<Vec<_>>(), order, "{from} -> {to}");
            assert_eq!(bm.selected_index, now_selected, "{from} -> {to}");
        }
    }
    // ── selection lifecycle ─────────────────────────────────────────────

    fn style(fill: f32) -> crate::gui::background_io::SelectionStyle {
        crate::gui::background_io::SelectionStyle::with_fill(fill)
    }

    #[test]
    fn ensure_selection_texture_requests_once_per_key() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        let ctx = egui::Context::default();
        bm.ensure_selection_texture(style(0.15), &ctx);
        assert!(bm.items[0].selection_tex_pending.is_none(), "no selection: nothing to build");

        let mask = prunr_core::selection::MaskArtifact::new_empty(8, 8);
        let hash = mask.content_hash();
        bm.commit_selection(1, mask, crate::gui::timeline::StepKind::Stroke);
        bm.ensure_selection_texture(style(0.15), &ctx);
        assert_eq!(bm.items[0].selection_tex_pending, Some((hash, style(0.15))));
        bm.ensure_selection_texture(style(0.15), &ctx);
        assert_eq!(bm.items[0].selection_tex_pending, Some((hash, style(0.15))), "same key: single-flight");

        // A style change while the first build is in flight re-keys the request.
        bm.ensure_selection_texture(style(0.5), &ctx);
        assert_eq!(bm.items[0].selection_tex_pending, Some((hash, style(0.5))));
    }

    /// The overlay keeps its last texture while a new one builds, so it
    /// never flickers. With no selection left (undo of the first stroke)
    /// nothing replaces it: the texture must go, or the stroke stays drawn.
    #[test]
    fn the_overlay_texture_goes_with_the_last_selection() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        let ctx = egui::Context::default();
        let mask = std::sync::Arc::new(prunr_core::selection::MaskArtifact::new_empty(8, 8));
        let handle = ctx.load_texture("sel", egui::ColorImage::new([8, 8], vec![egui::Color32::WHITE; 64]), Default::default());
        let item = &mut bm.items[0];
        item.selection_texture = Some(crate::gui::item::SelectionTexture {
            key: (1, style(0.15)), texture: handle.into(), shown: mask, feathered: None,
        });
        assert!(item.selection_hash.is_none(), "the selection was undone");
        bm.ensure_selection_texture(style(0.15), &ctx);
        assert!(bm.items[0].selection_texture.is_none());
    }

    /// A cut-out's re-cut applies the selection at the photo's own size,
    /// feathered once the overlay has built the feathered plane; never a
    /// copy shrunk to the model's working size.
    #[test]
    fn a_recut_applies_the_full_size_selection_and_its_feather() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        let ctx = egui::Context::default();
        let cells: Vec<i8> = (0..64).map(|i| if i % 8 < 4 { -127 } else { 0 }).collect();
        let selection = std::sync::Arc::new(prunr_core::selection::MaskArtifact::from_cells(8, 8, cells));
        let feathered = std::sync::Arc::new(prunr_core::selection::MaskArtifact::from_cells(8, 8, vec![-60; 64]));
        let item = &mut bm.items[0];
        item.commit_selection_mask(std::sync::Arc::clone(&selection), crate::gui::timeline::StepKind::Stroke);
        let hash = item.selection_hash.expect("committed");
        let plane = item.correction_plane(style(0.15), crate::gui::brush_state::CutoutAction::Erase).expect("a selection");
        assert_eq!((plane.width, plane.height), (8, 8), "the photo's size, not the model's");
        assert!(std::sync::Arc::ptr_eq(&plane, &selection));

        let handle = ctx.load_texture("sel", egui::ColorImage::new([8, 8], vec![egui::Color32::WHITE; 64]), Default::default());
        item.selection_texture = Some(crate::gui::item::SelectionTexture {
            key: (hash, style(0.15)), texture: handle.into(), shown: selection, feathered: Some(std::sync::Arc::clone(&feathered)),
        });
        assert!(std::sync::Arc::ptr_eq(&item.correction_plane(style(0.15), crate::gui::brush_state::CutoutAction::Erase).unwrap(), &feathered), "the feathered plane once built");
    }

    /// A drag carries the ticked rows when it starts on one, else its own
    /// row; one with a run under way too, as Save writes it.
    #[test]
    fn a_drag_carries_the_ticked_rows_or_its_own() {
        let mut bm = fixture();
        for id in 0..3 {
            bm.items.push(item_with_cache(id, 0));
        }
        bm.items[0].ticked = true;
        bm.items[2].ticked = true;
        assert_eq!(bm.drag_targets(0), [0, 2]);
        assert_eq!(bm.drag_targets(1), [1]);
        crate::gui::tests::fixtures::put_in(&mut bm.items[2], BatchStatus::Processing);
        assert_eq!(bm.drag_targets(0), [0, 2]);
        crate::gui::tests::fixtures::put_in(&mut bm.items[1], BatchStatus::Processing);
        assert_eq!(bm.drag_targets(1), [1], "like Save, a drag writes what the image shows mid-run");
    }

    /// Cancel Selected counts a ticked cut waiting for its worker slot
    /// as well as one under way: both are the user's to stop.
    #[test]
    fn cancel_selected_counts_waiting_and_running_cuts() {
        let mut bm = fixture();
        for id in 0..3 {
            bm.items.push(item_with_cache(id, 0));
        }
        crate::gui::tests::fixtures::put_in(&mut bm.items[1], BatchStatus::Processing);
        bm.items[2].start_cut(Some(crate::gui::timeline::StepKind::Process), crate::gui::item::Cutter::Model(prunr_core::ModelKind::Silueta));
        assert!(bm.ticked_awaiting_cut().next().is_none());
        bm.items[0].ticked = true;
        assert!(bm.ticked_awaiting_cut().next().is_none(), "ticked but nothing awaited");
        bm.items[2].ticked = true;
        assert_eq!(bm.ticked_awaiting_cut().collect::<Vec<_>>(), [2], "a waiting cut counts");
        bm.set_all_ticked(true);
        assert_eq!(bm.ticked_awaiting_cut().collect::<Vec<_>>(), [1, 2]);
        assert!(bm.any_awaiting_cut());
    }

    #[test]
    fn commit_selection_sets_hash_in_lockstep() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        let mask = prunr_core::selection::MaskArtifact::new_empty(32, 32);
        let expected_hash = mask.content_hash();
        assert!(bm.commit_selection(1, mask, crate::gui::timeline::StepKind::Stroke));
        let item = bm.find_by_id(1).unwrap();
        assert_eq!(
            item.selection_hash,
            Some(expected_hash),
            "selection_hash must mirror content_hash() of the committed mask",
        );
        assert!(item.selection_mask.is_some());
    }

    /// A decode answers even when it panics, so the image waiting on it
    /// does not wait for good.
    #[test]
    fn a_decode_that_panics_still_answers() {
        let bm = fixture();
        bm.spawn_decode(1, || panic!("a malformed file"));
        let (id, result) = bm.bg_io.decode_rx.recv_timeout(std::time::Duration::from_secs(10)).expect("an answer");
        assert_eq!(id, 1);
        assert!(result.is_err());
    }
}
