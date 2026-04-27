//! Processing-pipeline coordinator: subprocess worker channels, admission
//! control, live preview, and per-batch dispatch state.
//!
//! Owns:
//! - The worker bridge channels (`worker_tx` / `worker_rx`) — UI thread sends
//!   `WorkerMessage`, drains `WorkerResult` non-blockingly each frame.
//! - The shared cancellation flag (`Arc<AtomicBool>`) — read by the worker
//!   bridge to short-circuit a batch in flight.
//! - The in-process Tier 2 live-preview dispatcher.
//! - Admission controller state during streaming batches.
//! - The dispatch-time recipe snapshot (used to attribute completed results
//!   to the settings that produced them, even if the user keeps editing).
//! - The periodic history-cleanup timestamp.
//!
//! Does NOT own:
//! - The worker bridge thread itself — that's spawned by `worker::spawn_worker`
//!   at app startup. We just hold the channel ends.
//! - `BatchManager` (per the cross-coordinator borrow rule). Methods that
//!   operate on the batch take `&mut BatchManager` per call, never as a field.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use prunr_core::ProcessingRecipe;

use super::live_preview::LivePreview;
use super::memory::AdmissionController;
use super::worker::{WorkerMessage, WorkerResult, WorkItem};

#[derive(Clone)]
pub struct CancelRegistry {
    global: Arc<AtomicBool>,
    // Short-circuit for the common zero-cancel case: `is_cancelled` skips the
    // mutex entirely unless some per-item entry has been requested.
    has_per_item: Arc<AtomicBool>,
    per_item: Arc<Mutex<HashMap<u64, Arc<AtomicBool>>>>,
}

impl CancelRegistry {
    pub(crate) fn new() -> Self {
        Self {
            global: Arc::new(AtomicBool::new(false)),
            has_per_item: Arc::new(AtomicBool::new(false)),
            per_item: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub(crate) fn is_global_cancelled(&self) -> bool {
        self.global.load(Ordering::Acquire)
    }

    pub(crate) fn is_cancelled(&self, item_id: u64) -> bool {
        if self.is_global_cancelled() {
            return true;
        }
        if !self.has_per_item.load(Ordering::Acquire) {
            return false;
        }
        let guard = self.per_item.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.get(&item_id).is_some_and(|f| f.load(Ordering::Acquire))
    }

    pub(crate) fn request_global_cancel(&self) {
        self.global.store(true, Ordering::Release);
    }

    pub(crate) fn request_item_cancel(&self, item_id: u64) {
        let mut guard = self.per_item.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.entry(item_id)
            .or_insert_with(|| Arc::new(AtomicBool::new(false)))
            .store(true, Ordering::Release);
        self.has_per_item.store(true, Ordering::Release);
    }

    pub(crate) fn reset(&self) {
        self.global.store(false, Ordering::Release);
        let mut guard = self.per_item.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.clear();
        self.has_per_item.store(false, Ordering::Release);
    }
}

/// Active dispatch's recipe + the set of items still expected to deliver.
/// All items in a batch share one recipe (the toolbar broadcasts current
/// settings at dispatch). `take_recipe` removes an item; the slot self-
/// cleans when the last pending item completes — so a late ImageDone after
/// a settings edit can't pick up the wrong recipe.
struct InFlightBatch {
    recipe: ProcessingRecipe,
    pending: HashSet<u64>,
}

pub(crate) struct InpaintResult {
    pub item_id: u64,
    pub rgba: image::RgbaImage,
    pub generation: u64,
}

/// Eraser-specific tuning passed from `BrushSettings` into the dispatch.
/// Bundled into a struct to keep `dispatch_inpaint` from sprawling.
#[derive(Clone, Debug)]
pub(crate) struct InpaintTuning {
    pub sharpen: f32,
    pub feather_px: f32,
    pub grow_px: f32,
    /// Which inpaint backend to use (LaMaFp32, BigLaMa, …).
    pub backend: prunr_models::ModelId,
    /// SD-only: text prompt; ignored for LaMa-family backends.
    pub sd_prompt: String,
    pub sd_negative_prompt: String,
    pub sd_guidance_scale: f32,
}

impl Default for InpaintTuning {
    fn default() -> Self {
        Self {
            sharpen: 0.0,
            feather_px: 0.0,
            grow_px: 0.0,
            backend: prunr_models::ModelId::LaMaFp32,
            sd_prompt: String::new(),
            sd_negative_prompt: String::new(),
            sd_guidance_scale: 1.0,
        }
    }
}

pub(crate) struct Processor {
    pub(crate) worker_tx: mpsc::Sender<WorkerMessage>,
    pub(crate) worker_rx: mpsc::Receiver<WorkerResult>,
    /// Cancellation state shared with the worker bridge. `global` stops the
    /// whole batch; per-item entries drop individual items at the next
    /// dispatch check.
    pub(crate) cancels: CancelRegistry,
    pub(crate) live_preview: LivePreview,
    /// Active admission controller (present only during streaming batches).
    pub(crate) admission: Option<AdmissionController>,
    /// Sender for streaming additional items to the worker.
    pub(crate) admission_tx: Option<mpsc::Sender<WorkItem>>,
    /// In-flight batch: recipe + pending IDs. `None` between batches.
    in_flight: Option<InFlightBatch>,
    /// Last time periodic history cleanup ran.
    pub(crate) last_history_cleanup: Instant,
    /// Inpaint dispatch state. Per-item generation counter discards
    /// stale results when the user paints a fresh stroke before the
    /// previous one finishes. `inpaint_pending` is the count of
    /// dispatches not yet drained — the canvas reads it via
    /// `is_inpaint_in_flight` to show a progress overlay.
    inpaint_tx: mpsc::Sender<InpaintResult>,
    inpaint_rx: mpsc::Receiver<InpaintResult>,
    inpaint_latest_gen: HashMap<u64, u64>,
    inpaint_pending: HashMap<u64, u32>,
}

impl Processor {
    pub(crate) fn new(
        worker_tx: mpsc::Sender<WorkerMessage>,
        worker_rx: mpsc::Receiver<WorkerResult>,
    ) -> Self {
        let (inpaint_tx, inpaint_rx) = mpsc::channel();
        Self {
            worker_tx,
            worker_rx,
            cancels: CancelRegistry::new(),
            live_preview: LivePreview::default(),
            admission: None,
            admission_tx: None,
            in_flight: None,
            last_history_cleanup: Instant::now(),
            inpaint_tx,
            inpaint_rx,
            inpaint_latest_gen: HashMap::new(),
            inpaint_pending: HashMap::new(),
        }
    }

    /// Per-item generation counter ensures a fresh stroke supersedes the
    /// previous in-flight job at drain time — see `drain_inpaint_results`.
    pub(crate) fn dispatch_inpaint(
        &mut self,
        item_id: u64,
        image: std::sync::Arc<image::RgbaImage>,
        correction: std::sync::Arc<prunr_core::brush::MaskCorrection>,
        tuning: InpaintTuning,
    ) {
        let generation = self.inpaint_latest_gen.entry(item_id).or_insert(0);
        *generation += 1;
        let gen = *generation;
        *self.inpaint_pending.entry(item_id).or_insert(0) += 1;
        let tx = self.inpaint_tx.clone();
        rayon::spawn(move || {
            let raw_mask = correction.to_binary_mask(image.width(), image.height());
            // Pre-process: grow/erode the painted area before LaMa runs.
            let mask = if tuning.grow_px != 0.0 {
                prunr_core::inpaint::grow_mask(&raw_mask, tuning.grow_px.round() as i32)
            } else {
                raw_mask
            };
            // SD vs LCM: 20 steps + user-CFG for the standard checkpoint;
            // 4 steps + guidance forced to 1.0 for the LCM-distilled
            // backend (LCM bakes guidance into training, runtime CFG
            // would double-count it). use_taesd is set for LCM only —
            // the same fast-mode gate that picked LCM also opts into
            // TAESD VAE; if the TAESD bundle isn't installed yet, the
            // dispatcher silently falls back to standard VAE.
            let sd_req = match tuning.backend {
                prunr_models::ModelId::SdV15InpaintFp16 => Some(prunr_core::inpaint_sd::SdInpaintRequest {
                    prompt: tuning.sd_prompt.clone(),
                    negative_prompt: tuning.sd_negative_prompt.clone(),
                    num_inference_steps: 20,
                    guidance_scale: tuning.sd_guidance_scale,
                    seed: None,
                    use_taesd: false,
                }),
                prunr_models::ModelId::SdV15LcmInpaintFp16 => Some(prunr_core::inpaint_sd::SdInpaintRequest {
                    prompt: tuning.sd_prompt.clone(),
                    negative_prompt: String::new(), // LCM ignores negative prompt
                    num_inference_steps: 4,
                    guidance_scale: 1.0,
                    seed: None,
                    use_taesd: prunr_models::is_available(prunr_models::ModelId::TaesdFp16),
                }),
                _ => None,
            };
            match prunr_core::inpaint::process_inpaint_with(&image, &mask, tuning.backend, sd_req) {
                Ok(rgba) => {
                    // Post-process: sharpen first (LaMa's blur is the
                    // pixel content), then feather the boundary so the
                    // sharpen doesn't crisp up an already-soft seam.
                    let mut out = rgba;
                    if tuning.sharpen > 0.0 {
                        out = prunr_core::inpaint::sharpen_inpainted(&out, &mask, tuning.sharpen);
                    }
                    if tuning.feather_px > 0.0 {
                        out = prunr_core::inpaint::feather_inpainted(&out, &image, &mask, tuning.feather_px);
                    }
                    let _ = tx.send(InpaintResult { item_id, rgba: out, generation: gen });
                }
                Err(e) => {
                    tracing::error!(item_id, %e, "inpaint dispatch failed");
                    // Best-effort: send an empty marker so pending count
                    // decrements. Use generation 0 so it's always treated
                    // as stale and the result itself is dropped.
                    let _ = tx.send(InpaintResult {
                        item_id,
                        rgba: image::RgbaImage::new(0, 0),
                        generation: 0,
                    });
                }
            }
        });
    }

    pub(crate) fn drain_inpaint_results(&mut self) -> Vec<InpaintResult> {
        let mut out = Vec::new();
        while let Ok(result) = self.inpaint_rx.try_recv() {
            // Every drained result decrements pending — stale ones
            // count too, since the rayon job that produced them has
            // run to completion.
            if let Some(c) = self.inpaint_pending.get_mut(&result.item_id) {
                *c = c.saturating_sub(1);
            }
            let latest = self.inpaint_latest_gen.get(&result.item_id).copied().unwrap_or(0);
            if result.generation == latest && result.generation > 0 {
                out.push(result);
            }
        }
        out
    }

    /// True while a dispatched inpaint job hasn't drained yet for `item_id`.
    /// Canvas reads this to render a "Erasing..." overlay during LaMa work.
    pub(crate) fn is_inpaint_in_flight(&self, item_id: u64) -> bool {
        self.inpaint_pending.get(&item_id).copied().unwrap_or(0) > 0
    }

    /// True while ANY item has an in-flight inpaint job. Status bar
    /// reads this to override the "All done" text during LaMa work.
    pub(crate) fn any_inpaint_in_flight(&self) -> bool {
        self.inpaint_pending.values().any(|&c| c > 0)
    }

    /// Register a batch's recipe + the IDs that should deliver against it.
    /// Replaces any prior in-flight state — callers ensure prior batches
    /// have completed before firing a new dispatch.
    pub(crate) fn track_dispatch(
        &mut self,
        recipe: ProcessingRecipe,
        ids: impl IntoIterator<Item = u64>,
    ) {
        self.in_flight = Some(InFlightBatch {
            recipe,
            pending: ids.into_iter().collect(),
        });
    }

    /// Add a streamed (admission-pool) item to the current batch. The
    /// `debug_assert` catches the "admission ran without a tracked batch"
    /// invariant breach in tests; release builds silently no-op so a
    /// single late delivery can't take down a real batch.
    pub(crate) fn track_streamed(&mut self, id: u64) {
        match self.in_flight.as_mut() {
            Some(b) => { b.pending.insert(id); }
            None => debug_assert!(false, "track_streamed called without active batch"),
        }
    }

    /// Take the recipe for a finished item. Returns `None` when the item
    /// wasn't in flight (late delivery after cancel/drain) — caller falls
    /// back. Self-cleans the in-flight slot when the last item completes.
    pub(crate) fn take_recipe(&mut self, id: u64) -> Option<ProcessingRecipe> {
        let batch = self.in_flight.as_mut()?;
        if !batch.pending.remove(&id) {
            return None;
        }
        let recipe = batch.recipe.clone();
        if batch.pending.is_empty() {
            self.in_flight = None;
        }
        Some(recipe)
    }

    /// Drop the in-flight slot regardless of pending. Called on user cancel
    /// or batch-complete signals so a late delivery can't reattribute.
    pub(crate) fn drain_recipes(&mut self) {
        self.in_flight = None;
    }

    /// Drop admission state so no further items are admitted. Called on
    /// cancel (user or worker-side) and by the cancelled-message handler.
    /// Leaves the cancel registry untouched — that's owned by the caller's
    /// cancel protocol.
    pub(crate) fn clear_admission(&mut self) {
        self.admission = None;
        self.admission_tx = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Processor {
        let (tx, _rx_unused) = mpsc::channel::<WorkerMessage>();
        let (_tx_unused, rx) = mpsc::channel::<WorkerResult>();
        Processor::new(tx, rx)
    }

    #[test]
    fn new_initialises_last_history_cleanup_recent() {
        // Verifies the periodic 600s cleanup gate isn't accidentally
        // triggered at startup — the Instant must be effectively-now.
        let p = fixture();
        assert!(p.last_history_cleanup.elapsed().as_secs() < 5);
    }

    #[test]
    fn drain_filters_stale_generations() {
        let mut p = fixture();
        p.inpaint_latest_gen.insert(7, 2);
        p.inpaint_tx.send(InpaintResult {
            item_id: 7, rgba: image::RgbaImage::new(1, 1), generation: 1,
        }).unwrap();
        p.inpaint_tx.send(InpaintResult {
            item_id: 7, rgba: image::RgbaImage::new(1, 1), generation: 2,
        }).unwrap();
        let drained = p.drain_inpaint_results();
        assert_eq!(drained.len(), 1, "stale gen=1 must be dropped");
        assert_eq!(drained[0].generation, 2);
    }

    #[test]
    fn clear_admission_drops_both_sides_but_leaves_cancels() {
        let mut p = fixture();
        let (tx, _rx) = mpsc::channel::<WorkItem>();
        p.admission_tx = Some(tx);
        p.cancels.request_global_cancel();
        assert!(p.admission_tx.is_some());

        p.clear_admission();

        assert!(p.admission.is_none());
        assert!(p.admission_tx.is_none());
        assert!(p.cancels.is_cancelled(999),
            "clear_admission must leave cancel registry untouched — that's the caller's protocol");
    }

    #[test]
    fn cancel_registry_clone_shares_state() {
        // Cloned into WorkerMessage::BatchProcess and read by the bridge —
        // a store on the parent must be visible via any clone.
        let r = CancelRegistry::new();
        let handle = r.clone();
        assert!(!handle.is_cancelled(5));
        r.request_global_cancel();
        assert!(handle.is_cancelled(5), "Clone must observe the global store");
    }

    #[test]
    fn cancel_registry_per_item_is_independent_of_global() {
        let r = CancelRegistry::new();
        r.request_item_cancel(42);
        assert!(r.is_cancelled(42));
        assert!(!r.is_cancelled(7), "per-item cancel must not leak to other ids");
    }

    #[test]
    fn cancel_registry_reset_clears_all_flags() {
        let r = CancelRegistry::new();
        r.request_global_cancel();
        r.request_item_cancel(42);
        r.reset();
        assert!(!r.is_cancelled(42));
        assert!(!r.is_cancelled(99));
    }

    #[test]
    fn global_cancel_short_circuits_per_item_lookup() {
        let r = CancelRegistry::new();
        r.request_global_cancel();
        // Any id reports cancelled when global is set, even ones with no per-item entry.
        assert!(r.is_cancelled(u64::MAX));
    }

    fn fixture_recipe() -> ProcessingRecipe {
        use prunr_core::{
            CompositeRecipe, EdgeRecipe, EdgeScale, ComposeMode, FillStyle, InferenceRecipe,
            InputTransform, LineStyle, MaskSettings, ModelKind,
        };
        ProcessingRecipe {
            inference: InferenceRecipe {
                model: ModelKind::Silueta,
                uses_segmentation: true,
                uses_edge_detection: false,
                input_transform: InputTransform::None,
            },
            edge: EdgeRecipe {
                line_strength_bits: 0.5f32.to_bits(),
                solid_line_color: None,
                edge_thickness: 0,
                edge_scale: EdgeScale::Fused,
                compose_mode: ComposeMode::LinesOnly,
                line_style: LineStyle::Solid,
            },
            mask: (&MaskSettings { fill_style: FillStyle::None, ..Default::default() }).into(),
            composite: CompositeRecipe { bg_color: None, solid_line_color: None },
            was_chain: false,
        }
    }

    #[test]
    fn track_dispatch_then_take_returns_recipe_per_item() {
        let mut p = fixture();
        p.track_dispatch(fixture_recipe(), [10, 20, 30].iter().copied());
        assert!(p.take_recipe(10).is_some());
        assert!(p.take_recipe(20).is_some());
        // Slot still alive while items remain.
        assert!(p.in_flight.is_some());
        assert!(p.take_recipe(30).is_some());
        // Last item drains the slot.
        assert!(p.in_flight.is_none());
    }

    #[test]
    fn take_recipe_unknown_id_is_none() {
        let mut p = fixture();
        p.track_dispatch(fixture_recipe(), [1].iter().copied());
        assert!(p.take_recipe(999).is_none(), "unknown id must not return a recipe");
        // Tracked id still works.
        assert!(p.take_recipe(1).is_some());
    }

    #[test]
    fn track_streamed_inherits_batch_recipe() {
        // Admission-pool items are added after dispatch; they inherit the
        // current batch's recipe so a late ImageDone for a streamed id
        // still attributes correctly.
        let mut p = fixture();
        p.track_dispatch(fixture_recipe(), [1].iter().copied());
        p.track_streamed(2);
        assert!(p.take_recipe(1).is_some());
        assert!(p.take_recipe(2).is_some(), "streamed item must have a recipe");
    }

    #[test]
    #[should_panic(expected = "track_streamed called without active batch")]
    fn track_streamed_without_dispatch_panics_in_debug() {
        let mut p = fixture();
        p.track_streamed(99);
    }

    #[test]
    fn drain_recipes_clears_pending() {
        let mut p = fixture();
        p.track_dispatch(fixture_recipe(), [1, 2, 3].iter().copied());
        p.drain_recipes();
        assert!(p.take_recipe(1).is_none(),
            "drain must drop the slot so late deliveries fall back");
    }

    #[test]
    fn zero_cancel_is_cancelled_does_not_touch_mutex() {
        // `is_cancelled` is called ~160×/s from the bridge loop. Until any
        // per-item entry is requested the mutex must stay cold — poisoning
        // the map from another thread and then calling `is_cancelled` on a
        // fresh registry must not panic.
        let r = CancelRegistry::new();
        // Poison the inner mutex from a panicking thread.
        let p = r.per_item.clone();
        let _ = std::thread::spawn(move || {
            let _guard = p.lock().unwrap();
            panic!("deliberate poison");
        }).join();
        // has_per_item is still false → no lock taken → no panic propagation.
        assert!(!r.is_cancelled(42));
    }
}
