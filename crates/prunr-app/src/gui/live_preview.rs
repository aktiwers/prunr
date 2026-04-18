//! In-process live-preview dispatcher for Tier 2 reruns.
//!
//! When the user tweaks a mask or edge knob in the adjustments toolbar,
//! `mark_tweak` registers a pending preview for the affected item. Every
//! frame, `tick` checks if 300ms have passed since the last tweak for that
//! item and, if so, spawns a background thread to run the Tier 2 postprocess
//! (mask) or edge finalize (edge) on the cached tensor.
//!
//! Staleness handling: if multiple dispatches are in flight for the same item
//! (possible during a continuous drag — a new dispatch fires every DEBOUNCE
//! while earlier ones are still running), `drain_results` drops any result
//! whose generation doesn't match the most-recently-dispatched generation. The
//! cancel token is still held per-dispatch but is only triggered on
//! `cancel_all` (batch clear / shutdown) — letting in-flight dispatches
//! complete during a drag is what produces live updates mid-drag.
//!
//! **Do not reintroduce per-tweak cancellation.** It was removed on purpose:
//! `postprocess` doesn't check the cancel token at stage boundaries, so the
//! token only gates the final `tx.send`. Cancelling mid-drag therefore dropped
//! every dispatch's result before it could be drained, which is exactly the
//! "no live preview during drag" bug this module was rewritten to fix.
//!
//! No subprocess involved — Tier 2 is pure CPU work, so running it in-process
//! saves ~20-50ms of IPC overhead per tick and works even outside a batch.
//! The rayon thread pool handles parallel previews across multiple items.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use image::{DynamicImage, GrayImage, RgbaImage};

use prunr_core::{ModelKind, postprocess_from_flat, tensor_to_edge_mask, compose_edges};

use crate::gui::item_settings::ItemSettings;
use crate::gui::worker::CompressedTensor;

/// Debounce before a preview actually dispatches. Short enough to feel
/// responsive on slider release; long enough to coalesce fast drag events
/// into a single rerun.
pub const DEBOUNCE: Duration = Duration::from_millis(150);

/// What kind of Tier 2 rerun a tweak needs. Two kinds because they touch
/// different cached tensors (segmentation vs DexiNed) and different pipeline
/// stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewKind {
    /// gamma / threshold / edge_shift / refine_edges → Tier 2 mask rerun.
    Mask,
    /// line_strength / solid_line_color → Tier 2 edge rerun.
    Edge,
}

/// A completed Tier 2 rerun delivered back to the UI thread.
pub struct PreviewResult {
    pub item_id: u64,
    pub rgba: RgbaImage,
    /// Generation counter — used to discard results from stale dispatches
    /// that completed after a newer tweak cancelled them.
    pub generation: u64,
    /// Edge mask built during this dispatch, for the parent to cache so
    /// subsequent edge_thickness / solid_line_color tweaks skip the resize.
    /// Some only when Kind was Edge and the mask was built (not reused).
    pub new_edge_mask: Option<(Arc<GrayImage>, u32 /* line_strength bits */)>,
    /// `true` when no further tweaks are pending for this item at drain
    /// time — the drag has settled and this is the last result of the
    /// session. Callers gate heavy side-effects (sidebar thumb rebuild)
    /// on this so the mid-drag sidebar doesn't flicker.
    pub is_final: bool,
}

/// State for a pending (debounced) preview dispatch.
struct Pending {
    last_tweak_at: Instant,
    kind: PreviewKind,
}

/// State for an in-flight dispatch. Used to cancel when a new tweak arrives.
struct InFlight {
    cancel: Arc<AtomicBool>,
    generation: u64,
}

pub struct LivePreview {
    pending: HashMap<u64, Pending>,
    in_flight: HashMap<u64, InFlight>,
    generation_counter: u64,
    result_tx: mpsc::Sender<PreviewResult>,
    result_rx: mpsc::Receiver<PreviewResult>,
}

impl Default for LivePreview {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            pending: HashMap::new(),
            in_flight: HashMap::new(),
            generation_counter: 0,
            result_tx: tx,
            result_rx: rx,
        }
    }
}

impl LivePreview {
    /// Register a tweak. During a continuous drag (many tweaks per frame), the
    /// debounce timer is only reset once per DEBOUNCE window — so dispatches
    /// actually fire mid-drag rather than being postponed indefinitely. In-
    /// flight dispatches are left alone; `drain_results` drops stale ones via
    /// the generation filter, and the next tick starts a fresh dispatch once
    /// the window elapses again.
    pub fn mark_tweak(&mut self, item_id: u64, kind: PreviewKind) {
        let now = Instant::now();
        match self.pending.get_mut(&item_id) {
            Some(p) => {
                // Cap dispatch cadence during a drag: only re-arm the timer if
                // the previous arm has already expired (and thus dispatched).
                // Continuous mid-drag tweaks then produce ~one dispatch per
                // DEBOUNCE window instead of holding the timer open forever.
                if now.saturating_duration_since(p.last_tweak_at) >= DEBOUNCE {
                    p.last_tweak_at = now;
                }
                p.kind = kind;
            }
            None => {
                self.pending.insert(item_id, Pending { last_tweak_at: now, kind });
            }
        }
    }

    /// Flush: expire the pending tweak timer so the next `tick` dispatches
    /// immediately. Used when an edit settles (slider released, checkbox
    /// toggled, color picked) so the user doesn't wait the full debounce.
    ///
    /// Idempotent — no-op if there's no pending tweak for `item_id`.
    pub fn flush(&mut self, item_id: u64) {
        if let Some(p) = self.pending.get_mut(&item_id) {
            // Anti-date past DEBOUNCE so next tick dispatches immediately.
            // checked_sub guards against early-boot monotonic clock underflow.
            p.last_tweak_at = Instant::now()
                .checked_sub(DEBOUNCE + Duration::from_millis(10))
                .unwrap_or_else(Instant::now);
        }
    }

    /// Per-frame tick. Returns the time until the next tween we're waiting on
    /// (useful for `ctx.request_repaint_after`), or `None` if nothing pending.
    ///
    /// `snapshot` provides per-item inputs (tensors, settings, original image)
    /// for any dispatches we fire this frame. Items that aren't in the
    /// snapshot (e.g. tensor cache evicted) are silently skipped.
    pub fn tick<F>(&mut self, mut snapshot: F) -> Option<Duration>
    where
        F: FnMut(u64, PreviewKind) -> Option<DispatchInputs>,
    {
        let now = Instant::now();
        let mut ready_ids: Vec<u64> = Vec::new();
        let mut wait_for: Option<Duration> = None;

        for (id, p) in &self.pending {
            let elapsed = now.saturating_duration_since(p.last_tweak_at);
            if elapsed >= DEBOUNCE {
                ready_ids.push(*id);
            } else {
                let remaining = DEBOUNCE - elapsed;
                wait_for = Some(wait_for.map_or(remaining, |w| w.min(remaining)));
            }
        }

        for id in ready_ids {
            // Peek the pending kind without removing — if the snapshot isn't
            // ready yet (e.g. `source_rgba` decode in flight after a view
            // switch), we keep the pending entry so the next tick retries.
            // Dropping it here silently swallowed the tweak and left the
            // user thinking live preview was dead.
            let kind = match self.pending.get(&id) {
                Some(p) => p.kind,
                None => continue,
            };
            let Some(inputs) = snapshot(id, kind) else {
                // Snapshot couldn't assemble inputs this frame. Possibilities:
                // (a) `source_rgba` is None (decode pending after view
                //     switch) — retry once the decode lands.
                // (b) required tensor cache is permanently absent (item
                //     never processed) — the retry is cheap and the pending
                //     entry stays inert until the user clicks Process or
                //     stops tweaking; no harm.
                continue;
            };
            self.pending.remove(&id);
            self.generation_counter = self.generation_counter.wrapping_add(1);
            let generation = self.generation_counter;
            let cancel = Arc::new(AtomicBool::new(false));
            self.in_flight.insert(id, InFlight { cancel: cancel.clone(), generation });

            let tx = self.result_tx.clone();
            rayon::spawn(move || {
                let ls_bits = inputs.settings.line_strength.to_bits();
                let is_edge = matches!(inputs.kind, PreviewKind::Edge);
                let had_cache = inputs.cached_edge_mask.is_some();
                let (result, built_mask) = run_preview(inputs, &cancel);
                if cancel.load(Ordering::Acquire) {
                    return;
                }
                if let Some(rgba) = result {
                    // Publish the mask for caching only if we just built it
                    // (cache miss path) — no point re-publishing what was cached.
                    let new_edge_mask = if is_edge && !had_cache {
                        built_mask.map(|m| (m, ls_bits))
                    } else {
                        None
                    };
                    // `is_final` is set by `drain_results`, where the UI
                    // thread can read `self.pending` atomically. The worker
                    // ships a placeholder and doesn't care.
                    let _ = tx.send(PreviewResult {
                        item_id: id, rgba, generation, new_edge_mask,
                        is_final: false,
                    });
                }
            });
        }

        wait_for
    }

    /// Drain any completed previews. Returned results are already filtered
    /// to the latest generation; each result's `is_final` is set to `true`
    /// iff no new tweak for that item is currently pending — i.e. the drag
    /// has settled and this is the last result of the session.
    pub fn drain_results(&mut self) -> Vec<PreviewResult> {
        let mut out = Vec::new();
        while let Ok(mut r) = self.result_rx.try_recv() {
            // Drop stale: if generation doesn't match the last dispatch for
            // this item, a newer tweak superseded it.
            let is_latest = self.in_flight
                .get(&r.item_id)
                .map_or(false, |f| f.generation == r.generation);
            if is_latest {
                self.in_flight.remove(&r.item_id);
                // If pending has a fresh entry for this item, the user is
                // still mid-drag (a tweak landed after the current dispatch
                // started). Empty pending = drag settled = final result.
                r.is_final = !self.pending.contains_key(&r.item_id);
                out.push(r);
            }
        }
        out
    }

    /// Cancel all in-flight + pending previews. Called on batch clear or shutdown.
    pub fn cancel_all(&mut self) {
        for f in self.in_flight.values() {
            f.cancel.store(true, Ordering::Release);
        }
        self.in_flight.clear();
        self.pending.clear();
    }
}

/// Inputs required to actually run a single preview. Snapshot captures these
/// on the UI thread (holding the item briefly) and hands them to the worker
/// so the worker doesn't need a `&mut BatchItem`.
pub struct DispatchInputs {
    pub kind: PreviewKind,
    pub original: Arc<DynamicImage>,
    pub settings: ItemSettings,
    /// Segmentation tensor (decompressed) + dimensions + model. Required for
    /// `PreviewKind::Mask` *or* `PreviewKind::Edge` in SubjectOutline mode
    /// (the edge was computed on top of a segmented subject; preview reuses
    /// that as the base image for finalize_edges).
    pub seg_tensor: Option<SegTensor>,
    pub edge_tensor: Option<EdgeTensor>,
    /// Pre-built edge mask (post-resize, pre-dilation) from a previous dispatch
    /// whose line_strength matches the current one. Populated when available
    /// so tweaks to edge_thickness / solid_line_color skip the resize.
    pub cached_edge_mask: Option<Arc<GrayImage>>,
}

pub struct SegTensor {
    pub data: Vec<f32>,
    pub height: u32,
    pub width: u32,
    pub model: ModelKind,
}

pub struct EdgeTensor {
    pub data: Vec<f32>,
    pub height: u32,
    pub width: u32,
}

/// Helper: decompress a `CompressedTensor` into the raw form the preview worker needs.
pub fn decompress_seg(ct: &CompressedTensor) -> Option<SegTensor> {
    Some(SegTensor {
        data: ct.decompress()?,
        height: ct.height,
        width: ct.width,
        model: ct.model,
    })
}

pub fn decompress_edge(ct: &CompressedTensor) -> Option<EdgeTensor> {
    Some(EdgeTensor {
        data: ct.decompress()?,
        height: ct.height,
        width: ct.width,
    })
}

/// Actually run the preview. Runs on a rayon worker thread. Returns the RGBA
/// plus, for Edge kind, the resized pre-dilation mask so the parent can cache
/// it (tied to the dispatch's line_strength).
///
/// Respects the user's full mask settings, including `refine_edges`. The
/// guided filter stage adds ~50-150ms on 4K when enabled, so drag cadence
/// slows accordingly — but the user opted into that cost when they toggled
/// Refine Edges on, and without running the filter the three refine knobs
/// (toggle, guided_radius, guided_epsilon) would be no-ops in live preview.
fn run_preview(inputs: DispatchInputs, cancel: &AtomicBool) -> (Option<RgbaImage>, Option<Arc<GrayImage>>) {
    if cancel.load(Ordering::Acquire) { return (None, None); }

    match inputs.kind {
        PreviewKind::Mask => {
            let Some(seg) = inputs.seg_tensor else { return (None, None); };
            let mask_settings = inputs.settings.mask_settings();
            let rgba = postprocess_from_flat(
                &seg.data,
                seg.height as usize,
                seg.width as usize,
                &inputs.original,
                &mask_settings,
                seg.model,
            ).ok();
            (rgba, None)
        }
        PreviewKind::Edge => {
            let Some(edge) = inputs.edge_tensor else { return (None, None); };
            let edge_settings = inputs.settings.edge_settings();
            // Fast path: cached mask still valid for this line_strength.
            // Skip sigmoid + Lanczos resize (~40-80ms on 4K) and go straight
            // to dilate + composite.
            let (mask, built) = if let Some(m) = inputs.cached_edge_mask {
                (m, None)
            } else {
                let m = Arc::new(tensor_to_edge_mask(
                    &edge.data,
                    edge.height,
                    edge.width,
                    inputs.original.width(),
                    inputs.original.height(),
                    edge_settings.line_strength,
                ));
                (m.clone(), Some(m))
            };
            let rgba = compose_edges(
                &mask,
                &inputs.original,
                edge_settings.solid_line_color,
                edge_settings.edge_thickness,
            );
            (Some(rgba), built)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debounce_is_reasonable() {
        // Balance: short enough to feel responsive, long enough to coalesce
        // mid-drag change events into one dispatch.
        assert!(DEBOUNCE >= Duration::from_millis(100));
        assert!(DEBOUNCE <= Duration::from_millis(300));
    }

    #[test]
    fn mark_tweak_inserts_pending() {
        let mut lp = LivePreview::default();
        lp.mark_tweak(42, PreviewKind::Mask);
        assert!(lp.pending.contains_key(&42));
    }

    #[test]
    fn mark_tweak_does_not_cancel_in_flight() {
        // Mid-drag behaviour: a running dispatch must be allowed to complete
        // so the user sees a preview update; `drain_results`' generation
        // filter drops results that were superseded.
        let mut lp = LivePreview::default();
        let cancel = Arc::new(AtomicBool::new(false));
        lp.in_flight.insert(
            7,
            InFlight { cancel: cancel.clone(), generation: 1 },
        );
        lp.mark_tweak(7, PreviewKind::Edge);
        assert!(
            !cancel.load(Ordering::Acquire),
            "in-flight dispatch must NOT be cancelled by a subsequent tweak",
        );
    }

    #[test]
    fn mark_tweak_does_not_reset_timer_mid_drag() {
        // Continuous-drag scenario: multiple tweaks within DEBOUNCE of each
        // other must not keep re-arming the timer, or the dispatch would
        // never fire until the user stopped moving.
        let mut lp = LivePreview::default();
        lp.mark_tweak(1, PreviewKind::Mask);
        let first_arm = lp.pending.get(&1).expect("armed").last_tweak_at;

        // Simulate a second tweak "soon after" (short of DEBOUNCE). The timer
        // should NOT move — otherwise the dispatch would be pushed out.
        std::thread::sleep(Duration::from_millis(5));
        lp.mark_tweak(1, PreviewKind::Mask);
        let second_arm = lp.pending.get(&1).expect("still armed").last_tweak_at;

        assert_eq!(
            first_arm, second_arm,
            "mid-drag tweaks must leave the original arm time in place",
        );
    }

    #[test]
    fn tick_before_debounce_does_not_dispatch() {
        let mut lp = LivePreview::default();
        lp.mark_tweak(1, PreviewKind::Mask);
        let wait = lp.tick(|_, _| None);
        assert!(wait.is_some(), "should still be pending");
        assert!(lp.in_flight.is_empty(), "no dispatch before debounce expires");
    }

    #[test]
    fn cancel_all_clears_state() {
        let mut lp = LivePreview::default();
        lp.mark_tweak(1, PreviewKind::Mask);
        lp.in_flight.insert(
            2,
            InFlight { cancel: Arc::new(AtomicBool::new(false)), generation: 1 },
        );
        lp.cancel_all();
        assert!(lp.pending.is_empty());
        assert!(lp.in_flight.is_empty());
    }

    #[test]
    fn stale_generation_result_is_dropped() {
        // In-flight carries generation 5 (latest dispatch). A stale worker
        // finishes with generation 1 — drain_results must drop it silently.
        let mut lp = LivePreview::default();
        lp.in_flight.insert(
            42,
            InFlight { cancel: Arc::new(AtomicBool::new(false)), generation: 5 },
        );
        lp.result_tx.send(PreviewResult {
            item_id: 42,
            rgba: RgbaImage::new(1, 1),
            generation: 1,
            new_edge_mask: None,
            is_final: false,
        }).unwrap();

        let drained = lp.drain_results();
        assert!(drained.is_empty(), "stale generation must not be returned");
        assert!(
            lp.in_flight.contains_key(&42),
            "in-flight entry is preserved — the real dispatch hasn't completed",
        );
    }

    #[test]
    fn matching_generation_result_is_accepted_and_clears_in_flight() {
        let mut lp = LivePreview::default();
        lp.in_flight.insert(
            99,
            InFlight { cancel: Arc::new(AtomicBool::new(false)), generation: 7 },
        );
        lp.result_tx.send(PreviewResult {
            item_id: 99,
            rgba: RgbaImage::new(2, 2),
            generation: 7,
            new_edge_mask: None,
            is_final: false,
        }).unwrap();

        let drained = lp.drain_results();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].item_id, 99);
        assert!(
            !lp.in_flight.contains_key(&99),
            "once a fresh result is accepted, the in-flight slot is cleared",
        );
    }

    #[test]
    fn result_for_unknown_item_is_dropped() {
        // No in-flight entry at all (user cancel_all'd or never dispatched).
        // A late result arriving for an unknown id must be dropped cleanly.
        let mut lp = LivePreview::default();
        lp.result_tx.send(PreviewResult {
            item_id: 777,
            rgba: RgbaImage::new(1, 1),
            generation: 1,
            new_edge_mask: None,
            is_final: false,
        }).unwrap();
        let drained = lp.drain_results();
        assert!(drained.is_empty());
    }

    #[test]
    fn drain_sets_is_final_true_when_no_more_pending() {
        // User released the knob — no new pending entry for this item.
        let mut lp = LivePreview::default();
        lp.in_flight.insert(
            11,
            InFlight { cancel: Arc::new(AtomicBool::new(false)), generation: 3 },
        );
        lp.result_tx.send(PreviewResult {
            item_id: 11,
            rgba: RgbaImage::new(1, 1),
            generation: 3,
            new_edge_mask: None,
            is_final: false,
        }).unwrap();

        let drained = lp.drain_results();
        assert_eq!(drained.len(), 1);
        assert!(drained[0].is_final, "empty pending at drain time → result is final");
    }

    #[test]
    fn drain_sets_is_final_false_while_user_still_tweaking() {
        // User is still dragging — a new tweak has landed in pending since
        // the dispatch was started.
        let mut lp = LivePreview::default();
        lp.in_flight.insert(
            22,
            InFlight { cancel: Arc::new(AtomicBool::new(false)), generation: 2 },
        );
        lp.pending.insert(
            22,
            Pending { last_tweak_at: Instant::now(), kind: PreviewKind::Mask },
        );
        lp.result_tx.send(PreviewResult {
            item_id: 22,
            rgba: RgbaImage::new(1, 1),
            generation: 2,
            new_edge_mask: None,
            is_final: false,
        }).unwrap();

        let drained = lp.drain_results();
        assert_eq!(drained.len(), 1);
        assert!(!drained[0].is_final, "pending for this item means drag is still active");
    }
}
