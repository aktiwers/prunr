//! In-process live-preview dispatcher for Tier 2 reruns.
//!
//! When the user tweaks a mask or edge knob in the adjustments toolbar,
//! `mark_tweak` registers a pending preview for the affected item. Every
//! frame, `tick` checks if 300ms have passed since the last tweak for that
//! item and, if so, spawns a background thread to run the Tier 2 postprocess
//! (mask) or edge finalize (edge) on the cached tensor.
//!
//! Cancel + restart semantics: when a new tweak arrives for an item whose
//! preview is still in flight, the old cancel token is flipped; the in-flight
//! worker drops its result on the next polling point (stage boundaries in the
//! postprocess pipeline). The new tweak starts a fresh dispatch.
//!
//! No subprocess involved — Tier 2 is pure CPU work, so running it in-process
//! saves ~20-50ms of IPC overhead per tick and works even outside a batch.
//! The rayon thread pool handles parallel previews across multiple items.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use image::{DynamicImage, RgbaImage};

use prunr_core::{ModelKind, postprocess_from_flat, finalize_edges};

use crate::gui::item_settings::ItemSettings;
use crate::gui::worker::CompressedTensor;

/// Debounce before a preview actually dispatches. Gives fast slider drags
/// a chance to settle so we don't fire 60 previews/second during a drag.
pub const DEBOUNCE: Duration = Duration::from_millis(300);

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
    /// Register a tweak. Debounce resets each call; when the user stops
    /// tweaking for 300ms, `tick` dispatches. A new tweak for the same item
    /// cancels any in-flight dispatch.
    pub fn mark_tweak(&mut self, item_id: u64, kind: PreviewKind) {
        // Cancel any in-flight work for this item — the new tweak will produce
        // a fresh dispatch that supersedes it.
        if let Some(f) = self.in_flight.get(&item_id) {
            f.cancel.store(true, Ordering::Release);
        }
        self.pending.insert(item_id, Pending { last_tweak_at: Instant::now(), kind });
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
            let Some(pending) = self.pending.remove(&id) else { continue };
            let Some(inputs) = snapshot(id, pending.kind) else {
                // No cache available — user must Process first. Silently skip.
                continue;
            };
            self.generation_counter = self.generation_counter.wrapping_add(1);
            let generation = self.generation_counter;
            let cancel = Arc::new(AtomicBool::new(false));
            self.in_flight.insert(id, InFlight { cancel: cancel.clone(), generation });

            let tx = self.result_tx.clone();
            rayon::spawn(move || {
                let result = run_preview(inputs, &cancel);
                if cancel.load(Ordering::Acquire) {
                    return;
                }
                if let Some(rgba) = result {
                    let _ = tx.send(PreviewResult { item_id: id, rgba, generation });
                }
            });
        }

        wait_for
    }

    /// Drain any completed previews. Returned results are already filtered
    /// to the latest generation — callers can apply them directly without
    /// further staleness checks.
    pub fn drain_results(&mut self) -> Vec<PreviewResult> {
        let mut out = Vec::new();
        while let Ok(r) = self.result_rx.try_recv() {
            // Drop stale: if generation doesn't match the last dispatch for
            // this item, a newer tweak superseded it.
            let is_latest = self.in_flight
                .get(&r.item_id)
                .map_or(false, |f| f.generation == r.generation);
            if is_latest {
                self.in_flight.remove(&r.item_id);
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

/// Actually run the preview. Runs on a rayon worker thread. Returns None
/// if the work couldn't be completed (cancelled or bad inputs).
///
/// Preview mode skips the guided filter stage in postprocess — the full-quality
/// refinement runs only on Process commit. Result looks slightly softer during
/// live-drag but full quality is restored the moment the user commits.
fn run_preview(inputs: DispatchInputs, cancel: &AtomicBool) -> Option<RgbaImage> {
    if cancel.load(Ordering::Acquire) { return None; }

    match inputs.kind {
        PreviewKind::Mask => {
            let seg = inputs.seg_tensor?;
            // For live preview we disable guided filter in the MaskSettings
            // copy — keeps preview fast. User re-enables automatically on Process.
            let mut mask_settings = inputs.settings.mask_settings();
            mask_settings.refine_edges = false;
            postprocess_from_flat(
                &seg.data,
                seg.height as usize,
                seg.width as usize,
                &inputs.original,
                &mask_settings,
                seg.model,
            ).ok()
        }
        PreviewKind::Edge => {
            let edge = inputs.edge_tensor?;
            Some(finalize_edges(
                &edge.data,
                edge.height,
                edge.width,
                &inputs.original,
                inputs.settings.line_strength,
                inputs.settings.solid_line_color,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debounce_is_300ms() {
        assert_eq!(DEBOUNCE, Duration::from_millis(300));
    }

    #[test]
    fn mark_tweak_inserts_pending() {
        let mut lp = LivePreview::default();
        lp.mark_tweak(42, PreviewKind::Mask);
        assert!(lp.pending.contains_key(&42));
    }

    #[test]
    fn mark_tweak_twice_cancels_in_flight() {
        let mut lp = LivePreview::default();
        // Simulate an in-flight dispatch
        let cancel = Arc::new(AtomicBool::new(false));
        lp.in_flight.insert(
            7,
            InFlight { cancel: cancel.clone(), generation: 1 },
        );
        lp.mark_tweak(7, PreviewKind::Edge);
        assert!(cancel.load(Ordering::Acquire), "old in-flight should be cancelled");
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
}
