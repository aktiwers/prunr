//! Magic Brush coordinator. Owns the tool-active flag, the encoder-pending
//! flag, and the in-progress stroke point buffer for SAM decoder prompts.
//!
//! Does NOT own the selection mask (lives on BatchItem) or the encoder
//! embedding cache (also BatchItem). This struct is purely transient UI
//! state — active tool + "preparing..." spinner gate + stroke buffer.

use std::time::{Duration, Instant};

use super::brush_state::Trail;

/// How long an image must stay shown before its embedding is encoded,
/// so paging past images does not start an encode for each (an encode
/// cannot be cancelled).
pub(crate) const ENCODE_SETTLE: Duration = Duration::from_millis(250);

/// Magic Brush tool coordinator.
#[derive(Default)]
pub(crate) struct MagicBrushState {
    active: bool,
    /// A re-threshold is on the pool; the next Confidence value waits for
    /// its result so a drag never queues more than one.
    pub(crate) rethreshold_in_flight: bool,
    /// True while Processor::dispatch_sam_encoder is in flight for the
    /// current item. Set on dispatch, cleared by pump_sam_encoder_results
    /// when the embedding lands.
    encoder_pending: bool,
    /// Points collected during an in-progress drag stroke, in SOURCE-IMAGE
    /// pixel coordinates. Populated while the user drags with Magic Brush
    /// active; dispatched on mouse-up via build_stroke_prompt.
    pub(crate) active_stroke: Vec<(f32, f32)>,
    /// The in-progress trail in SCREEN coordinates, cleared at the same
    /// lifecycle points as `active_stroke`. Distinct buffer because the
    /// SAM prompt path needs source-pixel points but the visual layer
    /// needs zoom-aware screen pixels — keeping the transform out of the
    /// dispatch path means a pan / zoom mid-stroke doesn't corrupt the
    /// SAM points.
    pub(crate) active_trail: Trail,
    /// The image on screen and since when, and the one before it.
    shown: Option<(u64, Instant)>,
    previous: Option<u64>,
}

impl MagicBrushState {
    pub fn is_active(&self) -> bool { self.active }

    /// Activate Magic Brush. Returns true if the active state changed
    /// (caller may need to dispatch encoder).
    pub fn activate(&mut self) -> bool {
        if !self.active {
            self.active = true;
            true
        } else { false }
    }

    /// Turn the tool off. An encode in flight keeps its flag until its
    /// result lands, so no second encode of the same image starts.
    pub fn deactivate(&mut self) {
        self.active = false;
        self.clear_stroke();
    }

    pub fn has_pending_encoder(&self) -> bool { self.encoder_pending }

    pub fn set_encoder_pending(&mut self, pending: bool) {
        self.encoder_pending = pending;
        if !pending {
            // Stroke that arrived while encoder was pending is discarded;
            // user can click/drag again now that the encoder is ready.
            self.clear_stroke();
        }
    }

    /// Record the image on screen; a change starts the encode settle.
    pub(crate) fn note_shown(&mut self, id: Option<u64>, now: Instant) {
        if id != self.shown.map(|(shown, _)| shown) {
            self.previous = self.shown.map(|(shown, _)| shown);
            self.shown = id.map(|id| (id, now));
        }
    }

    /// With the tool on, the image shown before this one keeps its
    /// embedding too, so flipping between two images stays instant.
    pub(crate) fn keeps_embedding(&self, id: u64) -> bool {
        self.active && self.previous == Some(id)
    }

    /// The tool cannot take a click yet: an encode runs, or the shown
    /// image (on with the tool) has no embedding, which covers the settle
    /// before its encode starts.
    pub(crate) fn preparing(&self, shown_has_embedding: bool) -> bool {
        self.encoder_pending || (self.active && !shown_has_embedding)
    }

    /// Time left before the shown image may start encoding.
    pub(crate) fn settle_remaining(&self, now: Instant) -> Option<Duration> {
        let (_, since) = self.shown?;
        ENCODE_SETTLE.checked_sub(now.saturating_duration_since(since)).filter(|d| !d.is_zero())
    }

    /// Clear both the source-coords stroke and the screen-coords trail.
    /// One method keeps the two buffers locked together — a clear of one
    /// without the other would leave a ghost trail or a stuck dispatch.
    pub(crate) fn clear_stroke(&mut self) {
        self.active_stroke.clear();
        self.active_trail.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_inactive() {
        let m = MagicBrushState::default();
        assert!(!m.is_active());
        assert!(!m.has_pending_encoder());
        assert!(m.active_stroke.is_empty());
        assert!(m.active_trail.stamps().next().is_none());
    }

    #[test]
    fn an_encode_waits_for_the_shown_image_to_settle() {
        let mut m = MagicBrushState::default();
        let t0 = Instant::now();
        assert_eq!(m.settle_remaining(t0), None, "nothing shown yet");
        m.note_shown(Some(1), t0);
        assert_eq!(m.settle_remaining(t0), Some(ENCODE_SETTLE));
        m.note_shown(Some(1), t0 + Duration::from_millis(200));
        assert_eq!(m.settle_remaining(t0 + Duration::from_millis(200)), Some(Duration::from_millis(50)), "the same image does not restart it");
        assert_eq!(m.settle_remaining(t0 + ENCODE_SETTLE), None);
        m.note_shown(Some(2), t0 + ENCODE_SETTLE);
        assert!(m.settle_remaining(t0 + ENCODE_SETTLE).is_some(), "a new image restarts it");
        assert!(!m.keeps_embedding(1), "only with the tool on");
        m.activate();
        assert!(m.keeps_embedding(1) && !m.keeps_embedding(2));
    }

    #[test]
    fn activate_idempotent() {
        let mut m = MagicBrushState::default();
        assert!(m.activate());   // first call returns true
        assert!(!m.activate());  // second call returns false (no change)
    }

    /// An encode in flight stays in flight when the tool turns off (Paint
    /// Brush turns Magic off): clearing the flag let the next frame start
    /// a second encode of the same image, and two encodes deadlocked.
    #[test]
    fn turning_the_tool_off_keeps_an_encode_in_flight() {
        let mut m = MagicBrushState::default();
        m.activate();
        m.set_encoder_pending(true);
        m.deactivate();
        assert!(m.has_pending_encoder(), "the encode is still running");
        assert!(!m.is_active());
    }

    /// Pins that `active_stroke` and `active_trail` are cleared in
    /// lockstep — a stale trail on the next stroke would render a ghost
    /// from the previous gesture; a stale stroke would dispatch SAM on
    /// the wrong path. Every entry point (deactivate, set_encoder_pending
    /// false, drag-start, drag-release) must use `clear_stroke()`.
    #[test]
    fn clear_stroke_clears_both_buffers() {
        let mut m = MagicBrushState::default();
        m.active_stroke.push((10.0, 20.0));
        m.active_trail.push_spaced(100.0, 200.0, 8.0);
        m.clear_stroke();
        assert!(m.active_stroke.is_empty());
        assert!(m.active_trail.stamps().next().is_none());
    }
}
