//! Magic Brush coordinator. Owns the tool-active flag, the encoder-pending
//! flag, and the in-progress stroke point buffer for SAM decoder prompts.
//!
//! Does NOT own the selection mask (lives on BatchItem) or the encoder
//! embedding cache (also BatchItem). This struct is purely transient UI
//! state — active tool + "preparing..." spinner gate + stroke buffer.

/// Magic Brush tool coordinator.
#[derive(Default)]
pub(crate) struct MagicBrushState {
    active: bool,
    /// True while Processor::dispatch_sam_encoder is in flight for the
    /// current item. Set on dispatch, cleared by pump_sam_encoder_results
    /// when the embedding lands.
    encoder_pending: bool,
    /// Points collected during an in-progress drag stroke, in SOURCE-IMAGE
    /// pixel coordinates. Populated while the user drags with Magic Brush
    /// active; dispatched on mouse-up via build_stroke_prompt.
    pub(crate) active_stroke: Vec<(f32, f32)>,
    /// In-progress trail stamps in SCREEN coordinates `(x, y, radius)`,
    /// drawn at 60 Hz via `brush_overlay::draw_trail_for` for visual
    /// feedback during the drag. Cleared at the same lifecycle points as
    /// `active_stroke`. Distinct buffer because the SAM prompt path needs
    /// source-pixel points but the visual layer needs zoom-aware screen
    /// pixels — keeping the transform out of the dispatch path means a
    /// pan / zoom mid-stroke doesn't corrupt the SAM points.
    pub(crate) active_trail: Vec<(f32, f32, f32)>,
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

    pub fn deactivate(&mut self) {
        self.active = false;
        self.encoder_pending = false;
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
        assert!(m.active_trail.is_empty());
    }

    #[test]
    fn activate_idempotent() {
        let mut m = MagicBrushState::default();
        assert!(m.activate());   // first call returns true
        assert!(!m.activate());  // second call returns false (no change)
    }

    #[test]
    fn deactivate_clears_pending_flag() {
        let mut m = MagicBrushState::default();
        m.activate();
        m.set_encoder_pending(true);
        m.deactivate();
        assert!(!m.has_pending_encoder());
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
        m.active_trail.push((100.0, 200.0, 8.0));
        m.clear_stroke();
        assert!(m.active_stroke.is_empty());
        assert!(m.active_trail.is_empty());
    }
}
