//! Magic Brush coordinator. Owns the tool-active flag, the encoder-pending
//! flag, and the in-progress stroke point buffer for SAM decoder prompts.
//! Mirrors `BrushState` so the [ Paint ]  [ Magic ] tool toggle pair has
//! symmetric ownership.
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
    /// Points collected during an in-progress drag stroke. Populated while
    /// the user drags with Magic Brush active; dispatched on mouse-up via
    /// build_stroke_prompt.
    pub(crate) active_stroke: Vec<(f32, f32)>,
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
        self.active_stroke.clear();
    }

    pub fn has_pending_encoder(&self) -> bool { self.encoder_pending }

    pub fn set_encoder_pending(&mut self, pending: bool) {
        self.encoder_pending = pending;
        if !pending {
            // Stroke that arrived while encoder was pending is discarded;
            // user can click/drag again now that the encoder is ready.
            self.active_stroke.clear();
        }
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
}
