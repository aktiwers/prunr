/// Zoom and pan state for the canvas viewer.
pub struct ZoomState {
    pub zoom: f32,
    pub pan_offset: egui::Vec2,
    pub previous_zoom: f32,
    pub is_panning: bool,
    pub pending_fit_zoom: bool,
    pub pending_actual_size: bool,
}

impl Default for ZoomState {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan_offset: egui::Vec2::ZERO,
            previous_zoom: 1.0,
            is_panning: false,
            pending_fit_zoom: false,
            pending_actual_size: false,
        }
    }
}

impl ZoomState {
    /// Reset to default state, requesting fit-to-window on next frame.
    pub fn reset(&mut self) {
        self.pan_offset = egui::Vec2::ZERO;
        self.previous_zoom = 1.0;
        self.zoom = 1.0;
        self.pending_fit_zoom = true;
    }
}
