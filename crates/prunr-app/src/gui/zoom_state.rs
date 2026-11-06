pub struct ZoomState {
    pub zoom: f32,
    pub pan_offset: egui::Vec2,
    pub previous_zoom: f32,
    pub is_panning: bool,
    pub pending_fit_zoom: bool,
    pub pending_actual_size: bool,
    /// The item shown and the size of its frame, as of the last frame.
    frame: Option<(u64, egui::Vec2)>,
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
            frame: None,
        }
    }
}

impl ZoomState {
    /// Note the frame `item` is shown in. When the same item's frame
    /// changes size (a 4× result lands), the zoom follows so the image
    /// keeps its place on screen.
    pub fn follow_frame(&mut self, item: u64, size: egui::Vec2) {
        if let Some((shown, before)) = self.frame {
            if shown == item && before != size && size.x > 0.0 {
                self.zoom *= before.x / size.x;
                // A zoom to toggle back to belonged to the old size.
                self.previous_zoom = 1.0;
            }
        }
        self.frame = Some((item, size));
    }

    /// Zoom to `zoom` (held within the zoom range) about `cursor` (from the
    /// canvas centre, in screen points): the image point under the cursor
    /// stays under it.
    pub fn zoom_about(&mut self, cursor: egui::Vec2, zoom: f32) {
        let zoom = zoom.clamp(crate::gui::theme::ZOOM_MIN, crate::gui::theme::ZOOM_MAX);
        self.pan_offset = cursor - (cursor - self.pan_offset) * (zoom / self.zoom);
        self.zoom = zoom;
    }

    /// Run a pending Fit to window (`fit` is the fitting zoom) or Actual
    /// size. Pressed again at its zoom it goes back to the zoom before; a
    /// fresh image (no zoom before) always fits.
    pub fn apply_pending(&mut self, fit: f32) {
        if std::mem::take(&mut self.pending_fit_zoom) {
            let may_return = self.previous_zoom != 1.0;
            self.toggle_to(fit, may_return);
        }
        if std::mem::take(&mut self.pending_actual_size) {
            self.toggle_to(1.0, true);
        }
    }

    fn toggle_to(&mut self, target: f32, may_return: bool) {
        if may_return && (self.zoom - target).abs() < 0.001 {
            self.zoom = self.previous_zoom;
        } else {
            self.previous_zoom = self.zoom;
            self.zoom = target;
            self.pan_offset = egui::Vec2::ZERO;
        }
    }

    /// Reset to default state, requesting fit-to-window on next frame.
    pub fn reset(&mut self) {
        self.pan_offset = egui::Vec2::ZERO;
        self.previous_zoom = 1.0;
        self.zoom = 1.0;
        self.pending_fit_zoom = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A result of another size (a 4× upscale) keeps the image where it
    /// was on screen: a fitted image stays fitted, a zoomed-in view keeps
    /// its spot. Another image starts afresh.
    #[test]
    fn a_resized_result_keeps_its_place_on_screen() {
        let canvas = egui::vec2(1000.0, 800.0);
        let source = egui::vec2(800.0, 1200.0);
        let upscaled = source * 4.0;
        let fit = |size: egui::Vec2| (canvas.x / size.x).min(canvas.y / size.y).min(1.0);

        let mut z = ZoomState { zoom: fit(source), pan_offset: egui::vec2(12.0, -4.0), ..ZoomState::default() };
        z.follow_frame(1, source);
        z.follow_frame(1, upscaled);
        assert!((z.zoom - fit(upscaled)).abs() < 1e-6, "fitted stays fitted: {} vs {}", z.zoom, fit(upscaled));
        assert_eq!(z.pan_offset, egui::vec2(12.0, -4.0), "the image stays where it was");
        assert_eq!(z.previous_zoom, 1.0, "Fit to window then fits the new size rather than toggling back");

        let before = z.zoom;
        z.follow_frame(2, source);
        assert_eq!(z.zoom, before, "another image is not rescaled here");
    }

    /// Fit to window and Actual size each go back to the zoom before when
    /// pressed again at their zoom; a fresh image fits and stays fitted.
    #[test]
    fn fit_and_actual_size_toggle_back_to_the_zoom_before() {
        let fit = 0.25;
        let mut z = ZoomState::default();
        z.reset();
        z.pan_offset = egui::vec2(5.0, 5.0);
        z.apply_pending(fit);
        assert_eq!((z.zoom, z.previous_zoom, z.pan_offset), (fit, 1.0, egui::Vec2::ZERO), "a fresh image fits");
        z.pending_fit_zoom = true;
        z.apply_pending(fit);
        assert_eq!(z.zoom, fit, "with no zoom before, Fit stays fitted");

        z.zoom = 2.0;
        z.pan_offset = egui::vec2(30.0, -10.0);
        z.pending_fit_zoom = true;
        z.apply_pending(fit);
        assert_eq!((z.zoom, z.previous_zoom, z.pan_offset), (fit, 2.0, egui::Vec2::ZERO));
        z.pending_fit_zoom = true;
        z.apply_pending(fit);
        assert_eq!(z.zoom, 2.0, "Fit again goes back");

        z.pending_actual_size = true;
        z.apply_pending(fit);
        assert_eq!((z.zoom, z.previous_zoom), (1.0, 2.0));
        z.pending_actual_size = true;
        z.apply_pending(fit);
        assert_eq!(z.zoom, 2.0, "Actual size again goes back");
        assert!(!z.pending_fit_zoom && !z.pending_actual_size, "both requests are spent");
    }

    /// Scroll-zoom keeps the image point under the cursor where it is, at
    /// any zoom: at an upscale's small fit zoom one notch threw the image
    /// far off centre.
    #[test]
    fn zooming_keeps_the_point_under_the_cursor() {
        let cursor = egui::vec2(300.0, 200.0);
        for (zoom, pan) in [(1.0, egui::Vec2::ZERO), (0.16, egui::Vec2::ZERO), (0.16, egui::vec2(-40.0, 25.0)), (3.0, egui::vec2(10.0, -10.0))] {
            let mut z = ZoomState { zoom, pan_offset: pan, ..ZoomState::default() };
            let under = (cursor - z.pan_offset) / z.zoom;
            for factor in [1.1, 1.1, 1.0 / 1.1, 2.0] {
                z.zoom_about(cursor, z.zoom * factor);
                let now = (cursor - z.pan_offset) / z.zoom;
                assert!((now - under).length() < 1e-3, "from {zoom} at {pan:?}: {under:?} moved to {now:?}");
            }
        }
    }
}
