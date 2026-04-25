//! Brush tool coordinator. Owns the toolbar toggle, current size /
//! hardness / mode, and the in-progress stroke buffer. Doesn't iterate
//! `BatchManager.items` — the caller hands it the active grid size
//! and writes the committed strokes back via `BatchItem`'s mutator.

use prunr_core::brush::{paint_circle, BrushMode, MaskCorrection};

#[derive(Clone, Copy, Debug)]
pub(crate) struct BrushSettings {
    pub radius: f32,
    pub hardness: f32,
    pub mode: BrushMode,
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            radius: 24.0,
            hardness: 0.7,
            mode: BrushMode::Subtract,
        }
    }
}

/// Mid-drag stroke buffer at the active item's model resolution.
struct ActiveStroke {
    grid: MaskCorrection,
    /// Set the first time `paint_circle` runs against `grid`. Lets
    /// `commit_stroke` skip an O(W·H) is_empty scan on click-without-drag.
    dirty: bool,
}

#[derive(Default)]
pub(crate) struct BrushState {
    enabled: bool,
    settings: BrushSettings,
    active: Option<ActiveStroke>,
}

impl BrushState {
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
        if !self.enabled {
            self.active = None;
        }
    }

    pub fn settings(&self) -> &BrushSettings {
        &self.settings
    }

    pub fn settings_mut(&mut self) -> &mut BrushSettings {
        &mut self.settings
    }

    /// True while the user is mid-drag.
    pub fn has_active_stroke(&self) -> bool {
        self.active.is_some()
    }

    pub fn begin_stroke(&mut self, width: u16, height: u16) {
        self.active = Some(ActiveStroke {
            grid: MaskCorrection::empty(width, height),
            dirty: false,
        });
    }

    /// Extend the active stroke with a sample at item-pixel coordinates.
    /// No-op if no stroke is active or if dimensions don't match.
    pub fn extend_stroke(&mut self, x: f32, y: f32) {
        let Some(active) = self.active.as_mut() else { return };
        paint_circle(
            &mut active.grid,
            x,
            y,
            self.settings.radius,
            self.settings.hardness,
            self.settings.mode,
        );
        active.dirty = true;
    }

    /// End the active stroke and return it. Returns `None` if no stroke
    /// was started or the stroke ended up empty (user clicked but didn't
    /// drag onto valid pixels).
    pub fn commit_stroke(&mut self) -> Option<MaskCorrection> {
        let active = self.active.take()?;
        if !active.dirty {
            return None;
        }
        Some(active.grid)
    }

    /// Cancel the active stroke without committing.
    pub fn cancel_stroke(&mut self) {
        self.active = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_disabled() {
        let s = BrushState::default();
        assert!(!s.is_enabled());
        assert!(!s.has_active_stroke());
    }

    #[test]
    fn toggle_flips_enabled() {
        let mut s = BrushState::default();
        s.toggle();
        assert!(s.is_enabled());
        s.toggle();
        assert!(!s.is_enabled());
    }

    #[test]
    fn toggle_off_drops_active_stroke() {
        let mut s = BrushState::default();
        s.toggle();
        s.begin_stroke(64, 64);
        s.extend_stroke(32.0, 32.0);
        assert!(s.has_active_stroke());
        s.toggle();
        assert!(!s.has_active_stroke());
    }

    #[test]
    fn extend_without_begin_is_no_op() {
        let mut s = BrushState::default();
        s.extend_stroke(10.0, 10.0);
        assert!(!s.has_active_stroke());
        assert!(s.commit_stroke().is_none());
    }

    #[test]
    fn empty_stroke_commit_returns_none() {
        let mut s = BrushState::default();
        s.begin_stroke(64, 64);
        // No extend_stroke calls — buffer stays empty.
        assert!(s.commit_stroke().is_none());
        assert!(!s.has_active_stroke(), "commit should clear active stroke");
    }

    #[test]
    fn populated_stroke_commit_returns_correction() {
        let mut s = BrushState::default();
        s.begin_stroke(64, 64);
        s.extend_stroke(32.0, 32.0);
        let c = s.commit_stroke().expect("populated stroke");
        assert_eq!(c.width, 64);
        assert_eq!(c.height, 64);
        assert!(!c.is_empty());
        assert!(!s.has_active_stroke());
    }

    #[test]
    fn cancel_drops_active_without_returning() {
        let mut s = BrushState::default();
        s.begin_stroke(32, 32);
        s.extend_stroke(16.0, 16.0);
        s.cancel_stroke();
        assert!(!s.has_active_stroke());
        assert!(s.commit_stroke().is_none());
    }

    #[test]
    fn settings_round_trip() {
        let mut s = BrushState::default();
        s.settings_mut().radius = 100.0;
        s.settings_mut().hardness = 0.3;
        s.settings_mut().mode = BrushMode::Add;
        assert_eq!(s.settings().radius, 100.0);
        assert_eq!(s.settings().hardness, 0.3);
        assert_eq!(s.settings().mode, BrushMode::Add);
    }
}
