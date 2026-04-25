//! Brush tool coordinator. Owns the toolbar toggle, current size /
//! hardness / mode, and the in-progress stroke buffer. Doesn't iterate
//! `BatchManager.items` — the caller hands it the active grid size
//! and writes the committed strokes back via `BatchItem`'s mutator.

use prunr_core::brush::{paint_circle, paint_line, paint_square, BrushMode, BrushShape, MaskCorrection};

#[derive(Clone, Copy, Debug)]
pub(crate) struct BrushSettings {
    pub radius: f32,
    /// Edge falloff: 0.0 = full smoothstep falloff, 1.0 = hard disc.
    pub hardness: f32,
    /// Stroke magnitude. 1.0 = "neutral / equivalent to a gamma push";
    /// lower values give a gentler local effect (`m → m * (1 - s)` for
    /// subtract). Decoupled from hardness so users can fine-tune
    /// intensity without changing edge softness.
    pub strength: f32,
    pub mode: BrushMode,
    pub shape: BrushShape,
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            radius: 24.0,
            hardness: 0.7,
            strength: 1.0,
            mode: BrushMode::Subtract,
            shape: BrushShape::Circle,
        }
    }
}

/// Mid-drag stroke buffer at the active item's model resolution.
struct ActiveStroke {
    grid: MaskCorrection,
    /// Set the first time the stamp runs against `grid`. Lets
    /// `commit_stroke` skip an O(W·H) is_empty scan on click-without-drag.
    dirty: bool,
    /// Screen-space stamps painted so far. Drawn each frame as the
    /// in-progress trail until the stroke commits.
    trail: Vec<(f32, f32, f32)>,
    /// Stroke-time snapshot of the brush shape — pinned at begin_stroke
    /// so a mid-stroke shape switch doesn't desync the grid.
    shape: BrushShape,
    /// Model-space press / current positions (Line tool only — paints
    /// once on commit using these endpoints).
    first_model_pos: Option<(f32, f32)>,
    last_model_pos: Option<(f32, f32)>,
    last_model_radius: f32,
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
            trail: Vec::new(),
            shape: self.settings.shape,
            first_model_pos: None,
            last_model_pos: None,
            last_model_radius: self.settings.radius,
        });
    }

    /// Active stroke's pinned shape (or `Circle` if no active stroke —
    /// used by the trail renderer to pick a draw style).
    pub fn active_shape(&self) -> BrushShape {
        self.active.as_ref().map_or(BrushShape::Circle, |a| a.shape)
    }

    /// Record a screen-space stamp for the in-progress stroke trail.
    /// Caller is the canvas-side overlay; `BrushState` doesn't compute
    /// screen coords itself.
    ///
    /// Spatial dedup: skip the push if the new stamp is within half-radius
    /// of the previous one. Pointer events at 60+ Hz over a slow drag
    /// stack near-identical stamps that paint to the same pixels — keeping
    /// only every-half-radius cuts the per-frame paint count without
    /// changing what the user sees.
    pub fn record_trail_stamp(&mut self, sx: f32, sy: f32, screen_radius: f32) {
        let Some(active) = self.active.as_mut() else { return };
        if let Some(&(px, py, _)) = active.trail.last() {
            let dx = sx - px;
            let dy = sy - py;
            let min_step_sq = (screen_radius * 0.5).max(1.0).powi(2);
            if dx * dx + dy * dy < min_step_sq {
                return;
            }
        }
        active.trail.push((sx, sy, screen_radius));
    }

    /// Iterator over `(sx, sy, screen_radius)` stamps in the active
    /// stroke's trail. Empty when no stroke is active.
    pub fn trail_stamps(&self) -> impl Iterator<Item = (f32, f32, f32)> + '_ {
        self.active
            .as_ref()
            .into_iter()
            .flat_map(|a| a.trail.iter().copied())
    }

    /// Extend the active stroke at model-space coordinates with an
    /// explicit model-space radius. Caller converts screen→model
    /// outside this type so screen-radius confusion can't reach the
    /// grid. Dispatch on the stroke's pinned shape:
    /// - Circle / Square: paint per-frame stamps (incremental drag).
    /// - Line: just record press / current; the actual line is painted
    ///   once at `commit_stroke` from press → release.
    pub fn extend_stroke_with_radius(&mut self, x: f32, y: f32, radius: f32) {
        let Some(active) = self.active.as_mut() else { return };
        if active.first_model_pos.is_none() {
            active.first_model_pos = Some((x, y));
        }
        active.last_model_pos = Some((x, y));
        active.last_model_radius = radius;

        match active.shape {
            BrushShape::Circle => {
                paint_circle(
                    &mut active.grid, x, y, radius,
                    self.settings.hardness, self.settings.strength, self.settings.mode,
                );
                active.dirty = true;
            }
            BrushShape::Square => {
                paint_square(
                    &mut active.grid, x, y, radius,
                    self.settings.hardness, self.settings.strength, self.settings.mode,
                );
                active.dirty = true;
            }
            BrushShape::Line => {
                // Wait for commit — `commit_stroke` paints press → release.
            }
        }
    }

    /// End the active stroke and return it. For Line shape, paints the
    /// final press → release segment now. Returns `None` if no stroke
    /// was started or it ended up empty.
    pub fn commit_stroke(&mut self) -> Option<MaskCorrection> {
        let mut active = self.active.take()?;
        if let (BrushShape::Line, Some((x1, y1)), Some((x2, y2))) =
            (active.shape, active.first_model_pos, active.last_model_pos)
        {
            paint_line(
                &mut active.grid,
                x1, y1, x2, y2,
                active.last_model_radius,
                self.settings.hardness,
                self.settings.strength,
                self.settings.mode,
            );
            active.dirty = true;
        }
        if !active.dirty {
            return None;
        }
        Some(active.grid)
    }

    /// Cancel the active stroke without committing.
    #[allow(dead_code)]
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
        s.extend_stroke_with_radius(32.0, 32.0, 8.0);
        assert!(s.has_active_stroke());
        s.toggle();
        assert!(!s.has_active_stroke());
    }

    #[test]
    fn extend_without_begin_is_no_op() {
        let mut s = BrushState::default();
        s.extend_stroke_with_radius(10.0, 10.0, 8.0);
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
        s.extend_stroke_with_radius(32.0, 32.0, 8.0);
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
        s.extend_stroke_with_radius(16.0, 16.0, 8.0);
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
