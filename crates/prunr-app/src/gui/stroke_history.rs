//! One stroke stack (undo or redo) that keeps full planes only where
//! they are needed. Its top holds the whole plane, so the last stroke's
//! "before" is at hand (the Confidence retune merges onto it) and nothing
//! depends on the live selection, which Clear or a retune may change
//! outside the history. Deeper entries hold only the cells that differ
//! from the entry above: a small stroke on a 4K image costs kilobytes,
//! not the 8 MB of a full plane.

use std::collections::VecDeque;
use std::sync::Arc;

use prunr_core::selection::{CellRect, MaskArtifact};

/// Per-item stroke history depth.
pub(crate) const STROKE_HISTORY_DEPTH: usize = 32;

/// One stroke's place in the history: the selection on the other side of
/// it, and whether its re-cut archived the pre-stroke image (chain mode),
/// which an undo then brings back instead of re-cutting.
#[derive(Clone)]
pub(crate) struct StrokeSnapshot {
    pub(crate) plane: Option<Arc<MaskArtifact>>,
    pub(crate) result_archived: bool,
}

/// An entry below the top, relative to the plane of the entry above it.
enum Delta {
    Whole(Option<Arc<MaskArtifact>>),
    /// The cells at `rect` as they were.
    Patch { rect: CellRect, cells: MaskArtifact },
    Same,
}

#[derive(Default)]
pub(crate) struct StrokeStack {
    top: Option<StrokeSnapshot>,
    below: VecDeque<(Delta, bool)>,
}

impl StrokeStack {
    /// Push a snapshot; returns `true` when the oldest one was dropped to
    /// stay within `STROKE_HISTORY_DEPTH`.
    pub(crate) fn push(&mut self, snap: StrokeSnapshot) -> bool {
        if let Some(old) = self.top.replace(snap) {
            let above = self.top.as_ref().and_then(|s| s.plane.as_ref());
            self.below.push_back((compact(old.plane, above), old.result_archived));
        }
        let mut dropped = false;
        while self.below.len() + 1 > STROKE_HISTORY_DEPTH {
            self.below.pop_front();
            dropped = true;
        }
        dropped
    }

    pub(crate) fn pop(&mut self) -> Option<StrokeSnapshot> {
        let top = self.top.take()?;
        self.top = self.below.pop_back().map(|(delta, result_archived)| StrokeSnapshot {
            plane: expand(delta, top.plane.as_ref()),
            result_archived,
        });
        Some(top)
    }

    /// The newest snapshot's plane.
    pub(crate) fn top_plane(&self) -> Option<Arc<MaskArtifact>> {
        self.top.as_ref()?.plane.clone()
    }

    pub(crate) fn mark_top_archived(&mut self) {
        if let Some(top) = self.top.as_mut() {
            top.result_archived = true;
        }
    }

    pub(crate) fn clear(&mut self) {
        self.top = None;
        self.below.clear();
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.top.is_none()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.top.as_ref().map_or(0, |_| 1 + self.below.len())
    }

    /// Cells held by the stack; full planes count once per entry.
    #[cfg(test)]
    fn cells_held(&self) -> usize {
        let whole = |p: &Option<Arc<MaskArtifact>>| p.as_ref().map_or(0, |p| p.cells().len());
        self.top.as_ref().map_or(0, |t| whole(&t.plane))
            + self.below.iter().map(|(d, _)| match d {
                Delta::Whole(p) => whole(p),
                Delta::Patch { cells, .. } => cells.cells().len(),
                Delta::Same => 0,
            }).sum::<usize>()
    }
}

/// `plane` as its difference from `above`, the plane pushed on top of it.
/// Planes of another size, or a missing one, stay whole.
fn compact(plane: Option<Arc<MaskArtifact>>, above: Option<&Arc<MaskArtifact>>) -> Delta {
    let (Some(p), Some(above)) = (plane.as_ref(), above) else { return Delta::Whole(plane) };
    if Arc::ptr_eq(p, above) {
        return Delta::Same;
    }
    if (p.width, p.height) != (above.width, above.height) {
        return Delta::Whole(plane);
    }
    match p.diff_bbox(above) {
        Some(rect) => Delta::Patch { rect, cells: p.crop(rect) },
        None => Delta::Same,
    }
}

/// Inverse of `compact`, once `above` is the top again.
fn expand(delta: Delta, above: Option<&Arc<MaskArtifact>>) -> Option<Arc<MaskArtifact>> {
    match delta {
        Delta::Whole(plane) => plane,
        Delta::Same => above.cloned(),
        // `compact` makes a patch only against a present plane.
        Delta::Patch { rect, cells } => above.map(|a| Arc::new(a.paste(rect, &cells))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(w: u32, h: u32, painted: &[(u32, u32)]) -> Arc<MaskArtifact> {
        let mut cells = vec![0i8; (w * h) as usize];
        for &(x, y) in painted {
            cells[(y * w + x) as usize] = prunr_core::selection::FULL;
        }
        Arc::new(MaskArtifact::from_cells(w, h, cells))
    }

    fn snap(p: Option<Arc<MaskArtifact>>) -> StrokeSnapshot {
        StrokeSnapshot { plane: p, result_archived: false }
    }

    /// Small strokes on a large plane keep one full plane, not one each,
    /// and every pop gives back exactly what was pushed.
    #[test]
    fn small_strokes_cost_their_area_and_pop_back_exactly() {
        let planes: Vec<Option<Arc<MaskArtifact>>> = std::iter::once(None)
            .chain((1..=5u32).map(|n| Some(plane(1000, 1000, &(0..n).map(|i| (i * 10, i * 10)).collect::<Vec<_>>()))))
            .collect();
        let mut stack = StrokeStack::default();
        for p in &planes {
            stack.push(snap(p.clone()));
        }
        assert!(stack.cells_held() < 1_000_000 + 1_000, "one whole plane plus patches, got {}", stack.cells_held());
        assert_eq!(stack.top_plane(), planes[5]);
        for p in planes.iter().rev() {
            assert_eq!(stack.pop().map(|s| s.plane), Some(p.clone()));
        }
        assert!(stack.pop().is_none());
    }

    #[test]
    fn unchanged_and_resized_planes_round_trip() {
        let a = plane(8, 8, &[(1, 1)]);
        let b = plane(4, 4, &[(2, 2)]);
        let mut stack = StrokeStack::default();
        for p in [&a, &a, &b] {
            stack.push(snap(Some(Arc::clone(p))));
        }
        assert_eq!(stack.pop().and_then(|s| s.plane), Some(b));
        assert_eq!(stack.pop().and_then(|s| s.plane), Some(Arc::clone(&a)));
        assert_eq!(stack.pop().and_then(|s| s.plane), Some(a));
    }

    #[test]
    fn the_archived_flag_travels_with_its_entry() {
        let mut stack = StrokeStack::default();
        stack.push(snap(Some(plane(4, 4, &[]))));
        stack.mark_top_archived();
        stack.push(snap(Some(plane(4, 4, &[(1, 1)]))));
        assert!(!stack.pop().unwrap().result_archived);
        assert!(stack.pop().unwrap().result_archived);
    }

    #[test]
    fn depth_is_bounded() {
        let mut stack = StrokeStack::default();
        let dropped = (0..STROKE_HISTORY_DEPTH + 2).map(|i| stack.push(snap(Some(plane(4, 4, &[((i % 4) as u32, 0)]))))).filter(|d| *d).count();
        assert_eq!((stack.len(), dropped), (STROKE_HISTORY_DEPTH, 2));
    }
}
