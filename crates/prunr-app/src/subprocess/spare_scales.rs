//! The line scales a Sketch run leaves for after its result: the worker
//! keeps the photo resized for them, sends the result, then makes them
//! for the image still on screen, so a scale switch there needs no run.

use std::sync::atomic::{AtomicU64, Ordering};

use image::DynamicImage;
use prunr_core::{CoreError, EdgeEngine, EdgeInferenceResult, EdgeInput, EdgeScale};

/// One run's spare scales and, once the run traced its lines, the photo
/// resized for them (3.7 MB with Fine).
pub struct SpareScales {
    scales: Vec<EdgeScale>,
    input: Option<EdgeInput>,
}

impl SpareScales {
    pub fn new(scales: Vec<EdgeScale>) -> Self {
        Self { scales, input: None }
    }

    /// The detector at `run`'s scales on `img`; the photo is resized for
    /// the spare scales in the same pass and kept.
    pub fn trace(&mut self, engine: &EdgeEngine, img: &DynamicImage, run: &[EdgeScale]) -> Result<EdgeInferenceResult, CoreError> {
        if self.scales.is_empty() {
            return engine.infer_scales(img, run);
        }
        let every: Vec<EdgeScale> = run.iter().chain(&self.scales).copied().collect();
        let input = EdgeInput::new(img, &every);
        let traced = engine.infer_input(&input, run);
        self.input = Some(input);
        traced
    }

    /// Whether the run kept a photo to make spare scales from.
    pub fn is_ready(&self) -> bool {
        self.input.is_some()
    }

    /// The spare scales, from the kept photo; `None` when none was kept.
    pub fn make(self, engine: &EdgeEngine) -> Option<Result<EdgeInferenceResult, CoreError>> {
        let input = self.input?;
        Some(engine.infer_input(&input, &self.scales))
    }
}

/// A token no other spare pass shares, in this worker or a later one:
/// the worker's pid in the high half, a count in the low.
pub fn next_token() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    (u64::from(std::process::id()) << 32) | (NEXT.fetch_add(1, Ordering::Relaxed) & 0xFFFF_FFFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_differ_and_carry_the_worker() {
        let (a, b) = (next_token(), next_token());
        assert_ne!(a, b);
        assert_eq!(a >> 32, u64::from(std::process::id()));
    }

    /// A run with no spare scales keeps no photo: nothing follows it.
    #[test]
    fn a_run_without_spare_scales_keeps_nothing() {
        let spare = SpareScales::new(Vec::new());
        assert!(!spare.is_ready());
    }
}
