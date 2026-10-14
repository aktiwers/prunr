//! How closely two outputs agree, how long the runs took, and the
//! samples each kind of output is compared on.

use image::{GrayImage, RgbaImage};
use serde::Serialize;

/// The difference between the CPU's output and the GPU path's, over the
/// values `samples_*` picked, in 0-255 levels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Agreement {
    pub max_abs: u8,
    pub mean_abs: f64,
    /// Share (0-1) of the values more than 2 levels apart.
    pub over_2: f64,
    pub samples: usize,
}

impl Agreement {
    /// `None` when the two have different lengths or nothing to compare.
    pub fn of(a: &[u8], b: &[u8]) -> Option<Self> {
        if a.len() != b.len() || a.is_empty() {
            return None;
        }
        let (mut max_abs, mut sum, mut over) = (0u8, 0u64, 0usize);
        for (&x, &y) in a.iter().zip(b) {
            let d = x.abs_diff(y);
            max_abs = max_abs.max(d);
            sum += u64::from(d);
            over += usize::from(d > 2);
        }
        let n = a.len() as f64;
        Some(Self { max_abs, mean_abs: sum as f64 / n, over_2: over as f64 / n, samples: a.len() })
    }

    pub fn within(&self, tolerance: Tolerance) -> bool {
        self.mean_abs <= tolerance.mean_abs && self.over_2 <= tolerance.over_2
    }
}

/// The most two providers' outputs may differ before the bench calls it
/// a mismatch. The maximum is not part of it: a few edge pixels of a
/// mask move by many levels between FP16 and FP32 weights.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Tolerance {
    pub mean_abs: f64,
    pub over_2: f64,
}

/// Session build, first run and the timed runs after it, in ms.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Timing {
    pub session_ms: f64,
    pub first_ms: f64,
    pub runs_ms: Vec<f64>,
}

impl Timing {
    /// The mean of the timed runs, else the first run.
    pub fn typical_ms(&self) -> f64 {
        if self.runs_ms.is_empty() {
            self.first_ms
        } else {
            self.runs_ms.iter().sum::<f64>() / self.runs_ms.len() as f64
        }
    }
}

/// How many times faster `gpu` ran than `cpu`.
pub fn speedup(cpu: &Timing, gpu: &Timing) -> Option<f64> {
    let gpu_ms = gpu.typical_ms();
    (gpu_ms > 0.0).then(|| cpu.typical_ms() / gpu_ms)
}

pub fn alpha_samples(img: &RgbaImage) -> Vec<u8> {
    img.pixels().map(|p| p.0[3]).collect()
}

pub fn rgb_samples(img: &RgbaImage) -> Vec<u8> {
    img.pixels().flat_map(|p| [p.0[0], p.0[1], p.0[2]]).collect()
}

/// The RGB of the pixels `mask` covers (≥ 128): an eraser leaves the rest
/// as it was, which would only dilute the difference.
pub fn rgb_samples_inside(img: &RgbaImage, mask: &GrayImage) -> Vec<u8> {
    img.pixels()
        .zip(mask.pixels())
        .filter(|(_, m)| m.0[0] >= 128)
        .flat_map(|(p, _)| [p.0[0], p.0[1], p.0[2]])
        .collect()
}

/// A 16-bit grey map in 0-255 levels.
pub fn gray16_samples(values: &[u16]) -> Vec<u8> {
    values.iter().map(|&v| (v >> 8) as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgba};

    #[test]
    fn agreement_counts_max_mean_and_the_share_over_two_levels() {
        let a = [10, 20, 30, 40];
        let b = [10, 23, 28, 49];
        let agreement = Agreement::of(&a, &b).unwrap();
        assert_eq!(agreement.max_abs, 9);
        assert_eq!(agreement.mean_abs, 14.0 / 4.0);
        assert_eq!(agreement.over_2, 0.5, "3 and 9 are over 2; 2 is not");
        assert_eq!(agreement.samples, 4);
        assert_eq!(Agreement::of(&a, &a).unwrap(), Agreement { max_abs: 0, mean_abs: 0.0, over_2: 0.0, samples: 4 });
        assert_eq!(Agreement::of(&a, &b[..3]), None, "different lengths do not compare");
        assert_eq!(Agreement::of(&[], &[]), None);
    }

    #[test]
    fn agreement_is_within_a_tolerance_on_mean_and_share_alone() {
        let tolerance = Tolerance { mean_abs: 2.0, over_2: 0.1 };
        let close = Agreement { max_abs: 255, mean_abs: 0.5, over_2: 0.01, samples: 100 };
        assert!(close.within(tolerance), "a large maximum alone is fine");
        assert!(!Agreement { mean_abs: 2.5, ..close }.within(tolerance));
        assert!(!Agreement { over_2: 0.2, ..close }.within(tolerance));
        assert!(Agreement { mean_abs: 2.0, over_2: 0.1, ..close }.within(tolerance), "the limits themselves pass");
    }

    #[test]
    fn typical_time_is_the_mean_of_the_timed_runs_or_the_first() {
        let timed = Timing { session_ms: 500.0, first_ms: 90.0, runs_ms: vec![10.0, 20.0, 30.0] };
        assert_eq!(timed.typical_ms(), 20.0);
        let once = Timing { runs_ms: vec![], ..timed.clone() };
        assert_eq!(once.typical_ms(), 90.0);
        let cpu = Timing { runs_ms: vec![200.0], ..Timing::default() };
        assert_eq!(speedup(&cpu, &timed), Some(10.0));
        assert_eq!(speedup(&cpu, &Timing::default()), None);
    }

    #[test]
    fn samples_pick_the_channels_each_output_is_compared_on() {
        let img = RgbaImage::from_fn(2, 1, |x, _| if x == 0 { Rgba([1, 2, 3, 4]) } else { Rgba([5, 6, 7, 8]) });
        assert_eq!(alpha_samples(&img), [4, 8]);
        assert_eq!(rgb_samples(&img), [1, 2, 3, 5, 6, 7]);
        let mask = GrayImage::from_fn(2, 1, |x, _| Luma([if x == 1 { 255 } else { 0 }]));
        assert_eq!(rgb_samples_inside(&img, &mask), [5, 6, 7]);
        assert_eq!(gray16_samples(&[0, 0x80ff, 0xffff]), [0, 0x80, 0xff]);
    }
}
