//! Microbench for `guided_filter_alpha`. Hot path during postprocess
//! edge-refine on every batch image. The two-pass parallel prefix sum
//! is the perf-critical inner loop; this bench surfaces ≥5% drift the
//! end-to-end CLI timing would hide under noise.
//!
//! Run: `cargo bench -p prunr-core --bench guided_filter`.
//! Reference numbers in `ARCHITECTURE.md` § Postprocess fast paths.
//!
//! NOT in CI: per-runner wall-clock variance produces false positives
//! cheaper to ignore than to investigate.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use image::{GrayImage, Luma, Rgba, RgbaImage};
use prunr_core::guided_filter::guided_filter_alpha;

fn make_inputs(w: u32, h: u32) -> (RgbaImage, GrayImage) {
    // Synthetic image: vertical gradient on R channel + speckle on G/B.
    // Synthetic mask: solid centre rectangle. Gradient + speckle keep
    // the integral-image variance term non-trivial (zero variance
    // would let the filter degenerate to a passthrough).
    let mut guide = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let r = ((x as f32 / w as f32) * 255.0) as u8;
            let g = ((x ^ y) & 0xff) as u8;
            let b = (((x + y) * 7) & 0xff) as u8;
            guide.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }
    let mut mask = GrayImage::new(w, h);
    let (mx0, mx1) = (w / 4, 3 * w / 4);
    let (my0, my1) = (h / 4, 3 * h / 4);
    for y in my0..my1 {
        for x in mx0..mx1 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    (guide, mask)
}

pub fn bench(c: &mut Criterion) {
    // Two sizes — 512² is the typical seg output, 2048² is a 4K-ish
    // case where allocation churn matters. Bigger sizes amplify
    // regressions in the prefix-sum loops.
    for &(w, h, label) in &[(512u32, 512u32, "512x512"), (2048, 2048, "2048x2048")] {
        let (guide, mask) = make_inputs(w, h);
        let mut group = c.benchmark_group("guided_filter_alpha");
        group.throughput(Throughput::Elements((w * h) as u64));
        group.bench_function(label, |b| {
            b.iter(|| {
                let out = guided_filter_alpha(black_box(&guide), black_box(&mask), 8, 1e-4);
                black_box(out);
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
