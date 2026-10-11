//! Microbench for the SAM decoder output → selection plane step: a
//! 256² logit map bilinearly upsampled to source resolution on every
//! Magic Brush click or stroke.
//!
//! Run: `cargo bench -p prunr-core --bench sam_decode`.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use prunr_core::sam::{decode_to_mask_artifact, MaskReading, SamDecoderOutput, SAM_MASK_RESOLUTION};
use prunr_core::selection::BrushMode;

pub fn bench(c: &mut Criterion) {
    let n = (SAM_MASK_RESOLUTION * SAM_MASK_RESOLUTION) as usize;
    // A soft disc of logits so the sign flips along a real contour.
    let m = SAM_MASK_RESOLUTION as f32;
    let masks: Vec<f32> = (0..3 * n)
        .map(|i| {
            let i = (i % n) as f32;
            let (x, y) = (i % m, (i / m).floor());
            0.4 * m - ((x - m / 2.0).powi(2) + (y - m / 2.0).powi(2)).sqrt()
        })
        .collect();
    let output = SamDecoderOutput { masks, iou_predictions: [0.2, 0.9, 0.3] };
    let (w, h) = (4096u32, 3072u32);
    let mut group = c.benchmark_group("sam_decode");
    group.throughput(Throughput::Elements((w * h) as u64));
    group.bench_function("4K", |b| {
        b.iter(|| black_box(decode_to_mask_artifact(black_box(&output), w, h, MaskReading { confidence: 0.5, remove_specks: true }, BrushMode::Add)));
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
