//! Microbench for the SAM 2 encoder input: a 4K RGBA photo resized to
//! 1024² and ImageNet-normalised into NCHW f32. Runs once per image
//! when Magic Brush is active.
//!
//! Run: `cargo bench -p prunr-core --bench sam_preprocess`.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use image::{Rgba, RgbaImage};
use prunr_core::sam::preprocess::preprocess_for_sam;

pub fn bench(c: &mut Criterion) {
    let (w, h) = (4096u32, 3072u32);
    let source = RgbaImage::from_fn(w, h, |x, y| Rgba([(x & 255) as u8, (y & 255) as u8, ((x ^ y) & 255) as u8, 255]));
    let mut group = c.benchmark_group("sam_preprocess");
    group.throughput(Throughput::Elements((w * h) as u64));
    group.sample_size(20);
    group.bench_function("4K", |b| b.iter(|| black_box(preprocess_for_sam(black_box(&source)))));
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
