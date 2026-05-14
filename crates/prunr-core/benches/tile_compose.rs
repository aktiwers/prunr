//! Microbench for `tile_compose`. The bbox-crop + smoothstep feather
//! accumulator is the LaMa stroke's per-tile critical path. Inpaint
//! closure is a passthrough — we measure compose overhead, not the
//! ORT call.
//!
//! Run: `cargo bench -p prunr-core --bench tile_compose`.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use image::{GrayImage, Luma, Rgba, RgbaImage};
use prunr_core::inpaint::tile_compose;

fn make_inputs(w: u32, h: u32, mask_frac: f32) -> (RgbaImage, GrayImage) {
    let mut image = RgbaImage::new(w, h);
    for p in image.pixels_mut() {
        *p = Rgba([128, 64, 200, 255]);
    }
    let mut mask = GrayImage::new(w, h);
    let inset_w = ((w as f32) * (1.0 - mask_frac) * 0.5) as u32;
    let inset_h = ((h as f32) * (1.0 - mask_frac) * 0.5) as u32;
    for y in inset_h..(h - inset_h) {
        for x in inset_w..(w - inset_w) {
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    (image, mask)
}

pub fn bench(c: &mut Criterion) {
    // Two coverage points: a small centred mask (bbox-crop dominates)
    // and a near-full mask (no crop savings — worst case for this
    // path). Together they bound the realistic cost surface.
    for &(label, mask_frac) in &[("4K_small_mask_10pct", 0.10), ("4K_full_mask_90pct", 0.90)] {
        let (image, mask) = make_inputs(4096, 3072, mask_frac);
        let mut group = c.benchmark_group("tile_compose");
        group.throughput(Throughput::Elements(
            (image.width() * image.height()) as u64,
        ));
        group.sample_size(10); // 4K compose is multi-second per iter — keep run time bounded
        group.bench_function(label, |b| {
            b.iter(|| {
                let out = tile_compose(
                    black_box(&image),
                    black_box(&mask),
                    |tile_rgba, _tile_mask| tile_rgba.clone(),
                )
                .expect("tile_compose passthrough must not fail");
                black_box(out);
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
