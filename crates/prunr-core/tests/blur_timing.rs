//! Timings behind the blur choices, on a 24 MP photo. Ignored: run with
//! `cargo test -p prunr-core --release --test blur_timing -- --ignored --nocapture`.

use std::time::Instant;

use image::{Rgba, RgbaImage};

fn photo(w: u32, h: u32) -> RgbaImage {
    RgbaImage::from_fn(w, h, |x, y| {
        let v = ((x * 7 + y * 13) % 251) as u8;
        Rgba([v, v.wrapping_mul(3), v.wrapping_add(90), 255])
    })
}

/// Largest and mean per-byte difference, as "max/255, mean x.xx".
fn diff(a: &RgbaImage, b: &RgbaImage) -> String {
    let (mut max, mut sum) = (0u8, 0u64);
    for (&x, &y) in a.as_raw().iter().zip(b.as_raw()) {
        let d = x.abs_diff(y);
        max = max.max(d);
        sum += u64::from(d);
    }
    format!("max {max}/255, mean {:.2}", sum as f64 / a.as_raw().len() as f64)
}

/// `blur_rgba` against `image::imageops::fast_blur` at the two radii the
/// app uses them with: the Blurred background's and Sharpen's.
#[test]
#[ignore]
fn blur_rgba_against_fast_blur() {
    let src = photo(6000, 4000);
    for sigma in [12.0f32, 0.5] {
        let t = Instant::now();
        let a = image::imageops::fast_blur(&src, sigma);
        let fast = t.elapsed();
        let t = Instant::now();
        let b = prunr_core::blur::blur_rgba(&src, sigma);
        let ours = t.elapsed();
        println!("sigma {sigma}: fast_blur {fast:?}, blur_rgba {ours:?}, {}", diff(&a, &b));
    }
}

/// The lens look at full size against the quick render a drag uses.
#[test]
#[ignore]
fn lens_look_full_against_quick() {
    use prunr_core::depth::{DepthLook, DepthMap, LensBlurMode};
    let src = photo(6000, 4000);
    let depth = DepthMap::from_raw(60, 40, (0..2400).map(|i| (i % 40) as f32 / 40.0).collect());
    let look = DepthLook { blur: LensBlurMode::Point, blur_pct: 60, fog_pct: 20, ..DepthLook::default() };
    let t = Instant::now();
    let full = look.apply(&src, &depth).expect("active");
    let full_t = t.elapsed();
    let t = Instant::now();
    let quick = look.apply_quick(&src, &depth, 2048).expect("active");
    let quick_t = t.elapsed();
    println!("lens look 24 MP: full {full_t:?}, quick(2048) {quick_t:?}, {}", diff(&full, &quick));
}
