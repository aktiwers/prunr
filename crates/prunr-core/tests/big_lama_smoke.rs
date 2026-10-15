//! Smoke test for Big-LaMa: loads the on-disk model, runs one tile,
//! and checks that the painted region has been modified vs the source.
//!
//! Skipped automatically when the model isn't installed, so this won't
//! fail CI on a fresh checkout. Run with:
//!   cargo test -p prunr-core --test big_lama_smoke -- --nocapture

mod test_common;

use image::{Rgba, RgbaImage};
use prunr_core::inpaint;
use test_common::{changed_in, rect_mask, skip_if_model_missing, skip_if_no_ort};

#[test]
fn big_lama_modifies_painted_region() {
    if skip_if_no_ort("big_lama_smoke") { return; }
    let id = prunr_models::ModelId::BigLaMa;
    if skip_if_model_missing("big_lama_smoke", id) { return; }

    // 256×256 solid red. Mask covers the centre 64×64 region.
    let w: u32 = 256;
    let h: u32 = 256;
    let mut image = RgbaImage::new(w, h);
    for p in image.pixels_mut() { *p = Rgba([200, 50, 50, 255]); }
    let mask = rect_mask(w, h, 96, 96, 64, 64);

    let result = inpaint::process_inpaint(&image, &mask, id)
        .expect("Big-LaMa inference should succeed");

    assert_eq!(result.dimensions(), image.dimensions());

    let diffs = changed_in(&image, &result, 96, 96, 64, 64);
    let total = 64 * 64;
    eprintln!("Big-LaMa: {diffs}/{total} masked pixels differ from source");
    eprintln!("Sample painted pixel at (128,128): {:?} → {:?}",
        image.get_pixel(128, 128), result.get_pixel(128, 128));
    assert!(diffs > total / 2,
        "Big-LaMa returned the input unchanged for {} of {} masked pixels",
        total - diffs, total);
}
