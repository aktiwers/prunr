//! Smoke test for Big-LaMa: loads the on-disk model, runs one tile,
//! and checks that the painted region has been modified vs the source.
//!
//! Skipped automatically when the model isn't installed, so this won't
//! fail CI on a fresh checkout. Run with:
//!   cargo test -p prunr-core --test big_lama_smoke -- --nocapture

use image::{GrayImage, Luma, Rgba, RgbaImage};
use prunr_core::inpaint;

#[test]
fn big_lama_modifies_painted_region() {

    let id = prunr_models::ModelId::BigLaMa;
    if !prunr_models::is_available(id) {
        eprintln!("SKIP: Big-LaMa not installed at {:?}",
            prunr_models::on_demand_dir());
        return;
    }

    // 256×256 solid red. Mask covers the centre 64×64 region.
    let w: u32 = 256;
    let h: u32 = 256;
    let mut image = RgbaImage::new(w, h);
    for p in image.pixels_mut() { *p = Rgba([200, 50, 50, 255]); }
    let mut mask = GrayImage::new(w, h);
    for y in 96..160 {
        for x in 96..160 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }

    let result = inpaint::process_inpaint(&image, &mask, id)
        .expect("Big-LaMa inference should succeed");

    assert_eq!(result.dimensions(), image.dimensions());

    // Sample inside the mask: the painted output should differ from
    // the solid red source. If they're identical, the model is
    // returning the input unchanged — the bug we're investigating.
    let mut diffs = 0u32;
    for y in 96..160 {
        for x in 96..160 {
            let src = image.get_pixel(x, y);
            let dst = result.get_pixel(x, y);
            if src.0 != dst.0 {
                diffs += 1;
            }
        }
    }
    let total = 64 * 64;
    eprintln!("Big-LaMa: {diffs}/{total} masked pixels differ from source");
    eprintln!("Sample painted pixel at (128,128): {:?} → {:?}",
        image.get_pixel(128, 128), result.get_pixel(128, 128));
    assert!(diffs > total / 2,
        "Big-LaMa returned the input unchanged for {} of {} masked pixels",
        total - diffs, total);
}
