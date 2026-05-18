//! End-to-end smoke test for SD 1.5 Inpaint. Auto-skips when the bundle
//! isn't installed so this won't fail CI on a fresh checkout. Once the
//! 4-part bundle finishes downloading, run with:
//!
//!   cargo test -p prunr-core --test sd_inpaint_smoke -- --nocapture
//!
//! Validates that:
//!   - The session bundle loads (4 ONNX files in the multi-part dir)
//!   - One full denoising loop runs without panicking
//!   - The painted region differs materially from the source
//!     (i.e. the pipeline is doing real work, not returning input)

use image::{GrayImage, Luma, Rgba, RgbaImage};
use prunr_core::inpaint_sd::{self, SdInpaintRequest};

#[test]
fn sd_inpaint_modifies_painted_region() {
    let _ = tracing_subscriber::fmt::try_init();
    let id = prunr_models::ModelId::SdV15InpaintFp16;
    if !prunr_models::is_available(id) {
        eprintln!(
            "SKIP: SD 1.5 Inpaint bundle not installed at {:?}",
            prunr_models::on_demand_dir()
        );
        return;
    }

    // 512×512 solid blue. Mask covers the centre 128×128 region.
    let w: u32 = 512;
    let h: u32 = 512;
    let mut image = RgbaImage::new(w, h);
    for p in image.pixels_mut() {
        *p = Rgba([40, 80, 200, 255]);
    }
    let mut mask = GrayImage::new(w, h);
    for y in 192..320 {
        for x in 192..320 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }

    let req = SdInpaintRequest {
        num_inference_steps: 20,
        seed: Some(42),
        ..Default::default()
    };
    let hooks = prunr_core::inpaint::InpaintHooks::default();
    let result = inpaint_sd::process_inpaint_with(&image, &mask, id, req, &hooks)
        .expect("SD inpaint should succeed");

    assert_eq!(result.dimensions(), image.dimensions());

    let mut diffs = 0u32;
    for y in 192..320 {
        for x in 192..320 {
            let src = image.get_pixel(x, y);
            let dst = result.get_pixel(x, y);
            if src.0 != dst.0 {
                diffs += 1;
            }
        }
    }
    let total = 128 * 128;
    eprintln!("SD inpaint: {diffs}/{total} masked pixels differ from source");
    eprintln!(
        "Sample painted pixel at (256,256): {:?} → {:?}",
        image.get_pixel(256, 256),
        result.get_pixel(256, 256)
    );
    assert!(
        diffs > total / 2,
        "SD returned ≤ half-changed for {} of {} masked pixels",
        total - diffs,
        total
    );

    // Outside the mask: source must be preserved exactly (no scaling
    // drift from the VAE round-trip leaking out of the masked region).
    for (x, y) in [(0, 0), (10, 10), (500, 500), (100, 400)] {
        assert_eq!(
            image.get_pixel(x, y).0,
            result.get_pixel(x, y).0,
            "unmasked pixel at ({x},{y}) drifted",
        );
    }
}
