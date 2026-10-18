//! Timings behind turning ORT's intra-op spinning off, for two in-process
//! sessions alone and at once. Ignored: run with
//! `cargo test -p prunr-core --profile profiling --test spin_probe -- --ignored --nocapture`
//! (needs LaMa and GFPGAN installed); flip `with_intra_op_spinning` in
//! `ort_runtime::session_builder` to compare.
use std::time::Instant;
use image::{GrayImage, Luma};
use prunr_models::ModelId;

#[test]
#[ignore]
fn spin_probe() {
    let photo = image::open("/data/Coding/Vibe/Private/BgPrunr/test_cli/portrait.jpg").unwrap().into_rgba8();
    let photo = image::imageops::resize(&photo, 1200, 1800, image::imageops::FilterType::Triangle);
    let mut mask = GrayImage::new(1200, 1800);
    for y in 300..650 { for x in 700..1050 { mask.put_pixel(x, y, Luma([255])); } }
    let p = prunr_core::Progress::none();
    let lama = || prunr_core::inpaint::process_lama(&photo, &mask, ModelId::LaMaFp32, true, prunr_core::inpaint::FillScale::Full, &p).unwrap();
    let face = || prunr_core::face::find_and_restore(&photo, ModelId::FaceGfpgan, prunr_core::face::FaceBlend::default(), &p).unwrap();
    lama(); face(); // warm sessions
    for _ in 0..2 {
        let t = Instant::now(); lama(); let l = t.elapsed();
        let t = Instant::now(); face(); let f = t.elapsed();
        let t = Instant::now();
        std::thread::scope(|s| { s.spawn(lama); s.spawn(face); });
        let both = t.elapsed();
        println!("lama alone {l:?}, face alone {f:?}, both at once {both:?}");
    }
}
