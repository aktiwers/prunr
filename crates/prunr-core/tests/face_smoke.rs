//! The face finder built into the app loads and runs, without any face
//! model downloaded. Skips without an ONNX Runtime.
//!
//! Run with:
//!   cargo test -p prunr-core --test face_smoke -- --nocapture

mod test_common;

use image::{Rgba, RgbaImage};
use prunr_core::face::FaceFinder;
use test_common::skip_if_no_ort;

#[test]
fn the_built_in_face_finder_runs() {
    if skip_if_no_ort("face") {
        return;
    }
    let mut finder = FaceFinder::new().expect("the face finder is built in");
    let sky = RgbaImage::from_pixel(640, 480, Rgba([150, 190, 235, 255]));
    assert!(finder.find(&sky).expect("it runs").is_empty(), "no face in a clear sky");
}

/// An installed face model opens with the built-in finder, including an
/// install whose folder still holds the copy it once downloaded.
#[test]
fn an_installed_face_model_finds_with_the_built_in_finder() {
    use prunr_models::ModelId;
    if skip_if_no_ort("face") {
        return;
    }
    let sky = RgbaImage::from_pixel(640, 480, Rgba([150, 190, 235, 255]));
    for id in [ModelId::FaceGfpgan, ModelId::FaceRestoreFormerPp] {
        if !prunr_models::is_available(id) {
            eprintln!("[face] SKIP: {id:?} is not installed");
            continue;
        }
        let mut restorer = prunr_core::face::FaceRestorer::new(id).expect("an installed face model opens");
        let found = restorer.find_and_restore(&sky, Default::default(), &prunr_core::Progress::none()).expect("it runs");
        assert!(found.is_empty(), "{id:?}: no face in a clear sky");
    }
}
