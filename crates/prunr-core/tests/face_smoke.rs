//! The face finder built into the app loads and runs, without any face
//! model downloaded. Skips without an ONNX Runtime.
//!
//! Run with:
//!   cargo test -p prunr-core --test face_smoke -- --nocapture
//! and, on a portrait of your own (not kept in the repo):
//!   PRUNR_FACE_PHOTO=/path/to/portrait.jpg cargo test -p prunr-core --test face_smoke -- --nocapture

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

/// On the portrait `PRUNR_FACE_PHOTO` names: the finder finds its face,
/// its depth map focuses on it, and an installed face model reports the
/// faces as found and then each one as it is restored.
#[test]
fn a_portrait_is_found_focused_on_and_restored_face_by_face() {
    use prunr_core::progress::{check_contract, ProgressUpdate, RecordingSink};
    let Some(path) = std::env::var_os("PRUNR_FACE_PHOTO") else {
        eprintln!("[face] SKIP: set PRUNR_FACE_PHOTO to a portrait");
        return;
    };
    if skip_if_no_ort("face") {
        return;
    }
    let photo = prunr_core::load_image_from_path(std::path::Path::new(&path)).expect("the portrait loads").to_rgba8();
    let faces = FaceFinder::new().expect("built in").find(&photo).expect("it runs");
    assert!(!faces.is_empty(), "a face in the portrait");
    eprintln!("[face] {} face(s), best {:?}", faces.len(), faces[0].bbox);

    if prunr_models::is_available(prunr_core::depth::MODEL.into()) {
        let map = prunr_core::depth::Measurer::default().measure(&photo).expect("depth");
        let near = map.face_nearness().expect("the map knows the face");
        eprintln!("[face] auto focus on the face, at nearness {near:.3}");
        assert_eq!(prunr_core::depth::auto_focus(&map), near);
        assert_eq!(prunr_core::depth::Measurer::lines_only().measure(&photo).expect("depth").face_nearness(), None);
    }

    let Some(id) = [prunr_models::ModelId::FaceGfpgan, prunr_models::ModelId::FaceRestoreFormerPp].into_iter().find(|&id| prunr_models::is_available(id)) else {
        eprintln!("[face] SKIP restore: no face model installed");
        return;
    };
    let sink = std::sync::Arc::new(RecordingSink::default());
    let progress = prunr_core::Progress::new(sink.clone()).with_tile_previews(1536);
    let restored = prunr_core::face::find_and_restore(&photo, id, Default::default(), &progress).expect("restored");
    let updates = sink.updates();
    assert_eq!(check_contract(&updates), Ok(()));
    let found_at = updates.iter().position(|u| matches!(u, ProgressUpdate::Faces(_))).expect("the faces reported as found");
    let previews: Vec<usize> = updates.iter().enumerate().filter_map(|(i, u)| matches!(u, ProgressUpdate::TilePreview(_)).then_some(i)).collect();
    assert_eq!(previews.len(), restored.len(), "one preview per face");
    assert!(previews.iter().all(|&i| i > found_at), "found before any is restored");
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
