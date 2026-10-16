//! File dialogs never hold the frame: asking returns at once, and the
//! answer, drained from the frame loop, resumes the action that asked.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use image::GenericImageView;

use super::fixtures::push_test_item;
use super::harness::{click, harness, settle};
use crate::gui::app::PrunrApp;
use crate::gui::item::BatchStatus;
use crate::gui::system_bridge::DialogRequest;

/// Long enough for a save thread on a loaded CI runner.
const LANDS_WITHIN: Duration = Duration::from_secs(10);
/// How long a test waits to be sure nothing was saved.
const NOTHING_WITHIN: Duration = Duration::from_millis(300);

fn done_item(app: &mut PrunrApp, id: u64) -> &mut crate::gui::item::BatchItem {
    let item = push_test_item(app, id);
    item.set_status_for_test(BatchStatus::Done);
    item.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(3, 2, image::Rgba([10, 20, 30, 255]))));
    item
}

fn png_on_disk(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    image::RgbaImage::from_pixel(5, 4, image::Rgba([200, 100, 50, 255])).save(&path).expect("write png");
    path
}

fn answer(app: &mut PrunrApp, paths: Option<Vec<PathBuf>>) {
    app.system.answer_for_test(paths);
    app.drain_dialog_answer();
}

fn asked(app: &PrunrApp) -> Option<&DialogRequest> {
    app.system.asked_for_test().map(|(request, _)| request)
}

#[test]
fn open_answer_imports_the_picked_image() {
    let mut app = PrunrApp::new_for_test();
    app.handle_open_dialog();
    assert_eq!(asked(&app), Some(&DialogRequest::OpenImages));
    assert!(app.batch.items.is_empty(), "asking imports nothing yet");

    let dir = tempfile::tempdir().unwrap();
    let picked = png_on_disk(dir.path(), "picked.png");
    answer(&mut app, Some(vec![picked]));

    assert_eq!(app.batch.items.len(), 1);
    assert_eq!(app.batch.items[0].filename, "picked.png");
    assert_eq!(app.last_open_dir.as_deref(), Some(dir.path()));
    assert!(!app.system.dialog_open());
}

#[test]
fn dialogs_start_in_the_last_opened_folder() {
    let mut app = PrunrApp::new_for_test();
    let dir = tempfile::tempdir().unwrap();
    app.last_open_dir = Some(dir.path().to_path_buf());
    app.handle_open_dialog();
    let (_, start_dir) = app.system.asked_for_test().expect("asked");
    assert_eq!(start_dir, Some(dir.path()));
}

#[test]
fn background_answer_loads_the_picture_for_the_image_that_asked() {
    let mut app = PrunrApp::new_for_test();
    push_test_item(&mut app, 1);
    push_test_item(&mut app, 2);
    app.handle_pick_bg_image(0);
    assert_eq!(asked(&app), Some(&DialogRequest::Background { item_id: 1 }));
    // The image moves while the dialog is up; the answer follows it.
    app.batch.reorder(0, 1);

    let dir = tempfile::tempdir().unwrap();
    let picked = png_on_disk(dir.path(), "backdrop.png");
    answer(&mut app, Some(vec![picked.clone()]));

    let load = app.batch.bg_io.bg_image_rx.recv_timeout(LANDS_WITHIN).expect("background load");
    assert_eq!(load.item_id, 1);
    assert_eq!(load.path, picked);
    assert!(load.picked);
}

#[test]
fn save_as_answer_writes_that_image_to_the_picked_file() {
    let mut app = PrunrApp::new_for_test();
    done_item(&mut app, 1);
    app.batch.selected_index = 0;
    app.handle_save_selected();
    assert_eq!(
        asked(&app),
        Some(&DialogRequest::SaveImage {
            item_id: 1,
            default_name: "test1.prunr.png".into(),
            preferred: prunr_core::ExportFormat::Png,
        }),
    );

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.png");
    answer(&mut app, Some(vec![out.clone()]));

    let msg = app.batch.bg_io.save_done_rx.recv_timeout(LANDS_WITHIN).expect("save done");
    assert_eq!(msg, Ok("Saved".to_string()));
    assert_eq!(image::open(&out).expect("saved png").dimensions(), (3, 2));
}

#[test]
fn sidebar_save_answer_writes_that_rows_image() {
    let mut app = PrunrApp::new_for_test();
    done_item(&mut app, 1);
    done_item(&mut app, 2).result_rgba = Some(Arc::new(image::RgbaImage::new(7, 7)));
    app.batch.selected_index = 0;
    app.save_item_to_file(1);
    assert!(matches!(asked(&app), Some(DialogRequest::SaveImage { item_id: 2, .. })));

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("row.png");
    answer(&mut app, Some(vec![out.clone()]));

    app.batch.bg_io.save_done_rx.recv_timeout(LANDS_WITHIN).expect("save done").expect("saved");
    assert_eq!(image::open(&out).expect("saved png").dimensions(), (7, 7));
}

#[test]
fn save_layers_answer_writes_into_the_picked_folder() {
    let mut app = PrunrApp::new_for_test();
    app.settings.export_split_layers = true;
    assert!(app.saves_layers(), "the default model cuts out a subject");
    done_item(&mut app, 1).ticked = true;
    app.handle_save_selected();
    assert_eq!(asked(&app), Some(&DialogRequest::SaveLayers { targets: vec![1] }));

    let dir = tempfile::tempdir().unwrap();
    answer(&mut app, Some(vec![dir.path().to_path_buf()]));

    let msg = app.batch.bg_io.save_done_rx.recv_timeout(LANDS_WITHIN).expect("save done");
    assert_eq!(msg, Ok("Saved 1 file(s)".to_string()));
    assert!(dir.path().join("test1.prunr.png").is_file());
}

#[test]
fn save_selected_answer_writes_each_into_the_picked_folder() {
    let mut app = PrunrApp::new_for_test();
    done_item(&mut app, 1).ticked = true;
    done_item(&mut app, 2).ticked = true;
    done_item(&mut app, 3);
    app.handle_save_selected();
    assert_eq!(asked(&app), Some(&DialogRequest::SaveSelected { targets: vec![1, 2] }));

    let dir = tempfile::tempdir().unwrap();
    answer(&mut app, Some(vec![dir.path().to_path_buf()]));

    let msg = app.batch.bg_io.save_done_rx.recv_timeout(LANDS_WITHIN).expect("save done");
    assert_eq!(msg, Ok("Saved 2 image(s)".to_string()));
    assert!(dir.path().join("test1.prunr.png").is_file());
    assert!(dir.path().join("test2.prunr.png").is_file());
    assert!(!dir.path().join("test3.prunr.png").exists(), "unticked image stays out");
}

#[test]
fn cancel_does_nothing_and_frees_the_dialog() {
    let mut app = PrunrApp::new_for_test();
    app.handle_open_dialog();
    answer(&mut app, None);
    assert!(app.batch.items.is_empty());
    assert!(app.last_open_dir.is_none());
    assert!(!app.system.dialog_open());

    done_item(&mut app, 1);
    app.batch.selected_index = 0;
    app.handle_save_selected();
    assert!(matches!(asked(&app), Some(DialogRequest::SaveImage { item_id: 1, .. })), "a cancel frees the slot");
    answer(&mut app, None);
    assert!(app.batch.bg_io.save_done_rx.recv_timeout(NOTHING_WITHIN).is_err(), "a cancelled save writes nothing");
    assert!(!app.system.dialog_open());
}

/// A request while a dialog is up is not asked, and the user is told so
/// rather than the click going nowhere.
#[test]
fn a_second_request_while_one_is_open_is_ignored() {
    let mut app = PrunrApp::new_for_test();
    done_item(&mut app, 1);
    app.batch.selected_index = 0;
    app.handle_pick_bg_image(0);
    let toasts = app.toasts.len_for_test();
    app.handle_open_dialog();
    app.handle_save_selected();
    assert_eq!(asked(&app), Some(&DialogRequest::Background { item_id: 1 }));
    assert_eq!(app.toasts.len_for_test(), toasts + 2, "each ignored click says why");

    let dir = tempfile::tempdir().unwrap();
    let picked = png_on_disk(dir.path(), "backdrop.png");
    answer(&mut app, Some(vec![picked]));

    assert_eq!(app.batch.bg_io.bg_image_rx.recv_timeout(LANDS_WITHIN).expect("background load").item_id, 1);
    assert_eq!(app.batch.items.len(), 1, "the ignored Open imported nothing");
    assert!(app.batch.bg_io.file_load_rx.try_recv().is_err());
}

#[test]
fn a_save_answered_after_its_image_was_removed_saves_nothing() {
    let mut app = PrunrApp::new_for_test();
    done_item(&mut app, 1);
    done_item(&mut app, 2);
    app.batch.selected_index = 0;
    app.handle_save_selected();
    assert!(matches!(asked(&app), Some(DialogRequest::SaveImage { item_id: 1, .. })));
    // Image 2 now sits where image 1 was, selected.
    app.remove_batch_item(0);

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.png");
    answer(&mut app, Some(vec![out.clone()]));

    assert!(app.batch.bg_io.save_done_rx.recv_timeout(NOTHING_WITHIN).is_err());
    assert!(!out.exists(), "the other image was not saved in its place");
    assert!(!app.system.dialog_open());
}

#[test]
fn a_background_answered_after_its_image_was_removed_loads_nothing() {
    let mut app = PrunrApp::new_for_test();
    push_test_item(&mut app, 1);
    push_test_item(&mut app, 2);
    app.handle_pick_bg_image(0);
    app.remove_batch_item(0);

    let dir = tempfile::tempdir().unwrap();
    let picked = png_on_disk(dir.path(), "backdrop.png");
    answer(&mut app, Some(vec![picked]));

    assert!(app.batch.bg_io.bg_image_rx.recv_timeout(NOTHING_WITHIN).is_err());
}

/// The whole frame loop: the toolbar's Open asks, frames keep running
/// while the dialog is up, and the answer lands through `logic()`.
#[test]
fn open_button_asks_and_the_frame_loop_lands_the_answer() {
    let mut harness = harness();
    click(&mut harness, "Open");
    assert_eq!(asked(harness.state()), Some(&DialogRequest::OpenImages));
    assert!(harness.state().open_modals().contains(&"file_dialog"));
    settle(&mut harness);
    assert!(harness.state().system.dialog_open(), "frames run while the dialog is up");

    let dir = tempfile::tempdir().unwrap();
    let picked = png_on_disk(dir.path(), "picked.png");
    harness.state_mut().system.answer_for_test(Some(vec![picked]));
    settle(&mut harness);

    assert_eq!(harness.state().batch.items.len(), 1);
    assert!(!harness.state().open_modals().contains(&"file_dialog"));
}
