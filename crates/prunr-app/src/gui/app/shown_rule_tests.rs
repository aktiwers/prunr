//! The contract between what an image shows and the work its controls
//! start, for every state an image can be in and every trigger that
//! starts work: no request is left waiting for good, an unprocessed
//! cut-out never becomes a result, and Faces runs on what is shown.

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::PrunrApp;
use crate::gui::capability::{ItemFacts, Shown};
use crate::gui::item::{BatchStatus, Engine};
use crate::gui::item_settings::FaceModel;
use crate::gui::knob_catalog::DispatchKind;
use crate::gui::settings::SettingsModel;
use crate::gui::tests::fixtures::push_test_item;
use crate::gui::timeline::StepKind;
use crate::gui::views::adjustments_toolbar::ToolbarChange;
use crate::gui::views::faces_chip::FacesChange;

fn px(v: u8) -> Arc<image::RgbaImage> {
    Arc::new(image::RgbaImage::from_pixel(8, 8, image::Rgba([v, v, v, 255])))
}

#[derive(Clone, Copy, Debug)]
enum State {
    /// Opened, never processed.
    Unprocessed,
    /// Cut out, its segmentation kept.
    Cut,
    /// Delete on the unprocessed photo: a hand edit with no cut under it.
    DeletedOnPhoto,
    /// An upscale on screen whose kept run is gone (a model round trip).
    UpscaleNotKept,
}

fn app_in(state: State, model: SettingsModel) -> PrunrApp {
    let mut app = PrunrApp::new_for_test();
    app.settings.model = model;
    let item = push_test_item(&mut app, 1);
    item.dimensions = (8, 8);
    item.source_rgba = Some(px(10));
    match state {
        State::Unprocessed => {}
        State::Cut => {
            item.result_rgba = Some(px(20));
            item.commit(StepKind::Process);
            crate::gui::tests::fixtures::cut_out(item, prunr_core::ModelKind::Silueta);
        }
        State::DeletedOnPhoto => {
            item.result_rgba = Some(px(30));
            item.set_status_for_test(BatchStatus::Done);
            item.commit(StepKind::Edit);
        }
        State::UpscaleNotKept => {
            item.result_rgba = Some(px(40));
            item.set_status_for_test(BatchStatus::Done);
            item.commit(StepKind::Upscale);
            item.record_origin(Engine::Upscale, None, None);
        }
    }
    app.batch.select_item(0);
    app
}

#[derive(Clone, Copy, Debug)]
enum Trigger {
    MaskKnob,
    OutlineKnob,
    FacesOn,
    FaceModelSwitch,
}

fn pull(app: &mut PrunrApp, trigger: Trigger) {
    let ctx = egui::Context::default();
    let before = app.batch.items[0].settings;
    let s = &mut app.batch.items[0].settings;
    let change = match trigger {
        Trigger::MaskKnob => {
            s.gamma = 1.4;
            ToolbarChange { auto_dispatch: DispatchKind::LivePreviewMask, commit: true, ..Default::default() }
        }
        Trigger::OutlineKnob => {
            s.decoration.outline_width = 6.0;
            ToolbarChange { auto_dispatch: DispatchKind::LivePreviewMask, commit: true, ..Default::default() }
        }
        Trigger::FacesOn => {
            s.faces.on = true;
            ToolbarChange { faces: FacesChange { search: true, ..Default::default() }, commit: true, ..Default::default() }
        }
        Trigger::FaceModelSwitch => {
            s.faces.on = true;
            s.faces.model = FaceModel::RestoreFormerPp;
            ToolbarChange { faces: FacesChange { search: true, ..Default::default() }, commit: true, ..Default::default() }
        }
    };
    app.apply_toolbar_change(&ctx, change, before);
}

/// Frames until the live preview has nothing left, or two seconds.
fn settle(app: &mut PrunrApp) -> bool {
    let ctx = egui::Context::default();
    let until = Instant::now() + Duration::from_secs(2);
    loop {
        app.pump_results(&ctx);
        app.pump_live_preview(&ctx);
        if !app.processor.live_preview.is_busy() {
            return true;
        }
        if Instant::now() > until {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

const STATES: [State; 4] = [State::Unprocessed, State::Cut, State::DeletedOnPhoto, State::UpscaleNotKept];
const TRIGGERS: [Trigger; 4] = [Trigger::MaskKnob, Trigger::OutlineKnob, Trigger::FacesOn, Trigger::FaceModelSwitch];
const MODELS: [SettingsModel; 3] = [SettingsModel::Silueta, SettingsModel::None, SettingsModel::RealEsrganUpscale];

#[test]
fn no_request_waits_for_good() {
    let mut stuck = Vec::new();
    for model in MODELS {
        for state in STATES {
            for trigger in TRIGGERS {
                let mut app = app_in(state, model);
                pull(&mut app, trigger);
                if !settle(&mut app) {
                    stuck.push(format!("{model:?} / {state:?} / {trigger:?}"));
                }
            }
        }
    }
    assert!(stuck.is_empty(), "renders that never finish:\n{}", stuck.join("\n"));
}

#[test]
fn an_unprocessed_cutout_never_becomes_a_result() {
    let mut faked = Vec::new();
    for trigger in TRIGGERS {
        let mut app = app_in(State::Unprocessed, SettingsModel::Silueta);
        pull(&mut app, trigger);
        settle(&mut app);
        let item = &app.batch.items[0];
        if ItemFacts::of(item).shown != Shown::Photo || item.status() == BatchStatus::Done {
            faked.push(format!("{trigger:?}: shown {:?}, status {:?}", ItemFacts::of(item).shown, item.status()));
        }
    }
    assert!(faked.is_empty(), "an unprocessed cut-out turned into a result:\n{}", faked.join("\n"));
}

#[test]
fn faces_run_on_what_is_shown() {
    let mut idle = Vec::new();
    for (model, state) in [
        (SettingsModel::Silueta, State::Unprocessed),
        (SettingsModel::Silueta, State::Cut),
        (SettingsModel::None, State::Unprocessed),
    ] {
        let mut app = app_in(state, model);
        pull(&mut app, Trigger::FacesOn);
        app.pump_results(&egui::Context::default());
        if app.batch.items[0].faces_pending.is_none() && app.batch.items[0].found_faces.is_none() {
            idle.push(format!("{model:?} / {state:?}"));
        }
    }
    assert!(idle.is_empty(), "Faces turned on but no search started:\n{}", idle.join("\n"));
}

#[test]
fn an_export_writes_what_the_canvas_shows() {
    use crate::gui::item::ExportImage;
    let app = app_in(State::Unprocessed, SettingsModel::Silueta);
    assert!(app.batch.items[0].export_image().is_untouched_photo(), "the photo as opened");

    let mut app = app_in(State::Cut, SettingsModel::Silueta);
    assert!(matches!(app.batch.items[0].export_image(), ExportImage::Shown(_)), "the cut-out on screen");
    assert!(app.batch.items[0].park_result());
    assert!(matches!(app.batch.items[0].export_image(), ExportImage::Parked(_)), "off screen: the parked cut-out, not the photo");
}

/// A failed re-run leaves the result it had on screen: the failure is
/// told, never shown in place of the image.
#[test]
fn a_failure_never_hides_a_result() {
    let mut app = app_in(State::Cut, SettingsModel::Silueta);
    let item = &mut app.batch.items[0];
    item.start_cut(None, crate::gui::item::Cutter::Model(prunr_core::ModelKind::Silueta));
    item.fail("boom".into());
    assert_eq!(item.status(), BatchStatus::Done, "the cut-out stays");
    assert!(item.error().is_none());
    let mut app = app_in(State::Unprocessed, SettingsModel::Silueta);
    app.batch.items[0].fail("boom".into());
    assert_eq!(app.batch.items[0].error(), Some("boom"), "with nothing to show, the failure is what the image is");
}

/// A cut waiting for a worker slot shows the result it has, and Cancel
/// All ends it like one under way: nothing is left awaited.
#[test]
fn cancel_all_ends_waiting_cuts_too() {
    let mut app = app_in(State::Cut, SettingsModel::Silueta);
    let item = &mut app.batch.items[0];
    item.start_cut(Some(StepKind::Process), crate::gui::item::Cutter::Model(prunr_core::ModelKind::Silueta));
    assert_eq!(item.status(), BatchStatus::Done, "waiting, it shows its cut-out");
    assert_eq!(ItemFacts::of(item).activity, crate::gui::capability::Activity::Queued);
    app.handle_cancel_all_and_reset();
    let item = &app.batch.items[0];
    assert!(item.job.is_none() && !item.is_busy(), "nothing awaited after Cancel All");
    assert_eq!(item.status(), BatchStatus::Done);
}

/// An erase on an image off screen starts from the result it shows, kept
/// parked in its timeline, not from the photo: earlier fills stay.
#[test]
fn an_erase_off_screen_starts_from_the_parked_result() {
    let mut app = app_in(State::Cut, SettingsModel::Silueta);
    let item = &mut app.batch.items[0];
    let shown = item.result_rgba.clone().expect("a cut-out");
    assert!(item.park_result());
    let source = item.source_for_inpaint().expect("pixels to erase");
    assert_eq!(*source, *shown, "the cut-out, not the photo");
}

/// Lines run on use on No model's photo, which is its own result; on a
/// cut-out model they wait for the cut, like the other cut-out knobs.
#[test]
fn lines_run_on_use_on_no_models_photo() {
    for (model, runs) in [(SettingsModel::None, true), (SettingsModel::Silueta, false)] {
        let mut app = app_in(State::Unprocessed, model);
        let before = app.batch.items[0].settings;
        app.batch.items[0].settings.line_mode = prunr_core::LineMode::EdgesOnly;
        let change = ToolbarChange { auto_dispatch: DispatchKind::SubprocessFullPipeline, commit: true, ..Default::default() };
        app.apply_toolbar_change(&egui::Context::default(), change, before);
        assert_eq!(app.batch.items[0].job.is_some(), runs, "{model:?}");
    }
}

/// Save writes what the canvas shows: the photo as it is before Process,
/// the cut-out after, and a parked cut-out for an image off screen.
#[test]
fn save_writes_what_the_canvas_shows() {
    let dir = std::env::temp_dir().join(format!("prunr-save-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let saved = |app: &mut PrunrApp, name: &str| {
        let path = dir.join(name);
        app.on_save_image_picked(1, prunr_core::ExportFormat::Png, path.clone());
        let outcome = app.batch.bg_io.save_done_rx.recv_timeout(Duration::from_secs(20)).expect("the save reports");
        assert!(outcome.is_ok(), "{name}: {outcome:?}");
        image::open(&path).expect("a readable file").to_rgba8()
    };
    let mut app = app_in(State::Unprocessed, SettingsModel::Silueta);
    assert_eq!(saved(&mut app, "photo.png"), *px(10), "the photo as it is");
    let mut app = app_in(State::Cut, SettingsModel::Silueta);
    assert_eq!(saved(&mut app, "cut.png"), *px(20), "the cut-out");
    assert!(app.batch.items[0].park_result());
    assert_eq!(saved(&mut app, "parked.png"), *px(20), "the parked cut-out, not the photo");
    let _ = std::fs::remove_dir_all(&dir);
}
