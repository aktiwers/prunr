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
            crate::gui::tests::fixtures::put_in(item, BatchStatus::Done);
            item.commit(StepKind::Edit);
        }
        State::UpscaleNotKept => {
            item.result_rgba = Some(px(40));
            crate::gui::tests::fixtures::put_in(item, BatchStatus::Done);
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
    use crate::gui::item::Pixels;
    let app = app_in(State::Unprocessed, SettingsModel::Silueta);
    assert!(app.batch.items[0].export_image().is_untouched_photo(), "the photo as opened");

    let mut app = app_in(State::Cut, SettingsModel::Silueta);
    assert!(matches!(app.batch.items[0].export_image().pixels, Pixels::Resident(_)), "the cut-out on screen");
    assert!(app.batch.items[0].park_result());
    assert!(matches!(app.batch.items[0].export_image().pixels, Pixels::Parked(_)), "off screen: the parked cut-out, not the photo");
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
    let pixels = item.edit_pixels();
    assert!(matches!(pixels, crate::gui::item::Pixels::Parked(_)), "handed over parked: the erase's thread unpacks it, not the UI");
    let source = pixels.load().expect("pixels to erase");
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
        let outcome = app.batch.bg_io.save_done_rx.recv_timeout(Duration::from_secs(20)).expect("the save reports").outcome;
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

/// A run on an image holds edits to it: brushes, Magic Brush and the
/// selection actions wait, or the step they take makes the run's result
/// stale and it is dropped.
#[test]
fn edits_wait_for_a_run_on_the_image() {
    let mut app = app_in(State::Cut, SettingsModel::Silueta);
    let mut cells = vec![0i8; 64];
    cells[9] = prunr_core::selection::FULL;
    app.batch.commit_selection(1, prunr_core::selection::MaskArtifact::from_cells(8, 8, cells), StepKind::Stroke);
    use crate::gui::views::selection_action_bar::SelectionAction;
    assert!(app.batch.items[0].is_idle() && app.can_selection_action(SelectionAction::Delete));
    app.batch.items[0].start_erase(None);
    assert!(!app.batch.items[0].is_idle() && !app.can_selection_action(SelectionAction::Delete), "an erase under way");
    assert!(!app.can_selection_action(SelectionAction::Clear), "clearing is a step too");
    assert!(app.can_selection_action(SelectionAction::Copy), "a copy changes nothing: it may go");
    app.batch.items[0].abandon_job();
    crate::gui::tests::fixtures::put_in(&mut app.batch.items[0], BatchStatus::Processing);
    assert!(!app.can_selection_action(SelectionAction::Cut), "a cut under way");
}

/// The Process shortcut and the Process button are one gate: with a
/// ticked image to run, Cmd+R starts it even while the shown one cuts.
#[test]
fn the_process_shortcut_does_what_the_button_does() {
    let mut app = app_in(State::Unprocessed, SettingsModel::Silueta);
    let item = push_test_item(&mut app, 2);
    item.dimensions = (8, 8);
    item.source_rgba = Some(px(10));
    item.ticked = true;
    crate::gui::tests::fixtures::put_in(&mut app.batch.items[0], BatchStatus::Processing);
    assert!(app.can_process_intent(), "the ticked image can run");
    app.perform(crate::gui::views::shortcuts::Action::Process, &egui::Context::default());
    assert!(app.batch.items[1].awaits_cut(), "Cmd+R started it");
}

/// Process never starts over a run already on an image: a cut waiting
/// for its slot keeps its turn, and an erase under way keeps its fill.
#[test]
fn process_never_starts_over_a_run_on_the_image() {
    let mut app = app_in(State::Cut, SettingsModel::Silueta);
    app.batch.items[0].start_erase(None);
    assert!(!app.can_process_intent());
    app.process_items(|_| true, Some(StepKind::Process));
    assert_eq!(app.batch.items[0].job.as_ref().map(|j| j.engine), Some(Engine::Erase), "the erase keeps its job");
}

/// After an undo to the photo the kept segmentation made nothing shown:
/// No model's preview does not cut the photo with it, and a layers save
/// has no subject to cut.
#[test]
fn an_undone_cut_is_not_applied_again() {
    let mut app = app_in(State::Unprocessed, SettingsModel::Silueta);
    let item = &mut app.batch.items[0];
    item.result_rgba = Some(px(20));
    item.commit(StepKind::Process);
    crate::gui::tests::fixtures::cut_out(item, prunr_core::ModelKind::Silueta);
    assert!(item.shows_cutout());
    assert!(item.undo_step().is_ok());
    assert!(!item.shows_cutout() && item.hot_seg_tensor().is_none(), "the photo shows: the mask made nothing on screen");
    let style = crate::gui::background_io::SelectionStyle::with_fill(0.0);
    assert!(!crate::gui::drag_export::LayerJob::of(item, style, crate::gui::brush_state::CutoutAction::Erase).has_layers(), "no subject layer from the undone cut");
    app.settings.model = SettingsModel::None;
    let inputs = PrunrApp::build_preview_inputs(app.batch.find_by_id_mut(1).expect("the item"), crate::gui::live_preview::PreviewKind::Mask, crate::gui::knob_catalog::ModelRole::Filters, style, app.settings.brush.cutout_action);
    assert!(inputs.is_ok_and(|i| i.seg_tensor.is_none()), "No model renders the photo itself");
}

/// The render door never copies the photo: every render gets the
/// decoded photo itself, and nothing is kept on the item for it.
#[test]
fn the_render_door_hands_the_photo_over_as_it_is() {
    use crate::gui::live_preview::PreviewKind;
    let mut app = app_in(State::Unprocessed, SettingsModel::None);
    let style = crate::gui::background_io::SelectionStyle::with_fill(0.0);
    let action = app.settings.brush.cutout_action;
    let role = crate::gui::knob_catalog::ModelRole::Filters;
    let photo = app.batch.items[0].source_rgba.clone().expect("decoded");
    for _ in 0..2 {
        let render = PrunrApp::build_preview_inputs(app.batch.find_by_id_mut(1).expect("the item"), PreviewKind::Mask, role, style, action).expect("a render");
        assert!(render.original_from.is_none() && Arc::ptr_eq(&render.original, &photo), "the photo itself");
    }
    assert!(app.batch.items[0].start_photo.is_none(), "nothing held besides it");
}

/// Cancelling erases ends the queue after them too, from Escape and the
/// capsule alike: a stopped erase does not start the next image.
#[test]
fn cancelling_erases_ends_their_queue() {
    let mut app = app_in(State::Cut, SettingsModel::Inpaint);
    app.processor.queue_runs(vec![7, 8, 9]);
    app.cancel_erases();
    assert_eq!(app.processor.next_queued_run(), None, "nothing waits to start");
}

/// A source that failed to decode is not asked for again every frame;
/// its renders are refused, not left waiting.
#[test]
fn an_unreadable_photo_is_not_decoded_again() {
    let mut app = PrunrApp::new_for_test();
    app.settings.model = SettingsModel::None;
    push_test_item(&mut app, 1).dimensions = (8, 8);
    app.batch.select_item(0);
    app.batch.items[0].land_decode(Err("corrupt".into()));
    app.ensure_source_decoded(0);
    assert!(!app.batch.items[0].decode_pending, "no second decode");
    let style = crate::gui::background_io::SelectionStyle::with_fill(0.0);
    let inputs = PrunrApp::build_preview_inputs(app.batch.find_by_id_mut(1).expect("the item"), crate::gui::live_preview::PreviewKind::Mask, crate::gui::knob_catalog::ModelRole::Filters, style, app.settings.brush.cutout_action);
    assert!(matches!(inputs, Err(crate::gui::capability::Verdict::Never(_))), "refused for good, so the busy pill goes");
}

/// Try another runs one fill at a time, as Process does: it waits for an
/// erase on another image.
#[test]
fn try_another_waits_for_an_erase_elsewhere() {
    let mut app = app_in(State::Cut, SettingsModel::SdInpaint);
    assert_eq!(app.reroll_refusal(), Some("Erase something first; this generates that fill again."));
    let mut cells = vec![0i8; 64];
    cells[9] = prunr_core::selection::FULL;
    app.batch.commit_selection(1, prunr_core::selection::MaskArtifact::from_cells(8, 8, cells), StepKind::Stroke);
    let item = &mut app.batch.items[0];
    item.start_erase(Some(7));
    item.result_rgba = Some(px(60));
    item.invalidate_selection();
    item.finish_job();
    assert!(app.can_reroll_intent(), "a generated fill on screen");
    app.processor.set_inpaint_in_flight_for_test(99, true);
    assert_eq!(app.reroll_refusal(), Some("Wait for the erase to finish, or cancel it."));
    assert!(!app.can_reroll_intent(), "one fill at a time");
}

/// A step recorded or undone on an image off screen keeps its result
/// parked (packed in the timeline), and the run's start knows its size
/// without unpacking it.
#[test]
fn a_parked_result_stays_parked_through_steps() {
    let mut app = app_in(State::Cut, SettingsModel::Silueta);
    let item = &mut app.batch.items[0];
    assert!(item.park_result());
    item.settings.gamma = 1.3;
    item.commit(StepKind::Adjust);
    assert!(item.result_rgba.is_none() && item.has_result(), "the step shares the stored result, still parked");
    assert!(item.undo_step().is_ok());
    assert!(item.result_rgba.is_none() && item.has_result(), "undone off screen: still parked");
    assert_eq!(item.run_input_dims(Engine::Upscale), (8, 8), "the stored step knows its size");
}
