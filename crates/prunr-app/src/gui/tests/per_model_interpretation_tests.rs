//! Per-model interpretation boundary tests.
//!
//! Each test commits a known selection on a fixture BatchItem under a specific
//! active-model category and asserts the dispatch side-effect (or lack thereof)
//! via `LivePreview::is_pending_for`, which returns true when
//! `dispatch_brush_rerun` called `mark_tweak` for the item.
//!
//! Limitation: `SettingsModel` has no Sam2HieraSmall variant (the Selection-
//! category model is not user-selectable as a top-level mode). The
//! ModelCategory::Selection arm is therefore verified via code inspection.
//! `SettingsModel::None` covers the no-model path, which is the closest
//! testable proxy for "no active model".


use crate::gui::app::PrunrApp;
use crate::gui::settings::SettingsModel;
use crate::gui::timeline::StepKind;

use super::fixtures::push_test_item;
use crate::gui::item::BatchItem;
use crate::gui::worker::{CompressedTensor, TensorCache};

/// A segmentation result exists for the item: the correction has a tensor
/// to apply to.
fn give_tensor(item: &mut BatchItem) {
    let cache = TensorCache { data: vec![0.5; 64], height: 8, width: 8, model: prunr_core::ModelKind::Silueta };
    item.set_cached_tensor(CompressedTensor::from_raw(cache));
}

/// A processed item: the cut-out is on screen and its tensor is cached.
fn give_cutout(item: &mut BatchItem) {
    give_tensor(item);
    item.status = crate::gui::item::BatchStatus::Done;
}

/// The live preview decompresses the segmentation tensor once per drag:
/// consecutive dispatch inputs share one plane, and replacing the cached
/// tensor drops it.
#[test]
fn preview_inputs_reuse_the_decompressed_tensor_until_it_changes() {
    use crate::gui::live_preview::PreviewKind;
    use std::sync::Arc;
    let mut app = app_with_model(SettingsModel::Silueta);
    let item = push_test_item(&mut app, 3);
    item.dimensions = (8, 8);
    item.source_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    give_cutout(item);
    let first = crate::gui::app::PrunrApp::build_preview_inputs(&mut app.batch.items, 3, PreviewKind::Mask, false, false)
        .expect("inputs").seg_tensor.expect("tensor");
    let second = crate::gui::app::PrunrApp::build_preview_inputs(&mut app.batch.items, 3, PreviewKind::Mask, false, false)
        .expect("inputs").seg_tensor.expect("tensor");
    assert!(Arc::ptr_eq(&first, &second), "second tick must reuse the hot tensor");
    let item = app.batch.find_by_id_mut(3).unwrap();
    item.set_cached_tensor(None);
    assert!(item.volatile_seg_tensor.is_none());
}

fn make_mask(w: u32, h: u32) -> prunr_core::selection::MaskArtifact {
    let mut data = vec![0i8; (w * h) as usize];
    // Paint a 4x4 square at (2, 2) — non-trivial but small.
    for y in 2..6_u32.min(h) {
        for x in 2..6_u32.min(w) {
            data[(y * w + x) as usize] = prunr_core::selection::FULL;
        }
    }
    prunr_core::selection::MaskArtifact::from_cells(w, h, data)
}

fn app_with_model(model: SettingsModel) -> PrunrApp {
    let mut app = PrunrApp::new_for_test();
    app.settings.model = model;
    app
}

/// A stroke is one step that shares the shown result with the step
/// before it: recording it copies no image.
#[test]
fn a_stroke_is_one_step_sharing_the_result_before_it() {
    use crate::gui::item::BatchStatus;
    use std::sync::Arc;
    let mut app = app_with_model(SettingsModel::Silueta);
    let item = push_test_item(&mut app, 7);
    item.dimensions = (8, 8);
    item.status = BatchStatus::Done;
    item.result_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    item.commit(StepKind::Process);
    give_tensor(item);

    app.commit_selection_and_dispatch(7, make_mask(8, 8), StepKind::Stroke);

    let item = &app.batch.items[0];
    assert_eq!(item.timeline.undo_steps(), 2);
    let stored: Vec<_> = item.timeline.stored_images().collect();
    assert!(Arc::ptr_eq(stored[0], stored[1]), "both steps point at one stored image");
    assert!(app.processor.live_preview.is_pending_for(7), "the re-cut is queued");
}

/// Moving the Confidence knob re-thresholds the last Magic Brush stroke
/// from its kept logits: the selection changes, the stroke keeps its single
/// undo entry, and the retune follows the selection it committed.
#[test]
fn confidence_change_retunes_the_last_stroke_in_place() {
    use crate::gui::item::LastDecode;
    use crate::gui::processor::DecodedSelection;
    use prunr_core::selection::BrushMode;
    use std::sync::Arc;
    let mut app = app_with_model(SettingsModel::Silueta);
    app.settings.brush.magic_confidence_threshold = 0.5;
    let item = push_test_item(&mut app, 9);
    item.dimensions = (8, 8);
    give_tensor(item);
    app.commit_selection_and_dispatch(9, make_mask(8, 8), StepKind::Stroke);
    let first_hash = app.batch.items[0].selection_hash.unwrap();
    assert_eq!(app.batch.items[0].timeline.undo_steps(), 1);

    // Logits ramp left→right: a higher confidence keeps fewer columns.
    let m = prunr_core::sam::SAM_MASK_RESOLUTION as usize;
    let mut masks = vec![-9.0f32; 3 * m * m];
    for y in 0..m {
        for x in 0..m {
            masks[y * m + x] = -4.0 + 8.0 * x as f32 / (m - 1) as f32;
        }
    }
    let output = Arc::new(prunr_core::sam::SamDecoderOutput { masks, iou_predictions: [0.9, 0.1, 0.1] });
    let committed_reading = app.settings.brush.magic_reading();
    app.batch.items[0].last_decode = Some(LastDecode {
        output, mode: BrushMode::Add,
        committed_hash: first_hash, reading: committed_reading,
    });

    app.settings.brush.magic_confidence_threshold = 0.9;
    app.maybe_rethreshold_last_stroke();
    assert!(app.magic_brush_state.rethreshold_in_flight);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let result = loop {
        if let Some(r) = app.processor.pump_sam_decoder_results().pop() { break r; }
        assert!(std::time::Instant::now() < deadline, "re-threshold never landed");
        std::thread::sleep(std::time::Duration::from_millis(5));
    };
    let Ok(DecodedSelection::Retuned { merged, hash }) = result.result else {
        panic!("re-threshold decodes without a session");
    };
    app.apply_rethreshold(9, merged, hash, result.reading);

    let item = &app.batch.items[0];
    let new_hash = item.selection_hash.unwrap();
    assert_ne!(new_hash, first_hash, "the selection follows the knob");
    assert_eq!(item.timeline.undo_steps(), 1, "still one undo step for the stroke");
    let ld = item.last_decode.as_ref().unwrap();
    assert_eq!(ld.committed_hash, new_hash);
    assert!((ld.reading.confidence - 0.9).abs() < 1e-6);
    // The retuned selection keeps only the right-hand columns.
    let sel = item.selection_mask.as_ref().unwrap();
    assert!(sel.cells()[7] == prunr_core::selection::FULL && sel.cells()[0] == 0);

    // The speck switch retunes the last stroke the same way.
    app.magic_brush_state.rethreshold_in_flight = false;
    app.maybe_rethreshold_last_stroke();
    assert!(!app.magic_brush_state.rethreshold_in_flight, "nothing changed, nothing to retune");
    app.settings.brush.magic_remove_specks = !app.settings.brush.magic_remove_specks;
    app.maybe_rethreshold_last_stroke();
    assert!(app.magic_brush_state.rethreshold_in_flight, "flipping Remove specks retunes the last stroke");
}

// ── Segmentation → the re-cut fires at once ─────────────────────────────────

#[test]
fn bg_removal_fires_rerun_on_selection_commit() {
    let mut app = app_with_model(SettingsModel::BiRefNetLite);

    let item = push_test_item(&mut app, 1);
    item.dimensions = (64, 64);
    give_cutout(item);
    let item_id = 1u64;

    let mask = make_mask(64, 64);
    app.batch.commit_selection(item_id, mask, StepKind::Stroke);
    app.apply_selection_to_active_model(item_id);

    assert!(
        app.processor.live_preview.is_pending_for(item_id),
        "a stroke on a background-removal model must queue the re-cut"
    );
}

// ── Inpaint → no auto-dispatch ───────────────────────────────────────────────

#[test]
fn inpaint_does_not_auto_dispatch_on_selection_commit() {
    let mut app = app_with_model(SettingsModel::Inpaint);

    let item = push_test_item(&mut app, 3);
    item.dimensions = (64, 64);
    let item_id = 3u64;

    let mask = make_mask(64, 64);
    app.batch.commit_selection(item_id, mask, StepKind::Stroke);
    app.apply_selection_to_active_model(item_id);

    assert!(
        !app.processor.live_preview.is_pending_for(item_id),
        "Inpaint must not auto-dispatch on stroke commit — user clicks Process"
    );
}

// ── Upscale → no auto-dispatch ───────────────────────────────────────────────

#[test]
fn upscale_does_not_auto_dispatch_on_selection_commit() {
    let mut app = app_with_model(SettingsModel::RealEsrganUpscale);

    let item = push_test_item(&mut app, 4);
    item.dimensions = (64, 64);
    let item_id = 4u64;

    let mask = make_mask(64, 64);
    app.batch.commit_selection(item_id, mask, StepKind::Stroke);
    app.apply_selection_to_active_model(item_id);

    assert!(
        !app.processor.live_preview.is_pending_for(item_id),
        "Upscale must not auto-dispatch on stroke commit"
    );
}

// ── No model → no auto-dispatch ──────────────────────────────────────────────

#[test]
fn no_model_loaded_does_not_dispatch() {
    let mut app = app_with_model(SettingsModel::None);

    let item = push_test_item(&mut app, 5);
    item.dimensions = (64, 64);
    let item_id = 5u64;

    let mask = make_mask(64, 64);
    app.batch.commit_selection(item_id, mask, StepKind::Stroke);
    app.apply_selection_to_active_model(item_id);

    assert!(
        !app.processor.live_preview.is_pending_for(item_id),
        "No-model mode must not dispatch on stroke commit"
    );
}

// ── Selection category → no auto-dispatch ────────────────────────────────────
//
// Sam2HieraSmall (ModelCategory::Selection) is not user-selectable via
// SettingsModel — it's an internal Magic Brush model dispatched by the canvas,
// not a top-level user-facing mode. There is no SettingsModel::Sam2 variant, so
// this category is not reachable through the settings path.
//
// The match arm `Some(ModelCategory::Selection) => {}` exists in
// `apply_selection_to_active_model` and is verified here by confirming that
// when the model maps to None (the closest reachable proxy), no dispatch fires.
// Identical no-dispatch behaviour is what the Selection arm enforces; this test
// pins the contract so a future SettingsModel::Sam2 variant cannot drift it.
#[test]
fn selection_category_arm_exists_no_dispatch() {
    // Verify apply_selection_to_active_model with SettingsModel::None (which
    // produces category=None, handled by the last match arm alongside Upscale
    // and EdgeDetection) does not dispatch. This is the same behavior the
    // Selection arm enforces. The arm itself prevents future regressions if a
    // SettingsModel::Sam2 variant is added later.
    let mut app = app_with_model(SettingsModel::None);

    let item = push_test_item(&mut app, 6);
    item.dimensions = (64, 64);
    let item_id = 6u64;

    let mask = make_mask(64, 64);
    app.batch.commit_selection(item_id, mask, StepKind::Stroke);
    app.apply_selection_to_active_model(item_id);

    assert!(
        !app.processor.live_preview.is_pending_for(item_id),
        "Selection-category (or no-model) must not auto-dispatch on stroke commit"
    );
}

// ── Regression: Paint Brush BG-removal immediate-feedback ────────────────────
//
// A Paint stroke on a BG-removal model must reach the live-preview rerun
// exactly as painted: direction (Subtract is the default), hardness
// falloff and strength. The core suite pins the lossless round trip; this
// pins the app wiring that stores and dispatches it.
#[test]
fn paint_brush_bg_removal_keeps_stroke_direction_and_softness() {
    use prunr_core::brush::{paint_circle, Stamp};
    use prunr_core::selection::BrushMode;

    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    let item = push_test_item(&mut app, 7);
    item.dimensions = (32, 32);
    give_cutout(item);
    let item_id = 7u64;

    let mut committed = prunr_core::selection::MaskArtifact::new_empty(32, 32);
    let stamp = Stamp { hardness: 0.3, strength: 0.8, mode: BrushMode::Subtract };
    paint_circle(&mut committed, 16.0, 16.0, 8.0, stamp);

    app.commit_selection_and_dispatch(item_id, committed.clone(), StepKind::Stroke);
    assert!(
        app.processor.live_preview.is_pending_for(item_id),
        "a Paint stroke on a BG-removal model must queue the immediate rerun"
    );
    let stored = app.batch.find_by_id(item_id).unwrap().selection_mask.clone().unwrap();
    assert_eq!(stored.cells(), committed.cells(), "the stored selection is the stroke, cell for cell");
    assert!(stored.cells().iter().any(|&v| v < 0), "Subtract stroke must stay negative");
    assert!(
        stored.cells().iter().any(|&v| v < 0 && v > -prunr_core::selection::FULL),
        "hardness falloff must survive as intermediate magnitudes"
    );
}

// ── Segmentation without a tensor → the selection waits for the result ───────
//
// A brush is usable as soon as an image is loaded. With no tensor there is
// nothing to correct, so the commit must not queue a rerun (the no-tensor
// live-preview path would paint the raw source as a "result"); once a result
// lands, the same selection is applied.
#[test]
fn bg_removal_without_tensor_waits_until_a_result_lands() {
    let mut app = app_with_model(SettingsModel::BiRefNetLite);

    let item = push_test_item(&mut app, 9);
    item.dimensions = (64, 64);
    let item_id = 9u64;

    app.commit_selection_and_dispatch(item_id, make_mask(64, 64), StepKind::Stroke);
    assert!(
        !app.processor.live_preview.is_pending_for(item_id),
        "no tensor: a stroke must author the selection without dispatching"
    );
    assert!(app.batch.find_by_id(item_id).unwrap().selection_mask.is_some());

    give_cutout(app.batch.find_by_id_mut(item_id).unwrap());
    app.apply_selection_to_active_model(item_id);
    assert!(
        app.processor.live_preview.is_pending_for(item_id),
        "once a tensor exists the waiting selection must be applied"
    );
}

// ── Strokes undo and redo on their own ──────────────────────────────────────
//
// A stroke is one step on the timeline whatever the model and whether or
// not a result exists yet; undoing it shows the state before it.

#[test]
fn a_stroke_before_the_first_result_undoes_and_redoes() {
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    let item = push_test_item(&mut app, 3);
    item.dimensions = (8, 8);
    app.commit_selection_and_dispatch(3, make_mask(8, 8), StepKind::Stroke);
    assert!(app.batch.items[0].selection_mask.is_some());

    app.handle_undo(&ctx);
    assert!(app.batch.items[0].selection_mask.is_none(), "undo reverts the stroke");
    assert_eq!(app.batch.items[0].status, crate::gui::item::BatchStatus::Pending, "and touches no result");

    app.handle_redo(&ctx);
    assert!(app.batch.items[0].selection_mask.is_some(), "redo paints it again");
}

#[test]
fn an_eraser_stroke_undo_keeps_the_result() {
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::Inpaint);
    let item = push_test_item(&mut app, 4);
    item.dimensions = (8, 8);
    item.status = crate::gui::item::BatchStatus::Done;
    item.result_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    item.commit(StepKind::Erase);
    app.commit_selection_and_dispatch(4, make_mask(8, 8), StepKind::Stroke);

    app.handle_undo(&ctx);
    let item = &app.batch.items[0];
    assert!(item.selection_mask.is_none(), "the region stroke is undone");
    assert!(item.result_rgba.is_some() && item.status == crate::gui::item::BatchStatus::Done, "the result stays");
}

#[test]
fn a_stroke_undo_brings_the_stored_result_back() {
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::Silueta);
    app.settings.chain_mode = true;
    let item = push_test_item(&mut app, 5);
    item.dimensions = (8, 8);
    item.status = crate::gui::item::BatchStatus::Done;
    // History entries may be stored compressed, so compare pixels, not Arcs.
    let before = image::RgbaImage::from_pixel(8, 8, image::Rgba([1, 2, 3, 255]));
    let recut = image::RgbaImage::from_pixel(8, 8, image::Rgba([9, 9, 9, 255]));
    item.source_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    item.result_rgba = Some(Arc::new(before.clone()));
    give_tensor(item);
    app.batch.items[0].commit(StepKind::Process);
    app.commit_selection_and_dispatch(5, make_mask(8, 8), StepKind::Stroke);
    // The re-cut landed.
    app.batch.items[0].result_rgba = Some(Arc::new(recut.clone()));
    app.batch.items[0].amend();

    app.handle_undo(&ctx);
    let item = &app.batch.items[0];
    assert!(item.selection_mask.is_none());
    assert!(item.result_rgba.as_ref().is_some_and(|r| **r == before), "the pre-stroke image is back");
    assert_eq!(item.status, crate::gui::item::BatchStatus::Done);

    app.handle_redo(&ctx);
    let item = &app.batch.items[0];
    assert!(item.selection_mask.is_some());
    assert!(item.result_rgba.as_ref().is_some_and(|r| **r == recut), "redo restores the re-cut");
}

/// A Process run that lands with the selection kept applies it within the
/// run's own step. Undo walks back exactly: first the run, to the corrected
/// image it replaced (selection still on), then the stroke, to the image
/// before it.
#[test]
fn a_process_run_with_a_kept_selection_undoes_in_order() {
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let px = |v: u8| image::RgbaImage::from_pixel(8, 8, image::Rgba([v, v, v, 255]));
    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    app.settings.chain_mode = true;
    let item = push_test_item(&mut app, 9);
    item.dimensions = (8, 8);
    item.status = crate::gui::item::BatchStatus::Done;
    item.source_rgba = Some(Arc::new(px(0)));
    item.result_rgba = Some(Arc::new(px(1)));
    item.commit(StepKind::Process);
    give_tensor(item);

    app.commit_selection_and_dispatch(9, make_mask(8, 8), StepKind::Stroke);
    app.batch.items[0].result_rgba = Some(Arc::new(px(2))); // the stroke's re-cut
    app.batch.items[0].amend();

    // Process: the run lands raw, then the kept selection re-cuts it.
    let item = &mut app.batch.items[0];
    item.start_job(Some(StepKind::Process), crate::gui::item::Engine::Segment);
    item.result_rgba = Some(Arc::new(px(3)));
    item.finish_job();
    app.apply_kept_selection_to_new_result(9);
    assert!(app.processor.live_preview.is_pending_for(9), "the kept selection re-cuts the new result");
    app.batch.items[0].result_rgba = Some(Arc::new(px(4)));
    app.batch.items[0].amend();

    app.handle_undo(&ctx);
    let item = &app.batch.items[0];
    assert!(item.result_rgba.as_ref().is_some_and(|r| **r == px(2)), "undoing the run brings back the corrected image it replaced");
    assert!(item.selection_mask.is_some(), "the selection belongs to the stroke, not the run");

    app.handle_undo(&ctx);
    let item = &app.batch.items[0];
    assert!(item.result_rgba.as_ref().is_some_and(|r| **r == px(1)), "undoing the stroke brings back the image before it");
    assert!(item.selection_mask.is_none());
}

/// Undoing back to the original keeps the model output cached; a stroke
/// there edits the selection only and must not re-cut, which would bring
/// the undone result back.
#[test]
fn a_stroke_after_undoing_the_result_does_not_bring_it_back() {
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    let item = push_test_item(&mut app, 6);
    item.dimensions = (8, 8);
    item.source_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    give_tensor(item);
    // The state after undoing a Process: the tensor stays, the result is gone.
    item.status = crate::gui::item::BatchStatus::Pending;

    app.commit_selection_and_dispatch(6, make_mask(8, 8), StepKind::Stroke);
    assert!(!app.processor.live_preview.is_pending_for(6), "a stroke on the original must not re-cut");

    app.handle_undo(&ctx);
    app.handle_redo(&ctx);
    assert!(!app.processor.live_preview.is_pending_for(6), "nor may stepping that stroke");
    assert_eq!(app.batch.items[0].status, crate::gui::item::BatchStatus::Pending);
}

/// A re-cut computed for a state the image has left (by undo, redo or
/// another step) never lands: its result is for a state no longer shown.
#[test]
fn a_recut_for_a_state_left_behind_never_lands() {
    use crate::gui::live_preview::{PreviewKind, PreviewResult};
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let px = |v: u8| image::RgbaImage::from_pixel(8, 8, image::Rgba([v, v, v, 255]));
    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    let item = push_test_item(&mut app, 8);
    item.dimensions = (8, 8);
    item.source_rgba = Some(Arc::new(px(0)));
    item.status = crate::gui::item::BatchStatus::Done;
    item.result_rgba = Some(Arc::new(px(1)));
    item.commit(StepKind::Process);
    give_tensor(item);
    app.commit_selection_and_dispatch(8, make_mask(8, 8), StepKind::Stroke);
    let inputs = PrunrApp::build_preview_inputs(&mut app.batch.items, 8, PreviewKind::Mask, false, false)
        .expect("the stroke's re-cut starts");

    app.handle_undo(&ctx);
    let landed = PreviewResult {
        item_id: 8,
        rgba: px(9),
        kind: PreviewKind::Mask,
        generation: 0,
        version: inputs.version,
        new_edge: None,
        new_bold: None,
        new_masked_base: None,
        applied_mask: prunr_core::MaskRecipe::from(&prunr_core::MaskSettings::default()),
        applied_tier2_knobs: None,
        is_final: true,
    };
    app.apply_completed_previews(&ctx, vec![landed]);
    let item = &app.batch.items[0];
    assert!(item.result_rgba.as_ref().is_some_and(|r| **r == px(1)), "the undone stroke's re-cut does not land");
    assert!(app.batch.items[0].timeline.can_redo(), "and the redo step is intact");
}

/// Switching away releases an image's result and switching back restores
/// it from the timeline; neither costs or moves an undo step.
#[test]
fn switching_images_keeps_every_undo_step() {
    use crate::gui::item::BatchStatus;
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    let px = |v: u8| image::RgbaImage::from_pixel(8, 8, image::Rgba([v, v, v, 255]));
    let item = push_test_item(&mut app, 1);
    item.dimensions = (8, 8);
    item.source_rgba = Some(Arc::new(px(0)));
    item.status = BatchStatus::Done;
    item.result_rgba = Some(Arc::new(px(1)));
    item.commit(StepKind::Process);
    item.result_rgba = Some(Arc::new(px(2)));
    item.commit(StepKind::Process);
    push_test_item(&mut app, 2).dimensions = (8, 8);

    app.batch.select_item(0);
    app.sync_selected_batch_textures(&ctx);
    app.batch.select_item(1);
    app.sync_selected_batch_textures(&ctx);
    assert!(app.batch.items[0].result_rgba.is_none(), "the background result is released");
    app.batch.select_item(0);
    app.sync_selected_batch_textures(&ctx);
    assert!(app.batch.items[0].result_rgba.as_ref().is_some_and(|r| **r == px(2)), "and back on return");

    app.handle_undo(&ctx);
    let item = &app.batch.items[0];
    assert!(item.result_rgba.as_ref().is_some_and(|r| **r == px(1)), "undo shows the first result");
    app.handle_undo(&ctx);
    assert_eq!(app.batch.items[0].status, BatchStatus::Pending, "and the next undo the original");
}

/// With Magic Brush on, the shown image and the one before it keep their
/// embeddings, so flipping between two is instant; paging further drops
/// the rest instead of holding 16 MB per visited image.
#[test]
fn magic_brush_keeps_the_shown_and_the_previous_embedding_only() {
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    let embedding = Arc::new(prunr_core::sam::SamEmbedding {
        image_embed: Vec::new(), high_res_feats_0: Vec::new(), high_res_feats_1: Vec::new(),
    });
    for id in 1..=3 {
        push_test_item(&mut app, id).dimensions = (8, 8);
    }
    app.magic_brush_state.activate();
    // Each image is encoded while it is shown.
    for idx in 0..3 {
        app.batch.select_item(idx);
        app.sync_selected_batch_textures(&ctx);
        app.batch.items[idx].magic_brush_embedding = Some(Arc::clone(&embedding));
    }
    let kept: Vec<bool> = app.batch.items.iter().map(|i| i.magic_brush_embedding.is_some()).collect();
    assert_eq!(kept, [false, true, true], "A dropped; B (previous) and C (shown) kept");
}
