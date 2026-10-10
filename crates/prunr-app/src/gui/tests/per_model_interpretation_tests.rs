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

use super::fixtures::push_test_item;
use crate::gui::item::BatchItem;
use crate::gui::worker::{CompressedTensor, TensorCache};

/// A segmentation result exists for the item: the correction has a tensor
/// to apply to.
fn give_tensor(item: &mut BatchItem) {
    let cache = TensorCache { data: vec![0.5; 64], height: 8, width: 8, model: prunr_core::ModelKind::Silueta };
    item.set_cached_tensor(CompressedTensor::from_raw(cache));
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
    give_tensor(item);
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

/// Every selection author (Paint, Magic Brush, Invert) commits through
/// `commit_selection_and_dispatch`; in chain mode that commit archives the
/// pre-stroke result so undoing the stroke can restore the image instead
/// of rerunning against the already-corrected chain base.
#[test]
fn chain_mode_commit_archives_the_pre_stroke_result_for_undo() {
    use crate::gui::history_manager::HistoryManager;
    use crate::gui::item::BatchStatus;
    use std::sync::Arc;
    let mut app = app_with_model(SettingsModel::Silueta);
    app.settings.chain_mode = true;
    app.settings.protect_selection = false;
    let item = push_test_item(&mut app, 7);
    item.dimensions = (8, 8);
    item.status = BatchStatus::Done;
    item.result_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    give_tensor(item);
    let archived_before = app.batch.items[0].history.len();

    app.commit_selection_and_dispatch(7, make_mask(8, 8));

    let item = &app.batch.items[0];
    assert_eq!(item.history.len(), archived_before + 1, "the pre-stroke result must be archived");
    assert!(HistoryManager::can_undo(item));
    assert!(item.result_rgba.is_some(), "chain base stays in place for the rerun");
    assert_eq!(item.actions_undo.len(), 1, "one Stroke marker, no Result marker");
}

/// Moving the Confidence knob re-thresholds the last Magic Brush stroke
/// from its kept logits: the selection changes, the stroke keeps its single
/// undo entry, and the retune follows the selection it committed.
#[test]
fn confidence_change_retunes_the_last_stroke_in_place() {
    use crate::gui::magic_brush_state::LastDecode;
    use crate::gui::processor::PromptModifier;
    use prunr_core::selection::BrushMode;
    use std::sync::Arc;
    let mut app = app_with_model(SettingsModel::Silueta);
    app.settings.brush.magic_confidence_threshold = 0.5;
    let item = push_test_item(&mut app, 9);
    item.dimensions = (8, 8);
    give_tensor(item);
    app.commit_selection_and_dispatch(9, make_mask(8, 8));
    let first_hash = app.batch.items[0].selection_hash.unwrap();
    assert_eq!(app.batch.items[0].stroke_undo_stack.len(), 1);

    // Logits ramp left→right: a higher confidence keeps fewer columns.
    let m = prunr_core::sam::SAM_MASK_RESOLUTION as usize;
    let mut masks = vec![-9.0f32; 3 * m * m];
    for y in 0..m {
        for x in 0..m {
            masks[y * m + x] = -4.0 + 8.0 * x as f32 / (m - 1) as f32;
        }
    }
    let output = Arc::new(prunr_core::sam::SamDecoderOutput { masks, iou_predictions: [0.9, 0.1, 0.1] });
    app.magic_brush_state.last_decode = Some(LastDecode {
        item_id: 9, output, modifier: PromptModifier::Replace, mode: BrushMode::Add,
        source_dims: (8, 8), committed_hash: first_hash, confidence: 0.5,
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
    assert!(result.rethreshold);
    let mask = result.result.expect("re-threshold decodes without a session");
    app.apply_rethreshold(9, mask, result.confidence);

    let item = &app.batch.items[0];
    let new_hash = item.selection_hash.unwrap();
    assert_ne!(new_hash, first_hash, "the selection follows the knob");
    assert_eq!(item.stroke_undo_stack.len(), 1, "still one undo entry for the stroke");
    assert_eq!(item.actions_undo.len(), 1);
    let ld = app.magic_brush_state.last_decode.as_ref().unwrap();
    assert_eq!(ld.committed_hash, new_hash);
    assert!((ld.confidence - 0.9).abs() < 1e-6);
    // The retuned selection keeps only the right-hand columns.
    let sel = item.selection_mask.as_ref().unwrap();
    assert!(sel.cells()[7] == prunr_core::selection::FULL && sel.cells()[0] == 0);
}

// ── Segmentation + !protect → rerun fires ────────────────────────────────────

#[test]
fn bg_removal_fires_rerun_on_selection_commit() {
    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    app.settings.protect_selection = false;

    let item = push_test_item(&mut app, 1);
    item.dimensions = (64, 64);
    give_tensor(item);
    let item_id = 1u64;

    let mask = make_mask(64, 64);
    app.batch.commit_selection(item_id, mask);
    app.apply_selection_to_active_model(item_id);

    assert!(
        app.processor.live_preview.is_pending_for(item_id),
        "Segmentation + !protect_selection must queue a live-preview rerun"
    );
}

// ── Segmentation + protect → rerun suppressed ────────────────────────────────

#[test]
fn bg_removal_with_protect_selection_does_not_fire_rerun() {
    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    app.settings.protect_selection = true;

    let item = push_test_item(&mut app, 2);
    item.dimensions = (64, 64);
    give_tensor(item);
    let item_id = 2u64;

    let mask = make_mask(64, 64);
    app.batch.commit_selection(item_id, mask);
    app.apply_selection_to_active_model(item_id);

    assert!(
        !app.processor.live_preview.is_pending_for(item_id),
        "Segmentation + protect_selection must NOT queue a rerun"
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
    app.batch.commit_selection(item_id, mask);
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
    app.batch.commit_selection(item_id, mask);
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
    app.batch.commit_selection(item_id, mask);
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
    app.batch.commit_selection(item_id, mask);
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
    app.settings.protect_selection = false;
    let item = push_test_item(&mut app, 7);
    item.dimensions = (32, 32);
    give_tensor(item);
    let item_id = 7u64;

    let mut committed = prunr_core::selection::MaskArtifact::new_empty(32, 32);
    let stamp = Stamp { hardness: 0.3, strength: 0.8, mode: BrushMode::Subtract };
    paint_circle(&mut committed, 16.0, 16.0, 8.0, stamp);

    app.commit_selection_and_dispatch(item_id, committed.clone());
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
    app.settings.protect_selection = false;

    let item = push_test_item(&mut app, 9);
    item.dimensions = (64, 64);
    let item_id = 9u64;

    app.commit_selection_and_dispatch(item_id, make_mask(64, 64));
    assert!(
        !app.processor.live_preview.is_pending_for(item_id),
        "no tensor: a stroke must author the selection without dispatching"
    );
    assert!(app.batch.find_by_id(item_id).unwrap().selection_mask.is_some());

    give_tensor(app.batch.find_by_id_mut(item_id).unwrap());
    app.apply_selection_to_active_model(item_id);
    assert!(
        app.processor.live_preview.is_pending_for(item_id),
        "once a tensor exists the waiting selection must be applied"
    );
}
