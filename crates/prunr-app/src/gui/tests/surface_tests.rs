//! Placement contracts for popovers, flyouts and overlays: what the canvas
//! does underneath them, and how each kind dismisses.

use egui_kittest::Harness;

use super::fixtures::push_test_item;
use super::tree_tests::{click, harness, settle};
use crate::gui::app::PrunrApp;
use crate::gui::item::BatchStatus;

/// A point on the canvas, below any popover and left of the sidebar.
const CANVAS_POINT: egui::Pos2 = egui::pos2(600.0, 820.0);

pub(super) fn loaded() -> Harness<'static, PrunrApp> {
    let mut h = harness();
    {
        let app = h.state_mut();
        push_test_item(app, 1).status = BatchStatus::Done;
        app.batch.select_item(0);
    }
    settle(&mut h);
    h
}

fn pointer_to(h: &mut Harness<'_, PrunrApp>, pos: egui::Pos2) {
    h.input_mut().events.push(egui::Event::PointerMoved(pos));
    h.step();
}

fn press(h: &mut Harness<'_, PrunrApp>, pos: egui::Pos2, pressed: bool) {
    h.input_mut().events.push(egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    });
    h.step();
}

fn scroll(h: &mut Harness<'_, PrunrApp>, lines: f32) {
    h.input_mut().events.push(egui::Event::MouseWheel {
        unit: egui::MouseWheelUnit::Line,
        delta: egui::vec2(0.0, lines),
        phase: egui::TouchPhase::Move,
        modifiers: egui::Modifiers::NONE,
    });
    h.step();
}

#[test]
fn canvas_zooms_while_a_popover_is_open() {
    let mut h = loaded();
    click(&mut h, "Mask");
    assert!(egui::Popup::is_any_open(&h.ctx), "the Mask popover opened");
    let before = h.state().zoom_state.zoom;
    pointer_to(&mut h, CANVAS_POINT);
    scroll(&mut h, 1.0);
    settle(&mut h);
    assert_ne!(h.state().zoom_state.zoom, before, "scroll over the canvas zoomed");
    assert!(egui::Popup::is_any_open(&h.ctx), "scrolling did not close the popover");
}

#[test]
fn the_press_that_dismisses_a_popover_does_not_pan() {
    let mut h = loaded();
    click(&mut h, "Preset");
    assert!(egui::Popup::is_any_open(&h.ctx), "the Preset popover opened");
    pointer_to(&mut h, CANVAS_POINT);
    press(&mut h, CANVAS_POINT, true);
    assert!(!h.state().zoom_state.is_panning, "the press under an open popover does not pan");
    press(&mut h, CANVAS_POINT, false);
    settle(&mut h);
    assert!(!egui::Popup::is_any_open(&h.ctx), "the click closed it");
    assert!(!h.state().zoom_state.is_panning);
    press(&mut h, CANVAS_POINT, true);
    assert!(h.state().zoom_state.is_panning, "the next press pans");
    press(&mut h, CANVAS_POINT, false);
}

#[test]
fn a_flyout_stays_open_while_the_canvas_is_used() {
    let mut h = loaded();
    click(&mut h, "Mask");
    assert!(egui::Popup::is_any_open(&h.ctx), "the Mask flyout opened");
    pointer_to(&mut h, CANVAS_POINT);
    press(&mut h, CANVAS_POINT, true);
    assert!(h.state().zoom_state.is_panning, "the canvas pans under the flyout");
    press(&mut h, CANVAS_POINT, false);
    settle(&mut h);
    assert!(egui::Popup::is_any_open(&h.ctx), "the canvas click did not close it");
    click(&mut h, "Mask");
    assert!(!egui::Popup::is_any_open(&h.ctx), "its chip closes it");
}

#[test]
fn escape_closes_the_flyout_and_keeps_the_selection() {
    let mut h = loaded();
    {
        let app = h.state_mut();
        let item = &mut app.batch.items[0];
        item.selection_mask = Some(std::sync::Arc::new(prunr_core::selection::MaskArtifact::from_cells(1, 1, vec![100])));
    }
    settle(&mut h);
    click(&mut h, "Mask");
    assert!(egui::Popup::is_any_open(&h.ctx));
    h.key_press(egui::Key::Escape);
    settle(&mut h);
    assert!(!egui::Popup::is_any_open(&h.ctx), "Escape closed the flyout");
    assert!(h.state().batch.items[0].selection_mask.is_some(), "and did not clear the selection");
    h.key_press(egui::Key::Escape);
    settle(&mut h);
    assert!(h.state().batch.items[0].selection_mask.is_none(), "the next Escape clears it");
}

#[test]
fn the_tool_strip_follows_the_paint_brush() {
    use super::tree_tests::{assert_all_controls_named, names};
    let mut h = loaded();
    assert!(!names(&h).iter().any(|n| n == "Hardness"), "no strip without a tool");
    h.state_mut().brush_state.toggle();
    settle(&mut h);
    let present = names(&h);
    for knob in ["Add", "Subtract", "Size", "Hardness", "Opacity", "More Paint Brush settings"] {
        assert!(present.iter().any(|n| n == knob), "{knob} missing from the strip: {present:?}");
    }
    assert_all_controls_named(&h, "paint brush strip");
    click(&mut h, "More Paint Brush settings");
    assert!(egui::Popup::is_any_open(&h.ctx), "the dots open the full panel");
    assert!(names(&h).iter().any(|n| n == "Feather"), "{:?}", names(&h));
    assert_all_controls_named(&h, "paint brush panel");
    h.state_mut().brush_state.disable();
    settle(&mut h);
    assert!(!names(&h).iter().any(|n| n == "Hardness"), "the strip leaves with the tool");
}

#[test]
fn the_tool_strip_follows_the_magic_brush() {
    use super::tree_tests::{assert_all_controls_named, names};
    let mut h = loaded();
    h.state_mut().magic_brush_state.activate();
    settle(&mut h);
    let present = names(&h);
    for knob in ["Size", "Confidence", "Circle", "Square", "Line", "More Magic Brush settings"] {
        assert!(present.iter().any(|n| n == knob), "{knob} missing from the strip: {present:?}");
    }
    assert_all_controls_named(&h, "magic brush strip");
    click(&mut h, "More Magic Brush settings");
    assert!(egui::Popup::is_any_open(&h.ctx));
    assert_all_controls_named(&h, "magic brush panel");
}

#[test]
fn bracket_keys_step_the_brush_size() {
    let mut h = loaded();
    h.state_mut().brush_state.toggle();
    settle(&mut h);
    let start = h.state().settings.brush.radius;
    h.key_press(egui::Key::CloseBracket);
    settle(&mut h);
    let larger = h.state().settings.brush.radius;
    assert!(larger > start, "] grows the brush: {start} -> {larger}");
    h.key_press(egui::Key::OpenBracket);
    settle(&mut h);
    assert!(h.state().settings.brush.radius < larger, "[ shrinks it");
}
