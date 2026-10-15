//! The app under the AccessKit test harness, and the queries every
//! harness-driven test module shares.

use egui_kittest::kittest::Queryable;
use egui_kittest::{Harness, HarnessBuilder};

use super::fixtures::push_test_item;
use crate::gui::app::PrunrApp;
use crate::gui::automation::tree::{readable_name, ControlNode};
use crate::gui::item::BatchStatus;

pub(super) const HARNESS_SIZE: egui::Vec2 = egui::vec2(1400.0, 900.0);

/// A builder at the default window size, for tests that need one more
/// option before `build_eframe(new_app)`.
pub(super) fn harness_builder() -> HarnessBuilder<PrunrApp> {
    Harness::builder().with_size(HARNESS_SIZE)
}

/// The app as eframe would create it, with its icon font installed.
pub(super) fn new_app(cc: &eframe::CreationContext<'_>) -> PrunrApp {
    egui_material_icons::initialize(&cc.egui_ctx);
    PrunrApp::new_for_test()
}

pub(super) fn harness() -> Harness<'static, PrunrApp> {
    harness_sized(HARNESS_SIZE)
}

pub(super) fn harness_sized(size: egui::Vec2) -> Harness<'static, PrunrApp> {
    Harness::builder().with_size(size).build_eframe(|cc| new_app(cc))
}

/// One finished item, selected.
pub(super) fn loaded() -> Harness<'static, PrunrApp> {
    loaded_with(|_| {})
}

/// One finished item, selected, then `setup` before the first frames.
pub(super) fn loaded_with(setup: impl FnOnce(&mut PrunrApp)) -> Harness<'static, PrunrApp> {
    let mut h = harness();
    {
        let app = h.state_mut();
        push_test_item(app, 1).status = BatchStatus::Done;
        app.batch.select_item(0);
        setup(app);
    }
    settle(&mut h);
    h
}

pub(super) fn tree<S>(harness: &Harness<'_, S>) -> ControlNode {
    ControlNode::from_node(&harness.kittest_state().root())
}

/// Runs a few frames so clicks, animations and repaint requests settle.
/// `Harness::run` would stop at the first idle frame and panic on a
/// spinner, which the batch list shows for any item still in flight.
pub(super) fn settle(harness: &mut Harness<'_, PrunrApp>) {
    harness.run_steps(4);
}

/// Clicks the control whose readable name is `name`, then lets it settle.
pub(super) fn click(harness: &mut Harness<'_, PrunrApp>, name: &str) {
    harness
        .get_by(|n| n.label().as_deref().and_then(readable_name).as_deref() == Some(name))
        .click();
    settle(harness);
}

pub(super) fn names<S>(harness: &Harness<'_, S>) -> Vec<String> {
    tree(harness).iter().filter(|n| n.is_control()).filter_map(|n| n.name.clone()).collect()
}

/// Every control in the tree has a readable name. Fails with the list of
/// nameless controls: an icon-only button, an unlabelled slider.
pub(super) fn assert_all_controls_named<S>(harness: &Harness<'_, S>, surface: &str) {
    let root = tree(harness);
    let nameless: Vec<String> = root
        .iter()
        .filter(|n| n.is_control() && n.name.is_none())
        .map(|n| format!("{:?} {:?} at {:?}", n.role, n.id, n.rect))
        .collect();
    assert!(nameless.is_empty(), "{surface}: controls without a readable name:\n{}", nameless.join("\n"));
}
