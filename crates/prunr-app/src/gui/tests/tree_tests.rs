//! The app under the AccessKit test harness: real widgets, found by name.

use egui_kittest::kittest::Queryable;
use egui_kittest::Harness;

use super::fixtures::push_test_item;
use crate::gui::app::PrunrApp;
use crate::gui::automation::tree::{readable_name, ControlNode};
use crate::gui::item::BatchStatus;
use crate::gui::views::settings::SettingsTab;

pub(super) fn harness() -> Harness<'static, PrunrApp> {
    Harness::builder()
        .with_size(egui::vec2(1400.0, 900.0))
        .build_eframe(|cc| {
            egui_material_icons::initialize(&cc.egui_ctx);
            PrunrApp::new_for_test()
        })
}

pub(super) fn tree(harness: &Harness<'_, PrunrApp>) -> ControlNode {
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

fn names(harness: &Harness<'_, PrunrApp>) -> Vec<String> {
    tree(harness).iter().filter(|n| n.is_control()).filter_map(|n| n.name.clone()).collect()
}

/// Every control in the tree has a readable name. Fails with the list of
/// nameless controls: an icon-only button, an unlabelled slider.
fn assert_all_controls_named(harness: &Harness<'_, PrunrApp>, surface: &str) {
    let root = tree(harness);
    let nameless: Vec<String> = root
        .iter()
        .filter(|n| n.is_control() && n.name.is_none())
        .map(|n| format!("{} {:?} at {:?}", n.role, n.id, n.rect))
        .collect();
    assert!(nameless.is_empty(), "{surface}: controls without a readable name:\n{}", nameless.join("\n"));
}

#[test]
fn empty_window_names_every_control() {
    let mut harness = harness();
    settle(&mut harness);
    assert_all_controls_named(&harness, "empty window");
    let names = names(&harness);
    for expected in ["Open", "Settings"] {
        assert!(names.iter().any(|n| n == expected), "{expected} missing from {names:?}");
    }
}

#[test]
fn loaded_item_names_every_control() {
    let mut harness = harness();
    {
        let app = harness.state_mut();
        let item = push_test_item(app, 1);
        item.status = BatchStatus::Done;
        app.batch.select_item(0);
    }
    settle(&mut harness);
    assert_all_controls_named(&harness, "loaded item");
}

#[test]
fn settings_tabs_name_every_control() {
    let mut harness = harness();
    settle(&mut harness);
    click(&mut harness, "Settings");
    for tab in SettingsTab::ALL {
        harness.state_mut().settings_tab = tab;
        settle(&mut harness);
        assert_all_controls_named(&harness, &format!("settings › {tab:?}"));
    }
}

#[test]
fn settings_opens_from_its_toolbar_button() {
    let mut harness = harness();
    settle(&mut harness);
    assert!(!harness.state().show_settings);
    click(&mut harness, "Settings");
    assert!(harness.state().show_settings);
    assert!(names(&harness).iter().any(|n| n == "Behavior"), "{:?}", names(&harness));
}

#[test]
fn help_modals_name_every_control() {
    let mut harness = harness();
    for (flag, surface) in [(0, "shortcuts"), (1, "cli help"), (2, "pipeline flow")] {
        {
            let app = harness.state_mut();
            app.show_shortcuts = flag == 0;
            app.show_cli_help = flag == 1;
            app.show_pipeline_flow = flag == 2;
        }
        settle(&mut harness);
        assert_all_controls_named(&harness, surface);
    }
}
