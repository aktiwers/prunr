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

#[test]
fn mirror_sees_what_the_harness_sees() {
    use std::collections::BTreeSet;
    let mirror = crate::gui::automation::tree::Mirror::default();
    let plugin = mirror.clone();
    let mut harness = Harness::builder()
        .with_size(egui::vec2(1400.0, 900.0))
        .build_eframe(move |cc| {
            egui_material_icons::initialize(&cc.egui_ctx);
            cc.egui_ctx.add_plugin(plugin);
            PrunrApp::new_for_test()
        });
    settle(&mut harness);
    let from_harness: BTreeSet<String> = names(&harness).into_iter().collect();
    let root = mirror.snapshot().expect("a frame ran");
    let from_mirror: BTreeSet<String> = crate::gui::automation::controls(&root).into_iter().filter_map(|n| n.name).collect();
    assert_eq!(from_harness, from_mirror);
}

#[test]
fn perform_routes_an_action_without_a_key() {
    use crate::gui::views::shortcuts::Action;
    let mut harness = harness();
    settle(&mut harness);
    let ctx = harness.ctx.clone();
    let app = harness.state_mut();
    assert!(!app.sidebar_hidden);
    app.perform(Action::ToggleQueue, &ctx);
    assert!(app.sidebar_hidden);
    app.perform(Action::Settings, &ctx);
    assert!(app.show_settings);
}

#[test]
fn control_socket_clicks_by_name_and_reports_state() {
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream};
    use crate::gui::automation::Automation;

    let mut harness = harness();
    settle(&mut harness);
    let port = TcpListener::bind(("127.0.0.1", 0)).unwrap().local_addr().unwrap().port();
    let ctx = harness.ctx.clone();
    harness.state_mut().automation = Some(Automation::start(&ctx, port).expect("bind the test port"));

    fn roundtrip(w: &mut TcpStream, r: &mut BufReader<TcpStream>, request: &str) -> serde_json::Value {
        writeln!(w, "{request}").unwrap();
        let mut line = String::new();
        r.read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line:?}"))
    }
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut w = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let mut r = BufReader::new(w.try_clone().unwrap());
        let ping = roundtrip(&mut w, &mut r, r#"{"cmd":"ping"}"#);
        let click = roundtrip(&mut w, &mut r, r#"{"cmd":"click","name":"Settings"}"#);
        let state = roundtrip(&mut w, &mut r, r#"{"cmd":"state"}"#);
        let missing = roundtrip(&mut w, &mut r, r#"{"cmd":"click","name":"No such control"}"#);
        done_tx.send((ping, click, state, missing)).unwrap();
    });
    // The socket needs frames to run; feed the injected input through the
    // harness the way raw_input_hook does for eframe.
    let mut result = None;
    for _ in 0..3000 {
        let events = harness.state_mut().automation.as_mut().unwrap().take_events();
        harness.input_mut().events.extend(events);
        harness.step();
        if let Ok(r) = done_rx.try_recv() {
            result = Some(r);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let (ping, click, state, missing) = result.expect("the client finished");
    assert_eq!(ping["result"], "pong");
    assert_eq!(click["ok"], true, "{click}");
    assert!(harness.state().show_settings, "the click opened Settings");
    let modals = state["result"]["modals"].as_array().expect("modals");
    assert!(modals.iter().any(|m| m == "settings"), "{state}");
    assert_eq!(missing["ok"], false);
}
