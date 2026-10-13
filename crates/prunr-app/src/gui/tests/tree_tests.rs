//! The app under the AccessKit test harness: real widgets, found by name.

use egui_kittest::kittest::Queryable;
use egui_kittest::Harness;

use super::fixtures::push_test_item;
use super::surface_tests::loaded;
use crate::gui::app::PrunrApp;
use crate::gui::automation::tree::{readable_name, ControlNode};
use crate::gui::item::BatchStatus;
use crate::gui::views::settings::SettingsTab;

pub(super) fn harness() -> Harness<'static, PrunrApp> {
    harness_sized(egui::vec2(1400.0, 900.0))
}

pub(super) fn harness_sized(size: egui::Vec2) -> Harness<'static, PrunrApp> {
    Harness::builder()
        .with_size(size)
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

pub(super) fn names(harness: &Harness<'_, PrunrApp>) -> Vec<String> {
    tree(harness).iter().filter(|n| n.is_control()).filter_map(|n| n.name.clone()).collect()
}

/// Every control in the tree has a readable name. Fails with the list of
/// nameless controls: an icon-only button, an unlabelled slider.
pub(super) fn assert_all_controls_named(harness: &Harness<'_, PrunrApp>, surface: &str) {
    let root = tree(harness);
    let nameless: Vec<String> = root
        .iter()
        .filter(|n| n.is_control() && n.name.is_none())
        .map(|n| format!("{:?} {:?} at {:?}", n.role, n.id, n.rect))
        .collect();
    assert!(nameless.is_empty(), "{surface}: controls without a readable name:\n{}", nameless.join("\n"));
}

#[test]
fn empty_window_names_every_control() {
    let mut harness = harness();
    settle(&mut harness);
    assert_all_controls_named(&harness, "empty window");
    let names = names(&harness);
    // The Model picker shows before any image, so a model can be picked
    // or downloaded first (its last row opens the Model Store).
    for expected in ["Open", "Settings", "Model"] {
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
    // A screen reader, and the control socket, pick an image by its name.
    assert!(names(&harness).iter().any(|n| n == "test1.png"), "the sidebar row is named after its file");
}

/// The real toolbar, per model, shows exactly the chips the surface
/// table says act on that model's image.
#[test]
fn each_model_toolbar_shows_only_its_chips() {
    use crate::gui::knob_catalog::{shows, Surface};
    use crate::gui::settings::SettingsModel;
    let chips: &[(&str, Surface)] = &[
        ("Mask", Surface::Mask),
        ("Sketch", Surface::Lines),
        ("Effect", Surface::Effect),
        ("Faces", Surface::Faces),
        ("Depth", Surface::Depth),
        ("Background", Surface::Background),
        ("Scale", Surface::Upscale),
        ("Refine", Surface::Upscale),
        ("Strength", Surface::Colorize),
        ("Prompt", Surface::SdChips),
        ("Paint Brush", Surface::Brushes),
        ("Magic Brush", Surface::Brushes),
    ];
    for model in SettingsModel::ALL {
        let mut harness = harness();
        {
            let app = harness.state_mut();
            app.settings.model = model;
            let item = push_test_item(app, 1);
            item.status = BatchStatus::Done;
            app.batch.select_item(0);
        }
        settle(&mut harness);
        let names = names(&harness);
        for &(chip, surface) in chips {
            assert_eq!(names.iter().any(|n| n == chip), shows(model, surface), "{model:?}: {chip}");
        }
    }
}

#[test]
fn a_run_in_flight_names_its_cancel_button() {
    use crate::gui::dispatch_progress::DispatchProgress;
    use prunr_core::{ProgressUpdate, Step, TileRect};
    let mut harness = harness();
    {
        let app = harness.state_mut();
        let item = push_test_item(app, 1);
        item.status = BatchStatus::Done;
        app.batch.select_item(0);
        let mut progress = DispatchProgress { inner: (1, 2), ..DispatchProgress::new(crate::gui::dispatch_progress::ProgressKind::Upscale, Step::Upscaling.label()) };
        progress.apply(&ProgressUpdate::Tiles(vec![
            TileRect { x: 0.0, y: 0.0, w: 0.5, h: 1.0 },
            TileRect { x: 0.5, y: 0.0, w: 0.5, h: 1.0 },
        ]));
        progress.apply(&ProgressUpdate::Tile { index: 1, done: false });
        app.processor.show_progress_for_test(progress);
    }
    settle(&mut harness);
    assert_all_controls_named(&harness, "run in flight");
    let names = names(&harness);
    assert!(names.iter().any(|n| n == "Cancel"), "the capsule's Cancel is named: {names:?}");
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
    assert!(names(&harness).iter().any(|n| n == "Editing"), "{:?}", names(&harness));
}

#[test]
fn help_modals_name_every_control() {
    let mut harness = harness();
    for (flag, surface) in [(1, "cli help"), (2, "pipeline flow")] {
        {
            let app = harness.state_mut();
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
        if let Some(frame) = harness.state_mut().automation.as_mut().unwrap().take_frame() {
            harness.input_mut().events.extend(frame.events);
            if let Some(modifiers) = frame.modifiers {
                harness.input_mut().modifiers = modifiers;
            }
        }
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

/// While an upscale runs, Scale and Refine can't be touched: a change
/// would make the landing run stale and throw it away.
#[test]
fn an_upscale_run_holds_its_scale_and_refine_chips() {
    use crate::gui::settings::SettingsModel;
    for running in [false, true] {
        let mut harness = harness();
        {
            let app = harness.state_mut();
            app.settings.model = SettingsModel::RealEsrganUpscale;
            push_test_item(app, 1).status = BatchStatus::Done;
            app.batch.select_item(0);
            app.processor.set_upscale_in_flight_for_test(running);
        }
        settle(&mut harness);
        let tree = tree(&harness);
        for chip in ["Scale", "Refine"] {
            let node = tree.iter().find(|n| n.name.as_deref() == Some(chip)).unwrap_or_else(|| panic!("{chip} shown"));
            assert_eq!(node.disabled, running, "{chip} while running: {running}");
        }
    }
}


/// While a colorize runs, Strength can't be touched, like Refine.
#[test]
fn a_colorize_run_holds_its_strength_chip() {
    use crate::gui::settings::SettingsModel;
    for running in [false, true] {
        let mut harness = harness();
        {
            let app = harness.state_mut();
            app.settings.model = SettingsModel::ColorizeDdcolor;
            push_test_item(app, 1).status = BatchStatus::Done;
            app.batch.select_item(0);
            app.processor.set_colorize_in_flight_for_test(running);
        }
        settle(&mut harness);
        let tree = tree(&harness);
        let node = tree.iter().find(|n| n.name.as_deref() == Some("Strength")).expect("Strength shown");
        assert_eq!(node.disabled, running, "Strength while running: {running}");
    }
}

/// Each quick strip and the full panel its dots open name every control.
#[test]
fn quick_strips_name_every_control() {
    for (chip, on, more, in_panel) in [
        ("Mask", None, "All mask settings", "Contrast"),
        ("Sketch", Some("Whole photo"), "All line settings", "Style"),
        ("Effect", Some("Duotone"), "All effects", "Shadows"),
        ("Faces", None, "More face settings", "Include hair"),
        ("Background", Some("Cream"), "All background settings", "Custom color"),
    ] {
        let mut h = loaded();
        click(&mut h, chip);
        if let Some(on) = on {
            click(&mut h, on);
        }
        assert_all_controls_named(&h, &format!("{chip} strip"));
        click(&mut h, more);
        assert!(names(&h).iter().any(|n| n == in_panel), "{chip}: {in_panel} missing from the full panel");
        assert_all_controls_named(&h, &format!("{chip} panel"));
    }
}

/// No model's Depth chip: its strip and full panel name every control,
/// and Set focus waits for a click on the image.
#[test]
fn the_depth_chip_names_every_control() {
    let mut h = harness();
    {
        let app = h.state_mut();
        app.settings.model = crate::gui::settings::SettingsModel::None;
        app.hardware_install_cache.depth = true;
        push_test_item(app, 1).status = BatchStatus::Done;
        app.batch.select_item(0);
    }
    settle(&mut h);
    click(&mut h, "Depth");
    for knob in ["Off", "Point", "Tilt-shift", "Amount", "Fog"] {
        assert!(names(&h).iter().any(|n| n == knob), "{knob} missing from the Depth strip");
    }
    assert_all_controls_named(&h, "Depth strip");
    click(&mut h, "Point");
    assert_eq!(h.state().batch.items[0].settings.depth.blur, prunr_core::depth::LensBlurMode::Point);
    click(&mut h, "All depth settings");
    for knob in ["Set focus", "Aerial perspective"] {
        assert!(names(&h).iter().any(|n| n == knob), "{knob} missing from the Depth panel");
    }
    assert_all_controls_named(&h, "Depth panel");
    click(&mut h, "Set focus");
    assert!(h.state().picking_depth_focus, "the next click on the image sets the focus");
}

/// Sketch's full panel names Fade with distance.
#[test]
fn the_sketch_panel_names_fade_with_distance() {
    let mut h = super::surface_tests::loaded();
    click(&mut h, "Sketch");
    click(&mut h, "Whole photo");
    click(&mut h, "All line settings");
    assert!(names(&h).iter().any(|n| n == "Fade with distance"));
    assert_all_controls_named(&h, "Sketch panel");
}

/// A dropdown in a flyout opens its list in place: a popup list would
/// replace the flyout, the one popup egui keeps open.
#[test]
fn a_dropdown_keeps_its_flyout_open() {
    let mut h = loaded();
    click(&mut h, "Sketch");
    click(&mut h, "Whole photo");
    click(&mut h, "All line settings");
    click(&mut h, "Style");
    assert!(egui::Popup::is_any_open(&h.ctx), "the flyout stayed open");
    assert!(names(&h).iter().any(|n| n.starts_with("Rainbow")), "the list opened");
}

/// The Model Store names every control, a card's details open in place.
#[test]
fn the_model_store_names_every_control() {
    let mut h = harness();
    h.state_mut().model_store = Some(crate::gui::views::adjustments_toolbar::ModelStoreRequest { filter: None });
    settle(&mut h);
    assert_all_controls_named(&h, "model store");
    click(&mut h, "Details Silueta");
    assert!(h.state().model_store.is_some(), "the store stays open");
    let toggle = tree(&h).iter().find(|n| n.name.as_deref() == Some("Details Silueta")).map(|n| n.toggled);
    assert_eq!(toggle, Some(Some(true)), "the card opened");
    assert_all_controls_named(&h, "model store, a card open");
}
