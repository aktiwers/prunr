//! The app under the AccessKit test harness: real widgets, found by name.

use egui_kittest::Harness;

use super::harness::{
    assert_all_controls_named, click, enabled, harness, harness_builder, loaded, loaded_with, names, new_app,
    settle, texts, tree,
};
use crate::gui::app::PrunrApp;
use crate::gui::views::settings::SettingsTab;

/// The real toolbar, per model, shows exactly the chips the surface
/// table says act on that model's image.
#[test]
fn each_model_toolbar_shows_only_its_chips() {
    use crate::gui::knob_catalog::{shows, Surface};
    use crate::gui::settings::SettingsModel;
    // A chip shows when any of its surfaces does: both generative
    // erasers have an Advanced chip.
    let chips: &[(&str, &[Surface])] = &[
        ("Mask", &[Surface::Mask]),
        ("Sketch", &[Surface::Lines]),
        ("Effect", &[Surface::Effect]),
        ("Faces", &[Surface::Faces]),
        ("Depth", &[Surface::Depth]),
        ("Background", &[Surface::Background]),
        ("Scale", &[Surface::Upscale]),
        ("Refine", &[Surface::Upscale]),
        ("Strength", &[Surface::Colorize]),
        ("Prompt", &[Surface::SdChips]),
        ("Advanced", &[Surface::SdChips, Surface::RoremChips, Surface::FillChips]),
        ("Paint Brush", &[Surface::Brushes]),
        ("Magic Brush", &[Surface::Brushes]),
    ];
    for model in SettingsModel::ALL {
        let harness = loaded_with(|app| app.settings.model = model);
        let names = names(&harness);
        for &(chip, surfaces) in chips {
            let expected = surfaces.iter().any(|&surface| shows(model, surface));
            assert_eq!(names.iter().any(|n| n == chip), expected, "{model:?}: {chip}");
        }
    }
}

#[test]
fn a_run_in_flight_names_its_cancel_button() {
    use crate::gui::dispatch_progress::DispatchProgress;
    use prunr_core::{ProgressUpdate, Step, TileRect};
    let harness = loaded_with(|app| {
        let mut progress = DispatchProgress { inner: (1, 2), ..DispatchProgress::new(crate::gui::dispatch_progress::ProgressKind::Upscale, Step::Upscaling.label()) };
        progress.apply(&ProgressUpdate::Tiles(vec![
            TileRect { x: 0.0, y: 0.0, w: 0.5, h: 1.0 },
            TileRect { x: 0.5, y: 0.0, w: 0.5, h: 1.0 },
        ]));
        progress.apply(&ProgressUpdate::Tile { index: 1, done: false });
        app.processor.show_progress_for_test(progress);
    });
    assert_all_controls_named(&harness, "run in flight");
    let names = names(&harness);
    assert!(names.iter().any(|n| n == "Cancel"), "the capsule's Cancel is named: {names:?}");
}

/// Settings › Hardware as Windows shows it, which offers both runtimes:
/// every runtime row and the runtime choice are named. The page is drawn
/// on its own because the app reads this machine's platform and store.
#[test]
fn the_hardware_page_names_every_runtime_control() {
    use crate::gui::hardware_cache::HardwareInstallCache;
    use crate::gui::views::settings::{render_page_hardware, HardwareSectionContext};
    use crate::runtime_install::RuntimeId;
    use crate::runtime_install::RuntimeChoice;
    let mut both = HardwareInstallCache::all_installed();
    both.next_runtime = Some(RuntimeChoice::Store(RuntimeId::OpenVino));
    let mut built_in_only = HardwareInstallCache::default();
    built_in_only.built_in_gpu = true;
    let installing = Some((crate::runtime_install::InstallTarget::Runtime(RuntimeId::OpenVino), "Downloading 40% (32/80 MB)".to_string()));
    let shapes = [
        ("both installed", both, None, vec!["Uninstall OpenVINO Runtime", "Built in", "OpenVINO"]),
        ("OpenVINO installing", built_in_only, installing, vec!["Cancel"]),
        ("none installed", HardwareInstallCache::default(), None, vec!["Install OpenVINO Runtime"]),
    ];
    // Where no runtime is offered for download (Windows), the page has no
    // install, uninstall or cancel; a runtime installed anyway still gives
    // the choice. Every control is still named.
    let offered = RuntimeId::OFFERED.contains(&RuntimeId::OpenVino);
    for (shape, installs, installing, expected) in shapes {
        let download_controls = |name: &&str| name.ends_with("OpenVINO Runtime") || *name == "Cancel";
        let expected: Vec<&str> = expected.into_iter().filter(|n| offered || !download_controls(n)).collect();
        let ctx = HardwareSectionContext {
            offered: RuntimeId::OFFERED,
            installs,
            installing,
            sd_margin_mb: 2048,
            cpu_only: false,
            test: Default::default(),
            gpu_failure: None,
            testable: true,
            test_waits: false,
        };
        let mut harness = Harness::new_ui(move |ui| { render_page_hardware(ui, &ctx); });
        harness.run_steps(2);
        assert_all_controls_named(&harness, shape);
        let present = names(&harness);
        for name in expected {
            assert!(present.iter().any(|n| n == name), "{shape}: {name} missing: {present:?}");
        }
        let choice_shown = present.iter().any(|n| n == "Built in");
        assert_eq!(choice_shown, shape == "both installed", "{shape}: the runtime choice needs two to load");
    }
}

/// Test this computer: offered with an accelerator, Test before a test
/// and Test again with Forget after one; without an accelerator (or on
/// CPU only) the row is not there. After a test on an NVIDIA card without
/// CUDA, the table offers it. Every control is named.
#[test]
fn the_computer_test_is_offered_with_an_accelerator() {
    use crate::gui::hardware_cache::{Cell, Engine, HardwareInstallCache, TestResults, TestRow, Time};
    use crate::gui::views::settings::{render_page_hardware, HardwareSectionContext};
    use crate::runtime_install::RuntimeId;
    use prunr_models::ModelId;
    use std::sync::Arc;
    let directml = Engine { name: "DirectML", installed: true };
    let time = |ms| Cell::Time(Time::of(ms));
    let row = |model, cells: Vec<Cell>, runs_on| TestRow { model, cpu: Some(Time::of(670.0)), cells, runs_on, complete: true };
    let tested = Arc::new(TestResults {
        tested: Some(1),
        engines: vec![directml, Engine { name: "CUDA", installed: true }],
        rows: vec![row(ModelId::Silueta, vec![time(40.0), time(57.0)], "DirectML"), row(ModelId::BiRefNetLite, vec![Cell::Fails, time(450.0)], "CUDA")],
        ..Default::default()
    });
    let cancelled = Arc::new(TestResults { tested: None, ..(*tested).clone() });
    let without_cuda = Arc::new(TestResults {
        tested: Some(1),
        engines: vec![directml, Engine { name: "CUDA", installed: false }],
        rows: vec![row(ModelId::BiRefNetLite, vec![Cell::Fails, Cell::Untested], "CPU")],
        ..Default::default()
    });
    let never = Arc::new(TestResults { engines: vec![directml], ..Default::default() });
    let driver_changed = Arc::new(TestResults { engines: vec![directml], changed: vec![prunr_core::ep_compat::ScopePart::Drivers], ..Default::default() });
    let offer = format!("{} ({})", RuntimeId::Cuda.install_button_label(), RuntimeId::Cuda.download_label(false));
    let offer = offer.as_str();
    let shapes = [
        ("never tested", Arc::clone(&never), true, vec!["Test this computer"], vec!["Forget the test", offer], Some("Not tested yet")),
        ("not tested since the driver changed", driver_changed, true, vec!["Test this computer"], vec!["Forget the test"], Some("Not tested since the graphics driver changed")),
        ("tested", Arc::clone(&tested), true, vec!["Test this computer", "Forget the test"], vec![offer], None),
        ("cancelled after two models", cancelled, true, vec!["Test this computer", "Forget the test"], vec![], None),
        ("tested without CUDA", without_cuda, true, vec!["Test this computer", "Forget the test", offer], vec![], None),
        ("no accelerator", tested, false, vec![], vec!["Test this computer", "Forget the test"], None),
    ];
    for (shape, test, testable, present_names, absent_names, note) in shapes {
        let ctx = HardwareSectionContext {
            offered: RuntimeId::OFFERED,
            installs: HardwareInstallCache::default(),
            test,
            installing: None,
            sd_margin_mb: 2048,
            cpu_only: false,
            gpu_failure: None,
            testable,
            test_waits: false,
        };
        let mut harness = Harness::new_ui(move |ui| { render_page_hardware(ui, &ctx); });
        harness.run_steps(2);
        assert_all_controls_named(&harness, shape);
        let present = names(&harness);
        for name in present_names {
            assert!(present.iter().any(|n| n == name), "{shape}: {name} missing: {present:?}");
        }
        for name in absent_names {
            assert!(!present.iter().any(|n| n == name), "{shape}: {name} shown: {present:?}");
        }
        if let Some(note) = note {
            assert!(texts(&harness).iter().any(|t| t.starts_with(note)), "{shape}: no note {note:?}");
        }
    }
}

/// CUDA installed while an older runtime from the Runtime Store is loaded
/// says it waits for Prunr's own, not that it is in use.
#[test]
fn an_add_on_says_when_it_waits_for_prunrs_own_runtime() {
    use crate::gui::hardware_cache::HardwareInstallCache;
    use crate::gui::views::settings::{render_page_hardware, HardwareSectionContext};
    use crate::runtime_install::RuntimeId;
    if !RuntimeId::Cuda.is_addon() {
        return;
    }
    for (waits, line) in [(true, RuntimeId::Cuda.waiting_line()), (false, RuntimeId::Cuda.installed_line())] {
        let mut installs = HardwareInstallCache::default();
        (installs.cuda, installs.nvidia_libraries, installs.addon_waits) = (true, true, waits);
        let ctx = HardwareSectionContext {
            offered: RuntimeId::OFFERED,
            installs,
            installing: None,
            sd_margin_mb: 2048,
            cpu_only: false,
            test: Default::default(),
            gpu_failure: None,
            testable: false,
            test_waits: false,
        };
        let mut harness = Harness::new_ui(move |ui| { render_page_hardware(ui, &ctx); });
        harness.run_steps(2);
        assert!(texts(&harness).iter().any(|t| t == line), "waits {waits}: {line:?} missing");
    }
}

/// The NVIDIA libraries are offered whenever the CUDA runtime lacks its
/// own copy, whatever the system has: Download then, Cancel while they
/// install, neither once they are there or without a CUDA runtime.
#[test]
fn the_nvidia_libraries_are_offered_while_the_cuda_runtime_lacks_them() {
    use crate::gui::hardware_cache::HardwareInstallCache;
    use crate::gui::views::settings::{render_page_hardware, HardwareSectionContext};
    use crate::runtime_install::{InstallTarget, RuntimeChoice, RuntimeId};
    use prunr_core::engine::{GpuStack, GpuStartFailure, StartFailure};
    static MISSING: GpuStartFailure = GpuStartFailure { stack: GpuStack::Cuda, why: StartFailure::MissingLibrary(String::new()) };
    if !RuntimeId::OFFERED.contains(&RuntimeId::Cuda) {
        return;
    }
    let mut on_cuda = HardwareInstallCache::default();
    on_cuda.cuda = true;
    on_cuda.loaded_runtime = Some(RuntimeChoice::Store(RuntimeId::Cuda));
    let installing = Some((InstallTarget::NvidiaLibraries, "Downloading 10% (170/1700 MB)".to_string()));
    let mut with_libraries = on_cuda;
    with_libraries.nvidia_libraries = true;
    let shapes = [
        ("missing a library", on_cuda, Some(&MISSING), None, Some("Download NVIDIA libraries")),
        ("starts on the system's", on_cuda, None, None, Some("Download NVIDIA libraries")),
        ("installing", on_cuda, Some(&MISSING), installing, Some("Cancel")),
        ("installed", with_libraries, None, None, None),
        ("no CUDA runtime", HardwareInstallCache::default(), None, None, None),
    ];
    for (shape, installs, failure, installing, expected) in shapes {
        let ctx = HardwareSectionContext {
            offered: RuntimeId::OFFERED,
            installs,
            installing,
            sd_margin_mb: 2048,
            cpu_only: false,
            test: Default::default(),
            gpu_failure: failure,
            testable: false,
            test_waits: false,
        };
        let mut harness = Harness::new_ui(move |ui| { render_page_hardware(ui, &ctx); });
        harness.run_steps(2);
        assert_all_controls_named(&harness, shape);
        let present = names(&harness);
        for name in ["Download NVIDIA libraries", "Cancel"] {
            assert_eq!(present.iter().any(|n| n == name), expected == Some(name), "{shape}: {name}: {present:?}");
        }
    }
}

/// Settings speaks of the Stable Diffusion eraser only once one is
/// installed: its RAM check, its keep-loaded switch and memory guard,
/// and the CPU-only warning that it needs the accelerator. The memory
/// line names each eraser installed.
#[test]
fn stable_diffusion_settings_show_only_with_an_sd_eraser_installed() {
    use crate::gui::hardware_cache::HardwareInstallCache;
    use crate::gui::settings::SettingsModel;
    let mut harness = harness();
    let line_of = |harness: &Harness<'_, PrunrApp>, name: &str| texts(harness).iter().any(|t| t.starts_with(name));
    let rorem_only = HardwareInstallCache::with_models(&[SettingsModel::RoremInpaint]);
    for (installs, shown) in [(HardwareInstallCache::default(), false), (HardwareInstallCache::all_installed(), true), (rorem_only, true)] {
        let app = harness.state_mut();
        app.hardware_install_cache = installs;
        app.show_settings = true;
        app.settings_tab = SettingsTab::Performance;
        settle(&mut harness);
        let switch = names(&harness).iter().any(|n| n == "Keep the Stable Diffusion eraser loaded");
        assert_eq!(switch, shown, "Performance: the keep-loaded switch");
        harness.state_mut().settings_tab = SettingsTab::Hardware;
        settle(&mut harness);
        let sd15 = installs.has_model(SettingsModel::SdInpaint) || installs.lcm_bundle;
        assert_eq!(line_of(&harness, "Stable Diffusion 1.5"), sd15, "Hardware: the Stable Diffusion 1.5 memory line");
        assert_eq!(line_of(&harness, "RORem"), installs.has_model(SettingsModel::RoremInpaint), "Hardware: the RORem memory line");
    }
}

#[test]
fn the_model_store_opens_from_its_toolbar_button() {
    let mut harness = harness();
    settle(&mut harness);
    assert!(harness.state().model_store.is_none());
    click(&mut harness, "Model Store");
    assert_eq!(harness.state().model_store.map(|r| r.filter), Some(None), "the whole store, no category picked");
}

#[test]
fn mirror_sees_what_the_harness_sees() {
    use std::collections::BTreeSet;
    let mirror = crate::gui::automation::tree::Mirror::default();
    let plugin = mirror.clone();
    let mut harness = harness_builder().build_eframe(move |cc| {
        let app = new_app(cc);
        cc.egui_ctx.add_plugin(plugin);
        app
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

/// While an upscale runs, Scale, Refine and Faces can't be touched: a
/// change would make the landing run stale and throw it away.
#[test]
fn an_upscale_run_holds_the_chips_it_reads() {
    use crate::gui::settings::SettingsModel;
    for running in [false, true] {
        let harness = loaded_with(|app| {
            app.settings.model = SettingsModel::RealEsrganUpscale;
            app.processor.set_upscale_in_flight_for_test(running);
        });
        let tree = tree(&harness);
        for chip in ["Scale", "Refine", "Faces"] {
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
        let harness = loaded_with(|app| {
            app.settings.model = SettingsModel::ColorizeDdcolor;
            app.processor.set_colorize_in_flight_for_test(running);
        });
        let tree = tree(&harness);
        let node = tree.iter().find(|n| n.name.as_deref() == Some("Strength")).expect("Strength shown");
        assert_eq!(node.disabled, running, "Strength while running: {running}");
    }
}

/// No model's Depth chip: Point picks the lens blur, and Set focus waits
/// for a click on the image.
#[test]
fn the_depth_chip_s_controls_act() {
    let mut h = loaded_with(|app| {
        app.settings.model = crate::gui::settings::SettingsModel::None;
        app.hardware_install_cache.depth = true;
    });
    click(&mut h, "Depth");
    click(&mut h, "Point");
    assert_eq!(h.state().batch.items[0].settings.depth.blur, prunr_core::depth::LensBlurMode::Point);
    click(&mut h, "All depth settings");
    click(&mut h, "Set focus");
    assert!(h.state().picking_depth_focus, "the next click on the image sets the focus");
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

/// A Model Store card's details open in place, the store staying up.
#[test]
fn a_store_card_s_details_open_in_place() {
    let mut h = harness();
    h.state_mut().model_store = Some(crate::gui::views::adjustments_toolbar::ModelStoreRequest { filter: None });
    settle(&mut h);
    click(&mut h, "Details Silueta");
    assert!(h.state().model_store.is_some(), "the store stays open");
    let toggle = tree(&h).iter().find(|n| n.name.as_deref() == Some("Details Silueta")).map(|n| n.toggled);
    assert_eq!(toggle, Some(Some(true)), "the card opened");
}

/// The Model Store offers Test on an installed model this computer never
/// measured or measured on some accelerators only, and none on a measured
/// one, nor on one a test would not time here; while a model's own test runs, its card has Cancel and no window
/// covers the store. Every control is named.
#[test]
fn a_store_card_tests_its_own_model() {
    use crate::gui::hardware_cache::{Cell, Engine, TestResults, TestRow, Time};
    use prunr_models::ModelId;
    let mut h = harness();
    {
        let app = h.state_mut();
        app.settings.active_backend = "CUDA".into();
        app.test_results = std::sync::Arc::new(TestResults {
            tested: None,
            engines: vec![Engine { name: "CUDA", installed: true }],
            rows: vec![
                TestRow { model: ModelId::Silueta, cpu: Some(Time::of(620.0)), cells: vec![Cell::Time(Time::of(40.0))], runs_on: "CUDA", complete: true },
                TestRow { model: ModelId::Migan, cpu: Some(Time::of(970.0)), cells: vec![Cell::Untested], runs_on: "CPU", complete: false },
            ],
            testable: vec![ModelId::Silueta, ModelId::U2net, ModelId::Migan],
            after_probe: true,
            ..Default::default()
        });
        app.model_store = Some(crate::gui::views::adjustments_toolbar::ModelStoreRequest { filter: None });
    }
    settle(&mut h);
    assert_all_controls_named(&h, "the Model Store with tests to offer");
    let present = names(&h);
    assert!(present.iter().any(|n| n == "Test U2Net"), "an installed model never measured here: {present:?}");
    assert!(!present.iter().any(|n| n == "Test Silueta"), "a measured one has nothing to test");
    assert!(!present.iter().any(|n| n == "Test BiRefNet-lite"), "installed, but no accelerator here would try it");
    assert!(enabled(&h, "Test U2Net"), "nothing else uses the machine");
    assert!(texts(&h).iter().any(|t| t == "Fastest here: CUDA, 0.04 s"), "{:?}", texts(&h));
    assert!(present.iter().any(|n| n == "Test Eraser (MI-GAN)"), "a test cancelled before every accelerator offers Test again");
    assert!(texts(&h).iter().any(|t| t == "Partly tested on this computer."));

    let child = crate::gui::tests::fixtures::idle_child();
    h.state_mut().machine_test = Some(crate::gui::machine_test::MachineTest::for_test(child, Some(ModelId::U2net)));
    settle(&mut h);
    assert_all_controls_named(&h, "the Model Store while a model's test runs");
    let present = names(&h);
    assert!(present.iter().any(|n| n == "Cancel the test of U2Net"), "{present:?}");
    assert!(!h.state().open_modals().contains(&"machine_test"), "no window over the store");
    click(&mut h, "Cancel the test of U2Net");
    assert!(h.state().machine_test.is_none(), "Cancel ends it");
}
