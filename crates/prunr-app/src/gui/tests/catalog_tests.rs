//! Every surface in the catalog, for every model it shows for: opened
//! with the clicks a user would make, then checked on every variant.

use egui_kittest::kittest::Queryable;
use egui_kittest::Harness;

use super::harness::{harness, loaded_with, settle, tree};
use crate::gui::app::PrunrApp;
use crate::gui::automation::checks::Violation;
use crate::gui::automation::tree::readable_name;
use crate::gui::brush_state::BrushSettings;
use crate::gui::capability::ItemFacts;
use crate::gui::item_settings::ItemSettings;
use crate::gui::knob_catalog::{shows, surface_gate, GateContext, Surface};
use crate::gui::settings::SettingsModel;
use crate::gui::surface_catalog::{Kind, Start, Step, Visit, UNVISITED_MODALS, VISITS};
use crate::gui::views::shortcuts::Chord;

/// A fresh window for `visit`, on `model` (`None`: the app's own), its
/// tooltips and clipped text recorded for `check`.
fn start(visit: &Visit, model: Option<SettingsModel>) -> Harness<'static, PrunrApp> {
    crate::gui::automation::record::turn_on();
    let pick = |app: &mut PrunrApp| {
        if let Some(model) = model {
            app.settings.model = model;
        }
    };
    match visit.start {
        Start::Image => loaded_with(pick),
        Start::Empty => {
            let mut h = harness();
            pick(h.state_mut());
            settle(&mut h);
            h
        }
    }
}

fn run(h: &mut Harness<'_, PrunrApp>, step: Step) -> Result<(), String> {
    match step {
        Step::Click(name) | Step::Expand(name) => {
            let open = matches!(step, Step::Expand(_))
                && tree(h).iter().filter(|n| n.name.as_deref() == Some(name)).last().and_then(|n| n.toggled) == Some(true);
            let node = h
                .query_all_by(|n| n.label().as_deref().and_then(readable_name).as_deref() == Some(name))
                .last()
                .ok_or_else(|| format!("no control named {name:?}"))?;
            if !open {
                node.click();
            }
        }
        Step::Key(chord) => {
            let chord = Chord::parse(chord).ok_or_else(|| format!("unknown chord {chord:?}"))?;
            h.key_press_modifiers(chord.modifiers(), chord.key);
        }
    }
    settle(h);
    Ok(())
}

/// Whether the surface `kind` names is up.
fn is_open(h: &Harness<'_, PrunrApp>, kind: Kind) -> Option<bool> {
    match kind {
        Kind::Popover | Kind::Flyout => Some(egui::Popup::is_any_open(&h.ctx)),
        Kind::Modal(name) => Some(h.state().open_modals().contains(&name)),
        Kind::Window | Kind::Strip => None,
    }
}

/// The control socket's `check` on the harness's last frame.
fn check(h: &Harness<'_, PrunrApp>, expect: &[&str]) -> Vec<Violation> {
    crate::gui::automation::check(&tree(h), &h.ctx, expect)
}

/// Why the visit's surface waits on the image shown, if it does: then
/// its chip is greyed and there is nothing to open.
fn gated(h: &Harness<'_, PrunrApp>, visit: &Visit) -> Option<String> {
    let app = h.state();
    let ctx = GateContext {
        model: app.settings.model,
        facts: ItemFacts::of(app.batch.selected_item()?),
        processing: false,
    };
    surface_gate(visit.surface?, &ctx).map(Into::into)
}

/// Opens `visit` for `model` and checks it, then each variant on a fresh
/// window, then the dismissal; every problem as one line.
fn walk(visit: &Visit, model: Option<SettingsModel>) -> Vec<String> {
    let at = |state: &str, what: String| format!("{} ({model:?}) {state}: {what}", visit.id);
    let open = |h: &mut Harness<'_, PrunrApp>| visit.open.iter().try_for_each(|&s| run(h, s));
    let mut problems = Vec::new();

    let mut h = start(visit, model);
    if let Some(reason) = gated(&h, visit) {
        let opener = visit.open.first().and_then(|&step| match step {
            Step::Click(name) | Step::Expand(name) => tree(&h).iter().find(|n| n.name.as_deref() == Some(name)).map(|n| n.disabled),
            Step::Key(_) => None,
        });
        return match opener {
            Some(true) => Vec::new(),
            _ => vec![at("gated", format!("{reason:?}, yet its opener is not greyed"))],
        };
    }
    if let Err(e) = open(&mut h) {
        return vec![at("opening", e)];
    }
    if is_open(&h, visit.kind) == Some(false) {
        problems.push(at("opening", format!("{:?} did not open", visit.kind)));
    }
    problems.extend(check(&h, visit.expect).iter().map(|v| at("open", v.to_string())));
    if let Some(step) = visit.dismiss {
        match run(&mut h, step) {
            Err(e) => problems.push(at("dismissing", e)),
            Ok(()) if is_open(&h, visit.kind) == Some(true) => problems.push(at("dismissing", format!("{step:?} left it open"))),
            Ok(()) => {}
        }
    }

    for variant in visit.variants() {
        let mut h = start(visit, model);
        let ran = open(&mut h).and_then(|()| variant.steps.iter().try_for_each(|&s| run(&mut h, s)));
        match ran {
            Err(e) => problems.push(at(variant.name, e)),
            Ok(()) => problems.extend(check(&h, variant.expect).iter().map(|v| at(variant.name, v.to_string()))),
        }
    }
    problems
}

/// The models a visit is checked for: every one its surface shows for,
/// or once on the app's own when it shows for all.
fn models(visit: &Visit) -> Vec<Option<SettingsModel>> {
    match visit.surface {
        Some(surface) => SettingsModel::ALL.into_iter().filter(|&m| shows(m, surface)).map(Some).collect(),
        None => vec![None],
    }
}

/// Each visit for each of its models on its own fresh window, spread
/// over threads: walked one by one this is most of a minute.
#[test]
fn every_surface_names_its_controls_and_shows_what_it_must() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let jobs: Vec<(&Visit, Option<SettingsModel>)> = VISITS.iter().flat_map(|v| models(v).into_iter().map(move |m| (v, m))).collect();
    let next = AtomicUsize::new(0);
    let walkers = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
    let problems: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..walkers)
            .map(|_| {
                scope.spawn(|| {
                    let mut found = Vec::new();
                    while let Some(&(visit, model)) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                        found.extend(walk(visit, model));
                    }
                    found
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("a walker panicked")).collect()
    });
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}

/// A strip's full panel stays open across closing the strip: opening
/// it again by `Expand` finds it open rather than closing it, and its
/// toggle says so to the tree.
#[test]
fn a_panel_left_open_is_found_open_again() {
    let mut h = super::harness::loaded();
    let include_hair = |h: &Harness<'_, PrunrApp>| tree(h).iter().any(|n| n.name.as_deref() == Some("Include hair"));
    for _ in 0..2 {
        for step in [Step::Click("Faces"), Step::Expand("More face settings")] {
            run(&mut h, step).unwrap();
        }
        assert!(include_hair(&h), "the full panel is open");
        run(&mut h, Step::Key("Escape")).unwrap();
        run(&mut h, Step::Key("Escape")).unwrap();
    }
}

/// At the minimum window every strip's full panel can be reached: its
/// toggle lies inside the window (the Mask strip ran its '...' off the
/// right edge at 900 px live).
#[test]
fn every_panel_toggle_is_on_screen_at_the_minimum_width() {
    use crate::gui::theme::MIN_WINDOW_SIZE;
    // The harness measures text ~3% narrower than a live window, so the
    // minimum width is checked 3% narrower too.
    let width = (MIN_WINDOW_SIZE[0] * 0.97).floor();
    let mut off = Vec::new();
    for visit in VISITS.iter().filter(|v| v.start == Start::Image) {
        let toggles: Vec<&str> = visit.open.iter().copied()
            .chain(visit.variants().into_iter().flat_map(|v| v.steps))
            .filter_map(|step| match step { Step::Expand(name) => Some(name), _ => None })
            .collect();
        let Some(&toggle) = toggles.first() else { continue };
        for model in models(visit) {
            let mut h = super::harness::harness_sized(egui::vec2(width, MIN_WINDOW_SIZE[1]));
            {
                let app = h.state_mut();
                super::fixtures::cut_out(super::fixtures::push_test_item(app, 1), prunr_core::ModelKind::Silueta);
                app.batch.select_item(0);
                if let Some(model) = model {
                    app.settings.model = model;
                }
            }
            settle(&mut h);
            if gated(&h, visit).is_some() || visit.open.iter().try_for_each(|&s| run(&mut h, s)).is_err() {
                continue;
            }
            match tree(&h).iter().filter(|n| n.name.as_deref() == Some(toggle)).last().and_then(|n| n.rect) {
                Some(r) if r[2] <= width => {}
                r => off.push(format!("{} ({model:?}): {toggle:?} at {r:?}", visit.id)),
            }
        }
    }
    assert!(off.is_empty(), "{off:#?}");
}

#[test]
fn every_modal_is_visited_or_says_why_not() {
    for (name, ..) in PrunrApp::MODALS {
        let visited = VISITS.iter().any(|v| v.kind == Kind::Modal(name));
        let excused = UNVISITED_MODALS.iter().any(|(n, _)| *n == name);
        assert!(visited != excused, "{name}: visited {visited}, excused {excused}; exactly one");
    }
    for (name, _) in UNVISITED_MODALS {
        assert!(PrunrApp::MODALS.iter().any(|(n, ..)| n == name), "{name} is no modal");
    }
}

#[test]
fn every_surface_is_visited() {
    // A new Surface fails to compile here until it is placed; give it
    // its slot in `Surface::ALL` too.
    let slot = |s: Surface| match s {
        Surface::Mask => 0,
        Surface::Lines => 1,
        Surface::Effect => 2,
        Surface::Decoration => 3,
        Surface::Background => 4,
        Surface::Upscale => 5,
        Surface::Colorize => 6,
        Surface::SdChips => 7,
        Surface::RoremChips => 8,
        Surface::FillChips => 9,
        Surface::Brushes => 10,
        Surface::SelectionStrokes => 11,
        Surface::FillRegionKnobs => 12,
        Surface::Faces => 13,
        Surface::Depth => 14,
    };
    for (i, surface) in Surface::ALL.into_iter().enumerate() {
        assert_eq!(slot(surface), i, "{surface:?} out of place in Surface::ALL");
        assert!(VISITS.iter().any(|v| v.surface == Some(surface)), "{surface:?} has no visit in the surface catalog");
        assert!(SettingsModel::ALL.into_iter().any(|m| shows(m, surface)), "{surface:?} shows for no model");
    }
}

#[test]
fn visit_ids_are_unique() {
    let mut ids: Vec<&str> = VISITS.iter().map(|v| v.id).collect();
    ids.sort_unstable();
    let before = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), before, "two visits share an id");
}

/// One panel of `every_panel_reset_returns_to_the_default_preset`.
struct PanelReset {
    model: SettingsModel,
    /// What the default preset sets off Prunr's, for the panel.
    set: fn(&mut ItemSettings, &mut BrushSettings),
    open: &'static [Step],
    /// The panel's settings, as a comparable text.
    part: fn(&ItemSettings, &BrushSettings) -> String,
}

/// Every panel's Reset puts back the default preset's settings, not the
/// code's: each starts on Prunr's, the default preset sets it otherwise,
/// and after Reset it reads the preset's.
#[test]
fn every_panel_reset_returns_to_the_default_preset() {
    use crate::gui::presets::{prunr_view, PresetFile};
    use SettingsModel as M;
    use Step::{Click, Expand};
    let panels = [
        PanelReset { model: M::BiRefNetLite, set: |s, _| s.effect = prunr_core::SubjectEffect::ALL[1],
            open: &[Click("Effect"), Expand("All effects"), Click("Reset")], part: |s, _| format!("{:?}", s.effect) },
        PanelReset { model: M::BiRefNetLite, set: |s, _| s.decoration.outline_width = 6.0,
            open: &[Click("Outline and shadow"), Expand("All outline and shadow settings"), Click("Reset")], part: |s, _| format!("{:?}", s.decoration) },
        PanelReset { model: M::BiRefNetLite, set: |s, _| s.bg = Some([10, 20, 30, 255]),
            open: &[Click("Background"), Expand("All background settings"), Click("Reset")], part: |s, _| format!("{:?}", s.backdrop()) },
        PanelReset { model: M::BiRefNetLite, set: |s, _| s.faces.strength_pct = 80,
            open: &[Click("Faces"), Expand("More face settings"), Click("Reset")], part: |s, _| format!("{:?}", s.faces) },
        PanelReset { model: M::None, set: |s, _| s.depth.fog_pct = 30,
            open: &[Click("Depth"), Expand("All depth settings"), Click("Reset")], part: |s, _| format!("{:?}", s.depth) },
        PanelReset { model: M::Nomos2Upscale, set: |s, _| s.sharpen = 0.4,
            open: &[Click("Refine"), Expand("All refine settings"), Click("Reset")], part: |s, _| format!("{:?}", s.sharpen) },
        PanelReset { model: M::Nomos2Upscale, set: |s, _| s.output_scale = prunr_core::OutputScale::X2,
            open: &[Click("Scale"), Click("Reset")], part: |s, _| format!("{:?}", s.output_scale) },
        PanelReset { model: M::ColorizeDdcolor, set: |s, _| s.color_strength_pct = 120,
            open: &[Click("Strength"), Click("Reset")], part: |s, _| format!("{:?}", s.color_strength_pct) },
        PanelReset { model: M::Inpaint, set: |_, b| b.inpaint_sharpen = 0.2,
            open: &[Click("Advanced"), Click("Reset")], part: |_, b| format!("{:?}", b.inpaint_sharpen) },
        PanelReset { model: M::RoremInpaint, set: |_, b| b.rorem_steps = 2,
            open: &[Click("Advanced"), Click("Reset")], part: |_, b| format!("{:?}", b.rorem_steps) },
        PanelReset { model: M::SdInpaint, set: |_, b| b.sd_prompt = "a meadow".into(),
            open: &[Click("Prompt"), Click("Reset")], part: |_, b| format!("{:?}", b.sd_prompt) },
    ];
    for panel in panels {
        let view = prunr_view(panel.model, None);
        let (mut look, mut brush) = (view.item_settings, view.brush.clone());
        (panel.set)(&mut look, &mut brush);
        let preset = PresetFile::default().with_settings(panel.model, &look, &brush);
        let mut h = loaded_with(|app| {
            app.settings.model = panel.model;
            app.settings.presets.insert("Look".into(), preset.into());
            app.settings.default_preset = "Look".into();
            app.settings.brush = view.brush.clone();
        });
        let shown = |h: &Harness<'_, PrunrApp>| (panel.part)(&h.state().batch.items[0].settings, &h.state().settings.brush);
        let before = shown(&h);
        for &step in panel.open {
            run(&mut h, step).unwrap_or_else(|err| panic!("{:?}: {err}", panel.model));
        }
        let after = shown(&h);
        assert_ne!(before, after, "{:?} {:?}: Reset changed nothing", panel.model, panel.open);
        assert_eq!(after, (panel.part)(&look, &brush), "{:?} {:?}: the default preset's", panel.model, panel.open);
    }
}
