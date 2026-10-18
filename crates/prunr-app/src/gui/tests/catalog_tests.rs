//! Every surface in the catalog, for every model it shows for: opened
//! with the clicks a user would make, then checked on every variant.

use egui_kittest::kittest::Queryable;
use egui_kittest::Harness;

use super::harness::{harness, loaded_with, settle, tree};
use crate::gui::app::PrunrApp;
use crate::gui::automation::checks::{self, Violation};
use crate::gui::automation::tree::readable_name;
use crate::gui::capability::ItemFacts;
use crate::gui::knob_catalog::{shows, surface_gate, GateContext, Surface};
use crate::gui::settings::SettingsModel;
use crate::gui::surface_catalog::{Kind, Start, Step, Visit, UNVISITED_MODALS, VISITS};
use crate::gui::views::shortcuts::Chord;

/// A fresh window for `visit`, on `model` (`None`: the app's own).
fn start(visit: &Visit, model: Option<SettingsModel>) -> Harness<'static, PrunrApp> {
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
        Step::Click(name) => {
            let node = h
                .query_all_by(|n| n.label().as_deref().and_then(readable_name).as_deref() == Some(name))
                .last()
                .ok_or_else(|| format!("no control named {name:?}"))?;
            node.click();
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

fn check(h: &Harness<'_, PrunrApp>, expect: &[&str]) -> Vec<Violation> {
    let root = tree(h);
    let mut found = checks::nameless(&root);
    found.extend(checks::missing(&root, expect));
    found
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
            Step::Click(name) => tree(&h).iter().find(|n| n.name.as_deref() == Some(name)).map(|n| n.disabled),
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

#[test]
fn every_modal_is_visited_or_says_why_not() {
    for (name, _) in PrunrApp::MODALS {
        let visited = VISITS.iter().any(|v| v.kind == Kind::Modal(name));
        let excused = UNVISITED_MODALS.iter().any(|(n, _)| *n == name);
        assert!(visited != excused, "{name}: visited {visited}, excused {excused}; exactly one");
    }
    for (name, _) in UNVISITED_MODALS {
        assert!(PrunrApp::MODALS.iter().any(|(n, _)| n == name), "{name} is no modal");
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
        Surface::Brushes => 9,
        Surface::SelectionStrokes => 10,
        Surface::FillRegionKnobs => 11,
        Surface::Faces => 12,
        Surface::Depth => 13,
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
