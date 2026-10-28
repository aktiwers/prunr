//! Settings › Hardware › Test this computer: the bench in a process of
//! its own (`prunr bench --here`), so a provider that crashes takes only
//! the test down. Each model's place is written to this machine's store
//! (`ep_compat::set_measured`) as it lands, so a cancelled test keeps
//! what it measured; the app follows on one JSON line per event on stdout.

use prunr_models::{Column, ModelId};
use serde::{Deserialize, Serialize};

use super::{Event, Inputs};

/// Timed runs on each side, as `prunr bench` makes by default.
const RUNS: u32 = 3;

/// What the test tells the app.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Told {
    Step(Step),
    /// What a run of `model` on the accelerator in `column` found, as it
    /// is saved.
    Placed { model: ModelId, column: Column, measured: prunr_core::ep_compat::Measured },
    /// All done. `left_out`: installed models a test does not run, which
    /// take many minutes on the processor.
    Finished { left_out: Vec<ModelId> },
}

/// Where a test is: the model at `index` of `total`, on its `side` of
/// `sides`: 0 the CPU, then each accelerator it tries; `column` the
/// accelerator that runs now, `None` while the CPU does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub index: usize,
    pub total: usize,
    pub model: ModelId,
    pub side: usize,
    pub sides: usize,
    pub column: Option<Column>,
}

impl Step {
    /// How far the test is: each side a model runs on an equal share of
    /// its turn.
    pub fn fraction(&self) -> f32 {
        let within = self.side as f32 / self.sides.max(1) as f32;
        ((self.index.saturating_sub(1) as f32 + within) / self.total.max(1) as f32).clamp(0.0, 1.0)
    }
}

impl Told {
    pub fn line(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    pub fn parse(line: &str) -> Option<Self> {
        serde_json::from_str(line.trim()).ok()
    }
}

/// The column `id`'s GPU side runs on here, before it runs: a failure the
/// run records would otherwise hide which provider it was.
pub(crate) fn column_of(id: ModelId) -> Option<Column> {
    let ladder = prunr_core::engine::gpu_ladder(id);
    prunr_core::engine::column_for(id, super::first_tried(&ladder)?)
}

/// The column a test measures `id` in first; `None` when no accelerator
/// would try it.
fn first_column_in_a_test(id: ModelId) -> Option<Column> {
    prunr_core::engine::tries_in_a_test(id).first().map(|&(_, column)| column)
}

/// Run the test: every installed model the accelerator would try (a place
/// the registry keeps on the CPU for a slower, different or no-gain result
/// is tried again: `remeasure_places`), but the heavy ones; or only the
/// models `only` names (Settings' test of one model), which leaves the
/// whole computer's test date alone.
pub fn run(only: &[String], on: &mut dyn FnMut(Told)) {
    prunr_core::engine::remeasure_places();
    let installed = super::select(only, true, prunr_models::is_available).unwrap_or_default().into_iter().filter(|&id| prunr_models::is_available(id)).collect();
    let Plan { models, left_out } = plan(installed, first_column_in_a_test);
    let inputs = Inputs::built_in();
    // Each model once on each accelerator it may try here: a place for
    // every one, so the fastest can be told apart.
    let passes = |id| prunr_core::engine::tries_in_a_test(id).into_iter().map(|(provider, _)| Some(provider)).collect();
    super::run_passes(&models, &inputs, RUNS, passes, &mut |event| match event {
        Event::Stage { index, total, side, sides, provider, .. } => {
            let model = models[index - 1];
            let column = provider.and_then(|p| prunr_core::engine::column_for(model, p));
            on(Told::Step(Step { index, total, model, side, sides, column }));
        }
        Event::Done { row, provider, .. } => {
            let Some(id) = models.iter().copied().find(|id| id.stable_name() == row.name) else { return };
            let Some(column) = provider.and_then(|p| prunr_core::engine::column_for(id, p)) else { return };
            if let Some(place) = super::table::found_here(&row, id, column) {
                tracing::info!(model = ?id, ?column, kind = ?place.kind, text = %place, "test placed the model");
                prunr_core::ep_compat::set_measured(id, column, place.clone());
                on(Told::Placed { model: id, column, measured: place });
            }
        }
    });
    if only.is_empty() {
        prunr_core::ep_compat::set_tested();
    }
    on(Told::Finished { left_out });
}

/// After a download onto an accelerator no one measured `id` on: one run
/// there on the bench picture, judged alone (`table::found_alone`), with
/// no CPU run. A wrong result keeps the model off the accelerator here; a
/// right one leaves it unmeasured, so it runs there. The run also builds
/// what the accelerator compiles, so the first real run starts sooner.
pub fn check(id: ModelId, on: &mut dyn FnMut(Told)) {
    if let Some((column, place)) = checked(id) {
        tracing::info!(model = ?id, ?column, kind = ?place.kind, text = %place, "the check keeps the model off the accelerator");
        prunr_core::ep_compat::set_measured(id, column, place);
    }
    on(Told::Finished { left_out: Vec::new() });
}

fn checked(id: ModelId) -> Option<(Column, prunr_core::ep_compat::Measured)> {
    let column = column_of(id)?;
    let side = super::gpu_alone(id, &Inputs::built_in())?;
    let place = super::table::found_alone(&side, super::Family::of(id)?);
    if place.is_none() {
        tracing::info!(model = ?id, ?column, "the check found the model's result right");
    }
    Some((column, place?))
}

/// What a test runs.
#[derive(Debug, PartialEq)]
struct Plan {
    models: Vec<ModelId>,
    /// The heavy models it does not run.
    left_out: Vec<ModelId>,
}

/// What a test does with one model.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Turn {
    /// Times it.
    Runs,
    /// Leaves it out: a heavy model takes many minutes on the CPU.
    LeftOut,
    /// Nothing to measure: no accelerator would try it (`column`, from
    /// `column_of`, is `None`), so its CPU run could take minutes for
    /// nothing; or it runs only inside another model's run.
    Passed,
}

fn turn(id: ModelId, column: Option<Column>) -> Turn {
    match column {
        Some(_) if !super::in_catalog(id) => Turn::Passed,
        Some(_) if super::HEAVY.contains(&id) => Turn::LeftOut,
        Some(_) => Turn::Runs,
        None => Turn::Passed,
    }
}

/// A test would time `id` here: what the Model Store's Test offers.
pub(crate) fn times(id: ModelId) -> bool {
    turn(id, first_column_in_a_test(id)) == Turn::Runs
}

fn plan(installed: Vec<ModelId>, column_of: impl Fn(ModelId) -> Option<Column>) -> Plan {
    let mut plan = Plan { models: Vec::new(), left_out: Vec::new() };
    for id in installed {
        match turn(id, column_of(id)) {
            Turn::Runs => plan.models.push(id),
            Turn::LeftOut => plan.left_out.push(id),
            Turn::Passed => {}
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The test runs each model an accelerator would try, but the heavy
    /// ones, which it names as left out, and TAESD, which runs only inside
    /// a Stable Diffusion erase.
    #[test]
    fn a_test_runs_what_an_accelerator_would_try_but_the_heavy_models() {
        let installed = vec![ModelId::Silueta, ModelId::BiRefNetLite, ModelId::Nomos8kSchatL, ModelId::SdV15InpaintFp16, ModelId::TaesdFp16, ModelId::U2net];
        let column_of = |id| (id != ModelId::BiRefNetLite && id != ModelId::SdV15InpaintFp16).then_some(Column::Cuda);
        assert_eq!(plan(installed, column_of), Plan {
            models: vec![ModelId::Silueta, ModelId::U2net],
            left_out: vec![ModelId::Nomos8kSchatL],
        });
    }

    /// A test of chosen models leaves the whole computer's test date alone:
    /// Settings still says the computer was not tested.
    #[test]
    fn a_test_of_chosen_models_leaves_the_computer_untested() {
        let before = prunr_core::ep_compat::tested();
        let mut finished = false;
        run(&["silueta".to_string()], &mut |told| finished |= matches!(told, Told::Finished { .. }));
        assert!(finished);
        assert_eq!(prunr_core::ep_compat::tested(), before);
    }

    /// The bar moves an equal share of a model's turn per side it runs on.
    #[test]
    fn the_bar_moves_a_share_of_the_model_per_side() {
        let step = |index, side, sides| Step { index, total: 4, model: ModelId::U2net, side, sides, column: (side > 0).then_some(Column::Cuda) };
        assert_eq!(step(1, 0, 3).fraction(), 0.0);
        assert!((step(1, 1, 2).fraction() - 0.125).abs() < 1e-6);
        assert!((step(4, 2, 3).fraction() - (3.0 + 2.0 / 3.0) / 4.0).abs() < 1e-6);
    }

    /// Each event survives the line it travels on.
    #[test]
    fn every_event_reads_back_from_its_line() {
        for told in [
            Told::Step(Step { index: 2, total: 9, model: ModelId::U2net, side: 1, sides: 2, column: Some(Column::OpenVinoGpu) }),
            Told::Step(Step { index: 1, total: 9, model: ModelId::Silueta, side: 0, sides: 3, column: None }),
            Told::Placed { model: ModelId::Silueta, column: Column::DirectMl, measured: prunr_core::ep_compat::Measured::timed(prunr_core::ep_compat::PlaceKind::There, 40.0, 670.0, "") },
            Told::Placed {
                model: ModelId::BiRefNetLite,
                column: Column::DirectMl,
                measured: prunr_core::ep_compat::Measured::new(prunr_core::ep_compat::PlaceKind::Fails, "E_OUTOFMEMORY").with_times(None, Some(10_100.0)),
            },
            Told::Finished { left_out: vec![ModelId::Nomos8kSchatL] },
        ] {
            assert_eq!(Told::parse(&told.line()), Some(told.clone()), "{}", told.line());
        }
        assert_eq!(Told::parse("[3/9] U2Net: CPU session"), None, "anything else on the line is not an event");
    }
}
