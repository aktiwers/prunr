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
    /// Model `index` of `total` is at a step; `on_gpu` while its GPU side
    /// runs, on `column`.
    Step { index: usize, total: usize, model: ModelId, on_gpu: bool, column: Column },
    /// All done. `left_out`: installed models a test does not run, which
    /// take many minutes on the processor.
    Finished { left_out: Vec<ModelId> },
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
fn column_of(id: ModelId) -> Option<Column> {
    let ladder = prunr_core::engine::gpu_ladder(id);
    prunr_core::engine::column_for(id, super::first_tried(&ladder)?)
}

/// Run the test: every installed model the accelerator would try (a place
/// the registry keeps on the CPU for a slower, different or no-gain result
/// is tried again: `remeasure_places`), but the heavy ones.
pub fn run(on: &mut dyn FnMut(Told)) {
    prunr_core::engine::remeasure_places();
    let installed = super::select(&[], true, prunr_models::is_available).unwrap_or_default();
    let Plan { models, columns, left_out } = plan(installed, column_of);
    let inputs = Inputs::built_in();
    super::run(&models, &inputs, RUNS, &mut |event| match event {
        Event::Stage { index, total, stage, .. } => {
            let (model, column) = (models[index - 1], columns[index - 1]);
            on(Told::Step { index, total, model, on_gpu: stage.starts_with("GPU"), column });
        }
        Event::Done(row, _) => {
            let Some(i) = models.iter().position(|id| id.stable_name() == row.name) else { return };
            let (id, column) = (models[i], columns[i]);
            if let Some(place) = super::table::found_here(&row, id, column) {
                tracing::info!(model = ?id, ?column, kind = ?place.kind, text = %place.text, "test placed the model");
                prunr_core::ep_compat::set_measured(id, column, place);
            }
        }
    });
    prunr_core::ep_compat::set_tested(now_secs());
    on(Told::Finished { left_out });
}

/// After a download onto an accelerator no one measured `id` on: one run
/// there on the bench picture, judged alone (`table::found_alone`), with
/// no CPU run. A wrong result keeps the model off the accelerator here; a
/// right one leaves it unmeasured, so it runs there. The run also builds
/// what the accelerator compiles, so the first real run starts sooner.
pub fn check(id: ModelId, on: &mut dyn FnMut(Told)) {
    if let Some((column, place)) = checked(id) {
        tracing::info!(model = ?id, ?column, kind = ?place.kind, text = %place.text, "the check keeps the model off the accelerator");
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

/// What a test runs: each model with the column it is measured in.
#[derive(Debug, PartialEq)]
struct Plan {
    models: Vec<ModelId>,
    columns: Vec<Column>,
    /// The heavy models it does not run.
    left_out: Vec<ModelId>,
}

/// A model no accelerator would try (`column_of` is `None`) has nothing
/// to measure: its CPU run alone could take minutes for nothing. Heavy
/// models take many minutes on the CPU, so the test leaves them out.
fn plan(installed: Vec<ModelId>, column_of: impl Fn(ModelId) -> Option<Column>) -> Plan {
    let mut plan = Plan { models: Vec::new(), columns: Vec::new(), left_out: Vec::new() };
    for id in installed {
        let Some(column) = column_of(id) else { continue };
        if super::HEAVY.contains(&id) {
            plan.left_out.push(id);
        } else {
            plan.models.push(id);
            plan.columns.push(column);
        }
    }
    plan
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The test runs each model an accelerator would try, but the heavy
    /// ones, which it names as left out.
    #[test]
    fn a_test_runs_what_an_accelerator_would_try_but_the_heavy_models() {
        let installed = vec![ModelId::Silueta, ModelId::BiRefNetLite, ModelId::Nomos8kSchatL, ModelId::SdV15InpaintFp16, ModelId::U2net];
        let column_of = |id| (id != ModelId::BiRefNetLite && id != ModelId::SdV15InpaintFp16).then_some(Column::Cuda);
        assert_eq!(plan(installed, column_of), Plan {
            models: vec![ModelId::Silueta, ModelId::U2net],
            columns: vec![Column::Cuda, Column::Cuda],
            left_out: vec![ModelId::Nomos8kSchatL],
        });
    }

    /// Each event survives the line it travels on.
    #[test]
    fn every_event_reads_back_from_its_line() {
        for told in [
            Told::Step { index: 2, total: 9, model: ModelId::U2net, on_gpu: true, column: Column::OpenVinoGpu },
            Told::Step { index: 1, total: 9, model: ModelId::Silueta, on_gpu: false, column: Column::Cuda },
            Told::Finished { left_out: vec![ModelId::Nomos8kSchatL] },
        ] {
            assert_eq!(Told::parse(&told.line()), Some(told.clone()), "{}", told.line());
        }
        assert_eq!(Told::parse("[3/9] U2Net: CPU session"), None, "anything else on the line is not an event");
    }
}
