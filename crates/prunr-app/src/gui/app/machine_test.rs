//! Settings › Hardware › Test this computer and the check after a
//! download: starting them, following them, and what the app does with
//! the places they measure.

use prunr_models::{Column, ModelId};

use super::PrunrApp;
use crate::gui::machine_test::{Ended, MachineTest};

impl PrunrApp {
    /// The accelerator a test measures; `None` without one, or with
    /// Settings › Processor on CPU only, where there is nothing to test.
    pub(crate) fn machine_test_accelerator(&self) -> Option<&str> {
        (self.settings.is_gpu() && !self.settings.cpu_only()).then_some(self.settings.active_backend.as_str())
    }

    /// Nothing else uses the machine: no work of the user's, and no test or
    /// check under way. A test, or a check, starts only then.
    fn machine_idle(&self) -> bool {
        self.machine_test.is_none() && !self.download_manager.checking() && !crate::gui::automation::state::busy(self)
    }

    pub(crate) fn can_machine_test_intent(&self) -> bool {
        self.machine_test_accelerator().is_some() && self.machine_idle()
    }

    /// The Model Store's Test can time `model` here: an accelerator to
    /// test, its weights installed, and a model the computer test runs
    /// (not a heavy one, nor one that runs only inside another's run).
    pub(crate) fn model_testable(&self, model: ModelId) -> bool {
        self.machine_test_accelerator().is_some()
            && self.hardware_install_cache.has_installed(model)
            && crate::bench::Family::of(model).is_some()
            && !crate::bench::HEAVY.contains(&model)
    }

    pub(crate) fn can_model_test_intent(&self, model: ModelId) -> bool {
        self.model_testable(model) && self.machine_idle()
    }

    /// Test `model` alone, as the computer test would, on every engine.
    pub(crate) fn handle_model_test_intent(&mut self, ctx: &egui::Context, model: ModelId) {
        if !self.can_model_test_intent(model) {
            return;
        }
        self.free_sessions_for_new_places();
        let ctx = ctx.clone();
        match MachineTest::start_for(model, move || ctx.request_repaint()) {
            Ok(test) => self.machine_test = Some(test),
            Err(e) => {
                tracing::warn!(?model, %e, "the model's test did not start");
                self.toasts.error(e);
            }
        }
    }

    /// Start a test. The sessions the app holds go first: the test needs
    /// their memory, and the ones built after it follow its places.
    pub(crate) fn handle_machine_test_intent(&mut self, ctx: &egui::Context) {
        if !self.can_machine_test_intent() {
            return;
        }
        self.free_sessions_for_new_places();
        let ctx = ctx.clone();
        match MachineTest::start(move || ctx.request_repaint()) {
            Ok(test) => self.machine_test = Some(test),
            Err(e) => {
                tracing::warn!(%e, "the computer test did not start");
                self.toasts.error(e);
            }
        }
    }

    /// Forget what tests measured: the places Prunr ships apply again.
    pub(crate) fn forget_machine_test(&mut self, ctx: &egui::Context) {
        self.change_places(ctx, || {
            let removed = prunr_core::ep_compat::clear_measured();
            tracing::info!(removed, "forgot the computer test's places");
            Ok("Each model runs where Prunr's own tests put it".to_string())
        });
        self.free_sessions_for_new_places();
    }

    /// Each frame: take in the test's progress, and its end.
    pub(crate) fn pump_machine_test(&mut self, ctx: &egui::Context) {
        let Some(ended) = self.machine_test.as_mut().and_then(MachineTest::pump) else { return };
        let only = self.machine_test.take().and_then(|test| test.only());
        self.places_changed();
        if let (Ended::Finished { .. }, Some(model)) = (&ended, only) {
            let on = prunr_core::memory::provider_now(model, self.settings.cpu_only());
            let on = if on == prunr_core::engine::CPU_NAME { "the processor" } else { on };
            self.toasts.success(format!("{} tested: it runs on {on} here.", prunr_models::display_name(model)));
            return;
        }
        match ended {
            Ended::Finished { left_out } => {
                let mut text = "Test done: each model now runs on the fastest that gives the same picture. Settings › Hardware has the times.".to_string();
                if !left_out.is_empty() {
                    let names: Vec<&str> = left_out.iter().map(|&id| prunr_models::display_name(id)).collect();
                    text += &format!(" Not tested, as each takes many minutes: {}.", names.join(", "));
                }
                self.toasts.success(text);
            }
            Ended::Stopped { during: Some(step) } if step.on_gpu() => {
                tracing::warn!(model = ?step.model, column = ?step.column, "the computer test stopped while a model ran on the accelerator");
                let text = format!(
                    "The test stopped while {} ran with {}, so it runs without it. Test again to finish the rest.",
                    prunr_models::display_name(step.model),
                    step.column.name(),
                );
                self.keep_off_after_stop(ctx, step.model, step.column, text);
            }
            Ended::Stopped { during } => {
                tracing::warn!(model = ?during.map(|s| s.model), "the computer test stopped");
                let at = during.map(|step| format!(" at {}", prunr_models::display_name(step.model))).unwrap_or_default();
                self.toasts.error(format!("The test stopped{at}. The log has the details."));
            }
        }
    }

    /// Cancel the test; what it measured so far is kept.
    pub(crate) fn cancel_machine_test(&mut self) {
        if self.machine_test.take().is_some() {
            tracing::info!("the computer test was cancelled");
            self.places_changed();
        }
    }

    /// The column a download of `id` is checked in: the accelerator it
    /// runs on first here, when no one measured it there (`needs_check`).
    pub(crate) fn download_check_column(&self, id: ModelId) -> Option<Column> {
        self.machine_test_accelerator()?;
        crate::bench::here::column_of(id).filter(|&column| needs_check(id, column))
    }

    /// Each frame: a download's check, started while nothing else uses the
    /// machine; a model it keeps off the accelerator is said once.
    pub(crate) fn pump_download_checks(&mut self, ctx: &egui::Context) {
        if !self.download_manager.has_checks() {
            return;
        }
        let idle = self.machine_idle();
        let Some((id, column, ended)) = self.download_manager.pump_checks(idle) else { return };
        let name = prunr_models::display_name(id);
        let accelerator = column.name();
        match ended {
            Ended::Finished { .. } => {
                prunr_core::ep_compat::refresh();
                if let Some(place) = prunr_core::ep_compat::measured_for(id)[column.index()] {
                    self.toasts.warning(check_found(name, accelerator, place));
                    self.places_changed();
                }
            }
            Ended::Stopped { .. } => {
                tracing::warn!(?id, ?column, "a download's check stopped while the model ran on the accelerator");
                self.keep_off_after_stop(ctx, id, column, format!("{name} stopped Prunr's check with {accelerator} on this computer, so it runs without it."));
            }
        }
    }

    /// Keep `model` off the accelerator in `column` after a test or check
    /// stopped while it ran there, and say so once that is saved.
    fn keep_off_after_stop(&mut self, ctx: &egui::Context, model: ModelId, column: Column, toast: String) {
        self.change_places(ctx, move || {
            prunr_core::ep_compat::set_measured(model, column, crate::gui::machine_test::stopped_here());
            Err(toast)
        });
        self.free_sessions_for_new_places();
    }

    /// Where models run may have changed: Settings shows it, and sessions
    /// are built afresh.
    pub(super) fn places_changed(&mut self) {
        self.refresh_install_cache();
        self.free_sessions_for_new_places();
    }

    /// Drop every session, the Magic Brush's too: the next one of each
    /// model is built where the places now say.
    fn free_sessions_for_new_places(&mut self) {
        self.processor.release_all_sessions();
        self.processor.release_sam_sessions();
    }
}

/// Whether a download of `id` is checked in `column`: no one measured it
/// there. Not a heavy model (its run alone takes minutes and gigabytes),
/// nor one that runs only inside another's run.
fn needs_check(id: ModelId, column: Column) -> bool {
    !crate::bench::HEAVY.contains(&id)
        && crate::bench::Family::of(id).is_some()
        && prunr_core::engine::places_here(id).get(column) == prunr_models::Place::Unmeasured
}

/// What a download's check found that keeps the model off `accelerator`.
fn check_found(name: &str, accelerator: &str, place: prunr_models::Place) -> String {
    match place {
        prunr_models::Place::Cpu(prunr_models::Why::Fails(_)) => format!("{name} does not run with {accelerator} on this computer, so it runs without it."),
        _ => format!("{name} gave a wrong result with {accelerator} on this computer, so it runs without it."),
    }
}

#[cfg(test)]
mod tests {
    use prunr_models::ModelId;

    /// A model the accelerator refuses or breaks on is not said to give a
    /// wrong result: four on DirectML were, and none had run.
    #[test]
    fn a_check_says_what_it_found() {
        use prunr_models::{Place, Why};
        assert_eq!(super::check_found("LaMa", "DirectML", Place::Cpu(Why::Fails("x"))), "LaMa does not run with DirectML on this computer, so it runs without it.");
        assert_eq!(super::check_found("Nomos8k", "CUDA", Place::Cpu(Why::Differs("x"))), "Nomos8k gave a wrong result with CUDA on this computer, so it runs without it.");
    }

    /// A download is checked only where it runs on an accelerator no one
    /// measured it on: not on a measured one, not on CPU only, and never a
    /// heavy model or one that runs inside another's run.
    #[test]
    fn a_download_is_checked_only_on_an_unmeasured_accelerator() {
        use prunr_models::Column;
        assert!(super::needs_check(ModelId::U2net, Column::MiGraphX), "unmeasured there");
        assert!(!super::needs_check(ModelId::SdV15InpaintFp16, Column::MiGraphX), "a heavy model");
        assert!(!super::needs_check(ModelId::TaesdFp16, Column::MiGraphX), "runs inside a Stable Diffusion erase");
        assert!(!super::needs_check(ModelId::U2net, Column::Cuda), "measured on CUDA");
        let mut app = super::PrunrApp::new_for_test();
        app.settings.active_backend = "MIGraphX".into();
        app.settings.processor = crate::gui::settings::ComputeMode::CpuOnly;
        assert_eq!(app.download_check_column(ModelId::U2net), None, "CPU only");
    }

    /// A test starts only with an accelerator and nothing else under way.
    #[test]
    fn a_test_waits_for_an_accelerator_and_an_idle_machine() {
        use crate::gui::tests::fixtures::{push_test_item, put_in};
        let mut app = super::PrunrApp::new_for_test();
        assert!(!app.can_machine_test_intent(), "the CPU alone: nothing to test");
        app.settings.active_backend = "CUDA".into();
        assert!(app.can_machine_test_intent());
        put_in(push_test_item(&mut app, 1), crate::gui::item::BatchStatus::Processing);
        assert!(!app.can_machine_test_intent(), "a run is under way");
    }
}
