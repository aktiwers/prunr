//! Settings › Hardware › Test this computer: starting it, following it,
//! and what the app does with its places.

use super::PrunrApp;
use crate::gui::machine_test::{Ended, MachineTest};

impl PrunrApp {
    /// The accelerator a test measures; `None` without one, or with
    /// Settings › Processor on CPU only, where there is nothing to test.
    pub(crate) fn machine_test_accelerator(&self) -> Option<String> {
        (self.settings.is_gpu() && !self.settings.cpu_only()).then(|| self.settings.active_backend.clone())
    }

    /// Start a test. The sessions the app holds go first: the test needs
    /// their memory, and the ones built after it follow its places.
    pub(crate) fn start_machine_test(&mut self, ctx: &egui::Context) {
        if self.machine_test.is_some() || self.machine_test_accelerator().is_none() {
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
    pub(crate) fn forget_machine_test(&mut self) {
        let removed = prunr_core::ep_compat::clear_measured();
        tracing::info!(removed, "forgot the computer test's places");
        self.free_sessions_for_new_places();
        self.refresh_install_cache();
        self.toasts.info("Each model runs where Prunr's own tests put it");
    }

    /// Each frame: take in the test's progress, and its end.
    pub(crate) fn pump_machine_test(&mut self, ctx: &egui::Context) {
        let Some(ended) = self.machine_test.as_mut().and_then(MachineTest::pump) else { return };
        self.machine_test = None;
        match ended {
            Ended::Finished { left_out } => {
                self.refresh_install_cache();
                let test = self.hardware_install_cache.test;
                let accelerator = self.settings.active_backend.as_str();
                let mut text = format!(
                    "Test done: {} with {accelerator}, {} without it",
                    crate::gui::views::fmt::count(test.there, "model", "models"),
                    crate::gui::views::fmt::count(test.off, "model", "models"),
                );
                if !left_out.is_empty() {
                    let names: Vec<&str> = left_out.iter().map(|&id| prunr_models::display_name(id)).collect();
                    text += &format!(". Not tested, as each takes many minutes: {}", names.join(", "));
                }
                self.toasts.success(text);
            }
            Ended::Stopped { during } => {
                match during.and_then(crate::gui::machine_test::stopped_place) {
                    Some((model, column, place)) => {
                        tracing::warn!(?model, ?column, "the computer test stopped while a model ran on the accelerator");
                        let text = format!(
                            "The test stopped while {} ran with {}, so it runs without it. Test again to finish the rest.",
                            prunr_models::display_name(model),
                            self.settings.active_backend,
                        );
                        self.off_thread(ctx, move || {
                            prunr_core::ep_compat::set_measured(model, column, place);
                            Err(text)
                        });
                    }
                    None => {
                        tracing::warn!(model = ?during.map(|s| s.model), "the computer test stopped");
                        let name = during.map(|step| prunr_models::display_name(step.model));
                        self.toasts.error(format!(
                            "The test stopped{}. The log has the details.",
                            name.map(|n| format!(" at {n}")).unwrap_or_default(),
                        ));
                    }
                }
                self.refresh_install_cache();
            }
        }
        self.free_sessions_for_new_places();
    }

    /// Cancel the test; what it measured so far is kept.
    pub(crate) fn cancel_machine_test(&mut self) {
        if self.machine_test.take().is_some() {
            tracing::info!("the computer test was cancelled");
            self.refresh_install_cache();
            self.free_sessions_for_new_places();
        }
    }

    /// Whether a download of `id` ends with a check of its result: it runs
    /// on an accelerator here no one measured it on. Not a heavy model: its
    /// run alone takes minutes and gigabytes.
    pub(crate) fn download_needs_check(&self, id: prunr_models::ModelId) -> bool {
        let Some(accelerator) = self.machine_test_accelerator() else { return false };
        let unmeasured = |column| prunr_core::engine::places_here(id).get(column) == prunr_models::Place::Unmeasured;
        !crate::bench::HEAVY.contains(&id)
            && crate::bench::Family::of(id).is_some()
            && prunr_core::engine::column_for(id, &accelerator).is_some_and(unmeasured)
    }

    /// Each frame: a download's check, run while nothing else is; a model
    /// it keeps off the accelerator is said once.
    pub(crate) fn pump_download_checks(&mut self, ctx: &egui::Context) {
        let idle = self.machine_test.is_none() && !crate::gui::automation::state::busy(self);
        let Some((id, ended)) = self.download_manager.pump_checks(idle) else { return };
        let name = prunr_models::display_name(id);
        let accelerator = self.settings.active_backend.clone();
        let column = prunr_core::engine::column_for(id, &accelerator);
        match ended {
            Ended::Finished { .. } => {
                prunr_core::ep_compat::refresh();
                if column.and_then(|c| prunr_core::ep_compat::measured(id, c)).is_some() {
                    self.toasts.warning(format!("{name} gave a wrong result with {accelerator} on this computer, so it runs without it."));
                    self.free_sessions_for_new_places();
                }
            }
            Ended::Stopped { .. } => {
                let Some(column) = column else { return };
                tracing::warn!(?id, ?column, "a download's check stopped while the model ran on the accelerator");
                let place = prunr_core::ep_compat::Measured {
                    kind: prunr_core::ep_compat::PlaceKind::Fails,
                    text: "the check's process stopped while it ran here".into(),
                };
                let text = format!("{name} stopped Prunr's check with {accelerator} on this computer, so it runs without it.");
                self.off_thread(ctx, move || {
                    prunr_core::ep_compat::set_measured(id, column, place);
                    Err(text)
                });
                self.free_sessions_for_new_places();
            }
        }
    }

    /// Drop every session, the Magic Brush's too: the next one of each
    /// model is built where the places now say.
    fn free_sessions_for_new_places(&mut self) {
        self.processor.release_all_sessions();
        self.processor.release_sam_sessions();
    }
}

#[cfg(test)]
mod tests {
    use prunr_models::ModelId;

    /// A download is checked only where it runs on an accelerator no one
    /// measured it on: not on a measured one, not on CPU only, and never a
    /// heavy model or one that runs inside another's run.
    #[test]
    fn a_download_is_checked_only_on_an_unmeasured_accelerator() {
        let mut app = super::PrunrApp::new_for_test();
        app.settings.active_backend = "MIGraphX".into();
        assert!(app.download_needs_check(ModelId::U2net), "unmeasured there");
        assert!(!app.download_needs_check(ModelId::SdV15InpaintFp16), "a heavy model");
        assert!(!app.download_needs_check(ModelId::TaesdFp16), "runs inside a Stable Diffusion erase");
        app.settings.active_backend = "CUDA".into();
        assert!(!app.download_needs_check(ModelId::U2net), "measured on CUDA");
        app.settings.active_backend = "MIGraphX".into();
        app.settings.processor = crate::gui::settings::ComputeMode::CpuOnly;
        assert!(!app.download_needs_check(ModelId::U2net), "CPU only");
    }
}
