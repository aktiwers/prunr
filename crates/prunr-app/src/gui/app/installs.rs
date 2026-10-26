//! Removing what is installed: models, runtimes and their caches. Each
//! runs on a worker; the toast and the installed set follow when it lands.

use prunr_models::{ModelId, ModelSource};

use super::PrunrApp;

impl PrunrApp {
    /// Remove a downloaded model's weights. Gigabytes take a while to
    /// unlink.
    pub(crate) fn remove_model(&mut self, id: ModelId, ctx: &egui::Context) {
        let Some(desc) = prunr_models::descriptor(id) else { return };
        let Some(dir) = prunr_models::on_demand_dir() else {
            self.toasts.error("Could not find Prunr's data folder.");
            return;
        };
        let path = match desc.source {
            ModelSource::OnDemand { filename, .. } => dir.join(filename),
            ModelSource::MultiPartOnDemand { subdir, .. } => dir.join(subdir),
            ModelSource::Bundled { .. } | ModelSource::MultiPartBundled { .. } => return,
        };
        let name = desc.display_name;
        self.off_thread(ctx, move || {
            let removed = if path.is_dir() { std::fs::remove_dir_all(&path) } else { std::fs::remove_file(&path) };
            removed.map_err(|e| format!("Could not remove {name}: {e}"))?;
            tracing::info!(?id, "deleted on-demand model");
            // The compiled-model cache (CUDA optimized.onnx, CoreML
            // mlmodelc) is useless without the source weights.
            let bytes = prunr_core::cache::clear_for_model(id);
            if bytes > 0 {
                tracing::info!(?id, bytes, "cleared compiled-model cache after uninstall");
            }
            Ok(format!("{name} removed"))
        });
    }

    pub(crate) fn uninstall_runtime(&mut self, rt: crate::runtime_install::RuntimeId, ctx: &egui::Context) {
        self.off_thread(ctx, move || {
            crate::runtime_install::uninstall(rt)
                .map(|()| format!("{} uninstalled", rt.display_name()))
                .map_err(|e| format!("Could not uninstall {}: {e}", rt.display_name()))
        });
    }

    /// Clear the compiled-model cache (a multi-GB tree on a slow disk).
    pub(crate) fn clear_compiled_cache(&mut self) {
        std::thread::spawn(|| {
            let bytes = prunr_core::cache::clear_all();
            tracing::info!(bytes, "cleared compiled-model cache");
        });
        self.toasts.info("Clearing compiled-model cache…");
    }

    pub(crate) fn forget_gpu_failures(&mut self, ctx: &egui::Context) {
        self.change_places(ctx, || {
            let entries = prunr_core::ep_compat::clear();
            tracing::info!(entries, "forgot recorded GPU failures");
            Ok("Models that failed on the graphics card will try it again".to_string())
        });
    }

    /// Change what the app learned about this machine (`ep_compat`) on the
    /// disk queue, in order with every other such change.
    pub(super) fn change_places(&self, ctx: &egui::Context, change: impl FnOnce() -> Result<String, String> + Send + 'static) {
        crate::gui::disk_queue::call(self.landing(ctx, change));
    }

    /// Run `remove` on a worker of its own.
    fn off_thread(&self, ctx: &egui::Context, remove: impl FnOnce() -> Result<String, String> + Send + 'static) {
        std::thread::spawn(self.landing(ctx, remove));
    }

    /// `change`, its toast text landing through `bg_io.removed_rx`, which
    /// reads the installed set again.
    fn landing(&self, ctx: &egui::Context, change: impl FnOnce() -> Result<String, String> + Send + 'static) -> impl FnOnce() + Send + 'static {
        let (done, ctx) = (self.batch.bg_io.removed_tx.clone(), ctx.clone());
        move || {
            let _ = done.send(change());
            ctx.request_repaint();
        }
    }
}
