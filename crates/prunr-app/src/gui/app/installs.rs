//! Removing what is installed: models, runtimes and their caches. Each
//! runs on a worker; the toast and the installed set follow when it lands.

use std::path::{Path, PathBuf};

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
        let Some((path, companion)) = downloaded_files(desc.source, &dir) else { return };
        let name = desc.display_name;
        self.off_thread(ctx, move || {
            remove_weights(&path, companion.as_deref()).map_err(|e| format!("Could not remove {name}: {e}"))?;
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
            use crate::runtime_install::Uninstalled;
            crate::runtime_install::uninstall(rt)
                .map(|when| match when {
                    Uninstalled::Now => format!("{} uninstalled", rt.display_name()),
                    Uninstalled::AtNextStart => format!("{} is removed when Prunr next starts", rt.display_name()),
                })
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

/// What a download of `source` put in the models folder `dir`: its weights
/// (a file, or a bundle's folder) and a single file's half-precision
/// companion. `None` for a built-in model.
fn downloaded_files(source: ModelSource, dir: &Path) -> Option<(PathBuf, Option<PathBuf>)> {
    match source {
        ModelSource::OnDemand { filename, fp16, .. } => Some((dir.join(filename), fp16.map(|v| dir.join(v.filename)))),
        ModelSource::MultiPartOnDemand { subdir, .. } => Some((dir.join(subdir), None)),
        ModelSource::Bundled { .. } | ModelSource::MultiPartBundled { .. } => None,
    }
}

/// Remove the weights at `path` and their `companion`, which may never
/// have come.
fn remove_weights(path: &Path, companion: Option<&Path>) -> std::io::Result<()> {
    // The companion first: one that cannot go leaves the model installed,
    // so Remove can be tried again.
    if let Some(companion) = companion {
        match std::fs::remove_file(companion) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            removed => removed?,
        }
    }
    if path.is_dir() { std::fs::remove_dir_all(path) } else { std::fs::remove_file(path) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Remove takes a model's half-precision file with its weights: the
    /// 33 MB file stayed on disk, counted in the Model Store, after each
    /// upscaler that brings one was removed. A companion that never came
    /// does not stop the removal.
    #[test]
    fn removing_a_model_takes_its_half_precision_file_too() {
        let dir = tempfile::tempdir().unwrap();
        let source = prunr_models::descriptor(ModelId::RealEsrganX4Plus).expect("in the registry").source;
        let (path, companion) = downloaded_files(source, dir.path()).expect("a download");
        let companion = companion.expect("the upscaler brings a half-precision file");
        std::fs::write(&path, b"fp32").unwrap();
        std::fs::write(&companion, b"fp16").unwrap();
        remove_weights(&path, Some(&companion)).unwrap();
        assert!(!path.exists() && !companion.exists(), "both go");
        std::fs::write(&path, b"fp32").unwrap();
        remove_weights(&path, Some(&companion)).unwrap();
        assert!(!path.exists(), "one whose half-precision file never came is removed");
        let lama = prunr_models::descriptor(ModelId::LaMaFp32).expect("in the registry").source;
        assert_eq!(downloaded_files(lama, dir.path()).map(|(_, companion)| companion), Some(None), "a model without one");
    }
}
