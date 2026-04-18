pub mod app;
pub mod item;
pub mod history_manager;
pub mod state;
pub mod settings;
pub mod item_settings;
pub mod presets_fs;
pub mod worker;
pub mod theme;
pub mod views;
pub mod zoom_state;
pub mod status_state;
pub mod background_io;
pub mod drag_export;
pub mod history_disk;
pub mod memory;
pub mod live_preview;

#[cfg(test)]
mod tests;

pub fn run() -> eframe::Result {
    let icon = {
        let png_bytes = include_bytes!("../../../../assets/prunr-256.png");
        let img = image::load_from_memory(png_bytes).expect("Failed to decode app icon");
        let rgba = img.to_rgba8();
        let (w, h) = (rgba.width(), rgba.height());
        egui::IconData {
            rgba: rgba.into_raw(),
            width: w,
            height: h,
        }
    };

    let make_options = || eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Prunr")
            .with_inner_size(theme::DEFAULT_WINDOW_SIZE)
            .with_min_inner_size(theme::MIN_WINDOW_SIZE)
            .with_drag_and_drop(true)
            .with_icon(icon.clone())
            .with_app_id("prunr"),
        ..Default::default()
    };

    let app_factory = || Box::new(|cc: &eframe::CreationContext<'_>| Ok(Box::new(app::PrunrApp::new(cc)) as Box<dyn eframe::App>));

    // Try default renderer (glow/OpenGL). On Windows-on-Mac VMs and older Intel
    // GPUs OpenGL drivers are unreliable; fall back to wgpu (DX12/Metal/Vulkan)
    // which tends to work better in virtualized environments.
    let primary = eframe::run_native("prunr", make_options(), app_factory());
    if let Err(ref e) = primary {
        log_startup_error("glow/OpenGL renderer failed", e);
        let mut opts = make_options();
        opts.renderer = eframe::Renderer::Wgpu;
        let fallback = eframe::run_native("prunr", opts, app_factory());
        if let Err(ref e2) = fallback {
            log_startup_error("wgpu renderer also failed", e2);
        }
        return fallback;
    }
    primary
}

/// Write startup failures to a log file next to the executable so users on
/// machines without a console (Windows GUI subsystem) can still diagnose.
fn log_startup_error(stage: &str, err: &eframe::Error) {
    tracing::error!(stage, %err, "startup failed");
    let Ok(exe) = std::env::current_exe() else { return };
    let Some(dir) = exe.parent() else { return };
    let path = dir.join("prunr-startup-error.log");
    let _ = std::fs::write(
        &path,
        format!(
            "Prunr startup error\nStage: {stage}\nError: {err}\nPlatform: {}\n\n\
             Please report this at https://github.com/aktiwers/prunr/issues\n",
            std::env::consts::OS,
        ),
    );
}
