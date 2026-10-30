pub mod app;
pub mod toasts;
pub mod item;
pub mod drag_export_state;
pub mod batch_manager;
pub mod processor;
pub mod state;
pub mod settings;
pub mod item_settings;
pub mod presets;
pub mod presets_fs;
pub mod worker;
pub mod inpaint_bridge;
pub mod theme;
pub mod views;
pub mod zoom_state;
pub mod brush_state;
pub(crate) mod magic_brush_state;
pub(crate) mod timeline;
pub mod status_state;
pub mod background_io;
pub mod drag_export;
pub mod history_disk;
pub mod memory;
pub mod live_preview;
pub mod knob_catalog;
pub mod surface_catalog;
pub mod system_bridge;
pub mod download_manager;
pub mod hardware_cache;
pub mod machine_test;
pub mod env_overrides;
pub mod dispatch_progress;
pub(crate) mod first_look;
pub mod automation;
pub(crate) mod capability;
pub(crate) mod disk_queue;
pub(crate) mod reclaim;
pub(crate) mod image_texture;
pub(crate) mod selection_queue;
#[cfg(target_os = "macos")]
pub(crate) mod macos_quit;

#[cfg(test)]
pub(crate) mod tests;

pub fn run() -> eframe::Result {
    crate::panics::use_window_pool();
    // The GPU provider probe (DirectML starts a device) and the hardware
    // profile (DXGI lists the adapters on Windows) run while the window
    // opens; `PrunrApp::new` and the first frame read what they found.
    let probes: [(&str, fn()); 2] = [
        ("provider-probe", || { let _ = prunr_core::OrtEngine::detect_active_provider(); }),
        ("hardware-probe", || { let _ = crate::hardware::profile(); }),
    ];
    for (name, probe) in probes {
        if let Err(e) = std::thread::Builder::new().name(name.into()).spawn(probe) {
            tracing::warn!(%e, name, "could not start the probe; it runs on first use");
        }
    }
    let icon = {
        let png_bytes = include_bytes!("../../../../assets/prunr-256.png");
        let img = image::load_from_memory(png_bytes).expect("Failed to decode app icon");
        let rgba = img.into_rgba8();
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

    let primary = eframe::run_native("prunr", make_options(), app_factory());
    if let Err(ref e) = primary {
        log_startup_error("the graphics renderer failed", e);
        let fallback = eframe::run_native("prunr", on_opengl(make_options()), app_factory());
        if let Err(ref e2) = fallback {
            log_startup_error("the OpenGL renderer also failed", e2);
        }
        return fallback;
    }
    primary
}

/// `options` on wgpu's OpenGL backend alone: wgpu picks Vulkan, DirectX 12
/// or Metal first, and a driver that fails there (a virtual machine, older
/// Intel graphics) may still run OpenGL.
fn on_opengl(mut options: eframe::NativeOptions) -> eframe::NativeOptions {
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
        setup.instance_descriptor.backends = eframe::wgpu::Backends::GL;
    }
    options
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

#[cfg(test)]
mod renderer_tests {
    /// The second try asks wgpu for OpenGL alone, not again for the
    /// backend that just failed.
    #[test]
    fn the_fallback_runs_on_opengl_alone() {
        let options = super::on_opengl(eframe::NativeOptions::default());
        let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &options.wgpu_options.wgpu_setup else { panic!("a new wgpu setup") };
        assert_eq!(setup.instance_descriptor.backends, eframe::wgpu::Backends::GL);
    }
}
