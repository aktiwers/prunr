pub mod app;
pub mod state;
pub mod settings;
pub mod worker;
pub mod theme;
pub mod views;
pub mod zoom_state;
pub mod status_state;
pub mod background_io;

#[cfg(test)]
mod tests;

pub fn run() -> eframe::Result {
    // Load app icon from embedded PNG
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

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Prunr")
            .with_inner_size(theme::DEFAULT_WINDOW_SIZE)
            .with_min_inner_size(theme::MIN_WINDOW_SIZE)
            .with_drag_and_drop(true)
            .with_icon(icon),
        ..Default::default()
    };
    eframe::run_native(
        "prunr",
        native_options,
        Box::new(|cc| Ok(Box::new(app::PrunrApp::new(cc)))),
    )
}
