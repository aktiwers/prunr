pub mod app;
pub mod state;
pub mod worker;
pub mod theme;
pub mod views;

#[cfg(test)]
mod tests;

pub fn run() -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("BgPrunR")
            .with_inner_size(theme::DEFAULT_WINDOW_SIZE)
            .with_min_inner_size(theme::MIN_WINDOW_SIZE)
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "bgprunr",
        native_options,
        Box::new(|cc| Ok(Box::new(app::BgPrunrApp::new(cc)))),
    )
}
