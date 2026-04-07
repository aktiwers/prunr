use egui::{Pos2, Rect, RichText};

use crate::gui::app::BgPrunrApp;
use crate::gui::state::AppState;
use crate::gui::theme;

pub fn render(ui: &mut egui::Ui, app: &BgPrunrApp) {
    ui.horizontal(|ui| {
        ui.add_space(theme::SPACE_SM);

        // Left side: status text
        let status_text = match app.state {
            AppState::Empty | AppState::Loaded => {
                if app.status_is_temporary {
                    app.status_text.clone()
                } else {
                    "Ready".to_string()
                }
            }
            AppState::Processing => {
                format!("Processing... {}", app.progress_stage)
            }
            AppState::Animating | AppState::Done => app.status_text.clone(),
        };

        ui.label(RichText::new(&status_text).color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));

        // Center: progress bar only during Processing
        if app.state == AppState::Processing {
            let available_width = ui.available_width() - 200.0; // leave space for right side
            let bar_width = available_width.max(50.0);
            let bar_height = theme::PROGRESS_BAR_HEIGHT;

            // Allocate space for the progress bar
            let (rect, _) = ui.allocate_exact_size(
                egui::Vec2::new(bar_width, bar_height),
                egui::Sense::hover(),
            );

            // Background
            ui.painter().rect_filled(rect, 2.0, egui::Color32::from_rgb(0x30, 0x30, 0x30));

            // Fill
            let fill_w = rect.width() * app.progress_pct.clamp(0.0, 1.0);
            if fill_w > 0.0 {
                let fill_rect = Rect::from_min_max(
                    rect.min,
                    Pos2::new(rect.min.x + fill_w, rect.max.y),
                );
                ui.painter().rect_filled(fill_rect, 2.0, theme::PROGRESS_FILL);
            }
        }

        // Right side -- push to right
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(theme::SPACE_SM);

            // Backend badge
            ui.label(
                RichText::new(&app.active_backend)
                    .monospace()
                    .size(theme::FONT_SIZE_MONO)
                    .color(theme::TEXT_SECONDARY),
            );

            // Zoom percentage (right of dimensions, left of backend badge)
            if app.state != AppState::Empty {
                ui.add_space(theme::SPACE_SM);
                let zoom_pct = (app.zoom * 100.0).round() as u32;
                ui.label(
                    RichText::new(format!("{zoom_pct}%"))
                        .size(theme::FONT_SIZE_BODY)
                        .color(theme::TEXT_SECONDARY),
                );
            }

            // Image dimensions
            if let Some((w, h)) = app.image_dimensions {
                ui.add_space(theme::SPACE_SM);
                ui.label(
                    RichText::new(format!("{w}×{h}"))
                        .size(theme::FONT_SIZE_BODY)
                        .color(theme::TEXT_SECONDARY),
                );
            }
        });
    });
}
