use egui::{Pos2, Rect, RichText};

use crate::gui::app::{PrunrApp, BatchStatus};
use crate::gui::state::AppState;
use crate::gui::theme;

pub fn render(ui: &mut egui::Ui, app: &PrunrApp) {
    ui.horizontal(|ui| {
        ui.add_space(theme::SPACE_XS);

        // Mini logo
        let logo_h = theme::STATUS_BAR_HEIGHT - 4.0;
        let logo_w = logo_h * theme::LOGO_ASPECT;
        ui.add(
            egui::Image::new(egui::include_image!("../../../../../img/logo-nobg.png"))
                .fit_to_exact_size(egui::vec2(logo_w, logo_h))
        );
        ui.add_space(theme::SPACE_XS);

        // Left side: status text
        let batch_processing = app.batch_items.iter().any(|i| i.status == BatchStatus::Processing);
        let batch_done_count = app.batch_items.iter().filter(|i| i.status == BatchStatus::Done).count();
        let batch_total = app.batch_items.len();

        let status_text = if batch_processing {
            format!("Processing {batch_done_count}/{batch_total} images...")
        } else if batch_total >= 2 && batch_done_count == batch_total {
            format!("All done \u{2014} {batch_total} images processed")
        } else if batch_total >= 2 && batch_done_count > 0 {
            format!("{batch_done_count} of {batch_total} images processed")
        } else {
            match app.state {
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
            }
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
            ui.painter().rect_filled(rect, 2.0, theme::PROGRESS_BAR_BG);

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
                RichText::new(format!("Backend: {}", app.settings.active_backend))
                    .monospace()
                    .size(theme::FONT_SIZE_MONO)
                    .color(theme::TEXT_SECONDARY),
            );

            // Zoom percentage with magnifier icon
            if app.state != AppState::Empty {
                ui.add_space(theme::SPACE_SM);
                let zoom_pct = (app.zoom * 100.0).round() as u32;
                ui.label(
                    RichText::new(format!("{} {zoom_pct}%", egui_material_icons::icons::ICON_SEARCH.codepoint))
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
