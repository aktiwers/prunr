use egui::{Pos2, Rect, RichText};

use crate::gui::app::PrunrApp;
use crate::gui::state::AppState;
use crate::gui::theme;

pub fn render(ui: &mut egui::Ui, app: &PrunrApp) {
    ui.horizontal(|ui| {
        ui.add_space(theme::SPACE_XS);

        let logo_h = theme::STATUS_BAR_HEIGHT - 4.0;
        let logo_w = logo_h * theme::LOGO_ASPECT;
        ui.add(
            egui::Image::new(egui::include_image!("../../../../../img/logo-nobg.png"))
                .fit_to_exact_size(egui::vec2(logo_w, logo_h))
        );
        ui.add_space(theme::SPACE_XS);

        let counts = app.batch.status_counts();
        let batch_processing = counts.processing > 0;
        let batch_done_count = counts.done;
        // Only count items involved in processing (not idle Pending items)
        let batch_total = counts.batch_total();

        let status_text = if batch_processing {
            format!("Processing {batch_done_count}/{batch_total} images...")
        } else if batch_total >= 2 && batch_done_count == batch_total {
            format!("All done \u{2014} {batch_total} images processed")
        } else if batch_total >= 2 && batch_done_count > 0 {
            format!("{batch_done_count} of {batch_total} images processed")
        } else {
            match app.state {
                AppState::Empty | AppState::Loaded => {
                    if app.status.is_temporary {
                        app.status.text.clone()
                    } else {
                        "Ready".to_string()
                    }
                }
                AppState::Processing => {
                    format!("Processing... {}", app.status.stage)
                }
                AppState::Done => app.status.text.clone(),
            }
        };

        ui.label(RichText::new(&status_text).color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));

        // History/undo depth indicator (visible when chain_mode is on and history exists)
        if app.settings.chain_mode {
            if let Some(item) = app.batch.selected_item() {
                let depth = item.history.len();
                if depth > 0 {
                    ui.add_space(theme::SPACE_SM);
                    ui.label(
                        RichText::new(format!("{depth} undo"))
                            .monospace()
                            .size(theme::FONT_SIZE_MONO)
                            .color(theme::TEXT_SECONDARY),
                    );
                }
            }
        }

        if app.state == AppState::Processing {
            let pct = (app.status.pct * 100.0).round() as u32;
            ui.label(
                RichText::new(format!("{pct}%"))
                    .monospace()
                    .size(theme::FONT_SIZE_MONO)
                    .color(theme::TEXT_SECONDARY),
            );

            let available_width = ui.available_width() - 320.0;
            let bar_width = available_width.max(50.0);
            let bar_height = theme::PROGRESS_BAR_HEIGHT;

            let (rect, _) = ui.allocate_exact_size(
                egui::Vec2::new(bar_width, bar_height),
                egui::Sense::hover(),
            );

            ui.painter().rect_filled(rect, 2.0, theme::PROGRESS_BAR_BG);

            let fill_w = rect.width() * app.status.pct.clamp(0.0, 1.0);
            if fill_w > 0.0 {
                let fill_rect = Rect::from_min_max(
                    rect.min,
                    Pos2::new(rect.min.x + fill_w, rect.max.y),
                );
                ui.painter().rect_filled(fill_rect, 2.0, theme::PROGRESS_FILL);
            }
        } else {
            // Teach-the-user hint when ≥ 2 checkboxes are set: the canvas
            // live-preview only updates the viewed image, not the whole
            // selection. Sits in the statusbar center (between the left
            // status text and the right metadata) — only visible when the
            // progress bar isn't occupying that slot.
            let selected_count = app.batch.items.iter().filter(|i| i.selected).count();
            if selected_count >= 2 {
                ui.add_space(theme::SPACE_SM);
                ui.label(
                    RichText::new(
                        "Live editing applies to the viewed image only \u{2014} \
                         click Process to apply to all selected."
                    )
                    .size(theme::FONT_SIZE_MONO)
                    .color(theme::TEXT_SECONDARY),
                );
            }
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(theme::SPACE_SM);

            // `force_cpu` overrides the detected backend so the toggle reflects
            // the override instantly, not after the next batch reinit.
            let backend_display = if app.settings.force_cpu {
                "CPU (forced)"
            } else {
                &app.settings.active_backend
            };
            ui.label(
                RichText::new(format!("Backend: {backend_display}"))
                    .monospace()
                    .size(theme::FONT_SIZE_MONO)
                    .color(theme::TEXT_SECONDARY),
            );

            if app.state != AppState::Empty {
                ui.add_space(theme::SPACE_SM);
                let zoom_pct = (app.zoom_state.zoom * 100.0).round() as u32;
                ui.label(
                    RichText::new(format!("{} {zoom_pct}%", egui_material_icons::icons::ICON_SEARCH.codepoint))
                        .size(theme::FONT_SIZE_BODY)
                        .color(theme::TEXT_SECONDARY),
                );
            }

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
