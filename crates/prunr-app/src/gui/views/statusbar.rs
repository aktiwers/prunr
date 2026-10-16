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

        let app_state = app.batch.app_state();
        let counts = app.batch.status_counts();
        let batch_processing = app.batch.any_awaiting_cut();
        let batch_done_count = counts.done;
        // Only count items involved in processing (not idle Pending items)
        let batch_total = counts.batch_total();
        let inpaint_busy = app.processor.any_inpaint_in_flight();

        // The canvas capsule carries the step and counts; this text is
        // the terse fallback for eyes off the canvas, without counters.
        let running = |step: prunr_core::Step| format!("{}\u{2026}", step.label());
        let status_text = if app.processor.is_upscale_in_flight() {
            running(prunr_core::Step::Upscaling)
        } else if app.processor.is_colorize_in_flight() {
            running(prunr_core::Step::Colorizing)
        } else if inpaint_busy {
            // Eraser dispatch lives outside the seg-pipeline status track,
            // so check it first — otherwise the "All done" branch wins.
            running(prunr_core::Step::Erasing)
        } else if batch_processing {
            format!("Processing {batch_done_count}/{batch_total} images...")
        } else if batch_total >= 2 && batch_done_count == batch_total {
            format!("All done \u{2014} {batch_total} images processed")
        } else if batch_total >= 2 && batch_done_count > 0 {
            format!("{batch_done_count} of {batch_total} images processed")
        } else {
            match app_state {
                AppState::Empty | AppState::Loaded => {
                    if app.status.is_temporary() {
                        app.status.text.clone()
                    } else {
                        crate::gui::status_state::READY.to_string()
                    }
                }
                AppState::Processing => {
                    format!("Processing... {}", app.status.stage)
                }
                AppState::Done => app.status.text.clone(),
            }
        };

        ui.label(RichText::new(&status_text).color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));

        // Steps Cmd+Z can walk back.
        let depth = app.batch.selected_item().map_or(0, |item| item.timeline.undo_steps());
        if depth > 0 {
            ui.add_space(theme::SPACE_SM);
            ui.label(
                RichText::new(format!("{depth} undo"))
                    .monospace()
                    .size(theme::FONT_SIZE_SECONDARY)
                    .color(theme::TEXT_SECONDARY),
            );
        }

        if app_state == AppState::Processing {
            let pct = (app.status.pct * 100.0).round() as u32;
            ui.label(
                RichText::new(format!("{pct}%"))
                    .monospace()
                    .size(theme::FONT_SIZE_SECONDARY)
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
            let ticked_count = app.batch.ticked_count();
            if ticked_count >= 2 {
                ui.add_space(theme::SPACE_SM);
                ui.label(
                    RichText::new(
                        "Live editing applies to the viewed image only \u{2014} \
                         click Process to apply to all selected."
                    )
                    .size(theme::FONT_SIZE_SECONDARY)
                    .color(theme::TEXT_SECONDARY),
                );
            }
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(theme::SPACE_SM);

            let backend_display = if app.settings.cpu_only() { "CPU only" } else { &app.settings.active_backend };
            ui.label(
                RichText::new(format!("Backend: {backend_display}"))
                    .monospace()
                    .size(theme::FONT_SIZE_SECONDARY)
                    .color(theme::TEXT_SECONDARY),
            );

            if app_state != AppState::Empty {
                ui.add_space(theme::SPACE_SM);
                let zoom_pct = (app.zoom_state.zoom * 100.0).round() as u32;
                ui.label(
                    RichText::new(format!("{} {zoom_pct}%", egui_material_icons::icons::ICON_SEARCH.codepoint))
                        .size(theme::FONT_SIZE_BODY)
                        .color(theme::TEXT_SECONDARY),
                );
            }

            // Dimensions of what's currently shown on canvas: the processed
            // result if there is one (so an upscale's 4× output reads
            // correctly), otherwise the source dimensions captured at decode.
            if let Some(item) = app.batch.selected_item() {
                let (w, h) = item.result_rgba
                    .as_ref()
                    .map(|r| r.dimensions())
                    .unwrap_or(item.dimensions);
                ui.add_space(theme::SPACE_SM);
                ui.label(
                    RichText::new(format!("{w}×{h}"))
                        .size(theme::FONT_SIZE_BODY)
                        .color(theme::TEXT_SECONDARY),
                );
                let erase_scale = item.timeline.current().origin.as_ref()
                    .filter(|o| o.engine == crate::gui::item::Engine::Erase)
                    .and_then(|o| o.fill_scale);
                if let Some(scale) = erase_scale {
                    ui.add_space(theme::SPACE_SM);
                    ui.label(
                        RichText::new(format!("Fill {}", super::fmt::percent(scale)))
                            .size(theme::FONT_SIZE_BODY)
                            .color(theme::TEXT_SECONDARY),
                    ).on_hover_text("How large the eraser filled its region. 100% is full size; Auto shrinks a large region to fit the model.");
                }
            }
        });
    });
}
