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
            format!("Processing {batch_done_count}/{}\u{2026}", super::fmt::count(batch_total, "image", "images"))
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
                    format!("Processing\u{2026} {}", app.status.stage)
                }
                AppState::Done => app.status.text.clone(),
            }
        };

        ui.label(RichText::new(&status_text).color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));

        if let Some(line) = downloads_line(app) {
            ui.add_space(theme::SPACE_SM);
            ui.label(RichText::new(line).size(theme::FONT_SIZE_SECONDARY).color(theme::TEXT_SECONDARY));
        }

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
            // With two or more images ticked, say that live preview only
            // updates the viewed image, not the whole selection.
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

            ui.label(
                RichText::new(format!("Accelerator: {}", model_engine(ui.ctx(), app)))
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
                let (w, h) = item.shown_dims();
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

/// The downloads and the install under way, for eyes off the Model Store
/// and Settings: "Downloading DDColor 44% · 2 more waiting".
fn downloads_line(app: &PrunrApp) -> Option<String> {
    use crate::gui::download_manager::DownloadState;
    let install = app.runtime_install.as_ref().map(|p| format!("{}: {}", p.target.display_name(), p.last_event.status_text()));
    let download = app.download_manager.active().map(|id| {
        let name = prunr_models::display_name(id);
        let doing = match app.download_manager.state(id) {
            DownloadState::InProgress { bytes_so_far, total_bytes } if total_bytes > 0 => format!("Downloading {name} {}%", bytes_so_far * 100 / total_bytes),
            DownloadState::Verifying => format!("Verifying {name}"),
            _ => format!("Downloading {name}"),
        };
        match app.download_manager.queued() {
            0 => doing,
            n => format!("{doing} \u{b7} {n} more waiting"),
        }
    });
    match (install, download) {
        (Some(install), Some(download)) => Some(format!("{install} \u{b7} {download}")),
        (install, download) => install.or(download),
    }
}

/// What the status bar names as the accelerator: "CPU only" when the user
/// chose it, "Checking" until the GPU probe has answered, else the engine
/// the chosen model runs on, which each model picks for itself; read again
/// only when the model, the probe's answer or what was measured changes.
fn model_engine<'a>(ctx: &egui::Context, app: &'a PrunrApp) -> &'a str {
    let backend = app.settings.active_backend.as_str();
    if app.settings.cpu_only() {
        return "CPU only";
    }
    if backend.is_empty() {
        return "Checking";
    }
    let Some(model) = app.settings.model.to_model_id() else { return backend };
    // The probe answers once per process, and the results are read again
    // whenever what was measured changes.
    let key = (model, std::sync::Arc::as_ptr(&app.test_results) as usize);
    let id = egui::Id::new("statusbar_model_engine");
    if let Some((_, engine)) = ctx.data(|d| d.get_temp::<((prunr_models::ModelId, usize), &'static str)>(id)).filter(|(k, _)| *k == key) {
        return engine;
    }
    let engine = app.settings.engine_of(model);
    ctx.data_mut(|d| d.insert_temp(id, (key, engine)));
    engine
}

#[cfg(test)]
mod tests {
    use crate::gui::settings::{ComputeMode, SettingsModel};

    /// The status bar names the chosen model's engine: the user's CPU
    /// only first, the probe's wait, and with no model the runtime's own.
    #[test]
    fn the_status_bar_names_the_chosen_models_engine() {
        let ctx = egui::Context::default();
        let mut app = crate::gui::app::PrunrApp::new_for_test();
        app.settings.active_backend = String::new();
        assert_eq!(super::model_engine(&ctx, &app), "Checking");
        app.settings.active_backend = "DirectML".into();
        app.settings.model = SettingsModel::None;
        assert_eq!(super::model_engine(&ctx, &app), "DirectML", "no model: the runtime's");
        app.settings.model = SettingsModel::Silueta;
        app.settings.processor = ComputeMode::CpuOnly;
        assert_eq!(super::model_engine(&ctx, &app), "CPU only");
        app.settings.processor = ComputeMode::Auto;
        let engine = super::model_engine(&ctx, &app);
        assert_eq!(engine, prunr_core::memory::provider_now(prunr_models::ModelId::Silueta, false), "the model's own");
    }

    use prunr_models::ModelId;

    /// A download under way shows in the footer with what waits behind
    /// it; with none, the footer shows nothing for it.
    #[test]
    fn the_footer_follows_the_downloads() {
        let mut app = crate::gui::app::PrunrApp::new_for_test();
        assert_eq!(super::downloads_line(&app), None);
        app.download_manager.begin_for_test(ModelId::DdcolorTiny);
        assert_eq!(super::downloads_line(&app).as_deref(), Some("Downloading DDColor 0%"));
    }
}
