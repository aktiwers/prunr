use egui::{Color32, RichText};
use egui_material_icons::icons::*;

use crate::gui::app::PrunrApp;
use crate::gui::batch_manager::ProcessButtonLabel;
use crate::gui::history_manager::HistoryManager;
use crate::gui::item::BatchStatus;
use crate::gui::state::AppState;
use crate::gui::theme;

use crate::kb;
use super::KB_MOD;

pub fn render(ui: &mut egui::Ui, app: &mut PrunrApp) {
    ui.horizontal_centered(|ui| {
        ui.spacing_mut().item_spacing.x = theme::SPACE_SM;
        ui.spacing_mut().button_padding = egui::vec2(8.0, 4.0);

        let can_save_copy = app.batch.app_state() == AppState::Done;
        let has_selected = app.batch.has_any_selected();

        // ── Left: Open ──
        let open_btn = egui::Button::new(
            RichText::new(format!("{}  Open", ICON_FOLDER_OPEN.codepoint)).color(theme::TEXT_PRIMARY),
        )
        .fill(theme::BG_SECONDARY)
        .corner_radius(theme::BUTTON_ROUNDING)
        .min_size(egui::vec2(0.0, theme::BTN_HEIGHT));
        if ui.add(open_btn).on_hover_text(kb!("Open image(s)", "O")).clicked() {
            app.pending_open_dialog = true;
        }

        // ── Settings gear + Model dropdown ──

        let gear_btn = egui::Button::new(
            RichText::new(ICON_SETTINGS.codepoint)
                .size(theme::ICON_SIZE_BUTTON)
                .color(theme::TEXT_PRIMARY),
        )
        .fill(theme::BG_SECONDARY)
        .corner_radius(theme::BUTTON_ROUNDING)
        .min_size(egui::vec2(theme::BTN_HEIGHT, theme::BTN_HEIGHT));
        if ui.add(gear_btn).on_hover_text(kb!("Settings", "Space")).clicked() {
            if app.show_settings {
                app.close_settings(ui.ctx());
            } else {
                app.open_settings();
            }
        }

        // Model + Preset dropdowns live on row 2 (adjustments_toolbar); the
        // Lines mode selector lives on row 3's left edge alongside its own
        // chips. Row 1 stays minimal: Open, Settings, and the action cluster.

        if !app.batch.items.is_empty() {
            let can_undo = app.batch.any_target_can(HistoryManager::can_undo);
            let can_redo = app.batch.any_target_can(HistoryManager::can_redo);
            let icon_btn = |icon: &'static str| egui::Button::new(
                RichText::new(icon).size(theme::ICON_SIZE_BUTTON).color(theme::TEXT_PRIMARY),
            )
            .fill(theme::BG_SECONDARY)
            .corner_radius(theme::BUTTON_ROUNDING)
            .min_size(egui::vec2(theme::BTN_HEIGHT, theme::BTN_HEIGHT));

            if ui.add_enabled(can_undo, icon_btn(ICON_UNDO.codepoint))
                .on_hover_text(kb!("Undo", "Z"))
                .clicked()
            {
                app.handle_undo(ui.ctx());
            }
            if ui.add_enabled(can_redo, icon_btn(ICON_REDO.codepoint))
                .on_hover_text(kb!("Redo", "Y"))
                .clicked()
            {
                app.handle_redo(ui.ctx());
            }
        }

        // ── Right group: action buttons ──
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if has_selected {
                let remove_sel_btn = egui::Button::new(
                    RichText::new(format!("{}  Remove Selected", ICON_DELETE.codepoint)).color(Color32::WHITE),
                )
                .fill(theme::DESTRUCTIVE)
                .corner_radius(theme::BUTTON_ROUNDING);
                if ui.add(remove_sel_btn).clicked() {
                    app.remove_selected();
                }
            }

            let has_saveable_selected = has_selected
                && app.batch.items.iter().any(|i| i.selected && i.status == BatchStatus::Done);
            let show_save = can_save_copy || has_saveable_selected;
            if show_save {
                let save_label = if has_selected {
                    format!("{}  Save Selected", ICON_SAVE.codepoint)
                } else {
                    format!("{}  Save", ICON_SAVE.codepoint)
                };
                let save_btn = egui::Button::new(
                    RichText::new(save_label).color(theme::TEXT_PRIMARY),
                )
                .fill(theme::BG_SECONDARY)
                .corner_radius(theme::BUTTON_ROUNDING);
                if ui.add(save_btn).on_hover_text(kb!("Save result", "S")).clicked() {
                    app.handle_save_selected();
                }
            }


            let is_batch_processing = app.batch.status_counts().processing > 0;
            let selected_processing = app.batch.items.iter()
                .any(|i| i.selected && i.status == BatchStatus::Processing);

            if is_batch_processing {
                let partial = selected_processing;
                let cancel_label = if partial { "Cancel Selected" } else { "Cancel All" };
                let cancel_btn = egui::Button::new(
                    RichText::new(format!("{}  {cancel_label}", ICON_CANCEL.codepoint)).color(Color32::WHITE),
                )
                .fill(theme::DESTRUCTIVE)
                .corner_radius(theme::BUTTON_ROUNDING)
                .min_size(egui::vec2(0.0, theme::BTN_HEIGHT));
                let tip = if partial {
                    "Stop the selected items — others keep running"
                } else {
                    "Cancel all processing (Escape)"
                };
                if ui.add(cancel_btn).on_hover_text(tip).clicked() {
                    if partial {
                        app.handle_cancel_selected();
                    } else {
                        app.handle_cancel_all_and_reset();
                    }
                }
            }

            // Process button — always visible. Clicking while a batch is
            // already running enqueues the new work on the worker bridge;
            // it runs once the current batch finishes (no auto-cancel).
            //
            // Eraser-mode (SD / LaMa / MI-GAN inpaint) repurposes Process
            // as "Reprocess stroke": dispatch the current item's committed
            // mask correction through the inpaint pipeline using the
            // current toolbar settings. Enabled only when there's a stroke
            // to reprocess. The seg pipeline (`handle_remove_bg`) doesn't
            // apply to inpaint backends and was incorrectly hiding the
            // canvas image when clicked here.
            {
                let inpaint_mode = app.settings.model.is_inpaint();
                let has_correction = app.batch.selected_item()
                    .and_then(|i| i.mask_correction.as_ref())
                    .is_some();
                let label = app.batch.process_button_label();
                // Seg pipeline gating: at least one target exists and isn't
                // already Processing. `any_target_can` is the no-alloc
                // primitive — building `items_to_process()` here ran a
                // `Vec<u64>::collect` every frame just to throw it away.
                let has_processable = if inpaint_mode {
                    has_correction
                } else {
                    app.batch.any_target_can(|it|
                        !matches!(it.status, BatchStatus::Processing)
                    )
                };

                let (label_text, is_all) = if inpaint_mode {
                    ("Reprocess stroke".to_string(), false)
                } else {
                    match label {
                        ProcessButtonLabel::ProcessViewed => ("Process".to_string(), false),
                        ProcessButtonLabel::ProcessSelected(1) => ("Process 1 selected".to_string(), false),
                        ProcessButtonLabel::ProcessSelected(n) => (format!("Process {n} selected"), false),
                        ProcessButtonLabel::ProcessAll(n) => (format!("Process All [{n}]"), true),
                    }
                };

                let text_color = if has_processable {
                    Color32::WHITE
                } else {
                    Color32::from_rgba_unmultiplied(255, 255, 255, 102)
                };
                let fill = if has_processable { theme::ACCENT } else { theme::ACCENT_DISABLED };

                let icon = if is_all {
                    egui::Image::new(egui::include_image!("../../../../../img/batch-icon.png"))
                } else {
                    egui::Image::new(egui::include_image!("../../../../../img/logo-nobg.png"))
                }
                .fit_to_exact_size(egui::vec2(22.0, 22.0));

                let btn = egui::Button::image_and_text(icon, RichText::new(label_text).color(text_color))
                    .fill(fill)
                    .corner_radius(theme::BUTTON_ROUNDING)
                    .min_size(egui::vec2(0.0, theme::BTN_HEIGHT));

                let tooltip: std::borrow::Cow<'static, str> = if inpaint_mode {
                    if has_correction {
                        "Re-dispatch the current stroke through the inpaint model with the current toolbar settings (prompt / scheduler / steps / strength).".into()
                    } else {
                        "Paint a stroke first — Process re-dispatches the painted region with current settings.".into()
                    }
                } else {
                    match label {
                        ProcessButtonLabel::ProcessAll(n) => format!("Process all {n} images ({}+R)", KB_MOD).into(),
                        ProcessButtonLabel::ProcessSelected(n) if n > 1 => {
                            format!("Process {n} selected images ({}+R)", KB_MOD).into()
                        }
                        _ => {
                            let target_has_result = app.batch.first_target_item()
                                .is_some_and(|i| i.result_rgba.is_some());
                            if app.settings.chain_mode && target_has_result {
                                kb!("Process current result", "R").into()
                            } else {
                                kb!("Process original", "R").into()
                            }
                        }
                    }
                };

                if ui.add_enabled(has_processable, btn).on_hover_text(tooltip).clicked() {
                    if inpaint_mode {
                        if let Some(idx) = app.batch.selected_idx_clamped() {
                            app.dispatch_inpaint_for_item(idx);
                        }
                    } else {
                        app.handle_remove_bg();
                    }
                }
            }

        });
    });
}
