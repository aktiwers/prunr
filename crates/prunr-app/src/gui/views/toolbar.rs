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
        let has_selected = app.batch.items.iter().any(|i| i.selected);

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
                app.open_settings(ui.ctx());
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
                        app.handle_cancel();
                        for item in &mut app.batch.items {
                            if item.status == BatchStatus::Processing {
                                item.status = BatchStatus::Pending;
                            }
                        }
                        app.status.text = "Cancelled".to_string();
                    }
                }
            }

            // Process button — always visible. Clicking while a batch is
            // already running enqueues the new work on the worker bridge;
            // it runs once the current batch finishes (no auto-cancel).
            {
                let label = app.batch.process_button_label();
                let target_ids = app.batch.items_to_process();
                // Enabled when at least one target exists and isn't already
                // Processing. Empty `target_ids` (empty batch) naturally → false.
                let has_processable = target_ids.iter().any(|id| {
                    app.batch.find_by_id(*id)
                        .is_some_and(|it| !matches!(it.status, BatchStatus::Processing))
                });

                let (label_text, is_all) = match label {
                    ProcessButtonLabel::ProcessViewed => ("Process".to_string(), false),
                    ProcessButtonLabel::ProcessSelected(1) => ("Process 1 selected".to_string(), false),
                    ProcessButtonLabel::ProcessSelected(n) => (format!("Process {n} selected"), false),
                    ProcessButtonLabel::ProcessAll(n) => (format!("Process All [{n}]"), true),
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

                let tooltip: std::borrow::Cow<'static, str> = match label {
                    ProcessButtonLabel::ProcessAll(n) => format!("Process all {n} images ({}+R)", KB_MOD).into(),
                    ProcessButtonLabel::ProcessSelected(n) if n > 1 => {
                        format!("Process {n} selected images ({}+R)", KB_MOD).into()
                    }
                    _ => {
                        let target_has_result = target_ids.first().and_then(|id| app.batch.find_by_id(*id))
                            .is_some_and(|i| i.result_rgba.is_some());
                        if app.settings.chain_mode && target_has_result {
                            kb!("Process current result", "R").into()
                        } else {
                            kb!("Process original", "R").into()
                        }
                    }
                };

                if ui.add_enabled(has_processable, btn).on_hover_text(tooltip).clicked() {
                    app.handle_remove_bg();
                }
            }

        });
    });
}
