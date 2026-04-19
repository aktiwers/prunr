use egui::{Color32, RichText};
use egui_material_icons::icons::*;

use crate::gui::app::PrunrApp;
use crate::gui::batch_manager::ProcessButtonLabel;
use crate::gui::history_manager::HistoryManager;
use crate::gui::item::BatchStatus;
use crate::gui::state::AppState;
use crate::gui::theme;

use super::modifier_key;

pub fn render(ui: &mut egui::Ui, app: &mut PrunrApp) {
    ui.horizontal_centered(|ui| {
        ui.spacing_mut().item_spacing.x = theme::SPACE_SM;
        ui.spacing_mut().button_padding = egui::vec2(8.0, 4.0);

        let can_save_copy = app.state == AppState::Done;
        let has_selected = app.batch.items.iter().any(|i| i.selected);
        let m = modifier_key();

        // ── Left: Open ──
        let open_btn = egui::Button::new(
            RichText::new(format!("{}  Open", ICON_FOLDER_OPEN.codepoint)).color(theme::TEXT_PRIMARY),
        )
        .fill(theme::BG_SECONDARY)
        .corner_radius(theme::BUTTON_ROUNDING)
        .min_size(egui::vec2(0.0, theme::BTN_HEIGHT));
        if ui.add(open_btn).on_hover_text(format!("Open image(s) ({m}+O)")).clicked() {
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
        if ui.add(gear_btn).on_hover_text(format!("Settings ({m}+Space)")).clicked() {
            if app.show_settings {
                app.close_settings(ui.ctx());
            } else {
                app.show_settings = true;
                app.settings_opened_at = ui.ctx().input(|i| i.time);
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
                .on_hover_text(format!("Undo ({m}+Z)"))
                .clicked()
            {
                app.handle_undo(ui.ctx());
            }
            if ui.add_enabled(can_redo, icon_btn(ICON_REDO.codepoint))
                .on_hover_text(format!("Redo ({m}+Y)"))
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
                if ui.add(save_btn).on_hover_text(format!("Save result ({m}+S)")).clicked() {
                    app.handle_save_selected();
                }
            }

            let is_batch_processing = app.batch.status_counts().processing > 0;

            // Cancel All replaces the Process button while any item is
            // mid-processing — single-action slot, no ambiguity.
            if is_batch_processing {
                let cancel_btn = egui::Button::new(
                    RichText::new(format!("{}  Cancel All", ICON_CANCEL.codepoint)).color(Color32::WHITE),
                )
                .fill(theme::DESTRUCTIVE)
                .corner_radius(theme::BUTTON_ROUNDING)
                .min_size(egui::vec2(0.0, theme::BTN_HEIGHT));
                if ui.add(cancel_btn).on_hover_text("Cancel all processing (Escape)").clicked() {
                    app.handle_cancel();
                    for item in &mut app.batch.items {
                        if item.status == BatchStatus::Processing {
                            item.status = BatchStatus::Pending;
                        }
                    }
                    app.state = AppState::Loaded;
                    app.status.text = "Cancelled".to_string();
                }
            } else {
                let label = app.batch.process_button_label();
                let target_ids = app.batch.items_to_process();
                // Enabled when at least one target exists and isn't already
                // Processing. Empty `target_ids` (empty batch) naturally → false.
                let has_processable = target_ids.iter().any(|id| {
                    app.batch.find_by_id(*id)
                        .map_or(false, |it| !matches!(it.status, BatchStatus::Processing))
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

                let tooltip = match label {
                    ProcessButtonLabel::ProcessAll(n) => format!("Process all {n} images ({m}+R)"),
                    ProcessButtonLabel::ProcessSelected(n) if n > 1 => {
                        format!("Process {n} selected images ({m}+R)")
                    }
                    _ => {
                        // ProcessViewed or ProcessSelected(1): single-image dispatch.
                        // Chain-mode tooltip varies on whether the target already has a result.
                        let target_has_result = target_ids.first().and_then(|id| app.batch.find_by_id(*id))
                            .map_or(false, |i| i.result_rgba.is_some());
                        if app.settings.chain_mode && target_has_result {
                            format!("Process current result ({m}+R)")
                        } else {
                            format!("Process original ({m}+R)")
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
