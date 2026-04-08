use egui::{Color32, RichText};
use egui_material_icons::icons::*;

use crate::gui::app::{PrunrApp, BatchStatus};
use crate::gui::settings::SettingsModel;
use crate::gui::state::AppState;
use crate::gui::theme;

pub fn render(ui: &mut egui::Ui, app: &mut PrunrApp) {
    ui.horizontal_centered(|ui| {
        ui.spacing_mut().item_spacing.x = theme::SPACE_SM;
        ui.spacing_mut().button_padding = egui::vec2(8.0, 4.0);

        let is_processing = app.state == AppState::Processing;
        let can_save_copy = app.state == AppState::Done;
        let has_selected = app.batch_items.iter().any(|i| i.selected);

        // ── Left group: Open + Model ──
        let open_btn = egui::Button::new(
            RichText::new(format!("{} Open", ICON_FOLDER_OPEN.codepoint)).color(theme::TEXT_PRIMARY),
        )
        .fill(theme::BG_SECONDARY)
        .corner_radius(theme::BUTTON_ROUNDING);
        if ui.add(open_btn).clicked() {
            app.pending_open_dialog = true;
        }

        ui.add_enabled_ui(!is_processing, |ui| {
            let icon_h = ui.spacing().interact_size.y;
            ui.add(
                egui::Image::new(egui::include_image!("../../../../../img/aicon.jpeg"))
                    .fit_to_exact_size(egui::vec2(icon_h, icon_h))
                    .corner_radius(egui::CornerRadius::same(3)),
            );
            let model_text = match app.settings.model {
                SettingsModel::Silueta => "silueta (fast)",
                SettingsModel::U2net => "u2net (quality)",
            };
            egui::ComboBox::from_id_salt("model")
                .selected_text(model_text)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut app.settings.model,
                        SettingsModel::Silueta,
                        "silueta (fast)",
                    );
                    ui.selectable_value(
                        &mut app.settings.model,
                        SettingsModel::U2net,
                        "u2net (quality)",
                    );
                });
        });

        // ── Center: Settings gear (bigger) ──
        let gear_size = 32.0;
        let toolbar_center_x = ui.max_rect().center().x;
        let current_x = ui.cursor().min.x;
        let offset = (toolbar_center_x - current_x - gear_size / 2.0).max(0.0);
        ui.add_space(offset);

        let gear_btn = egui::Button::new(
            RichText::new(ICON_SETTINGS.codepoint)
                .size(20.0)
                .color(theme::TEXT_PRIMARY),
        )
        .fill(theme::BG_SECONDARY)
        .corner_radius(theme::BUTTON_ROUNDING)
        .min_size(egui::vec2(gear_size, gear_size));
        if ui.add(gear_btn).on_hover_text("Settings (Ctrl+Space)").clicked() {
            if app.show_settings {
                app.close_settings();
            } else {
                app.show_settings = true;
                app.settings_opened_at = ui.ctx().input(|i| i.time);
            }
        }

        // ── Right group: action buttons ──
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Remove Selected (rightmost)
            if has_selected {
                let remove_sel_btn = egui::Button::new(
                    RichText::new(format!("{} Remove Selected", ICON_DELETE.codepoint)).color(Color32::WHITE),
                )
                .fill(theme::DESTRUCTIVE)
                .corner_radius(theme::BUTTON_ROUNDING);
                if ui.add(remove_sel_btn).clicked() {
                    app.remove_selected();
                }
            }

            // Save — only visible when there's something to save
            let has_saveable_selected = has_selected
                && app.batch_items.iter().any(|i| i.selected && i.status == BatchStatus::Done);
            let show_save = can_save_copy || has_saveable_selected;
            if show_save {
                let save_icon = ICON_SAVE.codepoint;
                let save_label = if has_selected {
                    format!("{save_icon} Save Selected")
                } else {
                    format!("{save_icon} Save")
                };
                let save_btn = egui::Button::new(
                    RichText::new(save_label).color(theme::TEXT_PRIMARY),
                )
                .fill(theme::BG_SECONDARY)
                .corner_radius(theme::BUTTON_ROUNDING);
                if ui.add(save_btn).clicked() {
                    app.handle_save_selected();
                }
            }

            // Process All (right-to-left: rendered first = rightmost)
            if app.batch_items.len() >= 2 {
                let has_pending = app.batch_items.iter().any(|i| i.status == BatchStatus::Pending);
                let is_batch_processing = app.batch_items.iter().any(|i| i.status == BatchStatus::Processing);
                let enabled = has_pending && !is_batch_processing;

                let fill = if enabled { theme::ACCENT } else { theme::ACCENT_DISABLED };
                let text_color = if enabled {
                    Color32::WHITE
                } else {
                    Color32::from_rgba_unmultiplied(255, 255, 255, 102)
                };
                let batch_icon = egui::Image::new(egui::include_image!("../../../../../img/batch-icon.png"))
                    .fit_to_exact_size(egui::vec2(22.0, 22.0));
                let btn = egui::Button::image_and_text(batch_icon, RichText::new("Process All").color(text_color))
                    .fill(fill)
                    .corner_radius(theme::BUTTON_ROUNDING);
                if ui.add_enabled(enabled, btn).clicked() {
                    app.handle_process_all();
                }
            }

            // Remove BG / Process Selected (right-to-left: rendered second = left of Process All)
            let has_processable = if has_selected {
                app.batch_items.iter().any(|i| i.selected && matches!(i.status, BatchStatus::Pending | BatchStatus::Error(_)))
            } else {
                app.batch_items.get(app.selected_batch_index)
                    .map_or(app.state == AppState::Loaded, |item| matches!(item.status, BatchStatus::Pending | BatchStatus::Error(_)))
            };
            let remove_label = if has_selected { "Process Selected" } else { "Remove BG" };
            let remove_text = if !has_processable || is_processing {
                RichText::new(remove_label).color(Color32::from_rgba_unmultiplied(255, 255, 255, 102))
            } else {
                RichText::new(remove_label).color(Color32::WHITE)
            };
            let remove_fill = if has_processable && !is_processing {
                theme::ACCENT
            } else {
                theme::ACCENT_DISABLED
            };
            let logo_icon = egui::Image::new(egui::include_image!("../../../../../img/logo-nobg.png"))
                .fit_to_exact_size(egui::vec2(22.0, 22.0));
            let remove_btn = egui::Button::image_and_text(logo_icon, remove_text)
                .fill(remove_fill)
                .corner_radius(theme::BUTTON_ROUNDING);
            if ui.add_enabled(has_processable && !is_processing, remove_btn).clicked() {
                app.handle_remove_bg();
            }
        });
    });
}
