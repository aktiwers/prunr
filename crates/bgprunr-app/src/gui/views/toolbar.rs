use egui::{Color32, RichText};
use egui_material_icons::icons::*;

use crate::gui::app::{BgPrunrApp, BatchStatus};
use crate::gui::settings::SettingsModel;
use crate::gui::state::AppState;
use crate::gui::theme;

pub fn render(ui: &mut egui::Ui, app: &mut BgPrunrApp) {
    ui.horizontal_centered(|ui| {
        ui.spacing_mut().item_spacing.x = theme::SPACE_SM;
        ui.spacing_mut().button_padding = egui::vec2(8.0, 4.0);

        let is_processing = app.state == AppState::Processing;
        let can_remove = matches!(app.state, AppState::Loaded | AppState::Done);
        let can_save_copy = app.state == AppState::Done;

        let open_btn = egui::Button::new(
            RichText::new(format!("{} Open", ICON_FOLDER_OPEN.codepoint)).color(theme::TEXT_PRIMARY),
        )
        .fill(theme::BG_SECONDARY)
        .corner_radius(theme::BUTTON_ROUNDING);

        if ui.add(open_btn).clicked() {
            app.pending_open_dialog = true;
        }

        // Remove BG button — processes checked items, or current if none checked
        let has_selected = app.batch_items.iter().any(|i| i.selected);
        let has_processable = if has_selected {
            app.batch_items.iter().any(|i| i.selected && matches!(i.status, BatchStatus::Pending | BatchStatus::Error(_)))
        } else {
            can_remove
        };
        let remove_label = if has_selected { "Remove BG Selected" } else { "Remove BG" };
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
        let remove_btn = egui::Button::new(remove_text)
            .fill(remove_fill)
            .corner_radius(theme::BUTTON_ROUNDING);

        if ui.add_enabled(has_processable && !is_processing, remove_btn).clicked() {
            app.handle_remove_bg();
        }

        // Process All button — only shown when batch items exist
        if app.batch_items.len() >= 2 {
            let has_pending = app.batch_items.iter().any(|i| i.status == BatchStatus::Pending);
            let is_batch_processing = app.batch_items.iter().any(|i| i.status == BatchStatus::Processing);
            let process_all_enabled = has_pending && !is_batch_processing;

            let process_all_fill = if process_all_enabled {
                theme::ACCENT
            } else {
                theme::ACCENT_DISABLED
            };
            let process_all_text = if process_all_enabled {
                RichText::new("Process All").color(Color32::WHITE)
            } else {
                RichText::new("Process All").color(Color32::from_rgba_unmultiplied(255, 255, 255, 102))
            };
            let process_all_btn = egui::Button::new(process_all_text)
                .fill(process_all_fill).corner_radius(theme::BUTTON_ROUNDING);

            if ui.add_enabled(process_all_enabled, process_all_btn).clicked() {
                app.handle_process_all();
            }
        }

        // Right-aligned group: settings + model selector + save/copy
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let btn_height = ui.spacing().interact_size.y;

            // Settings gear button
            let gear_btn = egui::Button::new(
                RichText::new(ICON_SETTINGS.codepoint).size(theme::FONT_SIZE_BODY).color(theme::TEXT_PRIMARY),
            )
            .fill(theme::BG_SECONDARY)
            .corner_radius(theme::BUTTON_ROUNDING)
            .min_size(egui::vec2(0.0, btn_height));
            if ui.add(gear_btn).on_hover_text("Settings (Ctrl+,)").clicked() {
                app.show_settings = !app.show_settings;
            }
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

            // Save All button — only shown when batch has 2+ done items
            {
                let done_count = app.batch_items.iter()
                    .filter(|i| i.status == BatchStatus::Done && i.result_rgba.is_some())
                    .count();
                if done_count >= 2 {
                    let save_all_btn = egui::Button::new(
                        RichText::new(format!("{} Save All", ICON_SAVE.codepoint)).color(Color32::WHITE),
                    )
                    .fill(theme::ACCENT)
                    .corner_radius(theme::BUTTON_ROUNDING);
                    if ui.add(save_all_btn).clicked() {
                        app.handle_save_all();
                    }
                }
            }

            // Save Selected button (saves current if none checked, or all checked)
            let save_icon = ICON_SAVE.codepoint;
            let save_label = if has_selected { format!("{save_icon} Save Selected") } else { format!("{save_icon} Save") };
            let save_btn = egui::Button::new(
                RichText::new(save_label).color(theme::TEXT_PRIMARY),
            )
            .fill(theme::BG_SECONDARY)
            .corner_radius(theme::BUTTON_ROUNDING);
            if ui.add_enabled(can_save_copy || has_selected, save_btn).clicked() {
                app.handle_save_selected();
            }

            // Model selector -- disabled during processing
            ui.add_enabled_ui(!is_processing, |ui| {
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
                ui.label(RichText::new("Model:").color(theme::TEXT_SECONDARY));
            });
        });
    });
}
