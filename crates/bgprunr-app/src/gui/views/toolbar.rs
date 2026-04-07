use egui::{Color32, RichText};

use crate::gui::app::{BgPrunrApp, BatchStatus};
use crate::gui::settings::SettingsModel;
use crate::gui::state::AppState;
use crate::gui::theme;

pub fn render(ui: &mut egui::Ui, app: &mut BgPrunrApp) {
    ui.horizontal_centered(|ui| {
        ui.spacing_mut().item_spacing.x = theme::SPACE_SM;

        let is_processing = app.state == AppState::Processing;
        let can_remove = matches!(app.state, AppState::Loaded | AppState::Done);
        let can_save_copy = app.state == AppState::Done;

        // Open Image button -- always enabled
        let open_btn = egui::Button::new(
            RichText::new("Open Image").color(theme::TEXT_PRIMARY),
        )
        .fill(theme::BG_SECONDARY)
        .corner_radius(theme::BUTTON_ROUNDING);

        if ui.add(open_btn).clicked() {
            app.pending_open_dialog = true;
        }

        // Remove BG button -- primary CTA, enabled only when Loaded or Done
        let remove_text = if is_processing {
            RichText::new("Remove BG").color(Color32::from_rgba_unmultiplied(255, 255, 255, 102))
        } else {
            RichText::new("Remove BG").color(Color32::WHITE)
        };
        let remove_fill = if can_remove {
            theme::ACCENT
        } else {
            theme::ACCENT_DISABLED
        };
        let remove_btn = egui::Button::new(remove_text)
            .fill(remove_fill)
            .corner_radius(theme::BUTTON_ROUNDING);

        if ui.add_enabled(can_remove, remove_btn).clicked() {
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
            // Settings gear button
            let gear_btn = egui::Button::new(
                RichText::new("\u{2699}").size(18.0).color(theme::TEXT_PRIMARY),
            )
            .fill(theme::BG_SECONDARY)
            .corner_radius(theme::BUTTON_ROUNDING);
            if ui.add(gear_btn).on_hover_text("Settings (Ctrl+,)").clicked() {
                app.show_settings = !app.show_settings;
            }
            // Save All button — only shown when batch has done items
            {
                let has_done_batch = app.batch_items.iter()
                    .filter(|i| i.status == BatchStatus::Done && i.result_rgba.is_some())
                    .count() >= 2;
                if has_done_batch {
                    let save_all_btn = egui::Button::new(
                        RichText::new("Save All").color(Color32::WHITE),
                    )
                    .fill(theme::ACCENT)
                    .corner_radius(theme::BUTTON_ROUNDING);
                    if ui.add(save_all_btn).clicked() {
                        app.handle_save_all();
                    }
                }
            }

            // Copy Image button -- enabled only when Done
            let copy_btn = egui::Button::new(
                RichText::new("Copy Image").color(theme::TEXT_PRIMARY),
            )
            .fill(theme::BG_SECONDARY)
            .corner_radius(theme::BUTTON_ROUNDING);
            if ui.add_enabled(can_save_copy, copy_btn).clicked() {
                app.handle_copy();
            }

            // Save PNG button -- enabled only when Done
            let save_btn = egui::Button::new(
                RichText::new("Save PNG").color(theme::TEXT_PRIMARY),
            )
            .fill(theme::BG_SECONDARY)
            .corner_radius(theme::BUTTON_ROUNDING);
            if ui.add_enabled(can_save_copy, save_btn).clicked() {
                app.handle_save();
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
