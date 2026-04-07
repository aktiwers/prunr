use egui::{Color32, RichText};

use bgprunr_core::ModelKind;

use crate::gui::app::{BgPrunrApp, BatchStatus};
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
            app.handle_open_dialog();
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
            Color32::from_rgba_unmultiplied(
                theme::ACCENT.r(),
                theme::ACCENT.g(),
                theme::ACCENT.b(),
                102, // ~40% opacity
            )
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
                Color32::from_rgba_unmultiplied(theme::ACCENT.r(), theme::ACCENT.g(), theme::ACCENT.b(), 102)
            };
            let process_all_btn = egui::Button::new(
                RichText::new("Process All").color(Color32::WHITE),
            ).fill(process_all_fill).corner_radius(theme::BUTTON_ROUNDING);

            if ui.add_enabled(process_all_enabled, process_all_btn).clicked() {
                app.handle_process_all();
            }
        }

        // Spacer to push right-side items to the right
        let right_items_width = {
            // Approximate width: Save PNG + Copy Image + Model label + ComboBox
            // Use remaining space minus estimated right side width
            let available = ui.available_width();
            // Right side: "Save PNG" (~70) + "Copy Image" (~80) + "Model:" (~50) + ComboBox (~140) + spacing
            let right_estimate = 350.0_f32;
            (available - right_estimate).max(0.0)
        };
        ui.add_space(right_items_width);

        // Model selector -- disabled during processing
        ui.add_enabled_ui(!is_processing, |ui| {
            ui.label(RichText::new("Model:").color(theme::TEXT_SECONDARY));
            let model_text = match app.selected_model {
                ModelKind::Silueta => "silueta (fast)",
                ModelKind::U2net => "u2net (quality)",
            };
            egui::ComboBox::from_id_salt("model")
                .selected_text(model_text)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut app.selected_model,
                        ModelKind::Silueta,
                        "silueta (fast)",
                    );
                    ui.selectable_value(
                        &mut app.selected_model,
                        ModelKind::U2net,
                        "u2net (quality)",
                    );
                });
        });

        // Save PNG button -- enabled only when Done
        let save_btn = egui::Button::new(
            RichText::new("Save PNG").color(theme::TEXT_PRIMARY),
        )
        .fill(theme::BG_SECONDARY)
        .corner_radius(theme::BUTTON_ROUNDING);
        if ui.add_enabled(can_save_copy, save_btn).clicked() {
            app.handle_save();
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
    });
}
