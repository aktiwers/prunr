use egui::{Color32, RichText};
use egui_material_icons::icons::*;

use crate::gui::app::{PrunrApp, BatchStatus};
use crate::gui::settings::SettingsModel;
use crate::gui::state::AppState;
use crate::gui::theme;

use super::{model_label, model_name, modifier_key};

const BTN_HEIGHT: f32 = 32.0;

pub fn render(ui: &mut egui::Ui, app: &mut PrunrApp) {
    ui.horizontal_centered(|ui| {
        ui.spacing_mut().item_spacing.x = theme::SPACE_SM;
        ui.spacing_mut().button_padding = egui::vec2(8.0, 4.0);

        let is_processing = app.state == AppState::Processing;
        let can_save_copy = app.state == AppState::Done;
        let has_selected = app.batch_items.iter().any(|i| i.selected);
        let m = modifier_key();

        // ── Left: Open ──
        let open_btn = egui::Button::new(
            RichText::new(format!("{}  Open", ICON_FOLDER_OPEN.codepoint)).color(theme::TEXT_PRIMARY),
        )
        .fill(theme::BG_SECONDARY)
        .corner_radius(theme::BUTTON_ROUNDING)
        .min_size(egui::vec2(0.0, BTN_HEIGHT));
        if ui.add(open_btn).on_hover_text(format!("Open image(s) ({m}+O)")).clicked() {
            app.pending_open_dialog = true;
        }

        // ── Settings gear + Model dropdown ──

        let gear_btn = egui::Button::new(
            RichText::new(ICON_SETTINGS.codepoint)
                .size(20.0)
                .color(theme::TEXT_PRIMARY),
        )
        .fill(theme::BG_SECONDARY)
        .corner_radius(theme::BUTTON_ROUNDING)
        .min_size(egui::vec2(BTN_HEIGHT, BTN_HEIGHT));
        if ui.add(gear_btn).on_hover_text(format!("Settings ({m}+Space)")).clicked() {
            if app.show_settings {
                app.close_settings(ui.ctx());
            } else {
                app.show_settings = true;
                app.settings_opened_at = ui.ctx().input(|i| i.time);
                app.bg_settings_snapshot = (app.settings.apply_bg_color, app.settings.bg_color);
            }
        }

        let prev_model = app.settings.model;
        ui.add_enabled_ui(!is_processing, |ui| {
            {
                // Force light text on all ComboBox states — prevents dark-on-dark
                // when the OS is in light mode (egui may inherit OS text color).
                let vis = ui.visuals_mut();
                vis.widgets.inactive.weak_bg_fill = theme::BG_SECONDARY;
                vis.widgets.inactive.fg_stroke.color = theme::TEXT_PRIMARY;
                vis.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(0x30, 0x2e, 0x32);
                vis.widgets.hovered.fg_stroke.color = theme::TEXT_PRIMARY;
                vis.widgets.open.weak_bg_fill = theme::BG_SECONDARY;
                vis.widgets.open.fg_stroke.color = theme::TEXT_PRIMARY;
                vis.widgets.active.fg_stroke.color = theme::TEXT_PRIMARY;
                vis.widgets.noninteractive.fg_stroke.color = theme::TEXT_SECONDARY;
                vis.window_fill = theme::BG_PRIMARY;
            }
            ui.spacing_mut().interact_size.y = BTN_HEIGHT;
            egui::ComboBox::from_id_salt("model")
                .selected_text(RichText::new(model_label(app.settings.model, true)).color(theme::TEXT_PRIMARY))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut app.settings.model,
                        SettingsModel::Silueta,
                        RichText::new(model_label(SettingsModel::Silueta, false)).color(theme::TEXT_PRIMARY),
                    );
                    ui.selectable_value(
                        &mut app.settings.model,
                        SettingsModel::U2net,
                        RichText::new(model_label(SettingsModel::U2net, false)).color(theme::TEXT_PRIMARY),
                    );
                    ui.selectable_value(
                        &mut app.settings.model,
                        SettingsModel::BiRefNetLite,
                        RichText::new(model_label(SettingsModel::BiRefNetLite, false)).color(theme::TEXT_PRIMARY),
                    );
                });
        });

        if app.settings.model != prev_model {
            // Clamp parallel jobs to safe max for the new model
            let max = app.settings.max_jobs();
            if app.settings.parallel_jobs > max {
                app.settings.parallel_jobs = max;
            }
            app.toasts.info(format!("{} loaded", model_name(app.settings.model)));
            app.settings.save();
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
                && app.batch_items.iter().any(|i| i.selected && i.status == BatchStatus::Done);
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

            if app.batch_items.len() >= 2 {
                let is_batch_processing = app.batch_items.iter().any(|i| i.status == BatchStatus::Processing);

                if is_batch_processing {
                    // Show Cancel All while processing
                    let cancel_btn = egui::Button::new(
                        RichText::new(format!("{}  Cancel All", ICON_CANCEL.codepoint)).color(Color32::WHITE),
                    )
                    .fill(theme::DESTRUCTIVE)
                    .corner_radius(theme::BUTTON_ROUNDING)
                    .min_size(egui::vec2(0.0, BTN_HEIGHT));
                    if ui.add(cancel_btn).on_hover_text("Cancel all processing (Escape)").clicked() {
                        app.handle_cancel();
                        for item in &mut app.batch_items {
                            if item.status == BatchStatus::Processing {
                                item.status = BatchStatus::Pending;
                            }
                        }
                        app.state = AppState::Loaded;
                        app.status.text = "Cancelled".to_string();
                    }
                } else {
                    let has_pending = app.batch_items.iter().any(|i| i.status == BatchStatus::Pending);
                    let fill = if has_pending { theme::ACCENT } else { theme::ACCENT_DISABLED };
                    let text_color = if has_pending {
                        Color32::WHITE
                    } else {
                        Color32::from_rgba_unmultiplied(255, 255, 255, 102)
                    };
                    let batch_icon = egui::Image::new(egui::include_image!("../../../../../img/batch-icon.png"))
                        .fit_to_exact_size(egui::vec2(22.0, 22.0));
                    let btn = egui::Button::image_and_text(batch_icon, RichText::new("Process All").color(text_color))
                        .fill(fill)
                        .corner_radius(theme::BUTTON_ROUNDING)
                        .min_size(egui::vec2(0.0, BTN_HEIGHT));
                    if ui.add_enabled(has_pending, btn).on_hover_text("Process all pending images").clicked() {
                        app.handle_process_all();
                    }
                }
            }

            let has_processable = if has_selected {
                app.batch_items.iter().any(|i| i.selected && !matches!(i.status, BatchStatus::Processing))
            } else {
                app.selected_item()
                    .map_or(app.state == AppState::Loaded, |item| !matches!(item.status, BatchStatus::Processing))
            };
            let remove_label = if has_selected { "Process Selected" } else { "Process" };
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
                .corner_radius(theme::BUTTON_ROUNDING)
                .min_size(egui::vec2(0.0, BTN_HEIGHT));
            let process_tooltip = if app.settings.chain_mode
                && app.selected_item().map_or(false, |i| i.result_rgba.is_some())
            {
                format!("Process current result ({m}+R)")
            } else {
                format!("Process original ({m}+R)")
            };
            if ui.add_enabled(has_processable && !is_processing, remove_btn)
                .on_hover_text(process_tooltip)
                .clicked()
            {
                app.handle_remove_bg();
            }
        });
    });
}
