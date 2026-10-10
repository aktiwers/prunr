use egui::{Color32, RichText};
use egui_material_icons::icons::*;

use crate::gui::app::PrunrApp;
use crate::gui::batch_manager::ProcessButtonLabel;
use crate::gui::history_manager::HistoryManager;
use crate::gui::item::BatchStatus;
use crate::gui::state::AppState;
use crate::gui::theme;

use super::adjustments_toolbar::ModelStoreRequest;
use super::chip::{picker_row, popover_header, popup_for, tooltip, with_fill};
use super::shortcuts::{keys, Action};

/// Row-1 icon button: taller than a chip, filled like one.
fn icon_button(ui: &mut egui::Ui, icon: &'static str) -> egui::Response {
    let btn = egui::Button::new(
        RichText::new(icon).size(theme::ICON_SIZE_BUTTON).color(theme::TEXT_PRIMARY),
    )
    .corner_radius(theme::BUTTON_ROUNDING)
    .min_size(egui::vec2(theme::BTN_HEIGHT, theme::BTN_HEIGHT));
    with_fill(ui, theme::BG_SECONDARY, |ui| ui.add(btn))
}

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
        .corner_radius(theme::BUTTON_ROUNDING)
        .min_size(egui::vec2(0.0, theme::BTN_HEIGHT));
        if tooltip(with_fill(ui, theme::BG_SECONDARY, |ui| ui.add(open_btn)), "Open", "Open one or more images.", Some(Action::Open)).clicked() {
            app.pending_open_dialog = true;
        }

        if tooltip(icon_button(ui, ICON_SETTINGS.codepoint), "Settings", "Hardware, performance and behavior.", Some(Action::Settings)).clicked() {
            if app.show_settings {
                app.close_settings(ui.ctx());
            } else {
                app.open_settings();
            }
        }

        let help_resp = tooltip(
            icon_button(ui, ICON_HELP.codepoint),
            "Help",
            "Shortcuts, the command-line reference, the pipelines and the Model Store.",
            None,
        );
        let help_id = egui::Id::new("help_menu");
        popup_for(ui, help_id, &help_resp, |ui| {
            popover_header(ui, "Help", None);
            let entries = [
                ("Keyboard shortcuts", Action::Shortcuts),
                ("Command-line reference", Action::CliHelp),
                ("Pipelines", Action::PipelineFlow),
            ];
            let mut chosen = false;
            for (label, action) in entries {
                if picker_row(ui, false, label, &keys(ui.ctx(), action)).clicked() {
                    if let Some(open) = app.help_modal_mut(action) {
                        *open = true;
                    }
                    chosen = true;
                }
            }
            if picker_row(ui, false, "Model Store", "Download and manage models").clicked() {
                app.model_store = Some(ModelStoreRequest::default());
                chosen = true;
            }
            if chosen {
                egui::Popup::close_id(ui.ctx(), help_id);
            }
        });

        // Model, groups and presets live on the adjustments toolbar; this
        // row stays minimal: Open, Settings, Help, and the action cluster.

        if !app.batch.items.is_empty() {
            let can_undo = app.batch.any_target_can(HistoryManager::can_undo);
            let can_redo = app.batch.any_target_can(HistoryManager::can_redo);
            if tooltip(ui.add_enabled_ui(can_undo, |ui| icon_button(ui, ICON_UNDO.codepoint)).inner, "Undo", "", Some(Action::Undo))
                .clicked()
            {
                app.handle_undo(ui.ctx());
            }
            if tooltip(ui.add_enabled_ui(can_redo, |ui| icon_button(ui, ICON_REDO.codepoint)).inner, "Redo", "", Some(Action::Redo))
                .clicked()
            {
                app.handle_redo(ui.ctx());
            }
        }

        // ── Right group: action buttons ──
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if has_selected {
                let remove_sel_btn = egui::Button::new(
                    RichText::new(format!("{}  Remove selected", ICON_DELETE.codepoint)).color(Color32::WHITE),
                )
                .corner_radius(theme::BUTTON_ROUNDING);
                if tooltip(with_fill(ui, theme::DESTRUCTIVE, |ui| ui.add(remove_sel_btn)), "Remove selected", "Take the selected images out of the queue.", None).clicked() {
                    app.remove_selected();
                }
            }

            let has_saveable_selected = has_selected
                && app.batch.items.iter().any(|i| i.selected && i.status == BatchStatus::Done);
            let show_save = can_save_copy || has_saveable_selected;
            if show_save {
                let (save_title, save_label) = if has_selected {
                    ("Save selected", format!("{}  Save selected", ICON_SAVE.codepoint))
                } else {
                    ("Save", format!("{}  Save", ICON_SAVE.codepoint))
                };
                let save_btn = egui::Button::new(
                    RichText::new(save_label).color(theme::TEXT_PRIMARY),
                )
                .corner_radius(theme::BUTTON_ROUNDING);
                if tooltip(with_fill(ui, theme::BG_SECONDARY, |ui| ui.add(save_btn)), save_title, "Save the result as PNG.", Some(Action::Save)).clicked() {
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
                .corner_radius(theme::BUTTON_ROUNDING)
                .min_size(egui::vec2(0.0, theme::BTN_HEIGHT));
                let (title, body) = if partial {
                    ("Cancel selected", "Stop the selected images; the others keep running.")
                } else {
                    ("Cancel all", "Stop all processing.")
                };
                if tooltip(with_fill(ui, theme::DESTRUCTIVE, |ui| ui.add(cancel_btn)), title, body, Some(Action::Cancel)).clicked() {
                    if partial {
                        app.handle_cancel_selected();
                    } else {
                        app.handle_cancel_all_and_reset();
                    }
                }
            }

            // Process button — always visible; routing and gating live in
            // `handle_process_intent` / `can_process_intent`.
            {
                let inpaint_mode = app.settings.model.is_inpaint();
                let label = app.batch.process_button_label();
                let has_processable = app.can_process_intent();

                let (label_text, is_all) = if inpaint_mode {
                    ("Process".to_string(), false)
                } else {
                    match label {
                        ProcessButtonLabel::ProcessViewed => ("Process".to_string(), false),
                        ProcessButtonLabel::ProcessSelected(1) => ("Process 1 selected".to_string(), false),
                        ProcessButtonLabel::ProcessSelected(n) => (format!("Process {n} selected"), false),
                        ProcessButtonLabel::ProcessAll(n) => (format!("Process all ({n})"), true),
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
                    .corner_radius(theme::BUTTON_ROUNDING)
                    .min_size(egui::vec2(0.0, theme::BTN_HEIGHT));

                let body: std::borrow::Cow<'static, str> = if inpaint_mode {
                    if has_processable {
                        "Run the eraser again over the painted region with the current settings.".into()
                    } else {
                        "Paint a region first; Process then runs the eraser over it.".into()
                    }
                } else {
                    match label {
                        ProcessButtonLabel::ProcessAll(n) => format!("Process all {n} images.").into(),
                        ProcessButtonLabel::ProcessSelected(n) if n > 1 => {
                            format!("Process the {n} selected images.").into()
                        }
                        _ => {
                            let target_has_result = app.batch.first_target_item()
                                .is_some_and(|i| i.result_rgba.is_some());
                            if app.settings.chain_mode && target_has_result {
                                "Process the current result again.".into()
                            } else {
                                "Process the current image.".into()
                            }
                        }
                    }
                };

                if tooltip(with_fill(ui, fill, |ui| ui.add_enabled(has_processable, btn)), "Process", &body, Some(Action::Process)).clicked() {
                    app.handle_process_intent();
                }
            }

        });
    });
}
