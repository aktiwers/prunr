//! Row 1 "Preset" dropdown: quick-apply named settings snapshots and save
//! the current item's settings as a new preset.
//!
//! View Component discipline: takes `&mut AppSettings` (for the preset map
//! and the naming dialog state) + `&mut ItemSettings` (the current image's
//! settings — source of a Save, target of an Apply).
//!
//! Returns `true` when the current item's settings changed (Apply), so the
//! caller can invalidate textures / reset cached tensors.
//!
//! The "Save preset" naming dialog lives on `AppSettings.preset_name_buffer`
//! (a transient field, `#[serde(skip)]`). It opens on "Save current as..." and
//! closes when the user confirms or cancels.

use egui::{RichText, Ui};
use egui_material_icons::icons::*;

use crate::gui::item_settings::ItemSettings;
use crate::gui::settings::Settings;
use crate::gui::theme;

const BTN_HEIGHT: f32 = 32.0;
const POPOVER_WIDTH: f32 = 260.0;

/// Label for the dropdown button.
fn button_label(settings: &Settings) -> String {
    let name = settings
        .default_preset
        .as_deref()
        .unwrap_or("Default");
    format!("{}  Preset: {}", ICON_BOOKMARK.codepoint, name)
}

/// Sort preset names case-insensitively so the list is stable.
fn sorted_preset_names(settings: &Settings) -> Vec<String> {
    let mut names: Vec<String> = settings.presets.keys().cloned().collect();
    names.sort_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()));
    names
}

/// Render the Preset dropdown. Returns `true` when the current item's
/// settings were modified by an Apply action.
#[allow(deprecated)]
pub fn render(
    ui: &mut Ui,
    settings: &mut Settings,
    current_item: &mut ItemSettings,
) -> bool {
    let pop_id = egui::Id::new("preset_popover");
    let save_dialog_id = egui::Id::new("preset_save_dialog");

    let btn = egui::Button::new(
        RichText::new(button_label(settings))
            .color(theme::TEXT_PRIMARY)
            .size(theme::FONT_SIZE_BODY),
    )
    .fill(theme::BG_SECONDARY)
    .corner_radius(theme::BUTTON_ROUNDING)
    .min_size(egui::vec2(0.0, BTN_HEIGHT));
    let resp = ui.add(btn).on_hover_text("Apply or save preset for the current image");

    if resp.clicked() {
        ui.memory_mut(|m| m.toggle_popup(pop_id));
    }

    let mut applied = false;
    egui::popup_below_widget(
        ui,
        pop_id,
        &resp,
        egui::PopupCloseBehavior::CloseOnClickOutside,
        |ui| {
            ui.set_min_width(POPOVER_WIDTH);
            ui.label(RichText::new("Presets").strong().color(theme::TEXT_PRIMARY));
            ui.add_space(theme::SPACE_XS);

            let names = sorted_preset_names(settings);
            if names.is_empty() {
                ui.label(
                    RichText::new("No presets saved yet.")
                        .color(theme::TEXT_HINT)
                        .size(theme::FONT_SIZE_MONO),
                );
            } else {
                for name in &names {
                    let is_default = settings.default_preset.as_deref() == Some(name.as_str());
                    ui.horizontal(|ui| {
                        let label_text = RichText::new(name)
                            .color(theme::TEXT_PRIMARY)
                            .size(theme::FONT_SIZE_BODY);
                        if ui.selectable_label(false, label_text)
                            .on_hover_text("Click to apply these settings to the current image")
                            .clicked()
                        {
                            if let Some(values) = settings.presets.get(name) {
                                *current_item = *values;
                                applied = true;
                            }
                        }
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                // Right-to-left: delete (rightmost), then default-toggle.
                                let delete_btn = ui.small_button(
                                    RichText::new(ICON_DELETE.codepoint)
                                        .size(theme::FONT_SIZE_MONO)
                                        .color(theme::DESTRUCTIVE),
                                );
                                if delete_btn.on_hover_text("Delete preset").clicked() {
                                    settings.presets.remove(name);
                                    if is_default {
                                        settings.default_preset = None;
                                    }
                                }
                                // Star icon toggles "is default preset for new images."
                                // Filled when default, outlined otherwise.
                                let star_icon = if is_default {
                                    ICON_STAR.codepoint
                                } else {
                                    ICON_STAR_OUTLINE.codepoint
                                };
                                let star_color = if is_default {
                                    theme::ACCENT
                                } else {
                                    theme::TEXT_SECONDARY
                                };
                                let star_tooltip = if is_default {
                                    "Default for new images — click to clear"
                                } else {
                                    "Set as default for new images"
                                };
                                let star_btn = ui.small_button(
                                    RichText::new(star_icon)
                                        .size(theme::FONT_SIZE_MONO)
                                        .color(star_color),
                                );
                                if star_btn.on_hover_text(star_tooltip).clicked() {
                                    if is_default {
                                        settings.default_preset = None;
                                    } else {
                                        settings.default_preset = Some(name.clone());
                                    }
                                }
                            },
                        );
                    });
                }
            }

            ui.add_space(theme::SPACE_SM);
            ui.separator();
            ui.add_space(theme::SPACE_SM);

            // Save current as… → opens dialog
            let save_btn = egui::Button::new(
                RichText::new(format!(
                    "{}  Save current as…",
                    ICON_BOOKMARK_ADD.codepoint
                ))
                .color(theme::TEXT_PRIMARY)
                .size(theme::FONT_SIZE_BODY),
            )
            .fill(theme::BG_SECONDARY);
            if ui.add(save_btn).clicked() {
                ui.memory_mut(|m| m.data.insert_temp::<String>(save_dialog_id, String::new()));
                ui.memory_mut(|m| {
                    m.data.insert_temp::<bool>(save_dialog_id.with("open"), true);
                });
            }

            // Set-as-default toggle for currently-applied preset (if any)
            if let Some(ref name) = settings.default_preset.clone() {
                ui.add_space(theme::SPACE_XS);
                ui.label(
                    RichText::new(format!("Default for new images: {name}"))
                        .color(theme::TEXT_HINT)
                        .size(theme::FONT_SIZE_MONO),
                );
                if ui.small_button("Clear default").clicked() {
                    settings.default_preset = None;
                }
            }
        },
    );

    // ── Naming dialog ──
    let dialog_open = ui
        .ctx()
        .memory(|m| m.data.get_temp::<bool>(save_dialog_id.with("open")).unwrap_or(false));
    if dialog_open {
        let mut name_buf = ui
            .ctx()
            .memory(|m| m.data.get_temp::<String>(save_dialog_id).unwrap_or_default());
        let mut commit = false;
        let mut cancel = false;
        let mut overwrite_target: Option<String> = None;

        // Snapshot existing preset names so we can offer overwrite targets
        // without holding a borrow on `settings.presets` across the dialog.
        let existing_names = sorted_preset_names(settings);

        egui::Window::new("Save preset")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ui.ctx(), |ui| {
                ui.label("Name for this preset:");
                let text_resp = ui.text_edit_singleline(&mut name_buf);
                if text_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    commit = true;
                }
                // Warn if the typed name would overwrite an existing preset.
                let trimmed = name_buf.trim();
                if !trimmed.is_empty() && existing_names.iter().any(|n| n == trimmed) {
                    ui.label(
                        RichText::new(format!("⚠ Will overwrite \"{trimmed}\""))
                            .color(theme::DESTRUCTIVE)
                            .size(theme::FONT_SIZE_MONO),
                    );
                }

                ui.add_space(theme::SPACE_SM);
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        commit = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });

                // Show existing presets as one-click overwrite targets. Clicking
                // one saves the CURRENT item.settings under that existing name.
                if !existing_names.is_empty() {
                    ui.add_space(theme::SPACE_SM);
                    ui.separator();
                    ui.add_space(theme::SPACE_XS);
                    ui.label(
                        RichText::new("Or overwrite an existing preset:")
                            .color(theme::TEXT_HINT)
                            .size(theme::FONT_SIZE_MONO),
                    );
                    ui.add_space(theme::SPACE_XS);
                    for name in &existing_names {
                        let btn = ui.button(
                            RichText::new(format!("{}  {name}", ICON_BOOKMARK.codepoint))
                                .color(theme::TEXT_PRIMARY)
                                .size(theme::FONT_SIZE_BODY),
                        );
                        if btn.on_hover_text("Overwrite with current settings").clicked() {
                            overwrite_target = Some(name.clone());
                        }
                    }
                }
            });

        let close_dialog = || {
            ui.ctx().memory_mut(|m| {
                m.data.insert_temp::<bool>(save_dialog_id.with("open"), false);
                m.data.remove::<String>(save_dialog_id);
            });
        };

        if let Some(target) = overwrite_target {
            settings.presets.insert(target, *current_item);
            close_dialog();
        } else if commit && !name_buf.trim().is_empty() {
            settings.presets.insert(name_buf.trim().to_string(), *current_item);
            close_dialog();
        } else if cancel {
            close_dialog();
        } else {
            // Persist typed text across frames while the dialog is open.
            ui.ctx()
                .memory_mut(|m| m.data.insert_temp::<String>(save_dialog_id, name_buf));
        }
    }

    applied
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn button_label_shows_default_when_none() {
        let s = Settings::default();
        assert_eq!(button_label(&s), format!("{}  Preset: Default", ICON_BOOKMARK.codepoint));
    }

    #[test]
    fn button_label_shows_named_preset() {
        let mut s = Settings::default();
        s.default_preset = Some("Portrait".to_string());
        assert_eq!(button_label(&s), format!("{}  Preset: Portrait", ICON_BOOKMARK.codepoint));
    }

    #[test]
    fn sorted_preset_names_case_insensitive() {
        let mut s = Settings::default();
        let mut map: HashMap<String, ItemSettings> = HashMap::new();
        map.insert("Zeta".to_string(), ItemSettings::default());
        map.insert("alpha".to_string(), ItemSettings::default());
        map.insert("Beta".to_string(), ItemSettings::default());
        s.presets = map;
        assert_eq!(sorted_preset_names(&s), vec!["alpha", "Beta", "Zeta"]);
    }
}
