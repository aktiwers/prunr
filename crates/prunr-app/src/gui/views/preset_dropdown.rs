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
use crate::gui::settings::{PRUNR_PRESET, Settings};
use crate::gui::theme;

const BTN_HEIGHT: f32 = 32.0;
const POPOVER_WIDTH: f32 = 260.0;

/// Label for the dropdown button — reflects the CURRENT item's effective preset:
///   - Matches "Prunr" (factory defaults) → "Prunr"
///   - Matches a saved preset exactly → that preset's name
///   - Otherwise → "Custom" (user has diverged from the last-applied preset)
fn button_label(settings: &Settings, current: &ItemSettings) -> String {
    let name = active_preset_name(settings, current);
    format!("{}  Preset: {name}", ICON_BOOKMARK.codepoint)
}

/// Determine which preset matches the current item's settings. Checks Prunr
/// first (factory defaults), then the saved preset map. Returns "Custom" on
/// no match — the signal that the user has diverged from any saved state.
fn active_preset_name<'a>(settings: &'a Settings, current: &ItemSettings) -> &'a str {
    if *current == ItemSettings::default() {
        return PRUNR_PRESET;
    }
    for (name, values) in &settings.presets {
        if *values == *current {
            return name.as_str();
        }
    }
    "Custom"
}

/// Sort USER preset names case-insensitively. The synthetic "Prunr" preset
/// is NOT included — callers prepend it manually since it always appears
/// first in the dropdown regardless of alphabetical order.
pub(super) fn sorted_preset_names(settings: &Settings) -> Vec<String> {
    let mut names: Vec<String> = settings.presets.keys().cloned().collect();
    names.sort_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()));
    names
}

/// All preset names in display order: Prunr first, then user presets
/// alphabetically. Used by the dropdown list and anywhere we need the
/// complete set of valid `default_preset` values.
pub(super) fn all_preset_names(settings: &Settings) -> Vec<String> {
    let mut names = Vec::with_capacity(settings.presets.len() + 1);
    names.push(PRUNR_PRESET.to_string());
    names.extend(sorted_preset_names(settings));
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
        RichText::new(button_label(settings, current_item))
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

            // List in display order: Prunr first, then user presets. Prunr is
            // a synthetic entry — applies ItemSettings::default() and cannot be
            // deleted or overwritten.
            for name in all_preset_names(settings) {
                let is_prunr = name == PRUNR_PRESET;
                let is_default = settings.default_preset == name;
                ui.horizontal(|ui| {
                    let label_text = RichText::new(&name)
                        .color(theme::TEXT_PRIMARY)
                        .size(theme::FONT_SIZE_BODY);
                    if ui.selectable_label(false, label_text)
                        .on_hover_text("Click to apply these settings to the current image")
                        .clicked()
                    {
                        *current_item = settings.preset_values(&name);
                        applied = true;
                    }
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            // Delete — hidden for Prunr (non-deletable).
                            if !is_prunr {
                                let delete_btn = ui.small_button(
                                    RichText::new(ICON_DELETE.codepoint)
                                        .size(theme::FONT_SIZE_MONO)
                                        .color(theme::DESTRUCTIVE),
                                );
                                if delete_btn.on_hover_text("Delete preset").clicked() {
                                    settings.presets.remove(&name);
                                    // If the deleted preset was the app's default,
                                    // fall back to Prunr so default_preset stays valid.
                                    if is_default {
                                        settings.default_preset = PRUNR_PRESET.to_string();
                                    }
                                }
                            }
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
                                "Default for new images (click another preset's star to switch)"
                            } else {
                                "Set as default for new images"
                            };
                            let star_btn = ui.small_button(
                                RichText::new(star_icon)
                                    .size(theme::FONT_SIZE_MONO)
                                    .color(star_color),
                            );
                            if star_btn.on_hover_text(star_tooltip).clicked() && !is_default {
                                settings.default_preset = name.clone();
                            }
                        },
                    );
                });
            }

            ui.add_space(theme::SPACE_SM);
            ui.separator();
            ui.add_space(theme::SPACE_SM);

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

            ui.add_space(theme::SPACE_XS);
            ui.label(
                RichText::new(format!("Default for new images: {}", settings.default_preset))
                    .color(theme::TEXT_HINT)
                    .size(theme::FONT_SIZE_MONO),
            );
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
                let trimmed = name_buf.trim();
                let is_prunr_name = trimmed.eq_ignore_ascii_case(PRUNR_PRESET);
                if is_prunr_name {
                    ui.label(
                        RichText::new("\"Prunr\" is reserved — pick another name.")
                            .color(theme::DESTRUCTIVE)
                            .size(theme::FONT_SIZE_MONO),
                    );
                } else if !trimmed.is_empty() && existing_names.iter().any(|n| n == trimmed) {
                    ui.label(
                        RichText::new(format!("⚠ Will overwrite \"{trimmed}\""))
                            .color(theme::DESTRUCTIVE)
                            .size(theme::FONT_SIZE_MONO),
                    );
                }

                ui.add_space(theme::SPACE_SM);
                ui.horizontal(|ui| {
                    if ui.add_enabled(!is_prunr_name, egui::Button::new("Save")).clicked() {
                        commit = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });

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

        let trimmed = name_buf.trim();
        if let Some(target) = overwrite_target {
            settings.presets.insert(target, *current_item);
            close_dialog();
        } else if commit
            && !trimmed.is_empty()
            && !trimmed.eq_ignore_ascii_case(PRUNR_PRESET)
        {
            settings.presets.insert(trimmed.to_string(), *current_item);
            close_dialog();
        } else if cancel {
            close_dialog();
        } else {
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
    fn button_label_shows_prunr_for_factory_defaults() {
        let s = Settings::default();
        let current = ItemSettings::default();
        assert_eq!(
            button_label(&s, &current),
            format!("{}  Preset: Prunr", ICON_BOOKMARK.codepoint),
        );
    }

    #[test]
    fn button_label_shows_matching_preset_name() {
        let mut s = Settings::default();
        let mut values = ItemSettings::default();
        values.gamma = 2.0;
        s.presets.insert("Portrait".to_string(), values);
        // Current item matches "Portrait" exactly → label tracks it.
        let current = values;
        assert_eq!(
            button_label(&s, &current),
            format!("{}  Preset: Portrait", ICON_BOOKMARK.codepoint),
        );
    }

    #[test]
    fn button_label_shows_custom_when_no_match() {
        let mut s = Settings::default();
        s.presets.insert("Portrait".to_string(), {
            let mut v = ItemSettings::default();
            v.gamma = 2.0;
            v
        });
        // Current item matches neither factory nor any saved preset.
        let mut current = ItemSettings::default();
        current.gamma = 1.7;
        assert_eq!(
            button_label(&s, &current),
            format!("{}  Preset: Custom", ICON_BOOKMARK.codepoint),
        );
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

    #[test]
    fn all_preset_names_prunr_first() {
        let mut s = Settings::default();
        s.presets.insert("Zeta".to_string(), ItemSettings::default());
        s.presets.insert("alpha".to_string(), ItemSettings::default());
        let names = all_preset_names(&s);
        assert_eq!(names[0], PRUNR_PRESET);
        assert_eq!(names[1..], vec!["alpha".to_string(), "Zeta".to_string()]);
    }
}
