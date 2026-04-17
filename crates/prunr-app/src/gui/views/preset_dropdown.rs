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

/// Label for the dropdown button — shows the `applied_preset` name and
/// whether current settings still match it.
///   - Match → `Preset: Portrait ✓` (check icon, clean)
///   - Diverged → `Preset: Portrait ✎` (edit icon, modified)
///
/// Tracks the preset the user LAST APPLIED (via dropdown click or Reset All),
/// not whichever preset happens to match current settings. This way
/// "Portrait ✎" keeps saying Portrait while you tweak — you know your base
/// and that you've diverged from it.
fn button_label(settings: &Settings, current: &ItemSettings, applied_preset: &str) -> String {
    // If applied_preset was deleted out from under us, fall back gracefully.
    let exists = applied_preset == PRUNR_PRESET
        || settings.presets.contains_key(applied_preset);
    if !exists {
        return format!("{}  Preset: Custom  {}", ICON_BOOKMARK.codepoint, ICON_EDIT.codepoint);
    }
    let is_modified = *current != settings.preset_values(applied_preset);
    let state_icon = if is_modified { ICON_EDIT.codepoint } else { ICON_CHECK.codepoint };
    format!("{}  Preset: {applied_preset}  {state_icon}", ICON_BOOKMARK.codepoint)
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

/// Render the Preset dropdown. Returns `Some(name)` when the user applied a
/// preset (click on a row, or Save that swaps in the new name), so the caller
/// can update `BatchItem.applied_preset`. Returns `None` for no-op frames or
/// non-apply interactions (saves, deletes, star toggles).
#[allow(deprecated)]
pub fn render(
    ui: &mut Ui,
    settings: &mut Settings,
    current_item: &mut ItemSettings,
    applied_preset: &str,
) -> Option<String> {
    let pop_id = egui::Id::new("preset_popover");
    let save_dialog_id = egui::Id::new("preset_save_dialog");

    let btn = egui::Button::new(
        RichText::new(button_label(settings, current_item, applied_preset))
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

    let mut applied: Option<String> = None;
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
                        applied = Some(name.clone());
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
            settings.presets.insert(target.clone(), *current_item);
            // Saving the current settings AS a preset makes that preset the
            // now-applied one — the label should read "Target ✓" (clean).
            applied = Some(target);
            close_dialog();
        } else if commit
            && !trimmed.is_empty()
            && !trimmed.eq_ignore_ascii_case(PRUNR_PRESET)
        {
            let name = trimmed.to_string();
            settings.presets.insert(name.clone(), *current_item);
            applied = Some(name);
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
    fn button_label_clean_icon_when_settings_match_applied() {
        let s = Settings::default();
        let current = ItemSettings::default();
        let label = button_label(&s, &current, PRUNR_PRESET);
        assert!(label.contains("Prunr"), "{label}");
        assert!(label.ends_with(ICON_CHECK.codepoint), "{label}");
    }

    #[test]
    fn button_label_modified_icon_when_diverged_from_applied() {
        let s = Settings::default();
        let mut current = ItemSettings::default();
        current.gamma = 1.7;
        let label = button_label(&s, &current, PRUNR_PRESET);
        assert!(label.contains("Prunr"), "{label}");
        assert!(label.ends_with(ICON_EDIT.codepoint), "{label}");
    }

    #[test]
    fn button_label_falls_back_to_custom_when_applied_was_deleted() {
        let s = Settings::default();
        let current = ItemSettings::default();
        // "Portrait" is NOT in presets, not Prunr — applied_preset dangles.
        let label = button_label(&s, &current, "Portrait");
        assert!(label.contains("Custom"), "{label}");
        assert!(label.ends_with(ICON_EDIT.codepoint), "{label}");
    }

    #[test]
    fn button_label_tracks_applied_even_when_current_matches_different_preset() {
        let mut s = Settings::default();
        let mut portrait = ItemSettings::default();
        portrait.gamma = 2.0;
        s.presets.insert("Portrait".to_string(), portrait);
        // applied_preset = "Prunr" but current happens to match factory.
        // Label should stay on "Prunr" (the applied one).
        let current = ItemSettings::default();
        let label = button_label(&s, &current, PRUNR_PRESET);
        assert!(label.contains("Prunr"), "{label}");
        // …and when current matches the OTHER preset but applied is Prunr,
        // we still show "Prunr ✎" (modified relative to the applied base).
        let label = button_label(&s, &portrait, PRUNR_PRESET);
        assert!(label.contains("Prunr"), "{label}");
        assert!(label.ends_with(ICON_EDIT.codepoint), "{label}");
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
