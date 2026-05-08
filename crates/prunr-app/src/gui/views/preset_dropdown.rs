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

/// Label for the dropdown button — shows the `applied_preset` name and
/// whether current settings still match it.
///   - Match → `🔖 Portrait ✓` (check icon, clean)
///   - Diverged → `🔖 Portrait ✎` (edit icon, modified)
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
        return format!("{}  Custom  {}", ICON_BOOKMARK.codepoint, ICON_EDIT.codepoint);
    }
    let resolved = resolve_preset_view(settings, applied_preset);
    let item_diverged = *current != resolved.item_settings;
    let brush_diverged = settings.brush != resolved.brush;
    let is_modified = item_diverged || brush_diverged;
    let state_icon = if is_modified { ICON_EDIT.codepoint } else { ICON_CHECK.codepoint };
    format!("{}  {applied_preset}  {state_icon}", ICON_BOOKMARK.codepoint)
}

/// Resolve an arbitrary preset name to a `ResolvedView` using the
/// active model. The dirty indicator may check a preset other than
/// the active one, so the active-preset shortcut doesn't apply here.
fn resolve_preset_view(
    settings: &Settings,
    name: &str,
) -> crate::gui::presets::ResolvedView {
    use crate::gui::presets;
    let empty = presets::PresetFile::default();
    let file = if name == PRUNR_PRESET {
        &empty
    } else {
        settings.presets.get(name).unwrap_or(&empty)
    };
    let model_id = settings.model.to_model_id()
        .or_else(|| Settings::default().model.to_model_id())
        .expect("Settings::default().model always has a model_id");
    presets::resolve_preset_for_model(file, model_id, None)
}

/// Sort USER preset names case-insensitively. The synthetic "Prunr" preset
/// is NOT included — callers prepend it manually since it always appears
/// first in the dropdown regardless of alphabetical order.
pub(super) fn sorted_preset_names(settings: &Settings) -> Vec<String> {
    let mut names: Vec<String> = settings.presets.keys().cloned().collect();
    names.sort_by_key(|a| a.to_lowercase());
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
    .min_size(egui::vec2(0.0, theme::BTN_HEIGHT));
    let applied_label = applied_preset.to_string();
    let resp = ui.add(btn).on_hover_ui(|ui| {
        ui.label(RichText::new("Preset").strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        ui.label(
            RichText::new(format!("Apply or save a preset. Currently applied: {applied_label}"))
                .color(theme::TEXT_PRIMARY)
                .size(theme::FONT_SIZE_MONO),
        );
    });

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
            ui.set_min_width(theme::POPOVER_WIDTH);
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
                                    // Remove the on-disk file too; presets are
                                    // the filesystem-store's source of truth.
                                    let _ = crate::gui::presets_fs::delete(&name);
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
                            .color(theme::TEXT_PRIMARY)
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
        let is_overwrite = overwrite_target.is_some();
        // Resolve the target name: overwrite button wins, else a valid typed
        // name on commit. Saving the current settings AS a preset makes that
        // preset the now-applied one — label reads "Name ✓" after.
        let target_name = overwrite_target.or_else(|| {
            (commit && !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case(PRUNR_PRESET))
                .then(|| trimmed.to_string())
        });
        if let Some(name) = target_name {
            // Falls back to the workspace default model in filter-only
            // mode (no model_id) so the file is still keyed correctly.
            let model_id = settings.model.to_model_id()
                .or_else(|| Settings::default().model.to_model_id())
                .expect("Settings::default().model always has a model_id");
            let (brush, sd) = crate::gui::presets::split_brush_for_save(
                &settings.brush,
                model_id,
            );
            let mp = crate::gui::presets::ModelPreset {
                item_settings: *current_item,
                brush,
                sd,
            };

            let result = if is_overwrite {
                // Merge into existing on disk so other-model entries +
                // other-scheduler bundles stay intact.
                crate::gui::presets_fs::save_merged(&name, model_id, mp)
            } else {
                let mut models = std::collections::HashMap::new();
                models.insert(crate::gui::presets::model_id_key(model_id), mp);
                let file = crate::gui::presets::PresetFile {
                    format_version: crate::gui::presets::PRESET_FORMAT_VERSION,
                    models,
                };
                crate::gui::presets_fs::save(&name, &file)
            };

            if let Err(e) = result {
                tracing::error!(preset = %name, %e, "failed to save preset to disk");
            }

            // Refresh just the saved entry from disk so dirty-tracking
            // round-trips against the merged file. Avoids the O(n)
            // directory rescan that load_all() would do.
            if let Some(path) = crate::gui::presets_fs::preset_path(&name) {
                if let Some(reloaded) = crate::gui::presets_fs::load_from_path(&path) {
                    settings.presets.insert(name.clone(), reloaded);
                }
            }

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
    use crate::gui::item_settings::item_with_gamma;
    use crate::gui::presets::{model_id_key, ModelPreset, PresetFile, PRESET_FORMAT_VERSION};
    use std::collections::HashMap;

    /// Wrap an `ItemSettings` into a v2 single-entry `PresetFile` keyed by
    /// the active model — tests use this where `s.presets.insert` once
    /// took a bare `ItemSettings`.
    fn wrap(s: &Settings, item: ItemSettings) -> PresetFile {
        let mid = s.model.to_model_id().expect("test settings have a model_id");
        let mut models = HashMap::new();
        models.insert(model_id_key(mid), ModelPreset {
            item_settings: item,
            brush: Default::default(),
            sd: None,
        });
        PresetFile { format_version: PRESET_FORMAT_VERSION, models }
    }

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
        let current = item_with_gamma(1.7);
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
        let portrait_item = item_with_gamma(2.0);
        s.presets.insert("Portrait".to_string(), wrap(&s, portrait_item));
        // applied_preset = "Prunr" but current happens to match factory.
        // Label should stay on "Prunr" (the applied one).
        let current = ItemSettings::default();
        let label = button_label(&s, &current, PRUNR_PRESET);
        assert!(label.contains("Prunr"), "{label}");
        // …and when current matches the OTHER preset but applied is Prunr,
        // we still show "Prunr ✎" (modified relative to the applied base).
        let label = button_label(&s, &portrait_item, PRUNR_PRESET);
        assert!(label.contains("Prunr"), "{label}");
        assert!(label.ends_with(ICON_EDIT.codepoint), "{label}");
    }

    #[test]
    fn button_label_dirty_indicator_includes_brush_diff() {
        use crate::gui::brush_state::BrushSettings;
        use crate::gui::settings::SettingsModel;
        use crate::gui::presets::{model_id_key, ModelPreset, PresetFile, PRESET_FORMAT_VERSION};

        let mut s = Settings::default();
        s.model = SettingsModel::Silueta;
        let mut preset_brush = BrushSettings::default();
        preset_brush.radius = 80.0;
        let mp = ModelPreset {
            item_settings: ItemSettings::default(),
            brush: preset_brush,
            sd: None,
        };
        let mut models = HashMap::new();
        models.insert(model_id_key(prunr_models::ModelId::Silueta), mp);
        let file = PresetFile { format_version: PRESET_FORMAT_VERSION, models };
        s.presets.insert("Foo".to_string(), file);

        // ItemSettings still equals the preset's; only brush diverges.
        let current = ItemSettings::default();
        s.brush.radius = 10.0;

        let label = button_label(&s, &current, "Foo");
        assert!(label.contains("Foo"), "{label}");
        assert!(
            label.ends_with(ICON_EDIT.codepoint),
            "brush-only diff must trigger the modified indicator: {label}",
        );
    }

    #[test]
    fn sorted_preset_names_case_insensitive() {
        let mut s = Settings::default();
        let empty = wrap(&s, ItemSettings::default());
        let mut map: HashMap<String, PresetFile> = HashMap::new();
        map.insert("Zeta".to_string(), empty.clone());
        map.insert("alpha".to_string(), empty.clone());
        map.insert("Beta".to_string(), empty);
        s.presets = map;
        assert_eq!(sorted_preset_names(&s), vec!["alpha", "Beta", "Zeta"]);
    }

    #[test]
    fn all_preset_names_prunr_first() {
        let mut s = Settings::default();
        let empty = wrap(&s, ItemSettings::default());
        s.presets.insert("Zeta".to_string(), empty.clone());
        s.presets.insert("alpha".to_string(), empty);
        let names = all_preset_names(&s);
        assert_eq!(names[0], PRUNR_PRESET);
        assert_eq!(names[1..], vec!["alpha".to_string(), "Zeta".to_string()]);
    }
}
