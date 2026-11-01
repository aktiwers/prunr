//! The "Preset" dropdown: quick-apply named settings snapshots and save
//! the current item's settings as a new preset.
//!
//! Takes `&mut AppSettings` and `&mut ItemSettings`, never `&mut PrunrApp`.

use egui::{RichText, Ui};
use egui_material_icons::icons::*;

use crate::gui::item_settings::ItemSettings;
use crate::gui::preset_changes::Changes;
use crate::gui::presets::{draft, entry_name, Entry, PresetFile, Shown};
use crate::gui::settings::{PRUNR_PRESET, Settings};
use crate::gui::theme;
use crate::gui::views::chip::{button, ButtonKind};

/// Label for the dropdown button: the preset the user last applied, with a
/// check while current settings still match it and an edit mark once they
/// diverge, so "Portrait ✎" keeps naming the base while you tweak.
fn button_label(settings: &Settings, current: &ItemSettings, applied_preset: &str) -> String {
    // If applied_preset was deleted out from under us, fall back gracefully.
    let exists = crate::gui::presets_fs::is_prunr(applied_preset)
        || settings.presets.contains_key(applied_preset);
    if !exists {
        return format!("Custom  {}", ICON_EDIT.codepoint);
    }
    let resolved = settings.resolve_preset(applied_preset, None);
    let item_diverged = *current != resolved.item_settings;
    let brush_diverged = settings.brush != resolved.brush;
    let is_modified = item_diverged || brush_diverged;
    let state_icon = if is_modified { ICON_EDIT.codepoint } else { ICON_CHECK.codepoint };
    format!("{applied_preset}  {state_icon}")
}

/// The presets a save may write over: not the built-in ones.
fn writable_preset_names(settings: &Settings) -> Vec<String> {
    let writable = settings.presets.iter().filter(|(_, preset)| !preset.is_built_in()).map(|(name, _)| name.clone());
    crate::gui::presets_fs::sorted_names(writable)
}

/// Take preset `name` out of the list: a file of the user's is deleted,
/// a built-in one hidden, and so is the built-in a file of that name stood
/// in for, or it would be back at the next read. The default falls back
/// to Prunr when it goes.
fn delete_preset(settings: &mut Settings, name: &str) {
    if !settings.presets.remove(name).is_some_and(|preset| preset.is_built_in()) {
        crate::gui::presets_fs::queue_delete(name);
    }
    if crate::gui::presets_fs::is_showcase(name) {
        settings.hidden_showcases.insert(name.to_string());
    }
    if settings.default_preset == name {
        settings.default_preset = PRUNR_PRESET.to_string();
    }
}

/// All preset names in display order: Prunr first, then the others by
/// name, case ignored.
pub(super) fn all_preset_names(settings: &Settings) -> Vec<String> {
    let mut names = Vec::with_capacity(settings.presets.len() + 1);
    names.push(PRUNR_PRESET.to_string());
    names.extend(crate::gui::presets_fs::sorted_names(settings.presets.keys().cloned()));
    names
}

const SAVE_DIALOG_ID: &str = "preset_save_dialog";
const SAVE_DIALOG_SIZE: [f32; 2] = [420.0, 500.0];
const OVERWRITE_LIST_HEIGHT: f32 = 120.0;
/// The preset rows scroll past this, keeping Save, Import and Show folder
/// in a short window.
const PRESET_LIST_HEIGHT: f32 = 360.0;
const MODEL_LIST_HEIGHT: f32 = 110.0;

/// What the preset list asks the app to do: with preset files, or to save
/// the settings it changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PresetRequest {
    /// The default preset changed or a built-in preset was hidden: save
    /// the settings, which keep both.
    SaveSettings,
    /// The list opened: read the presets folder again.
    Rescan,
    /// Pick preset files to add.
    Import,
    /// Save this preset as a file to share.
    Export(String),
    /// Open the presets folder.
    ShowFolder,
}

/// The preset state the toolbar edits: the item's applied preset name,
/// what was changed since, and the Save-preset dialog.
pub(crate) struct PresetUi<'a> {
    pub applied: &'a mut String,
    pub changes: &'a Changes<ItemSettings>,
    pub save_dialog: &'a mut Option<SaveDialog>,
}

/// The Save-preset dialog while it is up: the name typed, and the models
/// the save writes, each with its entry and whether it is ticked.
#[derive(Debug, Clone, Default)]
pub(crate) struct SaveDialog {
    pub name: String,
    pub models: Vec<ModelRow>,
}

#[derive(Debug, Clone)]
pub(crate) struct ModelRow {
    pub key: String,
    /// How the row names the model.
    pub name: String,
    pub entry: Entry,
    pub ticked: bool,
}

impl SaveDialog {
    /// The dialog for the shown image: every model whose settings it
    /// would save, ticked.
    fn of(settings: &Settings, item: &ItemSettings, applied: &str, changes: &Changes<ItemSettings>) -> Self {
        let shown = Shown { model: settings.model, item, brush: &settings.brush, item_changes: changes, brush_changes: &settings.brush_changes };
        let models = draft(settings.preset(applied), &shown)
            .into_iter()
            .map(|(key, entry)| ModelRow { name: entry_name(&key).to_string(), key, entry, ticked: true })
            .collect();
        Self { name: String::new(), models }
    }

    /// The ticked models' entries.
    fn ticked(&self) -> impl Iterator<Item = (String, Entry)> + '_ {
        self.models.iter().filter(|row| row.ticked).map(|row| (row.key.clone(), row.entry.clone()))
    }

    /// The file a save as `name` writes: the ticked models over the preset
    /// of that name when there is one, whose other models keep their
    /// entries.
    fn file_for(&self, settings: &Settings, name: &str) -> PresetFile {
        settings.preset(name).file().clone().with_entries(self.ticked())
    }
}

/// Render the Preset dropdown. Returns `Some(name)` when the user applied a
/// preset (click on a row, or Save that swaps in the new name), so the caller
/// can update `BatchItem.applied_preset`. Returns `None` for no-op frames or
/// non-apply interactions (saves, deletes, star toggles).
// Legacy `toggle_popup` / `popup_below_widget` wiring: the Preset chip
// opens like the chip popovers but draws no open-chip outline.
#[allow(deprecated)]
pub(crate) fn render(
    ui: &mut Ui,
    settings: &mut Settings,
    current_item: &mut ItemSettings,
    presets: &mut PresetUi<'_>,
    request: &mut Option<PresetRequest>,
) -> Option<String> {
    let pop_id = egui::Id::new("preset_popover");
    // A delete armed in the list is forgotten once the list closes.
    let delete_scope = pop_id.with("delete_armed");
    if !egui::Popup::is_id_open(ui.ctx(), pop_id) {
        ui.data_mut(|d| d.remove::<String>(delete_scope));
    }

    let label = button_label(settings, current_item, presets.applied);
    let resp = crate::gui::views::chip::tooltip(
        crate::gui::views::chip::chip_button(ui, ICON_BOOKMARK.codepoint, &label, false),
        "Preset",
        "Apply a saved preset to the current image, or save the current settings as one.",
        None,
    );

    if resp.clicked() {
        if !egui::Popup::is_id_open(ui.ctx(), pop_id) {
            *request = Some(PresetRequest::Rescan);
        }
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
            crate::gui::views::chip::popover_header(ui, "Presets", None);

            // Prunr's is built in: it cannot be deleted or overwritten.
            egui::ScrollArea::vertical().id_salt("preset_rows").max_height(PRESET_LIST_HEIGHT).show(ui, |ui| {
                for name in all_preset_names(settings) {
                    let is_prunr = crate::gui::presets_fs::is_prunr(&name);
                    let is_default = settings.default_preset == name;
                    ui.horizontal(|ui| {
                        let label_text = RichText::new(&name)
                            .color(theme::TEXT_PRIMARY)
                            .size(theme::FONT_SIZE_BODY);
                        if ui.selectable_label(false, label_text)
                            .on_hover_text("Click to apply these settings to the current image")
                            .clicked()
                        {
                            // Applying sets what Reset to this preset would.
                            settings.apply_preset(&name, current_item);
                            applied = Some(name.clone());
                            egui::Popup::close_id(ui.ctx(), pop_id);
                        }
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if !is_prunr {
                                    let idle = RichText::new(ICON_DELETE.codepoint).size(theme::FONT_SIZE_SECONDARY).color(theme::DESTRUCTIVE);
                                    let armed = RichText::new("Delete?").size(theme::FONT_SIZE_SECONDARY).color(theme::DESTRUCTIVE);
                                    if crate::gui::views::chip::confirm_button(ui, delete_scope, &name, idle, armed, "Delete preset") {
                                        delete_preset(settings, &name);
                                        *request = Some(PresetRequest::SaveSettings);
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
                                        .size(theme::FONT_SIZE_SECONDARY)
                                        .color(star_color),
                                );
                                let star_btn = crate::gui::views::chip::named(star_btn, "Set as default");
                                if star_btn.on_hover_text(star_tooltip).clicked() && !is_default {
                                    settings.default_preset = name.clone();
                                    *request = Some(PresetRequest::SaveSettings);
                                }
                                let export = ui.small_button(RichText::new(ICON_IOS_SHARE.codepoint).size(theme::FONT_SIZE_SECONDARY).color(theme::TEXT_SECONDARY));
                                if crate::gui::views::chip::named(export, "Export preset").on_hover_text("Save this preset as a file to share").clicked() {
                                    *request = Some(PresetRequest::Export(name.clone()));
                                    egui::Popup::close_id(ui.ctx(), pop_id);
                                }
                            },
                        );
                    });
                    if let Some(covers) = settings.presets.get(&name).map(|p| p.covers()).filter(|c| !c.is_empty()) {
                        ui.horizontal(|ui| {
                            // Under the name, which sits inside its row's padding.
                            ui.add_space(ui.spacing().button_padding.x);
                            ui.label(RichText::new(covers).color(theme::TEXT_HINT).size(theme::FONT_SIZE_SECONDARY));
                        });
                    }
                }
            });

            ui.add_space(theme::SPACE_SM);
            ui.separator();
            ui.add_space(theme::SPACE_SM);

            let save_label = format!("{}  Save current as…", ICON_BOOKMARK_ADD.codepoint);
            if button(ui, ButtonKind::Secondary, &save_label).clicked() {
                *presets.save_dialog = Some(SaveDialog::of(settings, current_item, presets.applied, presets.changes));
                egui::Popup::close_id(ui.ctx(), pop_id);
            }
            ui.horizontal(|ui| {
                let import = format!("{}  Import…", ICON_FILE_OPEN.codepoint);
                if button(ui, ButtonKind::Secondary, &import).on_hover_text("Add preset files someone shared").clicked() {
                    *request = Some(PresetRequest::Import);
                    egui::Popup::close_id(ui.ctx(), pop_id);
                }
                let folder = format!("{}  Show folder", ICON_FOLDER_OPEN.codepoint);
                if button(ui, ButtonKind::Secondary, &folder).on_hover_text("Open the folder the presets are kept in").clicked() {
                    *request = Some(PresetRequest::ShowFolder);
                    egui::Popup::close_id(ui.ctx(), pop_id);
                }
            });

            ui.add_space(theme::SPACE_XS);
            ui.label(
                RichText::new(format!("Default for new images: {}", settings.default_preset))
                    .color(theme::TEXT_HINT)
                    .size(theme::FONT_SIZE_SECONDARY),
            );
        },
    );

    if let Some(dialog) = presets.save_dialog.as_mut() {
        let SaveDialog { name: name_buf, models } = dialog;
        let mut commit = false;
        let mut cancel = false;
        let mut overwrite_target: Option<String> = None;

        // Snapshot existing preset names so we can offer overwrite targets
        // without holding a borrow on `settings.presets` across the dialog.
        let existing_names = writable_preset_names(settings);

        let backdrop_closed = theme::standard_modal_window(ui.ctx(), SAVE_DIALOG_ID, "Save preset", SAVE_DIALOG_SIZE, |ui| {
                let name_label = ui.label("Name for this preset:");
                let text_resp = ui.text_edit_singleline(name_buf).labelled_by(name_label.id);
                if text_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    commit = true;
                }
                // Opens ready for a name; a click elsewhere in the dialog keeps its focus.
                // After the Enter check: focusing first would hide the loss Enter caused.
                if !commit && ui.memory(|m| m.focused().is_none()) {
                    text_resp.request_focus();
                }
                let name = crate::gui::presets_fs::preset_name(name_buf);
                let refusal = name.as_deref().and_then(|name| crate::gui::presets_fs::name_refusal(name, &settings.newer_presets, &settings.presets));
                if let Some(why) = refusal {
                    ui.label(RichText::new(why).color(theme::DESTRUCTIVE).size(theme::FONT_SIZE_SECONDARY));
                } else if let Some(name) = name.as_ref().filter(|name| existing_names.contains(name)) {
                    ui.label(
                        RichText::new(format!("⚠ Will overwrite \"{name}\""))
                            .color(theme::DESTRUCTIVE)
                            .size(theme::FONT_SIZE_SECONDARY),
                    );
                } else if let Some(name) = name.as_ref().filter(|name| *name != name_buf.trim()) {
                    ui.label(RichText::new(format!("Saves as \"{name}\"")).size(theme::FONT_SIZE_SECONDARY));
                }

                ui.add_space(theme::SPACE_SM);
                if models.is_empty() {
                    ui.label(RichText::new("Nothing differs from Prunr's settings.").color(theme::TEXT_HINT).size(theme::FONT_SIZE_SECONDARY));
                } else {
                    ui.label(RichText::new("Saves the settings of:").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_SECONDARY));
                    egui::ScrollArea::vertical().id_salt("save_models").max_height(MODEL_LIST_HEIGHT).show(ui, |ui| {
                        for row in models.iter_mut() {
                            ui.checkbox(&mut row.ticked, row.name.as_str());
                        }
                    });
                }

                ui.add_space(theme::SPACE_SM);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled_ui(refusal.is_none(), |ui| button(ui, ButtonKind::Primary, "Save"))
                        .inner
                        .clicked()
                    {
                        commit = true;
                    }
                    if button(ui, ButtonKind::Secondary, "Cancel").clicked() {
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
                            .size(theme::FONT_SIZE_SECONDARY),
                    );
                    ui.add_space(theme::SPACE_XS);
                    egui::ScrollArea::vertical().id_salt("overwrite_presets").max_height(OVERWRITE_LIST_HEIGHT).show(ui, |ui| {
                        for name in &existing_names {
                            let label = format!("{}  {name}", ICON_BOOKMARK.codepoint);
                            let btn = button(ui, ButtonKind::Secondary, &label);
                            if btn.on_hover_text("Overwrite with current settings").clicked() {
                                overwrite_target = Some(name.clone());
                            }
                        }
                    });
                }
        });
        cancel |= backdrop_closed;

        // Resolve the target name: overwrite button wins, else a valid typed
        // name on commit. Saving the current settings AS a preset makes that
        // preset the now-applied one — label reads "Name ✓" after.
        let target_name = overwrite_target.or_else(|| {
            crate::gui::presets_fs::preset_name(name_buf)
                .filter(|name| commit && crate::gui::presets_fs::name_refusal(name, &settings.newer_presets, &settings.presets).is_none())
        });
        if let Some(name) = target_name {
            // The store keeps what is written, for the preset button's mark.
            let file = dialog.file_for(settings, &name);
            crate::gui::presets_fs::queue_save(&name, &file);
            settings.presets.insert(name.clone(), file.into());

            applied = Some(name);
            cancel = true;
        }
        if cancel {
            *presets.save_dialog = None;
        }
    }

    applied
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::gui::item_settings::item_with_gamma;
    use crate::gui::presets::{Preset, PresetFile};

    /// A preset of `item` for the active model, saved as the app saves.
    fn wrap(s: &Settings, item: ItemSettings) -> Preset {
        let brush = crate::gui::presets::prunr_view(s.model, None).brush;
        PresetFile::default().with_settings(s.model, &item, &brush).into()
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
    fn button_label_dirty_indicator_includes_brush_diff() {
        use crate::gui::brush_state::BrushSettings;
        use crate::gui::settings::SettingsModel;

        let mut s = Settings::default();
        s.model = SettingsModel::Silueta;
        let preset_brush = BrushSettings { radius: 80.0, ..crate::gui::presets::prunr_view(s.model, None).brush };
        let foo = PresetFile::default().with_settings(s.model, &ItemSettings::default(), &preset_brush);
        s.presets.insert("Foo".to_string(), foo.into());

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

    /// Saving under an existing preset's name, typed or picked, merges
    /// into it: the models the save does not hold keep their entries.
    #[test]
    fn saving_under_a_presets_name_keeps_its_other_models() {
        use crate::gui::presets::Preset;
        let mut s = Settings::default();
        let portrait: Preset = serde_json::from_str(r#"{ "format_version": 2, "models": {
            "Silueta": { "item_settings": { "gamma": 1.5 } },
            "FourXNomos2RealPlksr": { "item_settings": { "sharpen": 0.4 } }
        } }"#).unwrap();
        s.presets.insert("Portrait".into(), portrait);
        let entry: Entry = serde_json::from_str(r#"{ "item_settings": { "gamma": 1.2 } }"#).unwrap();
        let dialog = SaveDialog { name: "Portrait".into(), models: vec![ModelRow { key: "BiRefNetLite".into(), name: "BiRefNet".into(), entry, ticked: true }] };
        let mut keys: Vec<String> = dialog.file_for(&s, "Portrait").models.into_keys().collect();
        keys.sort();
        assert_eq!(keys, ["BiRefNetLite", "FourXNomos2RealPlksr", "Silueta"]);
    }

    /// Starring a preset asks for the settings to be saved, which keep
    /// the default: without that a restart lost it unless something else
    /// saved them first.
    #[test]
    fn starring_a_preset_saves_the_settings() {
        use egui_kittest::kittest::Queryable;
        struct State { settings: Settings, item: ItemSettings, applied: String, changes: Changes<ItemSettings>, dialog: Option<SaveDialog>, request: Option<PresetRequest> }
        let mut settings = Settings::default();
        settings.presets.insert("Mine".into(), PresetFile::default().into());
        let state = State { settings, item: ItemSettings::default(), applied: PRUNR_PRESET.into(), changes: Changes::new(ItemSettings::default()), dialog: None, request: None };
        let mut h = egui_kittest::Harness::new_ui_state(|ui, s: &mut State| {
            let mut presets = PresetUi { applied: &mut s.applied, changes: &s.changes, save_dialog: &mut s.dialog };
            render(ui, &mut s.settings, &mut s.item, &mut presets, &mut s.request);
        }, state);
        h.get_by_label("Preset").click();
        h.run();
        // Prunr's row comes first and is the default already.
        h.get_all_by_label("Set as default").nth(1).expect("Mine's star").click();
        h.run();
        assert_eq!(h.state().settings.default_preset, "Mine");
        assert_eq!(h.state().request, Some(PresetRequest::SaveSettings));
    }

    /// Deleting a built-in preset hides it and writes no file; the
    /// default falls back to Prunr when it was the one deleted. Saving
    /// never offers to write over a built-in one.
    #[test]
    fn deleting_a_built_in_preset_hides_it() {
        let mut s = Settings::default();
        let built_in = PresetFile::default();
        s.presets.insert("Comic".into(), crate::gui::presets::Preset::from(built_in).built_in());
        s.presets.insert("Mine".into(), PresetFile::default().into());
        assert_eq!(writable_preset_names(&s), ["Mine"]);
        s.default_preset = "Comic".into();
        delete_preset(&mut s, "Comic");
        assert!(!s.presets.contains_key("Comic"));
        assert!(s.hidden_showcases.contains("Comic"));
        assert_eq!(s.default_preset, PRUNR_PRESET);

        let mut s = Settings::default();
        s.presets.insert("Comic".into(), PresetFile::default().into());
        delete_preset(&mut s, "Comic");
        assert!(s.hidden_showcases.contains("Comic"), "a file standing in for the built-in takes it along");
    }

    /// Prunr first, then the user's presets by name ignoring case
    /// ("alpha" before "Zeta").
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
