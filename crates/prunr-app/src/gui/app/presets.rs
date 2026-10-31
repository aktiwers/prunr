//! The preset list's files: reading the folder again, Import (picked or
//! dropped), Export and Show folder.

use std::path::PathBuf;

use crate::gui::presets_fs::{self, ImportSource, Imported};
use crate::gui::settings::PRUNR_PRESET;
use crate::gui::system_bridge::DialogRequest;
use crate::gui::views::preset_dropdown::PresetRequest;
use super::PrunrApp;

impl PrunrApp {
    pub(super) fn handle_preset_request(&mut self, request: PresetRequest, ctx: &egui::Context) {
        match request {
            PresetRequest::Rescan => self.preset_store.rescan(&self.settings, ctx),
            PresetRequest::Import => self.ask_dialog(DialogRequest::ImportPresets),
            PresetRequest::Export(name) => self.ask_dialog(DialogRequest::ExportPreset { name }),
            PresetRequest::ShowFolder => {
                let shown = presets_fs::presets_dir()
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no settings folder"))
                    .and_then(|dir| crate::gui::system_bridge::show_folder(&dir));
                if let Err(err) = shown {
                    self.toasts.error(format!("Couldn't open the presets folder: {}", prunr_core::plain::reason(err)));
                }
            }
        }
    }

    /// Add the preset files picked or dropped.
    pub(super) fn import_presets(&mut self, sources: Vec<ImportSource>, ctx: &egui::Context) {
        if !sources.is_empty() {
            self.preset_store.import(sources, &self.settings, ctx);
        }
    }

    /// Write preset `name` to the file picked for it.
    pub(super) fn on_preset_export_picked(&mut self, name: &str, path: PathBuf) {
        let json = if name == PRUNR_PRESET {
            Ok(crate::gui::presets::prunr_file_json().to_string())
        } else if let Some(preset) = self.settings.presets.get(name) {
            serde_json::to_string_pretty(preset.file())
        } else {
            return;
        };
        match json {
            Ok(json) => {
                crate::gui::disk_queue::write(path, json.into_bytes());
                self.toasts.info(format!("Exported \"{name}\"."));
            }
            Err(err) => {
                tracing::error!(preset = %name, %err, "preset not exported");
            }
        }
    }

    /// Lay the last read of the presets folder over the list, and say what
    /// an import did.
    pub(super) fn drain_preset_scans(&mut self, ctx: &egui::Context) {
        for imported in self.preset_store.drain(&mut self.settings, ctx) {
            match imported {
                Imported::As(name) => self.toasts.info(format!("Imported the preset \"{name}\".")),
                Imported::Newer(name) => self.toasts.warning(format!("\"{name}\" was saved by a newer Prunr; update Prunr to use it.")),
                Imported::NotAPreset(file) => self.toasts.warning(format!("\"{file}\" is not a Prunr preset.")),
                Imported::NotCopied(file) => self.toasts.error(format!("\"{file}\" could not be copied into the presets folder.")),
            };
        }
    }
}
