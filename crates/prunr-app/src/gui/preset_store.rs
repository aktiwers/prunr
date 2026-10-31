//! The presets folder, read off the UI thread: again whenever the preset
//! list opens, so files added by hand show up, and after an import has
//! copied preset files in. Both run on the disk queue, after every save
//! and delete asked for before them, so a read never misses a preset just
//! saved or finds one just deleted.

use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc};

use super::presets::{Preset, PresetFile};
use super::presets_fs::{self, ImportSource, Imported};
use super::settings::Settings;

/// The folder as read, and what an import before it did.
struct Scan {
    /// The presets the app held when the read was asked for.
    held: HashMap<String, Arc<PresetFile>>,
    presets: HashMap<String, Preset>,
    newer: HashSet<String>,
    imported: Vec<Imported>,
}

pub(crate) struct PresetStore {
    tx: mpsc::Sender<Scan>,
    rx: mpsc::Receiver<Scan>,
    /// Off in a test window, whose presets are made in memory.
    #[cfg(test)]
    reads_folder: bool,
}

impl PresetStore {
    pub(crate) fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tx,
            rx,
            #[cfg(test)]
            reads_folder: true,
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_test() -> Self {
        Self { reads_folder: false, ..Self::new() }
    }

    #[cfg(test)]
    fn reads_folder(&self) -> bool {
        self.reads_folder
    }

    #[cfg(not(test))]
    fn reads_folder(&self) -> bool {
        true
    }

    /// Read the folder again.
    pub(crate) fn rescan(&self, settings: &Settings, ctx: &egui::Context) {
        self.import(Vec::new(), settings, ctx);
    }

    /// Copy `sources` into the folder, then read it again.
    pub(crate) fn import(&self, sources: Vec<ImportSource>, settings: &Settings, ctx: &egui::Context) {
        if !self.reads_folder() {
            return;
        }
        let (tx, ctx, held) = (self.tx.clone(), ctx.clone(), held(settings));
        super::disk_queue::call(move || {
            let imported = presets_fs::import(sources);
            let (presets, newer) = presets_fs::load_all();
            let _ = tx.send(Scan { held, presets, newer, imported });
            ctx.request_repaint();
        });
    }

    /// Lay the last read that landed over `settings`' presets; what every
    /// import that landed did, for the toasts. A read the app's own save or
    /// delete overtook is read again.
    pub(crate) fn drain(&self, settings: &mut Settings, ctx: &egui::Context) -> Vec<Imported> {
        let mut imported = Vec::new();
        let mut last = None;
        for mut scan in self.rx.try_iter() {
            imported.append(&mut scan.imported);
            last = Some(scan);
        }
        let Some(scan) = last else { return imported };
        let unchanged = settings.presets.len() == scan.held.len()
            && settings.presets.iter().all(|(name, preset)| scan.held.get(name).is_some_and(|file| Arc::ptr_eq(file, &preset.shared_file())));
        if unchanged {
            settings.presets = scan.presets;
            settings.newer_presets = scan.newer;
        } else {
            self.rescan(settings, ctx);
        }
        imported
    }
}

fn held(settings: &Settings) -> HashMap<String, Arc<PresetFile>> {
    settings.presets.iter().map(|(name, preset)| (name.clone(), preset.shared_file())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn landed(store: &PresetStore, settings: &mut Settings) {
        super::super::disk_queue::flush();
        store.drain(settings, &egui::Context::default());
    }

    /// A preset written into the folder shows once the folder is read
    /// again, after a save asked for before the read; one deleted goes.
    #[test]
    fn a_rescan_finds_what_was_saved_and_deleted_before_it() {
        let (store, mut settings, ctx) = (PresetStore::new(), Settings::default(), egui::Context::default());
        let name = format!("Rescan {}", std::process::id());
        presets_fs::queue_save(&name, &PresetFile::default());
        store.rescan(&settings, &ctx);
        landed(&store, &mut settings);
        assert!(settings.presets.contains_key(&name), "the queued save landed before the read");
        presets_fs::queue_delete(&name);
        store.rescan(&settings, &ctx);
        landed(&store, &mut settings);
        assert!(!settings.presets.contains_key(&name));
    }

    /// Two reads landing together say what each import did.
    #[test]
    fn every_import_that_landed_is_told() {
        let (store, mut settings, ctx) = (PresetStore::new(), Settings::default(), egui::Context::default());
        let bytes = |name: &str| ImportSource::Bytes { name: name.into(), bytes: std::sync::Arc::from(&b"{ \"hello\": 1 }"[..]) };
        store.import(vec![bytes("one.json")], &settings, &ctx);
        store.import(vec![bytes("two.json")], &settings, &ctx);
        super::super::disk_queue::flush();
        let told = store.drain(&mut settings, &ctx);
        assert_eq!(told, [Imported::NotAPreset("one.json".into()), Imported::NotAPreset("two.json".into())]);
    }

    /// A read that lands after the app saved a preset of its own is read
    /// again rather than dropping that preset from the list.
    #[test]
    fn a_read_overtaken_by_a_save_is_read_again() {
        let (store, mut settings, ctx) = (PresetStore::new(), Settings::default(), egui::Context::default());
        let name = format!("Overtaken {}", std::process::id());
        store.rescan(&settings, &ctx);
        super::super::disk_queue::flush();
        presets_fs::queue_save(&name, &PresetFile::default());
        settings.presets.insert(name.clone(), PresetFile::default().into());
        store.drain(&mut settings, &ctx);
        assert!(settings.presets.contains_key(&name), "the stale read was dropped");
        landed(&store, &mut settings);
        assert!(settings.presets.contains_key(&name), "the read asked again finds the file");
        presets_fs::queue_delete(&name);
        super::super::disk_queue::flush();
    }
}
