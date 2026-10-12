//! Thin shim over `rfd` (file dialogs) and `arboard` (clipboard) so the
//! rest of the GUI doesn't carry their import surface.

use std::path::{Path, PathBuf};

pub struct SystemBridge {
    /// `None` on Wayland-without-data-control or headless test envs —
    /// `copy_image` no-ops gracefully and the caller falls back to a
    /// "save instead" toast.
    clipboard: Option<arboard::Clipboard>,
}

impl Default for SystemBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemBridge {
    pub fn new() -> Self {
        Self { clipboard: arboard::Clipboard::new().ok() }
    }

    /// Multi-file image picker. Returns `None` on cancel.
    pub fn open_files_dialog(&self, start_dir: Option<&Path>) -> Option<Vec<PathBuf>> {
        let mut dlg = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp", "svg"])
            .set_title("Open Image(s)");
        if let Some(dir) = start_dir {
            dlg = dlg.set_directory(dir);
        }
        dlg.pick_files()
    }

    /// Save-as dialog offering every export format, `preferred` first.
    /// `None` on cancel.
    pub fn save_image_dialog(
        &self,
        start_dir: Option<&Path>,
        default_name: &str,
        preferred: prunr_core::ExportFormat,
    ) -> Option<PathBuf> {
        let mut dlg = rfd::FileDialog::new().set_file_name(default_name).set_title("Save image");
        let others = prunr_core::ExportFormat::ALL.into_iter().filter(|f| *f != preferred);
        for f in std::iter::once(preferred).chain(others) {
            dlg = dlg.add_filter(format!("{} image", f.label()), f.extensions());
        }
        if let Some(dir) = start_dir {
            dlg = dlg.set_directory(dir);
        }
        dlg.save_file()
    }

    /// Single-file image picker with a custom title. Returns `None` on cancel.
    pub fn pick_image_dialog(
        &self,
        start_dir: Option<&Path>,
        title: &str,
    ) -> Option<PathBuf> {
        let mut dlg = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp"])
            .set_title(title);
        if let Some(dir) = start_dir {
            dlg = dlg.set_directory(dir);
        }
        dlg.pick_file()
    }

    /// Folder picker with custom title. Returns `None` on cancel.
    pub fn pick_folder_dialog(
        &self,
        start_dir: Option<&Path>,
        title: &str,
    ) -> Option<PathBuf> {
        let mut dlg = rfd::FileDialog::new().set_title(title);
        if let Some(dir) = start_dir {
            dlg = dlg.set_directory(dir);
        }
        dlg.pick_folder()
    }

    /// Copy raw RGBA pixels to the system clipboard. Returns whether the
    /// clipboard was available — `false` means no platform clipboard
    /// (callers fall back to a "save instead" suggestion).
    pub fn copy_image(&mut self, rgba: &image::RgbaImage) -> bool {
        let Some(clipboard) = self.clipboard.as_mut() else { return false };
        let samples = rgba.as_flat_samples();
        let image_data = arboard::ImageData {
            width: rgba.width() as usize,
            height: rgba.height() as usize,
            bytes: std::borrow::Cow::Borrowed(samples.as_slice()),
        };
        clipboard.set_image(image_data).is_ok()
    }
}

