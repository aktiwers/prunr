//! Thin shim over `rfd` (file dialogs) and `arboard` (clipboard) so the
//! rest of the GUI doesn't carry their import surface.
//!
//! A file dialog never holds the frame. Any surface asks for one;
//! `logic()` shows it attached to the window, a helper thread waits for
//! the user, and `logic()` drains the answer and resumes what asked.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::mpsc;

use prunr_core::ExportFormat;

/// What a file dialog is for; its answer resumes exactly this. Images
/// go by id: the batch can change while the dialog is up.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum DialogRequest {
    /// Open: import the picked images.
    OpenImages,
    /// The picked picture becomes this image's background.
    Background { item_id: u64 },
    /// Save as: this image's result goes to the picked file.
    SaveImage { item_id: u64, default_name: String, preferred: ExportFormat },
    /// These images' layers go into the picked folder.
    SaveLayers { targets: Vec<u64> },
    /// These images' results go into the picked folder.
    SaveSelected { targets: Vec<u64> },
}

/// A dialog the user answered: what it was for and the paths picked,
/// never empty.
pub(crate) struct DialogAnswer {
    pub(crate) request: DialogRequest,
    pub(crate) paths: Vec<PathBuf>,
}

/// The paths a dialog sends back; `None` when it was cancelled.
type Reply = Option<Vec<PathBuf>>;

/// Shows a dialog without blocking; the reply comes on the receiver.
/// `None` leaves the dialog asked for and unshown.
type ShowDialog = fn(&DialogRequest, Option<&Path>, &eframe::Frame, &egui::Context) -> Option<mpsc::Receiver<Reply>>;

struct Dialog {
    request: DialogRequest,
    start_dir: Option<PathBuf>,
    /// `None` until shown.
    reply: Option<mpsc::Receiver<Reply>>,
}

pub struct SystemBridge {
    /// `None` on Wayland-without-data-control or headless test envs —
    /// `copy_image` no-ops gracefully and the caller falls back to a
    /// "save instead" toast.
    clipboard: Option<arboard::Clipboard>,
    /// The one file dialog asked for or up.
    dialog: Option<Dialog>,
    show: ShowDialog,
}

impl Default for SystemBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemBridge {
    pub fn new() -> Self {
        Self { clipboard: arboard::Clipboard::new().ok(), dialog: None, show: show_native }
    }

    /// Ask for a file dialog starting in `start_dir`. Ignored while
    /// another is asked for or up: one dialog at a time.
    pub(crate) fn ask_dialog(&mut self, request: DialogRequest, start_dir: Option<PathBuf>) {
        if let Some(open) = &self.dialog {
            tracing::debug!(?request, open = ?open.request, "a file dialog is already up; request ignored");
            return;
        }
        self.dialog = Some(Dialog { request, start_dir, reply: None });
    }

    /// A file dialog is asked for or up.
    pub(crate) fn dialog_open(&self) -> bool {
        self.dialog.is_some()
    }

    /// A dialog is asked for and waits for `show_dialog`.
    pub(crate) fn dialog_unshown(&self) -> bool {
        self.dialog.as_ref().is_some_and(|d| d.reply.is_none())
    }

    /// Show the asked-for dialog, attached to the app window.
    pub(crate) fn show_dialog(&mut self, ctx: &egui::Context, window: &eframe::Frame) {
        let Some(dialog) = self.dialog.as_mut().filter(|d| d.reply.is_none()) else { return };
        dialog.reply = (self.show)(&dialog.request, dialog.start_dir.as_deref(), window, ctx);
    }

    /// The dialog's answer once the user picked. A cancel closes the
    /// dialog and answers nothing.
    pub(crate) fn take_dialog_answer(&mut self) -> Option<DialogAnswer> {
        let reply = match self.dialog.as_ref()?.reply.as_ref()?.try_recv() {
            Ok(reply) => reply,
            Err(mpsc::TryRecvError::Empty) => return None,
            Err(mpsc::TryRecvError::Disconnected) => {
                tracing::warn!("file dialog ended without an answer");
                None
            }
        };
        let request = self.dialog.take()?.request;
        let paths = reply.filter(|paths| !paths.is_empty())?;
        Some(DialogAnswer { request, paths })
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

#[cfg(test)]
impl SystemBridge {
    /// No native dialog ever shows: a test answers with `answer_for_test`.
    pub(crate) fn new_for_test() -> Self {
        Self { show: |_, _, _, _| None, ..Self::new() }
    }

    /// The dialog asked for, and the folder it starts in.
    pub(crate) fn asked_for_test(&self) -> Option<(&DialogRequest, Option<&Path>)> {
        self.dialog.as_ref().map(|d| (&d.request, d.start_dir.as_deref()))
    }

    /// Answer the dialog as the user would: the paths picked, `None` to cancel.
    pub(crate) fn answer_for_test(&mut self, reply: Reply) {
        let Some(dialog) = self.dialog.as_mut() else { return };
        let (tx, rx) = mpsc::channel();
        let _ = tx.send(reply);
        dialog.reply = Some(rx);
    }
}

/// Which native dialog a request shows.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Pick {
    Files,
    File,
    Folder,
    SaveFile,
}

/// How a request's dialog looks, apart from showing it.
#[derive(Debug, PartialEq)]
struct Look {
    pick: Pick,
    title: &'static str,
    filters: Vec<(String, &'static [&'static str])>,
    file_name: Option<String>,
}

const OPEN_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "svg"];
const BACKGROUND_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp"];

fn look(request: &DialogRequest) -> Look {
    let images = |extensions| vec![("Images".to_string(), extensions)];
    let folder = |title| Look { pick: Pick::Folder, title, filters: Vec::new(), file_name: None };
    match request {
        DialogRequest::OpenImages => {
            Look { pick: Pick::Files, title: "Open Image(s)", filters: images(OPEN_EXTENSIONS), file_name: None }
        }
        DialogRequest::Background { .. } => Look {
            pick: Pick::File,
            title: "Choose background image",
            filters: images(BACKGROUND_EXTENSIONS),
            file_name: None,
        },
        // Every export format, the one Save options names first.
        DialogRequest::SaveImage { default_name, preferred, .. } => {
            let others = ExportFormat::ALL.into_iter().filter(|f| f != preferred);
            Look {
                pick: Pick::SaveFile,
                title: "Save image",
                filters: std::iter::once(*preferred)
                    .chain(others)
                    .map(|f| (format!("{} image", f.label()), f.extensions()))
                    .collect(),
                file_name: Some(default_name.clone()),
            }
        }
        DialogRequest::SaveLayers { .. } => folder("Save layers \u{2014} Choose folder"),
        DialogRequest::SaveSelected { .. } => folder("Save Selected \u{2014} Choose Folder"),
    }
}

type PendingReply = Pin<Box<dyn Future<Output = Reply> + Send>>;

/// Start the native dialog, parented to the window, and a helper thread
/// that waits for it, sends the reply and wakes the frame loop.
fn show_native(
    request: &DialogRequest,
    start_dir: Option<&Path>,
    window: &eframe::Frame,
    ctx: &egui::Context,
) -> Option<mpsc::Receiver<Reply>> {
    let Look { pick, title, filters, file_name } = look(request);
    let mut dialog = rfd::AsyncFileDialog::new().set_parent(window).set_title(title);
    for (name, extensions) in filters {
        dialog = dialog.add_filter(name, extensions);
    }
    if let Some(name) = file_name {
        dialog = dialog.set_file_name(name);
    }
    if let Some(dir) = start_dir {
        dialog = dialog.set_directory(dir);
    }
    // Called here, on the UI thread: on macOS rfd begins the dialog as a
    // sheet right away, and AppKit takes that from the main thread only.
    // The helper thread only waits.
    let pending: PendingReply = match pick {
        Pick::Files => {
            let files = dialog.pick_files();
            Box::pin(async move { files.await.map(|files| files.into_iter().map(PathBuf::from).collect()) })
        }
        Pick::File => one_path(dialog.pick_file()),
        Pick::Folder => one_path(dialog.pick_folder()),
        Pick::SaveFile => one_path(dialog.save_file()),
    };
    let (tx, rx) = mpsc::channel();
    let ctx = ctx.clone();
    let waiter = std::thread::Builder::new().name("file-dialog".into()).spawn(move || {
        let _ = tx.send(pollster::block_on(pending));
        ctx.request_repaint();
    });
    if let Err(err) = waiter {
        // The sender went down with the closure, so the dialog reads as cancelled.
        tracing::error!(%err, "file dialog thread failed to start");
    }
    Some(rx)
}

fn one_path(picked: impl Future<Output = Option<rfd::FileHandle>> + Send + 'static) -> PendingReply {
    Box::pin(async move { picked.await.map(|file| vec![PathBuf::from(file)]) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exts(name: &str, extensions: &'static [&'static str]) -> (String, &'static [&'static str]) {
        (name.to_string(), extensions)
    }

    #[test]
    fn each_dialog_keeps_its_title_filters_and_file_name() {
        assert_eq!(
            look(&DialogRequest::OpenImages),
            Look {
                pick: Pick::Files,
                title: "Open Image(s)",
                filters: vec![exts("Images", &["png", "jpg", "jpeg", "webp", "bmp", "svg"])],
                file_name: None,
            },
        );
        assert_eq!(
            look(&DialogRequest::Background { item_id: 3 }),
            Look {
                pick: Pick::File,
                title: "Choose background image",
                filters: vec![exts("Images", &["png", "jpg", "jpeg", "webp", "bmp"])],
                file_name: None,
            },
        );
        assert_eq!(
            look(&DialogRequest::SaveImage { item_id: 3, default_name: "cat.prunr.webp".into(), preferred: ExportFormat::Webp }),
            Look {
                pick: Pick::SaveFile,
                title: "Save image",
                filters: vec![
                    exts("WebP image", &["webp"]),
                    exts("PNG image", &["png"]),
                    exts("JPEG image", &["jpg", "jpeg"]),
                ],
                file_name: Some("cat.prunr.webp".into()),
            },
        );
        let folder = |title| Look { pick: Pick::Folder, title, filters: Vec::new(), file_name: None };
        assert_eq!(look(&DialogRequest::SaveLayers { targets: vec![1] }), folder("Save layers \u{2014} Choose folder"));
        assert_eq!(look(&DialogRequest::SaveSelected { targets: vec![1] }), folder("Save Selected \u{2014} Choose Folder"));
    }

    #[test]
    fn one_dialog_at_a_time_and_a_cancel_frees_the_slot() {
        let mut bridge = SystemBridge::new_for_test();
        bridge.ask_dialog(DialogRequest::OpenImages, None);
        bridge.ask_dialog(DialogRequest::Background { item_id: 1 }, None);
        assert_eq!(bridge.asked_for_test().map(|(r, _)| r), Some(&DialogRequest::OpenImages));
        assert!(bridge.take_dialog_answer().is_none(), "no answer before the user picks");

        bridge.answer_for_test(None);
        assert!(bridge.take_dialog_answer().is_none());
        assert!(!bridge.dialog_open());

        bridge.ask_dialog(DialogRequest::Background { item_id: 1 }, None);
        assert!(bridge.dialog_open());
    }

    #[test]
    fn a_dialog_whose_thread_died_closes_as_cancelled() {
        let mut bridge = SystemBridge::new_for_test();
        bridge.ask_dialog(DialogRequest::OpenImages, None);
        let (tx, rx) = mpsc::channel::<Reply>();
        drop(tx);
        bridge.dialog.as_mut().unwrap().reply = Some(rx);
        assert!(bridge.take_dialog_answer().is_none());
        assert!(!bridge.dialog_open());
    }

    #[test]
    fn an_empty_pick_reads_as_cancelled() {
        let mut bridge = SystemBridge::new_for_test();
        bridge.ask_dialog(DialogRequest::OpenImages, None);
        bridge.answer_for_test(Some(Vec::new()));
        assert!(bridge.take_dialog_answer().is_none());
        assert!(!bridge.dialog_open());
    }
}
