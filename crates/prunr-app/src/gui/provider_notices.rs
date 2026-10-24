//! The provider check's notices (`prunr_core::provider_check`), from this
//! process's sessions and the worker's, held until the window shows them.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use prunr_core::provider_check::Notice;

static WAITING: Mutex<Vec<Notice>> = Mutex::new(Vec::new());
static ANY: AtomicBool = AtomicBool::new(false);
static WINDOW: OnceLock<egui::Context> = OnceLock::new();

/// This process's observer.
pub fn observe(notice: Notice) {
    WAITING.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(notice);
    ANY.store(true, Ordering::Release);
    if let Some(ctx) = WINDOW.get() {
        ctx.request_repaint();
    }
}

/// The notices so far, for the window to show; the window is woken for
/// the next. Called every frame.
pub(crate) fn take(ctx: &egui::Context) -> Vec<Notice> {
    WINDOW.get_or_init(|| ctx.clone());
    if !ANY.swap(false, Ordering::Acquire) {
        return Vec::new();
    }
    std::mem::take(&mut *WAITING.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
}

/// How long a notice stays up: a check runs the model on the CPU, about
/// 17-19 s for BiRefNet on an i5-11600KF, so its notice lasts as long.
pub(crate) fn shown_for(notice: &Notice) -> std::time::Duration {
    std::time::Duration::from_secs(match notice {
        Notice::Checking { .. } => 20,
        Notice::PassedOver { .. } => 10,
    })
}
