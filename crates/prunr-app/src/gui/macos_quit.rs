//! Quit (the app menu, Cmd+Q, the Dock, a logout) asks AppKit's
//! `applicationShouldTerminate:`, which winit's app delegate does not
//! answer, so the app ended without the unsaved-work prompt. This answers
//! it: with work no save wrote, the quit becomes a close of the window,
//! which the prompt holds as it does the title bar's button.

use std::sync::atomic::{AtomicBool, Ordering};

use objc2::runtime::{AnyClass, AnyObject, Imp, Sel};
use objc2::{sel, MainThreadMarker};
use objc2_app_kit::{NSApplication, NSApplicationTerminateReply};

static UNSAVED: AtomicBool = AtomicBool::new(false);

/// Whether a quit now would lose work; the app sets it every frame.
pub(crate) fn set_unsaved(unsaved: bool) {
    UNSAVED.store(unsaved, Ordering::Relaxed);
}

type ShouldTerminate = unsafe extern "C-unwind" fn(&AnyObject, Sel, Option<&AnyObject>) -> NSApplicationTerminateReply;

/// Answer `applicationShouldTerminate:` on winit's app delegate. Runs once
/// the event loop has made the delegate, as the app is created.
pub(crate) fn install() {
    let Some(class) = AnyClass::get(c"WinitApplicationDelegate") else {
        tracing::warn!("winit's app delegate not found; Quit skips the unsaved-work prompt");
        return;
    };
    let answer: ShouldTerminate = should_terminate;
    // SAFETY: "Q@:@" is `should_terminate`'s signature (an NSUInteger
    // reply; self, _cmd and the sender), and the class is winit's, which
    // does not answer the selector, so the method is added, not swapped.
    let added = unsafe {
        objc2::ffi::class_addMethod(
            (class as *const AnyClass).cast_mut(),
            sel!(applicationShouldTerminate:),
            std::mem::transmute::<ShouldTerminate, Imp>(answer),
            c"Q@:@".as_ptr(),
        )
    };
    if !added.as_bool() {
        tracing::warn!("winit's app delegate already answers Quit; the unsaved-work prompt may be skipped");
    }
}

unsafe extern "C-unwind" fn should_terminate(_this: &AnyObject, _cmd: Sel, _sender: Option<&AnyObject>) -> NSApplicationTerminateReply {
    // Nothing to lose: let a logout or a Dock quit go through untouched.
    if !UNSAVED.load(Ordering::Relaxed) {
        return NSApplicationTerminateReply::TerminateNow;
    }
    let Some(mtm) = MainThreadMarker::new() else { return NSApplicationTerminateReply::TerminateNow };
    let app = NSApplication::sharedApplication(mtm);
    let Some(window) = app.mainWindow().or_else(|| app.keyWindow()).or_else(|| app.windows().firstObject()) else {
        return NSApplicationTerminateReply::TerminateNow;
    };
    // The prompt shows in the window, so the user must see it.
    if window.isMiniaturized() {
        window.deminiaturize(None);
    }
    app.activate();
    window.performClose(None);
    NSApplicationTerminateReply::TerminateCancel
}
