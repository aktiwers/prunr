//! Panics on threads that owe an answer. rayon aborts the process when a
//! spawned job panics, and a thread that dies without answering leaves
//! whatever waits on it (a busy pill, an image) waiting for good.

/// `f`'s result, or `None` when it panicked, its message logged under
/// `what`.
pub fn caught<T>(what: &'static str, f: impl FnOnce() -> T) -> Option<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .map_err(|payload| tracing::error!(why = %message(&*payload), what, "panicked"))
        .ok()
}

fn message(payload: &(dyn std::any::Any + Send)) -> String {
    payload.downcast_ref::<&str>().map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default()
}

/// The window's pool logs a spawned job's panic instead of aborting:
/// each job answers its own panic where something waits on it, and one
/// that slips through must not close the window and lose unsaved work.
pub(crate) fn window_pool() -> rayon::ThreadPoolBuilder {
    rayon::ThreadPoolBuilder::new().panic_handler(|payload| {
        tracing::error!(why = %message(&*payload), "a background job panicked");
    })
}

/// Makes `window_pool` rayon's global pool; before any job uses it.
pub(crate) fn use_window_pool() {
    if let Err(e) = window_pool().build_global() {
        tracing::warn!(%e, "the background pool started without its panic handler");
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_panic_is_caught_and_its_message_read() {
        assert_eq!(super::caught("a test", || 7), Some(7));
        assert_eq!(super::caught("a test", || -> u32 { panic!("a driver bug") }), None);
        let payload = |f: fn()| std::panic::catch_unwind(f).unwrap_err();
        assert_eq!(super::message(&*payload(|| panic!("a driver bug"))), "a driver bug");
        assert_eq!(super::message(&*payload(|| panic!("{}", String::from("owned")))), "owned");
    }

    /// Without the handler rayon aborts the process here, taking the
    /// test run with it.
    #[test]
    fn a_job_that_panics_on_the_window_pool_leaves_the_app_running() {
        let pool = super::window_pool().num_threads(1).build().expect("a one-thread pool");
        pool.spawn(|| panic!("a bug in a background job"));
        let (tx, rx) = std::sync::mpsc::channel();
        pool.spawn(move || {
            let _ = tx.send(7);
        });
        assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(5)), Ok(7));
    }
}
