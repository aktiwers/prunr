//! Large values freed off the UI thread. Freeing a full-size image gives
//! its pages back to the OS one by one (a 48 MB buffer takes a few ms, an
//! upscale's output or a model session far longer); the UI thread hands
//! such a value here instead of dropping it.

use std::sync::{mpsc, OnceLock};

type Garbage = Box<dyn Send>;

fn bin() -> &'static mpsc::Sender<Garbage> {
    static BIN: OnceLock<mpsc::Sender<Garbage>> = OnceLock::new();
    BIN.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Garbage>();
        std::thread::Builder::new()
            .name("reclaim".into())
            .spawn(move || rx.into_iter().for_each(drop))
            .map_err(|err| tracing::error!(%err, "reclaim thread did not start"))
            .ok();
        tx
    })
}

/// Drop `value` on the reclaim thread. A value still shared elsewhere
/// only loses this handle there.
pub(crate) fn later<T: Send + 'static>(value: T) {
    if let Err(mpsc::SendError(value)) = bin().send(Box::new(value)) {
        drop(value);
    }
}

#[cfg(test)]
mod tests {
    /// A value handed over is dropped, on another thread.
    #[test]
    fn a_value_is_dropped_on_the_reclaim_thread() {
        struct Marker(std::sync::mpsc::Sender<std::thread::ThreadId>);
        impl Drop for Marker {
            fn drop(&mut self) {
                let _ = self.0.send(std::thread::current().id());
            }
        }
        let (tx, rx) = std::sync::mpsc::channel();
        super::later(Marker(tx));
        let dropped_on = rx.recv_timeout(std::time::Duration::from_secs(5)).expect("dropped");
        assert_ne!(dropped_on, std::thread::current().id());
    }
}
