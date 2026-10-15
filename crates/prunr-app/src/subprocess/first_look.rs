//! The worker's first look at the subject of the image on screen: the
//! fast model's session, kept beside the worker's own engines while first
//! looks are on and released with them, loaded when a run first takes one.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use prunr_core::{ModelKind, OrtEngine};

/// No image.
const NONE: u64 = u64::MAX;

/// An image id, or none, read and set from several threads: the image on
/// screen, when its run would show a first look.
#[derive(Clone, Debug)]
pub struct Shown(Arc<AtomicU64>);

impl Default for Shown {
    fn default() -> Self {
        Self(Arc::new(AtomicU64::new(NONE)))
    }
}

impl Shown {
    pub fn get(&self) -> Option<u64> {
        Some(self.0.load(Ordering::Acquire)).filter(|&id| id != NONE)
    }

    /// Set it; returns what it was.
    pub fn set(&self, item: Option<u64>) -> Option<u64> {
        Some(self.0.swap(item.unwrap_or(NONE), Ordering::AcqRel)).filter(|&id| id != NONE)
    }
}

type Session = Arc<Mutex<Option<Arc<OrtEngine>>>>;

/// Which image's run takes a first look, and the session it takes it
/// with. Clones share both.
#[derive(Clone, Default)]
pub struct FirstLook {
    shown: Shown,
    session: Session,
}

impl FirstLook {
    /// The image on screen whose run would show a first look; `None`
    /// (first looks off) lets the session go.
    pub fn show_for(&self, item: Option<u64>) {
        self.shown.set(item);
        if item.is_none() {
            *lock(&self.session) = None;
        }
    }

    /// The first look a run of `item_id` on `model`, starting now beside a
    /// pool of `pool_size` engines, takes: only when its image is the one
    /// on screen as it starts. A run already under way never starts one.
    pub fn for_run(&self, item_id: u64, model: ModelKind, pool_size: usize) -> Option<Look> {
        (self.shown.get() == Some(item_id) && prunr_core::subject::may_take_first_look(model)).then(|| Look {
            item_id,
            shown: self.shown.clone(),
            session: Arc::clone(&self.session),
            pool_size,
        })
    }

    /// Whether the first-look model is loaded.
    pub fn is_loaded(&self) -> bool {
        lock(&self.session).is_some()
    }
}

/// One run's first look, still to be taken.
pub struct Look {
    item_id: u64,
    shown: Shown,
    session: Session,
    pool_size: usize,
}

impl Look {
    /// Whether its image is still the one on screen: one switched away
    /// from drops its look as stale.
    pub fn is_current(&self) -> bool {
        self.shown.get() == Some(self.item_id)
    }

    /// The first-look model's engine, loaded on the first run that needs
    /// it, while the look is current; `None` once it is stale, or when the
    /// model fails to load and the run goes on without it.
    pub fn engine(&self) -> Option<Arc<OrtEngine>> {
        if !self.is_current() {
            return None;
        }
        let mut slot = lock(&self.session);
        if slot.is_none() {
            let started = std::time::Instant::now();
            match prunr_core::subject::first_look_engine(self.pool_size) {
                Ok(engine) => {
                    tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "worker: first-look model loaded");
                    *slot = Some(Arc::new(engine));
                }
                Err(err) => tracing::warn!(%err, "worker: first-look model failed to load; the run goes on without it"),
            }
        }
        slot.clone()
    }
}

fn lock(session: &Session) -> MutexGuard<'_, Option<Arc<OrtEngine>>> {
    session.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shown_holds_one_image_or_none() {
        let shown = Shown::default();
        assert_eq!(shown.get(), None);
        assert_eq!(shown.set(Some(0)), None);
        assert_eq!(shown.clone().get(), Some(0), "clones share it");
        assert_eq!(shown.set(None), Some(0));
        assert_eq!(shown.get(), None);
    }

    /// The first-look model loads on the run that takes the look, is
    /// shared by later ones, and goes as soon as first looks are off; a
    /// look switched away from loads nothing.
    #[test]
    fn the_first_look_model_loads_on_need_and_goes_when_looks_are_off() {
        let look = FirstLook::default();
        look.show_for(Some(4));
        let stale = look.for_run(4, ModelKind::BiRefNetLite, 1).expect("the shown image's run");
        look.show_for(Some(5));
        assert!(stale.engine().is_none() && !look.is_loaded(), "switched away before the look: nothing loads");

        let run = look.for_run(5, ModelKind::BiRefNetLite, 1).expect("the shown image's run");
        let engine = run.engine().expect("the first-look model loads");
        assert_eq!(engine.model_kind(), prunr_core::subject::FIRST_LOOK_MODEL);
        assert!(look.is_loaded());
        let again = look.for_run(5, ModelKind::U2net, 1).and_then(|r| r.engine()).expect("loaded");
        assert!(Arc::ptr_eq(&engine, &again), "one session per worker");
        look.show_for(None);
        assert!(!look.is_loaded(), "no memory held once first looks are off");
    }
}
