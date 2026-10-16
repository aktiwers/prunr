use crate::gui::item::{BatchItem, Engine};

/// What the image on screen is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shown {
    /// The unprocessed photo.
    Photo,
    /// A run's output: the shown step's own origin.
    Run(Engine),
    /// A hand edit (stroke, Delete, Cut) on top of `on`, the latest run
    /// before it (`None`: on the photo).
    Edited { on: Option<Engine> },
}

impl Shown {
    /// The run whose output the image builds on, if any.
    pub(crate) fn base_run(self) -> Option<Engine> {
        match self {
            Shown::Photo => None,
            Shown::Run(engine) => Some(engine),
            Shown::Edited { on } => on,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Activity {
    Idle,
    /// Waiting for a worker slot.
    Queued,
    Running(Engine),
}

/// The decoded photo the image starts from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Photo {
    Decoded,
    /// Decoding, or about to be.
    Decoding,
    /// The source could not be read.
    Unreadable,
}

/// The line detector's output for the scales the settings draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lines {
    Ready,
    /// The worker is still making the missing scales.
    Coming,
    /// Only a new run of the line detector makes them.
    Missing,
}

/// Everything the capability check reads about one image. Cheap to build:
/// no decode, no hash, no copy of pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ItemFacts {
    pub(crate) shown: Shown,
    pub(crate) activity: Activity,
    pub(crate) photo: Photo,
    /// The kept segmentation made the cut-out the image shows.
    pub(crate) cutout_mask: bool,
    pub(crate) lines: Lines,
    pub(crate) kept_upscale: bool,
    pub(crate) kept_colorize: bool,
    pub(crate) painted: bool,
    pub(crate) failed: bool,
}

impl ItemFacts {
    pub(crate) fn of(item: &BatchItem) -> Self {
        let current = item.timeline.current();
        let shown = match (&current.result, &current.origin) {
            (None, _) => Shown::Photo,
            (Some(_), Some(origin)) => Shown::Run(origin.engine),
            (Some(_), None) => Shown::Edited { on: item.timeline.latest_run().map(|o| o.engine) },
        };
        let activity = match &item.job {
            Some(job) if job.running => Activity::Running(job.engine),
            Some(_) => Activity::Queued,
            None => Activity::Idle,
        };
        let failed = item.error().is_some();
        let photo = if item.source_rgba.is_some() || item.source_dyn.is_some() {
            Photo::Decoded
        } else if failed && !item.decode_pending {
            Photo::Unreadable
        } else {
            Photo::Decoding
        };
        let lines = match &item.cached_edge_tensors {
            Some(kept) if kept.has_all(item.settings.edge_settings().drawn_scales()) => Lines::Ready,
            Some(kept) if kept.awaits_spare() => Lines::Coming,
            _ => Lines::Missing,
        };
        Self {
            shown,
            activity,
            photo,
            cutout_mask: shown.base_run() == Some(Engine::Segment) && item.tensor_made_shown_step(),
            lines,
            kept_upscale: item.kept_upscale().is_some(),
            kept_colorize: item.kept_colorize().is_some(),
            painted: item.has_painted_region(),
            failed,
        }
    }
}

#[cfg(test)]
impl ItemFacts {
    /// An unprocessed, decoded photo with nothing kept: tests change the
    /// fields they are about.
    pub(crate) fn photo() -> Self {
        Self {
            shown: Shown::Photo,
            activity: Activity::Idle,
            photo: Photo::Decoded,
            cutout_mask: false,
            lines: Lines::Missing,
            kept_upscale: false,
            kept_colorize: false,
            painted: false,
            failed: false,
        }
    }
}
