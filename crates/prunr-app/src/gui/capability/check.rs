use super::facts::{Activity, ItemFacts, Lines, Photo};
use crate::gui::item::Engine;
use crate::gui::knob_catalog::ModelRole;
use prunr_core::LineMode;

/// What a control or a job needs from the image before it can act.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Need {
    /// The pixels on screen: the photo, or the result.
    ShownImage,
    /// A model whose result is the photo re-rendered (a cut-out model or
    /// No model): what a Mask, Edge or Photo render draws for.
    PhotoRender,
    /// The cut-out's mask under the image on screen: Mask, Outline &
    /// Shadow, a brush re-cut.
    Cutout,
    /// A subject to restyle: the cut-out, or No model's photo; never
    /// while Sketch covers the whole photo.
    Subject,
    /// Something transparent to fill behind: a cut on screen, or lines
    /// on No model's photo.
    Transparency,
    /// Lines at every scale the settings draw.
    DrawnLines,
    /// The run a model's settings re-run: a cut on screen for a cut-out
    /// model; No model's photo is its own.
    Processed,
    /// The upscale run the image shows, for Refine.
    KeptUpscale,
    /// The colorize run the image shows, for Strength.
    KeptColorize,
    PaintedRegion,
    /// No run of this image under way.
    Idle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    Ready,
    /// Becomes ready by itself: keep the request.
    Wait(Reason),
    /// Not until the user acts: refuse, and say why.
    Never(Reason),
}

impl Verdict {
    pub(crate) fn is_ready(self) -> bool {
        self == Verdict::Ready
    }

    /// Work may be asked for: ready now, or once it waits out its reason
    /// (the queue keeps a request that waits).
    pub(crate) fn allows_request(self) -> bool {
        !matches!(self, Verdict::Never(_))
    }

    /// Why it cannot act now, if it cannot.
    pub(crate) fn reason(self) -> Option<Reason> {
        match self {
            Verdict::Ready => None,
            Verdict::Wait(r) | Verdict::Never(r) => Some(r),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reason {
    Decoding,
    Unreadable,
    NotForModel,
    ProcessFirst,
    /// The kept cut-out is gone (memory, another model's step): only a
    /// new Process brings it back.
    ProcessAgain,
    Processing,
    Queued,
    LinesComing,
    LinesNotRun,
    UpscaleFirst,
    ColorizeFirst,
    PaintFirst,
    Running,
    Removed,
    SketchCoversPhoto,
    NoSubject,
    NothingTransparent,
    /// A selection edit (a stroke, Invert, a fold into the cut-out) is
    /// still being made.
    FinishingSelection,
}

impl Reason {
    pub(crate) fn text(self) -> &'static str {
        match self {
            Reason::Decoding => "Loading the image.",
            Reason::Unreadable => "The image could not be read.",
            Reason::NotForModel => "Not available for this model.",
            Reason::ProcessFirst => "Process first.",
            Reason::ProcessAgain => "Process again to adjust the cut-out.",
            Reason::Processing => "Processing.",
            Reason::Queued => "Waiting to process.",
            Reason::LinesComing => "Finding the lines.",
            Reason::LinesNotRun => "The lines are found when Sketch runs.",
            Reason::UpscaleFirst => "Upscale first.",
            Reason::ColorizeFirst => "Colorize first.",
            Reason::PaintFirst => "Paint the area first.",
            Reason::Running => "Working on this image.",
            Reason::Removed => "The image was removed.",
            Reason::SketchCoversPhoto => "Sketch is on Whole photo, so nothing is cut out.",
            Reason::NoSubject => "Sketch is on Whole photo, so there is no subject to restyle.",
            Reason::NothingTransparent => "Nothing is transparent in this mode, so there is no background to fill.",
            Reason::FinishingSelection => "Finishing the selection.",
        }
    }
}

/// Whether `need` is met for an image with `facts` under a model of `role`.
pub(crate) fn check(facts: &ItemFacts, need: Need, role: ModelRole) -> Verdict {
    match need {
        Need::ShownImage => match (facts.shown.base_run(), facts.photo) {
            (Some(_), _) | (None, Photo::Decoded) => Verdict::Ready,
            (None, Photo::Decoding) => Verdict::Wait(Reason::Decoding),
            (None, Photo::Unreadable) => Verdict::Never(Reason::Unreadable),
        },
        Need::PhotoRender if matches!(role, ModelRole::Cutout | ModelRole::Filters) => Verdict::Ready,
        Need::PhotoRender => Verdict::Never(Reason::NotForModel),
        Need::Cutout if facts.line_mode == LineMode::EdgesOnly => Verdict::Never(Reason::SketchCoversPhoto),
        // No model's photo is its own result: what cut-out controls act on.
        Need::Cutout if role == ModelRole::Filters => Verdict::Ready,
        Need::Subject if facts.line_mode == LineMode::EdgesOnly => Verdict::Never(Reason::NoSubject),
        Need::Subject => check(facts, Need::Cutout, role),
        Need::Transparency if role == ModelRole::Filters && facts.line_mode == LineMode::Off => Verdict::Never(Reason::NothingTransparent),
        Need::Transparency => check(facts, Need::Processed, role),
        Need::Cutout => {
            if role != ModelRole::Cutout {
                return Verdict::Never(Reason::NotForModel);
            }
            match facts.activity {
                Activity::Running(Engine::Segment) => return Verdict::Wait(Reason::Processing),
                Activity::Queued => return Verdict::Wait(Reason::Queued),
                _ => {}
            }
            match facts.shown.base_run() {
                Some(Engine::Segment) if facts.cutout_mask => Verdict::Ready,
                Some(Engine::Segment) => Verdict::Never(Reason::ProcessAgain),
                _ => Verdict::Never(Reason::ProcessFirst),
            }
        }
        Need::Processed => match (role, facts.shown.base_run()) {
            (ModelRole::Filters, _) => check(facts, Need::ShownImage, role),
            (ModelRole::Cutout, Some(Engine::Segment)) => Verdict::Ready,
            (ModelRole::Cutout, _) => Verdict::Never(Reason::ProcessFirst),
            _ => Verdict::Never(Reason::NotForModel),
        },
        Need::DrawnLines => match facts.lines {
            Lines::Ready => Verdict::Ready,
            Lines::Coming => Verdict::Wait(Reason::LinesComing),
            Lines::Missing => Verdict::Never(Reason::LinesNotRun),
        },
        Need::KeptUpscale => kept(facts.kept_upscale, facts.activity, Engine::Upscale, Reason::UpscaleFirst),
        Need::KeptColorize => kept(facts.kept_colorize, facts.activity, Engine::Colorize, Reason::ColorizeFirst),
        Need::PaintedRegion if facts.painted => Verdict::Ready,
        Need::PaintedRegion => Verdict::Never(Reason::PaintFirst),
        Need::Idle => match facts.activity {
            Activity::Idle => Verdict::Ready,
            Activity::Queued => Verdict::Wait(Reason::Queued),
            Activity::Running(_) => Verdict::Wait(Reason::Running),
        },
    }
}

fn kept(present: bool, activity: Activity, engine: Engine, missing: Reason) -> Verdict {
    if present {
        Verdict::Ready
    } else if activity == Activity::Running(engine) {
        Verdict::Wait(Reason::Running)
    } else {
        Verdict::Never(missing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::capability::facts::Shown;
    use ModelRole::*;

    fn cut(cutout_mask: bool) -> ItemFacts {
        ItemFacts { shown: Shown::Run(Engine::Segment), cutout_mask, ..ItemFacts::photo() }
    }

    #[test]
    fn the_cutout_is_there_only_under_a_cut_its_mask_made() {
        let rows: [(&str, ItemFacts, ModelRole, Verdict); 10] = [
            ("unprocessed photo", ItemFacts::photo(), Cutout, Verdict::Never(Reason::ProcessFirst)),
            ("cut with its mask", cut(true), Cutout, Verdict::Ready),
            ("cut, mask evicted", cut(false), Cutout, Verdict::Never(Reason::ProcessAgain)),
            ("hand edit on a cut", ItemFacts { shown: Shown::Edited { on: Some(Engine::Segment) }, cutout_mask: true, ..ItemFacts::photo() }, Cutout, Verdict::Ready),
            ("Delete on the photo", ItemFacts { shown: Shown::Edited { on: None }, ..ItemFacts::photo() }, Cutout, Verdict::Never(Reason::ProcessFirst)),
            ("upscale shown, old mask kept", ItemFacts { shown: Shown::Run(Engine::Upscale), cutout_mask: false, ..ItemFacts::photo() }, Cutout, Verdict::Never(Reason::ProcessFirst)),
            ("processing", ItemFacts { activity: Activity::Running(Engine::Segment), ..ItemFacts::photo() }, Cutout, Verdict::Wait(Reason::Processing)),
            ("waiting for a slot", ItemFacts { activity: Activity::Queued, ..cut(true) }, Cutout, Verdict::Wait(Reason::Queued)),
            ("No model: the photo is the result", ItemFacts::photo(), Filters, Verdict::Ready),
            ("an upscaler", cut(true), Upscale, Verdict::Never(Reason::NotForModel)),
        ];
        for (name, facts, role, want) in rows {
            assert_eq!(check(&facts, Need::Cutout, role), want, "{name}");
        }
    }

    #[test]
    fn the_shown_image_waits_only_for_the_photos_decode() {
        let rows = [
            ("decoded photo", ItemFacts::photo(), Verdict::Ready),
            ("decoding", ItemFacts { photo: Photo::Decoding, ..ItemFacts::photo() }, Verdict::Wait(Reason::Decoding)),
            ("unreadable", ItemFacts { photo: Photo::Unreadable, ..ItemFacts::photo() }, Verdict::Never(Reason::Unreadable)),
            ("result, photo not decoded", ItemFacts { photo: Photo::Decoding, ..cut(true) }, Verdict::Ready),
        ];
        for (name, facts, want) in rows {
            assert_eq!(check(&facts, Need::ShownImage, Filters), want, "{name}");
        }
    }

    #[test]
    fn kept_runs_wait_for_their_own_run_and_refuse_otherwise() {
        let up = |kept_upscale, activity| ItemFacts { kept_upscale, activity, ..ItemFacts::photo() };
        assert_eq!(check(&up(true, Activity::Idle), Need::KeptUpscale, Upscale), Verdict::Ready);
        assert_eq!(check(&up(false, Activity::Running(Engine::Upscale)), Need::KeptUpscale, Upscale), Verdict::Wait(Reason::Running));
        assert_eq!(check(&up(false, Activity::Idle), Need::KeptUpscale, Upscale), Verdict::Never(Reason::UpscaleFirst));
        assert_eq!(check(&up(false, Activity::Running(Engine::Colorize)), Need::KeptUpscale, Upscale), Verdict::Never(Reason::UpscaleFirst));
        let col = ItemFacts { activity: Activity::Running(Engine::Colorize), ..ItemFacts::photo() };
        assert_eq!(check(&col, Need::KeptColorize, Colorize), Verdict::Wait(Reason::Running));
    }

    #[test]
    fn processed_means_a_cut_on_screen_or_no_models_photo() {
        assert_eq!(check(&ItemFacts::photo(), Need::Processed, Cutout), Verdict::Never(Reason::ProcessFirst));
        assert_eq!(check(&cut(false), Need::Processed, Cutout), Verdict::Ready, "a cut whose mask is evicted re-runs");
        assert_eq!(check(&ItemFacts::photo(), Need::Processed, Filters), Verdict::Ready);
        assert_eq!(check(&ItemFacts { photo: Photo::Decoding, ..ItemFacts::photo() }, Need::Processed, Filters), Verdict::Wait(Reason::Decoding));
    }

    #[test]
    fn lines_wait_while_the_worker_makes_them() {
        let lines = |lines| ItemFacts { lines, ..ItemFacts::photo() };
        assert_eq!(check(&lines(Lines::Ready), Need::DrawnLines, Cutout), Verdict::Ready);
        assert_eq!(check(&lines(Lines::Coming), Need::DrawnLines, Cutout), Verdict::Wait(Reason::LinesComing));
        assert_eq!(check(&lines(Lines::Missing), Need::DrawnLines, Filters), Verdict::Never(Reason::LinesNotRun));
    }

    #[test]
    fn idle_and_painted() {
        assert_eq!(check(&ItemFacts::photo(), Need::Idle, Erase), Verdict::Ready);
        assert_eq!(check(&ItemFacts { activity: Activity::Queued, ..ItemFacts::photo() }, Need::Idle, Erase), Verdict::Wait(Reason::Queued));
        assert_eq!(check(&ItemFacts { activity: Activity::Running(Engine::Erase), ..ItemFacts::photo() }, Need::Idle, Erase), Verdict::Wait(Reason::Running));
        assert_eq!(check(&ItemFacts::photo(), Need::PaintedRegion, Erase), Verdict::Never(Reason::PaintFirst));
        assert_eq!(check(&ItemFacts { painted: true, ..ItemFacts::photo() }, Need::PaintedRegion, Erase), Verdict::Ready);
    }

    #[test]
    fn every_reason_has_text() {
        use Reason::*;
        for r in [Decoding, Unreadable, NotForModel, ProcessFirst, ProcessAgain, Processing, Queued, LinesComing, LinesNotRun, UpscaleFirst, ColorizeFirst, PaintFirst, Running, Removed, SketchCoversPhoto, NoSubject, NothingTransparent] {
            assert!(r.text().ends_with('.'), "{r:?}");
        }
    }
}
