//! A model is run on a GPU provider only where the provider beats the
//! CPU at the CPU's quality, found out once per machine: the first time a
//! session of the model would land on the provider, the app's checker
//! runs it on both and the verdict is kept with the provider failures
//! (`ep_compat`), under the same scope (app, runtime build, drivers).

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use crate::engine::{EpKind, Weights};

/// What a check of a model on a provider found: the provider and weights
/// its GPU side ran, and why the provider is passed over, if it is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Check {
    pub provider: String,
    pub weights: String,
    pub rejected: Option<Rejection>,
}

/// Why a check passes a provider over: in plain words for the window,
/// and in full (the provider's own error, the numbers) for the log and
/// the failure store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rejection {
    pub plain: String,
    pub detail: String,
}

impl Rejection {
    /// A reason plain enough for the window as it is.
    pub fn same(reason: String) -> Self {
        Self { plain: reason.clone(), detail: reason }
    }
}

/// Checks `model` on `provider` with `weights` against the CPU; `None`
/// when it cannot now (not enough memory for both sides, a model the
/// check does not cover).
pub type Checker = fn(prunr_models::ModelId, &str, Weights) -> Option<Check>;

static CHECKER: OnceLock<Checker> = OnceLock::new();

/// What the person using the app is told: a check starting (it takes
/// seconds, once), and a provider passed over.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Notice {
    /// `on`: where the model is checked, as the window names it.
    Checking { model: prunr_models::ModelId, on: String },
    PassedOver { model: prunr_models::ModelId, reason: String },
}

impl Notice {
    /// What the person using the app is told.
    pub fn text(&self) -> String {
        match self {
            Notice::Checking { model, on } => {
                format!("Checking {on} with {}, once. This takes a few seconds.", prunr_models::display_name(*model))
            }
            Notice::PassedOver { model, reason } => format!("{} runs on the CPU here: {reason}.", prunr_models::display_name(*model)),
        }
    }
}

static OBSERVER: OnceLock<fn(Notice)> = OnceLock::new();

/// Where this process sends its notices: the window, the worker's link
/// to it, or the command line.
pub fn set_observer(observer: fn(Notice)) {
    let _ = OBSERVER.set(observer);
}

/// `notice` to this process's observer: its own, or one a worker it
/// runs relayed.
pub fn tell(notice: Notice) {
    if let Some(observer) = OBSERVER.get() {
        observer(notice);
    }
}

/// The app's checker, set once at start by every process that builds
/// sessions for use; without one (the bench, the check's own process)
/// nothing is checked.
pub fn set_checker(checker: Checker) {
    let _ = CHECKER.set(checker);
}

/// Why a session of `model` with `weights` passes over `ep`: the check
/// against the CPU rejected it. Runs the check when the pair has no
/// verdict yet, once per process; another build of the same pair waits
/// for it and reads its verdict.
pub(crate) fn rejection(model: prunr_models::ModelId, weights: Weights, ep: EpKind) -> Option<String> {
    rejection_by(*CHECKER.get()?, model, weights, ep)
}

type Pair = (EpKind, prunr_models::ModelId, Weights);

/// Each pair's lock, holding whether this process asked for its check.
static PAIRS: LazyLock<Mutex<HashMap<Pair, Arc<Mutex<bool>>>>> = LazyLock::new(Mutex::default);

fn rejection_by(checker: Checker, model: prunr_models::ModelId, weights: Weights, ep: EpKind) -> Option<String> {
    let lock = Arc::clone(PAIRS.lock().unwrap_or_else(std::sync::PoisonError::into_inner).entry((ep, model, weights)).or_default());
    let mut asked = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    crate::ep_compat::refresh();
    if crate::ep_compat::has_passed(ep, model, weights) {
        return None;
    }
    if !std::mem::replace(&mut *asked, true) {
        tracing::info!(?model, ep = %ep, ?weights, "checking the provider against the CPU, once");
        tell(Notice::Checking { model, on: ep.checked_on().to_string() });
        if let Some(check) = checker(model, ep.as_str(), weights) {
            record(model, ep, weights, check);
        }
    }
    let rejected = crate::ep_compat::failure(ep, model, weights)?;
    tracing::info!(?model, ep = %ep, ?weights, %rejected, "the provider is passed over");
    Some(rejected)
}

/// `check`'s verdict for the pair asked, when its GPU side ran that pair
/// (its ladder walks the same providers, failures and weights in the same
/// order, so it does unless one changed meanwhile).
fn record(model: prunr_models::ModelId, ep: EpKind, weights: Weights, check: Check) {
    if check.provider != ep.as_str() || check.weights != weights.label() {
        tracing::warn!(?model, ep = %ep, ?weights, ?check, "the check ran another provider; nothing recorded");
        return;
    }
    match check.rejected {
        None => crate::ep_compat::record_pass(ep, model, weights),
        Some(rejection) => {
            crate::ep_compat::record_failure(ep, model, weights, &rejection.detail);
            tell(Notice::PassedOver { model, reason: rejection.plain });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn a_notice_names_the_model_and_the_reason() {
        let model = prunr_models::ModelId::DdcolorTiny;
        let on = |on: &str| Notice::Checking { model, on: on.into() }.text();
        assert_eq!(on(crate::engine::GRAPHICS_CARD), "Checking the graphics card with DDColor, once. This takes a few seconds.");
        assert_eq!(on("OpenVINO"), "Checking OpenVINO with DDColor, once. This takes a few seconds.");
        let passed_over = Notice::PassedOver { model, reason: "it runs slower on CoreML than on the CPU (950 ms against 1000 ms)".into() };
        assert_eq!(passed_over.text(), "DDColor runs on the CPU here: it runs slower on CoreML than on the CPU (950 ms against 1000 ms).");
    }

    #[cfg(not(target_os = "macos"))]
    const EP: EpKind = EpKind::Cuda;
    #[cfg(target_os = "macos")]
    const EP: EpKind = EpKind::CoreMl;

    static CHECKS: AtomicUsize = AtomicUsize::new(0);
    static TOLD: Mutex<Vec<Notice>> = Mutex::new(Vec::new());

    fn tell(notice: Notice) {
        TOLD.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(notice);
    }

    fn passes(_: prunr_models::ModelId, provider: &str, weights: Weights) -> Option<Check> {
        CHECKS.fetch_add(1, Ordering::Relaxed);
        Some(Check { provider: provider.into(), weights: weights.label().into(), rejected: None })
    }

    fn slower(_: prunr_models::ModelId, provider: &str, weights: Weights) -> Option<Check> {
        let rejected = Rejection { plain: "it runs slower there".into(), detail: "it runs slower there (950 ms against 1000 ms)".into() };
        Some(Check { provider: provider.into(), weights: weights.label().into(), rejected: Some(rejected) })
    }

    /// The check runs once for a pair and says so; a pass is kept and
    /// lets the provider run, a rejection is kept in full, said plainly,
    /// and passes the provider over.
    #[test]
    fn a_provider_is_checked_once_and_its_verdict_kept() {
        set_observer(tell);
        let model = prunr_models::ModelId::DepthAnythingV2Small;
        assert_eq!(rejection_by(passes, model, Weights::Fp16, EP), None);
        assert_eq!(rejection_by(passes, model, Weights::Fp16, EP), None);
        assert_eq!(CHECKS.load(Ordering::Relaxed), 1, "checked once");
        assert!(crate::ep_compat::has_passed(EP, model, Weights::Fp16));

        let other = prunr_models::ModelId::Migan;
        assert_eq!(rejection_by(slower, other, Weights::Fp16, EP).as_deref(), Some("it runs slower there (950 ms against 1000 ms)"));
        assert!(crate::ep_compat::is_known_failure(EP, other, Weights::Fp16));
        let told = TOLD.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        assert_eq!(told, [
            Notice::Checking { model, on: crate::engine::GRAPHICS_CARD.into() },
            Notice::Checking { model: other, on: crate::engine::GRAPHICS_CARD.into() },
            Notice::PassedOver { model: other, reason: "it runs slower there".into() },
        ]);
    }
}
