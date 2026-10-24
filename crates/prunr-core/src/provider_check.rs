//! A model is run on a GPU provider only where the provider beats the
//! CPU at the CPU's quality, found out once per machine: the first time a
//! session of the model would land on the provider, the app's checker
//! runs it on both and the verdict is kept with the provider failures
//! (`ep_compat`), under the same scope (app, runtime build, drivers).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use crate::engine::{EpKind, Weights};

/// What a check of a model on a provider found: the provider and weights
/// its GPU side ran, and why the provider is passed over, if it is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Check {
    pub provider: String,
    pub weights: String,
    pub rejected: Option<String>,
}

/// Checks `model` on `provider` with `weights` against the CPU; `None`
/// when it cannot now (not enough memory for both sides, a model the
/// check does not cover).
pub type Checker = fn(prunr_models::ModelId, &str, Weights) -> Option<Check>;

static CHECKER: OnceLock<Checker> = OnceLock::new();
static OFF: AtomicBool = AtomicBool::new(false);

/// The app's checker, set once at start by every process that builds
/// sessions.
pub fn set_checker(checker: Checker) {
    let _ = CHECKER.set(checker);
}

/// No checks in this process: the check's own runs, and the bench.
pub fn turn_off() {
    OFF.store(true, Ordering::Relaxed);
}

/// Why a session of `model` with `weights` passes over `ep`: the check
/// against the CPU rejected it. Runs the check when the pair has no
/// verdict yet, once per process; another build of the same pair waits
/// for it and reads its verdict.
pub(crate) fn rejection(model: prunr_models::ModelId, weights: Weights, ep: EpKind) -> Option<String> {
    if OFF.load(Ordering::Relaxed) {
        return None;
    }
    rejection_by(*CHECKER.get()?, model, weights, ep)
}

fn rejection_by(checker: Checker, model: prunr_models::ModelId, weights: Weights, ep: EpKind) -> Option<String> {
    let pair = (ep, model, weights);
    let lock = {
        static PAIRS: Mutex<Option<HashMap<Pair, Arc<Mutex<bool>>>>> = Mutex::new(None);
        let mut pairs = PAIRS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        Arc::clone(pairs.get_or_insert_with(HashMap::new).entry(pair).or_default())
    };
    let mut asked = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    crate::ep_compat::refresh();
    if crate::ep_compat::has_passed(ep, model, weights) {
        return None;
    }
    if !std::mem::replace(&mut *asked, true) {
        tracing::info!(?model, ep = %ep, ?weights, "checking the provider against the CPU, once");
        if let Some(check) = checker(model, ep.as_str(), weights) {
            record(model, &check);
        }
    }
    let rejected = crate::ep_compat::failure(ep, model, weights)?;
    tracing::info!(?model, ep = %ep, ?weights, %rejected, "the provider is passed over");
    Some(rejected)
}

type Pair = (EpKind, prunr_models::ModelId, Weights);

fn record(model: prunr_models::ModelId, check: &Check) {
    let (Some(ep), Some(weights)) = (crate::engine::ep_named(&check.provider), Weights::labelled(&check.weights)) else {
        tracing::warn!(?model, ?check, "a check named no provider this build has");
        return;
    };
    match &check.rejected {
        None => crate::ep_compat::record_pass(ep, model, weights),
        Some(reason) => crate::ep_compat::record_failure(ep, model, weights, reason),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[cfg(not(target_os = "macos"))]
    const EP: EpKind = EpKind::Cuda;
    #[cfg(target_os = "macos")]
    const EP: EpKind = EpKind::CoreMl;

    static CHECKS: AtomicUsize = AtomicUsize::new(0);

    fn passes(_: prunr_models::ModelId, provider: &str, weights: Weights) -> Option<Check> {
        CHECKS.fetch_add(1, Ordering::Relaxed);
        Some(Check { provider: provider.into(), weights: weights.label().into(), rejected: None })
    }

    fn slower(_: prunr_models::ModelId, provider: &str, weights: Weights) -> Option<Check> {
        Some(Check { provider: provider.into(), weights: weights.label().into(), rejected: Some("it runs slower there".into()) })
    }

    /// The check runs once for a pair; a pass is kept and lets the
    /// provider run, a rejection is kept and passes the provider over.
    #[test]
    fn a_provider_is_checked_once_and_its_verdict_kept() {
        let model = prunr_models::ModelId::DepthAnythingV2Small;
        assert_eq!(rejection_by(passes, model, Weights::Fp16, EP), None);
        assert_eq!(rejection_by(passes, model, Weights::Fp16, EP), None);
        assert_eq!(CHECKS.load(Ordering::Relaxed), 1, "checked once");
        assert!(crate::ep_compat::has_passed(EP, model, Weights::Fp16));

        let other = prunr_models::ModelId::Migan;
        assert_eq!(rejection_by(slower, other, Weights::Fp16, EP).as_deref(), Some("it runs slower there"));
        assert!(crate::ep_compat::is_known_failure(EP, other, Weights::Fp16));
    }
}
