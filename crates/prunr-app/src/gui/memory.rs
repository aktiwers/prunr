//! Memory-aware admission for batch processing: images are admitted
//! best-fit against a RAM budget, and each release frees room for more.

use std::collections::HashMap;

use prunr_core::ModelKind;

pub struct ImageMemCost {
    pub item_id: u64,
    /// Estimated bytes while in flight.
    pub total: usize,
}

/// Tracks which items are currently admitted (in the processing window)
/// and enforces a RAM budget.
pub struct AdmissionController {
    budget_bytes: usize,
    committed_bytes: usize,
    admitted: HashMap<u64, usize>,
    /// Pending items sorted descending by cost (largest first for best-fit).
    pending: Vec<ImageMemCost>,
}

/// Compressed undo states per image the admission estimate counts in
/// RAM; older ones spill to disk under pressure.
pub(crate) const RESIDENT_HISTORY_STEPS: usize = 10;

impl AdmissionController {
    /// Budget: available RAM less the reserve kept for other apps and the
    /// batch's engines.
    pub fn new(engines: CutoutEngines) -> Self {
        let available = crate::hardware::available_ram_bytes_now() as usize;
        let reserve = (prunr_core::memory::RAM_RESERVE_MB * 1024 * 1024) as usize;
        let budget = available.saturating_sub(reserve).saturating_sub(engines.need_bytes());
        Self {
            budget_bytes: budget,
            committed_bytes: 0,
            admitted: HashMap::new(),
            pending: Vec::new(),
        }
    }

    /// Cost = `2 × rgba_size + file_bytes + history_residue`, the residue
    /// at zstd-1's ~2:1 on RGBA per resident undo state. Deliberately
    /// pessimistic: without the history term long batches with deep
    /// history OOM.
    pub fn estimate_cost(
        item_id: u64,
        dimensions: (u32, u32),
        file_bytes_len: usize,
        history_depth: usize,
    ) -> ImageMemCost {
        let pixels = dimensions.0 as usize * dimensions.1 as usize;
        let rgba_size = pixels * 4;
        let history_residue = (rgba_size / 2) * history_depth;
        let total = rgba_size * 2 + file_bytes_len + history_residue;
        ImageMemCost { item_id, total }
    }

    /// A controller with a fixed budget, for tests that must not read
    /// the machine's RAM or the active provider.
    #[cfg(test)]
    pub(crate) fn with_budget(budget_bytes: usize) -> Self {
        Self { budget_bytes, committed_bytes: 0, admitted: HashMap::new(), pending: Vec::new() }
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn enqueue(&mut self, costs: Vec<ImageMemCost>) {
        self.pending.extend(costs);
        self.pending.sort_by(|a, b| b.total.cmp(&a.total));
    }

    /// The largest pending image that fits the remaining budget.
    pub fn try_admit_next(&mut self) -> Option<u64> {
        let remaining = self.budget_bytes.saturating_sub(self.committed_bytes);

        if let Some(idx) = self.pending.iter().position(|c| c.total <= remaining) {
            let cost = self.pending.remove(idx);
            self.committed_bytes += cost.total;
            self.admitted.insert(cost.item_id, cost.total);
            return Some(cost.item_id);
        }

        // Nothing fits. If nothing is in-flight, force-admit the smallest
        // to guarantee forward progress (avoids deadlock).
        if self.admitted.is_empty() && !self.pending.is_empty() {
            // invariant: !pending.is_empty() checked in the guard above.
            let cost = self.pending.pop().unwrap(); // pop last = smallest
            self.committed_bytes += cost.total;
            self.admitted.insert(cost.item_id, cost.total);
            return Some(cost.item_id);
        }

        None // wait for in-flight items to release budget
    }

    pub fn release(&mut self, item_id: u64) {
        if let Some(cost) = self.admitted.remove(&item_id) {
            self.committed_bytes = self.committed_bytes.saturating_sub(cost);
        }
    }

    /// The images still waiting for room, not admitted.
    pub(crate) fn pending_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.pending.iter().map(|c| c.item_id)
    }

    pub fn is_complete(&self) -> bool {
        self.pending.is_empty() && self.admitted.is_empty()
    }
}

/// Memory short: the system's own word where it gives one (macOS), else
/// available RAM below 20% of total. Undo history then spills to disk.
pub fn under_memory_pressure() -> bool {
    if let Some(pressed) = prunr_core::sysmem::system_says_pressure() {
        return pressed;
    }
    let total = crate::hardware::total_ram_bytes();
    let available = crate::hardware::available_ram_bytes_throttled();
    if total == 0 { return false; }
    (available as f64 / total as f64) < 0.20
}

/// Per inference slot: session weights, ORT workspace, tensors and image
/// buffers in system RAM, as the model needs them where it will run (on
/// a card only its share of system RAM). Until the GPU probe answers, the
/// CPU's figure.
fn per_engine_cost(model: ModelKind, cpu_only: bool) -> usize {
    let id: prunr_models::ModelId = model.into();
    let provider = prunr_core::memory::provider_now(id, cpu_only);
    engine_cost_on(model, provider)
}

fn engine_cost_on(model: ModelKind, provider: &str) -> usize {
    cost_on(model.into(), provider)
}

/// System RAM `id` takes on `provider`, in bytes.
fn cost_on(id: prunr_models::ModelId, provider: &str) -> usize {
    let card_free = crate::hardware::card_free_mb_throttled(provider);
    // unreachable fallback: every model has a descriptor.
    let mb = prunr_core::memory::charge_with(id, provider, card_free).map_or(800, |c| c.ram_mb);
    mb as usize * 1024 * 1024
}

pub(crate) fn cpu_need_mb(model: ModelKind) -> u32 {
    let id: prunr_models::ModelId = model.into();
    // unreachable fallback: every ModelKind has a descriptor.
    prunr_models::descriptor(id).map_or(800, |d| d.needs.host_cpu().host_mb)
}

/// Whether free RAM bounds what a cut-out may take here. macOS makes room
/// by compressing other apps' memory, so its free figure is not that
/// bound: BiRefNet-lite peaks at 12.9 GB and runs on a 16 GB Mac. There
/// the system's own memory management stays in charge, as before.
const FREE_RAM_BOUNDS_A_CUTOUT: bool = !cfg!(target_os = "macos");

/// Free RAM for a cut-out worker's watchdog, where free RAM bounds a run
/// (`FREE_RAM_BOUNDS_A_CUTOUT`); `None` lets it run on.
pub fn cutout_watch_free_ram() -> Option<u64> {
    FREE_RAM_BOUNDS_A_CUTOUT.then(crate::hardware::watchdog_free_ram_bytes).flatten()
}

/// The engines a cut-out worker builds for a batch.
#[derive(Clone, Copy, Debug)]
pub struct CutoutEngines {
    pub model: ModelKind,
    pub jobs: usize,
    pub cpu_only: bool,
    /// The first-look model may load beside them.
    pub first_look: bool,
    pub line_mode: prunr_core::LineMode,
    /// Lines fade with depth: the depth model loads beside them.
    pub line_fade: bool,
}

impl CutoutEngines {
    /// System RAM they take: the cut-out engines unless lines alone run,
    /// with the first look beside them (on the CPU), and with lines on the
    /// line model and, for a fade, the depth model.
    pub(crate) fn need_bytes(self) -> usize {
        use prunr_core::LineMode;
        let cuts = self.line_mode != LineMode::EdgesOnly;
        let looks = cuts && self.first_look && prunr_core::subject::may_take_first_look(self.model);
        let lines = self.line_mode != LineMode::Off;
        let fades = lines && self.line_fade;
        let edge = prunr_models::ModelId::Teed;
        let depth = prunr_core::depth::MODEL;
        usize::from(cuts) * model_overhead(self.model, self.jobs, self.cpu_only)
            + usize::from(looks) * per_engine_cost(prunr_core::subject::FIRST_LOOK_MODEL, true)
            + usize::from(lines) * cost_on(edge, prunr_core::memory::provider_now(edge, self.cpu_only))
            + usize::from(fades) * per_engine_cost(depth, self.cpu_only)
    }

    /// The most jobs, up to `self.jobs`, whose engines fit what is free now
    /// with the reserve kept for other apps; `Err`: not even one. `held_mb`:
    /// what a warm worker already holds, its own again for the run. No
    /// reading of free memory, or macOS, lets all of them run.
    pub fn jobs_that_fit(self, held_mb: u64) -> Result<usize, prunr_core::memory::Short> {
        let free = FREE_RAM_BOUNDS_A_CUTOUT.then(prunr_core::sysmem::available_ram_bytes).flatten();
        let Some(free) = free else { return Ok(self.jobs) };
        most_jobs(self.jobs, |jobs| fits_beside_held(Self { jobs, ..self }.need_bytes() as u64, free, held_mb))
    }

    /// The cut-out models lighter than these engines' that fit now at one
    /// job, the lightest first: what a refusal for memory offers instead.
    /// `held_mb`: a warm worker's memory, freed by the refusal.
    pub fn lighter_that_fit(self, held_mb: u64) -> Vec<super::settings::SettingsModel> {
        let Some(free) = prunr_core::sysmem::available_ram_bytes() else { return Vec::new() };
        let mut lighter: Vec<(super::settings::SettingsModel, ModelKind)> = super::settings::SettingsModel::ALL
            .into_iter()
            .filter_map(|m| m.to_model_kind().map(|kind| (m, kind)))
            .filter(|&(_, kind)| {
                let one = Self { model: kind, jobs: 1, first_look: false, ..self };
                cpu_need_mb(kind) < cpu_need_mb(self.model) && fits_beside_held(one.need_bytes() as u64, free, held_mb).is_ok()
            })
            .collect();
        lighter.sort_by_key(|&(_, kind)| cpu_need_mb(kind));
        lighter.into_iter().map(|(m, _)| m).collect()
    }
}

/// The most of up to `max_jobs` that `fits`; one's shortfall when none.
fn most_jobs(max_jobs: usize, fits: impl Fn(usize) -> Result<(), prunr_core::memory::Short>) -> Result<usize, prunr_core::memory::Short> {
    (2..=max_jobs).rev().find(|&jobs| fits(jobs).is_ok()).map_or_else(|| fits(1).map(|()| 1), Ok)
}

fn fits_beside_held(need_bytes: u64, free_bytes: u64, held_mb: u64) -> Result<(), prunr_core::memory::Short> {
    const MB: u64 = 1024 * 1024;
    let charge = prunr_core::memory::Charge { ram_mb: need_bytes.div_ceil(MB), card_mb: 0 };
    prunr_core::memory::fit(charge, prunr_core::memory::Room { ram_mb: free_bytes / MB + held_mb, card_mb: None })
}

/// What a run refused for memory tells the user: what `subject` needs and
/// what is free, and the `lighter` models that fit, offered as `offer`
/// words them.
pub fn refusal_text(subject: &str, short: prunr_core::memory::Short, lighter: &[String], offer: impl FnOnce(&str) -> String) -> String {
    let instead = if lighter.is_empty() { String::new() } else { format!(", or {}", offer(&lighter.join(" or "))) };
    format!("{subject} {short}. Close other apps{instead}.")
}

fn model_overhead(model: ModelKind, jobs: usize, cpu_only: bool) -> usize {
    // Not waiting on the accelerator probe here (the UI thread): until it
    // answers the engines are counted as on the CPU, the larger figure.
    let id: prunr_models::ModelId = model.into();
    let on = prunr_core::memory::provider_now(id, cpu_only);
    let engines = prunr_core::batch::engine_pool_size(model, jobs, on);
    engine_cost_on(model, on) * prunr_core::memory::engines_that_fit(id, on, engines, crate::hardware::card_free_mb_throttled(on))
}

/// Images at once by default: half the cores on the processor, two on a
/// graphics card, held to what memory allows for `model`. The app and the
/// command line start from it.
pub fn default_jobs(model: Option<ModelKind>, on_host_cpu: bool, cpu_only: bool) -> usize {
    cap_jobs(model, if on_host_cpu { (num_cpus::get() / 2).max(1) } else { 2 }, cpu_only)
}

/// `jobs`, held to what memory allows for `model`.
pub fn cap_jobs(model: Option<ModelKind>, jobs: usize, cpu_only: bool) -> usize {
    model.map_or(jobs, |model| jobs.min(safe_max_jobs(model, cpu_only)))
}

/// Engines take at most 50% of available RAM, leaving the rest for image
/// buffers, history and OS headroom.
pub fn safe_max_jobs(model: ModelKind, cpu_only: bool) -> usize {
    let available = crate::hardware::available_ram_bytes_throttled() as usize;
    let budget = available / 2; // 50% of available RAM for engines
    let cost = per_engine_cost(model, cpu_only);
    if cost == 0 { return 4; }
    (budget / cost).clamp(1, 8)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Half the cores on the processor, two on a card, never more than
    /// memory allows for the model; the command line starts from the same.
    #[test]
    fn the_default_job_count_follows_where_the_model_runs() {
        assert_eq!(default_jobs(None, true, false), (num_cpus::get() / 2).max(1));
        assert_eq!(default_jobs(None, false, false), 2);
        for model in [ModelKind::Silueta, ModelKind::BiRefNetLite] {
            assert!(default_jobs(Some(model), true, true) <= safe_max_jobs(model, true), "{model:?}");
        }
    }

    /// A cut-out batch is charged the engines it builds: the first look
    /// beside a slow cut-out model, the line model with lines on, no
    /// cut-out engines for lines alone. With 7.9 GB free, BiRefNet-lite on
    /// the CPU and its first look are refused; a warm worker's memory
    /// counts as theirs.
    #[test]
    fn a_cut_out_batch_is_charged_the_engines_it_builds() {
        use prunr_core::LineMode;
        const MB: u64 = 1024 * 1024;
        let birefnet = CutoutEngines { model: ModelKind::BiRefNetLite, jobs: 1, cpu_only: true, first_look: true, line_mode: LineMode::Off, line_fade: false };
        let need = |e: CutoutEngines| e.need_bytes() as u64;
        let look = engine_cost_on(prunr_core::subject::FIRST_LOOK_MODEL, prunr_core::engine::CPU_NAME) as u64;
        let teed = cost_on(prunr_models::ModelId::Teed, prunr_core::engine::CPU_NAME) as u64;
        assert_eq!(need(birefnet) - need(CutoutEngines { first_look: false, ..birefnet }), look, "the first look, on the CPU");
        let silueta = CutoutEngines { model: ModelKind::Silueta, ..birefnet };
        assert_eq!(need(silueta), need(CutoutEngines { first_look: false, ..silueta }), "Silueta takes no first look");
        assert_eq!(need(CutoutEngines { line_mode: LineMode::SubjectOutline, ..birefnet }), need(birefnet) + teed, "lines on the subject");
        assert_eq!(need(CutoutEngines { line_mode: LineMode::EdgesOnly, ..birefnet }), teed, "lines alone");
        let depth = per_engine_cost(prunr_core::depth::MODEL, true) as u64;
        assert_eq!(need(CutoutEngines { line_mode: LineMode::EdgesOnly, line_fade: true, ..birefnet }), teed + depth, "lines that fade");
        assert_eq!(need(CutoutEngines { line_fade: true, ..birefnet }), need(birefnet), "no lines, no fade");

        let need_mb = need(birefnet).div_ceil(MB);
        assert!(fits_beside_held(need(birefnet), 7_881 * MB, 0).is_err());
        let free = need_mb + prunr_core::memory::RAM_RESERVE_MB - 1_000;
        assert!(fits_beside_held(need(birefnet), free * MB, 0).is_err());
        assert!(fits_beside_held(need(birefnet), free * MB, 1_000).is_ok(), "a warm worker's own memory");
    }

    /// A batch that does not fit at its job count runs with fewer, and is
    /// refused only when one does not fit.
    #[test]
    fn a_batch_runs_with_fewer_jobs_before_it_is_refused() {
        let short = prunr_core::memory::Short::Ram { need_mb: 2_000, kept_mb: 1_536, free_mb: 3_000 };
        let up_to = |limit: usize| move |jobs: usize| if jobs <= limit { Ok(()) } else { Err(short) };
        assert_eq!(most_jobs(4, up_to(4)), Ok(4));
        assert_eq!(most_jobs(4, up_to(2)), Ok(2));
        assert_eq!(most_jobs(4, up_to(1)), Ok(1));
        assert_eq!(most_jobs(4, up_to(0)), Err(short));
        assert_eq!(most_jobs(1, up_to(0)), Err(short));
    }

    /// A refusal says what the model needs, what Prunr keeps free and
    /// what is free, and offers the lighter models that fit; a toast shows
    /// it as written.
    #[test]
    fn a_refusal_for_memory_says_what_to_do() {
        let short = prunr_core::memory::Short::Ram { need_mb: 7_700, kept_mb: prunr_core::memory::RAM_RESERVE_MB, free_mb: 7_881 };
        let lighter = ["Silueta".to_string(), "U2Net".to_string()];
        let text = refusal_text("BiRefNet", short, &lighter, |models| format!("pick {models} in the model list"));
        assert_eq!(
            text,
            "BiRefNet needs about 7.5 GB of free memory, and Prunr keeps 1.5 GB free for your other apps; 7.7 GB is free. \
             Close other apps, or pick Silueta or U2Net in the model list.",
        );
        assert_eq!(prunr_core::plain::error(&text), text);
        assert!(refusal_text("Silueta", short, &[], |_| unreachable!()).ends_with("7.7 GB is free. Close other apps."));
    }

    /// An engine on a card is charged its share of system RAM, measured
    /// there; one where it is not measured yet, its CPU figure.
    #[test]
    fn an_engine_is_charged_what_it_takes_where_it_runs() {
        const MB: usize = 1024 * 1024;
        let cpu = engine_cost_on(ModelKind::BiRefNetLite, "CPU");
        let cuda = engine_cost_on(ModelKind::BiRefNetLite, "CUDA");
        assert!(cuda < cpu / 2, "on CUDA its weights and activations sit on the card: {} vs {} MB", cuda / MB, cpu / MB);
        assert_eq!(engine_cost_on(ModelKind::BiRefNetLite, "DirectML"), cpu, "unmeasured there: the CPU's figure");
    }

    #[test]
    fn force_admits_when_deadlocked() {
        let mut ctrl = AdmissionController::with_budget(10_000); // tiny budget
        ctrl.enqueue(vec![
            AdmissionController::estimate_cost(1, (1000, 1000), 100, 0), // ~8 MB
        ]);
        // Way over budget but nothing in-flight → force admit
        assert_eq!(ctrl.try_admit_next(), Some(1));
    }

    #[test]
    fn release_frees_budget() {
        let mut ctrl = AdmissionController::with_budget(100_000_000);
        ctrl.enqueue(vec![
            AdmissionController::estimate_cost(1, (3000, 2000), 1_000_000, 0),
            AdmissionController::estimate_cost(2, (3000, 2000), 1_000_000, 0),
            AdmissionController::estimate_cost(3, (3000, 2000), 1_000_000, 0),
        ]);
        let id1 = ctrl.try_admit_next().unwrap();
        let _id2 = ctrl.try_admit_next().unwrap();
        assert!(ctrl.try_admit_next().is_none());
        ctrl.release(id1);
        assert!(ctrl.try_admit_next().is_some());
    }

    #[test]
    fn is_complete_after_all_released() {
        let mut ctrl = AdmissionController::with_budget(100_000_000);
        ctrl.enqueue(vec![
            AdmissionController::estimate_cost(1, (100, 100), 100, 0),
        ]);
        let id = ctrl.try_admit_next().unwrap();
        assert!(!ctrl.is_complete());
        ctrl.release(id);
        assert!(ctrl.is_complete());
    }

    #[test]
    fn best_fit_prefers_largest_that_fits() {
        // Budget ~35 MB; three items:
        //   - big:  ~38 MB (oversized — won't fit)
        //   - mid:  ~23 MB (fits — and is the best-fit largest)
        //   - smol: ~8 MB  (also fits but smaller)
        let mut ctrl = AdmissionController::with_budget(35_000_000);
        let big = AdmissionController::estimate_cost(1, (2200, 2200), 0, 0);
        let mid = AdmissionController::estimate_cost(2, (1700, 1700), 0, 0);
        let smol = AdmissionController::estimate_cost(3, (1000, 1000), 0, 0);
        ctrl.enqueue(vec![big, mid, smol]);
        let admitted = ctrl.try_admit_next();
        assert_eq!(admitted, Some(2), "should prefer the largest fitting item");
    }

    #[test]
    fn release_of_unadmitted_is_noop() {
        // Defensive: releasing an id that was never admitted must not underflow.
        let mut ctrl = AdmissionController::with_budget(100_000_000);
        ctrl.release(42);
        assert_eq!(ctrl.committed_bytes, 0);
    }

    #[test]
    fn double_release_does_not_underflow() {
        let mut ctrl = AdmissionController::with_budget(100_000_000);
        ctrl.enqueue(vec![AdmissionController::estimate_cost(1, (100, 100), 100, 0)]);
        let id = ctrl.try_admit_next().unwrap();
        ctrl.release(id);
        ctrl.release(id); // second release must be a no-op
        assert_eq!(ctrl.committed_bytes, 0);
    }

    #[test]
    fn drains_all_items_even_when_every_one_exceeds_budget() {
        // Every item is bigger than the budget. Force-admit must kick in each
        // time the window is empty, so the queue still drains completely.
        let mut ctrl = AdmissionController::with_budget(5_000_000);
        let items: Vec<ImageMemCost> = (1..=10)
            .map(|i| AdmissionController::estimate_cost(i, (2000, 2000), 100, 0))
            .collect();
        ctrl.enqueue(items);

        let mut drained = 0;
        let mut safety = 0;
        while !ctrl.is_complete() {
            safety += 1;
            if safety > 1000 {
                panic!("admission loop did not converge");
            }
            if let Some(id) = ctrl.try_admit_next() {
                drained += 1;
                ctrl.release(id);
            } else {
                break;
            }
        }
        assert_eq!(drained, 10);
        assert!(ctrl.is_complete());
    }

    /// History residue is included in the cost estimate. Without this
    /// term, long batches with deep history under-count committed RAM
    /// and over-admit, leading to OOM. Test pins the (rgba_size / 2)
    /// × depth contract so a future tweak to the divisor is visible.
    #[test]
    fn estimate_cost_includes_history_residue() {
        let dims = (1000, 1000);
        let pixels = (dims.0 * dims.1) as usize;
        let rgba_size = pixels * 4; // 4 MB
        let cost_zero = AdmissionController::estimate_cost(1, dims, 0, 0);
        let cost_depth_5 = AdmissionController::estimate_cost(2, dims, 0, 5);
        // depth 0 → only 2× rgba.
        assert_eq!(cost_zero.total, 2 * rgba_size);
        // depth 5 → adds (rgba_size / 2) × 5.
        let expected_residue = (rgba_size / 2) * 5;
        assert_eq!(cost_depth_5.total, 2 * rgba_size + expected_residue);
        // Non-trivial OOM-prevention property: depth 50 should add at
        // least 25× the rgba_size in committed bytes.
        let cost_deep = AdmissionController::estimate_cost(3, dims, 0, 50);
        assert!(cost_deep.total >= 2 * rgba_size + 25 * rgba_size);
    }

    /// Pins descriptor-driven dispatch — a future "regress to const"
    /// refactor would silently mis-charge admission for new models.
    #[test]
    fn per_engine_cost_is_descriptor_driven() {
        for model in [ModelKind::Silueta, ModelKind::U2net, ModelKind::BiRefNetLite] {
            let registry = prunr_models::descriptor(model.into()).map(|d| d.needs.host_cpu().host_mb).unwrap();
            assert_eq!(engine_cost_on(model, "CPU"), registry as usize * 1024 * 1024, "{model:?}: the registry's MB, as bytes");
        }
        // BiRefNet's bigger activation footprint must charge more than
        // Silueta: the whole point of model-aware admission. Measured
        // figures order the small models differently per platform.
        let (silueta, birefnet) = (engine_cost_on(ModelKind::Silueta, "CPU"), engine_cost_on(ModelKind::BiRefNetLite, "CPU"));
        assert!(birefnet > silueta, "BiRefNet ({birefnet}) ≤ Silueta ({silueta})");
    }
}
