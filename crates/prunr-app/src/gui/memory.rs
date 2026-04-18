//! Memory-aware admission controller for batch image processing.
//!
//! Instead of loading all images into memory at once, the controller
//! maintains a RAM budget and greedily admits images using best-fit
//! selection. As each image finishes processing and its buffers are
//! evicted, the freed budget is used to admit new images.

use std::collections::HashMap;

use prunr_core::ModelKind;

/// Estimated memory cost for a single image.
pub struct ImageMemCost {
    pub item_id: u64,
    /// Total estimated bytes this image will consume while in-flight.
    pub total: usize,
}

/// Tracks which items are currently admitted (in the processing window)
/// and enforces a RAM budget.
pub struct AdmissionController {
    budget_bytes: usize,
    committed_bytes: usize,
    /// Maps admitted item_id → its cost, so we can subtract on release.
    admitted: HashMap<u64, usize>,
    /// Pending items sorted descending by cost (largest first for best-fit).
    pending: Vec<ImageMemCost>,
    total_items: usize,
}

impl AdmissionController {
    /// Create a new controller. Queries available system RAM and applies safety margin.
    pub fn new(model: ModelKind, jobs: usize) -> Self {
        let available = available_ram();
        let overhead = model_overhead(model, jobs);
        let budget = ((available as f64) * 0.85) as usize;
        let budget = budget.saturating_sub(overhead);
        Self {
            budget_bytes: budget,
            committed_bytes: 0,
            admitted: HashMap::new(),
            pending: Vec::new(),
            total_items: 0,
        }
    }

    /// Estimate memory cost for a single image from dimensions and file size.
    /// Cost = source_rgba + result_rgba + compressed bytes.
    pub fn estimate_cost(item_id: u64, dimensions: (u32, u32), file_bytes_len: usize) -> ImageMemCost {
        let pixels = dimensions.0 as usize * dimensions.1 as usize;
        let rgba_size = pixels * 4;
        // source_rgba (decoded) + result_rgba (output) + compressed bytes already in RAM
        let total = rgba_size * 2 + file_bytes_len;
        ImageMemCost { item_id, total }
    }

    /// Add pending items. Sorts them for best-fit (largest first).
    pub fn enqueue(&mut self, mut costs: Vec<ImageMemCost>) {
        self.total_items += costs.len();
        costs.sort_by(|a, b| b.total.cmp(&a.total));
        self.pending.extend(costs);
        // Re-sort after extend in case there were existing pending items
        self.pending.sort_by(|a, b| b.total.cmp(&a.total));
    }

    /// Try to admit the next best-fitting image.
    /// Returns `Some(item_id)` if one fits the remaining budget.
    pub fn try_admit_next(&mut self) -> Option<u64> {
        let remaining = self.budget_bytes.saturating_sub(self.committed_bytes);

        // Best-fit: find the largest pending item that fits.
        if let Some(idx) = self.pending.iter().position(|c| c.total <= remaining) {
            let cost = self.pending.remove(idx);
            self.committed_bytes += cost.total;
            self.admitted.insert(cost.item_id, cost.total);
            return Some(cost.item_id);
        }

        // Nothing fits. If nothing is in-flight, force-admit the smallest
        // to guarantee forward progress (avoids deadlock).
        if self.admitted.is_empty() && !self.pending.is_empty() {
            let cost = self.pending.pop().unwrap(); // pop last = smallest
            self.committed_bytes += cost.total;
            self.admitted.insert(cost.item_id, cost.total);
            return Some(cost.item_id);
        }

        None // wait for in-flight items to release budget
    }

    /// Release an item's budget after its processing completes.
    pub fn release(&mut self, item_id: u64) {
        if let Some(cost) = self.admitted.remove(&item_id) {
            self.committed_bytes = self.committed_bytes.saturating_sub(cost);
        }
    }

    /// True when all items have been admitted and released.
    pub fn is_complete(&self) -> bool {
        self.pending.is_empty() && self.admitted.is_empty()
    }
}

/// Cached sysinfo::System instance to avoid repeated allocations.
fn with_system<T>(f: impl FnOnce(&sysinfo::System) -> T) -> T {
    use std::sync::{Mutex, OnceLock};
    static SYS: OnceLock<Mutex<sysinfo::System>> = OnceLock::new();
    let mtx = SYS.get_or_init(|| Mutex::new(sysinfo::System::new()));
    let mut sys = mtx.lock().unwrap();
    sys.refresh_memory();
    f(&sys)
}

/// Query available system RAM (cross-platform via sysinfo).
fn available_ram() -> usize {
    with_system(|sys| sys.available_memory() as usize)
}

/// Returns true when available RAM drops below 20% of total.
/// Used to trigger Tier 2 → Tier 3 demotion of history entries.
pub fn under_memory_pressure() -> bool {
    with_system(|sys| {
        let total = sys.total_memory();
        let available = sys.available_memory();
        if total == 0 { return false; }
        (available as f64 / total as f64) < 0.20
    })
}

/// Total cost per concurrent inference slot: session weights + ORT workspace +
/// inference tensors + decoded image buffers. Measured empirically from OOM
/// at 4 jobs / 20 GB RSS with BiRefNet on a 32 GB system.
fn per_engine_cost(model: ModelKind) -> usize {
    match model {
        ModelKind::Silueta => 200 * 1024 * 1024,        //  200 MB (320x320 tensors)
        ModelKind::U2net => 800 * 1024 * 1024,           //  800 MB (320x320 + large model)
        ModelKind::BiRefNetLite => 2500 * 1024 * 1024,   // 2.5 GB (1024x1024 tensors + ORT workspace)
    }
}

/// Fixed overhead for the engine pool at the requested job count.
fn model_overhead(model: ModelKind, jobs: usize) -> usize {
    let is_gpu = !prunr_core::OrtEngine::detect_active_provider().eq_ignore_ascii_case("CPU");
    let num_engines = if is_gpu { jobs.min(2) } else { jobs };
    per_engine_cost(model) * num_engines
}

/// Maximum safe parallel jobs for the given model and available RAM.
/// Conservative: engines should use at most 50% of available RAM,
/// leaving the rest for image buffers, history, and OS headroom.
pub fn safe_max_jobs(model: ModelKind) -> usize {
    let available = available_ram();
    let budget = available / 2; // 50% of available RAM for engines
    let cost = per_engine_cost(model);
    if cost == 0 { return 4; }
    (budget / cost).max(1).min(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctrl(budget: usize) -> AdmissionController {
        AdmissionController {
            budget_bytes: budget,
            committed_bytes: 0,
            admitted: HashMap::new(),
            pending: Vec::new(),
            total_items: 0,
        }
    }

    #[test]
    fn admits_items_within_budget() {
        let mut ctrl = make_ctrl(100_000_000);
        ctrl.enqueue(vec![
            AdmissionController::estimate_cost(1, (3000, 2000), 1_000_000),
            AdmissionController::estimate_cost(2, (3000, 2000), 1_000_000),
            AdmissionController::estimate_cost(3, (3000, 2000), 1_000_000),
        ]);
        // Each image ~49 MB, budget 100 MB → 2 fit
        assert!(ctrl.try_admit_next().is_some());
        assert!(ctrl.try_admit_next().is_some());
        assert!(ctrl.try_admit_next().is_none());
    }

    #[test]
    fn force_admits_when_deadlocked() {
        let mut ctrl = make_ctrl(10_000); // tiny budget
        ctrl.enqueue(vec![
            AdmissionController::estimate_cost(1, (1000, 1000), 100), // ~8 MB
        ]);
        // Way over budget but nothing in-flight → force admit
        assert_eq!(ctrl.try_admit_next(), Some(1));
    }

    #[test]
    fn release_frees_budget() {
        let mut ctrl = make_ctrl(100_000_000);
        ctrl.enqueue(vec![
            AdmissionController::estimate_cost(1, (3000, 2000), 1_000_000),
            AdmissionController::estimate_cost(2, (3000, 2000), 1_000_000),
            AdmissionController::estimate_cost(3, (3000, 2000), 1_000_000),
        ]);
        let id1 = ctrl.try_admit_next().unwrap();
        let _id2 = ctrl.try_admit_next().unwrap();
        assert!(ctrl.try_admit_next().is_none());
        ctrl.release(id1);
        assert!(ctrl.try_admit_next().is_some());
    }

    #[test]
    fn is_complete_after_all_released() {
        let mut ctrl = make_ctrl(100_000_000);
        ctrl.enqueue(vec![
            AdmissionController::estimate_cost(1, (100, 100), 100),
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
        let mut ctrl = make_ctrl(35_000_000);
        let big = AdmissionController::estimate_cost(1, (2200, 2200), 0);
        let mid = AdmissionController::estimate_cost(2, (1700, 1700), 0);
        let smol = AdmissionController::estimate_cost(3, (1000, 1000), 0);
        ctrl.enqueue(vec![big, mid, smol]);
        let admitted = ctrl.try_admit_next();
        assert_eq!(admitted, Some(2), "should prefer the largest fitting item");
    }

    #[test]
    fn release_of_unadmitted_is_noop() {
        // Defensive: releasing an id that was never admitted must not underflow.
        let mut ctrl = make_ctrl(100_000_000);
        ctrl.release(42);
        assert_eq!(ctrl.committed_bytes, 0);
    }

    #[test]
    fn double_release_does_not_underflow() {
        let mut ctrl = make_ctrl(100_000_000);
        ctrl.enqueue(vec![AdmissionController::estimate_cost(1, (100, 100), 100)]);
        let id = ctrl.try_admit_next().unwrap();
        ctrl.release(id);
        ctrl.release(id); // second release must be a no-op
        assert_eq!(ctrl.committed_bytes, 0);
    }

    #[test]
    fn drains_all_items_even_when_every_one_exceeds_budget() {
        // Every item is bigger than the budget. Force-admit must kick in each
        // time the window is empty, so the queue still drains completely.
        let mut ctrl = make_ctrl(5_000_000);
        let items: Vec<ImageMemCost> = (1..=10)
            .map(|i| AdmissionController::estimate_cost(i, (2000, 2000), 100))
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
}
