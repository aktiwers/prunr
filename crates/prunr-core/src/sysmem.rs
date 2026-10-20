//! What the system has free and what this process holds, for the memory
//! guards.

use std::sync::{Mutex, OnceLock};

/// One `sysinfo::System` for both probes; they refresh disjoint fields.
fn with_system<T>(f: impl FnOnce(&mut sysinfo::System) -> T) -> T {
    use std::sync::PoisonError;
    static SYS: OnceLock<Mutex<sysinfo::System>> = OnceLock::new();
    let mtx = SYS.get_or_init(|| Mutex::new(sysinfo::System::new()));
    let mut sys = mtx.lock().unwrap_or_else(PoisonError::into_inner);
    f(&mut sys)
}

/// The memory free to take: what the system reports available, else its
/// total; `None` when it reports neither.
pub fn available_ram_bytes() -> Option<u64> {
    let (available, total) = with_system(|sys| {
        sys.refresh_memory();
        (sys.available_memory(), sys.total_memory())
    });
    first_reported(available, total)
}

/// The first figure the system reported (0 is none).
fn first_reported(available: u64, total: u64) -> Option<u64> {
    [available, total].into_iter().find(|&bytes| bytes > 0)
}

/// The system's memory; `None` when it doesn't report it.
pub fn total_ram_bytes() -> Option<u64> {
    let total = with_system(|sys| {
        sys.refresh_memory();
        sys.total_memory()
    });
    (total > 0).then_some(total)
}

/// Current process RSS in MB; `None` when sysinfo can't read the process.
pub fn process_rss_mb() -> Option<u64> {
    let pid = sysinfo::get_current_pid().ok()?;
    with_system(|sys| {
        sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        sys.process(pid).map(|p| p.memory() / (1024 * 1024))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Available memory counts; the total stands in when it isn't
    /// reported; neither reported is unknown.
    #[test]
    fn available_memory_falls_back_to_the_total() {
        assert_eq!(first_reported(4, 16), Some(4));
        assert_eq!(first_reported(0, 16), Some(16));
        assert_eq!(first_reported(0, 0), None);
    }
}
