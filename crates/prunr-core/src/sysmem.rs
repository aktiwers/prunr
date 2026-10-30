//! What the system has free and what this process holds, for the memory
//! and disk guards.

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
    memory_now().0
}

/// What a photo's buffers may use: the memory available now plus free
/// swap; `None` when free memory cannot be read. Buffers touched once or
/// twice sit in swap at little cost, so the system pushing idle memory
/// there serves them; a model's weights, read every step, are held to RAM
/// by the model checks instead.
pub fn room_for_buffers_bytes() -> Option<u64> {
    let (available, free_swap) = memory_now();
    available.map(|available| available + free_swap)
}

/// `available_ram_bytes` and the free swap, from one reading.
fn memory_now() -> (Option<u64>, u64) {
    let (available, total, free_swap) = with_system(|sys| {
        sys.refresh_memory();
        (sys.available_memory(), sys.total_memory(), sys.free_swap())
    });
    (first_reported(available, total), free_swap)
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

/// Hand the heap's free pages back to the system. Once glibc frees a large
/// block it raises its mmap threshold to that size, so the next large
/// buffers come from its heap, which it keeps after they are dropped: a
/// Stable Diffusion session loaded and dropped for each fill kept about
/// 1.5 GB more each time on Linux with CUDA, until the fourth fill was
/// refused for memory. Other allocators give large blocks back on free.
pub fn return_freed_memory() {
    // SAFETY: malloc_trim only releases pages malloc holds free.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
}

/// Current process RSS in MB; `None` when sysinfo can't read the process.
pub fn process_rss_mb() -> Option<u64> {
    let pid = sysinfo::get_current_pid().ok()?;
    with_system(|sys| {
        sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        sys.process(pid).map(|p| p.memory() / (1024 * 1024))
    })
}

/// Free bytes on the disk holding `path` (or its nearest existing
/// parent): the mount with the longest prefix of it. `None` when no
/// mount matches.
pub fn free_disk_bytes(path: &std::path::Path) -> Option<u64> {
    let existing = path.ancestors().find(|p| p.exists())?;
    // std's canonicalize gives `\\?\C:\…` on Windows, which no mount
    // point (`C:\`) prefixes; dunce keeps the plain form.
    let path = dunce::canonicalize(existing).unwrap_or_else(|_| existing.to_path_buf());
    sysinfo::Disks::new_with_refreshed_list()
        .list()
        .iter()
        .filter(|d| path.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(sysinfo::Disk::available_space)
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
