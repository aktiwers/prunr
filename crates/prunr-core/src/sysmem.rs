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

/// The memory free to take, as the system reports it available; `None`
/// when it reports no memory at all.
pub fn available_ram_bytes() -> Option<u64> {
    let (available, total, _) = memory_now();
    reported_available(available, total)
}

/// What a photo's buffers may use: the memory available now plus free
/// swap; `None` when free memory cannot be read. Buffers touched once or
/// twice sit in swap at little cost, so the system pushing idle memory
/// there serves them; a model's weights, read every step, are held to RAM
/// by the model checks instead.
pub fn room_for_buffers_bytes() -> Option<u64> {
    let (available, total, free_swap) = memory_now();
    // macOS compresses memory and makes swap files as it needs them, so
    // the free swap it reports (part of one file at rest) says nothing of
    // the room: a 294 MP Refine needing 13.1 GB was refused on an idle
    // 16 GB Mac, and ran there in 11 s, also beside 12.9 GB held by
    // another process. A photo is held to the machine's memory instead.
    if cfg!(target_os = "macos") {
        return (total > 0).then_some(total);
    }
    reported_available(available, total).map(|available| available + free_swap)
}

/// Available memory, total memory and free swap, from one reading.
fn memory_now() -> (u64, u64, u64) {
    with_system(|sys| {
        sys.refresh_memory();
        (available_of(sys), sys.total_memory(), sys.free_swap())
    })
}

/// On macOS all memory less what is used (apps, wired, compressed), as
/// Activity Monitor counts it. sysinfo's own figure there is free +
/// inactive + purgeable less the compressor's pages, floored at 0, which
/// it reaches while the system still makes room by swapping: under a
/// 13 GB load on a 16 GB Mac it read 0 where this reads 3.4 GB, and
/// idle the two agree within half a gigabyte.
fn available_of(sys: &sysinfo::System) -> u64 {
    if cfg!(target_os = "macos") {
        sys.total_memory().saturating_sub(sys.used_memory())
    } else {
        sys.available_memory()
    }
}

/// Whether the system itself says memory is short, where it says so:
/// macOS's pressure level (Activity Monitor's yellow or red). Neither
/// free-memory figure tells it there: under a load that left swap nearly
/// full, sysinfo read 0 free and total less used read 27%. `None`
/// elsewhere, and when the level cannot be read.
pub fn system_says_pressure() -> Option<bool> {
    #[cfg(target_os = "macos")]
    {
        let mut level: libc::c_int = 0;
        let mut size = std::mem::size_of::<libc::c_int>();
        // SAFETY: the out-parameter is a c_int and `size` says so.
        let read = unsafe {
            libc::sysctlbyname(c"kern.memorystatus_vm_pressure_level".as_ptr(), (&raw mut level).cast(), &mut size, std::ptr::null_mut(), 0)
        };
        (read == 0).then(|| pressed(level))
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// macOS's pressure levels: 1 normal, 2 warning, 4 critical.
#[cfg(any(target_os = "macos", test))]
fn pressed(level: i32) -> bool {
    level >= 2
}

/// `available` when the system reports its memory at all: a 0 is nothing
/// free, not a figure missing.
fn reported_available(available: u64, total: u64) -> Option<u64> {
    (total > 0).then_some(available)
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

    /// macOS names its pressure 1 (normal), 2 (warning) or 4 (critical):
    /// the warning is when memory is short.
    #[test]
    fn the_systems_warning_level_is_pressure() {
        assert!(!pressed(1));
        assert!(pressed(2));
        assert!(pressed(4));
    }

    /// A 0 available is a figure; no total reported is unknown.
    #[test]
    fn available_memory_counts_also_when_it_is_none() {
        assert_eq!(reported_available(4, 16), Some(4));
        assert_eq!(reported_available(0, 16), Some(0));
        assert_eq!(reported_available(0, 0), None);
    }
}
