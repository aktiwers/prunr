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
    let (available, total, _) = memory_now(false);
    reported_available(available, total)
}

/// Free memory a run may still take before the computer runs out: RAM
/// available now plus free swap, zram's at what compression frees. A
/// model run pushed into swap is slower, not stopped; `None` when the
/// system reports no memory.
pub fn free_ram_and_swap_bytes() -> Option<u64> {
    free_ram_then_swap_bytes().map(|(ram, swap)| ram + swap)
}

/// Free swap alone, as `swap_room` counts it; reading it on Windows walks
/// the process table.
pub fn free_swap_bytes() -> u64 {
    swap_room(with_system(|sys| {
        sys.refresh_memory_specifics(sysinfo::MemoryRefreshKind::nothing().with_swap());
        sys.free_swap()
    }))
}

/// The room `free_swap` makes: less the RAM a Linux zram device would take
/// to hold its free space, as zram keeps its pages in RAM, compressed.
/// Elsewhere (no /proc/swaps) all of it.
fn swap_room(free_swap: u64) -> u64 {
    let Ok(swaps) = std::fs::read_to_string("/proc/swaps") else { return free_swap };
    let ratio = |device: &str| std::fs::read_to_string(format!("/sys/block/{device}/mm_stat")).ok().and_then(|stat| zram_ratio(&stat));
    free_swap.saturating_sub(zram_held(&swaps, ratio))
}

/// zram's compression assumed before a device holds any pages.
const ZRAM_RATIO_UNKNOWN: f64 = 2.0;
/// The most zram's compression is counted at: zero-filled pages take no
/// memory, so a device holding few pages can report far more than the
/// pages a run pushes out will compress to.
const ZRAM_RATIO_MAX: f64 = 4.0;

/// The RAM the zram devices `swaps` (/proc/swaps) lists would take to
/// hold their free space: free over `ratio_of` the device's compression.
fn zram_held(swaps: &str, ratio_of: impl Fn(&str) -> Option<f64>) -> u64 {
    swaps
        .lines()
        .skip(1)
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let device = fields.next()?.strip_prefix("/dev/").filter(|d| d.starts_with("zram"))?;
            let (size_kb, used_kb) = (fields.nth(1)?.parse::<u64>().ok()?, fields.next()?.parse::<u64>().ok()?);
            let ratio = ratio_of(device).unwrap_or(ZRAM_RATIO_UNKNOWN).clamp(1.0, ZRAM_RATIO_MAX);
            Some((size_kb.saturating_sub(used_kb) as f64 * 1024.0 / ratio) as u64)
        })
        .sum()
}

/// A zram device's compression from its `mm_stat`: the data it holds
/// over the memory it takes; `None` while it holds none.
fn zram_ratio(mm_stat: &str) -> Option<f64> {
    let mut fields = mm_stat.split_whitespace().map(|field| field.parse::<u64>().ok());
    let (held, _compressed, taken) = (fields.next()??, fields.next()??, fields.next()??);
    (held > 0 && taken > 0).then(|| held as f64 / taken as f64)
}

/// `free_ram_and_swap_bytes` in its two parts: RAM available, then free
/// swap.
pub fn free_ram_then_swap_bytes() -> Option<(u64, u64)> {
    let (available, total, free_swap) = memory_now(true);
    reported_available(available, total).map(|available| (available, free_swap))
}

/// What a photo's buffers may use: the memory available now plus free
/// swap; `None` when free memory cannot be read. Buffers touched once or
/// twice sit in swap at little cost, so the system pushing idle memory
/// there serves them; a model's weights, read every step, are held to RAM
/// by the model checks instead.
pub fn room_for_buffers_bytes() -> Option<u64> {
    let (available, total, free_swap) = memory_now(true);
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

/// Available memory, total memory and, with `swap`, free swap as
/// `swap_room` counts it (else 0), from one reading. Swap is read only
/// when asked: on Windows it walks the process table, and the run
/// watchdogs read memory several times a second.
fn memory_now(swap: bool) -> (u64, u64, u64) {
    let (available, total, free_swap) = with_system(|sys| {
        let ram = sysinfo::MemoryRefreshKind::nothing().with_ram();
        sys.refresh_memory_specifics(if swap { ram.with_swap() } else { ram });
        (available_of(sys), sys.total_memory(), if swap { sys.free_swap() } else { 0 })
    });
    (available, total, if swap { swap_room(free_swap) } else { 0 })
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

/// The memory Windows can still commit: its limit (RAM plus page file)
/// less what processes have committed. CUDA's allocations are committed
/// too (a BiRefNet worker on a 12 GB card held 8.9 GB), so with the page
/// file unable to grow (a full system drive) runs fail with RAM free.
/// `None` elsewhere, and when it cannot be read.
pub fn available_commit_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut status = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
        // SAFETY: `status` is a MEMORYSTATUSEX with its length set, as the
        // call requires; it only writes that struct.
        unsafe { GlobalMemoryStatusEx(&mut status) }.ok()?;
        Some(status.ullAvailPageFile)
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// Whether Windows has less left to commit than a run's stop line
/// (`memory::STOP_BELOW_FREE_MB`): a run failing then failed for that,
/// not for the card's memory or the model. Never elsewhere.
pub fn commit_is_short() -> bool {
    available_commit_bytes().is_some_and(|free| free < crate::memory::STOP_BELOW_FREE_MB * 1024 * 1024)
}

/// The system's memory; `None` when it doesn't report it.
pub fn total_ram_bytes() -> Option<u64> {
    let total = with_system(|sys| {
        sys.refresh_memory_specifics(sysinfo::MemoryRefreshKind::nothing().with_ram());
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

/// Current process RSS in MB; `None` when it cannot be read.
pub fn process_rss_mb() -> Option<u64> {
    rss_mb_of(std::process::id())
}

/// Process `pid`'s RSS in MB; `None` when it cannot be read.
pub fn rss_mb_of(pid: u32) -> Option<u64> {
    let pid = sysinfo::Pid::from_u32(pid);
    with_system(|sys| {
        let memory = sysinfo::ProcessRefreshKind::nothing().with_memory();
        sys.refresh_processes_specifics(sysinfo::ProcessesToUpdate::Some(&[pid]), true, memory);
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

    /// Windows reports the memory it can still commit; elsewhere there is
    /// none to report.
    #[test]
    fn only_windows_reports_its_commit() {
        assert_eq!(available_commit_bytes().is_some(), cfg!(windows));
    }

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

    /// zram keeps its pages in RAM, so its free space counts at what
    /// compression frees; disk swap counts in full.
    #[test]
    fn zram_counts_at_what_compression_frees() {
        let swaps = "Filename\t\t\t\tType\t\tSize\t\tUsed\t\tPriority\n\
            /dev/nvme0n1p4                          partition\t974844\t\t902212\t\t-2\n\
            /dev/zram0                              partition\t16269400\t6480352\t\t5\n";
        let free_zram: u64 = (16_269_400 - 6_480_352) * 1024;
        assert_eq!(zram_held(swaps, |device| (device == "zram0").then_some(2.5)), (free_zram as f64 / 2.5) as u64);
        assert_eq!(zram_held(swaps, |_| None), free_zram / 2, "2:1 before it holds any");
        assert_eq!(zram_held(swaps, |_| Some(50.0)), free_zram / 4, "a few zero pages make it no better than 4:1");
        assert_eq!(zram_held("Filename Type Size Used Priority\n/swap.img file 4194300 1794304 -1\n", |_| Some(3.0)), 0, "disk swap counts in full");
        assert_eq!(zram_ratio("6420135936 2656512693 2711789568 0 2843983872 20630 49390 174730 236841"), Some(6_420_135_936.0 / 2_711_789_568.0));
        assert_eq!(zram_ratio("0 0 0 0 0 0 0 0 0"), None);
    }
}
