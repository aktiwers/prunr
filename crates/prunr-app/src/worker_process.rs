//! Subprocess worker entry point.
//!
//! Invoked by the parent process as `prunr --worker`. Reads commands from
//! stdin, processes images using ORT, writes events to stdout. If this
//! process OOMs, the parent detects the crash and retries with reduced
//! concurrency.

use std::io::{BufReader, BufWriter};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use prunr_core::{
    create_engine_pool,
    OrtEngine, ProcessResult, EdgeEngine,
};

use prunr_app::subprocess::protocol::*;
use prunr_app::subprocess::ipc::{read_image, read_message, write_message};
use prunr_app::subprocess::first_look::{FirstLook, Look, Shown};
use prunr_app::subprocess::spare_scales::{next_token, SpareScales};
use prunr_app::subprocess::cancels::{DispatchCancel, DispatchCancels};
use prunr_app::gui::settings::LineMode;

const SEG_ENGINE_MISSING: &str = "segmentation engine not initialized";

/// Weighted memory semaphore for processing.
/// Instead of a binary lock, this tracks "pixel units" (1 unit = 1M pixels).
/// Small images run in parallel; large images throttle automatically.
/// Total capacity is set based on available RAM at worker startup.
///
/// Fairness: the wait queue is FIFO. New acquirers may proceed concurrently
/// with existing holders only if their request still leaves enough units
/// for the head-of-queue waiter — so a giant image isn't repeatedly skipped
/// past by a stream of arriving small ones (`notify_all` previously woke
/// every waiter and the smallest grabbed the lock first).
struct WeightedSemaphore {
    state: Mutex<SemState>,
    available: std::sync::Condvar,
    total_units: usize,
}

struct SemState {
    units_remaining: usize,
    next_ticket: u64,
    /// FIFO queue of (ticket, weight) for waiters that haven't yet acquired.
    /// `front()` is the head; new admissions must leave room for its weight.
    waiters: std::collections::VecDeque<(u64, usize)>,
}

impl WeightedSemaphore {
    fn new(total_units: usize) -> Self {
        Self {
            state: Mutex::new(SemState {
                units_remaining: total_units,
                next_ticket: 0,
                waiters: std::collections::VecDeque::new(),
            }),
            available: std::sync::Condvar::new(),
            total_units,
        }
    }

    /// Acquire units, returning a RAII guard that releases on drop (panic-safe).
    fn acquire(self: &Arc<Self>, weight: usize) -> SemaphoreGuard {
        let capped = weight.min(self.total_units);
        // Poison recovery: the semaphore state is just bookkeeping. A
        // panicking user of the semaphore must not deadlock the worker pool.
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let my_ticket = state.next_ticket;
        state.next_ticket += 1;
        state.waiters.push_back((my_ticket, capped));
        loop {
            let am_head = state.waiters.front().map(|(t, _)| *t) == Some(my_ticket);
            let head_weight = state.waiters.front().map(|(_, w)| *w).unwrap_or(0);
            let can_go = if am_head {
                state.units_remaining >= capped
            } else {
                // Leave room for the head waiter so it isn't starved.
                state.units_remaining >= capped + head_weight
            };
            if can_go { break; }
            state = self.available.wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        // Remove our entry (could be head or interior — interior happens when
        // a smaller-than-head waiter behind the head got admitted concurrently).
        state.waiters.retain(|(t, _)| *t != my_ticket);
        state.units_remaining -= capped;
        // Wake other waiters in case one of them now fits behind us.
        self.available.notify_all();
        SemaphoreGuard { sem: self.clone(), acquired: capped }
    }
}

/// RAII guard: releases semaphore units on drop (including panics).
struct SemaphoreGuard {
    sem: Arc<WeightedSemaphore>,
    acquired: usize,
}

impl Drop for SemaphoreGuard {
    fn drop(&mut self) {
        // Poison recovery: drop must not panic (would abort under double-panic).
        let mut state = self.sem.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.units_remaining += self.acquired;
        self.sem.available.notify_all();
    }
}

/// How often a model being built reports, well inside the parent's hang
/// watchdog.
const HEARTBEAT_EVERY: std::time::Duration = std::time::Duration::from_secs(10);

/// Reports "Loading model" for `item_id` every `every` until dropped.
struct Heartbeat {
    stop: Arc<AtomicBool>,
    beating: std::thread::Thread,
}

impl Heartbeat {
    fn start(events: std::sync::mpsc::Sender<SubprocessEvent>, item_id: u64, every: std::time::Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let beating = std::thread::spawn(move || {
            while !stopped.load(Ordering::Acquire) {
                let report = SubprocessEvent::Report { item_id, update: prunr_core::ProgressUpdate::Step(prunr_core::Step::LoadingModel) };
                if events.send(report).is_err() {
                    return;
                }
                std::thread::park_timeout(every);
            }
        });
        Self { stop, beating: beating.thread().clone() }
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.beating.unpark();
    }
}

/// Write a result for the parent to read. One that fails (a full temp
/// disk) leaves no half file behind; the caller reports the error, or the
/// parent would wait for an answer that never comes.
fn write_result(path: &std::path::Path, rgba: &image::RgbaImage) -> Result<(), String> {
    std::fs::write(path, rgba.as_raw()).map_err(|e| {
        let _ = std::fs::remove_file(path);
        format!("Failed to write result: {e}")
    })
}

/// Remove each temp path that lives under the IPC directory. Silent on
/// missing files — subprocess cancel branches call this before the normal
/// post-spawn cleanup would have fired.
fn cleanup_ipc_temps(ipc: &std::path::Path, paths: &[&std::path::Path]) {
    for p in paths {
        if p.starts_with(ipc) {
            let _ = std::fs::remove_file(p);
        }
    }
}

fn send_rss(evt_tx: &mpsc::Sender<SubprocessEvent>) {
    if let Some(stats) = memory_stats::memory_stats() {
        let _ = evt_tx.send(SubprocessEvent::RssUpdate { rss_bytes: stats.physical_mem as u64 });
    }
}

/// Remove an image's IPC inputs: its file and, for a chained run, its start.
fn cleanup_image_inputs(ipc: &std::path::Path, image: &std::path::Path, chain: Option<&std::path::Path>) {
    match chain {
        Some(c) => cleanup_ipc_temps(ipc, &[image, c]),
        None => cleanup_ipc_temps(ipc, &[image]),
    }
}

/// The edge detector, built on first need.
fn ensure_edge_engine(edge_engine: &mut Option<Arc<EdgeEngine>>) -> Result<(), String> {
    if edge_engine.is_none() {
        *edge_engine = Some(Arc::new(EdgeEngine::new().map_err(|e| format!("Edge engine failed: {e}"))?));
    }
    Ok(())
}

/// The detector's planes the run made, written to `path`; `None` when
/// it made none or the write failed.
fn write_edge_cache(
    path: std::path::PathBuf,
    item_id: u64,
    res: &prunr_core::EdgeInferenceResult,
) -> Option<EdgeCacheFile> {
    let scales: Vec<prunr_core::EdgeScale> = prunr_core::EdgeScale::ALL.into_iter()
        .filter(|&s| !res.tensors[s as usize].is_empty())
        .collect();
    if scales.is_empty() {
        return None;
    }
    match write_edge_tensors(&path, res) {
        Ok(()) => Some(EdgeCacheFile { path, height: res.height, width: res.width, scales }),
        Err(err) => {
            tracing::error!(item_id, %err, "edge cache write failed");
            None
        }
    }
}

/// Stream an `EdgeInferenceResult`'s non-empty planes to a path as one LE
/// byte sequence in `EdgeScale` order — the layout `EdgeCacheFile`
/// names — without the ~16 MB intermediate `Vec<u8>` that
/// `std::fs::write(path, pack_edge_tensors(res))` would have built.
fn write_edge_tensors(
    path: &std::path::Path,
    res: &prunr_core::EdgeInferenceResult,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for t in &res.tensors {
        f.write_all(prunr_app::subprocess::ipc::f32s_as_le_bytes(t))?;
    }
    Ok(())
}

/// Make the scales a run left for after its result and send them under
/// `token`; `engine` is `None` to skip them (the image left the screen,
/// a cancel, a shutdown), and the event goes all the same so the app
/// stops waiting for them.
fn send_spare_scales(
    spare: SpareScales,
    token: u64,
    item_id: u64,
    engine: Option<&EdgeEngine>,
    ipc: &std::path::Path,
    evt_tx: &mpsc::Sender<SubprocessEvent>,
) {
    let started = std::time::Instant::now();
    let outcome = match engine.and_then(|e| spare.make(e)) {
        Some(Ok(res)) => write_edge_cache(IpcKind::SpareEdge.path_for(ipc, token), item_id, &res).map_or(SpareOutcome::Failed, SpareOutcome::Made),
        Some(Err(err)) => {
            tracing::warn!(item_id, %err, "spare line scales failed");
            SpareOutcome::Failed
        }
        None => SpareOutcome::Skipped,
    };
    let made = match &outcome {
        SpareOutcome::Made(file) => format!("{:?}", file.scales),
        SpareOutcome::Skipped => "skipped".to_string(),
        SpareOutcome::Failed => "failed".to_string(),
    };
    tracing::info!(item_id, token, %made, elapsed_ms = started.elapsed().as_millis() as u64, "worker: spare line scales sent");
    let _ = evt_tx.send(SubprocessEvent::SpareEdgeScales { item_id, token, outcome });
}

/// The token a run's spare scales go under, when it owes them: its edge
/// cache was written, it kept a photo for them, and its image is on screen.
fn owed_spare_token(edge_cached: bool, spare: &SpareScales, on_screen: &Shown, item_id: u64) -> Option<u64> {
    (edge_cached && spare.is_ready() && on_screen.get() == Some(item_id)).then(next_token)
}

/// Whether owed spare scales are skipped after all: the run was
/// cancelled, the worker is stopping, or the image left the screen.
fn spare_skipped(cancel: &DispatchCancel, stopping: &AtomicBool, on_screen: &Shown, item_id: u64) -> bool {
    cancel.is_cancelled() || stopping.load(Ordering::Acquire) || on_screen.get() != Some(item_id)
}

/// Calculate pixel weight for an image (1 unit = 1M pixels, minimum 1).
fn pixel_weight(image_bytes: &[u8]) -> usize {
    match prunr_core::image_dimensions_from_bytes(image_bytes) {
        Some((w, h)) => ((w as usize * h as usize) / 1_000_000).max(1),
        None => 4, // conservative fallback (~4MP)
    }
}

/// Entry point for `prunr --worker`.
pub fn run_worker() -> ! {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();

    let mut reader = BufReader::new(stdin.lock());
    let (evt_tx, evt_rx) = mpsc::channel::<SubprocessEvent>();

    // Writer thread: evt_rx → stdout (bincode frames). Flush after
    // every event — the parent blocks on `recv` waiting for the next
    // frame, and a coalesced BufWriter would deadlock low-rate event
    // flows (most importantly the post-Init `Ready` which is the
    // only frame between the worker reaching the command loop and the
    // first user dispatch). Per-event flush is one extra `write`
    // syscall per frame; events are sparse (a few per image plus
    // occasional Progress), so the cost is unmeasurable.
    let writer_handle = std::thread::Builder::new()
        .name("worker-writer".into())
        .spawn(move || {
            let mut writer = BufWriter::new(stdout.lock());
            while let Ok(evt) = evt_rx.recv() {
                if write_message(&mut writer, &evt).is_err() {
                    break;
                }
                if prunr_app::subprocess::ipc::flush_writer(&mut writer).is_err() {
                    break;
                }
            }
            let _ = prunr_app::subprocess::ipc::flush_writer(&mut writer);
        })
        .expect("failed to spawn writer thread");

    // Read Init command
    let Ok(Some(SubprocessCommand::Init {
        model, jobs, mask, cpu_only, line_mode, edge, ipc_dir, inpaint_only, line_fade_pct, edge_scales,
    })) = read_message::<_, SubprocessCommand>(&mut reader)
    else {
        let _ = evt_tx.send(SubprocessEvent::InitError {
            error: "Expected Init command".to_string(),
        });
        drop(evt_tx);
        let _ = writer_handle.join();
        std::process::exit(1);
    };
    tracing::info!(
        ?model, jobs, cpu_only, ?line_mode, inpaint_only,
        "worker: Init received, beginning init sequence",
    );

    // Detect backend. `detect_active_provider` probes
    // `OpenVINO::is_available` on Linux which has been
    // observed to hang on some iGPU setups — bracket with tracing so a
    // future hang is diagnosable from the worker log.
    let probe_started = std::time::Instant::now();
    tracing::info!("worker: probing detect_active_provider…");
    let has_gpu = !OrtEngine::detect_active_provider().eq_ignore_ascii_case("CPU");
    tracing::info!(
        elapsed_ms = probe_started.elapsed().as_millis() as u64,
        has_gpu,
        "worker: detect_active_provider returned",
    );
    let on_cpu = cpu_only || !has_gpu;

    // Inpaint-only subprocess: skip seg/edge engine creation entirely.
    // Inpaint sessions are built lazily by `process_inpaint_with` on
    // first dispatch, so the worker can enter the command loop now.
    // The run settings a `Configure` command replaces; the engines a
    // later mode needs are built at its first dispatch.
    let (mut mask, mut line_mode, mut edge) = (mask, line_mode, edge);
    let needs_edge = !inpaint_only && line_mode != LineMode::Off;
    let needs_segmentation = !inpaint_only && line_mode != LineMode::EdgesOnly;

    let line_depth = (!inpaint_only && line_fade_pct > 0).then(|| Arc::new(LineDepth::new(line_fade_pct)));
    let mut traced_scales = edge_scales.of(&edge);
    let mut edge_engine: Option<Arc<EdgeEngine>> = if needs_edge {
        match EdgeEngine::new() {
            Ok(e) => Some(Arc::new(e)),
            Err(e) => {
                let _ = evt_tx.send(SubprocessEvent::InitError {
                    error: format!("Edge engine failed: {e}"),
                });
                drop(evt_tx);
                let _ = writer_handle.join();
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    // Create segmentation engine pool
    let mut engines: Vec<Arc<OrtEngine>> = if needs_segmentation {
        match create_engine_pool(model, jobs, on_cpu) {
            Ok(e) => e,
            Err(e) => {
                let _ = evt_tx.send(SubprocessEvent::InitError {
                    error: format!("Engine pool failed: {e}"),
                });
                drop(evt_tx);
                let _ = writer_handle.join();
                std::process::exit(1);
            }
        }
    } else {
        Vec::new()
    };

    let active_provider = OrtEngine::detect_active_provider();
    tracing::info!(
        active_provider = %active_provider,
        inpaint_only,
        "worker: init complete, sending Ready",
    );
    let _ = evt_tx.send(SubprocessEvent::Ready {
        active_provider: active_provider.clone(),
    });

    // Build rayon pool for parallel inference, sized for the engines a
    // later `Configure` may still build.
    let pool_size = if inpaint_only { 1 } else { prunr_core::batch::engine_pool_size(model, jobs, on_cpu, &active_provider) };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(pool_size)
        .build()
        // Fallback to a default (unsized) rayon pool if the sized builder
        // somehow fails. Default-pool construction failing means rayon itself
        // can't function; no useful recovery possible.
        .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

    // One per dispatch, seg and inpaint alike: a `CancelItem` stops a
    // queued run before it starts and a running one between stages.
    let cancels = DispatchCancels::default();
    let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut engine_idx: usize = 0;
    let first_look = FirstLook::default();
    let on_screen = Shown::default();
    // Set by Shutdown: a run's spare scales are skipped rather than
    // holding the exit for another detector run.
    let stopping = Arc::new(AtomicBool::new(false));

    let ipc_dir = Arc::new(ipc_dir);

    // Weighted memory semaphore: capacity in megapixel units.
    // ~64 units = allows parallel processing of many small images,
    // but throttles large images (40MP = 40 units → only ~1 at a time).
    let semaphore = Arc::new(WeightedSemaphore::new(64));

    // Main command loop
    loop {
        let cmd = match read_message::<_, SubprocessCommand>(&mut reader) {
            Ok(Some(cmd)) => cmd,
            Ok(None) => break, // stdin closed — parent exited
            Err(_) => break,
        };

        match cmd {
            SubprocessCommand::Configure { mask: new_mask, line_mode: new_mode, edge: new_edge, first_look: shown, on_screen: screen } => {
                mask = new_mask;
                line_mode = new_mode;
                edge = new_edge;
                traced_scales = edge_scales.of(&edge);
                first_look.show_for(shown);
                on_screen.set(screen);
                tracing::info!(?line_mode, ?shown, ?screen, "worker: run settings replaced");
            }
            SubprocessCommand::ProcessImage { item_id, image_path, chain_input } => {
                let built = (|| -> Result<(), String> {
                    // A model built here (a compile can run minutes) says so
                    // as it goes, or the parent's hang watchdog kills it.
                    let builds = (line_mode != LineMode::Off && edge_engine.is_none())
                        || (line_mode != LineMode::EdgesOnly && engines.is_empty());
                    let _loading = builds.then(|| Heartbeat::start(evt_tx.clone(), item_id, HEARTBEAT_EVERY));
                    if line_mode != LineMode::Off {
                        ensure_edge_engine(&mut edge_engine)?;
                    }
                    if line_mode != LineMode::EdgesOnly && engines.is_empty() {
                        let started = std::time::Instant::now();
                        engines = create_engine_pool(model, jobs, on_cpu).map_err(|e| format!("Engine pool failed: {e}"))?;
                        tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "worker: engine pool built on first use");
                    }
                    Ok(())
                })();
                if let Err(error) = built {
                    let _ = evt_tx.send(SubprocessEvent::ImageError { item_id, error });
                    cleanup_image_inputs(&ipc_dir, &image_path, chain_input.as_ref().map(|c| c.path.as_path()));
                    continue;
                }
                let engine = if !engines.is_empty() {
                    Some(engines[engine_idx % engines.len()].clone())
                } else {
                    None
                };
                engine_idx += 1;
                in_flight.fetch_add(1, Ordering::AcqRel);

                let evt_tx = evt_tx.clone();
                let in_flight = in_flight.clone();
                let edge_eng = edge_engine.clone();
                let line_depth = line_depth.clone();
                let provider = active_provider.clone();
                let sem = semaphore.clone();
                let ipc = ipc_dir.clone();
                let first_look = first_look.clone();
                let on_screen = on_screen.clone();
                let stopping = stopping.clone();
                let engine_count = engines.len();
                let cancel = cancels.register(item_id);

                pool.spawn(move || {
                    // Hoist chain_path so every early-return path can clean up
                    // the parent-written temp files (img + optional chain) via
                    // `cleanup_ipc_temps`. Forgetting the chain leaks one
                    // ~50 MB file in `/dev/shm` per cancelled / failed dispatch.
                    let chain_path = chain_input.as_ref().map(|ci| ci.path.clone());
                    let chain_img: Option<image::DynamicImage> = chain_input.and_then(|ci| {
                        let data = std::fs::read(&ci.path).ok()?;
                        let rgba = image::RgbaImage::from_raw(ci.width, ci.height, data)?;
                        Some(image::DynamicImage::ImageRgba8(rgba))
                    });
                    let cleanup_inputs = || cleanup_image_inputs(&ipc, &image_path, chain_path.as_deref());
                    // An early exit: report the error, drop the inputs, leave the in-flight count.
                    let fail = |error: String| {
                        let _ = evt_tx.send(SubprocessEvent::ImageError { item_id, error });
                        cleanup_inputs();
                        in_flight.fetch_sub(1, Ordering::AcqRel);
                    };

                    if cancel.is_cancelled() {
                        fail(CANCELLED_ERR_MSG.into());
                        return;
                    }

                    // Read image bytes from temp file
                    let img_bytes = match std::fs::read(&image_path) {
                        Ok(b) => b,
                        Err(e) => {
                            fail(format!("Failed to read image: {e}"));
                            return;
                        }
                    };

                    // Weighted semaphore: acquire pixel units proportional to image size.
                    // Small images run in parallel; large images throttle automatically.
                    // Guard releases on drop (panic-safe).
                    let weight = pixel_weight(&img_bytes);
                    let sem_guard = sem.acquire(weight);
                    let progress = ipc_progress(item_id, evt_tx.clone()).with_cancel(cancel.flag());
                    progress.step(prunr_core::Step::ReadingImage);
                    // Whether this image is the one on screen is settled
                    // as its run starts, here.
                    let look = first_look.for_run(item_id, model, engine_count);
                    let mut spare = SpareScales::new(edge_scales.spare(&edge, item_id, on_screen.get()));

                    // Off and non-chained SubjectOutline keep the seg tensor
                    // for mask reruns.
                    let mut tensor_for_cache: Option<(Vec<f32>, u32, u32)> = None;
                    let mut edge_tensor_for_cache: Option<prunr_core::EdgeInferenceResult> = None;

                    let result = match line_mode {
                        LineMode::EdgesOnly => {
                            let decoded;
                            let img_ref = if let Some(ref img) = chain_img {
                                Ok(img as &image::DynamicImage)
                            } else {
                                match prunr_core::load_image_from_bytes(&img_bytes) {
                                    Ok(img) => { decoded = img; Ok(&decoded as &image::DynamicImage) }
                                    Err(e) => Err(e),
                                }
                            };
                            img_ref.and_then(|img| {
                                progress.step(prunr_core::Step::DrawingLines);
                                // invariant: ensure_edge_engine ran above for every line mode but Off.
                                let eng_ref = edge_eng.as_ref().unwrap();
                                let transformed = prunr_core::apply_input_transform(img, edge.input_transform);
                                spare.trace(eng_ref, &transformed, traced_scales).map(|res| {
                                    let measured = line_depth.as_ref().and_then(|d| d.measure(img));
                                    let rgba_image = prunr_core::compose_whole_photo(&res, img, &edge, measured.as_ref().map(Measured::fade));
                                    edge_tensor_for_cache = Some(res);
                                    ProcessResult {
                                        rgba_image,
                                        active_provider: provider.clone(),
                                    }
                                })
                            })
                        }
                        LineMode::SubjectOutline => {
                            if let Some(ref img) = chain_img {
                                // A run on an earlier result: that image already is the
                                // cut-out, effect included, so the lines are traced from
                                // it directly. No seg cache — no seg tensor made it.
                                progress.step(prunr_core::Step::DrawingLines);
                                // invariant: ensure_edge_engine ran above for every line mode but Off.
                                let eng_ref = edge_eng.as_ref().unwrap();
                                let measured = line_depth.as_ref().and_then(|d| d.measure(img));
                                prunr_core::outline_subject(img.to_rgba8(), prunr_core::SubjectEffect::None, &edge, measured.as_ref().map(Measured::fade), |i| spare.trace(eng_ref, i, traced_scales))
                                    .map(|(rgba_image, res)| {
                                        edge_tensor_for_cache = Some(res);
                                        ProcessResult { rgba_image, active_provider: provider.clone() }
                                    })
                            } else {
                                // Non-chain: split pipeline so the seg tensor is captured
                                // for Tier 2 mask reruns and cross-mode transitions.
                                let Some(eng) = engine.as_ref() else {
                                    fail(SEG_ENGINE_MISSING.into());
                                    return;
                                };
                                let decoded = match prunr_core::load_image_from_bytes(&img_bytes) {
                                    Ok(img) => img,
                                    Err(e) => {
                                        fail(e.to_string());
                                        return;
                                    }
                                };
                                let original = &decoded;
                                if let Some(err) = prunr_core::check_large_image(original) {
                                    Err(err)
                                } else {
                                    let look = look.as_ref().and_then(Look::engine);
                                    prunr_core::infer_after_first_look(original, eng, look.as_deref(), &progress).and_then(|ir| {
                                        progress.step(prunr_core::Step::RefiningEdges);
                                        let th = ir.tensor_height;
                                        let tw = ir.tensor_width;
                                        let active_provider = ir.active_provider.clone();
                                        let plain = prunr_core::MaskSettings { effect: prunr_core::SubjectEffect::None, ..mask };
                                        let subject = prunr_core::postprocess_from_flat(
                                            &ir.tensor_data, th, tw, original,
                                            &prunr_core::PostprocessOpts::new(&plain, model),
                                        )?;
                                        // Seg tensor captured → Tier 2 mask reruns work in
                                        // SubjectOutline; Off ↔ SubjectOutline can reuse it.
                                        tensor_for_cache = Some((ir.tensor_data, th as u32, tw as u32));
                                        progress.step(prunr_core::Step::DrawingLines);
                                        // invariant: ensure_edge_engine ran above for every line mode but Off.
                                        let eng_ref = edge_eng.as_ref().unwrap();
                                        let measured = line_depth.as_ref().and_then(|d| d.measure(original));
                                        let (rgba_image, edge_res) = prunr_core::outline_subject(
                                            subject, mask.effect, &edge, measured.as_ref().map(Measured::fade), |img| spare.trace(eng_ref, img, traced_scales),
                                        )?;
                                        edge_tensor_for_cache = Some(edge_res);
                                        Ok(ProcessResult { rgba_image, active_provider })
                                    })
                                }
                            }
                        }
                        LineMode::Off => {
                            // Split pipeline: infer_only → tensor_to_mask → apply_mask
                            // This captures the raw tensor for future Tier 2 mask reruns.
                            let Some(eng) = engine.as_ref() else {
                                fail(SEG_ENGINE_MISSING.into());
                                return;
                            };
                            let decoded;
                            let original: &image::DynamicImage = if let Some(ref img) = chain_img {
                                img
                            } else {
                                match prunr_core::load_image_from_bytes(&img_bytes) {
                                    Ok(img) => { decoded = img; &decoded }
                                    Err(e) => {
                                        fail(e.to_string());
                                        return;
                                    }
                                }
                            };

                            // Check large image limit
                            if let Some(err) = prunr_core::check_large_image(original) {
                                Err(err)
                            } else {
                                let look = look.as_ref().and_then(Look::engine);
                                prunr_core::infer_after_first_look(original, eng, look.as_deref(), &progress).and_then(|ir| {
                                    progress.step(prunr_core::Step::RefiningEdges);

                                    let th = ir.tensor_height;
                                    let tw = ir.tensor_width;
                                    tracing::debug!(
                                        item_id, th, tw, tensor_len = ir.tensor_data.len(),
                                        provider = %ir.active_provider,
                                        "worker about to postprocess_from_flat",
                                    );
                                    let rgba_image = prunr_core::postprocess_from_flat(
                                        &ir.tensor_data, th, tw, original,
                                        &prunr_core::PostprocessOpts::new(&mask, model),
                                    )?;
                                    let (w, h) = (rgba_image.width(), rgba_image.height());
                                    let (mut a_min, mut a_max) = (255u8, 0u8);
                                    let mut a_sum = 0u64;
                                    for p in rgba_image.pixels() {
                                        let a = p.0[3];
                                        if a < a_min { a_min = a; }
                                        if a > a_max { a_max = a; }
                                        a_sum += a as u64;
                                    }
                                    let a_mean = a_sum / (w as u64 * h as u64).max(1);
                                    tracing::debug!(
                                        item_id, w, h, a_min, a_max, a_mean,
                                        "worker postprocess result alpha stats",
                                    );

                                    // Stash tensor for cache output
                                    tensor_for_cache = Some((
                                        ir.tensor_data,
                                        th as u32,
                                        tw as u32,
                                    ));

                                    Ok(ProcessResult {
                                        rgba_image,
                                        active_provider: ir.active_provider,
                                    })
                                })
                            }
                        }
                    };

                    // Background color is applied by the parent at display/export time
                    // (non-destructive — result_rgba always stores the transparent version)

                    let mut spare_token = None;
                    match result {
                        Ok(pr) => {
                            // Write tensor cache to temp file if available
                            let (tcp, tch, tcw) = if let Some((ref tdata, th, tw)) = tensor_for_cache {
                                let tp = IpcKind::Tensor.path_for(&ipc, item_id);
                                let bytes = prunr_app::subprocess::ipc::f32s_as_le_bytes(tdata);
                                tracing::debug!(item_id, th, tw, tensor_len = tdata.len(), bytes_len = bytes.len(), path = %tp.display(), "writing seg tensor");
                                match std::fs::write(&tp, bytes) {
                                    Ok(()) => (Some(tp), Some(th), Some(tw)),
                                    Err(e) => {
                                        tracing::error!(item_id, %e, path = %tp.display(), "seg tensor write failed");
                                        (None, None, None)
                                    }
                                }
                            } else {
                                (None, None, None)
                            };

                            let edge_cache = edge_tensor_for_cache.as_ref()
                                .and_then(|res| write_edge_cache(IpcKind::Edge.path_for(&ipc, item_id), item_id, res));
                            let spare_scales = owed_spare_token(edge_cache.is_some(), &spare, &on_screen, item_id);

                            // Write result RGBA to temp file
                            let (w, h) = (pr.rgba_image.width(), pr.rgba_image.height());
                            let active_provider = pr.active_provider.clone();
                            let result_path = IpcKind::Result.path_for(&ipc, item_id);
                            let written = write_result(&result_path, &pr.rgba_image);
                            // Free the result RGBA (~50 MB at 4 K) before
                            // posting the event — under N-job parallelism
                            // this drops N×50 MB of redundant retention
                            // during the brief window before the next
                            // dispatch overwrites `pr`.
                            drop(pr);
                            match written {
                                Ok(()) => {
                                    spare_token = spare_scales;
                                    let _ = evt_tx.send(SubprocessEvent::ImageDone {
                                        item_id,
                                        result_path,
                                        width: w,
                                        height: h,
                                        active_provider,
                                        tensor_cache_path: tcp,
                                        tensor_cache_height: tch,
                                        tensor_cache_width: tcw,
                                        edge_cache,
                                        spare_scales,
                                    });
                                }
                                Err(error) => {
                                    let _ = evt_tx.send(SubprocessEvent::ImageError { item_id, error });
                                }
                            }
                        }
                        Err(e) => {
                            let _ = evt_tx.send(SubprocessEvent::ImageError {
                                item_id,
                                error: e.to_string(),
                            });
                        }
                    }

                    send_rss(&evt_tx);

                    cleanup_inputs();
                    if let Some(token) = spare_token {
                        drop(sem_guard);
                        let skip = spare_skipped(&cancel, &stopping, &on_screen, item_id);
                        send_spare_scales(spare, token, item_id, edge_eng.as_deref().filter(|_| !skip), &ipc, &evt_tx);
                    }
                    in_flight.fetch_sub(1, Ordering::AcqRel);
                });
            }

            SubprocessCommand::RePostProcess {
                item_id, tensor_path, tensor_height, tensor_width,
                model: cmd_model, original_image_path, mask: cmd_mask,
            } => {
                let evt_tx = evt_tx.clone();
                let in_flight = in_flight.clone();
                let sem = semaphore.clone();
                let ipc = ipc_dir.clone();
                let cancel = cancels.register(item_id);

                in_flight.fetch_add(1, Ordering::AcqRel);
                pool.spawn(move || {
                    if cancel.is_cancelled() {
                        let _ = evt_tx.send(SubprocessEvent::ImageError {
                            item_id, error: CANCELLED_ERR_MSG.into(),
                        });
                        cleanup_ipc_temps(&ipc, &[&tensor_path, &original_image_path]);
                        in_flight.fetch_sub(1, Ordering::AcqRel);
                        return;
                    }
                    // Conservative weight — postprocess upscales tensor to original resolution
                    // which allocates significant memory for Lanczos3 + guided filter
                    let _sem_guard = sem.acquire(10);
                    let progress = ipc_progress(item_id, evt_tx.clone());

                    // Read tensor from temp file
                    let tensor_result = (|| -> Result<image::RgbaImage, String> {
                        let raw_bytes = std::fs::read(&tensor_path)
                            .map_err(|e| format!("Failed to read tensor: {e}"))?;
                        let floats = prunr_app::subprocess::ipc::le_bytes_to_f32s(&raw_bytes);
                        let img_bytes = std::fs::read(&original_image_path)
                            .map_err(|e| format!("Failed to read original: {e}"))?;
                        let original = prunr_core::load_image_from_bytes(&img_bytes)
                            .map_err(|e| format!("Failed to decode original: {e}"))?;
                        progress.step(prunr_core::Step::RefiningEdges);
                        prunr_core::postprocess_from_flat(
                            &floats, tensor_height as usize, tensor_width as usize,
                            &original, &prunr_core::PostprocessOpts::new(&cmd_mask, cmd_model),
                        ).map_err(|e| e.to_string())
                    })();

                    match tensor_result {
                        Ok(rgba) => {
                            let (w, h) = (rgba.width(), rgba.height());
                            let result_path = IpcKind::Result.path_for(&ipc, item_id);
                            let written = write_result(&result_path, &rgba);
                            drop(rgba);
                            let _ = evt_tx.send(match written {
                                Ok(()) => SubprocessEvent::ImageDone {
                                    item_id,
                                    result_path,
                                    width: w,
                                    height: h,
                                    active_provider: String::new(),
                                    tensor_cache_path: None,
                                    tensor_cache_height: None,
                                    tensor_cache_width: None,
                                    edge_cache: None,
                                    spare_scales: None,
                                },
                                Err(error) => SubprocessEvent::ImageError { item_id, error },
                            });
                        }
                        Err(e) => {
                            let _ = evt_tx.send(SubprocessEvent::ImageError { item_id, error: e });
                        }
                    }

                    cleanup_ipc_temps(&ipc, &[&tensor_path, &original_image_path]);

                    send_rss(&evt_tx);

                    in_flight.fetch_sub(1, Ordering::AcqRel);
                });
            }

            SubprocessCommand::AddEdgeInference {
                item_id, image_path, seg_tensor_path,
                seg_tensor_height, seg_tensor_width, model: cmd_model,
                mask: cmd_mask,
            } => {
                if let Err(error) = ensure_edge_engine(&mut edge_engine) {
                    let _ = evt_tx.send(SubprocessEvent::ImageError { item_id, error });
                    cleanup_ipc_temps(&ipc_dir, &[&seg_tensor_path, &image_path]);
                    continue;
                }
                let evt_tx = evt_tx.clone();
                let in_flight = in_flight.clone();
                let sem = semaphore.clone();
                let ipc = ipc_dir.clone();
                let edge_eng = edge_engine.clone();
                let line_depth = line_depth.clone();
                let on_screen = on_screen.clone();
                let stopping = stopping.clone();
                let cancel = cancels.register(item_id);

                in_flight.fetch_add(1, Ordering::AcqRel);
                pool.spawn(move || {
                    if cancel.is_cancelled() {
                        let _ = evt_tx.send(SubprocessEvent::ImageError {
                            item_id, error: CANCELLED_ERR_MSG.into(),
                        });
                        cleanup_ipc_temps(&ipc, &[&seg_tensor_path, &image_path]);
                        in_flight.fetch_sub(1, Ordering::AcqRel);
                        return;
                    }
                    let sem_guard = sem.acquire(10);
                    let progress = ipc_progress(item_id, evt_tx.clone());
                    let mut spare = SpareScales::new(edge_scales.spare(&edge, item_id, on_screen.get()));

                    // seg_tensor_path is both our input and the output we hand
                    // back as tensor_cache_path — the parent's read_raw_seg_tensor
                    // reads-and-deletes it. So here we `read`, not `read_and_delete`.
                    let produce = (|| -> Result<(image::RgbaImage, prunr_core::EdgeInferenceResult), String> {
                        let raw_bytes = std::fs::read(&seg_tensor_path)
                            .map_err(|e| format!("Failed to read seg tensor: {e}"))?;
                        let seg_data = prunr_app::subprocess::ipc::le_bytes_to_f32s(&raw_bytes);

                        let img_bytes = std::fs::read(&image_path)
                            .map_err(|e| format!("Failed to read image: {e}"))?;
                        let original = prunr_core::load_image_from_bytes(&img_bytes)
                            .map_err(|e| format!("Failed to decode image: {e}"))?;

                        progress.step(prunr_core::Step::RefiningEdges);
                        let plain = prunr_core::MaskSettings { effect: prunr_core::SubjectEffect::None, ..cmd_mask };
                        let subject = prunr_core::postprocess_from_flat(
                            &seg_data, seg_tensor_height as usize, seg_tensor_width as usize,
                            &original, &prunr_core::PostprocessOpts::new(&plain, cmd_model),
                        ).map_err(|e| e.to_string())?;

                        progress.step(prunr_core::Step::DrawingLines);
                        let eng_ref = edge_eng.as_ref()
                            .ok_or_else(|| "edge engine not initialized in this subprocess".to_string())?;
                        let measured = line_depth.as_ref().and_then(|d| d.measure(&original));
                        prunr_core::outline_subject(subject, cmd_mask.effect, &edge, measured.as_ref().map(Measured::fade), |img| spare.trace(eng_ref, img, traced_scales))
                            .map_err(|e| e.to_string())
                    })();

                    let mut spare_token = None;
                    match produce {
                        Ok((rgba, edge_res)) => {
                            let (w, h) = (rgba.width(), rgba.height());
                            let result_path = IpcKind::Result.path_for(&ipc, item_id);

                            let edge_cache = write_edge_cache(IpcKind::Edge.path_for(&ipc, item_id), item_id, &edge_res);
                            let spare_scales = owed_spare_token(edge_cache.is_some(), &spare, &on_screen, item_id);

                            let written = write_result(&result_path, &rgba);
                            drop(rgba);
                            if let Err(error) = written {
                                if let Some(file) = &edge_cache {
                                    let _ = std::fs::remove_file(&file.path);
                                }
                                cleanup_ipc_temps(&ipc, &[&seg_tensor_path]);
                                let _ = evt_tx.send(SubprocessEvent::ImageError { item_id, error });
                            } else {
                                spare_token = spare_scales;
                                let _ = evt_tx.send(SubprocessEvent::ImageDone {
                                    item_id,
                                    result_path,
                                    width: w,
                                    height: h,
                                    // Empty active_provider — seg didn't run this
                                    // tier; parent treats it like a Tier 2 rerun
                                    // and keeps the existing backend label.
                                    active_provider: String::new(),
                                    tensor_cache_path: Some(seg_tensor_path.clone()),
                                    tensor_cache_height: Some(seg_tensor_height),
                                    tensor_cache_width: Some(seg_tensor_width),
                                    edge_cache,
                                    spare_scales,
                                });
                            }
                        }
                        Err(e) => {
                            let _ = evt_tx.send(SubprocessEvent::ImageError { item_id, error: e });
                            cleanup_ipc_temps(&ipc, &[&seg_tensor_path]);
                        }
                    }

                    cleanup_ipc_temps(&ipc, &[&image_path]);
                    send_rss(&evt_tx);
                    if let Some(token) = spare_token {
                        drop(sem_guard);
                        let skip = spare_skipped(&cancel, &stopping, &on_screen, item_id);
                        send_spare_scales(spare, token, item_id, edge_eng.as_deref().filter(|_| !skip), &ipc, &evt_tx);
                    }
                    in_flight.fetch_sub(1, Ordering::AcqRel);
                });
            }

            SubprocessCommand::CancelItem { item_id } => cancels.cancel(item_id),

            SubprocessCommand::Inpaint { item_id, model_id, image_path, mask_path, sd_req, feather_px, sharpen } => {
                // Inpaint runs independent of the seg/edge engine pool —
                // a fresh thread per dispatch keeps inference from
                // blocking batched seg work. Bump `in_flight` so a
                // concurrent `Shutdown` waits for the SD UNet loop to
                // actually finish rather than declaring drain done while
                // the spawned thread is still alive and queueing events.
                // Decrement is in `Cleanup::drop` so panics still pair
                // correctly.
                in_flight.fetch_add(1, Ordering::AcqRel);
                let evt_tx = evt_tx.clone();
                let inpaint_cancel = cancels.register(item_id);
                // Reports go out on this stroke's own thread, ahead of its
                // InpaintDone on the same channel.
                let progress = ipc_progress(item_id, evt_tx.clone())
                    .with_cancel(inpaint_cancel.flag())
                    .with_tile_previews(TILE_PREVIEW_EDGE);
                let in_flight_for_thread = in_flight.clone();
                // Clones reserved for the spawn-failure unwind path
                // — the closure below moves the originals.
                let evt_tx_on_spawn_err = evt_tx.clone();
                // Pre-clone the parent-written PNG paths so the spawn-
                // failure path can delete them. Without this, a thread-
                // exhaustion error left ~20 MB of `inpaint-img-*` /
                // `inpaint-mask-*` per failed dispatch in /dev/shm
                // until the parent restarted (the dead-PID sweep on
                // next launch was the only collector).
                let image_path_on_spawn_err = image_path.clone();
                let mask_path_on_spawn_err = mask_path.clone();
                // Use the parent-supplied ipc_dir so the result PNG lands in
                // the same /dev/shm/prunr-ipc-<parent_pid>/ that the parent
                // reads from. Calling ipc_temp_dir() here would resolve via
                // the CHILD's pid, leaking an orphan dir per spawn (no
                // cleanup ever sweeps it because the child exits).
                let ipc_dir_for_spawn = ipc_dir.clone();
                let spawn_result = std::thread::Builder::new()
                    .name("prunr-inpaint".into())
                    .spawn(move || {
                    // Unregisters the cancel and then decrements in_flight,
                    // panics included, so a `Shutdown` waiting on zero
                    // never sees this run still registered.
                    struct Cleanup {
                        in_flight: Arc<std::sync::atomic::AtomicUsize>,
                        cancel: Option<DispatchCancel>,
                    }
                    impl Drop for Cleanup {
                        fn drop(&mut self) {
                            drop(self.cancel.take());
                            self.in_flight.fetch_sub(1, Ordering::AcqRel);
                        }
                    }
                    let _cleanup = Cleanup { in_flight: in_flight_for_thread, cancel: Some(inpaint_cancel) };
                    let result = (|| -> Result<(image::RgbaImage, f32), String> {
                        let image: image::RgbaImage = read_image(&image_path)
                            .ok_or_else(|| "read source: missing or short".to_string())?;
                        let mask: image::GrayImage = read_image(&mask_path)
                            .ok_or_else(|| "read mask: missing or short".to_string())?;
                        let filled = prunr_core::inpaint::process_inpaint_with(
                            &image, &mask, model_id, sd_req.clone(),
                            sd_req.as_ref().is_some_and(|r| r.tuning.cpu_only), &progress,
                        ).map_err(|e| e.message())?;
                        progress.step(prunr_core::Step::Blending);
                        Ok((prunr_core::inpaint_blend::finalize_inpaint(
                            filled.rgba, &image, &mask, feather_px, sharpen,
                        ), filled.scale))
                    })();
                    drop(_cleanup);
                    match result {
                        Ok((rgba, fill_scale)) => {
                            let path = IpcKind::InpaintOut.path_for(&ipc_dir_for_spawn, item_id);
                            let event = match write_result(&path, &rgba) {
                                Ok(()) => SubprocessEvent::InpaintDone { item_id, rgba_path: path, width: rgba.width(), height: rgba.height(), fill_scale },
                                Err(error) => SubprocessEvent::InpaintError { item_id, error },
                            };
                            let _ = evt_tx.send(event);
                        }
                        Err(e) => {
                            let _ = evt_tx.send(SubprocessEvent::InpaintError {
                                item_id,
                                error: e,
                            });
                        }
                    }
                });
                // Builder::spawn returns Result; std::thread::spawn would
                // panic on resource exhaustion. On failure we've already
                // bumped `in_flight` and inserted the cancel flag, so
                // unwind both before erroring out — otherwise the worker
                // never completes its drain.
                if let Err(e) = spawn_result {
                    // The dropped closure took the cancel with it.
                    in_flight.fetch_sub(1, Ordering::AcqRel);
                    // The closure never ran, so its input PNGs are still
                    // there — clean them before the InpaintError lands so
                    // /dev/shm doesn't accumulate.
                    let _ = std::fs::remove_file(&image_path_on_spawn_err);
                    let _ = std::fs::remove_file(&mask_path_on_spawn_err);
                    let _ = evt_tx_on_spawn_err.send(SubprocessEvent::InpaintError {
                        item_id,
                        error: format!("inpaint thread spawn failed: {e}"),
                    });
                }
            }
            SubprocessCommand::Shutdown => {
                stopping.store(true, Ordering::Release);
                // Wait for in-flight to drain
                while in_flight.load(Ordering::Acquire) > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                let _ = evt_tx.send(SubprocessEvent::Finished);
                drop(evt_tx);
                let _ = writer_handle.join();
                std::process::exit(0);
            }

            SubprocessCommand::Init { .. } => {
                // Duplicate Init — ignore
            }
        }
    }

    // stdin closed — clean exit
    drop(evt_tx);
    let _ = writer_handle.join();
    std::process::exit(0);
}

/// Fade with distance in the worker (the CLI's runs): the depth model,
/// built on first use and shared by the pool, and how much to fade.
struct LineDepth {
    measurer: prunr_core::depth::Measurer,
    amount: f32,
}

impl LineDepth {
    fn new(fade_pct: u8) -> Self {
        Self { measurer: prunr_core::depth::Measurer::lines_only(), amount: prunr_core::math::pct(fade_pct) }
    }

    /// `photo`'s depth map; `None` (lines drawn unfaded) when the model
    /// is missing or the run fails, which is logged.
    fn measure(&self, photo: &image::DynamicImage) -> Option<Measured> {
        self.measurer.measure_decoded(photo)
            .map_err(|err| tracing::warn!(%err, "worker: depth run failed; lines drawn without fading"))
            .ok()
            .map(|map| Measured { map, amount: self.amount })
    }
}

/// One photo's depth map and how much its lines fade.
struct Measured {
    map: prunr_core::depth::DepthMap,
    amount: f32,
}

impl Measured {
    fn fade(&self) -> prunr_core::depth::LineFade<'_> {
        prunr_core::depth::LineFade { depth: &self.map, amount: self.amount }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// While a model builds, the worker reports it again and again, so a
    /// long compile is not taken for a hang; it stops with the build.
    #[test]
    fn a_model_being_built_keeps_reporting() {
        use std::sync::mpsc::RecvTimeoutError;
        use std::time::Duration;
        // Waits on events, not sleeps: a loaded CI runner stalls threads
        // for longer than any fixed sleep.
        let wait = Duration::from_secs(10);
        let (tx, rx) = std::sync::mpsc::channel();
        let beat = Heartbeat::start(tx, 7, Duration::from_millis(20));
        for _ in 0..3 {
            let report = rx.recv_timeout(wait).expect("a report while building");
            assert!(matches!(report, SubprocessEvent::Report { item_id: 7, .. }));
        }
        drop(beat);
        // The beat thread owns the sender, so the channel closes when it ends.
        let end = loop {
            if let Err(e) = rx.recv_timeout(wait) {
                break e;
            }
        };
        assert_eq!(end, RecvTimeoutError::Disconnected, "it stops with the build");
    }

    /// A result that cannot be written is an error for the parent, with no
    /// half file left; one that can is the whole image.
    #[test]
    fn a_result_that_cannot_be_written_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let rgba = image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 4]));
        let blocked = dir.path().join("result_0.raw");
        std::fs::create_dir(&blocked).unwrap();
        let err = write_result(&blocked, &rgba).unwrap_err();
        assert!(err.starts_with("Failed to write result"), "{err}");
        let missing = dir.path().join("gone").join("result_1.raw");
        assert!(write_result(&missing, &rgba).is_err());
        assert!(!missing.exists());
        let fine = dir.path().join("result_2.raw");
        write_result(&fine, &rgba).unwrap();
        assert_eq!(std::fs::read(&fine).unwrap(), rgba.as_raw().as_slice());
    }
    use std::time::{Duration, Instant};

    /// Block until the semaphore has at least `n` waiters parked. Polls
    /// the internal waiter queue rather than sleeping a fixed window, so
    /// the test stays deterministic on slow CI runners.
    fn wait_for_waiters(sem: &WeightedSemaphore, n: usize) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            let len = sem.state.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .waiters.len();
            if len >= n { return; }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("expected ≥{n} waiters in 2s — test setup wedged");
    }

    /// Pins the no-starvation invariant: a giant acquirer at the head
    /// of the queue must not be repeatedly skipped past by smaller
    /// arrivals. Smaller waiters behind the giant only proceed if their
    /// weight still leaves room for the giant's weight.
    #[test]
    fn weighted_semaphore_giant_at_head_blocks_smaller_arrivals() {
        let sem = Arc::new(WeightedSemaphore::new(64));

        // Hold 50 units so a 40-unit acquirer cannot proceed yet.
        let _hold = sem.acquire(50);

        // Spawn a 40-unit acquirer — it queues at head, blocked on the
        // 14 free units.
        let sem_giant = sem.clone();
        let giant_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let giant_done_for_thread = giant_done.clone();
        let giant_thread = std::thread::spawn(move || {
            let _g = sem_giant.acquire(40);
            giant_done_for_thread.store(true, std::sync::atomic::Ordering::Release);
        });
        wait_for_waiters(&sem, 1);

        // Now spawn a 5-unit acquirer. Pre-fix it would have grabbed the
        // 14 free units past the giant. Post-fix it must wait — its
        // 5 + giant's 40 = 45 > 14.
        let sem_small = sem.clone();
        let small_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let small_done_for_thread = small_done.clone();
        let small_thread = std::thread::spawn(move || {
            let _g = sem_small.acquire(5);
            small_done_for_thread.store(true, std::sync::atomic::Ordering::Release);
        });
        wait_for_waiters(&sem, 2);

        // Both waiters are now parked. Give the runtime a chance to wake
        // a buggy small-acquirer past the giant — if the fairness rule
        // were broken, the small would have completed by now.
        std::thread::sleep(Duration::from_millis(20));
        assert!(!small_done.load(std::sync::atomic::Ordering::Acquire),
            "small acquirer skipped past starving giant");
        assert!(!giant_done.load(std::sync::atomic::Ordering::Acquire),
            "giant should still be waiting (50 + 40 > 64)");

        // Releasing the 50-unit hold lets the giant proceed.
        drop(_hold);

        // Both should complete shortly after the release.
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if giant_done.load(std::sync::atomic::Ordering::Acquire)
                && small_done.load(std::sync::atomic::Ordering::Acquire)
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let _ = giant_thread.join();
        let _ = small_thread.join();
        assert!(giant_done.load(std::sync::atomic::Ordering::Acquire),
            "giant should have completed once head's hold released");
        assert!(small_done.load(std::sync::atomic::Ordering::Acquire),
            "small should have completed after giant cleared the head");
    }

    /// Concurrent small acquirers must run in parallel (the fairness rule
    /// only kicks in when a larger waiter is at the head).
    #[test]
    fn weighted_semaphore_small_acquirers_run_concurrently() {
        let sem = Arc::new(WeightedSemaphore::new(64));
        let live = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let max_live = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let sem = sem.clone();
            let live = live.clone();
            let max_live = max_live.clone();
            handles.push(std::thread::spawn(move || {
                let _g = sem.acquire(2);
                let now = live.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1;
                let _ = max_live.fetch_max(now, std::sync::atomic::Ordering::AcqRel);
                std::thread::sleep(Duration::from_millis(40));
                live.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            }));
        }
        for h in handles { h.join().unwrap(); }
        assert!(max_live.load(std::sync::atomic::Ordering::Acquire) >= 4,
            "expected at least 4 concurrent small acquirers, got {}",
            max_live.load(std::sync::atomic::Ordering::Acquire));
    }
}
