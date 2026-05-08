//! Processing-pipeline coordinator: subprocess worker channels, admission
//! control, live preview, and per-batch dispatch state.
//!
//! Owns:
//! - The worker bridge channels (`worker_tx` / `worker_rx`) — UI thread sends
//!   `WorkerMessage`, drains `WorkerResult` non-blockingly each frame.
//! - The shared cancellation flag (`Arc<AtomicBool>`) — read by the worker
//!   bridge to short-circuit a batch in flight.
//! - The in-process Tier 2 live-preview dispatcher.
//! - Admission controller state during streaming batches.
//! - The dispatch-time recipe snapshot (used to attribute completed results
//!   to the settings that produced them, even if the user keeps editing).
//! - The periodic history-cleanup timestamp.
//!
//! Does NOT own:
//! - The worker bridge thread itself — that's spawned by `worker::spawn_worker`
//!   at app startup. We just hold the channel ends.
//! - `BatchManager` (per the cross-coordinator borrow rule). Methods that
//!   operate on the batch take `&mut BatchManager` per call, never as a field.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use prunr_core::ProcessingRecipe;

use super::inpaint_bridge::{InpaintBridgeMsg, InpaintBridgeResult, spawn_inpaint_bridge};
use super::live_preview::LivePreview;
use super::memory::AdmissionController;
use super::worker::{WorkerMessage, WorkerResult, WorkItem};

#[derive(Clone)]
pub struct CancelRegistry {
    global: Arc<AtomicBool>,
    // Short-circuit for the common zero-cancel case: `is_cancelled` skips the
    // mutex entirely unless some per-item entry has been requested.
    has_per_item: Arc<AtomicBool>,
    per_item: Arc<Mutex<HashMap<u64, Arc<AtomicBool>>>>,
}

impl CancelRegistry {
    pub(crate) fn new() -> Self {
        Self {
            global: Arc::new(AtomicBool::new(false)),
            has_per_item: Arc::new(AtomicBool::new(false)),
            per_item: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub(crate) fn is_global_cancelled(&self) -> bool {
        self.global.load(Ordering::Acquire)
    }

    pub(crate) fn is_cancelled(&self, item_id: u64) -> bool {
        if self.is_global_cancelled() {
            return true;
        }
        if !self.has_per_item.load(Ordering::Acquire) {
            return false;
        }
        let guard = self.per_item.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.get(&item_id).is_some_and(|f| f.load(Ordering::Acquire))
    }

    pub(crate) fn request_global_cancel(&self) {
        self.global.store(true, Ordering::Release);
    }

    pub(crate) fn request_item_cancel(&self, item_id: u64) {
        let mut guard = self.per_item.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.entry(item_id)
            .or_insert_with(|| Arc::new(AtomicBool::new(false)))
            .store(true, Ordering::Release);
        self.has_per_item.store(true, Ordering::Release);
    }

    pub(crate) fn reset(&self) {
        self.global.store(false, Ordering::Release);
        let mut guard = self.per_item.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.clear();
        self.has_per_item.store(false, Ordering::Release);
    }
}

/// Active dispatch's recipe + the set of items still expected to deliver.
/// All items in a batch share one recipe (the toolbar broadcasts current
/// settings at dispatch). `take_recipe` removes an item; the slot self-
/// cleans when the last pending item completes — so a late ImageDone after
/// a settings edit can't pick up the wrong recipe.
struct InFlightBatch {
    recipe: ProcessingRecipe,
    pending: HashSet<u64>,
}

pub(crate) struct InpaintResult {
    pub item_id: u64,
    pub rgba: image::RgbaImage,
    pub generation: u64,
    /// True when the worker exited via `CoreError::Cancelled`. Drives
    /// the "Erase cancelled" toast in `drain_inpaint_results` so the
    /// user gets explicit feedback that Esc / the Cancel button took
    /// effect (vs. a silent failure or a stale-result drop).
    pub cancelled: bool,
    /// Set when the worker returned a non-Cancelled error (e.g. the
    /// SD-bundle RAM guard refused to load). Surfaced as a toast in
    /// `drain_inpaint_results` so the user sees WHY the stroke did
    /// nothing instead of guessing it was lost.
    pub error: Option<String>,
}

/// Eraser-specific tuning passed from `BrushSettings` into the dispatch.
/// Bundled into a struct to keep `dispatch_inpaint` from sprawling.
#[derive(Clone, Debug)]
pub(crate) struct InpaintTuning {
    pub sharpen: f32,
    pub feather_px: f32,
    pub grow_px: f32,
    /// Which inpaint backend to use (LaMaFp32, BigLaMa, …). For
    /// SD-family backends, the choice between `SdV15InpaintFp16`
    /// (standard SD weights) and `SdV15LcmInpaintFp16` (LCM weights)
    /// is driven by the user's scheduler pick upstream
    /// (`Settings::lcm_routing_active`).
    pub backend: prunr_models::ModelId,
    /// SD-only: text prompt; ignored for LaMa-family backends.
    pub sd_prompt: String,
    pub sd_negative_prompt: String,
    pub sd_guidance_scale: f32,
    /// SD-only: scheduler kind. Carried through to the worker via
    /// `SdInpaintRequest::scheduler` so the right denoise math runs.
    pub sd_scheduler: super::brush_state::SdScheduler,
    /// SD-only: number of denoise steps.
    pub sd_steps: u32,
    /// SD-only: pinned RNG seed for reproducibility. `None` = random.
    pub sd_seed: Option<u64>,
    /// SD-only: inpaint strength in [0, 1]. 1.0 = pure noise init,
    /// fully creative rewrite. <1.0 preserves the source proportionally.
    pub sd_strength: f32,
    /// LCM-only: Karras sigma schedule. Default false (linear, matches
    /// distillation training).
    pub sd_use_karras_sigmas: bool,
    /// SD-only: Gaussian blur sigma applied to mask before VAE encoding.
    /// 0.0 = hard-cliff binary mask (original behavior).
    pub sd_mask_blur: f32,
}

impl Default for InpaintTuning {
    fn default() -> Self {
        Self {
            sharpen: 0.0,
            feather_px: 0.0,
            grow_px: 0.0,
            backend: prunr_models::ModelId::LaMaFp32,
            sd_prompt: String::new(),
            sd_negative_prompt: String::new(),
            sd_guidance_scale: 1.0,
            sd_scheduler: super::brush_state::SdScheduler::Lcm,
            sd_steps: 8,
            sd_seed: None,
            sd_strength: 1.0,
            sd_use_karras_sigmas: false,
            sd_mask_blur: 0.0,
        }
    }
}

pub(crate) struct Processor {
    pub(crate) worker_tx: mpsc::Sender<WorkerMessage>,
    pub(crate) worker_rx: mpsc::Receiver<WorkerResult>,
    /// Cancellation state shared with the worker bridge. `global` stops the
    /// whole batch; per-item entries drop individual items at the next
    /// dispatch check.
    pub(crate) cancels: CancelRegistry,
    pub(crate) live_preview: LivePreview,
    /// Active admission controller (present only during streaming batches).
    pub(crate) admission: Option<AdmissionController>,
    /// Sender for streaming additional items to the worker.
    pub(crate) admission_tx: Option<mpsc::Sender<WorkItem>>,
    /// In-flight batch: recipe + pending IDs. `None` between batches.
    in_flight: Option<InFlightBatch>,
    /// Last time periodic history cleanup ran.
    pub(crate) last_history_cleanup: Instant,
    /// Inpaint dispatch state. Per-item generation counter discards
    /// stale results when the user paints a fresh stroke before the
    /// previous one finishes. `inpaint_pending` is the count of
    /// dispatches not yet drained — the canvas reads it via
    /// `is_inpaint_in_flight` to show a progress overlay.
    inpaint_tx: mpsc::Sender<InpaintResult>,
    inpaint_rx: mpsc::Receiver<InpaintResult>,
    inpaint_latest_gen: HashMap<u64, u64>,
    inpaint_pending: HashMap<u64, u32>,
    /// Per-item cancel flag for the in-flight inpaint stroke. The flag
    /// is checked between LaMa tiles and between SD UNet steps; when
    /// set, the rayon job returns `CoreError::Cancelled` early and the
    /// drain path ignores the result. Cancel button + Esc key both
    /// flip the flag for the currently-selected item.
    inpaint_cancels: HashMap<u64, std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Per-item progress sink for the in-flight stroke. Worker writes
    /// `current` step between SD UNet iterations; the canvas banner
    /// reads `(current, total)` to show "Erasing — step N of M". Same
    /// lifetime as `inpaint_cancels` — both are replaced on every
    /// dispatch.
    inpaint_progress: HashMap<u64, std::sync::Arc<prunr_core::inpaint::InpaintProgress>>,
    /// Channels to the dedicated SD-inpaint subprocess bridge thread.
    /// LaMa / Big-LaMa / MI-GAN stay on the in-process rayon path; only
    /// SD-family dispatches go through these. Bridge spawns the
    /// subprocess lazily and drops it on a 5-min idle timer.
    inpaint_bridge_tx: mpsc::Sender<InpaintBridgeMsg>,
    inpaint_bridge_rx: mpsc::Receiver<InpaintBridgeResult>,
}

impl Processor {
    pub(crate) fn new(
        worker_tx: mpsc::Sender<WorkerMessage>,
        worker_rx: mpsc::Receiver<WorkerResult>,
    ) -> Self {
        let (inpaint_tx, inpaint_rx) = mpsc::channel();
        let (inpaint_bridge_tx, inpaint_bridge_rx) = spawn_inpaint_bridge();
        Self {
            worker_tx,
            worker_rx,
            cancels: CancelRegistry::new(),
            live_preview: LivePreview::default(),
            admission: None,
            admission_tx: None,
            in_flight: None,
            last_history_cleanup: Instant::now(),
            inpaint_tx,
            inpaint_rx,
            inpaint_latest_gen: HashMap::new(),
            inpaint_pending: HashMap::new(),
            inpaint_cancels: HashMap::new(),
            inpaint_progress: HashMap::new(),
            inpaint_bridge_tx,
            inpaint_bridge_rx,
        }
    }

    /// Per-item generation counter ensures a fresh stroke supersedes the
    /// previous in-flight job at drain time — see `drain_inpaint_results`.
    /// SD-family models route through the inpaint subprocess bridge for
    /// process isolation; LaMa / Big-LaMa / MI-GAN stay on the in-process
    /// rayon path (low RAM footprint, no isolation pressure).
    pub(crate) fn dispatch_inpaint(
        &mut self,
        item_id: u64,
        image: std::sync::Arc<image::RgbaImage>,
        correction: std::sync::Arc<prunr_core::brush::MaskCorrection>,
        tuning: InpaintTuning,
    ) {
        let generation = self.inpaint_latest_gen.entry(item_id).or_insert(0);
        *generation += 1;
        let gen = *generation;
        *self.inpaint_pending.entry(item_id).or_insert(0) += 1;
        // Replace any prior cancel flag + progress sink — the new
        // dispatch supersedes its predecessor anyway, so wiring fresh
        // ones avoids a stale earlier-stroke cancel firing the moment
        // a new stroke starts (and avoids the banner showing the prior
        // stroke's last step count for one frame).
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let progress = std::sync::Arc::new(prunr_core::inpaint::InpaintProgress::new());
        self.inpaint_cancels.insert(item_id, cancel.clone());
        self.inpaint_progress.insert(item_id, progress.clone());
        if tuning.backend.is_sd_family() {
            self.dispatch_inpaint_sd(item_id, gen, &image, &correction, &tuning);
            return;
        }
        let tx = self.inpaint_tx.clone();
        rayon::spawn(move || {
            let raw_mask = correction.to_binary_mask(image.width(), image.height());
            // Pre-process: grow/erode the painted area before LaMa runs.
            let mask = if tuning.grow_px != 0.0 {
                prunr_core::inpaint::grow_mask(&raw_mask, tuning.grow_px.round() as i32)
            } else {
                raw_mask
            };
            // This path only sees LaMa / Big-LaMa / MI-GAN — sd_req is unread.
            let sd_req = None;
            let hooks = prunr_core::inpaint::InpaintHooks {
                cancel: Some(cancel.clone()),
                progress: Some(progress.clone()),
            };
            match prunr_core::inpaint::process_inpaint_with(&image, &mask, tuning.backend, sd_req, &hooks) {
                Ok(rgba) => {
                    let out = prunr_core::inpaint_blend::finalize_inpaint(
                        &rgba, &image, &mask, tuning.feather_px, tuning.sharpen,
                    );
                    let _ = tx.send(InpaintResult { item_id, rgba: out, generation: gen, cancelled: false, error: None });
                }
                Err(prunr_core::CoreError::Cancelled) => {
                    tracing::info!(item_id, "inpaint cancelled by user");
                    // Marker with cancelled=true so the drain path can
                    // surface a toast. Gen 0 keeps it treated as stale.
                    let _ = tx.send(InpaintResult {
                        item_id,
                        rgba: image::RgbaImage::new(0, 0),
                        generation: 0,
                        cancelled: true,
                        error: None,
                    });
                }
                Err(e) => {
                    let msg = e.to_string();
                    tracing::error!(item_id, e = %msg, "inpaint dispatch failed");
                    // Surface the error to the GUI as a toast — without
                    // it the user just sees the brush stroke "do nothing"
                    // (e.g. when the SD-bundle RAM guard refuses to
                    // load). Gen 0 still routes the result through the
                    // stale-drop branch in drain_inpaint_results.
                    let _ = tx.send(InpaintResult {
                        item_id,
                        rgba: image::RgbaImage::new(0, 0),
                        generation: 0,
                        cancelled: false,
                        error: Some(msg),
                    });
                }
            }
        });
    }

    /// Cancel every in-flight inpaint stroke. Used by `handle_cancel`
    /// (Esc) so it doesn't have to iterate `batch.items` on the GUI
    /// side — coordinator pattern: PrunrApp delegates, Processor owns
    /// the in-flight set. Idempotent and cheap (HashMap walk).
    pub(crate) fn cancel_all_inpaints(&self) {
        for flag in self.inpaint_cancels.values() {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
        for &item_id in self.inpaint_cancels.keys() {
            let _ = self.inpaint_bridge_tx.send(InpaintBridgeMsg::Cancel { item_id });
        }
    }

    /// Cancel any in-flight inpaint stroke for `item_id`. The local
    /// flag drives the "Cancelling…" banner state immediately; for SD
    /// strokes the bridge also forwards `CancelItem` to the subprocess
    /// so its inference loop sees the flag too. Latency to actually-
    /// stopping is one tile (LaMa) or one UNet step (SD); ORT has no
    /// per-op cancel hook. Idempotent.
    pub(crate) fn cancel_inpaint(&self, item_id: u64) {
        if let Some(flag) = self.inpaint_cancels.get(&item_id) {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
        let _ = self.inpaint_bridge_tx.send(InpaintBridgeMsg::Cancel { item_id });
    }

    /// Tell the inpaint bridge to drop its cached subprocess. Idempotent;
    /// no-op when the bridge is already idle. Called by
    /// `apply_toolbar_change` on every model switch so a previous
    /// SD-family backend's ~200 MB residual subprocess (bundle was
    /// already released by `inpaint_sd::release` on dispatch
    /// completion, but the worker process itself stays alive for the
    /// 5-min idle window) doesn't pin RAM the user no longer wants
    /// allocated.
    pub(crate) fn release_inpaint_subprocess(&self) {
        let _ = self.inpaint_bridge_tx.send(InpaintBridgeMsg::Release);
    }

    /// Tell the seg/edge worker bridge to drop its warm subprocess.
    /// Idempotent; no-op when nothing is warm. Symmetric with
    /// `release_inpaint_subprocess` — both fire on model switch so a
    /// previous backend's engine pool (BiRefNetLite ~2 GB, U2Net
    /// ~800 MB, …) isn't held across the user's "I'm done with that
    /// model" signal.
    pub(crate) fn release_seg_warm(&self) {
        let _ = self.worker_tx.send(WorkerMessage::ReleaseWarm);
    }

    /// SD inpaint dispatch via the dedicated subprocess bridge. Encodes
    /// the (already-Arc'd) image + binary mask to PNG temp files,
    /// writes them, and sends an `InpaintBridgeMsg::Dispatch`. The
    /// bridge handles subprocess lifecycle. Bridge results stream back
    /// via `pump_inpaint_subprocess` (called once per frame).
    fn dispatch_inpaint_sd(
        &mut self,
        item_id: u64,
        gen: u64,
        image: &std::sync::Arc<image::RgbaImage>,
        correction: &std::sync::Arc<prunr_core::brush::MaskCorrection>,
        tuning: &InpaintTuning,
    ) {
        // PNG-encode + temp-file write is 150-300 ms per stroke at 4K
        // — moving it onto rayon keeps the egui frame loop responsive
        // mid-paint. Mask construction (`to_binary_mask`, `grow_mask`)
        // is also CPU work and rides along.
        let image = image.clone();
        let correction = correction.clone();
        let tuning = tuning.clone();
        let bridge_tx = self.inpaint_bridge_tx.clone();
        let inpaint_tx = self.inpaint_tx.clone();
        // Cancel flag the parent already inserted into `inpaint_cancels`
        // — the rayon job checks it before any work and before the
        // bridge dispatch so a mid-encode Esc doesn't end up running
        // the SD bundle on a stroke the user already discarded.
        let cancel = self.inpaint_cancels.get(&item_id).cloned();
        rayon::spawn(move || {
            let cancelled = || cancel.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::Acquire));
            let send_cancelled = || {
                let _ = inpaint_tx.send(InpaintResult {
                    item_id,
                    rgba: image::RgbaImage::new(0, 0),
                    generation: 0,
                    cancelled: true,
                    error: None,
                });
            };
            if cancelled() { send_cancelled(); return; }
            let raw_mask = correction.to_binary_mask(image.width(), image.height());
            let mask = if tuning.grow_px != 0.0 {
                prunr_core::inpaint::grow_mask(&raw_mask, tuning.grow_px.round() as i32)
            } else {
                raw_mask
            };
            let sd_req = match tuning.backend {
                prunr_models::ModelId::SdV15InpaintFp16
                | prunr_models::ModelId::SdV15LcmInpaintFp16 => {
                    use prunr_core::inpaint_sd::SchedulerKind;
                    let scheduler: SchedulerKind = tuning.sd_scheduler.into();
                    let lcm = matches!(scheduler, SchedulerKind::Lcm);
                    // LCM weights are calibrated for low CFG. The LCM
                    // model card recommends staying in 1.0–2.0; values
                    // above degrade output. Clamp so the toolbar slider's
                    // full 1.0–15.0 range doesn't pipe an out-of-range
                    // value into LCM.
                    let cfg = if lcm {
                        tuning.sd_guidance_scale.clamp(1.0, 2.0)
                    } else {
                        tuning.sd_guidance_scale
                    };
                    // TAESD is the distilled fast VAE substitute paired
                    // with LCM. Standard SD weights expect the full VAE.
                    let use_taesd = lcm
                        && prunr_models::is_available(prunr_models::ModelId::TaesdFp16);
                    Some(prunr_core::inpaint_sd::SdInpaintRequest {
                        prompt: tuning.sd_prompt.clone(),
                        negative_prompt: tuning.sd_negative_prompt.clone(),
                        num_inference_steps: tuning.sd_steps,
                        guidance_scale: cfg,
                        seed: tuning.sd_seed,
                        use_taesd,
                        scheduler,
                        strength: tuning.sd_strength,
                        use_karras_sigmas: tuning.sd_use_karras_sigmas,
                        mask_blur: tuning.sd_mask_blur,
                    })
                }
                _ => None,
            };
            let dir = crate::subprocess::protocol::ipc_temp_dir();
            let image_path = crate::subprocess::protocol::IpcKind::InpaintImg.path_for_gen(dir, item_id, gen);
            let mask_path  = crate::subprocess::protocol::IpcKind::InpaintMask.path_for_gen(dir, item_id, gen);
            let res: Result<(), String> = (|| {
                let img_bytes = prunr_core::encode_rgba_png(&image)
                    .map_err(|e| format!("encode source: {e:?}"))?;
                std::fs::write(&image_path, img_bytes)
                    .map_err(|e| format!("write source: {e}"))?;
                let mask_bytes = prunr_core::encode_gray_png(&mask)
                    .map_err(|e| format!("encode mask: {e:?}"))?;
                std::fs::write(&mask_path, mask_bytes)
                    .map_err(|e| format!("write mask: {e}"))
            })();
            if let Err(e) = res {
                let _ = inpaint_tx.send(InpaintResult {
                    item_id,
                    rgba: image::RgbaImage::new(0, 0),
                    generation: 0,
                    cancelled: false,
                    error: Some(e),
                });
                return;
            }
            // Esc landed during the encode — drop the work before the
            // bridge sees it. Temp files we just wrote are abandoned;
            // the next dispatch's gen-bump renames over them.
            if cancelled() {
                let _ = std::fs::remove_file(&image_path);
                let _ = std::fs::remove_file(&mask_path);
                send_cancelled();
                return;
            }
            let _ = bridge_tx.send(InpaintBridgeMsg::Dispatch {
                item_id, gen, model_id: tuning.backend, image_path, mask_path, sd_req,
                feather_px: tuning.feather_px,
                sharpen: tuning.sharpen,
            });
        });
    }

    /// Drain bridge events, forward into the existing inpaint result
    /// channel + progress sinks. Called once per frame from app pump.
    pub(crate) fn pump_inpaint_subprocess(&mut self) {
        while let Ok(evt) = self.inpaint_bridge_rx.try_recv() {
            match evt {
                InpaintBridgeResult::Progress { item_id, current, total } => {
                    if let Some(p) = self.inpaint_progress.get(&item_id) {
                        p.set_total(total);
                        p.set_step(current);
                    }
                }
                InpaintBridgeResult::Done { item_id, gen, rgba_path, width, height } => {
                    let result = match super::worker::read_and_delete(&rgba_path) {
                        Some(b) => match image::load_from_memory(&b) {
                            Ok(img) => {
                                let rgba = img.to_rgba8();
                                debug_assert_eq!((rgba.width(), rgba.height()), (width, height));
                                // Stamp with the dispatch's own gen — drain
                                // drops it as stale if a fresher stroke
                                // bumped `inpaint_latest_gen` while this
                                // one was in the subprocess.
                                InpaintResult {
                                    item_id, rgba, generation: gen,
                                    cancelled: false, error: None,
                                }
                            }
                            Err(e) => InpaintResult {
                                item_id,
                                rgba: image::RgbaImage::new(0, 0),
                                generation: 0, cancelled: false,
                                error: Some(format!("decode SD result: {e}")),
                            },
                        },
                        None => InpaintResult {
                            item_id,
                            rgba: image::RgbaImage::new(0, 0),
                            generation: 0, cancelled: false,
                            error: Some(format!("read SD result missing: {}", rgba_path.display())),
                        },
                    };
                    let _ = self.inpaint_tx.send(result);
                }
                InpaintBridgeResult::Error { item_id, error } => {
                    // Translate bridge sentinels to user-facing text at
                    // this seam so `drain_inpaint_results` doesn't have
                    // to know about IPC strings.
                    use crate::subprocess::protocol::{CANCELLED_ERR_MSG, MEMORY_PRESSURE_ABORT_MSG};
                    let cancelled = error == CANCELLED_ERR_MSG;
                    let user_error = if cancelled {
                        None
                    } else if error == MEMORY_PRESSURE_ABORT_MSG {
                        Some(
                            "Erase aborted — system memory low. \
                             Close other apps or use LaMa instead \
                             (Settings → Eraser)."
                                .to_string(),
                        )
                    } else {
                        Some(error)
                    };
                    let _ = self.inpaint_tx.send(InpaintResult {
                        item_id,
                        rgba: image::RgbaImage::new(0, 0),
                        generation: 0,
                        cancelled,
                        error: user_error,
                    });
                }
            }
        }
    }

    /// Read the in-flight inpaint stroke's progress for `item_id` as
    /// `(current_step, total_steps)`. Returns `(0, 0)` when no stroke
    /// is in flight or the worker hasn't started stepping yet (LaMa
    /// stays here for the whole stroke; only SD's UNet loop publishes
    /// step counts).
    pub(crate) fn inpaint_progress(&self, item_id: u64) -> (u32, u32) {
        self.inpaint_progress.get(&item_id)
            .map(|p| p.read())
            .unwrap_or((0, 0))
    }

    /// Drain in-flight inpaint results.
    ///
    /// Returns `(committed_results, cancelled_item_ids, errors)`.
    /// - `committed_results` only carries the latest-gen finished
    ///   strokes (stale ones are silently dropped).
    /// - `cancelled_item_ids` lists items whose stroke was cancelled
    ///   by the user — the GUI surfaces a toast for each so the user
    ///   gets explicit feedback that Esc/Cancel took effect.
    /// - `errors` carries the user-visible message for any non-Cancelled
    ///   dispatch failure (e.g. SD RAM-guard refusal). The GUI shows
    ///   each as an error toast — without this surface the user sees
    ///   the stroke "do nothing" with no idea why.
    pub(crate) fn drain_inpaint_results(&mut self) -> (Vec<InpaintResult>, Vec<u64>, Vec<String>) {
        let mut out = Vec::new();
        let mut cancelled = Vec::new();
        let mut errors = Vec::new();
        while let Ok(result) = self.inpaint_rx.try_recv() {
            let item_id = result.item_id;
            // Every drained result decrements pending — stale ones
            // count too, since the rayon job that produced them has
            // run to completion.
            if let Some(c) = self.inpaint_pending.get_mut(&item_id) {
                *c = c.saturating_sub(1);
            }
            // Reclaim the per-item cancel/progress entries once nothing
            // else is in flight for this id. Without this they accumulate
            // for the life of the session — `cancel_all_inpaints` walks
            // them all, and Esc-after-50-strokes ends up firing 50
            // no-op IPC cancels.
            if self.inpaint_pending.get(&item_id).copied().unwrap_or(0) == 0 {
                self.inpaint_cancels.remove(&item_id);
                self.inpaint_progress.remove(&item_id);
            }
            if result.cancelled {
                cancelled.push(item_id);
                continue;
            }
            if let Some(msg) = result.error {
                errors.push(msg);
                continue;
            }
            let latest = self.inpaint_latest_gen.get(&item_id).copied().unwrap_or(0);
            if result.generation == latest && result.generation > 0 {
                out.push(result);
            }
        }
        (out, cancelled, errors)
    }

    /// True while a dispatched inpaint job hasn't drained yet for `item_id`.
    /// Canvas reads this to render a "Erasing..." overlay during LaMa work.
    pub(crate) fn is_inpaint_in_flight(&self, item_id: u64) -> bool {
        self.inpaint_pending.get(&item_id).copied().unwrap_or(0) > 0
    }

    /// True while ANY item has an in-flight inpaint job. Status bar
    /// reads this to override the "All done" text during LaMa work.
    pub(crate) fn any_inpaint_in_flight(&self) -> bool {
        self.inpaint_pending.values().any(|&c| c > 0)
    }

    /// True after Cancel/Esc clicked but before the worker's atomic
    /// Acquire load observes the flag. Drives the "Cancelling…" banner
    /// state — without this signal the click looks unacknowledged
    /// during the multi-second latency to the next worker checkpoint.
    pub(crate) fn is_inpaint_cancelling(&self, item_id: u64) -> bool {
        self.inpaint_cancels.get(&item_id)
            .is_some_and(|f| f.load(std::sync::atomic::Ordering::Acquire))
    }

    /// Register a batch's recipe + the IDs that should deliver against it.
    /// Replaces any prior in-flight state — callers ensure prior batches
    /// have completed before firing a new dispatch.
    pub(crate) fn track_dispatch(
        &mut self,
        recipe: ProcessingRecipe,
        ids: impl IntoIterator<Item = u64>,
    ) {
        self.in_flight = Some(InFlightBatch {
            recipe,
            pending: ids.into_iter().collect(),
        });
    }

    /// Add a streamed (admission-pool) item to the current batch. The
    /// `debug_assert` catches the "admission ran without a tracked batch"
    /// invariant breach in tests; release builds silently no-op so a
    /// single late delivery can't take down a real batch.
    pub(crate) fn track_streamed(&mut self, id: u64) {
        match self.in_flight.as_mut() {
            Some(b) => { b.pending.insert(id); }
            None => debug_assert!(false, "track_streamed called without active batch"),
        }
    }

    /// Take the recipe for a finished item. Returns `None` when the item
    /// wasn't in flight (late delivery after cancel/drain) — caller falls
    /// back. Self-cleans the in-flight slot when the last item completes.
    pub(crate) fn take_recipe(&mut self, id: u64) -> Option<ProcessingRecipe> {
        let batch = self.in_flight.as_mut()?;
        if !batch.pending.remove(&id) {
            return None;
        }
        let recipe = batch.recipe.clone();
        if batch.pending.is_empty() {
            self.in_flight = None;
        }
        Some(recipe)
    }

    /// Drop the in-flight slot regardless of pending. Called on user cancel
    /// or batch-complete signals so a late delivery can't reattribute.
    pub(crate) fn drain_recipes(&mut self) {
        self.in_flight = None;
    }

    /// Drop admission state so no further items are admitted. Called on
    /// cancel (user or worker-side) and by the cancelled-message handler.
    /// Leaves the cancel registry untouched — that's owned by the caller's
    /// cancel protocol.
    pub(crate) fn clear_admission(&mut self) {
        self.admission = None;
        self.admission_tx = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Processor {
        let (tx, _rx_unused) = mpsc::channel::<WorkerMessage>();
        let (_tx_unused, rx) = mpsc::channel::<WorkerResult>();
        Processor::new(tx, rx)
    }

    #[test]
    fn new_initialises_last_history_cleanup_recent() {
        // Verifies the periodic 600s cleanup gate isn't accidentally
        // triggered at startup — the Instant must be effectively-now.
        let p = fixture();
        assert!(p.last_history_cleanup.elapsed().as_secs() < 5);
    }

    #[test]
    fn drain_filters_stale_generations() {
        let mut p = fixture();
        p.inpaint_latest_gen.insert(7, 2);
        p.inpaint_tx.send(InpaintResult {
            item_id: 7, rgba: image::RgbaImage::new(1, 1), generation: 1, cancelled: false, error: None,
        }).unwrap();
        p.inpaint_tx.send(InpaintResult {
            item_id: 7, rgba: image::RgbaImage::new(1, 1), generation: 2, cancelled: false, error: None,
        }).unwrap();
        let (drained, cancelled, errors) = p.drain_inpaint_results();
        assert_eq!(drained.len(), 1, "stale gen=1 must be dropped");
        assert_eq!(drained[0].generation, 2);
        assert!(cancelled.is_empty(), "no cancellation events expected");
        assert!(errors.is_empty(), "no dispatch errors expected");
    }

    #[test]
    fn drain_routes_cancelled_results_to_cancellation_list() {
        let mut p = fixture();
        p.inpaint_latest_gen.insert(7, 5);
        p.inpaint_pending.insert(7, 1);
        p.inpaint_tx.send(InpaintResult {
            item_id: 7, rgba: image::RgbaImage::new(0, 0),
            generation: 0, cancelled: true, error: None,
        }).unwrap();
        let (drained, cancelled, errors) = p.drain_inpaint_results();
        assert!(drained.is_empty(),
            "cancelled stroke must not commit a result");
        assert_eq!(cancelled, vec![7],
            "cancellation event must surface for the toast");
        assert!(errors.is_empty(),
            "cancel must not be reported as a dispatch error");
    }

    #[test]
    fn drain_routes_dispatch_errors_to_error_list() {
        let mut p = fixture();
        p.inpaint_pending.insert(7, 1);
        p.inpaint_tx.send(InpaintResult {
            item_id: 7, rgba: image::RgbaImage::new(0, 0),
            generation: 0, cancelled: false,
            error: Some("SD inpaint refused to load: only 13.4 GB free".to_string()),
        }).unwrap();
        let (drained, cancelled, errors) = p.drain_inpaint_results();
        assert!(drained.is_empty(),
            "errored stroke must not commit a result");
        assert!(cancelled.is_empty(),
            "errored stroke is not a cancel");
        assert_eq!(errors.len(), 1,
            "dispatch error must surface for the user toast");
        assert!(errors[0].contains("13.4 GB"));
    }

    #[test]
    fn clear_admission_drops_both_sides_but_leaves_cancels() {
        let mut p = fixture();
        let (tx, _rx) = mpsc::channel::<WorkItem>();
        p.admission_tx = Some(tx);
        p.cancels.request_global_cancel();
        assert!(p.admission_tx.is_some());

        p.clear_admission();

        assert!(p.admission.is_none());
        assert!(p.admission_tx.is_none());
        assert!(p.cancels.is_cancelled(999),
            "clear_admission must leave cancel registry untouched — that's the caller's protocol");
    }

    #[test]
    fn cancel_registry_clone_shares_state() {
        // Cloned into WorkerMessage::BatchProcess and read by the bridge —
        // a store on the parent must be visible via any clone.
        let r = CancelRegistry::new();
        let handle = r.clone();
        assert!(!handle.is_cancelled(5));
        r.request_global_cancel();
        assert!(handle.is_cancelled(5), "Clone must observe the global store");
    }

    #[test]
    fn cancel_registry_per_item_is_independent_of_global() {
        let r = CancelRegistry::new();
        r.request_item_cancel(42);
        assert!(r.is_cancelled(42));
        assert!(!r.is_cancelled(7), "per-item cancel must not leak to other ids");
    }

    #[test]
    fn cancel_registry_reset_clears_all_flags() {
        let r = CancelRegistry::new();
        r.request_global_cancel();
        r.request_item_cancel(42);
        r.reset();
        assert!(!r.is_cancelled(42));
        assert!(!r.is_cancelled(99));
    }

    #[test]
    fn global_cancel_short_circuits_per_item_lookup() {
        let r = CancelRegistry::new();
        r.request_global_cancel();
        // Any id reports cancelled when global is set, even ones with no per-item entry.
        assert!(r.is_cancelled(u64::MAX));
    }

    fn fixture_recipe() -> ProcessingRecipe {
        use prunr_core::{
            CompositeRecipe, EdgeRecipe, EdgeScale, ComposeMode, FillStyle, InferenceRecipe,
            InputTransform, LineStyle, MaskSettings, ModelKind,
        };
        ProcessingRecipe {
            inference: InferenceRecipe {
                model: ModelKind::Silueta,
                uses_segmentation: true,
                uses_edge_detection: false,
                input_transform: InputTransform::None,
            },
            edge: EdgeRecipe {
                line_strength_bits: 0.5f32.to_bits(),
                solid_line_color: None,
                edge_thickness: 0,
                edge_scale: EdgeScale::Fused,
                compose_mode: ComposeMode::LinesOnly,
                line_style: LineStyle::Solid,
            },
            mask: (&MaskSettings { fill_style: FillStyle::None, ..Default::default() }).into(),
            composite: CompositeRecipe::default(),
            was_chain: false,
        }
    }

    #[test]
    fn track_dispatch_then_take_returns_recipe_per_item() {
        let mut p = fixture();
        p.track_dispatch(fixture_recipe(), [10, 20, 30].iter().copied());
        assert!(p.take_recipe(10).is_some());
        assert!(p.take_recipe(20).is_some());
        // Slot still alive while items remain.
        assert!(p.in_flight.is_some());
        assert!(p.take_recipe(30).is_some());
        // Last item drains the slot.
        assert!(p.in_flight.is_none());
    }

    #[test]
    fn take_recipe_unknown_id_is_none() {
        let mut p = fixture();
        p.track_dispatch(fixture_recipe(), [1].iter().copied());
        assert!(p.take_recipe(999).is_none(), "unknown id must not return a recipe");
        // Tracked id still works.
        assert!(p.take_recipe(1).is_some());
    }

    #[test]
    fn track_streamed_inherits_batch_recipe() {
        // Admission-pool items are added after dispatch; they inherit the
        // current batch's recipe so a late ImageDone for a streamed id
        // still attributes correctly.
        let mut p = fixture();
        p.track_dispatch(fixture_recipe(), [1].iter().copied());
        p.track_streamed(2);
        assert!(p.take_recipe(1).is_some());
        assert!(p.take_recipe(2).is_some(), "streamed item must have a recipe");
    }

    #[test]
    #[should_panic(expected = "track_streamed called without active batch")]
    fn track_streamed_without_dispatch_panics_in_debug() {
        let mut p = fixture();
        p.track_streamed(99);
    }

    #[test]
    fn drain_recipes_clears_pending() {
        let mut p = fixture();
        p.track_dispatch(fixture_recipe(), [1, 2, 3].iter().copied());
        p.drain_recipes();
        assert!(p.take_recipe(1).is_none(),
            "drain must drop the slot so late deliveries fall back");
    }

    // ── Brush gate boundary contract (M-GUI-9) ───────────────────────────────
    //
    // `canvas::render` computes:
    //   brush_active = is_enabled && !inpaint_in_flight_for_selected && …
    //
    // If is_inpaint_in_flight returns true for the selected item, brush_active
    // must be false — so handle_brush_input is never entered. This test pins
    // the Processor half of that contract so a future decoupling of
    // is_inpaint_in_flight / is_inpaint_cancelling doesn't silently break the gate.

    #[test]
    fn inpaint_in_flight_blocks_brush_gate_for_selected_item() {
        let mut p = fixture();
        // Simulate an in-flight inpaint for item 42 by bumping the pending counter.
        *p.inpaint_pending.entry(42).or_insert(0) += 1;
        // The gate must block: is_inpaint_in_flight returns true.
        assert!(p.is_inpaint_in_flight(42),
            "pending count > 0 must report in-flight for the canvas brush gate");
        // A different item must not be affected.
        assert!(!p.is_inpaint_in_flight(99),
            "in-flight state must be item-scoped, not global");
    }

    #[test]
    fn inpaint_cancelling_is_independent_of_in_flight_count() {
        // Cancelling starts before the worker has seen the flag; is_inpaint_in_flight
        // is still true during this window. A future refactor that decouples them
        // must not remove the separate is_inpaint_cancelling check.
        let mut p = fixture();
        *p.inpaint_pending.entry(7).or_insert(0) += 1;
        p.inpaint_cancels.insert(7, Arc::new(AtomicBool::new(true)));
        assert!(p.is_inpaint_in_flight(7), "in-flight must still be true during cancel");
        assert!(p.is_inpaint_cancelling(7), "cancelling must reflect the atomic flag");
    }

    #[test]
    fn zero_cancel_is_cancelled_does_not_touch_mutex() {
        // `is_cancelled` is called ~160×/s from the bridge loop. Until any
        // per-item entry is requested the mutex must stay cold — poisoning
        // the map from another thread and then calling `is_cancelled` on a
        // fresh registry must not panic.
        let r = CancelRegistry::new();
        // Poison the inner mutex from a panicking thread.
        let p = r.per_item.clone();
        let _ = std::thread::spawn(move || {
            let _guard = p.lock().unwrap();
            panic!("deliberate poison");
        }).join();
        // has_per_item is still false → no lock taken → no panic propagation.
        assert!(!r.is_cancelled(42));
    }
}

