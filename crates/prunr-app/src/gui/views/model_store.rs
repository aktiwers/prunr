//! Model Store modal — single user-facing surface for browsing,
//! installing, cancelling, and deleting on-demand models.

use egui::RichText;

use prunr_models::{
    descriptor as model_descriptor, on_demand_dir, ModelCategory, ModelDescriptor, ModelId,
    ModelSource, REGISTRY,
};

use crate::gui::app::PrunrApp;
use crate::gui::download_manager::DownloadState;
use crate::gui::theme;
use super::format_byte_size;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CardAction {
    /// Bundled model — no action available, button reads "Built-in".
    Bundled,
    Download,
    Cancel,
    Verifying,
    Delete,
    /// `retryable=false` renders disabled (e.g. SHA-mismatch is fatal).
    Retry { retryable: bool },
}

pub(crate) fn card_action(
    desc: &prunr_models::ModelDescriptor,
    is_installed: bool,
    download_state: &DownloadState,
) -> CardAction {
    if matches!(desc.source, ModelSource::Bundled) {
        return CardAction::Bundled;
    }
    // Exhaustive on purpose — adding a new `DownloadState` variant
    // forces this match to grow with it instead of silently falling
    // through to `Download` and re-queueing a "paused" download.
    match download_state {
        DownloadState::InProgress { .. } | DownloadState::Queued => CardAction::Cancel,
        DownloadState::Verifying => CardAction::Verifying,
        DownloadState::Failed { retryable, .. } if !is_installed => {
            CardAction::Retry { retryable: *retryable }
        }
        DownloadState::Failed { .. } | DownloadState::Done | DownloadState::Idle => {
            if is_installed { CardAction::Delete } else { CardAction::Download }
        }
    }
}

fn disk_usage_bytes_uncached() -> u64 {
    let Some(dir) = on_demand_dir() else { return 0 };
    REGISTRY.iter().filter_map(|d| match d.source {
        ModelSource::OnDemand { filename, .. } => {
            std::fs::metadata(dir.join(filename)).ok().map(|m| m.len())
        }
        _ => None,
    }).sum()
}

/// 1 s TTL cache around the per-OnDemand-model `fs::metadata` walk —
/// the modal repaints every frame while the user drags a scrollbar
/// or hovers a row, which fired ~7 stat syscalls per frame (~420
/// stat/sec on cold-cache disk). Same TTL pattern as
/// `hardware::available_ram_bytes_throttled`.
fn disk_usage_bytes() -> u64 {
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;
    static CACHE: OnceLock<Mutex<Option<(Instant, u64)>>> = OnceLock::new();
    let cell = CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let now = Instant::now();
    if let Some((at, bytes)) = *guard {
        if now.duration_since(at).as_secs_f32() < 1.0 {
            return bytes;
        }
    }
    let bytes = disk_usage_bytes_uncached();
    *guard = Some((now, bytes));
    bytes
}

/// Returns true when the user closed the modal this frame.
pub fn render(ctx: &egui::Context, app: &mut PrunrApp) -> bool {
    let mut close_requested = false;
    let provider = app.settings.active_backend.clone();

    let initial_filter = app.model_store.as_ref().and_then(|r| r.filter);
    let mut new_filter = initial_filter;
    let mut pending_actions: Vec<(ModelId, CardAction)> = Vec::with_capacity(REGISTRY.len());

    let backdrop_closed = theme::standard_modal_window(
        ctx, "model_store", "Model Store",
        [theme::SETTINGS_DIALOG_WIDTH, 560.0],
        |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Model Store")
                        .size(theme::FONT_SIZE_HEADING)
                        .strong()
                        .color(theme::TEXT_PRIMARY),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Close").clicked() {
                        close_requested = true;
                    }
                });
            });
            ui.add_space(theme::SPACE_SM);

            ui.horizontal(|ui| {
                filter_chip(ui, "All", None, &mut new_filter);
                filter_chip(ui, "Background", Some(ModelCategory::Segmentation), &mut new_filter);
                filter_chip(ui, "Lines", Some(ModelCategory::EdgeDetection), &mut new_filter);
                filter_chip(ui, "Eraser", Some(ModelCategory::Inpaint), &mut new_filter);
            });
            ui.add_space(theme::SPACE_SM);
            ui.separator();

            egui::ScrollArea::vertical()
                .max_height(420.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let mut shown = 0;
                    for desc in REGISTRY {
                        if let Some(c) = new_filter {
                            if desc.category != c { continue; }
                        }
                        if shown > 0 {
                            ui.add_space(theme::SPACE_XS);
                            ui.separator();
                        }
                        shown += 1;
                        ui.add_space(theme::SPACE_SM);
                        let state = app.download_manager.state(desc.id);
                        if let Some(action) = render_card(ui, desc, &state, &provider) {
                            pending_actions.push((desc.id, action));
                        }
                    }
                });

            ui.add_space(theme::SPACE_SM);
            ui.separator();
            ui.horizontal(|ui| {
                let total = disk_usage_bytes();
                ui.label(
                    RichText::new(format!("Disk usage: {}", format_byte_size(total)))
                        .size(theme::FONT_SIZE_MONO)
                        .color(theme::TEXT_SECONDARY),
                );
            });
        },
    );

    for (id, action) in pending_actions {
        match action {
            CardAction::Download | CardAction::Retry { retryable: true } => {
                if requires_license_gate(id, &app.settings) {
                    app.pending_license_request = Some(id);
                } else {
                    app.download_manager.start_download(id);
                }
            }
            CardAction::Cancel => {
                app.download_manager.cancel_download(id);
            }
            CardAction::Delete => {
                delete_installed_model(app, id);
            }
            CardAction::Bundled
            | CardAction::Verifying
            | CardAction::Retry { retryable: false } => {}
        }
    }

    if new_filter != initial_filter {
        if let Some(req) = app.model_store.as_mut() {
            req.filter = new_filter;
        }
    }

    backdrop_closed || close_requested
}

fn filter_chip(
    ui: &mut egui::Ui,
    label: &str,
    category: Option<ModelCategory>,
    current: &mut Option<ModelCategory>,
) {
    let selected = *current == category;
    let text = RichText::new(label)
        .size(theme::FONT_SIZE_BODY)
        .color(if selected { theme::TEXT_PRIMARY } else { theme::TEXT_SECONDARY });
    let btn = egui::Button::new(text)
        .fill(if selected { theme::BG_SECONDARY } else { egui::Color32::TRANSPARENT })
        .corner_radius(theme::BUTTON_ROUNDING)
        .min_size(egui::vec2(0.0, theme::CHIP_HEIGHT));
    if ui.add(btn).clicked() {
        *current = category;
    }
}

fn render_card(
    ui: &mut egui::Ui,
    desc: &ModelDescriptor,
    state: &DownloadState,
    provider: &str,
) -> Option<CardAction> {
    let installed = prunr_models::is_available(desc.id);
    let action = card_action(desc, installed, state);
    let advisory = desc.hardware_advisory(provider);

    let mut clicked = None;

    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.label(
                RichText::new(desc.display_name)
                    .size(theme::FONT_SIZE_HEADING)
                    .strong()
                    .color(theme::TEXT_PRIMARY),
            );
            ui.label(
                RichText::new(desc.description)
                    .size(theme::FONT_SIZE_BODY)
                    .color(theme::TEXT_PRIMARY),
            );
            let meta = match desc.source {
                ModelSource::Bundled => "Built-in".to_string(),
                ModelSource::OnDemand { license, size_mb, .. } => {
                    format!("{} · {} · {size_mb} MB", license.license, license.source_url)
                }
                ModelSource::MultiPartOnDemand { license, parts, .. } => {
                    let total_mb = parts.iter().map(|p| p.size_bytes).sum::<u64>() / (1024 * 1024);
                    format!("{} · {} · {total_mb} MB", license.license, license.source_url)
                }
            };
            ui.label(
                RichText::new(meta)
                    .size(theme::FONT_SIZE_MONO)
                    .color(theme::TEXT_SECONDARY),
            );
            if let DownloadState::InProgress { bytes_so_far, total_bytes } = *state {
                let pct = if total_bytes > 0 {
                    (bytes_so_far as f32 / total_bytes as f32 * 100.0).round() as u32
                } else { 0 };
                ui.label(
                    RichText::new(format!(
                        "Downloading: {} / {} ({pct}%)",
                        format_byte_size(bytes_so_far), format_byte_size(total_bytes),
                    ))
                    .size(theme::FONT_SIZE_MONO)
                    .color(theme::TEXT_SECONDARY),
                );
            } else if let DownloadState::Failed { error, .. } = state {
                ui.label(
                    RichText::new(format!("Last attempt failed: {error}"))
                        .size(theme::FONT_SIZE_MONO)
                        .color(theme::TEXT_HINT),
                );
            }
            if let Some(msg) = advisory {
                ui.label(
                    RichText::new(format!("⚠ {msg}"))
                        .size(theme::FONT_SIZE_MONO)
                        .color(theme::TEXT_HINT),
                );
            }
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if let Some(label) = button_label(action) {
                let mut resp = ui.add_enabled(action_enabled(action), egui::Button::new(label));
                if let Some(tip) = advisory {
                    resp = resp.on_disabled_hover_text(tip);
                }
                if resp.clicked() {
                    clicked = Some(action);
                }
            }
        });
    });

    clicked
}

fn button_label(action: CardAction) -> Option<&'static str> {
    match action {
        CardAction::Bundled => Some("Built-in"),
        CardAction::Download => Some("Download"),
        CardAction::Cancel => Some("Cancel"),
        CardAction::Verifying => Some("Verifying…"),
        CardAction::Delete => Some("Delete"),
        CardAction::Retry { .. } => Some("Retry"),
    }
}

fn action_enabled(action: CardAction) -> bool {
    !matches!(action,
        CardAction::Bundled
        | CardAction::Verifying
        | CardAction::Retry { retryable: false }
    )
}

fn delete_installed_model(app: &mut PrunrApp, id: ModelId) {
    let Some(desc) = model_descriptor(id) else { return };
    let Some(dir) = on_demand_dir() else {
        app.toasts.error("Could not resolve user data directory");
        return;
    };
    let result = match desc.source {
        ModelSource::OnDemand { filename, .. } => std::fs::remove_file(dir.join(filename)),
        ModelSource::MultiPartOnDemand { subdir, .. } => std::fs::remove_dir_all(dir.join(subdir)),
        ModelSource::Bundled => return,
    };
    match result {
        Ok(()) => {
            prunr_models::evict_on_demand_cache(id);
            // Compiled-model cache (CUDA optimized.onnx, CoreML mlmodelc) is
            // useless without the source weights. Off-thread because a
            // multi-GB rmtree on a slow disk would stall the GUI frame.
            std::thread::spawn(move || {
                let bytes = prunr_core::cache::clear_for_model(id);
                if bytes > 0 {
                    tracing::info!(?id, bytes, "cleared compiled-model cache after uninstall");
                }
            });
            app.toasts.success(format!("{} removed", desc.display_name));
            tracing::info!(?id, "deleted on-demand model");
        }
        Err(e) => {
            app.toasts.error(format!("Could not delete {}: {e}", desc.display_name));
        }
    }
}

/// True when the descriptor for `id` declares a restrictive license that
/// the user hasn't yet accepted in `Settings::accepted_licenses`.
pub(crate) fn requires_license_gate(id: ModelId, settings: &super::super::settings::Settings) -> bool {
    let Some(desc) = model_descriptor(id) else { return false };
    desc.requires_license_acceptance() && !settings.has_accepted_license(id)
}

/// License-acceptance modal. Returns `(close_requested, accepted)`:
/// - `close_requested`: the modal should be dismissed (Cancel or backdrop)
/// - `accepted`: the user clicked Accept — caller persists + starts download
pub fn render_license_dialog(
    ctx: &egui::Context,
    id: ModelId,
) -> (bool, bool) {
    let Some(desc) = model_descriptor(id) else { return (true, false) };
    let license = match desc.source {
        ModelSource::MultiPartOnDemand { license, .. } => license,
        ModelSource::OnDemand { license, .. } => license,
        // Bundled never requires acceptance — defensive, shouldn't reach here.
        ModelSource::Bundled => return (true, false),
    };

    let mut close_requested = false;
    let mut accepted = false;

    let backdrop_closed = theme::standard_modal_window(
        ctx, "license_dialog", "License acceptance",
        [theme::SETTINGS_DIALOG_WIDTH, 360.0],
        |ui| {
            ui.label(
                RichText::new("License acceptance")
                    .size(theme::FONT_SIZE_HEADING)
                    .strong()
                    .color(theme::TEXT_PRIMARY),
            );
            ui.add_space(theme::SPACE_SM);
            ui.label(
                RichText::new(desc.display_name)
                    .size(theme::FONT_SIZE_BODY)
                    .strong()
                    .color(theme::TEXT_PRIMARY),
            );
            ui.label(
                RichText::new(format!("License: {}", license.license))
                    .size(theme::FONT_SIZE_BODY)
                    .color(theme::TEXT_PRIMARY),
            );
            ui.add_space(theme::SPACE_XS);
            ui.label(
                RichText::new("This model is distributed under terms that you must \
                     review and accept before downloading. The full text \
                     is published at the URL below.".to_string())
                .size(theme::FONT_SIZE_BODY)
                .color(theme::TEXT_SECONDARY),
            );
            ui.add_space(theme::SPACE_XS);
            ui.hyperlink_to(license.license_url, license.license_url);
            ui.hyperlink_to(license.source_url, license.source_url);
            ui.add_space(theme::SPACE_MD);

            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    close_requested = true;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Accept and download").clicked() {
                        accepted = true;
                    }
                });
            });
        },
    );

    (backdrop_closed || close_requested, accepted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dl_idle() -> DownloadState { DownloadState::Idle }
    fn dl_in_progress() -> DownloadState { DownloadState::InProgress { bytes_so_far: 1, total_bytes: 100 } }
    fn dl_verifying() -> DownloadState { DownloadState::Verifying }
    fn dl_done() -> DownloadState { DownloadState::Done }
    fn dl_failed_retryable() -> DownloadState {
        DownloadState::Failed { error: "x".into(), retryable: true }
    }
    fn dl_failed_fatal() -> DownloadState {
        DownloadState::Failed { error: "x".into(), retryable: false }
    }
    fn synthetic(gpu: prunr_models::GpuRequirement) -> prunr_models::ModelDescriptor {
        prunr_models::ModelDescriptor {
            id: prunr_models::ModelId::Migan, // any id; tests don't dispatch
            display_name: "test",
            description: "test",
            category: prunr_models::ModelCategory::Inpaint,
            source: ModelSource::OnDemand {
                filename: "x.onnx",
                url: "https://example.test/x",
                sha256: "0".repeat(64).leak(),
                size_mb: 100,
                license: prunr_models::LicenseInfo {
                    license: "Apache-2.0",
                    license_url: "https://example.test/lic",
                    source_url: "https://example.test/src",
                },
            },
            version: "1.0.0",
            gpu,
            incompatible_eps: &[],
            working_set_mb: 100,
        }
    }

    fn synthetic_bundled() -> prunr_models::ModelDescriptor {
        prunr_models::ModelDescriptor {
            id: prunr_models::ModelId::Silueta,
            display_name: "test",
            description: "test",
            category: prunr_models::ModelCategory::Segmentation,
            source: ModelSource::Bundled,
            version: "1.0.0",
            gpu: prunr_models::GpuRequirement::None,
            incompatible_eps: &[],
            working_set_mb: 100,
        }
    }

    #[test]
    fn bundled_always_renders_as_none() {
        let d = synthetic_bundled();
        assert_eq!(card_action(&d, true, &dl_idle()), CardAction::Bundled);
        assert_eq!(card_action(&d, false, &dl_failed_retryable()), CardAction::Bundled);
    }

    #[test]
    fn ondemand_idle_uninstalled_is_download() {
        let d = synthetic(prunr_models::GpuRequirement::None);
        assert_eq!(card_action(&d, false, &dl_idle()), CardAction::Download);
    }

    #[test]
    fn ondemand_done_installed_is_delete() {
        let d = synthetic(prunr_models::GpuRequirement::None);
        assert_eq!(card_action(&d, true, &dl_done()), CardAction::Delete);
    }

    #[test]
    fn ondemand_in_progress_is_cancel() {
        let d = synthetic(prunr_models::GpuRequirement::None);
        assert_eq!(card_action(&d, false, &dl_in_progress()), CardAction::Cancel);
    }

    #[test]
    fn ondemand_verifying_is_verifying_disabled() {
        let d = synthetic(prunr_models::GpuRequirement::None);
        let a = card_action(&d, false, &dl_verifying());
        assert_eq!(a, CardAction::Verifying);
        assert!(!action_enabled(a));
    }

    #[test]
    fn ondemand_failed_retryable_uninstalled_is_retry_clickable() {
        let d = synthetic(prunr_models::GpuRequirement::None);
        let a = card_action(&d, false, &dl_failed_retryable());
        assert_eq!(a, CardAction::Retry { retryable: true });
        assert!(action_enabled(a));
    }

    #[test]
    fn ondemand_failed_fatal_uninstalled_is_retry_disabled() {
        let d = synthetic(prunr_models::GpuRequirement::None);
        let a = card_action(&d, false, &dl_failed_fatal());
        assert_eq!(a, CardAction::Retry { retryable: false });
        assert!(!action_enabled(a));
    }

    #[test]
    fn ondemand_failed_but_already_installed_shows_delete() {
        let d = synthetic(prunr_models::GpuRequirement::None);
        assert_eq!(card_action(&d, true, &dl_failed_retryable()), CardAction::Delete);
    }

    // ── Hardware advisory ─────────────────────────────────────────────
    // `card_action` no longer reads `gpu` — the dispatch is gated only by
    // source/install/download_state. Tests here pin the *advisory* surface.

    #[test]
    fn hardware_advisory_required_warns_on_cpu() {
        let d = synthetic(prunr_models::GpuRequirement::Required);
        let tip = d.hardware_advisory("CPU").expect("Required must warn on CPU");
        assert!(tip.contains("Very slow") || tip.to_lowercase().contains("gpu"),
            "advisory should warn about CPU performance: {tip}");
        assert!(d.hardware_advisory("CUDA").is_none());
        assert!(d.hardware_advisory("CoreML").is_none());
    }

    #[test]
    fn hardware_advisory_recommended_warns_on_cpu_only() {
        let d = synthetic(prunr_models::GpuRequirement::Recommended);
        assert!(d.hardware_advisory("CPU").is_some());
        assert!(d.hardware_advisory("CUDA").is_none());
    }

    #[test]
    fn hardware_advisory_none_never_warns() {
        let d = synthetic(prunr_models::GpuRequirement::None);
        assert!(d.hardware_advisory("CPU").is_none());
        assert!(d.hardware_advisory("CUDA").is_none());
    }

    // ── License gating ─────────────────────────────────────────────────

    #[test]
    fn license_gate_blocks_when_descriptor_requires_acceptance_and_settings_empty() {
        // SD V1.5 is the only license-gated entry today; relies on REGISTRY.
        let mut s = super::super::super::settings::Settings::default();
        assert!(requires_license_gate(prunr_models::ModelId::SdV15InpaintFp16, &s));
        // Push directly into the field — `accept_license` would persist
        // to the user's real config file from test scope, which we don't
        // want; behaviour under test is the gate itself, not the writer.
        s.accepted_licenses.push(format!("{:?}", prunr_models::ModelId::SdV15InpaintFp16));
        assert!(!requires_license_gate(prunr_models::ModelId::SdV15InpaintFp16, &s));
    }

    #[test]
    fn license_gate_passes_for_non_restricted_models() {
        let s = super::super::super::settings::Settings::default();
        assert!(!requires_license_gate(prunr_models::ModelId::U2net, &s));
        assert!(!requires_license_gate(prunr_models::ModelId::LaMaFp32, &s));
    }
}
