//! Model Store modal — single user-facing surface for browsing,
//! installing, cancelling, and deleting on-demand models.

use egui::{Color32, RichText};
use egui_material_icons::icons::*;

use prunr_models::{
    descriptor as model_descriptor, ModelCategory, ModelDescriptor, ModelId,
    ModelSource, REGISTRY,
};

use crate::gui::app::PrunrApp;
use crate::gui::download_manager::DownloadState;
use crate::gui::theme;
use super::chip::{button, ButtonKind};
use super::format_byte_size;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CardAction {
    /// Bundled model — no action available; the footer reads "Built-in"
    /// and the badge "Ready".
    Bundled,
    Download,
    Cancel,
    Verifying,
    Delete,
    Retry,
}

pub(crate) fn card_action(
    desc: &prunr_models::ModelDescriptor,
    is_installed: bool,
    download_state: &DownloadState,
) -> CardAction {
    if matches!(desc.source, ModelSource::Bundled { .. } | ModelSource::MultiPartBundled { .. }) {
        return CardAction::Bundled;
    }
    // Exhaustive on purpose — adding a new `DownloadState` variant
    // forces this match to grow with it instead of silently falling
    // through to `Download` and re-queueing a "paused" download.
    match download_state {
        DownloadState::InProgress { .. } | DownloadState::Queued => CardAction::Cancel,
        DownloadState::Verifying => CardAction::Verifying,
        DownloadState::Failed { .. } if !is_installed => CardAction::Retry,
        DownloadState::Failed { .. } | DownloadState::Done | DownloadState::Idle => {
            if is_installed { CardAction::Delete } else { CardAction::Download }
        }
    }
}

/// Returns true when the user closed the modal this frame.
pub fn render(ctx: &egui::Context, app: &mut PrunrApp) -> bool {
    let provider = app.settings.effective_backend().to_string();

    let initial_filter = app.model_store.as_ref().and_then(|r| r.filter);
    let mut new_filter = initial_filter;
    let mut pending_actions: Vec<(ModelId, CardAction)> = Vec::with_capacity(REGISTRY.len());

    let closed = theme::bare_modal_window(ctx, "model_store", "Model Store", STORE_SIZE, |ui| {
        let subtitle = format!("Everything runs on this computer. Download what you need. {} on disk.", format_byte_size(app.hardware_install_cache.models_on_disk));
        super::chip::modal_page(ui, "Model Store", &subtitle, |ui| {
            super::chip::choice_buttons(ui, &FILTERS, &mut new_filter);
            ui.add_space(14.0);

            // The scroll bar takes its own lane instead of overlapping the cards.
            let lane = ui.spacing().scroll.bar_width + ui.spacing().scroll.bar_outer_margin;
            let card_width = ((ui.available_width() - 2.0 * CARD_GAP - lane) / 3.0).floor();
            // Cards in a row stretch to its tallest, measured last frame,
            // so a row's footers line up.
            let heights_id = egui::Id::new(("store_row_heights", new_filter.map(|c| c as u8)));
            let last: Vec<f32> = ui.data(|d| d.get_temp(heights_id)).unwrap_or_default();
            let mut heights = vec![0.0f32; REGISTRY.len().div_ceil(3)];
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                egui::Grid::new("store_cards").spacing([CARD_GAP, CARD_GAP]).show(ui, |ui| {
                    let shown = REGISTRY.iter().filter(|d| new_filter.is_none_or(|c| d.category == c));
                    for (i, desc) in shown.enumerate() {
                        let row = i / 3;
                        let state = app.download_manager.state(desc.id);
                        let height = last.get(row).copied().unwrap_or(0.0);
                        let installed = app.hardware_install_cache.has_installed(desc.id);
                        let (action, natural) = render_card(ui, desc, installed, &state, &provider, egui::vec2(card_width, height));
                        heights[row] = heights[row].max(natural);
                        if let Some(action) = action {
                            pending_actions.push((desc.id, action));
                        }
                        if i % 3 == 2 {
                            ui.end_row();
                        }
                    }
                });
            });
            if heights != last {
                ui.data_mut(|d| d.insert_temp(heights_id, heights));
                ui.ctx().request_repaint();
            }
        })
    });

    for (id, action) in pending_actions {
        match action {
            CardAction::Download | CardAction::Retry => {
                if requires_license_gate(id, &app.settings) {
                    app.pending_license_request = Some(LicenseRequest::Model(id));
                } else {
                    app.download_manager.start_download(id);
                }
            }
            CardAction::Cancel => {
                app.download_manager.cancel_download(id);
            }
            CardAction::Delete => {
                app.remove_model(id, ctx);
            }
            CardAction::Bundled | CardAction::Verifying => {}
        }
    }

    if new_filter != initial_filter {
        if let Some(req) = app.model_store.as_mut() {
            req.filter = new_filter;
        }
    }

    closed
}

const STORE_SIZE: [f32; 2] = [1000.0, 700.0];
const CARD_GAP: f32 = 16.0;
const CARD_MIN_HEIGHT: f32 = 150.0;
const CARD_PADDING: f32 = 16.0;
const FOOTER_HEIGHT: f32 = 30.0;
const BAR_HEIGHT: f32 = 4.0;
const READY_FILL: Color32 = Color32::from_rgba_premultiplied(13, 25, 13, 31);
const READY_TEXT: Color32 = Color32::from_rgb(0x9f, 0xe0, 0x9f);
const FAILED_TEXT: Color32 = Color32::from_rgb(0xd8, 0x8e, 0x8e);

/// Filter tabs in display order; `None` is "All".
const FILTERS: [super::chip::Choice<Option<ModelCategory>>; 8] = [
    super::chip::Choice { value: None, name: "All", description: "", enabled: true },
    filter(ModelCategory::Segmentation),
    filter(ModelCategory::EdgeDetection),
    filter(ModelCategory::Inpaint),
    filter(ModelCategory::Upscale),
    filter(ModelCategory::Face),
    filter(ModelCategory::Colorize),
    filter(ModelCategory::Depth),
];

const fn filter(c: ModelCategory) -> super::chip::Choice<Option<ModelCategory>> {
    super::chip::Choice { value: Some(c), name: category_name(c), description: "", enabled: true }
}

/// A category's name on its filter tab and its cards.
const fn category_name(c: ModelCategory) -> &'static str {
    match c {
        ModelCategory::Segmentation => "Background removal",
        ModelCategory::EdgeDetection => "Sketch",
        ModelCategory::Inpaint => "Eraser",
        ModelCategory::Upscale => "Upscale",
        ModelCategory::Selection => "Selection",
        ModelCategory::Face => "Faces",
        ModelCategory::Colorize => "Colorize",
        ModelCategory::Depth => "Depth",
    }
}

fn category_icon(c: ModelCategory) -> &'static str {
    match c {
        ModelCategory::Segmentation => ICON_CONTENT_CUT.codepoint,
        ModelCategory::EdgeDetection => ICON_DRAW.codepoint,
        ModelCategory::Inpaint => ICON_BRUSH.codepoint,
        ModelCategory::Upscale => ICON_ARROW_UPWARD.codepoint,
        ModelCategory::Selection => ICON_AUTO_AWESOME.codepoint,
        ModelCategory::Face => ICON_FACE.codepoint,
        ModelCategory::Colorize => ICON_PALETTE.codepoint,
        ModelCategory::Depth => ICON_LANDSCAPE.codepoint,
    }
}

/// One card: icon, name and kind; what it is for; and a footer with its
/// size and what can be done with it. The name row opens the licence,
/// author and source in place.
fn render_card(
    ui: &mut egui::Ui,
    desc: &ModelDescriptor,
    installed: bool,
    state: &DownloadState,
    provider: &str,
    size: egui::Vec2,
) -> (Option<CardAction>, f32) {
    let action = card_action(desc, installed, state);
    let advisory = desc.hardware_advisory(provider);
    let border = if installed {
        theme::ACCENT_BRIGHT.gamma_multiply(0.35)
    } else {
        theme::FIELD_FILL
    };
    let mut clicked = None;
    let mut natural = 0.0;
    egui::Frame::NONE
        .fill(theme::CARD_FILL)
        .stroke(egui::Stroke::new(1.0, border))
        .corner_radius(16.0)
        .inner_margin(CARD_PADDING)
        .show(ui, |ui| ui.vertical(|ui| {
            ui.set_width(size.x - 2.0 * CARD_PADDING);
            let open_id = ui.id().with(("store_card", desc.id));
            let mut open = ui.data(|d| d.get_temp::<bool>(open_id).unwrap_or(false));
            let (icon, tag) = (category_icon(desc.category), category_name(desc.category));
            let top = ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(36.0, 36.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 18.0, theme::ACCENT_BRIGHT.gamma_multiply(0.18));
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, icon, egui::FontId::proportional(17.0), theme::GLOW);
                ui.vertical(|ui| {
                    ui.label(RichText::new(desc.display_name).size(theme::FONT_SIZE_BODY).strong().color(theme::TEXT_PRIMARY));
                    ui.label(RichText::new(tag).size(theme::FONT_SIZE_CAPTION).color(theme::TEXT_SECONDARY));
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                    let chevron = if open { ICON_EXPAND_LESS.codepoint } else { ICON_EXPAND_MORE.codepoint };
                    ui.label(RichText::new(chevron).size(theme::FONT_SIZE_HEADING).color(theme::TEXT_SECONDARY));
                });
            }).response;
            let details = ui.interact(top.rect, open_id.with("toggle"), egui::Sense::click())
                .on_hover_cursor(egui::CursorIcon::PointingHand);
            details.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, true, open, format!("Details {}", desc.display_name)));
            if details.clicked() {
                open = !open;
            }
            ui.add(egui::Label::new(RichText::new(desc.description).size(theme::FONT_SIZE_SECONDARY).color(theme::TEXT_PRIMARY.gamma_multiply(0.8))).wrap());
            if let Some(msg) = advisory {
                ui.label(RichText::new(msg).size(theme::FONT_SIZE_SECONDARY).color(theme::ADVISORY_TEXT));
            }
            if open {
                card_details(ui, desc, state);
            }
            let downloading = matches!(state, DownloadState::InProgress { .. });
            let footer = FOOTER_HEIGHT + if downloading { BAR_HEIGHT + ui.spacing().item_spacing.y } else { 0.0 };
            let content = ui.min_rect().height() + theme::SPACE_XS + footer;
            natural = (content + 2.0 * CARD_PADDING).max(CARD_MIN_HEIGHT);
            ui.add_space((size.y.max(natural) - 2.0 * CARD_PADDING - content).max(0.0) + theme::SPACE_XS);
            if let DownloadState::InProgress { bytes_so_far, total_bytes } = *state {
                let fraction = if total_bytes > 0 { bytes_so_far as f32 / total_bytes as f32 } else { 0.0 };
                let (bar, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), BAR_HEIGHT), egui::Sense::hover());
                ui.painter().rect_filled(bar, 2.0, theme::RAIL);
                ui.painter().rect_filled(egui::Rect::from_min_size(bar.min, egui::vec2(bar.width() * fraction, BAR_HEIGHT)), 2.0, theme::ACCENT_BRIGHT);
            }
            let footer_size = egui::vec2(ui.available_width(), FOOTER_HEIGHT);
            ui.allocate_ui_with_layout(footer_size, egui::Layout::left_to_right(egui::Align::Center), |ui| {
                let (meta, color) = card_meta(desc, installed, state, advisory.is_some());
                ui.label(RichText::new(meta).size(theme::FONT_SIZE_SECONDARY).color(color));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if matches!(action, CardAction::Bundled) || (installed && action == CardAction::Delete) {
                        badge(ui, if action == CardAction::Bundled { "Ready" } else { "Installed" });
                    }
                    if let Some(label) = button_label(action) {
                        let kind = match action {
                            CardAction::Delete => ButtonKind::Ghost,
                            CardAction::Download => ButtonKind::Primary,
                            _ => ButtonKind::Secondary,
                        };
                        let reason = (!action_enabled(action)).then_some(advisory.unwrap_or("Nothing to do yet."));
                        let resp = super::chip::gated(ui, reason, |ui| button(ui, kind, label));
                        let resp = super::chip::named_with(resp, || format!("{label} {}", desc.display_name));
                        if resp.clicked() {
                            clicked = Some(action);
                        }
                    }
                });
            });
            ui.data_mut(|d| d.insert_temp(open_id, open));
        }));
    (clicked, natural)
}

/// The footer's text: the size, or what the download is doing.
fn card_meta(desc: &ModelDescriptor, installed: bool, state: &DownloadState, advisory: bool) -> (String, Color32) {
    let size = desc.size_mb().map(super::get_model::size_label);
    match state {
        DownloadState::InProgress { bytes_so_far, total_bytes } => {
            let pct = if *total_bytes > 0 { bytes_so_far * 100 / total_bytes } else { 0 };
            (format!("{} of {} \u{00b7} {pct}%", format_byte_size(*bytes_so_far), format_byte_size(*total_bytes)), theme::TEXT_SECONDARY)
        }
        DownloadState::Queued => ("Queued".into(), theme::TEXT_SECONDARY),
        DownloadState::Verifying => ("Verifying\u{2026}".into(), theme::TEXT_SECONDARY),
        DownloadState::Failed { .. } if !installed => ("Download failed".into(), FAILED_TEXT),
        _ => match size {
            Some(s) => (s, if advisory { theme::ADVISORY_TEXT } else { theme::TEXT_SECONDARY }),
            None => ("Built-in".into(), theme::TEXT_SECONDARY),
        },
    }
}

/// What the name row opens: the licence, the author and the source, and
/// why the last download failed.
fn card_details(ui: &mut egui::Ui, desc: &ModelDescriptor, state: &DownloadState) {
    let license = desc.license();
    ui.add_space(theme::SPACE_XS);
    let line = |ui: &mut egui::Ui, key: &str, value: &str| {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(key).size(theme::FONT_SIZE_SECONDARY).color(theme::TEXT_SECONDARY));
            ui.label(RichText::new(value).size(theme::FONT_SIZE_SECONDARY).color(theme::TEXT_PRIMARY));
        });
    };
    line(ui, "License", license.license);
    line(ui, "Author", license.author);
    ui.hyperlink_to(RichText::new("Source \u{2197}").size(theme::FONT_SIZE_SECONDARY), license.source_url);
    if let DownloadState::Failed { error } = state {
        ui.label(RichText::new(format!("Last attempt: {error}")).size(theme::FONT_SIZE_SECONDARY).color(FAILED_TEXT));
    }
}

fn badge(ui: &mut egui::Ui, text: &str) {
    egui::Frame::NONE.fill(READY_FILL).corner_radius(theme::PILL_RADIUS).inner_margin(egui::Margin::symmetric(10, 4)).show(ui, |ui| {
        ui.label(RichText::new(text).size(theme::FONT_SIZE_CAPTION).color(READY_TEXT));
    });
}

fn button_label(action: CardAction) -> Option<&'static str> {
    match action {
        CardAction::Bundled => None,
        CardAction::Download => Some("Download"),
        CardAction::Cancel => Some("Cancel"),
        CardAction::Verifying => None,
        CardAction::Delete => Some("Remove"),
        CardAction::Retry => Some("Try again"),
    }
}

fn action_enabled(action: CardAction) -> bool {
    !matches!(action, CardAction::Bundled | CardAction::Verifying)
}


/// True when the descriptor for `id` declares a restrictive license that
/// the user hasn't yet accepted in `Settings::accepted_licenses`.
pub(crate) fn requires_license_gate(id: ModelId, settings: &super::super::settings::Settings) -> bool {
    let Some(desc) = model_descriptor(id) else { return false };
    desc.requires_license_acceptance() && !settings.has_accepted_license(id)
}

/// What a license dialog asks the user to accept before a download.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LicenseRequest {
    Model(ModelId),
    /// NVIDIA's CUDA and cuDNN libraries for the CUDA runtime.
    NvidiaLibraries,
}

/// License-acceptance modal. Returns `(close_requested, accepted)`:
/// - `close_requested`: the modal should be dismissed (Cancel or backdrop)
/// - `accepted`: the user clicked Accept — caller persists + starts download
pub(crate) fn render_license_dialog(
    ctx: &egui::Context,
    request: LicenseRequest,
) -> (bool, bool) {
    let (name, license, links): (&str, &str, Vec<&str>) = match request {
        LicenseRequest::Model(id) => {
            let Some(desc) = model_descriptor(id) else { return (true, false) };
            // A bundled model was accepted at build time and never asks.
            if desc.source.is_bundled() {
                return (true, false);
            }
            let license = desc.license();
            (desc.display_name, license.license, vec![license.license_url, license.source_url])
        }
        LicenseRequest::NvidiaLibraries => (
            "NVIDIA CUDA 12 and cuDNN 9 libraries",
            "NVIDIA CUDA Toolkit EULA and cuDNN Software License Agreement",
            crate::runtime_install::NVIDIA_LICENSE_URLS.to_vec(),
        ),
    };

    let mut close_requested = false;
    let mut accepted = false;

    let backdrop_closed = theme::standard_modal_window(
        ctx, "license_dialog", "License acceptance",
        [theme::STANDARD_MODAL_WIDTH, 360.0],
        |ui| {
            ui.label(
                RichText::new(name)
                    .size(theme::FONT_SIZE_BODY)
                    .strong()
                    .color(theme::TEXT_PRIMARY),
            );
            ui.label(
                RichText::new(format!("License: {license}"))
                    .size(theme::FONT_SIZE_BODY)
                    .color(theme::TEXT_PRIMARY),
            );
            ui.add_space(theme::SPACE_XS);
            ui.label(
                RichText::new("This download comes under terms that you must \
                     review and accept before downloading. The full text \
                     is published at the links below.".to_string())
                .size(theme::FONT_SIZE_BODY)
                .color(theme::TEXT_SECONDARY),
            );
            ui.add_space(theme::SPACE_XS);
            for link in &links {
                ui.hyperlink_to(*link, *link);
            }
            ui.add_space(theme::SPACE_MD);

            ui.horizontal(|ui| {
                if button(ui, ButtonKind::Secondary, "Cancel").clicked() {
                    close_requested = true;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if button(ui, ButtonKind::Primary, "Accept and download").clicked() {
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
    fn dl_failed_network() -> DownloadState {
        DownloadState::Failed { error: "timeout".into() }
    }
    fn dl_failed_disk_full() -> DownloadState {
        DownloadState::Failed { error: "No space left on device".into() }
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
                    author: "test",
                    license: "Apache-2.0",
                    license_url: "https://example.test/lic",
                    source_url: "https://example.test/src",
                },
                fp16: None,
            },
            version: "1.0.0",
            gpu,
            incompatible_eps: &[],
            working_set_mb: 100,
            parallel_jobs_pay_on_cpu: false,
            tile_size_multiple: None,
            recommended_tile: None,
            attribution_required: false,
            upscale: None,
        }
    }

    fn synthetic_bundled() -> prunr_models::ModelDescriptor {
        prunr_models::ModelDescriptor {
            id: prunr_models::ModelId::Silueta,
            display_name: "test",
            description: "test",
            category: prunr_models::ModelCategory::Segmentation,
            source: ModelSource::Bundled {
                license: prunr_models::LicenseInfo { author: "test", license: "MIT", license_url: "", source_url: "" },
            },
            version: "1.0.0",
            gpu: prunr_models::GpuRequirement::None,
            incompatible_eps: &[],
            working_set_mb: 100,
            parallel_jobs_pay_on_cpu: false,
            tile_size_multiple: None,
            recommended_tile: None,
            attribution_required: false,
            upscale: None,
        }
    }

    #[test]
    fn bundled_always_renders_as_none() {
        let d = synthetic_bundled();
        assert_eq!(card_action(&d, true, &dl_idle()), CardAction::Bundled);
        assert_eq!(card_action(&d, false, &dl_failed_network()), CardAction::Bundled);
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
    fn ondemand_failed_network_uninstalled_is_retry_clickable() {
        let d = synthetic(prunr_models::GpuRequirement::None);
        let a = card_action(&d, false, &dl_failed_network());
        assert_eq!(a, CardAction::Retry);
        assert!(action_enabled(a));
    }

    /// A failure the network retry gave up on (disk full, 404, bad
    /// checksum) can still be retried by hand once the cause is fixed.
    #[test]
    fn ondemand_failed_disk_full_uninstalled_is_retry_clickable() {
        let d = synthetic(prunr_models::GpuRequirement::None);
        let a = card_action(&d, false, &dl_failed_disk_full());
        assert_eq!(a, CardAction::Retry);
        assert!(action_enabled(a));
    }

    #[test]
    fn ondemand_failed_but_already_installed_shows_delete() {
        let d = synthetic(prunr_models::GpuRequirement::None);
        assert_eq!(card_action(&d, true, &dl_failed_network()), CardAction::Delete);
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
