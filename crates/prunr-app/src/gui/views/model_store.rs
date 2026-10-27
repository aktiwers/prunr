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

/// What a click on a card asks for.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Click {
    Footer(CardAction),
    Test,
    CancelTest,
}

/// Where a card's model stands with a test on this computer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CardTest {
    /// Nothing to offer: not installed, not a model the test runs, or no
    /// accelerator to test.
    Off,
    /// Never measured here; `ready` while nothing else uses the machine.
    Untested { ready: bool },
    /// Its test runs: on what now, and how far.
    Testing { on: &'static str, fraction: f32 },
    /// Measured: the engine it runs on, and its time there.
    Tested { on: &'static str, ms: Option<f32> },
}

/// `id`'s test line, from the test under way and what tests measured.
pub(crate) fn card_test(app: &PrunrApp, id: ModelId) -> CardTest {
    use crate::gui::hardware_cache::Cell;
    if let Some(test) = app.machine_test.as_ref().filter(|t| t.only() == Some(id)) {
        let step = test.step().filter(|s| s.model == id);
        let on = step.filter(|s| s.on_gpu()).map_or("the processor", |s| s.column.provider());
        return CardTest::Testing { on, fraction: step.map_or(0.0, |s| s.fraction()) };
    }
    if !app.model_testable(id) {
        return CardTest::Off;
    }
    let Some(row) = app.test_results.rows.iter().find(|r| r.model == id) else {
        return CardTest::Untested { ready: app.can_model_test_intent(id) };
    };
    let on = if app.settings.cpu_only() { prunr_core::engine::CPU_NAME } else { row.runs_on };
    let ms = match on {
        prunr_core::engine::CPU_NAME => row.cpu_ms,
        engine => app.test_results.engines.iter().zip(&row.cells).find(|(e, _)| e.name == engine).and_then(|(_, cell)| match cell {
            Cell::Time(ms) => Some(*ms),
            _ => None,
        }),
    };
    CardTest::Tested { on, ms }
}

/// Returns true when the user closed the modal this frame.
pub fn render(ctx: &egui::Context, app: &mut PrunrApp) -> bool {
    let provider = app.settings.effective_backend().to_string();
    let engines = super::fmt::and_list(std::iter::once("the processor").chain(app.test_results.engines.iter().filter(|e| e.installed).map(|e| e.name)));

    let initial_filter = app.model_store.as_ref().and_then(|r| r.filter);
    let mut new_filter = initial_filter;
    let mut pending_actions: Vec<(ModelId, Click)> = Vec::with_capacity(REGISTRY.len());

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
                        let card = Card { desc, installed: app.hardware_install_cache.has_installed(desc.id), state: &state, test: card_test(app, desc.id) };
                        let (action, natural) = render_card(ui, &card, (&provider, &engines), egui::vec2(card_width, height));
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
        let action = match action {
            Click::Footer(action) => action,
            Click::Test => {
                app.handle_model_test_intent(ctx, id);
                continue;
            }
            Click::CancelTest => {
                app.cancel_machine_test();
                continue;
            }
        };
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

/// A card's model and what it shows of it.
struct Card<'a> {
    desc: &'a ModelDescriptor,
    installed: bool,
    state: &'a DownloadState,
    test: CardTest,
}

/// One card: icon, name and kind; what it is for; where it runs fastest
/// here, or Test; and a footer with its size and what can be done with
/// it. The name row opens the licence, author and source in place.
/// `(provider, engines)`: the runtime's provider, for the hardware
/// advisory, and what a test tries, for Test's tooltip.
fn render_card(ui: &mut egui::Ui, card: &Card<'_>, (provider, engines): (&str, &str), size: egui::Vec2) -> (Option<Click>, f32) {
    let Card { desc, installed, state, test } = *card;
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
            if let Some(click) = test_line(ui, desc, test, engines) {
                clicked = Some(click);
            }
            let downloading = matches!(state, DownloadState::InProgress { .. });
            let footer = FOOTER_HEIGHT + if downloading { BAR_HEIGHT + ui.spacing().item_spacing.y } else { 0.0 };
            let content = ui.min_rect().height() + theme::SPACE_XS + footer;
            natural = (content + 2.0 * CARD_PADDING).max(CARD_MIN_HEIGHT);
            ui.add_space((size.y.max(natural) - 2.0 * CARD_PADDING - content).max(0.0) + theme::SPACE_XS);
            if let DownloadState::InProgress { bytes_so_far, total_bytes } = *state {
                let fraction = if total_bytes > 0 { bytes_so_far as f32 / total_bytes as f32 } else { 0.0 };
                super::chip::progress_bar(ui, fraction, BAR_HEIGHT);
            }
            let footer_size = egui::vec2(ui.available_width(), FOOTER_HEIGHT);
            ui.allocate_ui_with_layout(footer_size, egui::Layout::left_to_right(egui::Align::Center), |ui| {
                let (meta, color, why) = card_meta(desc, installed, state, advisory.is_some());
                let line = ui.label(RichText::new(&meta).size(theme::FONT_SIZE_SECONDARY).color(color));
                if let Some(why) = why {
                    super::chip::tooltip(line, &meta, why, None);
                }
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
                            clicked = Some(Click::Footer(action));
                        }
                    }
                });
            });
            ui.data_mut(|d| d.insert_temp(open_id, open));
        }));
    (clicked, natural)
}

/// A card's test line: "Not tested on this computer." with Test, the test
/// under way with Cancel, or where the model runs fastest here.
fn test_line(ui: &mut egui::Ui, desc: &ModelDescriptor, test: CardTest, engines: &str) -> Option<Click> {
    let text = |text: &str, color: Color32| RichText::new(text).size(theme::FONT_SIZE_SECONDARY).color(color);
    let mut clicked = None;
    match test {
        CardTest::Off => {}
        CardTest::Untested { ready } => {
            ui.horizontal(|ui| {
                ui.label(text("Not tested on this computer.", theme::ADVISORY_TEXT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let reason = (!ready).then_some("Waits until the current work is done.");
                    let resp = super::chip::gated(ui, reason, |ui| button(ui, ButtonKind::Secondary, "Test"));
                    let body = format!(
                        "Times {} on {engines}, checks each gives the same picture, and runs it on the fastest from now on. Takes about a minute.",
                        desc.display_name,
                    );
                    let resp = super::chip::tooltip(resp, "Test", &body, None);
                    if super::chip::named_with(resp, || format!("Test {}", desc.display_name)).clicked() {
                        clicked = Some(Click::Test);
                    }
                });
            });
        }
        CardTest::Testing { on, fraction } => {
            ui.horizontal(|ui| {
                ui.label(text(&format!("Testing on {on}\u{2026}"), theme::TEXT_SECONDARY));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let resp = button(ui, ButtonKind::Secondary, "Cancel");
                    if super::chip::named_with(resp, || format!("Cancel the test of {}", desc.display_name)).clicked() {
                        clicked = Some(Click::CancelTest);
                    }
                });
            });
            super::chip::progress_bar(ui, fraction, BAR_HEIGHT);
        }
        CardTest::Tested { on, ms } => {
            ui.horizontal(|ui| {
                ui.label(RichText::new(ICON_CHECK.codepoint).size(theme::FONT_SIZE_SECONDARY).color(theme::ACCENT_BRIGHT));
                let on = if on == prunr_core::engine::CPU_NAME { "the processor" } else { on };
                let line = match ms {
                    Some(ms) => format!("Fastest here: {on}, {}", super::fmt::seconds(ms)),
                    None => format!("Fastest here: {on}"),
                };
                ui.label(text(&line, theme::TEXT_SECONDARY));
            });
        }
    }
    clicked
}

/// The footer's text: the size, or what the download is doing.
/// The card's footer line, its colour, and what its hover says (why a
/// download failed).
fn card_meta<'a>(desc: &ModelDescriptor, installed: bool, state: &'a DownloadState, advisory: bool) -> (String, Color32, Option<&'a str>) {
    let size = desc.size_mb().map(super::get_model::size_label);
    match state {
        DownloadState::InProgress { bytes_so_far, total_bytes } => {
            let pct = if *total_bytes > 0 { bytes_so_far * 100 / total_bytes } else { 0 };
            (format!("{} of {} \u{00b7} {pct}%", format_byte_size(*bytes_so_far), format_byte_size(*total_bytes)), theme::TEXT_SECONDARY, None)
        }
        DownloadState::Queued => ("Queued".into(), theme::TEXT_SECONDARY, None),
        DownloadState::Verifying => ("Verifying\u{2026}".into(), theme::TEXT_SECONDARY, None),
        DownloadState::Failed { error } if !installed => ("Download failed".into(), FAILED_TEXT, Some(error)),
        _ => match size {
            Some(s) => (s, if advisory { theme::ADVISORY_TEXT } else { theme::TEXT_SECONDARY }, None),
            None => ("Built-in".into(), theme::TEXT_SECONDARY, None),
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
    /// An install that downloads NVIDIA's CUDA and cuDNN libraries.
    Nvidia(crate::runtime_install::InstallTarget),
}

/// License-acceptance modal. Returns `(close_requested, accepted)`:
/// - `close_requested`: the modal should be dismissed (Cancel or backdrop)
/// - `accepted`: the user clicked Accept — caller persists + starts download
pub(crate) fn render_license_dialog(
    ctx: &egui::Context,
    request: LicenseRequest,
) -> (bool, bool) {
    let (name, license, links): (&str, &str, [&str; 2]) = match request {
        LicenseRequest::Model(id) => {
            let Some(desc) = model_descriptor(id) else { return (true, false) };
            // A bundled model was accepted at build time and never asks.
            if desc.source.is_bundled() {
                return (true, false);
            }
            let license = desc.license();
            (desc.display_name, license.license, [license.license_url, license.source_url])
        }
        LicenseRequest::Nvidia(_) => (
            "NVIDIA CUDA 12 and cuDNN 9 libraries",
            "NVIDIA CUDA Toolkit EULA and cuDNN Software License Agreement",
            crate::runtime_install::NVIDIA_LICENSE_URLS,
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
            places: prunr_models::Places::UNMEASURED,
            needs: prunr_models::Needs {
                cpu: prunr_models::Cost::estimate(100, 0, "a test fixture"),
                mac_cpu: prunr_models::Priced::Unmeasured,
                cuda: prunr_models::Priced::Unmeasured,
                directml: prunr_models::Priced::Unmeasured,
                coreml: prunr_models::Priced::Unmeasured,
                openvino: prunr_models::Priced::Unmeasured,
                migraphx: prunr_models::Priced::Unmeasured,
            },
            parallel_jobs_pay_on_cpu: false,
            tile_size_multiple: None,
            recommended_tile: None,
            free_dims: &[],
            attribution_required: false,
            upscale: None,
        }
    }

    fn synthetic_bundled() -> prunr_models::ModelDescriptor {
        prunr_models::ModelDescriptor {
            id: prunr_models::ModelId::Silueta,
            category: prunr_models::ModelCategory::Segmentation,
            source: ModelSource::Bundled {
                license: prunr_models::LicenseInfo { author: "test", license: "MIT", license_url: "", source_url: "" },
            },
            ..synthetic(prunr_models::GpuRequirement::None)
        }
    }

    /// What a card's button offers, by install and download state. A
    /// failure the network retry gave up on (disk full, 404, bad checksum)
    /// can still be retried by hand once the cause is fixed.
    #[test]
    fn a_cards_action_follows_its_install_and_download() {
        let failed = |error: &str| DownloadState::Failed { error: error.into() };
        let bundled = synthetic_bundled();
        assert_eq!(card_action(&bundled, true, &DownloadState::Idle), CardAction::Bundled);
        assert_eq!(card_action(&bundled, false, &failed("timeout")), CardAction::Bundled);
        let d = synthetic(prunr_models::GpuRequirement::None);
        for (installed, state, action, enabled) in [
            (false, DownloadState::Idle, CardAction::Download, true),
            (true, DownloadState::Done, CardAction::Delete, true),
            (false, DownloadState::InProgress { bytes_so_far: 1, total_bytes: 100 }, CardAction::Cancel, true),
            (false, DownloadState::Verifying, CardAction::Verifying, false),
            (false, failed("timeout"), CardAction::Retry, true),
            (false, failed("No space left on device"), CardAction::Retry, true),
            (true, failed("timeout"), CardAction::Delete, true),
        ] {
            let a = card_action(&d, installed, &state);
            assert_eq!((a, action_enabled(a)), (action, enabled), "installed={installed} {state:?}");
        }
    }

    // ── Hardware advisory ─────────────────────────────────────────────
    // `card_action` no longer reads `gpu` — the dispatch is gated only by
    // source/install/download_state. Tests here pin the *advisory* surface.

    #[test]
    fn a_gpu_model_warns_on_the_cpu_only() {
        use prunr_models::GpuRequirement::{None as NoGpu, Recommended, Required};
        let tip = synthetic(Required).hardware_advisory("CPU").expect("Required must warn on CPU");
        assert!(tip.contains("Very slow") || tip.to_lowercase().contains("gpu"),
            "advisory should warn about CPU performance: {tip}");
        for (gpu, backend, warns) in [
            (Required, "CUDA", false),
            (Required, "CoreML", false),
            (Recommended, "CPU", true),
            (Recommended, "CUDA", false),
            (NoGpu, "CPU", false),
            (NoGpu, "CUDA", false),
        ] {
            assert_eq!(synthetic(gpu).hardware_advisory(backend).is_some(), warns, "{gpu:?} on {backend}");
        }
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
        assert!(!requires_license_gate(prunr_models::ModelId::U2net, &s), "no license to accept");
        assert!(!requires_license_gate(prunr_models::ModelId::LaMaFp32, &s));
    }

}
