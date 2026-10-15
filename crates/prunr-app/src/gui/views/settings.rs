//! Settings modal: a sidebar of pages, each a few groups of rows.
//! - **Performance**: what runs and how much memory it may take
//! - **Hardware**: this computer, the accelerator runtime, the cache
//! - **Images**: what happens when images arrive
//! - **Editing**: how the canvas looks while editing
//! - **Hotkeys**: the shortcut table, each chord rebindable in place
//! - **Credits**: installed-model attribution (CC-BY compliance)
//!
//! Per-image knobs (gamma, threshold, line mode, …) live on the persistent
//! adjustments toolbar (rows 2 + 3), not here.

use std::sync::Arc;

use egui::{Color32, RichText};
use egui_material_icons::icons::*;

use crate::gui::app::PrunrApp;
use crate::gui::hardware_cache::HardwareInstallCache;
use crate::gui::settings::Settings;
use crate::gui::theme;
use crate::runtime_install::{RuntimeChoice, RuntimeId};

use super::chip::{button, ButtonKind};
use super::{hint, kv_row};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsTab {
    Performance,
    Hardware,
    Images,
    Editing,
    Hotkeys,
    Credits,
}

impl SettingsTab {
    fn label(self) -> &'static str {
        match self {
            Self::Performance => "Performance",
            Self::Hardware => "Hardware",
            Self::Images => "Images",
            Self::Editing => "Editing",
            Self::Hotkeys => "Hotkeys",
            Self::Credits => "Credits",
        }
    }

    fn icon(self) -> &'static str {
        match self {
            Self::Performance => ICON_BOLT.codepoint,
            Self::Hardware => ICON_MEMORY.codepoint,
            Self::Images => ICON_IMAGE.codepoint,
            Self::Editing => ICON_EDIT.codepoint,
            Self::Hotkeys => ICON_KEYBOARD.codepoint,
            Self::Credits => ICON_INFO.codepoint,
        }
    }

    pub(crate) const ALL: [SettingsTab; 6] = [
        Self::Performance, Self::Hardware, Self::Images, Self::Editing, Self::Hotkeys, Self::Credits,
    ];

    /// Parse a label string into a tab. Used by PRUNR_OPEN_TAB.
    pub fn from_label(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|t| t.label() == s)
    }
}

/// Read-only snapshot of runtime-install state passed into the Hardware
/// page so its renderer doesn't need a `&mut PrunrApp`.
pub(crate) struct HardwareSectionContext {
    /// The runtimes this platform offers to download.
    pub offered: &'static [RuntimeId],
    pub installs: HardwareInstallCache,
    /// The runtime being installed, with its progress line.
    pub installing: Option<(RuntimeId, String)>,
    pub active_provider: String,
    pub sd_margin_mb: u64,
}

/// View-layer intent the Hardware page returns so platform side effects
/// (rfd, runtime install) live on the orchestrator, not the view.
#[derive(Debug, Clone, Copy)]
pub(crate) enum HardwareSectionIntent {
    StartInstall(RuntimeId),
    CancelInstall,
    Uninstall(RuntimeId),
    PreferRuntime(RuntimeChoice),
    ClearCompiledCache,
    ForgetGpuFailures,
}

const SIDEBAR_WIDTH: f32 = 220.0;
/// A row control's width before it has been measured once.
const CONTROL_GUESS: f32 = 160.0;
const SIDEBAR_FILL: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 31);
const PAGE_MARGIN: egui::Margin = egui::Margin { left: 24, right: 24, top: 18, bottom: 20 };

pub fn render(ctx: &egui::Context, app: &mut PrunrApp) {
    let mut hardware_intent: Option<HardwareSectionIntent> = None;
    let keep_sd_before = app.settings.keep_sd_loaded;
    let processor_before = app.settings.processor;
    let closed = theme::bare_modal_window(ctx, "settings", "Settings", theme::SETTINGS_MODAL_SIZE, |ui| {
        theme::apply_modal_visuals(ui);
        let height = ui.available_height();
        let mut close = false;
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            render_sidebar(ui, app, height);
            egui::Frame::NONE.inner_margin(PAGE_MARGIN).show(ui, |ui| ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.x = theme::SPACE_SM;
                ui.set_width(ui.available_width());
                ui.set_height(height - (PAGE_MARGIN.top + PAGE_MARGIN.bottom) as f32);
                ui.horizontal(|ui| {
                    ui.label(RichText::new(app.settings_tab.label()).size(theme::FONT_SIZE_MODAL_TITLE).color(theme::TEXT_PRIMARY));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let x = super::chip::icon_square_button(ui, ICON_CLOSE.codepoint, theme::TEXT_SECONDARY, theme::FIELD_FILL);
                        close = super::chip::named(x, "Close settings").clicked();
                    });
                });
                let footer = if app.pending_reset_confirm { 44.0 } else { 0.0 };
                hardware_intent = egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .max_height((ui.available_height() - footer).max(0.0))
                    .show(ui, |ui| render_page(ui, app))
                    .inner;
                if app.pending_reset_confirm {
                    render_reset_confirm(ui, app);
                }
            }));
        });
        close
    });

    if let Some(intent) = hardware_intent {
        dispatch_hardware_intent(app, intent);
    }
    // Turning the switch off frees the resident bundle now, not on the
    // next stroke.
    if keep_sd_before && !app.settings.keep_sd_loaded {
        app.processor.release_inpaint_subprocess();
    }
    if app.settings.processor != processor_before {
        app.apply_processor();
    }
    if closed {
        app.close_settings();
    }
}

fn dispatch_hardware_intent(app: &mut PrunrApp, intent: HardwareSectionIntent) {
    match intent {
        HardwareSectionIntent::StartInstall(rt) => app.start_runtime_install(rt),
        HardwareSectionIntent::CancelInstall => {
            if let Some(p) = app.runtime_install.as_ref() {
                p.cancel.store(true, std::sync::atomic::Ordering::Release);
            }
        }
        HardwareSectionIntent::Uninstall(rt) => {
            match crate::runtime_install::uninstall(rt) {
                Ok(()) => app.toasts.success(format!("{} removed", rt.display_name())),
                Err(e) => app.toasts.error(format!("Uninstall failed: {e}")),
            };
            app.refresh_install_cache();
        }
        HardwareSectionIntent::PreferRuntime(choice) => {
            match crate::runtime_install::prefer(choice) {
                Ok(()) if app.hardware_install_cache.loaded_runtime != Some(choice) => {
                    app.toasts.info(format!("Restart Prunr to switch to {}", choice.name()));
                }
                Ok(()) => {}
                Err(e) => {
                    app.toasts.error(format!("Could not choose {}: {e}", choice.name()));
                }
            }
            app.refresh_install_cache();
        }
        HardwareSectionIntent::ClearCompiledCache => {
            // Off-thread: a multi-GB rmtree on a slow disk would stall
            // the GUI frame.
            std::thread::spawn(|| {
                let bytes = prunr_core::cache::clear_all();
                tracing::info!(bytes, "cleared compiled-model cache");
            });
            app.toasts.info("Clearing compiled-model cache…");
        }
        HardwareSectionIntent::ForgetGpuFailures => {
            std::thread::spawn(|| {
                let entries = prunr_core::ep_compat::clear();
                tracing::info!(entries, "forgot recorded GPU failures");
            });
            app.toasts.info("Every model will try the graphics card again");
        }
    }
}

fn render_page(ui: &mut egui::Ui, app: &mut PrunrApp) -> Option<HardwareSectionIntent> {
    match app.settings_tab {
        SettingsTab::Performance => { render_page_performance(ui, &mut app.settings, app.hardware_install_cache.has_sd_eraser()); None }
        SettingsTab::Hardware => {
            let hw_ctx = HardwareSectionContext {
                offered: RuntimeId::OFFERED,
                installs: app.hardware_install_cache,
                installing: app.runtime_install.as_ref().map(|p| (p.runtime, p.last_event.status_text())),
                active_provider: app.settings.active_backend.clone(),
                sd_margin_mb: app.settings.sd_memory.margin_mb(),
            };
            render_page_hardware(ui, &hw_ctx)
        }
        SettingsTab::Images => { render_page_images(ui, &mut app.settings); None }
        SettingsTab::Editing => { render_page_editing(ui, &mut app.settings); None }
        SettingsTab::Hotkeys => { render_tab_hotkeys(ui, app); None }
        SettingsTab::Credits => { render_tab_model_credits(ui); None }
    }
}

/// The page list, and Reset to defaults at its foot.
fn render_sidebar(ui: &mut egui::Ui, app: &mut PrunrApp, height: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(SIDEBAR_WIDTH, height), egui::Sense::hover());
    let painter = ui.painter();
    let left = egui::CornerRadius { nw: theme::PANEL_RADIUS, sw: theme::PANEL_RADIUS, ne: 0, se: 0 };
    painter.rect_filled(rect, left, SIDEBAR_FILL);
    painter.vline(rect.right(), rect.y_range(), egui::Stroke::new(1.0, theme::FIELD_FILL));
    let mut ui = ui.new_child(egui::UiBuilder::new().max_rect(rect.shrink2(egui::vec2(12.0, 18.0))).layout(egui::Layout::top_down(egui::Align::Min)));
    ui.spacing_mut().item_spacing.y = 2.0;
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.add_space(12.0);
        ui.label(RichText::new("Settings").size(theme::FONT_SIZE_HEADING).color(theme::TEXT_PRIMARY));
    });
    ui.add_space(14.0);
    for tab in SettingsTab::ALL {
        if nav_item(&mut ui, Some(tab.icon()), tab.label(), app.settings_tab == tab).clicked() {
            app.settings_tab = tab;
        }
    }
    ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
        if nav_item(ui, None, "Reset to defaults", false).clicked() {
            app.pending_reset_confirm = true;
        }
    });
}

/// One sidebar entry: a 34 px pill, purple while its page shows.
fn nav_item(ui: &mut egui::Ui, icon: Option<&str>, label: &str, selected: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 34.0), egui::Sense::click());
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, true, selected, label));
    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        let fill = if selected {
            theme::ROW_PICKED
        } else if resp.hovered() {
            theme::ROW_HOVER
        } else {
            Color32::TRANSPARENT
        };
        painter.rect_filled(rect, theme::PILL_RADIUS, fill);
        let mut x = rect.left() + 12.0;
        if let Some(icon) = icon {
            painter.text(egui::pos2(x, rect.center().y), egui::Align2::LEFT_CENTER, icon, egui::FontId::proportional(16.0), theme::ACCENT_RING);
            x += 26.0;
        }
        let color = if selected { Color32::WHITE } else if icon.is_some() { theme::TEXT_PRIMARY } else { theme::TEXT_SECONDARY };
        painter.text(egui::pos2(x, rect.center().y), egui::Align2::LEFT_CENTER, label, egui::FontId::proportional(theme::FONT_SIZE_BODY), color);
    }
    resp
}

/// Inline confirm-then-reset so the destructive scope is spelled out
/// before the user can act on it.
fn render_reset_confirm(ui: &mut egui::Ui, app: &mut PrunrApp) {
    ui.separator();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Reset settings? Presets, licences, background pictures and the model stay.")
            .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_SECONDARY));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if button(ui, ButtonKind::Secondary, "Cancel").clicked() {
                app.pending_reset_confirm = false;
            }
            if button(ui, ButtonKind::Destructive, "Reset").clicked() {
                app.settings.reset_preserving_identity();
                // The shipped keys go live now, not at the next launch.
                app.apply_hotkeys(ui.ctx(), super::shortcuts::Bindings::shipped());
                app.pending_reset_confirm = false;
            }
        });
    });
}

/// A page's small heading over a group.
fn group_heading(ui: &mut egui::Ui, text: &str) {
    ui.add_space(18.0);
    ui.horizontal(|ui| {
        // Lines up with the rows' text: the card's 16 px padding and 1 px edge.
        ui.add_space(17.0);
        ui.label(RichText::new(text).size(theme::FONT_SIZE_SECONDARY).color(theme::TEXT_SECONDARY));
    });
    ui.add_space(4.0);
}

/// A group of rows in one dim rounded box.
fn group<R>(ui: &mut egui::Ui, body: impl FnOnce(&mut Rows<'_>) -> R) -> R {
    egui::Frame::NONE
        .fill(theme::CARD_FILL)
        .stroke(egui::Stroke::new(1.0, theme::CARD_EDGE))
        .corner_radius(14.0)
        .inner_margin(egui::Margin::symmetric(16, 4))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            body(&mut Rows { ui, first: true })
        })
        .inner
}

/// The rows of a `group`, a hairline between each.
struct Rows<'a> {
    ui: &'a mut egui::Ui,
    first: bool,
}

impl Rows<'_> {
    /// A row: its title over a one-line description on the left, the
    /// control `control` draws on the right.
    fn row<R>(&mut self, title: &str, description: &str, control: impl FnOnce(&mut egui::Ui) -> R) -> R {
        let ui = &mut *self.ui;
        if !std::mem::replace(&mut self.first, false) {
            let y = ui.cursor().top();
            ui.painter().hline(ui.max_rect().x_range(), y, egui::Stroke::new(1.0, theme::CARD_EDGE));
        }
        // The control sits flush right but lays out left to right, so its
        // width comes from the last frame: a right-to-left layout would
        // reverse the parts of a stepper or a segmented choice.
        let width_id = ui.id().with(("settings_row", title));
        let control_width = ui.data(|d| d.get_temp::<f32>(width_id)).unwrap_or(CONTROL_GUESS);
        let (out, drawn) = ui
            .horizontal(|ui| {
                ui.set_min_height(46.0);
                ui.vertical(|ui| {
                    ui.set_width((ui.available_width() - control_width - theme::SPACE_MD).max(120.0));
                    ui.add_space(6.0);
                    ui.label(RichText::new(title).size(theme::FONT_SIZE_BODY).color(theme::TEXT_PRIMARY));
                    if !description.is_empty() {
                        ui.add(egui::Label::new(RichText::new(description).size(theme::FONT_SIZE_SECONDARY).color(theme::TEXT_SECONDARY)).wrap());
                    }
                    ui.add_space(6.0);
                });
                ui.add_space((ui.available_width() - control_width).max(0.0));
                let r = ui.horizontal(control);
                (r.inner, r.response.rect.width())
            })
            .inner;
        if (drawn - control_width).abs() > 0.5 {
            ui.data_mut(|d| d.insert_temp(width_id, drawn));
            ui.ctx().request_repaint();
        }
        out
    }

    fn ui(&mut self) -> &mut egui::Ui {
        self.ui
    }
}

/// A stepper: − value + in one dim pill.
fn stepper(ui: &mut egui::Ui, name: &str, value: &mut usize, range: std::ops::RangeInclusive<usize>) {
    egui::Frame::NONE.fill(theme::FIELD_FILL).corner_radius(theme::PILL_RADIUS).inner_margin(3.0).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            let step = |ui: &mut egui::Ui, enabled: bool, glyph: &str, tip: &str| {
                let b = egui::Button::new(RichText::new(glyph).size(theme::FONT_SIZE_HEADING).color(theme::TEXT_PRIMARY))
                    .frame(false)
                    .min_size(egui::vec2(26.0, 26.0));
                super::chip::named(ui.add_enabled(enabled, b), tip).clicked()
            };
            if step(ui, *value > *range.start(), "\u{2212}", "Fewer") {
                *value -= 1;
            }
            let (rect, _) = ui.allocate_exact_size(egui::vec2(26.0, 26.0), egui::Sense::hover());
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, value.to_string(), egui::FontId::proportional(theme::FONT_SIZE_BODY), theme::TEXT_PRIMARY);
            if step(ui, *value < *range.end(), "+", "More") {
                *value += 1;
            }
        });
    })
    .response
    .on_hover_text(name);
}

fn jobs_hint(on_cpu: bool, max_jobs: usize) -> String {
    if on_cpu {
        format!("Images at once. 1\u{2013}{max_jobs}. On the CPU only {} run several; the other models take one at a time, since one run already uses every core.", models_running_several_on_cpu())
    } else {
        format!("Images at once. 1\u{2013}{max_jobs}; on GPU 1\u{2013}2 is optimal.")
    }
}

/// "Silueta and U2Net": the registry's models that gain from parallel
/// jobs on the CPU, built once.
fn models_running_several_on_cpu() -> &'static str {
    static NAMES: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    NAMES.get_or_init(|| {
        let names: Vec<_> = prunr_models::REGISTRY.iter().filter(|d| d.parallel_jobs_pay_on_cpu).map(|d| d.display_name).collect();
        match names.split_last() {
            Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
            _ => names.concat(),
        }
    })
}

fn render_page_performance(ui: &mut egui::Ui, settings: &mut Settings, sd_installed: bool) {
    use crate::gui::settings::{ComputeMode, SdMemoryGuard};
    use super::chip::Choice;
    group_heading(ui, "Processing");
    let max_jobs = settings.max_jobs();
    let jobs_hint = jobs_hint(settings.runs_on_host_cpu(), max_jobs);
    let gpu = settings.is_gpu();
    group(ui, |rows| {
        rows.row("Parallel jobs", &jobs_hint, |ui| stepper(ui, "Parallel jobs", &mut settings.parallel_jobs, 1..=max_jobs));
        if gpu {
            rows.row("Processor", "Auto runs each model on the accelerator when it supports one.", |ui| {
                super::chip::choice_buttons(ui, &[
                    Choice { value: ComputeMode::Auto, name: "Auto", description: "Each model on the accelerator when it supports one, else the CPU", enabled: true },
                    Choice {
                        value: ComputeMode::CpuOnly,
                        name: "CPU only",
                        description: if sd_installed { "Every model on the CPU. The Stable Diffusion eraser needs the accelerator" } else { "Every model on the CPU" },
                        enabled: true,
                    },
                ], &mut settings.processor);
            });
        }
    });
    if !sd_installed {
        return;
    }
    group_heading(ui, "Stable Diffusion eraser");
    group(ui, |rows| {
        rows.row("Keep it loaded", "Faster repeated erases. Holds about 16 GB of RAM while the app is open.", |ui| {
            super::chip::switch(ui, "Keep the Stable Diffusion eraser loaded", &mut settings.keep_sd_loaded);
        });
        rows.row("Memory guard", "How much memory to leave free for other apps while it runs.", |ui| {
            super::chip::choice_buttons(ui, &[
                Choice { value: SdMemoryGuard::Relaxed, name: "Relaxed", description: "Leaves about 1 GB free for other apps. Fits the Stable Diffusion eraser on a tight machine", enabled: true },
                Choice { value: SdMemoryGuard::Balanced, name: "Balanced", description: "Leaves about 2 GB free for other apps", enabled: true },
                Choice { value: SdMemoryGuard::Strict, name: "Strict", description: "Leaves about 4 GB free, for when other apps grab memory while it runs", enabled: true },
            ], &mut settings.sd_memory);
        });
    });
}

pub(crate) fn render_page_hardware(ui: &mut egui::Ui, ctx: &HardwareSectionContext) -> Option<HardwareSectionIntent> {
    use crate::hardware;
    let mut intent = None;
    let p = hardware::profile();
    let avail_ram = hardware::available_ram_bytes_throttled();

    group_heading(ui, "This computer");
    group(ui, |rows| {
        let ui = rows.ui();
        ui.add_space(4.0);
        let kv = |ui: &mut egui::Ui, key: &str, value: &str| {
            ui.horizontal(|ui| {
                ui.set_min_height(26.0);
                ui.label(RichText::new(key).size(theme::FONT_SIZE_BODY).color(theme::TEXT_SECONDARY));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(value).size(theme::FONT_SIZE_BODY).color(theme::TEXT_PRIMARY));
                });
            });
        };
        kv(ui, "CPU", &crate::hardware::cpu_label(p.cpu_vendor, &p.cpu_brand));
        let gpu_label = match (p.dgpu, p.igpu) {
            (Some(d), Some(i)) => format!("{d}, {i} (iGPU)"),
            (Some(d), None) => format!("{d}"),
            (None, Some(i)) => format!("{i} (iGPU)"),
            (None, None) => "none detected".to_string(),
        };
        kv(ui, "GPU", &gpu_label);
        kv(ui, "Accelerator", &ctx.active_provider);
        let total_ram = hardware::total_ram_bytes();
        kv(ui, "RAM", &format!("{:.1} / {:.1} GB free", avail_ram as f64 / 1e9, total_ram as f64 / 1e9));

        // The same rule as the gate an erase passes, with the user's margin.
        use prunr_core::inpaint_sd::{sd_ram_fit, SdRamFit};
        for (name, id) in ctx.installs.sd_erasers() {
            let (color, fit) = match sd_ram_fit(id, avail_ram / (1024 * 1024), ctx.sd_margin_mb) {
                SdRamFit::Full => (Color32::from_rgb(0x6c, 0xd1, 0x6c), "enough memory free"),
                SdRamFit::TilesOnly => (
                    theme::ADVISORY_TEXT,
                    "runs, but tall regions are filled in tiles. Close other apps for one pass",
                ),
                SdRamFit::Refused => (
                    Color32::from_rgb(0xd8, 0x6e, 0x6e),
                    "not enough memory free. Close other apps or use Big-LaMa",
                ),
            };
            ui.horizontal(|ui| {
                ui.set_min_height(28.0);
                let (dot, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                ui.painter().circle_filled(dot.center(), 4.0, color);
                ui.label(RichText::new(format!("{name}: {fit}")).size(theme::FONT_SIZE_SECONDARY).color(color));
            });
        }
        ui.add_space(4.0);
    });

    group_heading(ui, "Acceleration");
    let recommended = p.recommended_runtime();
    group(ui, |rows| {
        if ctx.installs.built_in_gpu {
            rows.row("DirectML Runtime", "Built in. Runs the models on any DirectX 12 graphics card, and on the processor without one.", |_| {});
        }
        for &rt in ctx.offered {
            runtime_row(rows, ctx, rt, recommended == Some(rt), &mut intent);
        }
        if RuntimeChoice::ALL.iter().filter(|&&c| ctx.installs.can_load(c)).count() > 1 {
            rows.row("Runtime", "Which one Prunr uses. A change takes effect when you restart Prunr.", |ui| {
                let mut chosen = ctx.installs.next_runtime;
                let choices = RuntimeChoice::ALL.map(|choice| super::chip::Choice {
                    value: Some(choice),
                    name: choice.name(),
                    description: choice.description(),
                    enabled: ctx.installs.can_load(choice),
                });
                if super::chip::choice_buttons(ui, &choices, &mut chosen) {
                    intent = chosen.map(HardwareSectionIntent::PreferRuntime);
                }
            });
        }
        rows.row(
            "Compiled-model cache",
            "Models are compiled for your graphics hardware the first time they run. Clear it if a compiled file is corrupt or to free disk space.",
            |ui| {
                if button(ui, ButtonKind::Secondary, "Clear").clicked() {
                    intent = Some(HardwareSectionIntent::ClearCompiledCache);
                }
            },
        );
        rows.row(
            "GPU failures",
            "A model that failed on the graphics card runs on the processor until the graphics driver or Prunr changes. Forget them to try the graphics card again.",
            |ui| {
                if super::chip::named(button(ui, ButtonKind::Secondary, "Forget"), "Forget GPU failures").clicked() {
                    intent = Some(HardwareSectionIntent::ForgetGpuFailures);
                }
            },
        );
    });
    intent
}

/// One runtime: its state, and Install, Uninstall or Cancel. One install
/// runs at a time.
fn runtime_row(rows: &mut Rows<'_>, ctx: &HardwareSectionContext, rt: RuntimeId, recommended: bool, intent: &mut Option<HardwareSectionIntent>) {
    let installed = ctx.installs.has_runtime(rt);
    let installing_this = ctx.installing.as_ref().filter(|(r, _)| *r == rt);
    let status = match installing_this {
        Some((_, progress)) => progress.as_str(),
        None if installed => "Installed.",
        None => rt.not_installed(recommended),
    };
    rows.row(rt.display_name(), status, |ui| {
        if installing_this.is_some() {
            if button(ui, ButtonKind::Secondary, "Cancel").clicked() {
                *intent = Some(HardwareSectionIntent::CancelInstall);
            }
        } else if installed {
            if super::chip::named(button(ui, ButtonKind::Destructive, "Uninstall"), rt.uninstall_label()).clicked() {
                *intent = Some(HardwareSectionIntent::Uninstall(rt));
            }
        } else {
            let resp = ui.add_enabled_ui(ctx.installing.is_none(), |ui| button(ui, ButtonKind::Primary, "Install")).inner;
            if super::chip::named(resp, rt.install_button_label()).clicked() {
                *intent = Some(HardwareSectionIntent::StartInstall(rt));
            }
        }
    });
}

fn render_page_images(ui: &mut egui::Ui, settings: &mut Settings) {
    group_heading(ui, "When opening images");
    let preset_names = super::preset_dropdown::all_preset_names(settings);
    let current = settings.default_preset.clone();
    group(ui, |rows| {
        rows.row("Auto-process on import", "Run Process automatically as each image arrives.", |ui| {
            super::chip::switch(ui, "Auto-process on import", &mut settings.auto_process_on_import);
        });
        rows.row("Default preset", "New images inherit this preset. The toolbar's reset restores its values.", |ui| {
            ui.spacing_mut().interact_size.y = theme::CHIP_HEIGHT;
            let combo = egui::ComboBox::from_id_salt("default_preset")
                .selected_text(RichText::new(&current).color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY))
                .show_ui(ui, |ui| {
                    for name in &preset_names {
                        let selected = settings.default_preset == *name;
                        if ui.selectable_label(selected, name).clicked() {
                            settings.default_preset = name.clone();
                        }
                    }
                });
            super::chip::named(combo.response, "Default preset");
        });
    });
}

fn render_page_editing(ui: &mut egui::Ui, settings: &mut Settings) {
    group_heading(ui, "Canvas");
    group(ui, |rows| {
        rows.row("Dark checkerboard", "Dark tones for the transparency pattern, for light results.", |ui| {
            super::chip::switch(ui, "Dark checkerboard", &mut settings.dark_checker);
        });
        rows.row("Selection fill", "How strongly a selection is tinted on the image.", |ui| {
            ui.spacing_mut().slider_width = 140.0;
            super::chip::bare_slider_f32(ui, "Selection fill", &mut settings.selection_fill, 0.0..=1.0, super::fmt::percent);
        });
        rows.row("Outline the selection", "", |ui| {
            super::chip::switch(ui, "Outline the selection", &mut settings.selection_outline);
        });
    });
    group_heading(ui, "Motion");
    group(ui, |rows| {
        rows.row("Animations", "Reveals, glows and moving highlights. Off shows each result at once.", |ui| {
            super::chip::switch(ui, "Animations", &mut settings.animations);
        });
    });
    group_heading(ui, "Toolbar");
    group(ui, |rows| {
        rows.row("Auto-hide adjustments", "Collapse the toolbar when the cursor leaves it. Toggle with Shift+H.", |ui| {
            super::chip::switch(ui, "Auto-hide adjustments toolbar", &mut settings.auto_hide_adjustments);
        });
    });
}

/// One row per action: label, the key slots, Reset. Clicking a slot
/// starts a capture that `PrunrApp::handle_hotkey_capture` completes on
/// the next key press.
fn render_tab_hotkeys(ui: &mut egui::Ui, app: &mut PrunrApp) {
    use super::shortcuts::{self, Bindings, SHORTCUTS, SLOTS};

    let current = Arc::clone(&app.bindings);
    let mut reset: Option<Bindings> = None;

    hint(ui, "Click a key to change it, then press the new keys. Escape cancels, Backspace clears.");
    ui.add_space(theme::SPACE_SM);
    group(ui, |rows| {
    let ui = rows.ui();
    egui::Grid::new("hotkeys_grid")
        .num_columns(2 + SLOTS)
        .spacing([theme::SPACE_MD, theme::SPACE_XS])
        .show(ui, |ui| {
            for s in SHORTCUTS {
                let action = s.action;
                let rebindable = shortcuts::rebindable(action);
                ui.label(RichText::new(s.label).color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
                let texts = current.chord_texts(action);
                for slot in 0..SLOTS {
                    let waiting = app.hotkey_capture == Some((action, slot));
                    let text: &str = if waiting {
                        "Press keys\u{2026}"
                    } else {
                        texts.get(slot).map_or("\u{2014}", |t| t)
                    };
                    let color = if waiting { theme::ACCENT } else if rebindable { theme::TEXT_PRIMARY } else { theme::TEXT_HINT };
                    let label = RichText::new(text).monospace().size(theme::FONT_SIZE_SECONDARY).color(color);
                    if rebindable {
                        let slot_button = egui::Button::new(label).min_size(egui::vec2(theme::HOTKEY_SLOT_WIDTH, theme::CHIP_HEIGHT));
                        if ui.add(slot_button).clicked() {
                            app.hotkey_capture = Some((action, slot));
                        }
                    } else {
                        ui.label(label);
                    }
                }
                if rebindable && !current.is_default(action) {
                    if button(ui, ButtonKind::Secondary, "Reset").clicked() {
                        let mut next = (*current).clone();
                        next.reset(action);
                        reset = Some(next);
                    }
                } else {
                    ui.label("");
                }
                ui.end_row();
            }
        });
    });
    group_heading(ui, "Mouse");
    group(ui, |rows| {
        let ui = rows.ui();
        egui::Grid::new("gestures_grid")
            .num_columns(2)
            .spacing([theme::SPACE_LG, theme::SPACE_SM])
            .show(ui, |ui| {
                for (gesture, what) in shortcuts::gestures() {
                    kv_row(ui, gesture, what, theme::TEXT_PRIMARY);
                }
            });
    });
    ui.add_space(theme::SPACE_MD);
    if app.settings.hotkeys.is_empty() {
        ui.label(RichText::new("Copy and Escape are fixed.").color(theme::TEXT_HINT).size(theme::FONT_SIZE_SECONDARY));
    } else if button(ui, ButtonKind::Secondary, "Reset all hotkeys").clicked() {
        reset = Some(Bindings::shipped());
    }
    if let Some(next) = reset {
        app.apply_hotkeys(ui.ctx(), next);
    }
}

// ── Model Credits tab ────────────────────────────────────────────────────────

/// Sort installed model descriptors for the Model Credits tab.
/// Primary: `attribution_required == true` first (CC-BY at top for
/// legal-compliance scannability). Secondary: alphabetical by `display_name`.
fn sort_for_model_credits<'a>(
    entries: &[&'a prunr_models::ModelDescriptor],
) -> Vec<&'a prunr_models::ModelDescriptor> {
    let mut out = entries.to_vec();
    out.sort_by(|a, b| {
        b.attribution_required
            .cmp(&a.attribution_required)
            .then_with(|| a.display_name.cmp(b.display_name))
    });
    out
}

fn render_tab_model_credits(ui: &mut egui::Ui) {
    super::hint(ui, "All installed models, their authors, and licenses.");
    ui.add_space(theme::SPACE_SM);

    let installed_raw: Vec<&prunr_models::ModelDescriptor> = prunr_models::REGISTRY
        .iter()
        .filter(|d| prunr_models::is_available(d.id))
        .collect();
    let installed = sort_for_model_credits(&installed_raw);

    if installed.is_empty() {
        ui.vertical_centered(|ui| {
            ui.add_space(theme::SPACE_LG);
            ui.label(
                RichText::new("No models installed yet.")
                    .color(theme::TEXT_SECONDARY),
            );
            super::hint(ui, "Get models in the Model Store.");
        });
        return;
    }

    egui::ScrollArea::vertical().show(ui, |ui| {
        group(ui, |rows| {
            let ui = rows.ui();
            for (i, descriptor) in installed.iter().enumerate() {
                if i > 0 {
                    ui.separator();
                }
                render_model_credit_row(ui, descriptor);
            }
            ui.add_space(theme::SPACE_SM);
        });
    });
}

fn render_model_credit_row(ui: &mut egui::Ui, descriptor: &prunr_models::ModelDescriptor) {
    let prunr_models::LicenseInfo { author, license: license_name, source_url, .. } = descriptor.license();

    ui.add_space(theme::SPACE_SM);
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(descriptor.display_name)
                .size(theme::FONT_SIZE_BODY)
                .strong()
                .color(theme::TEXT_PRIMARY),
        );
        if descriptor.attribution_required {
            ui.label(
                RichText::new(" CC-BY")
                    .size(theme::FONT_SIZE_SECONDARY)
                    .color(theme::ACCENT_RING),
            );
        }
    });
    ui.add_space(theme::SPACE_XS);
    ui.horizontal(|ui| {
        ui.label(RichText::new("Author:").size(theme::FONT_SIZE_SECONDARY).color(theme::TEXT_SECONDARY));
        ui.label(RichText::new(author).size(theme::FONT_SIZE_SECONDARY).color(theme::TEXT_SECONDARY));
    });
    ui.horizontal(|ui| {
        ui.label(RichText::new("License:").size(theme::FONT_SIZE_SECONDARY).color(theme::TEXT_SECONDARY));
        ui.label(RichText::new(license_name).size(theme::FONT_SIZE_SECONDARY).color(theme::TEXT_SECONDARY));
        if !source_url.is_empty() {
            ui.hyperlink_to("↗ source", source_url);
        }
    });
}

#[cfg(test)]
mod model_credits_tests {
    use super::sort_for_model_credits;
    use prunr_models::{ModelCategory, ModelDescriptor, ModelId, ModelSource, GpuRequirement};

    fn make_descriptor(
        id: ModelId,
        display_name: &'static str,
        attribution_required: bool,
    ) -> ModelDescriptor {
        ModelDescriptor {
            id,
            display_name,
            description: "",
            category: ModelCategory::Segmentation,
            source: ModelSource::Bundled {
                license: prunr_models::LicenseInfo { author: "test", license: "MIT", license_url: "", source_url: "" },
            },
            version: "1.0.0",
            gpu: GpuRequirement::None,
            incompatible_eps: &[],
            working_set_mb: 100,
            parallel_jobs_pay_on_cpu: false,
            tile_size_multiple: None,
            recommended_tile: None,
            attribution_required,
            upscale: None,
        }
    }

    #[test]
    fn sort_puts_cc_by_first_then_alphabetical() {
        let z_ccby = make_descriptor(ModelId::Silueta, "Z model", true);
        let a_bsd = make_descriptor(ModelId::U2net, "A model", false);
        let b_bsd = make_descriptor(ModelId::BiRefNetLite, "B model", false);

        let entries = vec![&a_bsd, &b_bsd, &z_ccby];
        let sorted = sort_for_model_credits(&entries);

        assert_eq!(sorted[0].display_name, "Z model", "CC-BY must come first");
        assert_eq!(sorted[1].display_name, "A model", "alphabetical within non-CC-BY");
        assert_eq!(sorted[2].display_name, "B model", "alphabetical within non-CC-BY");
    }

    #[test]
    fn sort_multiple_cc_by_sorted_alphabetically() {
        let z_ccby = make_descriptor(ModelId::Silueta, "Z CC model", true);
        let a_ccby = make_descriptor(ModelId::U2net, "A CC model", true);
        let m_bsd = make_descriptor(ModelId::BiRefNetLite, "M BSD model", false);

        let entries = vec![&m_bsd, &z_ccby, &a_ccby];
        let sorted = sort_for_model_credits(&entries);

        assert_eq!(sorted[0].display_name, "A CC model");
        assert_eq!(sorted[1].display_name, "Z CC model");
        assert_eq!(sorted[2].display_name, "M BSD model");
    }

    #[test]
    fn sort_empty_input_returns_empty() {
        let sorted = sort_for_model_credits(&[]);
        assert!(sorted.is_empty());
    }
}

#[cfg(test)]
mod jobs_hint_tests {
    use super::jobs_hint;

    #[test]
    fn the_cpu_jobs_hint_names_the_models_that_run_several_at_once() {
        let hint = jobs_hint(true, 8);
        for d in prunr_models::REGISTRY.iter().filter(|d| d.parallel_jobs_pay_on_cpu) {
            assert!(hint.contains(d.display_name), "{hint:?} names {}", d.display_name);
        }
        assert!(!hint.contains("BiRefNet"), "{hint:?} reads as if only BiRefNet took one at a time");
    }
}
