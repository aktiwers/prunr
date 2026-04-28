//! Settings modal — three tabs:
//! - **General**: hardware + performance knobs that change *what runs*
//! - **Behavior**: toggles that change *how editing feels*
//! - **Hotkeys**: read-only shortcut reference (rebinding TBD)
//!
//! Per-image knobs (gamma, threshold, line mode, …) live on the persistent
//! adjustments toolbar (rows 2 + 3), not here.

use egui::{Align2, RichText};

use crate::gui::app::PrunrApp;
use crate::gui::settings::Settings;
use crate::gui::theme;

use super::{hint, section_heading};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsTab {
    General,
    Behavior,
    Hotkeys,
}

impl SettingsTab {
    fn label(self) -> &'static str {
        match self {
            Self::General => "General",
            Self::Behavior => "Behavior",
            Self::Hotkeys => "Hotkeys",
        }
    }
    const ALL: [SettingsTab; 3] = [Self::General, Self::Behavior, Self::Hotkeys];

    /// Parse a label-style name into a tab. Used by PRUNR_OPEN_TAB.
    pub fn from_debug_name(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|t| t.label() == s)
    }
}

/// Read-only snapshot of runtime-install state passed into the General
/// tab so its renderer doesn't need a `&mut PrunrApp`.
pub(crate) struct HardwareSectionContext {
    pub openvino_installed: bool,
    pub install_in_progress: bool,
    pub install_status_text: Option<String>,
}

/// View-layer intent the General tab returns so platform side effects
/// (rfd, runtime install) live on the orchestrator, not the view.
#[derive(Debug, Clone, Copy)]
pub(crate) enum HardwareSectionIntent {
    StartInstall(crate::runtime_install::RuntimeId),
    CancelInstall,
    Uninstall(crate::runtime_install::RuntimeId),
}

/// Slider row: label left, slider fills middle, value right.
fn slider_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    value_text: &str,
    step: Option<f64>,
) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
        let avail = ui.available_width() - 52.0;
        let mut slider = egui::Slider::new(value, range).show_value(false);
        if let Some(s) = step { slider = slider.step_by(s); }
        ui.add_sized([avail.max(100.0), 18.0], slider);
        ui.label(RichText::new(value_text).monospace().size(theme::FONT_SIZE_MONO).color(theme::TEXT_PRIMARY));
    });
}

pub fn render(ctx: &egui::Context, app: &mut PrunrApp) {
    theme::draw_modal_backdrop(ctx, "settings_backdrop");

    let mut open = true;
    let window_response = egui::Window::new("Settings")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([theme::SETTINGS_DIALOG_WIDTH, 620.0])
        .frame(theme::overlay_frame())
        .show(ctx, |ui| {
            apply_modal_visuals(ui);

            render_tab_strip(ui, &mut app.settings_tab);
            ui.add_space(theme::SPACE_SM);

            // Tab content scrolls inside its own region so the footer
            // (Reset row) doesn't overlap long content.
            const FOOTER_RESERVED: f32 = 56.0;
            let scroll_height = (ui.available_height() - FOOTER_RESERVED).max(0.0);
            let hardware_intent = egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .max_height(scroll_height)
                .show(ui, |ui| {
                    match app.settings_tab {
                        SettingsTab::General => {
                            let ctx = HardwareSectionContext {
                                openvino_installed: app.hardware_install_cache.openvino,
                                install_in_progress: app.runtime_install.is_some(),
                                install_status_text: app.runtime_install.as_ref()
                                    .map(|p| p.last_event.status_text()),
                            };
                            render_tab_general(ui, &mut app.settings, &ctx)
                        }
                        SettingsTab::Behavior => { render_tab_behavior(ui, &mut app.settings); None }
                        SettingsTab::Hotkeys => { render_tab_hotkeys(ui); None }
                    }
                })
                .inner;
            if let Some(intent) = hardware_intent {
                dispatch_hardware_intent(app, intent);
            }

            ui.separator();
            render_modal_footer(ui, app);
        });

    let now = ctx.input(|i| i.time);
    let debounce_passed = (now - app.settings_opened_at) > theme::MODAL_BACKDROP_DEBOUNCE_SECS;
    let close_via_backdrop = debounce_passed && theme::backdrop_clicked(ctx, &window_response);

    if !open || close_via_backdrop {
        app.close_settings(ctx);
    }
}

/// Tighten egui's default visuals for the modal: subdued borders + the
/// button-fill colors used elsewhere in the app.
fn apply_modal_visuals(ui: &mut egui::Ui) {
    let vis = ui.visuals_mut();
    vis.widgets.inactive.bg_stroke =
        egui::Stroke::new(theme::STROKE_DEFAULT, egui::Color32::from_rgb(0x60, 0x60, 0x60));
    vis.widgets.hovered.bg_stroke =
        egui::Stroke::new(theme::STROKE_DEFAULT, egui::Color32::from_rgb(0x80, 0x80, 0x80));
    vis.widgets.inactive.bg_fill = theme::WIDGET_INACTIVE_BG;
    vis.widgets.inactive.fg_stroke =
        egui::Stroke::new(theme::STROKE_DEFAULT, theme::TEXT_PRIMARY);
    vis.widgets.hovered.bg_fill = theme::WIDGET_HOVER_BG;
}

/// Footer = Reset-to-defaults row. When `pending_reset_confirm` is set,
/// shows an inline "Reset everything? [Reset] [Cancel]" instead of the
/// initial trigger button. No separate modal; the destructive scope is
/// spelled out so users can't accidentally nuke their config.
fn render_modal_footer(ui: &mut egui::Ui, app: &mut PrunrApp) {
    ui.horizontal(|ui| {
        if app.pending_reset_confirm {
            ui.label(RichText::new("Reset all settings to defaults?")
                .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button(RichText::new("Cancel")
                    .color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_BODY)).clicked()
                {
                    app.pending_reset_confirm = false;
                }
                if ui.small_button(RichText::new("Reset")
                    .color(theme::DESTRUCTIVE).size(theme::FONT_SIZE_BODY).strong()).clicked()
                {
                    reset_settings_to_defaults(&mut app.settings);
                    app.pending_reset_confirm = false;
                }
            });
        } else {
            ui.label(RichText::new("Auto-saved")
                .color(theme::TEXT_HINT).size(theme::FONT_SIZE_MONO));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button(RichText::new("Reset to defaults")
                    .color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO)).clicked()
                {
                    app.pending_reset_confirm = true;
                }
            });
        }
    });
}

/// Reset Settings to its `Default` while preserving identity-bearing fields
/// (active backend probe, presets the user authored, the chosen default
/// preset). `parallel_jobs` re-derives from the current backend so it
/// snaps to a sane value for whatever GPU/CPU is detected.
fn reset_settings_to_defaults(settings: &mut Settings) {
    let backend = settings.active_backend.clone();
    let presets = std::mem::take(&mut settings.presets);
    let default_preset = settings.default_preset.clone();
    *settings = Settings::default();
    settings.active_backend = backend;
    settings.parallel_jobs = settings.default_jobs();
    settings.presets = presets;
    settings.default_preset = default_preset;
}

fn dispatch_hardware_intent(app: &mut PrunrApp, intent: HardwareSectionIntent) {
    use crate::runtime_install::{InstallEvent, start_install};
    match intent {
        HardwareSectionIntent::StartInstall(rt) => {
            let h = start_install(rt);
            app.runtime_install = Some(crate::gui::app::RuntimeInstallProgress {
                runtime: rt,
                rx: h.events,
                cancel: h.cancel,
                last_event: InstallEvent::Preparing,
            });
        }
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
            app.hardware_install_cache =
                crate::gui::hardware_cache::HardwareInstallCache::refresh();
        }
    }
}

fn render_hardware_section(
    ui: &mut egui::Ui,
    ctx: &HardwareSectionContext,
) -> Option<HardwareSectionIntent> {
    use crate::hardware;
    use crate::runtime_install::RuntimeId;
    let mut intent = None;
    section_heading(ui, "Hardware");

    let p = hardware::profile();
    // Two-column key/value grid for scannable hardware facts.
    egui::Grid::new("hardware_grid")
        .num_columns(2)
        .spacing([theme::SPACE_LG, theme::SPACE_XS])
        .show(ui, |ui| {
            hw_row(ui, "CPU", &format!("{} ({})", p.cpu_vendor, p.cpu_brand));
            let gpu_label = match (p.dgpu, p.igpu) {
                (Some(d), Some(i)) => format!("{d}, {i} (iGPU)"),
                (Some(d), None) => format!("{d}"),
                (None, Some(i)) => format!("{i} (iGPU)"),
                (None, None) => "none detected".to_string(),
            };
            hw_row(ui, "GPU", &gpu_label);
            let active_provider = prunr_core::OrtEngine::detect_active_provider();
            hw_row(ui, "Active EP", &active_provider);
            let total_ram = hardware::total_ram_bytes();
            let avail_ram = hardware::available_ram_bytes_now();
            hw_row(ui, "RAM",
                &format!("{:.1} / {:.1} GB free",
                    avail_ram as f64 / 1e9, total_ram as f64 / 1e9));
        });

    // SD-specific verdict on its own line, color-coded so it pops.
    let sd_working_set: u64 = 7 * 1024 * 1024 * 1024;
    let avail_ram = hardware::available_ram_bytes_now();
    let verdict = hardware::ram_verdict(sd_working_set, avail_ram);
    let (color, text) = match verdict {
        hardware::RamVerdict::Comfortable => (
            egui::Color32::from_rgb(0x6c, 0xd1, 0x6c),
            "SD 1.5: comfortable headroom",
        ),
        hardware::RamVerdict::Tight => (
            egui::Color32::from_rgb(0xe1, 0xb3, 0x4e),
            "SD 1.5: tight — close other apps before running",
        ),
        hardware::RamVerdict::Insufficient => (
            egui::Color32::from_rgb(0xd8, 0x6e, 0x6e),
            "SD 1.5: insufficient — try Big-LaMa instead",
        ),
    };
    ui.add_space(theme::SPACE_XS);
    ui.label(RichText::new(text).color(color).size(theme::FONT_SIZE_MONO));
    ui.add_space(theme::SPACE_SM);

    let rt = RuntimeId::OpenVino;
    let installed = ctx.openvino_installed;
    let status = match (ctx.install_in_progress, ctx.install_status_text.as_deref()) {
        (true, Some(s)) => s.to_string(),
        _ if installed => "Installed".to_string(),
        _ => format!("Not installed ({} MB)", rt.approx_download_mb()),
    };

    ui.horizontal(|ui| {
        ui.label(RichText::new(rt.display_name())
            .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
        ui.label(RichText::new(status)
            .color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ctx.install_in_progress {
                if ui.button(RichText::new("Cancel")
                    .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY)).clicked() {
                    intent = Some(HardwareSectionIntent::CancelInstall);
                }
            } else if installed {
                if ui.button(RichText::new("Uninstall")
                    .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY)).clicked() {
                    intent = Some(HardwareSectionIntent::Uninstall(rt));
                }
            } else if ui.button(RichText::new("Install")
                .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY)).clicked() {
                intent = Some(HardwareSectionIntent::StartInstall(rt));
            }
        });
    });
    if p.recommends_openvino() && !installed && !ctx.install_in_progress {
        hint(ui, "Recommended for Intel hardware — 2-3× faster inference, plus iGPU acceleration for SD inpaint.");
    }

    ui.add_space(theme::SPACE_MD);
    intent
}

/// One key/value row inside the hardware grid.
fn hw_row(ui: &mut egui::Ui, key: &str, value: &str) {
    ui.label(RichText::new(key)
        .color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));
    ui.label(RichText::new(value)
        .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
    ui.end_row();
}

fn render_tab_strip(ui: &mut egui::Ui, current: &mut SettingsTab) {
    ui.horizontal(|ui| {
        for tab in SettingsTab::ALL.iter().copied() {
            let is_active = *current == tab;
            let label = RichText::new(tab.label())
                .color(if is_active { theme::TEXT_PRIMARY } else { theme::TEXT_SECONDARY })
                .size(theme::FONT_SIZE_BODY)
                .strong();
            if ui.selectable_label(is_active, label).clicked() {
                *current = tab;
            }
        }
    });
    ui.separator();
}

fn render_tab_general(
    ui: &mut egui::Ui,
    settings: &mut Settings,
    ctx: &HardwareSectionContext,
) -> Option<HardwareSectionIntent> {
    let intent = render_hardware_section(ui, ctx);

    section_heading(ui, "Performance");
    let max_jobs = settings.max_jobs();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Parallel jobs")
            .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.add_enabled(
                settings.parallel_jobs < max_jobs,
                egui::Button::new(RichText::new("+")
                    .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY))
                    .fill(theme::BG_SECONDARY)
                    .min_size(egui::vec2(theme::CHIP_HEIGHT, theme::CHIP_HEIGHT)),
            ).clicked() {
                settings.parallel_jobs += 1;
            }
            ui.label(RichText::new(format!("{}", settings.parallel_jobs))
                .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY).strong());
            if ui.add_enabled(
                settings.parallel_jobs > 1,
                egui::Button::new(RichText::new("\u{2212}")
                    .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY))
                    .fill(theme::BG_SECONDARY)
                    .min_size(egui::vec2(theme::CHIP_HEIGHT, theme::CHIP_HEIGHT)),
            ).clicked() {
                settings.parallel_jobs -= 1;
            }
        });
    });
    let jobs_hint = if settings.is_gpu() {
        format!("Images at once. 1\u{2013}{max_jobs}; on GPU 1\u{2013}2 is optimal.")
    } else {
        format!("Images at once. 1\u{2013}{max_jobs}.")
    };
    hint(ui, &jobs_hint);
    ui.add_space(theme::SPACE_MD);

    render_sd_fast_mode_row(ui, &mut settings.sd_fast_mode);
    ui.add_space(theme::SPACE_MD);

    let has_gpu = !prunr_core::OrtEngine::detect_active_provider().eq_ignore_ascii_case("CPU");
    if has_gpu {
        ui.checkbox(&mut settings.force_cpu, RichText::new("Force CPU this session")
            .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
        hint(ui, "Resets to GPU on next launch. Useful for debugging GPU misbehaviour.");
        ui.add_space(theme::SPACE_MD);
    }

    ui.checkbox(&mut settings.live_preview, RichText::new("Live preview")
        .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
    hint(ui, "Auto-rerun mask and edge tweaks while you drag sliders.");
    intent
}

fn render_sd_fast_mode_row(ui: &mut egui::Ui, user_override: &mut Option<bool>) {
    use crate::hardware;
    let profile = hardware::profile();
    let auto = hardware::sd_fast_mode_auto_default(profile);
    let mut effective = user_override.unwrap_or(auto);

    let prev = effective;
    let resp = ui.checkbox(&mut effective, RichText::new("Fast SD inpaint")
        .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
    resp.on_hover_text(
        "LCM-distilled SD weights — ~5× faster on CPU / Intel iGPU at \
         lower fidelity. Negative-prompt and Guidance grey out (LCM bakes \
         guidance into training)."
    );
    if effective != prev {
        // Setting back to the auto value collapses to None so the toggle
        // keeps tracking hardware changes (e.g. user installs OpenVINO).
        *user_override = if effective == auto { None } else { Some(effective) };
    }

    let mode_label = match user_override {
        None => format!("auto: {}", if auto { "on" } else { "off" }),
        Some(true) => "user: on".to_string(),
        Some(false) => "user: off".to_string(),
    };
    hint(ui, &format!("Trade quality for speed when SD runs on CPU / Intel iGPU. ({mode_label})"));
}

fn render_tab_behavior(ui: &mut egui::Ui, settings: &mut Settings) {
    section_heading(ui, "When opening images");
    ui.checkbox(&mut settings.auto_process_on_import, RichText::new("Auto-process on import")
        .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
    hint(ui, "Run Process automatically as each image arrives.");
    ui.add_space(theme::SPACE_MD);

    section_heading(ui, "Editing");
    ui.checkbox(&mut settings.dark_checker, RichText::new("Dark checkerboard")
        .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
    hint(ui, "Use dark tones for the transparency pattern (helps on light results).");
    ui.add_space(theme::SPACE_SM);

    ui.checkbox(&mut settings.auto_hide_adjustments, RichText::new("Auto-hide adjustments toolbar")
        .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
    hint(ui, "Collapse the toolbar when the cursor leaves it. Toggle with Shift+H.");
    ui.add_space(theme::SPACE_MD);

    section_heading(ui, "Stack passes (chain mode)");
    ui.checkbox(&mut settings.chain_mode, RichText::new("Use last result as input")
        .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
    hint(ui, "Process feeds on the previous output instead of the original — stack effects.");
    if settings.chain_mode {
        ui.add_space(theme::SPACE_SM);
        let mut depth_f32 = settings.history_depth as f32;
        let depth_text = format!("{}", settings.history_depth);
        slider_row(ui, "History depth", &mut depth_f32, 1.0..=50.0, &depth_text, Some(1.0));
        settings.history_depth = depth_f32 as usize;
        hint(ui, "Maximum undo steps per image. Higher = more memory.");
    }
    ui.add_space(theme::SPACE_MD);

    section_heading(ui, "Defaults");
    ui.checkbox(&mut settings.export_split_layers, RichText::new("Split exports into layers")
        .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY))
        .on_hover_text(
            "Drag-out and Save emit subject / lines / mask as separate \
             PNGs — useful for Photoshop / Procreate. On Linux (no \
             drag-out) this only affects Save.");
    hint(ui, "Subject / lines / mask as separate PNGs.");
    ui.add_space(theme::SPACE_SM);

    let preset_names = super::preset_dropdown::all_preset_names(settings);
    let current = settings.default_preset.clone();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Default preset")
            .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
        egui::ComboBox::from_id_salt("default_preset")
            .selected_text(RichText::new(&current).color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY))
            .show_ui(ui, |ui| {
                for name in &preset_names {
                    let selected = settings.default_preset == *name;
                    if ui.selectable_label(selected, name).clicked() {
                        settings.default_preset = name.clone();
                    }
                }
            });
    });
    hint(ui, "New images inherit this preset. The reset button on the toolbar restores its values.");
}

fn render_tab_hotkeys(ui: &mut egui::Ui) {
    super::shortcuts::render_shortcut_grid(ui);
    ui.add_space(theme::SPACE_MD);
    ui.label(RichText::new("Rebinding will land in a future release.")
        .color(theme::TEXT_HINT).size(theme::FONT_SIZE_MONO));
}
