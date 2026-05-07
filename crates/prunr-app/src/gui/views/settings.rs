//! Settings modal — three tabs:
//! - **General**: hardware + performance knobs that change *what runs*
//! - **Behavior**: toggles that change *how editing feels*
//! - **Hotkeys**: read-only shortcut reference (rebinding TBD)
//!
//! Per-image knobs (gamma, threshold, line mode, …) live on the persistent
//! adjustments toolbar (rows 2 + 3), not here.

use egui::RichText;

use crate::gui::app::PrunrApp;
use crate::gui::settings::Settings;
use crate::gui::theme;

use super::{hint, kv_row, section_heading};

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

    /// Parse a label string into a tab. Used by PRUNR_OPEN_TAB.
    pub fn from_label(s: &str) -> Option<Self> {
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
    ClearCompiledCache,
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
    let mut hardware_intent: Option<HardwareSectionIntent> = None;
    let closed = theme::standard_modal_window(
        ctx,
        "settings",
        "Settings",
        [theme::SETTINGS_DIALOG_WIDTH, theme::SETTINGS_DIALOG_HEIGHT],
        |ui| {
            theme::apply_modal_visuals(ui);
            render_tab_strip(ui, &mut app.settings_tab);
            ui.add_space(theme::SPACE_SM);

            // Tab content scrolls inside its own region so the footer
            // (Reset row) doesn't overlap long content.
            let scroll_height = (ui.available_height() - theme::MODAL_FOOTER_RESERVED).max(0.0);
            hardware_intent = egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .max_height(scroll_height)
                .show(ui, |ui| {
                    match app.settings_tab {
                        SettingsTab::General => {
                            let hw_ctx = HardwareSectionContext {
                                openvino_installed: app.hardware_install_cache.openvino,
                                install_in_progress: app.runtime_install.is_some(),
                                install_status_text: app.runtime_install.as_ref()
                                    .map(|p| p.last_event.status_text()),
                            };
                            render_tab_general(ui, &mut app.settings, &hw_ctx)
                        }
                        SettingsTab::Behavior => { render_tab_behavior(ui, &mut app.settings); None }
                        SettingsTab::Hotkeys => { render_tab_hotkeys(ui); None }
                    }
                })
                .inner;

            ui.separator();
            render_modal_footer(ui, app);
        },
    );

    if let Some(intent) = hardware_intent {
        dispatch_hardware_intent(app, intent);
    }
    if closed {
        app.close_settings(ctx);
    }
}

/// Inline confirm-then-reset (no separate modal) so the destructive
/// scope is spelled out before the user can act on it.
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
                    app.settings.reset_preserving_identity();
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
        HardwareSectionIntent::ClearCompiledCache => {
            // Off-thread: a multi-GB rmtree on a slow disk would stall
            // the GUI frame.
            std::thread::spawn(|| {
                let bytes = prunr_core::cache::clear_all();
                tracing::info!(bytes, "cleared compiled-model cache");
            });
            app.toasts.info("Clearing compiled-model cache…");
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
    let avail_ram = hardware::available_ram_bytes_throttled();
    egui::Grid::new("hardware_grid")
        .num_columns(2)
        .spacing([theme::SPACE_LG, theme::SPACE_XS])
        .show(ui, |ui| {
            kv_row(ui, "CPU", &format!("{} ({})", p.cpu_vendor, p.cpu_brand), theme::TEXT_SECONDARY);
            let gpu_label = match (p.dgpu, p.igpu) {
                (Some(d), Some(i)) => format!("{d}, {i} (iGPU)"),
                (Some(d), None) => format!("{d}"),
                (None, Some(i)) => format!("{i} (iGPU)"),
                (None, None) => "none detected".to_string(),
            };
            kv_row(ui, "GPU", &gpu_label, theme::TEXT_SECONDARY);
            let active_provider = prunr_core::OrtEngine::detect_active_provider();
            kv_row(ui, "Active EP", &active_provider, theme::TEXT_SECONDARY);
            let total_ram = hardware::total_ram_bytes();
            kv_row(ui, "RAM",
                &format!("{:.1} / {:.1} GB free",
                    avail_ram as f64 / 1e9, total_ram as f64 / 1e9),
                theme::TEXT_SECONDARY);
        });

    // SD-specific verdict on its own line, color-coded so it pops.
    let sd_working_set: u64 = 7 * 1024 * 1024 * 1024;
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

    ui.horizontal(|ui| {
        ui.label(RichText::new("Compiled-model cache")
            .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button(RichText::new("Clear")
                .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY)).clicked() {
                intent = Some(HardwareSectionIntent::ClearCompiledCache);
            }
        });
    });
    hint(ui, "Wipes per-EP compiled artifacts (CUDA optimized graphs, CoreML mlmodelc). Models recompile on next use. Use if a cached file is corrupt or you want the disk space back.");

    ui.add_space(theme::SPACE_MD);
    intent
}

fn render_ram_safety_margin_row(ui: &mut egui::Ui, value: &mut f32) {
    slider_row(
        ui,
        "SD safety margin",
        value,
        0.0..=8.0,
        &format!("{value:.1} GB"),
        Some(0.5),
    );
    hint(ui, "Free RAM the SD eraser keeps clear on top of the model's working set. Higher = more conservative on systems where other apps spike during inference. Default 2 GB.");
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
    render_ram_safety_margin_row(ui, &mut settings.ram_safety_margin_gb);
    ui.add_space(theme::SPACE_MD);

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
