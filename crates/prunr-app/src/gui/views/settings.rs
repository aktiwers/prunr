//! Settings modal — app-wide config split across General / Appearance /
//! Processing / Defaults / Hotkeys tabs. Per-image knobs live on the
//! persistent adjustments toolbar (rows 2 + 3).

use egui::{Align2, RichText};

use crate::gui::app::PrunrApp;
use crate::gui::settings::Settings;
use crate::gui::theme;

use super::{hint, section_heading};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsTab {
    General,
    Appearance,
    Processing,
    Defaults,
    Hotkeys,
}

impl SettingsTab {
    fn label(self) -> &'static str {
        match self {
            Self::General => "General",
            Self::Appearance => "Appearance",
            Self::Processing => "Processing",
            Self::Defaults => "Defaults",
            Self::Hotkeys => "Hotkeys",
        }
    }
    const ALL: [SettingsTab; 5] = [
        Self::General, Self::Appearance, Self::Processing,
        Self::Defaults, Self::Hotkeys,
    ];
}

/// Slider row: label left, slider fills middle, value right.
fn slider_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    value_text: &str,
    logarithmic: bool,
    step: Option<f64>,
) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(label)
                .color(theme::TEXT_PRIMARY)
                .size(theme::FONT_SIZE_BODY),
        );
        let avail = ui.available_width() - 52.0;
        let mut slider = egui::Slider::new(value, range).show_value(false);
        if logarithmic { slider = slider.logarithmic(true); }
        if let Some(s) = step { slider = slider.step_by(s); }
        ui.add_sized([avail.max(100.0), 18.0], slider);
        ui.label(
            RichText::new(value_text)
                .monospace()
                .size(theme::FONT_SIZE_MONO)
                .color(theme::TEXT_PRIMARY),
        );
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
        .fixed_size([theme::SETTINGS_DIALOG_WIDTH, 520.0])
        .frame(theme::overlay_frame())
        .show(ctx, |ui| {
            {
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

            // Header with Reset-all action on the right.
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("App settings")
                        .size(theme::FONT_SIZE_HEADING)
                        .strong()
                        .color(theme::TEXT_PRIMARY),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button(
                        RichText::new("Reset all")
                            .size(theme::FONT_SIZE_MONO)
                            .color(theme::TEXT_SECONDARY),
                    ).clicked() {
                        // Preserve presets + default pointer across a
                        // Reset of app-wide settings — those are per-user
                        // artifacts that don't belong to "app defaults."
                        let backend = app.settings.active_backend.clone();
                        let presets = std::mem::take(&mut app.settings.presets);
                        let default_preset = app.settings.default_preset.clone();
                        app.settings = Settings::default();
                        app.settings.active_backend = backend;
                        app.settings.parallel_jobs = app.settings.default_jobs();
                        app.settings.presets = presets;
                        app.settings.default_preset = default_preset;
                    }
                });
            });
            ui.separator();
            ui.add_space(theme::SPACE_SM);

            render_tab_strip(ui, &mut app.settings_tab);
            ui.add_space(theme::SPACE_SM);

            match app.settings_tab {
                SettingsTab::General => render_tab_general(ui, app),
                SettingsTab::Appearance => render_tab_appearance(ui, &mut app.settings),
                SettingsTab::Processing => render_tab_processing(ui, &mut app.settings),
                SettingsTab::Defaults => render_tab_defaults(ui, &mut app.settings),
                SettingsTab::Hotkeys => render_tab_hotkeys(ui),
            }

            // Backend info at the bottom
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.label(
                    RichText::new(format!("Backend: {}", app.settings.active_backend))
                        .monospace()
                        .size(theme::FONT_SIZE_MONO)
                        .color(theme::TEXT_HINT),
                );
                ui.separator();
            });
        });

    let now = ctx.input(|i| i.time);
    let debounce_passed = (now - app.settings_opened_at) > theme::MODAL_BACKDROP_DEBOUNCE_SECS;
    let close_via_backdrop = debounce_passed && theme::backdrop_clicked(ctx, &window_response);

    if !open || close_via_backdrop {
        app.close_settings(ctx);
    }
}

fn render_hardware_section(ui: &mut egui::Ui, app: &mut PrunrApp) {
    use crate::hardware;
    use crate::runtime_install::{InstallEvent, RuntimeId, start_install};
    section_heading(ui, "Hardware");

    let p = hardware::profile();
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("CPU: {} ({})", p.cpu_vendor, p.cpu_brand))
            .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
    });
    let gpu_label = match (p.dgpu, p.igpu) {
        (Some(d), Some(i)) => format!("dGPU: {d}, iGPU: {i}"),
        (Some(d), None) => format!("dGPU: {d}"),
        (None, Some(i)) => format!("iGPU: {i}"),
        (None, None) => "GPU: none detected".to_string(),
    };
    ui.label(RichText::new(gpu_label)
        .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));

    let active_provider = prunr_core::OrtEngine::detect_active_provider();
    ui.label(RichText::new(format!("Active EP: {active_provider}"))
        .color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));

    let total_ram = hardware::total_ram_bytes();
    let avail_ram = hardware::available_ram_bytes_now();
    ui.label(RichText::new(format!(
        "RAM: {:.1} / {:.1} GB free",
        avail_ram as f64 / 1e9, total_ram as f64 / 1e9,
    ))
        .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));

    // Per-heavy-model RAM headroom verdict. SD 1.5 is the only model
    // whose working set rivals the user's RAM (~6-10 GB on CPU); for
    // the lighter eraser models the verdict is always comfortable on
    // any system that has SD installed at all, so we skip them here.
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
    ui.label(RichText::new(text).color(color).size(theme::FONT_SIZE_MONO));
    ui.add_space(theme::SPACE_SM);

    // OpenVINO Runtime row. Compute the status string up front so the
    // borrow on `app.runtime_install` doesn't extend into the closure
    // (which needs mutable access for the Install button).
    let rt = RuntimeId::OpenVino;
    let installed = rt.is_installed();
    let status = match app.runtime_install.as_ref().filter(|p| p.runtime == rt) {
        Some(p) => p.last_event.status_text(),
        None if installed => "Installed".to_string(),
        None => format!("Not installed ({} MB)", rt.approx_download_mb()),
    };

    ui.horizontal(|ui| {
        ui.label(RichText::new(rt.display_name())
            .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
        ui.label(RichText::new(status)
            .color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let busy = app.runtime_install.is_some();
            if busy {
                if ui.button(RichText::new("Cancel")
                    .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY)).clicked() {
                    if let Some(p) = app.runtime_install.as_ref() {
                        p.cancel.store(true, std::sync::atomic::Ordering::Release);
                    }
                }
            } else if installed {
                if ui.button(RichText::new("Uninstall")
                    .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY)).clicked() {
                    match crate::runtime_install::uninstall(rt) {
                        Ok(()) => { app.toasts.success(format!("{} removed", rt.display_name())); }
                        Err(e) => { app.toasts.error(format!("Uninstall failed: {e}")); }
                    }
                }
            } else if ui.button(RichText::new("Install")
                .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY)).clicked() {
                let h = start_install(rt);
                app.runtime_install = Some(crate::gui::app::RuntimeInstallProgress {
                    runtime: rt,
                    rx: h.events,
                    cancel: h.cancel,
                    last_event: InstallEvent::Preparing,
                });
            }
        });
    });
    // Hint hidden once installing or installed — it's a recommendation,
    // not a permanent label.
    if p.recommends_openvino() && !installed && app.runtime_install.is_none() {
        hint(ui, "Recommended for your Intel hardware — 2-3× faster inference on most models, plus iGPU acceleration for SD inpaint.");
    }

    ui.add_space(theme::SPACE_MD);
    ui.separator();
    ui.add_space(theme::SPACE_SM);
}


fn render_tab_strip(ui: &mut egui::Ui, current: &mut SettingsTab) {
    ui.horizontal(|ui| {
        for tab in SettingsTab::ALL.iter().copied() {
            let is_active = *current == tab;
            let label = RichText::new(tab.label())
                .color(if is_active { theme::TEXT_PRIMARY } else { theme::TEXT_SECONDARY })
                .size(theme::FONT_SIZE_BODY);
            if ui.selectable_label(is_active, label).clicked() {
                *current = tab;
            }
        }
    });
    ui.separator();
}

fn render_tab_general(ui: &mut egui::Ui, app: &mut PrunrApp) {
    render_hardware_section(ui, app);

    section_heading(ui, "Performance");
    let max_jobs = app.settings.max_jobs();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Parallel jobs")
            .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.add_enabled(
                app.settings.parallel_jobs < max_jobs,
                egui::Button::new(RichText::new("+").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY))
                    .fill(theme::BG_SECONDARY).min_size(egui::vec2(theme::CHIP_HEIGHT, theme::CHIP_HEIGHT)),
            ).clicked() {
                app.settings.parallel_jobs += 1;
            }
            ui.label(RichText::new(format!("{}", app.settings.parallel_jobs))
                .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY).strong());
            if ui.add_enabled(
                app.settings.parallel_jobs > 1,
                egui::Button::new(RichText::new("\u{2212}").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY))
                    .fill(theme::BG_SECONDARY).min_size(egui::vec2(theme::CHIP_HEIGHT, theme::CHIP_HEIGHT)),
            ).clicked() {
                app.settings.parallel_jobs -= 1;
            }
        });
    });
    ui.add_space(-10.0);
    let jobs_hint = if app.settings.is_gpu() {
        format!("Images processed at the same time (1\u{2013}{max_jobs}, GPU: 1\u{2013}2 is optimal)")
    } else {
        format!("Images processed at the same time (1\u{2013}{max_jobs})")
    };
    hint(ui, &jobs_hint);
    ui.add_space(theme::SPACE_MD);

    let has_gpu = !prunr_core::OrtEngine::detect_active_provider().eq_ignore_ascii_case("CPU");
    if has_gpu {
        ui.checkbox(&mut app.settings.force_cpu,
            RichText::new("Force CPU").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
        hint(ui, "Use CPU even when GPU is available (resets each launch).");
        ui.add_space(theme::SPACE_MD);
    }

    ui.checkbox(&mut app.settings.live_preview,
        RichText::new("Live preview").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
    hint(ui, "Auto-rerun mask and edge tweaks as you adjust them.");
    ui.add_space(theme::SPACE_MD);

    render_sd_fast_mode_row(ui, &mut app.settings.sd_fast_mode);
}

fn render_sd_fast_mode_row(ui: &mut egui::Ui, user_override: &mut Option<bool>) {
    use crate::hardware;
    let profile = hardware::profile();
    let auto = hardware::sd_fast_mode_auto_default(profile);
    let mut effective = user_override.unwrap_or(auto);

    let prev = effective;
    ui.checkbox(&mut effective,
        RichText::new("Fast SD inpaint (CPU optimization)")
            .color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
    if effective != prev {
        // Record explicit user choice. Setting it back to the auto value
        // collapses to None so the toggle keeps tracking hardware
        // changes (e.g. user installs OpenVINO + iGPU later).
        *user_override = if effective == auto { None } else { Some(effective) };
    }

    let mode_label = match user_override {
        None => format!("auto: {}", if auto { "on" } else { "off" }),
        Some(true) => "user: on".to_string(),
        Some(false) => "user: off".to_string(),
    };
    hint(ui, &format!(
        "Trade quality for speed when SD inpaint runs on CPU / Intel iGPU. Uses LCM-distilled weights (~5\u{00d7} faster, lower fidelity) when available; the Guidance slider greys out — LCM bakes guidance into training. Default tracks your hardware ({mode_label})."
    ));
}

fn render_tab_appearance(ui: &mut egui::Ui, settings: &mut Settings) {
    ui.checkbox(&mut settings.dark_checker,
        RichText::new("Dark checkerboard").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
    hint(ui, "Use dark tones for the transparency pattern \u{2014} helps when viewing light results.");
    ui.add_space(theme::SPACE_MD);

    ui.checkbox(&mut settings.auto_hide_adjustments,
        RichText::new("Auto-hide adjustments toolbar").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
    hint(ui, "Collapse the adjustments toolbar when the cursor leaves it. Toggle manually with Shift+H.");
}

fn render_tab_processing(ui: &mut egui::Ui, settings: &mut Settings) {
    ui.checkbox(&mut settings.auto_process_on_import,
        RichText::new("Auto-process on import").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
    hint(ui, "When enabled, each image kicks off Process automatically on import. The full pipeline runs \u{2014} BG removal or line extraction, whichever matches the current Line mode.");
    ui.add_space(theme::SPACE_MD);

    ui.checkbox(&mut settings.chain_mode,
        RichText::new("Chain mode").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
    hint(ui, "Process the current result instead of the original \u{2014} stacks effects.");

    if settings.chain_mode {
        ui.add_space(theme::SPACE_SM);
        let mut depth_f32 = settings.history_depth as f32;
        let depth_text = format!("{}", settings.history_depth);
        slider_row(ui, "History depth", &mut depth_f32, 1.0..=50.0, &depth_text, false, Some(1.0));
        settings.history_depth = depth_f32 as usize;
        hint(ui, "Maximum undo steps per image. Higher = more memory.");
    }
}

fn render_tab_defaults(ui: &mut egui::Ui, settings: &mut Settings) {
    ui.checkbox(&mut settings.export_split_layers,
        RichText::new("Split exports into layers").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
    hint(ui, "Drag-out and Save emit subject / lines / mask as separate PNGs instead of one composite \u{2014} useful for Photoshop / Procreate workflows. On Linux (no drag-out), Save is the way.");
    ui.add_space(theme::SPACE_MD);

    section_heading(ui, "Default preset");
    let preset_names = super::preset_dropdown::all_preset_names(settings);
    let current = settings.default_preset.clone();
    ui.horizontal(|ui| {
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
    hint(ui, "New images inherit this preset. Reset-all-knobs on row 2 also restores this preset's values.");
}

fn render_tab_hotkeys(ui: &mut egui::Ui) {
    ui.label(RichText::new("Rebindable shortcuts \u{2014} coming soon")
        .color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_BODY));
    hint(ui, "The current shortcut map is fixed. Press F1 anywhere in the app to see the full list.");
}
