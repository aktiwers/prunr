//! A serde view of the app for the control socket: what is loaded, what
//! is selected, which tool and modals are up, the settings that change
//! what a command does.

use serde::Serialize;

use crate::gui::app::PrunrApp;
use crate::gui::item_settings::ItemSettings;

#[derive(Debug, Serialize)]
pub struct StateDump {
    pub app_state: String,
    pub model: String,
    pub tool: &'static str,
    pub selected: Option<u64>,
    pub items: Vec<ItemDump>,
    pub show_original: bool,
    pub sidebar_hidden: bool,
    pub adjustments_hidden: bool,
    pub modals: Vec<&'static str>,
    pub settings: SettingsDump,
}

#[derive(Debug, Serialize)]
pub struct ItemDump {
    pub id: u64,
    pub filename: String,
    pub status: String,
    pub selected: bool,
    pub dimensions: (u32, u32),
    pub has_result: bool,
    pub settings: ItemSettings,
}

#[derive(Debug, Serialize)]
pub struct SettingsDump {
    pub auto_process_on_import: bool,
    pub parallel_jobs: usize,
    pub live_preview: bool,
    pub chain_mode: bool,
    pub keep_sd_loaded: bool,
}

pub fn dump(app: &PrunrApp) -> StateDump {
    StateDump {
        app_state: format!("{:?}", app.batch.app_state()),
        model: format!("{:?}", app.settings.model),
        tool: if app.magic_brush_state.is_active() {
            "magic_brush"
        } else if app.brush_state.is_enabled() {
            "paint_brush"
        } else {
            "none"
        },
        selected: app.batch.selected_item().map(|it| it.id),
        items: app
            .batch
            .items
            .iter()
            .map(|it| ItemDump {
                id: it.id,
                filename: it.filename.clone(),
                status: format!("{:?}", it.status),
                selected: it.selected,
                dimensions: it.dimensions,
                has_result: it.result_rgba.is_some(),
                settings: it.settings,
            })
            .collect(),
        show_original: app.show_original,
        sidebar_hidden: app.sidebar_hidden,
        adjustments_hidden: app.adjustments_hidden,
        modals: app.open_modals(),
        settings: SettingsDump {
            auto_process_on_import: app.settings.auto_process_on_import,
            parallel_jobs: app.settings.parallel_jobs,
            live_preview: app.settings.live_preview,
            chain_mode: app.settings.chain_mode,
            keep_sd_loaded: app.settings.keep_sd_loaded,
        },
    }
}
