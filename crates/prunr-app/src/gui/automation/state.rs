//! A serde view of the app for the control socket: what is loaded, what
//! is selected, which tool and modals are up, the settings that change
//! what a command does.

use serde::Serialize;

use crate::gui::app::PrunrApp;
use crate::gui::brush_state::Tool;
use crate::gui::item_settings::ItemSettings;

#[derive(Debug, Serialize)]
pub struct StateDump {
    pub app_state: String,
    pub model: String,
    pub tool: &'static str,
    pub selected: Option<u64>,
    /// Work still owed to the screen: a preview, an upscale, a run, or a
    /// decode or texture. A script waits for this to clear.
    pub busy: bool,
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
    pub result_dims: Option<(u32, u32)>,
    pub has_selection: bool,
    pub undo_steps: usize,
    pub settings: ItemSettings,
}

#[derive(Debug, Serialize)]
pub struct SettingsDump {
    pub auto_process_on_import: bool,
    pub parallel_jobs: usize,
    pub keep_sd_loaded: bool,
}

pub fn dump(app: &PrunrApp) -> StateDump {
    StateDump {
        app_state: format!("{:?}", app.batch.app_state()),
        model: format!("{:?}", app.settings.model),
        tool: match app.active_tool() {
            Some(Tool::Magic) => "magic_brush",
            Some(Tool::Paint) => "paint_brush",
            None if app.picking_depth_focus => "set_focus",
            None => "none",
        },
        selected: app.batch.selected_item().map(|it| it.id),
        busy: app.quiet_background_work().is_some()
            || app.processor.is_run_in_flight()
            || app.batch.items.iter().any(|it| it.is_busy()),
        items: app
            .batch
            .items
            .iter()
            .map(|it| ItemDump {
                id: it.id,
                filename: it.filename.clone(),
                status: format!("{:?}", it.status()),
                selected: it.ticked,
                dimensions: it.dimensions,
                has_result: it.has_result(),
                result_dims: it.result_rgba.as_ref().map(|r| r.dimensions()),
                has_selection: it.selection_mask.is_some(),
                undo_steps: it.timeline.undo_steps(),
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
            keep_sd_loaded: app.settings.keep_sd_loaded,
        },
    }
}
