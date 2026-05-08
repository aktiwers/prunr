//! Per-model preset schema for v2 preset files.
//!
//! `ModelPreset` carries item knobs + brush knobs + (for SD-family
//! models) an `SdPreset` with per-scheduler tuning bundles. The
//! one-`ItemSettings`-per-file v1 layout in `presets_fs` is migrated
//! into a v2 `models[model_id] = ModelPreset` envelope at load time.
//!
//! ## Forward-compatibility contract (read before editing any field)
//!
//! Every new field on these structs MUST be reachable from default —
//! that is what the tripwire tests at the bottom enforce. If you add
//! a field and one of these tests breaks, the fix is `#[serde(default)]`
//! on the field (or a Default impl that covers the whole struct).
//!
//! `models` is keyed by `String` (the `ModelId` Debug name), NOT by
//! `ModelId` directly: a v2 preset file produced by a future binary
//! that has a `ModelId::FutureModel` variant we don't know about
//! must still deserialize cleanly on this binary — unknown string
//! keys round-trip and are silently skipped at lookup time.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use prunr_models::ModelId;

use super::brush_state::{BrushSettings, SdScheduler};
use super::item_settings::ItemSettings;

pub(crate) const PRESET_FORMAT_VERSION: u32 = 2;

fn default_format_version() -> u32 { PRESET_FORMAT_VERSION }

/// Convert a `ModelId` into the JSON key used in `PresetFile.models`.
/// Single source of truth — every `models.insert` / `.get` site routes
/// through this function. Debug-name format matches the on-disk shape
/// of `Settings.accepted_licenses` so both consent state and preset
/// keys stay in lockstep.
pub(crate) fn model_id_key(id: ModelId) -> String {
    format!("{id:?}")
}

/// Inverse of `model_id_key`. Returns `None` for unknown keys —
/// future-binary entries the receiver doesn't know about round-trip
/// in the JSON but resolve to nothing at lookup time.
pub(crate) fn model_id_from_key(s: &str) -> Option<ModelId> {
    ModelId::ALL.iter().copied().find(|id| format!("{id:?}") == s)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct PresetFile {
    pub format_version: u32,
    pub models: HashMap<String, ModelPreset>,
}

impl Default for PresetFile {
    fn default() -> Self {
        Self {
            format_version: default_format_version(),
            models: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct ModelPreset {
    pub item_settings: ItemSettings,
    pub brush: BrushSettings,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sd: Option<SdPreset>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SdPreset {
    pub prompt: String,
    pub negative_prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_taesd: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub active_scheduler: SdScheduler,
    pub schedulers: HashMap<SdScheduler, SdSchedulerBundle>,
}

impl Default for SdPreset {
    fn default() -> Self {
        use super::brush_state::{DEFAULT_SD_NEGATIVE_PROMPT, DEFAULT_SD_PROMPT};
        Self {
            prompt: DEFAULT_SD_PROMPT.to_string(),
            negative_prompt: DEFAULT_SD_NEGATIVE_PROMPT.to_string(),
            use_taesd: None,
            seed: None,
            active_scheduler: SdScheduler::Lcm,
            schedulers: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SdSchedulerBundle {
    pub steps: u32,
    pub guidance_scale: f32,
    pub use_karras_sigmas: bool,
    pub strength: f32,
}

impl Default for SdSchedulerBundle {
    fn default() -> Self {
        Self::default_for(SdScheduler::Lcm)
    }
}

impl SdSchedulerBundle {
    /// Factory tuning for a scheduler — Prunr's floor when an active
    /// preset omits a `schedulers[S]` entry. The numbers mirror
    /// `SdQualityPreset::Balanced` for LCM and `SdQualityPreset::Quality`
    /// for DPM++; DDIM / UniPC / Euler-A use scheduler-typical defaults.
    pub fn default_for(scheduler: SdScheduler) -> Self {
        match scheduler {
            SdScheduler::Lcm => Self {
                steps: 8,
                guidance_scale: 1.5,
                use_karras_sigmas: false,
                strength: 1.0,
            },
            SdScheduler::Ddim => Self {
                steps: 20,
                guidance_scale: 7.5,
                use_karras_sigmas: false,
                strength: 1.0,
            },
            SdScheduler::DpmPlusPlus2MKarras => Self {
                steps: 25,
                guidance_scale: 4.0,
                use_karras_sigmas: true,
                strength: 1.0,
            },
            SdScheduler::UniPc => Self {
                steps: 10,
                guidance_scale: 4.0,
                use_karras_sigmas: false,
                strength: 1.0,
            },
            SdScheduler::EulerA => Self {
                steps: 25,
                guidance_scale: 7.5,
                use_karras_sigmas: false,
                strength: 1.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_file_default_is_empty_models_map() {
        let f = PresetFile::default();
        assert_eq!(f.format_version, 2);
        assert!(f.models.is_empty());
    }

    #[test]
    fn model_preset_default_round_trips() {
        let original = ModelPreset::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ModelPreset = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, original);
    }

    #[test]
    fn sd_preset_default_round_trips() {
        let original = SdPreset::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: SdPreset = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, original);
    }

    #[test]
    fn sd_scheduler_bundle_default_for_lcm_matches_balanced_apply_to() {
        let b = SdSchedulerBundle::default_for(SdScheduler::Lcm);
        assert_eq!(b.steps, 8);
        assert!((b.guidance_scale - 1.5).abs() < f32::EPSILON);
        assert!(!b.use_karras_sigmas);
        assert!((b.strength - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sd_scheduler_bundle_default_for_ddim_uses_ddim_tuning() {
        let b = SdSchedulerBundle::default_for(SdScheduler::Ddim);
        assert_eq!(b.steps, 20);
        assert!((b.guidance_scale - 7.5).abs() < f32::EPSILON);
        assert!(!b.use_karras_sigmas);
        assert!((b.strength - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sd_scheduler_bundle_default_for_dpmpp_matches_quality_preset() {
        let b = SdSchedulerBundle::default_for(SdScheduler::DpmPlusPlus2MKarras);
        assert_eq!(b.steps, 25);
        assert!((b.guidance_scale - 4.0).abs() < f32::EPSILON);
        assert!(b.use_karras_sigmas);
        assert!((b.strength - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sd_scheduler_bundle_default_for_unipc_uses_unipc_tuning() {
        let b = SdSchedulerBundle::default_for(SdScheduler::UniPc);
        assert_eq!(b.steps, 10);
        assert!((b.guidance_scale - 4.0).abs() < f32::EPSILON);
        assert!(!b.use_karras_sigmas);
        assert!((b.strength - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sd_scheduler_bundle_default_for_eulera_uses_eulera_tuning() {
        let b = SdSchedulerBundle::default_for(SdScheduler::EulerA);
        assert_eq!(b.steps, 25);
        assert!((b.guidance_scale - 7.5).abs() < f32::EPSILON);
        assert!(!b.use_karras_sigmas);
        assert!((b.strength - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn presets_loads_empty_object_as_defaults_modelpreset() {
        let parsed: ModelPreset = serde_json::from_str("{}")
            .expect("ModelPreset must deserialize from `{}` — add #[serde(default)] to any new field");
        assert_eq!(parsed, ModelPreset::default());
    }

    #[test]
    fn presets_loads_unknown_fields_modelpreset() {
        let json = r#"{ "made_up_field": 42, "another_future_field": "hello" }"#;
        let parsed: ModelPreset = serde_json::from_str(json)
            .expect("ModelPreset must ignore unknown fields — do NOT add #[serde(deny_unknown_fields)]");
        assert_eq!(parsed, ModelPreset::default());
    }

    #[test]
    fn presets_loads_empty_object_as_defaults_sdpreset() {
        let parsed: SdPreset = serde_json::from_str("{}")
            .expect("SdPreset must deserialize from `{}` — add #[serde(default)] to any new field");
        assert_eq!(parsed, SdPreset::default());
    }

    #[test]
    fn presets_loads_unknown_fields_sdpreset() {
        let json = r#"{ "future_field": true, "another": [1, 2, 3] }"#;
        let parsed: SdPreset = serde_json::from_str(json)
            .expect("SdPreset must ignore unknown fields");
        assert_eq!(parsed, SdPreset::default());
    }

    #[test]
    fn presets_loads_empty_object_as_defaults_sdschedulerbundle() {
        let parsed: SdSchedulerBundle = serde_json::from_str("{}")
            .expect("SdSchedulerBundle must deserialize from `{}` — add #[serde(default)] to any new field");
        assert_eq!(parsed, SdSchedulerBundle::default());
    }

    #[test]
    fn presets_loads_unknown_fields_sdschedulerbundle() {
        let json = r#"{ "tomorrows_field": 1.5 }"#;
        let parsed: SdSchedulerBundle = serde_json::from_str(json)
            .expect("SdSchedulerBundle must ignore unknown fields");
        assert_eq!(parsed, SdSchedulerBundle::default());
    }

    #[test]
    fn presets_loads_empty_object_as_defaults_presetfile() {
        let parsed: PresetFile = serde_json::from_str("{}")
            .expect("PresetFile must deserialize from `{}`");
        assert_eq!(parsed, PresetFile::default());
    }

    #[test]
    fn presets_loads_unknown_fields_presetfile() {
        let json = r#"{ "format_version": 2, "models": {}, "extra_top_level": "ignored" }"#;
        let parsed: PresetFile = serde_json::from_str(json)
            .expect("PresetFile must ignore unknown top-level fields");
        assert_eq!(parsed.format_version, 2);
        assert!(parsed.models.is_empty());
    }

    #[test]
    fn preset_file_round_trips_with_one_model_entry() {
        let mut sd = SdPreset::default();
        sd.active_scheduler = SdScheduler::Lcm;
        sd.schedulers.insert(
            SdScheduler::Lcm,
            SdSchedulerBundle::default_for(SdScheduler::Lcm),
        );
        let model_preset = ModelPreset {
            sd: Some(sd),
            ..ModelPreset::default()
        };
        let mut original = PresetFile::default();
        original
            .models
            .insert(model_id_key(ModelId::SdV15InpaintFp16), model_preset);

        let json = serde_json::to_string(&original).expect("serialize");
        let restored: PresetFile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, original);
    }

    #[test]
    fn preset_file_serializes_modelid_as_debug_name() {
        let mut f = PresetFile::default();
        f.models.insert(
            model_id_key(ModelId::SdV15InpaintFp16),
            ModelPreset::default(),
        );
        let json = serde_json::to_string(&f).expect("serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse back");
        let models = value["models"].as_object().expect("models is object");
        assert!(
            models.contains_key("SdV15InpaintFp16"),
            "models key must be the ModelId Debug name; got keys {:?}",
            models.keys().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn sd_preset_skip_serializing_none_omits_use_taesd_field() {
        let sd = SdPreset {
            use_taesd: None,
            seed: None,
            ..SdPreset::default()
        };
        let json = serde_json::to_string(&sd).expect("serialize");
        assert!(
            !json.contains("\"use_taesd\""),
            "use_taesd: None must be skipped in JSON; got {json}",
        );
        assert!(
            !json.contains("\"seed\""),
            "seed: None must be skipped in JSON; got {json}",
        );
    }

    #[test]
    fn model_id_key_round_trips_via_from_key() {
        for &id in ModelId::ALL {
            let key = model_id_key(id);
            assert_eq!(
                model_id_from_key(&key),
                Some(id),
                "round-trip failed for {id:?}",
            );
        }
    }

    #[test]
    fn model_id_from_key_returns_none_for_unknown() {
        assert_eq!(model_id_from_key("FutureModel"), None);
        assert_eq!(model_id_from_key("not a real key"), None);
        assert_eq!(model_id_from_key(""), None);
    }
}
