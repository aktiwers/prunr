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

/// Resolved view of `(file, model, scheduler)` — the values the GUI
/// applies right now.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedView {
    pub item_settings: ItemSettings,
    pub brush: BrushSettings,
}

/// Overwrite the SD-tuning fields of `brush` from `sd` + the chosen
/// scheduler's bundle. Shared between `resolve_preset_for_model` and
/// `fuse_brush_for_apply` — the only two paths that materialise SD
/// state from a stored `SdPreset` into a live `BrushSettings`.
fn apply_sd_to_brush(
    brush: &mut BrushSettings,
    sd: &SdPreset,
    scheduler: Option<SdScheduler>,
) {
    let chosen = scheduler.unwrap_or(sd.active_scheduler);
    let bundle = sd
        .schedulers
        .get(&chosen)
        .copied()
        .unwrap_or_else(|| SdSchedulerBundle::default_for(chosen));
    brush.sd_scheduler = chosen;
    brush.sd_steps = bundle.steps;
    brush.sd_guidance_scale = bundle.guidance_scale;
    brush.sd_use_karras_sigmas = bundle.use_karras_sigmas;
    brush.sd_strength = bundle.strength;
    brush.sd_prompt = sd.prompt.clone();
    brush.sd_negative_prompt = sd.negative_prompt.clone();
    brush.sd_seed = sd.seed;
    brush.sd_use_taesd = sd.use_taesd;
}

/// Resolve a preset to live values for the given model.
///
/// `scheduler = None` means "use the preset's stored `active_scheduler`"
/// for SD-family models, and is ignored for non-SD models. `Some(_)`
/// overrides — the in-session scheduler-change handler passes the
/// user's pick so the bundle swap doesn't rewrite the preset on disk.
pub(crate) fn resolve_preset_for_model(
    file: &PresetFile,
    model: ModelId,
    scheduler: Option<SdScheduler>,
) -> ResolvedView {
    let key = model_id_key(model);
    let mp = file.models.get(&key);

    let item_settings = mp.map(|m| m.item_settings).unwrap_or_default();
    let mut brush = mp.map(|m| m.brush.clone()).unwrap_or_default();

    if model.is_sd_family() {
        let sd_default = SdPreset::default();
        let sd = mp.and_then(|m| m.sd.as_ref()).unwrap_or(&sd_default);
        apply_sd_to_brush(&mut brush, sd, scheduler);
    }

    ResolvedView { item_settings, brush }
}

/// Save-time split: produce the on-disk shape from a unified live
/// `BrushSettings`. For non-SD models the SD fields stay on the brush
/// struct (they're session state) but no `SdPreset` is emitted.
pub(crate) fn split_brush_for_save(
    brush: &BrushSettings,
    model: ModelId,
) -> (BrushSettings, Option<SdPreset>) {
    if !model.is_sd_family() {
        return (brush.clone(), None);
    }
    let mut schedulers = HashMap::new();
    schedulers.insert(
        brush.sd_scheduler,
        SdSchedulerBundle {
            steps: brush.sd_steps,
            guidance_scale: brush.sd_guidance_scale,
            use_karras_sigmas: brush.sd_use_karras_sigmas,
            strength: brush.sd_strength,
        },
    );
    let sd = SdPreset {
        prompt: brush.sd_prompt.clone(),
        negative_prompt: brush.sd_negative_prompt.clone(),
        use_taesd: brush.sd_use_taesd,
        seed: brush.sd_seed,
        active_scheduler: brush.sd_scheduler,
        schedulers,
    };
    (brush.clone(), Some(sd))
}

/// Apply-time fuse: produce a unified `BrushSettings` from a stored
/// `ModelPreset`. `scheduler = Some(_)` overrides `sd.active_scheduler`
/// so an in-session scheduler swap can pick a different bundle without
/// mutating the preset.
pub(super) fn fuse_brush_for_apply(
    mp: &ModelPreset,
    scheduler: Option<SdScheduler>,
) -> BrushSettings {
    let mut brush = mp.brush.clone();
    if let Some(sd) = mp.sd.as_ref() {
        apply_sd_to_brush(&mut brush, sd, scheduler);
    }
    brush
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

    fn brush_with(radius: f32) -> BrushSettings {
        BrushSettings { radius, ..BrushSettings::default() }
    }

    fn item_with(gamma: f32) -> ItemSettings {
        ItemSettings { gamma, ..ItemSettings::default() }
    }

    #[test]
    fn resolve_full_entry_returns_preset_values() {
        let mp = ModelPreset {
            item_settings: item_with(2.5),
            brush: brush_with(80.0),
            sd: None,
        };
        let mut file = PresetFile::default();
        file.models.insert(model_id_key(ModelId::Silueta), mp);

        let r = resolve_preset_for_model(&file, ModelId::Silueta, None);
        assert!((r.item_settings.gamma - 2.5).abs() < f32::EPSILON);
        assert!((r.brush.radius - 80.0).abs() < f32::EPSILON);
    }

    #[test]
    fn resolve_missing_model_falls_back_to_prunr() {
        let file = PresetFile::default();
        let r = resolve_preset_for_model(&file, ModelId::U2net, None);
        assert_eq!(r.item_settings, ItemSettings::default());
        assert_eq!(r.brush, BrushSettings::default());
    }

    #[test]
    fn resolve_missing_field_falls_back_per_field() {
        let json = r#"{
            "format_version": 2,
            "models": {
                "Silueta": { "item_settings": { "gamma": 2.5 } }
            }
        }"#;
        let file: PresetFile = serde_json::from_str(json).expect("parse");

        let r = resolve_preset_for_model(&file, ModelId::Silueta, None);
        assert!((r.item_settings.gamma - 2.5).abs() < f32::EPSILON);
        assert_eq!(r.item_settings.threshold, ItemSettings::default().threshold);
        assert_eq!(r.brush, BrushSettings::default());
    }

    #[test]
    fn resolve_sd_full_bundle_uses_bundle_values() {
        let mut schedulers = HashMap::new();
        schedulers.insert(SdScheduler::Lcm, SdSchedulerBundle {
            steps: 4,
            guidance_scale: 1.0,
            use_karras_sigmas: false,
            strength: 1.0,
        });
        let sd = SdPreset {
            active_scheduler: SdScheduler::Lcm,
            schedulers,
            ..SdPreset::default()
        };
        let mp = ModelPreset { sd: Some(sd), ..ModelPreset::default() };
        let mut file = PresetFile::default();
        file.models.insert(model_id_key(ModelId::SdV15InpaintFp16), mp);

        let r = resolve_preset_for_model(&file, ModelId::SdV15InpaintFp16, Some(SdScheduler::Lcm));
        assert_eq!(r.brush.sd_steps, 4);
        assert!((r.brush.sd_guidance_scale - 1.0).abs() < f32::EPSILON);
        assert_eq!(r.brush.sd_scheduler, SdScheduler::Lcm);
    }

    #[test]
    fn resolve_sd_missing_bundle_uses_default_for_scheduler() {
        let sd = SdPreset {
            active_scheduler: SdScheduler::Lcm,
            schedulers: HashMap::new(),
            ..SdPreset::default()
        };
        let mp = ModelPreset { sd: Some(sd), ..ModelPreset::default() };
        let mut file = PresetFile::default();
        file.models.insert(model_id_key(ModelId::SdV15InpaintFp16), mp);

        let r = resolve_preset_for_model(&file, ModelId::SdV15InpaintFp16, Some(SdScheduler::Ddim));
        let ddim = SdSchedulerBundle::default_for(SdScheduler::Ddim);
        assert_eq!(r.brush.sd_steps, ddim.steps);
        assert!((r.brush.sd_guidance_scale - ddim.guidance_scale).abs() < f32::EPSILON);
        assert_eq!(r.brush.sd_scheduler, SdScheduler::Ddim);
    }

    #[test]
    fn resolve_sd_no_sd_entry_uses_prunr_sdpreset() {
        let mp = ModelPreset { sd: None, ..ModelPreset::default() };
        let mut file = PresetFile::default();
        file.models.insert(model_id_key(ModelId::SdV15InpaintFp16), mp);

        let r = resolve_preset_for_model(&file, ModelId::SdV15InpaintFp16, Some(SdScheduler::Lcm));
        let lcm = SdSchedulerBundle::default_for(SdScheduler::Lcm);
        assert_eq!(r.brush.sd_scheduler, SdScheduler::Lcm);
        assert_eq!(r.brush.sd_steps, lcm.steps);
        assert!((r.brush.sd_guidance_scale - lcm.guidance_scale).abs() < f32::EPSILON);
    }

    #[test]
    fn resolve_non_sd_model_ignores_scheduler() {
        let mp = ModelPreset {
            brush: brush_with(80.0),
            ..ModelPreset::default()
        };
        let mut file = PresetFile::default();
        file.models.insert(model_id_key(ModelId::Silueta), mp.clone());

        let r = resolve_preset_for_model(&file, ModelId::Silueta, Some(SdScheduler::Lcm));
        assert_eq!(r.brush, mp.brush);
    }

    #[test]
    fn resolve_sd_uses_active_scheduler_when_caller_passes_none() {
        let mut schedulers = HashMap::new();
        schedulers.insert(SdScheduler::Ddim, SdSchedulerBundle {
            steps: 20,
            guidance_scale: 7.5,
            use_karras_sigmas: false,
            strength: 1.0,
        });
        let sd = SdPreset {
            active_scheduler: SdScheduler::Ddim,
            schedulers,
            ..SdPreset::default()
        };
        let mp = ModelPreset { sd: Some(sd), ..ModelPreset::default() };
        let mut file = PresetFile::default();
        file.models.insert(model_id_key(ModelId::SdV15InpaintFp16), mp);

        let r = resolve_preset_for_model(&file, ModelId::SdV15InpaintFp16, None);
        assert_eq!(r.brush.sd_scheduler, SdScheduler::Ddim);
        assert_eq!(r.brush.sd_steps, 20);
        assert!((r.brush.sd_guidance_scale - 7.5).abs() < f32::EPSILON);
    }

    #[test]
    fn split_brush_for_save_non_sd_model_returns_brush_only() {
        let mut brush = BrushSettings::default();
        brush.sd_prompt = "carry-over".into();
        brush.sd_steps = 12;

        let (out_brush, sd) = split_brush_for_save(&brush, ModelId::Silueta);
        assert_eq!(out_brush, brush);
        assert!(sd.is_none());
    }

    #[test]
    fn split_brush_for_save_sd_model_extracts_sd_fields() {
        let brush = BrushSettings {
            sd_prompt: "hello".into(),
            sd_negative_prompt: "world".into(),
            sd_steps: 12,
            sd_guidance_scale: 6.5,
            sd_scheduler: SdScheduler::Ddim,
            sd_use_karras_sigmas: true,
            sd_strength: 0.85,
            sd_seed: Some(7),
            sd_use_taesd: Some(true),
            ..BrushSettings::default()
        };

        let (_, sd) = split_brush_for_save(&brush, ModelId::SdV15InpaintFp16);
        let sd = sd.expect("SD model produces an SdPreset");
        assert_eq!(sd.prompt, "hello");
        assert_eq!(sd.negative_prompt, "world");
        assert_eq!(sd.use_taesd, Some(true));
        assert_eq!(sd.seed, Some(7));
        assert_eq!(sd.active_scheduler, SdScheduler::Ddim);

        let bundle = sd.schedulers.get(&SdScheduler::Ddim).copied().expect("Ddim bundle");
        assert_eq!(bundle.steps, 12);
        assert!((bundle.guidance_scale - 6.5).abs() < f32::EPSILON);
        assert!(bundle.use_karras_sigmas);
        assert!((bundle.strength - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn fuse_brush_for_apply_uses_sd_bundle_values() {
        let mut schedulers = HashMap::new();
        schedulers.insert(SdScheduler::Ddim, SdSchedulerBundle {
            steps: 30,
            guidance_scale: 6.0,
            use_karras_sigmas: false,
            strength: 0.9,
        });
        let sd = SdPreset {
            prompt: "x".into(),
            negative_prompt: "y".into(),
            active_scheduler: SdScheduler::Ddim,
            schedulers,
            ..SdPreset::default()
        };
        let mp = ModelPreset {
            brush: brush_with(80.0),
            sd: Some(sd),
            ..ModelPreset::default()
        };

        let fused = fuse_brush_for_apply(&mp, Some(SdScheduler::Ddim));
        assert!((fused.radius - 80.0).abs() < f32::EPSILON);
        assert_eq!(fused.sd_steps, 30);
        assert!((fused.sd_guidance_scale - 6.0).abs() < f32::EPSILON);
        assert_eq!(fused.sd_prompt, "x");
        assert_eq!(fused.sd_negative_prompt, "y");
        assert_eq!(fused.sd_scheduler, SdScheduler::Ddim);
        assert!((fused.sd_strength - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn fuse_brush_for_apply_no_sd_entry_uses_brush_sd_fields() {
        let mut brush = BrushSettings::default();
        brush.sd_prompt = "kept".into();
        brush.sd_steps = 11;
        let mp = ModelPreset { brush: brush.clone(), sd: None, ..ModelPreset::default() };

        let fused = fuse_brush_for_apply(&mp, Some(SdScheduler::Ddim));
        assert_eq!(fused, brush);
    }

    #[test]
    fn reset_button_parity_top_right_matches_brush_popover_subset() {
        use prunr_core::brush::{BrushMode, BrushShape};
        let preset_brush = BrushSettings {
            radius: 80.0,
            hardness: 0.3,
            mode: BrushMode::Add,
            shape: BrushShape::Square,
            inpaint_grow: 5.0,
            inpaint_feather: 6.0,
            inpaint_sharpen: 1.5,
            ..BrushSettings::default()
        };
        let mp = ModelPreset {
            item_settings: ItemSettings::default(),
            brush: preset_brush.clone(),
            sd: None,
        };
        let mut file = PresetFile::default();
        file.models.insert(model_id_key(ModelId::Silueta), mp);

        // (a) Top-right ↻ path: full re-resolve via the resolver. Brush
        //     comes back exactly as the preset stored it.
        let top_right_path_brush =
            resolve_preset_for_model(&file, ModelId::Silueta, None).brush;

        // (b) Brush popover Reset path: caller has a live `Settings.brush`
        //     with a user mutation (radius=10) and resets the popover-visible
        //     subset using the resolver's brush as the source.
        let mut live_brush = BrushSettings::default();
        live_brush.radius = 10.0;
        let mut popover_target = live_brush.clone();
        popover_target.reset_popover_fields_from(&top_right_path_brush);

        // Parity: every popover-visible knob agrees between the two paths.
        assert!((top_right_path_brush.radius - 80.0).abs() < f32::EPSILON);
        assert!((popover_target.radius - top_right_path_brush.radius).abs() < f32::EPSILON);
        assert!((popover_target.hardness - top_right_path_brush.hardness).abs() < f32::EPSILON);
        assert_eq!(popover_target.shape, top_right_path_brush.shape);
        assert!(
            (popover_target.inpaint_grow - top_right_path_brush.inpaint_grow).abs()
                < f32::EPSILON,
        );
        assert!(
            (popover_target.inpaint_feather - top_right_path_brush.inpaint_feather).abs()
                < f32::EPSILON,
        );
        assert!(
            (popover_target.inpaint_sharpen - top_right_path_brush.inpaint_sharpen).abs()
                < f32::EPSILON,
        );
    }

    #[test]
    fn split_then_fuse_round_trips_sd_model() {
        let original = BrushSettings {
            radius: 33.0,
            sd_prompt: "rt-prompt".into(),
            sd_negative_prompt: "rt-neg".into(),
            sd_steps: 17,
            sd_guidance_scale: 4.25,
            sd_scheduler: SdScheduler::Ddim,
            sd_use_karras_sigmas: true,
            sd_strength: 0.5,
            sd_seed: Some(99),
            sd_use_taesd: Some(false),
            ..BrushSettings::default()
        };

        let (brush_out, sd) = split_brush_for_save(&original, ModelId::SdV15InpaintFp16);
        let mp = ModelPreset { brush: brush_out, sd, ..ModelPreset::default() };
        let fused = fuse_brush_for_apply(&mp, Some(original.sd_scheduler));
        assert_eq!(fused, original);
    }
}
