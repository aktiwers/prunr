use crate::gui::settings::{Settings, SettingsModel};

#[test]
fn settings_default_parallel_jobs_is_half_cpus() {
    let s = Settings::default();
    let expected = (num_cpus::get() / 2).max(1);
    assert_eq!(s.parallel_jobs, expected);
}

#[test]
fn settings_serializes_to_json() {
    let s = Settings::default();
    let json = serde_json::to_string(&s).unwrap();
    assert!(json.contains("model"));
    assert!(json.contains("parallel_jobs"));
    // active_backend should be skipped
    assert!(!json.contains("active_backend"));
}

#[test]
fn settings_roundtrip_json() {
    let s = Settings {
        model: SettingsModel::U2net,
        parallel_jobs: 4,
        ..Settings::default()
    };
    let json = serde_json::to_string(&s).unwrap();
    let restored: Settings = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.model, SettingsModel::U2net);
    assert_eq!(restored.parallel_jobs, 4);
}
