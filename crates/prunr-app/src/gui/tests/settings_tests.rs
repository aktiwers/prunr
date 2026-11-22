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

/// Once a check is due the app asks from a frame, and the answer lands in
/// the settings, a newer version told; with the switch off it asks nothing.
#[test]
fn a_due_update_check_asks_from_a_frame_and_keeps_its_answer() {
    let mut h = super::harness::harness();
    h.state_mut().update_check.due_now();
    h.run();
    let sent = &h.state().update_check.sent;
    assert_eq!(sent.len(), 1, "asked from the frame");
    assert!(sent[0].0.starts_with(&format!("app=prunr&v={}&os=", crate::gui::update_check::VERSION)), "{}", sent[0].0);
    sent[0].1.send(Some("99.0.0".into())).unwrap();
    // One frame lands it; the settings' save then waits for input to pause.
    h.step();
    let kept = &h.state().settings.update_check;
    assert_eq!((kept.latest.as_str(), kept.told.as_str()), ("99.0.0", "99.0.0"));
    assert_eq!(kept.newer(), Some("99.0.0"));

    let mut off = super::harness::harness();
    off.state_mut().settings.check_updates = false;
    off.state_mut().update_check.due_now();
    off.run();
    assert!(off.state().update_check.sent.is_empty(), "the switch off");
}
