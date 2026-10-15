//! Test-harness escape hatches: env-var overrides for per-image knobs.
//!
//! Each `PRUNR_*` var is consulted by `apply_to_item_settings` whenever a
//! new `BatchItem` is created, so a harness can preconfigure the knob state
//! without driving sliders through xdotool (which is unreliable under
//! Xephyr's coord-space mismatch).
//!
//! Production users never touch these — unset env vars are no-ops.

use prunr_core::{BgEffect, SubjectEffect, LineMode};

use super::item_settings::ItemSettings;

/// Apply `PRUNR_*` knob overrides on top of `item`. Unset vars leave fields
/// untouched. Bad values are silently ignored — defensive parse so a typo
/// in a harness script doesn't kill the launch.
pub fn apply_to_item_settings(item: &mut ItemSettings) {
    if let Some(v) = read("PRUNR_GAMMA") { item.gamma = v; }
    if let Some(v) = read("PRUNR_CONTRAST") { item.contrast = v; }
    if let Some(v) = read("PRUNR_EDGE_SHIFT") { item.edge_shift = prunr_core::MaskSettings::edge_shift_for_grow(v); }
    if let Some(v) = prunr_core::env::flag("PRUNR_REFINE_EDGES") { item.refine_edges = v; }
    if let Some(v) = read("PRUNR_FEATHER") { item.feather = v; }
    if let Some(v) = read_hex_rgba("PRUNR_BG_COLOR") { item.bg = Some(v); }
    if let Some(v) = read_line_mode("PRUNR_LINE_MODE") { item.line_mode = v; }
    if let Some(v) = read("PRUNR_LINE_STRENGTH") { item.line_strength = v; }
    if let Some(v) = read_subject_effect("PRUNR_FILL_STYLE") { item.effect = v; }
    if let Some(v) = read_bg_effect("PRUNR_BG_EFFECT") { item.bg_effect = v; }
}

/// `PRUNR_AUTO_PROCESS=1` flips `auto_process_on_import` so imported images
/// process without the harness having to send `Ctrl+R`. Mirrored on the
/// `Settings` rather than per-item because that's where the toggle lives.
pub fn auto_process_override() -> Option<bool> {
    prunr_core::env::flag("PRUNR_AUTO_PROCESS")
}

/// `PRUNR_CONTROL_PORT=<port>` opens the control socket on localhost.
pub fn control_port() -> Option<u16> {
    read("PRUNR_CONTROL_PORT")
}

fn read<T: std::str::FromStr>(var: &str) -> Option<T> {
    std::env::var(var).ok()?.trim().parse().ok()
}

/// Hex RGB ("ffffff") or RGBA ("ffffff80"). Leading `#` allowed.
fn read_hex_rgba(var: &str) -> Option<[u8; 4]> {
    let raw = std::env::var(var).ok()?;
    let s = raw.trim().trim_start_matches('#');
    let bytes = match s.len() {
        6 | 8 => (0..s.len()).step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
            .collect::<Option<Vec<_>>>()?,
        _ => return None,
    };
    let mut out = [0, 0, 0, 255];
    for (i, b) in bytes.into_iter().enumerate() { out[i] = b; }
    Some(out)
}

fn read_line_mode(var: &str) -> Option<LineMode> {
    let s = std::env::var(var).ok()?;
    match s.trim().to_ascii_lowercase().as_str() {
        "off" => Some(LineMode::Off),
        "edges" | "edges_only" | "edgesonly" => Some(LineMode::EdgesOnly),
        "subject" | "subject_outline" | "subjectoutline" => Some(LineMode::SubjectOutline),
        _ => None,
    }
}

fn read_subject_effect(var: &str) -> Option<SubjectEffect> {
    let s = std::env::var(var).ok()?;
    match s.trim().to_ascii_lowercase().as_str() {
        "none" => Some(SubjectEffect::None),
        "desaturate" => Some(SubjectEffect::Desaturate),
        "invert" => Some(SubjectEffect::Invert),
        "sepia" => Some(SubjectEffect::Sepia),
        _ => None,
    }
}

fn read_bg_effect(var: &str) -> Option<BgEffect> {
    let s = std::env::var(var).ok()?;
    match s.trim().to_ascii_lowercase().as_str() {
        "none" => Some(BgEffect::None),
        "desaturated" | "desaturated_source" => Some(BgEffect::DesaturatedSource),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each test mutates process-wide env state, so they share a Mutex to
    /// run one at a time even under cargo's parallel test runner.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_env<F: FnOnce()>(vars: &[(&str, &str)], f: F) {
        let _guard = ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for (k, v) in vars { unsafe { std::env::set_var(k, v); } }
        f();
        for (k, _) in vars { unsafe { std::env::remove_var(k); } }
    }

    #[test]
    fn unset_env_leaves_defaults_unchanged() {
        with_env(&[], || {
            let mut item = ItemSettings::default();
            apply_to_item_settings(&mut item);
            assert_eq!(item, ItemSettings::default());
        });
    }

    #[test]
    fn gamma_contrast_edge_shift_parse() {
        with_env(
            &[("PRUNR_GAMMA", "2.5"), ("PRUNR_CONTRAST", "0.4"),
              ("PRUNR_EDGE_SHIFT", "-3")],
            || {
                let mut item = ItemSettings::default();
                apply_to_item_settings(&mut item);
                assert_eq!(item.gamma, 2.5);
                assert_eq!(item.contrast, 0.4);
                assert_eq!(item.edge_shift, 3.0, "-3 shrinks, like the slider and the CLI");
            },
        );
    }

    #[test]
    fn refine_edges_accepts_truthy_strings() {
        for s in ["1", "true", "yes", "on", "TRUE"] {
            with_env(&[("PRUNR_REFINE_EDGES", s)], || {
                let mut item = ItemSettings::default();
                apply_to_item_settings(&mut item);
                assert!(item.refine_edges, "{s} should parse as true");
            });
        }
    }

    #[test]
    fn bg_color_hex_rgb_and_rgba() {
        with_env(&[("PRUNR_BG_COLOR", "ff0080")], || {
            let mut item = ItemSettings::default();
            apply_to_item_settings(&mut item);
            assert_eq!(item.bg, Some([0xff, 0x00, 0x80, 0xff]));
        });
        with_env(&[("PRUNR_BG_COLOR", "#11223344")], || {
            let mut item = ItemSettings::default();
            apply_to_item_settings(&mut item);
            assert_eq!(item.bg, Some([0x11, 0x22, 0x33, 0x44]));
        });
    }

    #[test]
    fn bad_value_is_ignored() {
        with_env(&[("PRUNR_GAMMA", "not-a-number")], || {
            let mut item = ItemSettings::default();
            apply_to_item_settings(&mut item);
            assert_eq!(item.gamma, ItemSettings::default().gamma);
        });
    }

    #[test]
    fn line_mode_aliases() {
        for (s, expect) in [
            ("Off", LineMode::Off),
            ("edges", LineMode::EdgesOnly),
            ("subject_outline", LineMode::SubjectOutline),
        ] {
            with_env(&[("PRUNR_LINE_MODE", s)], || {
                let mut item = ItemSettings::default();
                apply_to_item_settings(&mut item);
                assert_eq!(item.line_mode, expect);
            });
        }
    }
}
