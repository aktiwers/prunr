//! Test-harness escape hatches: env-var overrides for per-image knobs.
//!
//! Each `PRUNR_*` var is consulted by `apply_to_item_settings` whenever a
//! new `BatchItem` is created, so a harness can preconfigure the knob state
//! without driving sliders through xdotool (which is unreliable under
//! Xephyr's coord-space mismatch — see task #83).
//!
//! Production users never touch these — unset env vars are no-ops.

use prunr_core::{BgEffect, FillStyle, LineMode};

use super::item_settings::ItemSettings;

/// Apply `PRUNR_*` knob overrides on top of `item`. Unset vars leave fields
/// untouched. Bad values are silently ignored — defensive parse so a typo
/// in a harness script doesn't kill the launch.
pub fn apply_to_item_settings(item: &mut ItemSettings) {
    if let Some(v) = read_f32("PRUNR_GAMMA") {
        item.gamma = v;
    }
    if let Some(v) = read_f32("PRUNR_THRESHOLD") {
        item.threshold = Some(v);
    }
    if let Some(v) = read_f32("PRUNR_EDGE_SHIFT") {
        item.edge_shift = v;
    }
    if let Some(v) = read_bool("PRUNR_REFINE_EDGES") {
        item.refine_edges = v;
    }
    if let Some(v) = read_f32("PRUNR_FEATHER") {
        item.feather = v;
    }
    if let Some(v) = read_hex_rgba("PRUNR_BG_COLOR") {
        item.bg = Some(v);
    }
    if let Some(v) = read_line_mode("PRUNR_LINE_MODE") {
        item.line_mode = v;
    }
    if let Some(v) = read_f32("PRUNR_LINE_STRENGTH") {
        item.line_strength = v;
    }
    if let Some(v) = read_fill_style("PRUNR_FILL_STYLE") {
        item.fill_style = v;
    }
    if let Some(v) = read_bg_effect("PRUNR_BG_EFFECT") {
        item.bg_effect = v;
    }
}

/// `PRUNR_AUTO_PROCESS=1` flips `auto_process_on_import` so imported images
/// process without the harness having to send `Ctrl+R`. Mirrored on the
/// `Settings` rather than per-item because that's where the toggle lives.
pub fn auto_process_override() -> Option<bool> {
    read_bool("PRUNR_AUTO_PROCESS")
}

fn read_f32(var: &str) -> Option<f32> {
    std::env::var(var).ok()?.parse().ok()
}

fn read_bool(var: &str) -> Option<bool> {
    let s = std::env::var(var).ok()?;
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Hex RGB ("ffffff") or RGBA ("ffffff80"). Leading `#` allowed.
fn read_hex_rgba(var: &str) -> Option<[u8; 4]> {
    let raw = std::env::var(var).ok()?;
    let s = raw.trim().trim_start_matches('#');
    let bytes = match s.len() {
        6 | 8 => (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
            .collect::<Option<Vec<_>>>()?,
        _ => return None,
    };
    let mut out = [0, 0, 0, 255];
    for (i, b) in bytes.into_iter().enumerate() {
        out[i] = b;
    }
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

fn read_fill_style(var: &str) -> Option<FillStyle> {
    let s = std::env::var(var).ok()?;
    match s.trim().to_ascii_lowercase().as_str() {
        "none" => Some(FillStyle::None),
        "desaturate" => Some(FillStyle::Desaturate),
        "invert" => Some(FillStyle::Invert),
        "sepia" => Some(FillStyle::Sepia),
        _ => None,
    }
}

fn read_bg_effect(var: &str) -> Option<BgEffect> {
    let s = std::env::var(var).ok()?;
    match s.trim().to_ascii_lowercase().as_str() {
        "none" => Some(BgEffect::None),
        "inverted" | "inverted_source" => Some(BgEffect::InvertedSource),
        "desaturated" | "desaturated_source" => Some(BgEffect::DesaturatedSource),
        // BlurredSource{radius} not exposed — needs a parameter; defer.
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
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (k, v) in vars {
            unsafe {
                std::env::set_var(k, v);
            }
        }
        f();
        for (k, _) in vars {
            unsafe {
                std::env::remove_var(k);
            }
        }
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
    fn gamma_threshold_edge_shift_parse() {
        with_env(
            &[
                ("PRUNR_GAMMA", "2.5"),
                ("PRUNR_THRESHOLD", "0.4"),
                ("PRUNR_EDGE_SHIFT", "-3"),
            ],
            || {
                let mut item = ItemSettings::default();
                apply_to_item_settings(&mut item);
                assert_eq!(item.gamma, 2.5);
                assert_eq!(item.threshold, Some(0.4));
                assert_eq!(item.edge_shift, -3.0);
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
