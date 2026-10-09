//! Value text for chip faces and slider boxes. One place decides how a
//! number reads so every control agrees: "Off" capitalized, a space before
//! px and EV, none before %, an explicit sign where direction matters, and
//! no Greek prefixes.

/// "Off" below `off_below`, otherwise `f(v)`.
pub fn off_or(v: f32, off_below: f32, f: impl FnOnce(f32) -> String) -> String {
    if v.abs() < off_below { "Off".to_string() } else { f(v) }
}

/// "1.5 px" / "20 px" depending on `decimals`.
pub fn px(v: f32, decimals: usize) -> String {
    format!("{v:.decimals$} px")
}

/// "+2 px" / "\u{2212}3 px" / "0 px". Uses the typographic minus so the
/// sign reads at chip size.
pub fn signed_px(v: f32, decimals: usize) -> String {
    signed(v, decimals, " px")
}

/// "+0.5 EV" / "0 EV".
pub fn ev(v: f32) -> String {
    if v == 0.0 { "0 EV".to_string() } else { signed(v, 1, " EV") }
}

/// 0–1 as "50%".
pub fn percent(v01: f32) -> String {
    format!("{:.0}%", v01 * 100.0)
}

/// 0–1 as "50.0%" for knobs that need tenths (Hardness).
pub fn percent_tenths(v01: f32) -> String {
    format!("{:.1}%", v01 * 100.0)
}

/// "+0.25" / "\u{2212}0.25" for symmetric ranges around zero.
pub fn signed_plain(v: f32, decimals: usize) -> String {
    signed(v, decimals, "")
}

/// "1.00" for plain magnitudes (gamma, line detail).
pub fn plain(v: f32, decimals: usize) -> String {
    format!("{v:.decimals$}")
}

fn signed(v: f32, decimals: usize, unit: &str) -> String {
    let rounded = format!("{:.decimals$}", v.abs());
    let is_zero = rounded.trim_matches(|c| c == '0' || c == '.').is_empty();
    if is_zero {
        format!("0{unit}")
    } else if v > 0.0 {
        format!("+{rounded}{unit}")
    } else {
        format!("\u{2212}{rounded}{unit}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_is_capitalized_and_threshold_respected() {
        assert_eq!(off_or(0.05, 0.1, |v| plain(v, 2)), "Off");
        assert_eq!(off_or(0.5, 0.1, |v| plain(v, 2)), "0.50");
    }

    #[test]
    fn units_get_a_space_except_percent() {
        assert_eq!(px(1.5, 1), "1.5 px");
        assert_eq!(px(20.0, 0), "20 px");
        assert_eq!(percent(0.5), "50%");
        assert_eq!(percent_tenths(0.505), "50.5%");
        assert_eq!(ev(0.5), "+0.5 EV");
        assert_eq!(ev(0.0), "0 EV");
    }

    #[test]
    fn signs_use_plus_and_typographic_minus() {
        assert_eq!(signed_px(2.0, 0), "+2 px");
        assert_eq!(signed_px(-3.0, 0), "\u{2212}3 px");
        assert_eq!(signed_px(0.0, 0), "0 px");
        assert_eq!(signed_px(-0.04, 1), "0 px", "rounds to zero, so no sign");
        assert_eq!(signed_plain(0.25, 2), "+0.25");
        assert_eq!(signed_plain(-0.25, 2), "\u{2212}0.25");
    }
}
