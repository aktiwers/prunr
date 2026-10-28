//! Value text for chip faces and slider boxes. One place decides how a
//! number reads so every control agrees: "Off" capitalized, a space before
//! px and EV, none before %, an explicit sign where direction matters, and
//! no Greek prefixes.

/// "1 image", "3 images".
pub fn count(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// "today", "yesterday", "3 days ago": how long before `now` the time
/// `then` was, both in seconds since 1970.
pub fn days_ago(then: u64, now: u64) -> String {
    match now.saturating_sub(then) / 86_400 {
        0 => "today".to_string(),
        1 => "yesterday".to_string(),
        days => format!("{days} days ago"),
    }
}

/// A run's time per picture: "0.04 s" under a second, "10.1 s" above;
/// "3.8 ms" where two decimals of a second would read nothing, and
/// "2m 06s" past a minute.
pub fn seconds(ms: f32) -> String {
    match ms {
        ms if !(10.0..60_000.0).contains(&ms) => prunr_models::fmt_ms(f64::from(ms)),
        ms if ms < 1000.0 => format!("{:.2} s", ms / 1000.0),
        ms => format!("{:.1} s", ms / 1000.0),
    }
}

/// An engine as a sentence says it: "the processor" for the CPU provider,
/// else the provider's own name.
pub fn engine(provider: &str) -> &str {
    if provider == prunr_core::engine::CPU_NAME { "the processor" } else { provider }
}

/// An engine as a table names it: "Processor", or the provider's name.
pub fn engine_heading(provider: &str) -> &str {
    if provider == prunr_core::engine::CPU_NAME { "Processor" } else { provider }
}

/// `text` with its first letter capitalized: a phrase opening a sentence.
pub fn capitalized(text: &str) -> String {
    let mut chars = text.chars();
    chars.next().map(|first| first.to_uppercase().chain(chars).collect()).unwrap_or_default()
}

/// "CUDA", "DirectML and CUDA", "the processor, DirectML and CUDA".
pub fn and_list<'a>(items: impl IntoIterator<Item = &'a str>) -> String {
    let items: Vec<&str> = items.into_iter().collect();
    match items.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
        _ => items.concat(),
    }
}

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
    if (v * 10f32.powi(decimals as i32)).round() == 0.0 {
        return format!("0{unit}");
    }
    let sign = if v > 0.0 { "+" } else { "\u{2212}" };
    format!("{sign}{:.decimals$}{unit}", v.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_ago_reads_today_yesterday_then_a_count() {
        let now = 1_000_000_000;
        assert_eq!(days_ago(now - 60, now), "today");
        assert_eq!(days_ago(now - 86_400, now), "yesterday");
        assert_eq!(days_ago(now - 3 * 86_400 - 5, now), "3 days ago");
        assert_eq!(days_ago(now + 60, now), "today", "a clock set back is not the future");
    }

    #[test]
    fn seconds_keep_two_figures_under_a_second() {
        assert_eq!(seconds(40.0), "0.04 s");
        assert_eq!(seconds(450.0), "0.45 s");
        assert_eq!(seconds(10_100.0), "10.1 s");
        assert_eq!(seconds(126_000.0), "2m 06s");
        assert_eq!(seconds(3.84), "3.8 ms", "the face finder's time is not nothing");
    }

    #[test]
    fn capitalized_opens_a_sentence() {
        assert_eq!(capitalized("the graphics driver"), "The graphics driver");
        assert_eq!(capitalized(""), "");
    }

    #[test]
    fn and_list_joins_the_last_with_and() {
        assert_eq!(and_list([]), "");
        assert_eq!(and_list(["CUDA"]), "CUDA");
        assert_eq!(and_list(["DirectML", "CUDA"]), "DirectML and CUDA");
        assert_eq!(and_list(["the processor", "DirectML", "CUDA"]), "the processor, DirectML and CUDA");
    }

    #[test]
    fn count_says_one_image_and_two_images() {
        assert_eq!(count(1, "image", "images"), "1 image");
        assert_eq!(count(0, "image", "images"), "0 images");
        assert_eq!(count(2, "image", "images"), "2 images");
    }

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
