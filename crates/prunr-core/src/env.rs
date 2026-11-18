//! Switches read from the environment, for experiments and the harness.

/// `name` as a switch: 1/true/yes/on or 0/false/no/off in any case,
/// surrounding blanks ignored. `None` when unset or anything else.
pub fn flag(name: &str) -> Option<bool> {
    parse_flag(&std::env::var(name).ok()?)
}

fn parse_flag(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::parse_flag;

    #[test]
    fn a_flag_reads_its_words_in_any_case() {
        for on in ["1", "true", "YES", " On "] { assert_eq!(parse_flag(on), Some(true), "{on:?}"); }
        for off in ["0", "False", "no", "off\n"] { assert_eq!(parse_flag(off), Some(false), "{off:?}"); }
        for other in ["", "2", "maybe", "y"] { assert_eq!(parse_flag(other), None, "{other:?}"); }
    }
}
