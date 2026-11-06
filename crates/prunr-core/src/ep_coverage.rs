//! How much of a session's graph CoreML took, read from ONNX Runtime's
//! own log. CoreML's `GetCapability` reports it once per session, at
//! INFO for one partition and WARNING for more. Every partition edge is
//! a round trip between the CPU and the GPU / Neural Engine, so the
//! counts say whether CoreML pays for a model.

use std::sync::{Arc, OnceLock};

use ort::logging::LogLevel;
use ort::session::builder::SessionBuilder;

use crate::engine::EpKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Coverage {
    pub partitions: usize,
    pub nodes: usize,
    pub claimed: usize,
}

const CAPABILITY_PREFIX: &str = "CoreMLExecutionProvider::GetCapability,";

/// CoreML's summary line, e.g. "CoreMLExecutionProvider::GetCapability,
/// number of partitions supported by CoreML: 14 number of nodes in the
/// graph: 870 number of nodes supported by CoreML: 812".
fn parse_capability(message: &str) -> Option<Coverage> {
    let rest = message.split_once(CAPABILITY_PREFIX)?.1;
    let count = |label: &str| -> Option<usize> {
        rest.split_once(label)?.1.split_whitespace().next()?.parse().ok()
    };
    Some(Coverage {
        partitions: count("number of partitions supported by CoreML:")?,
        nodes: count("number of nodes in the graph:")?,
        claimed: count("number of nodes supported by CoreML:")?,
    })
}

/// The session's log handler. It replaces ORT's default sink, which
/// prints errors only, so errors pass on to `tracing`.
fn on_ort_log(seen: &OnceLock<Coverage>, level: LogLevel, message: &str) {
    if let Some(coverage) = parse_capability(message) {
        let _ = seen.set(coverage);
    } else if matches!(level, LogLevel::Error | LogLevel::Fatal) {
        tracing::error!(%message, "ORT");
    }
}

/// A session's coverage, readable once it has committed. Empty for a
/// provider that reports none.
#[derive(Default)]
pub(crate) struct CoverageWatch(Option<Arc<OnceLock<Coverage>>>);

impl CoverageWatch {
    pub(crate) fn report(&self, model: prunr_models::ModelId, part: Option<&str>) {
        let Some(seen) = &self.0 else { return };
        match seen.get() {
            Some(c) => tracing::info!(
                ?model, part, ep = "CoreML",
                partitions = c.partitions, nodes = c.nodes, coreml_nodes = c.claimed,
                "CoreML coverage",
            ),
            None => tracing::info!(?model, part, ep = "CoreML", "CoreML coverage not reported"),
        }
    }
}

impl EpKind {
    fn reports_coverage(self) -> bool {
        match self {
            #[cfg(target_os = "macos")]
            EpKind::CoreMl => true,
            #[cfg(not(target_os = "macos"))]
            _ => false,
        }
    }
}

/// Hooks `builder`'s log for `ep` so the committed session's coverage
/// can be reported. Leaves the builder as it was for other providers,
/// or when ORT refuses the hook.
pub(crate) fn watch(ep: EpKind, builder: SessionBuilder) -> (SessionBuilder, CoverageWatch) {
    if !ep.reports_coverage() {
        return (builder, CoverageWatch::default());
    }
    let seen = Arc::new(OnceLock::new());
    let sink = Arc::clone(&seen);
    let logger: ort::logging::LoggerFunction = Arc::new(
        move |level: LogLevel, _category: &str, _id: &str, _location: &str, message: &str| {
            on_ort_log(&sink, level, message)
        },
    );
    match builder.with_logger(logger).and_then(|b| b.with_log_level(LogLevel::Info)) {
        Ok(b) => (b, CoverageWatch(Some(seen))),
        Err(e) => {
            tracing::debug!(ep = %ep, %e, "coverage log not attached");
            (e.recover(), CoverageWatch::default())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SINGLE: &str = "CoreMLExecutionProvider::GetCapability, number of partitions supported by CoreML: 1 \
                          number of nodes in the graph: 312 number of nodes supported by CoreML: 312";
    const MULTI: &str = "CoreMLExecutionProvider::GetCapability, number of partitions supported by CoreML: 14 \
                         number of nodes in the graph: 870 number of nodes supported by CoreML: 812";

    #[test]
    fn records_what_coreml_reports_at_either_level_and_nothing_else() {
        let unrelated = [
            (LogLevel::Info, "Successfully registered `CoreMLExecutionProvider`"),
            (LogLevel::Warning, "Some nodes were not assigned to the preferred execution providers"),
            (LogLevel::Verbose, "Operator type: [Pad] index: [7] name: [pad1] supported: [0]"),
        ];
        let cases = [
            ("single partition at INFO", Some((LogLevel::Info, SINGLE)),
             Some(Coverage { partitions: 1, nodes: 312, claimed: 312 })),
            ("several partitions at WARNING", Some((LogLevel::Warning, MULTI)),
             Some(Coverage { partitions: 14, nodes: 870, claimed: 812 })),
            ("no summary line", None, None),
        ];
        for (name, summary, expected) in cases {
            let seen = OnceLock::new();
            for (level, message) in unrelated.into_iter().chain(summary) {
                on_ort_log(&seen, level, message);
            }
            assert_eq!(seen.get().copied(), expected, "{name}");
        }
    }

    #[test]
    fn a_truncated_summary_is_not_a_coverage() {
        assert_eq!(
            parse_capability("CoreMLExecutionProvider::GetCapability, number of partitions supported by CoreML: 2"),
            None,
        );
    }
}
