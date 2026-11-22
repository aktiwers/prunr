//! Asks the version check service whether a newer Prunr is out: at most
//! once a day of UTC's calendar, some time after the app has started, off
//! the UI thread. The request names Prunr's version, the operating system
//! and its version where known, and how many days ago this computer last
//! got an answer (0: never); from that number the service counts the
//! computers in use without telling one from another. The day of the last
//! answered check is kept in the settings with the version that asked: a
//! Prunr updated later that day asks once more and names the version it
//! ran before, so the service counts the computer once, on its new version.
//! A newer version shows in Settings › About and is told in a notice when
//! it is first seen, then at most once a week while it is out. With the
//! switch off nothing is asked and nothing newer is known. A check that
//! fails says nothing and is made again on the next day or the next start.
//! The command line never asks.

use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The service's address, without a path or a slash at its end.
#[cfg(not(test))]
const CHECK_URL: &str = "https://check.bolli-version.workers.dev";
/// Where a version's download is.
const RELEASE_URL: &str = "https://github.com/aktiwers/prunr/releases/tag/v";
/// A check waits this long after the app has started.
const START_AFTER: Duration = Duration::from_secs(30);
/// The service answers at once: a request still open after this long is
/// given up.
#[cfg(not(test))]
const TIMEOUT: Duration = Duration::from_secs(5);
/// The longest time since the last check a request names, in days (a
/// little over three years): longer ago is said as this.
const GAP_MAX: i64 = 1100;
/// A notice of a version already told comes again no sooner than this
/// many days after.
const NOTICE_DAYS: i64 = 7;
/// A system's version as a check names it: one to four numbers, no longer
/// than this.
const OS_VERSION_LENGTH: usize = 15;
const SECONDS_A_DAY: u64 = 86_400;

/// This build's version.
pub(crate) const VERSION: &str = env!("CARGO_PKG_VERSION");

/// What is kept between runs: the UTC day (days since 1970) of the last
/// answered check, the latest version it named (`""`: none), the newer
/// version last told of and the day it was told, and Prunr's version the
/// check was sent with.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Kept {
    #[serde(default)]
    pub day: Option<i64>,
    #[serde(default)]
    pub latest: String,
    #[serde(default)]
    pub told: String,
    #[serde(default)]
    pub told_day: Option<i64>,
    #[serde(default)]
    pub sent_with: String,
}

impl Kept {
    /// The newer version the last answer named, if there is one.
    pub fn newer(&self) -> Option<&str> {
        newer(&self.latest, VERSION).then_some(self.latest.as_str())
    }
}

/// Where version `version` is downloaded.
pub fn release_page(version: &str) -> String {
    format!("{RELEASE_URL}{version}")
}

/// Today by UTC's calendar, in days since 1970.
pub fn today() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |since| (since.as_secs() / SECONDS_A_DAY) as i64)
}

/// How long until the next UTC day begins.
fn until_tomorrow() -> Duration {
    let since = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |since| since.as_secs());
    Duration::from_secs(SECONDS_A_DAY - since % SECONDS_A_DAY)
}

/// Three numbers: a version a check may name.
fn is_version(text: &str) -> bool {
    parts(text).is_some()
}

fn parts(text: &str) -> Option<(u32, u32, u32)> {
    let mut numbers = text.split('.').map(|part| (!part.is_empty() && part.bytes().all(|b| b.is_ascii_digit())).then(|| part.parse::<u32>().ok()).flatten());
    let version = (numbers.next()??, numbers.next()??, numbers.next()??);
    numbers.next().is_none().then_some(version)
}

/// Whether `version` is a later one than `than`.
fn newer(version: &str, than: &str) -> bool {
    matches!((parts(version), parts(than)), (Some(a), Some(b)) if a > b)
}

/// `said` if it is a system's version as a check may name it, else `""`.
fn system_version(said: &str) -> &str {
    let numbers = said.split('.').count();
    let fits = said.len() <= OS_VERSION_LENGTH && (1..=4).contains(&numbers) && said.split('.').all(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
    if fits { said } else { "" }
}

/// The operating system as a check names it, and its version where known:
/// read once (Linux reads a file for it).
fn system() -> (&'static str, &'static str) {
    static SYSTEM: std::sync::OnceLock<(&'static str, String)> = std::sync::OnceLock::new();
    let (os, version) = SYSTEM.get_or_init(|| {
        let os = if cfg!(target_os = "macos") { "macos" } else if cfg!(windows) { "windows" } else { "linux" };
        // Windows names itself "11 (26100)"; its build number is the number.
        let said = if cfg!(windows) { sysinfo::System::kernel_version() } else { sysinfo::System::os_version() };
        (os, system_version(&said.unwrap_or_default()).to_string())
    });
    (os, version)
}

/// The query of a check from `kept` on `today` by Prunr `version` on `os`
/// (version `os_version`, `""` when unknown), or `None` when none is due:
/// one a day, and one more on the day of an update, naming the version
/// the day's first check was sent with.
fn query(kept: &Kept, today: i64, version: &str, os: &str, os_version: &str) -> Option<String> {
    let updated_today = kept.day == Some(today) && is_version(&kept.sent_with) && kept.sent_with != version;
    if kept.day == Some(today) && !updated_today {
        return None;
    }
    let gap = kept.day.map_or(0, |day| (today - day).clamp(1, GAP_MAX));
    let mut query = format!("app=prunr&v={version}&os={os}");
    if !os_version.is_empty() {
        query.push_str(&format!("&ov={os_version}"));
    }
    query.push_str(&format!("&g={gap}&n=1"));
    if updated_today {
        query.push_str(&format!("&from={}", kept.sent_with));
    }
    Some(query)
}

/// The latest version an answer names: `Some("")` when the service knows
/// none, `None` when it is no answer of the service.
fn latest_of(body: &str) -> Option<String> {
    let latest = serde_json::from_str::<serde_json::Value>(body).ok()?.get("latest")?.as_str()?.to_string();
    (latest.is_empty() || is_version(&latest)).then_some(latest)
}

/// What is kept after an answer naming `latest` on `today` to Prunr
/// `version`, and whether to tell of a newer version now: at once when it
/// is not the one told last, else once `NOTICE_DAYS` have gone by.
fn landed(kept: &Kept, latest: &str, today: i64, version: &str) -> (Kept, bool) {
    let mut next = Kept { day: Some(today), latest: latest.to_string(), sent_with: version.to_string(), ..kept.clone() };
    let due = kept.told != latest || kept.told_day.is_none_or(|day| today - day >= NOTICE_DAYS);
    let tell = newer(latest, version) && due;
    if tell {
        next.told = latest.to_string();
        next.told_day = Some(today);
    }
    (next, tell)
}

/// The check of this run: when it may look, and the answer on its way.
pub struct UpdateCheck {
    /// No look before this: the start's wait, then the next UTC day. A
    /// look that asked, answered or not, waits for the next day.
    look_at: Instant,
    asking: Option<mpsc::Receiver<Option<String>>>,
    /// What tests' checks asked, with the way to answer each.
    #[cfg(test)]
    pub(crate) sent: Vec<(String, mpsc::Sender<Option<String>>)>,
}

impl UpdateCheck {
    pub fn new() -> Self {
        Self {
            look_at: Instant::now() + START_AFTER,
            asking: None,
            #[cfg(test)]
            sent: Vec::new(),
        }
    }

    /// Asks the service when a check is due (`on`: the setting); until the
    /// next look, only wakes the window for it.
    pub fn ask_when_due(&mut self, on: bool, kept: &Kept, ctx: &egui::Context) {
        if !on || self.asking.is_some() {
            return;
        }
        let now = Instant::now();
        if now < self.look_at {
            ctx.request_repaint_after(self.look_at - now);
            return;
        }
        self.look_at = now + until_tomorrow();
        let (os, os_version) = system();
        if let Some(query) = query(kept, today(), VERSION, os, os_version) {
            self.ask(query, ctx);
        }
    }

    /// Tests: the next look is now.
    #[cfg(test)]
    pub(crate) fn due_now(&mut self) {
        self.look_at = Instant::now();
    }

    /// The answer of the check on its way, once it is in: `Some(latest)`
    /// when the service answered.
    pub fn answer(&mut self) -> Option<Option<String>> {
        let answer = match self.asking.as_ref()?.try_recv() {
            Ok(answer) => answer,
            Err(mpsc::TryRecvError::Empty) => return None,
            Err(mpsc::TryRecvError::Disconnected) => None,
        };
        self.asking = None;
        Some(answer)
    }

    #[cfg(not(test))]
    fn ask(&mut self, query: String, ctx: &egui::Context) {
        let (tx, rx) = mpsc::channel();
        self.asking = Some(rx);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(ask_service(&query));
            ctx.request_repaint();
        });
    }

    /// Tests ask nothing: the query is kept, with the way to answer it.
    #[cfg(test)]
    fn ask(&mut self, query: String, _ctx: &egui::Context) {
        let (tx, rx) = mpsc::channel();
        self.asking = Some(rx);
        self.sent.push((query, tx));
    }
}

impl Default for UpdateCheck {
    fn default() -> Self {
        Self::new()
    }
}

/// The latest version the service names for `query`, `None` when it does
/// not answer as the service does.
#[cfg(not(test))]
fn ask_service(query: &str) -> Option<String> {
    let client = prunr_runtime_install::http_client(crate::runtime_install::USER_AGENT).ok()?;
    let reply = client.get(format!("{CHECK_URL}/v1/check?{query}")).timeout(TIMEOUT).send().ok()?;
    if !reply.status().is_success() {
        tracing::debug!(status = %reply.status(), "version check not answered");
        return None;
    }
    latest_of(&reply.text().ok()?)
}

/// Lands an answer on the kept state: the notice it gives, if any.
pub fn notice(kept: &mut Kept, latest: &str) -> Option<String> {
    let (next, tell) = landed(kept, latest, today(), VERSION);
    *kept = next;
    tell.then(|| format!("Prunr {latest} is out. Settings › About has the download."))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: i64 = 20_400;

    /// The first check says it never asked; the next day's says one day;
    /// one long ago says at most three years; no second check on a day.
    #[test]
    fn a_check_names_the_days_since_the_last_answer_once_a_day() {
        let never = Kept::default();
        assert_eq!(query(&never, DAY, "0.5.0", "linux", "24.04").as_deref(), Some("app=prunr&v=0.5.0&os=linux&ov=24.04&g=0&n=1"));
        let yesterday = Kept { day: Some(DAY - 1), sent_with: "0.5.0".into(), ..Kept::default() };
        assert_eq!(query(&yesterday, DAY, "0.5.0", "macos", "").as_deref(), Some("app=prunr&v=0.5.0&os=macos&g=1&n=1"));
        let long_ago = Kept { day: Some(DAY - 5000), ..yesterday.clone() };
        assert!(query(&long_ago, DAY, "0.5.0", "windows", "26100").is_some_and(|q| q.contains("&g=1100&")));
        let today = Kept { day: Some(DAY), ..yesterday };
        assert_eq!(query(&today, DAY, "0.5.0", "linux", ""), None, "once a day");
    }

    /// Updated on the day of its check, Prunr asks once more, naming the
    /// version it ran before; then not again that day.
    #[test]
    fn an_update_on_the_day_of_a_check_asks_once_more() {
        let checked = Kept { day: Some(DAY), sent_with: "0.5.0".into(), ..Kept::default() };
        assert!(query(&checked, DAY, "0.5.1", "linux", "").is_some_and(|q| q.ends_with("&from=0.5.0")));
        let (after, _) = landed(&checked, "0.5.1", DAY, "0.5.1");
        assert_eq!(query(&after, DAY, "0.5.1", "linux", ""), None);
    }

    /// A newer version is told when first seen, then at most once a week;
    /// the latest one, or an older one, is not told at all.
    #[test]
    fn a_newer_version_is_told_at_once_then_once_a_week() {
        let (kept, told) = landed(&Kept::default(), "0.5.1", DAY, "0.5.0");
        assert!(told);
        assert_eq!((kept.told.as_str(), kept.told_day), ("0.5.1", Some(DAY)));
        assert!(!landed(&kept, "0.5.1", DAY + 6, "0.5.0").1, "not within a week");
        assert!(landed(&kept, "0.5.1", DAY + 7, "0.5.0").1, "again after a week");
        assert!(landed(&kept, "0.5.2", DAY + 1, "0.5.0").1, "a newer one at once");
        assert!(!landed(&Kept::default(), "0.5.0", DAY, "0.5.0").1, "the version running");
        assert!(!landed(&Kept::default(), "", DAY, "0.5.0").1, "none known");
    }

    #[test]
    fn versions_compare_by_their_numbers() {
        assert!(newer("0.10.0", "0.9.9") && newer("1.0.0", "0.99.99") && !newer("0.5.0", "0.5.0"));
        assert!(!newer("0.5", "0.4.0") && !newer("0.5.0-rc1", "0.4.0") && !newer("v0.6.0", "0.5.0"), "not versions");
    }

    #[test]
    fn only_the_services_answer_is_read() {
        assert_eq!(latest_of(r#"{"latest":"0.5.1"}"#).as_deref(), Some("0.5.1"));
        assert_eq!(latest_of(r#"{"latest":""}"#).as_deref(), Some(""));
        assert_eq!(latest_of(r#"{"latest":"<script>"}"#), None);
        assert_eq!(latest_of("<html>"), None);
    }

    #[test]
    fn a_system_version_is_named_only_as_numbers() {
        assert_eq!(system_version("24.04"), "24.04");
        assert_eq!(system_version("10.0.26100.4652"), "10.0.26100.4652");
        assert_eq!(system_version("11 (26100)"), "");
        assert_eq!(system_version("1.2.3.4.5"), "");
        assert_eq!(system_version("rolling"), "");
    }

    /// Off, nothing is asked; on, nothing before the start's wait is over,
    /// then one check, and after its answer none until the next day.
    #[test]
    fn nothing_is_asked_with_the_switch_off_before_the_wait_or_twice_a_day() {
        let ctx = egui::Context::default();
        let mut check = UpdateCheck::new();
        check.ask_when_due(true, &Kept::default(), &ctx);
        assert!(check.sent.is_empty(), "the start's wait");
        check.look_at = Instant::now();
        check.ask_when_due(false, &Kept::default(), &ctx);
        assert!(check.sent.is_empty(), "the switch off");
        check.ask_when_due(true, &Kept::default(), &ctx);
        assert_eq!(check.sent.len(), 1);
        assert_eq!(check.answer(), None, "not in yet");
        check.sent[0].1.send(Some("0.9.0".into())).unwrap();
        assert_eq!(check.answer(), Some(Some("0.9.0".into())));
        check.ask_when_due(true, &Kept::default(), &ctx);
        assert_eq!(check.sent.len(), 1, "the next look is tomorrow");
    }
}
