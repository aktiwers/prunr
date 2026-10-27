//! Settings › Hardware › Test this computer, followed from the window.
//! The test runs in a process of its own (`prunr bench --here`) and owns
//! the machine while it runs: a run of the user's would skew its timings
//! and take the memory it measures with, so the window shows it as a
//! modal until it ends. A download's check (`bench --check`) runs the
//! same way, in the background.

use std::io::BufRead;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;

use prunr_core::ep_compat::{Measured, PlaceKind};
use prunr_models::{Column, ModelId};

use crate::bench::here::{Step, Told};

/// How a test ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Ended {
    /// Every model it runs was placed; `left_out` it does not run. A
    /// check ends this way too.
    Finished { left_out: Vec<ModelId> },
    /// Its process stopped during `during` (`None`: before the first).
    Stopped { during: Option<Step> },
}

pub(crate) struct MachineTest {
    /// Taken when it is dropped, to be waited on off the UI thread.
    child: Option<Child>,
    told: mpsc::Receiver<Told>,
    step: Option<Step>,
    finished: Option<Vec<ModelId>>,
    /// What it measured so far, a model at a time.
    measured: Vec<Measuring>,
}

/// A model the test measured: its time on the processor, and its fastest
/// engine that gives the processor's picture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Measuring {
    pub model: ModelId,
    pub cpu_ms: Option<f32>,
    /// The engine's provider name and its time.
    pub fastest: Option<(&'static str, f32)>,
}

impl Measuring {
    /// Take in what a run on `column` found.
    fn add(&mut self, column: Column, measured: &Measured) {
        self.cpu_ms = self.cpu_ms.or(measured.cpu_ms);
        let Some(ms) = measured.ms.filter(|_| measured.kind == PlaceKind::There) else { return };
        if self.fastest.is_none_or(|(_, best)| ms < best) {
            self.fastest = Some((column.provider(), ms));
        }
    }
}

impl MachineTest {
    /// Start the test's process; `wake` repaints the window as it reports.
    pub(crate) fn start(wake: impl Fn() + Send + 'static) -> Result<Self, String> {
        Self::spawn(&["bench", "--here"], wake)
    }

    /// Start the check of a download's result (`bench::here::check`).
    pub(crate) fn check(model: ModelId, wake: impl Fn() + Send + 'static) -> Result<Self, String> {
        Self::spawn(&["bench", "--check", model.stable_name()], wake)
    }

    fn spawn(args: &[&str], wake: impl Fn() + Send + 'static) -> Result<Self, String> {
        let exe = crate::subprocess::manager::worker_exe()?;
        let mut child = Command::new(exe)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("The test could not start: {e}"))?;
        let stdout = child.stdout.take().ok_or("The test could not start: no output")?;
        let (tx, told) = mpsc::channel();
        std::thread::Builder::new()
            .name("machine-test".into())
            .spawn(move || {
                for line in std::io::BufReader::new(stdout).lines().map_while(Result::ok) {
                    if let Some(event) = Told::parse(&line) {
                        if tx.send(event).is_err() {
                            break;
                        }
                        wake();
                    }
                }
                wake();
            })
            .map_err(|e| format!("The test could not start: {e}"))?;
        Ok(Self { child: Some(child), told, step: None, finished: None, measured: Vec::new() })
    }

    /// A test whose process is `child` and that reports nothing.
    #[cfg(test)]
    pub(crate) fn for_test(child: Child) -> Self {
        let (_, told) = mpsc::channel();
        Self { child: Some(child), told, step: None, finished: None, measured: Vec::new() }
    }

    pub(crate) fn step(&self) -> Option<Step> {
        self.step
    }

    /// The models it has finished, in the order it ran them.
    pub(crate) fn done(&self) -> impl Iterator<Item = &Measuring> {
        let now = self.step.map(|s| s.model);
        self.measured.iter().filter(move |m| Some(m.model) != now)
    }

    /// Take in what the test told; `Some` once its process has ended.
    pub(crate) fn pump(&mut self) -> Option<Ended> {
        let closed = loop {
            match self.told.try_recv() {
                Ok(Told::Step(step)) => self.step = Some(step),
                Ok(Told::Placed { model, column, measured }) => self.place(model, column, &measured),
                Ok(Told::Finished { left_out }) => self.finished = Some(left_out),
                Err(mpsc::TryRecvError::Empty) => break false,
                Err(mpsc::TryRecvError::Disconnected) => break true,
            }
        };
        // Its output closes as it ends; until then there is nothing to wait for.
        let exited = closed && self.child.as_mut().is_some_and(|child| matches!(child.try_wait(), Ok(Some(_))));
        if !exited {
            return None;
        }
        Some(match self.finished.take() {
            Some(left_out) => Ended::Finished { left_out },
            None => Ended::Stopped { during: self.step },
        })
    }
}

impl MachineTest {
    fn place(&mut self, model: ModelId, column: Column, measured: &Measured) {
        let at = match self.measured.iter().position(|m| m.model == model) {
            Some(at) => at,
            None => {
                self.measured.push(Measuring { model, cpu_ms: None, fastest: None });
                self.measured.len() - 1
            }
        };
        self.measured[at].add(column, measured);
    }
}

impl Drop for MachineTest {
    /// Cancel, or close the app mid-test: the process stops; what it
    /// measured so far is already saved.
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            std::thread::spawn(move || child.wait());
        }
    }
}

/// The place a model gets when Prunr's test or check stopped while it ran
/// on the accelerator: a process that died running it there would die
/// running it in the app.
pub(crate) fn stopped_here() -> Measured {
    Measured::new(PlaceKind::Fails, "Prunr's test stopped while it ran here")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model's line keeps the processor's time from any run, and the
    /// fastest engine whose picture matched: never one that differs, nor
    /// one slower than the processor.
    #[test]
    fn a_finished_model_shows_its_fastest_engine_with_the_same_picture() {
        let mut m = Measuring { model: ModelId::Silueta, cpu_ms: None, fastest: None };
        m.add(Column::Cuda, &Measured::new(PlaceKind::Differs, "x").with_times(Some(10.0), Some(670.0)));
        assert_eq!((m.cpu_ms, m.fastest), (Some(670.0), None));
        m.add(Column::DirectMl, &Measured::timed(PlaceKind::There, 40.0, 670.0, ""));
        m.add(Column::OpenVino, &Measured::timed(PlaceKind::Slower, 900.0, 670.0, ""));
        m.add(Column::MiGraphX, &Measured::timed(PlaceKind::There, 57.0, 670.0, ""));
        assert_eq!(m.fastest, Some(("DirectML", 40.0)));
    }
}
