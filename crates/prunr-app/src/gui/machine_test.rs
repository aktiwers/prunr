//! Settings › Hardware › Test this computer, followed from the window.
//! The test runs in a process of its own (`prunr bench --here`) and owns
//! the machine while it runs: a run of the user's would skew its timings
//! and take the memory it measures with, so the window shows it as a
//! modal until it ends.

use std::io::BufRead;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;

use prunr_core::ep_compat::{Measured, PlaceKind};
use prunr_models::{Column, ModelId};

use crate::bench::here::Told;

/// Where the test is: the model at `index` of `total`, on the GPU side or
/// the CPU's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Step {
    pub index: usize,
    pub total: usize,
    pub model: ModelId,
    pub on_gpu: bool,
    pub column: Column,
}

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
    /// Its output closed: the process is ending.
    closed: bool,
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
        Ok(Self { child: Some(child), told, step: None, finished: None, closed: false })
    }

    /// A test whose process is `child` and that reports nothing.
    #[cfg(test)]
    pub(crate) fn for_test(child: Child) -> Self {
        let (_, told) = mpsc::channel();
        Self { child: Some(child), told, step: None, finished: None, closed: false }
    }

    pub(crate) fn step(&self) -> Option<Step> {
        self.step
    }

    /// Take in what the test told; `Some` once its process has ended.
    pub(crate) fn pump(&mut self) -> Option<Ended> {
        loop {
            match self.told.try_recv() {
                Ok(Told::Step { index, total, model, on_gpu, column }) => {
                    self.step = Some(Step { index, total, model, on_gpu, column });
                }
                Ok(Told::Finished { left_out }) => self.finished = Some(left_out),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.closed = true;
                    break;
                }
            }
        }
        let exited = self.child.as_mut().is_some_and(|child| matches!(child.try_wait(), Ok(Some(_))));
        if !self.closed || !exited {
            return None;
        }
        Some(match self.finished.take() {
            Some(left_out) => Ended::Finished { left_out },
            None => Ended::Stopped { during: self.step },
        })
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

/// The place a test that stopped during `step` leaves: a process that died
/// running a model on the graphics card would die running it in the app.
pub(crate) fn stopped_place(step: Step) -> Option<(ModelId, Column, Measured)> {
    step.on_gpu.then(|| (step.model, step.column, Measured { kind: PlaceKind::Fails, text: "the test's process stopped while it ran here".into() }))
}
