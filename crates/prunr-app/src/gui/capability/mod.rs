//! What an image shows and what may run on it: the one answer every
//! control, shortcut and dispatch reads. Facts are derived from the item
//! each time, never stored, so nothing can fall out of step with them.

mod check;
mod facts;

pub(crate) use check::{check, Need, Reason, Verdict};
pub(crate) use facts::ItemFacts;
#[cfg(test)]
pub(crate) use facts::{Activity, Shown};
