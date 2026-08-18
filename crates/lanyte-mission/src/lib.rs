//! Domain contracts and invariants for durable Lanyte missions.
//!
//! This crate contains no persistence, transport, process control, or harness
//! implementation. It mirrors the versioned mission-v0 contracts and provides
//! pure validation and transition rules for downstream control-plane code.
//! JSON inputs must first pass the corresponding Crucible schema so required
//! nullable fields and canonical lexical forms are checked before deserialization.

mod control;
mod driver;
mod invariant;
mod model;
mod transition;
mod wave3;

pub use control::*;
pub use driver::{DriverDescriptor, HarnessDriver, NormalizedHarnessEvent};
pub use invariant::{validate_history, InvariantError, Validate};
pub use model::*;
pub use transition::{AttemptTransition, MissionTransition, TransitionError};
pub use wave3::semantic_violation_codes;
