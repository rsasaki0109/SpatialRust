//! Platform stability, conformance, security checklists, performance budgets, and LTS policy.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod budget;
mod conformance;
mod error;
mod gate;
mod lts;
mod security;
mod stability;
mod streaming;
mod vision;
mod vision2;
mod visual;

pub use budget::{BudgetKind, PerformanceBudget, PerformanceBudgetReport, PerformanceSample};
pub use conformance::{ConformanceCase, ConformanceReport, ConformanceStatus};
pub use error::{PlatformError, PlatformResult};
pub use gate::{ReleaseGate, ReleaseGateDecision};
pub use lts::{LtsPolicy, SupportWindow};
pub use security::{SecurityAuditItem, SecurityChecklist};
pub use stability::{ApiStabilityClass, ApiSurfaceItem, StabilityRegistry};
pub use streaming::{Streaming12Measurements, Streaming12ReleaseEvidence, Streaming12ReleaseGate};
pub use vision::{Vision1Measurements, Vision1ReleaseEvidence, Vision1ReleaseGate};
pub use vision2::{Vision2Measurements, Vision2ReleaseEvidence, Vision2ReleaseGate};
pub use visual::{
    VisualMeasurements, VisualReceiptEvidence, VisualReleaseEvidence, VisualReleaseGate,
};
