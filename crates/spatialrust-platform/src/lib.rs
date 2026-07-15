//! Platform stability, conformance, security checklists, and LTS policy.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod conformance;
mod error;
mod lts;
mod security;
mod stability;

pub use conformance::{ConformanceCase, ConformanceReport, ConformanceStatus};
pub use error::{PlatformError, PlatformResult};
pub use lts::{LtsPolicy, SupportWindow};
pub use security::{SecurityAuditItem, SecurityChecklist};
pub use stability::{ApiStabilityClass, ApiSurfaceItem, StabilityRegistry};
