//! Bounded pipelines, tracing, diagnostics, and ROS 2 adaptation contracts.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod diagnostics;
mod error;
mod pipeline;
mod trace;

#[cfg(feature = "ros2")]
mod ros2;

pub use diagnostics::{DiagnosticCode, FailureDiagnostic};
pub use error::{RuntimeError, RuntimeResult};
pub use pipeline::{BoundedPipeline, PipelineConfig, PipelineStage};
pub use trace::{TraceEvent, TraceLevel, TraceLog};

#[cfg(feature = "ros2")]
pub use ros2::{CatalogRos2Adapter, Ros2Adapter, Ros2MessageHint};
