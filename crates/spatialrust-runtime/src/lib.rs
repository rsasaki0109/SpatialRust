//! Bounded pipelines, tracing, diagnostics, and ROS 2 CDR/loopback adapters.
//!
//! Enable `ros2` for PointCloud2 CDR codecs and negotiation without linking
//! `rclrs`. Native ROS 2 executors remain an install-time toolchain concern.

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
pub use ros2::{
    decode_point_cloud2_xyz, encode_point_cloud2_xyz, CatalogRos2Adapter, LoopbackRos2Node,
    PointCloud2Xyz, Ros2Adapter, Ros2MessageHint, POINT_CLOUD2_TYPE,
};
