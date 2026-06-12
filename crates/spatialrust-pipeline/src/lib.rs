//! Composable processing pipelines for SpatialRust.

#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "pipeline-mvp")]
mod mvp;

#[cfg(feature = "pipeline-mvp")]
pub use mvp::{
    MvpIcpConfig, MvpPipeline, MvpPipelineConfig, MvpPipelineResult,
};
