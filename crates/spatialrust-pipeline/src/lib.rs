//! Composable processing pipelines for SpatialRust.

#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "pipeline-mvp")]
mod mvp;

#[cfg(feature = "pipeline-mvp")]
pub use mvp::{
    MvpIcpConfig, MvpPipeline, MvpPipelineConfig, MvpPipelineResult, MvpRegistrationMethod,
};

#[cfg(feature = "pipeline-streaming")]
mod streaming;
#[cfg(feature = "pipeline-streaming")]
mod workflow;

#[cfg(feature = "pipeline-streaming")]
pub use streaming::{
    reduce_positions, ChunkMapOperation, ChunkMapSource, PositionReduction, StreamingVoxelConfig,
    StreamingVoxelSource,
};
#[cfg(feature = "pipeline-streaming")]
pub use workflow::{StreamingPipeline, StreamingPipelineIter};
