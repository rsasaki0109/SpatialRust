//! Core data model, metadata, and algorithm traits for SpatialRust.
//!
//! This crate intentionally stays lightweight: no IO, GPU, ROS2, or AI runtimes.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod algorithm;
mod buffer;
mod capabilities;
mod device;
mod error;
mod execution;
mod metadata;
mod pointcloud;
mod schema;
mod tensor;

#[cfg(feature = "tensor-aoso")]
mod tensor_aoso;

pub use algorithm::SpatialAlgorithm;
pub use buffer::{PointBuffer, PointBufferSet};
pub use capabilities::{HasIntensity, HasNormals3, HasPositions3};
pub use device::{CpuDevice, Device, DeviceKind};
pub use error::{SpatialError, SpatialResult};
pub use execution::ExecutionPolicy;
pub use metadata::{FrameId, SpatialMetadata, Timestamp};
pub use pointcloud::{PointCloud, PointCloudBuilder};
pub use schema::{DType, FieldSemantic, PointField, PointSchema, StandardSchemas};
pub use tensor::{
    SpatialTensor, SpatialTensorChunk, SpatialTensorFieldChunk, DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE,
};

#[cfg(feature = "tensor-aoso")]
pub use tensor_aoso::{AoSoAAttributeChunk, AoSoAAttributeLayout, AoSoAXyzChunk};
