//! Point cloud filters for SpatialRust.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod filter;

#[cfg(feature = "filter-voxel")]
mod voxel;

pub use filter::PointCloudFilter;

#[cfg(feature = "filter-voxel")]
pub use voxel::{
    AttributeAggregation, DEFAULT_GPU_MIN_POINTS, VoxelAggregationMode, VoxelGridDownsample,
    VoxelGridDownsampleConfig,
};
