//! Point cloud filters for SpatialRust.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod filter;

#[cfg(feature = "filter-voxel")]
mod voxel;

pub use filter::PointCloudFilter;

#[cfg(feature = "filter-voxel")]
pub use voxel::{
    AttributeAggregation, VoxelAggregationMode, VoxelGridDownsample, VoxelGridDownsampleConfig,
    DEFAULT_GPU_MIN_POINTS, DEFAULT_GPU_MIN_POINTS_APPROXIMATE,
};
