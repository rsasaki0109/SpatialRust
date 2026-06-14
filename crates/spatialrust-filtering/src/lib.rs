//! Point cloud filters for SpatialRust.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod filter;

#[cfg(feature = "filter-outlier")]
mod outlier;

#[cfg(feature = "filter-voxel")]
mod voxel;

pub use filter::PointCloudFilter;

#[cfg(feature = "filter-outlier")]
pub use outlier::{
    RadiusOutlierConfig, RadiusOutlierRemoval, StatisticalOutlierConfig, StatisticalOutlierRemoval,
};

#[cfg(feature = "filter-voxel")]
pub use voxel::{
    AttributeAggregation, VoxelAggregationMode, VoxelGridDownsample, VoxelGridDownsampleConfig,
    DEFAULT_GPU_MIN_POINTS, DEFAULT_GPU_MIN_POINTS_APPROXIMATE,
};
