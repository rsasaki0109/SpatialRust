//! Point cloud filters for SpatialRust.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod filter;

#[cfg(feature = "filter-crop")]
mod crop;

#[cfg(feature = "filter-fps")]
mod fps;

#[cfg(feature = "filter-mls")]
mod mls;

#[cfg(feature = "filter-outlier")]
mod outlier;

#[cfg(feature = "filter-voxel")]
mod voxel;

pub use filter::PointCloudFilter;

#[cfg(feature = "filter-crop")]
pub use crop::{Aabb, CropBox, PassThrough};

#[cfg(feature = "filter-fps")]
pub use fps::{FarthestPointSampling, FarthestPointSamplingConfig};

#[cfg(feature = "filter-mls")]
pub use mls::{MlsConfig, MlsSmoothing};

#[cfg(feature = "filter-outlier")]
pub use outlier::{
    RadiusOutlierConfig, RadiusOutlierRemoval, StatisticalOutlierConfig, StatisticalOutlierRemoval,
};

#[cfg(feature = "filter-voxel")]
pub use voxel::{
    AttributeAggregation, VoxelAggregationMode, VoxelGridDownsample, VoxelGridDownsampleConfig,
    DEFAULT_GPU_MIN_POINTS, DEFAULT_GPU_MIN_POINTS_APPROXIMATE,
};
