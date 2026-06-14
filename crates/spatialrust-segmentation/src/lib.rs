//! Point cloud segmentation for SpatialRust.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod cloud;
mod segmenter;

#[cfg(feature = "segment-ransac-plane")]
mod plane;

#[cfg(feature = "segment-euclidean")]
mod cluster;

#[cfg(feature = "segment-region-growing")]
mod region_growing;

pub use cloud::{extract_indices, extract_mask, with_labels};
pub use segmenter::PointCloudSegmenter;

#[cfg(feature = "segment-ransac-plane")]
pub use plane::{
    PlaneModel, RansacPlaneConfig, RansacPlaneSegmentation, RansacPlaneSegmenter,
};

#[cfg(feature = "segment-euclidean")]
pub use cluster::{
    EuclideanClusterConfig, EuclideanClusterResult, EuclideanClusterExtractor,
};

#[cfg(feature = "segment-region-growing")]
pub use region_growing::{
    RegionGrowingConfig, RegionGrowingResult, RegionGrowingSegmenter,
};
