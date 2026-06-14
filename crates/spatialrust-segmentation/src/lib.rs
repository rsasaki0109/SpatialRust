//! Point cloud segmentation for SpatialRust.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod cloud;
mod segmenter;

#[cfg(feature = "segment-ransac-plane")]
mod plane;

#[cfg(feature = "segment-euclidean")]
mod cluster;

#[cfg(feature = "segment-dbscan")]
mod dbscan;

#[cfg(feature = "segment-ground")]
mod ground;

#[cfg(feature = "segment-ransac-primitives")]
mod primitives;

#[cfg(feature = "segment-region-growing")]
mod region_growing;

pub use cloud::{extract_indices, extract_mask, with_labels};
pub use segmenter::PointCloudSegmenter;

#[cfg(feature = "segment-ransac-plane")]
pub use plane::{PlaneModel, RansacPlaneConfig, RansacPlaneSegmentation, RansacPlaneSegmenter};

#[cfg(feature = "segment-euclidean")]
pub use cluster::{EuclideanClusterConfig, EuclideanClusterExtractor, EuclideanClusterResult};

#[cfg(feature = "segment-dbscan")]
pub use dbscan::{DbscanConfig, DbscanResult, DbscanSegmenter};

#[cfg(feature = "segment-ground")]
pub use ground::{GroundConfig, GroundSegmentation, GroundSegmenter, UpAxis};

#[cfg(feature = "segment-ransac-primitives")]
pub use primitives::{
    CylinderModel, PrimitiveSegmentation, RansacCylinderSegmenter, RansacPrimitiveConfig,
    RansacSphereSegmenter, SphereModel,
};

#[cfg(feature = "segment-region-growing")]
pub use region_growing::{RegionGrowingConfig, RegionGrowingResult, RegionGrowingSegmenter};
