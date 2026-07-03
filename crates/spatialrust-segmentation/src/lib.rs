//! Point cloud segmentation for SpatialRust.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod cloud;
mod segmenter;

#[cfg(feature = "segment-ransac-plane")]
mod plane;

#[cfg(feature = "segment-ransac-plane")]
mod plane_ransac;

#[cfg(all(feature = "segment-ransac-plane", feature = "segment-ransac-plane-gpu"))]
mod plane_gpu;

#[cfg(feature = "segment-multi-plane")]
mod multi_plane;

#[cfg(feature = "segment-euclidean")]
mod cluster;

#[cfg(all(feature = "segment-euclidean", feature = "segment-euclidean-gpu"))]
mod cluster_gpu;

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
pub use plane::{
    PlaneModel, RansacPlaneConfig, RansacPlaneSegmentation, RansacPlaneSegmenter,
    DEFAULT_GPU_MIN_POINTS_PLANE,
};

#[cfg(all(feature = "segment-ransac-plane", feature = "segment-ransac-plane-gpu"))]
pub use plane_gpu::GpuRansacPlaneSegmenter;

#[cfg(feature = "segment-multi-plane")]
pub use multi_plane::{MultiPlaneConfig, MultiPlaneSegmentation, MultiPlaneSegmenter};

#[cfg(feature = "segment-euclidean")]
pub use cluster::{
    EuclideanClusterConfig, EuclideanClusterExtractor, EuclideanClusterResult,
    DEFAULT_GPU_KDTREE_MIN_POINTS, DEFAULT_GPU_MIN_POINTS_EUCLIDEAN,
};

#[cfg(all(feature = "segment-euclidean", feature = "segment-euclidean-gpu"))]
pub use cluster_gpu::GpuEuclideanClusterExtractor;

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
