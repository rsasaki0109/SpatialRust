//! Feature estimation for SpatialRust point clouds.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod estimator;
mod neighborhood;

#[cfg(feature = "feature-normal")]
mod normal;

#[cfg(feature = "feature-normal-gpu")]
mod normal_gpu;

pub use estimator::FeatureEstimator;
pub use neighborhood::{KdTreeNeighborhood, NeighborhoodProvider};

#[cfg(feature = "feature-normal")]
pub use normal::{
    orient_normal_towards_viewpoint, NormalEstimationConfig, NormalEstimationResult,
    NormalEstimator,
};

#[cfg(feature = "feature-normal-gpu")]
pub use normal_gpu::GpuNormalEstimator;
