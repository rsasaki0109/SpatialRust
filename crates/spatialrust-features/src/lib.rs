//! Feature estimation for SpatialRust point clouds.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod estimator;
mod neighborhood;

#[cfg(feature = "feature-normal")]
mod normal;

pub use estimator::FeatureEstimator;
pub use neighborhood::{KdTreeNeighborhood, NeighborhoodProvider};

#[cfg(feature = "feature-normal")]
pub use normal::{
    NormalEstimationConfig, NormalEstimationResult, NormalEstimator, orient_normal_towards_viewpoint,
};
