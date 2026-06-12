use spatialrust_core::{PointCloud, SpatialResult};

/// Computes point-wise features from an input point cloud.
pub trait FeatureEstimator {
    /// Human-readable estimator name.
    fn name(&self) -> &'static str;

    /// Runs feature estimation and returns an output point cloud.
    fn estimate(&self, input: &PointCloud) -> SpatialResult<PointCloud>;
}
