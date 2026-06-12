use spatialrust_core::{PointCloud, SpatialResult};

/// Point cloud filter interface.
pub trait PointCloudFilter {
    /// Human-readable filter name.
    fn name(&self) -> &'static str;

    /// Applies the filter and returns a new point cloud.
    fn filter(&self, input: &PointCloud) -> SpatialResult<PointCloud>;
}
