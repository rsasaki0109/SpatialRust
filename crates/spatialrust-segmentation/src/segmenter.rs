/// Common trait for point cloud segmentation algorithms.
pub trait PointCloudSegmenter {
    /// Human-readable segmenter name.
    fn name(&self) -> &'static str;
}
