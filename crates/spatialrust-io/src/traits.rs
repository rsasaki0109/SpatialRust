use spatialrust_core::{PointCloud, PointSchema, SpatialMetadata, SpatialResult};

use crate::{ReadOptions, WriteOptions};

/// Reads a point cloud from a file or stream.
pub trait PointReader {
    /// Returns the schema declared by the source.
    fn schema(&self) -> SpatialResult<PointSchema>;

    /// Returns spatial metadata associated with the source.
    fn metadata(&self) -> SpatialResult<SpatialMetadata>;

    /// Reads the full point cloud.
    fn read(&mut self, options: &ReadOptions) -> SpatialResult<PointCloud>;
}

/// Writes a point cloud to a file or stream.
pub trait PointWriter {
    /// Writes the point cloud using the provided options.
    fn write(&mut self, cloud: &PointCloud, options: &WriteOptions) -> SpatialResult<()>;
}

/// Reads point clouds in chunks.
pub trait PointStream {
    /// Advances to the next chunk, returning `false` when finished.
    fn next_chunk(&mut self, options: &ReadOptions) -> SpatialResult<bool>;
}

/// Accepts point cloud chunks for streaming writes.
pub trait PointSink {
    /// Accepts one chunk for writing.
    fn write_chunk(&mut self, cloud: &PointCloud, options: &WriteOptions) -> SpatialResult<()>;
}
