//! Provisional chunked views over schema-aware column storage.
//!
//! `SpatialTensor` is the architecture-level name for zero-copy iteration over
//! fixed-size point chunks (AoSoA-style slices). The API is **provisional** —
//! see [`docs/API_STABILITY.md`](../../docs/API_STABILITY.md).

use std::ops::Range;

use crate::{PointBuffer, PointCloud, PointSchema, SpatialResult};

/// Default chunk size for [`PointCloud::spatial_tensor_chunks`].
pub const DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE: usize = 16_384;

/// Borrowed chunked view over a [`PointCloud`].
#[derive(Clone, Copy, Debug)]
pub struct SpatialTensor<'a> {
    cloud: &'a PointCloud,
    chunk_size: usize,
}

/// One contiguous index range within a [`SpatialTensor`] chunk iteration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpatialTensorChunk {
    range: Range<usize>,
}

impl<'a> SpatialTensor<'a> {
    /// Creates a chunked view with an explicit chunk size (must be > 0).
    pub fn new(cloud: &'a PointCloud, chunk_size: usize) -> SpatialResult<Self> {
        if chunk_size == 0 {
            return Err(crate::SpatialError::InvalidArgument(
                "SpatialTensor chunk_size must be positive".into(),
            ));
        }
        Ok(Self { cloud, chunk_size })
    }

    /// Returns the underlying point cloud.
    #[must_use]
    pub const fn cloud(&self) -> &'a PointCloud {
        self.cloud
    }

    /// Returns the configured chunk size.
    #[must_use]
    pub const fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Returns the point schema shared by every chunk.
    #[must_use]
    pub fn schema(&self) -> &PointSchema {
        self.cloud.schema()
    }

    /// Returns the total number of points in the view.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cloud.len()
    }

    /// Returns whether the view spans zero points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cloud.is_empty()
    }

    /// Iterates contiguous index ranges covering the full cloud.
    pub fn chunks(&self) -> impl Iterator<Item = SpatialTensorChunk> + 'a {
        let len = self.cloud.len();
        let chunk_size = self.chunk_size;
        (0..len).step_by(chunk_size).map(move |start| {
            let end = (start + chunk_size).min(len);
            SpatialTensorChunk { range: start..end }
        })
    }
}

impl SpatialTensorChunk {
    /// Returns the half-open index range `[start, end)` for this chunk.
    #[must_use]
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Returns the number of points in this chunk.
    #[must_use]
    pub fn len(&self) -> usize {
        self.range.len()
    }

    /// Returns whether this chunk is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// Returns a slice of an `f32` field column for this chunk.
    pub fn field_f32<'a>(&self, cloud: &'a PointCloud, name: &str) -> SpatialResult<&'a [f32]> {
        let values = cloud.field(name)?.as_f32()?;
        Ok(&values[self.range.clone()])
    }

    /// Returns the underlying buffer slice for any field dtype in this chunk.
    pub fn field_buffer<'a>(
        &self,
        cloud: &'a PointCloud,
        name: &str,
    ) -> SpatialResult<SpatialTensorFieldChunk<'a>> {
        let buffer = cloud.field(name)?;
        Ok(SpatialTensorFieldChunk { buffer, range: self.range.clone() })
    }
}

/// Borrowed slice of one column buffer within a chunk range.
#[derive(Clone, Debug)]
pub struct SpatialTensorFieldChunk<'a> {
    buffer: &'a PointBuffer,
    range: Range<usize>,
}

impl<'a> SpatialTensorFieldChunk<'a> {
    /// Returns the field slice as `f32` when the column dtype allows it.
    pub fn as_f32(&self) -> SpatialResult<&[f32]> {
        Ok(&self.buffer.as_f32()?[self.range.clone()])
    }

    /// Returns the half-open index range for this field chunk.
    #[must_use]
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

impl PointCloud {
    /// Returns a provisional chunked view for AoSoA-style iteration.
    pub fn spatial_tensor_chunks(&self, chunk_size: usize) -> SpatialResult<SpatialTensor<'_>> {
        SpatialTensor::new(self, chunk_size)
    }

    /// Returns a chunked view using [`DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE`].
    pub fn spatial_tensor(&self) -> SpatialResult<SpatialTensor<'_>> {
        self.spatial_tensor_chunks(DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::{SpatialTensor, DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE};
    use crate::PointCloudBuilder;

    #[test]
    fn chunks_cover_all_points() {
        let mut builder = PointCloudBuilder::xyz();
        for index in 0..5 {
            builder.push_point([index as f32, 0.0, 0.0]).unwrap();
        }
        let cloud = builder.build().unwrap();
        let tensor = SpatialTensor::new(&cloud, 2).unwrap();
        let chunks: Vec<_> = tensor.chunks().collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 2);
        assert_eq!(chunks[2].len(), 1);
        let covered: usize = chunks.iter().map(super::SpatialTensorChunk::len).sum();
        assert_eq!(covered, cloud.len());
    }

    #[test]
    fn field_f32_slice_matches_column() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([1.0, 2.0, 3.0]).unwrap();
        builder.push_point([4.0, 5.0, 6.0]).unwrap();
        let cloud = builder.build().unwrap();
        let chunk = SpatialTensor::new(&cloud, DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE)
            .unwrap()
            .chunks()
            .next()
            .unwrap();
        assert_eq!(chunk.field_f32(&cloud, "y").unwrap(), &[2.0, 5.0]);
    }

    #[test]
    fn rejects_zero_chunk_size() {
        let cloud = crate::PointCloud::xyz();
        assert!(SpatialTensor::new(&cloud, 0).is_err());
    }
}
