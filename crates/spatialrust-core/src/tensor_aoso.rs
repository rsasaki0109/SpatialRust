//! AoSoA chunk packing: interleaved `[x,y,z, …]` buffers for SIMD/GPU staging.
//!
//! Enabled by the `tensor-aoso` feature. The underlying cloud stays schema-SoA;
//! packing copies one chunk at a time on demand.

use crate::{HasPositions3, PointCloud, SpatialResult, SpatialTensorChunk};

/// Interleaved XYZ layout for one [`SpatialTensorChunk`] (`3 * point_count` floats).
#[derive(Clone, Debug, PartialEq)]
pub struct AoSoAXyzChunk {
    data: Vec<f32>,
    point_count: usize,
}

impl AoSoAXyzChunk {
    /// Packs chunk points into interleaved `[x,y,z, …]` order.
    pub fn pack(chunk: &SpatialTensorChunk, cloud: &PointCloud) -> SpatialResult<Self> {
        let (x, y, z) = cloud.positions3()?;
        let range = chunk.range();
        let point_count = range.len();
        let mut data = Vec::with_capacity(point_count * 3);
        for index in range {
            data.push(x[index]);
            data.push(y[index]);
            data.push(z[index]);
        }
        Ok(Self { data, point_count })
    }

    /// Returns the number of points in this chunk.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.point_count
    }

    /// Returns whether this chunk contains zero points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.point_count == 0
    }

    /// Returns the interleaved `[x,y,z, …]` slice (`len == 3 * point_count`).
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Returns one point by local chunk index.
    #[must_use]
    pub fn point(&self, local_index: usize) -> [f32; 3] {
        let base = local_index * 3;
        [self.data[base], self.data[base + 1], self.data[base + 2]]
    }
}

impl SpatialTensorChunk {
    /// Packs this chunk into an owned interleaved XYZ buffer.
    pub fn pack_xyz(&self, cloud: &PointCloud) -> SpatialResult<AoSoAXyzChunk> {
        AoSoAXyzChunk::pack(self, cloud)
    }

    /// Packs this chunk into `out`, returning the number of points written.
    ///
    /// `out` is cleared first and resized to `3 * chunk.len()` floats.
    pub fn pack_xyz_into(&self, cloud: &PointCloud, out: &mut Vec<f32>) -> SpatialResult<usize> {
        let (x, y, z) = cloud.positions3()?;
        let range = self.range();
        out.clear();
        out.reserve(range.len() * 3);
        for index in range {
            out.push(x[index]);
            out.push(y[index]);
            out.push(z[index]);
        }
        Ok(out.len() / 3)
    }
}

#[cfg(test)]
mod tests {
    use crate::{PointCloudBuilder, SpatialTensor};

    #[test]
    fn pack_matches_column_slices() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([1.0, 2.0, 3.0]).unwrap();
        builder.push_point([4.0, 5.0, 6.0]).unwrap();
        builder.push_point([7.0, 8.0, 9.0]).unwrap();
        let cloud = builder.build().unwrap();
        let chunk = SpatialTensor::new(&cloud, 2).unwrap().chunks().next().unwrap();

        let packed = chunk.pack_xyz(&cloud).unwrap();
        assert_eq!(packed.len(), 2);
        assert_eq!(packed.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(packed.point(1), [4.0, 5.0, 6.0]);
    }

    #[test]
    fn pack_into_reuses_buffer() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([0.0, 1.0, 2.0]).unwrap();
        let cloud = builder.build().unwrap();
        let chunk = SpatialTensor::new(&cloud, 4).unwrap().chunks().next().unwrap();

        let mut buffer = vec![f32::NAN; 99];
        let count = chunk.pack_xyz_into(&cloud, &mut buffer).unwrap();
        assert_eq!(count, 1);
        assert_eq!(buffer, vec![0.0, 1.0, 2.0]);
    }
}
