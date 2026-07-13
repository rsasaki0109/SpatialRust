//! AoSoA chunk packing: interleaved `[x,y,z, …]` buffers for SIMD/GPU staging.
//!
//! Enabled by the `tensor-aoso` feature. The underlying cloud stays schema-SoA;
//! packing copies one chunk at a time on demand.

use crate::{
    HasIntensity, HasNormals3, HasPositions3, PointCloud, SpatialResult, SpatialTensorChunk,
};

/// Explicit interleaved field layout for an [`AoSoAAttributeChunk`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AoSoAAttributeLayout {
    stride_f32: usize,
    position_offsets: [usize; 3],
    intensity_offset: Option<usize>,
    normal_offsets: Option<[usize; 3]>,
}

impl AoSoAAttributeLayout {
    /// Interleaved `[x, y, z, intensity]` layout.
    pub const XYZ_INTENSITY: Self = Self {
        stride_f32: 4,
        position_offsets: [0, 1, 2],
        intensity_offset: Some(3),
        normal_offsets: None,
    };

    /// Interleaved `[x, y, z, nx, ny, nz]` layout.
    pub const XYZ_NORMALS: Self = Self {
        stride_f32: 6,
        position_offsets: [0, 1, 2],
        intensity_offset: None,
        normal_offsets: Some([3, 4, 5]),
    };

    /// Interleaved `[x, y, z, intensity, nx, ny, nz]` layout.
    pub const XYZ_INTENSITY_NORMALS: Self = Self {
        stride_f32: 7,
        position_offsets: [0, 1, 2],
        intensity_offset: Some(3),
        normal_offsets: Some([4, 5, 6]),
    };

    /// Returns the distance between adjacent points in `f32` elements.
    #[must_use]
    pub const fn stride_f32(self) -> usize {
        self.stride_f32
    }

    /// Returns the x/y/z offsets within one point record.
    #[must_use]
    pub const fn position_offsets(self) -> [usize; 3] {
        self.position_offsets
    }

    /// Returns the intensity offset when present.
    #[must_use]
    pub const fn intensity_offset(self) -> Option<usize> {
        self.intensity_offset
    }

    /// Returns the normal x/y/z offsets when present.
    #[must_use]
    pub const fn normal_offsets(self) -> Option<[usize; 3]> {
        self.normal_offsets
    }
}

/// Owned interleaved positions and optional capability attributes for one chunk.
#[derive(Clone, Debug, PartialEq)]
pub struct AoSoAAttributeChunk {
    data: Vec<f32>,
    point_count: usize,
    layout: AoSoAAttributeLayout,
}

impl AoSoAAttributeChunk {
    fn pack(
        chunk: &SpatialTensorChunk,
        cloud: &PointCloud,
        layout: AoSoAAttributeLayout,
    ) -> SpatialResult<Self> {
        let (x, y, z) = cloud.positions3()?;
        let intensity = layout.intensity_offset().map(|_| cloud.intensity()).transpose()?;
        let normals = layout.normal_offsets().map(|_| cloud.normals3()).transpose()?;
        let range = chunk.range();
        let point_count = range.len();
        let mut data = Vec::with_capacity(point_count * layout.stride_f32());
        for index in range {
            data.extend_from_slice(&[x[index], y[index], z[index]]);
            if let Some(intensity) = intensity {
                data.push(intensity[index]);
            }
            if let Some((nx, ny, nz)) = normals {
                data.extend_from_slice(&[nx[index], ny[index], nz[index]]);
            }
        }
        Ok(Self { data, point_count, layout })
    }

    /// Packs `[x, y, z, intensity]` records using [`HasIntensity`].
    pub fn pack_xyz_intensity(
        chunk: &SpatialTensorChunk,
        cloud: &PointCloud,
    ) -> SpatialResult<Self> {
        Self::pack(chunk, cloud, AoSoAAttributeLayout::XYZ_INTENSITY)
    }

    /// Packs `[x, y, z, nx, ny, nz]` records using [`HasNormals3`].
    pub fn pack_xyz_normals(chunk: &SpatialTensorChunk, cloud: &PointCloud) -> SpatialResult<Self> {
        Self::pack(chunk, cloud, AoSoAAttributeLayout::XYZ_NORMALS)
    }

    /// Packs `[x, y, z, intensity, nx, ny, nz]` capability records.
    pub fn pack_xyz_intensity_normals(
        chunk: &SpatialTensorChunk,
        cloud: &PointCloud,
    ) -> SpatialResult<Self> {
        Self::pack(chunk, cloud, AoSoAAttributeLayout::XYZ_INTENSITY_NORMALS)
    }

    /// Returns the number of packed points.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.point_count
    }

    /// Returns whether this chunk contains no points.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.point_count == 0
    }

    /// Returns the explicit stride and field-offset metadata.
    #[must_use]
    pub const fn layout(&self) -> AoSoAAttributeLayout {
        self.layout
    }

    /// Returns the packed `f32` records.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

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

    /// Packs interleaved XYZ and intensity records.
    pub fn pack_xyz_intensity(&self, cloud: &PointCloud) -> SpatialResult<AoSoAAttributeChunk> {
        AoSoAAttributeChunk::pack_xyz_intensity(self, cloud)
    }

    /// Packs interleaved XYZ and normal records.
    pub fn pack_xyz_normals(&self, cloud: &PointCloud) -> SpatialResult<AoSoAAttributeChunk> {
        AoSoAAttributeChunk::pack_xyz_normals(self, cloud)
    }

    /// Packs interleaved XYZ, intensity, and normal records.
    pub fn pack_xyz_intensity_normals(
        &self,
        cloud: &PointCloud,
    ) -> SpatialResult<AoSoAAttributeChunk> {
        AoSoAAttributeChunk::pack_xyz_intensity_normals(self, cloud)
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
    use crate::{AoSoAAttributeLayout, PointCloudBuilder, SpatialTensor, StandardSchemas};

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

    #[test]
    fn packs_composite_capabilities_with_explicit_layout() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzinormal());
        builder.push_point([1.0, 2.0, 3.0, 0.5, 0.0, 1.0, 0.0]).unwrap();
        builder.push_point([4.0, 5.0, 6.0, 0.8, 1.0, 0.0, 0.0]).unwrap();
        let cloud = builder.build().unwrap();
        let chunk = SpatialTensor::new(&cloud, 8).unwrap().chunks().next().unwrap();

        let packed = chunk.pack_xyz_intensity_normals(&cloud).unwrap();
        assert_eq!(packed.layout(), AoSoAAttributeLayout::XYZ_INTENSITY_NORMALS);
        assert_eq!(packed.layout().stride_f32(), 7);
        assert_eq!(packed.layout().intensity_offset(), Some(3));
        assert_eq!(packed.layout().normal_offsets(), Some([4, 5, 6]));
        assert_eq!(
            packed.as_slice(),
            &[1.0, 2.0, 3.0, 0.5, 0.0, 1.0, 0.0, 4.0, 5.0, 6.0, 0.8, 1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn attribute_packers_reject_missing_capabilities() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([1.0, 2.0, 3.0]).unwrap();
        let cloud = builder.build().unwrap();
        let chunk = SpatialTensor::new(&cloud, 4).unwrap().chunks().next().unwrap();

        assert!(chunk.pack_xyz_intensity(&cloud).is_err());
        assert!(chunk.pack_xyz_normals(&cloud).is_err());
    }
}
