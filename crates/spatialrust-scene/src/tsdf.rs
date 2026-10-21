//! Dense TSDF volume integration and mesh extraction.

use spatialrust_core::{HasPositions3, PointCloud};
use spatialrust_math::{Isometry3, TransformPoint, Vec3};

use crate::marching_cubes::{polygonise_tet, tetrahedra};
use crate::{SceneError, SceneResult, TriangleMesh};

/// Axis-aligned truncated signed distance volume.
#[derive(Clone, Debug, PartialEq)]
pub struct TsdfVolume {
    origin: Vec3<f32>,
    voxel_size: f32,
    dims: [usize; 3],
    distance: Vec<f32>,
    weight: Vec<f32>,
    truncation: f32,
}

impl TsdfVolume {
    /// Creates an empty TSDF volume initialized to truncation distance.
    ///
    /// Rejects non-finite origins or voxel centers, invalid scales/dimensions
    /// and field capacities that cannot be represented. Both fields are
    /// reserved fallibly before initialization; reservation failures return
    /// [`SceneError::InvalidConfiguration`]. This does not impose a memory
    /// budget or guarantee recovery from operating-system overcommit.
    pub fn try_new(
        origin: Vec3<f32>,
        voxel_size: f32,
        dims: [usize; 3],
        truncation: f32,
    ) -> SceneResult<Self> {
        if !(voxel_size.is_finite() && voxel_size > 0.0) {
            return Err(SceneError::InvalidConfiguration("voxel_size must be > 0".into()));
        }
        if !(truncation.is_finite() && truncation > 0.0) {
            return Err(SceneError::InvalidConfiguration("truncation must be > 0".into()));
        }
        if dims.contains(&0) {
            return Err(SceneError::InvalidConfiguration("dims must be non-zero".into()));
        }
        if ![origin.x, origin.y, origin.z].into_iter().all(f32::is_finite) {
            return Err(SceneError::InvalidConfiguration("origin must be finite".into()));
        }
        let len = dims[0]
            .checked_mul(dims[1])
            .and_then(|count| count.checked_mul(dims[2]))
            .filter(|&count| count <= isize::MAX as usize / std::mem::size_of::<f32>())
            .ok_or_else(|| {
                SceneError::InvalidConfiguration("TSDF field capacity is not representable".into())
            })?;
        for (coordinate, dim) in [origin.x, origin.y, origin.z].into_iter().zip(dims) {
            // Match index_to_world's f32 evaluation, including intermediate
            // offset overflow. Positive offsets are monotonic on each axis.
            let last_center = coordinate + ((dim - 1) as f32 + 0.5) * voxel_size;
            if !last_center.is_finite() {
                return Err(SceneError::InvalidConfiguration(
                    "voxel centers must be finite".into(),
                ));
            }
        }
        let mut distance = Vec::new();
        let mut weight = Vec::new();
        distance.try_reserve_exact(len).map_err(|_| {
            SceneError::InvalidConfiguration("unable to reserve TSDF distance storage".into())
        })?;
        weight.try_reserve_exact(len).map_err(|_| {
            SceneError::InvalidConfiguration("unable to reserve TSDF weight storage".into())
        })?;
        distance.resize(len, truncation);
        weight.resize(len, 0.0);
        Ok(Self { origin, voxel_size, dims, distance, weight, truncation })
    }

    /// Returns voxel dimensions `[nx, ny, nz]`.
    #[must_use]
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Integrates one metric depth sample at a world-space point.
    ///
    /// Updates valid cells in the inclusive floor-index range of the cube
    /// `point +/- truncation`, using a view-ray signed distance for meshing.
    /// Fully disjoint support is ignored; partial overlap is clipped to the
    /// grid. This retains the discretized cube rather than a spherical band.
    pub fn integrate_point(&mut self, point: Vec3<f32>, sensor_origin: Vec3<f32>) {
        let depth = (point - sensor_origin).length();
        if !(depth.is_finite() && depth > 1e-5) {
            return;
        }
        let ray = (point - sensor_origin).normalize();
        let radius = self.truncation;
        let min = point - Vec3::new(radius, radius, radius);
        let max = point + Vec3::new(radius, radius, radius);
        let i0 = self.world_to_index(min);
        let i1 = self.world_to_index(max);
        // Reject disjoint raw ranges before either endpoint is clamped onto
        // a boundary cell. Keep the original quantization for every overlap.
        if (0..3).any(|axis| i1[axis] < 0 || i0[axis] >= self.dims[axis] as isize) {
            return;
        }
        let i0: [usize; 3] =
            std::array::from_fn(|axis| i0[axis].clamp(0, self.dims[axis] as isize - 1) as usize);
        let i1: [usize; 3] =
            std::array::from_fn(|axis| i1[axis].clamp(0, self.dims[axis] as isize - 1) as usize);
        for z in i0[2]..=i1[2] {
            for y in i0[1]..=i1[1] {
                for x in i0[0]..=i1[0] {
                    let center = self.index_to_world([x, y, z]);
                    let sdf = (point - center).dot(ray).clamp(-self.truncation, self.truncation);
                    let flat = self.flat([x, y, z]);
                    let w_old = self.weight[flat];
                    let w_new = w_old + 1.0;
                    self.distance[flat] = (self.distance[flat] * w_old + sdf) / w_new;
                    self.weight[flat] = w_new;
                }
            }
        }
    }

    /// Integrates every XYZ triple from interleaved storage.
    pub fn integrate_xyz(&mut self, xyz: &[f32], sensor_origin: Vec3<f32>) -> SceneResult<()> {
        if xyz.len() % 3 != 0 {
            return Err(SceneError::InvalidConfiguration(
                "xyz length must be a multiple of 3".into(),
            ));
        }
        for chunk in xyz.chunks_exact(3) {
            self.integrate_point(Vec3::new(chunk[0], chunk[1], chunk[2]), sensor_origin);
        }
        Ok(())
    }

    /// Integrates XYZ columns directly from a point cloud without interleaving
    /// or allocating a temporary coordinate vector.
    ///
    /// Returns the number of point samples visited. Invalid metric samples are
    /// ignored by [`TsdfVolume::integrate_point`] with the same semantics as
    /// [`TsdfVolume::integrate_xyz`].
    pub fn integrate_cloud(
        &mut self,
        cloud: &PointCloud,
        sensor_origin: Vec3<f32>,
    ) -> SceneResult<usize> {
        let (x, y, z) = cloud.positions3()?;
        for index in 0..cloud.len() {
            self.integrate_point(Vec3::new(x[index], y[index], z[index]), sensor_origin);
        }
        Ok(cloud.len())
    }

    /// Integrates a point cloud after an explicit sensor-to-volume pose.
    ///
    /// Both positions and the supplied sensor origin are transformed by
    /// `volume_t_sensor`; no transformed point-cloud allocation is created.
    pub fn integrate_cloud_with_pose(
        &mut self,
        cloud: &PointCloud,
        volume_t_sensor: Isometry3<f32>,
        sensor_origin: Vec3<f32>,
    ) -> SceneResult<usize> {
        let (x, y, z) = cloud.positions3()?;
        let volume_sensor_origin = volume_t_sensor.transform_point(sensor_origin);
        for index in 0..cloud.len() {
            let point = volume_t_sensor.transform_point(Vec3::new(x[index], y[index], z[index]));
            self.integrate_point(point, volume_sensor_origin);
        }
        Ok(cloud.len())
    }

    /// Extracts the zero isolevel surface with marching tetrahedra.
    ///
    /// Voxels with weight `< min_weight` are treated as free space (`+truncation`).
    pub fn extract_mesh(&self, min_weight: f32) -> TriangleMesh {
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        if self.dims[0] < 2 || self.dims[1] < 2 || self.dims[2] < 2 {
            return TriangleMesh { positions, indices };
        }

        for z in 0..self.dims[2] - 1 {
            for y in 0..self.dims[1] - 1 {
                for x in 0..self.dims[0] - 1 {
                    let mut corner_pos = [Vec3::new(0.0, 0.0, 0.0); 8];
                    let mut corner_val = [0.0f32; 8];
                    for (corner, offset) in (0..8).zip(CORNER_OFFSETS.iter()) {
                        let idx = [x + offset[0], y + offset[1], z + offset[2]];
                        corner_pos[corner] = self.index_to_world(idx);
                        corner_val[corner] = self.sample(idx, min_weight);
                    }
                    for tet in tetrahedra() {
                        polygonise_tet(
                            &mut positions,
                            &mut indices,
                            [
                                corner_pos[tet[0]],
                                corner_pos[tet[1]],
                                corner_pos[tet[2]],
                                corner_pos[tet[3]],
                            ],
                            [
                                corner_val[tet[0]],
                                corner_val[tet[1]],
                                corner_val[tet[2]],
                                corner_val[tet[3]],
                            ],
                            0.0,
                        );
                    }
                }
            }
        }
        TriangleMesh { positions, indices }
    }

    fn sample(&self, index: [usize; 3], min_weight: f32) -> f32 {
        let flat = self.flat(index);
        if self.weight[flat] < min_weight {
            self.truncation
        } else {
            self.distance[flat]
        }
    }

    fn world_to_index(&self, point: Vec3<f32>) -> [isize; 3] {
        let ix = ((point.x - self.origin.x) / self.voxel_size).floor() as isize;
        let iy = ((point.y - self.origin.y) / self.voxel_size).floor() as isize;
        let iz = ((point.z - self.origin.z) / self.voxel_size).floor() as isize;
        [ix, iy, iz]
    }

    fn index_to_world(&self, index: [usize; 3]) -> Vec3<f32> {
        Vec3::new(
            self.origin.x + (index[0] as f32 + 0.5) * self.voxel_size,
            self.origin.y + (index[1] as f32 + 0.5) * self.voxel_size,
            self.origin.z + (index[2] as f32 + 0.5) * self.voxel_size,
        )
    }

    fn flat(&self, index: [usize; 3]) -> usize {
        index[0] + self.dims[0] * (index[1] + self.dims[1] * index[2])
    }
}

const CORNER_OFFSETS: [[usize; 3]; 8] =
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];

#[cfg(test)]
#[path = "tsdf_support_tests.rs"]
mod support_tests;

#[cfg(test)]
#[path = "tsdf_grid_tests.rs"]
mod grid_tests;

#[cfg(test)]
mod tests {
    use super::TsdfVolume;
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};
    use spatialrust_math::Vec3;

    #[test]
    fn integrates_and_extracts_non_empty_mesh() {
        let mut volume =
            TsdfVolume::try_new(Vec3::new(-1.0, -1.0, -1.0), 0.25, [8, 8, 8], 0.5).unwrap();
        volume.integrate_xyz(&[0.0, 0.0, 0.0, 0.2, 0.0, 0.0], Vec3::new(0.0, 0.0, -1.0)).unwrap();
        let mesh = volume.extract_mesh(0.5);
        assert!(!mesh.positions.is_empty());
        assert_eq!(mesh.indices.len() % 3, 0);
        assert!(mesh.triangle_count() >= 1);
    }

    #[test]
    fn empty_weight_yields_empty_mesh() {
        let volume =
            TsdfVolume::try_new(Vec3::new(-1.0, -1.0, -1.0), 0.25, [8, 8, 8], 0.5).unwrap();
        let mesh = volume.extract_mesh(1.0);
        assert!(mesh.positions.is_empty());
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn integrates_point_cloud_columns_without_interleave() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        builder.push_point([0.2, 0.0, 0.0]).unwrap();
        let cloud = builder.build().unwrap();
        let origin = Vec3::new(0.0, 0.0, -1.0);
        let mut from_cloud =
            TsdfVolume::try_new(Vec3::new(-1.0, -1.0, -1.0), 0.25, [8, 8, 8], 0.5).unwrap();
        let mut from_xyz = from_cloud.clone();
        assert_eq!(from_cloud.integrate_cloud(&cloud, origin).unwrap(), 2);
        from_xyz.integrate_xyz(&[0.0, 0.0, 0.0, 0.2, 0.0, 0.0], origin).unwrap();
        assert_eq!(from_cloud, from_xyz);
    }

    #[test]
    fn integrates_point_cloud_with_explicit_sensor_pose() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        builder.push_point([0.2, 0.0, 0.0]).unwrap();
        let cloud = builder.build().unwrap();
        let mut volume =
            TsdfVolume::try_new(Vec3::new(-1.0, -1.0, -1.0), 0.25, [8, 8, 8], 0.5).unwrap();
        let pose = spatialrust_math::Isometry3::new(
            spatialrust_math::Quat::<f32>::identity(),
            Vec3::new(0.5, 0.0, 0.0),
        );
        assert_eq!(
            volume.integrate_cloud_with_pose(&cloud, pose, Vec3::new(0.0, 0.0, -1.0)).unwrap(),
            2
        );
    }
}
