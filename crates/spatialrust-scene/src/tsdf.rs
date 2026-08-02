//! Dense TSDF volume integration and mesh extraction.

use spatialrust_math::Vec3;

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
        let len = dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]);
        Ok(Self {
            origin,
            voxel_size,
            dims,
            distance: vec![truncation; len],
            weight: vec![0.0; len],
            truncation,
        })
    }

    /// Returns voxel dimensions `[nx, ny, nz]`.
    #[must_use]
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Integrates one metric depth sample at a world-space point.
    ///
    /// Updates every voxel whose center lies within the truncation band, using
    /// a view-ray signed distance so the field has zero crossings for meshing.
    pub fn integrate_point(&mut self, point: Vec3<f32>, sensor_origin: Vec3<f32>) {
        let depth = (point - sensor_origin).length();
        if !(depth.is_finite() && depth > 1e-5) {
            return;
        }
        let ray = (point - sensor_origin).normalize();
        let radius = self.truncation;
        let min = point - Vec3::new(radius, radius, radius);
        let max = point + Vec3::new(radius, radius, radius);
        let i0 = self.world_to_index_clamped(min);
        let i1 = self.world_to_index_clamped(max);
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

    fn world_to_index_clamped(&self, point: Vec3<f32>) -> [usize; 3] {
        let ix = ((point.x - self.origin.x) / self.voxel_size).floor() as isize;
        let iy = ((point.y - self.origin.y) / self.voxel_size).floor() as isize;
        let iz = ((point.z - self.origin.z) / self.voxel_size).floor() as isize;
        [
            ix.clamp(0, self.dims[0] as isize - 1) as usize,
            iy.clamp(0, self.dims[1] as isize - 1) as usize,
            iz.clamp(0, self.dims[2] as isize - 1) as usize,
        ]
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
mod tests {
    use super::TsdfVolume;
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
}
