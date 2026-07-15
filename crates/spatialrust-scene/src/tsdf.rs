//! Dense TSDF volume integration and mesh extraction.

use spatialrust_math::Vec3;

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
        if dims.iter().any(|d| *d == 0) {
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
    pub fn integrate_point(&mut self, point: Vec3<f32>, sensor_origin: Vec3<f32>) {
        let Some(index) = self.world_to_index(point) else {
            return;
        };
        let voxel_center = self.index_to_world(index);
        let depth = (point - sensor_origin).length();
        let voxel_depth = (voxel_center - sensor_origin).length();
        let sdf = (depth - voxel_depth).clamp(-self.truncation, self.truncation);
        let flat = self.flat(index);
        let w_old = self.weight[flat];
        let w_new = w_old + 1.0;
        self.distance[flat] = (self.distance[flat] * w_old + sdf) / w_new;
        self.weight[flat] = w_new;
    }

    /// Integrates every XYZ triple from interleaved storage.
    pub fn integrate_xyz(&mut self, xyz: &[f32], sensor_origin: Vec3<f32>) -> SceneResult<()> {
        if xyz.len() % 3 != 0 {
            return Err(SceneError::InvalidConfiguration("xyz length must be a multiple of 3".into()));
        }
        for chunk in xyz.chunks_exact(3) {
            self.integrate_point(Vec3::new(chunk[0], chunk[1], chunk[2]), sensor_origin);
        }
        Ok(())
    }

    /// Extracts an occupied-voxel surface proxy as a triangle mesh.
    ///
    /// Each voxel with weight >= `min_weight` and near-zero distance contributes
    /// a small axis-aligned triangle fan at its center (deterministic proxy until
    /// a marching-cubes backend lands).
    pub fn extract_mesh(&self, min_weight: f32) -> TriangleMesh {
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let flat = self.flat([x, y, z]);
                    if self.weight[flat] < min_weight {
                        continue;
                    }
                    let p = self.index_to_world([x, y, z]);
                    let base = (positions.len() / 3) as u32;
                    let s = self.voxel_size * 0.25;
                    positions.extend_from_slice(&[
                        p.x - s,
                        p.y - s,
                        p.z,
                        p.x + s,
                        p.y - s,
                        p.z,
                        p.x,
                        p.y + s,
                        p.z,
                    ]);
                    indices.extend_from_slice(&[base, base + 1, base + 2]);
                }
            }
        }
        TriangleMesh { positions, indices }
    }

    fn world_to_index(&self, point: Vec3<f32>) -> Option<[usize; 3]> {
        let ix = ((point.x - self.origin.x) / self.voxel_size).floor() as isize;
        let iy = ((point.y - self.origin.y) / self.voxel_size).floor() as isize;
        let iz = ((point.z - self.origin.z) / self.voxel_size).floor() as isize;
        if ix < 0 || iy < 0 || iz < 0 {
            return None;
        }
        let x = ix as usize;
        let y = iy as usize;
        let z = iz as usize;
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return None;
        }
        Some([x, y, z])
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

#[cfg(test)]
mod tests {
    use super::TsdfVolume;
    use spatialrust_math::Vec3;

    #[test]
    fn integrates_and_extracts_non_empty_mesh() {
        let mut volume =
            TsdfVolume::try_new(Vec3::new(-1.0, -1.0, -1.0), 0.25, [8, 8, 8], 0.5).unwrap();
        volume
            .integrate_xyz(&[0.0, 0.0, 0.0, 0.2, 0.0, 0.0], Vec3::new(0.0, 0.0, -1.0))
            .unwrap();
        let mesh = volume.extract_mesh(0.5);
        assert!(!mesh.positions.is_empty());
        assert_eq!(mesh.indices.len() % 3, 0);
    }
}
