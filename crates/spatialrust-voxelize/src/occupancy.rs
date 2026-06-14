//! Dense voxel occupancy / count grids.

use spatialrust_core::{HasPositions3, PointCloud, SpatialError, SpatialResult};

/// How a voxel's value is filled from the points that fall in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelFill {
    /// `1.0` if any point falls in the voxel, else `0.0`.
    Occupancy,
    /// The number of points that fall in the voxel.
    Count,
}

/// Configuration for [`voxelize`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelGridConfig {
    /// Side length of each cubic voxel.
    pub voxel_size: f32,
    /// Grid origin (lower corner). `None` uses the cloud's minimum corner.
    pub origin: Option<[f32; 3]>,
    /// Grid dimensions `(nx, ny, nz)`. `None` derives them from the cloud bounds.
    pub dims: Option<[usize; 3]>,
    /// How to fill each voxel.
    pub fill: VoxelFill,
}

impl Default for VoxelGridConfig {
    fn default() -> Self {
        Self { voxel_size: 0.1, origin: None, dims: None, fill: VoxelFill::Occupancy }
    }
}

impl VoxelGridConfig {
    /// Creates a config from the voxel size (auto origin/dims, occupancy fill).
    #[must_use]
    pub fn new(voxel_size: f32) -> Self {
        Self { voxel_size, ..Self::default() }
    }
}

/// A dense 3D grid in row-major `(z, y, x)` order.
#[derive(Clone, Debug, PartialEq)]
pub struct OccupancyGrid {
    /// Grid dimensions `(nx, ny, nz)`.
    pub dims: [usize; 3],
    /// Lower corner of voxel `(0, 0, 0)`.
    pub origin: [f32; 3],
    /// Voxel side length.
    pub voxel_size: f32,
    /// Values, indexed `z * (ny * nx) + y * nx + x`.
    pub data: Vec<f32>,
}

impl OccupancyGrid {
    /// Total number of voxels (`nx * ny * nz`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the grid has no voxels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Number of voxels with a non-zero value.
    #[must_use]
    pub fn occupied_count(&self) -> usize {
        self.data.iter().filter(|&&v| v != 0.0).count()
    }

    /// Value at voxel `(x, y, z)`, or `None` if out of range.
    #[must_use]
    pub fn get(&self, x: usize, y: usize, z: usize) -> Option<f32> {
        let [nx, ny, nz] = self.dims;
        if x >= nx || y >= ny || z >= nz {
            return None;
        }
        Some(self.data[z * ny * nx + y * nx + x])
    }
}

/// Voxelizes a cloud into a dense occupancy / count grid.
pub fn voxelize(cloud: &PointCloud, config: VoxelGridConfig) -> SpatialResult<OccupancyGrid> {
    if config.voxel_size <= 0.0 || config.voxel_size.is_nan() {
        return Err(SpatialError::InvalidArgument("voxel_size must be positive".to_owned()));
    }
    let (x, y, z) = cloud.positions3()?;
    let len = cloud.len();

    let origin = match config.origin {
        Some(o) => o,
        None => {
            if len == 0 {
                [0.0, 0.0, 0.0]
            } else {
                [
                    x.iter().copied().fold(f32::INFINITY, f32::min),
                    y.iter().copied().fold(f32::INFINITY, f32::min),
                    z.iter().copied().fold(f32::INFINITY, f32::min),
                ]
            }
        }
    };

    let inv = 1.0 / config.voxel_size;
    let dims = match config.dims {
        Some(d) => d,
        None => {
            if len == 0 {
                [1, 1, 1]
            } else {
                let span = |vals: &[f32], o: f32| {
                    let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    (((max - o) * inv).floor() as usize) + 1
                };
                [span(x, origin[0]), span(y, origin[1]), span(z, origin[2])]
            }
        }
    };
    let [nx, ny, nz] = dims;
    let total = nx.checked_mul(ny).and_then(|v| v.checked_mul(nz)).ok_or_else(|| {
        SpatialError::InvalidArgument("voxel grid dimensions overflow".to_owned())
    })?;

    let mut data = vec![0.0_f32; total];
    for i in 0..len {
        let vx = ((x[i] - origin[0]) * inv).floor();
        let vy = ((y[i] - origin[1]) * inv).floor();
        let vz = ((z[i] - origin[2]) * inv).floor();
        if vx < 0.0 || vy < 0.0 || vz < 0.0 {
            continue;
        }
        let (vx, vy, vz) = (vx as usize, vy as usize, vz as usize);
        if vx >= nx || vy >= ny || vz >= nz {
            continue;
        }
        let idx = vz * ny * nx + vy * nx + vx;
        match config.fill {
            VoxelFill::Occupancy => data[idx] = 1.0,
            VoxelFill::Count => data[idx] += 1.0,
        }
    }

    Ok(OccupancyGrid { dims, origin, voxel_size: config.voxel_size, data })
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};

    fn cloud(points: &[[f32; 3]]) -> PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for p in points {
            builder.push_point(*p).unwrap();
        }
        builder.build().unwrap()
    }

    #[test]
    fn occupancy_marks_filled_voxels() {
        // Two points one voxel apart along x.
        let c = cloud(&[[0.05, 0.05, 0.05], [0.15, 0.05, 0.05]]);
        let grid = voxelize(&c, VoxelGridConfig::new(0.1)).unwrap();
        assert_eq!(grid.dims, [2, 1, 1]);
        assert_eq!(grid.occupied_count(), 2);
        assert_eq!(grid.get(0, 0, 0), Some(1.0));
        assert_eq!(grid.get(1, 0, 0), Some(1.0));
    }

    #[test]
    fn count_accumulates_points_per_voxel() {
        // Three points in the same voxel, one in another.
        let c = cloud(&[
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [0.03, 0.03, 0.03],
            [0.55, 0.01, 0.01],
        ]);
        let config = VoxelGridConfig { fill: VoxelFill::Count, ..VoxelGridConfig::new(0.1) };
        let grid = voxelize(&c, config).unwrap();
        assert_eq!(grid.get(0, 0, 0), Some(3.0));
        assert_eq!(grid.occupied_count(), 2);
        // The total count equals the number of in-bounds points.
        let total: f32 = grid.data.iter().sum();
        assert_eq!(total, 4.0);
    }

    #[test]
    fn fixed_dims_drop_out_of_bounds_points() {
        let c = cloud(&[[0.05, 0.05, 0.05], [100.0, 0.0, 0.0]]);
        let config = VoxelGridConfig {
            origin: Some([0.0, 0.0, 0.0]),
            dims: Some([2, 2, 2]),
            ..VoxelGridConfig::new(0.1)
        };
        let grid = voxelize(&c, config).unwrap();
        assert_eq!(grid.dims, [2, 2, 2]);
        assert_eq!(grid.len(), 8);
        // The far point is outside the fixed grid and is dropped.
        assert_eq!(grid.occupied_count(), 1);
    }

    #[test]
    fn rejects_bad_voxel_size() {
        let c = cloud(&[[0.0, 0.0, 0.0]]);
        assert!(voxelize(&c, VoxelGridConfig::new(0.0)).is_err());
    }
}
