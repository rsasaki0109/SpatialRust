use spatialrust_core::{HasPositions3, PointCloud, SpatialResult};
use spatialrust_gpu::{estimate_normals_gpu, WgpuRuntime};
use spatialrust_math::Vec3;

use crate::neighborhood::{KdTreeNeighborhood, NeighborhoodProvider};
use crate::normal::{build_output_cloud, orient_normal_towards_viewpoint, NormalEstimationConfig};

/// GPU-accelerated normal estimator.
///
/// Neighbor search runs on the CPU (KD-tree); the per-point covariance analysis
/// and eigen-decomposition run on the GPU via wgpu. Output matches
/// [`crate::NormalEstimator`]: `normal_x/y/z` and `curvature` fields, optionally
/// oriented toward a viewpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuNormalEstimator {
    config: NormalEstimationConfig,
}

impl GpuNormalEstimator {
    /// Creates a GPU normal estimator from config.
    #[must_use]
    pub const fn new(config: NormalEstimationConfig) -> Self {
        Self { config }
    }

    /// Returns the config.
    #[must_use]
    pub const fn config(&self) -> NormalEstimationConfig {
        self.config
    }

    /// Estimates normals on the GPU, returning a cloud with normal/curvature fields.
    pub fn estimate(&self, input: &PointCloud) -> SpatialResult<PointCloud> {
        if input.is_empty() {
            return Ok(input.clone());
        }

        let (x, y, z) = input.positions3()?;
        let n = input.len();
        let k = self.config.k_neighbors.max(1);

        // CPU neighbor search; flatten to a fixed `n * k` index grid for the GPU.
        let neighborhood = KdTreeNeighborhood::from_point_cloud(input)?;
        let mut flat = Vec::with_capacity(n * k);
        for index in 0..n {
            let mut neighbors = neighborhood.query_k(index, k)?;
            if neighbors.is_empty() {
                neighbors.push(index);
            }
            for slot in 0..k {
                flat.push(neighbors[slot % neighbors.len()] as u32);
            }
        }

        let runtime = WgpuRuntime::shared()?;
        let gpu_normals = estimate_normals_gpu(&runtime, x, y, z, &flat, k as u32)?;

        let mut nx = Vec::with_capacity(n);
        let mut ny = Vec::with_capacity(n);
        let mut nz = Vec::with_capacity(n);
        let mut curvature = Vec::with_capacity(n);
        for (index, gpu_normal) in gpu_normals.iter().enumerate() {
            let mut normal =
                Vec3::new(gpu_normal.normal[0], gpu_normal.normal[1], gpu_normal.normal[2]);
            if let Some(viewpoint) = self.config.viewpoint {
                normal = orient_normal_towards_viewpoint(
                    normal,
                    Vec3::new(x[index], y[index], z[index]),
                    viewpoint,
                );
            }
            nx.push(normal.x);
            ny.push(normal.y);
            nz.push(normal.z);
            curvature.push(gpu_normal.curvature);
        }

        build_output_cloud(input, nx, ny, nz, curvature)
    }
}

#[cfg(test)]
mod tests {
    use super::GpuNormalEstimator;
    use crate::normal::NormalEstimationConfig;
    use spatialrust_core::{HasNormals3, PointCloudBuilder, StandardSchemas};
    use spatialrust_gpu::WgpuRuntime;
    use spatialrust_math::Vec3;

    #[test]
    fn estimates_plane_normals_on_gpu() {
        // Skip gracefully when no GPU/software adapter is available.
        if WgpuRuntime::shared().is_err() {
            return;
        }

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for i in 0..8 {
            for j in 0..8 {
                builder.push_point([i as f32 * 0.1, j as f32 * 0.1, 0.0]).unwrap();
            }
        }
        let cloud = builder.build().unwrap();

        let estimator = GpuNormalEstimator::new(NormalEstimationConfig {
            k_neighbors: 8,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.0, 0.0, 10.0)),
            ..Default::default()
        });
        let output = estimator.estimate(&cloud).unwrap();

        let (nx, ny, nz) = output.normals3().unwrap();
        for index in 0..output.len() {
            // Plane normals point up toward the viewpoint.
            assert!(nz[index] > 0.99, "normal not vertical: {}", nz[index]);
            assert!(nx[index].abs() < 0.1 && ny[index].abs() < 0.1);
        }
        assert!(output.field("curvature").is_ok());
    }
}
