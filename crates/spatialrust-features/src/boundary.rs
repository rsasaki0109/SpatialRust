//! Boundary / edge point detection via tangent-plane angle gaps.
//!
//! A point on the interior of a well-sampled surface has neighbors spread all
//! around it in its tangent plane; a point on a boundary (a hole rim or the edge
//! of a scan) has a large empty angular wedge. Projecting neighbors into the
//! tangent plane and measuring the largest gap between consecutive directions
//! flags those boundary points.

use spatialrust_core::{
    HasNormals3, HasPositions3, PointBuffer, PointBufferSet, PointCloud, SpatialError,
    SpatialResult,
};
use spatialrust_math::Vec3;
use spatialrust_search::{parallel_worker_count, KdTree, RadiusSearchIndex};

/// Configuration for [`BoundaryDetector`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryConfig {
    /// Neighborhood radius used to gather tangent-plane directions.
    pub search_radius: f32,
    /// A point is a boundary if its largest angular gap (radians) exceeds this.
    pub angle_threshold: f32,
    /// Minimum neighbors required to evaluate a point.
    pub min_neighbors: usize,
}

impl Default for BoundaryConfig {
    fn default() -> Self {
        // ~90°: interior points have small gaps, boundary points a large wedge.
        Self { search_radius: 0.1, angle_threshold: std::f32::consts::FRAC_PI_2, min_neighbors: 5 }
    }
}

impl BoundaryConfig {
    /// Creates a config from the search radius (90° threshold).
    #[must_use]
    pub fn with_radius(search_radius: f32) -> Self {
        Self { search_radius, ..Self::default() }
    }
}

/// Result of boundary detection.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryResult {
    /// Indices of the boundary points in the input cloud.
    pub indices: Vec<usize>,
    /// Sub-cloud containing only the boundary points (schema preserved).
    pub boundary: PointCloud,
}

/// Tangent-plane boundary point detector. The input cloud must carry normals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryDetector {
    config: BoundaryConfig,
}

impl BoundaryDetector {
    /// Creates a detector from config.
    #[must_use]
    pub const fn new(config: BoundaryConfig) -> Self {
        Self { config }
    }

    /// Returns the detector config.
    #[must_use]
    pub const fn config(&self) -> BoundaryConfig {
        self.config
    }

    /// Computes the per-point boundary mask (`true` = boundary).
    ///
    /// Shares one KD-tree across bounded scoped CPU workers using the existing
    /// point-staging policy. Small clouds and one-worker calls run on the caller.
    /// Point order, neighborhood order and per-point arithmetic are preserved.
    pub fn boundary_mask(&self, input: &PointCloud) -> SpatialResult<Vec<bool>> {
        self.boundary_mask_with_workers(input, parallel_worker_count(input.len()))
    }

    fn boundary_mask_with_workers(
        &self,
        input: &PointCloud,
        workers: usize,
    ) -> SpatialResult<Vec<bool>> {
        if self.config.search_radius <= 0.0 || self.config.search_radius.is_nan() {
            return Err(SpatialError::InvalidArgument("search_radius must be positive".to_owned()));
        }
        let len = input.len();
        if len == 0 {
            return Ok(Vec::new());
        }

        let (x, y, z) = input.positions3()?;
        let (nx, ny, nz) = input.normals3()?;
        let tree = KdTree::from_slices(x, y, z);

        let mut mask = vec![false; len];
        let fill = |start: usize, band: &mut [bool]| {
            // Reuse angular scratch within each worker, without retaining an
            // all-point neighbor table or allocating a worker result vector.
            let mut angles: Vec<f32> = Vec::new();
            for (offset, is_boundary) in band.iter_mut().enumerate() {
                let i = start + offset;
                let p = Vec3::new(x[i], y[i], z[i]);
                let normal = Vec3::new(nx[i], ny[i], nz[i]);
                let (u, v) = tangent_basis(normal);

                let neighbors = tree.radius_search(p.x, p.y, p.z, self.config.search_radius);
                angles.clear();
                for neighbor in &neighbors {
                    let j = neighbor.index;
                    if j == i {
                        continue;
                    }
                    let d = Vec3::new(x[j] - p.x, y[j] - p.y, z[j] - p.z);
                    let (du, dv) = (d.dot(u), d.dot(v));
                    if du * du + dv * dv > 1e-12 {
                        angles.push(dv.atan2(du));
                    }
                }
                if angles.len() < self.config.min_neighbors {
                    continue;
                }
                *is_boundary = max_angle_gap(&mut angles) > self.config.angle_threshold;
            }
        };
        let workers = workers.max(1).min(len);
        if workers == 1 {
            fill(0, &mut mask);
        } else {
            let chunk_size = len.div_ceil(workers);
            std::thread::scope(|scope| {
                for (band, chunk) in mask.chunks_mut(chunk_size).enumerate() {
                    let fill = &fill;
                    scope.spawn(move || fill(band * chunk_size, chunk));
                }
            });
        }
        Ok(mask)
    }

    /// Detects boundary points, returning their indices and a sub-cloud.
    pub fn detect(&self, input: &PointCloud) -> SpatialResult<BoundaryResult> {
        let mask = self.boundary_mask(input)?;
        let indices: Vec<usize> =
            mask.iter().enumerate().filter_map(|(i, &b)| b.then_some(i)).collect();
        let boundary = gather_indices(input, &indices)?;
        Ok(BoundaryResult { indices, boundary })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "../tests/fixtures/boundary_serial.rs"]
#[allow(dead_code)]
mod previous;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "boundary_parallel_tests.rs"]
mod parallel_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "boundary_parallel_benchmark.rs"]
mod parallel_benchmark;

/// Builds an orthonormal basis `(u, v)` spanning the plane perpendicular to `n`.
fn tangent_basis(n: Vec3<f32>) -> (Vec3<f32>, Vec3<f32>) {
    let helper = if n.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
    let u = n.cross(helper).normalize();
    let v = n.cross(u);
    (u, v)
}

/// Largest gap between consecutive sorted angles, wrapping around the circle.
fn max_angle_gap(angles: &mut [f32]) -> f32 {
    // A point with no in-plane neighbors has no gap to measure (reachable when
    // `min_neighbors` is 0 and a point is isolated); treat it as non-salient.
    if angles.is_empty() {
        return 0.0;
    }
    angles.sort_by(f32::total_cmp);
    let mut max_gap = 0.0_f32;
    for w in angles.windows(2) {
        max_gap = max_gap.max(w[1] - w[0]);
    }
    // Wrap-around gap between the last and first direction.
    let wrap = (angles[0] + std::f32::consts::TAU) - angles[angles.len() - 1];
    max_gap.max(wrap)
}

fn gather_indices(input: &PointCloud, indices: &[usize]) -> SpatialResult<PointCloud> {
    let mut buffers = PointBufferSet::new();
    for field in input.schema().fields() {
        let source = input.field(&field.name)?;
        buffers.insert(field.name.clone(), gather_buffer(source, indices));
    }
    PointCloud::try_from_parts(input.schema().clone(), buffers, input.metadata().clone())
}

fn gather_buffer(source: &PointBuffer, indices: &[usize]) -> PointBuffer {
    match source {
        PointBuffer::F32(v) => PointBuffer::from_f32(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::F64(v) => PointBuffer::F64(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::U8(v) => PointBuffer::U8(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::U16(v) => PointBuffer::U16(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::U32(v) => PointBuffer::U32(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::I32(v) => PointBuffer::I32(indices.iter().map(|&i| v[i]).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::{BoundaryConfig, BoundaryDetector};
    use spatialrust_core::{DType, FieldSemantic, PointCloudBuilder, PointField, PointSchema};

    fn schema() -> PointSchema {
        PointSchema::new()
            .with_field(PointField::scalar("x", FieldSemantic::PositionX, DType::F32))
            .with_field(PointField::scalar("y", FieldSemantic::PositionY, DType::F32))
            .with_field(PointField::scalar("z", FieldSemantic::PositionZ, DType::F32))
            .with_field(PointField::scalar("normal_x", FieldSemantic::NormalX, DType::F32))
            .with_field(PointField::scalar("normal_y", FieldSemantic::NormalY, DType::F32))
            .with_field(PointField::scalar("normal_z", FieldSemantic::NormalZ, DType::F32))
    }

    /// A square planar patch (normal +Z): rim points are boundaries, the
    /// interior is not.
    fn square_patch(n: usize) -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(schema());
        for i in 0..n {
            for j in 0..n {
                builder.push_point([i as f32 * 0.1, j as f32 * 0.1, 0.0, 0.0, 0.0, 1.0]).unwrap();
            }
        }
        builder.build().unwrap()
    }

    #[test]
    fn detects_patch_rim_not_interior() {
        let n = 12;
        let cloud = square_patch(n);
        // min_neighbors=3 so sparse corners (only 3 in-radius neighbors) are
        // still evaluated.
        let config = BoundaryConfig { min_neighbors: 3, ..BoundaryConfig::with_radius(0.18) };
        let result = BoundaryDetector::new(config).detect(&cloud).unwrap();
        let mask = BoundaryDetector::new(config).boundary_mask(&cloud).unwrap();

        // The center point is interior; a corner is on the boundary.
        let center = (n / 2) * n + (n / 2);
        let corner = 0;
        assert!(!mask[center], "interior point flagged as boundary");
        assert!(mask[corner], "corner not flagged as boundary");
        // Boundaries are a minority of the patch.
        assert!(!result.indices.is_empty());
        assert!(result.indices.len() < cloud.len() / 2);
        assert_eq!(result.boundary.len(), result.indices.len());
    }

    #[test]
    fn rejects_bad_radius() {
        let cloud = square_patch(5);
        assert!(BoundaryDetector::new(BoundaryConfig::with_radius(0.0)).detect(&cloud).is_err());
    }

    #[test]
    fn handles_isolated_point_with_zero_min_neighbors() {
        // A lone point far from the patch has no in-radius neighbors. With
        // min_neighbors = 0 this used to reach the angle-gap code with an empty
        // angle list and panic; it must now just run.
        let mut builder = PointCloudBuilder::new(schema());
        builder.push_point([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        builder.push_point([100.0, 100.0, 100.0, 0.0, 0.0, 1.0]).unwrap();
        let cloud = builder.build().unwrap();

        let config = BoundaryConfig { min_neighbors: 0, ..BoundaryConfig::with_radius(0.1) };
        let mask = BoundaryDetector::new(config).boundary_mask(&cloud).unwrap();
        // Isolated points (no neighbors) are not flagged as surface boundaries.
        assert_eq!(mask, vec![false, false]);
    }
}
