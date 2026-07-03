#[cfg(feature = "feature-normal-gpu")]
use spatialrust_core::DeviceKind;
use spatialrust_core::{
    DType, ExecutionPolicy, FieldSemantic, HasPositions3, PointBuffer, PointBufferSet, PointCloud,
    PointField, PointSchema, SpatialError, SpatialResult,
};
use spatialrust_math::{symmetric_eigen3, Mat3, Vec3};
use spatialrust_search::{KdTree, Neighbor, RadiusSearchIndex};

use crate::estimator::FeatureEstimator;

/// Minimum point count before GPU normal estimation is selected under `Auto`.
///
/// The k-NN GPU path (default MVP config) modestly wins at large counts; the
/// radius/grid GPU path is much faster but requires `search_radius`.
pub const DEFAULT_GPU_MIN_POINTS_NORMAL: usize = 10_000;

/// Configuration for covariance-based normal estimation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalEstimationConfig {
    /// Number of nearest neighbors to use when `search_radius` is `None`.
    pub k_neighbors: usize,
    /// Optional radius search instead of fixed `k`.
    pub search_radius: Option<f32>,
    /// Minimum number of neighbors required to estimate a valid normal.
    pub min_neighbors: usize,
    /// Optional viewpoint used to orient normals consistently.
    pub viewpoint: Option<Vec3<f32>>,
    /// Minimum input point count before GPU execution is considered under `Auto`.
    ///
    /// `None` always uses GPU when requested.
    pub gpu_min_points: Option<usize>,
}

impl Default for NormalEstimationConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 20,
            search_radius: None,
            min_neighbors: 3,
            viewpoint: None,
            gpu_min_points: Some(DEFAULT_GPU_MIN_POINTS_NORMAL),
        }
    }
}

impl NormalEstimationConfig {
    /// Creates a k-NN normal estimation config.
    #[must_use]
    pub const fn k_neighbors(k_neighbors: usize) -> Self {
        Self {
            k_neighbors,
            search_radius: None,
            min_neighbors: 3,
            viewpoint: None,
            gpu_min_points: Some(DEFAULT_GPU_MIN_POINTS_NORMAL),
        }
    }

    /// Disables the GPU point-count heuristic so GPU is always used when requested.
    #[must_use]
    pub const fn without_gpu_min_points(mut self) -> Self {
        self.gpu_min_points = None;
        self
    }

    /// Returns the point-count threshold used by [`ExecutionPolicy::Auto`].
    #[must_use]
    pub const fn effective_gpu_min_points(&self) -> Option<usize> {
        self.gpu_min_points
    }
}

/// Result metadata for normal estimation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NormalEstimationResult {
    /// Number of points with valid normals.
    pub valid_count: usize,
    /// Number of points with invalid normals.
    pub invalid_count: usize,
}

/// Covariance-based normal estimator.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalEstimator {
    config: NormalEstimationConfig,
}

impl NormalEstimator {
    /// Creates a normal estimator from config.
    #[must_use]
    pub const fn new(config: NormalEstimationConfig) -> Self {
        Self { config }
    }

    /// Returns the estimator config.
    #[must_use]
    pub const fn config(&self) -> NormalEstimationConfig {
        self.config
    }

    /// Estimates normals and curvature, returning output cloud and diagnostics.
    pub fn estimate_with_diagnostics(
        &self,
        input: &PointCloud,
    ) -> SpatialResult<(PointCloud, NormalEstimationResult)> {
        if input.is_empty() {
            return Ok((input.clone(), NormalEstimationResult::default()));
        }
        if self.config.search_radius.is_some_and(|radius| radius < 0.0) {
            return Err(SpatialError::InvalidArgument("search_radius must be non-negative".into()));
        }

        let (x, y, z) = input.positions3()?;
        let tree = KdTree::from_slices(x, y, z);

        let mut nx = vec![f32::NAN; input.len()];
        let mut ny = vec![f32::NAN; input.len()];
        let mut nz = vec![f32::NAN; input.len()];
        let mut curvature = vec![0.0_f32; input.len()];
        let mut valid_count = 0usize;
        let mut invalid_count = 0usize;

        let worker_count = normal_worker_count(input.len());
        if worker_count == 1 {
            let chunk = estimate_normal_range(self.config, &tree, x, y, z, 0, input.len());
            nx = chunk.nx;
            ny = chunk.ny;
            nz = chunk.nz;
            curvature = chunk.curvature;
            valid_count = chunk.valid_count;
            invalid_count = chunk.invalid_count;
        } else {
            let chunk_size = input.len().div_ceil(worker_count);
            let chunks = std::thread::scope(|scope| {
                let mut handles = Vec::new();
                let config = self.config;
                let tree_ref = &tree;
                for start in (0..input.len()).step_by(chunk_size) {
                    let end = (start + chunk_size).min(input.len());
                    handles.push(scope.spawn(move || {
                        estimate_normal_range(config, tree_ref, x, y, z, start, end)
                    }));
                }

                handles
                    .into_iter()
                    .map(|handle| handle.join().expect("normal estimation worker panicked"))
                    .collect::<Vec<_>>()
            });

            for chunk in chunks {
                let end = chunk.start + chunk.nx.len();
                nx[chunk.start..end].copy_from_slice(&chunk.nx);
                ny[chunk.start..end].copy_from_slice(&chunk.ny);
                nz[chunk.start..end].copy_from_slice(&chunk.nz);
                curvature[chunk.start..end].copy_from_slice(&chunk.curvature);
                valid_count += chunk.valid_count;
                invalid_count += chunk.invalid_count;
            }
        }

        let output = build_output_cloud(input, nx, ny, nz, curvature)?;
        Ok((output, NormalEstimationResult { valid_count, invalid_count }))
    }

    /// Estimates normals using the given execution policy.
    ///
    /// With the `feature-normal-gpu` feature, [`ExecutionPolicy::Auto`] and
    /// [`ExecutionPolicy::Gpu`] run covariance analysis on wgpu when the input
    /// meets [`NormalEstimationConfig::effective_gpu_min_points`].
    pub fn estimate_with_policy(
        &self,
        input: &PointCloud,
        policy: ExecutionPolicy,
    ) -> SpatialResult<PointCloud> {
        #[cfg(feature = "feature-normal-gpu")]
        {
            let resolved = self.resolve_policy(input, policy);
            if matches!(resolved, ExecutionPolicy::Gpu(DeviceKind::Wgpu)) {
                return crate::normal_gpu::GpuNormalEstimator::new(self.config).estimate(input);
            }
        }

        let _ = policy;
        self.estimate(input)
    }

    /// Returns whether the given policy selects the GPU backend for this input.
    #[cfg(feature = "feature-normal-gpu")]
    pub fn selects_gpu_backend(&self, input: &PointCloud, policy: ExecutionPolicy) -> bool {
        matches!(self.resolve_policy(input, policy), ExecutionPolicy::Gpu(DeviceKind::Wgpu))
    }

    #[cfg(feature = "feature-normal-gpu")]
    fn should_use_gpu(&self, input: &PointCloud) -> bool {
        self.config.effective_gpu_min_points().map_or(true, |min_points| input.len() >= min_points)
    }

    #[cfg(feature = "feature-normal-gpu")]
    fn resolve_policy(&self, input: &PointCloud, policy: ExecutionPolicy) -> ExecutionPolicy {
        match policy {
            ExecutionPolicy::Auto => {
                if self.should_use_gpu(input) {
                    ExecutionPolicy::Gpu(DeviceKind::Wgpu)
                } else {
                    ExecutionPolicy::CpuSingle
                }
            }
            ExecutionPolicy::Gpu(DeviceKind::Wgpu) if !self.should_use_gpu(input) => {
                ExecutionPolicy::CpuSingle
            }
            other => other,
        }
    }
}

impl FeatureEstimator for NormalEstimator {
    fn name(&self) -> &'static str {
        "NormalEstimator"
    }

    fn estimate(&self, input: &PointCloud) -> SpatialResult<PointCloud> {
        self.estimate_with_diagnostics(input).map(|(cloud, _)| cloud)
    }
}

#[derive(Debug)]
struct NormalChunk {
    start: usize,
    nx: Vec<f32>,
    ny: Vec<f32>,
    nz: Vec<f32>,
    curvature: Vec<f32>,
    valid_count: usize,
    invalid_count: usize,
}

fn normal_worker_count(len: usize) -> usize {
    let available = std::thread::available_parallelism().map_or(1, |count| count.get());
    let useful = (len / 16_384).max(1);
    available.min(useful)
}

fn estimate_normal_range(
    config: NormalEstimationConfig,
    tree: &KdTree,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    start: usize,
    end: usize,
) -> NormalChunk {
    let len = end - start;
    let mut nx = vec![f32::NAN; len];
    let mut ny = vec![f32::NAN; len];
    let mut nz = vec![f32::NAN; len];
    let mut curvature = vec![0.0_f32; len];
    let mut valid_count = 0usize;
    let mut invalid_count = 0usize;
    let mut neighbor_buffer = Vec::with_capacity(config.k_neighbors.saturating_add(1));
    let mut index_buffer = Vec::with_capacity(config.k_neighbors);

    for index in start..end {
        query_neighbors_into(config, tree, x, y, z, index, &mut neighbor_buffer, &mut index_buffer);
        let local = index - start;
        if index_buffer.len() < config.min_neighbors {
            invalid_count += 1;
            continue;
        }

        let Some((normal, curv)) = estimate_normal_from_neighbors(x, y, z, index, &index_buffer)
        else {
            invalid_count += 1;
            continue;
        };

        let oriented = if let Some(viewpoint) = config.viewpoint {
            orient_normal_towards_viewpoint(normal, point_xyz(x, y, z, index), viewpoint)
        } else {
            normal
        };

        nx[local] = oriented.x;
        ny[local] = oriented.y;
        nz[local] = oriented.z;
        curvature[local] = curv;
        valid_count += 1;
    }

    NormalChunk { start, nx, ny, nz, curvature, valid_count, invalid_count }
}

fn query_neighbors_into(
    config: NormalEstimationConfig,
    tree: &KdTree,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    index: usize,
    neighbor_buffer: &mut Vec<Neighbor>,
    index_buffer: &mut Vec<usize>,
) {
    index_buffer.clear();
    if let Some(radius) = config.search_radius {
        for neighbor in tree.radius_search(x[index], y[index], z[index], radius) {
            if neighbor.index != index {
                index_buffer.push(neighbor.index);
            }
        }
    } else {
        tree.nearest_k_unsorted_into(
            x[index],
            y[index],
            z[index],
            config.k_neighbors.saturating_add(1),
            neighbor_buffer,
        );
        for neighbor in neighbor_buffer.iter() {
            if neighbor.index != index {
                index_buffer.push(neighbor.index);
                if index_buffer.len() == config.k_neighbors {
                    break;
                }
            }
        }
    }
}

/// Orients a normal to point towards the viewpoint when possible.
#[must_use]
pub fn orient_normal_towards_viewpoint(
    mut normal: Vec3<f32>,
    point: Vec3<f32>,
    viewpoint: Vec3<f32>,
) -> Vec3<f32> {
    let view_direction =
        Vec3::new(viewpoint.x - point.x, viewpoint.y - point.y, viewpoint.z - point.z);
    if normal.dot(view_direction) < 0.0 {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = -normal.z;
    }
    normal.normalize()
}

fn point_xyz(x: &[f32], y: &[f32], z: &[f32], index: usize) -> Vec3<f32> {
    Vec3::new(x[index], y[index], z[index])
}

fn estimate_normal_from_neighbors(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    _center_index: usize,
    neighbors: &[usize],
) -> Option<(Vec3<f32>, f32)> {
    let mut mean_x = 0.0_f32;
    let mut mean_y = 0.0_f32;
    let mut mean_z = 0.0_f32;
    for &index in neighbors {
        mean_x += x[index];
        mean_y += y[index];
        mean_z += z[index];
    }
    let count = neighbors.len() as f32;
    mean_x /= count;
    mean_y /= count;
    mean_z /= count;

    let mut c00 = 0.0_f32;
    let mut c11 = 0.0_f32;
    let mut c22 = 0.0_f32;
    let mut c01 = 0.0_f32;
    let mut c02 = 0.0_f32;
    let mut c12 = 0.0_f32;
    for &index in neighbors {
        let dx = x[index] - mean_x;
        let dy = y[index] - mean_y;
        let dz = z[index] - mean_z;
        c00 += dx * dx;
        c11 += dy * dy;
        c22 += dz * dz;
        c01 += dx * dy;
        c02 += dx * dz;
        c12 += dy * dz;
    }
    let inv = 1.0 / count;
    smallest_eigenpair_for_covariance(
        c00 * inv,
        c11 * inv,
        c22 * inv,
        c01 * inv,
        c02 * inv,
        c12 * inv,
    )
}

fn smallest_eigenpair_for_covariance(
    c00: f32,
    c11: f32,
    c22: f32,
    c01: f32,
    c02: f32,
    c12: f32,
) -> Option<(Vec3<f32>, f32)> {
    let eigenvalues = symmetric_eigenvalues3(c00, c11, c22, c01, c02, c12);
    let lambda = eigenvalues[0];
    let normal =
        eigenvector_for_eigenvalue(c00, c11, c22, c01, c02, c12, lambda).unwrap_or_else(|| {
            let covariance = Mat3::<f64>::from_rows(
                [c00 as f64, c01 as f64, c02 as f64],
                [c01 as f64, c11 as f64, c12 as f64],
                [c02 as f64, c12 as f64, c22 as f64],
            );
            let eigen = symmetric_eigen3(covariance);
            Vec3::new(
                eigen.eigenvectors.m[0][0] as f32,
                eigen.eigenvectors.m[1][0] as f32,
                eigen.eigenvectors.m[2][0] as f32,
            )
            .normalize()
        });

    let sum = eigenvalues[0] + eigenvalues[1] + eigenvalues[2];
    let curvature = if sum > 0.0 { eigenvalues[0] / sum } else { 0.0 };
    Some((normal.normalize(), curvature))
}

fn symmetric_eigenvalues3(c00: f32, c11: f32, c22: f32, c01: f32, c02: f32, c12: f32) -> [f32; 3] {
    let p1 = c01 * c01 + c02 * c02 + c12 * c12;
    if p1 <= f32::EPSILON {
        let mut values = [c00, c11, c22];
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        return values;
    }

    let q = (c00 + c11 + c22) / 3.0;
    let b00 = c00 - q;
    let b11 = c11 - q;
    let b22 = c22 - q;
    let p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    if p <= f32::EPSILON {
        return [q, q, q];
    }

    let inv_p = 1.0 / p;
    let n00 = b00 * inv_p;
    let n11 = b11 * inv_p;
    let n22 = b22 * inv_p;
    let n01 = c01 * inv_p;
    let n02 = c02 * inv_p;
    let n12 = c12 * inv_p;
    let det = n00 * (n11 * n22 - n12 * n12) - n01 * (n01 * n22 - n12 * n02)
        + n02 * (n01 * n12 - n11 * n02);
    let r = (det * 0.5).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;

    let largest = q + 2.0 * p * phi.cos();
    let smallest = q + 2.0 * p * (phi + 2.0 * std::f32::consts::PI / 3.0).cos();
    let middle = 3.0 * q - largest - smallest;
    let mut values = [smallest, middle, largest];
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values
}

fn eigenvector_for_eigenvalue(
    c00: f32,
    c11: f32,
    c22: f32,
    c01: f32,
    c02: f32,
    c12: f32,
    lambda: f32,
) -> Option<Vec3<f32>> {
    let row0 = Vec3::new(c00 - lambda, c01, c02);
    let row1 = Vec3::new(c01, c11 - lambda, c12);
    let row2 = Vec3::new(c02, c12, c22 - lambda);

    let candidates = [row0.cross(row1), row0.cross(row2), row1.cross(row2)];
    let mut best = candidates[0];
    let mut best_norm = best.length_squared();
    for candidate in candidates.into_iter().skip(1) {
        let norm = candidate.length_squared();
        if norm > best_norm {
            best = candidate;
            best_norm = norm;
        }
    }

    if best_norm <= 1e-24 {
        None
    } else {
        Some(best.normalize())
    }
}

pub(crate) fn build_output_cloud(
    input: &PointCloud,
    nx: Vec<f32>,
    ny: Vec<f32>,
    nz: Vec<f32>,
    curvature: Vec<f32>,
) -> SpatialResult<PointCloud> {
    let mut schema = input.schema().clone();
    ensure_field(&mut schema, "normal_x", FieldSemantic::NormalX, DType::F32);
    ensure_field(&mut schema, "normal_y", FieldSemantic::NormalY, DType::F32);
    ensure_field(&mut schema, "normal_z", FieldSemantic::NormalZ, DType::F32);
    ensure_field(&mut schema, "curvature", FieldSemantic::Curvature, DType::F32);

    let mut buffers = PointBufferSet::new();
    for field in input.schema().fields() {
        let source = input.field(&field.name)?;
        buffers.insert(field.name.clone(), clone_buffer(source)?);
    }
    buffers.insert("normal_x".to_owned(), PointBuffer::from_f32(nx));
    buffers.insert("normal_y".to_owned(), PointBuffer::from_f32(ny));
    buffers.insert("normal_z".to_owned(), PointBuffer::from_f32(nz));
    buffers.insert("curvature".to_owned(), PointBuffer::from_f32(curvature));

    PointCloud::try_from_parts(schema, buffers, input.metadata().clone())
}

fn ensure_field(schema: &mut PointSchema, name: &str, semantic: FieldSemantic, dtype: DType) {
    if schema.find_semantic(semantic).is_none() {
        *schema = schema.clone().with_field(PointField::scalar(name, semantic, dtype));
    }
}

fn clone_buffer(buffer: &PointBuffer) -> SpatialResult<PointBuffer> {
    Ok(match buffer {
        PointBuffer::F32(values) => PointBuffer::from_f32(values.clone()),
        PointBuffer::F64(values) => PointBuffer::F64(values.clone()),
        PointBuffer::U8(values) => PointBuffer::U8(values.clone()),
        PointBuffer::U16(values) => PointBuffer::U16(values.clone()),
        PointBuffer::U32(values) => PointBuffer::U32(values.clone()),
        PointBuffer::I32(values) => PointBuffer::I32(values.clone()),
    })
}

#[cfg(test)]
mod tests {
    use super::{orient_normal_towards_viewpoint, NormalEstimationConfig, NormalEstimator};
    use crate::FeatureEstimator;
    use spatialrust_core::{HasNormals3, PointCloudBuilder, StandardSchemas};
    use spatialrust_math::Vec3;

    fn plane_cloud() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for x in 0..5 {
            for y in 0..5 {
                builder.push_point([x as f32, y as f32, 0.0]).unwrap();
            }
        }
        builder.build().unwrap()
    }

    fn tilted_plane_cloud() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for x in 0..7 {
            for y in 0..7 {
                let fx = x as f32 * 0.2;
                let fy = y as f32 * 0.2;
                let z = 0.2 * fx - 0.3 * fy + 0.1;
                builder.push_point([fx, fy, z]).unwrap();
            }
        }
        builder.build().unwrap()
    }

    #[test]
    fn estimates_plane_normals_upwards() {
        let input = plane_cloud();
        let estimator = NormalEstimator::new(NormalEstimationConfig {
            k_neighbors: 8,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.0, 0.0, 10.0)),
            ..NormalEstimationConfig::default()
        });
        let (output, stats) = estimator.estimate_with_diagnostics(&input).unwrap();
        assert_eq!(stats.valid_count, input.len());
        assert_eq!(stats.invalid_count, 0);

        let (_, _, nz) = output.normals3().unwrap();
        for value in nz {
            assert!((*value - 1.0).abs() < 0.1, "expected upward normal, got {value}");
        }
    }

    #[test]
    fn estimates_tilted_plane_normals() {
        let input = tilted_plane_cloud();
        let estimator = NormalEstimator::new(NormalEstimationConfig {
            k_neighbors: 12,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.0, 0.0, 10.0)),
            ..NormalEstimationConfig::default()
        });
        let output = estimator.estimate(&input).unwrap();
        let (nx, ny, nz) = output.normals3().unwrap();
        let expected = Vec3::new(-0.2, 0.3, 1.0).normalize();

        for index in 0..input.len() {
            let actual = Vec3::new(nx[index], ny[index], nz[index]).normalize();
            assert!(actual.dot(expected) > 0.98, "tilted plane normal was {actual:?}");
        }
    }

    #[test]
    fn orient_normal_towards_viewpoint_works() {
        let normal = Vec3::new(0.0, 0.0, -1.0);
        let point = Vec3::new(0.0, 0.0, 0.0);
        let viewpoint = Vec3::new(0.0, 0.0, 1.0);
        let oriented = orient_normal_towards_viewpoint(normal, point, viewpoint);
        assert!(oriented.z > 0.0);
    }

    #[test]
    fn adds_curvature_field() {
        let input = plane_cloud();
        let estimator = NormalEstimator::new(NormalEstimationConfig::k_neighbors(10));
        let output = estimator.estimate(&input).unwrap();
        assert!(output.field("curvature").is_ok());
    }
}
