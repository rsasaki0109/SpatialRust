// Explicit row/column indexing reads more clearly than iterators for fixed 3x3
// and 3x6 linear-algebra kernels.
#![allow(clippy::needless_range_loop)]

use spatialrust_core::{HasPositions3, PointCloud, SpatialError, SpatialResult};
use spatialrust_math::{
    solve_linear_system, symmetric_eigen3, Isometry3, LeastSquaresResult, Mat3, Quat,
    TransformPoint, Vec3,
};
use spatialrust_search::{KdTree, NearestNeighborIndex};

use crate::registration::{PointCloudRegistration, RegistrationResult};

type M3 = [[f64; 3]; 3];

/// Configuration for Generalized ICP (plane-to-plane).
///
/// GICP models each point's local surface as an anisotropic Gaussian and
/// minimizes the Mahalanobis distance between correspondences, combining the
/// robustness of point-to-plane ICP for both clouds. Local covariances are
/// estimated from k-nearest neighbors and regularized toward planar disks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GicpConfig {
    /// Maximum number of GICP iterations.
    pub max_iterations: usize,
    /// Maximum correspondence distance.
    pub max_correspondence_distance: f32,
    /// Number of neighbors used to estimate per-point covariance.
    pub k_neighbors: usize,
    /// Planar regularization: eigenvalue assigned along the surface normal.
    pub epsilon: f64,
    /// Stop when the transform update is smaller than this threshold.
    pub transformation_epsilon: f64,
    /// Stop when the fitness is smaller than this threshold.
    pub fitness_epsilon: f64,
    /// Minimum number of correspondences required per iteration.
    pub min_correspondences: usize,
    /// Initial transform guess mapping source into target frame.
    pub initial_guess: Isometry3<f32>,
}

impl Default for GicpConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            max_correspondence_distance: 1.0,
            k_neighbors: 20,
            epsilon: 1e-3,
            transformation_epsilon: 1e-8,
            fitness_epsilon: 1e-6,
            min_correspondences: 6,
            initial_guess: Isometry3::identity(),
        }
    }
}

impl GicpConfig {
    /// Creates a config with the given correspondence distance.
    #[must_use]
    pub fn with_correspondence_distance(max_correspondence_distance: f32) -> Self {
        Self { max_correspondence_distance, ..Self::default() }
    }
}

/// Generalized ICP registration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GicpRegistration {
    config: GicpConfig,
}

impl GicpRegistration {
    /// Creates a GICP algorithm from config.
    #[must_use]
    pub const fn new(config: GicpConfig) -> Self {
        Self { config }
    }

    /// Returns the config.
    #[must_use]
    pub const fn config(&self) -> GicpConfig {
        self.config
    }

    /// Aligns `source` to `target` using Generalized ICP.
    pub fn align_with_diagnostics(
        &self,
        source: &PointCloud,
        target: &PointCloud,
    ) -> SpatialResult<RegistrationResult> {
        if source.is_empty() || target.is_empty() {
            return Err(SpatialError::InvalidArgument(
                "GICP requires non-empty source and target point clouds".to_owned(),
            ));
        }
        if self.config.k_neighbors < 3 {
            return Err(SpatialError::InvalidArgument(
                "GICP needs k_neighbors >= 3 for covariance estimation".to_owned(),
            ));
        }

        let (sx, sy, sz) = source.positions3()?;
        let (tx, ty, tz) = target.positions3()?;
        let source_tree = KdTree::from_slices(sx, sy, sz);
        let target_tree = KdTree::from_slices(tx, ty, tz);
        let source_cov =
            covariances(sx, sy, sz, &source_tree, self.config.k_neighbors, self.config.epsilon);
        let target_cov =
            covariances(tx, ty, tz, &target_tree, self.config.k_neighbors, self.config.epsilon);

        let max_distance_squared =
            self.config.max_correspondence_distance * self.config.max_correspondence_distance;

        let mut transform = self.config.initial_guess;
        let mut transformed = vec![Vec3::new(0.0, 0.0, 0.0); source.len()];
        apply_transform(&mut transformed, sx, sy, sz, transform);

        let mut iterations = 0usize;
        let mut converged = false;

        for _ in 0..self.config.max_iterations {
            iterations += 1;
            let rot = mat3_to_f64(transform.rotation().to_mat3());
            let rot_t = transpose(rot);

            let mut hessian = [[0.0_f64; 6]; 6];
            let mut gradient = [0.0_f64; 6];
            let mut count = 0usize;

            for (i, point) in transformed.iter().enumerate() {
                let Some(neighbor) = target_tree.nearest_one(point.x, point.y, point.z) else {
                    continue;
                };
                if neighbor.distance_squared > max_distance_squared {
                    continue;
                }
                let j = neighbor.index;

                // C = C_target + R C_source R^T, then M = C^-1.
                let rotated_source = mat_mul(mat_mul(rot, source_cov[i]), rot_t);
                let combined = mat_add(target_cov[j], rotated_source);
                let Some(m) = inverse3(combined) else {
                    continue;
                };

                let e = [
                    f64::from(point.x) - f64::from(tx[j]),
                    f64::from(point.y) - f64::from(ty[j]),
                    f64::from(point.z) - f64::from(tz[j]),
                ];
                // Jacobian rows (3x6): [-skew(point) | I].
                let jac = jacobian([f64::from(point.x), f64::from(point.y), f64::from(point.z)]);
                let mj = mat3x6_premul(m, &jac);
                let me = mat_vec(m, e);

                for a in 0..6 {
                    for k in 0..3 {
                        gradient[a] += jac[k][a] * me[k];
                        for b in 0..6 {
                            hessian[a][b] += jac[k][a] * mj[k][b];
                        }
                    }
                }
                count += 1;
            }

            if count < self.config.min_correspondences {
                return Err(SpatialError::InvalidArgument(format!(
                    "GICP found only {} correspondences, minimum is {}",
                    count, self.config.min_correspondences
                )));
            }

            let a_rows: Vec<Vec<f64>> = hessian.iter().map(|row| row.to_vec()).collect();
            let neg_g: Vec<f64> = gradient.iter().map(|value| -value).collect();
            let LeastSquaresResult::Solved(solution) = solve_linear_system(a_rows, neg_g) else {
                return Err(SpatialError::InvalidArgument(
                    "GICP normal equations were singular".to_owned(),
                ));
            };

            transform = delta_from_solution(&solution).compose(transform);
            apply_transform(&mut transformed, sx, sy, sz, transform);

            if update_magnitude(&solution) < self.config.transformation_epsilon {
                converged = true;
                break;
            }
            if euclidean_fitness(&transformed, tx, ty, tz, &target_tree, max_distance_squared)
                < self.config.fitness_epsilon
            {
                converged = true;
                break;
            }
        }

        Ok(RegistrationResult {
            transform,
            fitness: euclidean_fitness(
                &transformed,
                tx,
                ty,
                tz,
                &target_tree,
                max_distance_squared,
            ),
            iterations,
            converged,
        })
    }
}

impl PointCloudRegistration for GicpRegistration {
    fn name(&self) -> &'static str {
        "GicpRegistration"
    }

    fn align(&self, source: &PointCloud, target: &PointCloud) -> SpatialResult<RegistrationResult> {
        self.align_with_diagnostics(source, target)
    }
}

/// Computes per-point plane-regularized covariances from k-nearest neighbors.
fn covariances(x: &[f32], y: &[f32], z: &[f32], tree: &KdTree, k: usize, epsilon: f64) -> Vec<M3> {
    let len = x.len();
    let mut out = Vec::with_capacity(len);
    for index in 0..len {
        let neighbors = tree.nearest_k(x[index], y[index], z[index], k);
        let mut acc = spatialrust_math::CovarianceAccumulator3::new();
        for neighbor in &neighbors {
            acc.push(Vec3::new(x[neighbor.index], y[neighbor.index], z[neighbor.index]));
        }
        let cov = acc
            .covariance()
            .unwrap_or_else(|| Mat3::from_rows([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]));
        out.push(regularize(cov, epsilon));
    }
    out
}

/// Replaces a covariance's eigenvalues with (epsilon, 1, 1) to model a planar disk.
fn regularize(cov: Mat3<f64>, epsilon: f64) -> M3 {
    let eigen = symmetric_eigen3(cov);
    // Eigenvectors are columns; eigenvalues are ascending, so column 0 is the normal.
    let v = eigen.eigenvectors.m;
    let mut result = [[0.0_f64; 3]; 3];
    for (col, scale) in [(0usize, epsilon), (1, 1.0), (2, 1.0)] {
        let axis = [v[0][col], v[1][col], v[2][col]];
        for r in 0..3 {
            for c in 0..3 {
                result[r][c] += scale * axis[r] * axis[c];
            }
        }
    }
    result
}

fn jacobian(p: [f64; 3]) -> [[f64; 6]; 3] {
    // [-skew(p) | I3]
    [
        [0.0, p[2], -p[1], 1.0, 0.0, 0.0],
        [-p[2], 0.0, p[0], 0.0, 1.0, 0.0],
        [p[1], -p[0], 0.0, 0.0, 0.0, 1.0],
    ]
}

fn apply_transform(
    transformed: &mut [Vec3<f32>],
    x: &[f32],
    y: &[f32],
    z: &[f32],
    transform: Isometry3<f32>,
) {
    for (index, point) in transformed.iter_mut().enumerate() {
        *point = transform.transform_point(Vec3::new(x[index], y[index], z[index]));
    }
}

fn delta_from_solution(solution: &[f64]) -> Isometry3<f32> {
    let (rx, ry, rz) = (solution[0], solution[1], solution[2]);
    let angle = (rx * rx + ry * ry + rz * rz).sqrt();
    let rotation = if angle > 1e-12 {
        let axis = Vec3::new((rx / angle) as f32, (ry / angle) as f32, (rz / angle) as f32);
        Quat::from_axis_angle(axis, angle as f32)
    } else {
        Quat::<f32>::identity()
    };
    Isometry3::new(rotation, Vec3::new(solution[3] as f32, solution[4] as f32, solution[5] as f32))
}

fn update_magnitude(solution: &[f64]) -> f64 {
    solution.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn euclidean_fitness(
    transformed: &[Vec3<f32>],
    tx: &[f32],
    ty: &[f32],
    tz: &[f32],
    tree: &KdTree,
    max_distance_squared: f32,
) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for point in transformed {
        let Some(neighbor) = tree.nearest_one(point.x, point.y, point.z) else {
            continue;
        };
        if neighbor.distance_squared > max_distance_squared {
            continue;
        }
        let dx = f64::from(point.x - tx[neighbor.index]);
        let dy = f64::from(point.y - ty[neighbor.index]);
        let dz = f64::from(point.z - tz[neighbor.index]);
        sum += dx * dx + dy * dy + dz * dz;
        count += 1;
    }
    if count == 0 {
        return f64::MAX;
    }
    sum / count as f64
}

fn mat3_to_f64(matrix: Mat3<f32>) -> M3 {
    let mut out = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = f64::from(matrix.m[r][c]);
        }
    }
    out
}

fn transpose(a: M3) -> M3 {
    let mut out = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[c][r];
        }
    }
    out
}

fn mat_mul(a: M3, b: M3) -> M3 {
    let mut out = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            for k in 0..3 {
                out[r][c] += a[r][k] * b[k][c];
            }
        }
    }
    out
}

fn mat_add(a: M3, b: M3) -> M3 {
    let mut out = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][c] + b[r][c];
        }
    }
    out
}

fn mat_vec(a: M3, v: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

/// Computes `m * j` where `m` is 3x3 and `j` is 3x6.
fn mat3x6_premul(m: M3, j: &[[f64; 6]; 3]) -> [[f64; 6]; 3] {
    let mut out = [[0.0_f64; 6]; 3];
    for r in 0..3 {
        for c in 0..6 {
            for k in 0..3 {
                out[r][c] += m[r][k] * j[k][c];
            }
        }
    }
    out
}

fn inverse3(a: M3) -> Option<M3> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-18 {
        return None;
    }
    let inv_det = 1.0 / det;
    let mut out = [[0.0_f64; 3]; 3];
    out[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det;
    out[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det;
    out[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det;
    out[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_det;
    out[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det;
    out[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det;
    out[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det;
    out[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det;
    out[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det;
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{GicpConfig, GicpRegistration};
    use crate::registration::PointCloudRegistration;
    use crate::transform::transform_point_cloud;
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};
    use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};

    /// Three perpendicular faces (a box corner) giving full 6-DoF constraint.
    fn box_corner() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for i in 0..10 {
            for j in 0..10 {
                let (a, b) = (i as f32 * 0.08, j as f32 * 0.08);
                builder.push_point([a, b, 0.0]).unwrap();
                builder.push_point([a, 0.0, b + 0.04]).unwrap();
                builder.push_point([0.0, a + 0.04, b + 0.04]).unwrap();
            }
        }
        builder.build().unwrap()
    }

    #[test]
    fn aligns_rotated_and_translated_source() {
        let target = box_corner();
        let misalignment = Isometry3::new(
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.06),
            Vec3::new(0.015, -0.01, 0.012),
        );
        let source = transform_point_cloud(&target, misalignment).unwrap();

        let gicp = GicpRegistration::new(GicpConfig {
            max_correspondence_distance: 0.2,
            max_iterations: 40,
            k_neighbors: 12,
            ..GicpConfig::default()
        });
        let result = gicp.align(&source, &target).unwrap();
        assert!(result.fitness < 1e-4, "fitness too high: {}", result.fitness);

        let composed = result.transform.compose(misalignment);
        let probe = Vec3::new(0.3, 0.4, 0.2);
        let restored = composed.transform_point(probe);
        assert!((restored.x - probe.x).abs() < 1e-2);
        assert!((restored.y - probe.y).abs() < 1e-2);
        assert!((restored.z - probe.z).abs() < 1e-2);
    }

    #[test]
    fn rejects_tiny_neighbor_count() {
        let cloud = box_corner();
        let gicp = GicpRegistration::new(GicpConfig { k_neighbors: 2, ..GicpConfig::default() });
        assert!(gicp.align(&cloud, &cloud).is_err());
    }
}
