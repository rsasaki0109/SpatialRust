// Explicit row/column indexing reads more clearly than iterators for fixed 3x3
// and 3x6 linear-algebra kernels.
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

use spatialrust_core::{HasPositions3, PointCloud, SpatialError, SpatialResult};
use spatialrust_math::{
    solve_linear_system, symmetric_eigen3, CovarianceAccumulator3, Isometry3, LeastSquaresResult,
    Mat3, Quat, TransformPoint, Vec3,
};

use crate::registration::{PointCloudRegistration, RegistrationResult};

type M3 = [[f64; 3]; 3];

/// Configuration for NDT (Normal Distributions Transform) registration.
///
/// The target cloud is discretized into a voxel grid; each cell with enough
/// points becomes a Gaussian (mean + covariance). The source is aligned by
/// minimizing the Mahalanobis distance of each transformed point to its target
/// cell distribution via Gauss-Newton (point-to-distribution NDT).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NdtConfig {
    /// Maximum number of optimization iterations.
    pub max_iterations: usize,
    /// Voxel size used to build the target distributions.
    pub resolution: f32,
    /// Planar regularization: smallest eigenvalue floor as a fraction of the largest.
    pub epsilon: f64,
    /// Minimum points required for a target voxel to form a distribution.
    pub min_points_per_voxel: usize,
    /// Stop when the transform update is smaller than this threshold.
    pub transformation_epsilon: f64,
    /// Stop when the fitness is smaller than this threshold.
    pub fitness_epsilon: f64,
    /// Minimum number of matched points required per iteration.
    pub min_correspondences: usize,
    /// Initial transform guess mapping source into target frame.
    pub initial_guess: Isometry3<f32>,
}

impl Default for NdtConfig {
    fn default() -> Self {
        Self {
            max_iterations: 35,
            resolution: 1.0,
            epsilon: 1e-3,
            min_points_per_voxel: 5,
            transformation_epsilon: 1e-8,
            fitness_epsilon: 1e-6,
            min_correspondences: 6,
            initial_guess: Isometry3::identity(),
        }
    }
}

impl NdtConfig {
    /// Creates a config with the given voxel resolution.
    #[must_use]
    pub fn with_resolution(resolution: f32) -> Self {
        Self { resolution, ..Self::default() }
    }
}

/// NDT registration (point-to-distribution).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NdtRegistration {
    config: NdtConfig,
}

struct VoxelDistribution {
    mean: [f64; 3],
    information: M3,
}

impl NdtRegistration {
    /// Creates an NDT algorithm from config.
    #[must_use]
    pub const fn new(config: NdtConfig) -> Self {
        Self { config }
    }

    /// Returns the config.
    #[must_use]
    pub const fn config(&self) -> NdtConfig {
        self.config
    }

    /// Aligns `source` to `target` using NDT.
    pub fn align_with_diagnostics(
        &self,
        source: &PointCloud,
        target: &PointCloud,
    ) -> SpatialResult<RegistrationResult> {
        if source.is_empty() || target.is_empty() {
            return Err(SpatialError::InvalidArgument(
                "NDT requires non-empty source and target point clouds".to_owned(),
            ));
        }
        if self.config.resolution <= 0.0 {
            return Err(SpatialError::InvalidArgument(
                "NDT resolution must be positive".to_owned(),
            ));
        }

        let (sx, sy, sz) = source.positions3()?;
        let (tx, ty, tz) = target.positions3()?;
        let inv_res = 1.0 / self.config.resolution;
        let distributions = build_distributions(
            tx,
            ty,
            tz,
            inv_res,
            self.config.min_points_per_voxel,
            self.config.epsilon,
        );
        if distributions.is_empty() {
            return Err(SpatialError::InvalidArgument(
                "NDT found no target voxels with enough points; increase resolution".to_owned(),
            ));
        }

        let mut transform = self.config.initial_guess;
        let mut transformed = vec![Vec3::new(0.0, 0.0, 0.0); source.len()];
        apply_transform(&mut transformed, sx, sy, sz, transform);

        let mut iterations = 0usize;
        let mut converged = false;
        let mut lambda = 1e-3_f64;
        let mut cost = mahalanobis_fitness(&transformed, &distributions, inv_res);

        for _ in 0..self.config.max_iterations {
            iterations += 1;
            let mut hessian = [[0.0_f64; 6]; 6];
            let mut gradient = [0.0_f64; 6];
            let mut count = 0usize;

            for point in &transformed {
                let cell = cell_of(point, inv_res);
                let Some(dist) = distributions.get(&cell) else {
                    continue;
                };
                let d = [
                    f64::from(point.x) - dist.mean[0],
                    f64::from(point.y) - dist.mean[1],
                    f64::from(point.z) - dist.mean[2],
                ];
                let jac = jacobian([f64::from(point.x), f64::from(point.y), f64::from(point.z)]);
                let mj = mat3x6_premul(dist.information, &jac);
                let md = mat_vec(dist.information, d);
                for a in 0..6 {
                    for k in 0..3 {
                        gradient[a] += jac[k][a] * md[k];
                        for b in 0..6 {
                            hessian[a][b] += jac[k][a] * mj[k][b];
                        }
                    }
                }
                count += 1;
            }

            if count < self.config.min_correspondences {
                return Err(SpatialError::InvalidArgument(format!(
                    "NDT matched only {} points to target voxels, minimum is {}",
                    count, self.config.min_correspondences
                )));
            }

            // Levenberg-Marquardt: damp the Hessian and only accept steps that
            // lower the cost, otherwise increase damping and retry. Hard voxel
            // assignment makes the objective non-smooth, so a plain Gauss-Newton
            // step can overshoot and diverge.
            let mut step_accepted = false;
            for _ in 0..8 {
                let mut a_rows: Vec<Vec<f64>> = hessian.iter().map(|row| row.to_vec()).collect();
                for d in 0..6 {
                    a_rows[d][d] += lambda * hessian[d][d].max(1e-9);
                }
                let neg_g: Vec<f64> = gradient.iter().map(|value| -value).collect();
                let LeastSquaresResult::Solved(solution) = solve_linear_system(a_rows, neg_g)
                else {
                    lambda *= 4.0;
                    continue;
                };

                let candidate = delta_from_solution(&solution).compose(transform);
                let mut candidate_points = transformed.clone();
                apply_transform(&mut candidate_points, sx, sy, sz, candidate);
                let candidate_cost =
                    mahalanobis_fitness(&candidate_points, &distributions, inv_res);

                if candidate_cost < cost {
                    transform = candidate;
                    transformed = candidate_points;
                    let step = update_magnitude(&solution);
                    cost = candidate_cost;
                    lambda = (lambda * 0.5).max(1e-9);
                    step_accepted = true;
                    if step < self.config.transformation_epsilon
                        || cost < self.config.fitness_epsilon
                    {
                        converged = true;
                    }
                    break;
                }
                lambda *= 4.0;
            }

            if converged || !step_accepted {
                converged = converged || step_accepted;
                break;
            }
        }

        Ok(RegistrationResult { transform, fitness: cost, iterations, converged })
    }
}

impl PointCloudRegistration for NdtRegistration {
    fn name(&self) -> &'static str {
        "NdtRegistration"
    }

    fn align(&self, source: &PointCloud, target: &PointCloud) -> SpatialResult<RegistrationResult> {
        self.align_with_diagnostics(source, target)
    }
}

fn cell_of(point: &Vec3<f32>, inv_res: f32) -> (i64, i64, i64) {
    (
        (point.x * inv_res).floor() as i64,
        (point.y * inv_res).floor() as i64,
        (point.z * inv_res).floor() as i64,
    )
}

/// Builds per-voxel Gaussian distributions (mean + inverse covariance) for the target.
fn build_distributions(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    inv_res: f32,
    min_points: usize,
    epsilon: f64,
) -> HashMap<(i64, i64, i64), VoxelDistribution> {
    let mut accumulators: HashMap<(i64, i64, i64), CovarianceAccumulator3> = HashMap::new();
    for index in 0..x.len() {
        let point = Vec3::new(x[index], y[index], z[index]);
        let cell = cell_of(&point, inv_res);
        accumulators.entry(cell).or_default().push(point);
    }

    let mut distributions = HashMap::with_capacity(accumulators.len());
    for (cell, acc) in accumulators {
        if (acc.count() as usize) < min_points.max(3) {
            continue;
        }
        let (Some(mean), Some(cov)) = (acc.mean(), acc.covariance()) else {
            continue;
        };
        let regularized = regularize(cov, epsilon);
        let Some(information) = inverse3(regularized) else {
            continue;
        };
        distributions
            .insert(cell, VoxelDistribution { mean: [mean.x, mean.y, mean.z], information });
    }
    distributions
}

/// Floors small eigenvalues so a near-planar cell still yields an invertible covariance.
fn regularize(cov: Mat3<f64>, epsilon: f64) -> M3 {
    let eigen = symmetric_eigen3(cov);
    let max_eig = eigen.eigenvalues[2].max(1e-12);
    let floor = epsilon * max_eig;
    let v = eigen.eigenvectors.m;
    let mut result = [[0.0_f64; 3]; 3];
    for col in 0..3 {
        let lambda = eigen.eigenvalues[col].max(floor);
        let axis = [v[0][col], v[1][col], v[2][col]];
        for r in 0..3 {
            for c in 0..3 {
                result[r][c] += lambda * axis[r] * axis[c];
            }
        }
    }
    result
}

fn mahalanobis_fitness(
    transformed: &[Vec3<f32>],
    distributions: &HashMap<(i64, i64, i64), VoxelDistribution>,
    inv_res: f32,
) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for point in transformed {
        let cell = cell_of(point, inv_res);
        let Some(dist) = distributions.get(&cell) else {
            continue;
        };
        let d = [
            f64::from(point.x) - dist.mean[0],
            f64::from(point.y) - dist.mean[1],
            f64::from(point.z) - dist.mean[2],
        ];
        let md = mat_vec(dist.information, d);
        sum += d[0] * md[0] + d[1] * md[1] + d[2] * md[2];
        count += 1;
    }
    if count == 0 {
        return f64::MAX;
    }
    sum / count as f64
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

fn jacobian(p: [f64; 3]) -> [[f64; 6]; 3] {
    [
        [0.0, p[2], -p[1], 1.0, 0.0, 0.0],
        [-p[2], 0.0, p[0], 0.0, 1.0, 0.0],
        [p[1], -p[0], 0.0, 0.0, 0.0, 1.0],
    ]
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

fn mat_vec(a: M3, v: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

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
    use super::{NdtConfig, NdtRegistration};
    use crate::registration::PointCloudRegistration;
    use crate::transform::transform_point_cloud;
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};
    use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};

    /// Dense box corner so each voxel holds enough points for a distribution.
    fn box_corner() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for i in 0..30 {
            for j in 0..30 {
                let (a, b) = (i as f32 * 0.05, j as f32 * 0.05);
                builder.push_point([a, b, 0.0]).unwrap();
                builder.push_point([a, 0.0, b + 0.02]).unwrap();
                builder.push_point([0.0, a + 0.02, b + 0.02]).unwrap();
            }
        }
        builder.build().unwrap()
    }

    #[test]
    fn aligns_rotated_and_translated_source() {
        let target = box_corner();
        let misalignment = Isometry3::new(
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.03),
            Vec3::new(0.02, -0.015, 0.01),
        );
        let source = transform_point_cloud(&target, misalignment).unwrap();

        let ndt = NdtRegistration::new(NdtConfig {
            resolution: 0.2,
            max_iterations: 60,
            min_points_per_voxel: 4,
            ..NdtConfig::default()
        });
        let result = ndt.align(&source, &target).unwrap();

        let composed = result.transform.compose(misalignment);
        let probe = Vec3::new(0.4, 0.5, 0.3);
        let restored = composed.transform_point(probe);
        assert!((restored.x - probe.x).abs() < 2e-2, "x off: {}", restored.x);
        assert!((restored.y - probe.y).abs() < 2e-2, "y off: {}", restored.y);
        assert!((restored.z - probe.z).abs() < 2e-2, "z off: {}", restored.z);
    }

    #[test]
    fn rejects_nonpositive_resolution() {
        let cloud = box_corner();
        let ndt = NdtRegistration::new(NdtConfig { resolution: 0.0, ..NdtConfig::default() });
        assert!(ndt.align(&cloud, &cloud).is_err());
    }
}
