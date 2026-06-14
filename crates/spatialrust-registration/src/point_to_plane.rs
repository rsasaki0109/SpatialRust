use spatialrust_core::{HasNormals3, HasPositions3, PointCloud, SpatialError, SpatialResult};
use spatialrust_math::{
    solve_linear_system, Isometry3, LeastSquaresResult, Quat, TransformPoint, Vec3,
};
use spatialrust_search::{KdTree, NearestNeighborIndex};

use crate::registration::{PointCloudRegistration, RegistrationResult};

/// Configuration for point-to-plane ICP.
///
/// Point-to-plane ICP minimizes the distance from each transformed source point
/// to the tangent plane of its target correspondence, which converges faster and
/// more accurately than point-to-point ICP on locally planar surfaces. The target
/// cloud must carry normals (e.g. from normal estimation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointToPlaneIcpConfig {
    /// Maximum number of ICP iterations.
    pub max_iterations: usize,
    /// Maximum correspondence distance.
    pub max_correspondence_distance: f32,
    /// Stop when the transform update is smaller than this threshold.
    pub transformation_epsilon: f64,
    /// Stop when the point-to-plane fitness is smaller than this threshold.
    pub fitness_epsilon: f64,
    /// Minimum number of correspondences required per iteration.
    pub min_correspondences: usize,
    /// Initial transform guess mapping source into target frame.
    pub initial_guess: Isometry3<f32>,
}

impl Default for PointToPlaneIcpConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            max_correspondence_distance: 1.0,
            transformation_epsilon: 1e-8,
            fitness_epsilon: 1e-6,
            min_correspondences: 6,
            initial_guess: Isometry3::identity(),
        }
    }
}

impl PointToPlaneIcpConfig {
    /// Creates a config with the given correspondence distance.
    #[must_use]
    pub fn with_correspondence_distance(max_correspondence_distance: f32) -> Self {
        Self { max_correspondence_distance, ..Self::default() }
    }
}

/// Point-to-plane ICP registration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointToPlaneIcp {
    config: PointToPlaneIcpConfig,
}

impl PointToPlaneIcp {
    /// Creates a point-to-plane ICP algorithm from config.
    #[must_use]
    pub const fn new(config: PointToPlaneIcpConfig) -> Self {
        Self { config }
    }

    /// Returns the config.
    #[must_use]
    pub const fn config(&self) -> PointToPlaneIcpConfig {
        self.config
    }

    /// Aligns `source` to `target` minimizing point-to-plane distance.
    ///
    /// `target` must provide normals.
    pub fn align_with_diagnostics(
        &self,
        source: &PointCloud,
        target: &PointCloud,
    ) -> SpatialResult<RegistrationResult> {
        if source.is_empty() || target.is_empty() {
            return Err(SpatialError::InvalidArgument(
                "ICP requires non-empty source and target point clouds".to_owned(),
            ));
        }

        let (source_x, source_y, source_z) = source.positions3()?;
        let (target_x, target_y, target_z) = target.positions3()?;
        let (normal_x, normal_y, normal_z) = target.normals3()?;
        let tree = KdTree::from_slices(target_x, target_y, target_z);
        let max_distance_squared =
            self.config.max_correspondence_distance * self.config.max_correspondence_distance;

        let mut transform = self.config.initial_guess;
        let mut transformed = vec![Vec3::new(0.0, 0.0, 0.0); source.len()];
        apply_transform(&mut transformed, source_x, source_y, source_z, transform);

        let mut iterations = 0usize;
        let mut converged = false;

        for _ in 0..self.config.max_iterations {
            iterations += 1;

            // Accumulate the 6x6 normal equations for the linearized problem.
            let mut ata = [[0.0_f64; 6]; 6];
            let mut atb = [0.0_f64; 6];
            let mut count = 0usize;

            for point in &transformed {
                let Some(neighbor) = tree.nearest_one(point.x, point.y, point.z) else {
                    continue;
                };
                if neighbor.distance_squared > max_distance_squared {
                    continue;
                }
                let q = Vec3::new(
                    target_x[neighbor.index],
                    target_y[neighbor.index],
                    target_z[neighbor.index],
                );
                let n = Vec3::new(
                    normal_x[neighbor.index],
                    normal_y[neighbor.index],
                    normal_z[neighbor.index],
                );

                // Jacobian row for x = [rx, ry, rz, tx, ty, tz]: [p x n, n].
                let c = point.cross(n);
                let row = [
                    f64::from(c.x),
                    f64::from(c.y),
                    f64::from(c.z),
                    f64::from(n.x),
                    f64::from(n.y),
                    f64::from(n.z),
                ];
                let residual = f64::from((*point - q).dot(n));

                for i in 0..6 {
                    atb[i] -= row[i] * residual;
                    for j in 0..6 {
                        ata[i][j] += row[i] * row[j];
                    }
                }
                count += 1;
            }

            if count < self.config.min_correspondences {
                return Err(SpatialError::InvalidArgument(format!(
                    "point-to-plane ICP found only {} correspondences, minimum is {}",
                    count, self.config.min_correspondences
                )));
            }

            let a_rows: Vec<Vec<f64>> = ata.iter().map(|row| row.to_vec()).collect();
            let LeastSquaresResult::Solved(solution) = solve_linear_system(a_rows, atb.to_vec())
            else {
                return Err(SpatialError::InvalidArgument(
                    "point-to-plane ICP normal equations were singular".to_owned(),
                ));
            };

            let delta = delta_from_solution(&solution);
            transform = delta.compose(transform);
            apply_transform(&mut transformed, source_x, source_y, source_z, transform);

            if update_magnitude(&solution) < self.config.transformation_epsilon {
                converged = true;
                break;
            }
            let fitness = point_to_plane_fitness(
                &transformed,
                target_x,
                target_y,
                target_z,
                normal_x,
                normal_y,
                normal_z,
                &tree,
                max_distance_squared,
            );
            if fitness < self.config.fitness_epsilon {
                converged = true;
                break;
            }
        }

        Ok(RegistrationResult {
            transform,
            fitness: point_to_plane_fitness(
                &transformed,
                target_x,
                target_y,
                target_z,
                normal_x,
                normal_y,
                normal_z,
                &tree,
                max_distance_squared,
            ),
            iterations,
            converged,
        })
    }
}

impl PointCloudRegistration for PointToPlaneIcp {
    fn name(&self) -> &'static str {
        "PointToPlaneIcp"
    }

    fn align(&self, source: &PointCloud, target: &PointCloud) -> SpatialResult<RegistrationResult> {
        self.align_with_diagnostics(source, target)
    }
}

fn apply_transform(
    transformed: &mut [Vec3<f32>],
    source_x: &[f32],
    source_y: &[f32],
    source_z: &[f32],
    transform: Isometry3<f32>,
) {
    for (index, point) in transformed.iter_mut().enumerate() {
        *point =
            transform.transform_point(Vec3::new(source_x[index], source_y[index], source_z[index]));
    }
}

/// Builds the incremental isometry from the linearized rotation/translation solution.
fn delta_from_solution(solution: &[f64]) -> Isometry3<f32> {
    let (rx, ry, rz) = (solution[0], solution[1], solution[2]);
    let angle = (rx * rx + ry * ry + rz * rz).sqrt();
    let rotation = if angle > 1e-12 {
        let axis = Vec3::new((rx / angle) as f32, (ry / angle) as f32, (rz / angle) as f32);
        Quat::from_axis_angle(axis, angle as f32)
    } else {
        Quat::<f32>::identity()
    };
    let translation = Vec3::new(solution[3] as f32, solution[4] as f32, solution[5] as f32);
    Isometry3::new(rotation, translation)
}

fn update_magnitude(solution: &[f64]) -> f64 {
    solution.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[allow(clippy::too_many_arguments)]
fn point_to_plane_fitness(
    transformed: &[Vec3<f32>],
    target_x: &[f32],
    target_y: &[f32],
    target_z: &[f32],
    normal_x: &[f32],
    normal_y: &[f32],
    normal_z: &[f32],
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
        let q =
            Vec3::new(target_x[neighbor.index], target_y[neighbor.index], target_z[neighbor.index]);
        let n =
            Vec3::new(normal_x[neighbor.index], normal_y[neighbor.index], normal_z[neighbor.index]);
        let residual = f64::from((*point - q).dot(n));
        sum += residual * residual;
        count += 1;
    }
    if count == 0 {
        return f64::MAX;
    }
    sum / count as f64
}

#[cfg(test)]
mod tests {
    use super::{PointToPlaneIcp, PointToPlaneIcpConfig};
    use crate::registration::PointCloudRegistration;
    use crate::transform::transform_point_cloud;
    use spatialrust_core::{DType, FieldSemantic, PointCloudBuilder, PointField, PointSchema};
    use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};

    fn schema_with_normals() -> PointSchema {
        PointSchema::new()
            .with_field(PointField::scalar("x", FieldSemantic::PositionX, DType::F32))
            .with_field(PointField::scalar("y", FieldSemantic::PositionY, DType::F32))
            .with_field(PointField::scalar("z", FieldSemantic::PositionZ, DType::F32))
            .with_field(PointField::scalar("normal_x", FieldSemantic::NormalX, DType::F32))
            .with_field(PointField::scalar("normal_y", FieldSemantic::NormalY, DType::F32))
            .with_field(PointField::scalar("normal_z", FieldSemantic::NormalZ, DType::F32))
    }

    /// Two perpendicular faces with normals, giving full 6-DoF constraint.
    fn box_corner() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(schema_with_normals());
        for i in 0..8 {
            for j in 0..8 {
                let (a, b) = (i as f32 * 0.1, j as f32 * 0.1);
                // floor z=0, normal +Z
                builder.push_point([a, b, 0.0, 0.0, 0.0, 1.0]).unwrap();
                // wall y=0, normal +Y
                builder.push_point([a, 0.0, b + 0.05, 0.0, 1.0, 0.0]).unwrap();
                // wall x=0, normal +X
                builder.push_point([0.0, a + 0.05, b + 0.05, 1.0, 0.0, 0.0]).unwrap();
            }
        }
        builder.build().unwrap()
    }

    #[test]
    fn aligns_rotated_and_translated_source() {
        let target = box_corner();
        let misalignment = Isometry3::new(
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.08),
            Vec3::new(0.02, -0.015, 0.01),
        );
        let source = transform_point_cloud(&target, misalignment).unwrap();

        let icp = PointToPlaneIcp::new(PointToPlaneIcpConfig {
            max_correspondence_distance: 0.2,
            max_iterations: 40,
            ..PointToPlaneIcpConfig::default()
        });
        let result = icp.align(&source, &target).unwrap();
        assert!(result.fitness < 1e-5, "fitness too high: {}", result.fitness);

        // Composing the recovered transform with the misalignment restores points.
        let composed = result.transform.compose(misalignment);
        let probe = Vec3::new(0.3, 0.4, 0.2);
        let restored = composed.transform_point(probe);
        assert!((restored.x - probe.x).abs() < 5e-3);
        assert!((restored.y - probe.y).abs() < 5e-3);
        assert!((restored.z - probe.z).abs() < 5e-3);
    }

    #[test]
    fn requires_target_normals() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        let cloud = builder.build().unwrap();
        let icp = PointToPlaneIcp::new(PointToPlaneIcpConfig::default());
        assert!(icp.align(&cloud, &cloud).is_err());
    }
}
