use spatialrust_core::{HasPositions3, PointCloud, SpatialError, SpatialResult};
use spatialrust_math::{Isometry3, TransformPoint, Vec3};
use spatialrust_search::{KdTree, NearestNeighborIndex};

use crate::kabsch::estimate_rigid_transform;
use crate::registration::{PointCloudRegistration, RegistrationResult};

/// Configuration for point-to-point ICP.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IcpConfig {
    /// Maximum number of ICP iterations.
    pub max_iterations: usize,
    /// Maximum correspondence distance.
    pub max_correspondence_distance: f32,
    /// Stop when the transform update is smaller than this threshold.
    pub transformation_epsilon: f64,
    /// Stop when fitness improvement is smaller than this threshold.
    pub fitness_epsilon: f64,
    /// Minimum number of correspondences required per iteration.
    pub min_correspondences: usize,
    /// Initial transform guess mapping source into target frame.
    pub initial_guess: Isometry3<f32>,
}

impl Default for IcpConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            max_correspondence_distance: 1.0,
            transformation_epsilon: 1e-8,
            fitness_epsilon: 1e-6,
            min_correspondences: 3,
            initial_guess: Isometry3::identity(),
        }
    }
}

impl IcpConfig {
    /// Creates a config with the given correspondence distance.
    #[must_use]
    pub fn with_correspondence_distance(max_correspondence_distance: f32) -> Self {
        Self {
            max_correspondence_distance,
            ..Self::default()
        }
    }
}

/// Point-to-point ICP registration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IcpRegistration {
    config: IcpConfig,
}

impl IcpRegistration {
    /// Creates an ICP registration algorithm from config.
    #[must_use]
    pub const fn new(config: IcpConfig) -> Self {
        Self { config }
    }

    /// Returns the ICP config.
    #[must_use]
    pub const fn config(&self) -> IcpConfig {
        self.config
    }

    /// Aligns `source` to `target` using iterative closest point.
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
        let tree = KdTree::from_slices(target_x, target_y, target_z);
        let max_distance_squared =
            self.config.max_correspondence_distance * self.config.max_correspondence_distance;

        let mut transform = self.config.initial_guess;
        let mut transformed = Vec::with_capacity(source.len());
        for index in 0..source.len() {
            transformed.push(Vec3::new(source_x[index], source_y[index], source_z[index]));
        }
        apply_transform_in_place(&mut transformed, source_x, source_y, source_z, transform);

        let mut iterations = 0usize;
        let mut converged = false;

        for _ in 0..self.config.max_iterations {
            iterations += 1;
            let mut pairs_source = Vec::new();
            let mut pairs_target = Vec::new();

            for point in &transformed {
                let Some(neighbor) = tree.nearest_one(point.x, point.y, point.z) else {
                    continue;
                };
                if neighbor.distance_squared <= max_distance_squared {
                    pairs_source.push(*point);
                    pairs_target.push(Vec3::new(
                        target_x[neighbor.index],
                        target_y[neighbor.index],
                        target_z[neighbor.index],
                    ));
                }
            }

            if pairs_source.len() < self.config.min_correspondences {
                return Err(SpatialError::InvalidArgument(format!(
                    "ICP found only {} correspondences, minimum is {}",
                    pairs_source.len(),
                    self.config.min_correspondences
                )));
            }

            let Some(delta) = estimate_rigid_transform(&pairs_source, &pairs_target) else {
                return Err(SpatialError::InvalidArgument(
                    "ICP failed to estimate a rigid transform".to_owned(),
                ));
            };

            transform = delta.compose(transform);
            apply_transform_in_place(&mut transformed, source_x, source_y, source_z, transform);

            let fitness = final_fitness(
                &transformed,
                target_x,
                target_y,
                target_z,
                &tree,
                max_distance_squared,
            );
            if fitness < self.config.fitness_epsilon {
                converged = true;
                break;
            }
            if transform_delta_below_epsilon(delta, self.config.transformation_epsilon) {
                converged = true;
                break;
            }
        }

        Ok(RegistrationResult {
            transform,
            fitness: final_fitness(
                &transformed,
                target_x,
                target_y,
                target_z,
                &tree,
                max_distance_squared,
            ),
            iterations,
            converged,
        })
    }
}

impl PointCloudRegistration for IcpRegistration {
    fn name(&self) -> &'static str {
        "IcpRegistration"
    }

    fn align(&self, source: &PointCloud, target: &PointCloud) -> SpatialResult<RegistrationResult> {
        self.align_with_diagnostics(source, target)
    }
}

fn apply_transform_in_place(
    transformed: &mut [Vec3<f32>],
    source_x: &[f32],
    source_y: &[f32],
    source_z: &[f32],
    transform: Isometry3<f32>,
) {
    for (index, point) in transformed.iter_mut().enumerate() {
        *point = transform.transform_point(Vec3::new(source_x[index], source_y[index], source_z[index]));
    }
}

fn mean_squared_error(source: &[Vec3<f32>], target: &[Vec3<f32>]) -> f64 {
    let mut sum = 0.0_f64;
    for (src, dst) in source.iter().zip(target) {
        let dx = f64::from(src.x - dst.x);
        let dy = f64::from(src.y - dst.y);
        let dz = f64::from(src.z - dst.z);
        sum += dx * dx + dy * dy + dz * dz;
    }
    sum / source.len() as f64
}

fn final_fitness(
    transformed: &[Vec3<f32>],
    target_x: &[f32],
    target_y: &[f32],
    target_z: &[f32],
    tree: &KdTree,
    max_distance_squared: f32,
) -> f64 {
    let mut pairs_source = Vec::new();
    let mut pairs_target = Vec::new();
    for point in transformed {
        let Some(neighbor) = tree.nearest_one(point.x, point.y, point.z) else {
            continue;
        };
        if neighbor.distance_squared <= max_distance_squared {
            pairs_source.push(*point);
            pairs_target.push(Vec3::new(
                target_x[neighbor.index],
                target_y[neighbor.index],
                target_z[neighbor.index],
            ));
        }
    }
    if pairs_source.is_empty() {
        return f64::MAX;
    }
    mean_squared_error(&pairs_source, &pairs_target)
}

fn transform_delta_below_epsilon(delta: Isometry3<f32>, epsilon: f64) -> bool {
    let translation = delta.translation();
    let translation_norm = f64::from(
        (translation.x * translation.x + translation.y * translation.y + translation.z * translation.z).sqrt(),
    );
    translation_norm < epsilon
}

#[cfg(test)]
mod tests {
    use super::{IcpConfig, IcpRegistration};
    use crate::registration::PointCloudRegistration;
    use crate::transform::transform_point_cloud;
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};
    use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};

    fn plane_cloud() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for x in 0..6 {
            for y in 0..6 {
                for z in 0..3 {
                    builder
                        .push_point([x as f32 * 0.05, y as f32 * 0.05, z as f32 * 0.05])
                        .unwrap();
                }
            }
        }
        builder.build().unwrap()
    }

    #[test]
    fn aligns_translated_source_to_target() {
        let target = plane_cloud();
        let shift = Isometry3::new(Quat::<f32>::identity(), Vec3::new(0.02, -0.01, 0.0));
        let source = transform_point_cloud(&target, shift).unwrap();

        let registration = IcpRegistration::new(IcpConfig {
            max_correspondence_distance: 0.1,
            max_iterations: 30,
            ..IcpConfig::default()
        });
        let result = registration.align(&source, &target).unwrap();
        assert!(result.fitness < 1e-4);
        assert!(result.converged);

        let composed = result.transform.compose(shift);
        let probe = Vec3::new(0.2, 0.3, 0.0);
        let restored = composed.transform_point(probe);
        assert!((restored.x - probe.x).abs() < 5e-3);
        assert!((restored.y - probe.y).abs() < 5e-3);
    }

    #[test]
    fn aligns_rotated_and_translated_source() {
        let target = plane_cloud();
        let misalignment = Isometry3::new(
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.15),
            Vec3::new(0.02, 0.01, 0.0),
        );
        let source = transform_point_cloud(&target, misalignment).unwrap();

        let registration = IcpRegistration::new(IcpConfig {
            max_correspondence_distance: 0.1,
            max_iterations: 40,
            ..IcpConfig::default()
        });
        let result = registration.align(&source, &target).unwrap();
        assert!(result.fitness < 1e-3);

        let composed = result.transform.compose(misalignment);
        let probe = Vec3::new(0.15, 0.25, 0.0);
        let restored = composed.transform_point(probe);
        assert!((restored.x - probe.x).abs() < 1e-2);
        assert!((restored.y - probe.y).abs() < 1e-2);
    }
}
