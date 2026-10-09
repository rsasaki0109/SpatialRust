use spatialrust_core::{HasPositions3, PointCloud, SpatialError, SpatialResult};
use spatialrust_math::{Isometry3, TransformPoint, Vec3};
use spatialrust_search::KdTree;

use crate::kabsch::estimate_rigid_transform;
use crate::registration::{PointCloudRegistration, RegistrationResult};

/// Why a successful ICP call stopped; convergence does not certify the pose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcpStopReason {
    /// The absolute mean squared correspondence distance was small enough.
    FitnessThreshold,
    /// Both translation length and rotation angle of the update were small enough.
    TransformThreshold,
    /// The iteration budget was exhausted without meeting a convergence threshold.
    IterationLimit,
}

/// Measurements for one completed update (no additional nearest-neighbor pass).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IcpIteration {
    /// One-based update number.
    pub iteration: usize,
    /// Gated correspondences used to estimate this update.
    pub correspondences: usize,
    /// Gated correspondences after applying the update and rematching.
    pub evaluated_correspondences: usize,
    /// Post-update mean squared distance, or f64::MAX if there are no matches.
    pub fitness: f64,
    /// Previous post-update fitness minus this fitness; None for the first update.
    /// Membership can change, so this is not an error on a fixed set of points.
    pub fitness_change: Option<f64>,
    /// Update translation length in the input coordinate units.
    pub translation_delta: f64,
    /// Shortest update rotation angle in radians.
    pub rotation_delta_radians: f64,
}

/// Opt-in ICP history. Ordinary alignment does not allocate this history.
#[derive(Clone, Debug, PartialEq)]
pub struct IcpDiagnostics {
    /// Ordinary registration result from the same updates.
    pub result: RegistrationResult,
    /// One row per completed update, in execution order.
    pub history: Vec<IcpIteration>,
    /// Criterion that stopped this successful call.
    pub stop_reason: IcpStopReason,
}

/// Configuration for point-to-point ICP.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IcpConfig {
    /// Maximum number of ICP iterations.
    pub max_iterations: usize,
    /// Maximum correspondence distance.
    pub max_correspondence_distance: f32,
    /// Stop when both translation length (coordinate units) and rotation angle
    /// (radians) of the transform update are smaller than this threshold.
    pub transformation_epsilon: f64,
    /// Stop when the mean squared correspondence distance is below this threshold.
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
        Self { max_correspondence_distance, ..Self::default() }
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
        self.align_impl(source, target, |_| {}).map(|(result, _)| result)
    }

    /// Aligns with an opt-in history; errors use the same path as ordinary alignment.
    pub fn align_with_trace(
        &self,
        source: &PointCloud,
        target: &PointCloud,
    ) -> SpatialResult<IcpDiagnostics> {
        let mut history = Vec::new();
        let (result, stop_reason) = self.align_impl(source, target, |row| history.push(row))?;
        Ok(IcpDiagnostics { result, history, stop_reason })
    }

    fn align_impl(
        &self,
        source: &PointCloud,
        target: &PointCloud,
        mut observe: impl FnMut(IcpIteration),
    ) -> SpatialResult<(RegistrationResult, IcpStopReason)> {
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
        let mut stop_reason = IcpStopReason::IterationLimit;
        let mut previous_fitness = None;

        for _ in 0..self.config.max_iterations {
            iterations += 1;
            let mut pairs_source = Vec::new();
            let mut pairs_target = Vec::new();

            for point in &transformed {
                let Some(neighbor) =
                    tree.nearest_one_within(point.x, point.y, point.z, max_distance_squared)
                else {
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

            let (fitness, evaluated_correspondences) = evaluate_fitness(
                &transformed,
                target_x,
                target_y,
                target_z,
                &tree,
                max_distance_squared,
            );
            let (translation_delta, rotation_delta_radians) = transform_delta_magnitudes(delta);
            observe(IcpIteration {
                iteration: iterations,
                correspondences: pairs_source.len(),
                evaluated_correspondences,
                fitness,
                fitness_change: previous_fitness.map(|previous| previous - fitness),
                translation_delta,
                rotation_delta_radians,
            });
            previous_fitness = Some(fitness);
            if fitness < self.config.fitness_epsilon {
                converged = true;
                stop_reason = IcpStopReason::FitnessThreshold;
                break;
            }
            if transform_delta_below_epsilon(delta, self.config.transformation_epsilon) {
                converged = true;
                stop_reason = IcpStopReason::TransformThreshold;
                break;
            }
        }

        Ok((
            RegistrationResult {
                transform,
                fitness: evaluate_fitness(
                    &transformed,
                    target_x,
                    target_y,
                    target_z,
                    &tree,
                    max_distance_squared,
                )
                .0,
                iterations,
                converged,
            },
            stop_reason,
        ))
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
        *point =
            transform.transform_point(Vec3::new(source_x[index], source_y[index], source_z[index]));
    }
}

fn evaluate_fitness(
    transformed: &[Vec3<f32>],
    target_x: &[f32],
    target_y: &[f32],
    target_z: &[f32],
    tree: &KdTree,
    max_distance_squared: f32,
) -> (f64, usize) {
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for point in transformed {
        let Some(neighbor) =
            tree.nearest_one_within(point.x, point.y, point.z, max_distance_squared)
        else {
            continue;
        };
        if neighbor.distance_squared <= max_distance_squared {
            let dx = f64::from(point.x - target_x[neighbor.index]);
            let dy = f64::from(point.y - target_y[neighbor.index]);
            let dz = f64::from(point.z - target_z[neighbor.index]);
            sum += dx * dx + dy * dy + dz * dz;
            count += 1;
        }
    }
    (if count == 0 { f64::MAX } else { sum / count as f64 }, count)
}

fn transform_delta_below_epsilon(delta: Isometry3<f32>, epsilon: f64) -> bool {
    let (translation_norm, rotation_angle) = transform_delta_magnitudes(delta);
    translation_norm < epsilon && rotation_angle < epsilon
}

fn transform_delta_magnitudes(delta: Isometry3<f32>) -> (f64, f64) {
    let translation = delta.translation();
    let translation_norm =
        f64::from(translation.x).hypot(f64::from(translation.y)).hypot(f64::from(translation.z));
    let rotation = delta.rotation();
    let vector_norm =
        f64::from(rotation.x).hypot(f64::from(rotation.y)).hypot(f64::from(rotation.z));
    // atan2 retains small angles when the f32 scalar component rounds to one.
    // abs(w) gives the same shortest angle for equivalent q and -q rotations.
    let rotation_angle = 2.0 * vector_norm.atan2(f64::from(rotation.w).abs());
    (translation_norm, rotation_angle)
}

#[cfg(test)]
mod tests {
    use super::{transform_delta_below_epsilon, IcpConfig, IcpRegistration, IcpStopReason};
    use crate::registration::PointCloudRegistration;
    use crate::transform::transform_point_cloud;
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};
    use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};

    #[test]
    fn convergence_requires_small_translation_and_rotation() {
        let zero = Vec3::new(0.0, 0.0, 0.0);
        let rotation = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 1e-4);
        assert_eq!(rotation.w, 1.0); // acos(w) would incorrectly report zero.
        assert!(!transform_delta_below_epsilon(Isometry3::new(rotation, zero), 1e-6));
        let opposite = Quat::new(-rotation.x, -rotation.y, -rotation.z, -rotation.w);
        assert!(!transform_delta_below_epsilon(Isometry3::new(opposite, zero), 1e-6));
        let tiny = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 1e-8);
        assert!(transform_delta_below_epsilon(
            Isometry3::new(tiny, Vec3::new(1e-8, 0.0, 0.0)),
            1e-6
        ));
        assert!(!transform_delta_below_epsilon(
            Isometry3::new(tiny, Vec3::new(1e-4, 0.0, 0.0)),
            1e-6
        ));
        assert!(!transform_delta_below_epsilon(Isometry3::identity(), 0.0));
    }

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
    fn pure_rotation_does_not_report_convergence_after_one_update() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for point in [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, -3.0],
        ] {
            builder.push_point(point).unwrap();
        }
        let target = builder.build().unwrap();
        let source = transform_point_cloud(
            &target,
            Isometry3::new(
                Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.05),
                Vec3::new(0.0, 0.0, 0.0),
            ),
        )
        .unwrap();
        let config = IcpConfig {
            max_iterations: 1,
            max_correspondence_distance: 0.2,
            transformation_epsilon: 1e-5,
            fitness_epsilon: 0.0,
            ..IcpConfig::default()
        };
        let first = IcpRegistration::new(config).align(&source, &target).unwrap();
        assert!(first.fitness < 1e-10);
        assert!(!first.converged);
        let settled = IcpRegistration::new(IcpConfig { max_iterations: 3, ..config })
            .align(&source, &target)
            .unwrap();
        assert!(settled.converged);
        assert!(settled.iterations > 1);
        let trace = IcpRegistration::new(IcpConfig { max_iterations: 3, ..config })
            .align_with_trace(&source, &target)
            .unwrap();
        assert_eq!(trace.result, settled);
        assert_eq!(trace.stop_reason, IcpStopReason::TransformThreshold);
        assert_eq!(trace.history.len(), settled.iterations);
        assert!(trace.history[0].rotation_delta_radians > 0.04);
        assert!(trace.history[0].translation_delta < config.transformation_epsilon);
        assert_eq!(trace.history[0].fitness_change, None);
        for rows in trace.history.windows(2) {
            assert_eq!(rows[1].fitness_change, Some(rows[0].fitness - rows[1].fitness));
        }
        let capped = IcpRegistration::new(config).align_with_trace(&source, &target).unwrap();
        assert_eq!(capped.stop_reason, IcpStopReason::IterationLimit);
        assert_eq!(capped.result, first);
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
        let trace = registration.align_with_trace(&source, &target).unwrap();
        assert_eq!(trace.result, result);
        assert_eq!(trace.history.last().unwrap().fitness, result.fitness);
        assert_eq!(trace.stop_reason, IcpStopReason::FitnessThreshold);

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
