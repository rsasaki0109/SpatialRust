//! Deterministic camera calibration contracts and small dense solvers.

use spatialrust_math::{solve_linear_system, LeastSquaresResult, Mat3, Vec2, Vec3};

use crate::{CameraIntrinsics, KannalaBrandt4, PinholeCamera};

/// Calibration input or numerical failure.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum CalibrationError {
    /// The dataset is empty, non-finite, inconsistent, or underconstrained.
    #[error("invalid calibration dataset: {0}")]
    InvalidDataset(String),
    /// A normal equation was singular or ill-conditioned.
    #[error("calibration normal equation is singular")]
    Singular,
    /// Projection failed while evaluating residuals.
    #[error("calibration projection failed: {0}")]
    Projection(String),
}

/// Shared robust least-squares controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationOptions {
    /// Maximum robust/refinement iterations.
    pub max_iterations: usize,
    /// Huber transition in residual units; must be finite and positive.
    pub huber_delta: f64,
    /// Parameter-step convergence threshold.
    pub convergence_tolerance: f64,
}

impl Default for CalibrationOptions {
    fn default() -> Self {
        Self { max_iterations: 12, huber_delta: 2.0, convergence_tolerance: 1e-10 }
    }
}

/// Common numerical receipt returned by calibration solvers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CalibrationReport {
    /// Root-mean-square residual in pixels or the solver's documented units.
    pub rms_residual: f64,
    /// Maximum residual magnitude.
    pub max_residual: f64,
    /// Number of observations evaluated.
    pub observation_count: usize,
    /// Iterations executed.
    pub iterations: usize,
    /// Whether the parameter step reached the configured tolerance.
    pub converged: bool,
}

/// Known camera-space point and observed image pixel for mono calibration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PinholeObservation {
    /// Point expressed in the camera frame.
    pub camera_point: Vec3<f64>,
    /// Measured image pixel.
    pub pixel: Vec2<f64>,
}

/// Fits `fx, fy, cx, cy` from known camera-space points with robust reweighting.
pub fn calibrate_pinhole(
    observations: &[PinholeObservation],
    width: usize,
    height: usize,
    options: CalibrationOptions,
) -> Result<(PinholeCamera, CalibrationReport), CalibrationError> {
    validate_options(options)?;
    if observations.len() < 4 {
        return Err(CalibrationError::InvalidDataset(
            "pinhole calibration needs at least four observations".to_owned(),
        ));
    }
    let normalized = observations
        .iter()
        .map(|observation| {
            validate_point_pixel(observation.camera_point, observation.pixel)?;
            Ok((
                observation.camera_point.x / observation.camera_point.z,
                observation.camera_point.y / observation.camera_point.z,
            ))
        })
        .collect::<Result<Vec<_>, CalibrationError>>()?;
    let mut weights = vec![1.0; observations.len()];
    let mut previous = [0.0; 4];
    let mut parameters = [0.0; 4];
    let mut converged = false;
    let mut iterations = 0;
    for iteration in 0..options.max_iterations.max(1) {
        let x_values = normalized.iter().map(|value| value.0).collect::<Vec<_>>();
        let y_values = normalized.iter().map(|value| value.1).collect::<Vec<_>>();
        let u_values = observations.iter().map(|value| value.pixel.x).collect::<Vec<_>>();
        let v_values = observations.iter().map(|value| value.pixel.y).collect::<Vec<_>>();
        let (fx, cx) = fit_slope_intercept(&x_values, &u_values, &weights)?;
        let (fy, cy) = fit_slope_intercept(&y_values, &v_values, &weights)?;
        parameters = [fx, fy, cx, cy];
        if fx <= 0.0 || fy <= 0.0 {
            return Err(CalibrationError::InvalidDataset(
                "calibrated focal lengths are not positive".to_owned(),
            ));
        }
        iterations = iteration + 1;
        let step = parameters
            .iter()
            .zip(previous)
            .map(|(current, old)| (current - old).abs())
            .fold(0.0, f64::max);
        let residuals = observations
            .iter()
            .zip(&normalized)
            .map(|(observation, &(x, y))| {
                (fx.mul_add(x, cx) - observation.pixel.x)
                    .hypot(fy.mul_add(y, cy) - observation.pixel.y)
            })
            .collect::<Vec<_>>();
        update_huber_weights(&residuals, options.huber_delta, &mut weights);
        if iteration > 0 && step <= options.convergence_tolerance {
            converged = true;
            break;
        }
        previous = parameters;
    }
    let intrinsics = CameraIntrinsics::try_new(
        parameters[0],
        parameters[1],
        parameters[2],
        parameters[3],
        width,
        height,
    )
    .map_err(|error| CalibrationError::InvalidDataset(error.to_string()))?;
    let camera = PinholeCamera::new(intrinsics);
    let residuals = observations
        .iter()
        .map(|observation| {
            camera
                .project(observation.camera_point)
                .map(|pixel| pixel_distance(pixel, observation.pixel))
                .map_err(|error| CalibrationError::Projection(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((camera, report(&residuals, iterations, converged)))
}

/// Fisheye angle/radius sample in normalized image coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FisheyeObservation {
    /// Incident ray angle in radians.
    pub theta: f64,
    /// Measured distorted normalized radius.
    pub distorted_radius: f64,
}

/// Fits a Kannala–Brandt four-coefficient angle polynomial.
pub fn calibrate_fisheye(
    observations: &[FisheyeObservation],
) -> Result<(KannalaBrandt4, CalibrationReport), CalibrationError> {
    if observations.len() < 4 {
        return Err(CalibrationError::InvalidDataset(
            "fisheye calibration needs at least four non-zero angles".to_owned(),
        ));
    }
    let mut normal = vec![vec![0.0; 4]; 4];
    let mut rhs = vec![0.0; 4];
    for observation in observations {
        if !observation.theta.is_finite()
            || !observation.distorted_radius.is_finite()
            || observation.theta <= 0.0
        {
            return Err(CalibrationError::InvalidDataset(
                "fisheye samples must have finite positive theta".to_owned(),
            ));
        }
        let theta2 = observation.theta * observation.theta;
        let row = [theta2, theta2.powi(2), theta2.powi(3), theta2.powi(4)];
        let target = observation.distorted_radius / observation.theta - 1.0;
        accumulate_normal(&mut normal, &mut rhs, &row, target, 1.0);
    }
    let values = solved(normal, rhs)?;
    let model = KannalaBrandt4 { k1: values[0], k2: values[1], k3: values[2], k4: values[3] };
    let residuals = observations
        .iter()
        .map(|observation| {
            let theta2 = observation.theta * observation.theta;
            let predicted = observation.theta
                * (1.0
                    + theta2
                        * (model.k1
                            + theta2 * (model.k2 + theta2 * (model.k3 + theta2 * model.k4))));
            (predicted - observation.distorted_radius).abs()
        })
        .collect::<Vec<_>>();
    Ok((model, report(&residuals, 1, true)))
}

/// Rigid transform represented as a rotation matrix and translation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidTransform3 {
    /// Source-to-destination rotation.
    pub rotation: Mat3<f64>,
    /// Source-to-destination translation.
    pub translation: Vec3<f64>,
}

impl RigidTransform3 {
    /// Applies the transform to a point.
    #[must_use]
    pub fn transform_point(self, point: Vec3<f64>) -> Vec3<f64> {
        self.rotation.mul_vec3(point) + self.translation
    }
}

/// Matched 3D point expressed in left and right camera frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoPointPair {
    /// Point in the left camera frame.
    pub left: Vec3<f64>,
    /// Same point in the right camera frame.
    pub right: Vec3<f64>,
}

/// Stereo calibration result with explicit right-from-left transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoCalibration {
    /// Left calibrated camera.
    pub left: PinholeCamera,
    /// Right calibrated camera.
    pub right: PinholeCamera,
    /// Transform mapping left-camera points into the right camera.
    pub right_from_left: RigidTransform3,
    /// 3D alignment residual receipt.
    pub report: CalibrationReport,
}

/// Fits stereo translation for a supplied relative rotation.
pub fn calibrate_stereo_translation(
    left: PinholeCamera,
    right: PinholeCamera,
    rotation: Mat3<f64>,
    pairs: &[StereoPointPair],
) -> Result<StereoCalibration, CalibrationError> {
    if pairs.len() < 3 {
        return Err(CalibrationError::InvalidDataset(
            "stereo calibration needs at least three point pairs".to_owned(),
        ));
    }
    validate_rotation(rotation)?;
    let mut translation = Vec3::new(0.0, 0.0, 0.0);
    for pair in pairs {
        validate_vec3(pair.left)?;
        validate_vec3(pair.right)?;
        translation = translation + pair.right - rotation.mul_vec3(pair.left);
    }
    let inverse_count = 1.0 / pairs.len() as f64;
    translation = Vec3::new(
        translation.x * inverse_count,
        translation.y * inverse_count,
        translation.z * inverse_count,
    );
    let transform = RigidTransform3 { rotation, translation };
    let residuals = pairs
        .iter()
        .map(|pair| (transform.transform_point(pair.left) - pair.right).length())
        .collect::<Vec<_>>();
    Ok(StereoCalibration {
        left,
        right,
        right_from_left: transform,
        report: report(&residuals, 1, true),
    })
}

/// Robot/camera relative motions for `A X = X B` hand-eye calibration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandEyeMotionPair {
    /// Robot/end-effector motion `A`.
    pub robot_motion: RigidTransform3,
    /// Camera motion `B`.
    pub camera_motion: RigidTransform3,
}

/// Solves hand-eye translation for a supplied hand-eye rotation.
pub fn calibrate_hand_eye_translation(
    pairs: &[HandEyeMotionPair],
    hand_eye_rotation: Mat3<f64>,
) -> Result<(RigidTransform3, CalibrationReport), CalibrationError> {
    if pairs.len() < 2 {
        return Err(CalibrationError::InvalidDataset(
            "hand-eye translation needs at least two motion pairs".to_owned(),
        ));
    }
    validate_rotation(hand_eye_rotation)?;
    let mut normal = vec![vec![0.0; 3]; 3];
    let mut rhs = vec![0.0; 3];
    for pair in pairs {
        validate_rotation(pair.robot_motion.rotation)?;
        validate_rotation(pair.camera_motion.rotation)?;
        let target = hand_eye_rotation.mul_vec3(pair.camera_motion.translation)
            - pair.robot_motion.translation;
        for row in 0..3 {
            let coefficients = [
                pair.robot_motion.rotation.m[row][0] - if row == 0 { 1.0 } else { 0.0 },
                pair.robot_motion.rotation.m[row][1] - if row == 1 { 1.0 } else { 0.0 },
                pair.robot_motion.rotation.m[row][2] - if row == 2 { 1.0 } else { 0.0 },
            ];
            accumulate_normal(
                &mut normal,
                &mut rhs,
                &coefficients,
                [target.x, target.y, target.z][row],
                1.0,
            );
        }
    }
    let values = solved(normal, rhs)?;
    let result = RigidTransform3 {
        rotation: hand_eye_rotation,
        translation: Vec3::new(values[0], values[1], values[2]),
    };
    let residuals = pairs
        .iter()
        .map(|pair| {
            let left = pair.robot_motion.rotation.mul_vec3(result.translation)
                + pair.robot_motion.translation;
            let right =
                result.rotation.mul_vec3(pair.camera_motion.translation) + result.translation;
            let rotation_error = matrix_distance(
                pair.robot_motion.rotation.mul_mat3(result.rotation),
                result.rotation.mul_mat3(pair.camera_motion.rotation),
            );
            (left - right).length().hypot(rotation_error)
        })
        .collect::<Vec<_>>();
    Ok((result, report(&residuals, 1, true)))
}

/// Calibrated camera and world-to-camera pose used by bundle adjustment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BundleView {
    /// Camera intrinsics/distortion.
    pub camera: PinholeCamera,
    /// World-to-camera transform.
    pub camera_from_world: RigidTransform3,
}

/// One 2D observation of a bundle point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BundleObservation {
    /// Index into [`BundleProblem::views`].
    pub view_index: usize,
    /// Index into [`BundleProblem::points`].
    pub point_index: usize,
    /// Measured image pixel.
    pub pixel: Vec2<f64>,
}

/// Sparse fixed-camera bundle problem.
#[derive(Clone, Debug, PartialEq)]
pub struct BundleProblem {
    /// Fixed calibrated camera views.
    pub views: Vec<BundleView>,
    /// Mutable world-space points.
    pub points: Vec<Vec3<f64>>,
    /// Sparse image observations.
    pub observations: Vec<BundleObservation>,
}

/// Refines world-space points while keeping calibrated camera poses fixed.
pub fn bundle_adjust_points(
    problem: &mut BundleProblem,
    options: CalibrationOptions,
) -> Result<CalibrationReport, CalibrationError> {
    validate_options(options)?;
    validate_bundle(problem)?;
    let mut converged = false;
    let mut iterations = 0;
    for iteration in 0..options.max_iterations.max(1) {
        let mut max_step: f64 = 0.0;
        for point_index in 0..problem.points.len() {
            let mut normal = vec![vec![0.0; 3]; 3];
            let mut rhs = vec![0.0; 3];
            let mut count = 0;
            for observation in
                problem.observations.iter().filter(|value| value.point_index == point_index)
            {
                let view = problem.views[observation.view_index];
                let point = problem.points[point_index];
                let predicted = project_world(view, point)?;
                let residual =
                    [predicted.x - observation.pixel.x, predicted.y - observation.pixel.y];
                let magnitude = residual[0].hypot(residual[1]);
                let weight = huber_weight(magnitude, options.huber_delta);
                const STEP: f64 = 1e-6;
                let shifted_x = project_world(view, point + Vec3::new(STEP, 0.0, 0.0))?;
                let shifted_y = project_world(view, point + Vec3::new(0.0, STEP, 0.0))?;
                let shifted_z = project_world(view, point + Vec3::new(0.0, 0.0, STEP))?;
                let jacobian = [
                    [
                        (shifted_x.x - predicted.x) / STEP,
                        (shifted_y.x - predicted.x) / STEP,
                        (shifted_z.x - predicted.x) / STEP,
                    ],
                    [
                        (shifted_x.y - predicted.y) / STEP,
                        (shifted_y.y - predicted.y) / STEP,
                        (shifted_z.y - predicted.y) / STEP,
                    ],
                ];
                for row in 0..2 {
                    accumulate_normal(
                        &mut normal,
                        &mut rhs,
                        &jacobian[row],
                        -residual[row],
                        weight,
                    );
                }
                count += 1;
            }
            if count >= 2 {
                let step = solved(normal, rhs)?;
                let delta = Vec3::new(step[0], step[1], step[2]);
                problem.points[point_index] = problem.points[point_index] + delta;
                max_step = max_step.max(delta.length());
            }
        }
        iterations = iteration + 1;
        if max_step <= options.convergence_tolerance {
            converged = true;
            break;
        }
    }
    let residuals = bundle_residuals(problem)?;
    Ok(report(&residuals, iterations, converged))
}

fn validate_bundle(problem: &BundleProblem) -> Result<(), CalibrationError> {
    if problem.views.is_empty() || problem.points.is_empty() || problem.observations.is_empty() {
        return Err(CalibrationError::InvalidDataset(
            "bundle problem must be non-empty".to_owned(),
        ));
    }
    for view in &problem.views {
        validate_rotation(view.camera_from_world.rotation)?;
        validate_vec3(view.camera_from_world.translation)?;
    }
    for observation in &problem.observations {
        if observation.view_index >= problem.views.len()
            || observation.point_index >= problem.points.len()
        {
            return Err(CalibrationError::InvalidDataset(
                "bundle observation index out of bounds".to_owned(),
            ));
        }
        if !observation.pixel.x.is_finite() || !observation.pixel.y.is_finite() {
            return Err(CalibrationError::InvalidDataset("bundle pixel must be finite".to_owned()));
        }
    }
    Ok(())
}

fn bundle_residuals(problem: &BundleProblem) -> Result<Vec<f64>, CalibrationError> {
    problem
        .observations
        .iter()
        .map(|observation| {
            project_world(
                problem.views[observation.view_index],
                problem.points[observation.point_index],
            )
            .map(|pixel| pixel_distance(pixel, observation.pixel))
        })
        .collect()
}

fn project_world(view: BundleView, point: Vec3<f64>) -> Result<Vec2<f64>, CalibrationError> {
    view.camera
        .project(view.camera_from_world.transform_point(point))
        .map_err(|error| CalibrationError::Projection(error.to_string()))
}

fn validate_options(options: CalibrationOptions) -> Result<(), CalibrationError> {
    if options.max_iterations == 0
        || !options.huber_delta.is_finite()
        || options.huber_delta <= 0.0
        || !options.convergence_tolerance.is_finite()
        || options.convergence_tolerance <= 0.0
    {
        return Err(CalibrationError::InvalidDataset("invalid solver options".to_owned()));
    }
    Ok(())
}

fn validate_point_pixel(point: Vec3<f64>, pixel: Vec2<f64>) -> Result<(), CalibrationError> {
    validate_vec3(point)?;
    if point.z <= 0.0 || !pixel.x.is_finite() || !pixel.y.is_finite() {
        return Err(CalibrationError::InvalidDataset(
            "camera points need positive depth and finite pixels".to_owned(),
        ));
    }
    Ok(())
}

fn validate_vec3(value: Vec3<f64>) -> Result<(), CalibrationError> {
    if !value.x.is_finite() || !value.y.is_finite() || !value.z.is_finite() {
        return Err(CalibrationError::InvalidDataset("3D values must be finite".to_owned()));
    }
    Ok(())
}

fn validate_rotation(rotation: Mat3<f64>) -> Result<(), CalibrationError> {
    if rotation.m.iter().flatten().any(|value| !value.is_finite()) {
        return Err(CalibrationError::InvalidDataset(
            "rotation coefficients must be finite".to_owned(),
        ));
    }
    let identity_error =
        matrix_distance(rotation.transpose().mul_mat3(rotation), Mat3::<f64>::identity());
    let determinant = rotation.m[0][0]
        * (rotation.m[1][1] * rotation.m[2][2] - rotation.m[1][2] * rotation.m[2][1])
        - rotation.m[0][1]
            * (rotation.m[1][0] * rotation.m[2][2] - rotation.m[1][2] * rotation.m[2][0])
        + rotation.m[0][2]
            * (rotation.m[1][0] * rotation.m[2][1] - rotation.m[1][1] * rotation.m[2][0]);
    if identity_error > 1e-6 || (determinant - 1.0).abs() > 1e-6 {
        return Err(CalibrationError::InvalidDataset(
            "rotation matrix must be right-handed and orthonormal".to_owned(),
        ));
    }
    Ok(())
}

fn matrix_distance(left: Mat3<f64>, right: Mat3<f64>) -> f64 {
    left.m
        .iter()
        .flatten()
        .zip(right.m.iter().flatten())
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn fit_slope_intercept(
    x: &[f64],
    y: &[f64],
    weights: &[f64],
) -> Result<(f64, f64), CalibrationError> {
    let mut normal = vec![vec![0.0; 2]; 2];
    let mut rhs = vec![0.0; 2];
    for ((&x, &y), &weight) in x.iter().zip(y).zip(weights) {
        accumulate_normal(&mut normal, &mut rhs, &[x, 1.0], y, weight);
    }
    let values = solved(normal, rhs)?;
    Ok((values[0], values[1]))
}

fn accumulate_normal(
    normal: &mut [Vec<f64>],
    rhs: &mut [f64],
    row: &[f64],
    target: f64,
    weight: f64,
) {
    for i in 0..row.len() {
        rhs[i] += weight * row[i] * target;
        for j in 0..row.len() {
            normal[i][j] += weight * row[i] * row[j];
        }
    }
}

fn solved(normal: Vec<Vec<f64>>, rhs: Vec<f64>) -> Result<Vec<f64>, CalibrationError> {
    match solve_linear_system(normal, rhs) {
        LeastSquaresResult::Solved(values) => Ok(values),
        LeastSquaresResult::Singular => Err(CalibrationError::Singular),
    }
}

fn update_huber_weights(residuals: &[f64], delta: f64, weights: &mut [f64]) {
    for (weight, &residual) in weights.iter_mut().zip(residuals) {
        *weight = huber_weight(residual, delta);
    }
}

fn huber_weight(residual: f64, delta: f64) -> f64 {
    if residual <= delta || residual <= f64::EPSILON {
        1.0
    } else {
        delta / residual
    }
}

fn pixel_distance(left: Vec2<f64>, right: Vec2<f64>) -> f64 {
    (left.x - right.x).hypot(left.y - right.y)
}

fn report(residuals: &[f64], iterations: usize, converged: bool) -> CalibrationReport {
    let sum_squared = residuals.iter().map(|value| value * value).sum::<f64>();
    CalibrationReport {
        rms_residual: (sum_squared / residuals.len().max(1) as f64).sqrt(),
        max_residual: residuals.iter().copied().fold(0.0, f64::max),
        observation_count: residuals.len(),
        iterations,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> PinholeCamera {
        PinholeCamera::new(CameraIntrinsics::try_new(500.0, 510.0, 320.0, 240.0, 640, 480).unwrap())
    }

    fn rotation_z(angle: f64) -> Mat3<f64> {
        let (sin, cos) = angle.sin_cos();
        Mat3::from_rows([cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0])
    }

    fn rotation_x(angle: f64) -> Mat3<f64> {
        let (sin, cos) = angle.sin_cos();
        Mat3::from_rows([1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos])
    }

    fn rotation_y(angle: f64) -> Mat3<f64> {
        let (sin, cos) = angle.sin_cos();
        Mat3::from_rows([cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos])
    }

    #[test]
    fn robust_mono_recovers_intrinsics_with_one_outlier() {
        let expected = camera();
        let mut observations = Vec::new();
        for y in -3_i32..=3 {
            for x in -4_i32..=4 {
                let point =
                    Vec3::new(x as f64 * 0.08, y as f64 * 0.07, 1.5 + (x + y).abs() as f64 * 0.03);
                observations.push(PinholeObservation {
                    camera_point: point,
                    pixel: expected.project(point).unwrap(),
                });
            }
        }
        observations[0].pixel.x += 80.0;
        let (calibrated, report) = calibrate_pinhole(
            &observations,
            640,
            480,
            CalibrationOptions { huber_delta: 1.0, ..CalibrationOptions::default() },
        )
        .unwrap();
        assert!((calibrated.intrinsics.fx - 500.0).abs() < 0.2);
        assert!((calibrated.intrinsics.fy - 510.0).abs() < 1e-8);
        assert!((calibrated.intrinsics.cx - 320.0).abs() < 0.1);
        assert_eq!(report.observation_count, observations.len());
    }

    #[test]
    fn fisheye_fit_recovers_angle_polynomial() {
        let expected = KannalaBrandt4 { k1: 0.03, k2: -0.004, k3: 0.0005, k4: -0.00003 };
        let observations = (1..=12)
            .map(|index| {
                let theta = index as f64 * 0.08;
                let theta2 = theta * theta;
                FisheyeObservation {
                    theta,
                    distorted_radius: theta
                        * (1.0
                            + theta2
                                * (expected.k1
                                    + theta2
                                        * (expected.k2
                                            + theta2 * (expected.k3 + theta2 * expected.k4)))),
                }
            })
            .collect::<Vec<_>>();
        let (actual, report) = calibrate_fisheye(&observations).unwrap();
        assert!((actual.k1 - expected.k1).abs() < 1e-9);
        assert!((actual.k4 - expected.k4).abs() < 1e-9);
        assert!(report.rms_residual < 1e-12);
    }

    #[test]
    fn stereo_translation_matches_known_transform() {
        let camera = camera();
        let rotation = rotation_z(0.03);
        let translation = Vec3::new(-0.2, 0.01, 0.005);
        let pairs = (0..8)
            .map(|index| {
                let left = Vec3::new(index as f64 * 0.1 - 0.3, 0.05 * index as f64, 2.0);
                StereoPointPair { left, right: rotation.mul_vec3(left) + translation }
            })
            .collect::<Vec<_>>();
        let result = calibrate_stereo_translation(camera, camera, rotation, &pairs).unwrap();
        assert!((result.right_from_left.translation - translation).length() < 1e-12);
        assert!(result.report.rms_residual < 1e-12);
    }

    #[test]
    fn hand_eye_translation_satisfies_ax_xb() {
        let expected = Vec3::new(0.12, -0.04, 0.3);
        let pairs = [rotation_x(0.4), rotation_y(-0.7), rotation_z(1.1)]
            .into_iter()
            .enumerate()
            .map(|(index, rotation)| {
                let robot_translation = Vec3::new(0.03 * index as f64, 0.02, -0.01);
                HandEyeMotionPair {
                    robot_motion: RigidTransform3 { rotation, translation: robot_translation },
                    camera_motion: RigidTransform3 {
                        rotation,
                        translation: robot_translation + (rotation.mul_vec3(expected) - expected),
                    },
                }
            })
            .collect::<Vec<_>>();
        let (result, report) =
            calibrate_hand_eye_translation(&pairs, Mat3::<f64>::identity()).unwrap();
        assert!((result.translation - expected).length() < 1e-10);
        assert!(report.rms_residual < 1e-10);
    }

    #[test]
    fn fixed_camera_bundle_reduces_reprojection_error() {
        let camera = camera();
        let views = vec![
            BundleView {
                camera,
                camera_from_world: RigidTransform3 {
                    rotation: Mat3::<f64>::identity(),
                    translation: Vec3::new(0.0, 0.0, 0.0),
                },
            },
            BundleView {
                camera,
                camera_from_world: RigidTransform3 {
                    rotation: Mat3::<f64>::identity(),
                    translation: Vec3::new(-0.4, 0.0, 0.0),
                },
            },
        ];
        let truth = [Vec3::new(0.1, -0.1, 2.5), Vec3::new(-0.2, 0.15, 3.0)];
        let mut observations = Vec::new();
        for (point_index, &point) in truth.iter().enumerate() {
            for (view_index, &view) in views.iter().enumerate() {
                observations.push(BundleObservation {
                    view_index,
                    point_index,
                    pixel: project_world(view, point).unwrap(),
                });
            }
        }
        let mut problem = BundleProblem {
            views,
            points: truth.iter().map(|point| *point + Vec3::new(0.08, -0.04, 0.2)).collect(),
            observations,
        };
        let before = report(&bundle_residuals(&problem).unwrap(), 0, false).rms_residual;
        let after = bundle_adjust_points(&mut problem, CalibrationOptions::default()).unwrap();
        assert!(after.rms_residual < before * 1e-4);
        assert!(after.rms_residual < 1e-6);
    }

    #[test]
    fn calibration_contracts_reject_invalid_rotations_and_indices() {
        let camera = camera();
        let reflection = Mat3::from_rows([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]);
        let pairs =
            [StereoPointPair { left: Vec3::new(0.0, 0.0, 1.0), right: Vec3::new(0.0, 0.0, 1.0) };
                3];
        assert!(calibrate_stereo_translation(camera, camera, reflection, &pairs).is_err());

        let mut problem = BundleProblem {
            views: vec![BundleView {
                camera,
                camera_from_world: RigidTransform3 {
                    rotation: Mat3::<f64>::identity(),
                    translation: Vec3::new(0.0, 0.0, 0.0),
                },
            }],
            points: vec![Vec3::new(0.0, 0.0, 2.0)],
            observations: vec![BundleObservation {
                view_index: 1,
                point_index: 0,
                pixel: Vec2 { x: 0.0, y: 0.0 },
            }],
        };
        assert!(bundle_adjust_points(&mut problem, CalibrationOptions::default()).is_err());
    }
}
