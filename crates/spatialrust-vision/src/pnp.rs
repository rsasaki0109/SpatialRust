//! Absolute pose from 3D–2D correspondences (PnP) with deterministic RANSAC.

use spatialrust_math::{
    solve_linear_system, symmetric_eigen3, LeastSquaresResult, Mat3, Vec2, Vec3,
};

use crate::{
    AbsolutePose, CameraMatrix3, GeometricEstimate, ObjectImageCorrespondence,
    RobustEstimationOptions, VisionError, VisionResult,
};

/// Estimates an object-to-camera pose from at least four correspondences.
///
/// Uses a calibrated DLT initialization followed by Gauss–Newton refinement on
/// the SE(3) tangent space. Four points are accepted for the final refine path
/// when an initial pose is recoverable; RANSAC minimal samples use six points.
pub fn solve_pnp(
    correspondences: &[ObjectImageCorrespondence],
    camera: CameraMatrix3,
) -> VisionResult<AbsolutePose> {
    if correspondences.len() < 4 {
        return Err(VisionError::InvalidParameter(
            "PnP requires at least four object-image correspondences".into(),
        ));
    }
    let initial = estimate_pnp_dlt(correspondences, camera)?;
    refine_pnp(correspondences, camera, initial)
}

/// Robust PnP using deterministic six-point RANSAC and inlier refinement.
pub fn solve_pnp_ransac(
    correspondences: &[ObjectImageCorrespondence],
    camera: CameraMatrix3,
    options: RobustEstimationOptions,
) -> VisionResult<GeometricEstimate<AbsolutePose>> {
    let options = options.validate()?;
    const SAMPLE: usize = 6;
    if correspondences.len() < SAMPLE {
        return Err(VisionError::InvalidParameter(
            "robust PnP requires at least six correspondences".into(),
        ));
    }
    let mut rng = XorShift64::new(options.seed);
    let mut best: Option<(AbsolutePose, Vec<bool>, Vec<f64>, usize, f64)> = None;
    let mut iteration_limit = options.max_iterations;
    let mut iteration = 0;
    while iteration < iteration_limit {
        let indices = sample_unique(&mut rng, correspondences.len(), SAMPLE);
        let sample = indices.iter().map(|&index| correspondences[index]).collect::<Vec<_>>();
        if let Ok(model) = estimate_pnp_dlt(&sample, camera).and_then(|pose| {
            refine_pnp(&sample, camera, pose)
        }) {
            let residuals = correspondences
                .iter()
                .copied()
                .map(|pair| pnp_residual(model, pair, camera))
                .collect::<Vec<_>>();
            let inliers =
                residuals.iter().map(|&value| value <= options.threshold).collect::<Vec<_>>();
            let count = inliers.iter().filter(|&&value| value).count();
            let error = residuals
                .iter()
                .zip(&inliers)
                .filter_map(|(value, &is_inlier)| is_inlier.then_some(*value))
                .sum::<f64>();
            let improves = best.as_ref().map_or(true, |candidate| {
                count > candidate.3 || (count == candidate.3 && error < candidate.4)
            });
            if improves {
                if count >= SAMPLE {
                    let inlier_ratio = count as f64 / correspondences.len() as f64;
                    let success = inlier_ratio.powi(SAMPLE as i32).clamp(0.0, 1.0);
                    if success > 0.0 && success < 1.0 {
                        let required = ((1.0 - options.confidence).ln() / (1.0 - success).ln())
                            .ceil()
                            .max(1.0) as usize;
                        iteration_limit = iteration_limit.min(required.max(iteration + 1));
                    } else if success == 1.0 {
                        iteration_limit = iteration + 1;
                    }
                }
                best = Some((model, inliers, residuals, count, error));
            }
        }
        iteration += 1;
    }
    let (_, best_inliers, _, count, _) = best.ok_or_else(|| {
        VisionError::InvalidParameter("robust PnP found no valid model".into())
    })?;
    if count < 4 {
        return Err(VisionError::InvalidParameter(
            "robust PnP found too few inliers".into(),
        ));
    }
    let inlier_pairs = correspondences
        .iter()
        .zip(&best_inliers)
        .filter_map(|(&pair, &is_inlier)| is_inlier.then_some(pair))
        .collect::<Vec<_>>();
    let refined = solve_pnp(&inlier_pairs, camera)?;
    let residuals = correspondences
        .iter()
        .copied()
        .map(|pair| pnp_residual(refined, pair, camera))
        .collect::<Vec<_>>();
    let inliers = residuals.iter().map(|&value| value <= options.threshold).collect();
    GeometricEstimate::try_new(refined, correspondences.len(), inliers, residuals)
}

/// Projects an object point with an absolute pose into pixel coordinates.
pub fn project_object_point(
    pose: AbsolutePose,
    camera: CameraMatrix3,
    object: Vec3<f64>,
) -> VisionResult<Vec2<f64>> {
    let camera_point = pose.transform_point(object);
    if camera_point.z <= 1e-12 {
        return Err(VisionError::InvalidParameter(
            "projected point lies behind or on the camera plane".into(),
        ));
    }
    let normalized = Vec3::new(camera_point.x / camera_point.z, camera_point.y / camera_point.z, 1.0);
    let pixel = camera.matrix().mul_vec3(normalized);
    Ok(Vec2 { x: pixel.x / pixel.z, y: pixel.y / pixel.z })
}

fn estimate_pnp_dlt(
    correspondences: &[ObjectImageCorrespondence],
    camera: CameraMatrix3,
) -> VisionResult<AbsolutePose> {
    if correspondences.len() < 4 {
        return Err(VisionError::InvalidParameter(
            "PnP DLT requires at least four correspondences".into(),
        ));
    }
    let mut normal = vec![vec![0.0; 12]; 12];
    for pair in correspondences {
        let object = pair.object();
        let image = pair.image();
        let rows = [
            [
                object.x,
                object.y,
                object.z,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -image.x * object.x,
                -image.x * object.y,
                -image.x * object.z,
                -image.x,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                object.x,
                object.y,
                object.z,
                1.0,
                -image.y * object.x,
                -image.y * object.y,
                -image.y * object.z,
                -image.y,
            ],
        ];
        for row in rows {
            for first in 0..12 {
                for second in 0..12 {
                    normal[first][second] += row[first] * row[second];
                }
            }
        }
    }
    let vector = smallest_symmetric_eigenvector(normal).ok_or_else(|| {
        VisionError::InvalidParameter("PnP correspondences are degenerate".into())
    })?;
    let projection = Mat3::from_rows(
        [vector[0], vector[1], vector[2]],
        [vector[4], vector[5], vector[6]],
        [vector[8], vector[9], vector[10]],
    );
    let translation_part = Vec3::new(vector[3], vector[7], vector[11]);
    let calibrated = camera.inverse().mul_mat3(projection);
    let calibrated_t = camera.inverse().mul_vec3(translation_part);
    let (rotation, scale) = orthonormalize_rotation(calibrated)?;
    let translation = Vec3::new(
        calibrated_t.x / scale,
        calibrated_t.y / scale,
        calibrated_t.z / scale,
    );
    // Flip if most points have negative depth.
    let pose = AbsolutePose::try_new(rotation, translation)?;
    let positive = correspondences
        .iter()
        .filter(|pair| pose.transform_point(pair.object()).z > 0.0)
        .count();
    if positive * 2 < correspondences.len() {
        AbsolutePose::try_new(
            Mat3::from_rows(
                [-rotation.m[0][0], -rotation.m[0][1], -rotation.m[0][2]],
                [-rotation.m[1][0], -rotation.m[1][1], -rotation.m[1][2]],
                [-rotation.m[2][0], -rotation.m[2][1], -rotation.m[2][2]],
            ),
            Vec3::new(-translation.x, -translation.y, -translation.z),
        )
    } else {
        Ok(pose)
    }
}

fn refine_pnp(
    correspondences: &[ObjectImageCorrespondence],
    camera: CameraMatrix3,
    mut pose: AbsolutePose,
) -> VisionResult<AbsolutePose> {
    for _ in 0..20 {
        let mut normal = vec![vec![0.0; 6]; 6];
        let mut rhs = vec![0.0; 6];
        let mut residual_sum = 0.0;
        for pair in correspondences {
            let camera_point = pose.transform_point(pair.object());
            if camera_point.z <= 1e-12 {
                continue;
            }
            let predicted = project_object_point(pose, camera, pair.object())?;
            let error = [predicted.x - pair.image().x, predicted.y - pair.image().y];
            residual_sum += error[0] * error[0] + error[1] * error[1];
            let jacobian = projection_jacobian(pose, camera, pair.object(), camera_point)?;
            for row in 0..2 {
                for col in 0..6 {
                    rhs[col] -= jacobian[row][col] * error[row];
                    for other in 0..6 {
                        normal[col][other] += jacobian[row][col] * jacobian[row][other];
                    }
                }
            }
        }
        let LeastSquaresResult::Solved(delta) = solve_linear_system(normal, rhs) else {
            break;
        };
        let update = Vec3::new(delta[0], delta[1], delta[2]);
        if update.length() + Vec3::new(delta[3], delta[4], delta[5]).length() < 1e-10 {
            break;
        }
        let rotated = exp_so3(update).mul_mat3(pose.rotation());
        let (rotation, _) = orthonormalize_rotation(rotated)?;
        let translation = pose.translation() + Vec3::new(delta[3], delta[4], delta[5]);
        pose = AbsolutePose::try_new(rotation, translation)?;
        if residual_sum < 1e-18 {
            break;
        }
    }
    Ok(pose)
}

fn projection_jacobian(
    pose: AbsolutePose,
    camera: CameraMatrix3,
    object: Vec3<f64>,
    camera_point: Vec3<f64>,
) -> VisionResult<[[f64; 6]; 2]> {
    let z = camera_point.z;
    let z2 = z * z;
    let fx = camera.matrix().m[0][0];
    let fy = camera.matrix().m[1][1];
    // d(u,v)/dX_c
    let du_dx = fx / z;
    let du_dy = 0.0;
    let du_dz = -fx * camera_point.x / z2;
    let dv_dx = 0.0;
    let dv_dy = fy / z;
    let dv_dz = -fy * camera_point.y / z2;
    // dX_c / d(omega,t): omega acts as [omega]_x R X, t is additive.
    let rotated = pose.rotation().mul_vec3(object);
    let dx_domega = [
        Vec3::new(0.0, -rotated.z, rotated.y),
        Vec3::new(rotated.z, 0.0, -rotated.x),
        Vec3::new(-rotated.y, rotated.x, 0.0),
    ];
    let mut jacobian = [[0.0; 6]; 2];
    for axis in 0..3 {
        let d = dx_domega[axis];
        jacobian[0][axis] = du_dx * d.x + du_dy * d.y + du_dz * d.z;
        jacobian[1][axis] = dv_dx * d.x + dv_dy * d.y + dv_dz * d.z;
    }
    jacobian[0][3] = du_dx;
    jacobian[0][4] = du_dy;
    jacobian[0][5] = du_dz;
    jacobian[1][3] = dv_dx;
    jacobian[1][4] = dv_dy;
    jacobian[1][5] = dv_dz;
    Ok(jacobian)
}

fn pnp_residual(
    pose: AbsolutePose,
    pair: ObjectImageCorrespondence,
    camera: CameraMatrix3,
) -> f64 {
    match project_object_point(pose, camera, pair.object()) {
        Ok(pixel) => (pixel.x - pair.image().x).hypot(pixel.y - pair.image().y),
        Err(_) => f64::MAX,
    }
}

fn orthonormalize_rotation(matrix: Mat3<f64>) -> VisionResult<(Mat3<f64>, f64)> {
    let eigen = symmetric_eigen3(matrix.transpose().mul_mat3(matrix));
    let scale = ((eigen.eigenvalues[0].max(0.0).sqrt()
        + eigen.eigenvalues[1].max(0.0).sqrt()
        + eigen.eigenvalues[2].max(0.0).sqrt())
        / 3.0)
        .max(1e-12);
    let right = eigen.eigenvectors;
    let mut left_cols = [Vec3::new(0.0, 0.0, 0.0); 3];
    for (column, left_col) in left_cols.iter_mut().enumerate() {
        let right_col = Vec3::new(right.m[0][column], right.m[1][column], right.m[2][column]);
        let sigma = eigen.eigenvalues[column].max(0.0).sqrt().max(1e-12);
        *left_col = scale_vec(matrix.mul_vec3(right_col), 1.0 / (sigma * scale));
    }
    left_cols[0] = left_cols[0].normalize();
    left_cols[1] =
        (left_cols[1] - scale_vec(left_cols[0], left_cols[0].dot(left_cols[1]))).normalize();
    left_cols[2] = left_cols[0].cross(left_cols[1]).normalize();
    let right0 = Vec3::new(right.m[0][0], right.m[1][0], right.m[2][0]).normalize();
    let mut right1 = Vec3::new(right.m[0][1], right.m[1][1], right.m[2][1]);
    right1 = (right1 - scale_vec(right0, right0.dot(right1))).normalize();
    let right2 = right0.cross(right1).normalize();
    let mut rotation = Mat3::from_rows(
        [
            left_cols[0].x * right0.x + left_cols[1].x * right1.x + left_cols[2].x * right2.x,
            left_cols[0].x * right0.y + left_cols[1].x * right1.y + left_cols[2].x * right2.y,
            left_cols[0].x * right0.z + left_cols[1].x * right1.z + left_cols[2].x * right2.z,
        ],
        [
            left_cols[0].y * right0.x + left_cols[1].y * right1.x + left_cols[2].y * right2.x,
            left_cols[0].y * right0.y + left_cols[1].y * right1.y + left_cols[2].y * right2.y,
            left_cols[0].y * right0.z + left_cols[1].y * right1.z + left_cols[2].y * right2.z,
        ],
        [
            left_cols[0].z * right0.x + left_cols[1].z * right1.x + left_cols[2].z * right2.x,
            left_cols[0].z * right0.y + left_cols[1].z * right1.y + left_cols[2].z * right2.y,
            left_cols[0].z * right0.z + left_cols[1].z * right1.z + left_cols[2].z * right2.z,
        ],
    );
    if determinant(rotation) < 0.0 {
        for row in &mut rotation.m {
            for value in row {
                *value = -*value;
            }
        }
    }
    Ok((rotation, if determinant(matrix) < 0.0 { -scale } else { scale }))
}

fn exp_so3(omega: Vec3<f64>) -> Mat3<f64> {
    let theta = omega.length();
    if theta < 1e-12 {
        return Mat3::from_rows(
            [1.0, -omega.z, omega.y],
            [omega.z, 1.0, -omega.x],
            [-omega.y, omega.x, 1.0],
        );
    }
    let axis = omega.normalize();
    let skew = Mat3::from_rows(
        [0.0, -axis.z, axis.y],
        [axis.z, 0.0, -axis.x],
        [-axis.y, axis.x, 0.0],
    );
    let skew2 = skew.mul_mat3(skew);
    let mut result = Mat3::<f64>::identity();
    let s = theta.sin();
    let c = 1.0 - theta.cos();
    for row in 0..3 {
        for column in 0..3 {
            result.m[row][column] += s * skew.m[row][column] + c * skew2.m[row][column];
        }
    }
    result
}

#[allow(clippy::needless_range_loop)]
fn smallest_symmetric_eigenvector(mut matrix: Vec<Vec<f64>>) -> Option<[f64; 12]> {
    let size = matrix.len();
    if size != 12 || matrix.iter().any(|row| row.len() != size) {
        return None;
    }
    let mut vectors = vec![vec![0.0; size]; size];
    for (index, row) in vectors.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    for _ in 0..size * size * 32 {
        let mut pivot = (0, 1);
        let mut maximum = 0.0_f64;
        for (row, values) in matrix.iter().enumerate() {
            for (column, &value) in values.iter().enumerate().skip(row + 1) {
                if value.abs() > maximum {
                    maximum = value.abs();
                    pivot = (row, column);
                }
            }
        }
        if maximum < 1e-12 {
            break;
        }
        let (p, q) = pivot;
        let angle = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[q][q] - matrix[p][p]);
        let (sine, cosine) = angle.sin_cos();
        for row in 0..size {
            if row != p && row != q {
                let rp = matrix[row][p];
                let rq = matrix[row][q];
                matrix[row][p] = cosine * rp - sine * rq;
                matrix[p][row] = matrix[row][p];
                matrix[row][q] = sine * rp + cosine * rq;
                matrix[q][row] = matrix[row][q];
            }
        }
        let pp = matrix[p][p];
        let qq = matrix[q][q];
        let pq = matrix[p][q];
        matrix[p][p] = cosine * cosine * pp - 2.0 * sine * cosine * pq + sine * sine * qq;
        matrix[q][q] = sine * sine * pp + 2.0 * sine * cosine * pq + cosine * cosine * qq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        for row in &mut vectors {
            let rp = row[p];
            let rq = row[q];
            row[p] = cosine * rp - sine * rq;
            row[q] = sine * rp + cosine * rq;
        }
    }
    let index =
        (0..size).min_by(|&left, &right| matrix[left][left].total_cmp(&matrix[right][right]))?;
    let mut vector = vectors.iter().map(|row| row[index]).collect::<Vec<_>>();
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return None;
    }
    for value in &mut vector {
        *value /= norm;
    }
    let mut out = [0.0; 12];
    out.copy_from_slice(&vector);
    Some(out)
}

fn sample_unique(rng: &mut XorShift64, upper: usize, count: usize) -> Vec<usize> {
    let mut selected = Vec::with_capacity(count);
    while selected.len() < count {
        let value = rng.next_usize(upper);
        if !selected.contains(&value) {
            selected.push(value);
        }
    }
    selected
}

fn scale_vec(vector: Vec3<f64>, scale: f64) -> Vec3<f64> {
    Vec3::new(vector.x * scale, vector.y * scale, vector.z * scale)
}

fn determinant(matrix: Mat3<f64>) -> f64 {
    matrix.m[0][0] * (matrix.m[1][1] * matrix.m[2][2] - matrix.m[1][2] * matrix.m[2][1])
        - matrix.m[0][1] * (matrix.m[1][0] * matrix.m[2][2] - matrix.m[1][2] * matrix.m[2][0])
        + matrix.m[0][2] * (matrix.m[1][0] * matrix.m[2][1] - matrix.m[1][1] * matrix.m[2][0])
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::{project_object_point, solve_pnp, solve_pnp_ransac};
    use crate::{AbsolutePose, CameraMatrix3, ObjectImageCorrespondence, RobustEstimationOptions};
    use spatialrust_camera::CameraIntrinsics;
    use spatialrust_math::{Mat3, Vec2, Vec3};

    fn camera() -> CameraMatrix3 {
        let intrinsics = CameraIntrinsics::try_new(500.0, 500.0, 320.0, 240.0, 640, 480).unwrap();
        CameraMatrix3::from_intrinsics(intrinsics)
    }

    fn sample_pose() -> AbsolutePose {
        AbsolutePose::try_new(
            Mat3::from_rows(
                [0.936_293_4, -0.275_095_9, 0.218_350_8],
                [0.289_629_5, 0.956_425_1, -0.036_957_0],
                [-0.198_669_3, 0.097_843_4, 0.975_170_3],
            ),
            Vec3::new(0.15, -0.05, 2.5),
        )
        .unwrap()
    }

    #[test]
    fn solve_pnp_recovers_known_pose() {
        let camera = camera();
        let pose = sample_pose();
        let objects = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.4, 0.0, 0.0),
            Vec3::new(0.0, 0.3, 0.0),
            Vec3::new(0.0, 0.0, 0.2),
            Vec3::new(0.25, 0.2, 0.1),
            Vec3::new(-0.1, 0.15, -0.05),
            Vec3::new(0.1, -0.2, 0.05),
            Vec3::new(-0.2, -0.1, 0.15),
        ];
        let pairs = objects
            .into_iter()
            .map(|object| {
                let image = project_object_point(pose, camera, object).unwrap();
                ObjectImageCorrespondence::try_new(object, image).unwrap()
            })
            .collect::<Vec<_>>();
        let estimated = solve_pnp(&pairs, camera).unwrap();
        for (expected, actual) in pose
            .rotation()
            .m
            .iter()
            .flatten()
            .zip(estimated.rotation().m.iter().flatten())
        {
            assert!((expected - actual).abs() < 2e-3);
        }
        assert!((pose.translation().x - estimated.translation().x).abs() < 2e-3);
        assert!((pose.translation().y - estimated.translation().y).abs() < 2e-3);
        assert!((pose.translation().z - estimated.translation().z).abs() < 2e-3);
    }

    #[test]
    fn solve_pnp_ransac_rejects_outliers() {
        let camera = camera();
        let pose = sample_pose();
        let mut pairs = (0..20)
            .map(|index| {
                let object = Vec3::new(
                    (index % 5) as f64 * 0.1 - 0.2,
                    (index / 5) as f64 * 0.1 - 0.15,
                    (index % 3) as f64 * 0.05,
                );
                let image = project_object_point(pose, camera, object).unwrap();
                ObjectImageCorrespondence::try_new(object, image).unwrap()
            })
            .collect::<Vec<_>>();
        for pair in pairs.iter_mut().take(4) {
            *pair = ObjectImageCorrespondence::try_new(
                pair.object(),
                Vec2 { x: pair.image().x + 80.0, y: pair.image().y - 60.0 },
            )
            .unwrap();
        }
        let estimate = solve_pnp_ransac(
            &pairs,
            camera,
            RobustEstimationOptions {
                threshold: 2.0,
                confidence: 0.99,
                max_iterations: 500,
                seed: 7,
            },
        )
        .unwrap();
        assert!(estimate.inlier_count() >= 14);
        let recovered = *estimate.model();
        assert!((pose.translation().z - recovered.translation().z).abs() < 5e-2);
    }
}
