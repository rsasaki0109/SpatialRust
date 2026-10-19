//! Absolute pose from 3D–2D correspondences (PnP) with deterministic RANSAC.

use spatialrust_math::{
    solve_linear_system, symmetric_eigen3, LeastSquaresResult, Mat3, Vec2, Vec3,
};

use crate::{
    AbsolutePose, CameraMatrix3, GeometricEstimate, ObjectImageCorrespondence,
    PointCorrespondence2, RobustEstimationOptions, VisionError, VisionResult,
};

/// Estimates an object-to-camera pose from at least four correspondences.
///
/// Uses calibrated DLT, or plane-aware homography initialization for planar and
/// near-planar object geometry, followed by Gauss–Newton refinement on
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

/// Robust PnP using deterministic six-point RANSAC and optional inlier refinement.
///
/// Refinement preserves the selected hypothesis on failure or when its consensus
/// is worse. More inliers win; equal counts require a lower inlier residual sum.
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
    let mut best: Option<PnpConsensus> = None;
    let mut iteration_limit = options.max_iterations;
    let mut iteration = 0;
    while iteration < iteration_limit {
        let indices = sample_unique(&mut rng, correspondences.len(), SAMPLE);
        let sample = indices.iter().map(|&index| correspondences[index]).collect::<Vec<_>>();
        if let Ok(model) =
            estimate_pnp_dlt(&sample, camera).and_then(|pose| refine_pnp(&sample, camera, pose))
        {
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
    let best = best
        .ok_or_else(|| VisionError::InvalidParameter("robust PnP found no valid model".into()))?;
    if best.3 < 4 {
        return Err(VisionError::InvalidParameter("robust PnP found too few inliers".into()));
    }
    let inlier_pairs = correspondences
        .iter()
        .zip(&best.1)
        .filter_map(|(&pair, &is_inlier)| is_inlier.then_some(pair))
        .collect::<Vec<_>>();
    let refinement = solve_pnp(&inlier_pairs, camera);
    finish_pnp_consensus(best, refinement, correspondences, camera, options.threshold)
}

// Pose, original-order decisions/residuals, inlier count and inlier residual sum.
type PnpConsensus = (AbsolutePose, Vec<bool>, Vec<f64>, usize, f64);

fn finish_pnp_consensus(
    mut best: PnpConsensus,
    refinement: VisionResult<AbsolutePose>,
    correspondences: &[ObjectImageCorrespondence],
    camera: CameraMatrix3,
    threshold: f64,
) -> VisionResult<GeometricEstimate<AbsolutePose>> {
    if let Ok(refined) = refinement {
        let (count, error) =
            rescore_pnp(refined, correspondences, camera, threshold, &mut best.1, &mut best.2);
        if count > best.3 || (count == best.3 && error < best.4) {
            best.0 = refined;
            best.3 = count;
            best.4 = error;
        } else {
            // Reuse the selected hypothesis's existing buffers. A rejected
            // candidate costs one extra projection pass, not another pair of
            // full-row output allocations. Restore rows before publishing.
            rescore_pnp(best.0, correspondences, camera, threshold, &mut best.1, &mut best.2);
        }
    }
    GeometricEstimate::try_new(best.0, correspondences.len(), best.1, best.2)
}

fn rescore_pnp(
    pose: AbsolutePose,
    correspondences: &[ObjectImageCorrespondence],
    camera: CameraMatrix3,
    threshold: f64,
    inliers: &mut [bool],
    residuals: &mut [f64],
) -> (usize, f64) {
    debug_assert_eq!(correspondences.len(), inliers.len());
    debug_assert_eq!(correspondences.len(), residuals.len());
    let mut count = 0;
    let mut error = 0.0;
    for ((&pair, inlier), residual) in
        correspondences.iter().zip(inliers.iter_mut()).zip(residuals.iter_mut())
    {
        *residual = pnp_residual(pose, pair, camera);
        *inlier = *residual <= threshold;
        if *inlier {
            count += 1;
            error += *residual;
        }
    }
    (count, error)
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
    let normalized =
        Vec3::new(camera_point.x / camera_point.z, camera_point.y / camera_point.z, 1.0);
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
    if let Some(pose) = planar_initialization(correspondences, camera)? {
        return Ok(pose);
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
    let translation =
        Vec3::new(calibrated_t.x / scale, calibrated_t.y / scale, calibrated_t.z / scale);
    // Flip if most points have negative depth.
    let pose = AbsolutePose::try_new(rotation, translation)?;
    let positive =
        correspondences.iter().filter(|pair| pose.transform_point(pair.object()).z > 0.0).count();
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

/// Lift normalized plane coordinates through a homography. This avoids the
/// extra null directions of 3D DLT when object points have negligible thickness.
fn planar_initialization(
    correspondences: &[ObjectImageCorrespondence],
    camera: CameraMatrix3,
) -> VisionResult<Option<AbsolutePose>> {
    let origin = correspondences[0].object();
    let extent = correspondences
        .iter()
        .map(|pair| {
            let p = pair.object() - origin;
            p.x.abs().max(p.y.abs()).max(p.z.abs())
        })
        .fold(0.0_f64, f64::max);
    if !extent.is_finite() || extent == 0.0 {
        return Err(VisionError::InvalidParameter("PnP object extent is zero or overflows".into()));
    }
    let mut mean = Vec3::new(0.0, 0.0, 0.0);
    for pair in correspondences {
        mean = mean + scale_vec(pair.object() - origin, 1.0 / extent);
    }
    mean = scale_vec(mean, 1.0 / correspondences.len() as f64);
    let centroid = origin + scale_vec(mean, extent);
    let mut covariance = [[0.0; 3]; 3];
    for pair in correspondences {
        let p = scale_vec(pair.object() - origin, 1.0 / extent) - mean;
        let components = [p.x, p.y, p.z];
        for row in 0..3 {
            for column in 0..3 {
                covariance[row][column] += components[row] * components[column];
            }
        }
    }
    if covariance.iter().flatten().any(|value| !value.is_finite())
        || ![centroid.x, centroid.y, centroid.z].into_iter().all(f64::is_finite)
    {
        return Err(VisionError::InvalidParameter("PnP object covariance overflows".into()));
    }
    let eigen = symmetric_eigen3(Mat3::from_rows(covariance[0], covariance[1], covariance[2]));
    if eigen.eigenvalues[1] <= eigen.eigenvalues[2] * 1e-10 {
        return Err(VisionError::InvalidParameter("PnP object points are collinear".into()));
    }
    if eigen.eigenvalues[0] > eigen.eigenvalues[2] * 1e-6 {
        return Ok(None);
    }
    let vectors = eigen.eigenvectors;
    let u = Vec3::new(vectors.m[0][2], vectors.m[1][2], vectors.m[2][2]);
    let v = Vec3::new(vectors.m[0][1], vectors.m[1][1], vectors.m[2][1]);
    let n = u.cross(v);
    let basis = Mat3::from_rows([u.x, v.x, n.x], [u.y, v.y, n.y], [u.z, v.z, n.z]);
    let pairs = correspondences
        .iter()
        .map(|pair| {
            let p = scale_vec(pair.object() - origin, 1.0 / extent) - mean;
            let pixel = camera.normalize_pixel(pair.image());
            PointCorrespondence2::try_new(
                Vec2 { x: p.dot(u), y: p.dot(v) },
                Vec2 { x: pixel.x, y: pixel.y },
            )
        })
        .collect::<VisionResult<Vec<_>>>()?;
    let homography = crate::multiview::estimate_homography(&pairs)?.matrix();
    let a = Vec3::new(homography.m[0][0], homography.m[1][0], homography.m[2][0]);
    let b = Vec3::new(homography.m[0][1], homography.m[1][1], homography.m[2][1]);
    let c = Vec3::new(homography.m[0][2], homography.m[1][2], homography.m[2][2]);
    let scale = (a.length() + b.length()) * 0.5;
    if !scale.is_finite() || scale <= f64::EPSILON {
        return Err(VisionError::InvalidParameter(
            "PnP planar homography has invalid scale".into(),
        ));
    }
    let mut best: Option<(AbsolutePose, usize, f64)> = None;
    for sign in [1.0, -1.0] {
        let first = scale_vec(a, sign / scale);
        let second = scale_vec(b, sign / scale);
        let third = first.cross(second);
        let approximate = Mat3::from_rows(
            [first.x, second.x, third.x],
            [first.y, second.y, third.y],
            [first.z, second.z, third.z],
        );
        let (camera_basis, _) = orthonormalize_rotation(approximate)?;
        let rotation = camera_basis.mul_mat3(basis.transpose());
        let translation = scale_vec(c, sign * extent / scale) - rotation.mul_vec3(centroid);
        let pose = AbsolutePose::try_new(rotation, translation)?;
        let positive = correspondences
            .iter()
            .filter(|pair| pose.transform_point(pair.object()).z > 0.0)
            .count();
        let error =
            correspondences.iter().map(|pair| pnp_residual(pose, *pair, camera)).sum::<f64>();
        if best.as_ref().map_or(true, |previous| {
            positive > previous.1 || (positive == previous.1 && error < previous.2)
        }) {
            best = Some((pose, positive, error));
        }
    }
    let (pose, positive, _) = best.expect("two planar signs evaluated");
    if positive == 0 {
        return Err(VisionError::InvalidParameter("PnP planar pose has no positive depth".into()));
    }
    Ok(Some(pose))
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

fn pnp_residual(pose: AbsolutePose, pair: ObjectImageCorrespondence, camera: CameraMatrix3) -> f64 {
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
    let skew =
        Mat3::from_rows([0.0, -axis.z, axis.y], [axis.z, 0.0, -axis.x], [-axis.y, axis.x, 0.0]);
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
        for (expected, actual) in
            pose.rotation().m.iter().flatten().zip(estimated.rotation().m.iter().flatten())
        {
            assert!((expected - actual).abs() < 2e-3);
        }
        assert!((pose.translation().x - estimated.translation().x).abs() < 2e-3);
        assert!((pose.translation().y - estimated.translation().y).abs() < 2e-3);
        assert!((pose.translation().z - estimated.translation().z).abs() < 2e-3);
    }

    #[test]
    fn planar_initialization_recovers_arbitrary_plane_at_multiple_scales() {
        let camera = camera();
        let base = sample_pose();
        let u = Vec3::new(0.8, 0.6, 0.0);
        let v = Vec3::new(0.0, 0.0, 1.0);
        let n = u.cross(v);
        for scale in [1e-4, 1.0, 1e4] {
            for thickness in [0.0, 1e-4] {
                let pose = AbsolutePose::try_new(
                    base.rotation(),
                    super::scale_vec(base.translation(), scale),
                )
                .unwrap();
                let pairs = (0..25)
                    .map(|index| {
                        let object = super::scale_vec(
                            super::scale_vec(u, (index % 5) as f64 * 0.2 - 0.4)
                                + super::scale_vec(v, (index / 5) as f64 * 0.2 - 0.4)
                                + super::scale_vec(n, thickness * ((index % 3) as f64 - 1.0))
                                + Vec3::new(0.3, -0.2, 0.1),
                            scale,
                        );
                        ObjectImageCorrespondence::try_new(
                            object,
                            project_object_point(pose, camera, object).unwrap(),
                        )
                        .unwrap()
                    })
                    .collect::<Vec<_>>();
                let estimated = solve_pnp(&pairs, camera).unwrap();
                for (a, b) in
                    estimated.rotation().m.iter().flatten().zip(pose.rotation().m.iter().flatten())
                {
                    assert!((a - b).abs() < 1e-5, "scale={scale}, thickness={thickness}");
                }
                let error = estimated.translation() - pose.translation();
                assert!(error.length() / scale < 1e-5);
            }
        }
    }

    #[test]
    fn planar_four_points_work_and_collinear_points_are_rejected() {
        let camera = camera();
        let pose = sample_pose();
        let objects = [
            Vec3::new(-0.4, -0.3, 0.0),
            Vec3::new(0.4, -0.3, 0.0),
            Vec3::new(-0.4, 0.3, 0.0),
            Vec3::new(0.4, 0.3, 0.0),
        ];
        let pairs = objects
            .iter()
            .map(|&object| {
                ObjectImageCorrespondence::try_new(
                    object,
                    project_object_point(pose, camera, object).unwrap(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let estimated = solve_pnp(&pairs, camera).unwrap();
        assert!((estimated.translation() - pose.translation()).length() < 1e-5);
        let line = (0..8)
            .map(|index| {
                let object = Vec3::new(index as f64 * 0.1, 0.0, 0.0);
                ObjectImageCorrespondence::try_new(
                    object,
                    project_object_point(pose, camera, object).unwrap(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        assert!(solve_pnp(&line, camera).is_err());
    }

    #[test]
    fn planar_ransac_recovers_pose_with_incorrect_image_pairs() {
        let camera = camera();
        let pose = sample_pose();
        let pairs = (0..30)
            .map(|index| {
                let object =
                    Vec3::new((index % 6) as f64 * 0.2 - 0.5, (index / 6) as f64 * 0.2 - 0.4, 0.0);
                let mut image = project_object_point(pose, camera, object).unwrap();
                if index < 6 {
                    image.x += 80.0;
                    image.y -= 60.0;
                }
                ObjectImageCorrespondence::try_new(object, image).unwrap()
            })
            .collect::<Vec<_>>();
        let estimate = solve_pnp_ransac(
            &pairs,
            camera,
            RobustEstimationOptions {
                threshold: 2.0,
                max_iterations: 300,
                seed: 7,
                ..RobustEstimationOptions::default()
            },
        )
        .unwrap();
        assert_eq!(estimate.inlier_count(), 24);
        assert!((estimate.model().translation() - pose.translation()).length() < 1e-5);
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

    #[test]
    fn robust_pose_survives_failed_inlier_reinitialization() {
        // Procedural near-planar geometry, not a captured sensor fixture.
        // The selected six-point hypothesis is valid, while cold DLT over its
        // consensus can produce an invalid rotation on the parent implementation.
        let camera = CameraMatrix3::from_intrinsics(
            CameraIntrinsics::try_new(525.0, 525.0, 319.5, 239.5, 640, 480).unwrap(),
        );
        let truth = AbsolutePose::try_new(
            Mat3::from_rows(
                [0.9983758533596424, -0.03588494159791196, -0.04424846206277005],
                [0.034957430317503015, 0.999156210831159, -0.021560250998475538],
                [0.044984814037660234, 0.019978421460035138, 0.9987878799735018],
            ),
            Vec3::new(0.01, -0.004, 0.007),
        )
        .unwrap();
        let mut rng = super::XorShift64::new(5);
        let uniform = |rng: &mut super::XorShift64| {
            (rng.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
        };
        let pairs = (0..90)
            .map(|index| {
                let x = -0.95 + 1.9 * uniform(&mut rng);
                let y = -0.7 + 1.4 * uniform(&mut rng);
                let z = 2.0 + 0.2 * x + 0.1 * y + 0.002 * (uniform(&mut rng) - 0.5);
                let object = Vec3::new(x, y, z);
                let mut image = project_object_point(truth, camera, object).unwrap();
                image.x += 0.5 * (uniform(&mut rng) - 0.5);
                image.y += 0.5 * (uniform(&mut rng) - 0.5);
                if index % 3 == 0 {
                    image.x = 40.0 + 560.0 * uniform(&mut rng);
                    image.y = 40.0 + 400.0 * uniform(&mut rng);
                }
                ObjectImageCorrespondence::try_new(object, image).unwrap()
            })
            .collect::<Vec<_>>();
        let estimate = solve_pnp_ransac(
            &pairs,
            camera,
            RobustEstimationOptions { threshold: 2.0, ..Default::default() },
        )
        .expect("optional refitting must not discard a valid robust hypothesis");
        assert!(estimate.inlier_count() >= 58);
        assert_eq!(estimate.inliers().len(), pairs.len());
        assert_eq!(estimate.residuals().len(), pairs.len());
        for ((pair, &inlier), &residual) in
            pairs.iter().zip(estimate.inliers()).zip(estimate.residuals())
        {
            let expected = super::pnp_residual(*estimate.model(), *pair, camera);
            assert_eq!(residual, expected);
            assert_eq!(inlier, expected <= 2.0);
        }
    }

    fn consensus_fixture() -> (CameraMatrix3, Vec<ObjectImageCorrespondence>) {
        let camera = camera();
        let pairs = (0..32)
            .map(|index| {
                let object = Vec3::new(
                    (index % 8) as f64 * 0.04 - 0.14,
                    (index / 8) as f64 * 0.06 - 0.09,
                    (index % 3) as f64 * 0.1,
                );
                let image = Vec2 {
                    x: 500.0 * (object.x / (object.z + 3.0)) + 320.0,
                    y: 500.0 * (object.y / (object.z + 3.0)) + 240.0,
                };
                ObjectImageCorrespondence::try_new(object, image).unwrap()
            })
            .collect();
        (camera, pairs)
    }

    fn translated_pose(x: f64, z: f64) -> AbsolutePose {
        AbsolutePose::try_new(Mat3::<f64>::identity(), Vec3::new(x, 0.0, z)).unwrap()
    }

    fn consensus_for(
        pose: AbsolutePose,
        pairs: &[ObjectImageCorrespondence],
        camera: CameraMatrix3,
    ) -> super::PnpConsensus {
        let mut inliers = vec![false; pairs.len()];
        let mut residuals = vec![0.0; pairs.len()];
        let (count, error) =
            super::rescore_pnp(pose, pairs, camera, 2.0, &mut inliers, &mut residuals);
        (pose, inliers, residuals, count, error)
    }

    #[test]
    fn optional_refit_error_or_worse_pose_preserves_owned_rows() {
        let (camera, pairs) = consensus_fixture();
        let original_pairs = pairs.clone();
        let selected = translated_pose(0.003, 3.0);
        for candidate in [
            Err(crate::VisionError::InvalidParameter("optional refit failed".into())),
            Ok(translated_pose(0.1, 3.0)),   // Fewer inliers.
            Ok(translated_pose(0.006, 3.0)), // Same count, larger error.
            Ok(translated_pose(0.0, -3.0)),  // All projections behind camera.
        ] {
            let best = consensus_for(selected, &pairs, camera);
            let decisions = best.1.as_ptr();
            let residuals = best.2.as_ptr();
            let estimate =
                super::finish_pnp_consensus(best, candidate, &pairs, camera, 2.0).unwrap();
            assert_eq!(*estimate.model(), selected);
            assert_eq!(estimate.inlier_count(), pairs.len());
            assert_eq!(estimate.inliers().as_ptr(), decisions);
            assert_eq!(estimate.residuals().as_ptr(), residuals);
            for (&residual, pair) in estimate.residuals().iter().zip(&pairs) {
                let expected = 500.0 * 0.003 / (pair.object().z + 3.0);
                assert!((residual - expected).abs() < 1e-12);
            }
        }
        assert_eq!(pairs, original_pairs);
    }

    #[test]
    fn equal_count_lower_error_refit_is_retained() {
        let (camera, pairs) = consensus_fixture();
        let best = consensus_for(translated_pose(0.003, 3.0), &pairs, camera);
        let refined = translated_pose(0.0, 3.0);
        let estimate = super::finish_pnp_consensus(best, Ok(refined), &pairs, camera, 2.0).unwrap();
        assert_eq!(*estimate.model(), refined);
        assert_eq!(estimate.inlier_count(), pairs.len());
        assert!(estimate.residuals().iter().all(|&value| value < 1e-12));
    }

    #[test]
    fn more_inliers_win_even_with_a_larger_inlier_residual_sum() {
        let (camera, mut pairs) = consensus_fixture();
        for pair in pairs.iter_mut().skip(30) {
            let mut image = pair.image();
            image.x += 2.2;
            *pair = ObjectImageCorrespondence::try_new(pair.object(), image).unwrap();
        }
        let best = consensus_for(translated_pose(0.0, 3.0), &pairs, camera);
        assert_eq!(best.3, 30);
        assert!(best.4 < 1e-12);
        let refined = translated_pose(0.0048, 3.0);
        let estimate = super::finish_pnp_consensus(best, Ok(refined), &pairs, camera, 2.0).unwrap();
        assert_eq!(*estimate.model(), refined);
        assert_eq!(estimate.inlier_count(), 32);
        assert!(estimate.residuals().iter().sum::<f64>() > 20.0);
    }

    #[test]
    fn exact_consensus_ties_keep_the_selected_pose() {
        let camera = camera();
        let pair = ObjectImageCorrespondence::try_new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec2 { x: 320.0, y: 240.0 },
        )
        .unwrap();
        let pairs = vec![pair; 8];
        let selected = translated_pose(0.0, 3.0);
        let tied = translated_pose(0.0, 4.0);
        let best = consensus_for(selected, &pairs, camera);
        let estimate = super::finish_pnp_consensus(best, Ok(tied), &pairs, camera, 2.0).unwrap();
        assert_eq!(*estimate.model(), selected);
        assert!(estimate.residuals().iter().all(|&value| value == 0.0));
    }

    #[test]
    fn missing_robust_hypothesis_still_returns_an_error() {
        let camera = camera();
        let pair = ObjectImageCorrespondence::try_new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec2 { x: 320.0, y: 240.0 },
        )
        .unwrap();
        assert!(solve_pnp_ransac(
            &[pair; 8],
            camera,
            RobustEstimationOptions { max_iterations: 8, ..Default::default() },
        )
        .is_err());
    }
}
