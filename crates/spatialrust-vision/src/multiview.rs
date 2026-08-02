//! Linear two-view model estimation and deterministic robust sampling.

use spatialrust_math::{
    solve_linear_system, symmetric_eigen3, LeastSquaresResult, Mat3, Vec2, Vec3,
};

use crate::{
    CameraMatrix3, Essential3, Fundamental3, GeometricEstimate, Homography3, PointCorrespondence2,
    RelativePose, RelativePoseEstimate, RobustEstimationOptions, TriangulatedPoint, VisionError,
    VisionResult,
};

/// Estimates a homography from at least four correspondences using normalized DLT.
pub fn estimate_homography(correspondences: &[PointCorrespondence2]) -> VisionResult<Homography3> {
    if correspondences.len() < 4 {
        return Err(VisionError::InvalidParameter(
            "homography estimation requires at least four correspondences".into(),
        ));
    }
    let source = correspondences.iter().map(|pair| pair.source()).collect::<Vec<_>>();
    let target = correspondences.iter().map(|pair| pair.target()).collect::<Vec<_>>();
    let source_normalization = Normalization2::from_points(&source)?;
    let target_normalization = Normalization2::from_points(&target)?;
    let mut normal = vec![vec![0.0; 8]; 8];
    let mut rhs = vec![0.0; 8];
    for pair in correspondences {
        let source = source_normalization.apply(pair.source());
        let target = target_normalization.apply(pair.target());
        accumulate_least_squares(
            &mut normal,
            &mut rhs,
            &[source.x, source.y, 1.0, 0.0, 0.0, 0.0, -target.x * source.x, -target.x * source.y],
            target.x,
        );
        accumulate_least_squares(
            &mut normal,
            &mut rhs,
            &[0.0, 0.0, 0.0, source.x, source.y, 1.0, -target.y * source.x, -target.y * source.y],
            target.y,
        );
    }
    let LeastSquaresResult::Solved(solution) = solve_linear_system(normal, rhs) else {
        return Err(VisionError::InvalidParameter(
            "homography correspondences are degenerate".into(),
        ));
    };
    let normalized = Mat3::from_rows(
        [solution[0], solution[1], solution[2]],
        [solution[3], solution[4], solution[5]],
        [solution[6], solution[7], 1.0],
    );
    Homography3::try_new(
        target_normalization.inverse.mul_mat3(normalized).mul_mat3(source_normalization.matrix),
    )
}

/// Estimates a fundamental matrix from at least eight correspondences.
///
/// Points are Hartley-normalized before the linear solve and the result is
/// projected to rank two before denormalization.
pub fn estimate_fundamental(
    correspondences: &[PointCorrespondence2],
) -> VisionResult<Fundamental3> {
    if correspondences.len() < 8 {
        return Err(VisionError::InvalidParameter(
            "fundamental estimation requires at least eight correspondences".into(),
        ));
    }
    let source = correspondences.iter().map(|pair| pair.source()).collect::<Vec<_>>();
    let target = correspondences.iter().map(|pair| pair.target()).collect::<Vec<_>>();
    let source_normalization = Normalization2::from_points(&source)?;
    let target_normalization = Normalization2::from_points(&target)?;
    let mut normal = vec![vec![0.0; 9]; 9];
    for pair in correspondences {
        let source = source_normalization.apply(pair.source());
        let target = target_normalization.apply(pair.target());
        let row = [
            target.x * source.x,
            target.x * source.y,
            target.x,
            target.y * source.x,
            target.y * source.y,
            target.y,
            source.x,
            source.y,
            1.0,
        ];
        for row_index in 0..9 {
            for column in 0..9 {
                normal[row_index][column] += row[row_index] * row[column];
            }
        }
    }
    let vector = smallest_symmetric_eigenvector(normal).ok_or_else(|| {
        VisionError::InvalidParameter("fundamental correspondences are degenerate".into())
    })?;
    let normalized = enforce_rank_two(Mat3::from_rows(
        [vector[0], vector[1], vector[2]],
        [vector[3], vector[4], vector[5]],
        [vector[6], vector[7], vector[8]],
    ));
    Fundamental3::try_new(
        target_normalization
            .matrix
            .transpose()
            .mul_mat3(normalized)
            .mul_mat3(source_normalization.matrix),
    )
}

/// Robustly estimates a homography using deterministic four-point RANSAC.
pub fn estimate_homography_ransac(
    correspondences: &[PointCorrespondence2],
    options: RobustEstimationOptions,
) -> VisionResult<GeometricEstimate<Homography3>> {
    robust_estimate(correspondences, options, 4, estimate_homography, homography_residual)
}

/// Robustly estimates a fundamental matrix using deterministic eight-point RANSAC.
pub fn estimate_fundamental_ransac(
    correspondences: &[PointCorrespondence2],
    options: RobustEstimationOptions,
) -> VisionResult<GeometricEstimate<Fundamental3>> {
    robust_estimate(correspondences, options, 8, estimate_fundamental, fundamental_residual)
}

/// Estimates an essential matrix after normalizing pixels with both cameras.
pub fn estimate_essential(
    correspondences: &[PointCorrespondence2],
    source_camera: CameraMatrix3,
    target_camera: CameraMatrix3,
) -> VisionResult<Essential3> {
    let normalized = normalize_correspondences(correspondences, source_camera, target_camera)?;
    estimate_essential_normalized(&normalized)
}

/// Robustly estimates an essential matrix in normalized-camera coordinates.
///
/// The RANSAC threshold is therefore expressed on the normalized image plane,
/// not in pixels.
pub fn estimate_essential_ransac(
    correspondences: &[PointCorrespondence2],
    source_camera: CameraMatrix3,
    target_camera: CameraMatrix3,
    options: RobustEstimationOptions,
) -> VisionResult<GeometricEstimate<Essential3>> {
    let normalized = normalize_correspondences(correspondences, source_camera, target_camera)?;
    robust_estimate(&normalized, options, 8, estimate_essential_normalized, essential_residual)
}

/// Triangulates one calibrated correspondence for a known relative pose.
pub fn triangulate_correspondence(
    correspondence: PointCorrespondence2,
    source_camera: CameraMatrix3,
    target_camera: CameraMatrix3,
    pose: RelativePose,
) -> VisionResult<TriangulatedPoint> {
    let source = source_camera.normalize_pixel(correspondence.source());
    let target = target_camera.normalize_pixel(correspondence.target());
    triangulate_normalized(
        Vec2 { x: source.x, y: source.y },
        Vec2 { x: target.x, y: target.y },
        pose,
    )
    .ok_or_else(|| VisionError::InvalidParameter("triangulation is degenerate".into()))
}

/// Recovers the essential-matrix pose with the most positive-depth points.
pub fn recover_relative_pose(
    essential: Essential3,
    correspondences: &[PointCorrespondence2],
    source_camera: CameraMatrix3,
    target_camera: CameraMatrix3,
) -> VisionResult<RelativePoseEstimate> {
    if correspondences.is_empty() {
        return Err(VisionError::InvalidParameter(
            "pose recovery requires at least one correspondence".into(),
        ));
    }
    let normalized = normalize_correspondences(correspondences, source_camera, target_camera)?;
    let (mut left, mut right) = essential_singular_vectors(essential.matrix())?;
    if determinant(left) < 0.0 {
        negate_column(&mut left, 2);
    }
    if determinant(right) < 0.0 {
        negate_column(&mut right, 2);
    }
    let w = Mat3::from_rows([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let rotations = [
        proper_rotation(left.mul_mat3(w).mul_mat3(right.transpose())),
        proper_rotation(left.mul_mat3(w.transpose()).mul_mat3(right.transpose())),
    ];
    let translation = column(left, 2).normalize();
    let mut best: Option<(RelativePose, Vec<Option<TriangulatedPoint>>, usize, f64)> = None;
    for rotation in rotations {
        for translation in [translation, Vec3::new(-translation.x, -translation.y, -translation.z)]
        {
            let pose = RelativePose::try_new(rotation, translation)?;
            let points = normalized
                .iter()
                .map(|pair| triangulate_normalized(pair.source(), pair.target(), pose))
                .collect::<Vec<_>>();
            let count = points.iter().flatten().filter(|point| point.has_positive_depth()).count();
            let error = points
                .iter()
                .flatten()
                .filter(|point| point.has_positive_depth())
                .map(|point| point.reprojection_error())
                .sum::<f64>();
            if best.as_ref().map_or(true, |candidate| {
                count > candidate.2 || (count == candidate.2 && error < candidate.3)
            }) {
                best = Some((pose, points, count, error));
            }
        }
    }
    let (pose, points, _, _) = best.expect("four essential-pose candidates");
    Ok(RelativePoseEstimate::new(pose, points))
}

fn normalize_correspondences(
    correspondences: &[PointCorrespondence2],
    source_camera: CameraMatrix3,
    target_camera: CameraMatrix3,
) -> VisionResult<Vec<PointCorrespondence2>> {
    correspondences
        .iter()
        .map(|pair| {
            let source = source_camera.normalize_pixel(pair.source());
            let target = target_camera.normalize_pixel(pair.target());
            PointCorrespondence2::try_new(
                Vec2 { x: source.x, y: source.y },
                Vec2 { x: target.x, y: target.y },
            )
        })
        .collect()
}

fn estimate_essential_normalized(
    correspondences: &[PointCorrespondence2],
) -> VisionResult<Essential3> {
    let fundamental = estimate_fundamental(correspondences)?;
    Essential3::try_new(project_essential(fundamental.matrix())?)
}

fn robust_estimate<Model: Copy>(
    correspondences: &[PointCorrespondence2],
    options: RobustEstimationOptions,
    sample_size: usize,
    estimate: fn(&[PointCorrespondence2]) -> VisionResult<Model>,
    residual: fn(Model, PointCorrespondence2) -> f64,
) -> VisionResult<GeometricEstimate<Model>> {
    let options = options.validate()?;
    if correspondences.len() < sample_size {
        return Err(VisionError::InvalidParameter(format!(
            "robust estimation requires at least {sample_size} correspondences"
        )));
    }
    let mut rng = XorShift64::new(options.seed);
    let mut best: Option<(Model, Vec<bool>, Vec<f64>, usize, f64)> = None;
    let mut iteration_limit = options.max_iterations;
    let mut iteration = 0;
    while iteration < iteration_limit {
        let indices = sample_unique(&mut rng, correspondences.len(), sample_size);
        let sample = indices.iter().map(|&index| correspondences[index]).collect::<Vec<_>>();
        if let Ok(model) = estimate(&sample) {
            let residuals = correspondences
                .iter()
                .copied()
                .map(|pair| residual(model, pair))
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
                if count >= sample_size {
                    let inlier_ratio = count as f64 / correspondences.len() as f64;
                    let success = inlier_ratio.powi(sample_size as i32).clamp(0.0, 1.0);
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
        VisionError::InvalidParameter("robust geometry estimation found no valid model".into())
    })?;
    if count < sample_size {
        return Err(VisionError::InvalidParameter(
            "robust geometry estimation found too few inliers".into(),
        ));
    }
    let inlier_pairs = correspondences
        .iter()
        .zip(&best_inliers)
        .filter_map(|(&pair, &is_inlier)| is_inlier.then_some(pair))
        .collect::<Vec<_>>();
    let refined = estimate(&inlier_pairs)?;
    let residuals =
        correspondences.iter().copied().map(|pair| residual(refined, pair)).collect::<Vec<_>>();
    let inliers = residuals.iter().map(|&value| value <= options.threshold).collect();
    GeometricEstimate::try_new(refined, correspondences.len(), inliers, residuals)
}

fn homography_residual(model: Homography3, pair: PointCorrespondence2) -> f64 {
    let projected = model.matrix().mul_vec3(Vec3::new(pair.source().x, pair.source().y, 1.0));
    if projected.z.abs() <= f64::EPSILON {
        return f64::MAX;
    }
    let dx = projected.x / projected.z - pair.target().x;
    let dy = projected.y / projected.z - pair.target().y;
    dx.hypot(dy)
}

fn fundamental_residual(model: Fundamental3, pair: PointCorrespondence2) -> f64 {
    epipolar_residual(model.matrix(), pair)
}

fn essential_residual(model: Essential3, pair: PointCorrespondence2) -> f64 {
    epipolar_residual(model.matrix(), pair)
}

fn epipolar_residual(matrix: Mat3<f64>, pair: PointCorrespondence2) -> f64 {
    let source = Vec3::new(pair.source().x, pair.source().y, 1.0);
    let target = Vec3::new(pair.target().x, pair.target().y, 1.0);
    let line_target = matrix.mul_vec3(source);
    let line_source = matrix.transpose().mul_vec3(target);
    let numerator = target.dot(line_target).abs();
    let denominator = line_target.x * line_target.x
        + line_target.y * line_target.y
        + line_source.x * line_source.x
        + line_source.y * line_source.y;
    if denominator <= f64::EPSILON {
        f64::MAX
    } else {
        numerator / denominator.sqrt()
    }
}

fn project_essential(matrix: Mat3<f64>) -> VisionResult<Mat3<f64>> {
    let (left, right) = essential_singular_vectors(matrix)?;
    let covariance = matrix.transpose().mul_mat3(matrix);
    let eigen = symmetric_eigen3(covariance);
    let singular = [eigen.eigenvalues[2].max(0.0).sqrt(), eigen.eigenvalues[1].max(0.0).sqrt()];
    let scale = 0.5 * (singular[0] + singular[1]);
    let mut result = Mat3::from_rows([0.0; 3], [0.0; 3], [0.0; 3]);
    for component in 0..2 {
        let left_column = column(left, component);
        let right_column = column(right, component);
        let left_values = [left_column.x, left_column.y, left_column.z];
        let right_values = [right_column.x, right_column.y, right_column.z];
        for (row, &left_value) in left_values.iter().enumerate() {
            for (column, &right_value) in right_values.iter().enumerate() {
                result.m[row][column] += scale * left_value * right_value;
            }
        }
    }
    Ok(result)
}

fn essential_singular_vectors(matrix: Mat3<f64>) -> VisionResult<(Mat3<f64>, Mat3<f64>)> {
    let eigen = symmetric_eigen3(matrix.transpose().mul_mat3(matrix));
    let right0 = column(eigen.eigenvectors, 2).normalize();
    let right1 = column(eigen.eigenvectors, 1).normalize();
    let right2 = right0.cross(right1).normalize();
    let sigma0 = eigen.eigenvalues[2].max(0.0).sqrt();
    let sigma1 = eigen.eigenvalues[1].max(0.0).sqrt();
    if sigma0 <= 1e-12 || sigma1 <= 1e-12 {
        return Err(VisionError::InvalidParameter(
            "essential matrix has fewer than two non-zero singular values".into(),
        ));
    }
    let left0 = scale_vec(matrix.mul_vec3(right0), 1.0 / sigma0).normalize();
    let mut left1 = scale_vec(matrix.mul_vec3(right1), 1.0 / sigma1);
    left1 = (left1 - scale_vec(left0, left0.dot(left1))).normalize();
    let left2 = left0.cross(left1).normalize();
    Ok((matrix_from_columns(left0, left1, left2), matrix_from_columns(right0, right1, right2)))
}

fn triangulate_normalized(
    source: Vec2<f64>,
    target: Vec2<f64>,
    pose: RelativePose,
) -> Option<TriangulatedPoint> {
    let rotation = pose.rotation();
    let translation = pose.translation();
    let rows = [
        vec![-1.0, 0.0, source.x, 0.0],
        vec![0.0, -1.0, source.y, 0.0],
        vec![
            target.x * rotation.m[2][0] - rotation.m[0][0],
            target.x * rotation.m[2][1] - rotation.m[0][1],
            target.x * rotation.m[2][2] - rotation.m[0][2],
            target.x * translation.z - translation.x,
        ],
        vec![
            target.y * rotation.m[2][0] - rotation.m[1][0],
            target.y * rotation.m[2][1] - rotation.m[1][1],
            target.y * rotation.m[2][2] - rotation.m[1][2],
            target.y * translation.z - translation.y,
        ],
    ];
    let mut normal = vec![vec![0.0; 4]; 4];
    for row in rows {
        for first in 0..4 {
            for second in 0..4 {
                normal[first][second] += row[first] * row[second];
            }
        }
    }
    let homogeneous = smallest_symmetric_eigenvector(normal)?;
    if homogeneous[3].abs() <= 1e-12 {
        return None;
    }
    let position = Vec3::new(
        homogeneous[0] / homogeneous[3],
        homogeneous[1] / homogeneous[3],
        homogeneous[2] / homogeneous[3],
    );
    let target_position = rotation.mul_vec3(position) + translation;
    if position.z.abs() <= 1e-12 || target_position.z.abs() <= 1e-12 {
        return None;
    }
    let source_error =
        (position.x / position.z - source.x).hypot(position.y / position.z - source.y);
    let target_error = (target_position.x / target_position.z - target.x)
        .hypot(target_position.y / target_position.z - target.y);
    TriangulatedPoint::try_new(
        position,
        position.z,
        target_position.z,
        0.5 * (source_error + target_error),
    )
    .ok()
}

fn matrix_from_columns(first: Vec3<f64>, second: Vec3<f64>, third: Vec3<f64>) -> Mat3<f64> {
    Mat3::from_rows(
        [first.x, second.x, third.x],
        [first.y, second.y, third.y],
        [first.z, second.z, third.z],
    )
}

fn column(matrix: Mat3<f64>, index: usize) -> Vec3<f64> {
    Vec3::new(matrix.m[0][index], matrix.m[1][index], matrix.m[2][index])
}

fn scale_vec(vector: Vec3<f64>, scale: f64) -> Vec3<f64> {
    Vec3::new(vector.x * scale, vector.y * scale, vector.z * scale)
}

fn determinant(matrix: Mat3<f64>) -> f64 {
    matrix.m[0][0] * (matrix.m[1][1] * matrix.m[2][2] - matrix.m[1][2] * matrix.m[2][1])
        - matrix.m[0][1] * (matrix.m[1][0] * matrix.m[2][2] - matrix.m[1][2] * matrix.m[2][0])
        + matrix.m[0][2] * (matrix.m[1][0] * matrix.m[2][1] - matrix.m[1][1] * matrix.m[2][0])
}

fn negate_column(matrix: &mut Mat3<f64>, column: usize) {
    for row in 0..3 {
        matrix.m[row][column] = -matrix.m[row][column];
    }
}

fn proper_rotation(mut matrix: Mat3<f64>) -> Mat3<f64> {
    if determinant(matrix) < 0.0 {
        for row in &mut matrix.m {
            for value in row {
                *value = -*value;
            }
        }
    }
    matrix
}

fn enforce_rank_two(matrix: Mat3<f64>) -> Mat3<f64> {
    let covariance = matrix.transpose().mul_mat3(matrix);
    let eigen = symmetric_eigen3(covariance);
    let vector = Vec3::new(
        eigen.eigenvectors.m[0][0],
        eigen.eigenvectors.m[1][0],
        eigen.eigenvectors.m[2][0],
    );
    let image = matrix.mul_vec3(vector);
    let mut result = matrix;
    for row in 0..3 {
        for column in 0..3 {
            result.m[row][column] -=
                [image.x, image.y, image.z][row] * [vector.x, vector.y, vector.z][column];
        }
    }
    result
}

fn accumulate_least_squares(normal: &mut [Vec<f64>], rhs: &mut [f64], row: &[f64], value: f64) {
    for row_index in 0..row.len() {
        rhs[row_index] += row[row_index] * value;
        for column in 0..row.len() {
            normal[row_index][column] += row[row_index] * row[column];
        }
    }
}

struct Normalization2 {
    matrix: Mat3<f64>,
    inverse: Mat3<f64>,
}

impl Normalization2 {
    fn from_points(points: &[Vec2<f64>]) -> VisionResult<Self> {
        let count = points.len() as f64;
        let center = Vec2 {
            x: points.iter().map(|point| point.x).sum::<f64>() / count,
            y: points.iter().map(|point| point.y).sum::<f64>() / count,
        };
        let rms = (points
            .iter()
            .map(|point| {
                let dx = point.x - center.x;
                let dy = point.y - center.y;
                dx * dx + dy * dy
            })
            .sum::<f64>()
            / count)
            .sqrt();
        if !rms.is_finite() || rms <= f64::EPSILON {
            return Err(VisionError::InvalidParameter(
                "geometry points have zero spatial extent".into(),
            ));
        }
        let scale = 2.0_f64.sqrt() / rms;
        Ok(Self {
            matrix: Mat3::from_rows(
                [scale, 0.0, -scale * center.x],
                [0.0, scale, -scale * center.y],
                [0.0, 0.0, 1.0],
            ),
            inverse: Mat3::from_rows(
                [1.0 / scale, 0.0, center.x],
                [0.0, 1.0 / scale, center.y],
                [0.0, 0.0, 1.0],
            ),
        })
    }

    fn apply(&self, point: Vec2<f64>) -> Vec2<f64> {
        let normalized = self.matrix.mul_vec3(Vec3::new(point.x, point.y, 1.0));
        Vec2 { x: normalized.x, y: normalized.y }
    }
}

#[allow(clippy::needless_range_loop)]
fn smallest_symmetric_eigenvector(mut matrix: Vec<Vec<f64>>) -> Option<Vec<f64>> {
    let size = matrix.len();
    if size == 0 || matrix.iter().any(|row| row.len() != size) {
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
    Some(vector)
}

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed })
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn sample_unique(rng: &mut XorShift64, population: usize, count: usize) -> Vec<usize> {
    let mut sample = Vec::with_capacity(count);
    while sample.len() < count {
        let candidate = (rng.next() % population as u64) as usize;
        if !sample.contains(&candidate) {
            sample.push(candidate);
        }
    }
    sample
}

#[cfg(test)]
mod tests {
    use super::{
        estimate_essential, estimate_fundamental, estimate_homography, estimate_homography_ransac,
        fundamental_residual, homography_residual, recover_relative_pose,
        triangulate_correspondence,
    };
    use crate::{
        CameraMatrix3, Essential3, PointCorrespondence2, RelativePose, RobustEstimationOptions,
    };
    use spatialrust_camera::CameraIntrinsics;
    use spatialrust_math::{Mat3, Vec2, Vec3};

    fn correspondence(source: (f64, f64), target: (f64, f64)) -> PointCorrespondence2 {
        PointCorrespondence2::try_new(
            Vec2 { x: source.0, y: source.1 },
            Vec2 { x: target.0, y: target.1 },
        )
        .unwrap()
    }

    #[test]
    fn homography_recovers_known_projective_mapping() {
        let pairs = [
            ((0.0, 0.0), (3.0, -2.0)),
            ((10.0, 0.0), (23.0, -2.0)),
            ((0.0, 5.0), (3.0, 13.0)),
            ((10.0, 5.0), (23.0, 13.0)),
            ((4.0, 2.0), (11.0, 4.0)),
        ]
        .map(|(source, target)| correspondence(source, target));
        let model = estimate_homography(&pairs).unwrap();
        assert!(pairs.iter().all(|&pair| homography_residual(model, pair) < 1e-9));
    }

    #[test]
    fn homography_ransac_rejects_large_outliers_deterministically() {
        let mut pairs = (0..20)
            .map(|index| {
                let x = f64::from(index % 5) * 8.0;
                let y = f64::from(index / 5) * 7.0;
                correspondence((x, y), (1.5 * x + 4.0, 0.75 * y - 3.0))
            })
            .collect::<Vec<_>>();
        pairs.push(correspondence((5.0, 5.0), (500.0, -300.0)));
        pairs.push(correspondence((9.0, 11.0), (-200.0, 400.0)));
        let options = RobustEstimationOptions { threshold: 0.1, seed: 7, ..Default::default() };
        let first = estimate_homography_ransac(&pairs, options).unwrap();
        let second = estimate_homography_ransac(&pairs, options).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.inlier_count(), 20);
    }

    #[test]
    fn eight_point_model_satisfies_synthetic_epipolar_constraints() {
        let pairs = (0..16)
            .map(|index| {
                let x = f64::from(index % 4) * 0.3 - 0.4;
                let y = f64::from(index / 4) * 0.2 - 0.3;
                let depth = 2.0 + f64::from(index) * 0.05;
                let source = (x / depth, y / depth);
                let target = ((x + 0.2) / depth, y / depth);
                correspondence(source, target)
            })
            .collect::<Vec<_>>();
        let model = estimate_fundamental(&pairs).unwrap();
        assert!(pairs.iter().all(|&pair| fundamental_residual(model, pair) < 1e-7));
    }

    #[test]
    fn calibrated_triangulation_and_pose_recovery_choose_positive_depth() {
        let intrinsics = CameraIntrinsics::try_new(500.0, 500.0, 320.0, 240.0, 640, 480).unwrap();
        let camera = CameraMatrix3::from_intrinsics(intrinsics);
        let known_pose =
            RelativePose::try_new(Mat3::<f64>::identity(), Vec3::new(0.2, 0.0, 0.0)).unwrap();
        let points = (0..16)
            .map(|index| {
                Vec3::new(
                    f64::from(index % 4) * 0.25 - 0.4,
                    f64::from(index / 4) * 0.2 - 0.3,
                    2.0 + f64::from((index * index + 3 * index) % 17) * 0.07,
                )
            })
            .collect::<Vec<_>>();
        let pairs = points
            .iter()
            .map(|point| {
                let target = *point + known_pose.translation();
                correspondence(
                    (500.0 * point.x / point.z + 320.0, 500.0 * point.y / point.z + 240.0),
                    (500.0 * target.x / target.z + 320.0, 500.0 * target.y / target.z + 240.0),
                )
            })
            .collect::<Vec<_>>();
        let triangulated =
            triangulate_correspondence(pairs[3], camera, camera, known_pose).unwrap();
        assert!((triangulated.position().x - points[3].x).abs() < 1e-8);
        assert!((triangulated.position().z - points[3].z).abs() < 1e-8);
        assert!(triangulated.has_positive_depth());

        let estimated = estimate_essential(&pairs, camera, camera).unwrap();
        let maximum_residual = pairs
            .iter()
            .map(|pair| {
                let source = camera.normalize_pixel(pair.source());
                let target = camera.normalize_pixel(pair.target());
                let normalized = correspondence((source.x, source.y), (target.x, target.y));
                super::essential_residual(estimated, normalized)
            })
            .fold(0.0_f64, f64::max);
        assert!(maximum_residual < 1e-7, "maximum residual {maximum_residual}");
        let essential = Essential3::try_new(Mat3::from_rows(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -0.2],
            [0.0, 0.2, 0.0],
        ))
        .unwrap();
        let recovered = recover_relative_pose(essential, &pairs, camera, camera).unwrap();
        assert_eq!(recovered.positive_depth_count(), pairs.len());
        assert!(recovered.pose().translation().x.abs() > 0.99);
    }
}
