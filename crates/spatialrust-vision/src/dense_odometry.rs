//! Explicit two-frame CPU RGB-D projective odometry.

use spatialrust_image::ImageView;
use spatialrust_math::{solve_linear_system, LeastSquaresResult, Mat3, Vec3};

use crate::{AbsolutePose, CameraMatrix3, VisionError, VisionResult};

const TILE_ROWS: usize = 16;

/// Borrowed grayscale and metric-depth views for one registered RGB-D frame.
///
/// Depth is already in metres. Nonfinite, nonpositive and out-of-range samples
/// are missing data. Estimation builds owned CPU pyramids; no device copy occurs.
#[derive(Clone, Copy, Debug)]
pub struct DenseRgbdFrame<'a> {
    gray: ImageView<'a, u8, 1>,
    depth: ImageView<'a, f32, 1>,
}

impl<'a> DenseRgbdFrame<'a> {
    /// Checks that both nonempty images have the same geometry.
    pub fn try_new(gray: ImageView<'a, u8, 1>, depth: ImageView<'a, f32, 1>) -> VisionResult<Self> {
        if gray.width() < 3 || gray.height() < 3 {
            return Err(VisionError::InvalidDimensions("dense RGB-D requires at least 3x3".into()));
        }
        if gray.width() != depth.width() || gray.height() != depth.height() {
            return Err(VisionError::ShapeMismatch("dense RGB-D intensity/depth geometry".into()));
        }
        Ok(Self { gray, depth })
    }
}

/// Bounded dense photometric/projective-depth least-squares controls.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseRgbdOdometryOptions {
    /// Iterations from coarsest to finest; one to six levels, 1..=100 each.
    pub iterations: Vec<usize>,
    /// Inclusive minimum valid metric depth.
    pub min_depth: f64,
    /// Inclusive maximum valid metric depth.
    pub max_depth: f64,
    /// Maximum absolute target-depth / warped-source-depth difference, metres.
    pub max_depth_difference: f64,
    /// Relative squared depth weight; intensity weight is one minus this value.
    pub depth_weight: f64,
    /// Minimum projective correspondence count at every level and final pose.
    pub min_correspondences: usize,
    /// Minimum matched fraction of all valid source pixels at the returned pose.
    /// Initial/intermediate hypotheses need the minimum count and a solvable
    /// normal system; they are not yet accepted motion estimates.
    pub min_correspondence_fraction: f64,
    /// Stop a level when rotation-vector norm plus translation norm is smaller.
    pub epsilon: f64,
    /// Upper bound on logical owned pyramid/reduction buffer payloads.
    pub max_working_bytes: usize,
}

impl Default for DenseRgbdOdometryOptions {
    fn default() -> Self {
        Self {
            iterations: vec![20, 10, 5],
            min_depth: 0.1,
            max_depth: 5.0,
            max_depth_difference: 0.03,
            depth_weight: 0.968,
            min_correspondences: 30,
            min_correspondence_fraction: 0.5,
            epsilon: 1e-7,
            max_working_bytes: 128 * 1024 * 1024,
        }
    }
}

impl DenseRgbdOdometryOptions {
    /// Preflights dimensions, controls and logical buffers before allocation.
    ///
    /// Excludes input views/copies, allocator headers and worker/thread stacks.
    pub fn required_working_bytes(&self, width: usize, height: usize) -> VisionResult<usize> {
        let invalid =
            || VisionError::InvalidParameter("invalid or over-budget dense RGB-D controls".into());
        if self.iterations.is_empty()
            || self.iterations.len() > 6
            || self.iterations.iter().any(|&n| !(1..=100).contains(&n))
            || !self.min_depth.is_finite()
            || self.min_depth <= 0.0
            || !self.max_depth.is_finite()
            || self.max_depth < self.min_depth
            || !self.max_depth_difference.is_finite()
            || self.max_depth_difference <= 0.0
            || !self.depth_weight.is_finite()
            || !(0.0..=1.0).contains(&self.depth_weight)
            || self.min_correspondences < 6
            || !self.min_correspondence_fraction.is_finite()
            || self.min_correspondence_fraction <= 0.0
            || self.min_correspondence_fraction > 1.0
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
        {
            return Err(invalid());
        }
        let mut pixels = 0usize;
        let (mut w, mut h) = (width, height);
        for _ in &self.iterations {
            if w < 3 || h < 3 {
                return Err(invalid());
            }
            pixels =
                pixels.checked_add(w.checked_mul(h).ok_or_else(invalid)?).ok_or_else(invalid)?;
            w = w / 2 + w % 2;
            h = h / 2 + h % 2;
        }
        let pyramid = pixels.checked_mul(4 * std::mem::size_of::<f64>()).ok_or_else(invalid)?;
        let tiles = (height / TILE_ROWS + usize::from(height % TILE_ROWS != 0))
            .checked_mul(std::mem::size_of::<Normal>())
            .ok_or_else(invalid)?;
        let bytes = pyramid.checked_add(tiles).ok_or_else(invalid)?;
        if bytes > self.max_working_bytes {
            return Err(invalid());
        }
        Ok(bytes)
    }
}

/// Owned pose and final full-resolution projective diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseRgbdOdometryEstimate {
    /// Source-camera to target-camera metric pose.
    pub pose: AbsolutePose,
    /// Number of solved pose updates, bounded by the iteration schedule.
    pub iterations: usize,
    /// All valid full-resolution source depth samples, including unmatched rows.
    pub valid_source_pixels: usize,
    /// Accepted full-resolution projective correspondences.
    pub correspondences: usize,
    /// Correspondences divided by all valid source pixels.
    pub correspondence_fraction: f64,
    /// Unweighted normalized [0,1] intensity RMSE over final correspondences.
    pub photometric_rmse: f64,
    /// Unweighted depth RMSE in metres over final correspondences.
    pub depth_rmse_m: f64,
    /// Preflighted logical pyramid/reduction-buffer payload bound.
    pub working_bytes: usize,
    /// Maximum private CPU workers available to this call (one on WASM).
    pub workers: usize,
}

/// Estimates dense motion from both registered frames and an explicit initial pose.
///
/// Uses a centered Gaussian pyramid, bilinear target warps and analytic
/// photometric/depth derivatives with left rigid updates. Original inputs are
/// borrowed; owned CPU pyramids and fixed ordered row-tile reductions are bounded.
/// Singular systems, insufficient overlap and invalid updates return errors.
pub fn estimate_dense_rgbd_odometry(
    source: DenseRgbdFrame<'_>,
    target: DenseRgbdFrame<'_>,
    camera: CameraMatrix3,
    initial: AbsolutePose,
    options: &DenseRgbdOdometryOptions,
) -> VisionResult<DenseRgbdOdometryEstimate> {
    if source.gray.width() != target.gray.width() || source.gray.height() != target.gray.height() {
        return Err(VisionError::ShapeMismatch("dense RGB-D source/target geometry".into()));
    }
    let working_bytes =
        options.required_working_bytes(source.gray.width(), source.gray.height())?;
    let source = pyramid(source, options);
    let target = pyramid(target, options);
    let mut pose = initial;
    let mut iterations = 0;
    for (schedule, level) in (0..source.len()).rev().enumerate() {
        let intrinsics = scaled_camera(camera, level);
        for iteration in 0..options.iterations[schedule] {
            let normal = linearize(&source[level], &target[level], intrinsics, pose, options);
            check_support(&normal, options, SupportStage::Linearization { level, iteration })?;
            let mut matrix = normal.hessian;
            for (row, values) in matrix.iter_mut().enumerate() {
                for (col, value) in values.iter_mut().take(row).enumerate() {
                    *value = normal.hessian[col][row];
                }
            }
            if matrix.iter().flatten().chain(normal.rhs.iter()).any(|x| !x.is_finite()) {
                return Err(VisionError::InvalidParameter(
                    "nonfinite dense RGB-D normal system".into(),
                ));
            }
            let LeastSquaresResult::Solved(delta) = solve_linear_system(
                matrix.iter().map(|row| row.to_vec()).collect(),
                normal.rhs.to_vec(),
            ) else {
                return Err(VisionError::InvalidParameter(
                    "singular dense RGB-D normal system".into(),
                ));
            };
            if delta.iter().any(|x| !x.is_finite()) {
                return Err(VisionError::InvalidParameter("nonfinite dense RGB-D update".into()));
            }
            pose = update_pose(pose, &delta)?;
            iterations += 1;
            if Vec3::new(delta[0], delta[1], delta[2]).length()
                + Vec3::new(delta[3], delta[4], delta[5]).length()
                < options.epsilon
            {
                break;
            }
        }
    }
    let final_rows = linearize(&source[0], &target[0], scaled_camera(camera, 0), pose, options);
    check_support(&final_rows, options, SupportStage::Final)?;
    Ok(DenseRgbdOdometryEstimate {
        pose,
        iterations,
        valid_source_pixels: final_rows.valid,
        correspondences: final_rows.count,
        correspondence_fraction: final_rows.count as f64 / final_rows.valid as f64,
        photometric_rmse: (final_rows.photo_sse / final_rows.count as f64).sqrt(),
        depth_rmse_m: (final_rows.depth_sse / final_rows.count as f64).sqrt(),
        working_bytes,
        workers: worker_count(source[0].width * source[0].height),
    })
}

struct Level {
    width: usize,
    height: usize,
    color: Vec<f64>,
    depth: Vec<f64>,
}

fn pyramid(frame: DenseRgbdFrame<'_>, options: &DenseRgbdOdometryOptions) -> Vec<Level> {
    let (width, height) = (frame.gray.width(), frame.gray.height());
    let mut color = Vec::with_capacity(width * height);
    let mut depth = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            color
                .push(f64::from(frame.gray.get(x, y).expect("checked image rectangle")[0]) / 255.0);
            let z = f64::from(frame.depth.get(x, y).expect("checked image rectangle")[0]);
            depth.push(if z.is_finite() && z >= options.min_depth && z <= options.max_depth {
                z
            } else {
                f64::NAN
            });
        }
    }
    let mut levels = vec![Level { width, height, color, depth }];
    while levels.len() < options.iterations.len() {
        let previous = levels.last().expect("base level exists");
        let (width, height) =
            (previous.width / 2 + previous.width % 2, previous.height / 2 + previous.height % 2);
        let mut color = Vec::with_capacity(width * height);
        let mut depth = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let (mut intensity, mut zsum, mut weight) = (0.0, 0.0, 0.0);
                for (dy, wy) in [1.0, 2.0, 1.0].into_iter().enumerate() {
                    for (dx, wx) in [1.0, 2.0, 1.0].into_iter().enumerate() {
                        let sx = (2 * x + dx).saturating_sub(1).min(previous.width - 1);
                        let sy = (2 * y + dy).saturating_sub(1).min(previous.height - 1);
                        let index = sy * previous.width + sx;
                        let w = wy * wx;
                        intensity += w * previous.color[index];
                        if previous.depth[index].is_finite() {
                            zsum += w * previous.depth[index];
                            weight += w;
                        }
                    }
                }
                color.push(intensity / 16.0);
                depth.push(if weight >= 8.0 { zsum / weight } else { f64::NAN });
            }
        }
        levels.push(Level { width, height, color, depth });
    }
    levels
}

fn scaled_camera(camera: CameraMatrix3, level: usize) -> [f64; 4] {
    let k = camera.matrix().m;
    let scale = (1usize << level) as f64;
    [k[0][0] / scale, k[1][1] / scale, k[0][2] / scale, k[1][2] / scale]
}

// Value and exact within-cell bilinear derivatives. Missing corners reject a row.
fn sample(data: &[f64], width: usize, height: usize, x: f64, y: f64) -> Option<[f64; 3]> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= (width - 1) as f64
        || y >= (height - 1) as f64
    {
        return None;
    }
    let (ix, iy) = (x as usize, y as usize);
    let (u, v) = (x - ix as f64, y - iy as f64);
    let [a, b, c, d] = [
        data[iy * width + ix],
        data[iy * width + ix + 1],
        data[(iy + 1) * width + ix],
        data[(iy + 1) * width + ix + 1],
    ];
    if [a, b, c, d].iter().any(|z| !z.is_finite()) {
        return None;
    }
    Some([
        (a * (1.0 - u) + b * u) * (1.0 - v) + (c * (1.0 - u) + d * u) * v,
        (b - a) * (1.0 - v) + (d - c) * v,
        (c - a) * (1.0 - u) + (d - b) * u,
    ])
}

#[derive(Clone, Default, Debug, PartialEq)]
struct Normal {
    hessian: [[f64; 6]; 6],
    rhs: [f64; 6],
    photo_sse: f64,
    depth_sse: f64,
    valid: usize,
    count: usize,
}

impl Normal {
    fn add(&mut self, jacobian: [f64; 6], residual: f64, weight: f64) {
        for a in 0..6 {
            self.rhs[a] -= weight * jacobian[a] * residual;
            for b in a..6 {
                self.hessian[a][b] += weight * jacobian[a] * jacobian[b];
            }
        }
    }
    fn merge(&mut self, other: Self) {
        for a in 0..6 {
            self.rhs[a] += other.rhs[a];
            for b in a..6 {
                self.hessian[a][b] += other.hessian[a][b];
            }
        }
        self.photo_sse += other.photo_sse;
        self.depth_sse += other.depth_sse;
        self.valid += other.valid;
        self.count += other.count;
    }
}

// Scalar residual gradient wrt transformed point, followed by left rigid action.
fn twist_gradient(point: Vec3<f64>, gradient: [f64; 3]) -> [f64; 6] {
    let [x, y, z] = gradient;
    [point.y * z - point.z * y, point.z * x - point.x * z, point.x * y - point.y * x, x, y, z]
}

fn residual_jacobians(
    point: Vec3<f64>,
    k: [f64; 4],
    color: [f64; 3],
    depth: [f64; 3],
) -> [[f64; 6]; 2] {
    let [fx, fy, _, _] = k;
    let image_gradient = |du: f64, dv: f64| {
        let a = du * fx / point.z;
        let b = dv * fy / point.z;
        [a, b, -(a * point.x + b * point.y) / point.z]
    };
    let photo = image_gradient(color[1], color[2]);
    let mut geometric = image_gradient(depth[1], depth[2]);
    geometric[2] -= 1.0;
    [twist_gradient(point, photo), twist_gradient(point, geometric)]
}

fn tile(
    source: &Level,
    target: &Level,
    k: [f64; 4],
    pose: AbsolutePose,
    options: &DenseRgbdOdometryOptions,
    block: usize,
) -> Normal {
    let mut out = Normal::default();
    let [fx, fy, cx, cy] = k;
    for y in block * TILE_ROWS..((block + 1) * TILE_ROWS).min(source.height) {
        for x in 0..source.width {
            let i = y * source.width + x;
            let z = source.depth[i];
            if !z.is_finite() {
                continue;
            }
            out.valid += 1;
            let point = pose.transform_point(Vec3::new(
                (x as f64 - cx) * z / fx,
                (y as f64 - cy) * z / fy,
                z,
            ));
            if !point.z.is_finite() || point.z <= 0.0 {
                continue;
            }
            let (u, v) = (fx * point.x / point.z + cx, fy * point.y / point.z + cy);
            let Some(depth) = sample(&target.depth, target.width, target.height, u, v) else {
                continue;
            };
            let depth_error = depth[0] - point.z;
            if depth_error.abs() > options.max_depth_difference {
                continue;
            }
            let Some(color) = sample(&target.color, target.width, target.height, u, v) else {
                continue;
            };
            let photo_error = color[0] - source.color[i];
            let jacobian = residual_jacobians(point, k, color, depth);
            out.add(jacobian[0], photo_error, 1.0 - options.depth_weight);
            out.add(jacobian[1], depth_error, options.depth_weight);
            out.photo_sse += photo_error * photo_error;
            out.depth_sse += depth_error * depth_error;
            out.count += 1;
        }
    }
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn pool() -> Option<&'static rayon::ThreadPool> {
    use std::sync::OnceLock;
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let workers = std::thread::available_parallelism().map_or(1, |n| n.get().min(4));
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|i| format!("dense-rgbd-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}

fn worker_count(pixels: usize) -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    if pixels >= 32768 {
        return pool().map_or(1, rayon::ThreadPool::current_num_threads);
    }
    let _ = pixels;
    1
}

fn linearize(
    source: &Level,
    target: &Level,
    k: [f64; 4],
    pose: AbsolutePose,
    options: &DenseRgbdOdometryOptions,
) -> Normal {
    let blocks = source.height / TILE_ROWS + usize::from(source.height % TILE_ROWS != 0);
    let evaluate = |b| tile(source, target, k, pose, options, b);
    let partials: Vec<Normal> = {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if worker_count(source.width * source.height) > 1 {
                use rayon::prelude::*;
                pool()
                    .expect("worker count requires a pool")
                    .install(|| (0..blocks).into_par_iter().map(evaluate).collect())
            } else {
                (0..blocks).map(evaluate).collect()
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            (0..blocks).map(evaluate).collect()
        }
    };
    let mut total = Normal::default();
    for part in partials {
        total.merge(part);
    }
    total
}

#[derive(Clone, Copy)]
enum SupportStage {
    Linearization { level: usize, iteration: usize },
    Final,
}

fn check_support(
    rows: &Normal,
    options: &DenseRgbdOdometryOptions,
    stage: SupportStage,
) -> VisionResult<()> {
    let fraction = if rows.valid == 0 { 0.0 } else { rows.count as f64 / rows.valid as f64 };
    if rows.valid == 0
        || rows.count < options.min_correspondences
        || (matches!(stage, SupportStage::Final) && fraction < options.min_correspondence_fraction)
        || !rows.photo_sse.is_finite()
        || !rows.depth_sse.is_finite()
    {
        let context = match stage {
            SupportStage::Linearization { level, iteration } => {
                format!("level {level}, iteration {iteration}")
            }
            SupportStage::Final => "returned full-resolution pose".to_string(),
        };
        return Err(VisionError::InvalidParameter(format!(
            "insufficient or nonfinite dense RGB-D support at {context}: {}/{} correspondences ({fraction}), minimum count {}, final fraction {}",
            rows.count,rows.valid,options.min_correspondences,options.min_correspondence_fraction)));
    }
    Ok(())
}

fn update_pose(pose: AbsolutePose, delta: &[f64]) -> VisionResult<AbsolutePose> {
    let omega = Vec3::new(delta[0], delta[1], delta[2]);
    let angle = omega.length();
    let a2 = angle * angle;
    let (s, c) = if angle < 1e-8 {
        (1.0 - a2 / 6.0, 0.5 - a2 / 24.0)
    } else {
        (angle.sin() / angle, (1.0 - angle.cos()) / a2)
    };
    let [x, y, z] = [omega.x, omega.y, omega.z];
    let rotation = Mat3::from_rows(
        [1.0 - c * (y * y + z * z), c * x * y - s * z, c * x * z + s * y],
        [c * x * y + s * z, 1.0 - c * (x * x + z * z), c * y * z - s * x],
        [c * x * z - s * y, c * y * z + s * x, 1.0 - c * (x * x + y * y)],
    );
    AbsolutePose::try_new(
        rotation.mul_mat3(pose.rotation()),
        rotation.mul_vec3(pose.translation()) + Vec3::new(delta[3], delta[4], delta[5]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatialrust_image::Image;

    fn identity() -> AbsolutePose {
        AbsolutePose::try_new(Mat3::<f64>::identity(), Vec3::new(0.0, 0.0, 0.0)).unwrap()
    }
    fn camera() -> CameraMatrix3 {
        CameraMatrix3::try_from_pinhole(64.0, 64.0, 48.0, 36.0).unwrap()
    }
    fn texture(x: f64, y: f64) -> u8 {
        (128.0 + 45.0 * (0.26 * x).sin() + 40.0 * (0.19 * y).cos() + 25.0 * (0.13 * (x + y)).sin())
            .round() as u8
    }
    fn images(shift: f64, z: f32) -> (Image<u8, 1>, Image<f32, 1>) {
        let gray = (0..72)
            .flat_map(|y| (0..96).map(move |x| texture(x as f64 - shift, y as f64)))
            .collect();
        (Image::try_new(96, 72, gray).unwrap(), Image::try_new(96, 72, vec![z; 96 * 72]).unwrap())
    }

    #[test]
    fn identity_and_metric_horizontal_motion_on_a_textured_plane() {
        let (a, depth) = images(0.0, 2.0);
        let (b, other) = images(1.0, 2.0);
        let source = DenseRgbdFrame::try_new(a.view(), depth.view()).unwrap();
        let target = DenseRgbdFrame::try_new(b.view(), other.view()).unwrap();
        let options = DenseRgbdOdometryOptions::default();
        let still =
            estimate_dense_rgbd_odometry(source, source, camera(), identity(), &options).unwrap();
        assert!(still.pose.translation().length() < 1e-10);
        assert!(still.photometric_rmse < 1e-12 && still.depth_rmse_m < 1e-12);
        let moved =
            estimate_dense_rgbd_odometry(source, target, camera(), identity(), &options).unwrap();
        // One pixel is exactly z/fx = 1/32 metre, in source-to-target direction.
        assert!(
            (moved.pose.translation() - Vec3::new(0.03125, 0.0, 0.0)).length() < 1e-4,
            "{:?}",
            moved.pose.translation()
        );
        assert!(moved.depth_rmse_m < 1e-5 && moved.photometric_rmse < 1e-4);
        assert_eq!(moved.valid_source_pixels, 96 * 72);
        assert!(moved.correspondence_fraction > 0.9);
        assert!(moved.iterations <= 35 && moved.working_bytes <= options.max_working_bytes);
    }

    #[test]
    fn left_rotation_also_rotates_existing_translation() {
        let initial =
            AbsolutePose::try_new(Mat3::<f64>::identity(), Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let pose =
            update_pose(initial, &[0.0, 0.0, std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0]).unwrap();
        assert!((pose.translation() - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-12);
        assert!(
            (pose.transform_point(Vec3::new(1.0, 0.0, 0.0)) - Vec3::new(0.0, 2.0, 0.0)).length()
                < 1e-12
        );
    }

    #[test]
    fn low_initial_overlap_can_recover_final_support() {
        let (source_gray, source_depth) = images(0.0, 2.0);
        let angle = 0.05_f64;
        let (s, c) = angle.sin_cos();
        let mut gray = Vec::new();
        let mut depth = Vec::new();
        for y in 0..72 {
            for x in 0..96 {
                let ray_x = (x as f64 - 48.0) / 64.0;
                let ray_y = (y as f64 - 36.0) / 64.0;
                let z = 2.0 / (c + s * ray_x);
                let source_x = 64.0 * z * (c * ray_x - s) / 2.0 + 48.0;
                let source_y = 64.0 * z * ray_y / 2.0 + 36.0;
                gray.push(texture(source_x, source_y));
                depth.push(z as f32);
            }
        }
        let target_gray = Image::<u8, 1>::try_new(96, 72, gray).unwrap();
        let target_depth = Image::<f32, 1>::try_new(96, 72, depth).unwrap();
        let source = DenseRgbdFrame::try_new(source_gray.view(), source_depth.view()).unwrap();
        let target = DenseRgbdFrame::try_new(target_gray.view(), target_depth.view()).unwrap();
        let options = DenseRgbdOdometryOptions::default();
        let a = pyramid(source, &options);
        let b = pyramid(target, &options);
        let initial = linearize(&a[0], &b[0], scaled_camera(camera(), 0), identity(), &options);
        assert!((initial.count as f64 / initial.valid as f64) < 0.5);
        let result =
            estimate_dense_rgbd_odometry(source, target, camera(), identity(), &options).unwrap();
        let expected = Mat3::from_rows([c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]);
        for (actual, truth) in
            result.pose.rotation().m.iter().flatten().zip(expected.m.iter().flatten())
        {
            assert!((actual - truth).abs() < 1e-3);
        }
        assert!(result.pose.translation().length() < 0.003);
        assert!(result.correspondence_fraction > 0.8 && result.photometric_rmse < 1.0 / 255.0);
    }

    #[test]
    fn an_exact_but_small_overlap_is_still_rejected_at_return() {
        let (gray, source_depth) = images(0.0, 2.0);
        let mut values = vec![0.0; 96 * 72];
        for y in 0..72 {
            for x in 30..55 {
                values[y * 96 + x] = 2.0;
            }
        }
        let target_depth = Image::<f32, 1>::try_new(96, 72, values).unwrap();
        let source = DenseRgbdFrame::try_new(gray.view(), source_depth.view()).unwrap();
        let target = DenseRgbdFrame::try_new(gray.view(), target_depth.view()).unwrap();
        let options = DenseRgbdOdometryOptions { iterations: vec![1], ..Default::default() };
        let error = estimate_dense_rgbd_odometry(source, target, camera(), identity(), &options)
            .unwrap_err();
        assert!(error.to_string().contains("returned full-resolution pose"));
    }

    #[test]
    fn both_residual_jacobians_match_six_finite_pose_differences() {
        let (width, height) = (96, 72);
        let k = [64.0, 80.0, 30.0, 24.0];
        let field = |a: f64, b: f64, c: f64| {
            (0..height)
                .flat_map(move |y| (0..width).map(move |x| a * x as f64 + b * y as f64 + c))
                .collect::<Vec<_>>()
        };
        let color = field(0.002, -0.001, 0.1);
        let depth = field(0.003, 0.004, 2.0);
        let object = Vec3::new(0.17, -0.22, 1.8);
        let pose =
            AbsolutePose::try_new(Mat3::<f64>::identity(), Vec3::new(0.03, -0.04, 0.2)).unwrap();
        let values = |pose: AbsolutePose| {
            let p = pose.transform_point(object);
            let u = k[0] * p.x / p.z + k[2];
            let v = k[1] * p.y / p.z + k[3];
            let i = sample(&color, width, height, u, v).unwrap();
            let d = sample(&depth, width, height, u, v).unwrap();
            ([i[0] - 0.3, d[0] - p.z], residual_jacobians(p, k, i, d))
        };
        let (_, jacobian) = values(pose);
        let epsilon = 1e-6;
        for axis in 0..6 {
            let mut delta = [0.0; 6];
            delta[axis] = epsilon;
            let plus = values(update_pose(pose, &delta).unwrap()).0;
            delta[axis] = -epsilon;
            let minus = values(update_pose(pose, &delta).unwrap()).0;
            for row in 0..2 {
                assert!(
                    ((plus[row] - minus[row]) / (2.0 * epsilon) - jacobian[row][axis]).abs() < 1e-8,
                    "row {row}, axis {axis}"
                );
            }
        }
    }

    #[test]
    fn missing_corners_and_unobservable_or_missing_scenes_fail() {
        let mut values = vec![2.0; 16];
        values[5] = f64::NAN;
        assert!(sample(&values, 4, 4, 0.0, 0.0).is_none());
        assert!(sample(&values, 4, 4, f64::NAN, 0.0).is_none());
        let gray = Image::<u8, 1>::try_new(96, 72, vec![100; 96 * 72]).unwrap();
        for z in [2.0, f32::NAN, 0.0, -1.0, f32::INFINITY] {
            let depth = Image::<f32, 1>::try_new(96, 72, vec![z; 96 * 72]).unwrap();
            let frame = DenseRgbdFrame::try_new(gray.view(), depth.view()).unwrap();
            assert!(estimate_dense_rgbd_odometry(
                frame,
                frame,
                camera(),
                identity(),
                &Default::default()
            )
            .is_err());
        }
    }

    #[test]
    fn preflight_rejects_invalid_controls_overflow_and_small_budget() {
        let options = DenseRgbdOdometryOptions::default();
        let bytes = options.required_working_bytes(96, 72).unwrap();
        assert!(options.required_working_bytes(usize::MAX, usize::MAX).is_err());
        assert!(options.required_working_bytes(3, 3).is_err());
        let bad = vec![
            DenseRgbdOdometryOptions { iterations: vec![], ..options.clone() },
            DenseRgbdOdometryOptions { iterations: vec![0], ..options.clone() },
            DenseRgbdOdometryOptions { iterations: vec![1; 7], ..options.clone() },
            DenseRgbdOdometryOptions { epsilon: f64::NAN, ..options.clone() },
            DenseRgbdOdometryOptions { depth_weight: 1.1, ..options.clone() },
            DenseRgbdOdometryOptions { min_correspondence_fraction: 0.0, ..options.clone() },
            DenseRgbdOdometryOptions { max_depth_difference: 0.0, ..options.clone() },
            DenseRgbdOdometryOptions { max_working_bytes: bytes - 1, ..options.clone() },
        ];
        for option in bad {
            assert!(option.required_working_bytes(96, 72).is_err());
        }
        let (gray, depth) = images(0.0, 2.0);
        let tiny = Image::<f32, 1>::try_new(3, 3, vec![2.0; 9]).unwrap();
        assert!(DenseRgbdFrame::try_new(gray.view(), tiny.view()).is_err());
        let tiny_gray = Image::<u8, 1>::try_new(3, 3, vec![0; 9]).unwrap();
        let a = DenseRgbdFrame::try_new(gray.view(), depth.view()).unwrap();
        let b = DenseRgbdFrame::try_new(tiny_gray.view(), tiny.view()).unwrap();
        assert!(estimate_dense_rgbd_odometry(a, b, camera(), identity(), &options).is_err());
    }

    #[test]
    fn centered_pyramid_camera_and_depth_missingness_are_explicit() {
        let gray = Image::<u8, 1>::try_new(5, 5, vec![0; 25]).unwrap();
        let mut z = vec![f32::NAN; 25];
        z[12] = 2.0;
        let depth = Image::<f32, 1>::try_new(5, 5, z).unwrap();
        let options = DenseRgbdOdometryOptions { iterations: vec![1, 1], ..Default::default() };
        let p = pyramid(DenseRgbdFrame::try_new(gray.view(), depth.view()).unwrap(), &options);
        assert_eq!((p[1].width, p[1].height), (3, 3));
        assert!(p[1].depth[4].is_nan());
        let k = scaled_camera(camera(), 1);
        assert_eq!(k, [32.0, 32.0, 24.0, 18.0]);
        // Source pixel 2x and level-one x describe exactly the same optical ray.
        assert_eq!((10.0 - 48.0) / 64.0, (5.0 - k[2]) / k[0]);
    }

    #[test]
    fn parallel_and_serial_row_tiles_reduce_bit_identically() {
        let (width, height) = (256, 160);
        let color = (0..height)
            .flat_map(|y| (0..width).map(move |x| f64::from(texture(x as f64, y as f64)) / 255.0))
            .collect();
        let level = Level { width, height, color, depth: vec![2.0; width * height] };
        let options = DenseRgbdOdometryOptions::default();
        let mut serial = Normal::default();
        for block in 0..height / TILE_ROWS {
            serial.merge(tile(
                &level,
                &level,
                [128.0, 128.0, 128.0, 80.0],
                identity(),
                &options,
                block,
            ));
        }
        let parallel = linearize(&level, &level, [128.0, 128.0, 128.0, 80.0], identity(), &options);
        assert_eq!(serial, parallel);
        assert!((1..=4).contains(&worker_count(width * height)));
    }
}
