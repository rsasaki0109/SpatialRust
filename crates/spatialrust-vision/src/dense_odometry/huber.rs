//! Explicit scalar Huber fitting on f32 intermediates / proper left Euler updates; raw final admission stays in the parent.
use super::*;

const PHOTO_HUBER_DELTA: f64 = 0.02;

pub(super) fn required_bytes(
    options: &DenseRgbdOdometryOptions,
    width: usize,
    height: usize,
) -> VisionResult<usize> {
    let bytes = reference_precision::required_bytes(options, width, height)?;
    let delta = options.max_depth_difference / 3.0;
    if !delta.is_finite() || delta <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "invalid or underflowed Huber depth delta".into(),
        ));
    }
    Ok(bytes)
}

// A nonfinite residual/parameter or an unrepresentable positive factor must
// reach the existing finite-system checks, never silently remove a row.
fn huber_factor(residual: f64, delta: f64) -> f64 {
    if !residual.is_finite() || !delta.is_finite() || delta <= 0.0 {
        return f64::NAN;
    }
    let magnitude = residual.abs();
    if magnitude <= delta {
        return 1.0;
    }
    let factor = delta / magnitude;
    if factor > 0.0 && factor.is_finite() {
        factor
    } else {
        f64::NAN
    }
}

fn rounded_coordinate(value: f64, length: usize) -> Option<usize> {
    let rounded = (value + 0.5).trunc();
    if !rounded.is_finite() || rounded < 0.0 || rounded >= length as f64 {
        None
    } else {
        Some(rounded as usize)
    }
}

fn correspondence(
    source: &Level,
    target: &Level,
    k: [f64; 4],
    pose: AbsolutePose,
    index: usize,
    options: &DenseRgbdOdometryOptions,
) -> Option<(Vec3<f64>, usize)> {
    let z = source.depth[index];
    if !z.is_finite() {
        return None;
    }
    let [fx, fy, cx, cy] = k;
    let point = pose.transform_point(Vec3::new(
        ((index % source.width) as f64 - cx) * z / fx,
        ((index / source.width) as f64 - cy) * z / fy,
        z,
    ));
    if !point.z.is_finite() || point.z <= 0.0 {
        return None;
    }
    let x = rounded_coordinate(fx * point.x / point.z + cx, target.width)?;
    let y = rounded_coordinate(fy * point.y / point.z + cy, target.height)?;
    let i = y * target.width + x;
    let depth = target.depth[i];
    (depth.is_finite() && (depth - point.z).abs() <= options.max_depth_difference)
        .then_some((point, i))
}

fn normalize(
    source: &mut Level,
    target: &mut Level,
    k: [f64; 4],
    initial: AbsolutePose,
    options: &DenseRgbdOdometryOptions,
) -> VisionResult<[f64; 2]> {
    let (mut a, mut b, mut count) = (0.0, 0.0, 0usize);
    for i in 0..source.depth.len() {
        if let Some((_, j)) = correspondence(source, target, k, initial, i, options) {
            a += source.color[i];
            b += target.color[j];
            count += 1;
        }
    }
    let means = [a / count as f64, b / count as f64];
    let gains = means.map(|mean| 0.5 / mean);
    if count < options.min_correspondences
        || means.iter().chain(gains.iter()).any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(VisionError::InvalidParameter(
            "reference-precision-huber pair means require minimum count and finite positive means/gains"
                .into(),
        ));
    }
    #[cfg(not(target_arch = "wasm32"))]
    if worker_count(source.width * source.height) > 1 {
        use rayon::prelude::*;
        pool().expect("worker count requires a pool").install(|| {
            source.color.par_iter_mut().zip(target.color.par_iter_mut()).for_each(|(a, b)| {
                *a = f64::from((*a * gains[0]) as f32);
                *b = f64::from((*b * gains[1]) as f32);
            });
        });
        return Ok(gains);
    }
    for (a, b) in source.color.iter_mut().zip(target.color.iter_mut()) {
        *a = f64::from((*a * gains[0]) as f32);
        *b = f64::from((*b * gains[1]) as f32);
    }
    Ok(gains)
}

fn downsample_tile(
    previous: &Level,
    smooth: &Level,
    width: usize,
    start: usize,
    color: &mut [f64],
    depth: &mut [f64],
) {
    for (i, (c, d)) in color.iter_mut().zip(depth.iter_mut()).enumerate() {
        let x = 2 * (i % width);
        let y = 2 * (start + i / width);
        let a = y * previous.width + x;
        let b = a + previous.width;
        *c = mean4(&smooth.color, a, b);
        *d = mean4(&previous.depth, a, b);
    }
}

fn downsample(previous: &Level) -> Level {
    let smooth = filter(previous, false);
    let (width, height) = (previous.width / 2, previous.height / 2);
    let mut out =
        Level { width, height, color: vec![0.0; width * height], depth: vec![0.0; width * height] };
    #[cfg(not(target_arch = "wasm32"))]
    if worker_count(previous.width * previous.height) > 1 {
        use rayon::prelude::*;
        let tile_pixels = width * TILE_ROWS;
        pool().expect("worker count requires a pool").install(|| {
            out.color
                .par_chunks_mut(tile_pixels)
                .zip(out.depth.par_chunks_mut(tile_pixels))
                .enumerate()
                .for_each(|(block, (color, depth))| {
                    downsample_tile(previous, &smooth, width, block * TILE_ROWS, color, depth);
                });
        });
        return out;
    }
    downsample_tile(previous, &smooth, width, 0, &mut out.color, &mut out.depth);
    out
}

fn pyramid(base: Level, count: usize) -> Vec<Level> {
    let mut out = vec![base];
    while out.len() < count {
        out.push(downsample(out.last().expect("base level exists")));
    }
    out
}

fn gradient_tile(level: &Level, start: usize, output: &mut [[f64; 4]]) {
    for (i, out) in output.iter_mut().enumerate() {
        let (x, y) = (i % level.width, start + i / level.width);
        *out = [
            separable(
                &level.color,
                level.width,
                level.height,
                x,
                y,
                [-1.0, 0.0, 1.0],
                [1.0, 2.0, 1.0],
            ),
            separable(
                &level.color,
                level.width,
                level.height,
                x,
                y,
                [1.0, 2.0, 1.0],
                [-1.0, 0.0, 1.0],
            ),
            separable(
                &level.depth,
                level.width,
                level.height,
                x,
                y,
                [-1.0, 0.0, 1.0],
                [1.0, 2.0, 1.0],
            ),
            separable(
                &level.depth,
                level.width,
                level.height,
                x,
                y,
                [1.0, 2.0, 1.0],
                [-1.0, 0.0, 1.0],
            ),
        ]
        .map(|v| v * 0.125);
        if !out[2].is_finite() {
            out[2] = 0.0;
        }
        if !out[3].is_finite() {
            out[3] = 0.0;
        }
    }
}

fn gradients(level: &Level) -> Vec<[f64; 4]> {
    let mut out = vec![[0.0; 4]; level.color.len()];
    #[cfg(not(target_arch = "wasm32"))]
    if worker_count(level.width * level.height) > 1 {
        use rayon::prelude::*;
        pool().expect("worker count requires a pool").install(|| {
            out.par_chunks_mut(level.width * TILE_ROWS).enumerate().for_each(|(block, part)| {
                gradient_tile(level, block * TILE_ROWS, part);
            });
        });
        return out;
    }
    gradient_tile(level, 0, &mut out);
    out
}

fn tile(
    source: &Level,
    target: &Level,
    gradients: &[[f64; 4]],
    k: [f64; 4],
    pose: AbsolutePose,
    options: &DenseRgbdOdometryOptions,
    block: usize,
) -> Normal {
    let mut normal = Normal::default();
    for y in block * TILE_ROWS..((block + 1) * TILE_ROWS).min(source.height) {
        for x in 0..source.width {
            let i = y * source.width + x;
            if !source.depth[i].is_finite() {
                continue;
            }
            normal.valid += 1;
            let Some((_, j)) = correspondence(source, target, k, pose, i, options) else {
                continue;
            };
            // Lookup uses the original f64 projection; fitting uses quantized XYZ.
            let z = source.depth[i];
            let point = pose.transform_point(Vec3::new(
                f64::from((((x as f64 - k[2]) * z) * (1.0 / k[0])) as f32),
                f64::from((((y as f64 - k[3]) * z) * (1.0 / k[1])) as f32),
                f64::from(z as f32),
            ));
            if !point.x.is_finite()
                || !point.y.is_finite()
                || !point.z.is_finite()
                || point.z <= 0.0
            {
                continue;
            }
            let g = gradients[j];
            let color = [target.color[j], g[0], g[1]];
            let depth = [target.depth[j], g[2], g[3]];
            let photo_error = f64::from(color[0] as f32 - source.color[i] as f32);
            let depth_error = depth[0] - point.z;
            let jacobian = residual_jacobians(point, k, color, depth);
            let photo_scale = (1.0 - options.depth_weight).sqrt()
                * huber_factor(photo_error, PHOTO_HUBER_DELTA).sqrt();
            let depth_scale = options.depth_weight.sqrt()
                * huber_factor(depth_error, options.max_depth_difference / 3.0).sqrt();
            normal.add(jacobian[0].map(|v| v * photo_scale), photo_error * photo_scale, 1.0);
            normal.add(jacobian[1].map(|v| v * depth_scale), depth_error * depth_scale, 1.0);
            normal.photo_sse += photo_error * photo_error;
            normal.depth_sse += depth_error * depth_error;
            normal.count += 1;
        }
    }
    normal
}

fn linearize(
    source: &Level,
    target: &Level,
    gradients: &[[f64; 4]],
    k: [f64; 4],
    pose: AbsolutePose,
    options: &DenseRgbdOdometryOptions,
) -> Normal {
    let blocks = source.height / TILE_ROWS + usize::from(source.height % TILE_ROWS != 0);
    let evaluate = |b| tile(source, target, gradients, k, pose, options, b);
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

pub(super) fn estimate(
    source: DenseRgbdFrame<'_>,
    target: DenseRgbdFrame<'_>,
    camera: CameraMatrix3,
    initial: AbsolutePose,
    options: &DenseRgbdOdometryOptions,
    working_bytes: usize,
) -> VisionResult<DenseRgbdOdometryEstimate> {
    let a = base_level(source, options);
    let b = base_level(target, options);
    let mut source = filter(&a, true);
    let mut target = filter(&b, true);
    normalize(&mut source, &mut target, scaled_camera(camera, 0), initial, options)?;
    let source = pyramid(source, options.iterations.len());
    let target = pyramid(target, options.iterations.len());
    let mut pose = initial;
    let mut iterations = 0;
    for (schedule, level) in (0..source.len()).rev().enumerate() {
        let k = scaled_camera(camera, level);
        let gradients = gradients(&target[level]);
        for iteration in 0..options.iterations[schedule] {
            let normal = linearize(&source[level], &target[level], &gradients, k, pose, options);
            check_support(&normal, options, SupportStage::Linearization { level, iteration })?;
            let mut matrix = normal.hessian;
            for (row, values) in matrix.iter_mut().enumerate() {
                for (col, value) in values.iter_mut().take(row).enumerate() {
                    *value = normal.hessian[col][row];
                }
            }
            if matrix.iter().flatten().chain(normal.rhs.iter()).any(|v| !v.is_finite()) {
                return Err(VisionError::InvalidParameter(
                    "nonfinite reference-precision-huber normal system".into(),
                ));
            }
            let LeastSquaresResult::Solved(delta) = solve_linear_system(
                matrix.iter().map(|r| r.to_vec()).collect(),
                normal.rhs.to_vec(),
            ) else {
                return Err(VisionError::InvalidParameter(
                    "singular reference-precision-huber normal system".into(),
                ));
            };
            if delta.iter().any(|v| !v.is_finite()) {
                return Err(VisionError::InvalidParameter(
                    "nonfinite reference-precision-huber update".into(),
                ));
            }
            pose = euler_update(pose, &delta)?;
            iterations += 1;
        }
    }
    let rows = super::linearize(&a, &b, scaled_camera(camera, 0), pose, options);
    check_support(&rows, options, SupportStage::Final)?;
    Ok(DenseRgbdOdometryEstimate {
        pose,
        iterations,
        valid_source_pixels: rows.valid,
        correspondences: rows.count,
        correspondence_fraction: rows.count as f64 / rows.valid as f64,
        photometric_rmse: (rows.photo_sse / rows.count as f64).sqrt(),
        depth_rmse_m: (rows.depth_sse / rows.count as f64).sqrt(),
        working_bytes,
        workers: worker_count(source[0].width * source[0].height),
    })
}

// Each pass has f32 products, an ordered f64 sum and one f32 store. Computing
// the three horizontal samples locally avoids an additional owned image plane.
#[allow(clippy::too_many_arguments)]
fn separable(
    values: &[f64],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    kx: [f32; 3],
    ky: [f32; 3],
) -> f64 {
    let mut vertical = 0.0f64;
    for (dy, wy) in ky.into_iter().enumerate() {
        let sy = (y + dy).saturating_sub(1).min(height - 1);
        let mut horizontal = 0.0f64;
        for (dx, wx) in kx.into_iter().enumerate() {
            let sx = (x + dx).saturating_sub(1).min(width - 1);
            horizontal += f64::from(values[sy * width + sx] as f32 * wx);
        }
        vertical += f64::from(horizontal as f32 * wy);
    }
    f64::from(vertical as f32)
}

fn filter_tile(
    raw: &Level,
    start: usize,
    color: &mut [f64],
    depth: &mut [f64],
    filter_depth: bool,
) {
    for (i, (c, d)) in color.iter_mut().zip(depth.iter_mut()).enumerate() {
        let (x, y) = (i % raw.width, start + i / raw.width);
        *c = separable(
            &raw.color,
            raw.width,
            raw.height,
            x,
            y,
            [0.25, 0.5, 0.25],
            [0.25, 0.5, 0.25],
        );
        *d = if filter_depth {
            separable(&raw.depth, raw.width, raw.height, x, y, [0.25, 0.5, 0.25], [0.25, 0.5, 0.25])
        } else {
            raw.depth[y * raw.width + x]
        };
    }
}

fn filter(raw: &Level, filter_depth: bool) -> Level {
    let mut out = Level {
        width: raw.width,
        height: raw.height,
        color: vec![0.0; raw.color.len()],
        depth: vec![0.0; raw.depth.len()],
    };
    #[cfg(not(target_arch = "wasm32"))]
    if worker_count(raw.width * raw.height) > 1 {
        use rayon::prelude::*;
        pool().expect("worker count requires a pool").install(|| {
            out.color
                .par_chunks_mut(raw.width * TILE_ROWS)
                .zip(out.depth.par_chunks_mut(raw.width * TILE_ROWS))
                .enumerate()
                .for_each(|(b, (c, d))| filter_tile(raw, b * TILE_ROWS, c, d, filter_depth));
        });
        return out;
    }
    filter_tile(raw, 0, &mut out.color, &mut out.depth, filter_depth);
    out
}

fn mean4(values: &[f64], a: usize, b: usize) -> f64 {
    f64::from(
        (values[a] as f32 + values[a + 1] as f32 + values[b] as f32 + values[b + 1] as f32)
            / 4.0f32,
    )
}

fn euler_update(pose: AbsolutePose, delta: &[f64]) -> VisionResult<AbsolutePose> {
    let (sx, cx) = delta[0].sin_cos();
    let (sy, cy) = delta[1].sin_cos();
    let (sz, cz) = delta[2].sin_cos();
    let rx = Mat3::from_rows([1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]);
    let ry = Mat3::from_rows([cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]);
    let rz = Mat3::from_rows([cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]);
    let rotation = rz.mul_mat3(ry).mul_mat3(rx);
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

    fn level(width: usize, height: usize) -> Level {
        Level {
            width,
            height,
            color: (0..width * height)
                .map(|i| f64::from(((i * 73 + 19) % 256) as f32 / 255.0))
                .collect(),
            depth: (0..width * height)
                .map(|i| f64::from(1.3f32 + (i % 17) as f32 * 0.002))
                .collect(),
        }
    }

    fn rho(r: f64, d: f64) -> f64 {
        if r.abs() <= d {
            0.5 * r * r
        } else {
            d * (r.abs() - 0.5 * d)
        }
    }

    #[test]
    fn weights_match_independent_loss_derivative_and_original_unclipped_scales() {
        for r in [-1., -0.02, -0.01, -0., 0., 0.01, 0.02, 1.] {
            let derivative = (rho(r + 1e-7, 0.02) - rho(r - 1e-7, 0.02)) / 2e-7;
            assert!((r * huber_factor(r, 0.02) - derivative).abs() < 3e-8);
            if r.abs() <= 0.02 {
                for w in [0f64, 0.032, 0.968, 1.] {
                    assert_eq!(
                        (w.sqrt() * huber_factor(r, 0.02).sqrt()).to_bits(),
                        w.sqrt().to_bits()
                    );
                }
            }
        }
        assert_eq!(huber_factor(-0.2, 0.02), huber_factor(0.2, 0.02));
        assert_eq!(huber_factor(0., 0.02), 1.);
    }

    #[test]
    fn nonfinite_and_underflowed_weights_cannot_silently_drop_rows() {
        for (r, d) in [
            (f64::NAN, 0.02),
            (f64::INFINITY, 0.02),
            (0., f64::NAN),
            (0., f64::INFINITY),
            (0., 0.),
            (0., -0.02),
            (f64::MAX, f64::from_bits(1)),
        ] {
            assert!(huber_factor(r, d).is_nan());
        }
        let finite = huber_factor(f64::MAX, f64::MIN_POSITIVE);
        assert!(finite.is_nan());
        assert!(huber_factor(f64::MAX, 1.).is_finite());
        assert!(required_bytes(
            &DenseRgbdOdometryOptions {
                max_depth_difference: f64::from_bits(1),
                ..Default::default()
            },
            640,
            480
        )
        .is_err());
    }

    #[test]
    fn scalar_irls_reduces_huber_objective_and_limits_one_outlier() {
        let samples = [0., 0., 0., 1.];
        let mut location = 0.25;
        let objective = |x: f64| samples.iter().map(|v| rho(v - x, 0.02)).sum::<f64>();
        for _ in 0..30 {
            let factors = samples.map(|v| huber_factor(v - location, 0.02));
            let next = samples.iter().zip(factors).map(|(v, w)| v * w).sum::<f64>()
                / factors.iter().sum::<f64>();
            assert!(objective(next) <= objective(location) + 1e-15);
            location = next;
        }
        assert!((location - 0.02 / 3.).abs() < 1e-12);
    }

    #[test]
    fn clipped_single_pixel_normal_matches_analytic_jacobians_and_keeps_raw_sse() {
        let mut source = level(8, 8);
        let mut target = level(8, 8);
        source.depth.fill(f64::NAN);
        source.depth[0] = 2.;
        target.depth.fill(f64::from(2.025f32));
        source.color.fill(f64::from(0.2f32));
        target.color.fill(f64::from(0.4f32));
        let options = DenseRgbdOdometryOptions::default();
        let observed = tile(
            &source,
            &target,
            &vec![[0.01, -0.02, 0.003, 0.004]; 64],
            [64., 80., 0., 0.],
            identity(),
            &options,
            0,
        );
        let jp = [1.6, 0.64, 0., 0.32, -0.8, 0.];
        let jd = [-0.32, 0.192, 0., 0.096, 0.16, -1.];
        let ep = f64::from(0.4f32 - 0.2f32);
        let ed = f64::from(2.025f32) - 2.;
        let wp = (1. - options.depth_weight) * 0.02 / ep.abs();
        let wd = options.depth_weight * (options.max_depth_difference / 3.) / ed.abs();
        assert_eq!((observed.valid, observed.count), (1, 1));
        assert_eq!(observed.photo_sse, ep * ep);
        assert_eq!(observed.depth_sse, ed * ed);
        for a in 0..6 {
            assert!((observed.rhs[a] - (-wp * jp[a] * ep - wd * jd[a] * ed)).abs() < 1e-15);
            for b in a..6 {
                assert!(
                    (observed.hessian[a][b] - (wp * jp[a] * jp[b] + wd * jd[a] * jd[b])).abs()
                        < 1e-15
                );
            }
        }
    }

    #[test]
    fn clipped_odd_dimension_parallel_reduction_equals_serial_bits() {
        let source = level(257, 145);
        let mut target = level(257, 145);
        for v in &mut target.color {
            *v += 0.2;
        }
        for v in &mut target.depth {
            *v += 0.025;
        }
        let k = [64., 80., 128., 72.];
        let opts = DenseRgbdOdometryOptions::default();
        let g = gradients(&target);
        let parallel = linearize(&source, &target, &g, k, identity(), &opts);
        let mut serial = Normal::default();
        for b in 0..source.height.div_ceil(TILE_ROWS) {
            serial.merge(tile(&source, &target, &g, k, identity(), &opts, b));
        }
        assert_eq!(parallel, serial);
        assert!(parallel.count > 30_000);
        assert!(parallel.photo_sse > 100.);
    }
    #[test]
    fn exact_working_boundary_tiny_solver_scratch_and_mixed_policies() {
        let options = DenseRgbdOdometryOptions::default();
        let n = required_bytes(&options, 640, 480).unwrap();
        assert_eq!(
            n,
            32 * (640 * 480 + 320 * 240 + 160 * 120)
                + 64 * 640 * 480
                + 30 * std::mem::size_of::<Normal>()
        );
        assert_eq!(
            required_bytes(
                &DenseRgbdOdometryOptions { max_working_bytes: n, ..options.clone() },
                640,
                480
            )
            .unwrap(),
            n
        );
        assert!(required_bytes(
            &DenseRgbdOdometryOptions { max_working_bytes: n - 1, ..options.clone() },
            640,
            480
        )
        .is_err());
        assert!(required_bytes(&options, usize::MAX, 480).is_err());
        let tiny = DenseRgbdOdometryOptions {
            iterations: vec![1],
            min_correspondences: 6,
            ..options.clone()
        };
        assert_eq!(required_bytes(&tiny, 3, 3).unwrap(), 96 * 9 + 48 * 8);
        for policy in [
            DenseRgbdPreprocessing::Gaussian3,
            DenseRgbdPreprocessing::Gaussian3GrayOnly,
            DenseRgbdPreprocessing::Gaussian3GrayOnlyPairMean,
        ] {
            assert!(options
                .required_working_bytes_with_model(
                    640,
                    480,
                    policy,
                    DenseRgbdFittingModel::ReferencePrecisionHuber
                )
                .is_err());
        }
        assert!(required_bytes(
            &DenseRgbdOdometryOptions { iterations: vec![1, 1], ..options },
            7,
            5
        )
        .is_err());
    }

    #[test]
    fn identity_executes_all_updates_and_uses_original_raw_denominator() {
        let (w, h) = (96, 72);
        let gray = Image::<u8, 1>::try_new(
            w,
            h,
            (0..w * h)
                .map(|i| {
                    let (x, y) = ((i % w) as f64, (i / w) as f64);
                    (128.
                        + 45. * (0.26 * x).sin()
                        + 40. * (0.19 * y).cos()
                        + 25. * (0.13 * (x + y)).sin())
                    .round() as u8
                })
                .collect(),
        )
        .unwrap();
        let mut depths = vec![2f32; w * h];
        depths[36 * w + 48] = f32::NAN;
        let depth = Image::<f32, 1>::try_new(w, h, depths).unwrap();
        let frame = DenseRgbdFrame::try_new(gray.view(), depth.view()).unwrap();
        let options = DenseRgbdOdometryOptions::default();
        let result = estimate_dense_rgbd_odometry_with_model(
            frame,
            frame,
            CameraMatrix3::try_from_pinhole(64., 64., 48., 36.).unwrap(),
            identity(),
            &options,
            DenseRgbdPreprocessing::Raw,
            DenseRgbdFittingModel::ReferencePrecisionHuber,
        )
        .unwrap();
        assert_eq!(result.iterations, 35);
        assert_eq!(result.pose, identity());
        assert_eq!(result.valid_source_pixels, w * h - 1);
        assert_eq!(result.working_bytes, required_bytes(&options, w, h).unwrap());
    }
}
