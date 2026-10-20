//! Complete rounded-pixel/Sobel fitting; raw final admission stays in the parent.
use super::*;

pub(super) fn required_bytes(
    options: &DenseRgbdOdometryOptions,
    width: usize,
    height: usize,
) -> VisionResult<usize> {
    // Validate the same controls first; the complete model adds a larger bound.
    options.required_working_bytes(width, height)?;
    let invalid =
        || VisionError::InvalidParameter("invalid or over-budget rounded-sobel buffers".into());
    let base = width.checked_mul(height).ok_or_else(invalid)?;
    let (mut w, mut h, mut pixels) = (width, height, 0usize);
    for _ in &options.iterations {
        if w < 3 || h < 3 {
            return Err(invalid());
        }
        pixels = pixels.checked_add(w.checked_mul(h).ok_or_else(invalid)?).ok_or_else(invalid)?;
        w /= 2;
        h /= 2;
    }
    let tiles = (height / TILE_ROWS + usize::from(height % TILE_ROWS != 0))
        .checked_mul(std::mem::size_of::<Normal>())
        .ok_or_else(invalid)?;
    let scratch = tiles.max(48 * std::mem::size_of::<f64>());
    pixels
        .checked_mul(32)
        .and_then(|n| base.checked_mul(64).and_then(|extra| n.checked_add(extra)))
        .and_then(|n| n.checked_add(scratch))
        .filter(|n| *n <= options.max_working_bytes)
        .ok_or_else(invalid)
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
            "rounded-sobel pair means require minimum count and finite positive means/gains".into(),
        ));
    }
    #[cfg(not(target_arch = "wasm32"))]
    if worker_count(source.width * source.height) > 1 {
        use rayon::prelude::*;
        pool().expect("worker count requires a pool").install(|| {
            source.color.par_iter_mut().zip(target.color.par_iter_mut()).for_each(|(a, b)| {
                *a *= gains[0];
                *b *= gains[1];
            });
        });
        return Ok(gains);
    }
    for (a, b) in source.color.iter_mut().zip(target.color.iter_mut()) {
        *a *= gains[0];
        *b *= gains[1];
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
        *c = (smooth.color[a] + smooth.color[a + 1] + smooth.color[b] + smooth.color[b + 1]) / 4.0;
        *d =
            (previous.depth[a] + previous.depth[a + 1] + previous.depth[b] + previous.depth[b + 1])
                / 4.0;
    }
}

fn downsample(previous: &Level) -> Level {
    let smooth = gaussian3(previous, false);
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
        let mut gradient = [0.0; 4];
        for (dy, wy) in [1.0, 2.0, 1.0].into_iter().enumerate() {
            for (dx, wx) in [1.0, 2.0, 1.0].into_iter().enumerate() {
                let sx = (x + dx).saturating_sub(1).min(level.width - 1);
                let sy = (y + dy).saturating_sub(1).min(level.height - 1);
                let index = sy * level.width + sx;
                let kx = (dx as f64 - 1.0) * wy;
                let ky = (dy as f64 - 1.0) * wx;
                gradient[0] += kx * level.color[index];
                gradient[1] += ky * level.color[index];
                gradient[2] += kx * level.depth[index];
                gradient[3] += ky * level.depth[index];
            }
        }
        *out = gradient.map(|v| v * 0.125);
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
            let Some((point, j)) = correspondence(source, target, k, pose, i, options) else {
                continue;
            };
            let g = gradients[j];
            let color = [target.color[j], g[0], g[1]];
            let depth = [target.depth[j], g[2], g[3]];
            let photo_error = color[0] - source.color[i];
            let depth_error = depth[0] - point.z;
            let jacobian = residual_jacobians(point, k, color, depth);
            normal.add(jacobian[0], photo_error, 1.0 - options.depth_weight);
            normal.add(jacobian[1], depth_error, options.depth_weight);
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
    let mut source = gaussian3(&a, true);
    let mut target = gaussian3(&b, true);
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
                    "nonfinite rounded-sobel normal system".into(),
                ));
            }
            let LeastSquaresResult::Solved(delta) = solve_linear_system(
                matrix.iter().map(|r| r.to_vec()).collect(),
                normal.rhs.to_vec(),
            ) else {
                return Err(VisionError::InvalidParameter(
                    "singular rounded-sobel normal system".into(),
                ));
            };
            if delta.iter().any(|v| !v.is_finite()) {
                return Err(VisionError::InvalidParameter("nonfinite rounded-sobel update".into()));
            }
            pose = update_pose(pose, &delta)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use spatialrust_image::Image;

    fn level(width: usize, height: usize) -> Level {
        Level {
            width,
            height,
            color: (0..width * height)
                .map(|i| (3 * (i % width) + 5 * (i / width) + 7) as f64 / 255.0)
                .collect(),
            depth: vec![2.0; width * height],
        }
    }

    fn identity() -> AbsolutePose {
        AbsolutePose::try_new(Mat3::<f64>::identity(), Vec3::new(0.0, 0.0, 0.0)).unwrap()
    }

    #[test]
    fn signed_rounding_nan_and_outer_border_rules_are_explicit() {
        for (value, expected) in [
            (-1.5, None),
            (-1.49, Some(0)),
            (-0.5, Some(0)),
            (0.49, Some(0)),
            (0.5, Some(1)),
            (7.0, Some(7)),
            (7.49, Some(7)),
            (7.5, None),
            (f64::NAN, None),
            (f64::INFINITY, None),
            (f64::NEG_INFINITY, None),
        ] {
            assert_eq!(rounded_coordinate(value, 8), expected);
        }
        let source = level(8, 8);
        let mut target = level(8, 8);
        target.depth[2 * 8 + 3] = f64::NAN;
        let options = DenseRgbdOdometryOptions::default();
        let k = [64.0, 80.0, 0.0, 0.0];
        assert!(correspondence(&source, &target, k, identity(), 2 * 8 + 2, &options).is_some());
        assert!(correspondence(&source, &target, k, identity(), 2 * 8 + 3, &options).is_none());
        assert!(correspondence(&source, &target, k, identity(), 2 * 8 + 7, &options).is_some());
    }

    #[test]
    fn floor_pyramid_affine_basis_and_nonfinite_depth_match_fixed_model() {
        let coarse = downsample(&level(8, 8));
        assert_eq!((coarse.width, coarse.height), (4, 4));
        assert!((coarse.color[5] - 27.0 / 255.0).abs() < 1e-14);
        let odd = downsample(&level(7, 5));
        assert_eq!((odd.width, odd.height), (3, 2));
        let mut hole = level(8, 8);
        hole.depth[2 * 8 + 2] = f64::NAN;
        assert!(!downsample(&hole).depth[5].is_finite());
        assert!(!gaussian3(&hole, true).depth[9].is_finite());
    }

    #[test]
    fn sobel_affine_border_and_missing_depth_surrogates_are_analytical() {
        let mut image = level(8, 8);
        let derivative = gradients(&image);
        for (observed, expected) in
            derivative[2 * 8 + 2].iter().zip([3.0 / 255.0, 5.0 / 255.0, 0.0, 0.0])
        {
            assert!((observed - expected).abs() < 1e-14);
        }
        assert!((derivative[0][0] - 1.5 / 255.0).abs() < 1e-14);
        assert!((derivative[0][1] - 2.5 / 255.0).abs() < 1e-14);
        image.depth[2 * 8 + 3] = f64::NAN;
        assert_eq!(&gradients(&image)[2 * 8 + 2][2..], &[0.0, 0.0]);
        let point = Vec3::new(0.0, 0.0, 2.0);
        let actual = residual_jacobians(
            point,
            [64.0, 80.0, 0.0, 0.0],
            [0.0, 0.002, -0.001],
            [2.0, 0.003, 0.004],
        );
        let expected =
            [[0.08, 0.128, 0.0, 0.064, -0.04, 0.0], [-0.32, 0.192, 0.0, 0.096, 0.16, -1.0]];
        for (a, b) in actual.iter().flatten().zip(expected.iter().flatten()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn preflight_exact_boundary_overflow_odd_levels_and_mixed_policy_reject() {
        let options = DenseRgbdOdometryOptions::default();
        let required = required_bytes(&options, 640, 480).unwrap();
        let pixels = 640 * 480 + 320 * 240 + 160 * 120;
        assert_eq!(required, 32 * pixels + 64 * 640 * 480 + 30 * std::mem::size_of::<Normal>());
        assert_eq!(
            required_bytes(
                &DenseRgbdOdometryOptions { max_working_bytes: required, ..options.clone() },
                640,
                480
            )
            .unwrap(),
            required
        );
        assert!(required_bytes(
            &DenseRgbdOdometryOptions { max_working_bytes: required - 1, ..options.clone() },
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
        assert_eq!(required_bytes(&tiny, 3, 3).unwrap(), 96 * 9 + 48 * std::mem::size_of::<f64>());

        assert!(required_bytes(
            &DenseRgbdOdometryOptions { iterations: vec![1, 1], ..options.clone() },
            7,
            5
        )
        .is_err());
        assert!(options
            .required_working_bytes_with_model(
                640,
                480,
                DenseRgbdPreprocessing::Gaussian3,
                DenseRgbdFittingModel::RoundedSobel
            )
            .is_err());
    }

    #[test]
    fn rejected_pair_means_and_count_do_not_mutate_inputs() {
        let mut a = level(8, 8);
        let mut b = level(8, 8);
        a.color.fill(0.0);
        let before = b.color.clone();
        assert!(normalize(
            &mut a,
            &mut b,
            [64.0, 80.0, 0.0, 0.0],
            identity(),
            &DenseRgbdOdometryOptions::default()
        )
        .is_err());
        assert!(a.color.iter().all(|v| *v == 0.0));
        assert_eq!(b.color, before);
        a = level(8, 8);
        let original = a.color.clone();
        let options = DenseRgbdOdometryOptions { min_correspondences: 65, ..Default::default() };
        assert!(normalize(&mut a, &mut b, [64.0, 80.0, 0.0, 0.0], identity(), &options).is_err());
        assert_eq!(a.color, original);
        assert_eq!(b.color, before);
    }

    #[test]
    fn parallel_gradients_downsample_and_ordered_normals_equal_serial_bits() {
        let image = level(257, 145);
        let mut expected = vec![[0.0; 4]; image.color.len()];
        gradient_tile(&image, 0, &mut expected);
        assert_eq!(gradients(&image), expected);
        let smooth = gaussian3(&image, false);
        let actual = downsample(&image);
        let mut color = vec![0.0; actual.color.len()];
        let mut depth = vec![0.0; actual.depth.len()];
        downsample_tile(&image, &smooth, actual.width, 0, &mut color, &mut depth);
        assert_eq!(actual.color, color);
        assert_eq!(actual.depth, depth);
        let k = [64.0, 80.0, 128.0, 72.0];
        let options = DenseRgbdOdometryOptions::default();
        let actual = linearize(&image, &image, &expected, k, identity(), &options);
        let mut serial = Normal::default();
        for block in 0..image.height.div_ceil(TILE_ROWS) {
            serial.merge(tile(&image, &image, &expected, k, identity(), &options, block));
        }
        assert_eq!(actual.hessian, serial.hessian);
        assert_eq!(actual.rhs, serial.rhs);
        assert_eq!(
            (actual.valid, actual.count, actual.photo_sse, actual.depth_sse),
            (serial.valid, serial.count, serial.photo_sse, serial.depth_sse)
        );
    }

    #[test]
    fn identity_executes_entire_schedule_and_bilinear_delegation_is_exact() {
        let (width, height) = (96, 72);
        let gray = Image::<u8, 1>::try_new(
            width,
            height,
            (0..width * height)
                .map(|i| {
                    let (x, y) = ((i % width) as f64, (i / width) as f64);
                    (128.0
                        + 45.0 * (0.26 * x).sin()
                        + 40.0 * (0.19 * y).cos()
                        + 25.0 * (0.13 * (x + y)).sin())
                    .round() as u8
                })
                .collect(),
        )
        .unwrap();
        let depth = Image::<f32, 1>::try_new(width, height, vec![2.0; width * height]).unwrap();
        let frame = DenseRgbdFrame::try_new(gray.view(), depth.view()).unwrap();
        let camera = CameraMatrix3::try_from_pinhole(64.0, 64.0, 48.0, 36.0).unwrap();
        let options = DenseRgbdOdometryOptions::default();
        let result = estimate_dense_rgbd_odometry_with_model(
            frame,
            frame,
            camera,
            identity(),
            &options,
            DenseRgbdPreprocessing::Raw,
            DenseRgbdFittingModel::RoundedSobel,
        )
        .unwrap();
        assert_eq!(result.iterations, 35);
        assert_eq!(result.pose, identity());
        assert_eq!(result.valid_source_pixels, width * height);
        assert_eq!(result.working_bytes, required_bytes(&options, width, height).unwrap());
        for policy in [
            DenseRgbdPreprocessing::Raw,
            DenseRgbdPreprocessing::Gaussian3,
            DenseRgbdPreprocessing::Gaussian3GrayOnly,
            DenseRgbdPreprocessing::Gaussian3GrayOnlyPairMean,
        ] {
            let before = estimate_dense_rgbd_odometry_with_preprocessing(
                frame,
                frame,
                camera,
                identity(),
                &options,
                policy,
            )
            .unwrap();
            let after = estimate_dense_rgbd_odometry_with_model(
                frame,
                frame,
                camera,
                identity(),
                &options,
                policy,
                DenseRgbdFittingModel::Bilinear,
            )
            .unwrap();
            assert_eq!(before, after);
        }
    }
}
