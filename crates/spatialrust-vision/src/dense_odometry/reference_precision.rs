//! Explicit f32-intermediate / proper left Euler fitting; raw final admission stays in the parent.
use super::*;

pub(super) fn required_bytes(
    options: &DenseRgbdOdometryOptions,
    width: usize,
    height: usize,
) -> VisionResult<usize> {
    // Validate the same controls first; the complete model adds a larger bound.
    options.required_working_bytes(width, height)?;
    let invalid = || {
        VisionError::InvalidParameter("invalid or over-budget reference-precision buffers".into())
    };
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
            "reference-precision pair means require minimum count and finite positive means/gains"
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
            let photo_scale = (1.0 - options.depth_weight).sqrt();
            let depth_scale = options.depth_weight.sqrt();
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
                    "nonfinite reference-precision normal system".into(),
                ));
            }
            let LeastSquaresResult::Solved(delta) = solve_linear_system(
                matrix.iter().map(|r| r.to_vec()).collect(),
                normal.rhs.to_vec(),
            ) else {
                return Err(VisionError::InvalidParameter(
                    "singular reference-precision normal system".into(),
                ));
            };
            if delta.iter().any(|v| !v.is_finite()) {
                return Err(VisionError::InvalidParameter(
                    "nonfinite reference-precision update".into(),
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

    // Independent two-plane implementation: unlike production this materializes
    // the complete horizontal pass and then visits columns.
    fn oracle(a: &[f64], w: usize, h: usize, kx: [f32; 3], ky: [f32; 3]) -> Vec<f64> {
        let mut horizontal = vec![0f32; a.len()];
        for y in 0..h {
            for x in 0..w {
                let indices = [x.saturating_sub(1), x, (x + 1).min(w - 1)];
                horizontal[y * w + x] = (f64::from(a[y * w + indices[0]] as f32 * kx[0])
                    + f64::from(a[y * w + indices[1]] as f32 * kx[1])
                    + f64::from(a[y * w + indices[2]] as f32 * kx[2]))
                    as f32;
            }
        }
        let mut result = vec![0.; a.len()];
        for x in 0..w {
            for y in 0..h {
                let indices = [y.saturating_sub(1), y, (y + 1).min(h - 1)];
                result[y * w + x] = f64::from(
                    (f64::from(horizontal[indices[0] * w + x] * ky[0])
                        + f64::from(horizontal[indices[1] * w + x] * ky[1])
                        + f64::from(horizontal[indices[2] * w + x] * ky[2]))
                        as f32,
                );
            }
        }
        result
    }

    #[test]
    fn f32_operators_equal_independent_passes_at_borders_and_holes() {
        let mut a = level(31, 25);
        a.depth[12 * 31 + 15] = f64::NAN;
        for values in [&a.color, &a.depth] {
            for (kx, ky) in [
                ([0.25, 0.5, 0.25], [0.25, 0.5, 0.25]),
                ([-1., 0., 1.], [1., 2., 1.]),
                ([1., 2., 1.], [-1., 0., 1.]),
            ] {
                let expected = oracle(values, a.width, a.height, kx, ky);
                for (i, b) in expected.iter().enumerate() {
                    let actual =
                        separable(values, a.width, a.height, i % a.width, i / a.width, kx, ky);
                    assert!(actual.to_bits() == b.to_bits() || (actual.is_nan() && b.is_nan()));
                }
            }
        }
        assert!(filter(&a, true).depth[12 * 31 + 14].is_nan());
        assert_eq!(&gradients(&a)[12 * 31 + 15][2..], &[0., 0.]);
    }

    #[test]
    fn u8_normalization_and_left_associated_f32_pyramid_are_explicit() {
        for v in 0u8..=255 {
            assert_eq!((f64::from(v) / 255.) as f32, f32::from(v) / 255.);
        }
        assert_eq!(mean4(&[16777216., 1., -16777216., 1.], 0, 2), 0.25);
        let a = level(31, 25);
        let b = downsample(&a);
        assert_eq!((b.width, b.height), (15, 12));
        let smooth = oracle(&a.color, 31, 25, [0.25, 0.5, 0.25], [0.25, 0.5, 0.25]);
        for y in 0..12 {
            for x in 0..15 {
                let indices = [
                    2 * y * 31 + 2 * x,
                    2 * y * 31 + 2 * x + 1,
                    (2 * y + 1) * 31 + 2 * x,
                    (2 * y + 1) * 31 + 2 * x + 1,
                ];
                for (values, observed) in [(&smooth, &b.color), (&a.depth, &b.depth)] {
                    let expected =
                        indices.into_iter().map(|i| values[i] as f32).fold(0f32, |s, v| s + v) / 4.;
                    assert_eq!(observed[y * 15 + x], f64::from(expected));
                }
            }
        }
    }

    #[test]
    fn proper_left_euler_action_matches_independent_axis_rotations() {
        let old = super::super::update_pose(identity(), &[0.2, -0.1, 0.3, 1., 2., 3.]).unwrap();
        let delta = [0.13, -0.21, 0.08, 0.4, -0.7, 0.9];
        let actual = euler_update(old, &delta).unwrap();
        for p in [Vec3::new(0., 0., 0.), Vec3::new(1., -2., 3.), Vec3::new(-0.2, 0.4, 2.)] {
            let p = old.transform_point(p);
            let (sx, cx) = delta[0].sin_cos();
            let (sy, cy) = delta[1].sin_cos();
            let (sz, cz) = delta[2].sin_cos();
            let q = Vec3::new(p.x, cx * p.y - sx * p.z, sx * p.y + cx * p.z);
            let r = Vec3::new(cy * q.x + sy * q.z, q.y, -sy * q.x + cy * q.z);
            let expected = Vec3::new(
                cz * r.x - sz * r.y + delta[3],
                sz * r.x + cz * r.y + delta[4],
                r.z + delta[5],
            );
            // Recover the original point for the actual composed transformation.
            let inverse = old.rotation().transpose().mul_vec3(p - old.translation());
            assert!((actual.transform_point(inverse) - expected).length() < 1e-12);
        }
        assert!(euler_update(old, &[f64::NAN, 0., 0., 0., 0., 0.]).is_err());
        assert!(euler_update(old, &[0., 0., 0., f64::INFINITY, 0., 0.]).is_err());
    }

    #[test]
    fn signed_lookup_and_invalid_means_preserve_original_admission() {
        for (v, expected) in [
            (-1.5, None),
            (-1.49, Some(0)),
            (-0.5, Some(0)),
            (7.49, Some(7)),
            (7.5, None),
            (f64::NAN, None),
            (f64::INFINITY, None),
        ] {
            assert_eq!(rounded_coordinate(v, 8), expected);
        }
        let mut a = level(8, 8);
        let mut b = level(8, 8);
        a.color.fill(0.);
        let before = b.color.clone();
        assert!(
            normalize(&mut a, &mut b, [64., 80., 0., 0.], identity(), &Default::default()).is_err()
        );
        assert_eq!(b.color, before);
        assert!(a.color.iter().all(|v| *v == 0.));
        b.depth[63] = f64::NAN;
        assert!(correspondence(&a, &b, [64., 80., 0., 0.], identity(), 63, &Default::default())
            .is_none());
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
                    DenseRgbdFittingModel::ReferencePrecision
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
    fn parallel_filters_gradients_pyramid_and_normals_equal_serial_bits() {
        let image = level(257, 145);
        let actual = filter(&image, true);
        let mut color = vec![0.; image.color.len()];
        let mut depth = vec![0.; image.depth.len()];
        filter_tile(&image, 0, &mut color, &mut depth, true);
        assert_eq!(actual.color, color);
        assert_eq!(actual.depth, depth);
        let gradient = gradients(&image);
        let mut serial = vec![[0.; 4]; gradient.len()];
        gradient_tile(&image, 0, &mut serial);
        assert_eq!(gradient, serial);
        let coarse = downsample(&image);
        color.resize(coarse.color.len(), 0.);
        depth.resize(coarse.depth.len(), 0.);
        downsample_tile(&image, &filter(&image, false), coarse.width, 0, &mut color, &mut depth);
        assert_eq!(coarse.color, color);
        assert_eq!(coarse.depth, depth);
        let k = [64., 80., 128., 72.];
        let opts = DenseRgbdOdometryOptions::default();
        let actual = linearize(&image, &image, &gradient, k, identity(), &opts);
        let mut serial = Normal::default();
        for b in 0..image.height.div_ceil(TILE_ROWS) {
            serial.merge(tile(&image, &image, &gradient, k, identity(), &opts, b));
        }
        assert_eq!(actual.hessian, serial.hessian);
        assert_eq!(actual.rhs, serial.rhs);
        assert_eq!(actual.count, serial.count);
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
            DenseRgbdFittingModel::ReferencePrecision,
        )
        .unwrap();
        assert_eq!(result.iterations, 35);
        assert_eq!(result.pose, identity());
        assert_eq!(result.valid_source_pixels, w * h - 1);
        assert_eq!(result.working_bytes, required_bytes(&options, w, h).unwrap());
    }
}
