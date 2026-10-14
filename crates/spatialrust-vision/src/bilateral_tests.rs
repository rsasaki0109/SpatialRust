use super::*;
fn previous<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    diameter: usize,
    sigma_color: f64,
    sigma_space: f64,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    if diameter == 0 {
        return Err(VisionError::InvalidParameter("bilateral diameter must be non-zero".into()));
    }
    validate_positive_finite(sigma_color, "sigma_color")?;
    validate_positive_finite(sigma_space, "sigma_space")?;
    let radius = (diameter / 2) as isize;
    let color_factor = -0.5 / (sigma_color * sigma_color);
    let space_factor = -0.5 / (sigma_space * sigma_space);
    let mut output = Vec::with_capacity(input.width() * input.height() * CHANNELS);
    for y in 0..input.height() {
        for x in 0..input.width() {
            let center = fetch(input, x as isize, y as isize, border);
            let mut sums = [0.0; CHANNELS];
            let mut total_weight = 0.0;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let dx_f64 = dx as f64;
                    let dy_f64 = dy as f64;
                    let spatial_distance = dx_f64.mul_add(dx_f64, dy_f64 * dy_f64);
                    if spatial_distance > (radius as f64) * (radius as f64) {
                        continue;
                    }
                    let pixel = fetch(input, x as isize + dx, y as isize + dy, border);
                    let color_distance = (0..CHANNELS)
                        .map(|channel| pixel[channel].to_f64() - center[channel].to_f64())
                        .map(f64::abs)
                        .sum::<f64>();
                    let weight = ((color_distance * color_distance)
                        .mul_add(color_factor, spatial_distance * space_factor))
                    .exp();
                    for channel in 0..CHANNELS {
                        sums[channel] += pixel[channel].to_f64() * weight;
                    }
                    total_weight += weight;
                }
            }
            output.extend(sums.map(|sum| T::from_f64(sum / total_weight)));
        }
    }
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}
fn compare<T: PixelComponent, const C: usize>(
    v: ImageView<'_, T, C>,
    d: usize,
    sc: f64,
    ss: f64,
    b: BorderMode<T, C>,
    workers: usize,
    bits: fn(T) -> u64,
) {
    let old = previous(v, d, sc, ss, b).unwrap();
    let new = run_with_workers(v, d, sc, ss, b, Some(workers)).unwrap();
    assert_eq!(old.metadata(), new.metadata());
    assert_eq!(
        old.as_slice().iter().copied().map(bits).collect::<Vec<_>>(),
        new.as_slice().iter().copied().map(bits).collect::<Vec<_>>()
    );
}
fn fixture<const C: usize>(w: usize, h: usize, p: usize) -> Image<u8, C> {
    Image::try_new(
        w,
        h,
        (0..w * h * C)
            .map(|i| match p {
                0 => {
                    (i as u32).wrapping_mul(747_796_405).wrapping_add(2_891_336_453).rotate_left(13)
                        as u8
                }
                1 => 0,
                2 => 255,
                _ => {
                    if i % 23 == 0 {
                        255
                    } else {
                        17
                    }
                }
            })
            .collect(),
    )
    .unwrap()
}
fn small<const C: usize>() {
    for (w, h) in [(0, 0), (0, 3), (1, 1), (1, 13), (17, 11), (33, 17)] {
        for p in 0..4 {
            let im = fixture::<C>(w, h, p);
            for d in [1, 2, 3, 5, 7] {
                for (sc, ss) in [(1.0, 1.0), (40.0, 3.0), (f64::MAX, f64::MAX), (1e-200, 1e-200)] {
                    for b in [
                        BorderMode::Replicate,
                        BorderMode::Reflect,
                        BorderMode::Reflect101,
                        BorderMode::Wrap,
                        BorderMode::Constant([43; C]),
                    ] {
                        compare(im.view(), d, sc, ss, b, 6, u64::from);
                    }
                }
            }
        }
    }
}
#[test]
fn bilateral_frozen_small_even_apertures_borders_and_extreme_sigmas() {
    small::<1>();
    small::<2>();
    small::<3>();
    small::<4>();
}
#[test]
fn bilateral_parallel_pools_roi_and_float_bits_match_frozen() {
    for (w, h) in [(511, 513), (512, 512), (513, 512), (1, 262145), (262145, 1)] {
        let im = fixture::<1>(w, h, 0);
        let old = previous(im.view(), 5, 40.0, 3.0, BorderMode::Reflect101).unwrap();
        for n in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
            assert_eq!(
                old,
                pool.install(|| run(im.view(), 5, 40.0, 3.0, BorderMode::Reflect101).unwrap())
            );
        }
    }
    let vals = [
        -0.0,
        0.0,
        -1.0,
        1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff8_0000_0000_0013),
        f64::from_bits(1),
    ];
    let im =
        Image::<f64, 1>::try_new(513, 512, (0..513 * 512).map(|i| vals[i % vals.len()]).collect())
            .unwrap();
    for n in [1, 2, 6] {
        compare(im.view(), 3, 40.0, 3.0, BorderMode::Wrap, n, f64::to_bits);
    }
    let w = 257;
    let h = 257;
    let stride = w * 4 + 19;
    let packed = fixture::<4>(w, h, 3);
    let mut data = vec![211; stride * h];
    for (row, src) in data.chunks_mut(stride).zip(packed.as_slice().chunks(w * 4)) {
        row[..w * 4].copy_from_slice(src);
    }
    let meta = spatialrust_image::ImageMetadata {
        color_space: spatialrust_image::ColorSpace::Bgra,
        color_range: spatialrust_image::ColorRange::Limited,
        alpha_mode: spatialrust_image::AlphaMode::Straight,
    };
    let view = ImageView::<u8, 4>::new_with_metadata(w, h, stride, &data, meta).unwrap();
    for v in [view, view.subview(spatialrust_image::ImageRegion::new(1, 1, 255, 255)).unwrap()] {
        for d in [2, 3, 5, 7] {
            for n in [1, 2, 6] {
                compare(v, d, 40.0, 3.0, BorderMode::Constant([19, 73, 129, 251]), n, u64::from);
            }
        }
    }
    let wide = Image::<u16, 4>::try_new(w, h, (0..w * h * 4).map(|i| (i * 7919) as u16).collect())
        .unwrap();
    compare(wide.view(), 3, 40.0, 3.0, BorderMode::Reflect101, 6, u64::from);
    let f = Image::<f32, 4>::try_new(
        w,
        h,
        wide.as_slice().iter().map(|&v| f32::from(v) / 255.0).collect(),
    )
    .unwrap();
    compare(f.view(), 3, 40.0, 3.0, BorderMode::Reflect101, 6, |v| u64::from(v.to_bits()));
}
#[test]
fn bilateral_errors_and_analytic_disk_l1_color_match() {
    for (w, h) in [(0, 0), (1, 1)] {
        let im = fixture::<1>(w, h, 0);
        for (d, sc, ss) in [
            (0, 1.0, 1.0),
            (3, 0.0, 1.0),
            (3, 1.0, 0.0),
            (3, f64::NAN, 1.0),
            (3, 1.0, f64::INFINITY),
        ] {
            assert_eq!(
                previous(im.view(), d, sc, ss, BorderMode::Replicate).err().unwrap().to_string(),
                run(im.view(), d, sc, ss, BorderMode::Replicate).err().unwrap().to_string()
            );
        }
    }
    let im = Image::<u8, 1>::try_new(3, 3, vec![255, 0, 255, 0, 0, 0, 255, 0, 255]).unwrap();
    assert_eq!(run(im.view(), 3, f64::MAX, f64::MAX, BorderMode::Replicate).unwrap()[(1, 1)][0], 0);
    assert_eq!(
        run(im.view(), 2, 40.0, 3.0, BorderMode::Replicate).unwrap(),
        run(im.view(), 3, 40.0, 3.0, BorderMode::Replicate).unwrap()
    );
}

#[path = "bilateral_benchmark.rs"]
mod benchmark;
