//! Frozen full-sort median from merged PR #192, plus independent numeric cases.
use super::*;
use spatialrust_image::{ImageMetadata, ImageRegion};

fn previous<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    kernel_size: usize,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    validate_odd_size(kernel_size, "median")?;
    let radius = (kernel_size / 2) as isize;
    let area = kernel_size
        .checked_mul(kernel_size)
        .ok_or_else(|| VisionError::InvalidParameter("median kernel area overflows".into()))?;
    let mut samples = (0..CHANNELS).map(|_| Vec::<f64>::with_capacity(area)).collect::<Vec<_>>();
    let mut output = Vec::with_capacity(input.width() * input.height() * CHANNELS);
    for y in 0..input.height() {
        for x in 0..input.width() {
            for channel in &mut samples {
                channel.clear();
            }
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let pixel = fetch(input, x as isize + dx, y as isize + dy, border);
                    for channel in 0..CHANNELS {
                        samples[channel].push(pixel[channel].to_f64());
                    }
                }
            }
            for channel in &mut samples {
                channel.sort_by(f64::total_cmp);
                output.push(T::from_f64(channel[area / 2]));
            }
        }
    }
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

fn compare<T: PixelComponent, const C: usize>(
    input: ImageView<'_, T, C>,
    kernel: usize,
    border: BorderMode<T, C>,
    available: usize,
    bits: fn(T) -> u64,
) {
    let old = previous(input, kernel, border).unwrap();
    let new = run_with_workers(input, kernel, border, Some(available)).unwrap();
    assert_eq!(old.metadata(), new.metadata());
    assert_eq!((old.width(), old.height()), (new.width(), new.height()));
    assert_eq!(
        old.as_slice().iter().copied().map(bits).collect::<Vec<_>>(),
        new.as_slice().iter().copied().map(bits).collect::<Vec<_>>()
    );
}

fn bytes<const C: usize>(w: usize, h: usize, profile: usize) -> Image<u8, C> {
    Image::try_new(
        w,
        h,
        (0..w * h * C)
            .map(|i| match profile {
                0 => {
                    (i as u32).wrapping_mul(747_796_405).wrapping_add(2_891_336_453).rotate_left(13)
                        as u8
                }
                1 => 0,
                2 => 255,
                3 => {
                    if (i / C % w + i / (C * w)) % 2 == 0 {
                        255
                    } else {
                        0
                    }
                }
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

fn channels<const C: usize>() -> usize {
    let mut checked = 0;
    for (w, h) in [(0, 0), (0, 3), (3, 0), (1, 1), (1, 9), (9, 1), (2, 3), (17, 11), (65, 33)] {
        for profile in 0..5 {
            let image = bytes::<C>(w, h, profile);
            for kernel in [1, 3, 5, 7, 9] {
                for b in [
                    BorderMode::Replicate,
                    BorderMode::Reflect,
                    BorderMode::Reflect101,
                    BorderMode::Wrap,
                    BorderMode::Constant([43; C]),
                ] {
                    for available in [0, 1, 2, 6] {
                        compare(image.view(), kernel, b, available, u64::from);
                        checked += w * h * C;
                    }
                }
            }
        }
    }
    checked
}

#[test]
fn median_matches_frozen_sort_channels_borders_empty_and_small_images() {
    let checked = channels::<1>() + channels::<2>() + channels::<3>() + channels::<4>();
    println!("CPU_MEDIAN_SMALL exact_components={checked} cases=18000");
}

#[test]
fn median_preserves_float_total_order_zero_nan_payloads_and_wide_dtype() {
    let values = [
        f64::NEG_INFINITY,
        -f64::MAX,
        -1.0,
        -f64::from_bits(1),
        -0.0,
        0.0,
        f64::from_bits(1),
        1.0,
        f64::MAX,
        f64::INFINITY,
        f64::from_bits(0x7ff8_0000_0000_0013),
        f64::from_bits(0xfff8_0000_0000_0017),
        f64::from_bits(0x7ff0_0000_0000_0001),
    ];
    let image = Image::<f64, 4>::try_new(
        17,
        11,
        (0..17 * 11 * 4).map(|i| values[(i * 7 + i / 68) % values.len()]).collect(),
    )
    .unwrap();
    let image32 =
        Image::<f32, 4>::try_new(17, 11, image.as_slice().iter().map(|&v| v as f32).collect())
            .unwrap();
    let wide =
        Image::<u16, 4>::try_new(17, 11, (0..17 * 11 * 4).map(|i| (i * 7919) as u16).collect())
            .unwrap();
    for kernel in [1, 3, 5, 7, 9, 17] {
        for workers in [1, 2, 6] {
            for b in [
                BorderMode::Replicate,
                BorderMode::Reflect,
                BorderMode::Reflect101,
                BorderMode::Wrap,
                BorderMode::Constant([-0.0; 4]),
            ] {
                compare(image.view(), kernel, b, workers, f64::to_bits);
            }
            compare(image32.view(), kernel, BorderMode::Reflect101, workers, |v| {
                u64::from(v.to_bits())
            });
            compare(wide.view(), kernel, BorderMode::Constant([u16::MAX; 4]), workers, u64::from);
        }
    }
    let signed =
        Image::<f64, 1>::try_new(3, 3, vec![-0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();
    assert_eq!(
        run(signed.view(), 3, BorderMode::Replicate).unwrap()[(1, 1)][0].to_bits(),
        (-0.0_f64).to_bits()
    );
    let analytic = Image::<u8, 1>::try_new(3, 3, vec![9, 8, 7, 6, 5, 4, 3, 2, 1]).unwrap();
    assert_eq!(run(analytic.view(), 3, BorderMode::Replicate).unwrap()[(1, 1)][0], 5);
}

#[test]
fn median_parallel_boundaries_pools_and_strided_roi_are_bit_exact() {
    let pools: [rayon::ThreadPool; 3] =
        [1, 2, 4].map(|n| rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap());
    let mut checked = 0;
    for (w, h) in [(511, 513), (512, 512), (513, 512), (1, 262_145), (262_145, 1)] {
        let image = bytes::<1>(w, h, 0);
        let old = previous(image.view(), 3, BorderMode::Reflect101).unwrap();
        for pool in &pools {
            let new = pool.install(|| run(image.view(), 3, BorderMode::Reflect101).unwrap());
            assert_eq!(old, new);
            checked += w * h;
        }
    }
    let w = 257;
    let h = 257;
    let stride = w * 4 + 19;
    let image = bytes::<4>(w, h, 4);
    let mut data = vec![211; stride * h];
    for (row, input) in data.chunks_mut(stride).zip(image.as_slice().chunks(w * 4)) {
        row[..w * 4].copy_from_slice(input);
    }
    let metadata = ImageMetadata {
        color_space: spatialrust_image::ColorSpace::Bgra,
        color_range: spatialrust_image::ColorRange::Limited,
        alpha_mode: spatialrust_image::AlphaMode::Straight,
    };
    let view = ImageView::<u8, 4>::new_with_metadata(w, h, stride, &data, metadata).unwrap();
    for input in [view, view.subview(ImageRegion::new(1, 1, 255, 255)).unwrap()] {
        for kernel in [1, 3, 5, 7] {
            for available in [1, 2, 6] {
                compare(
                    input,
                    kernel,
                    BorderMode::Constant([19, 73, 129, 251]),
                    available,
                    u64::from,
                );
                checked += input.width() * input.height() * 4;
            }
        }
    }
    // Float support and exact total ordering also cross the parallel cutoff.
    let values = [
        -0.0,
        0.0,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::from_bits(0x7ff8_0000_0000_0013),
        f64::from_bits(0xfff8_0000_0000_0017),
        f64::from_bits(1),
        -1.0,
        1.0,
    ];
    let float = Image::<f64, 1>::try_new(
        513,
        512,
        (0..513 * 512).map(|i| values[i % values.len()]).collect(),
    )
    .unwrap();
    for available in [1, 2, 6] {
        compare(float.view(), 3, BorderMode::Wrap, available, f64::to_bits);
        checked += 513 * 512;
    }
    for components in [0, 262_143, 262_144, 262_145, usize::MAX] {
        for height in [0, 1, 2, 257] {
            for available in [0, 1, 2, 6, usize::MAX] {
                assert_eq!(
                    worker_count(components, height, available),
                    if components >= 262_144 && height > 1 {
                        available.max(1).min(height)
                    } else {
                        1
                    }
                );
            }
        }
    }
    println!("CPU_MEDIAN_PARALLEL exact_components={checked}");
}

#[test]
fn median_rejects_bad_kernels_without_losing_empty_image_validation() {
    for (w, h) in [(0, 0), (1, 1)] {
        let image = bytes::<1>(w, h, 0);
        for kernel in [0, 2, 4, usize::MAX] {
            assert_eq!(
                previous(image.view(), kernel, BorderMode::Replicate).err().unwrap().to_string(),
                run(image.view(), kernel, BorderMode::Replicate).err().unwrap().to_string()
            );
        }
    }
    let empty = bytes::<1>(0, 3, 0);
    assert_eq!(run(empty.view(), 1, BorderMode::Replicate).unwrap().as_slice(), &[]);
}

#[path = "median_benchmark.rs"]
mod benchmark;
