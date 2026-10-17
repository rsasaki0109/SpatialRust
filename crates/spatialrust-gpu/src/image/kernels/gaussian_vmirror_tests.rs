use super::*;

#[path = "gaussian_vmirror_reference.rs"]
mod old;

fn row(y: i32, height: usize, border: GpuGaussianBorder) -> Option<usize> {
    let n = height as i32;
    if (0..n).contains(&y) {
        return Some(y as usize);
    }
    match border {
        GpuGaussianBorder::Constant(_) => None,
        GpuGaussianBorder::Replicate => Some(y.clamp(0, n - 1) as usize),
        _ if n == 1 => Some(0),
        GpuGaussianBorder::Wrap => Some(y.rem_euclid(n) as usize),
        GpuGaussianBorder::Reflect => {
            let y = y.rem_euclid(2 * n);
            Some(if y < n { y } else { 2 * n - 1 - y } as usize)
        }
        GpuGaussianBorder::Reflect101 => {
            let y = y.rem_euclid(2 * (n - 1));
            Some(if y < n { y } else { 2 * (n - 1) - y } as usize)
        }
    }
}

fn raw_vertical<const C: usize>(
    r: &WgpuRuntime,
    prior: &old::ParentGaussianPipelines,
    w: usize,
    h: usize,
    options: GpuGaussianBlurOptions,
    hot: bool,
) -> usize {
    let mut p = plan(w as u32, h as u32, C as u32, options, &r.device().limits()).unwrap();

    let mass: u64 = p.params.weights_y.iter().flatten().map(|&w| u64::from(w)).sum();
    assert_eq!(mass, 1u64 << p.params.bits);
    assert!(65535 * mass < (1 << 31));
    assert!(127 * mass < (1 << 22));
    p.params.groups = 1;
    let words = w * h * if p.packed { C.div_ceil(2) } else { C };
    let edges =
        [0, 1, 16383, 16384, 16385, 32767, 32768, 32769, 65535, 65536, 65537, 8_355_840, 8_388_607];
    let mut sums: Vec<u32> = (0..words)
        .map(|i| {
            if hot {
                if p.packed {
                    u32::MAX
                } else {
                    (1u32 << (p.params.bits + 8)) - 1
                }
            } else if p.packed {
                (i as u32).wrapping_mul(747_796_405).wrapping_add(2_891_336_453).rotate_left(13)
            } else if i % 2 == 0 {
                edges[(i / 2) % edges.len()] & ((1u32 << (p.params.bits + 8)) - 1)
            } else {
                (i as u32).wrapping_mul(747_796_405).wrapping_add(2_891_336_453).rotate_left(13)
                    & ((1u32 << (p.params.bits + 8)) - 1)
            }
        })
        .collect();
    // Probe full u16 or 23-bit bounds, including dirty inactive packed halves.
    sums.extend_from_slice(&[0xdead_beef; 8]);
    let input = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gaussian-vmirror-raw-input"),
        contents: bytemuck::cast_slice(&sums),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let uniform = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gaussian-vmirror-raw-params"),
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let cache = GaussianPipelines::new();
    let outputs: [_; 2] = std::array::from_fn(|_| {
        create_texture(r, w as u32, h as u32, "gaussian-vmirror-raw-output")
    });
    let mut encoder = r.device().create_command_encoder(&Default::default());
    for (pipeline, output) in [
        (&prior.pipelines(r.device(), p.packed)[1], &outputs[0]),
        (&cache.pipelines(r.device(), p.packed)[1], &outputs[1]),
    ] {
        let group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gaussian-vmirror-raw-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &output.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry { binding: 2, resource: input.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    r.queue().submit(Some(encoder.finish()));
    let constant =
        if let GpuGaussianBorder::Constant(pixel) = options.border { pixel } else { [0; 4] };
    let mut expected = vec![0; w * h * 4];
    for pixel in 0..w * h {
        for c in 0..C {
            let total: u64 = (0..options.kernel_height as usize)
                .map(|tap| {
                    let horizontal = row(
                        (pixel / w) as i32 + tap as i32 - (options.kernel_height / 2) as i32,
                        h,
                        options.border,
                    )
                    .map_or(u32::from(constant[c]) << p.params.bits, |sy| {
                        if p.packed {
                            let word = sums[(sy * w + pixel % w) * C.div_ceil(2) + c / 2];
                            (word >> (16 * (c % 2))) & 65535
                        } else {
                            sums[(sy * w + pixel % w) * C + c]
                        }
                    });
                    u64::from(horizontal) * u64::from(p.params.weights_y[tap / 4][tap % 4])
                })
                .sum();
            expected[pixel * 4 + c] =
                ((total + (1u64 << (2 * p.params.bits - 1))) >> (2 * p.params.bits)).min(255) as u8;
        }
    }
    for texture in outputs {
        let mut output = GpuImage::from_parts(
            r,
            w as u32,
            h as u32,
            4,
            texture,
            Default::default(),
            Default::default(),
        )
        .unwrap();
        assert_eq!(output.readback_u8::<4>(r).unwrap().as_slice(), expected);
        output.recycle(r);
    }
    w * h * 8
}

#[test]
fn gaussian_vmirror_raw_vertical_matches_parent_and_u64_all_layouts() {
    let r = WgpuRuntime::new_headless().unwrap();
    let prior = old::ParentGaussianPipelines::new();
    let mut bytes = 0;
    let mut cases = 0;
    for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 3), (17, 19), (257, 259)] {
        for (kw, kh) in [(3, 3), (3, 5), (7, 3), (7, 5), (7, 7)] {
            for border in [
                GpuGaussianBorder::Replicate,
                GpuGaussianBorder::Reflect,
                GpuGaussianBorder::Reflect101,
                GpuGaussianBorder::Wrap,
                GpuGaussianBorder::Constant([19, 73, 129, 251]),
            ] {
                for sigma in [0.17, 7.0] {
                    let options = GpuGaussianBlurOptions {
                        kernel_width: kw,
                        kernel_height: kh,
                        sigma_y: sigma,
                        border,
                        ..Default::default()
                    };
                    bytes += raw_vertical::<1>(&r, &prior, w, h, options, false)
                        + raw_vertical::<2>(&r, &prior, w, h, options, false)
                        + raw_vertical::<3>(&r, &prior, w, h, options, false)
                        + raw_vertical::<4>(&r, &prior, w, h, options, false);
                    cases += 4;
                }
            }
        }
    }
    for kw in [3, 7] {
        let options = GpuGaussianBlurOptions {
            kernel_width: kw,
            border: GpuGaussianBorder::Constant([255; 4]),
            ..Default::default()
        };
        bytes += raw_vertical::<1>(&r, &prior, 1025, 1025, options, true)
            + raw_vertical::<2>(&r, &prior, 1025, 1025, options, true)
            + raw_vertical::<3>(&r, &prior, 1025, 1025, options, true)
            + raw_vertical::<4>(&r, &prior, 1025, 1025, options, true);
        cases += 4;
    }
    println!("GAUSSIAN_VMIRROR_RAW verified {bytes} physical RGBA bytes across {cases} paired parent/u64 fixtures, full u16/23-bit inputs, dirty inactive halves, all borders and one-group stride.");
}

#[test]
fn gaussian_vmirror_pair_digits_keep_exact_wide_sums_and_rounding() {
    let mut sigmas =
        vec![f64::from_bits(1), 1e-300, 1e-100, 0.05, 0.17, 0.5, 1.2, 7., 1e100, f64::MAX];
    let mut seed = 0x3141_5926_5358_9793u64;
    for _ in 0..512 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        sigmas.push(f64::from_bits(((seed % 2047) << 52) | (seed & ((1 << 52) - 1)) | 1));
    }
    let mut cases = 0;
    for size in [3, 5, 7] {
        for bits in [8, 15] {
            for &sigma in &sigmas {
                let w = weights(size, sigma, bits).into_iter().flatten().collect::<Vec<_>>();
                let mass = 1u64 << bits;
                assert_eq!(w.iter().map(|&v| u64::from(v)).sum::<u64>(), mass);
                for tap in 0..size as usize {
                    assert_eq!(w[tap], w[size as usize - 1 - tap]);
                }
                for pattern in 0..16u32 {
                    let maximum = (1u32 << (bits + 8)) - 1;
                    let values: Vec<_> = (0..size)
                        .map(|tap| match pattern {
                            0 => maximum,
                            1 => {
                                if tap % 2 == 0 {
                                    maximum
                                } else {
                                    0
                                }
                            }
                            2 => 65535 & maximum,
                            3 => 65536 & maximum,
                            _ => {
                                tap.wrapping_mul(747_796_405)
                                    .wrapping_add(pattern.wrapping_mul(2_891_336_453))
                                    .rotate_left(13)
                                    & maximum
                            }
                        })
                        .collect();
                    let radius = size as usize / 2;
                    let mut lo = if bits == 8 { values[radius] } else { values[radius] & 65535 }
                        .checked_mul(w[radius])
                        .unwrap();
                    let mut hi = if bits == 8 {
                        0
                    } else {
                        (values[radius] >> 16).checked_mul(w[radius]).unwrap()
                    };
                    for tap in 0..radius {
                        let a = values[tap];
                        let b = values[size as usize - 1 - tap];
                        if bits == 8 {
                            lo = lo.checked_add((a + b).checked_mul(w[tap]).unwrap()).unwrap();
                        } else {
                            lo = lo
                                .checked_add(
                                    ((a & 65535) + (b & 65535)).checked_mul(w[tap]).unwrap(),
                                )
                                .unwrap();
                            hi = hi
                                .checked_add(((a >> 16) + (b >> 16)).checked_mul(w[tap]).unwrap())
                                .unwrap();
                        }
                    }
                    let expected: u64 = values
                        .iter()
                        .zip(&w)
                        .map(|(&v, &weight)| u64::from(v) * u64::from(weight))
                        .sum();
                    let reconstructed = if bits == 8 {
                        u64::from(lo)
                    } else {
                        u64::from(lo) + (u64::from(hi) << 16)
                    };
                    assert_eq!(reconstructed, expected);
                    let rounded =
                        if bits == 8 { (lo + 32768) >> 16 } else { (hi + (lo >> 16) + 8192) >> 14 };
                    assert_eq!(
                        u64::from(rounded),
                        (expected + (1u64 << (2 * bits - 1))) >> (2 * bits)
                    );
                    assert!(u64::from(lo) <= 65535 * mass);
                    assert!(u64::from(hi) <= 127 * mass);
                    cases += 1;
                }
            }
        }
    }
    println!("GAUSSIAN_VMIRROR_DIGITS verified {cases} symmetric paired wide-sum/carry/rounding fixtures with checked u32 arithmetic.");
}

fn complete_parent<const C: usize>(r: &WgpuRuntime, prior: &old::ParentGaussianPipelines) -> usize {
    let mut bytes = 0;
    for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 3), (17, 19), (257, 259)] {
        let image = fixture::<C>(w, h, false);
        let source = GpuImage::upload_u8(r, image.view()).unwrap();
        for kw in [3, 5, 7] {
            for kh in [3, 5, 7] {
                for b in [
                    GpuGaussianBorder::Replicate,
                    GpuGaussianBorder::Reflect,
                    GpuGaussianBorder::Reflect101,
                    GpuGaussianBorder::Wrap,
                    GpuGaussianBorder::Constant([19, 73, 129, 251]),
                ] {
                    let options = GpuGaussianBlurOptions {
                        kernel_width: kw,
                        kernel_height: kh,
                        sigma_x: 0.17,
                        sigma_y: 7.,
                        border: b,
                    };
                    let expected = gaussian_blur_u8(
                        image.view(),
                        kw as usize,
                        kh as usize,
                        options.sigma_x,
                        options.sigma_y,
                        border(b),
                    )
                    .unwrap();
                    let mut previous = old::previous(prior, r, &source, options, None).unwrap();
                    let mut current = gaussian_blur_gpu(r, &source, options).unwrap();
                    assert_eq!(previous.receipt(), current.receipt());
                    assert_eq!(previous.metadata(), current.metadata());
                    assert_eq!(previous.readback_u8::<C>(r).unwrap(), expected);
                    assert_eq!(current.readback_u8::<C>(r).unwrap(), expected);
                    assert_eq!(previous.receipt(), current.receipt());
                    previous.recycle(r);
                    current.recycle(r);
                    bytes += w * h * C * 2;
                }
            }
        }
        source.recycle(r);
    }
    bytes
}

#[test]
fn gaussian_vmirror_complete_parent_cpu_keep_all_channels_borders_and_mixed_axes() {
    let r = WgpuRuntime::new_headless().unwrap();
    let prior = old::ParentGaussianPipelines::new();
    let bytes = complete_parent::<1>(&r, &prior)
        + complete_parent::<2>(&r, &prior)
        + complete_parent::<3>(&r, &prior)
        + complete_parent::<4>(&r, &prior);
    println!("GAUSSIAN_VMIRROR_PUBLIC verified {bytes} complete parent/CPU bytes, both scratch layouts, mixed axes, receipts and metadata.");
}

#[path = "gaussian_vmirror_benchmark.rs"]
mod benchmark;
