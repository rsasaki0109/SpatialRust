use super::*;

#[path = "gaussian_split_reference.rs"]
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
    assert!(!p.packed);
    assert_eq!(p.params.bits, 15);
    let mass: u64 = p.params.weights_y.iter().flatten().map(|&w| u64::from(w)).sum();
    assert_eq!(mass, 32768);
    assert!(65535 * mass < (1 << 31));
    assert!(127 * mass < (1 << 22));
    p.params.groups = 1;
    let words = w * h * C;
    let edges =
        [0, 1, 16383, 16384, 16385, 32767, 32768, 32769, 65535, 65536, 65537, 8_355_840, 8_388_607];
    let mut sums: Vec<u32> = (0..words)
        .map(|i| {
            if hot {
                8_388_607
            } else if i % 2 == 0 {
                edges[(i / 2) % edges.len()]
            } else {
                (i as u32).wrapping_mul(747_796_405).wrapping_add(2_891_336_453).rotate_left(13)
                    & 0x7f_ffff
            }
        })
        .collect();
    // Probe the whole 23-bit bound, including half-up boundaries and clamp cases.
    sums.extend_from_slice(&[0xdead_beef; 8]);
    let input = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gaussian-split-raw-input"),
        contents: bytemuck::cast_slice(&sums),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let uniform = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gaussian-split-raw-params"),
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let cache = GaussianPipelines::new();
    let outputs: [_; 2] =
        std::array::from_fn(|_| create_texture(r, w as u32, h as u32, "gaussian-split-raw-output"));
    let mut encoder = r.device().create_command_encoder(&Default::default());
    for (pipeline, output) in [
        (&prior.pipelines(r.device(), false)[1], &outputs[0]),
        (&cache.pipelines(r.device(), false)[1], &outputs[1]),
    ] {
        let group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gaussian-split-raw-group"),
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
                    .map_or(u32::from(constant[c]) * 32768, |sy| {
                        sums[(sy * w + pixel % w) * C + c]
                    });
                    u64::from(horizontal) * u64::from(p.params.weights_y[tap / 4][tap % 4])
                })
                .sum();
            expected[pixel * 4 + c] = ((total + (1 << 29)) / (1 << 30)).min(255) as u8;
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
fn gaussian_split_raw_vertical_matches_parent_and_wide_cpu_with_rounding_and_carries() {
    let r = WgpuRuntime::new_headless().unwrap();
    let prior = old::ParentGaussianPipelines::new();
    let mut bytes = 0;
    for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 3), (17, 19), (257, 259)] {
        for height in [3, 5, 7] {
            for border in [
                GpuGaussianBorder::Replicate,
                GpuGaussianBorder::Reflect,
                GpuGaussianBorder::Reflect101,
                GpuGaussianBorder::Wrap,
                GpuGaussianBorder::Constant([19, 73, 129, 251]),
            ] {
                for sigma in [0.17, 7.0] {
                    let options = GpuGaussianBlurOptions {
                        kernel_width: 7,
                        kernel_height: height,
                        sigma_y: sigma,
                        border,
                        ..Default::default()
                    };
                    bytes += raw_vertical::<1>(&r, &prior, w, h, options, false)
                        + raw_vertical::<2>(&r, &prior, w, h, options, false)
                        + raw_vertical::<3>(&r, &prior, w, h, options, false)
                        + raw_vertical::<4>(&r, &prior, w, h, options, false);
                }
            }
        }
    }
    let options = GpuGaussianBlurOptions {
        kernel_width: 7,
        kernel_height: 7,
        border: GpuGaussianBorder::Constant([255; 4]),
        ..Default::default()
    };
    bytes += raw_vertical::<1>(&r, &prior, 1025, 1025, options, true)
        + raw_vertical::<2>(&r, &prior, 1025, 1025, options, true)
        + raw_vertical::<3>(&r, &prior, 1025, 1025, options, true)
        + raw_vertical::<4>(&r, &prior, 1025, 1025, options, true);
    eprintln!("GAUSSIAN_SPLIT_RAW verified {bytes} exact parent/u64 CPU physical bytes, full 23-bit inputs, half-up boundaries and clamping, all borders and one-group stride");
}

fn complete_parent<const C: usize>(r: &WgpuRuntime, prior: &old::ParentGaussianPipelines) -> usize {
    let mut bytes = 0;
    for (w, h) in [(1, 1), (1, 7), (7, 1), (17, 19), (257, 259)] {
        let image = fixture::<C>(w, h, false);
        let source = GpuImage::upload_u8(r, image.view()).unwrap();
        for (kw, kh) in [(3, 3), (3, 5), (3, 7), (5, 3), (5, 5), (5, 7), (7, 3), (7, 5), (7, 7)] {
            for border in [
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
                    sigma_y: 7.0,
                    border,
                };
                let expected = gaussian_blur_u8(
                    image.view(),
                    kw as usize,
                    kh as usize,
                    options.sigma_x,
                    options.sigma_y,
                    super::border(border),
                )
                .unwrap();
                let mut before = old::previous(prior, r, &source, options, None).unwrap();
                let mut after = gaussian_blur_gpu(r, &source, options).unwrap();
                assert_eq!(before.receipt(), after.receipt());
                assert_eq!(before.metadata(), after.metadata());
                assert_eq!(before.readback_u8::<C>(r).unwrap(), expected);
                assert_eq!(after.readback_u8::<C>(r).unwrap(), expected);
                assert_eq!(before.receipt(), after.receipt());
                before.recycle(r);
                after.recycle(r);
                bytes += w * h * C * 2;
            }
        }
        source.recycle(r);
    }
    bytes
}

#[test]
fn gaussian_split_complete_parent_and_cpu_keep_all_channels_borders_and_q8_controls() {
    let r = WgpuRuntime::new_headless().unwrap();
    let prior = old::ParentGaussianPipelines::new();
    let bytes = complete_parent::<1>(&r, &prior)
        + complete_parent::<2>(&r, &prior)
        + complete_parent::<3>(&r, &prior)
        + complete_parent::<4>(&r, &prior);
    eprintln!("GAUSSIAN_SPLIT_PUBLIC verified {bytes} exact complete-parent/CPU bytes, mixed axes, metadata, receipts, all channels/borders and Q15 and unchanged Q8 controls");
}

#[path = "gaussian_split_benchmark.rs"]
mod benchmark;
