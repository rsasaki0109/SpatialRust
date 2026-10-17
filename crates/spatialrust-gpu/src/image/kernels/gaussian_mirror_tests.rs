use super::*;

#[path = "gaussian_mirror_reference.rs"]
mod old;

fn mapped_column(x: i32, width: usize, b: GpuGaussianBorder) -> Option<usize> {
    let n = width as i32;
    if (0..n).contains(&x) {
        return Some(x as usize);
    }
    match b {
        GpuGaussianBorder::Constant(_) => None,
        GpuGaussianBorder::Replicate => Some(x.clamp(0, n - 1) as usize),
        _ if n == 1 => Some(0),
        GpuGaussianBorder::Wrap => Some(x.rem_euclid(n) as usize),
        GpuGaussianBorder::Reflect => {
            let x = x.rem_euclid(2 * n);
            Some(if x < n { x } else { 2 * n - 1 - x } as usize)
        }
        GpuGaussianBorder::Reflect101 => {
            let x = x.rem_euclid(2 * (n - 1));
            Some(if x < n { x } else { 2 * (n - 1) - x } as usize)
        }
    }
}

fn read_words(r: &WgpuRuntime, input: &wgpu::Buffer) -> Vec<u32> {
    let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: input.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = r.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(input, 0, &staging, 0, input.size());
    r.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    r.wait_idle();
    rx.recv().unwrap().unwrap();
    let view = slice.get_mapped_range();
    let words = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();
    words
}

#[test]
fn gaussian_mirror_weights_keep_symmetric_nonnegative_exact_mass() {
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
                let flat = weights(size, sigma, bits).into_iter().flatten().collect::<Vec<_>>();
                let mass = 1u64 << bits;
                assert_eq!(flat.iter().map(|&n| u64::from(n)).sum::<u64>(), mass);
                for tap in 0..size as usize {
                    assert_eq!(flat[tap], flat[size as usize - 1 - tap]);
                    assert!(u64::from(flat[tap]) <= mass);
                }
                assert!(flat[size as usize..].iter().all(|&n| n == 0));
                assert!(255 * mass < 1 << 23);
                cases += 1;
            }
        }
    }
    println!("GAUSSIAN_MIRROR_WEIGHTS verified {cases} exact symmetric bounded-mass quantizers.");
}

fn raw_horizontal<const C: usize>(
    r: &WgpuRuntime,
    prior: &old::ParentGaussianPipelines,
    w: usize,
    h: usize,
    options: GpuGaussianBlurOptions,
    hot: bool,
) -> usize {
    let image = fixture::<C>(w, h, hot);
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let mut p = plan(w as u32, h as u32, C as u32, options, &r.device().limits()).unwrap();
    // Force multiple grid-stride iterations and a partial final workgroup.
    p.params.groups = 1;
    let words = p.scratch_bytes as usize / 4;
    let mut expected = vec![0u32; words];
    let constant = if let GpuGaussianBorder::Constant(v) = options.border { v } else { [0; 4] };
    for pixel in 0..w * h {
        for c in 0..C {
            let sum: u64 = (0..options.kernel_width as usize)
                .map(|tap| {
                    let value = mapped_column(
                        (pixel % w) as i32 + tap as i32 - (options.kernel_width / 2) as i32,
                        w,
                        options.border,
                    )
                    .map_or(constant[c], |x| image.as_slice()[((pixel / w) * w + x) * C + c]);
                    u64::from(value) * u64::from(p.params.weights_x[tap / 4][tap % 4])
                })
                .sum();
            assert!(sum <= 255 * (1u64 << p.params.bits));
            let value = u32::try_from(sum).unwrap();
            if p.packed {
                assert!(value < 1 << 16);
                expected[pixel * C.div_ceil(2) + c / 2] |= value << (c % 2 * 16);
            } else {
                expected[pixel * C + c] = value;
            }
        }
    }
    expected.extend([0xdead_beef; 8]);
    let outputs: [_; 2] = std::array::from_fn(|_| {
        r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gaussian-mirror-guarded-sums"),
            contents: bytemuck::cast_slice(&vec![0xdead_beefu32; words + 8]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    });
    let uniform = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let cache = GaussianPipelines::new();
    let pipelines =
        [&prior.pipelines(r.device(), p.packed)[0], &cache.pipelines(r.device(), p.packed)[0]];
    let mut encoder = r.device().create_command_encoder(&Default::default());
    for (pipeline, output) in pipelines.into_iter().zip(&outputs) {
        let group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&source.view()),
                },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    r.queue().submit(Some(encoder.finish()));
    for output in outputs {
        assert_eq!(read_words(r, &output), expected, "{w}x{h} C={C} {options:?}");
    }
    source.recycle(r);
    expected.len() * 2
}

#[test]
fn gaussian_mirror_raw_horizontal_matches_parent_u64_sums_packing_and_guards() {
    let r = WgpuRuntime::new_headless().unwrap();
    let prior = old::ParentGaussianPipelines::new();
    let mut words = 0;
    for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 3), (17, 19), (257, 259)] {
        for kernel_width in [3, 5, 7] {
            for kernel_height in [3, 7] {
                for sigma_x in [0.17, 7.] {
                    for border in [
                        GpuGaussianBorder::Replicate,
                        GpuGaussianBorder::Reflect,
                        GpuGaussianBorder::Reflect101,
                        GpuGaussianBorder::Wrap,
                        GpuGaussianBorder::Constant([19, 73, 129, 251]),
                    ] {
                        let options = GpuGaussianBlurOptions {
                            kernel_width,
                            kernel_height,
                            sigma_x,
                            border,
                            ..Default::default()
                        };
                        words += raw_horizontal::<1>(&r, &prior, w, h, options, false)
                            + raw_horizontal::<2>(&r, &prior, w, h, options, false)
                            + raw_horizontal::<3>(&r, &prior, w, h, options, false)
                            + raw_horizontal::<4>(&r, &prior, w, h, options, false);
                    }
                }
            }
        }
    }
    for (kernel_width, kernel_height) in [(3, 3), (5, 5), (7, 7), (3, 7)] {
        let options = GpuGaussianBlurOptions {
            kernel_width,
            kernel_height,
            border: GpuGaussianBorder::Constant([255; 4]),
            ..Default::default()
        };
        words += raw_horizontal::<1>(&r, &prior, 1025, 257, options, true)
            + raw_horizontal::<2>(&r, &prior, 1025, 257, options, true)
            + raw_horizontal::<3>(&r, &prior, 1025, 257, options, true)
            + raw_horizontal::<4>(&r, &prior, 1025, 257, options, true);
    }
    println!("GAUSSIAN_MIRROR_RAW verified {words} exact parent/u64 horizontal/guard words, packed odd halves, one-group strides and maximum input mass.");
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
fn gaussian_mirror_complete_parent_cpu_keep_all_channels_borders_and_mixed_axes() {
    let r = WgpuRuntime::new_headless().unwrap();
    let prior = old::ParentGaussianPipelines::new();
    let bytes = complete_parent::<1>(&r, &prior)
        + complete_parent::<2>(&r, &prior)
        + complete_parent::<3>(&r, &prior)
        + complete_parent::<4>(&r, &prior);
    println!("GAUSSIAN_MIRROR_PUBLIC verified {bytes} complete parent/CPU bytes, both scratch layouts, mixed axes, receipts and metadata.");
}

#[path = "gaussian_mirror_benchmark.rs"]
mod benchmark;
