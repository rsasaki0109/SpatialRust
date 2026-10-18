use super::*;
use spatialrust_image::Image;

fn oracle(image: &Image<u8, 1>, kw: u32, kh: u32, border: GpuImageBorder) -> Vec<u8> {
    let (w, h) = (image.width() as i64, image.height() as i64);
    let area = u64::from(kw) * u64::from(kh);
    (0..w * h)
        .map(|index| {
            let (x, y) = (index % w, index / w);
            let mut sum = 0_u64;
            for dy in -(i64::from(kh) / 2)..=i64::from(kh) / 2 {
                for dx in -(i64::from(kw) / 2)..=i64::from(kw) / 2 {
                    let (mut sx, mut sy) = (x + dx, y + dy);
                    if border == GpuImageBorder::Replicate {
                        sx = sx.clamp(0, w - 1);
                        sy = sy.clamp(0, h - 1);
                    } else if sx < 0 || sx >= w || sy < 0 || sy >= h {
                        continue;
                    }
                    sum += u64::from(image.as_slice()[(sy * w + sx) as usize]);
                }
            }
            ((sum + area / 2) / area) as u8
        })
        .collect()
}

fn fixture(w: usize, h: usize, hot: bool) -> Image<u8, 1> {
    Image::try_new(
        w,
        h,
        (0..w * h).map(|i| if hot { 255 } else { ((i * 7919 + i / w * 31) % 256) as u8 }).collect(),
    )
    .unwrap()
}

fn previous_pipeline(runtime: &WgpuRuntime, cache: &BlurPipeline) -> wgpu::ComputePipeline {
    let layout = &cache.variant(runtime.device(), 0).layout;
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("frozen-blur"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../../tests/fixtures/box_blur_previous.wgsl").into(),
        ),
    });
    let pipeline_layout =
        runtime.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[layout],
            push_constant_ranges: &[],
        });
    runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_previous(
    runtime: &WgpuRuntime,
    cache: &BlurPipeline,
    previous: &wgpu::ComputePipeline,
    p: &Plan,
    source: &wgpu::TextureView,
    output: &wgpu::TextureView,
    uniform: &wgpu::Buffer,
    encoder: &mut wgpu::CommandEncoder,
    mut profiler: Option<&mut GpuStageProfiler>,
) {
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &cache.variant(runtime.device(), 0).layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(source),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(output),
            },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, "previous_blur"),
    });
    pass.set_pipeline(previous);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(p.params.pixel_count.div_ceil(256), 1, 1);
}

#[test]
fn exact_box_means_match_frozen_gpu_and_independent_integer_oracle() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cache = BlurPipeline::new();
    let previous = previous_pipeline(&runtime, &cache);
    let mut checked = 0;
    for (w, h) in [(1, 1), (1, 33), (33, 1), (2, 3), (17, 19), (257, 3), (33, 35)] {
        for hot in [false, true] {
            let image = fixture(w, h, hot);
            let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
            for (kw, kh) in
                [(1, 1), (3, 3), (7, 7), (9, 9), (3, 27), (27, 3), (1, 81), (81, 1), (31, 33)]
            {
                for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                    let mut p =
                        plan(w as u32, h as u32, kw, kh, border, &runtime.device().limits())
                            .unwrap();
                    // Force a single workgroup to prove stride coverage and complete scratch writes.
                    p.params.groups = 1;
                    p.separable = u64::from(kw) * u64::from(kh) >= 81;
                    let uniform =
                        runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: bytemuck::bytes_of(&p.params),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let mut scratch = ScratchBuffers::new(&runtime);
                    let sums = p.separable.then(|| scratch.acquire("dirty-blur-test", p.bytes));
                    if let Some(buffer) = &sums {
                        runtime.queue().write_buffer(buffer, 0, &vec![255; p.bytes as usize]);
                    }
                    let outputs: [_; 2] = std::array::from_fn(|_| {
                        create_texture(&runtime, w as u32, h as u32, "blur-test")
                    });
                    let mut encoder = runtime.device().create_command_encoder(&Default::default());
                    encode_previous(
                        &runtime,
                        &cache,
                        &previous,
                        &p,
                        &source.view(),
                        &outputs[0].create_view(&Default::default()),
                        &uniform,
                        &mut encoder,
                        None,
                    );
                    encode(
                        &runtime,
                        &cache,
                        &p,
                        &source.view(),
                        &outputs[1].create_view(&Default::default()),
                        &uniform,
                        sums.as_ref(),
                        &mut encoder,
                        None,
                    );
                    scratch.submit_and_recycle(encoder);
                    let expected = oracle(&image, kw, kh, border);
                    for output in outputs {
                        let mut gpu = GpuImage::from_parts(
                            &runtime,
                            w as u32,
                            h as u32,
                            4,
                            output,
                            source.metadata(),
                            Default::default(),
                        )
                        .unwrap();
                        let rgba: Vec<_> =
                            expected.iter().flat_map(|&value| [value, 0, 0, 0]).collect();
                        assert_eq!(
                            gpu.readback_u8::<4>(&runtime).unwrap().as_slice(),
                            rgba,
                            "{w}x{h} {kw}x{kh} {border:?}"
                        );
                        gpu.recycle(&runtime);
                    }
                    checked += w * h;
                }
            }
            source.recycle(&runtime);
        }
    }
    eprintln!("Exact frozen/oracle blur pixels checked: {checked}");
}

#[test]
fn blur_plan_bounds_pixels_sums_scratch_and_dispatch() {
    let limits = wgpu::Limits::default();
    for (kw, kh, passes) in [(1, 1, 1), (7, 7, 1), (9, 9, 2), (3, 27, 2), (81, 1, 2)] {
        assert_eq!(blur_pass_count(1024, 1, kw, kh), passes);
    }
    assert_eq!(blur_pass_count(809, 1, 9, 9), 1);
    assert_eq!(blur_pass_count(810, 1, 9, 9), 2);
    assert_eq!(blur_pass_count(1, 1, 257, 257), 2);
    assert_eq!(blur_pass_count(u32::MAX, u32::MAX, u32::MAX, u32::MAX), 2);
    for (w, h, kw, kh) in [
        (0, 1, 3, 3),
        (1, 0, 3, 3),
        (1, 1, 0, 3),
        (1, 1, 2, 3),
        (1, 1, 3, 4),
        (1, 1, 4101, 4101),
        (1, 1, u32::MAX, u32::MAX),
        (8193, 1, 3, 3),
    ] {
        assert!(plan(w, h, kw, kh, GpuImageBorder::Replicate, &limits).is_err());
    }
    assert!(plan(1, 1, 4099, 4099, GpuImageBorder::Replicate, &limits).is_ok());
    let mut huge = limits.clone();
    huge.max_texture_dimension_2d = u32::MAX;
    assert!(plan(u32::MAX, u32::MAX, 3, 3, GpuImageBorder::Replicate, &huge).is_err());
    assert!(plan(65536, 65536, 3, 3, GpuImageBorder::Replicate, &huge).is_err());
    assert!(plan(i32::MAX as u32, 1, 3, 3, GpuImageBorder::Replicate, &huge).is_ok());
    assert!(plan(i32::MAX as u32, 1, 5, 3, GpuImageBorder::Replicate, &huge).is_err());
    for field in 0..10 {
        let mut bad = limits.clone();
        match field {
            0 => bad.max_compute_invocations_per_workgroup = 255,
            1 => bad.max_compute_workgroup_size_x = 255,
            2 => bad.max_compute_workgroups_per_dimension = 0,
            3 => bad.max_sampled_textures_per_shader_stage = 0,
            4 => bad.max_storage_textures_per_shader_stage = 0,
            5 => bad.max_uniform_buffers_per_shader_stage = 0,
            6 => bad.max_uniform_buffer_binding_size = 31,
            7 => bad.max_storage_buffers_per_shader_stage = 0,
            8 => bad.max_buffer_size = 3,
            _ => bad.max_storage_buffer_binding_size = 3,
        }
        assert!(plan(64, 64, 9, 9, GpuImageBorder::Replicate, &bad).is_err(), "field {field}");
        if field >= 7 {
            assert!(plan(1, 1, 3, 3, GpuImageBorder::Replicate, &bad).is_ok());
        }
    }
    let mut reduced = limits;
    reduced.max_compute_workgroups_per_dimension = 1;
    assert_eq!(plan(257, 259, 9, 9, GpuImageBorder::Replicate, &reduced).unwrap().params.groups, 1);
    let maximal = plan(65535, 65537, 3, 3, GpuImageBorder::Replicate, &huge).unwrap();
    assert_eq!(maximal.params.pixel_count, u32::MAX);
    assert_eq!(maximal.params.groups, 4096);
}

#[test]
fn invalid_public_blur_inputs_do_not_allocate() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let image = fixture(2, 3, false);
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let color = Image::<u8, 3>::try_new(1, 1, vec![1, 2, 3]).unwrap();
    let color = GpuImage::upload_u8(&runtime, color.view()).unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let before = runtime.image_texture_pool_stats();
    assert!(box_blur_gpu(&runtime, &color, 9, 9, GpuImageBorder::Replicate).is_err());
    assert!(box_blur_gpu(&other, &source, 9, 9, GpuImageBorder::Replicate).is_err());
    for (kw, kh) in [(0, 3), (2, 3), (u32::MAX, u32::MAX), (4101, 4101)] {
        assert!(box_blur_gpu(&runtime, &source, kw, kh, GpuImageBorder::Replicate).is_err());
    }
    assert_eq!(runtime.image_texture_pool_stats(), before);
    assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count, 0);
    assert!(runtime.image_blur_pipeline.get().is_none());
}

#[test]
fn roi_metadata_and_rounding_neighbors_match_integer_oracle() {
    use spatialrust_image::{ColorRange, ColorSpace, ImageMetadata, ImageRegion, ImageView};
    let runtime = WgpuRuntime::new_headless().unwrap();
    let stride = 37;
    let data: Vec<_> = (0..stride * 31).map(|i| ((i * 31 + 7) % 256) as u8).collect();
    let metadata = ImageMetadata {
        color_space: ColorSpace::Gray,
        color_range: ColorRange::Limited,
        ..Default::default()
    };
    let view = ImageView::<u8, 1>::new_with_metadata(33, 31, stride, &data, metadata)
        .unwrap()
        .subview(ImageRegion::new(3, 2, 17, 19))
        .unwrap();
    let packed = Image::<u8, 1>::try_new(
        17,
        19,
        (0..19)
            .flat_map(|y| data[(y + 2) * stride + 3..(y + 2) * stride + 20].iter().copied())
            .collect(),
    )
    .unwrap();
    let source = GpuImage::upload_u8(&runtime, view).unwrap();
    for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
        let mut result = box_blur_gpu(&runtime, &source, 31, 33, border).unwrap();
        assert_eq!(result.metadata(), metadata);
        assert_eq!(
            result.readback_u8::<1>(&runtime).unwrap().as_slice(),
            oracle(&packed, 31, 33, border)
        );
        result.recycle(&runtime);
    }
    for (value, expected) in [(40, 0), (41, 1), (121, 1), (122, 2)] {
        let mut impulse = vec![0; 32 * 32];
        impulse[16 * 32 + 16] = value;
        let image = Image::<u8, 1>::try_new(32, 32, impulse).unwrap();
        let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        let mut result =
            box_blur_gpu(&runtime, &source, 9, 9, GpuImageBorder::ConstantZero).unwrap();
        assert_eq!(result.readback_u8::<1>(&runtime).unwrap().as_slice()[16 * 32 + 16], expected);
    }
}

#[test]
fn retained_outputs_concurrent_cache_and_large_border_masses_stay_exact() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let image = fixture(33, 35, false);
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let outputs = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    box_blur_gpu(&runtime, &source, 9, 9, GpuImageBorder::Replicate).unwrap()
                })
            })
            .collect();
        jobs.into_iter().map(|job| job.join().unwrap()).collect::<Vec<_>>()
    });
    let runtime = Box::new(runtime);
    let expected = oracle(&image, 9, 9, GpuImageBorder::Replicate);
    for mut output in outputs {
        assert_eq!(output.metadata(), source.metadata());
        assert_eq!(output.receipt().host_to_device_bytes(), 33 * 35 * 4);
        assert_eq!(
            output.receipt().gpu_to_gpu_bytes(),
            (33_u64 * 35).div_ceil(2) * 4 + 33 * 35 * 4
        );
        assert_eq!(output.receipt().device_to_host_bytes(), 0);
        assert_eq!(output.readback_u8::<1>(&runtime).unwrap().as_slice(), expected);
        output.recycle(&runtime);
    }
    let before = runtime.scratch_buffer_pool_stats().allocation_count;
    for _ in 0..4 {
        let mut result =
            box_blur_gpu(&runtime, &source, 9, 9, GpuImageBorder::ConstantZero).unwrap();
        assert_eq!(
            result.readback_u8::<1>(&runtime).unwrap().as_slice(),
            oracle(&image, 9, 9, GpuImageBorder::ConstantZero)
        );
        result.recycle(&runtime);
    }
    assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count, before);
    let cache = runtime.image_blur_pipeline.get().unwrap();
    assert!(cache.variants[0].get().is_none());
    assert!(cache.variants[1].get().is_none() && cache.variants[2].get().is_none());
    assert!(cache.variants[3].get().is_some() && cache.variants[4].get().is_some());
    let single = fixture(1, 1, true);
    let gpu = GpuImage::upload_u8(&runtime, single.view()).unwrap();
    for (border, value) in [(GpuImageBorder::Replicate, 255), (GpuImageBorder::ConstantZero, 0)] {
        let mut output = box_blur_gpu(&runtime, &gpu, 4099, 4099, border).unwrap();
        assert_eq!(output.readback_u8::<1>(&runtime).unwrap().as_slice(), [value]);
    }
}

#[path = "box_blur_benchmark.rs"]
mod benchmark;

#[path = "box_blur_packed_tests.rs"]
mod packed;

#[path = "box_blur_window_tests.rs"]
mod windows;

#[path = "box_blur_vwindow_tests.rs"]
mod vertical_windows;

#[path = "box_blur_wide_vwindow_tests.rs"]
mod wide_vertical_windows;
