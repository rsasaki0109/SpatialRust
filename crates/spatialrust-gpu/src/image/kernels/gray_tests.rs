use super::*;
use crate::image::gpu_image::read_staging_bytes;
use spatialrust_image::{ColorRange, Image, ImageRegion, ImageView};

pub(super) fn fixture(w: usize, h: usize, kind: usize) -> Image<u8, 3> {
    let data = (0..w * h)
        .flat_map(|i| match kind {
            0 => [(i * 37) as u8, (i * 11 / 3) as u8, (i * 71) as u8],
            1 => [0; 3],
            2 => [255; 3],
            _ => [(i >> 16) as u8, (i >> 8) as u8, i as u8],
        })
        .collect();
    Image::try_new_with_metadata(
        w,
        h,
        data,
        ImageMetadata { color_range: ColorRange::Limited, ..Default::default() },
    )
    .unwrap()
}

pub(super) fn oracle(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .map(|p| {
            ((77 * u32::from(p[0]) + 150 * u32::from(p[1]) + 29 * u32::from(p[2]) + 128) / 256)
                as u8
        })
        .collect()
}

fn physical_rgba(runtime: &WgpuRuntime, output: &GpuImage) -> Image<u8, 4> {
    // A temporary handle to the same texture exposes physical unused channels
    // for explicit test readback; it performs no texture allocation or copy.
    let mut physical = GpuImage::from_parts(
        runtime,
        output.width(),
        output.height(),
        4,
        output.texture().clone(),
        ImageMetadata { color_space: ColorSpace::Unknown, ..output.metadata() },
        output.receipt().clone(),
    )
    .unwrap();
    physical.readback_u8::<4>(runtime).unwrap()
}

fn assert_rgba(runtime: &WgpuRuntime, output: &GpuImage, expected: &[u8]) {
    let actual = physical_rgba(runtime, output);
    assert_eq!(actual.as_slice().len(), expected.len() * 4);
    for (pixel, &gray) in actual.as_slice().chunks_exact(4).zip(expected) {
        assert_eq!(pixel, [gray, 0, 0, 0]);
    }
}

#[test]
fn planner_bounds_both_axes_and_u32_coordinate_arithmetic() {
    let mut limits = wgpu::Limits::default();
    for (w, h) in [(0, 1), (1, 0), (u32::MAX, 1), (1, u32::MAX)] {
        assert!(gray_plan(w, h, &limits).is_err());
    }
    for field in 0..12 {
        let mut changed = limits.clone();
        match field {
            0 => changed.max_compute_invocations_per_workgroup = 255,
            1 => changed.max_compute_workgroup_size_x = 255,
            2 => changed.max_compute_workgroup_size_y = 0,
            3 => changed.max_compute_workgroup_size_z = 0,
            4 => changed.max_compute_workgroups_per_dimension = 0,
            5 => changed.max_sampled_textures_per_shader_stage = 0,
            6 => changed.max_storage_textures_per_shader_stage = 0,
            7 => changed.max_uniform_buffers_per_shader_stage = 0,
            8 => changed.max_uniform_buffer_binding_size = 15,
            9 => changed.max_buffer_size = 15,
            10 => changed.max_bind_groups = 0,
            _ => changed.max_bindings_per_bind_group = 2,
        }
        assert!(gray_plan(17, 19, &changed).is_err(), "field {field}");
    }
    limits.max_texture_dimension_2d = u32::MAX;
    limits.max_compute_workgroup_storage_size = 0;
    limits.max_storage_buffers_per_shader_stage = 0;
    assert!(gray_plan(65536, 65536, &limits).is_err());
    let maximum = gray_plan(65535, 65537, &limits).unwrap();
    assert_eq!(maximum.group_count, 16_777_216);
    assert!(
        maximum.groups_x <= limits.max_compute_workgroups_per_dimension
            && maximum.groups_y <= limits.max_compute_workgroups_per_dimension
    );
    assert!(gray_plan(i32::MAX as u32, 2, &limits).is_ok());
    assert!(gray_plan(i32::MAX as u32 + 1, 1, &limits).is_err());
    limits.max_compute_workgroups_per_dimension = 2;
    assert_eq!(gray_plan(1, 1024, &limits).unwrap().groups_y, 2);
    assert!(gray_plan(1, 1025, &limits).is_err());
    limits.max_texture_dimension_2d = 16;
    assert!(gray_plan(17, 1, &limits).is_err());
}

#[test]
fn forced_grid_and_frozen_linear_match_independent_full_rgba_oracle() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cache = runtime.image_gray_pipeline.get_or_init(|| create_gray_pipeline(runtime.device()));
    let frozen = create_pipeline(
        runtime.device(),
        &cache.bind_group_layout,
        include_str!("../../../tests/fixtures/gray_previous.wgsl"),
    );
    let grid =
        create_pipeline(runtime.device(), &cache.bind_group_layout, include_str!("gray_grid.wgsl"));
    let mut bytes = 0;
    let mut cases = 0;
    for (w, h) in
        [(1, 1), (1, 255), (1, 256), (1, 257), (257, 1), (17, 19), (31, 33), (65, 67), (257, 259)]
    {
        for kind in 0..3 {
            let image = fixture(w, h, kind);
            let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
            let expected = oracle(image.as_slice());
            let linear = gray_plan(w as u32, h as u32, &runtime.device().limits()).unwrap();
            let old = gray_dispatch(&runtime, &source, linear, cache, &frozen, None).unwrap();
            assert_rgba(&runtime, &old, &expected);
            old.recycle(&runtime);
            let current = rgb_to_gray_gpu(&runtime, &source).unwrap();
            assert_rgba(&runtime, &current, &expected);
            current.recycle(&runtime);
            bytes += w * h * 8;
            for cap in [2, 3, 7, 17, 31] {
                let mut limits = runtime.device().limits();
                limits.max_compute_workgroups_per_dimension = cap;
                if let Ok(plan) = gray_plan(w as u32, h as u32, &limits) {
                    let output =
                        gray_dispatch(&runtime, &source, plan, cache, &grid, None).unwrap();
                    assert_rgba(&runtime, &output, &expected);
                    assert_eq!(output.metadata().color_space, ColorSpace::Gray);
                    assert_eq!(output.metadata().color_range, image.metadata().color_range);
                    assert_eq!(output.receipt().stages(), ["upload_u8_texture", "rgb_to_gray_gpu"]);
                    assert_eq!(output.receipt().gpu_to_gpu_bytes(), (w * h * 4) as u64);
                    assert_eq!(output.receipt().device_to_host_bytes(), 0);
                    output.recycle(&runtime);
                    bytes += w * h * 4;
                    cases += 1;
                }
            }
            source.recycle(&runtime);
        }
    }
    eprintln!("Verified {cases} forced grid mappings and {bytes} exact RGBA bytes across frozen/public/grid variants.");
}

#[test]
fn maximum_pixel_index_and_padding_groups_do_not_wrap_on_device() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let shader = format!(
        "{}\n{}",
        include_str!("gray_grid.wgsl"),
        r#"
@group(0) @binding(3) var<storage, read> cases: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> answers: array<vec2<u32>>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&cases)) { return; }
    let c = cases[id.x];
    let index = flat_pixel(vec3<u32>(c.xy, 0u), c.z);
    answers[id.x] = vec2<u32>(index, select(0u, 1u, index < params.width * params.height));
}"#
    );
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut words = 0;
    for (w, h, cap) in
        [(1u32, 1u32, 2u32), (17, 33, 2), (65535, 65537, 65535), (i32::MAX as u32, 2, 65535)]
    {
        let mut limits = runtime.device().limits();
        limits.max_texture_dimension_2d = u32::MAX;
        limits.max_compute_workgroups_per_dimension = cap;
        let plan = gray_plan(w, h, &limits).unwrap();
        let mut cases = Vec::<[u32; 4]>::new();
        let mut expected = Vec::<[u32; 2]>::new();
        for flat in [0, plan.group_count - 1, plan.group_count] {
            for lane in [0, 1, 254, 255] {
                cases.push([flat % plan.groups_x, flat / plan.groups_x, lane, 0]);
                let index = if flat >= plan.group_count {
                    u32::MAX
                } else {
                    (u64::from(flat) * 256 + u64::from(lane)) as u32
                };
                expected.push([index, u32::from(u64::from(index) < u64::from(w) * u64::from(h))]);
            }
        }
        let params = GrayParams {
            width: w,
            height: h,
            groups_per_row: plan.groups_x,
            group_count: plan.group_count,
        };
        let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&cases),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let size = expected.len() as u64 * 8;
        let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: output.as_entire_binding() },
            ],
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, size);
        runtime.queue().submit(Some(encoder.finish()));
        let actual = read_staging_bytes(runtime.device(), &staging, size as usize).unwrap();
        assert_eq!(actual, bytemuck::cast_slice(&expected));
        words += expected.len() * 2;
    }
    eprintln!(
        "Verified {words} exact device index/validity words including u32::MAX and padding groups."
    );
}

#[test]
fn exhaustive_rgb_large_public_grid_is_lazy_and_exact() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let image = fixture(4096, 4096, 3); // All 2^24 RGB combinations, exactly once.
    let plan = gray_plan(4096, 4096, &runtime.device().limits()).unwrap();
    assert!(plan.groups_y > 1);
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let output = rgb_to_gray_gpu(&runtime, &source).unwrap();
    assert_rgba(&runtime, &output, &oracle(image.as_slice()));
    let cache = runtime.image_gray_pipeline.get().unwrap();
    assert!(cache.grid_pipeline.get().is_some() && cache.pipeline.get().is_none());
    assert_eq!(output.receipt().gpu_to_gpu_bytes(), 67_108_864);
    assert_eq!(output.receipt().device_to_host_bytes(), 0);
    output.recycle(&runtime);
    source.recycle(&runtime);
    eprintln!("Verified all 16777216 RGB values and 67108864 exact RGBA bytes above the single-axis dispatch limit.");
}

#[test]
fn invalid_ownership_channels_and_capacity_leave_gray_work_unallocated() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    let other =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    let source = GpuImage::upload_u8(&runtime, fixture(17, 19, 0).view()).unwrap();
    let gray =
        GpuImage::upload_u8(&runtime, Image::<u8, 1>::try_new(1, 1, vec![0]).unwrap().view())
            .unwrap();
    let before = runtime.image_texture_pool_stats();
    assert!(rgb_to_gray_gpu(&other, &source).is_err());
    assert!(rgb_to_gray_gpu(&runtime, &gray).is_err());
    let mut foreign = crate::GpuStageProfiler::new(&other, 1).unwrap();
    assert!(rgb_to_gray_gpu_with_profiler(&runtime, &source, Some(&mut foreign)).is_err());
    assert_eq!(runtime.image_texture_pool_stats(), before);
    assert!(
        runtime.image_gray_pipeline.get().is_none() && other.image_gray_pipeline.get().is_none()
    );
    let mut profiler = crate::GpuStageProfiler::new(&runtime, 1).unwrap();
    let first = crate::image::kernels::spatial::sobel_gpu_with_profiler(
        &runtime,
        &gray,
        Some(&mut profiler),
    )
    .unwrap();
    let before = runtime.image_texture_pool_stats();
    assert!(rgb_to_gray_gpu_with_profiler(&runtime, &source, Some(&mut profiler)).is_err());
    assert_eq!(runtime.image_texture_pool_stats(), before);
    assert!(runtime.image_gray_pipeline.get().is_none());
    let report = profiler.collect(&runtime).unwrap();
    assert_eq!(report.stages.len(), 1);
    assert_eq!(report.device_to_host_bytes, 16);
    first.recycle(&runtime);
    gray.recycle(&runtime);
    source.recycle(&runtime);
}

#[test]
fn strided_roi_cold_concurrency_live_outputs_and_reuse_are_exact() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let data: Vec<_> = (0..(67 * 3 + 11) * 79).map(|i| (i * 31 + 7) as u8).collect();
    let view = ImageView::<u8, 3>::new(67, 79, 67 * 3 + 11, &data)
        .unwrap()
        .subview(ImageRegion::new(2, 3, 17, 19))
        .unwrap();
    let expected =
        oracle(&(0..19).flat_map(|y| view.row(y).unwrap().iter().copied()).collect::<Vec<_>>());
    let source = GpuImage::upload_u8(&runtime, view).unwrap();
    let outputs = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..4)
            .map(|_| {
                let runtime = &runtime;
                let source = &source;
                scope.spawn(move || rgb_to_gray_gpu(runtime, source).unwrap())
            })
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect::<Vec<_>>()
    });
    let cache = runtime.image_gray_pipeline.get().unwrap();
    assert!(cache.pipeline.get().is_some() && cache.grid_pipeline.get().is_none());
    source.recycle(&runtime);
    let source = GpuImage::upload_u8(&runtime, fixture(17, 19, 2).view()).unwrap();
    for _ in 0..12 {
        rgb_to_gray_gpu(&runtime, &source).unwrap().recycle(&runtime);
    }
    let runtime = Box::new(runtime);
    for output in outputs {
        assert_rgba(&runtime, &output, &expected);
        output.recycle(&runtime);
    }
    let before = runtime.image_texture_pool_stats();
    for _ in 0..12 {
        rgb_to_gray_gpu(&runtime, &source).unwrap().recycle(&runtime);
    }
    assert_eq!(runtime.image_texture_pool_stats().allocation_count, before.allocation_count);
    runtime.set_image_texture_pool_budget(0);
    let output = rgb_to_gray_gpu(&runtime, &source).unwrap();
    assert_rgba(&runtime, &output, &vec![255; 17 * 19]);
    output.recycle(&runtime);
    source.recycle(&runtime);
    assert_eq!(runtime.image_texture_pool_stats().cached_bytes, 0);
}

#[path = "gray_benchmark.rs"]
mod benchmark;
