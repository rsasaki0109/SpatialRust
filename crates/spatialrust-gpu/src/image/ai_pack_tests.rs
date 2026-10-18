use super::*;
use spatialrust_image::{Image, ImageRegion, ImageView};

pub(super) fn fixture<const C: usize>(w: usize, h: usize) -> Image<u8, C> {
    Image::try_new(w, h, (0..w * h * C).map(|i| (i * 37 + i / C * 11 + i % C * 53) as u8).collect())
        .unwrap()
}

pub(super) fn assert_words(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "word {i}");
    }
}

fn cpu<const C: usize>(data: &[u8], scale: f32, mean: [f32; 4], std: [f32; 4]) -> Vec<f32> {
    (0..C)
        .flat_map(|c| {
            data.chunks_exact(C).map(move |pixel| (f32::from(pixel[c]) * scale - mean[c]) / std[c])
        })
        .collect()
}

pub(super) fn force_legacy(mut plan: AiPackPlan, w: u32, h: u32) -> AiPackPlan {
    plan.groups = [w.div_ceil(16), h.div_ceil(16), 1];
    plan
}

fn check_channel_cases<const C: usize>(
    runtime: &WgpuRuntime,
    cache: &AiPackPipeline,
    previous: &wgpu::ComputePipeline,
    parallel: &wgpu::ComputePipeline,
) -> usize {
    let specialized_pipeline = create_probe_pipeline(
        runtime.device(),
        &cache.layout,
        include_str!("../../tests/fixtures/ai_pack_specialized_experiment.wgsl"),
        Some(C as u32),
    );
    let mut words = 0;
    for (w, h) in [
        (1, 1),
        (1, 17),
        (17, 1),
        (7, 9),
        (8, 8),
        (9, 17),
        (15, 19),
        (16, 16),
        (17, 19),
        (255, 257),
        (256, 256),
        (257, 259),
    ] {
        let image = fixture::<C>(w, h);
        let source = GpuImage::upload_u8(runtime, image.view()).unwrap();
        let presets = [
            (1.0, [0.0; 4], [1.0; 4]),
            (0.25, [-0.0, 4.0, -8.0, 12.0], [0.5, -2.0, 4.0, 0.25]),
            (1.0 / 255.0, [0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.75]),
            (
                f32::MAX / 4.0,
                [f32::MAX / 2.0, -f32::MAX / 2.0, 0.0, 1.0],
                [f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 1.0, -1.0],
            ),
        ];
        for (p, (scale, mean, std)) in presets.into_iter().enumerate() {
            let plan = pack_plan(w as u32, h as u32, C as u32, &runtime.device().limits()).unwrap();
            let mut old = pack_dispatch(
                runtime,
                &source,
                scale,
                mean,
                std,
                force_legacy(plan, w as u32, h as u32),
                cache,
                previous,
                None,
            )
            .unwrap();
            let expected = old.readback_f32(runtime).unwrap();
            old.recycle(runtime);
            if p < 2 {
                assert_words(&expected, &cpu::<C>(image.as_slice(), scale, mean, std));
            }
            let forced =
                AiPackPlan { groups: [(w as u32).div_ceil(8), (h as u32).div_ceil(8), 1], ..plan };
            let tensors = [
                pack_dispatch(
                    runtime,
                    &source,
                    scale,
                    mean,
                    std,
                    force_legacy(plan, w as u32, h as u32),
                    cache,
                    &specialized_pipeline,
                    None,
                )
                .unwrap(),
                pack_dispatch(runtime, &source, scale, mean, std, forced, cache, parallel, None)
                    .unwrap(),
                pack_ai_chw_gpu(runtime, &source, scale, mean, std).unwrap(),
            ];
            for mut output in tensors {
                assert_eq!(output.shape(), [C as u32, h as u32, w as u32]);
                assert_eq!(output.receipt().stages(), ["upload_u8_texture", "pack_ai_chw_gpu"]);
                assert_eq!(output.receipt().gpu_to_gpu_bytes(), (w * h * C * 4) as u64);
                assert_eq!(output.receipt().device_to_host_bytes(), 0);
                assert_words(&output.readback_f32(runtime).unwrap(), &expected);
                assert_eq!(output.receipt().device_to_host_bytes(), (w * h * C * 4) as u64);
                output.recycle(runtime);
                words += w * h * C;
            }
        }
        source.recycle(runtime);
    }
    words
}

#[test]
fn all_channels_and_normalizations_match_frozen_words_and_independent_exact_math() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cache =
        runtime.image_ai_pack_pipeline.get_or_init(|| create_ai_pack_pipeline(runtime.device()));
    let previous = create_probe_pipeline(
        runtime.device(),
        &cache.layout,
        include_str!("../../tests/fixtures/ai_pack_previous.wgsl"),
        None,
    );
    let parallel = create_probe_pipeline(
        runtime.device(),
        &cache.layout,
        include_str!("../../tests/fixtures/ai_pack_channels_experiment.wgsl"),
        None,
    );
    let words = check_channel_cases::<1>(&runtime, cache, &previous, &parallel)
        + check_channel_cases::<2>(&runtime, cache, &previous, &parallel)
        + check_channel_cases::<3>(&runtime, cache, &previous, &parallel)
        + check_channel_cases::<4>(&runtime, cache, &previous, &parallel);
    eprintln!("Verified {words} exact output words across 192 channel/shape/normalization cases and forced/current paths.");
}

#[test]
fn buffer_index_dispatch_and_optional_channel_limits_are_checked() {
    let limits = wgpu::Limits::default();
    for (w, h, c) in
        [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 5), (u32::MAX, 1, 1), (1, u32::MAX, 1)]
    {
        assert!(pack_plan(w, h, c, &limits).is_err());
    }
    for field in 0..14 {
        let mut changed = limits.clone();
        match field {
            0 => changed.max_compute_invocations_per_workgroup = 255,
            1 => changed.max_compute_workgroup_size_x = 15,
            2 => changed.max_compute_workgroup_size_y = 15,
            3 => changed.max_compute_workgroup_size_z = 0,
            4 => changed.max_compute_workgroups_per_dimension = 0,
            5 => changed.max_sampled_textures_per_shader_stage = 0,
            6 => changed.max_storage_buffers_per_shader_stage = 0,
            7 => changed.max_uniform_buffers_per_shader_stage = 0,
            8 => changed.max_uniform_buffer_binding_size = 63,
            9 => changed.max_buffer_size = 63,
            10 => changed.max_bind_groups = 0,
            11 => changed.max_bindings_per_bind_group = 2,
            12 => changed.max_storage_buffer_binding_size = 63,
            _ => changed.max_texture_dimension_2d = 16,
        }
        assert!(pack_plan(17, 19, 3, &changed).is_err(), "field {field}");
    }
    let mut changed = limits.clone();
    changed.max_storage_buffer_binding_size = 1024;
    assert_eq!(pack_plan(16, 16, 1, &changed).unwrap().byte_len, 1024);
    assert!(pack_plan(17, 16, 1, &changed).is_err());
    changed.max_buffer_size = 1023;
    assert!(pack_plan(16, 16, 1, &changed).is_err());
    let mut optional = limits.clone();
    optional.max_compute_workgroup_storage_size = 0;
    optional.max_compute_workgroup_size_z = 1;
    assert_eq!(pack_plan(256, 256, 4, &optional).unwrap().groups, [16, 16, 1]);
    optional.max_compute_workgroup_size_z = 4;
    optional.max_compute_workgroups_per_dimension = 64;
    assert_eq!(pack_plan(1024, 64, 3, &optional).unwrap().groups, [64, 4, 1]);
    assert!(pack_plan(255, 257, 3, &limits).is_ok());
    assert!(pack_plan(256, 256, 3, &limits).is_ok());
    assert!(pack_plan(256, 256, 2, &limits).is_ok());
    let mut huge = limits;
    huge.max_texture_dimension_2d = u32::MAX;
    huge.max_compute_workgroups_per_dimension = u32::MAX;
    huge.max_buffer_size = u64::MAX;
    huge.max_storage_buffer_binding_size = u32::MAX;
    assert!(pack_plan(65536, 65536, 1, &huge).is_err());
    assert!(pack_plan(32768, 32768, 4, &huge).is_err());
    assert!(pack_plan(i32::MAX as u32 + 1, 1, 1, &huge).is_err());
    let boundary = pack_plan(16384, 16384, 3, &huge).unwrap();
    assert_eq!(boundary.byte_len, 3_221_225_472);
    assert!(pack_plan(16384, 16384, 4, &huge).is_err());
}

#[test]
fn invalid_parameters_ownership_and_capacity_do_not_allocate_or_initialize() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    let other =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    let source = GpuImage::upload_u8(&runtime, fixture::<3>(256, 256).view()).unwrap();
    let before = runtime.scratch_buffer_pool_stats();
    assert!(pack_ai_chw_gpu(&other, &source, 1.0, [0.0; 4], [1.0; 4]).is_err());
    for (scale, mean, std) in [
        (f32::NAN, [0.0; 4], [1.0; 4]),
        (1.0, [f32::INFINITY; 4], [1.0; 4]),
        (1.0, [0.0; 4], [0.0; 4]),
        (1.0, [0.0; 4], [-0.0; 4]),
        (1.0, [0.0; 4], [f32::NAN; 4]),
    ] {
        assert!(pack_ai_chw_gpu(&runtime, &source, scale, mean, std).is_err());
    }
    let mut foreign = crate::GpuStageProfiler::new(&other, 1).unwrap();
    assert!(pack_ai_chw_gpu_with_profiler(
        &runtime,
        &source,
        1.0,
        [0.0; 4],
        [1.0; 4],
        Some(&mut foreign)
    )
    .is_err());
    assert_eq!(runtime.scratch_buffer_pool_stats(), before);
    assert!(
        runtime.image_ai_pack_pipeline.get().is_none()
            && other.image_ai_pack_pipeline.get().is_none()
    );
    let gray = GpuImage::upload_u8(&runtime, fixture::<1>(17, 19).view()).unwrap();
    let mut profiler = crate::GpuStageProfiler::new(&runtime, 1).unwrap();
    let first = crate::image::kernels::spatial::sobel_gpu_with_profiler(
        &runtime,
        &gray,
        Some(&mut profiler),
    )
    .unwrap();
    let before = runtime.scratch_buffer_pool_stats();
    assert!(pack_ai_chw_gpu_with_profiler(
        &runtime,
        &source,
        1.0,
        [0.0; 4],
        [1.0; 4],
        Some(&mut profiler)
    )
    .is_err());
    assert_eq!(runtime.scratch_buffer_pool_stats(), before);
    assert!(runtime.image_ai_pack_pipeline.get().is_none());
    let report = profiler.collect(&runtime).unwrap();
    assert_eq!(report.stages.len(), 1);
    assert_eq!(report.device_to_host_bytes, 16);
    first.recycle(&runtime);
    gray.recycle(&runtime);
    source.recycle(&runtime);
}

#[test]
fn cold_channel_concurrency_strided_roi_and_live_tensor_reuse_preserve_words() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let stride = 273 * 3 + 13;
    let data: Vec<_> = (0..stride * 269).map(|i| (i * 31 + 7) as u8).collect();
    let view = ImageView::<u8, 3>::new(273, 269, stride, &data)
        .unwrap()
        .subview(ImageRegion::new(7, 3, 257, 259))
        .unwrap();
    let packed: Vec<_> = (0..259).flat_map(|y| view.row(y).unwrap().iter().copied()).collect();
    let expected = cpu::<3>(&packed, 0.25, [0.0; 4], [1.0; 4]);
    let source = GpuImage::upload_u8(&runtime, view).unwrap();
    let outputs = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..4)
            .map(|_| {
                let runtime = &runtime;
                let source = &source;
                scope.spawn(move || {
                    pack_ai_chw_gpu(runtime, source, 0.25, [0.0; 4], [1.0; 4]).unwrap()
                })
            })
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect::<Vec<_>>()
    });
    assert!(runtime.image_ai_pack_pipeline.get().is_some());
    source.recycle(&runtime);
    let source = GpuImage::upload_u8(
        &runtime,
        Image::<u8, 3>::try_new(257, 259, vec![255; 257 * 259 * 3]).unwrap().view(),
    )
    .unwrap();
    for _ in 0..12 {
        pack_ai_chw_gpu(&runtime, &source, 1.0, [0.0; 4], [1.0; 4]).unwrap().recycle(&runtime);
    }
    let runtime = Box::new(runtime);
    for mut tensor in outputs {
        assert_words(&tensor.readback_f32(&runtime).unwrap(), &expected);
        tensor.recycle(&runtime);
    }
    let before = runtime.scratch_buffer_pool_stats();
    for _ in 0..12 {
        pack_ai_chw_gpu(&runtime, &source, 1.0, [0.0; 4], [1.0; 4]).unwrap().recycle(&runtime);
    }
    assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count, before.allocation_count);
    runtime.set_scratch_buffer_pool_budget(0);
    let mut tensor = pack_ai_chw_gpu(&runtime, &source, 1.0, [0.0; 4], [1.0; 4]).unwrap();
    assert!(tensor.readback_f32(&runtime).unwrap().iter().all(|&v| v == 255.0));
    assert!(tensor.readback_f32(&WgpuRuntime::new_headless().unwrap()).is_err());
    tensor.recycle(&runtime);
    source.recycle(&runtime);
    assert_eq!(runtime.scratch_buffer_pool_stats().cached_bytes, 0);
}

#[test]
fn declared_output_limit_failure_happens_before_tensor_work() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let tiny = GpuImage::upload_u8(&runtime, fixture::<4>(1, 1).view()).unwrap();
    let limits = runtime.device().limits();
    let width = limits.max_texture_dimension_2d.min(16_384);
    let limit = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    let height = (limit / (u64::from(width) * 16) + 1) as u32;
    // A metadata-only negative probe avoids allocating a huge source image.
    // The pack call must reject these dimensions before touching the texture.
    let source = GpuImage::from_parts(
        &runtime,
        width,
        height,
        4,
        tiny.texture().clone(),
        tiny.metadata(),
        tiny.receipt().clone(),
    )
    .unwrap();
    let before = runtime.scratch_buffer_pool_stats();
    assert!(pack_ai_chw_gpu(&runtime, &source, 1.0, [0.0; 4], [1.0; 4]).is_err());
    assert_eq!(runtime.scratch_buffer_pool_stats(), before);
    assert!(runtime.image_ai_pack_pipeline.get().is_none());
    // The declared dimensions differ from its backing texture: never pool it.
    drop(source);
    tiny.recycle(&runtime);
}

#[test]
fn full_large_resident_vision_chain_keeps_five_stages_and_explicit_transfers() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    let image = Image::<u8, 3>::try_new(4096, 4096, vec![128; 4096 * 4096 * 3]).unwrap();
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let options = crate::GpuVisionChainOptions {
        width: 4096,
        height: 4096,
        blur_kernel: 1,
        morphology_kernel: 1,
        scale: 1.0,
        mean: [0.0; 4],
        std: [1.0; 4],
        ..Default::default()
    };
    let mut profiler = crate::GpuStageProfiler::new(&runtime, 5).unwrap();
    let mut tensor =
        crate::run_gpu_vision_chain_profiled(&runtime, &source, options, &mut profiler).unwrap();
    assert_eq!(tensor.shape(), [1, 4096, 4096]);
    assert_eq!(tensor.receipt().host_to_device_bytes(), 67_108_864);
    assert_eq!(tensor.receipt().gpu_to_gpu_bytes(), 335_544_320);
    assert_eq!(tensor.receipt().device_to_host_bytes(), 0);
    let report = profiler.collect(&runtime).unwrap();
    assert_eq!(report.stages.len(), 5);
    assert_eq!(report.device_to_host_bytes, 80);
    assert!(tensor.readback_f32(&runtime).unwrap().iter().all(|v| v.to_bits() == 0));
    assert_eq!(tensor.receipt().device_to_host_bytes(), 67_108_864);
    tensor.recycle(&runtime);
    source.recycle(&runtime);
    eprintln!("Verified 16777216 exact zero tensor words in a 4096x4096 five-stage resident chain, 80 explicit query bytes.");
}

#[path = "ai_pack_benchmark.rs"]
mod benchmark;

fn create_probe_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source: &str,
    channels: Option<u32>,
) -> wgpu::ComputePipeline {
    let constants = channels
        .map(|c| std::collections::HashMap::from([("PACK_CHANNELS".to_owned(), f64::from(c))]))
        .unwrap_or_default();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ai-pack-probe-shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gpu-ai-pack-layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gpu-ai-pack-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("pack_ai_chw"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..Default::default()
        },
        cache: None,
    })
}
