//! Frozen GPU and independent integer references for selector lane dispatch.
use super::*;

pub(super) fn frozen_cache(runtime: &WgpuRuntime) -> TrimPipelines {
    let mut cache = TrimPipelines::new(runtime.device());
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("frozen-trim-selector"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../tests/fixtures/icp_trim_selector_previous.wgsl").into(),
        ),
    });
    let layout = runtime.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&cache.layout],
        push_constant_ranges: &[],
    });
    let [histogram, select] = ["histogram_prefix", "select_digit"].map(|entry| {
        runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    });
    cache.histogram = histogram;
    cache.select = select;
    // Preserve the frozen 64-lane selector even on wide-capable devices.
    cache.wide_select.set(cache.select.clone()).unwrap();
    cache
}

pub(super) fn selection_snapshot(
    runtime: &WgpuRuntime,
    cache: &TrimPipelines,
    matches: &wgpu::Buffer,
    params: TrimParams,
) -> Vec<u32> {
    let selection = selection_resident(runtime, cache, matches, params);
    read(runtime, &selection, 32)
}

fn selection_resident(
    runtime: &WgpuRuntime,
    cache: &TrimPipelines,
    matches: &wgpu::Buffer,
    params: TrimParams,
) -> wgpu::Buffer {
    let selection = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut scratch = ScratchBuffers::new(runtime);
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    cache.encode(runtime, &mut encoder, &mut scratch, params, matches, &selection);
    scratch.submit_and_recycle(encoder);
    // Caller-owned output never enters the scratch pool.
    selection
}

#[test]
fn wide_selection_limits_and_lazy_small_input_path_are_exact() {
    let mut limits = wgpu::Limits::default();
    assert!(!use_wide_selector(63, &limits));
    assert!(use_wide_selector(64, &limits));
    limits.max_compute_invocations_per_workgroup = 255;
    assert!(!use_wide_selector(256, &limits));
    limits.max_compute_invocations_per_workgroup = 256;
    limits.max_compute_workgroup_size_x = 255;
    assert!(!use_wide_selector(256, &limits));
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cache = TrimPipelines::new(runtime.device());
    assert_eq!(cache.selector(runtime.device(), 63), &cache.select);
    assert!(cache.wide_select.get().is_none());
    let _ = cache.selector(runtime.device(), 64);
    assert_eq!(
        cache.wide_select.get().is_some(),
        use_wide_selector(64, &runtime.device().limits())
    );
}

#[test]
fn cold_wide_selector_concurrency_and_dirty_scratch_keep_owned_results() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let current = TrimPipelines::new(runtime.device());
    let previous = frozen_cache(&runtime);
    let inputs: Vec<_> = (0..4)
        .map(|job| {
            let n = 4097 + job * 67;
            let words: Vec<[u32; 2]> = (0..n)
                .map(|i| {
                    [
                        u32::from(i % 11 == 0) * u32::MAX,
                        ((i * 7919 % 1024) as f32 / 1024.0).to_bits(),
                    ]
                })
                .collect();
            let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&words),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let params = TrimParams::new(
                n as u32,
                plan_trim(n as u64, &runtime.device().limits()).unwrap(),
                [0.1, 0.3, 0.8, 0.9][job],
                0.5,
                1,
            );
            let expected = selection_snapshot(&runtime, &previous, &input, params);
            (input, params, expected)
        })
        .collect();
    assert!(current.wide_select.get().is_none());
    let mut held = vec![];
    for _ in 0..3 {
        let results = std::thread::scope(|scope| {
            let jobs: Vec<_> = inputs
                .iter()
                .map(|(input, params, _)| {
                    scope.spawn(|| selection_resident(&runtime, &current, input, *params))
                })
                .collect();
            jobs.into_iter().map(|job| job.join().unwrap()).collect::<Vec<_>>()
        });
        held.extend(results);
    }
    for (index, output) in held.iter().enumerate() {
        assert_eq!(read(&runtime, output, 32), inputs[index % 4].2);
    }
    assert_eq!(
        current.wide_select.get().is_some(),
        use_wide_selector(64, &runtime.device().limits())
    );
}

fn digit_snapshot(
    runtime: &WgpuRuntime,
    cache: &TrimPipelines,
    params: TrimParams,
    histogram: &[u32],
    initial: [u32; 8],
) -> Vec<u32> {
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let matches = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &[0; 8],
        usage: wgpu::BufferUsages::STORAGE,
    });
    let histogram = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(histogram),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let selection = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&initial),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let entries: Vec<_> = [&uniform, &matches, &histogram, &selection]
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &cache.layout,
        entries: &entries,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    // Force the wide pipeline for every group-count boundary when supported.
    dispatch(&mut encoder, cache.selector(runtime.device(), 64), &group, 1, 1);
    runtime.queue().submit(Some(encoder.finish()));
    read(runtime, &selection, 32)
}

#[test]
fn every_digit_matches_frozen_selector_and_u64_prefixes_through_maximum_mass() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let current = TrimPipelines::new(runtime.device());
    let previous = frozen_cache(&runtime);
    let mut comparisons = 0;
    for groups in [1, 2, 63, 64, 65, 255, 256] {
        for profile in 0..4 {
            let histogram: Vec<u32> = (0..groups * 256)
                .map(|i| match profile {
                    0 => 0,
                    1 => {
                        if i % 256 == 173 {
                            u32::MAX / groups + u32::from(i / 256 < u32::MAX % groups)
                        } else {
                            0
                        }
                    }
                    2 => (i * 7919 + i / 256 * 31 + 17) % 1024,
                    _ => u32::from(i % 11 == 0) * 37,
                })
                .collect();
            let bins: Vec<u64> = (0..256)
                .map(|b| (0..groups as usize).map(|g| u64::from(histogram[g * 256 + b])).sum())
                .collect();
            let mass: u64 = bins.iter().sum();
            assert!(mass <= u64::from(u32::MAX));
            for step in 0..8 {
                for fraction in [f64::from_bits(1), 0.3, 0.8, f64::from_bits(1.0f64.to_bits() - 1)]
                {
                    let params =
                        TrimParams { step, ..TrimParams::new(1, groups, fraction, 1.0, 1) };
                    let initial = if step == 0 {
                        [0; 8]
                    } else {
                        [mass as u32, u32::from(mass > 0), (mass / 3) as u32, 0, 0, 0, 0, 0]
                    };
                    let actual = digit_snapshot(&runtime, &current, params, &histogram, initial);
                    let old = digit_snapshot(&runtime, &previous, params, &histogram, initial);
                    assert_eq!(actual, old, "groups={groups} profile={profile} step={step}");
                    let mut expected = initial;
                    if step == 0 {
                        expected[0] = mass as u32;
                        expected[1] = (mass as f64 * fraction).floor() as u32;
                        expected[2] = expected[1].saturating_sub(1);
                    }
                    if expected[1] != 0 {
                        let mut preceding = 0;
                        for (digit, &total) in bins.iter().enumerate() {
                            if u64::from(expected[2]) - preceding < total {
                                expected[2] -= preceding as u32;
                                expected[if step < 4 { 3 } else { 4 }] |=
                                    (digit as u32) << (24 - (step % 4) * 8);
                                break;
                            }
                            preceding += total;
                        }
                    }
                    assert_eq!(
                        actual, expected,
                        "oracle groups={groups} profile={profile} step={step}"
                    );
                    comparisons += 8;
                }
            }
        }
    }
    eprintln!("GPU_ICP_SELECTOR exact_words={comparisons}");
}

#[path = "icp_trim_selector_benchmark.rs"]
mod benchmark;
