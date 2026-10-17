use super::*;

#[path = "voxel_radix_four_reference.rs"]
mod old;

struct ParentPipelines {
    small: [wgpu::ComputePipeline; 3],
    cached: [wgpu::ComputePipeline; 3],
}

impl ParentPipelines {
    fn new(r: &WgpuRuntime) -> Self {
        Self::from_source(r, include_str!("voxel_radix_four_reference.wgsl"))
    }

    fn candidate(r: &WgpuRuntime) -> Self {
        Self::from_source(r, include_str!("voxel_radix_four_candidate.wgsl"))
    }

    fn from_source(r: &WgpuRuntime, source: &str) -> Self {
        let module = r.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix-four-fixture"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = r.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&r.pipelines().voxel_radix.bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipelines = |cached: bool| {
            let constants = std::collections::HashMap::from([(
                "cached_prefix".to_owned(),
                if cached { 1.0 } else { 0.0 },
            )]);
            ["histogram", "prefix", "scatter"].map(|entry| {
                r.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    cache: None,
                })
            })
        };
        Self { small: pipelines(false), cached: pipelines(true) }
    }

    fn for_groups(&self, count: u32) -> [&wgpu::ComputePipeline; 3] {
        if count <= 2048 {
            [&self.small[0], &self.small[1], &self.small[2]]
        } else {
            [&self.cached[0], &self.cached[1], &self.small[2]]
        }
    }
}

fn masses(groups: usize, pattern: usize) -> Vec<u32> {
    let mut words = vec![0; groups * 16];
    let last = words.len() - 1;
    match pattern {
        0 => {}
        1 => words[0] = u32::MAX,
        2 => words[last] = u32::MAX,
        3 => {
            let n = words.len() as u32;
            words.fill(u32::MAX / n);
            words[last] += u32::MAX % n;
        }
        _ => {
            let mut remaining = u32::MAX;
            for (i, word) in words.iter_mut().enumerate() {
                *word = remaining.min((i.wrapping_mul(982_451_653) % 65536) as u32);
                remaining -= *word;
            }
            words[last] += remaining;
        }
    }
    words
}

#[test]
fn voxel_radix_four_prefix_matches_complete_parent_u64_offsets_and_guards() {
    assert_eq!(
        include_str!("../shaders/voxel_radix.wgsl"),
        include_str!("voxel_radix_four_reference.wgsl")
    );
    let r = WgpuRuntime::new_headless().unwrap();
    let prior = ParentPipelines::new(&r);
    let candidate = ParentPipelines::candidate(&r);
    let radix = &r.pipelines().voxel_radix;
    let variants = [
        &prior.small[1],
        &radix.prefix,
        &candidate.small[1],
        &prior.cached[1],
        radix.sort_pipelines(r.device(), u32::MAX)[1],
        &candidate.cached[1],
    ];
    let dummy: [_; 3] = std::array::from_fn(|_| buffer(&r, &[0x1234_5678; 4]));
    let mut comparisons = 0usize;
    let mut fixtures = 0usize;
    for groups in [
        1, 3, 4, 63, 64, 65, 255, 256, 257, 511, 512, 513, 2047, 2048, 2049, 4095, 4096, 4097,
        32769,
    ] {
        for pattern in 0..5 {
            let histogram = masses(groups, pattern);
            let mut earlier = vec![0; histogram.len()];
            for g in 0..groups {
                let mut total = 0u64;
                for bucket in 0..16 {
                    earlier[bucket * groups + g] = u32::try_from(total).unwrap();
                    total += u64::from(histogram[bucket * groups + g]);
                }
            }
            let mut total = 0u64;
            let mut expected: Vec<_> = histogram
                .iter()
                .map(|&n| {
                    let before = u32::try_from(total).unwrap();
                    total += u64::from(n);
                    before
                })
                .collect();
            assert_eq!(total, if pattern == 0 { 0 } else { u64::from(u32::MAX) });
            expected.extend([0xdead_beef; 8]);
            let mut guarded_histogram = histogram;
            guarded_histogram.extend([0xcafe_babe; 8]);
            let hist = buffer(&r, &guarded_histogram);
            for (mode, pipeline) in variants.iter().enumerate() {
                let mut initial =
                    if mode < 3 { vec![0xa5a5_a5a5; earlier.len()] } else { earlier.clone() };
                initial.extend([0xdead_beef; 8]);
                let offsets = buffer(&r, &initial);
                let bind = group(
                    &r,
                    RadixParams {
                        point_count: u32::MAX,
                        group_count: groups as u32,
                        axis: 0,
                        shift: 0,
                    },
                    [&dummy[0], &dummy[1], &hist, &offsets, &dummy[2]],
                );
                let mut encoder = r.device().create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(16, 1, 1);
                }
                r.queue().submit(Some(encoder.finish()));
                assert_eq!(
                    read_words(&r, &offsets),
                    expected,
                    "groups={groups} pattern={pattern} mode={mode}"
                );
                assert_eq!(read_words(&r, &hist), guarded_histogram);
                comparisons += expected.len() + guarded_histogram.len();
                fixtures += 1;
            }
        }
    }
    for unchanged in dummy {
        assert_eq!(read_words(&r, &unchanged), [0x1234_5678; 4]);
        comparisons += 4;
    }
    println!("RADIX_FOUR_RAW verified {comparisons} exact offset/histogram/guard words in {fixtures} executions.");
}

fn stable_inputs(count: usize, pattern: usize) -> (Vec<Entry>, Vec<Entry>) {
    let mut input = inputs(count, pattern);
    for (i, entry) in input.iter_mut().enumerate() {
        // The fourth word must follow input tie order, even when decreasing.
        entry[3] = (count - 1 - i) as u32;
    }
    let mut expected = input.clone();
    expected.sort_by_key(|e| (e[0] as i32, e[1] as i32, e[2] as i32));
    (input, expected)
}

#[test]
fn voxel_radix_four_complete_sort_keeps_parent_stability_threshold_and_live_outputs() {
    let r =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    let prior = ParentPipelines::new(&r);
    let candidate = ParentPipelines::candidate(&r);
    let radix = &r.pipelines().voxel_radix;
    assert!(std::ptr::eq(radix.sort_pipelines(r.device(), 2048)[1], &radix.prefix));
    assert!(!std::ptr::eq(radix.sort_pipelines(r.device(), 2049)[1], &radix.prefix));
    let mut retained = Vec::new();
    let mut comparisons = 0usize;
    let mut executions = 0usize;
    for count in [0, 1, 257, 65537, 524287, 524288, 524289, 1000003, 2000003] {
        if count == 2000003 {
            // Mixed sizes can fill the bounded retention budget. Start the
            // largest same-size reuse proof fresh while keeping live outputs.
            r.clear_scratch_buffer_pool();
        }
        for pattern in 0..3 {
            let (input, expected) = stable_inputs(count, pattern);
            for variant in 0..3 {
                let mut data = bytemuck::cast_slice::<Entry, u32>(&input).to_vec();
                // The zero no-op still owns a valid, unchanged sentinel allocation.
                if data.is_empty() {
                    data.extend([0xabcdef01; 4]);
                }
                let entries = buffer(&r, &data);
                let identity = entries.clone();
                let mut profiler = GpuStageProfiler::new(&r, 72).unwrap();
                let output = match variant {
                    0 => old::previous(&prior, &r, entries, count as u32, Some(&mut profiler)),
                    1 => sort_entries_radix_gpu(&r, entries, count as u32, Some(&mut profiler)),
                    _ => old::previous(&candidate, &r, entries, count as u32, Some(&mut profiler)),
                };
                assert_eq!(output, identity);
                let expected_words =
                    if count == 0 { &data } else { bytemuck::cast_slice::<Entry, u32>(&expected) };
                assert_eq!(
                    read_words(&r, &output),
                    expected_words,
                    "count={count} pattern={pattern} variant={variant}"
                );
                let report = profiler.collect(&r).unwrap();
                assert_eq!(report.stages.len(), if count <= 1 { 0 } else { 72 });
                assert_eq!(report.device_to_host_bytes, if count <= 1 { 0 } else { 1152 });
                comparisons += expected_words.len();
                executions += 1;
                if pattern == 0 && [257, 524289, 1000003, 2000003].contains(&count) {
                    retained.push((output, expected_words.to_vec()));
                }
            }
        }
    }
    let (input, expected) = stable_inputs(2000003, 0);
    for current in [false, true] {
        let before = r.scratch_buffer_pool_stats().allocation_count;
        let entries = buffer(&r, bytemuck::cast_slice(&input));
        let output = if current {
            sort_entries_radix_gpu(&r, entries, input.len() as u32, None)
        } else {
            old::previous(&candidate, &r, entries, input.len() as u32, None)
        };
        assert_eq!(read_words(&r, &output), bytemuck::cast_slice::<Entry, u32>(&expected));
        assert_eq!(r.scratch_buffer_pool_stats().allocation_count, before);
        comparisons += expected.len() * 4;
        executions += 1;
    }
    for (output, expected) in retained {
        assert_eq!(read_words(&r, &output), expected);
        comparisons += expected.len();
    }
    println!("RADIX_FOUR_SORT verified {comparisons} exact parent/CPU words in {executions} complete calls plus retained outputs.");
}

#[path = "voxel_radix_four_benchmark.rs"]
mod benchmark;
