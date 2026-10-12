use super::*;

fn previous_pipelines(runtime: &WgpuRuntime, cached: bool) -> [wgpu::ComputePipeline; 3] {
    source_pipelines(
        runtime,
        cached,
        include_str!("../../tests/fixtures/voxel_radix_ranks_previous.wgsl"),
    )
}

fn source_pipelines(
    runtime: &WgpuRuntime,
    cached: bool,
    source: &str,
) -> [wgpu::ComputePipeline; 3] {
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("radix-ranks-parent"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = runtime.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&runtime.pipelines().voxel_radix.bind_group_layout],
        push_constant_ranges: &[],
    });
    let constants = std::collections::HashMap::from([(
        "cached_prefix".to_owned(),
        if cached { 1.0 } else { 0.0 },
    )]);
    ["histogram", "prefix", "scatter"].map(|entry| {
        runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
}

#[test]
fn word_ranks_match_frozen_and_independent_histograms_on_every_axis() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let old_small = previous_pipelines(&runtime, false);
    let old_large = previous_pipelines(&runtime, true);
    let radix = &runtime.pipelines().voxel_radix;
    let variants = [
        [&old_small[0], &old_small[1], &old_small[2]],
        radix.sort_pipelines(runtime.device(), 1),
        [&old_large[0], &old_large[1], &old_large[2]],
        radix.sort_pipelines(runtime.device(), u32::MAX),
    ];
    let mut comparisons = 0usize;
    for count in [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 513, 65537] {
        for pattern in 0..3 {
            let input = inputs(count, pattern);
            let work = Work::new(&runtime, &input);
            let groups = count.div_ceil(256);
            let dirty = vec![u32::MAX; groups * 16];
            let dirty_ranks = vec![u32::MAX; count];
            for axis in 0usize..3 {
                for shift in [0usize, 4, 28] {
                    let mut counts = vec![0; groups * 16];
                    let mut earlier = vec![0; groups * 16];
                    let mut ranks = Vec::with_capacity(count);
                    for (g, chunk) in input.chunks(256).enumerate() {
                        let mut local = [0u32; 16];
                        for entry in chunk {
                            let bucket = ((entry[axis] ^ 0x80000000) >> shift & 15) as usize;
                            ranks.push(local[bucket]);
                            local[bucket] += 1;
                        }
                        let mut subtotal = 0;
                        for (bucket, mass) in local.into_iter().enumerate() {
                            counts[bucket * groups + g] = mass;
                            earlier[bucket * groups + g] = subtotal;
                            subtotal += mass;
                        }
                    }
                    let bind_index = (2 - axis) * 8 + shift / 4;
                    for (mode, pipelines) in variants.iter().enumerate() {
                        runtime.queue().write_buffer(
                            &work.histogram,
                            0,
                            bytemuck::cast_slice(&dirty),
                        );
                        runtime.queue().write_buffer(
                            &work.offsets,
                            0,
                            bytemuck::cast_slice(&dirty),
                        );
                        runtime.queue().write_buffer(
                            &work.ranks,
                            0,
                            bytemuck::cast_slice(&dirty_ranks),
                        );
                        let mut encoder =
                            runtime.device().create_command_encoder(&Default::default());
                        {
                            let mut pass = encoder.begin_compute_pass(&Default::default());
                            pass.set_pipeline(pipelines[0]);
                            pass.set_bind_group(0, &work.groups[bind_index], &[]);
                            pass.dispatch_workgroups(groups as u32, 1, 1);
                        }
                        runtime.queue().submit(Some(encoder.finish()));
                        assert_eq!(read_words(&runtime, &work.histogram), counts);
                        assert_eq!(
                            read_words(&runtime, &work.offsets),
                            if mode < 2 { &dirty } else { &earlier }[..]
                        );
                        assert_eq!(read_words(&runtime, &work.ranks), ranks, "count={count}, pattern={pattern}, axis={axis}, shift={shift}, mode={mode}");
                        comparisons += counts.len() + earlier.len() + ranks.len();
                    }
                }
            }
        }
    }
    eprintln!("Verified {comparisons} exact histogram/subtotal/rank words against frozen and independent CPU counts.");
}

#[test]
fn complete_word_rank_sorts_preserve_frozen_stable_order_and_scratch_reuse() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let old_small = previous_pipelines(&runtime, false);
    let old_large = previous_pipelines(&runtime, true);
    for count in [257, 524288, 524289, 1000003] {
        for pattern in 0..3 {
            let input = inputs(count, pattern);
            let expected = sorted(&input);
            let work = Work::new(&runtime, &input);
            let old = if count <= 524288 { &old_small } else { &old_large };
            let mut encoder = runtime.device().create_command_encoder(&Default::default());
            work.encode(&mut encoder, [&old[0], &old[1], &old[2]], &mut None);
            runtime.queue().submit(Some(encoder.finish()));
            assert_eq!(
                read_words(&runtime, &work.entries[0]),
                bytemuck::cast_slice::<Entry, u32>(&expected)
            );
            let current = sort_entries_radix_gpu(
                &runtime,
                buffer(&runtime, bytemuck::cast_slice(&input)),
                count as u32,
                None,
            );
            assert_eq!(
                read_words(&runtime, &current),
                bytemuck::cast_slice::<Entry, u32>(&expected)
            );
        }
    }
}

#[path = "voxel_rank_benchmark.rs"]
mod benchmark;

fn previous_sort(
    runtime: &WgpuRuntime,
    entries: wgpu::Buffer,
    point_count: u32,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
    pipelines: [&wgpu::ComputePipeline; 3],
) -> wgpu::Buffer {
    if point_count <= 1 {
        return entries;
    }

    let device = runtime.device();
    let group_count = point_count.div_ceil(WORKGROUP_SIZE);
    let [histogram_pipeline, prefix_pipeline, scatter_pipeline] = pipelines;
    let mut temporaries = crate::scratch::ScratchBuffers::new(runtime);
    let scratch = temporaries.acquire("voxel-radix-scratch", entries.size());
    let histograms =
        temporaries.acquire("voxel-radix-histograms", u64::from(group_count) * RADIX * 4);
    let offsets = temporaries.acquire("voxel-radix-offsets", histograms.size());
    let ranks = temporaries.acquire("voxel-radix-ranks", u64::from(point_count) * 4);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-radix-encoder"),
    });
    let mut input = &entries;
    let mut output = &scratch;

    // Entry construction supplies increasing point indices. Stable passes on
    // z, y, x preserve that order for equal keys without sorting a fourth word.
    for axis in (0..3).rev() {
        for shift in (0..32).step_by(4) {
            let params = RadixParams { point_count, group_count, axis, shift };
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel-radix-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("voxel-radix-bind-group"),
                layout: &runtime.pipelines().voxel_radix.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: histograms.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: offsets.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: ranks.as_entire_binding() },
                ],
            });
            // Separate dispatches provide device-wide producer/consumer
            // ordering. All 24 digits are submitted together without readback.
            for (pipeline, groups, stage) in [
                (histogram_pipeline, group_count, "voxel-radix-histogram"),
                (prefix_pipeline, RADIX as u32, "voxel-radix-prefix"),
                (scatter_pipeline, group_count, "voxel-radix-scatter"),
            ] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("voxel-radix-pass"),
                    timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, stage),
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            std::mem::swap(&mut input, &mut output);
        }
    }
    temporaries.submit_and_recycle(encoder);

    // Twenty-four ping-pong passes finish in the original allocation.
    entries
}
