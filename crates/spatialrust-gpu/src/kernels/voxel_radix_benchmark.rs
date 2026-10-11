//! Explicit paired old/new 24-digit sort measurement, on one GPU runtime.
use super::*;

#[test]
#[ignore = "native paired radix benchmark; requires explicit timestamp invocation"]
fn paired_previous_current_voxel_radix_timestamps() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    assert!(runtime.timestamp_queries_enabled());
    let current = &runtime.pipelines().voxel_radix;
    let shader = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("previous-radix-test-only"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../tests/fixtures/voxel_radix_previous.wgsl").into(),
        ),
    });
    let layout = runtime.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&current.bind_group_layout],
        push_constant_ranges: &[],
    });
    let previous: [wgpu::ComputePipeline; 3] = ["histogram", "prefix", "scatter"].map(|entry| {
        runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    });
    eprintln!("adapter={:?}; 3 warmups + 20 alternating paired rounds. GPU sums exclude reset/setup/gaps/readback; completed wall intervals include seed-to-input device copy, submission and wait only.",runtime.adapter_info());
    for count in [257, 65_537, 524_288, 524_289, 1_000_003] {
        let variants = [
            [&previous[0], &previous[1], &previous[2]],
            current.sort_pipelines(runtime.device(), (count as u32).div_ceil(256)),
        ];
        for pattern in 0..3 {
            let input = inputs(count, pattern);
            let expected = sorted(&input);
            let seed = buffer(&runtime, bytemuck::cast_slice(&input));
            let work = [Work::new(&runtime, &input), Work::new(&runtime, &input)];
            let mut stages: [Vec<[f64; 3]>; 2] = std::array::from_fn(|_| Vec::new());
            let mut wall: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            for round in 0..23 {
                let mut profiler = GpuStageProfiler::new(&runtime, 144).unwrap();
                for offset in 0..2 {
                    let i = (round + offset) % 2;
                    let mut encoder = runtime.device().create_command_encoder(&Default::default());
                    encoder.copy_buffer_to_buffer(&seed, 0, &work[i].entries[0], 0, seed.size());
                    work[i].encode(&mut encoder, variants[i], &mut Some(&mut profiler));
                    let commands = encoder.finish();
                    let start = std::time::Instant::now();
                    runtime.queue().submit(Some(commands));
                    runtime.wait_idle();
                    if round >= 3 {
                        wall[i].push(start.elapsed().as_secs_f64() * 1000.0);
                    }
                }
                let report = profiler.collect(&runtime).unwrap();
                assert_eq!(report.stages.len(), 144);
                assert_eq!(report.device_to_host_bytes, 2304);
                if round >= 3 {
                    for (offset, chunk) in report.stages.chunks_exact(72).enumerate() {
                        let i = (round + offset) % 2;
                        let mut total = [0.0; 3];
                        for (j, stage) in chunk.iter().enumerate() {
                            total[j % 3] += stage.elapsed_ns / 1000.0;
                        }
                        stages[i].push(total);
                    }
                }
            }
            for w in &work {
                assert_eq!(
                    read_words(&runtime, &w.entries[0]),
                    bytemuck::cast_slice::<Entry, u32>(&expected),
                    "count {count} pattern {pattern}"
                );
            }
            eprintln!("PAIRED {{\"count\":{count},\"pattern\":{pattern},\"reset_device_bytes\":{},\"previous_stage_us\":{:?},\"current_stage_us\":{:?},\"previous_completed_ms\":{:?},\"current_completed_ms\":{:?}}}",seed.size(),stages[0],stages[1],wall[0],wall[1]);
        }
    }
}
