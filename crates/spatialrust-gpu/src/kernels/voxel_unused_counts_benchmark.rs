use super::*;
use std::{hint::black_box, time::Instant};

fn write_passes(
    r: &WgpuRuntime,
    prior: &old::ParentCompactPipelines,
    data: &[[u32; 4]],
    expected: &crate::VoxelSegments,
) -> [Vec<f64>; 2] {
    let current = &r.pipelines().voxel_compact;
    let pipelines = [&prior.write, &current.write];
    let raw = [
        raw_dispatch(r, &prior.bind_group_layout, data, expected, false),
        raw_dispatch(r, &current.bind_group_layout, data, expected, true),
    ];
    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        let mut profiler = GpuStageProfiler::new(r, 2).unwrap();
        let mut encoder = r.device().create_command_encoder(&Default::default());
        // The parent's atomic counts require reset; it is outside pass timestamps.
        encoder.clear_buffer(&raw[0].counts, 0, Some(data.len().max(1) as u64 * 4));
        for offset in 0..2 {
            let route = (round + offset) % 2;
            let mut slot = Some(&mut profiler);
            let writes = crate::profiling::timestamp_writes(
                &mut slot,
                if route == 0 { "unused_counts_parent" } else { "unused_counts_current" },
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: writes,
            });
            pass.set_pipeline(pipelines[route]);
            pass.set_bind_group(0, &raw[route].group, &[]);
            pass.dispatch_workgroups((data.len() as u32).div_ceil(256), 1, 1);
        }
        r.queue().submit(Some(encoder.finish()));
        let report = profiler.collect(r).unwrap();
        assert_eq!(report.device_to_host_bytes, 32);
        assert_eq!(report.stages.len(), 2);
        if round >= 3 {
            for stage in report.stages {
                samples[usize::from(stage.stage == "unused_counts_current")]
                    .push(stage.elapsed_ns / 1_000_000.);
            }
        }
    }
    verify_raw(r, &raw[0], data, expected, false);
    verify_raw(r, &raw[1], data, expected, true);
    samples
}

fn measure(r: &WgpuRuntime, prior: &old::ParentCompactPipelines, n: usize, run: usize) {
    let (data, expected) = entries(n, run);
    let input = buffer(r, bytemuck::cast_slice(&data));
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        for offset in 0..4 {
            // Both routes measure fifteen pairs in each order after three warmups.
            let variant = offset ^ (round % 4);
            let full = variant >= 2;
            let current = variant % 2 == 1;
            let start = Instant::now();
            let uploaded =
                full.then(|| buffer(black_box(r), bytemuck::cast_slice(black_box(&data))));
            let source = uploaded.as_ref().unwrap_or(&input);
            let output = if current {
                compact_voxel_segments_gpu_buffers(black_box(r), black_box(source), n as u32)
                    .unwrap()
            } else {
                old::previous(prior, black_box(r), black_box(source), n as u32, None).unwrap()
            };
            // Both calls synchronously read the same four-byte cell-count scalar.
            let downloaded = if full { Some(output.to_voxel_segments(r).unwrap()) } else { None };
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            let downloaded = downloaded.unwrap_or_else(|| output.to_voxel_segments(r).unwrap());
            assert!(downloaded == expected, "count={n},run={run},variant={variant}");
            assert_eq!(output.construction_readback_bytes(), 4);
            if round >= 3 {
                samples[variant].push(elapsed);
            }
        }
    }
    let pass = write_passes(r, prior, &data, &expected);
    println!("VOXEL_UNUSED_COUNTS_RECORD {{\"count\":{n},\"run_length\":{run},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?},\"pass_old_ms\":{:?},\"pass_ms\":{:?}}}",samples[0],samples[1],samples[2],samples[3],pass[0],pass[1]);
}

#[test]
#[ignore = "paired write-pass, completed compactor and upload/compact/readback; run alone in release mode"]
fn benchmark_complete_voxel_unused_counts_calls() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    assert!(r.timestamp_queries_enabled());
    let prior = old::ParentCompactPipelines::new(r.device());
    println!("VOXEL_UNUSED_COUNTS_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 routes=write_gpu_resident_compactor_and_full wall_order=balanced checks_drop_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (n, run) in [
        (1, 1),
        (256, 4),
        (257, 4),
        (65537, 1),
        (65537, 257),
        (65537, 65537),
        (1_000_003, 1),
        (1_000_003, 4),
        (1_000_003, 257),
        (1_000_003, 1_000_003),
        (2_000_003, 0),
        (2_000_003, 2_000_003),
    ] {
        measure(&r, &prior, n, run);
    }
}
