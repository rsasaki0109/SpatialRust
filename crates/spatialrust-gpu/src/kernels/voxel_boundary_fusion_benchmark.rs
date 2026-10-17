use super::*;
use std::{hint::black_box, time::Instant};

fn scan_passes(
    r: &WgpuRuntime,
    compact: &old::ParentCompactPipelines,
    scan: &old_scan::ParentScanPipelines,
    data: &[[u32; 4]],
) -> [Vec<f64>; 2] {
    let n = data.len() as u32;
    let passes = scan_pass_count(n);
    let current = PrefixScanPipelines::new(r.device());
    let raw = [
        fixture(r, data, compact, &scan.layout, false),
        fixture(r, data, compact, &current.layout, true),
    ];
    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        let mut profiler = GpuStageProfiler::new(r, passes * 2 + 1).unwrap();
        let mut encoder = r.device().create_command_encoder(&Default::default());
        let mut temporaries = crate::scratch::ScratchBuffers::new(r);
        for offset in 0..2 {
            let route = (round + offset) % 2;
            encode_hierarchy(
                r,
                compact,
                scan,
                &raw[route],
                n,
                route == 1,
                &mut encoder,
                &mut Some(&mut profiler),
                &mut temporaries,
            );
        }
        temporaries.submit_and_recycle(encoder);
        let report = profiler.collect(r).unwrap();
        assert_eq!(report.device_to_host_bytes, u64::from(passes * 2 + 1) * 16);
        assert_eq!(report.stages.len(), (passes * 2 + 1) as usize);
        let mut cursor = 0;
        for offset in 0..2 {
            let route = (round + offset) % 2;
            let length = (passes + u32::from(route == 0)) as usize;
            let stages = &report.stages[cursor..cursor + length];
            assert_eq!(
                stages.iter().filter(|s| s.stage == "voxel-compact-mark").count(),
                usize::from(route == 0)
            );
            if round >= 3 {
                samples[route].push(stages.iter().map(|s| s.elapsed_ns).sum::<f64>() / 1_000_000.);
            }
            cursor += length;
        }
        assert_eq!(cursor, report.stages.len());
    }
    let expected = prefix_oracle(data);
    verify_scan(r, &raw[0], data, &expected, false);
    verify_scan(r, &raw[1], data, &expected, true);
    samples
}

fn measure(
    r: &WgpuRuntime,
    prior: &old::ParentCompactPipelines,
    scan: &old_scan::ParentScanPipelines,
    n: usize,
    run: usize,
) {
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
                old::previous(prior, scan, black_box(r), black_box(source), n as u32, None).unwrap()
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
    let pass = scan_passes(r, prior, scan, &data);
    println!("VOXEL_BOUNDARY_FUSION_RECORD {{\"count\":{n},\"run_length\":{run},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?},\"pass_old_ms\":{:?},\"pass_ms\":{:?}}}",samples[0],samples[1],samples[2],samples[3],pass[0],pass[1]);
}

#[test]
#[ignore = "paired mark/scan hierarchy, completed compactor and upload/compact/readback; run alone in release mode"]
fn benchmark_complete_voxel_boundary_fusion_calls() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    assert!(r.timestamp_queries_enabled());
    let prior = old::ParentCompactPipelines::new(r.device());
    let scan = old_scan::ParentScanPipelines::new(r.device());
    println!("VOXEL_BOUNDARY_FUSION_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 routes=mark_scan_gpu_resident_compactor_and_full wall_order=balanced checks_drop_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
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
        measure(&r, &prior, &scan, n, run);
    }
}
