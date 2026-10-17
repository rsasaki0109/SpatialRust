use super::*;
use std::{hint::black_box, time::Instant};

fn scan_passes(r: &WgpuRuntime, prior: &old::ParentScanPipelines, n: u32) -> [Vec<f64>; 2] {
    let data = values(n as usize, 1);
    let expected = oracle(&data);
    let input = buffer(r, &data);
    let outputs: [_; 2] = std::array::from_fn(|_| buffer(r, &vec![0xdead_beef; n as usize]));
    let passes = scan_pass_count(n);
    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        let mut profiler = GpuStageProfiler::new(r, passes * 2).unwrap();
        let mut encoder = r.device().create_command_encoder(&Default::default());
        let mut temporaries = ScratchBuffers::new(r);
        for offset in 0..2 {
            let route = (round + offset) % 2;
            if route == 0 {
                old::encode_inclusive_scan(
                    prior,
                    r,
                    &mut encoder,
                    &input,
                    &outputs[route],
                    n,
                    &mut Some(&mut profiler),
                    &mut temporaries,
                );
            } else {
                encode_inclusive_scan(
                    r,
                    &mut encoder,
                    &input,
                    &outputs[route],
                    n,
                    &mut Some(&mut profiler),
                    &mut temporaries,
                );
            }
        }
        temporaries.submit_and_recycle(encoder);
        let report = profiler.collect(r).unwrap();
        assert_eq!(report.device_to_host_bytes, u64::from(passes) * 32);
        assert_eq!(report.stages.len(), passes as usize * 2);
        if round >= 3 {
            for (sequence, stages) in report.stages.chunks_exact(passes as usize).enumerate() {
                samples[(round + sequence) % 2]
                    .push(stages.iter().map(|s| s.elapsed_ns).sum::<f64>() / 1_000_000.);
            }
        }
    }
    for output in outputs {
        assert_eq!(read_words(r, &output), expected);
    }
    samples
}

fn measure(r: &WgpuRuntime, prior: &old::ParentScanPipelines, n: usize, run: usize) {
    let (data, expected) = entries(n, run);
    let input = buffer(r, bytemuck::cast_slice(&data));
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        for offset in 0..4 {
            let variant = (round + offset) % 4;
            let full = variant >= 2;
            let current = variant % 2 == 1;
            let start = Instant::now();
            let uploaded =
                full.then(|| buffer(black_box(r), bytemuck::cast_slice(black_box(&data))));
            let source = uploaded.as_ref().unwrap_or(&input);
            let output = if current {
                super::super::super::voxel_compact::compact_voxel_segments_gpu_buffers(
                    black_box(r),
                    black_box(source),
                    n as u32,
                )
                .unwrap()
            } else {
                old_compact::previous(prior, black_box(r), black_box(source), n as u32, None)
                    .unwrap()
            };
            let downloaded = if full { Some(output.to_voxel_segments(r).unwrap()) } else { None };
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            let downloaded = downloaded.unwrap_or_else(|| output.to_voxel_segments(r).unwrap());
            assert!(downloaded == expected);
            assert_eq!(output.construction_readback_bytes(), 4);
            if round >= 3 {
                samples[variant].push(elapsed);
            }
        }
    }
    let pass = scan_passes(r, prior, n as u32);
    println!("VOXEL_SCAN_FOUR_RECORD {{\"count\":{n},\"run_length\":{run},\"scan_passes\":{},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?},\"pass_old_ms\":{:?},\"pass_ms\":{:?}}}",scan_pass_count(n as u32),samples[0],samples[1],samples[2],samples[3],pass[0],pass[1]);
}

#[test]
#[ignore = "paired scan hierarchy, completed resident compactor and upload/compact/readback; run alone in release mode"]
fn benchmark_complete_voxel_scan_four_calls() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    assert!(r.timestamp_queries_enabled());
    let prior = old::ParentScanPipelines::new(r.device());
    println!("VOXEL_SCAN_FOUR_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 routes=scan_gpu_resident_compactor_and_full checks_drop_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (n, run) in [
        (1, 1),
        (256, 4),
        (257, 4),
        (65536, 257),
        (65537, 257),
        (1_000_003, 1),
        (1_000_003, 1024),
        (1_000_003, 1_000_003),
    ] {
        measure(&r, &prior, n, run);
    }
}
