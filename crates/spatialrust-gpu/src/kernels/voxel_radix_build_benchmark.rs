use super::*;
use std::{hint::black_box, time::Instant};

fn first_passes(
    r: &WgpuRuntime,
    radix: &old_radix::ParentRadixPipelines,
    build: &old_radix::ParentBuildPipelines,
    data: &[Entry],
) -> [Vec<f64>; 2] {
    let n = data.len();
    let cached = n.div_ceil(256) > 2048;
    let fixtures = [raw(r, radix, build, data, false), raw(r, radix, build, data, true)];
    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        let mut profiler = GpuStageProfiler::new(r, 7).unwrap();
        let mut encoder = r.device().create_command_encoder(&Default::default());
        for offset in 0..2 {
            let variant = (round + offset) % 2;
            encode_raw(
                r,
                radix,
                build,
                &fixtures[variant],
                n,
                cached,
                variant == 1,
                &mut encoder,
                &mut Some(&mut profiler),
            );
        }
        r.queue().submit(Some(encoder.finish()));
        let report = profiler.collect(r).unwrap();
        assert_eq!(report.device_to_host_bytes, 112);
        assert_eq!(report.stages.len(), 7);
        let mut cursor = 0;
        for offset in 0..2 {
            let variant = (round + offset) % 2;
            let length = if variant == 0 { 4 } else { 3 };
            let stages = &report.stages[cursor..cursor + length];
            assert_eq!(
                stages.iter().filter(|s| s.stage == "voxel-sort-build").count(),
                usize::from(variant == 0)
            );
            if round >= 3 {
                samples[variant]
                    .push(stages.iter().map(|s| s.elapsed_ns).sum::<f64>() / 1_000_000.);
            }
            cursor += length;
        }
        assert_eq!(cursor, 7);
    }
    for (variant, fixture) in fixtures.iter().enumerate() {
        verify_raw(r, fixture, data, variant == 1);
    }
    samples
}

fn measure(
    r: &WgpuRuntime,
    radix: &old_radix::ParentRadixPipelines,
    build: &old_radix::ParentBuildPipelines,
    n: usize,
    pattern: usize,
) {
    let data = keys(n, pattern);
    let source = buffer(r, bytemuck::cast_slice(&data));
    let cpu_keys: Vec<_> = data
        .iter()
        .map(|e| (i64::from(e[0] as i32), i64::from(e[1] as i32), i64::from(e[2] as i32)))
        .collect();
    let expected = cpu_segments(&data);
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        for offset in 0..4 {
            let variant = offset ^ (round % 4);
            let full = variant >= 2;
            let current = variant % 2 == 1;
            let start = Instant::now();
            let output = if full {
                None
            } else if current {
                Some(
                    build_voxel_segments_gpu_from_keys_buffer(
                        black_box(r),
                        black_box(&source),
                        n as u32,
                        n as u32,
                    )
                    .unwrap(),
                )
            } else {
                Some(
                    old_sort::build_voxel_segments_gpu_from_keys_buffer(
                        radix,
                        build,
                        black_box(r),
                        black_box(&source),
                        n as u32,
                        n as u32,
                    )
                    .unwrap(),
                )
            };
            let downloaded = if full {
                Some(if current {
                    build_voxel_segments_gpu(black_box(r), black_box(&cpu_keys)).unwrap()
                } else {
                    old_sort::build_voxel_segments_gpu(
                        radix,
                        build,
                        black_box(r),
                        black_box(&cpu_keys),
                    )
                    .unwrap()
                })
            } else {
                None
            };
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            let downloaded = downloaded
                .unwrap_or_else(|| output.as_ref().unwrap().to_voxel_segments(r).unwrap());
            assert_eq!(downloaded, expected, "n={n},pattern={pattern},variant={variant}");
            if let Some(output) = &output {
                assert_eq!(output.construction_readback_bytes(), 4);
            }
            if round >= 3 {
                samples[variant].push(elapsed);
            }
        }
    }
    let pass = first_passes(r, radix, build, &data);
    println!("VOXEL_RADIX_BUILD_RECORD {{\"count\":{n},\"pattern\":{pattern},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?},\"pass_old_ms\":{:?},\"pass_ms\":{:?}}}", samples[0], samples[1], samples[2], samples[3], pass[0], pass[1]);
}

#[test]
#[ignore = "paired first digit, completed resident sort/compact and CPU-key full calls; run alone in release mode"]
fn benchmark_complete_voxel_radix_build_calls() {
    let r =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    assert!(r.timestamp_queries_enabled());
    let radix = old_radix::ParentRadixPipelines::new(r.device());
    let build = old_radix::ParentBuildPipelines::new(r.device());
    println!("VOXEL_RADIX_BUILD_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 wall_order=balanced first_gpu_order=alternating checks_drop_excluded=true", r.adapter_info().backend, r.adapter_info().device_type, std::thread::available_parallelism().map_or(1, |n|n.get()));
    for (n, pattern) in [
        (2, 0),
        (256, 2),
        (257, 2),
        (65_537, 0),
        (65_537, 1),
        (524_288, 0),
        (524_289, 0),
        (1_000_003, 0),
        (1_000_003, 1),
        (1_000_003, 2),
    ] {
        measure(&r, &radix, &build, n, pattern);
    }
}
