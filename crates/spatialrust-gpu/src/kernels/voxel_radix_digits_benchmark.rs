use super::*;
use std::{hint::black_box, time::Instant};

fn measure(r: &WgpuRuntime, n: usize, pattern: usize) {
    let data = fixture(n, pattern);
    let entries = prepare_cpu_sort_entries(&data);
    let mask = plan(&entries);
    let oracle = expected(&data);
    let mut wall: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    r.wait_idle();
    for round in 0..33 {
        for offset in 0..4 {
            let variant = offset ^ (round % 4);
            let current = variant % 2 == 1;
            let full = variant >= 2;
            let start = Instant::now();
            let result = if full {
                let result = if current {
                    build_voxel_segments_gpu(black_box(r), black_box(&data))
                } else {
                    parent::build_voxel_segments_gpu(black_box(r), black_box(&data))
                }
                .unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                assert_eq!(result, oracle);
                elapsed
            } else {
                let result = if current {
                    build_voxel_segments_gpu_from_keys(black_box(r), black_box(&data))
                } else {
                    parent::build_voxel_segments_gpu_from_keys(black_box(r), black_box(&data))
                }
                .unwrap();
                r.wait_idle();
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                assert_eq!(result.construction_readback_bytes(), 4);
                assert_eq!(result.to_voxel_segments(r).unwrap(), oracle);
                elapsed
            };
            if round >= 3 {
                wall[variant].push(result);
            }
        }
    }
    let mut gpu: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    let mut host: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    let mut sorted: Vec<_> =
        entries.iter().map(|e| [e.ix as u32, e.iy as u32, e.iz as u32, e.point_index]).collect();
    sorted.sort_by_key(|e| (e[0] as i32, e[1] as i32, e[2] as i32));
    for round in 0..33 {
        for offset in 0..2 {
            let current = (round + offset) % 2 == 1;
            let source = upload(r, &entries);
            let count = 3 * if current { mask.count_ones() } else { 24 };
            let mut profiler = GpuStageProfiler::new(r, count.max(1)).unwrap();
            let result = if current {
                sort_entries_radix_gpu_with_digits(r, source, n as u32, mask, Some(&mut profiler))
            } else {
                old_radix::sort_entries_radix_gpu(r, source, n as u32, Some(&mut profiler))
            };
            let report = profiler.collect(r).unwrap();
            assert_eq!(report.stages.len(), count as usize);
            assert_eq!(report.device_to_host_bytes, u64::from(count) * 16);
            let elapsed = report.stages.iter().map(|s| s.elapsed_ns).sum::<f64>() / 1_000_000.;
            assert_eq!(read_words(r, &result), bytemuck::cast_slice::<[u32; 4], u32>(&sorted));
            if round >= 3 {
                gpu[usize::from(current)].push(elapsed);
            }
        }
        for offset in 0..2 {
            let current = (round + offset) % 2 == 1;
            let start = Instant::now();
            let prepared = prepare_cpu_sort_entries(black_box(&data));
            let selected = current.then(|| black_box(plan(black_box(&prepared))));
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            if let Some(selected) = selected {
                assert_eq!(selected, mask);
            }
            assert_eq!(
                bytemuck::cast_slice::<VoxelSortEntry, u32>(&prepared),
                bytemuck::cast_slice::<VoxelSortEntry, u32>(&entries)
            );
            if round >= 3 {
                host[usize::from(current)].push(elapsed);
            }
        }
    }
    println!("VOXEL_DIGITS_RECORD {{\"n\":{n},\"pattern\":{pattern},\"digits\":{},\"build_old_ms\":{:?},\"build_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?},\"gpu_old_ms\":{:?},\"gpu_ms\":{:?},\"host_old_ms\":{:?},\"host_ms\":{:?}}}",mask.count_ones(),wall[0],wall[1],wall[2],wall[3],gpu[0],gpu[1],host[0],host[1]);
}

#[test]
#[ignore = "paired complete CPU-key GPU builds/full calls plus host and GPU-stage diagnostics; run alone in release"]
fn benchmark_voxel_constant_digits() {
    let r =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    println!("VOXEL_DIGITS_ENV backend={} device_type={} available_workers={} pairs=30 warmups=3 wall_order=balanced gpu_host_order=alternating checks_drop_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (n, pattern) in [
        (257, 0),
        (257, 1),
        (257, 2),
        (257, 5),
        (257, 9),
        (65_537, 0),
        (65_537, 2),
        (65_537, 3),
        (65_537, 5),
        (65_537, 8),
        (524_287, 2),
        (524_288, 2),
        (524_289, 2),
        (1_000_003, 0),
        (1_000_003, 2),
        (1_000_003, 4),
        (1_000_003, 5),
        (1_000_003, 8),
        (1_000_003, 6),
    ] {
        measure(&r, n, pattern);
    }
}

#[test]
#[ignore = "fresh runtime/device per variant; excludes runtime creation, retained driver cache; run alone in release"]
fn benchmark_voxel_constant_digits_cold() {
    println!("VOXEL_DIGITS_COLD_ENV pairs=20 warmups=0 fresh_runtime_per_variant=true driver_cache_reset=false runtime_creation_checks_drop_excluded=true");
    for (n, pattern) in [(257, 0), (65_537, 2), (65_537, 5), (524_289, 2)] {
        let data = fixture(n, pattern);
        let oracle = expected(&data);
        let count = plan(&prepare_cpu_sort_entries(&data)).count_ones();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..20 {
            for offset in 0..2 {
                let current = (round + offset) % 2 == 1;
                let r = WgpuRuntime::new_headless().unwrap();
                let start = Instant::now();
                let result = if current {
                    build_voxel_segments_gpu_from_keys(black_box(&r), black_box(&data))
                } else {
                    parent::build_voxel_segments_gpu_from_keys(black_box(&r), black_box(&data))
                }
                .unwrap();
                r.wait_idle();
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                assert_eq!(result.to_voxel_segments(&r).unwrap(), oracle);
                assert_eq!(result.construction_readback_bytes(), 4);
                samples[usize::from(current)].push(elapsed);
            }
        }
        println!("VOXEL_DIGITS_COLD_RECORD {{\"n\":{n},\"pattern\":{pattern},\"digits\":{count},\"old_ms\":{:?},\"ms\":{:?}}}",samples[0],samples[1]);
    }
}
