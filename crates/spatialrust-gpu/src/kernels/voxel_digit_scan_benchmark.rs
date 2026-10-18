use super::*;
use std::{hint::black_box, time::Instant};

fn measure(r: &WgpuRuntime, n: usize, p: usize) {
    let data = scan_fixture(n, p);
    let oracle = expected(&data);
    let entries = prepare_cpu_sort_entries(&data);
    let mask = plan(&entries);
    assert_eq!(mask, old_digits::plan(&entries));
    let mut wall: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    let mut host: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    r.wait_idle();
    for round in 0..43 {
        for offset in 0..4 {
            let variant = offset ^ (round % 4);
            let current = variant % 2 == 1;
            let full = variant >= 2;
            let start = Instant::now();
            let elapsed = if full {
                let out = if current {
                    build_voxel_segments_gpu(black_box(r), black_box(&data))
                } else {
                    parent::build_voxel_segments_gpu(black_box(r), black_box(&data))
                }
                .unwrap();
                let ms = start.elapsed().as_secs_f64() * 1000.;
                assert_eq!(out, oracle);
                ms
            } else {
                let out = if current {
                    build_voxel_segments_gpu_from_keys(black_box(r), black_box(&data))
                } else {
                    parent::build_voxel_segments_gpu_from_keys(black_box(r), black_box(&data))
                }
                .unwrap();
                r.wait_idle();
                let ms = start.elapsed().as_secs_f64() * 1000.;
                assert_eq!(out.construction_readback_bytes(), 4);
                assert_eq!(out.to_voxel_segments(r).unwrap(), oracle);
                ms
            };
            if round >= 3 {
                wall[variant].push(elapsed);
            }
        }
        for offset in 0..2 {
            let current = (round + offset) % 2 == 1;
            let start = Instant::now();
            let prepared = prepare_cpu_sort_entries(black_box(&data));
            let selected = black_box(if current {
                plan(black_box(&prepared))
            } else {
                old_digits::plan(black_box(&prepared))
            });
            let ms = start.elapsed().as_secs_f64() * 1000.;
            assert_eq!(selected, mask);
            assert_eq!(
                bytemuck::cast_slice::<VoxelSortEntry, u32>(&prepared),
                bytemuck::cast_slice::<VoxelSortEntry, u32>(&entries)
            );
            if round >= 3 {
                host[usize::from(current)].push(ms);
            }
        }
    }
    println!("VOXEL_SCAN_RECORD {{\"n\":{n},\"pattern\":{p},\"digits\":{},\"build_old_ms\":{:?},\"build_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?},\"host_old_ms\":{:?},\"host_ms\":{:?}}}",mask.count_ones(),wall[0],wall[1],wall[2],wall[3],host[0],host[1]);
}

#[test]
#[ignore = "paired complete resident/full/host paths, run alone in release"]
fn benchmark_voxel_digit_scan() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_SCAN_ENV backend={} device_type={} available_workers={} pairs=40 warmups=3 wall_order=balanced host_order=alternating checks_drop_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (n, p) in [
        (257, 5),
        (65_537, 5),
        (1_000_003, 5),
        (65_537, 21),
        (1_000_003, 21),
        (65_537, 22),
        (1_000_003, 22),
        (65_537, 0),
        (1_000_003, 0),
        (65_537, 6),
        (1_000_003, 6),
        (65_537, 8),
        (1_000_003, 8),
        (65_537, 23),
        (524_287, 23),
        (524_288, 23),
        (524_289, 23),
        (1_000_003, 23),
        (65_537, 24),
        (1_000_003, 24),
        (1_000_003, 29),
        (1_000_003, 30),
        (257, 7),
        (65_537, 25),
    ] {
        measure(&r, n, p);
    }
}

#[test]
#[ignore = "fresh runtime/device per variant, excludes runtime creation, no explicit driver cache reset; run alone in release"]
fn benchmark_voxel_digit_scan_cold() {
    println!("VOXEL_SCAN_COLD_ENV pairs=20 warmups=0 fresh_runtime_per_variant=true driver_cache_reset=false runtime_creation_checks_drop_excluded=true");
    for (n, p) in [(257, 5), (65_537, 22), (65_537, 23), (65_537, 24), (65_537, 6), (524_289, 23)] {
        let data = scan_fixture(n, p);
        let oracle = expected(&data);
        let entries = prepare_cpu_sort_entries(&data);
        let mask = plan(&entries);
        assert_eq!(mask, old_digits::plan(&entries));
        let count = mask.count_ones();
        let mut wall: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..20 {
            for offset in 0..2 {
                let current = (round + offset) % 2 == 1;
                let r = WgpuRuntime::new_headless().unwrap();
                let start = Instant::now();
                let out = if current {
                    build_voxel_segments_gpu_from_keys(black_box(&r), black_box(&data))
                } else {
                    parent::build_voxel_segments_gpu_from_keys(black_box(&r), black_box(&data))
                }
                .unwrap();
                r.wait_idle();
                let ms = start.elapsed().as_secs_f64() * 1000.;
                assert_eq!(out.to_voxel_segments(&r).unwrap(), oracle);
                wall[usize::from(current)].push(ms);
            }
        }
        println!("VOXEL_SCAN_COLD_RECORD {{\"n\":{n},\"pattern\":{p},\"digits\":{count},\"old_ms\":{:?},\"ms\":{:?}}}",wall[0],wall[1]);
    }
}
