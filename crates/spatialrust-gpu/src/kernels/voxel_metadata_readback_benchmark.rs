use super::*;
use std::{hint::black_box, time::Instant};

fn previous_full(r: &WgpuRuntime, keys: &[(i64, i64, i64)]) -> VoxelSegments {
    let segments = build_voxel_segments_gpu_from_keys(r, keys).unwrap();
    parent_segments(r, &segments)
}

fn measure(r: &WgpuRuntime, n: usize, pattern: usize) {
    let keys = input_keys(n, pattern);
    let expected = build_voxel_segments(&clamped(&keys));
    let segments = build_voxel_segments_gpu_from_keys(r, &keys).unwrap();
    let cells = segments.cell_count() as usize;
    let mut samples: [[Vec<f64>; 2]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    for round in 0..33 {
        for route_offset in 0..2 {
            let route = (round + route_offset) % 2;
            for offset in 0..2 {
                let current = (round + offset) % 2 == 1;
                r.wait_idle();
                let start = Instant::now();
                let output = if route == 0 {
                    if current {
                        black_box(&segments).to_voxel_segments(black_box(r)).unwrap()
                    } else {
                        parent_segments(black_box(r), black_box(&segments))
                    }
                } else if current {
                    build_voxel_segments_gpu(black_box(r), black_box(&keys)).unwrap()
                } else {
                    previous_full(black_box(r), black_box(&keys))
                };
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                assert_eq!(output, expected);
                if round >= 3 {
                    samples[route][usize::from(current)].push(elapsed);
                }
            }
        }
    }
    println!("VOXEL_METADATA_RECORD {{\"count\":{n},\"cells\":{cells},\"pattern\":{pattern},\"batched\":{},\"readback_old_ms\":{:?},\"readback_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?}}}",layout(cells,n,r.device().limits().max_buffer_size).is_some(),samples[0][0],samples[0][1],samples[1][0],samples[1][1]);
}

#[test]
#[ignore = "paired explicit metadata readbacks and complete CPU-key calls; run alone in release mode"]
fn benchmark_complete_voxel_metadata_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_METADATA_ENV backend={} device_type={} available_workers={} max_buffer_size={} staging_cap={} warmups=3 pairs=30 pair_order=alternating route_order=rotating checks_drop_excluded=true parent_build_shared=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()),r.device().limits().max_buffer_size,MAX_BATCH_BYTES);
    for (n, pattern) in [
        (0, 0),
        (1, 0),
        (2, 2),
        (257, 0),
        (65_537, 0),
        (65_537, 1),
        (65_537, 2),
        (1_000_003, 0),
        (1_000_003, 1),
        (1_000_003, 2),
    ] {
        measure(&r, n, pattern);
    }
}
