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
    let c = segments.cell_count() as usize;
    let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: c.max(1) as u64 * 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut e = r.device().create_command_encoder(&Default::default());
    if c > 0 {
        e.copy_buffer_to_buffer(segments.keys_buffer(), 0, &staging, 0, c as u64 * 16);
    }
    r.queue().submit(Some(e.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    r.wait_idle();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let mapped = bytemuck::cast_slice::<u8, VoxelKeyOutput>(&data[..c * 16]);
    let mut samples: [[Vec<f64>; 2]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    for round in 0..33 {
        for offset in 0..3 {
            let route = (round + offset) % 3;
            for order in 0..2 {
                let current = (round + order) % 2 == 1;
                r.wait_idle();
                let elapsed = if route == 0 {
                    let start = Instant::now();
                    let out = if current {
                        decode(black_box(mapped))
                    } else {
                        parent_decode(black_box(mapped))
                    };
                    let elapsed = start.elapsed().as_secs_f64() * 1000.;
                    assert_eq!(out, expected.keys);
                    elapsed
                } else {
                    let start = Instant::now();
                    let out = if route == 1 {
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
                    assert_eq!(out, expected);
                    elapsed
                };
                if round >= 3 {
                    samples[route][usize::from(current)].push(elapsed);
                }
            }
        }
    }
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("VOXEL_DECODE_RECORD {{\"count\":{n},\"cells\":{c},\"pattern\":{pattern},\"cpu_workers\":{},\"batched\":{},\"decode_old_ms\":{:?},\"decode_ms\":{:?},\"readback_old_ms\":{:?},\"readback_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?}}}",workers(c,available),super::super::super::layout(c,n,r.device().limits().max_buffer_size).is_some(),samples[0][0],samples[0][1],samples[1][0],samples[1][1],samples[2][0],samples[2][1]);
    drop(data);
    staging.unmap();
}

#[test]
#[ignore = "paired mapped key conversion/readback/complete CPU-key calls; run alone in release mode"]
fn benchmark_complete_voxel_decode_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_DECODE_ENV backend={} device_type={} available_workers={} threshold={} max_workers={} warmups=3 pairs=30 pair_order=alternating route_order=rotating checks_drop_excluded=true parent_build_shared=true mapped_key_fixture_retained=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()),MIN_PARALLEL_CELLS,MAX_WORKERS);
    for (n, pattern) in [
        (0, 0),
        (1, 0),
        (65_537, 2),
        (524_287, 2),
        (524_288, 2),
        (524_289, 2),
        (1_000_003, 2),
        (1_048_576, 2),
        (2_000_003, 2),
        (1_000_003, 0),
        (1_000_003, 1),
        (2_796_203, 2),
    ] {
        measure(&r, n, pattern);
    }
}
