use super::*;
use crate::{GpuPointToPlaneInput, WgpuRuntime};
use spatialrust_search::KdTree;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "requires isolated native release measurement"]
fn completed_serial_and_bounded_plane_preparation_uploads() {
    let r = WgpuRuntime::new_headless().unwrap();
    let info = r.adapter_info();
    let limits = r.device().limits();
    println!("PLANE_STAGING_ENV backend={} device_type={} available_workers={} max_workers={} storage_bytes={} dispatch_groups={}",info.backend,info.device_type,std::thread::available_parallelism().map_or(1,|n|n.get()),MAX_WORKERS,limits.max_storage_buffer_binding_size,limits.max_compute_workgroups_per_dimension);
    for (n, profile) in [
        (1, 0),
        (4097, 0),
        (65_537, 0),
        (262_143, 0),
        (262_144, 0),
        (262_145, 0),
        (524_289, 0),
        (1_048_577, 0),
        (524_289, 1),
        (524_289, 2),
    ] {
        let rows = fixture(n, profile);
        let source: Vec<_> = (0..n)
            .map(|i| {
                [(i % 17) as f32 * 0.125, (i / 17 % 17) as f32 * 0.125, (i / 289) as f32 * 0.001]
            })
            .collect();
        let axes: [Vec<_>; 3] = std::array::from_fn(|a| source.iter().map(|p| p[a]).collect());
        let tree = KdTree::from_slices(&axes[0], &axes[1], &axes[2]);
        let mut samples: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for family in 0..3 {
                for offset in 0..2 {
                    let route = (round + offset) % 2;
                    let start = Instant::now();
                    let batches = if family == 0 && n < 262_144 { 32 } else { 1 };
                    if family == 0 {
                        for _ in 0..batches {
                            let prepared = if route == 0 {
                                previous(black_box(&rows))
                            } else {
                                prepare(black_box(&rows))
                            }
                            .unwrap();
                            black_box(&prepared);
                            drop(prepared);
                        }
                    } else {
                        let gpu = family == 2;
                        let input = if route == 0 {
                            super::super::super::preparation_reference::previous_upload(
                                &r,
                                black_box(&source),
                                black_box(&tree),
                                black_box(&rows),
                                gpu,
                            )
                        } else if gpu {
                            GpuPointToPlaneInput::upload_with_gpu_gather(
                                &r,
                                black_box(&source),
                                black_box(&tree),
                                black_box(&rows),
                            )
                        } else {
                            GpuPointToPlaneInput::upload(
                                &r,
                                black_box(&source),
                                black_box(&tree),
                                black_box(&rows),
                            )
                        }
                        .unwrap();
                        r.wait_idle();
                        assert_eq!(input.source_count(), n as u32);
                        let receipt = input.upload_transfers();
                        assert_eq!(
                            receipt.host_to_device_bytes(),
                            n as u64 * if gpu { 48 } else { 64 }
                                + tree.layout().nodes().len() as u64 * 32
                        );
                        assert_eq!(
                            receipt.device_to_device_bytes(),
                            if gpu { n as u64 * 32 } else { 0 }
                        );
                        assert_eq!(receipt.device_to_host_bytes(), 0);
                        drop(input);
                    }
                    let ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(batches);
                    if round >= 3 {
                        samples[family * 2 + route].push(ms);
                    }
                }
            }
        }
        println!("PLANE_STAGING_RECORD {{\"points\":{n},\"profile\":{profile},\"host_old_ms\":{:?},\"host_ms\":{:?},\"default_old_ms\":{:?},\"default_ms\":{:?},\"gather_old_ms\":{:?},\"gather_ms\":{:?}}}",samples[0],samples[1],samples[2],samples[3],samples[4],samples[5]);
    }
}
