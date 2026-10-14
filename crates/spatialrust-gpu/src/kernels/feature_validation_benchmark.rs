use super::*;
use crate::{
    kernels::{fpfh, iss, mls},
    GpuFpfhInput, GpuIssInput, GpuMlsInput, WgpuRuntime,
};
use spatialrust_search::KdTree;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "requires isolated native release measurement"]
fn completed_serial_and_bounded_feature_validation_uploads() {
    let r = WgpuRuntime::new_headless().unwrap();
    let info = r.adapter_info();
    println!("FEATURE_VALIDATION_ENV backend={} device_type={} available_workers={} max_workers={} storage_bytes={} dispatch_groups={}", info.backend, info.device_type, std::thread::available_parallelism().map_or(1, |n| n.get()), MAX_WORKERS, r.device().limits().max_storage_buffer_binding_size, r.device().limits().max_compute_workgroups_per_dimension);
    for (n, profile) in [
        (4097, 0),
        (262_145, 0),
        (524_287, 0),
        (524_288, 0),
        (524_289, 0),
        (1_048_577, 0),
        (524_289, 1),
        (1_048_577, 2),
    ] {
        let axes: [Vec<_>; 3] = std::array::from_fn(|a| {
            (0..n)
                .map(|i| match profile {
                    1 => f32::from_bits(((i + a) as u32 & 1) << 31),
                    2 => f32::from_bits((i as u32 % 23) | (((i + a) as u32 & 1) << 31)),
                    _ => (i % (a * 5 + 17)) as f32 * 0.125,
                })
                .collect()
        });
        let positions = axes.each_ref().map(Vec::as_slice);
        let normals = vec![[0.0, -0.0, 1.0]; n];
        let tree = KdTree::from_slices(&axes[0], &axes[1], &axes[2]);
        let nodes = tree.layout().nodes().len() as u64;
        let mut samples: [Vec<f64>; 10] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for family in 0..5 {
                for offset in 0..2 {
                    let route = (round + offset) % 2;
                    let start = Instant::now();
                    let batches = if family < 2 && n < 524_288 { 32 } else { 1 };
                    match family {
                        0 | 1 => {
                            for _ in 0..batches {
                                let normals =
                                    if family == 1 { Some(normals.as_slice()) } else { None };
                                let accepted = if route == 0 {
                                    previous(black_box(positions), black_box(normals)).is_some()
                                } else {
                                    valid(black_box(positions), black_box(normals))
                                };
                                assert!(black_box(accepted));
                            }
                        }
                        2 => {
                            let input = if route == 0 {
                                fpfh::validation_reference::previous_upload(
                                    &r,
                                    black_box(&tree),
                                    black_box(&normals),
                                    true,
                                )
                            } else {
                                GpuFpfhInput::upload_with_gpu_gather(
                                    &r,
                                    black_box(&tree),
                                    black_box(&normals),
                                )
                            }
                            .unwrap();
                            r.wait_idle();
                            assert_eq!(
                                input.upload_transfers().host_to_device_bytes(),
                                n as u64 * 28 + nodes * 32
                            );
                            assert_eq!(
                                input.upload_transfers().device_to_device_bytes(),
                                n as u64 * 64
                            );
                            assert_eq!(input.upload_transfers().device_to_host_bytes(), 0);
                            assert_eq!(input.point_count(), n as u32);
                            drop(input);
                        }
                        3 => {
                            let input = if route == 0 {
                                iss::validation_reference::previous_upload(
                                    &r,
                                    black_box(&tree),
                                    true,
                                )
                            } else {
                                GpuIssInput::upload_with_gpu_gather(&r, black_box(&tree))
                            }
                            .unwrap();
                            r.wait_idle();
                            assert_eq!(
                                input.upload_transfers().host_to_device_bytes(),
                                n as u64 * 16 + nodes * 32
                            );
                            assert_eq!(
                                input.upload_transfers().device_to_device_bytes(),
                                n as u64 * 32
                            );
                            assert_eq!(input.upload_transfers().device_to_host_bytes(), 0);
                            assert_eq!(input.point_count(), n as u32);
                            drop(input);
                        }
                        _ => {
                            let input = if route == 0 {
                                mls::validation_reference::previous_upload(
                                    &r,
                                    black_box(&tree),
                                    true,
                                )
                            } else {
                                GpuMlsInput::upload_with_gpu_gather(&r, black_box(&tree))
                            }
                            .unwrap();
                            r.wait_idle();
                            assert_eq!(
                                input.upload_transfers().host_to_device_bytes(),
                                n as u64 * 16 + nodes * 32
                            );
                            assert_eq!(
                                input.upload_transfers().device_to_device_bytes(),
                                n as u64 * 32
                            );
                            assert_eq!(input.upload_transfers().device_to_host_bytes(), 0);
                            assert_eq!(input.point_count(), n as u32);
                            drop(input);
                        }
                    }
                    if round >= 3 {
                        samples[family * 2 + route]
                            .push(start.elapsed().as_secs_f64() * 1000.0 / f64::from(batches));
                    }
                }
            }
        }
        println!("FEATURE_VALIDATION_RECORD {{\"points\":{n},\"profile\":{profile},\"nodes\":{nodes},\"hostpos_old_ms\":{:?},\"hostpos_ms\":{:?},\"hostnorm_old_ms\":{:?},\"hostnorm_ms\":{:?},\"fpfh_old_ms\":{:?},\"fpfh_ms\":{:?},\"iss_old_ms\":{:?},\"iss_ms\":{:?},\"mls_old_ms\":{:?},\"mls_ms\":{:?}}}", samples[0],samples[1],samples[2],samples[3],samples[4],samples[5],samples[6],samples[7],samples[8],samples[9]);
    }
}
