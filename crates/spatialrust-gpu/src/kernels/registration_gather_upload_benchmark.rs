use super::*;
use std::{hint::black_box, time::Instant};

fn shuffled_fixture(n: usize) -> (Vec<[f32; 3]>, KdTree) {
    let (source, _, points) = fixture(n, 0);
    let mut order: Vec<_> = (0..n).collect();
    let mut state = 0x734b_d912_aba0_ae13_u64;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        order.swap(i, (state % (i as u64 + 1)) as usize);
    }
    let axes: [Vec<_>; 3] = std::array::from_fn(|a| order.iter().map(|&i| points[i][a]).collect());
    (source, KdTree::from_slices(&axes[0], &axes[1], &axes[2]))
}

#[test]
#[ignore = "requires isolated native release measurement"]
fn completed_default_and_gpu_registration_uploads() {
    let r = WgpuRuntime::new_headless().unwrap();
    let info = r.adapter_info();
    let limits = r.device().limits();
    println!("REGISTRATION_GATHER_ENV backend={} device_type={} available_workers={} storage_bytes={} dispatch_groups={}",info.backend,info.device_type,std::thread::available_parallelism().map_or(1,|n|n.get()),limits.max_storage_buffer_binding_size,limits.max_compute_workgroups_per_dimension);
    for (n, profile) in [
        (1, 0),
        (4097, 0),
        (65_537, 0),
        (262_145, 0),
        (524_289, 0),
        (1_048_577, 0),
        (524_289, 1),
        (524_289, 2),
        (524_289, 4),
        (1_048_577, 4),
    ] {
        let (source, tree) = if profile == 4 {
            shuffled_fixture(n)
        } else {
            let (s, t, _) = fixture(n, profile);
            (s, t)
        };
        let normals = normals(n);
        let cov = covariances(n);
        let nodes = tree.layout().nodes().len();
        let mut samples: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for family in 0..3 {
                for offset in 0..2 {
                    let route = (round + offset) % 2;
                    let start = Instant::now();
                    let receipt = match family {
                        0 => {
                            let input = if route == 0 {
                                icp::gather_reference::previous_upload(
                                    &r,
                                    black_box(&source),
                                    black_box(&tree),
                                )
                            } else {
                                GpuIcpInput::upload_with_gpu_gather(
                                    &r,
                                    black_box(&source),
                                    black_box(&tree),
                                )
                            }
                            .unwrap();
                            r.wait_idle();
                            assert_eq!(input.source_count(), n as u32);
                            let receipt = input.upload_transfers();
                            drop(input);
                            receipt
                        }
                        1 => {
                            let input = if route == 0 {
                                gicp::gather_reference::previous_upload(
                                    &r,
                                    black_box(&source),
                                    black_box(&cov),
                                    black_box(&tree),
                                    black_box(&cov),
                                )
                            } else {
                                GpuGicpInput::upload_with_gpu_gather(
                                    &r,
                                    black_box(&source),
                                    black_box(&cov),
                                    black_box(&tree),
                                    black_box(&cov),
                                )
                            }
                            .unwrap();
                            r.wait_idle();
                            assert_eq!(input.source_count(), n as u32);
                            let receipt = input.upload_transfers();
                            drop(input);
                            receipt
                        }
                        _ => {
                            let input = if route == 0 {
                                plane::gather_reference::previous_upload(
                                    &r,
                                    black_box(&source),
                                    black_box(&tree),
                                    black_box(&normals),
                                )
                            } else {
                                GpuPointToPlaneInput::upload_with_gpu_gather(
                                    &r,
                                    black_box(&source),
                                    black_box(&tree),
                                    black_box(&normals),
                                )
                            }
                            .unwrap();
                            r.wait_idle();
                            assert_eq!(input.source_count(), n as u32);
                            let receipt = input.upload_transfers();
                            drop(input);
                            receipt
                        }
                    };
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    let stride = match family {
                        1 => 112,
                        2 => 64,
                        _ => 48,
                    };
                    assert_eq!(
                        receipt.host_to_device_bytes(),
                        n as u64 * (stride - if route == 1 { 16 } else { 0 }) + nodes as u64 * 32
                    );
                    assert_eq!(
                        receipt.device_to_device_bytes(),
                        if route == 1 { n as u64 * 32 } else { 0 }
                    );
                    assert_eq!(receipt.device_to_host_bytes(), 0);
                    if round >= 3 {
                        samples[family * 2 + route].push(ms);
                    }
                }
            }
        }
        println!("REGISTRATION_GATHER_RECORD {{\"points\":{n},\"profile\":{profile},\"nodes\":{nodes},\"icp_old_ms\":{:?},\"icp_ms\":{:?},\"gicp_old_ms\":{:?},\"gicp_ms\":{:?},\"plane_old_ms\":{:?},\"plane_ms\":{:?}}}",samples[0],samples[1],samples[2],samples[3],samples[4],samples[5]);
    }
}
