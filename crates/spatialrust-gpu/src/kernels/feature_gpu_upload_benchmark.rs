use super::*;
use std::{hint::black_box, time::Instant};

enum Input {
    Fpfh(GpuFpfhInput),
    Iss(GpuIssInput),
    Mls(GpuMlsInput),
}
impl Input {
    fn receipt(&self) -> TransferStats {
        match self {
            Self::Fpfh(v) => v.upload_transfers(),
            Self::Iss(v) => v.upload_transfers(),
            Self::Mls(v) => v.upload_transfers(),
        }
    }
    fn count(&self) -> u32 {
        match self {
            Self::Fpfh(v) => v.point_count(),
            Self::Iss(v) => v.point_count(),
            Self::Mls(v) => v.point_count(),
        }
    }
}

fn shuffled_fixture(n: usize) -> spatialrust_search::KdTree {
    let mut order: Vec<_> = (0..n).collect();
    let mut state = 0x0123_4567_89ab_cdef_u64;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        order.swap(i, (state % (i as u64 + 1)) as usize);
    }
    let axes: [Vec<_>; 3] = std::array::from_fn(|axis| {
        order
            .iter()
            .map(|&i| match axis {
                0 => (i % 32) as f32 * 0.125,
                1 => (i / 32) as f32 * 0.125,
                _ => (i % 7) as f32 * 0.001,
            })
            .collect()
    });
    spatialrust_search::KdTree::from_slices(&axes[0], &axes[1], &axes[2])
}

#[test]
#[ignore = "requires isolated native release measurement"]
fn completed_default_and_gpu_feature_uploads() {
    let r = WgpuRuntime::new_headless().unwrap();
    let info = r.adapter_info();
    let limits = r.device().limits();
    println!("FEATURE_GPU_ENV backend={} device_type={} available_workers={} storage_bytes={} buffer_bytes={} dispatch_groups={} storage_bindings={}", info.backend, info.device_type, std::thread::available_parallelism().map_or(1, |n| n.get()), limits.max_storage_buffer_binding_size, limits.max_buffer_size, limits.max_compute_workgroups_per_dimension, limits.max_storage_buffers_per_shader_stage);
    for (n, profile) in [
        (1, 0),
        (4097, 0),
        (65_537, 0),
        (262_145, 0),
        (524_289, 0),
        (524_289, 1),
        (524_289, 2),
        (1_048_577, 0),
        (524_289, 4),
        (1_048_577, 4),
    ] {
        let tree = if profile == 4 { shuffled_fixture(n) } else { fixture(n, profile).0 };
        let normals = normals(n);
        let nodes = tree.layout().nodes().len() as u64;
        let mut samples: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..6 {
                let route = (round + offset) % 6;
                let start = Instant::now();
                let input = match route {
                    0 => Input::Fpfh(
                        GpuFpfhInput::upload(&r, black_box(&tree), black_box(&normals)).unwrap(),
                    ),
                    1 => Input::Fpfh(
                        GpuFpfhInput::upload_with_gpu_gather(
                            &r,
                            black_box(&tree),
                            black_box(&normals),
                        )
                        .unwrap(),
                    ),
                    2 => Input::Iss(GpuIssInput::upload(&r, black_box(&tree)).unwrap()),
                    3 => Input::Iss(
                        GpuIssInput::upload_with_gpu_gather(&r, black_box(&tree)).unwrap(),
                    ),
                    4 => Input::Mls(GpuMlsInput::upload(&r, black_box(&tree)).unwrap()),
                    _ => Input::Mls(
                        GpuMlsInput::upload_with_gpu_gather(&r, black_box(&tree)).unwrap(),
                    ),
                };
                r.wait_idle();
                let receipt = input.receipt();
                let count = input.count();
                drop(input);
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(count, n as u32);
                let stride = match route {
                    0 => 64,
                    1 => 28,
                    2 | 4 => 32,
                    _ => 16,
                };
                assert_eq!(receipt.host_to_device_bytes(), n as u64 * stride + nodes * 32);
                assert_eq!(
                    receipt.device_to_device_bytes(),
                    n as u64
                        * if route % 2 == 0 {
                            0
                        } else if route == 1 {
                            64
                        } else {
                            32
                        }
                );
                assert_eq!(receipt.device_to_host_bytes(), 0);
                if round >= 3 {
                    samples[route].push(ms);
                }
            }
        }
        println!("FEATURE_GPU_RECORD {{\"points\":{n},\"profile\":{profile},\"nodes\":{nodes},\"fpfh_old_ms\":{:?},\"fpfh_gpu_ms\":{:?},\"iss_old_ms\":{:?},\"iss_gpu_ms\":{:?},\"mls_old_ms\":{:?},\"mls_gpu_ms\":{:?}}}",samples[0],samples[1],samples[2],samples[3],samples[4],samples[5]);
    }
}
