use super::*;
use std::{hint::black_box, time::Instant};

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
fn completed_two_pass_and_fused_fpfh_uploads() {
    let r = WgpuRuntime::new_headless().unwrap();
    let info = r.adapter_info();
    let limits = r.device().limits();
    println!("FPFH_FUSED_ENV backend={} device_type={} available_workers={} storage_bytes={} dispatch_groups={} storage_bindings={}",info.backend,info.device_type,std::thread::available_parallelism().map_or(1,|n|n.get()),limits.max_storage_buffer_binding_size,limits.max_compute_workgroups_per_dimension,limits.max_storage_buffers_per_shader_stage);
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
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let route = (round + offset) % 2;
                let start = Instant::now();
                let input = if route == 0 {
                    fpfh::two_pass_reference::previous_gpu_upload(
                        &r,
                        black_box(&tree),
                        black_box(&normals),
                    )
                } else {
                    GpuFpfhInput::upload_with_gpu_gather(&r, black_box(&tree), black_box(&normals))
                }
                .unwrap();
                r.wait_idle();
                let receipt = input.upload_transfers();
                let count = input.point_count();
                drop(input);
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(count, n as u32);
                assert_eq!(receipt.host_to_device_bytes(), n as u64 * 28 + nodes * 32);
                assert_eq!(receipt.device_to_device_bytes(), n as u64 * 64);
                assert_eq!(receipt.device_to_host_bytes(), 0);
                if round >= 3 {
                    samples[route].push(ms);
                }
            }
        }
        println!("FPFH_FUSED_RECORD {{\"points\":{n},\"profile\":{profile},\"nodes\":{nodes},\"old_ms\":{:?},\"fused_ms\":{:?}}}",samples[0],samples[1]);
    }
}
