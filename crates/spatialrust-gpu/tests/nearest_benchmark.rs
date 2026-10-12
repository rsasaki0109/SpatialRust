#![cfg(all(feature = "gpu-wgpu", not(target_arch = "wasm32")))]

use spatialrust_gpu::{GpuKdTree, WgpuRuntime};
use spatialrust_search::KdTree;
use std::time::Instant;

#[test]
#[ignore = "native completed-query benchmark; run alone in release mode"]
fn completed_nearest_queries() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    println!("GPU_NEAREST_ENV backend={} device_type={}", info.backend, info.device_type);
    for n in [257, 4097, 16384, 65537] {
        for kind in 0..3 {
            let points: Vec<[f32; 3]> = (0..n)
                .map(|i| match kind {
                    0 => [(i % 41) as f32, (i / 41 % 41) as f32, (i / 1681) as f32],
                    1 => [(n - i) as f32 * 0.5, 0.0, 0.0],
                    _ => [(i % 7) as f32, (i / 7 % 3) as f32, 0.0],
                })
                .collect();
            let xyz: [Vec<_>; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).collect());
            let tree = KdTree::from_slices(&xyz[0], &xyz[1], &xyz[2]);
            let start = Instant::now();
            let target = GpuKdTree::upload(&runtime, &tree).unwrap();
            runtime.wait_idle();
            let upload_ms = start.elapsed().as_secs_f64() * 1000.0;
            let queries: Vec<_> =
                points.iter().map(|p| [p[0] + 0.125, p[1] - 0.0625, p[2]]).collect();
            let cpu = || {
                queries
                    .iter()
                    .map(|p| tree.nearest_one_within(p[0], p[1], p[2], 0.25))
                    .collect::<Vec<_>>()
            };
            let gpu = || target.nearest_one_within_download(&runtime, &queries, 0.25).unwrap().0;
            let expected = cpu();
            for _ in 0..3 {
                assert_eq!(gpu(), expected);
            }
            let mut cpu_ms = Vec::new();
            let mut gpu_ms = Vec::new();
            for pair in 0..20 {
                for run in 0..2 {
                    let is_gpu = (pair + run) % 2 == 0;
                    let start = Instant::now();
                    let result = if is_gpu { gpu() } else { cpu() };
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    if is_gpu {
                        gpu_ms.push(elapsed);
                    } else {
                        cpu_ms.push(elapsed);
                    }
                    assert_eq!(result, expected);
                }
            }
            println!("GPU_NEAREST_BENCH n={n} kind={kind} upload_ms={upload_ms:.6} cpu_ms={cpu_ms:?} gpu_ms={gpu_ms:?}");
        }
    }
}
