#![cfg(all(feature = "gpu-wgpu", not(target_arch = "wasm32")))]

use spatialrust_gpu::{GpuKdTree, GpuQueryBatch, WgpuRuntime};
use spatialrust_search::{KdTree, NearestNeighborIndex, RadiusSearchIndex};
use std::time::Instant;

#[test]
#[ignore = "completed and resident neighborhood benchmark; run alone in release mode"]
fn completed_and_resident_neighborhoods() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    println!(
        "GPU_NEIGHBORHOOD_ENV backend={} device_type={} cpu_workers=1",
        info.backend, info.device_type
    );
    for n in [257, 4097, 16384, 65537] {
        for geometry in 0..2 {
            let points: Vec<_> = (0..n)
                .map(|i| {
                    let j = n - i - 1;
                    if geometry == 0 {
                        [(j % 41) as f32 * 0.5, (j / 41 % 41) as f32 * 0.5, (j / 1681) as f32 * 0.5]
                    } else {
                        [j as f32 * 0.5, 0.0, 0.0]
                    }
                })
                .collect();
            let xyz: [Vec<_>; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).collect());
            let tree = KdTree::from_slices(&xyz[0], &xyz[1], &xyz[2]);
            let target = GpuKdTree::upload(&runtime, &tree).unwrap();
            let queries: Vec<_> =
                points.iter().map(|p| [p[0] + 0.0625, p[1] - 0.03125, p[2]]).collect();
            let resident_queries = GpuQueryBatch::upload(&runtime, &queries).unwrap();
            for kind in 0..2 {
                let cpu = || {
                    queries
                        .iter()
                        .map(|q| {
                            if kind == 0 {
                                tree.nearest_k(q[0], q[1], q[2], 16)
                            } else {
                                tree.radius_search(q[0], q[1], q[2], 0.75)
                            }
                        })
                        .collect::<Vec<_>>()
                };
                let run = |batch: &GpuQueryBatch| {
                    if kind == 0 {
                        target.nearest_k_resident(&runtime, batch, 16).unwrap()
                    } else {
                        target.radius_neighbors_resident(&runtime, batch, 0.75, 32).unwrap()
                    }
                };
                let gpu = || {
                    let batch = GpuQueryBatch::upload(&runtime, &queries).unwrap();
                    run(&batch).download(&runtime).unwrap().into_complete_rows().unwrap()
                };
                let expected = cpu();
                for _ in 0..3 {
                    assert_eq!(gpu(), expected);
                    assert_eq!(
                        run(&resident_queries)
                            .download(&runtime)
                            .unwrap()
                            .into_complete_rows()
                            .unwrap(),
                        expected
                    );
                }
                let (mut cpu_ms, mut gpu_ms, mut resident_ms) =
                    (Vec::new(), Vec::new(), Vec::new());
                for round in 0..20 {
                    for slot in 0..3 {
                        let stage = (round + slot) % 3;
                        let start = Instant::now();
                        if stage == 2 {
                            let output = run(&resident_queries);
                            runtime.wait_idle();
                            resident_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(
                                output.download(&runtime).unwrap().into_complete_rows().unwrap(),
                                expected
                            );
                        } else {
                            let rows = if stage == 0 { cpu() } else { gpu() };
                            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                            if stage == 0 {
                                cpu_ms.push(elapsed)
                            } else {
                                gpu_ms.push(elapsed)
                            }
                            assert_eq!(rows, expected);
                        }
                    }
                }
                println!("GPU_NEIGHBORHOOD_BENCH n={n} geometry={geometry} kind={kind} cpu_ms={cpu_ms:?} gpu_ms={gpu_ms:?} resident_ms={resident_ms:?}");
            }
        }
    }
}
