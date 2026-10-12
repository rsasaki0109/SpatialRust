#![cfg(all(feature = "feature-iss-gpu", not(target_arch = "wasm32")))]

#[path = "gpu_iss.rs"]
mod reference;

use spatialrust_features::{IssKeypointConfig, IssKeypointDetector};
use spatialrust_gpu::{GpuIssInput, WgpuRuntime};
use std::time::Instant;

#[test]
#[ignore = "completed ISS benchmark; run this exact test alone in release mode"]
fn completed_iss_calls() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    println!("GPU_ISS_ENV backend={} device_type={}", info.backend, info.device_type);
    for n in [257, 4097, 16384, 65537] {
        for kind in 0..3 {
            let radius = if kind == 0 { 0.11 } else { 0.16 };
            let salient_capacity = if kind == 2 { 96 } else { 64 };
            let non_max_capacity = 64;
            let cloud = reference::fixture(n, kind);
            let config = IssKeypointConfig::with_radii(radius, radius * 0.75);
            let detector = IssKeypointDetector::new(config);
            let cpu_expected = detector.detect(&cloud).unwrap();
            let tree = spatialrust_search::KdTree::from_point_cloud(&cloud).unwrap();
            let input = GpuIssInput::upload(&runtime, &tree).unwrap();
            let rows = input
                .neighborhoods_resident(
                    &runtime,
                    radius,
                    radius * 0.75,
                    salient_capacity,
                    non_max_capacity,
                )
                .unwrap();
            let checked = reference::audit(&runtime, &cloud, config, &input, &rows, &cpu_expected);
            let gpu_expected = detector
                .detect_with_gpu(&runtime, &cloud, salient_capacity, non_max_capacity)
                .unwrap()
                .result;
            assert_eq!(
                gpu_expected.indices,
                checked.indices.iter().map(|&i| i as usize).collect::<Vec<_>>()
            );
            let cpu = || detector.detect(&cloud).unwrap();
            let gpu = || {
                detector
                    .detect_with_gpu(&runtime, &cloud, salient_capacity, non_max_capacity)
                    .unwrap()
                    .result
            };
            for _ in 0..3 {
                reference::compare(&cpu(), &cpu_expected);
                reference::compare(&gpu(), &gpu_expected);
                assert_eq!(
                    input
                        .keypoints_resident(
                            &runtime,
                            &rows,
                            config.gamma_21,
                            config.gamma_32,
                            config.min_neighbors as u32
                        )
                        .unwrap()
                        .download(&runtime)
                        .unwrap()
                        .0,
                    checked.indices
                );
            }
            let (mut cpu_ms, mut gpu_ms, mut resident_ms) = (vec![], vec![], vec![]);
            for round in 0..20 {
                for offset in 0..3 {
                    let stage = (round + offset) % 3;
                    let start = Instant::now();
                    if stage == 2 {
                        let output = input
                            .keypoints_resident(
                                &runtime,
                                &rows,
                                config.gamma_21,
                                config.gamma_32,
                                config.min_neighbors as u32,
                            )
                            .unwrap();
                        runtime.wait_idle();
                        resident_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                        assert_eq!(output.download(&runtime).unwrap().0, checked.indices);
                    } else {
                        let result = if stage == 0 { cpu() } else { gpu() };
                        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                        if stage == 0 {
                            cpu_ms.push(elapsed);
                            reference::compare(&result, &cpu_expected);
                        } else {
                            gpu_ms.push(elapsed);
                            reference::compare(&result, &gpu_expected);
                        }
                    }
                }
            }
            let workers = spatialrust_search::parallel_worker_count(n);
            println!("GPU_ISS_BENCH n={n} kind={kind} workers={workers} salient_capacity={salient_capacity} non_max_capacity={non_max_capacity} cpu_selected={} gpu_selected={} selection_differences={} max_abs_eigen_error={} cpu_ms={cpu_ms:?} gpu_ms={gpu_ms:?} resident_ms={resident_ms:?}",cpu_expected.indices.len(),gpu_expected.indices.len(),checked.differences,checked.max_absolute_eigen_error);
        }
    }
}
