#![cfg(all(feature = "filter-mls-gpu", not(target_arch = "wasm32")))]

#[path = "gpu_mls.rs"]
mod reference;

use spatialrust_core::HasPositions3;
use spatialrust_filtering::{GpuMlsPositionResult, MlsConfig, MlsSmoothing, PointCloudFilter};
use spatialrust_gpu::{GpuMlsInput, WgpuRuntime};
use spatialrust_math::Vec3;
use std::time::Instant;

#[test]
#[ignore = "completed MLS benchmark; run this exact test alone in release mode"]
fn completed_mls_calls() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    println!("GPU_MLS_ENV backend={} device_type={}", info.backend, info.device_type);
    for n in [257, 4097, 16384, 65537] {
        for kind in 0..3 {
            let order = if kind == 0 { 1 } else { 2 };
            let capacity = if kind == 2 { 96 } else { 64 };
            let config =
                MlsConfig { search_radius: 0.16, polynomial_order: order, min_neighbors: 6 };
            let cloud = reference::fixture_with_width(n, kind, 128);
            let smoother = MlsSmoothing::new(config);
            let cpu_expected = smoother.filter(&cloud).unwrap();
            let cpu_positions = smoother.smoothed_positions(&cloud).unwrap();
            let gpu_expected = smoother.filter_with_gpu(&runtime, &cloud, capacity).unwrap();
            let (x, y, z) = gpu_expected.cloud.positions3().unwrap();
            let positions: Vec<_> =
                x.iter().zip(y).zip(z).map(|((&x, &y), &z)| Vec3::new(x, y, z)).collect();
            let audited = GpuMlsPositionResult {
                positions: positions.clone(),
                modes: gpu_expected.modes.clone(),
                diagnostics: gpu_expected.diagnostics,
                transfers: gpu_expected.transfers,
            };
            let (mode_reference_error, requested_cpu_error) =
                reference::audit(&cloud, config, &audited, &cpu_positions);
            let tree = spatialrust_search::KdTree::from_point_cloud(&cloud).unwrap();
            let input = GpuMlsInput::upload(&runtime, &tree).unwrap();
            let rows = input.neighborhoods_resident(&runtime, 0.16, capacity).unwrap();
            let resident_expected = input
                .smooth_resident(&runtime, &rows, order, 6)
                .unwrap()
                .download(&runtime)
                .unwrap();
            for _ in 0..3 {
                assert_eq!(smoother.filter(&cloud).unwrap(), cpu_expected);
                assert_eq!(
                    smoother.filter_with_gpu(&runtime, &cloud, capacity).unwrap(),
                    gpu_expected
                );
                assert_eq!(
                    input
                        .smooth_resident(&runtime, &rows, order, 6)
                        .unwrap()
                        .download(&runtime)
                        .unwrap(),
                    resident_expected
                );
            }
            let (mut cpu_ms, mut gpu_ms, mut resident_ms) = (vec![], vec![], vec![]);
            for round in 0..20 {
                for offset in 0..3 {
                    let stage = (round + offset) % 3;
                    let start = Instant::now();
                    if stage == 2 {
                        let output = input.smooth_resident(&runtime, &rows, order, 6).unwrap();
                        runtime.wait_idle();
                        resident_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                        assert_eq!(output.download(&runtime).unwrap(), resident_expected);
                    } else if stage == 0 {
                        let output = smoother.filter(&cloud).unwrap();
                        cpu_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                        assert_eq!(output, cpu_expected);
                    } else {
                        let output = smoother.filter_with_gpu(&runtime, &cloud, capacity).unwrap();
                        gpu_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                        assert_eq!(output, gpu_expected);
                    }
                }
            }
            let workers = spatialrust_search::parallel_worker_count(n);
            let d = gpu_expected.diagnostics;
            println!("GPU_MLS_BENCH n={n} kind={kind} order={order} workers={workers} capacity={capacity} unchanged={} plane={} polynomial={} max_mode_reference_error={mode_reference_error} max_requested_cpu_error={requested_cpu_error} cpu_ms={cpu_ms:?} gpu_ms={gpu_ms:?} resident_ms={resident_ms:?}",d.unchanged,d.plane_fallback,d.polynomial);
        }
    }
}
