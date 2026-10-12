#![cfg(all(feature = "register-gicp-gpu", not(target_arch = "wasm32")))]

use spatialrust_core::{PointCloudBuilder, StandardSchemas};
use spatialrust_gpu::WgpuRuntime;
use spatialrust_registration::{GicpConfig, GicpRegistration};
use std::time::Instant;

#[test]
#[ignore = "native completed-GICP benchmark; run alone in release mode"]
fn completed_gpu_correspondence_gicp() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    println!("GPU_GICP_ENV backend={} device_type={}", info.backend, info.device_type);
    for n in [257, 4097, 16384, 65537] {
        let mut source = PointCloudBuilder::new(StandardSchemas::point_xyz());
        let mut target = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for i in 0..n {
            let p = [(i % 41) as f32 * 0.5, (i / 41 % 41) as f32 * 0.5, (i / 1681) as f32 * 0.5];
            target.push_point(p).unwrap();
            source.push_point([p[0] + 0.03125, p[1] - 0.015625, p[2] + 0.0078125]).unwrap();
        }
        let (source, target) = (source.build().unwrap(), target.build().unwrap());
        for covariance_radius in [None, Some(0.625)] {
            let registration = GicpRegistration::new(GicpConfig {
                max_iterations: 3,
                max_correspondence_distance: 0.25,
                k_neighbors: 6,
                transformation_epsilon: 0.0,
                fitness_epsilon: 0.0,
                covariance_radius,
                ..GicpConfig::default()
            });
            let cpu = || registration.align_with_diagnostics(&source, &target).unwrap();
            let gpu =
                || registration.align_with_gpu_correspondences(&runtime, &source, &target).unwrap();
            let expected = cpu();
            assert_eq!(expected.iterations, 3);
            for _ in 0..3 {
                assert_eq!(gpu().result, expected);
            }
            let mut cpu_ms = Vec::new();
            let mut gpu_ms = Vec::new();
            for pair in 0..20 {
                for run in 0..2 {
                    let is_gpu = (pair + run) % 2 == 0;
                    let start = Instant::now();
                    let result = if is_gpu { gpu().result } else { cpu() };
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    if is_gpu {
                        gpu_ms.push(elapsed);
                    } else {
                        cpu_ms.push(elapsed);
                    }
                    assert_eq!(result, expected);
                }
            }
            println!("GPU_GICP_BENCH n={n} radius={covariance_radius:?} cpu_ms={cpu_ms:?} gpu_ms={gpu_ms:?}");
        }
    }
}
