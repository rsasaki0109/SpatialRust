#![cfg(all(feature = "register-gicp-gpu", not(target_arch = "wasm32")))]
#[path = "gpu_gicp_reduction.rs"]
mod reference;
use spatialrust_core::{HasPositions3, PointCloud, PointCloudBuilder};
use spatialrust_gpu::{GpuGicpInput, WgpuRuntime};
use spatialrust_math::{Isometry3, Quat, Vec3};
use spatialrust_registration::{transform_point_cloud, GicpConfig, GicpRegistration};
use std::time::Instant;

fn fixture(n: usize, profile: usize) -> PointCloud {
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..n {
        let j = if profile == 2 { i / 2 } else { i };
        builder
            .push_point([
                (j % 32) as f32 * 0.5,
                (j / 32 % 32) as f32 * 0.5,
                (j / 1024) as f32 * 0.5,
            ])
            .unwrap();
    }
    builder.build().unwrap()
}

#[test]
#[ignore = "completed resident grid covariance GICP benchmark; run this exact test alone in release mode"]
fn completed_resident_grid_covariance_gicp_calls() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    println!(
        "GPU_GICP_RESIDENT_GRID_COVARIANCES_ENV backend={} device_type={}",
        info.backend, info.device_type
    );
    let limits = runtime.device().limits();
    println!("GPU_GICP_RESIDENT_GRID_COVARIANCES_LIMITS storage_bytes={} workgroups_per_dimension={} storage_bindings={}",limits.max_storage_buffer_binding_size,limits.max_compute_workgroups_per_dimension,limits.max_storage_buffers_per_shader_stage);
    // `host` retains CPU matching/equations/solve; its radius covariances use
    // the legacy shared GPU runtime. It is not a pure CPU radius benchmark.
    for n in [257, 4097, 16384, 65537] {
        for profile in 0..3 {
            let cloud = fixture(n, profile);
            let target = fixture(n + if profile == 2 { 17 } else { 0 }, profile);
            let source = transform_point_cloud(
                &cloud,
                Isometry3::new(Quat::<f32>::identity(), Vec3::new(0.03125, -0.015625, 0.0078125)),
            )
            .unwrap();
            let registration = GicpRegistration::new(GicpConfig {
                max_iterations: 3,
                max_correspondence_distance: 0.25,
                k_neighbors: 20,
                transformation_epsilon: 0.0,
                fitness_epsilon: 0.0,
                covariance_radius: Some([0.51, 0.8, 1.05][profile]),
                ..Default::default()
            });
            let host = registration.align_with_diagnostics(&source, &target).unwrap();
            let legacy = registration.align_with_gpu_reduction(&runtime, &source, &target).unwrap();
            let bounded = registration
                .align_with_gpu_resident_radius_covariances(
                    &runtime,
                    &source,
                    &target,
                    [32, 64, 128][profile],
                )
                .unwrap();
            let gpu = registration
                .align_with_gpu_resident_grid_covariances(
                    &runtime,
                    &source,
                    &target,
                    [32, 64, 128][profile],
                )
                .unwrap();
            assert_eq!(gpu.result.iterations, 3);
            assert_eq!(gpu.evaluation_batches, 4);
            assert_eq!(gpu.covariance_batches, 2);
            let maximum =
                reference::compare_pose(&source, host.transform, gpu.result.transform, 1.0);
            reference::compare_pose(&source, legacy.result.transform, gpu.result.transform, 1.0);
            let (sx, sy, sz) = source.positions3().unwrap();
            let (tx, ty, tz) = target.positions3().unwrap();
            let tree = spatialrust_search::KdTree::from_slices(tx, ty, tz);
            let input = GpuGicpInput::upload_with_gpu_grid_covariances(
                &runtime,
                (sx, sy, sz),
                &tree,
                [0.51, 0.8, 1.05][profile],
                [32, 64, 128][profile],
                0.001,
            )
            .unwrap();
            let cells = [&source, &target]
                .iter()
                .map(|c| {
                    let (x, y, z) = c.positions3().unwrap();
                    spatialrust_search::grid_bounds(x, y, z, [0.51, 0.8, 1.05][profile])
                        .unwrap()
                        .1
                        .into_iter()
                        .map(u64::from)
                        .product::<u64>()
                })
                .sum::<u64>();
            assert_eq!(
                gpu.transfers.host_to_device_bytes(),
                20 * source.len() as u64
                    + 36 * target.len() as u64
                    + 32 * tree.layout().nodes().len() as u64
                    + 4 * (cells + 2)
            );
            assert_eq!(gpu.transfers.device_to_host_bytes(), 1024);
            let expected = input
                .equations_resident(&runtime, gpu.result.transform, 0.25)
                .unwrap()
                .download(&runtime)
                .unwrap();
            assert_eq!(expected.correspondences as usize, source.len());
            reference::compare_pose(&source, bounded.result.transform, gpu.result.transform, 1.0);
            for _ in 0..3 {
                assert_eq!(
                    registration
                        .align_with_gpu_resident_radius_covariances(
                            &runtime,
                            &source,
                            &target,
                            [32, 64, 128][profile]
                        )
                        .unwrap(),
                    bounded
                );
                assert_eq!(registration.align_with_diagnostics(&source, &target).unwrap(), host);
                assert_eq!(
                    registration.align_with_gpu_reduction(&runtime, &source, &target).unwrap(),
                    legacy
                );
                assert_eq!(
                    registration
                        .align_with_gpu_resident_grid_covariances(
                            &runtime,
                            &source,
                            &target,
                            [32, 64, 128][profile]
                        )
                        .unwrap(),
                    gpu
                );
                assert_eq!(
                    input
                        .equations_resident(&runtime, gpu.result.transform, 0.25)
                        .unwrap()
                        .download(&runtime)
                        .unwrap(),
                    expected
                );
            }
            let (mut host_ms, mut legacy_ms, mut bounded_ms, mut gpu_ms, mut resident_ms) =
                (vec![], vec![], vec![], vec![], vec![]);
            for round in 0..20 {
                for offset in 0..5 {
                    let stage = (round + offset) % 5;
                    let start = Instant::now();
                    match stage {
                        0 => {
                            let output =
                                registration.align_with_diagnostics(&source, &target).unwrap();
                            host_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output, host);
                        }
                        1 => {
                            let output = registration
                                .align_with_gpu_reduction(&runtime, &source, &target)
                                .unwrap();
                            legacy_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output, legacy);
                        }
                        2 => {
                            let output = registration
                                .align_with_gpu_resident_radius_covariances(
                                    &runtime,
                                    &source,
                                    &target,
                                    [32, 64, 128][profile],
                                )
                                .unwrap();
                            bounded_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output, bounded);
                        }
                        3 => {
                            let output = registration
                                .align_with_gpu_resident_grid_covariances(
                                    &runtime,
                                    &source,
                                    &target,
                                    [32, 64, 128][profile],
                                )
                                .unwrap();
                            gpu_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output, gpu);
                        }
                        _ => {
                            let output = input
                                .equations_resident(&runtime, gpu.result.transform, 0.25)
                                .unwrap();
                            runtime.wait_idle();
                            resident_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output.download(&runtime).unwrap(), expected);
                        }
                    }
                }
            }
            println!("GPU_GICP_RESIDENT_GRID_COVARIANCES_BENCH n={n} profile={profile} iterations=3 max_pose_error={maximum} radius={} capacity={} host_fitness={} gpu_fitness={} matches={} legacy_h2d={} legacy_d2h={} bounded_h2d={} bounded_d2h={} gpu_h2d={} gpu_d2h={} host_ms={host_ms:?} legacy_ms={legacy_ms:?} bounded_ms={bounded_ms:?} gpu_ms={gpu_ms:?} resident_ms={resident_ms:?}",[0.51,0.8,1.05][profile],[32,64,128][profile],host.fitness,gpu.result.fitness,expected.correspondences,legacy.transfers.host_to_device_bytes(),legacy.transfers.device_to_host_bytes(),bounded.transfers.host_to_device_bytes(),bounded.transfers.device_to_host_bytes(),gpu.transfers.host_to_device_bytes(),gpu.transfers.device_to_host_bytes());
        }
    }
}
