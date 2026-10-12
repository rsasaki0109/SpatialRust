#![cfg(all(feature = "register-icp-gpu", not(target_arch = "wasm32")))]
#[path = "gpu_icp_reduction.rs"]
mod reference;
use spatialrust_core::{HasPositions3, PointCloud, PointCloudBuilder};
use spatialrust_gpu::{GpuIcpInput, WgpuRuntime};
use spatialrust_math::{Isometry3, Quat, Vec3};
use spatialrust_registration::transform_point_cloud;
use std::time::Instant;

fn fixture(n: usize, kind: usize) -> PointCloud {
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..n {
        let j = if kind == 2 { i / 2 } else { i };
        let p = if kind == 1 {
            [(j % 128) as f32 * 0.5, (j / 128) as f32 * 0.5, 0.0]
        } else {
            [(j % 32) as f32 * 0.5, (j / 32 % 32) as f32 * 0.5, (j / 1024) as f32 * 0.5]
        };
        builder.push_point(p).unwrap();
    }
    builder.build().unwrap()
}

#[test]
#[ignore = "completed ICP reduction benchmark; run this exact test alone in release mode"]
fn completed_resident_icp_calls() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    println!("GPU_ICP_REDUCTION_ENV backend={} device_type={}", info.backend, info.device_type);
    for n in [257, 4097, 16384, 65537] {
        for kind in 0..3 {
            let cloud = fixture(n, kind);
            let target = fixture(n + if kind == 2 { 17 } else { 0 }, kind);
            let source = transform_point_cloud(
                &cloud,
                Isometry3::new(Quat::<f32>::identity(), Vec3::new(0.03125, -0.015625, 0.0078125)),
            )
            .unwrap();
            let registration = reference::registration(1.0);
            let expected_cpu = registration.align_with_trace(&source, &target).unwrap();
            let expected_legacy =
                registration.align_with_gpu_correspondences(&runtime, &source, &target).unwrap();
            let expected_gpu =
                registration.align_with_gpu_reduction(&runtime, &source, &target).unwrap();
            assert_eq!(expected_cpu, expected_legacy.diagnostics);
            let maximum = reference::compare_pose(
                &source,
                expected_cpu.result.transform,
                expected_gpu.diagnostics.result.transform,
                1.0,
            );
            assert_eq!(expected_gpu.diagnostics.result.iterations, 4);
            assert_eq!(expected_gpu.evaluation_batches, 5);
            let (sx, sy, sz) = source.positions3().unwrap();
            let (tx, ty, tz) = target.positions3().unwrap();
            let source_positions: Vec<_> = (0..n).map(|i| [sx[i], sy[i], sz[i]]).collect();
            let tree = spatialrust_search::KdTree::from_slices(tx, ty, tz);
            let input = GpuIcpInput::upload(&runtime, &source_positions, &tree).unwrap();
            let resident_pose = expected_gpu.diagnostics.result.transform;
            let expected_moments = input
                .moments_resident(&runtime, resident_pose, 0.25)
                .unwrap()
                .download(&runtime)
                .unwrap();
            for _ in 0..3 {
                assert_eq!(registration.align_with_trace(&source, &target).unwrap(), expected_cpu);
                assert_eq!(
                    registration
                        .align_with_gpu_correspondences(&runtime, &source, &target)
                        .unwrap(),
                    expected_legacy
                );
                assert_eq!(
                    registration.align_with_gpu_reduction(&runtime, &source, &target).unwrap(),
                    expected_gpu
                );
                assert_eq!(
                    input
                        .moments_resident(&runtime, resident_pose, 0.25)
                        .unwrap()
                        .download(&runtime)
                        .unwrap(),
                    expected_moments
                );
            }
            let (mut cpu_ms, mut legacy_ms, mut gpu_ms, mut resident_ms) =
                (vec![], vec![], vec![], vec![]);
            for round in 0..20 {
                for offset in 0..4 {
                    let stage = (round + offset) % 4;
                    let start = Instant::now();
                    match stage {
                        0 => {
                            let output = registration.align_with_trace(&source, &target).unwrap();
                            cpu_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output, expected_cpu);
                        }
                        1 => {
                            let output = registration
                                .align_with_gpu_correspondences(&runtime, &source, &target)
                                .unwrap();
                            legacy_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output, expected_legacy);
                        }
                        2 => {
                            let output = registration
                                .align_with_gpu_reduction(&runtime, &source, &target)
                                .unwrap();
                            gpu_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output, expected_gpu);
                        }
                        _ => {
                            let output =
                                input.moments_resident(&runtime, resident_pose, 0.25).unwrap();
                            runtime.wait_idle();
                            resident_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output.download(&runtime).unwrap(), expected_moments);
                        }
                    }
                }
            }
            println!("GPU_ICP_REDUCTION_BENCH n={n} kind={kind} iterations=4 max_pose_error={maximum} cpu_fitness={} gpu_fitness={} matches={} legacy_h2d={} legacy_d2h={} gpu_h2d={} gpu_d2h={} cpu_ms={cpu_ms:?} legacy_ms={legacy_ms:?} gpu_ms={gpu_ms:?} resident_ms={resident_ms:?}",
            expected_cpu.result.fitness,expected_gpu.diagnostics.result.fitness,expected_moments.correspondences,
            expected_legacy.transfers.host_to_device_bytes(),expected_legacy.transfers.device_to_host_bytes(),expected_gpu.transfers.host_to_device_bytes(),expected_gpu.transfers.device_to_host_bytes());
        }
    }
}
