#![cfg(all(feature = "register-icp-point-to-plane-gpu", not(target_arch = "wasm32")))]
#[path = "gpu_point_to_plane_reduction.rs"]
mod reference;
use spatialrust_core::{HasNormals3, HasPositions3, PointCloud, PointCloudBuilder};
use spatialrust_gpu::{GpuPointToPlaneInput, WgpuRuntime};
use spatialrust_math::{Isometry3, Quat, Vec3};
use spatialrust_registration::{transform_point_cloud, PointToPlaneIcp, PointToPlaneIcpConfig};
use std::time::Instant;

fn fixture(n: usize, profile: usize, side: usize) -> PointCloud {
    let mut builder = PointCloudBuilder::new(reference::schema());
    for i in 0..n {
        let index = if profile == 2 { i / 2 } else { i };
        let j = index / 3;
        let a = (j % side) as f32 * 0.25;
        let b = (j / side) as f32 * 0.25;
        let w = if profile == 0 { [1.0; 3] } else { [-1.0 - (j % 5) as f32 * 0.25, 0.5, 2.0] };
        let row = match index % 3 {
            0 => [a, b, 0.0, 0.0, 0.0, w[0]],
            1 => [a, 0.0, b + 0.125, 0.0, w[1], 0.0],
            _ => [0.0, a + 0.125, b + 0.125, w[2], 0.0, 0.0],
        };
        builder.push_point(row).unwrap();
    }
    builder.build().unwrap()
}

#[test]
#[ignore = "completed point-to-plane reduction benchmark; run this exact test alone in release mode"]
fn completed_resident_plane_calls() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    println!("GPU_PLANE_ENV backend={} device_type={}", info.backend, info.device_type);
    for n in [257, 4097, 16384, 65537] {
        let side = ((n as f64 / 3.0).sqrt().ceil() as usize).max(3);
        for profile in 0..3 {
            let original = fixture(n, profile, side);
            let target = fixture(n + if profile == 2 { 17 } else { 0 }, profile, side);
            let source = transform_point_cloud(
                &original,
                Isometry3::new(
                    Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.002),
                    Vec3::new(0.03125, -0.015625, 0.0078125),
                ),
            )
            .unwrap();
            let registration = PointToPlaneIcp::new(PointToPlaneIcpConfig {
                max_iterations: 3,
                max_correspondence_distance: 0.25,
                transformation_epsilon: 0.0,
                fitness_epsilon: 0.0,
                ..Default::default()
            });
            let cpu = registration.align_with_diagnostics(&source, &target).unwrap();
            let gpu = registration.align_with_gpu_reduction(&runtime, &source, &target).unwrap();
            assert_eq!(cpu.iterations, 3);
            assert_eq!(gpu.result.iterations, 3);
            assert_eq!(gpu.evaluation_batches, 4);
            let maximum =
                reference::compare_pose(&source, cpu.transform, gpu.result.transform, 1.0);
            let (sx, sy, sz) = source.positions3().unwrap();
            let positions: Vec<_> = (0..n).map(|i| [sx[i], sy[i], sz[i]]).collect();
            let (tx, ty, tz) = target.positions3().unwrap();
            let tree = spatialrust_search::KdTree::from_slices(tx, ty, tz);
            let (nx, ny, nz) = target.normals3().unwrap();
            let normals: Vec<_> = (0..target.len()).map(|i| [nx[i], ny[i], nz[i]]).collect();
            let input =
                GpuPointToPlaneInput::upload(&runtime, &positions, &tree, &normals).unwrap();
            let expected = input
                .equations_resident(&runtime, gpu.result.transform, 0.25)
                .unwrap()
                .download(&runtime)
                .unwrap();
            for _ in 0..3 {
                assert_eq!(registration.align_with_diagnostics(&source, &target).unwrap(), cpu);
                assert_eq!(
                    registration.align_with_gpu_reduction(&runtime, &source, &target).unwrap(),
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
            let (mut cpu_ms, mut gpu_ms, mut resident_ms) = (vec![], vec![], vec![]);
            for round in 0..20 {
                for offset in 0..3 {
                    let start = Instant::now();
                    match (round + offset) % 3 {
                        0 => {
                            let output =
                                registration.align_with_diagnostics(&source, &target).unwrap();
                            cpu_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output, cpu);
                        }
                        1 => {
                            let output = registration
                                .align_with_gpu_reduction(&runtime, &source, &target)
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
            println!("GPU_PLANE_BENCH n={n} profile={profile} iterations=3 max_pose_error={maximum} cpu_fitness={} gpu_fitness={} matches={} gpu_h2d={} gpu_d2h={} cpu_ms={cpu_ms:?} gpu_ms={gpu_ms:?} resident_ms={resident_ms:?}",cpu.fitness,gpu.result.fitness,expected.correspondences,gpu.transfers.host_to_device_bytes(),gpu.transfers.device_to_host_bytes());
        }
    }
}
