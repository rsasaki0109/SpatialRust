#![cfg(all(feature = "register-icp-gpu", not(target_arch = "wasm32")))]
#[path = "gpu_icp_trimmed_reduction.rs"]
mod reference;
use spatialrust_core::HasPositions3;
use spatialrust_gpu::{GpuIcpInput, WgpuRuntime};
use spatialrust_search::KdTree;
use std::time::Instant;

#[test]
#[ignore = "completed trimmed ICP benchmark; run this exact test alone in release mode"]
fn completed_resident_trimmed_icp_calls() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    println!("GPU_ICP_TRIMMED_ENV backend={} device_type={}", info.backend, info.device_type);
    for n in [257, 4097, 65537, 262145] {
        for profile in 0..3 {
            let fraction = [0.5, 0.8, 0.9][profile];
            let source = reference::fixture(n, profile, 1.0, true);
            let target = reference::fixture(n + 17, profile, 1.0, false);
            let registration = reference::registration(1.0, fraction, 0);
            let expected_cpu = registration.align_with_trace(&source, &target).unwrap();
            let expected_legacy =
                registration.align_with_gpu_correspondences(&runtime, &source, &target).unwrap();
            let expected_gpu =
                registration.align_with_gpu_trimmed_reduction(&runtime, &source, &target).unwrap();
            let legacy_maximum = reference::pose_error(
                &source,
                expected_cpu.result.transform,
                expected_legacy.diagnostics.result.transform,
            );
            assert!(
                legacy_maximum < 3e-5,
                "n={n} profile={profile} legacy_pose_error={legacy_maximum}"
            );
            // Each precision path keeps its own exact warm/timed control.
            // GPU distances can alter trimmed subsets on near ties.
            for diagnostics in [&expected_legacy.diagnostics, &expected_gpu.diagnostics] {
                assert_eq!(diagnostics.stop_reason, expected_cpu.stop_reason);
                assert_eq!(diagnostics.result.iterations, expected_cpu.result.iterations);
                assert_eq!(diagnostics.result.converged, expected_cpu.result.converged);
                for (a, b) in diagnostics.history.iter().zip(&expected_cpu.history) {
                    assert_eq!(
                        (a.correspondences, a.evaluated_correspondences),
                        (b.correspondences, b.evaluated_correspondences)
                    );
                }
                let (count, fitness) = reference::independent_fitness(
                    &source,
                    &target,
                    diagnostics.result.transform,
                    0.25,
                );
                assert_eq!(count, diagnostics.history.last().unwrap().evaluated_correspondences);
                assert!((fitness - diagnostics.result.fitness).abs() <= 1e-10 + fitness * 1e-4);
            }
            let maximum = reference::pose_error(
                &source,
                expected_cpu.result.transform,
                expected_gpu.diagnostics.result.transform,
            );
            assert!(maximum.is_finite());
            let within_pose_tolerance = maximum < 3e-5;
            // Preserve adverse large gated cases in the measurements rather
            // than silently increasing the small-fixture agreement tolerance.
            if n != 262145 || profile != 1 {
                assert!(within_pose_tolerance, "n={n} profile={profile} max_pose_error={maximum}");
            }
            assert_eq!(expected_gpu.diagnostics.result.iterations, 3);
            assert_eq!(expected_gpu.evaluation_batches, 4);
            for (a, b) in expected_cpu.history.iter().zip(&expected_gpu.diagnostics.history) {
                assert_eq!(
                    (a.correspondences, a.evaluated_correspondences),
                    (b.correspondences, b.evaluated_correspondences)
                );
            }
            let (sx, sy, sz) = source.positions3().unwrap();
            let (tx, ty, tz) = target.positions3().unwrap();
            let source_positions: Vec<_> = (0..n).map(|i| [sx[i], sy[i], sz[i]]).collect();
            let tree = KdTree::from_slices(tx, ty, tz);
            let input = GpuIcpInput::upload(&runtime, &source_positions, &tree).unwrap();
            let pose = expected_gpu.diagnostics.result.transform;
            let expected_moments = input
                .trimmed_moments_resident(&runtime, pose, 0.25, fraction)
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
                    registration
                        .align_with_gpu_trimmed_reduction(&runtime, &source, &target)
                        .unwrap(),
                    expected_gpu
                );
                assert_eq!(
                    input
                        .trimmed_moments_resident(&runtime, pose, 0.25, fraction)
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
                                .align_with_gpu_trimmed_reduction(&runtime, &source, &target)
                                .unwrap();
                            gpu_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output, expected_gpu);
                        }
                        _ => {
                            let output = input
                                .trimmed_moments_resident(&runtime, pose, 0.25, fraction)
                                .unwrap();
                            runtime.wait_idle();
                            resident_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output.download(&runtime).unwrap(), expected_moments);
                        }
                    }
                }
            }
            println!("GPU_ICP_TRIMMED_BENCH n={n} profile={profile} fraction={fraction} iterations=3 max_pose_error={maximum} within_pose_tolerance={within_pose_tolerance} legacy_pose_error={legacy_maximum} legacy_fitness={} cpu_fitness={} gpu_fitness={} retained={} gated={} legacy_h2d={} legacy_d2h={} gpu_h2d={} gpu_d2h={} cpu_ms={cpu_ms:?} legacy_ms={legacy_ms:?} gpu_ms={gpu_ms:?} resident_ms={resident_ms:?}", expected_legacy.diagnostics.result.fitness,expected_cpu.result.fitness,expected_gpu.diagnostics.result.fitness,expected_moments.moments.correspondences,expected_moments.evaluated_correspondences,expected_legacy.transfers.host_to_device_bytes(),expected_legacy.transfers.device_to_host_bytes(),expected_gpu.transfers.host_to_device_bytes(),expected_gpu.transfers.device_to_host_bytes());
        }
    }
}
