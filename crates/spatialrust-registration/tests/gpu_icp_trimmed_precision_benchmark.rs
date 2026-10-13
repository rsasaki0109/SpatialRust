#![cfg(all(feature = "register-icp-gpu", not(target_arch = "wasm32")))]
#[path = "gpu_icp_trimmed_reduction.rs"]
mod reference;
use spatialrust_core::HasPositions3;
use spatialrust_gpu::{GpuIcpInput, WgpuRuntime};
use spatialrust_registration::GpuIcpTrimPrecision;
use spatialrust_search::KdTree;
use std::time::Instant;

#[test]
#[ignore = "isolated completed-call precision survey; run this exact test alone in release mode"]
fn completed_trimmed_icp_precision_calls() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    println!("GPU_ICP_PRECISE_ENV backend={} device_type={}", info.backend, info.device_type);
    for n in [257, 4097, 65537, 262145] {
        for profile in 0..3 {
            let fraction = [0.5, 0.8, 0.9][profile];
            let source = reference::fixture(n, profile, 1.0, true);
            let target = reference::fixture(n + 17, profile, 1.0, false);
            let registration = reference::registration(1.0, fraction, 0);
            let cpu = registration.align_with_trace(&source, &target).unwrap();
            let legacy =
                registration.align_with_gpu_correspondences(&runtime, &source, &target).unwrap();
            let fast = registration
                .align_with_gpu_trimmed_reduction_with_precision(
                    &runtime,
                    &source,
                    &target,
                    GpuIcpTrimPrecision::CompensatedF32,
                )
                .unwrap();
            let precise =
                registration.align_with_gpu_trimmed_reduction(&runtime, &source, &target).unwrap();
            let fast_error = reference::pose_error(
                &source,
                cpu.result.transform,
                fast.diagnostics.result.transform,
            );
            let precise_error = reference::pose_error(
                &source,
                cpu.result.transform,
                precise.diagnostics.result.transform,
            );
            assert!(precise_error < 3e-5, "n={n} profile={profile} error={precise_error}");
            for diagnostics in [&legacy.diagnostics, &fast.diagnostics, &precise.diagnostics] {
                assert_eq!(diagnostics.stop_reason, cpu.stop_reason);
                assert_eq!(diagnostics.result.iterations, 3);
                assert_eq!(diagnostics.result.converged, cpu.result.converged);
                for (a, b) in diagnostics.history.iter().zip(&cpu.history) {
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
            assert_eq!(fast.transfers, precise.transfers);
            assert_eq!(precise.evaluation_batches, 4);
            let (sx, sy, sz) = source.positions3().unwrap();
            let (tx, ty, tz) = target.positions3().unwrap();
            let positions: Vec<_> = (0..n).map(|i| [sx[i], sy[i], sz[i]]).collect();
            let tree = KdTree::from_slices(tx, ty, tz);
            let input = GpuIcpInput::upload(&runtime, &positions, &tree).unwrap();
            let pose = precise.diagnostics.result.transform;
            let prepared = [GpuIcpTrimPrecision::CompensatedF32, GpuIcpTrimPrecision::DoubleSingle]
                .map(|p| {
                    input
                        .trimmed_moments_resident_with_precision(&runtime, pose, 0.25, fraction, p)
                        .unwrap()
                        .download(&runtime)
                        .unwrap()
                });
            assert_eq!(
                prepared[0].evaluated_correspondences,
                prepared[1].evaluated_correspondences
            );
            assert_eq!(prepared[0].moments.correspondences, prepared[1].moments.correspondences);
            assert_eq!(
                prepared[0].moments.squared_error_sum,
                prepared[1].moments.squared_error_sum
            );
            for _ in 0..3 {
                assert_eq!(registration.align_with_trace(&source, &target).unwrap(), cpu);
                assert_eq!(
                    registration
                        .align_with_gpu_correspondences(&runtime, &source, &target)
                        .unwrap(),
                    legacy
                );
                assert_eq!(
                    registration
                        .align_with_gpu_trimmed_reduction_with_precision(
                            &runtime,
                            &source,
                            &target,
                            GpuIcpTrimPrecision::CompensatedF32
                        )
                        .unwrap(),
                    fast
                );
                assert_eq!(
                    registration
                        .align_with_gpu_trimmed_reduction(&runtime, &source, &target)
                        .unwrap(),
                    precise
                );
                for (index, precision) in
                    [GpuIcpTrimPrecision::CompensatedF32, GpuIcpTrimPrecision::DoubleSingle]
                        .into_iter()
                        .enumerate()
                {
                    assert_eq!(
                        input
                            .trimmed_moments_resident_with_precision(
                                &runtime, pose, 0.25, fraction, precision
                            )
                            .unwrap()
                            .download(&runtime)
                            .unwrap(),
                        prepared[index]
                    );
                }
            }
            let (
                mut cpu_ms,
                mut legacy_ms,
                mut fast_ms,
                mut precise_ms,
                mut prepared_fast_ms,
                mut prepared_precise_ms,
            ) = (vec![], vec![], vec![], vec![], vec![], vec![]);
            for round in 0..20 {
                for offset in 0..6 {
                    let stage = (round + offset) % 6;
                    let start = Instant::now();
                    match stage {
                        0 => {
                            let output = registration.align_with_trace(&source, &target).unwrap();
                            cpu_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output, cpu);
                        }
                        1 => {
                            let output = registration
                                .align_with_gpu_correspondences(&runtime, &source, &target)
                                .unwrap();
                            legacy_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output, legacy);
                        }
                        2 => {
                            let output = registration
                                .align_with_gpu_trimmed_reduction_with_precision(
                                    &runtime,
                                    &source,
                                    &target,
                                    GpuIcpTrimPrecision::CompensatedF32,
                                )
                                .unwrap();
                            fast_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output, fast);
                        }
                        3 => {
                            let output = registration
                                .align_with_gpu_trimmed_reduction(&runtime, &source, &target)
                                .unwrap();
                            precise_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(output, precise);
                        }
                        _ => {
                            let index = stage - 4;
                            let precision = [
                                GpuIcpTrimPrecision::CompensatedF32,
                                GpuIcpTrimPrecision::DoubleSingle,
                            ][index];
                            let output = input
                                .trimmed_moments_resident_with_precision(
                                    &runtime, pose, 0.25, fraction, precision,
                                )
                                .unwrap();
                            runtime.wait_idle();
                            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                            if index == 0 {
                                prepared_fast_ms.push(elapsed);
                            } else {
                                prepared_precise_ms.push(elapsed);
                            }
                            assert_eq!(output.download(&runtime).unwrap(), prepared[index]);
                        }
                    }
                }
            }
            println!("GPU_ICP_PRECISE_BENCH n={n} profile={profile} fraction={fraction} iterations=3 fast_pose_error={fast_error} precise_pose_error={precise_error} cpu_fitness={} fast_fitness={} precise_fitness={} retained={} gated={} legacy_h2d={} legacy_d2h={} precise_h2d={} precise_d2h={} cpu_ms={cpu_ms:?} legacy_ms={legacy_ms:?} fast_ms={fast_ms:?} precise_ms={precise_ms:?} prepared_fast_ms={prepared_fast_ms:?} prepared_precise_ms={prepared_precise_ms:?}",cpu.result.fitness,fast.diagnostics.result.fitness,precise.diagnostics.result.fitness,prepared[1].moments.correspondences,prepared[1].evaluated_correspondences,legacy.transfers.host_to_device_bytes(),legacy.transfers.device_to_host_bytes(),precise.transfers.host_to_device_bytes(),precise.transfers.device_to_host_bytes());
        }
    }
}
