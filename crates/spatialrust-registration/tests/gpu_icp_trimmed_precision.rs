#![cfg(all(feature = "register-icp-gpu", not(target_arch = "wasm32")))]
#[path = "gpu_icp_trimmed_reduction.rs"]
mod reference;
use spatialrust_gpu::WgpuRuntime;
use spatialrust_registration::{
    GpuIcpTrimPrecision, IcpConfig, IcpConvergenceCriteria, IcpRegistration,
};

pub fn registration(iterations: usize, fraction: f64) -> IcpRegistration {
    IcpRegistration::new(IcpConfig {
        max_iterations: iterations,
        max_correspondence_distance: 0.25,
        ..Default::default()
    })
    .with_trim_fraction(fraction)
    .with_convergence_criteria(IcpConvergenceCriteria {
        translation_epsilon: 0.0,
        rotation_epsilon: 0.0,
        fitness_epsilon: 0.0,
    })
}

#[test]
#[ignore = "large native GPU precision regressions; run explicitly in release mode"]
fn precision_large_clouds_preserve_pose_counts_fitness_and_iteration_trajectories() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let (mut maximum, mut previous_maximum, mut conditions) = (0.0_f32, 0.0_f32, 0);
    for n in [4097, 65537, 262145] {
        for profile in 0..3 {
            let fraction = [0.5, 0.8, 0.9][profile];
            let source = reference::fixture(n, profile, 1.0, true);
            let target = reference::fixture(n + 17, profile, 1.0, false);
            for iterations in 1..=3 {
                let registration = registration(iterations, fraction);
                let cpu = registration.align_with_trace(&source, &target).unwrap();
                let precise = registration
                    .align_with_gpu_trimmed_reduction(&runtime, &source, &target)
                    .unwrap();
                let previous = registration
                    .align_with_gpu_trimmed_reduction_with_precision(
                        &runtime,
                        &source,
                        &target,
                        GpuIcpTrimPrecision::CompensatedF32,
                    )
                    .unwrap();
                let error = reference::pose_error(
                    &source,
                    cpu.result.transform,
                    precise.diagnostics.result.transform,
                );
                maximum = maximum.max(error);
                previous_maximum = previous_maximum.max(reference::pose_error(
                    &source,
                    cpu.result.transform,
                    previous.diagnostics.result.transform,
                ));
                assert!(
                    error < 3e-5,
                    "n={n} profile={profile} iterations={iterations} error={error}"
                );
                assert_eq!(precise.diagnostics.stop_reason, cpu.stop_reason);
                assert_eq!(precise.diagnostics.result.converged, cpu.result.converged);
                for (a, b) in precise.diagnostics.history.iter().zip(&cpu.history) {
                    assert_eq!(
                        (a.correspondences, a.evaluated_correspondences),
                        (b.correspondences, b.evaluated_correspondences)
                    );
                }
                let (count, fitness) = reference::independent_fitness(
                    &source,
                    &target,
                    precise.diagnostics.result.transform,
                    0.25,
                );
                assert_eq!(
                    count,
                    precise.diagnostics.history.last().unwrap().evaluated_correspondences
                );
                assert!(
                    (fitness - precise.diagnostics.result.fitness).abs() <= 1e-10 + fitness * 1e-4
                );
                assert_eq!(precise.evaluation_batches, iterations + 1);
                assert_eq!(precise.transfers.device_to_host_bytes(), 176 * (iterations + 1) as u64);
                assert_eq!(precise.transfers, previous.transfers);
                conditions += 1;
            }
        }
    }
    eprintln!("GPU_ICP_PRECISE_TRAJECTORIES conditions={conditions} maximum_pose_error={maximum} previous_maximum_pose_error={previous_maximum}");
}

#[test]
#[ignore = "requires a native GPU adapter; run explicitly"]
fn precision_choices_keep_fraction_one_and_mixed_concurrent_results() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let source = reference::fixture(4097, 1, 1.0, true);
    let target = reference::fixture(4114, 1, 1.0, false);
    let untrimmed = registration(3, 1.0);
    let ordinary = untrimmed.align_with_gpu_reduction(&runtime, &source, &target).unwrap();
    for precision in [GpuIcpTrimPrecision::CompensatedF32, GpuIcpTrimPrecision::DoubleSingle] {
        assert_eq!(
            untrimmed
                .align_with_gpu_trimmed_reduction_with_precision(
                    &runtime, &source, &target, precision
                )
                .unwrap(),
            ordinary
        );
    }
    let registration = registration(3, 0.8);
    let expected =
        [GpuIcpTrimPrecision::CompensatedF32, GpuIcpTrimPrecision::DoubleSingle].map(|p| {
            registration
                .align_with_gpu_trimmed_reduction_with_precision(&runtime, &source, &target, p)
                .unwrap()
        });
    assert_eq!(
        expected[1],
        registration.align_with_gpu_trimmed_reduction(&runtime, &source, &target).unwrap()
    );
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|i| {
                let registration = &registration;
                let runtime = &runtime;
                let source = &source;
                let target = &target;
                let expected = &expected;
                scope.spawn(move || {
                    for round in 0..2 {
                        let index = (i + round) % 2;
                        let precision = [
                            GpuIcpTrimPrecision::CompensatedF32,
                            GpuIcpTrimPrecision::DoubleSingle,
                        ][index];
                        assert_eq!(
                            registration
                                .align_with_gpu_trimmed_reduction_with_precision(
                                    runtime, source, target, precision
                                )
                                .unwrap(),
                            expected[index]
                        );
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
    });
}
