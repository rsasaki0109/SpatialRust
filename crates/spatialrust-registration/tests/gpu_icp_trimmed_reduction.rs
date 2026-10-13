#![cfg(all(feature = "register-icp-gpu", not(target_arch = "wasm32")))]
use spatialrust_core::{HasPositions3, PointCloud, PointCloudBuilder};
use spatialrust_gpu::WgpuRuntime;
use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};
use spatialrust_registration::{IcpConfig, IcpConvergenceCriteria, IcpRegistration, IcpStopReason};
use spatialrust_search::KdTree;

pub fn fixture(n: usize, profile: usize, scale: f32, source: bool) -> PointCloud {
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..n {
        let j = if profile == 2 { i / 2 } else { i };
        let mut p = [(j % 32) as f32 * 0.5, (j / 32 % 32) as f32 * 0.5, (j / 1024) as f32 * 0.5];
        if source {
            // Distinct inlier/outlier residuals; well separated nearest targets.
            let residual = if i % 7 == 0 { 0.1875 } else { 0.03125 };
            p[0] += residual;
            p[1] -= 0.015625;
            p[2] += 0.0078125;
            if profile == 1 && i % 11 == 0 {
                p[2] += 0.375;
            }
        }
        builder.push_point(p.map(|v| v * scale)).unwrap();
    }
    builder.build().unwrap()
}

pub fn registration(scale: f32, fraction: f64, stop: usize) -> IcpRegistration {
    IcpRegistration::new(IcpConfig {
        max_iterations: 3,
        max_correspondence_distance: 0.25 * scale,
        ..Default::default()
    })
    .with_trim_fraction(fraction)
    .with_convergence_criteria(IcpConvergenceCriteria {
        translation_epsilon: if stop == 2 { 1e10 } else { 0.0 },
        rotation_epsilon: if stop == 2 { 1e10 } else { 0.0 },
        fitness_epsilon: if stop == 1 { 1e30 } else { 0.0 },
    })
}

pub fn pose_error(source: &PointCloud, a: Isometry3<f32>, b: Isometry3<f32>) -> f32 {
    let (x, y, z) = source.positions3().unwrap();
    (0..source.len())
        .map(|i| {
            let p = Vec3::new(x[i], y[i], z[i]);
            let d = a.transform_point(p) - b.transform_point(p);
            d.x.abs().max(d.y.abs()).max(d.z.abs())
        })
        .fold(0.0, f32::max)
}

pub fn independent_fitness(
    source: &PointCloud,
    target: &PointCloud,
    pose: Isometry3<f32>,
    distance: f32,
) -> (usize, f64) {
    let (sx, sy, sz) = source.positions3().unwrap();
    let (tx, ty, tz) = target.positions3().unwrap();
    let tree = KdTree::from_slices(tx, ty, tz);
    let (mut count, mut sum) = (0, 0.0);
    for i in 0..source.len() {
        let p = pose.transform_point(Vec3::new(sx[i], sy[i], sz[i]));
        if let Some(neighbor) = tree.nearest_one_within(p.x, p.y, p.z, distance * distance) {
            count += 1;
            sum += f64::from(p.x - tx[neighbor.index]).powi(2)
                + f64::from(p.y - ty[neighbor.index]).powi(2)
                + f64::from(p.z - tz[neighbor.index]).powi(2);
        }
    }
    (count, if count == 0 { f64::MAX } else { sum / count as f64 })
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn resident_trimmed_icp_matches_pose_counts_fitness_and_stopping_across_profiles_scales() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let (mut maximum, mut conditions, mut points) = (0.0_f32, 0, 0);
    for n in [17, 63, 64, 65, 257, 4097] {
        for profile in 0..3 {
            for scale in [1e-4, 1.0, 1e4] {
                let source = fixture(n, profile, scale, true);
                let target = fixture(n + 17, profile, scale, false);
                let saved_source = source.clone();
                let saved_target = target.clone();
                for fraction in [0.5, 0.8, 1.0] {
                    for stop in 0..3 {
                        let registration = registration(scale, fraction, stop);
                        let cpu = registration.align_with_trace(&source, &target).unwrap();
                        let gpu = registration
                            .align_with_gpu_trimmed_reduction(&runtime, &source, &target)
                            .unwrap();
                        let error = pose_error(
                            &source,
                            cpu.result.transform,
                            gpu.diagnostics.result.transform,
                        ) / scale;
                        maximum = maximum.max(error);
                        assert!(error<3e-5,"n={n} profile={profile} scale={scale} fraction={fraction} stop={stop} error={error}");
                        assert_eq!(
                            gpu.diagnostics.stop_reason,
                            [
                                IcpStopReason::IterationLimit,
                                IcpStopReason::FitnessThreshold,
                                IcpStopReason::TransformThreshold
                            ][stop]
                        );
                        assert_eq!(gpu.diagnostics.stop_reason, cpu.stop_reason);
                        assert_eq!(gpu.diagnostics.result.iterations, cpu.result.iterations);
                        assert_eq!(gpu.diagnostics.result.converged, cpu.result.converged);
                        for (a, b) in gpu.diagnostics.history.iter().zip(&cpu.history) {
                            assert_eq!(
                                (a.correspondences, a.evaluated_correspondences),
                                (b.correspondences, b.evaluated_correspondences)
                            );
                        }
                        let (count, fitness) = independent_fitness(
                            &source,
                            &target,
                            gpu.diagnostics.result.transform,
                            0.25 * scale,
                        );
                        assert_eq!(
                            count,
                            gpu.diagnostics.history.last().unwrap().evaluated_correspondences
                        );
                        assert!(
                            (fitness - gpu.diagnostics.result.fitness).abs()
                                <= 1e-10 * f64::from(scale).powi(2) + fitness * 1e-4
                        );
                        assert_eq!(gpu.evaluation_batches, gpu.diagnostics.result.iterations + 1);
                        assert_eq!(
                            gpu.transfers.device_to_host_bytes(),
                            176 * gpu.evaluation_batches as u64
                        );
                        let (x, y, z) = target.positions3().unwrap();
                        let tree = KdTree::from_slices(x, y, z);
                        assert_eq!(
                            gpu.transfers.host_to_device_bytes(),
                            (16 * n + 32 * target.len() + 32 * tree.layout().nodes().len()) as u64
                        );
                        if fraction == 1.0 {
                            assert_eq!(
                                gpu,
                                registration
                                    .align_with_gpu_reduction(&runtime, &source, &target)
                                    .unwrap()
                            );
                        }
                        conditions += 1;
                        points += n;
                    }
                }
                assert_eq!(source, saved_source);
                assert_eq!(target, saved_target);
            }
        }
    }
    eprintln!("GPU_ICP_TRIMMED compared {conditions} conditions / {points} source points; maximum scale-normalized CPU pose-coordinate difference {maximum}");
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn resident_trimmed_validation_insufficient_retained_and_shared_runtime_are_preserved() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let target = fixture(65, 0, 1.0, false);
    for fraction in [0.0, 1.1, f64::NAN, f64::INFINITY, f64::from_bits(1), 0.001] {
        let registration = registration(1.0, fraction, 0);
        assert_eq!(
            registration.align_with_trace(&target, &target).unwrap_err().to_string(),
            registration
                .align_with_gpu_trimmed_reduction(&runtime, &target, &target)
                .unwrap_err()
                .to_string()
        );
    }
    for config in [
        IcpConfig { max_iterations: 0, ..Default::default() },
        IcpConfig { min_correspondences: 66, ..Default::default() },
        IcpConfig { max_correspondence_distance: f32::NAN, ..Default::default() },
        IcpConfig { max_correspondence_distance: f32::MAX, ..Default::default() },
        IcpConfig {
            initial_guess: Isometry3::new(Quat::new(0.0, 0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 0.0)),
            ..Default::default()
        },
    ] {
        let registration = IcpRegistration::new(config).with_trim_fraction(0.8);
        assert_eq!(
            registration.align_with_trace(&target, &target).unwrap_err().to_string(),
            registration
                .align_with_gpu_trimmed_reduction(&runtime, &target, &target)
                .unwrap_err()
                .to_string()
        );
    }
    let far = spatialrust_registration::transform_point_cloud(
        &target,
        Isometry3::new(Quat::<f32>::identity(), Vec3::new(100.0, 0.0, 0.0)),
    )
    .unwrap();
    let registration = registration(1.0, 0.8, 0);
    assert_eq!(
        registration.align_with_trace(&far, &target).unwrap_err().to_string(),
        registration
            .align_with_gpu_trimmed_reduction(&runtime, &far, &target)
            .unwrap_err()
            .to_string()
    );
    let source = fixture(257, 0, 1.0, true);
    let target = fixture(274, 0, 1.0, false);
    let expected =
        registration.align_with_gpu_trimmed_reduction(&runtime, &source, &target).unwrap();
    std::thread::scope(|scope| {
        let callers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    assert_eq!(
                        registration
                            .align_with_gpu_trimmed_reduction(&runtime, &source, &target)
                            .unwrap(),
                        expected
                    )
                })
            })
            .collect();
        for caller in callers {
            caller.join().unwrap();
        }
    });
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn resident_trimmed_rotated_initial_guesses_use_absolute_original_positions() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for n in [65, 257, 4097] {
        let target = fixture(n + 17, 0, 1.0, false);
        let source = spatialrust_registration::transform_point_cloud(
            &fixture(n, 0, 1.0, true),
            Isometry3::new(
                Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.003),
                Vec3::new(0.005, -0.002, 0.001),
            ),
        )
        .unwrap();
        let registration = IcpRegistration::new(IcpConfig {
            max_iterations: 3,
            max_correspondence_distance: 0.25,
            initial_guess: Isometry3::new(
                Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), -0.001),
                Vec3::new(-0.003, 0.002, 0.0),
            ),
            ..Default::default()
        })
        .with_trim_fraction(0.8)
        .with_convergence_criteria(IcpConvergenceCriteria {
            translation_epsilon: 0.0,
            rotation_epsilon: 0.0,
            fitness_epsilon: 0.0,
        });
        let expected = registration.align_with_trace(&source, &target).unwrap();
        let actual =
            registration.align_with_gpu_trimmed_reduction(&runtime, &source, &target).unwrap();
        assert!(
            pose_error(&source, expected.result.transform, actual.diagnostics.result.transform)
                < 3e-5
        );
        assert_eq!(actual.diagnostics.stop_reason, expected.stop_reason);
        for (a, b) in actual.diagnostics.history.iter().zip(&expected.history) {
            assert_eq!(
                (a.correspondences, a.evaluated_correspondences),
                (b.correspondences, b.evaluated_correspondences)
            );
        }
    }
}
