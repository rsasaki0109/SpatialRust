use super::*;
use crate::{transform_point_cloud, GicpConfig};
use spatialrust_core::PointCloudBuilder;
use spatialrust_math::{Quat, Vec3};

fn corner(n: usize, scale: f32) -> PointCloud {
    let mut b = PointCloudBuilder::xyz();
    for i in 0..n {
        let a = (i % 17) as f32 * 0.125 * scale;
        let c = (i / 17) as f32 * 0.125 * scale;
        for p in [
            [a, c, 0.0],
            [a, 0.0, c + 0.0625 * scale],
            [0.0, a + 0.0625 * scale, c + 0.0625 * scale],
        ] {
            b.push_point(p).unwrap();
        }
    }
    b.build().unwrap()
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn parallel_tree_preparation_preserves_complete_gicp_diagnostics_exactly() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for n in [17, 63, 64, 65, 257, 10923] {
        for scale in [1e-4_f32, 1.0, 1e4] {
            let target = corner(n, scale);
            let source = transform_point_cloud(
                &target,
                Isometry3::new(
                    Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.002),
                    Vec3::new(0.01 * scale, -0.005 * scale, 0.008 * scale),
                ),
            )
            .unwrap();
            for mode in 0..4 {
                for stop in 0..3 {
                    let registration = GicpRegistration::new(GicpConfig {
                        max_iterations: 3,
                        max_correspondence_distance: 0.25 * scale,
                        k_neighbors: 20,
                        covariance_radius: if mode >= 2 { Some(0.25 * scale) } else { None },
                        transformation_epsilon: if stop == 1 { scale.max(1.0) as f64 } else { 0.0 },
                        fitness_epsilon: if stop == 2 { f64::from(scale).powi(2) } else { 0.0 },
                        ..Default::default()
                    });
                    let prep = match mode {
                        0 => CovariancePreparation::Legacy,
                        1 => CovariancePreparation::ResidentKnn,
                        2 => CovariancePreparation::ResidentRadius(128),
                        _ => CovariancePreparation::ResidentGrid(128),
                    };
                    let before = registration
                        .align_reduction_with_tree_builder(
                            &runtime,
                            &source,
                            &target,
                            prep,
                            KdTree::from_slices,
                        )
                        .unwrap();
                    let actual = registration
                        .align_reduction_impl(&runtime, &source, &target, prep)
                        .unwrap();
                    assert_eq!(actual, before, "n={n} scale={scale} mode={mode} stop={stop}");
                    checked += source.len();
                }
            }
        }
    }
    eprintln!(
        "PARALLEL_GICP_TREE_BUILD exact_diagnostics_conditions=216 compared_points={checked}"
    );
}

fn fixture(n: usize, profile: usize) -> PointCloud {
    let mut b = PointCloudBuilder::xyz();
    for i in 0..n {
        let j = if profile == 2 { i / 2 } else { i };
        b.push_point([(j % 32) as f32 * 0.5, (j / 32 % 32) as f32 * 0.5, (j / 1024) as f32 * 0.5])
            .unwrap();
    }
    b.build().unwrap()
}

#[test]
#[ignore = "completed GICP calls with serial/parallel tree preparation; run this exact test alone in release mode"]
fn completed_parallel_tree_grid_gicp_calls() {
    use std::time::Instant;
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    println!(
        "PARALLEL_GICP_TREE_ENV backend={} device_type={} available_workers={}",
        info.backend,
        info.device_type,
        std::thread::available_parallelism().map_or(1, |n| n.get())
    );
    for n in [4097, 32768, 65537, 262145] {
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
                covariance_radius: Some([0.51, 0.8, 1.05][profile]),
                transformation_epsilon: 0.0,
                fitness_epsilon: 0.0,
                ..Default::default()
            });
            let capacity = [32, 64, 128][profile];
            let prep = CovariancePreparation::ResidentGrid(capacity);
            let expected = registration
                .align_reduction_with_tree_builder(
                    &runtime,
                    &source,
                    &target,
                    prep,
                    KdTree::from_slices,
                )
                .unwrap();
            for _ in 0..3 {
                assert_eq!(
                    registration
                        .align_reduction_with_tree_builder(
                            &runtime,
                            &source,
                            &target,
                            prep,
                            KdTree::from_slices
                        )
                        .unwrap(),
                    expected
                );
                assert_eq!(
                    registration
                        .align_with_gpu_resident_grid_covariances(
                            &runtime, &source, &target, capacity
                        )
                        .unwrap(),
                    expected
                );
            }
            let (mut serial_ms, mut parallel_ms) = (vec![], vec![]);
            for round in 0..20 {
                for offset in 0..2 {
                    let mode = (round + offset) % 2;
                    let start = Instant::now();
                    let result = if mode == 0 {
                        registration
                            .align_reduction_with_tree_builder(
                                &runtime,
                                &source,
                                &target,
                                prep,
                                KdTree::from_slices,
                            )
                            .unwrap()
                    } else {
                        registration
                            .align_with_gpu_resident_grid_covariances(
                                &runtime, &source, &target, capacity,
                            )
                            .unwrap()
                    };
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    if mode == 0 {
                        serial_ms.push(ms);
                    } else {
                        parallel_ms.push(ms);
                    }
                    assert_eq!(result, expected);
                }
            }
            println!("PARALLEL_GICP_TREE_BENCH n={n} profile={profile} iterations={} h2d={} d2h={} fitness={} serial_ms={serial_ms:?} parallel_ms={parallel_ms:?}",expected.result.iterations,expected.transfers.host_to_device_bytes(),expected.transfers.device_to_host_bytes(),expected.result.fitness);
        }
    }
}
