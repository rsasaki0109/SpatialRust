#![cfg(all(feature = "register-gicp-gpu", not(target_arch = "wasm32")))]
//! Paired old/new public API calls, coordinated through stdin/stdout.
//! Build this identical test before and after the GPU layout change. The local
//! controller keeps both processes alive and sends one command at a time; no
//! simultaneous computation or process-start time enters the samples.
use spatialrust_core::{PointCloud, PointCloudBuilder};
use spatialrust_gpu::WgpuRuntime;
use spatialrust_math::{Isometry3, Quat, Vec3};
use spatialrust_registration::{
    transform_point_cloud, GicpConfig, GicpRegistration, GpuGicpReductionDiagnostics,
};
use std::io::{BufRead, Write};
use std::time::Instant;

fn fixture(n: usize, duplicates: bool) -> PointCloud {
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..n {
        let j = if duplicates { i / 2 } else { i };
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

fn run(
    registration: &GicpRegistration,
    runtime: &WgpuRuntime,
    source: &PointCloud,
    target: &PointCloud,
    mode: usize,
) -> GpuGicpReductionDiagnostics {
    match mode {
        0 => registration.align_with_gpu_reduction(runtime, source, target),
        1 => registration.align_with_gpu_resident_covariances(runtime, source, target),
        2 => registration.align_with_gpu_resident_radius_covariances(runtime, source, target, 64),
        3 => registration.align_with_gpu_resident_grid_covariances(runtime, source, target, 64),
        _ => panic!("unsupported covariance mode"),
    }
    .unwrap()
}

fn emit(message: impl std::fmt::Display) {
    println!("GICP_REUSE_PROTOCOL {message}");
    std::io::stdout().flush().unwrap();
}

#[test]
#[ignore = "requires paired baseline/current stdin controller; exact release test only"]
fn paired_completed_registration_calls() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    emit(format!(
        "ready backend={} device_type={} scratch_budget={}",
        info.backend,
        info.device_type,
        runtime.scratch_buffer_pool_stats().budget_bytes
    ));
    let stdin = std::io::stdin();
    let mut commands = stdin.lock().lines();
    let mut measured = 0;
    loop {
        let command = commands.next().expect("paired controller required").unwrap();
        if command == "quit" {
            assert!(measured >= 20);
            emit(format!("finished measured={measured}"));
            break;
        }
        let fields: Vec<usize> = command
            .strip_prefix("case ")
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(fields.len(), 3);
        let (n, profile, mode) = (fields[0], fields[1], fields[2]);
        assert!([257, 4097, 16384, 65537].contains(&n));
        assert!(profile <= 1 && mode <= 3);
        runtime.clear_scratch_buffer_pool();
        let target = fixture(n + if profile == 1 { 17 } else { 0 }, profile == 1);
        let cloud = fixture(n, profile == 1);
        let source = transform_point_cloud(
            &cloud,
            Isometry3::new(Quat::<f32>::identity(), Vec3::new(0.03125, -0.015625, 0.0078125)),
        )
        .unwrap();
        let registration = GicpRegistration::new(GicpConfig {
            max_iterations: 3,
            max_correspondence_distance: 0.25,
            k_neighbors: 6,
            transformation_epsilon: 0.0,
            fitness_epsilon: 0.0,
            covariance_radius: if mode >= 2 { Some(0.8) } else { None },
            ..Default::default()
        });
        let expected = run(&registration, &runtime, &source, &target, mode);
        assert_eq!(expected.result.iterations, 3);
        assert_eq!(expected.evaluation_batches, 4);
        assert_eq!(expected.history.len(), 3);
        assert_eq!(expected.covariance_batches, if mode == 0 { 0 } else { 2 });
        assert_eq!(expected.transfers.device_to_host_bytes(), 1024);
        assert!(expected
            .history
            .iter()
            .all(|row| row.correspondences as usize == n
                && row.evaluated_correspondences as usize == n));
        // Includes complete pose/fitness/stopping/history/receipt fields. The
        // controller requires identical old/new strings as well as local checks.
        emit(format!("case_result n={n} profile={profile} mode={mode} result={expected:?}"));
        let mut samples = Vec::new();
        loop {
            match commands.next().expect("controller closed early").unwrap().as_str() {
                "warm" | "sample" => {
                    let start = Instant::now();
                    let actual = run(&registration, &runtime, &source, &target, mode);
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(actual, expected);
                    // All measured calls use the same public API and include
                    // CPU preparation, covariance/position transfers, all four
                    // evaluation downloads, solves and three pose updates.
                    emit(format!("sample ms={elapsed:.9}"));
                    samples.push(elapsed);
                }
                "finish" => {
                    assert_eq!(samples.len(), 23);
                    measured += 20;
                    emit(format!(
                        "case_done n={n} profile={profile} mode={mode} retained={} samples={:?}",
                        runtime.scratch_buffer_pool_stats().cached_bytes,
                        &samples[3..]
                    ));
                    break;
                }
                _ => panic!("invalid paired command"),
            }
        }
    }
}
