//! Completed old/new calls; frozen shader and submission retain separate pairs.
use super::*;
use std::time::Instant;

fn measure(
    runtimes: [&WgpuRuntime; 2],
    previous: &PointToPlaneReductionPipelines,
    n: usize,
    profile: u32,
    distance: f32,
) {
    for r in runtimes {
        r.clear_scratch_buffer_pool();
    }
    let (source, target, normals) = fixture(n, profile);
    let target_tree = tree(&target);
    let inputs =
        runtimes.map(|r| GpuPointToPlaneInput::upload(r, &source, &target_tree, &normals).unwrap());
    let expected = inputs[0]
        .previous_equations_resident(previous, runtimes[0], pose(), distance)
        .unwrap()
        .download(runtimes[0])
        .unwrap();
    assert_eq!(
        inputs[1]
            .equations_resident(runtimes[1], pose(), distance)
            .unwrap()
            .download(runtimes[1])
            .unwrap(),
        expected
    );
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| vec![]);
    for round in 0..23 {
        for offset in 0..4 {
            let stage = (round + offset) % 4;
            let variant = stage % 2;
            let runtime = runtimes[variant];
            let start = Instant::now();
            // Full calls build the CPU target tree, pack/upload input, create
            // controls/output/scratch, submit, download/decode and drop owners.
            // Resident calls use the prepared input, but include a completion
            // wait; their download and validation follow the timer.
            let preparation = (stage < 2).then(|| {
                let t = tree(&target);
                GpuPointToPlaneInput::upload(runtime, &source, &t, &normals).unwrap()
            });
            let input = preparation.as_ref().unwrap_or(&inputs[variant]);
            let output = if variant == 0 {
                input.previous_equations_resident(previous, runtime, pose(), distance)
            } else {
                input.equations_resident(runtime, pose(), distance)
            }
            .unwrap();
            let elapsed;
            if stage < 2 {
                let actual = output.download(runtime).unwrap();
                let upload = input.upload_transfers();
                drop(output);
                drop(preparation);
                elapsed = start.elapsed();
                assert_eq!(actual, expected);
                assert_eq!(actual.transfers.device_to_host_bytes(), 256);
                assert_eq!(
                    upload.host_to_device_bytes(),
                    (16 * n + 48 * target.len() + 32 * target_tree.layout().nodes().len()) as u64
                );
            } else {
                runtime.wait_idle();
                elapsed = start.elapsed();
                assert_eq!(output.download(runtime).unwrap(), expected);
            }
            if round >= 3 {
                samples[stage].push(elapsed.as_secs_f64() * 1000.0);
            }
        }
    }
    let groups = n.div_ceil(64).min(256) as u64;
    let old_scratch = n as u64 * 56 + groups * 240;
    let new_scratch = n as u64 * 40 + groups * 240;
    let retained = runtimes.map(|r| r.scratch_buffer_pool_stats().cached_bytes);
    assert!(runtimes
        .iter()
        .all(|r| r.scratch_buffer_pool_stats().budget_bytes == 64 * 1024 * 1024));
    println!("GPU_PLANE_REUSE sources={n} targets={} profile={profile} distance={distance} correspondences={} old_scratch={old_scratch} new_scratch={new_scratch} old_retained={} new_retained={} old_full_ms={:?} new_full_ms={:?} old_resident_ms={:?} new_resident_ms={:?}",
        target.len(), expected.correspondences, retained[0], retained[1], samples[0], samples[1], samples[2], samples[3]);
}

#[test]
#[ignore = "paired completed query/row reuse survey; run alone in release mode"]
fn completed_reused_query_calls() {
    let old = WgpuRuntime::new_headless().unwrap();
    let new = WgpuRuntime::new_headless().unwrap();
    let previous = previous_pipelines(&old);
    let info = old.adapter_info();
    assert_eq!(info.backend, new.adapter_info().backend);
    assert_eq!(info.device_type, new.adapter_info().device_type);
    println!(
        "GPU_PLANE_REUSE_ENV backend={} device_type={} cpu_workers={}",
        info.backend,
        info.device_type,
        rayon::current_num_threads()
    );
    for (n, profile, distance) in [
        (257, 1, 0.25),
        (4097, 1, 0.25),
        (4097, 2, 0.25),
        (65537, 1, 0.25),
        (65537, 2, 0.25),
        (65537, 0, 100.0),
        (262145, 0, 0.001),
        (262145, 0, 100.0),
        (1048577, 0, 0.001),
        (1048577, 0, 100.0),
    ] {
        measure([&old, &new], &previous, n, profile, distance);
    }
}
