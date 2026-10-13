//! Paired frozen/current completed evaluations with preparation phase receipts.
use super::*;
use std::time::Instant;

fn measure(
    runtimes: [&WgpuRuntime; 2],
    previous: &GicpReductionPipelines,
    n: usize,
    profile: u32,
    distance: f32,
) {
    for r in runtimes {
        r.clear_scratch_buffer_pool();
    }
    let (source, target, sc, tc) = fixture(n, profile, 1, 1.0);
    let target_tree = tree(&target);
    let inputs =
        runtimes.map(|r| GpuGicpInput::upload(r, &source, &sc, &target_tree, &tc).unwrap());
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
    // Full-call components: tree, upload/packing/validation, submission,
    // download/wait/decode, owner drop. Same clocks instrument both routes.
    let mut phases: [[Vec<f64>; 5]; 2] = std::array::from_fn(|_| std::array::from_fn(|_| vec![]));
    for round in 0..23 {
        for offset in 0..4 {
            let stage = (round + offset) % 4;
            let variant = stage % 2;
            let runtime = runtimes[variant];
            let start = Instant::now();
            let full_tree = (stage < 2).then(|| tree(&target));
            let tree_end = start.elapsed().as_secs_f64() * 1000.0;
            let uploaded = full_tree
                .as_ref()
                .map(|t| GpuGicpInput::upload(runtime, &source, &sc, t, &tc).unwrap());
            let upload_end = start.elapsed().as_secs_f64() * 1000.0;
            let input = uploaded.as_ref().unwrap_or(&inputs[variant]);
            let output = if variant == 0 {
                input.previous_equations_resident(previous, runtime, pose(), distance)
            } else {
                input.equations_resident(runtime, pose(), distance)
            }
            .unwrap();
            let submit_end = start.elapsed().as_secs_f64() * 1000.0;
            let elapsed;
            if stage < 2 {
                let actual = output.download(runtime).unwrap();
                let read_end = start.elapsed().as_secs_f64() * 1000.0;
                let receipt = input.upload_transfers();
                drop(output);
                drop(uploaded);
                drop(full_tree);
                elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(actual, expected);
                assert_eq!(
                    receipt.host_to_device_bytes(),
                    (48 * n + 64 * target.len() + 32 * target_tree.layout().nodes().len()) as u64
                );
                assert_eq!(actual.transfers.device_to_host_bytes(), 256);
                if round >= 3 {
                    let values = [
                        tree_end,
                        upload_end - tree_end,
                        submit_end - upload_end,
                        read_end - submit_end,
                        elapsed - read_end,
                    ];
                    for (phase, value) in phases[variant].iter_mut().zip(values) {
                        phase.push(value);
                    }
                }
            } else {
                runtime.wait_idle();
                elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(output.download(runtime).unwrap(), expected);
            }
            if round >= 3 {
                samples[stage].push(elapsed);
            }
        }
    }
    let groups = n.div_ceil(64).min(256) as u64;
    let retained = runtimes.map(|r| r.scratch_buffer_pool_stats().cached_bytes);
    assert!(runtimes
        .iter()
        .all(|r| r.scratch_buffer_pool_stats().budget_bytes == 64 * 1024 * 1024));
    println!("GPU_GICP_REUSE sources={n} targets={} profile={profile} distance={distance} correspondences={} old_scratch={} new_scratch={} old_retained={} new_retained={} old_full_ms={:?} new_full_ms={:?} old_resident_ms={:?} new_resident_ms={:?}",target.len(),expected.correspondences,88*n as u64+groups*240,72*n as u64+groups*240,retained[0],retained[1],samples[0],samples[1],samples[2],samples[3]);
    for (variant, values) in phases.into_iter().enumerate() {
        println!("GPU_GICP_REUSE_PHASE sources={n} profile={profile} distance={distance} variant={variant} tree_ms={:?} upload_ms={:?} submit_ms={:?} download_ms={:?} drop_ms={:?}",values[0],values[1],values[2],values[3],values[4]);
    }
}

#[test]
#[ignore = "paired completed GICP query reuse survey; exact isolated release invocation"]
fn completed_reused_query_calls() {
    let old = WgpuRuntime::new_headless().unwrap();
    let new = WgpuRuntime::new_headless().unwrap();
    let previous = previous_pipelines(&old);
    let info = old.adapter_info();
    assert_eq!(info.backend, new.adapter_info().backend);
    assert_eq!(info.device_type, new.adapter_info().device_type);
    println!(
        "GPU_GICP_REUSE_ENV backend={} device_type={} cpu_workers={}",
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
