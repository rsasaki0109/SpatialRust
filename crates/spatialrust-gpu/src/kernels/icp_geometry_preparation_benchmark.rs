//! Paired frozen/current geometry and completed ICP upload + one evaluation.
use super::*;
use std::time::Instant;

#[test]
#[ignore = "requires isolated release benchmark; exact test only"]
fn paired_completed_geometry_upload_calls() {
    let runtimes = [WgpuRuntime::new_headless().unwrap(), WgpuRuntime::new_headless().unwrap()];
    let info = runtimes[0].adapter_info();
    println!(
        "REGISTRATION_GEOMETRY_ENV backend={} device_type={} available_workers={}",
        info.backend,
        info.device_type,
        std::thread::available_parallelism().map_or(1, |n| n.get())
    );
    for (n, targets, profile) in [
        (257, 257, 0),
        (4097, 4097, 0),
        (32_767, 1, 0),
        (32_768, 1, 0),
        (65_537, 65_537, 0),
        (131_071, 1, 0),
        (131_072, 1, 0),
        (262_143, 1, 0),
        (262_144, 1, 0),
        (262_145, 1, 0),
        (1_048_577, 1, 0),
        (1_048_577, 1, 1),
        (1_048_577, 1, 2),
        (1_048_577, 1, 3),
        (1_048_577, 1, 4),
    ] {
        for runtime in &runtimes {
            runtime.clear_scratch_buffer_pool();
        }
        let points = fixture(n, profile);
        let target = tree(&fixture(targets, profile));
        let host_expected = previous_geometry(&points).unwrap();
        let input = previous_upload(&runtimes[0], &points, &target).unwrap();
        let receipt = input.upload_transfers();
        let expected = input
            .moments_resident(&runtimes[0], Isometry3::identity(), 100.0)
            .unwrap()
            .download(&runtimes[0])
            .unwrap();
        drop(input);
        let input = GpuIcpInput::upload(&runtimes[1], &points, &target).unwrap();
        assert_eq!(input.upload_transfers(), receipt);
        assert_eq!(
            input
                .moments_resident(&runtimes[1], Isometry3::identity(), 100.0)
                .unwrap()
                .download(&runtimes[1])
                .unwrap(),
            expected
        );
        drop(input);
        let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        let mut phases: [[Vec<f64>; 4]; 2] =
            std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
        for round in 0..23 {
            for offset in 0..4 {
                let stage = (round + offset) % 4;
                let variant = stage % 2;
                let start = Instant::now();
                let elapsed;
                if stage < 2 {
                    let actual =
                        if variant == 0 { previous_geometry(&points) } else { prepare(&points) }
                            .unwrap();
                    elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    let anchors = if variant == 0 {
                        actual.0.map(canonical_bits)
                    } else {
                        actual.0.map(f32::to_bits)
                    };
                    assert_eq!(anchors, host_expected.0.map(canonical_bits));
                    assert_eq!(actual.1.to_bits(), host_expected.1.to_bits());
                } else {
                    let runtime = &runtimes[variant];
                    let input = if variant == 0 {
                        previous_upload(runtime, &points, &target)
                    } else {
                        GpuIcpInput::upload(runtime, &points, &target)
                    }
                    .unwrap();
                    let upload = start.elapsed().as_secs_f64() * 1000.0;
                    let output =
                        input.moments_resident(runtime, Isometry3::identity(), 100.0).unwrap();
                    let submit = start.elapsed().as_secs_f64() * 1000.0;
                    let actual = output.download(runtime).unwrap();
                    let read = start.elapsed().as_secs_f64() * 1000.0;
                    let transfers = input.upload_transfers();
                    drop(output);
                    drop(input);
                    elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(actual, expected);
                    assert_eq!(transfers, receipt);
                    assert_eq!(actual.transfers.device_to_host_bytes(), 176);
                    if round >= 3 {
                        for (phase, ms) in phases[variant].iter_mut().zip([
                            upload,
                            submit - upload,
                            read - submit,
                            elapsed - read,
                        ]) {
                            phase.push(ms);
                        }
                    }
                }
                if round >= 3 {
                    samples[stage].push(elapsed);
                }
            }
        }
        println!("REGISTRATION_GEOMETRY_RECORD {{\"sources\":{n},\"targets\":{targets},\"profile\":{profile},\"workers\":{},\"host_old_ms\":{:?},\"host_new_ms\":{:?},\"full_old_ms\":{:?},\"full_new_ms\":{:?},\"old_phases_ms\":{:?},\"new_phases_ms\":{:?}}}",worker_count(n,std::thread::available_parallelism().map_or(1,|n|n.get())),samples[0],samples[1],samples[2],samples[3],phases[0],phases[1]);
    }
}
