//! Paired merged/current target staging and completed upload + one evaluation.
use super::*;
use std::time::Instant;

#[test]
#[ignore = "requires isolated release benchmark; exact test only"]
fn paired_completed_soa_upload_calls() {
    let runtimes = [WgpuRuntime::new_headless().unwrap(), WgpuRuntime::new_headless().unwrap()];
    let info = runtimes[0].adapter_info();
    println!(
        "REGISTRATION_SOA_ENV backend={} device_type={} available_workers={}",
        info.backend,
        info.device_type,
        std::thread::available_parallelism().map_or(1, |n| n.get())
    );
    for (sources, targets, profile) in [
        (1, 1, 0),
        (257, 257, 0),
        (4097, 4097, 0),
        (257, 65_537, 0),
        (65_537, 65_537, 0),
        (257, 131_073, 0),
        (257, 262_143, 0),
        (257, 262_144, 0),
        (257, 262_145, 0),
        (257, 524_289, 0),
        (257, 1_048_577, 0),
        (257, 1_048_577, 1),
        (257, 1_048_577, 2),
        (262_145, 1, 0),
    ] {
        for runtime in &runtimes {
            runtime.clear_scratch_buffer_pool();
        }
        let points = fixture(sources, profile);
        let target = tree(&fixture(targets, profile));
        let (x, y, z) = target.layout().positions();
        let old_positions: Vec<_> = (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect();
        let host_expected = geometry(&old_positions).unwrap();
        drop(old_positions);
        let input = previous_upload(&runtimes[0], &points, &target).unwrap();
        let receipt = input.upload_transfers();
        let expected = input
            .moments_resident(&runtimes[0], Isometry3::identity(), 100.0)
            .unwrap()
            .download(&runtimes[0])
            .unwrap();
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
                    let actual = if variant == 0 {
                        let positions: Vec<_> = (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect();
                        let result = geometry(&positions).unwrap();
                        drop(positions);
                        result
                    } else {
                        prepare_soa([x, y, z]).unwrap()
                    };
                    elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(actual.0.map(f32::to_bits), host_expected.0.map(f32::to_bits));
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
        println!("REGISTRATION_SOA_RECORD {{\"sources\":{sources},\"targets\":{targets},\"profile\":{profile},\"workers\":{},\"host_old_ms\":{:?},\"host_new_ms\":{:?},\"full_old_ms\":{:?},\"full_new_ms\":{:?},\"old_phases_ms\":{:?},\"new_phases_ms\":{:?}}}",worker_count(targets,std::thread::available_parallelism().map_or(1,|n|n.get())),samples[0],samples[1],samples[2],samples[3],phases[0],phases[1]);
    }
}
