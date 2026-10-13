//! Serial/current CPU staging and completed upload + one equation evaluation.
use super::*;
use std::time::Instant;

#[test]
#[ignore = "requires isolated release benchmark; exact test only"]
fn paired_completed_upload_calls() {
    let runtimes = [WgpuRuntime::new_headless().unwrap(), WgpuRuntime::new_headless().unwrap()];
    let info = runtimes[0].adapter_info();
    println!(
        "GICP_UPLOAD_ENV backend={} device_type={} available_workers={}",
        info.backend,
        info.device_type,
        std::thread::available_parallelism().map_or(1, |n| n.get())
    );
    for (n, targets) in [
        (257, 257),
        (4097, 4097),
        (16_383, 16_384),
        (32_768, 257),
        (65_537, 65_537),
        (262_145, 1),
        (1_048_577, 1),
    ] {
        for runtime in &runtimes {
            runtime.clear_scratch_buffer_pool();
        }
        let (source, target, sc, tc) = fixture(n, targets);
        let target = tree(&target);
        let old = previous_upload(&runtimes[0], &source, &sc, &target, &tc).unwrap();
        let new = GpuGicpInput::upload(&runtimes[1], &source, &sc, &target, &tc).unwrap();
        let expected_receipt = old.upload_transfers();
        assert_eq!(expected_receipt, new.upload_transfers());
        let expected = old
            .equations_resident(&runtimes[0], Isometry3::identity(), 1.0)
            .unwrap()
            .download(&runtimes[0])
            .unwrap();
        assert_eq!(
            new.equations_resident(&runtimes[1], Isometry3::identity(), 1.0)
                .unwrap()
                .download(&runtimes[1])
                .unwrap(),
            expected
        );
        let packed = previous_prepare(&sc, &tc).unwrap();
        drop(old);
        drop(new);
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
                        if variant == 0 { previous_prepare(&sc, &tc) } else { prepare(&sc, &tc) }
                            .unwrap();
                    elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(actual.1.to_bits(), packed.1.to_bits());
                    assert_eq!(
                        bytemuck::cast_slice::<_, u32>(&actual.0),
                        bytemuck::cast_slice::<_, u32>(&packed.0)
                    );
                } else {
                    let runtime = &runtimes[variant];
                    let input = if variant == 0 {
                        previous_upload(runtime, &source, &sc, &target, &tc)
                    } else {
                        GpuGicpInput::upload(runtime, &source, &sc, &target, &tc)
                    }
                    .unwrap();
                    let upload = start.elapsed().as_secs_f64() * 1000.0;
                    let output =
                        input.equations_resident(runtime, Isometry3::identity(), 1.0).unwrap();
                    let submit = start.elapsed().as_secs_f64() * 1000.0;
                    let actual = output.download(runtime).unwrap();
                    let read = start.elapsed().as_secs_f64() * 1000.0;
                    let receipt = input.upload_transfers();
                    drop(output);
                    drop(input);
                    elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(actual, expected);
                    assert_eq!(receipt, expected_receipt);
                    assert_eq!(actual.transfers.device_to_host_bytes(), 256);
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
        println!("GICP_UPLOAD_RECORD {{\"sources\":{n},\"targets\":{targets},\"workers\":{},\"host_old_ms\":{:?},\"host_new_ms\":{:?},\"full_old_ms\":{:?},\"full_new_ms\":{:?},\"old_phases_ms\":{:?},\"new_phases_ms\":{:?}}}", worker_count(n+targets, std::thread::available_parallelism().map_or(1, |n| n.get())), samples[0], samples[1], samples[2], samples[3], phases[0], phases[1]);
    }
}
