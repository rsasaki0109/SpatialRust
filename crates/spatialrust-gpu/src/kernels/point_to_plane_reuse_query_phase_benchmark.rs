//! Diagnose the retained full-call regressions without changing production APIs.
use super::*;
use std::time::Instant;

#[test]
#[ignore = "paired point-to-plane full-call phase diagnostic; isolated release only"]
fn completed_full_call_phases() {
    let old = WgpuRuntime::new_headless().unwrap();
    let new = WgpuRuntime::new_headless().unwrap();
    let runtimes = [&old, &new];
    let previous = previous_pipelines(&old);
    let info = old.adapter_info();
    assert_eq!(info.backend, new.adapter_info().backend);
    assert_eq!(info.device_type, new.adapter_info().device_type);
    println!(
        "GPU_PLANE_REUSE_PHASE_ENV backend={} device_type={} cpu_workers={}",
        info.backend,
        info.device_type,
        rayon::current_num_threads()
    );
    for n in [65537, 262145, 1048577] {
        for r in runtimes {
            r.clear_scratch_buffer_pool();
        }
        let (source, target, normals) = fixture(n, 0);
        let target_tree = tree(&target);
        let inputs = runtimes
            .map(|r| GpuPointToPlaneInput::upload(r, &source, &target_tree, &normals).unwrap());
        let expected = {
            inputs[0]
                .previous_equations_resident(&previous, &old, pose(), 100.0)
                .unwrap()
                .download(&old)
                .unwrap()
        };
        let mut samples: [[Vec<f64>; 6]; 2] =
            std::array::from_fn(|_| std::array::from_fn(|_| vec![]));
        for round in 0..23 {
            for offset in 0..4 {
                let stage = (round + offset) % 4;
                let variant = stage % 2;
                let r = runtimes[variant];
                // Preserve the original four-way full/resident interleaving.
                // Pool reuse and source lifetimes differ from full-only calls.
                if stage >= 2 {
                    let output = if variant == 0 {
                        inputs[variant].previous_equations_resident(&previous, r, pose(), 100.0)
                    } else {
                        inputs[variant].equations_resident(r, pose(), 100.0)
                    }
                    .unwrap();
                    r.wait_idle();
                    assert_eq!(output.download(r).unwrap(), expected);
                    continue;
                }
                let start = Instant::now();
                let target_tree = tree(&target);
                let tree_end = start.elapsed().as_secs_f64() * 1000.0;
                let input =
                    GpuPointToPlaneInput::upload(r, &source, &target_tree, &normals).unwrap();
                let upload_end = start.elapsed().as_secs_f64() * 1000.0;
                let output = if variant == 0 {
                    input.previous_equations_resident(&previous, r, pose(), 100.0)
                } else {
                    input.equations_resident(r, pose(), 100.0)
                }
                .unwrap();
                let submit_end = start.elapsed().as_secs_f64() * 1000.0;
                let actual = output.download(r).unwrap();
                let read_end = start.elapsed().as_secs_f64() * 1000.0;
                let receipt = input.upload_transfers();
                drop(output);
                drop(input);
                drop(target_tree);
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(actual, expected);
                assert_eq!(
                    receipt.host_to_device_bytes(),
                    (n * 16 + 48 + 32 * tree(&target).layout().nodes().len()) as u64
                );
                assert_eq!(actual.transfers.device_to_host_bytes(), 256);
                if round >= 3 {
                    for (phase, value) in samples[variant].iter_mut().zip([
                        tree_end,
                        upload_end - tree_end,
                        submit_end - upload_end,
                        read_end - submit_end,
                        elapsed - read_end,
                        elapsed,
                    ]) {
                        phase.push(value);
                    }
                }
            }
        }
        for (variant, phases) in samples.into_iter().enumerate() {
            println!("GPU_PLANE_REUSE_PHASE sources={n} variant={variant} tree_ms={:?} upload_ms={:?} submit_ms={:?} download_ms={:?} drop_ms={:?} full_ms={:?}",phases[0],phases[1],phases[2],phases[3],phases[4],phases[5]);
        }
    }
}
