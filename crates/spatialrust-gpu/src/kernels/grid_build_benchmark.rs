use super::*;

#[test]
#[ignore = "native paired completed GPU estimators with frozen/current CPU grid builders"]
fn paired_previous_current_cpu_grid_scratch() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    eprintln!("CONDITIONS backend={} device_type={} warmups=3 pairs=30 boundary=completed_public_api_including_validation_cpu_grid_uploads_compute_map_wait_and_decode shared_current_shaders=true",
        runtime.adapter_info().backend, runtime.adapter_info().device_type);
    for n in [257_usize, 4096, 65536] {
        for sparse in [false, true] {
            let xyz = fixture(n, sparse);
            let (_, dims) =
                spatialrust_search::grid_bounds(&xyz[0], &xyz[1], &xyz[2], 1.0).unwrap();
            let cells = dims.into_iter().map(u64::from).product::<u64>();
            for covariance in [false, true] {
                let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
                for round in 0..33 {
                    let mut output: [Vec<u32>; 2] = std::array::from_fn(|_| Vec::new());
                    for offset in 0..2 {
                        let mode = (round + offset) % 2;
                        let start = std::time::Instant::now();
                        let elapsed;
                        if covariance {
                            let result = if mode == 0 {
                                previous_covariances_grid(
                                    &runtime, &xyz[0], &xyz[1], &xyz[2], 1.0, 0.001,
                                )
                                .unwrap()
                            } else {
                                estimate_plane_covariances_grid_gpu(
                                    &runtime, &xyz[0], &xyz[1], &xyz[2], 1.0, 0.001,
                                )
                                .unwrap()
                            };
                            elapsed = start.elapsed().as_secs_f64() * 1000.0;
                            output[mode] =
                                result.into_iter().flat_map(|c| c.map(f32::to_bits)).collect();
                        } else {
                            let result = if mode == 0 {
                                previous_normals_grid(&runtime, &xyz[0], &xyz[1], &xyz[2], 1.0)
                                    .unwrap()
                            } else {
                                estimate_normals_grid_gpu(&runtime, &xyz[0], &xyz[1], &xyz[2], 1.0)
                                    .unwrap()
                            };
                            elapsed = start.elapsed().as_secs_f64() * 1000.0;
                            output[mode] = normal_words(result);
                        }
                        if round >= 3 {
                            samples[mode].push(elapsed);
                        }
                    }
                    assert_eq!(
                        output[0], output[1],
                        "n={n}, sparse={sparse}, covariance={covariance}, round={round}"
                    );
                }
                eprintln!("PAIRED {{\"points\":{n},\"cells\":{cells},\"sparse\":{sparse},\"covariance\":{covariance},\"saved_cursor_bytes\":{},\"previous_ms\":{:?},\"current_ms\":{:?}}}",
                    4 * (cells + 1), samples[0], samples[1]);
            }
        }
    }
}
