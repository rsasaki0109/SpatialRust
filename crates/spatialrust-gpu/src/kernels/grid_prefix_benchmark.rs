use super::*;

#[test]
#[ignore = "native paired completed GPU estimators with serial/parallel CPU prefix"]
fn paired_serial_current_cpu_grid_prefix() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    eprintln!("CONDITIONS backend={} device_type={} warmups=3 pairs=20 boundary=completed_public_api shared_current_shaders=true",
        runtime.adapter_info().backend,runtime.adapter_info().device_type);
    for n in [257_usize, 4096, 65536] {
        for large in [false, true] {
            let xyz = fixture(n, large);
            let (_, dims) =
                spatialrust_search::grid_bounds(&xyz[0], &xyz[1], &xyz[2], 1.0).unwrap();
            let cells = dims.into_iter().map(u64::from).product::<u64>();
            for covariance in [false, true] {
                let mut times: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
                for round in 0..23 {
                    let mut outputs: [Vec<u32>; 2] = std::array::from_fn(|_| Vec::new());
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
                            outputs[mode] =
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
                            outputs[mode] = normal_words(result);
                        }
                        if round >= 3 {
                            times[mode].push(elapsed);
                        }
                    }
                    assert_eq!(outputs[0], outputs[1]);
                }
                eprintln!("PAIRED {{\"points\":{n},\"cells\":{cells},\"covariance\":{covariance},\"previous_ms\":{:?},\"current_ms\":{:?}}}",times[0],times[1]);
            }
        }
    }
}
