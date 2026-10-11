use super::*;

fn timed<T, const FIELDS: usize>(
    call: impl FnOnce() -> Vec<T>,
    words: impl Fn(T) -> [u32; FIELDS],
) -> (f64, Vec<u32>) {
    let start = std::time::Instant::now();
    let output = call();
    let milliseconds = start.elapsed().as_secs_f64() * 1000.0;
    // Result-word conversion and comparison are outside the public API timer.
    (milliseconds, output.into_iter().flat_map(words).collect())
}

#[test]
#[ignore = "explicit native paired full-API point estimator diagnostic"]
fn paired_previous_current_standalone_point_estimators() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    eprintln!("backend={}; device_type={}; full API includes host grid/input validation, pipeline setup, allocation/uploads, compute/readback submissions, map wait and result decode; 3 warmups + 20 alternating paired rounds",runtime.adapter_info().backend,runtime.adapter_info().device_type);
    for n in [17, 257, 4096, 65536] {
        let xyz = fixture(n, 1);
        let k = 16;
        let indices = neighbors(n, k);
        for kind in [Kind::KnnNormals, Kind::RadiusNormals, Kind::PlaneCovariances] {
            let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            let mut outputs: [Vec<u32>; 2] = std::array::from_fn(|_| Vec::new());
            let before = runtime.scratch_buffer_pool_stats();
            let mut warmed = before;
            for round in 0..23 {
                for offset in 0..2 {
                    let i = (round + offset) % 2;
                    let (elapsed, output) = match kind {
                        Kind::KnnNormals if i == 0 => timed(
                            || {
                                previous_knn::estimate_normals_gpu(
                                    &runtime, &xyz[0], &xyz[1], &xyz[2], &indices, k,
                                )
                                .unwrap()
                            },
                            |p| {
                                [
                                    p.normal[0].to_bits(),
                                    p.normal[1].to_bits(),
                                    p.normal[2].to_bits(),
                                    p.curvature.to_bits(),
                                ]
                            },
                        ),
                        Kind::KnnNormals => timed(
                            || {
                                estimate_normals_gpu(
                                    &runtime, &xyz[0], &xyz[1], &xyz[2], &indices, k,
                                )
                                .unwrap()
                            },
                            |p| {
                                [
                                    p.normal[0].to_bits(),
                                    p.normal[1].to_bits(),
                                    p.normal[2].to_bits(),
                                    p.curvature.to_bits(),
                                ]
                            },
                        ),
                        Kind::RadiusNormals => timed(
                            || {
                                if i == 0 {
                                    previous_radius::estimate_normals_grid_gpu(
                                        &runtime, &xyz[0], &xyz[1], &xyz[2], 0.25,
                                    )
                                    .unwrap()
                                } else {
                                    estimate_normals_grid_gpu(
                                        &runtime, &xyz[0], &xyz[1], &xyz[2], 0.25,
                                    )
                                    .unwrap()
                                }
                            },
                            |p| {
                                [
                                    p.normal[0].to_bits(),
                                    p.normal[1].to_bits(),
                                    p.normal[2].to_bits(),
                                    p.curvature.to_bits(),
                                ]
                            },
                        ),
                        Kind::PlaneCovariances => timed(
                            || {
                                if i == 0 {
                                    previous_covariance::estimate_plane_covariances_grid_gpu(
                                        &runtime, &xyz[0], &xyz[1], &xyz[2], 0.25, 0.001,
                                    )
                                    .unwrap()
                                } else {
                                    estimate_plane_covariances_grid_gpu(
                                        &runtime, &xyz[0], &xyz[1], &xyz[2], 0.25, 0.001,
                                    )
                                    .unwrap()
                                }
                            },
                            |v| v.map(f32::to_bits),
                        ),
                    };
                    outputs[i] = output;
                    if round >= 3 {
                        timings[i].push(elapsed);
                    }
                }
                assert_eq!(outputs[0], outputs[1], "{kind:?}/{n}, round {round}");
                if round == 2 {
                    warmed = runtime.scratch_buffer_pool_stats();
                }
            }
            let after = runtime.scratch_buffer_pool_stats();
            assert_eq!(after.allocation_count, warmed.allocation_count);
            let fields = if matches!(kind, Kind::PlaneCovariances) { 6 } else { 4 };
            eprintln!("PAIRED {{\"points\":{n},\"kind\":\"{kind:?}\",\"fields\":{fields},\"previous_ms\":{:?},\"current_ms\":{:?},\"cold_storage_allocations\":{},\"timed_storage_allocations\":{},\"timed_storage_reuses\":{}}}",timings[0],timings[1],warmed.allocation_count-before.allocation_count,after.allocation_count-warmed.allocation_count,after.reuse_count-warmed.reuse_count);
        }
    }
}
