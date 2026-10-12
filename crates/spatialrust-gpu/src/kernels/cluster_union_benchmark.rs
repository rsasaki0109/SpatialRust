use super::*;

#[test]
#[ignore = "native completed GPU component labeling vs frozen hybrid and CPU; run alone in release mode"]
fn paired_gpu_hybrid_and_cpu_cluster_completion() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    eprintln!("CONDITIONS backend={} device_type={} cpu_workers={} warmups=3 triples=20 boundary=completed_public_clustering", runtime.adapter_info().backend, runtime.adapter_info().device_type, std::thread::available_parallelism().map_or(1, |n| n.get()));
    for n in [257_usize, 4096, 16384, 65537] {
        for kind in [0_usize, 3, 4] {
            let xyz = fixture(n, kind);
            let expected: Vec<_> =
                (0..n).map(|i| if kind == 4 { (i / 7 * 7) as u32 } else { 0 }).collect();
            let mut times: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
            for round in 0..23 {
                for offset in 0..3 {
                    let mode = (round + offset) % 3;
                    let start = std::time::Instant::now();
                    let output = match mode {
                        0 => {
                            previous::euclidean_cluster_roots_gpu_with_receipt(
                                &runtime, &xyz[0], &xyz[1], &xyz[2], 1.0,
                            )
                            .unwrap()
                            .0
                        }
                        1 => {
                            euclidean_cluster_roots_gpu_with_receipt(
                                &runtime, &xyz[0], &xyz[1], &xyz[2], 1.0,
                            )
                            .unwrap()
                            .0
                        }
                        _ => previous::cpu_roots(&xyz, 1.0),
                    };
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(output, expected);
                    std::hint::black_box(output);
                    if round >= 3 {
                        times[mode].push(elapsed);
                    }
                }
            }
            eprintln!("PAIRED {{\"points\":{n},\"kind\":{kind},\"previous_ms\":{:?},\"current_ms\":{:?},\"cpu_ms\":{:?}}}", times[0], times[1], times[2]);
        }
    }
}
