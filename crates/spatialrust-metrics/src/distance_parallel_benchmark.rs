use super::{
    parallel_tests::{fixture_pair, previous_words, words},
    *,
};
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "completed cloud-distance comparison; explicit native invocation only"]
fn paired_previous_serial_parallel_distances() {
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    eprintln!("available_parallelism={available}; optional_distance_bytes={}; 2 warmups + 20 rotating rounds; default shared staging policy", std::mem::size_of::<Option<f32>>());
    eprintln!("Timers include validation, both KD-tree builds, both directions, score-window allocation/reuse, worker spawn/join, ordered f64 accumulation and tree/scratch destruction; exclude fixtures and exact-word verification.");
    eprintln!("points,kind,target_points,workers,round,previous_ms,serial_ms,parallel_ms");
    for points in [512usize, 16384, 32768, 65536] {
        for kind in 0..3 {
            let (a, b) = fixture_pair(points, kind);
            let expected = previous_words(&a, &b);
            let wa = parallel_worker_count(a.len());
            let wb = parallel_worker_count(b.len());
            let run = |mode| {
                if mode == 0 {
                    let p = previous::cloud_distances(&a, &b).unwrap();
                    CloudDistances {
                        mean_sq_a_to_b: p.mean_sq_a_to_b,
                        mean_sq_b_to_a: p.mean_sq_b_to_a,
                        max_a_to_b: p.max_a_to_b,
                        max_b_to_a: p.max_b_to_a,
                    }
                } else if mode == 1 {
                    cloud_distances_with_workers(&a, &b, 1, 1).unwrap()
                } else {
                    cloud_distances(&a, &b).unwrap()
                }
            };
            for _ in 0..2 {
                for mode in 0..3 {
                    assert_eq!(words(run(mode)), expected);
                }
            }
            for round in 0..20 {
                let mut ms = [0.0; 3];
                for offset in 0..3 {
                    let mode = (round + offset) % 3;
                    let start = Instant::now();
                    let actual = run(mode);
                    ms[mode] = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(words(actual), expected);
                    black_box(actual);
                }
                eprintln!(
                    "{points},{kind},{},{wa}/{wb},{round},{:.6},{:.6},{:.6}",
                    b.len(),
                    ms[0],
                    ms[1],
                    ms[2]
                );
            }
        }
    }
}
