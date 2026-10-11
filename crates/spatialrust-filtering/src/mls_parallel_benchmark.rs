use super::{
    parallel_tests::{assert_words, fixture, previous_positions},
    *,
};
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "completed MLS API comparison; explicit native invocation only"]
fn paired_previous_serial_parallel_mls() {
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    eprintln!("available_parallelism={available}; 2 warmups + 20 rotating rounds; default shared staging policy");
    eprintln!("Timers include validation, KD-tree build, output allocation, fits, worker spawn/join; exclude fixture creation, exact-word checks and result drop.");
    eprintln!("points,kind,order,workers,round,previous_ms,serial_ms,parallel_ms");
    for points in [512usize, 16384, 32768, 65536] {
        for (kind, order) in [(0, 1), (1, 2), (2, 2)] {
            let cloud = fixture(points, kind);
            let config =
                MlsConfig { search_radius: 0.16, polynomial_order: order, min_neighbors: 6 };
            let smoother = MlsSmoothing::new(config);
            let expected = previous_positions(&cloud, config);
            let workers = parallel_worker_count(points);
            let run = |mode| match mode {
                0 => previous_positions(&cloud, config),
                1 => smoother.smoothed_positions_with_workers(&cloud, 1).unwrap(),
                _ => smoother.smoothed_positions(&cloud).unwrap(),
            };
            for _ in 0..2 {
                for mode in 0..3 {
                    assert_words(&run(mode), &expected);
                }
            }
            for round in 0..20 {
                let mut ms = [0.0; 3];
                for offset in 0..3 {
                    let mode = (round + offset) % 3;
                    let start = Instant::now();
                    let actual = run(mode);
                    ms[mode] = start.elapsed().as_secs_f64() * 1000.0;
                    assert_words(&actual, &expected);
                    black_box(&actual);
                }
                eprintln!(
                    "{points},{kind},{order},{workers},{round},{:.6},{:.6},{:.6}",
                    ms[0], ms[1], ms[2]
                );
            }
        }
    }
}
