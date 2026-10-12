use super::{
    parallel_tests::{fixture, previous_mask},
    *,
};
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "completed boundary API comparison; explicit native invocation only"]
fn paired_previous_serial_parallel_boundary() {
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    eprintln!("available_parallelism={available}; 2 warmups + 20 rotating rounds; default shared staging policy");
    eprintln!("Timers include validation, KD-tree build, mask allocation, angle-gap evaluation, worker spawn/join and scratch/tree destruction; exclude fixture creation, mask verification and returned mask drop.");
    eprintln!("points,kind,radius,workers,round,previous_ms,serial_ms,parallel_ms");
    for points in [512usize, 16384, 32768, 65536] {
        for kind in 0..3 {
            let config = BoundaryConfig {
                search_radius: if kind == 0 { 0.11 } else { 0.16 },
                min_neighbors: 3,
                ..Default::default()
            };
            let cloud = fixture(points, kind);
            let expected = previous_mask(&cloud, config);
            let detector = BoundaryDetector::new(config);
            let workers = parallel_worker_count(points);
            let run = |mode| match mode {
                0 => previous_mask(&cloud, config),
                1 => detector.boundary_mask_with_workers(&cloud, 1).unwrap(),
                _ => detector.boundary_mask(&cloud).unwrap(),
            };
            for _ in 0..2 {
                for mode in 0..3 {
                    assert_eq!(run(mode), expected);
                }
            }
            for round in 0..20 {
                let mut ms = [0.0; 3];
                for offset in 0..3 {
                    let mode = (round + offset) % 3;
                    let start = Instant::now();
                    let actual = run(mode);
                    ms[mode] = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(actual, expected);
                    black_box(&actual);
                }
                eprintln!(
                    "{points},{kind},{},{workers},{round},{:.6},{:.6},{:.6}",
                    config.search_radius, ms[0], ms[1], ms[2]
                );
            }
        }
    }
}
