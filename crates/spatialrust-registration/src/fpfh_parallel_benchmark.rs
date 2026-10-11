use super::{assert_words, fixture, previous_descriptors, *};
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "completed FPFH API comparison; explicit native invocation only"]
fn paired_previous_serial_parallel_fpfh() {
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    eprintln!("available_parallelism={available}; 2 warmups + 20 rotating rounds; default shared staging policy");
    eprintln!("Timers include validation, point/normal lifting, KD-tree build, both descriptor allocations and phases, worker spawn/join, intermediate/tree destruction; exclude fixture creation, exact-word checks and result drop.");
    eprintln!("points,kind,radius,workers,round,previous_ms,serial_ms,parallel_ms");
    for points in [512usize, 16384, 32768, 65536] {
        for kind in 0..3 {
            let radius = if kind == 0 { 0.11 } else { 0.16 };
            let cloud = fixture(points, kind);
            let expected = previous_descriptors(&cloud, radius);
            let workers = parallel_worker_count(points);
            let run = |mode| match mode {
                0 => previous_descriptors(&cloud, radius),
                1 => fpfh_descriptors_with_workers(&cloud, radius, 1).unwrap(),
                _ => fpfh_descriptors(&cloud, radius).unwrap(),
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
                    "{points},{kind},{radius},{workers},{round},{:.6},{:.6},{:.6}",
                    ms[0], ms[1], ms[2]
                );
            }
        }
    }
}
