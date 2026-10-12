use super::{
    parallel_tests::{assert_result, fixture, previous_result},
    *,
};
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "completed ISS detection comparison; explicit native invocation only"]
fn paired_previous_serial_parallel_iss() {
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    eprintln!("available_parallelism={available}; 2 warmups + 20 rotating rounds; default shared staging policy");
    eprintln!("Timers include validation, KD-tree build, saliency allocation/covariance, local-maximum selection, worker spawn/join, ordered indices, schema/attribute gathering and scratch/tree destruction; exclude fixture creation, result verification and returned result drop.");
    eprintln!("points,kind,radius,workers,round,previous_ms,serial_ms,parallel_ms");
    for points in [512usize, 16384, 32768, 65536] {
        for kind in 0..3 {
            let radius = if kind == 0 { 0.11 } else { 0.16 };
            let config = IssKeypointConfig::with_radii(radius, radius * 0.75);
            let cloud = fixture(points, kind);
            let expected = previous_result(&cloud, config);
            let detector = IssKeypointDetector::new(config);
            let workers = parallel_worker_count(points);
            let run = |mode| {
                if mode == 0 {
                    let previous = previous_result(&cloud, config);
                    IssKeypointResult { indices: previous.indices, keypoints: previous.keypoints }
                } else if mode == 1 {
                    detector.detect_with_workers(&cloud, 1).unwrap()
                } else {
                    detector.detect(&cloud).unwrap()
                }
            };
            for _ in 0..2 {
                for mode in 0..3 {
                    assert_result(&run(mode), &expected);
                }
            }
            for round in 0..20 {
                let mut ms = [0.0; 3];
                for offset in 0..3 {
                    let mode = (round + offset) % 3;
                    let start = Instant::now();
                    let actual = run(mode);
                    ms[mode] = start.elapsed().as_secs_f64() * 1000.0;
                    assert_result(&actual, &expected);
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
