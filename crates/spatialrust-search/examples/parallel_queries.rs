//! Completed-query comparison with rotating execution order and exact output checks.

mod common;

use common::{Fixture, Kind, Mode};
use std::{hint::black_box, time::Instant};

fn main() {
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("available_parallelism={available}; 2 warmups + 12 rotating rounds; radius=1.01, k=8");
    println!("Time includes allocation, queries, worker spawn/join and ordered merge; excludes setup, verification and result drop.");
    for (points, chunk_size) in [(16_384usize, 256usize), (65_536, 1024), (65_536, 1)] {
        let fixture = Fixture::new(points, chunk_size);
        let query = fixture.query();
        let chunks = points.div_ceil(chunk_size);
        let worker_cap = available.min(chunks);
        let spawned = if worker_cap == 1 { 0 } else { worker_cap };
        println!(
            "points={points}, chunk_size={chunk_size}, chunks={chunks}, bounded_spawned={spawned}"
        );
        let modes: &[Mode] = if chunks <= 256 {
            &[Mode::Serial, Mode::Bounded, Mode::Legacy]
        } else {
            println!("Legacy skipped: reference is capped at 256 chunks.");
            &[Mode::Serial, Mode::Bounded]
        };
        for kind in [Kind::Radius, Kind::Knn] {
            let expected = query.run(kind, Mode::Serial);
            for _ in 0..2 {
                for &mode in modes {
                    assert!(query.run(kind, mode) == expected, "warmup result/order mismatch");
                }
            }
            let mut samples = vec![Vec::new(); modes.len()];
            for round in 0..12 {
                for offset in 0..modes.len() {
                    let i = (round + offset) % modes.len();
                    let start = Instant::now();
                    let actual = query.run(kind, modes[i]);
                    let millis = start.elapsed().as_secs_f64() * 1000.0;
                    assert!(actual == expected, "measured result/order mismatch");
                    black_box(&actual);
                    samples[i].push(millis);
                }
            }
            for (mode, samples) in modes.iter().zip(&mut samples) {
                samples.sort_by(f64::total_cmp);
                let median = (samples[5] + samples[6]) / 2.0;
                println!("{kind:?}/{mode:?}: median={median:.3}ms, observed=[{:.3}, {:.3}]ms, neighbors={}", samples[0], samples[11], expected.len());
            }
        }
    }
}
