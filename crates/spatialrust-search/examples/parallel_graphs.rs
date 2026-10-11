//! Paired completed-API comparisons with frozen serial loops and exact edge order.
mod graph_common;
use graph_common::{fixture, run, Kind, Mode};
use std::{hint::black_box, time::Instant};

fn main() {
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("available_parallelism={available}; 2 warmups + 20 rotating rounds; chunk_size=256");
    println!("Time includes validation, KD-tree build, allocation, search, worker spawn/join and ordered merge; excludes fixture construction, verification and result drop.");
    println!("points,duplicates,kind,round,edges,previous_ms,serial_ms,parallel_ms");
    for points in [512usize, 4096, 16384, 65536] {
        for duplicates in [false, true] {
            let cloud = fixture(points, duplicates);
            for kind in [Kind::Knn8, Kind::Knn32, Kind::Radius] {
                let expected = run(&cloud, kind, Mode::Previous);
                let modes = [Mode::Previous, Mode::Serial, Mode::Parallel];
                for _ in 0..2 {
                    for mode in modes {
                        assert_eq!(run(&cloud, kind, mode), expected, "warmup edge order");
                    }
                }
                for round in 0..20 {
                    let mut ms = [0.0; 3];
                    for offset in 0..3 {
                        let i = (round + offset) % 3;
                        let start = Instant::now();
                        let actual = run(&cloud, kind, modes[i]);
                        ms[i] = start.elapsed().as_secs_f64() * 1000.0;
                        assert_eq!(actual, expected, "measured edge order");
                        black_box(&actual);
                    }
                    println!(
                        "{points},{duplicates},{kind:?},{round},{},{:.6},{:.6},{:.6}",
                        expected.num_edges(),
                        ms[0],
                        ms[1],
                        ms[2]
                    );
                }
            }
        }
    }
}
