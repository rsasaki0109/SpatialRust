use std::hint::black_box;
use std::time::Instant;

use super::{
    row_tests::{fixture, previous},
    *,
};

#[test]
#[ignore = "paired completed CPU clustering timing; run alone in release mode"]
fn paired_previous_current_grid_rows() {
    eprintln!("CONDITIONS available_workers={} parallel_feature={} warmups=3 pairs=30 boundary=validated_public_api_including_grid_and_roots",
        std::thread::available_parallelism().map_or(1, |n| n.get()), cfg!(feature = "parallel"));
    for n in [257_usize, 4096, 65536] {
        for kind in 0..4 {
            let xyz = fixture(n, kind);
            let expected =
                previous::euclidean_cluster_roots(&xyz[0], &xyz[1], &xyz[2], 1.0).unwrap();
            let run = |current: bool| {
                let start = Instant::now();
                let roots = if current {
                    euclidean_cluster_roots(black_box(&xyz[0]), &xyz[1], &xyz[2], 1.0).unwrap()
                } else {
                    previous::euclidean_cluster_roots(black_box(&xyz[0]), &xyz[1], &xyz[2], 1.0)
                        .unwrap()
                };
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(roots, expected);
                black_box(roots);
                elapsed
            };
            for _ in 0..3 {
                run(false);
                run(true);
            }
            let mut old = Vec::new();
            let mut new = Vec::new();
            for pair in 0..30 {
                if pair % 2 == 0 {
                    old.push(run(false));
                    new.push(run(true));
                } else {
                    new.push(run(true));
                    old.push(run(false));
                }
            }
            eprintln!("PAIRED {{\"points\":{n},\"distribution\":\"{}\",\"previous_ms\":{old:?},\"current_ms\":{new:?}}}",
                ["volume", "tilted_sheet", "eight_point_clusters", "chain"][kind]);
        }
    }
}
