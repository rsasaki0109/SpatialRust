use super::tests::previous;
use crate::uniform_grid::{build_grid, build_tests::fixture};
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "paired completed serial/parallel CPU grid construction; run alone in release mode"]
fn paired_serial_current_grid_prefix() {
    eprintln!("CONDITIONS available_workers={} parallel_feature={} warmups=3 pairs=20 boundary=completed_build_grid",
        std::thread::available_parallelism().map_or(1, |n| n.get()), cfg!(feature="parallel"));
    for n in [257_usize, 65536, 1048579] {
        for dims in [[128, 128, 64], [256, 128, 128], [256, 256, 128]] {
            let (xyz, _) = fixture(n, dims, 1.0);
            let cells = dims.into_iter().map(u64::from).product::<u64>();
            let expected = previous::build_grid(&xyz[0], &xyz[1], &xyz[2], [-4.0; 3], dims, 1.0);
            let run = |current: bool| {
                let start = Instant::now();
                let output = if current {
                    build_grid(black_box(&xyz[0]), &xyz[1], &xyz[2], [-4.0; 3], dims, 1.0)
                } else {
                    previous::build_grid(black_box(&xyz[0]), &xyz[1], &xyz[2], [-4.0; 3], dims, 1.0)
                };
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(output, expected);
                black_box(output);
                elapsed
            };
            for _ in 0..3 {
                run(false);
                run(true);
            }
            let mut old = Vec::new();
            let mut new = Vec::new();
            for pair in 0..20 {
                if pair % 2 == 0 {
                    old.push(run(false));
                    new.push(run(true));
                } else {
                    new.push(run(true));
                    old.push(run(false));
                }
            }
            eprintln!("PAIRED {{\"points\":{n},\"cells\":{cells},\"previous_ms\":{old:?},\"current_ms\":{new:?}}}");
        }
    }
}
