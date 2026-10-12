use std::{hint::black_box, time::Instant};

use super::{build_grid, build_tests::fixture, row_tests::previous};

#[test]
#[ignore = "paired completed CPU grid construction timing; run alone in release mode"]
fn paired_previous_current_grid_build() {
    eprintln!("CONDITIONS warmups=3 pairs=30 boundary=completed_build_grid_excluding_input_generation_checks_and_returned_output_drop");
    for n in [257_usize, 65536, 1048579] {
        for dims in [[1, 1, 1], [16, 16, 16], [128, 128, 64], [256, 256, 128]] {
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
            for pair in 0..30 {
                if pair % 2 == 0 {
                    old.push(run(false));
                    new.push(run(true));
                } else {
                    new.push(run(true));
                    old.push(run(false));
                }
            }
            eprintln!("PAIRED {{\"points\":{n},\"cells\":{cells},\"saved_cursor_bytes\":{},\"previous_ms\":{old:?},\"current_ms\":{new:?}}}",
                4 * (cells + 1));
        }
    }
}
