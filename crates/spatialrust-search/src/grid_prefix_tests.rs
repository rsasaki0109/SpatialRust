use super::*;

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
#[test]
fn worker_prefixes_match_independent_u64_sums_through_maximum_u32_mass() {
    let mut words = 0;
    for n in [
        0_usize,
        1,
        2,
        7,
        255,
        256,
        257,
        65537,
        MIN_PARALLEL_ENTRIES - 1,
        MIN_PARALLEL_ENTRIES,
        MIN_PARALLEL_ENTRIES + 1,
        8388609,
    ] {
        for kind in 0..4 {
            let mut counts: Vec<u32> = (0..n)
                .map(|i| match kind {
                    0 => 0,
                    1 => (i % 7) as u32,
                    2 => u32::from(i % 4096 == 0),
                    _ => 0,
                })
                .collect();
            if kind == 3 && n != 0 {
                counts[n / 2] = u32::MAX;
            }
            let mut total = 0u64;
            let expected: Vec<_> = counts
                .iter()
                .map(|&count| {
                    total += u64::from(count);
                    u32::try_from(total).unwrap()
                })
                .collect();
            for workers in [0, 1, 2, 6] {
                let mut actual = counts.clone();
                parallel_prefix(&mut actual, workers);
                assert_eq!(actual, expected, "n={n}, kind={kind}, workers={workers}");
                words += n;
            }
            let mut actual = counts;
            inclusive_grid_prefix(&mut actual, total as usize);
            assert_eq!(actual, expected, "public staging n={n}, kind={kind}");
            words += n;
        }
    }
    eprintln!("GRID_PREFIX verified {words} inclusive u32 words, including maximum mass and public staging.");
}

#[test]
fn caller_prefix_retains_small_and_empty_behavior() {
    let mut values = [3, 0, 5, 1, 0];
    inclusive_grid_prefix(&mut values, 9);
    assert_eq!(values, [3, 3, 8, 9, 9]);
    inclusive_grid_prefix(&mut [], 0);
}

pub(super) mod previous {
    include!("../tests/fixtures/grid_prefix_serial.rs");
}

#[test]
fn grid_builder_keeps_frozen_stable_indices_and_offsets_for_large_cell_tables() {
    use crate::uniform_grid::{build_grid, build_tests::fixture};
    for n in [17_usize, 257, 4097] {
        for dims in [[256, 128, 128], [256, 256, 128]] {
            let (xyz, _) = fixture(n, dims, 1.0);
            let expected = previous::build_grid(&xyz[0], &xyz[1], &xyz[2], [-4.0; 3], dims, 1.0);
            assert_eq!(build_grid(&xyz[0], &xyz[1], &xyz[2], [-4.0; 3], dims, 1.0), expected);
        }
    }
}

#[test]
fn concurrent_large_grid_calls_keep_independently_owned_csr_results() {
    use crate::uniform_grid::{build_grid, build_tests::fixture};
    let dims = [256, 128, 128];
    let (xyz, _) = fixture(257, dims, 1.0);
    let expected = previous::build_grid(&xyz[0], &xyz[1], &xyz[2], [-4.0; 3], dims, 1.0);
    std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    let first = build_grid(&xyz[0], &xyz[1], &xyz[2], [-4.0; 3], dims, 1.0);
                    let second = build_grid(&xyz[0], &xyz[1], &xyz[2], [-4.0; 3], dims, 1.0);
                    assert_eq!(first, expected);
                    assert_eq!(second, expected);
                })
            })
            .collect();
        for job in jobs {
            job.join().unwrap();
        }
    });
}
