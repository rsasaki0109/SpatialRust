use super::*;
use spatialrust_search::KdTree;
use std::{hint::black_box, time::Instant};

#[path = "tree_staging_gpu_tests.rs"]
mod gpu;

#[path = "tree_staging_benchmark.rs"]
mod benchmark;

#[test]
fn worker_counts_respect_cpu_grain_threshold_and_maximum() {
    for available in [0, 1, 2, 3, 6, 12, usize::MAX] {
        for n in [0, 1, 131_073, PARALLEL_MIN_POINTS - 1] {
            assert_eq!(worker_count(n, available), 1);
        }
        for n in [PARALLEL_MIN_POINTS, PARALLEL_MIN_POINTS + 1, 524_289, usize::MAX] {
            assert_eq!(
                worker_count(n, available),
                available.clamp(1, MAX_WORKERS).min(n / POINTS_PER_WORKER)
            );
        }
    }
}

#[test]
fn scan_only_validation_matches_packing_without_output_allocation() {
    for n in [0, 1, 17, 524_287, 524_288, 524_289] {
        let mut data = fixture(n, 3);
        let indices = order(n);
        for workers in [0, 1, 2, 3, 4, 12] {
            let axes = [&data[0][..], &data[1][..], &data[2][..]];
            assert!(validate_with_workers(axes, workers).is_ok());
            assert!(validate(axes).is_ok());
            if n != 0 {
                for axis in 0..3 {
                    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                        let i = n - 1;
                        let old = data[axis][i];
                        data[axis][i] = bad;
                        let axes = [&data[0][..], &data[1][..], &data[2][..]];
                        assert_eq!(
                            validate_with_workers(axes, workers).err().unwrap(),
                            previous_prepare(axes, &indices).err().unwrap()
                        );
                        data[axis][i] = old;
                    }
                }
            }
        }
    }
    assert!(validate([&[0.0], &[], &[0.0]]).is_err());
}

fn compare(axes: [&[f32]; 3], order: &[u32], rows: &[Point]) {
    assert_eq!(rows.len(), order.len());
    for (row, &index) in rows.iter().zip(order) {
        let i = index as usize;
        assert_eq!(row.index, index);
        assert_eq!(
            row.xyz.map(f32::to_bits),
            [axes[0][i].to_bits(), axes[1][i].to_bits(), axes[2][i].to_bits()]
        );
    }
}

fn order(n: usize) -> Vec<u32> {
    let mut order: Vec<_> = (0..n as u32).rev().collect();
    if n != 0 {
        order.rotate_left(n / 3);
    }
    order
}

#[test]
fn ordered_points_preserve_original_coordinate_bits_indices_and_capacity() {
    let mut words = 0;
    for n in
        [0, 1, 17, 65_537, 131_073, 262_143, 262_144, 262_145, 524_287, 524_288, 524_289, 1_048_577]
    {
        for profile in 0..4 {
            let data = fixture(n, profile);
            let axes = [&data[0][..], &data[1][..], &data[2][..]];
            let order = order(n);
            let old = previous_prepare(axes, &order).unwrap();
            compare(axes, &order, &old);
            for available in [0, 1, 2, 3, 6, 12] {
                let rows = prepare_with_workers(axes, &order, available).unwrap();
                assert_eq!(rows.capacity(), n);
                assert_eq!(
                    bytemuck::cast_slice::<_, u32>(&rows),
                    bytemuck::cast_slice::<_, u32>(&old)
                );
                compare(axes, &order, &rows);
                words += n * 4;
            }
            compare(axes, &order, &prepare(axes, &order).unwrap());
        }
    }
    println!("TREE_STAGING_CPU_WORDS {words}");
}

#[test]
fn invalid_original_coordinates_worker_boundaries_and_private_shapes_reject() {
    let n = 524_305;
    let mut data = fixture(n, 0);
    let order = order(n);
    let mut rejected = 0;
    for available in [1, 2, 3, 4, 12] {
        let chunk = n.div_ceil(worker_count(n, available));
        let mut offsets = vec![0, n - 1];
        for start in (0..n).step_by(chunk) {
            offsets.extend([start, (start + chunk).min(n) - 1]);
        }
        offsets.sort_unstable();
        offsets.dedup();
        for axis in 0..3 {
            for &offset in &offsets {
                let original = data[axis][offset];
                for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    data[axis][offset] = bad;
                    let axes = [&data[0][..], &data[1][..], &data[2][..]];
                    assert_eq!(
                        prepare_with_workers(axes, &order, available).err().unwrap(),
                        previous_prepare(axes, &order).err().unwrap()
                    );
                    rejected += 1;
                }
                data[axis][offset] = original;
            }
        }
        for short in 0..3 {
            let axes = std::array::from_fn(|a| &data[a][..n - usize::from(a == short)]);
            assert!(prepare_with_workers(axes, &order, available).is_err());
        }
        let axes = [&data[0][..], &data[1][..], &data[2][..]];
        assert!(prepare_with_workers(axes, &order[1..], available).is_err());
    }
    // Validation covers original axes, including unreferenced coordinates in
    // a malformed private order; it is independent of gather band membership.
    data[2][n - 1] = f32::NAN;
    let repeated = vec![0; n];
    for available in [1, 2, 4] {
        let axes = [&data[0][..], &data[1][..], &data[2][..]];
        assert!(prepare_with_workers(axes, &repeated, available).is_err());
    }
    println!("TREE_STAGING_FROZEN_ERRORS {rejected}");
}

#[test]
fn concurrent_callers_and_cpu_tree_leaf_order_remain_exact() {
    let data = fixture(PARALLEL_MIN_POINTS + 17, 3);
    let order = order(data[0].len());
    std::thread::scope(|scope| {
        for available in [1, 2, 3, 4] {
            let data = &data;
            let order = &order;
            scope.spawn(move || {
                let axes = [&data[0][..], &data[1][..], &data[2][..]];
                compare(axes, order, &prepare_with_workers(axes, order, available).unwrap());
            });
        }
    });
    for profile in 0..4 {
        let data = fixture(257, profile);
        let tree = KdTree::from_slices(&data[0], &data[1], &data[2]);
        let layout = tree.layout();
        let (x, y, z) = layout.positions();
        compare(
            [x, y, z],
            layout.point_order(),
            &prepare([x, y, z], layout.point_order()).unwrap(),
        );
    }
}

// Parent upload's validation and point packing, without device work or nodes.
fn previous_prepare(axes: [&[f32]; 3], order: &[u32]) -> SpatialResult<Vec<Point>> {
    let [x, y, z] = axes;
    if x.iter().chain(y).chain(z).any(|v| !v.is_finite()) {
        return Err(invalid());
    }
    Ok(order
        .iter()
        .map(|&index| {
            let i = index as usize;
            Point { xyz: [x[i], y[i], z[i]], index }
        })
        .collect())
}

fn fixture(n: usize, profile: usize) -> [Vec<f32>; 3] {
    std::array::from_fn(|axis| {
        (0..n)
            .map(|i| match profile {
                0 => ((i * 17 + axis * 13) % 100_003) as f32 * 0.125 - 6_000.0,
                1 => f32::from_bits(((i + axis) as u32 & 1) << 31),
                2 => f32::from_bits((i as u32 % 23) | (((i + axis) as u32 & 1) << 31)),
                _ => f32::from_bits(
                    (i as u32).wrapping_mul(0x9e37_79b9).rotate_left(axis as u32 * 7) & 0xff7f_ffff,
                ),
            })
            .collect()
    })
}

#[test]
#[ignore = "requires isolated native release measurement"]
fn survey_bounded_host_tree_packing() {
    println!(
        "TREE_STAGING_ENV available_workers={}",
        std::thread::available_parallelism().map_or(1, |n| n.get())
    );
    for (n, profile) in [
        (65_537, 0),
        (131_073, 0),
        (262_145, 0),
        (524_289, 0),
        (1_048_577, 0),
        (1_048_577, 1),
        (1_048_577, 2),
    ] {
        let data = fixture(n, profile);
        let tree = KdTree::from_slices(&data[0], &data[1], &data[2]);
        let layout = tree.layout();
        let (x, y, z) = layout.positions();
        let axes = [x, y, z];
        let order = layout.point_order();
        let mut samples: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..5 {
                let route = (round + offset) % 5;
                let start = Instant::now();
                let rows = match route {
                    0 => previous_prepare(black_box(axes), black_box(order)),
                    1 => prepare_with_workers(black_box(axes), black_box(order), 1),
                    2 => prepare_with_workers(black_box(axes), black_box(order), 2),
                    3 => prepare_with_workers(black_box(axes), black_box(order), 4),
                    _ => prepare(black_box(axes), black_box(order)),
                }
                .unwrap();
                black_box(&rows);
                drop(rows);
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                if round >= 3 {
                    samples[route].push(ms);
                }
            }
        }
        println!("TREE_STAGING_RECORD {{\"points\":{n},\"profile\":{profile},\"old_ms\":{:?},\"one_ms\":{:?},\"two_ms\":{:?},\"four_ms\":{:?},\"auto_ms\":{:?}}}",samples[0],samples[1],samples[2],samples[3],samples[4]);
    }
}
