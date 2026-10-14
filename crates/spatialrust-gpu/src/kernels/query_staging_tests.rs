use super::*;
use std::hint::black_box;
use std::time::Instant;

#[path = "query_staging_gpu_tests.rs"]
mod gpu;

#[path = "query_staging_benchmark.rs"]
mod benchmark;

#[test]
fn bounded_workers_respect_threshold_grain_and_cpu_budget() {
    for available in [0, 1, 2, 6, 12, usize::MAX] {
        for n in [0, 1, 65_537, PARALLEL_MIN_POINTS - 1] {
            assert_eq!(worker_count(n, available), 1);
        }
        for n in [PARALLEL_MIN_POINTS, PARALLEL_MIN_POINTS + 1, 1_048_577, usize::MAX] {
            let count = worker_count(n, available);
            assert_eq!(count, available.clamp(1, MAX_WORKERS).min(n / POINTS_PER_WORKER));
        }
    }
}

fn compare(axes: [&[f32]; 3], rows: &[[f32; 4]]) {
    assert_eq!(rows.len(), axes[0].len());
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(
            row.map(f32::to_bits),
            [axes[0][i].to_bits(), axes[1][i].to_bits(), axes[2][i].to_bits(), 0]
        );
    }
}

#[test]
fn original_bits_order_and_padding_match_at_worker_boundaries() {
    let mut words = 0;
    for n in [
        0,
        1,
        17,
        65_537,
        PARALLEL_MIN_POINTS - 1,
        PARALLEL_MIN_POINTS,
        PARALLEL_MIN_POINTS + 1,
        1_048_577,
    ] {
        for profile in 0..4 {
            let data = fixture(n, profile);
            let axes = [&data[0][..], &data[1][..], &data[2][..]];
            let old = previous_prepare(axes).unwrap();
            compare(axes, &old);
            for available in [0, 1, 2, 6, 12, usize::MAX] {
                let rows = prepare_with_workers(axes, available).unwrap();
                assert_eq!(
                    bytemuck::cast_slice::<_, u32>(&rows),
                    bytemuck::cast_slice::<_, u32>(&old)
                );
                compare(axes, &rows);
                words += n * 4;
            }
            compare(axes, &prepare(axes).unwrap());
        }
    }
    println!("QUERY_STAGING_CPU_WORDS {words}");
}

#[test]
fn invalid_axes_and_each_worker_band_return_the_frozen_error() {
    let n = 1_048_593;
    let mut data = fixture(n, 0);
    for available in [1, 2, 4, 12] {
        let chunk = n.div_ceil(worker_count(n, available));
        let mut offsets = vec![0, n - 1];
        for start in (0..n).step_by(chunk) {
            offsets.push(start);
            offsets.push((start + chunk).min(n) - 1);
        }
        offsets.sort_unstable();
        offsets.dedup();
        for axis in 0..3 {
            for &offset in &offsets {
                let original = data[axis][offset];
                for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    data[axis][offset] = value;
                    let axes = [&data[0][..], &data[1][..], &data[2][..]];
                    assert_eq!(
                        prepare_with_workers(axes, available).unwrap_err(),
                        previous_prepare(axes).unwrap_err()
                    );
                }
                data[axis][offset] = original;
            }
        }
        for short in 0..3 {
            let axes = std::array::from_fn(|a| &data[a][..n - usize::from(a == short)]);
            assert_eq!(
                prepare_with_workers(axes, available).unwrap_err(),
                previous_prepare(axes).unwrap_err()
            );
        }
    }
    let data = fixture(PARALLEL_MIN_POINTS + 17, 3);
    std::thread::scope(|scope| {
        for available in [1, 2, 4, 12] {
            let data = &data;
            scope.spawn(move || {
                let axes = [&data[0][..], &data[1][..], &data[2][..]];
                compare(axes, &prepare_with_workers(axes, available).unwrap());
            });
        }
    });
}

fn previous_prepare(axes: [&[f32]; 3]) -> SpatialResult<Vec<[f32; 4]>> {
    let [x, y, z] = axes;
    if x.len() != y.len()
        || x.len() != z.len()
        || axes.iter().any(|axis| axis.iter().any(|v| !v.is_finite()))
    {
        return Err(invalid());
    }
    Ok(x.iter().zip(y).zip(z).map(|((&x, &y), &z)| [x, y, z, 0.0]).collect())
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
fn survey_bounded_host_packing() {
    println!(
        "QUERY_STAGING_ENV available_workers={}",
        std::thread::available_parallelism().map_or(1, |n| n.get())
    );
    for (n, profile) in [
        (65_537, 0),
        (262_145, 0),
        (524_289, 0),
        (1_048_577, 0),
        (2_097_153, 0),
        (1_048_577, 1),
        (1_048_577, 2),
    ] {
        let data = fixture(n, profile);
        let axes = [&data[0][..], &data[1][..], &data[2][..]];
        let mut samples: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..5 {
                let route = (round + offset) % 5;
                let start = Instant::now();
                let rows = match route {
                    0 => previous_prepare(black_box(axes)),
                    1 => prepare_with_workers(black_box(axes), 1),
                    2 => prepare_with_workers(black_box(axes), 2),
                    3 => prepare_with_workers(black_box(axes), 4),
                    _ => prepare(black_box(axes)),
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
        println!("QUERY_STAGING_RECORD {{\"points\":{n},\"profile\":{profile},\"old_ms\":{:?},\"one_ms\":{:?},\"two_ms\":{:?},\"four_ms\":{:?},\"auto_ms\":{:?}}}", samples[0], samples[1], samples[2], samples[3], samples[4]);
    }
}
