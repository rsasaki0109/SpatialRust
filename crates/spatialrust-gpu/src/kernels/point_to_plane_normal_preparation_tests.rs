use super::super::preparation_reference::previous_normal_preparation as previous;
use super::*;

fn fixture(n: usize, profile: usize) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| {
            let zero = f32::from_bits((i as u32 & 1) << 31);
            match profile {
                0 => [1.0 + (i % 3) as f32 * 0.25, -0.5, zero],
                1 => [zero, -zero, zero],
                2 => [f32::MIN_POSITIVE, -f32::MIN_POSITIVE, zero],
                3 => [f32::MAX, -f32::MAX, zero],
                _ => [1.0, (i % 7) as f32 * 0.03125, zero],
            }
        })
        .collect()
}

fn compare(normals: &[[f32; 3]]) -> usize {
    let expected = previous(normals);
    let mut words = 0;
    for available in [0, 1, 2, 3, 4, 12, usize::MAX] {
        let actual = prepare_with_workers(normals, available);
        match (&expected, actual) {
            (Ok((old, os)), Ok((new, ns))) => {
                assert_eq!(os.to_bits(), ns.to_bits());
                for (a, b) in old.iter().zip(&new) {
                    assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
                    words += 4;
                }
                assert_eq!(old.len(), new.len());
            }
            (Err(old), Err(new)) => assert_eq!(*old, new),
            _ => panic!("normal preparation differs from frozen predicate"),
        }
    }
    assert_eq!(prepare(normals), expected);
    words
}

#[test]
fn exact_normal_words_and_scale_across_bounded_worker_shapes() {
    let mut words = 0;
    for n in [0, 1, 17, 65_537, 262_143, 262_144, 262_145, 524_289, 1_048_577] {
        for profile in 0..5 {
            words += compare(&fixture(n, profile));
        }
    }
    println!("PLANE_STAGING_HOST words={words}");
}

#[test]
fn numeric_boundaries_keep_subnormal_and_scale_decisions() {
    // Adjacent operands at the normalized-component underflow boundary, with
    // exact divisors instead of a reciprocal. Large rows exercise both phases.
    let mut normals = fixture(262_145, 0);
    for scale in [0.5, 1.0, 2.0, 3.0, 1e10, f32::MAX] {
        let threshold = (f64::from(f32::MIN_POSITIVE) * f64::from(scale)) as f32;
        if !threshold.is_finite() {
            continue;
        }
        for bits in threshold.to_bits().saturating_sub(4)..=threshold.to_bits().saturating_add(4) {
            let normal = [scale, f32::from_bits(bits), -0.0];
            compare(&[normal]);
            normals.fill([scale, 0.0, -0.0]);
            normals[262_144] = normal;
            assert_eq!(prepare_with_workers(&normals, 4), previous(&normals));
        }
    }
    for value in
        [f32::from_bits(1), f32::from_bits(f32::MIN_POSITIVE.to_bits() - 1), f32::MIN_POSITIVE]
    {
        compare(&[[value, 0.0, -0.0]]);
    }
    let mut state = 0x7382_b159_u32;
    let arbitrary: Vec<_> = (0..65_537)
        .map(|_| {
            std::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                f32::from_bits(if state & 0x7f80_0000 == 0x7f80_0000 {
                    state ^ 0x0080_0000
                } else {
                    state
                })
            })
        })
        .collect();
    compare(&arbitrary);
}

#[test]
fn nonfinite_and_underflowing_rows_in_each_worker_retain_errors() {
    let mut normals = fixture(262_145, 0);
    for index in [0, 65_536, 65_537, 131_074, 196_611, 262_144] {
        for component in 0..3 {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(1)] {
                normals[index][component] = value;
                compare(&normals);
                normals[index] = [1.0, -0.5, 0.0];
            }
        }
    }
}

#[test]
fn worker_policy_and_cold_parallel_calls_are_bounded() {
    for (n, available, count) in [
        (0, 12, 1),
        (262_143, 12, 1),
        (262_144, 0, 1),
        (262_144, 2, 2),
        (262_145, 3, 3),
        (262_145, 12, 4),
        (usize::MAX, usize::MAX, 4),
    ] {
        assert_eq!(workers(n, available), count);
    }
    let normals = fixture(262_145, 0);
    let expected = previous(&normals).unwrap();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..3).map(|_| scope.spawn(|| prepare(&normals).unwrap())).collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
    });
}

#[path = "point_to_plane_normal_benchmark.rs"]
mod benchmark;
