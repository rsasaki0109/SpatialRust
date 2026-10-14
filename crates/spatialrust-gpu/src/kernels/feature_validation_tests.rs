use super::*;

// Frozen pre-change predicate, including operation order and empty handling.
fn previous(axes: [&[f32]; 3], normals: Option<&[[f32; 3]]>) -> Option<f64> {
    if axes.into_iter().flatten().any(|v| !v.is_finite())
        || normals.is_some_and(|normals| {
            normals.iter().any(|n| {
                let norm = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
                n.iter().any(|v| !v.is_finite()) || !norm.is_finite() || (norm - 1.0).abs() > 1e-3
            })
        })
    {
        return None;
    }
    if axes[0].is_empty() {
        return Some(0.0);
    }
    let extent: f64 = axes
        .iter()
        .map(|axis| {
            let lo = axis.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = axis.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (f64::from(hi) - f64::from(lo)).powi(2)
        })
        .sum();
    (extent as f32).is_finite().then_some(extent)
}

fn compare(axes: [&[f32]; 3], normals: Option<&[[f32; 3]]>) {
    let expected = previous(axes, normals).map(f64::to_bits);
    for available in [0, 1, 2, 3, 4, 12, usize::MAX] {
        assert_eq!(prepare(axes, normals, available).map(f64::to_bits), expected);
    }
    assert_eq!(valid(axes, normals), expected.is_some());
}

#[test]
fn bounds_match_serial_bits_at_worker_and_extent_boundaries() {
    for n in [0, 1, 4097, 262_145, 524_287, 524_288, 524_289, 1_048_577] {
        for profile in 0..5 {
            let axes: [Vec<f32>; 3] = std::array::from_fn(|a| {
                (0..n)
                    .map(|i| match profile {
                        0 => (i % (a * 5 + 17)) as f32 * 0.125,
                        1 => f32::from_bits(((i + a) as u32 & 1) << 31),
                        2 => f32::from_bits((i as u32 % 31) | (((i + a) as u32 & 1) << 31)),
                        3 => f32::MAX,
                        _ => {
                            if i & 1 == 0 {
                                -f32::MAX
                            } else {
                                f32::MAX
                            }
                        }
                    })
                    .collect()
            });
            compare(axes.each_ref().map(Vec::as_slice), None);
        }
    }
    // Adjacent f32 values straddling representable squared extents.
    let boundary = f32::MAX.sqrt().to_bits();
    for bits in boundary - 8..=boundary + 8 {
        let x = [0.0, f32::from_bits(bits)];
        compare([&x, &[0.0; 2], &[0.0; 2]], None);
    }
    let mut state = 0x5e71_038d_u32;
    let values: Vec<_> = (0..65_537)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Arbitrary finite words, including extreme exponent ranges.
            f32::from_bits(if state & 0x7f80_0000 == 0x7f80_0000 {
                state ^ 0x0080_0000
            } else {
                state
            })
        })
        .collect();
    compare([&values; 3], None);
}

#[test]
fn invalid_values_in_every_worker_and_original_axis_are_rejected() {
    let n = 524_289;
    let mut axes: [Vec<_>; 3] = std::array::from_fn(|_| vec![0.0; n]);
    for a in 0..3 {
        for index in [0, 131_071, 131_072, 262_144, 393_216, n - 1] {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(0x7f80_0001)] {
                axes[a][index] = value;
                compare(axes.each_ref().map(Vec::as_slice), None);
                axes[a][index] = 0.0;
            }
        }
    }
}

#[test]
fn unit_normal_tolerance_and_nonfinite_arithmetic_match_serial() {
    let n = 524_289;
    let axis = vec![0.0; n];
    let mut normals = vec![[0.0, -0.0, 1.0]; n];
    compare([&axis; 3], Some(&normals));
    for index in [0, 131_072, 262_144, 393_216, n - 1] {
        for normal in [[0.0; 3], [f32::MAX; 3], [f32::NAN, 0.0, 1.0], [0.0, f32::INFINITY, 1.0]] {
            normals[index] = normal;
            compare([&axis; 3], Some(&normals));
        }
        normals[index] = [0.0, -0.0, 1.0];
    }
    for sign in [-1.0_f32, 1.0] {
        let center = (1.0 + sign * 1e-3).sqrt().to_bits();
        for bits in center - 32..=center + 32 {
            let normal = [0.0, -0.0, f32::from_bits(bits)];
            compare([&[0.0]; 3], Some(&[normal]));
            normals[n - 1] = normal;
            assert_eq!(
                prepare([&axis; 3], Some(&normals), 4).is_some(),
                previous([&axis; 3], Some(&normals)).is_some()
            );
        }
    }
}

#[test]
fn worker_limits_and_private_shape_checks() {
    for (len, available, expected) in [
        (0, 12, 1),
        (524_287, 12, 1),
        (524_288, 0, 1),
        (524_288, 1, 1),
        (524_288, 2, 2),
        (524_288, 3, 3),
        (524_288, 12, 4),
        (usize::MAX, usize::MAX, 4),
    ] {
        assert_eq!(workers(len, available), expected);
    }
    assert_eq!(prepare([&[], &[], &[]], Some(&[]), 4), Some(0.0));
    assert_eq!(prepare([&[], &[0.0], &[]], None, 4), None);
    assert_eq!(prepare([&[0.0]; 3], Some(&[]), 4), None);
}

#[path = "feature_validation_benchmark.rs"]
mod benchmark;
