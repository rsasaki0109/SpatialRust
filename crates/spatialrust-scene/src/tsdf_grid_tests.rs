//! Tiny grid admission checks; never attempt a large representable allocation.

use super::*;

fn origin() -> Vec3<f32> {
    Vec3::new(0., 0., 0.)
}

fn rejected(origin: Vec3<f32>, voxel: f32, dims: [usize; 3], truncation: f32) {
    let result = std::panic::catch_unwind(|| TsdfVolume::try_new(origin, voxel, dims, truncation));
    assert!(result.is_ok(), "invalid configuration must return an error, not panic");
    assert!(matches!(result.unwrap(), Err(SceneError::InvalidConfiguration(_))));
}

fn rejected_origins(value: f32) {
    for axis in 0..3 {
        let mut coordinates = [0.; 3];
        coordinates[axis] = value;
        rejected(Vec3::new(coordinates[0], coordinates[1], coordinates[2]), 1., [1; 3], 0.5);
    }
}

#[test]
fn rejects_nan_origins() {
    rejected_origins(f32::NAN);
}

#[test]
fn rejects_positive_infinite_origins() {
    rejected_origins(f32::INFINITY);
}

#[test]
fn rejects_negative_infinite_origins() {
    rejected_origins(f32::NEG_INFINITY);
}

#[test]
fn rejects_first_center_overflow() {
    for axis in 0..3 {
        let mut coordinates = [0.; 3];
        coordinates[axis] = f32::MAX;
        rejected(Vec3::new(coordinates[0], coordinates[1], coordinates[2]), f32::MAX, [1; 3], 0.5);
    }
}

#[test]
fn rejects_last_center_overflow() {
    // The first center is finite, but the second center is 1.5 * MAX.
    for axis in 0..3 {
        let mut dims = [1; 3];
        dims[axis] = 2;
        rejected(origin(), f32::MAX, dims, 0.5);
    }
}

#[test]
fn rejects_overflowing_f32_offset() {
    // Exact f64 world coordinates would be finite. The production f32 offset
    // overflows before adding the negative origin, so f64-only checks miss it.
    assert!((-(f32::MAX as f64) + 1.5 * f32::MAX as f64).abs() < f32::MAX as f64);
    for axis in 0..3 {
        let mut coordinates = [0.; 3];
        coordinates[axis] = -f32::MAX;
        let mut dims = [1; 3];
        dims[axis] = 2;
        rejected(Vec3::new(coordinates[0], coordinates[1], coordinates[2]), f32::MAX, dims, 0.5);
    }
}

#[test]
fn rejects_dimension_product_overflow() {
    // Every permutation overflows usize. The old Vec macro panics on capacity
    // before any allocator call; these are never large viable allocations.
    for dims in [[usize::MAX, 2, 1], [1, usize::MAX, 2], [2, 1, usize::MAX]] {
        rejected(origin(), 1., dims, 0.5);
    }
}

#[test]
fn rejects_field_byte_limit() {
    // Count fits usize and its byte product fits usize, but each Vec would
    // exceed Rust's isize::MAX byte limit. No physical allocation is attempted.
    let count = isize::MAX as usize / std::mem::size_of::<f32>() + 1;
    for dims in [[count, 1, 1], [1, count, 1], [1, 1, count]] {
        rejected(origin(), 1., dims, 0.5);
    }
}

#[test]
fn rejects_field_byte_overflow() {
    for dims in [[usize::MAX, 1, 1], [1, usize::MAX, 1], [1, 1, usize::MAX]] {
        rejected(origin(), 1., dims, 0.5);
    }
}

#[test]
fn rejects_invalid_voxel_sizes() {
    for voxel in [0., -0., -1., f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        rejected(origin(), voxel, [1; 3], 0.5);
    }
}

#[test]
fn rejects_invalid_truncations() {
    for truncation in [0., -0., -1., f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        rejected(origin(), 1., [1; 3], truncation);
    }
}

#[test]
fn rejects_zero_dimensions() {
    for dims in [[0, 1, 1], [1, 0, 1], [1, 1, 0], [0; 3]] {
        rejected(origin(), 1., dims, 0.5);
    }
}

#[test]
fn zero_dimensions_reject_before_capacity() {
    for dims in
        [[0, usize::MAX, usize::MAX], [usize::MAX, 0, usize::MAX], [usize::MAX, usize::MAX, 0]]
    {
        rejected(origin(), 1., dims, 0.5);
    }
}

#[test]
fn non_cubic_layout_has_expected_centers_and_offsets() {
    let volume = TsdfVolume::try_new(Vec3::new(-2., -1., -3.), 0.5, [3, 2, 4], 0.25).unwrap();
    assert_eq!(volume.dims(), [3, 2, 4]);
    assert_eq!(volume.distance.len(), 24);
    assert_eq!(volume.weight.len(), 24);
    assert_eq!(volume.index_to_world([0, 0, 0]), Vec3::new(-1.75, -0.75, -2.75));
    assert_eq!(volume.index_to_world([2, 1, 3]), Vec3::new(-0.75, -0.25, -1.25));
    for (offset, index) in [
        (0, [0, 0, 0]),
        (1, [1, 0, 0]),
        (2, [2, 0, 0]),
        (3, [0, 1, 0]),
        (6, [0, 0, 1]),
        (23, [2, 1, 3]),
    ] {
        assert_eq!(volume.flat(index), offset);
    }
}

#[test]
fn accepts_finite_extreme_centers() {
    for (origin, voxel, expected) in [
        (Vec3::new(f32::MAX, f32::MAX, f32::MAX), 1., f32::MAX),
        (Vec3::new(-f32::MAX, -f32::MAX, -f32::MAX), f32::MAX, -f32::MAX * 0.5),
    ] {
        let volume = TsdfVolume::try_new(origin, voxel, [1; 3], f32::MAX).unwrap();
        assert_eq!(volume.index_to_world([0; 3]), Vec3::new(expected, expected, expected));
        assert!(volume.extract_mesh(1.).positions.is_empty());
    }
}

#[test]
fn accepts_subnormal_positive_scales() {
    let tiny = f32::from_bits(1);
    let volume = TsdfVolume::try_new(origin(), tiny, [3; 3], tiny).unwrap();
    assert_eq!(volume.distance.len(), 27);
    assert!(volume.distance.iter().all(|value| value.to_bits() == tiny.to_bits()));
    assert!(volume.weight.iter().all(|value| value.to_bits() == 0));
    // Retain the existing rounded/coincident-center policy at tiny scales.
    assert_eq!(volume.index_to_world([0; 3]), origin());
    assert_eq!(volume.index_to_world([2; 3]).x.to_bits(), 2);
}

#[test]
fn accepted_initial_fields_keep_exact_values() {
    for dims in [[1; 3], [2, 3, 4], [4, 1, 2], [1, 3, 2]] {
        for truncation in [f32::from_bits(1), 0.125, 1., f32::MAX] {
            let volume =
                TsdfVolume::try_new(Vec3::new(-1., 2., -3.), 0.25, dims, truncation).unwrap();
            assert_eq!(volume.dims(), dims);
            let expected_len = dims[0] * dims[1] * dims[2];
            assert_eq!(volume.distance.len(), expected_len);
            assert_eq!(volume.weight.len(), expected_len);
            assert!(volume.distance.iter().all(|value| value.to_bits() == truncation.to_bits()));
            assert!(volume.weight.iter().all(|value| value.to_bits() == 0));
        }
    }
}
