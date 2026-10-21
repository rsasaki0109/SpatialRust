//! Synthetic discretized-support checks, not physical reconstruction evidence.

use super::*;
use spatialrust_core::{PointCloudBuilder, StandardSchemas};
use spatialrust_math::Quat;

fn volume() -> TsdfVolume {
    TsdfVolume::try_new(Vec3::new(0., 0., 0.), 1., [4, 4, 4], 0.5).unwrap()
}

fn sensor(point: Vec3<f32>) -> Vec3<f32> {
    point - Vec3::new(0., 0., 1.)
}

fn assert_words(actual: &TsdfVolume, expected: &TsdfVolume) {
    assert_eq!(actual.dims, expected.dims);
    for (index, (a, b)) in actual.distance.iter().zip(&expected.distance).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "distance {index}: {a} vs {b}");
    }
    for (index, (a, b)) in actual.weight.iter().zip(&expected.weight).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "weight {index}: {a} vs {b}");
    }
}

fn untouched(point: Vec3<f32>) {
    let mut actual = volume();
    let before = actual.clone();
    actual.integrate_point(point, sensor(point));
    assert_words(&actual, &before);
}

// Independent support enumeration: enumerate every valid cell, then test
// membership in unbounded floor ranges. Dyadic fixtures make these f64 ranges
// exact; no clamped endpoint or production traversal helper is reused.
fn reference(volume: &mut TsdfVolume, point: Vec3<f32>, origin: Vec3<f32>) {
    let depth = (point - origin).length();
    if !(depth.is_finite() && depth > 1e-5) {
        return;
    }
    let ray = (point - origin).normalize();
    let coordinates = [point.x, point.y, point.z];
    let grid = [volume.origin.x, volume.origin.y, volume.origin.z];
    let lo: [f64; 3] = std::array::from_fn(|axis| {
        ((coordinates[axis] as f64 - volume.truncation as f64 - grid[axis] as f64)
            / volume.voxel_size as f64)
            .floor()
    });
    let hi: [f64; 3] = std::array::from_fn(|axis| {
        ((coordinates[axis] as f64 + volume.truncation as f64 - grid[axis] as f64)
            / volume.voxel_size as f64)
            .floor()
    });
    for z in 0..volume.dims[2] {
        for y in 0..volume.dims[1] {
            for x in 0..volume.dims[0] {
                let index = [x, y, z];
                if (0..3)
                    .any(|axis| (index[axis] as f64) < lo[axis] || (index[axis] as f64) > hi[axis])
                {
                    continue;
                }
                let center = volume.index_to_world(index);
                let sdf = (point - center).dot(ray).clamp(-volume.truncation, volume.truncation);
                let flat = volume.flat(index);
                let old = volume.weight[flat];
                let new = old + 1.0;
                volume.distance[flat] = (volume.distance[flat] * old + sdf) / new;
                volume.weight[flat] = new;
            }
        }
    }
}

// Frozen parent integrator: original quantization, clamp, visitation and math.
fn parent_integrate(volume: &mut TsdfVolume, point: Vec3<f32>, sensor_origin: Vec3<f32>) {
    let depth = (point - sensor_origin).length();
    if !(depth.is_finite() && depth > 1e-5) {
        return;
    }
    let ray = (point - sensor_origin).normalize();
    let radius = volume.truncation;
    let min = point - Vec3::new(radius, radius, radius);
    let max = point + Vec3::new(radius, radius, radius);
    let i0 = parent_clamped(volume, min);
    let i1 = parent_clamped(volume, max);
    for z in i0[2]..=i1[2] {
        for y in i0[1]..=i1[1] {
            for x in i0[0]..=i1[0] {
                let center = volume.index_to_world([x, y, z]);
                let sdf = (point - center).dot(ray).clamp(-volume.truncation, volume.truncation);
                let flat = volume.flat([x, y, z]);
                let w_old = volume.weight[flat];
                let w_new = w_old + 1.0;
                volume.distance[flat] = (volume.distance[flat] * w_old + sdf) / w_new;
                volume.weight[flat] = w_new;
            }
        }
    }
}

fn parent_clamped(volume: &TsdfVolume, point: Vec3<f32>) -> [usize; 3] {
    let ix = ((point.x - volume.origin.x) / volume.voxel_size).floor() as isize;
    let iy = ((point.y - volume.origin.y) / volume.voxel_size).floor() as isize;
    let iz = ((point.z - volume.origin.z) / volume.voxel_size).floor() as isize;
    [
        ix.clamp(0, volume.dims[0] as isize - 1) as usize,
        iy.clamp(0, volume.dims[1] as isize - 1) as usize,
        iz.clamp(0, volume.dims[2] as isize - 1) as usize,
    ]
}

#[test]
fn outside_positive_x() {
    untouched(Vec3::new(10.5, 1.5, 1.5));
}
#[test]
fn outside_negative_x() {
    untouched(Vec3::new(-10.5, 1.5, 1.5));
}
#[test]
fn outside_positive_y() {
    untouched(Vec3::new(1.5, 10.5, 1.5));
}
#[test]
fn outside_negative_y() {
    untouched(Vec3::new(1.5, -10.5, 1.5));
}
#[test]
fn outside_positive_z() {
    untouched(Vec3::new(1.5, 1.5, 10.5));
}
#[test]
fn outside_negative_z() {
    untouched(Vec3::new(1.5, 1.5, -10.5));
}

#[test]
fn disjoint_edge_does_not_update_clamped_edge() {
    untouched(Vec3::new(10.5, -10.5, 1.5));
}

#[test]
fn disjoint_corner_does_not_update_clamped_corner() {
    untouched(Vec3::new(-10.5, 10.5, -10.5));
}

#[test]
fn finite_signed_index_saturation_stays_disjoint() {
    for x in [-1e30, 1e30] {
        untouched(Vec3::new(x, 1.5, 1.5));
    }
}

#[test]
fn partial_overlap_retains_each_boundary_plane() {
    for axis in 0..3 {
        for edge in [-0.25, 4.25] {
            let mut xyz = [1.5; 3];
            xyz[axis] = edge;
            let point = Vec3::new(xyz[0], xyz[1], xyz[2]);
            let mut actual = volume();
            let mut expected = actual.clone();
            actual.integrate_point(point, sensor(point));
            reference(&mut expected, point, sensor(point));
            assert_words(&actual, &expected);
            assert_eq!(actual.weight.iter().filter(|w| **w != 0.).count(), 4);
            for (flat, weight) in actual.weight.iter().enumerate() {
                let coordinate = match axis {
                    0 => flat % 4,
                    1 => (flat / 4) % 4,
                    _ => flat / 16,
                };
                if *weight != 0. {
                    assert_eq!(coordinate, if edge < 0. { 0 } else { 3 });
                }
            }
        }
    }
}

#[test]
fn exact_endpoints_preserve_upper_inclusive_floor_quantization() {
    let mut lower = volume();
    let point = Vec3::new(-0.5, 1.5, 1.5);
    lower.integrate_point(point, sensor(point));
    assert_eq!(lower.weight.iter().filter(|w| **w != 0.).count(), 4);
    untouched(Vec3::new(4.5, 1.5, 1.5));
}

#[test]
fn translated_scaled_grid_uses_its_own_bounds() {
    for point in [
        Vec3::new(-0.75, -0.75, -0.75),
        Vec3::new(0.375, -0.75, -0.75),
        Vec3::new(-2.375, -0.75, -0.75),
    ] {
        let mut actual =
            TsdfVolume::try_new(Vec3::new(-2., -2., -2.), 0.5, [4, 4, 4], 0.25).unwrap();
        let mut expected = actual.clone();
        actual.integrate_point(point, sensor(point));
        reference(&mut expected, point, sensor(point));
        assert_words(&actual, &expected);
    }
}

#[test]
fn exhaustive_dyadic_support_matches_independent_cell_enumeration() {
    let coordinates = [-2., -0.5, 0., 0.5, 1., 3.5, 4., 4.5, 6.];
    for x in coordinates {
        for y in coordinates {
            for z in coordinates {
                let point = Vec3::new(x, y, z);
                let mut actual = volume();
                let mut expected = actual.clone();
                actual.integrate_point(point, sensor(point));
                reference(&mut expected, point, sensor(point));
                assert_words(&actual, &expected);
            }
        }
    }
}

#[test]
fn overlapping_repeated_updates_match_frozen_parent_words() {
    let mut actual = volume();
    let mut expected = actual.clone();
    let coordinates = [-0.25, 0., 0.1, 0.5, 1.25, 3.75, 4.25];
    for x in coordinates {
        for y in coordinates {
            for z in coordinates {
                let point = Vec3::new(x, y, z);
                actual.integrate_point(point, sensor(point));
                parent_integrate(&mut expected, point, sensor(point));
                assert_words(&actual, &expected);
            }
        }
    }
}

#[test]
fn invalid_and_degenerate_depth_ignoring_is_unchanged() {
    let mut actual = volume();
    let expected = actual.clone();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for point in [Vec3::new(value, 1., 1.), Vec3::new(1., value, 1.), Vec3::new(1., 1., value)]
        {
            actual.integrate_point(point, Vec3::new(0., 0., 0.));
        }
    }
    for depth in [0., 1e-6, 1e-5] {
        actual.integrate_point(Vec3::new(depth, 0., 0.), Vec3::new(0., 0., 0.));
    }
    assert_words(&actual, &expected);
}

fn cloud(points: &[[f32; 3]]) -> PointCloud {
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
    for &point in points {
        builder.push_point(point).unwrap();
    }
    builder.build().unwrap()
}

#[test]
fn cloud_counts_all_submissions_but_only_updates_overlapping_support() {
    let cloud = cloud(&[[1.5, 1.5, 1.5], [10.5, 1.5, 1.5], [-10.5, 1.5, 1.5]]);
    let origin = Vec3::new(1.5, 1.5, 0.5);
    let mut actual = volume();
    let mut expected = actual.clone();
    assert_eq!(actual.integrate_cloud(&cloud, origin).unwrap(), 3);
    reference(&mut expected, Vec3::new(1.5, 1.5, 1.5), origin);
    assert_words(&actual, &expected);
}

#[test]
fn interleaved_xyz_skips_disjoint_samples() {
    let mut actual = volume();
    let mut expected = actual.clone();
    let origin = Vec3::new(1.5, 1.5, 0.5);
    actual.integrate_xyz(&[1.5, 1.5, 1.5, 10.5, 1.5, 1.5, -10.5, 1.5, 1.5], origin).unwrap();
    reference(&mut expected, Vec3::new(1.5, 1.5, 1.5), origin);
    assert_words(&actual, &expected);
}

#[test]
fn posed_cloud_tests_support_after_the_explicit_transform() {
    let cloud = cloud(&[[1.5, 1.5, 1.5], [2.5, 1.5, 1.5]]);
    let pose = Isometry3::new(Quat::<f32>::identity(), Vec3::new(20., 0., 0.));
    let mut actual = volume();
    let expected = actual.clone();
    assert_eq!(actual.integrate_cloud_with_pose(&cloud, pose, Vec3::new(0., 0., 0.)).unwrap(), 2);
    assert_words(&actual, &expected);
}

#[test]
fn rotated_translated_mixed_cloud_matches_world_support() {
    let points = [[1.5, -1.5, 1.5], [1.5, -10.5, 1.5]];
    let cloud = cloud(&points);
    let q = std::f32::consts::FRAC_1_SQRT_2;
    let pose = Isometry3::new(Quat::new(0., 0., q, q), Vec3::new(0.5, 0., 0.));
    let origin = Vec3::new(0., 0., 0.);
    let world_origin = pose.transform_point(origin);
    let mut actual = volume();
    let mut expected = actual.clone();
    assert_eq!(actual.integrate_cloud_with_pose(&cloud, pose, origin).unwrap(), 2);
    for p in points {
        reference(&mut expected, pose.transform_point(Vec3::new(p[0], p[1], p[2])), world_origin);
    }
    assert_words(&actual, &expected);
}

#[test]
fn disjoint_samples_cannot_create_a_spurious_mesh() {
    let mut actual = volume();
    let point = Vec3::new(10.5, 1.5, 1.75);
    actual.integrate_point(point, sensor(point));
    let mesh = actual.extract_mesh(0.5);
    assert!(mesh.positions.is_empty());
    assert!(mesh.indices.is_empty());
}

#[test]
fn skipped_support_does_not_disturb_existing_or_later_updates() {
    let mut actual = volume();
    let mut expected = actual.clone();
    for point in [Vec3::new(1.5, 1.5, 1.5), Vec3::new(10.5, 1.5, 1.5), Vec3::new(1.5, 1.5, 1.75)] {
        actual.integrate_point(point, sensor(point));
        reference(&mut expected, point, sensor(point));
    }
    assert_words(&actual, &expected);
}
