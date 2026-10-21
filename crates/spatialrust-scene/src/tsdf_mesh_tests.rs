//! Full extraction of tiny analytic planes; not physical accuracy evidence.

use super::*;
use std::collections::BTreeMap;

fn plane(axis: usize, origin: Vec3<f32>, voxel: f32, direction: f32) {
    let dims = [4; 3];
    let mut volume = TsdfVolume::try_new(origin, voxel, dims, 10.).unwrap();
    let base = [origin.x, origin.y, origin.z];
    let level = base[axis] + 0.75 * voxel;
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let position = volume.index_to_world([x, y, z]);
                let coordinate = [position.x, position.y, position.z][axis];
                let flat = volume.flat([x, y, z]);
                volume.distance[flat] = direction * (coordinate - level);
                volume.weight[flat] = 1.;
            }
        }
    }
    let mesh = volume.extract_mesh(0.5);
    assert!(!mesh.is_empty());
    assert_eq!(mesh.positions.len() % 3, 0);
    assert_eq!(mesh.indices.len() % 3, 0);
    type Key = [u32; 3];
    let mut edges = BTreeMap::<(Key, Key), usize>::new();
    let mut area = 0.;
    for triangle in mesh.indices.chunks_exact(3) {
        let p: [[f32; 3]; 3] = std::array::from_fn(|i| {
            let b = triangle[i] as usize * 3;
            assert!(b + 2 < mesh.positions.len());
            [mesh.positions[b], mesh.positions[b + 1], mesh.positions[b + 2]]
        });
        for point in p {
            assert_eq!(point[axis], level);
        }
        let a: [f64; 3] = std::array::from_fn(|i| p[1][i] as f64 - p[0][i] as f64);
        let b: [f64; 3] = std::array::from_fn(|i| p[2][i] as f64 - p[0][i] as f64);
        let n = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        assert!(n[axis] * direction as f64 > 0., "every plane triangle must face increasing field");
        area += n[axis].abs() * 0.5;
        let keys = p.map(|p| p.map(f32::to_bits));
        for (a, b) in [(keys[0], keys[1]), (keys[1], keys[2]), (keys[2], keys[0])] {
            assert_ne!(a, b);
            *edges.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let expected_area = (3. * voxel as f64).powi(2);
    assert!((area - expected_area).abs() < 1e-9, "{area} != {expected_area}");
    let mut perimeter = 0.;
    for ((a, b), count) in edges {
        let a = a.map(f32::from_bits);
        let b = b.map(f32::from_bits);
        let boundary = (0..3).filter(|&i| i != axis).any(|i| {
            [base[i] + 0.5 * voxel, base[i] + 3.5 * voxel].iter().any(|&v| a[i] == v && b[i] == v)
        });
        assert_eq!(
            count,
            if boundary { 1 } else { 2 },
            "unmatched/overlapping plane edge {a:?} {b:?}"
        );
        if boundary {
            perimeter += (0..3).map(|i| (a[i] as f64 - b[i] as f64).powi(2)).sum::<f64>().sqrt();
        }
    }
    assert!((perimeter - 12. * voxel as f64).abs() < 1e-9);
}

#[test]
fn extracts_complete_x_plane() {
    plane(0, Vec3::new(0., 0., 0.), 1., 1.);
}
#[test]
fn extracts_complete_y_plane() {
    plane(1, Vec3::new(0., 0., 0.), 1., 1.);
}
#[test]
fn extracts_complete_z_plane() {
    plane(2, Vec3::new(0., 0., 0.), 1., 1.);
}

#[test]
fn translated_scaled_planes_keep_area_and_shared_edges() {
    for axis in 0..3 {
        for voxel in [0.25, 2.] {
            plane(axis, Vec3::new(-2., 1., -3.), voxel, 1.);
        }
    }
}

#[test]
fn reversed_plane_fields_reverse_every_triangle() {
    for axis in 0..3 {
        plane(axis, Vec3::new(0., 0., 0.), 1., -1.);
    }
}
