//! Independent affine-cut geometry; no production case-table oracle.

use super::*;
use std::collections::BTreeMap;

type Point = [f64; 3];

fn point(p: Vec3<f32>) -> Point {
    [p.x as f64, p.y as f64, p.z as f64]
}

fn sub(a: Point, b: Point) -> Point {
    std::array::from_fn(|i| a[i] - b[i])
}

fn cross(a: Point, b: Point) -> Point {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: Point, b: Point) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}

fn normal(a: Point, b: Point, c: Point) -> Point {
    cross(sub(b, a), sub(c, a))
}

fn shares_corner(a: [usize; 2], b: [usize; 2]) -> bool {
    a.iter().any(|v| b.contains(v))
}

fn unit() -> [Vec3<f32>; 4] {
    [Vec3::new(0., 0., 0.), Vec3::new(1., 0., 0.), Vec3::new(0., 1., 0.), Vec3::new(0., 0., 1.)]
}

fn values(mask: u8, magnitudes: [f32; 4]) -> [f32; 4] {
    std::array::from_fn(|i| if mask & (1 << i) == 0 { magnitudes[i] } else { -magnitudes[i] })
}

fn emit(corners: [Vec3<f32>; 4], values: [f32; 4], level: f32) -> (Vec<f32>, Vec<u32>) {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    polygonise_tet(&mut positions, &mut indices, corners, values, level);
    (positions, indices)
}

fn verify(corners: [Vec3<f32>; 4], values: [f32; 4], level: f32) {
    // Enumerate all six tetrahedron edges independently of the case table.
    let mut cuts = Vec::new();
    for a in 0..4 {
        for b in a + 1..4 {
            if (values[a] < level) != (values[b] < level) {
                let t = (level as f64 - values[a] as f64) / (values[b] as f64 - values[a] as f64);
                let p = point(corners[a]);
                let q = point(corners[b]);
                cuts.push(([a, b], std::array::from_fn::<_, 3, _>(|i| p[i] + t * (q[i] - p[i]))));
            }
        }
    }
    let (positions, indices) = emit(corners, values, level);
    let expected_triangles = match cuts.len() {
        0 => 0,
        3 => 1,
        4 => 2,
        _ => panic!("invalid cut"),
    };
    assert_eq!(indices.len(), expected_triangles * 3);
    assert_eq!(positions.len(), indices.len() * 3);
    if cuts.is_empty() {
        return;
    }

    let p = corners.map(point);
    let e = [sub(p[1], p[0]), sub(p[2], p[0]), sub(p[3], p[0])];
    let reciprocal = [cross(e[1], e[2]), cross(e[2], e[0]), cross(e[0], e[1])];
    let determinant = dot(e[0], reciprocal[0]);
    assert!(determinant > 0.);
    let gradient: Point = std::array::from_fn(|i| {
        (0..3).map(|j| (values[j + 1] as f64 - values[0] as f64) * reciprocal[j][i]).sum::<f64>()
            / determinant
    });
    let scale = p.iter().flatten().fold(1.0f64, |a, b| a.max(b.abs()));
    let tolerance2 = (scale * 1e-6).powi(2);
    let mut incidence = BTreeMap::<(usize, usize), usize>::new();
    let mut area = 0.;
    for triangle in indices.chunks_exact(3) {
        let vertices: [Point; 3] = std::array::from_fn(|i| {
            let base = triangle[i] as usize * 3;
            assert!(base + 2 < positions.len());
            [positions[base] as f64, positions[base + 1] as f64, positions[base + 2] as f64]
        });
        let ids: [usize; 3] = std::array::from_fn(|i| {
            let matches: Vec<_> = cuts
                .iter()
                .enumerate()
                .filter(|(_, (_, p))| {
                    let delta = sub(vertices[i], *p);
                    dot(delta, delta) <= tolerance2
                })
                .map(|(j, _)| j)
                .collect();
            assert_eq!(matches.len(), 1, "vertex must be on exactly one cut edge");
            matches[0]
        });
        let n = normal(vertices[0], vertices[1], vertices[2]);
        assert!(dot(n, gradient) > 0., "winding must point toward increasing field: {values:?}");
        area += dot(n, n).sqrt() * 0.5;
        for (a, b) in [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
            assert_ne!(a, b);
            *incidence.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    // A surface boundary lies on one tetrahedron face: its cut edges share a
    // corner. A quad's shared diagonal joins disjoint tetrahedron edges.
    let mut diagonals = 0;
    for a in 0..cuts.len() {
        for b in a + 1..cuts.len() {
            let count = *incidence.get(&(a, b)).unwrap_or(&0);
            if shares_corner(cuts[a].0, cuts[b].0) {
                assert_eq!(count, 1, "polygon boundary must occur once: {values:?}");
            } else {
                assert!(count == 0 || count == 2);
                diagonals += usize::from(count == 2);
            }
        }
    }
    assert_eq!(diagonals, usize::from(cuts.len() == 4));
    // Construct an independent boundary cycle from face incidence, then find
    // its area. No triangle/table order or production interpolation is reused.
    let mut order = vec![0];
    while order.len() < cuts.len() {
        let previous = *order.last().unwrap();
        let next = (0..cuts.len())
            .find(|&i| !order.contains(&i) && shares_corner(cuts[previous].0, cuts[i].0))
            .unwrap();
        order.push(next);
    }
    assert!(shares_corner(cuts[*order.last().unwrap()].0, cuts[order[0]].0));
    let expected_area: f64 = (1..order.len() - 1)
        .map(|i| {
            let n = normal(cuts[order[0]].1, cuts[order[i]].1, cuts[order[i + 1]].1);
            dot(n, n).sqrt() * 0.5
        })
        .sum();
    assert!((area - expected_area).abs() <= expected_area * 2e-5, "area {area} != {expected_area}");
}

#[test]
fn quad_0011() {
    verify(unit(), values(0b0011, [1.; 4]), 0.);
}
#[test]
fn quad_0101() {
    verify(unit(), values(0b0101, [1.; 4]), 0.);
}
#[test]
fn quad_0110() {
    verify(unit(), values(0b0110, [1.; 4]), 0.);
}
#[test]
fn quad_1001() {
    verify(unit(), values(0b1001, [1.; 4]), 0.);
}
#[test]
fn quad_1010() {
    verify(unit(), values(0b1010, [1.; 4]), 0.);
}
#[test]
fn quad_1100() {
    verify(unit(), values(0b1100, [1.; 4]), 0.);
}
#[test]
fn outside_corner_two_has_outward_winding() {
    verify(unit(), values(0b1011, [1.; 4]), 0.);
}

#[test]
fn all_sixteen_masks_cover_their_cut_polygon() {
    for mask in 0..16 {
        verify(unit(), values(mask, [1.; 4]), 0.);
    }
}

#[test]
fn unequal_edge_magnitudes_cover_their_cut_polygon() {
    for mask in 0..16 {
        for shift in 0..4 {
            let magnitudes = std::array::from_fn(|i| [0.25, 0.5, 1., 2.][(i + shift) % 4]);
            verify(unit(), values(mask, magnitudes), 0.);
        }
    }
}

fn cube() -> [Vec3<f32>; 8] {
    [
        Vec3::new(0., 0., 0.),
        Vec3::new(1., 0., 0.),
        Vec3::new(1., 1., 0.),
        Vec3::new(0., 1., 0.),
        Vec3::new(0., 0., 1.),
        Vec3::new(1., 0., 1.),
        Vec3::new(1., 1., 1.),
        Vec3::new(0., 1., 1.),
    ]
}

#[test]
fn all_cube_tetrahedra_have_consistent_surface_topology() {
    for tet in tetrahedra() {
        for mask in 0..16 {
            verify(tet.map(|i| cube()[i]), values(mask, [1.; 4]), 0.);
        }
    }
}

#[test]
fn translated_scaled_tetrahedra_and_shifted_isolevel() {
    for (scale, offset) in
        [(0.25, Vec3::new(-2., 1., -3.)), (2., Vec3::new(3., -4., 2.)), (1., Vec3::new(0., 0., 0.))]
    {
        for tet in tetrahedra() {
            let corners = tet.map(|i| {
                let p = cube()[i];
                Vec3::new(p.x * scale + offset.x, p.y * scale + offset.y, p.z * scale + offset.z)
            });
            for mask in 0..16 {
                verify(corners, values(mask, [0.5, 1., 2., 0.25]).map(|v| v + 4.), 4.);
            }
        }
    }
}

#[test]
fn complementary_fields_reverse_surface_normals() {
    for mask in 1..15 {
        let v = values(mask, [1.; 4]);
        verify(unit(), v, 0.);
        verify(unit(), v.map(|x| -x), 0.);
        let summed_normal = |field| {
            let (p, ids) = emit(unit(), field, 0.);
            ids.chunks_exact(3).fold([0.; 3], |sum, tri| {
                let v: [Point; 3] = std::array::from_fn(|i| {
                    let b = tri[i] as usize * 3;
                    [p[b] as f64, p[b + 1] as f64, p[b + 2] as f64]
                });
                let n = normal(v[0], v[1], v[2]);
                std::array::from_fn(|i| sum[i] + n[i])
            })
        };
        let a = summed_normal(v);
        let b = summed_normal(v.map(|x| -x));
        for i in 0..3 {
            assert!((a[i] + b[i]).abs() < 1e-12);
        }
    }
}

#[test]
fn empty_cases_leave_existing_output_untouched() {
    for field in [[1.; 4], [-1.; 4], [0.; 4]] {
        let mut p = vec![9., 8., 7.];
        let mut ids = vec![0, 0, 0];
        polygonise_tet(&mut p, &mut ids, unit(), field, 0.);
        assert_eq!(p, [9., 8., 7.]);
        assert_eq!(ids, [0, 0, 0]);
    }
}

#[test]
fn appended_triangles_use_existing_vertex_offset() {
    for mask in 1..15 {
        let field = values(mask, [1.; 4]);
        let (expected_p, expected_ids) = emit(unit(), field, 0.);
        let mut p = vec![9., 8., 7.];
        let mut ids = vec![0, 0, 0];
        polygonise_tet(&mut p, &mut ids, unit(), field, 0.);
        assert_eq!(&p[3..], expected_p);
        assert_eq!(&ids[3..], expected_ids.iter().map(|i| i + 1).collect::<Vec<_>>());
    }
}

#[test]
fn interpolation_endpoints_and_epsilon_are_preserved() {
    let a = Vec3::new(1., 2., 3.);
    let b = Vec3::new(5., 6., 7.);
    assert_eq!(interp(0., a, b, 0., 1.), a);
    assert_eq!(interp(0., a, b, -1., 0.), b);
    assert_eq!(interp(0., a, b, 5e-9, 1.), a);
    assert_eq!(interp(10., a, b, 1e-9, 2e-9), a);
    assert_eq!(interp(0., a, b, -1., 1.), Vec3::new(3., 4., 5.));
}
