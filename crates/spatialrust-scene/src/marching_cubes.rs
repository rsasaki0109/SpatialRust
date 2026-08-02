//! Marching tetrahedra isosurface extraction (cube → six tetrahedra).

use spatialrust_math::Vec3;

/// Consistent six-tetrahedra covering of the unit cube (corner indices).
const TETS: [[usize; 4]; 6] =
    [[0, 2, 3, 7], [0, 6, 2, 7], [0, 4, 6, 7], [0, 6, 1, 2], [0, 1, 6, 4], [5, 6, 1, 4]];

/// Appends triangles for one tetrahedron to `positions` / `indices`.
pub(crate) fn polygonise_tet(
    positions: &mut Vec<f32>,
    indices: &mut Vec<u32>,
    corners: [Vec3<f32>; 4],
    values: [f32; 4],
    isolevel: f32,
) {
    let mut config = 0u8;
    for (bit, value) in values.iter().enumerate() {
        if *value < isolevel {
            config |= 1 << bit;
        }
    }

    // Triangle edge endpoint pairs for each non-empty configuration.
    // Winding keeps the "inside" (value < isolevel) on the same side.
    let edges: &[[usize; 2]] = match config {
        0b0001 => &[[0, 1], [0, 2], [0, 3]],
        0b0010 => &[[1, 0], [1, 3], [1, 2]],
        0b0011 => &[[0, 3], [0, 2], [1, 3], [1, 2]],
        0b0100 => &[[2, 0], [2, 1], [2, 3]],
        0b0101 => &[[0, 1], [2, 3], [0, 3], [1, 2]],
        0b0110 => &[[0, 1], [1, 3], [0, 2], [2, 3]],
        0b0111 => &[[3, 0], [3, 1], [3, 2]],
        0b1000 => &[[3, 0], [3, 2], [3, 1]],
        0b1001 => &[[0, 1], [0, 2], [1, 3], [2, 3]],
        0b1010 => &[[0, 1], [1, 2], [0, 3], [2, 3]],
        0b1011 => &[[2, 0], [2, 1], [2, 3]],
        0b1100 => &[[0, 2], [1, 2], [0, 3], [1, 3]],
        0b1101 => &[[1, 0], [1, 2], [1, 3]],
        0b1110 => &[[0, 1], [0, 3], [0, 2]],
        _ => return,
    };

    let mut verts = [[0.0f32; 3]; 4];
    let mut vert_count = 0usize;
    for edge in edges {
        let a = edge[0];
        let b = edge[1];
        let p = interp(isolevel, corners[a], corners[b], values[a], values[b]);
        verts[vert_count] = [p.x, p.y, p.z];
        vert_count += 1;
    }

    match vert_count {
        3 => push_triangle(positions, indices, verts[0], verts[1], verts[2]),
        4 => {
            push_triangle(positions, indices, verts[0], verts[1], verts[2]);
            push_triangle(positions, indices, verts[0], verts[2], verts[3]);
        }
        _ => {}
    }
}

fn push_triangle(
    positions: &mut Vec<f32>,
    indices: &mut Vec<u32>,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
) {
    let base = (positions.len() / 3) as u32;
    positions.extend_from_slice(&a);
    positions.extend_from_slice(&b);
    positions.extend_from_slice(&c);
    indices.extend_from_slice(&[base, base + 1, base + 2]);
}

fn interp(isolevel: f32, p1: Vec3<f32>, p2: Vec3<f32>, v1: f32, v2: f32) -> Vec3<f32> {
    if (isolevel - v1).abs() < 1e-8 {
        return p1;
    }
    if (isolevel - v2).abs() < 1e-8 {
        return p2;
    }
    if (v1 - v2).abs() < 1e-8 {
        return p1;
    }
    let t = (isolevel - v1) / (v2 - v1);
    Vec3::new(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y), p1.z + t * (p2.z - p1.z))
}

#[inline]
pub(crate) fn tetrahedra() -> &'static [[usize; 4]; 6] {
    &TETS
}

#[cfg(test)]
mod tests {
    use super::polygonise_tet;
    use spatialrust_math::Vec3;

    #[test]
    fn single_inside_corner_yields_one_triangle() {
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        polygonise_tet(
            &mut positions,
            &mut indices,
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            [-1.0, 1.0, 1.0, 1.0],
            0.0,
        );
        assert_eq!(positions.len(), 9);
        assert_eq!(indices, vec![0, 1, 2]);
    }
}
