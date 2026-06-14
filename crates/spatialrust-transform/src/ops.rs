//! Cloud transform and utility operations.

use spatialrust_core::{
    FieldSemantic, HasNormals3, HasPositions3, PointBuffer, PointBufferSet, PointCloud,
    SpatialError, SpatialResult,
};
use spatialrust_math::{symmetric_eigen3, Mat3, Mat4, Vec3};

use crate::bounds::{Aabb, Obb};

/// Applies a 4×4 affine transform to a cloud's positions (and normals, if
/// present — normals are rotated by the linear part and renormalized).
pub fn apply_transform(input: &PointCloud, transform: Mat4<f32>) -> SpatialResult<PointCloud> {
    if input.is_empty() {
        return Ok(input.clone());
    }
    let (x, y, z) = input.positions3()?;
    let len = input.len();

    let mut tx = Vec::with_capacity(len);
    let mut ty = Vec::with_capacity(len);
    let mut tz = Vec::with_capacity(len);
    for i in 0..len {
        let p = transform.transform_point(Vec3::new(x[i], y[i], z[i]));
        tx.push(p.x);
        ty.push(p.y);
        tz.push(p.z);
    }

    let has_normals = input.schema().find_semantic(FieldSemantic::NormalX).is_some();
    let normals = if has_normals {
        let (nx, ny, nz) = input.normals3()?;
        let (mut rx, mut ry, mut rz) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..len {
            let n = transform.transform_vector(Vec3::new(nx[i], ny[i], nz[i])).normalize();
            rx.push(n.x);
            ry.push(n.y);
            rz.push(n.z);
        }
        Some((rx, ry, rz))
    } else {
        None
    };

    let mut buffers = PointBufferSet::new();
    for field in input.schema().fields() {
        let buffer = match field.semantic {
            FieldSemantic::PositionX => PointBuffer::from_f32(tx.clone()),
            FieldSemantic::PositionY => PointBuffer::from_f32(ty.clone()),
            FieldSemantic::PositionZ => PointBuffer::from_f32(tz.clone()),
            FieldSemantic::NormalX => PointBuffer::from_f32(normals.as_ref().unwrap().0.clone()),
            FieldSemantic::NormalY => PointBuffer::from_f32(normals.as_ref().unwrap().1.clone()),
            FieldSemantic::NormalZ => PointBuffer::from_f32(normals.as_ref().unwrap().2.clone()),
            _ => clone_buffer(input.field(&field.name)?),
        };
        buffers.insert(field.name.clone(), buffer);
    }
    PointCloud::try_from_parts(input.schema().clone(), buffers, input.metadata().clone())
}

/// Centroid (mean position) of a cloud.
pub fn centroid(input: &PointCloud) -> SpatialResult<Vec3<f32>> {
    if input.is_empty() {
        return Err(SpatialError::InvalidArgument(
            "cannot take centroid of empty cloud".to_owned(),
        ));
    }
    let (x, y, z) = input.positions3()?;
    let n = input.len() as f64;
    let (mut sx, mut sy, mut sz) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..input.len() {
        sx += f64::from(x[i]);
        sy += f64::from(y[i]);
        sz += f64::from(z[i]);
    }
    Ok(Vec3::new((sx / n) as f32, (sy / n) as f32, (sz / n) as f32))
}

/// Axis-aligned bounding box of a cloud.
pub fn bounding_box(input: &PointCloud) -> SpatialResult<Aabb> {
    if input.is_empty() {
        return Err(SpatialError::InvalidArgument("cannot bound an empty cloud".to_owned()));
    }
    let (x, y, z) = input.positions3()?;
    let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for i in 0..input.len() {
        min.x = min.x.min(x[i]);
        min.y = min.y.min(y[i]);
        min.z = min.z.min(z[i]);
        max.x = max.x.max(x[i]);
        max.y = max.y.max(y[i]);
        max.z = max.z.max(z[i]);
    }
    Ok(Aabb::new(min, max))
}

/// Translates a cloud so its centroid sits at the origin.
pub fn recenter(input: &PointCloud) -> SpatialResult<PointCloud> {
    let c = centroid(input)?;
    apply_transform(input, translation(Vec3::new(-c.x, -c.y, -c.z)))
}

/// Uniformly scales a cloud about the origin by `factor`.
pub fn scale_cloud(input: &PointCloud, factor: f32) -> SpatialResult<PointCloud> {
    if !(factor.is_finite()) || factor == 0.0 {
        return Err(SpatialError::InvalidArgument(
            "scale factor must be finite and non-zero".to_owned(),
        ));
    }
    let m = Mat4::from_rows(
        [factor, 0.0, 0.0, 0.0],
        [0.0, factor, 0.0, 0.0],
        [0.0, 0.0, factor, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    );
    apply_transform(input, m)
}

/// Recenters a cloud and scales it so its farthest point is at unit distance —
/// the canonical normalization for learned point-cloud models.
pub fn normalize_unit_sphere(input: &PointCloud) -> SpatialResult<PointCloud> {
    let centered = recenter(input)?;
    let (x, y, z) = centered.positions3()?;
    let mut max_r = 0.0_f32;
    for i in 0..centered.len() {
        let r = (x[i] * x[i] + y[i] * y[i] + z[i] * z[i]).sqrt();
        if r > max_r {
            max_r = r;
        }
    }
    if max_r <= f32::EPSILON {
        return Ok(centered);
    }
    scale_cloud(&centered, 1.0 / max_r)
}

/// Concatenates clouds that share an identical schema into one cloud.
pub fn merge_clouds(clouds: &[&PointCloud]) -> SpatialResult<PointCloud> {
    let first = clouds.first().ok_or_else(|| {
        SpatialError::InvalidArgument("merge needs at least one cloud".to_owned())
    })?;
    let schema = first.schema();
    for cloud in &clouds[1..] {
        if cloud.schema() != schema {
            return Err(SpatialError::InvalidArgument(
                "all clouds must share the same schema to merge".to_owned(),
            ));
        }
    }

    let mut buffers = PointBufferSet::new();
    for field in schema.fields() {
        let sources: Vec<&PointBuffer> =
            clouds.iter().map(|c| c.field(&field.name)).collect::<SpatialResult<_>>()?;
        buffers.insert(field.name.clone(), concat_buffers(&sources));
    }
    PointCloud::try_from_parts(schema.clone(), buffers, first.metadata().clone())
}

/// Oriented bounding box via principal component analysis of the positions.
pub fn oriented_bounding_box(input: &PointCloud) -> SpatialResult<Obb> {
    let c = centroid(input)?;
    let (x, y, z) = input.positions3()?;

    let mut cov = [[0.0_f64; 3]; 3];
    for i in 0..input.len() {
        let d = [f64::from(x[i] - c.x), f64::from(y[i] - c.y), f64::from(z[i] - c.z)];
        for r in 0..3 {
            for col in 0..3 {
                cov[r][col] += d[r] * d[col];
            }
        }
    }
    let eigen = symmetric_eigen3(Mat3::<f64>::from_rows(cov[0], cov[1], cov[2]));
    let axis = |col: usize| {
        Vec3::new(
            eigen.eigenvectors.m[0][col] as f32,
            eigen.eigenvectors.m[1][col] as f32,
            eigen.eigenvectors.m[2][col] as f32,
        )
        .normalize()
    };
    // Eigenvalues are ascending, so column 2 is the principal (largest-variance)
    // axis. Order the box axes principal-first.
    let (a0, a1, a2) = (axis(2), axis(1), axis(0));

    // Project each point onto the axes to find the extents.
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for i in 0..input.len() {
        let d = Vec3::new(x[i] - c.x, y[i] - c.y, z[i] - c.z);
        for (k, ax) in [a0, a1, a2].iter().enumerate() {
            let p = d.dot(*ax);
            min[k] = min[k].min(p);
            max[k] = max[k].max(p);
        }
    }

    let axes = Mat3::from_rows([a0.x, a1.x, a2.x], [a0.y, a1.y, a2.y], [a0.z, a1.z, a2.z]);
    let mid = |k: usize| 0.5 * (min[k] + max[k]);
    let center = c + scale(a0, mid(0)) + scale(a1, mid(1)) + scale(a2, mid(2));
    let half = Vec3::new(0.5 * (max[0] - min[0]), 0.5 * (max[1] - min[1]), 0.5 * (max[2] - min[2]));
    Ok(Obb { center, axes, half_extents: half })
}

fn translation(t: Vec3<f32>) -> Mat4<f32> {
    Mat4::from_rows(
        [1.0, 0.0, 0.0, t.x],
        [0.0, 1.0, 0.0, t.y],
        [0.0, 0.0, 1.0, t.z],
        [0.0, 0.0, 0.0, 1.0],
    )
}

fn scale(v: Vec3<f32>, s: f32) -> Vec3<f32> {
    Vec3::new(v.x * s, v.y * s, v.z * s)
}

fn clone_buffer(buffer: &PointBuffer) -> PointBuffer {
    match buffer {
        PointBuffer::F32(v) => PointBuffer::from_f32(v.clone()),
        PointBuffer::F64(v) => PointBuffer::F64(v.clone()),
        PointBuffer::U8(v) => PointBuffer::U8(v.clone()),
        PointBuffer::U16(v) => PointBuffer::U16(v.clone()),
        PointBuffer::U32(v) => PointBuffer::U32(v.clone()),
        PointBuffer::I32(v) => PointBuffer::I32(v.clone()),
    }
}

fn concat_buffers(sources: &[&PointBuffer]) -> PointBuffer {
    match sources[0] {
        PointBuffer::F32(_) => {
            let mut out = Vec::new();
            for s in sources {
                if let PointBuffer::F32(v) = s {
                    out.extend_from_slice(v);
                }
            }
            PointBuffer::from_f32(out)
        }
        PointBuffer::F64(_) => {
            let mut out = Vec::new();
            for s in sources {
                if let PointBuffer::F64(v) = s {
                    out.extend_from_slice(v);
                }
            }
            PointBuffer::F64(out)
        }
        PointBuffer::U8(_) => concat_into(sources, |b| match b {
            PointBuffer::U8(v) => Some(v.as_slice()),
            _ => None,
        })
        .map(PointBuffer::U8)
        .unwrap(),
        PointBuffer::U16(_) => concat_into(sources, |b| match b {
            PointBuffer::U16(v) => Some(v.as_slice()),
            _ => None,
        })
        .map(PointBuffer::U16)
        .unwrap(),
        PointBuffer::U32(_) => concat_into(sources, |b| match b {
            PointBuffer::U32(v) => Some(v.as_slice()),
            _ => None,
        })
        .map(PointBuffer::U32)
        .unwrap(),
        PointBuffer::I32(_) => concat_into(sources, |b| match b {
            PointBuffer::I32(v) => Some(v.as_slice()),
            _ => None,
        })
        .map(PointBuffer::I32)
        .unwrap(),
    }
}

fn concat_into<T: Clone>(
    sources: &[&PointBuffer],
    extract: impl Fn(&PointBuffer) -> Option<&[T]>,
) -> Option<Vec<T>> {
    let mut out = Vec::new();
    for s in sources {
        out.extend_from_slice(extract(s)?);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};

    fn cloud(points: &[[f32; 3]]) -> PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for p in points {
            builder.push_point(*p).unwrap();
        }
        builder.build().unwrap()
    }

    #[test]
    fn apply_transform_translates() {
        let c = cloud(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let out = apply_transform(&c, translation(Vec3::new(2.0, 3.0, 4.0))).unwrap();
        let (x, y, z) = out.positions3().unwrap();
        assert!(
            (x[0] - 2.0).abs() < 1e-6 && (y[0] - 3.0).abs() < 1e-6 && (z[0] - 4.0).abs() < 1e-6
        );
    }

    #[test]
    fn recenter_moves_centroid_to_origin() {
        let c = cloud(&[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]);
        let out = recenter(&c).unwrap();
        let centroid = centroid(&out).unwrap();
        assert!(centroid.x.abs() < 1e-6 && centroid.y.abs() < 1e-6 && centroid.z.abs() < 1e-6);
    }

    #[test]
    fn normalize_unit_sphere_bounds_radius() {
        let c = cloud(&[[10.0, 0.0, 0.0], [12.0, 0.0, 0.0], [11.0, 5.0, 0.0]]);
        let out = normalize_unit_sphere(&c).unwrap();
        let (x, y, z) = out.positions3().unwrap();
        let max_r = (0..out.len())
            .map(|i| (x[i] * x[i] + y[i] * y[i] + z[i] * z[i]).sqrt())
            .fold(0.0_f32, f32::max);
        assert!((max_r - 1.0).abs() < 1e-5, "max radius {max_r}");
    }

    #[test]
    fn bounding_box_spans_extent() {
        let c = cloud(&[[0.0, -1.0, 2.0], [4.0, 3.0, 2.0]]);
        let aabb = bounding_box(&c).unwrap();
        assert_eq!(aabb.min, Vec3::new(0.0, -1.0, 2.0));
        assert_eq!(aabb.max, Vec3::new(4.0, 3.0, 2.0));
        assert_eq!(aabb.extent(), Vec3::new(4.0, 4.0, 0.0));
    }

    #[test]
    fn merge_concatenates() {
        let a = cloud(&[[0.0, 0.0, 0.0]]);
        let b = cloud(&[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]);
        let merged = merge_clouds(&[&a, &b]).unwrap();
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn obb_aligns_with_elongated_axis() {
        // Points spread mainly along x; the first OBB axis should be ~x and the
        // first half-extent the largest.
        let mut pts = Vec::new();
        for i in 0..20 {
            pts.push([i as f32, 0.05 * (i % 2) as f32, 0.0]);
        }
        let c = cloud(&pts);
        let obb = oriented_bounding_box(&c).unwrap();
        let axis0 = Vec3::new(obb.axes.m[0][0], obb.axes.m[1][0], obb.axes.m[2][0]);
        assert!(axis0.x.abs() > 0.99, "principal axis not along x: {axis0:?}");
        assert!(obb.half_extents.x > obb.half_extents.y);
        assert!(obb.half_extents.x > obb.half_extents.z);
    }
}
