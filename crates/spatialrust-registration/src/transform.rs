use spatialrust_core::{
    FieldSemantic, HasNormals3, HasPositions3, PointBuffer, PointBufferSet, PointCloud,
    SpatialResult,
};
use spatialrust_math::{Isometry3, TransformPoint, Vec3};

/// Applies a rigid transform to point positions and normals in a point cloud copy.
pub fn transform_point_cloud(
    input: &PointCloud,
    transform: Isometry3<f32>,
) -> SpatialResult<PointCloud> {
    if input.is_empty() {
        return Ok(input.clone());
    }

    let (x, y, z) = input.positions3()?;
    let mut tx = Vec::with_capacity(input.len());
    let mut ty = Vec::with_capacity(input.len());
    let mut tz = Vec::with_capacity(input.len());
    for index in 0..input.len() {
        let transformed = transform.transform_point(Vec3::new(x[index], y[index], z[index]));
        tx.push(transformed.x);
        ty.push(transformed.y);
        tz.push(transformed.z);
    }

    let mut buffers = PointBufferSet::new();
    for field in input.schema().fields() {
        let source = input.field(&field.name)?;
        let buffer = match field.semantic {
            FieldSemantic::PositionX => PointBuffer::from_f32(tx.clone()),
            FieldSemantic::PositionY => PointBuffer::from_f32(ty.clone()),
            FieldSemantic::PositionZ => PointBuffer::from_f32(tz.clone()),
            FieldSemantic::NormalX | FieldSemantic::NormalY | FieldSemantic::NormalZ => {
                transform_normal_buffer(input, transform, field.semantic)?
            }
            _ => clone_buffer(source)?,
        };
        buffers.insert(field.name.clone(), buffer);
    }

    PointCloud::try_from_parts(input.schema().clone(), buffers, input.metadata().clone())
}

fn transform_normal_buffer(
    input: &PointCloud,
    transform: Isometry3<f32>,
    semantic: FieldSemantic,
) -> SpatialResult<PointBuffer> {
    let (nx, ny, nz) = input.normals3()?;
    let mut values = Vec::with_capacity(input.len());
    for index in 0..input.len() {
        let normal =
            transform.transform_vector(Vec3::new(nx[index], ny[index], nz[index])).normalize();
        values.push(match semantic {
            FieldSemantic::NormalX => normal.x,
            FieldSemantic::NormalY => normal.y,
            FieldSemantic::NormalZ => normal.z,
            _ => 0.0,
        });
    }
    Ok(PointBuffer::from_f32(values))
}

fn clone_buffer(buffer: &PointBuffer) -> SpatialResult<PointBuffer> {
    Ok(match buffer {
        PointBuffer::F32(values) => PointBuffer::from_f32(values.clone()),
        PointBuffer::F64(values) => PointBuffer::F64(values.clone()),
        PointBuffer::U8(values) => PointBuffer::U8(values.clone()),
        PointBuffer::U16(values) => PointBuffer::U16(values.clone()),
        PointBuffer::U32(values) => PointBuffer::U32(values.clone()),
        PointBuffer::I32(values) => PointBuffer::I32(values.clone()),
    })
}

#[cfg(test)]
mod tests {
    use super::transform_point_cloud;
    use spatialrust_core::{HasPositions3, PointCloudBuilder};
    use spatialrust_math::{Isometry3, Vec3};

    #[test]
    fn transforms_positions() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        builder.push_point([1.0, 0.0, 0.0]).unwrap();
        let cloud = builder.build().unwrap();

        let transform =
            Isometry3::new(spatialrust_math::Quat::<f32>::identity(), Vec3::new(1.0, 2.0, 3.0));
        let transformed = transform_point_cloud(&cloud, transform).unwrap();
        let (x, y, z) = transformed.positions3().unwrap();
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((y[1] - 2.0).abs() < 1e-6);
        assert!((z[1] - 3.0).abs() < 1e-6);
    }
}
