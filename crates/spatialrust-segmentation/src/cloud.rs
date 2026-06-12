use spatialrust_core::{
    PointBuffer, PointBufferSet, PointCloud, PointField, SpatialError, SpatialResult,
};

/// Extracts a sub-cloud containing only the selected point indices.
pub fn extract_indices(input: &PointCloud, indices: &[usize]) -> SpatialResult<PointCloud> {
    if indices.is_empty() {
        let mut buffers = PointBufferSet::new();
        for field in input.schema().fields() {
            buffers.insert(field.name.clone(), PointBuffer::with_capacity(field.dtype, 0));
        }
        return PointCloud::try_from_parts(input.schema().clone(), buffers, input.metadata().clone());
    }

    let mut buffers = PointBufferSet::new();
    for field in input.schema().fields() {
        let source = input.field(&field.name)?;
        buffers.insert(field.name.clone(), gather_buffer(source, indices)?);
    }

    PointCloud::try_from_parts(input.schema().clone(), buffers, input.metadata().clone())
}

/// Extracts points where `mask[index]` is true.
pub fn extract_mask(input: &PointCloud, mask: &[bool]) -> SpatialResult<PointCloud> {
    if mask.len() != input.len() {
        return Err(SpatialError::InvalidArgument(format!(
            "mask length {} does not match point count {}",
            mask.len(),
            input.len()
        )));
    }

    let indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(index, selected)| selected.then_some(index))
        .collect();
    extract_indices(input, &indices)
}

/// Adds or replaces a per-point label field on a point cloud copy.
pub fn with_labels(
    input: &PointCloud,
    field_name: &str,
    labels: Vec<i32>,
) -> SpatialResult<PointCloud> {
    if labels.len() != input.len() {
        return Err(SpatialError::InvalidArgument(format!(
            "label length {} does not match point count {}",
            labels.len(),
            input.len()
        )));
    }

    let mut schema = input.schema().clone();
    if schema
        .find_semantic(spatialrust_core::FieldSemantic::Label)
        .is_none()
    {
        schema = schema.with_field(PointField::scalar(
            field_name,
            spatialrust_core::FieldSemantic::Label,
            spatialrust_core::DType::I32,
        ));
    }

    let mut buffers = PointBufferSet::new();
    for field in input.schema().fields() {
        let source = input.field(&field.name)?;
        buffers.insert(field.name.clone(), clone_buffer(source)?);
    }
    buffers.insert(field_name.to_owned(), PointBuffer::I32(labels));

    PointCloud::try_from_parts(schema, buffers, input.metadata().clone())
}

fn gather_buffer(source: &PointBuffer, indices: &[usize]) -> SpatialResult<PointBuffer> {
    Ok(match source {
        PointBuffer::F32(values) => {
            PointBuffer::from_f32(indices.iter().map(|&index| values[index]).collect())
        }
        PointBuffer::F64(values) => {
            PointBuffer::F64(indices.iter().map(|&index| values[index]).collect())
        }
        PointBuffer::U8(values) => {
            PointBuffer::U8(indices.iter().map(|&index| values[index]).collect())
        }
        PointBuffer::U16(values) => {
            PointBuffer::U16(indices.iter().map(|&index| values[index]).collect())
        }
        PointBuffer::U32(values) => {
            PointBuffer::U32(indices.iter().map(|&index| values[index]).collect())
        }
        PointBuffer::I32(values) => {
            PointBuffer::I32(indices.iter().map(|&index| values[index]).collect())
        }
    })
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
