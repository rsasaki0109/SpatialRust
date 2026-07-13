use super::*;

/// Packs and uploads every chunk in a [`SpatialTensor`].
pub fn upload_spatial_tensor_xyz_chunks(
    runtime: &WgpuRuntime,
    tensor: &SpatialTensor<'_>,
) -> SpatialResult<Vec<GpuAoSoXyzChunk>> {
    let cloud = tensor.cloud();
    tensor
        .chunks()
        .map(|chunk| GpuAoSoXyzChunk::pack_and_upload(runtime, "aoso-xyz-chunk", &chunk, cloud))
        .collect()
}

/// Packs and uploads every tensor chunk with an explicit attribute layout.
pub fn upload_spatial_tensor_attribute_chunks(
    runtime: &WgpuRuntime,
    tensor: &SpatialTensor<'_>,
    layout: AoSoAAttributeLayout,
) -> SpatialResult<Vec<GpuAoSoAttributeChunk>> {
    let cloud = tensor.cloud();
    tensor
        .chunks()
        .map(|chunk| {
            let packed = if layout == AoSoAAttributeLayout::XYZ_INTENSITY {
                chunk.pack_xyz_intensity(cloud)?
            } else if layout == AoSoAAttributeLayout::XYZ_NORMALS {
                chunk.pack_xyz_normals(cloud)?
            } else if layout == AoSoAAttributeLayout::XYZ_INTENSITY_NORMALS {
                chunk.pack_xyz_intensity_normals(cloud)?
            } else {
                return Err(SpatialError::InvalidArgument(
                    "unsupported AoSoA attribute layout".to_owned(),
                ));
            };
            GpuAoSoAttributeChunk::upload(runtime, "aoso-attribute-chunk", &packed)
        })
        .collect()
}
