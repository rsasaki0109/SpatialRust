//! Zero-copy bridges from schema-aware point-cloud columns.

use spatialrust_core::{SpatialError, SpatialTensor};

use crate::{DataType, Device, TensorDescriptor, TensorError, TensorView};

/// Errors raised while exposing a point-cloud column as a generic tensor.
#[derive(Debug, thiserror::Error)]
pub enum SpatialTensorBridgeError {
    /// The requested field is absent or does not have the required dtype.
    #[error(transparent)]
    Spatial(#[from] SpatialError),
    /// The generic tensor descriptor or storage is invalid.
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

/// Borrows one `f32` point field as a zero-copy `[point_count]` CPU tensor.
///
/// Point fields remain separate Schema-SoA columns. Interleaving XYZ or other
/// fields requires a separately named packing operation and is never hidden by
/// this bridge.
pub fn spatial_f32_field_view<'a>(
    tensor: &SpatialTensor<'a>,
    field_name: &str,
) -> Result<TensorView<'a>, SpatialTensorBridgeError> {
    let values = tensor.cloud().field(field_name)?.as_f32()?;
    let descriptor = TensorDescriptor::contiguous(DataType::F32, vec![values.len()], Device::CPU);
    Ok(TensorView::try_new(bytemuck::cast_slice(values), descriptor)?)
}

#[cfg(test)]
mod tests {
    use super::spatial_f32_field_view;
    use spatialrust_core::{PointCloudBuilder, SpatialTensor};

    #[test]
    fn point_field_is_borrowed_without_interleaving() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([1.0, 2.0, 3.0]).unwrap();
        builder.push_point([4.0, 5.0, 6.0]).unwrap();
        let cloud = builder.build().unwrap();
        let spatial = SpatialTensor::new(&cloud, 1).unwrap();
        let tensor = spatial_f32_field_view(&spatial, "x").unwrap();
        assert_eq!(tensor.descriptor().shape(), &[2]);
        assert_eq!(bytemuck::cast_slice::<u8, f32>(tensor.allocation_bytes()), &[1.0, 4.0]);
        assert_eq!(
            tensor.allocation_bytes().as_ptr(),
            cloud.field("x").unwrap().as_f32().unwrap().as_ptr().cast()
        );
    }
}
