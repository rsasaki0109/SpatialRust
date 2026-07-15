//! Device-resident image copy.

use spatialrust_core::SpatialResult;

use crate::image::gpu_image::{GpuImage, GpuImageReceipt};
use crate::WgpuRuntime;

/// Copies `source` into a new GPU image without host transfers.
pub fn copy_gpu_image(runtime: &WgpuRuntime, source: &GpuImage) -> SpatialResult<GpuImage> {
    source.validate_runtime(runtime)?;
    let storage_bytes = source.storage_bytes();
    let buffer = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-image-copy"),
        size: storage_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-image-copy-encoder"),
    });
    encoder.copy_buffer_to_buffer(source.buffer(), 0, &buffer, 0, storage_bytes);
    runtime.queue().submit(Some(encoder.finish()));
    let mut receipt = GpuImageReceipt::default();
    receipt.merge_from(source.receipt());
    receipt.record_gpu_to_gpu(storage_bytes, "copy_gpu_image");
    GpuImage::from_parts(
        runtime,
        source.width(),
        source.height(),
        source.channels(),
        buffer,
        storage_bytes,
        source.metadata(),
        receipt,
    )
}
