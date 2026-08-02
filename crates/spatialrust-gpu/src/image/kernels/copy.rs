//! Device-resident image copy.

use spatialrust_core::SpatialResult;

use crate::image::gpu_image::{create_texture, GpuImage, GpuImageReceipt};
use crate::WgpuRuntime;

/// Copies `source` into a new GPU image without host transfers.
pub fn copy_gpu_image(runtime: &WgpuRuntime, source: &GpuImage) -> SpatialResult<GpuImage> {
    source.validate_runtime(runtime)?;
    let storage_bytes = source.storage_bytes();
    let texture = create_texture(runtime, source.width(), source.height(), "gpu-image-copy");
    let mut encoder = runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-image-copy-encoder"),
    });
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: source.texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d { width: source.width(), height: source.height(), depth_or_array_layers: 1 },
    );
    runtime.queue().submit(Some(encoder.finish()));
    let mut receipt = GpuImageReceipt::default();
    receipt.merge_from(source.receipt());
    receipt.record_gpu_to_gpu(storage_bytes, "copy_gpu_image");
    GpuImage::from_parts(
        runtime,
        source.width(),
        source.height(),
        source.channels(),
        texture,
        source.metadata(),
        receipt,
    )
}
