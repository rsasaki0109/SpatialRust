// Frozen complete uploader from the parent merge; only the test method name changes.
use super::*;

impl GpuImage {
    pub(super) fn parent_upload_u8<const CHANNELS: usize>(
        runtime: &WgpuRuntime,
        view: ImageView<'_, u8, CHANNELS>,
    ) -> SpatialResult<Self> {
        let plan = upload_plan(view.width(), view.height(), CHANNELS, &runtime.device().limits())?;
        let (width, height) = (plan.width, plan.height);
        let texture = create_texture(runtime, width, height, "gpu-image-upload");
        let texels = pack_rgba8(view, plan.packed_len);
        runtime.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(plan.row_bytes),
                rows_per_image: Some(height),
            },
            extent(width, height),
        );
        let storage_bytes = plan.storage_bytes;
        let mut receipt = GpuImageReceipt::default();
        receipt.record_host_to_device(storage_bytes, "upload_u8_texture");
        Ok(Self {
            width,
            height,
            channels: CHANNELS as u32,
            device_key: runtime_device_key(runtime),
            texture,
            storage_bytes,
            metadata: view.metadata(),
            receipt,
        })
    }
}
