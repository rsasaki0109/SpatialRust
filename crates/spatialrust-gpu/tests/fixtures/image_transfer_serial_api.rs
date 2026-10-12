impl GpuImage {
pub(super) fn previous_upload_u8<const CHANNELS: usize>(
        runtime: &WgpuRuntime,
        view: ImageView<'_, u8, CHANNELS>,
    ) -> SpatialResult<Self> {
        let plan = upload_plan(view.width(), view.height(), CHANNELS, &runtime.device().limits())?;
        let (width, height) = (plan.width, plan.height);
        let texture = create_texture(runtime, width, height, "gpu-image-upload");
        let texels = previous::pack_rgba8(view, plan.packed_len);
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

pub(super) fn previous_readback_u8<const CHANNELS: usize>(
        &mut self,
        runtime: &WgpuRuntime,
    ) -> SpatialResult<Image<u8, CHANNELS>> {
        self.validate_runtime(runtime)?;
        if self.channels as usize != CHANNELS {
            return Err(SpatialError::InvalidArgument(format!(
                "GpuImage has {} channels but readback requested {CHANNELS}",
                self.channels
            )));
        }
        let plan = readback_plan(self.width, self.height, CHANNELS, &runtime.device().limits())?;
        let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-image-texture-readback"),
            size: plan.staging_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu-image-texture-readback-encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(plan.padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            extent(self.width, self.height),
        );
        runtime.queue().submit(Some(encoder.finish()));
        let rgba = read_staging_bytes(runtime.device(), &staging, plan.staging_len)?;
        let mut data = Vec::with_capacity(plan.output_len);
        for row in rgba.chunks_exact(plan.padded_row as usize).take(self.height as usize) {
            for texel in row[..plan.row_bytes as usize].chunks_exact(BYTES_PER_TEXEL as usize) {
                data.extend_from_slice(&texel[..CHANNELS]);
            }
        }
        self.receipt.record_device_to_host(self.storage_bytes, "readback_u8_texture");
        Image::try_new_with_metadata(self.width as usize, self.height as usize, data, self.metadata)
            .map_err(|error| SpatialError::InvalidArgument(error.to_string()))
    }
}
