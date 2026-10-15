// Complete parent readback and mapping helper; test-only names change.
use super::*;

impl GpuImage {
    pub(super) fn parent_readback_u8<const CHANNELS: usize>(
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
        let rgba = parent_read_staging_bytes(runtime.device(), &staging, plan.staging_len)?;
        let data = unpack_rgba8::<CHANNELS>(
            &rgba,
            plan.row_bytes as usize / BYTES_PER_TEXEL as usize,
            self.height as usize,
            plan.padded_row as usize,
            plan.output_len,
        );
        self.receipt.record_device_to_host(self.storage_bytes, "readback_u8_texture");
        Image::try_new_with_metadata(self.width as usize, self.height as usize, data, self.metadata)
            .map_err(|error| SpatialError::InvalidArgument(error.to_string()))
    }
}

fn parent_read_staging_bytes(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    len: usize,
) -> SpatialResult<Vec<u8>> {
    let slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| SpatialError::InvalidArgument("failed to receive wgpu map result".to_owned()))?
        .map_err(|error| {
            SpatialError::InvalidArgument(format!("failed to map wgpu texture: {error}"))
        })?;
    let data = slice.get_mapped_range();
    let values = data[..len].to_vec();
    drop(data);
    staging_buffer.unmap();
    Ok(values)
}
