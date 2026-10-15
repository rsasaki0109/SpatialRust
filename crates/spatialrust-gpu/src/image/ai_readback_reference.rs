// Complete parent tensor readback with its original owned-byte mapping behavior.
use super::*;

impl GpuAiTensor {
    pub(super) fn parent_readback_f32(&mut self, runtime: &WgpuRuntime) -> SpatialResult<Vec<f32>> {
        if !std::sync::Arc::ptr_eq(&self.device_key, &runtime_device_key(runtime)) {
            return Err(SpatialError::InvalidArgument(
                "GpuAiTensor belongs to a different runtime device".to_owned(),
            ));
        }
        let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-ai-tensor-readback"),
            size: self.byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu-ai-tensor-readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.byte_len);
        runtime.queue().submit(Some(encoder.finish()));
        let bytes = parent_read_staging_bytes(runtime.device(), &staging, self.byte_len as usize)?;
        self.receipt.record_device_to_host(self.byte_len, "readback_ai_chw_f32");
        Ok(bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte f32 chunk")))
            .collect())
    }
}

fn parent_read_staging_bytes(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    len: usize,
) -> SpatialResult<Vec<u8>> {
    parent_with_staging_bytes(device, staging_buffer, len, <[u8]>::to_vec)
}

// Declared before the mapped view, so unwinding drops the view before unmapping.
struct ParentStagingUnmap<'a>(&'a wgpu::Buffer);

impl Drop for ParentStagingUnmap<'_> {
    fn drop(&mut self) {
        self.0.unmap();
    }
}

fn parent_with_staging_bytes<T>(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    len: usize,
    decode: impl FnOnce(&[u8]) -> T,
) -> SpatialResult<T> {
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
    let _unmap = ParentStagingUnmap(staging_buffer);
    let data = slice.get_mapped_range();
    Ok(decode(&data[..len]))
}
