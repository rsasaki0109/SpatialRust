// Complete original histogram/CDF readbacks from merged PR 246.
use super::*;
use crate::image::gpu_image::read_staging_bytes;

pub(super) fn baseline_readback_bins_u32(
    runtime: &WgpuRuntime,
    buffer: &wgpu::Buffer,
) -> SpatialResult<[u32; 256]> {
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-image-histogram-readback"),
        size: HISTOGRAM_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-image-histogram-readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, HISTOGRAM_BYTES);
    runtime.queue().submit(Some(encoder.finish()));
    let bytes = read_staging_bytes(runtime.device(), &staging, HISTOGRAM_BYTES as usize)?;
    let mut result = [0; 256];
    for (count, bytes) in result.iter_mut().zip(bytes.chunks_exact(4)) {
        *count = u32::from_le_bytes(bytes.try_into().expect("four-byte histogram count"));
    }
    Ok(result)
}

impl GpuImageHistogram {
    pub(super) fn baseline_readback_u32(
        &mut self,
        runtime: &WgpuRuntime,
    ) -> SpatialResult<[u32; 256]> {
        self.validate_runtime(runtime)?;
        let result = baseline_readback_bins_u32(runtime, &self.buffer)?;
        self.receipt.record_device_to_host(HISTOGRAM_BYTES, "readback_histogram_u32");
        Ok(result)
    }
}

impl GpuImageCdf {
    pub(super) fn baseline_readback_u32(
        &mut self,
        runtime: &WgpuRuntime,
    ) -> SpatialResult<[u32; 256]> {
        self.validate_runtime(runtime)?;
        let counts = baseline_readback_bins_u32(runtime, &self.buffer)?;
        self.receipt.record_device_to_host(HISTOGRAM_BYTES, "readback_cdf_u32");
        Ok(counts)
    }
}
