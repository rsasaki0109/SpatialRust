//! Frozen explicit download from parent 331adc0f776aa3b4fbcbda72d42a428147a2f269.

use super::*;

pub(super) fn download(
    output: &GpuNeighborhoods,
    runtime: &WgpuRuntime,
) -> SpatialResult<GpuNeighborhoodReadback> {
    if !Arc::ptr_eq(&output.identity, &runtime.profiling_identity()) {
        return Err(invalid());
    }
    if output.queries == 0 {
        return Ok(GpuNeighborhoodReadback {
            rows: Vec::new(),
            totals: Vec::new(),
            transfers: TransferStats::default(),
        });
    }
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("neighborhood-readback"),
        size: output.bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&output.buffer, 0, &staging, 0, output.bytes);
    runtime.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    runtime.device().poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|e| SpatialError::InvalidArgument(e.to_string()))?
        .map_err(|e| SpatialError::InvalidArgument(e.to_string()))?;
    let mapped = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&mapped);
    let stride = 2 + 2 * output.capacity as usize;
    let decoded = parent_decode::decode(words, stride, output.capacity, output.kind);
    drop(mapped);
    staging.unmap();
    let (rows, totals) = decoded?;
    let mut transfers = TransferStats::default();
    transfers.record(TransferDirection::DeviceToHost, output.bytes);
    Ok(GpuNeighborhoodReadback { rows, totals, transfers })
}
