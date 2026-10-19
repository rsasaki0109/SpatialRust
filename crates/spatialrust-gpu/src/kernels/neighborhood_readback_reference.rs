//! Frozen explicit download from parent bf8e60831366a4e6f3ebb80f4b79734bdab46f8a.

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
    let mut rows = Vec::with_capacity(output.queries as usize);
    let mut totals = Vec::with_capacity(output.queries as usize);
    for row in words.chunks_exact(stride) {
        if row[0] > output.capacity
            || row[1] < row[0]
            || (output.kind == GpuNeighborhoodKind::NearestK && row[0] != row[1])
        {
            drop(mapped);
            staging.unmap();
            return Err(SpatialError::InvalidArgument(
                "GPU neighborhood metadata is invalid".to_owned(),
            ));
        }
        rows.push(
            (0..row[0] as usize)
                .map(|j| Neighbor {
                    index: row[2 + 2 * j] as usize,
                    distance_squared: f32::from_bits(row[3 + 2 * j]),
                })
                .collect(),
        );
        totals.push(row[1]);
    }
    drop(mapped);
    staging.unmap();
    let mut transfers = TransferStats::default();
    transfers.record(TransferDirection::DeviceToHost, output.bytes);
    Ok(GpuNeighborhoodReadback { rows, totals, transfers })
}
