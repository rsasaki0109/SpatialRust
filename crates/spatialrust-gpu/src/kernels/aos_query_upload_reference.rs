// Frozen query upload from merged PR 204; native tests only.
use super::*;

pub(crate) fn previous_prepare(points: &[[f32; 3]]) -> SpatialResult<Vec<[f32; 4]>> {
    if points.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid());
    }
    Ok(points.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect())
}

pub(crate) fn previous_upload(
    runtime: &WgpuRuntime,
    queries: &[[f32; 3]],
) -> SpatialResult<GpuQueryBatch> {
    let limits = runtime.device().limits();
    let bytes = buffer_bytes(queries.len() as u64, 16, &limits)?;
    check_compute(&limits, queries.len() as u64)?;
    if queries.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid());
    }
    let mut batch = GpuQueryBatch {
        buffer: None,
        count: queries.len() as u32,
        identity: runtime.profiling_identity(),
        transfers: TransferStats::default(),
    };
    if !queries.is_empty() {
        let packed: Vec<_> = queries.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        batch.buffer =
            Some(runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("neighborhood-queries"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE,
            }));
        batch.transfers.record(TransferDirection::HostToDevice, bytes);
    }
    Ok(batch)
}
