// Frozen constructor from merged PR 202; native tests only.
use super::*;

pub(crate) fn parts(input: &GpuGicpInput) -> (&GpuIcpInput, &wgpu::Buffer, u32, u32) {
    (&input.positions, &input.covariances, input.weight_scale.to_bits(), input.covariance_offset)
}

pub(crate) fn previous_upload(
    runtime: &WgpuRuntime,
    source: &[[f32; 3]],
    source_covariances: &[[f32; 6]],
    target: &KdTree,
    target_covariances: &[[f32; 6]],
) -> SpatialResult<GpuGicpInput> {
    if source.len() != source_covariances.len()
        || target.layout().positions().0.len() != target_covariances.len()
    {
        return Err(invalid());
    }
    let total =
        (source.len() as u64).checked_add(target_covariances.len() as u64).ok_or_else(invalid)?;
    plan(source.len() as u64, total, &runtime.device().limits())?;
    let (packed, weight_scale) =
        upload_preparation::prepare(source_covariances, target_covariances)?;
    let positions = GpuIcpInput::upload(runtime, source, target)?;
    let covariances = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gicp-covariances"),
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut transfers = positions.upload_transfers();
    transfers.record(TransferDirection::HostToDevice, total * 32);
    Ok(GpuGicpInput { positions, covariances, weight_scale, covariance_offset: 0, transfers })
}
