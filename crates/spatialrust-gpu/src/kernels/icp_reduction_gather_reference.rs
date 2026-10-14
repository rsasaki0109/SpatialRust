// Frozen constructor from merged PR 202; native tests only.
use super::*;

pub(crate) fn metadata(input: &GpuIcpInput) -> ([u32; 4], [u32; 4], u32) {
    (
        input.source_anchor.map(f32::to_bits),
        input.target_anchor.map(f32::to_bits),
        input.scale.to_bits(),
    )
}

pub(crate) fn previous_upload(
    runtime: &WgpuRuntime,
    source: &[[f32; 3]],
    target: &KdTree,
) -> SpatialResult<GpuIcpInput> {
    plan(source.len() as u64, &runtime.device().limits())?;
    let layout = target.layout();
    let (x, y, z) = layout.positions();
    plan(x.len() as u64, &runtime.device().limits())?;
    if x.is_empty() {
        return Err(invalid());
    }
    let (source_anchor, source_extent) = geometry(source)?;
    let (target_anchor, target_extent) = geometry_preparation::prepare_soa([x, y, z])?;
    let extent = source_extent.max(target_extent);
    if extent > 0.0 && (extent as f32) < f32::MIN_POSITIVE {
        return Err(invalid());
    }
    let scale = if extent == 0.0 { 1.0 } else { extent.sqrt() as f32 };
    let tree = GpuKdTree::upload(runtime, target)?;
    let source = GpuQueryBatch::upload(runtime, source)?;
    let target = GpuQueryBatch::upload_soa(runtime, [x, y, z])?;
    let mut transfers = tree.upload_transfers();
    transfers
        .record(TransferDirection::HostToDevice, source.upload_transfers().host_to_device_bytes());
    transfers
        .record(TransferDirection::HostToDevice, target.upload_transfers().host_to_device_bytes());
    Ok(GpuIcpInput {
        source,
        target,
        tree,
        source_anchor,
        target_anchor,
        scale,
        identity: runtime.profiling_identity(),
        transfers,
    })
}
