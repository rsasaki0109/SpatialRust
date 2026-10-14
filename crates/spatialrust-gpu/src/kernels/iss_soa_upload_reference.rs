//! Frozen upload from merged PR 195; constructor body retained.
use super::*;

pub(crate) fn previous_upload(runtime: &WgpuRuntime, tree: &KdTree) -> SpatialResult<GpuIssInput> {
    let layout = tree.layout();
    let (x, y, z) = layout.positions();
    let count = plan(x.len() as u64, &runtime.device().limits())?;
    if x.iter().chain(y).chain(z).any(|v| !v.is_finite()) {
        return Err(invalid());
    }
    if count > 0 {
        let extent: f64 = [x, y, z]
            .iter()
            .map(|axis| {
                let lo = axis.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = axis.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                (f64::from(hi) - f64::from(lo)).powi(2)
            })
            .sum();
        if !(extent as f32).is_finite() {
            return Err(invalid());
        }
    }
    let positions: Vec<_> = (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect();
    let tree = GpuKdTree::upload(runtime, tree)?;
    let queries = GpuQueryBatch::upload(runtime, &positions)?;
    let mut transfers = tree.upload_transfers();
    transfers
        .record(TransferDirection::HostToDevice, queries.upload_transfers().host_to_device_bytes());
    Ok(GpuIssInput {
        tree,
        queries,
        count,
        identity: runtime.profiling_identity(),
        input_identity: Arc::new(()),
        transfers,
    })
}

pub(crate) fn query_buffer(input: &GpuIssInput) -> Option<&wgpu::Buffer> {
    input.queries.buffer()
}
