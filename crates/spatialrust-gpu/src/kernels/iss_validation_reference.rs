// Frozen constructor from merged PR 201; test-only serial validation.
use super::*;

pub(crate) fn previous_upload(
    runtime: &WgpuRuntime,
    tree: &KdTree,
    gpu_gather: bool,
) -> SpatialResult<GpuIssInput> {
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
    let (tree, queries) = if gpu_gather {
        GpuKdTree::upload_and_queries_with_gpu_gather(runtime, tree)?
    } else {
        (GpuKdTree::upload(runtime, tree)?, GpuQueryBatch::upload_soa(runtime, [x, y, z])?)
    };
    let mut transfers = tree.upload_transfers();
    transfers
        .record(TransferDirection::HostToDevice, queries.upload_transfers().host_to_device_bytes());
    transfers.record(
        TransferDirection::DeviceToDevice,
        queries.upload_transfers().device_to_device_bytes(),
    );
    Ok(GpuIssInput {
        tree,
        queries,
        count,
        identity: runtime.profiling_identity(),
        input_identity: Arc::new(()),
        transfers,
    })
}
