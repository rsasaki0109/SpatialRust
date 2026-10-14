// Frozen two-pass GPU construction from merged PR 200, used only in tests.
use super::*;

pub(crate) fn previous_gpu_upload(
    runtime: &WgpuRuntime,
    tree: &KdTree,
    normals: &[[f32; 3]],
) -> SpatialResult<GpuFpfhInput> {
    let layout = tree.layout();
    let (x, y, z) = layout.positions();
    let (points, _) = plan(x.len() as u64, &runtime.device().limits())?;
    if normals.len() != x.len() {
        return Err(SpatialError::BufferLengthMismatch { expected: x.len(), found: normals.len() });
    }
    if x.iter().chain(y).chain(z).any(|v| !v.is_finite())
        || normals.iter().any(|n| {
            let norm = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
            n.iter().any(|v| !v.is_finite()) || !norm.is_finite() || (norm - 1.0).abs() > 1e-3
        })
    {
        return Err(invalid());
    }
    if points > 0 {
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
    let (tree, queries) = GpuKdTree::upload_and_queries_with_gpu_gather(runtime, tree)?;
    let mut transfers = tree.upload_transfers();
    transfers
        .record(TransferDirection::HostToDevice, queries.upload_transfers().host_to_device_bytes());
    transfers.record(
        TransferDirection::DeviceToDevice,
        queries.upload_transfers().device_to_device_bytes(),
    );
    let samples = if points == 0 {
        None
    } else {
        transfers.record(TransferDirection::HostToDevice, u64::from(points) * 12);
        transfers.record(TransferDirection::DeviceToDevice, u64::from(points) * 32);
        Some(sample_gather::upload(runtime, &queries, normals))
    };
    Ok(GpuFpfhInput {
        samples,
        tree,
        queries,
        points,
        identity: runtime.profiling_identity(),
        input_identity: Arc::new(()),
        transfers,
    })
}
