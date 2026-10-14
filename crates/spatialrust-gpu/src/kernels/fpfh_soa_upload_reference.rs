//! Frozen upload from merged PR 195; constructor body retained.
use super::*;

pub(crate) fn previous_upload(
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
    let positions: Vec<_> = (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect();
    let tree = GpuKdTree::upload(runtime, tree)?;
    let queries = GpuQueryBatch::upload(runtime, &positions)?;
    let mut transfers = tree.upload_transfers();
    transfers
        .record(TransferDirection::HostToDevice, queries.upload_transfers().host_to_device_bytes());
    let samples = if points == 0 {
        None
    } else {
        let samples: Vec<_> = positions
            .iter()
            .zip(normals)
            .map(|(p, n)| Sample {
                position: [p[0], p[1], p[2], 0.0],
                normal: [n[0], n[1], n[2], 0.0],
            })
            .collect();
        transfers.record(TransferDirection::HostToDevice, u64::from(points) * 32);
        Some(runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fpfh-samples"),
            contents: bytemuck::cast_slice(&samples),
            usage: wgpu::BufferUsages::STORAGE,
        }))
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

pub(crate) fn query_buffer(input: &GpuFpfhInput) -> Option<&wgpu::Buffer> {
    input.queries.buffer()
}
pub(crate) fn sample_buffer(input: &GpuFpfhInput) -> Option<&wgpu::Buffer> {
    input.samples.as_ref()
}
