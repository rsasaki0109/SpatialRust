use spatialrust_core::{SpatialError, SpatialResult, TransferDirection, TransferStats};

use crate::kernels::build_voxel_segments_from_positions_gpu;

pub use spatialrust_search::euclidean_cluster_roots as euclidean_cluster_roots_grid;

/// Runs the GPU sparse-grid stage and returns its transfer accounting together
/// with deterministic host-side component roots.
pub fn euclidean_cluster_roots_gpu_with_receipt(
    runtime: &crate::runtime::WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    cluster_tolerance: f32,
) -> SpatialResult<(Vec<u32>, TransferStats)> {
    if x.len() != y.len() || x.len() != z.len() {
        return Err(SpatialError::InvalidArgument("xyz arrays must have equal length".to_owned()));
    }
    if cluster_tolerance <= 0.0 || cluster_tolerance.is_nan() {
        return Err(SpatialError::InvalidArgument("cluster_tolerance must be positive".to_owned()));
    }
    if x.is_empty() {
        return Ok((Vec::new(), TransferStats::default()));
    }

    // Key generation, sorting, and sparse-cell compaction are GPU kernels. The
    // final connected-component union-find intentionally runs on the host so
    // the public GPU path has deterministic minimum-root semantics and does not
    // pretend that a CPU fallback is an accelerator kernel.
    let mut origin = [x[0], y[0], z[0]];
    for index in 1..x.len() {
        origin[0] = origin[0].min(x[index]);
        origin[1] = origin[1].min(y[index]);
        origin[2] = origin[2].min(z[index]);
    }
    let segments =
        build_voxel_segments_from_positions_gpu(runtime, x, y, z, origin, 1.0 / cluster_tolerance)?;
    let mut transfers = TransferStats::default();
    transfers
        .record(TransferDirection::HostToDevice, (x.len() * 3 * std::mem::size_of::<f32>()) as u64);
    let metadata_bytes = (segments.keys.len() * 4 * std::mem::size_of::<i32>()
        + segments.cell_starts.len() * std::mem::size_of::<u32>()
        + segments.point_indices.len() * std::mem::size_of::<u32>()
        + std::mem::size_of::<u32>()) as u64;
    transfers.record(TransferDirection::DeviceToHost, metadata_bytes);

    let roots = spatialrust_search::euclidean_cluster_roots_from_segments(
        x,
        y,
        z,
        cluster_tolerance,
        &segments.keys,
        &segments.point_indices,
        &segments.cell_starts,
        &segments.cell_counts,
    )?;
    Ok((roots, transfers))
}

/// Connected-component roots via uniform-grid union-find.
/// Computes Euclidean component roots using GPU sparse-grid construction.
pub fn euclidean_cluster_roots_gpu(
    runtime: &crate::runtime::WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    cluster_tolerance: f32,
) -> SpatialResult<Vec<u32>> {
    euclidean_cluster_roots_gpu_with_receipt(runtime, x, y, z, cluster_tolerance)
        .map(|(roots, _)| roots)
}
