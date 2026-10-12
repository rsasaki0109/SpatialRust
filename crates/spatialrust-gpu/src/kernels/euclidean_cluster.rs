use spatialrust_core::{SpatialError, SpatialResult, TransferStats};

pub use spatialrust_search::euclidean_cluster_roots as euclidean_cluster_roots_grid;

/// Builds sparse cells and labels minimum-index components on the GPU.
/// Returns roots on the host with explicit upload/readback accounting.
/// Nonempty inputs require finite XYZ and a positive tolerance with a finite
/// reciprocal. Positive infinity initializes one component on the device.
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

    super::cluster_union::run(runtime, x, y, z, cluster_tolerance)
}

/// Connected-component roots via uniform-grid union-find.
/// Sparse-grid construction and component union/find run on the GPU.
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
