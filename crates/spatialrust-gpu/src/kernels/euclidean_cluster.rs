use spatialrust_core::SpatialResult;
use spatialrust_search::euclidean_cluster_roots;

pub use spatialrust_search::euclidean_cluster_roots as euclidean_cluster_roots_grid;

/// Connected-component roots via uniform-grid union-find.
pub fn euclidean_cluster_roots_gpu(
    _runtime: &crate::runtime::WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    cluster_tolerance: f32,
) -> SpatialResult<Vec<u32>> {
    euclidean_cluster_roots(x, y, z, cluster_tolerance)
}
