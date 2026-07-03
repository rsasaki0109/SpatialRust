//! GPU compute kernels for SpatialRust algorithms.

#[cfg(feature = "gpu-wgpu")]
mod gpu_segments;

#[cfg(feature = "gpu-wgpu")]
mod normals;

#[cfg(feature = "gpu-wgpu")]
mod normals_grid;

#[cfg(feature = "gpu-wgpu")]
mod covariances_grid;

#[cfg(feature = "gpu-wgpu")]
mod voxel_keys;

#[cfg(feature = "gpu-wgpu")]
mod voxel_reduce;

#[cfg(feature = "gpu-wgpu")]
mod voxel_segments;

#[cfg(feature = "gpu-wgpu")]
mod voxel_sort;

#[cfg(feature = "gpu-wgpu")]
mod voxel_compact;

#[cfg(feature = "gpu-wgpu")]
mod voxel_segments_gpu;

#[cfg(feature = "gpu-wgpu")]
mod voxel_gather;

#[cfg(feature = "gpu-wgpu")]
mod euclidean_cluster;

#[cfg(feature = "gpu-wgpu")]
mod ransac_plane;

#[cfg(feature = "gpu-wgpu")]
mod voxel_pipeline;

#[cfg(feature = "gpu-wgpu")]
pub use gpu_segments::GpuVoxelSegments;

#[cfg(feature = "gpu-wgpu")]
pub use normals::{estimate_normals_gpu, GpuNormal};

#[cfg(feature = "gpu-wgpu")]
pub use normals_grid::estimate_normals_grid_gpu;

#[cfg(feature = "gpu-wgpu")]
pub use covariances_grid::{estimate_plane_covariances_grid_gpu, GpuCovariance};

#[cfg(feature = "gpu-wgpu")]
pub use voxel_keys::{compute_voxel_keys, compute_voxel_keys_gpu_buffers, GpuVoxelKeyBuffers};

#[cfg(feature = "gpu-wgpu")]
pub use voxel_reduce::{
    reduce_voxel_average_f32, reduce_voxel_average_f32_gpu, reduce_voxel_average_f32_gpu_buffers,
    reduce_voxel_average_f32_multi_gpu, reduce_voxel_centroids_xyz,
    reduce_voxel_centroids_xyz_and_average_multi_gpu,
    reduce_voxel_centroids_xyz_and_gather_first_multi_gpu, reduce_voxel_centroids_xyz_gpu_buffers,
};

#[cfg(feature = "gpu-wgpu")]
pub use voxel_gather::{
    gather_voxel_first_f32, gather_voxel_first_f32_gpu, gather_voxel_first_f32_gpu_buffers,
    gather_voxel_first_f32_multi_gpu, gather_voxel_first_xyz_and_average_multi_gpu,
    gather_voxel_first_xyz_and_multi_gpu, gather_voxel_first_xyz_gpu_buffers,
};

#[cfg(feature = "gpu-wgpu")]
pub use voxel_segments::{build_voxel_segments, compact_voxel_segments_from_sorted, VoxelSegments};

#[cfg(feature = "gpu-wgpu")]
pub use voxel_sort::{build_voxel_segments_gpu, build_voxel_segments_gpu_from_keys_buffer};

#[cfg(feature = "gpu-wgpu")]
pub use voxel_segments_gpu::{
    build_voxel_segments_from_positions_gpu, build_voxel_segments_from_positions_gpu_buffers,
};

#[cfg(feature = "gpu-wgpu")]
pub use euclidean_cluster::euclidean_cluster_roots_gpu;

#[cfg(feature = "gpu-wgpu")]
pub use ransac_plane::{score_ransac_plane_hypotheses_gpu, GpuPlaneScore};

#[cfg(feature = "gpu-wgpu")]
pub use voxel_pipeline::{
    downsample_voxel_approximate_first_gpu, downsample_voxel_centroid_gpu,
    VoxelApproximateFirstGpuResult, VoxelCentroidGpuResult,
};
