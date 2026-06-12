//! GPU compute kernels for SpatialRust algorithms.

#[cfg(feature = "gpu-wgpu")]
mod gpu_segments;

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
mod voxel_pipeline;

#[cfg(feature = "gpu-wgpu")]
pub use gpu_segments::GpuVoxelSegments;

#[cfg(feature = "gpu-wgpu")]
pub use voxel_keys::{compute_voxel_keys, compute_voxel_keys_gpu_buffers, GpuVoxelKeyBuffers};

#[cfg(feature = "gpu-wgpu")]
pub use voxel_reduce::{
    reduce_voxel_average_f32, reduce_voxel_average_f32_gpu, reduce_voxel_average_f32_gpu_buffers,
    reduce_voxel_centroids_xyz, reduce_voxel_centroids_xyz_gpu_buffers,
};

#[cfg(feature = "gpu-wgpu")]
pub use voxel_gather::{
    gather_voxel_first_f32, gather_voxel_first_f32_gpu, gather_voxel_first_f32_gpu_buffers,
    gather_voxel_first_f32_multi_gpu, gather_voxel_first_xyz_gpu_buffers,
};

#[cfg(feature = "gpu-wgpu")]
pub use voxel_segments::{build_voxel_segments, compact_voxel_segments_from_sorted, VoxelSegments};

#[cfg(feature = "gpu-wgpu")]
pub use voxel_sort::build_voxel_segments_gpu;

#[cfg(feature = "gpu-wgpu")]
pub use voxel_segments_gpu::{
    build_voxel_segments_from_positions_gpu, build_voxel_segments_from_positions_gpu_buffers,
};

#[cfg(feature = "gpu-wgpu")]
pub use voxel_pipeline::{
    downsample_voxel_approximate_first_gpu, downsample_voxel_centroid_gpu, VoxelApproximateFirstGpuResult,
    VoxelCentroidGpuResult,
};
