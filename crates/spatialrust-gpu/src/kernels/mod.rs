//! GPU compute kernels for SpatialRust algorithms.

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_mixed_readback_tests.rs"]
mod voxel_mixed_readback_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_channel_readback_tests.rs"]
mod voxel_channel_readback_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_xyz_readback_tests.rs"]
mod voxel_xyz_readback_tests;

#[cfg(feature = "gpu-wgpu")]
mod gpu_segments;

#[cfg(feature = "gpu-wgpu")]
mod normals;

#[cfg(feature = "gpu-wgpu")]
mod normals_grid;

#[cfg(feature = "gpu-wgpu")]
mod covariances_grid;

pub(crate) mod point_estimators;

#[cfg(feature = "gpu-wgpu")]
mod voxel_keys;

#[cfg(feature = "gpu-wgpu")]
mod voxel_reduce;

#[cfg(feature = "gpu-wgpu")]
mod voxel_segments;

#[cfg(feature = "gpu-wgpu")]
mod voxel_sort;

#[cfg(feature = "gpu-aoso-staging")]
pub(crate) use voxel_sort::build_voxel_segments_gpu_from_keys_buffer_with_profiler as build_voxel_segments_with_profiler;

#[cfg(feature = "gpu-wgpu")]
mod voxel_radix;

#[cfg(feature = "gpu-wgpu")]
mod voxel_compact;

#[cfg(feature = "gpu-wgpu")]
mod voxel_scan;

#[cfg(feature = "gpu-aoso-staging")]
pub(crate) use voxel_scan::scan_pass_count as voxel_scan_pass_count;

pub(crate) use voxel_scan::PrefixScanPipelines;

#[cfg(feature = "gpu-wgpu")]
mod voxel_segments_gpu;

#[cfg(feature = "gpu-wgpu")]
mod voxel_gather;

pub(crate) mod cluster_union;
mod feature_validation;
pub(crate) mod fpfh;
pub(crate) mod gicp_reduction;
pub(crate) mod point_to_plane_reduction;
pub use point_to_plane_reduction::{
    GpuPointToPlaneEquations, GpuPointToPlaneInput, GpuPointToPlaneNormalEquations,
};
pub(crate) mod icp_reduction;
pub use gicp_reduction::{GpuGicpEquations, GpuGicpInput, GpuGicpNormalEquations};
#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "feature_soa_upload_tests.rs"]
mod feature_soa_upload_tests;
pub(crate) mod iss;
pub(crate) mod mls;
pub use icp_reduction::trim::GpuIcpTrimPrecision;
pub use icp_reduction::{
    GpuIcpInput, GpuIcpMoments, GpuIcpStatistics, GpuIcpTrimmedMoments, GpuIcpTrimmedStatistics,
};
pub use iss::{GpuIssInput, GpuIssKeypoints, GpuIssNeighborhoods};
pub use mls::{
    GpuMlsDiagnostics, GpuMlsInput, GpuMlsMode, GpuMlsNeighborhoods, GpuMlsPositions,
    GpuMlsReadback,
};
pub(crate) mod nearest;
pub use fpfh::{GpuFpfhDescriptors, GpuFpfhInput, GpuFpfhNeighborhoods, GPU_FPFH_DESCRIPTOR_LEN};
pub use nearest::{
    GpuKdTree, GpuNeighborhoodKind, GpuNeighborhoodReadback, GpuNeighborhoods, GpuQueryBatch,
    MAX_GPU_K_NEIGHBORS,
};
#[cfg(feature = "gpu-wgpu")]
mod euclidean_cluster;

#[cfg(feature = "gpu-wgpu")]
mod ransac_plane;

#[cfg(feature = "gpu-wgpu")]
mod voxel_pipeline;

#[cfg(feature = "gpu-wgpu")]
pub use gpu_segments::GpuVoxelSegments;

#[cfg(feature = "gpu-aoso-staging")]
pub(crate) use normals::NORMALS_WGSL;
#[cfg(feature = "gpu-wgpu")]
pub use normals::{estimate_normals_gpu, GpuNormal};

#[cfg(feature = "gpu-wgpu")]
pub use normals_grid::{estimate_normals_grid_gpu, uniform_grid_fits};

#[cfg(feature = "gpu-wgpu")]
pub use covariances_grid::{
    estimate_plane_covariances_grid_gpu, estimate_plane_covariances_grid_gpu_with_receipt,
    GpuCovariance,
};

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
pub use euclidean_cluster::{
    euclidean_cluster_roots_gpu, euclidean_cluster_roots_gpu_with_receipt,
    euclidean_cluster_roots_grid,
};

#[cfg(feature = "gpu-wgpu")]
pub use ransac_plane::{score_ransac_plane_hypotheses_gpu, GpuPlaneScore};

#[cfg(feature = "gpu-wgpu")]
pub(crate) use ransac_plane::RansacPlanePipelines;

#[cfg(feature = "gpu-wgpu")]
pub use voxel_pipeline::{
    downsample_voxel_approximate_first_gpu, downsample_voxel_centroid_gpu,
    VoxelApproximateFirstGpuResult, VoxelCentroidGpuResult,
};

#[cfg(feature = "gpu-wgpu")]
pub(crate) mod voxel_reduce_fused;

#[cfg(feature = "gpu-wgpu")]
pub(crate) mod voxel_first_average_fused;

#[cfg(feature = "gpu-wgpu")]
pub(crate) mod voxel_centroid_first_fused;

#[cfg(feature = "gpu-wgpu")]
pub(crate) mod voxel_first_gather_fused;

#[cfg(feature = "gpu-wgpu")]
mod voxel_u8_batch;

#[cfg(feature = "gpu-wgpu")]
mod voxel_gather_submit;

#[cfg(feature = "gpu-wgpu")]
mod voxel_average_batch;

#[cfg(feature = "gpu-wgpu")]
mod voxel_gather_packed;

#[cfg(feature = "gpu-wgpu")]
mod voxel_standalone_packed;

#[cfg(feature = "gpu-wgpu")]
mod voxel_xyz_packed;

#[cfg(feature = "gpu-wgpu")]
mod voxel_wide_average;
