//! GPU runtime, device buffers, and kernel dispatch for SpatialRust.
//!
//! Unsafe code is restricted to FFI and GPU interop boundaries in this crate.

#![warn(missing_docs)]
#![deny(unsafe_code)]

mod buffer;
mod device;

#[cfg(feature = "gpu-wgpu")]
mod kernels;

#[cfg(feature = "gpu-wgpu")]
mod pipeline_cache;

#[cfg(feature = "gpu-wgpu")]
mod readback;

#[cfg(feature = "gpu-wgpu")]
mod upload_cache;

#[cfg(feature = "gpu-wgpu")]
mod runtime;

#[cfg(feature = "gpu-aoso-staging")]
mod aoso_staging;

#[cfg(feature = "gpu-aoso-staging")]
mod gpu_frame;

#[cfg(feature = "gpu-image")]
mod image;

pub use buffer::DeviceBuffer;
pub use device::{GpuDevice, WgpuDevice};

#[cfg(feature = "gpu-wgpu")]
pub use kernels::{
    build_voxel_segments, build_voxel_segments_from_positions_gpu,
    build_voxel_segments_from_positions_gpu_buffers, build_voxel_segments_gpu,
    build_voxel_segments_gpu_from_keys_buffer, compact_voxel_segments_from_sorted,
    compute_voxel_keys, compute_voxel_keys_gpu_buffers, downsample_voxel_approximate_first_gpu,
    downsample_voxel_centroid_gpu, estimate_normals_gpu, estimate_normals_grid_gpu,
    estimate_plane_covariances_grid_gpu, euclidean_cluster_roots_gpu, euclidean_cluster_roots_grid,
    gather_voxel_first_f32, gather_voxel_first_f32_gpu, gather_voxel_first_f32_gpu_buffers,
    gather_voxel_first_f32_multi_gpu, gather_voxel_first_xyz_and_average_multi_gpu,
    gather_voxel_first_xyz_and_multi_gpu, gather_voxel_first_xyz_gpu_buffers,
    reduce_voxel_average_f32, reduce_voxel_average_f32_gpu, reduce_voxel_average_f32_gpu_buffers,
    reduce_voxel_average_f32_multi_gpu, reduce_voxel_centroids_xyz,
    reduce_voxel_centroids_xyz_and_average_multi_gpu,
    reduce_voxel_centroids_xyz_and_gather_first_multi_gpu, reduce_voxel_centroids_xyz_gpu_buffers,
    score_ransac_plane_hypotheses_gpu, uniform_grid_fits, GpuCovariance, GpuNormal, GpuPlaneScore,
    GpuVoxelKeyBuffers, GpuVoxelSegments, VoxelApproximateFirstGpuResult, VoxelCentroidGpuResult,
    VoxelSegments,
};

#[cfg(feature = "gpu-wgpu")]
pub use runtime::{
    WgpuAdapterInfo, WgpuRuntime, MULTI_GATHER2_STORAGE_BUFFERS, MULTI_GATHER4_STORAGE_BUFFERS,
};

#[cfg(feature = "gpu-wgpu")]
pub use upload_cache::GpuBufferPool;

#[cfg(feature = "gpu-aoso-staging")]
pub use aoso_staging::{
    build_radius_grid_aoso_gpu, compute_voxel_keys_aoso_chunks,
    downsample_voxel_centroid_aoso_chunks, estimate_normals_aoso_gpu,
    estimate_normals_radius_grid_aoso_gpu, reduce_voxel_attributes_aoso_chunks,
    upload_spatial_tensor_attribute_chunks, upload_spatial_tensor_xyz_chunks,
    AoSoAAttributeAggregation, AoSoAAttributeReduction, AoSoAVoxelCentroidResult,
    GpuAoSoAttributeChunk, GpuAoSoNormals, GpuAoSoRadiusGrid, GpuAoSoXyzBuffer, GpuAoSoXyzChunk,
};

#[cfg(feature = "gpu-aoso-staging")]
pub use gpu_frame::{
    run_aoso_voxel_normal_frame, GpuExecutionReceipt, GpuFrameCapability, GpuSpatialFrame,
};

#[cfg(feature = "gpu-image")]
pub use image::{
    box_blur_gpu, copy_gpu_image, morphology_gpu, pack_ai_chw_gpu, resize_nearest_gpu,
    rgb_to_gray_gpu, run_gpu_vision_chain, sobel_gpu, GpuAiTensor, GpuImage, GpuImageBorder,
    GpuImageReceipt, GpuMorphology, GpuVisionChainOptions,
};
