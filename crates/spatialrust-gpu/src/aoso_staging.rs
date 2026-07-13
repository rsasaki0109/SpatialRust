//! Upload helpers for interleaved AoSoA XYZ chunks (`tensor-aoso` + `gpu-aoso-staging`).

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{
    AoSoAAttributeChunk, AoSoAAttributeLayout, AoSoAXyzChunk, PointCloud, SpatialError,
    SpatialResult, SpatialTensor, SpatialTensorChunk,
};
use wgpu::util::DeviceExt;
use wgpu::Buffer;

use crate::runtime::WgpuRuntime;
use crate::{build_voxel_segments_gpu_from_keys_buffer, GpuNormal, GpuVoxelSegments};

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct VoxelKeyUniform {
    origin: [f32; 4],
    inv_leaf: f32,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct VoxelKeyOutput {
    ix: i32,
    iy: i32,
    iz: i32,
    _pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct VoxelReduceUniform {
    cell_count: u32,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct AttributeReduceUniform {
    cell_count: u32,
    point_count: u32,
    stride: u32,
    first_mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct AoSoANormalsUniform {
    point_count: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SparseGridNormalsUniform {
    origin: [f32; 4],
    dims: [u32; 4],
    inv_cell: f32,
    radius_sq: f32,
    _pad0: f32,
    _pad1: f32,
}

mod attributes;
mod normals;
mod types;
mod upload;
mod voxel;

pub use attributes::*;
pub use normals::*;
pub use types::*;
pub use upload::*;
pub use voxel::*;

pub(crate) fn runtime_device_key(runtime: &WgpuRuntime) -> usize {
    runtime.device() as *const wgpu::Device as usize
}

#[cfg(test)]
mod tests;
