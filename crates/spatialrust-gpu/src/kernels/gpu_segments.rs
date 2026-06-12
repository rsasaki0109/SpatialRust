//! GPU-resident voxel segment buffers used across chained kernels.

use spatialrust_core::SpatialResult;

use crate::kernels::voxel_compact::{finalize_segments_from_readback, read_segment_metadata};
use crate::kernels::voxel_segments::VoxelSegments;
use crate::runtime::WgpuRuntime;

/// Voxel segment metadata kept on the GPU between sort/compact and reduce passes.
pub struct GpuVoxelSegments {
    cell_count: u32,
    point_count: u32,
    keys: wgpu::Buffer,
    point_indices: wgpu::Buffer,
    cell_starts: wgpu::Buffer,
}

impl GpuVoxelSegments {
    /// Creates GPU segment buffers.
    pub(crate) fn new(
        cell_count: u32,
        point_count: u32,
        keys: wgpu::Buffer,
        point_indices: wgpu::Buffer,
        cell_starts: wgpu::Buffer,
    ) -> Self {
        Self {
            cell_count,
            point_count,
            keys,
            point_indices,
            cell_starts,
        }
    }

    /// Returns the number of occupied voxel cells.
    #[must_use]
    pub fn cell_count(&self) -> u32 {
        self.cell_count
    }

    /// Returns the source point count represented by these segments.
    #[must_use]
    pub fn point_count(&self) -> u32 {
        self.point_count
    }

    /// Returns the GPU buffer of unique voxel keys.
    #[must_use]
    pub fn keys_buffer(&self) -> &wgpu::Buffer {
        &self.keys
    }

    /// Returns the GPU buffer of sorted point indices grouped by cell.
    #[must_use]
    pub fn point_indices_buffer(&self) -> &wgpu::Buffer {
        &self.point_indices
    }

    /// Returns the GPU buffer of per-cell start offsets into `point_indices`.
    #[must_use]
    pub fn cell_starts_buffer(&self) -> &wgpu::Buffer {
        &self.cell_starts
    }

    /// Readbacks segment metadata into the CPU-side representation.
    pub fn to_voxel_segments(&self, runtime: &WgpuRuntime) -> SpatialResult<VoxelSegments> {
        if self.cell_count == 0 {
            return Ok(VoxelSegments {
                keys: Vec::new(),
                point_indices: Vec::new(),
                cell_starts: Vec::new(),
                cell_counts: Vec::new(),
            });
        }

        let (keys, cell_starts, cell_counts, point_indices) =
            read_segment_metadata(runtime, self, self.cell_count as usize)?;

        Ok(finalize_segments_from_readback(
            keys,
            point_indices,
            cell_starts,
            cell_counts,
        ))
    }
}
