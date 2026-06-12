use spatialrust_core::SpatialResult;

use crate::kernels::gpu_segments::GpuVoxelSegments;
use crate::kernels::voxel_keys::{compute_voxel_keys_gpu_buffers, GpuVoxelKeyBuffers};
use crate::kernels::voxel_gather::gather_voxel_first_xyz_gpu_buffers;
use crate::kernels::voxel_reduce::reduce_voxel_centroids_xyz_gpu_buffers;
use crate::kernels::voxel_sort::build_voxel_segments_gpu_from_keys_buffer;
use crate::runtime::WgpuRuntime;

/// GPU-resident centroid downsample result.
pub struct VoxelCentroidGpuResult {
    /// Averaged x coordinates per voxel cell.
    pub out_x: Vec<f32>,
    /// Averaged y coordinates per voxel cell.
    pub out_y: Vec<f32>,
    /// Averaged z coordinates per voxel cell.
    pub out_z: Vec<f32>,
    /// Segment metadata kept on the GPU for follow-up reductions.
    pub segments: GpuVoxelSegments,
    /// Source position buffers kept on the GPU.
    pub positions: GpuVoxelKeyBuffers,
}

/// Runs the chained GPU voxel centroid pipeline without intermediate key/segment readbacks.
pub fn downsample_voxel_centroid_gpu(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: [f32; 3],
    inv_leaf: f32,
) -> SpatialResult<VoxelCentroidGpuResult> {
    let positions = compute_voxel_keys_gpu_buffers(runtime, x, y, z, origin, inv_leaf)?;
    let point_count = positions.point_count();
    let padded_count = point_count.next_power_of_two();
    let segments = build_voxel_segments_gpu_from_keys_buffer(
        runtime,
        positions.keys_buffer(),
        point_count,
        padded_count,
    )?;
    let (out_x, out_y, out_z) = reduce_voxel_centroids_xyz_gpu_buffers(
        runtime,
        positions.x_buffer(),
        positions.y_buffer(),
        positions.z_buffer(),
        &segments,
    )?;

    Ok(VoxelCentroidGpuResult {
        out_x,
        out_y,
        out_z,
        segments,
        positions,
    })
}

/// GPU-resident approximate-first downsample result.
pub struct VoxelApproximateFirstGpuResult {
    /// First-point x coordinates per voxel cell.
    pub out_x: Vec<f32>,
    /// First-point y coordinates per voxel cell.
    pub out_y: Vec<f32>,
    /// First-point z coordinates per voxel cell.
    pub out_z: Vec<f32>,
    /// Segment metadata kept on the GPU for follow-up gathers/reductions.
    pub segments: GpuVoxelSegments,
    /// Source position buffers kept on the GPU.
    pub positions: GpuVoxelKeyBuffers,
}

/// Runs the chained GPU approximate-first pipeline without intermediate readbacks.
pub fn downsample_voxel_approximate_first_gpu(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: [f32; 3],
    inv_leaf: f32,
) -> SpatialResult<VoxelApproximateFirstGpuResult> {
    let positions = compute_voxel_keys_gpu_buffers(runtime, x, y, z, origin, inv_leaf)?;
    let point_count = positions.point_count();
    let padded_count = point_count.next_power_of_two();
    let segments = build_voxel_segments_gpu_from_keys_buffer(
        runtime,
        positions.keys_buffer(),
        point_count,
        padded_count,
    )?;
    let (out_x, out_y, out_z) = gather_voxel_first_xyz_gpu_buffers(
        runtime,
        positions.x_buffer(),
        positions.y_buffer(),
        positions.z_buffer(),
        &segments,
    )?;

    Ok(VoxelApproximateFirstGpuResult {
        out_x,
        out_y,
        out_z,
        segments,
        positions,
    })
}

#[cfg(test)]
mod tests {
    use super::{downsample_voxel_approximate_first_gpu, downsample_voxel_centroid_gpu};
    use crate::kernels::voxel_segments::build_voxel_segments;
    use crate::kernels::voxel_reduce::reduce_voxel_centroids_xyz;
    use crate::runtime::WgpuRuntime;

    #[test]
    fn chained_pipeline_matches_staged_gpu_reference() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let x = [0.0_f32, 0.1, 1.0, 1.1];
        let y = [0.0_f32, 0.0, 0.0, 0.0];
        let z = [0.0_f32, 0.0, 0.0, 0.0];
        let origin = [0.0_f32, 0.0, 0.0];
        let inv_leaf = 2.0_f32;

        let chained = downsample_voxel_centroid_gpu(&runtime, &x, &y, &z, origin, inv_leaf)
            .expect("chained pipeline");

        let keys: Vec<(i64, i64, i64)> = x
            .iter()
            .zip(y.iter())
            .zip(z.iter())
            .map(|((x, y), z)| {
                (
                    ((x - origin[0]) * inv_leaf).floor() as i64,
                    ((y - origin[1]) * inv_leaf).floor() as i64,
                    ((z - origin[2]) * inv_leaf).floor() as i64,
                )
            })
            .collect();
        let segments = build_voxel_segments(&keys);
        let (ref_x, ref_y, ref_z) =
            reduce_voxel_centroids_xyz(&runtime, &x, &y, &z, &segments).expect("staged reduce");

        assert!((chained.out_x[0] - ref_x[0]).abs() < 1e-5);
        assert!((chained.out_x[1] - ref_x[1]).abs() < 1e-5);
        assert_eq!(chained.out_y, ref_y);
        assert_eq!(chained.out_z, ref_z);
    }

    #[test]
    fn approximate_first_pipeline_keeps_first_point() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let x = [0.0_f32, 0.1];
        let y = [0.0_f32, 0.0];
        let z = [0.0_f32, 0.0];
        let origin = [0.0_f32, 0.0, 0.0];
        let inv_leaf = 1.0_f32;

        let result = downsample_voxel_approximate_first_gpu(&runtime, &x, &y, &z, origin, inv_leaf)
            .expect("approximate-first pipeline");

        assert_eq!(result.out_x.len(), 1);
        assert!((result.out_x[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn centroid_pipeline_handles_non_pot_point_count() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let point_count = 100_000_usize;
        let mut x = Vec::with_capacity(point_count);
        let mut y = Vec::with_capacity(point_count);
        let mut z = Vec::with_capacity(point_count);
        for index in 0..point_count {
            let t = index as f32;
            x.push((t * 0.013).fract() * 100.0);
            y.push(((index % 97) as f32) * 0.017);
            z.push(((index % 53) as f32) * 0.019);
        }

        downsample_voxel_centroid_gpu(&runtime, &x, &y, &z, [0.0; 3], 0.25)
            .expect("centroid pipeline for 100k points");
    }
}
