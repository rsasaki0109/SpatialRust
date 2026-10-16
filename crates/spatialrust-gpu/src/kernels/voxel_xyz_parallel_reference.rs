// Complete standalone XYZ parents, packed controller and serial reader frozen from beecc3b7b7c87af99531889fe77e6169c1f93a26.
use super::*;
use crate::kernels::voxel_xyz_packed::{plan_packed_xyz, record_packed_xyz, XyzMode, XyzPlan};
struct StagingUnmap<'a>(&'a wgpu::Buffer);
impl Drop for StagingUnmap<'_> {
    fn drop(&mut self) {
        self.0.unmap();
    }
}
pub(super) fn read_staging_xyz(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    cells: usize,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| SpatialError::InvalidArgument("failed to receive wgpu map result".to_owned()))?
        .map_err(|error| {
            SpatialError::InvalidArgument(format!("failed to map wgpu buffer: {error}"))
        })?;

    // Declared before the view so it also unmaps after the view drops on unwind.
    let _unmap = StagingUnmap(staging_buffer);
    let data = slice.get_mapped_range();
    let values: &[f32] = bytemuck::cast_slice(&data);
    let values = &values[..cells * 3];
    Ok((values[..cells].to_vec(), values[cells..cells * 2].to_vec(), values[cells * 2..].to_vec()))
}
pub(super) fn try_voxel_xyz_packed(
    runtime: &WgpuRuntime,
    xyz: [&wgpu::Buffer; 3],
    segments: &GpuVoxelSegments,
    mode: XyzMode,
) -> Option<SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
    if segments.cell_count() == 0 {
        return None;
    }
    let plan =
        plan_packed_xyz(&runtime.device().limits(), segments.cell_count(), isize::MAX as u64)?;
    Some(dispatch_packed_xyz(runtime, xyz, segments, mode, plan))
}
fn dispatch_packed_xyz(
    runtime: &WgpuRuntime,
    xyz: [&wgpu::Buffer; 3],
    segments: &GpuVoxelSegments,
    mode: XyzMode,
    plan: XyzPlan,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let device = runtime.device();
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-xyz-packed-output"),
        size: plan.bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-xyz-packed-staging"),
        size: plan.bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-xyz-packed-encoder"),
    });
    record_packed_xyz(&mut encoder, runtime, xyz, segments, mode, &output);
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, plan.bytes);
    runtime.queue().submit(Some(encoder.finish()));
    read_staging_xyz(device, &staging, segments.cell_count() as usize)
}
fn dispatch_voxel_reduce_xyz_f32(
    runtime: &WgpuRuntime,
    values_x: &wgpu::Buffer,
    values_y: &wgpu::Buffer,
    values_z: &wgpu::Buffer,
    point_indices: &wgpu::Buffer,
    cell_starts: &wgpu::Buffer,
    cell_count: u32,
    point_count: u32,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if cell_count == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    let device = runtime.device();
    let queue = runtime.queue();
    let channel_len = cell_count as usize * std::mem::size_of::<f32>();
    let output_x = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-xyz-out-x"),
        size: channel_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_y = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-xyz-out-y"),
        size: channel_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_z = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-xyz-out-z"),
        size: channel_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-xyz-staging"),
        size: (channel_len * 3) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-reduce-xyz-encoder"),
    });
    record_voxel_reduce_xyz_pass(
        &mut encoder,
        runtime,
        values_x,
        values_y,
        values_z,
        point_indices,
        cell_starts,
        cell_count,
        point_count,
        &output_x,
        &output_y,
        &output_z,
    )?;
    encoder.copy_buffer_to_buffer(&output_x, 0, &staging_buffer, 0, channel_len as u64);
    encoder.copy_buffer_to_buffer(
        &output_y,
        0,
        &staging_buffer,
        channel_len as u64,
        channel_len as u64,
    );
    encoder.copy_buffer_to_buffer(
        &output_z,
        0,
        &staging_buffer,
        (channel_len * 2) as u64,
        channel_len as u64,
    );
    queue.submit(Some(encoder.finish()));

    read_staging_xyz(device, &staging_buffer, cell_count as usize)
}
pub(super) fn baseline_gather_voxel_first_xyz_gpu_buffers(
    runtime: &WgpuRuntime,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z: &wgpu::Buffer,
    segments: &GpuVoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if segments.cell_count() == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    if let Some(result) = try_voxel_xyz_packed(runtime, [x, y, z], segments, XyzMode::First) {
        return result;
    }

    let device = runtime.device();
    let queue = runtime.queue();
    let cell_count = segments.cell_count();
    let channel_len = cell_count as usize * std::mem::size_of::<f32>();
    let output_x = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-xyz-out-x"),
        size: channel_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_y = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-xyz-out-y"),
        size: channel_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_z = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-xyz-out-z"),
        size: channel_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-xyz-staging"),
        size: (channel_len * 3) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-gather-xyz-encoder"),
    });
    record_voxel_gather_xyz_pass(
        &mut encoder,
        runtime,
        x,
        y,
        z,
        segments,
        &output_x,
        &output_y,
        &output_z,
    )?;
    encoder.copy_buffer_to_buffer(&output_x, 0, &staging_buffer, 0, channel_len as u64);
    encoder.copy_buffer_to_buffer(
        &output_y,
        0,
        &staging_buffer,
        channel_len as u64,
        channel_len as u64,
    );
    encoder.copy_buffer_to_buffer(
        &output_z,
        0,
        &staging_buffer,
        (channel_len * 2) as u64,
        channel_len as u64,
    );
    queue.submit(Some(encoder.finish()));

    read_staging_xyz(device, &staging_buffer, cell_count as usize)
}
pub(super) fn baseline_reduce_voxel_centroids_xyz_gpu_buffers(
    runtime: &WgpuRuntime,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z: &wgpu::Buffer,
    segments: &GpuVoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if let Some(result) = try_voxel_xyz_packed(runtime, [x, y, z], segments, XyzMode::Centroid) {
        return result;
    }

    dispatch_voxel_reduce_xyz_f32(
        runtime,
        x,
        y,
        z,
        segments.point_indices_buffer(),
        segments.cell_starts_buffer(),
        segments.cell_count(),
        segments.point_count(),
    )
}
