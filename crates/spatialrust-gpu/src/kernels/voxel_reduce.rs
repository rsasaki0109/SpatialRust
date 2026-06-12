use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use wgpu::util::DeviceExt;

use crate::kernels::gpu_segments::GpuVoxelSegments;
use crate::kernels::voxel_segments::VoxelSegments;
use crate::runtime::WgpuRuntime;

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ReduceUniform {
    cell_count: u32,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Averages per-point `f32` values within each voxel cell on the GPU.
pub fn reduce_voxel_average_f32(
    runtime: &WgpuRuntime,
    values: &[f32],
    segments: &VoxelSegments,
) -> SpatialResult<Vec<f32>> {
    if segments.is_empty() {
        return Ok(Vec::new());
    }
    if values.is_empty() {
        return Err(SpatialError::InvalidArgument(
            "cannot reduce empty value buffer".to_owned(),
        ));
    }

    let device = runtime.device();
    let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-values"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-indices"),
        contents: bytemuck::cast_slice(&segments.point_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let starts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-starts"),
        contents: bytemuck::cast_slice(&segments.cell_starts),
        usage: wgpu::BufferUsages::STORAGE,
    });

    dispatch_voxel_reduce_f32(
        runtime,
        &values_buffer,
        &indices_buffer,
        &starts_buffer,
        segments.len() as u32,
        segments.point_indices.len() as u32,
    )
}

/// Averages per-point `f32` values using GPU-resident segment buffers.
pub fn reduce_voxel_average_f32_gpu_buffers(
    runtime: &WgpuRuntime,
    values: &wgpu::Buffer,
    segments: &GpuVoxelSegments,
) -> SpatialResult<Vec<f32>> {
    dispatch_voxel_reduce_f32(
        runtime,
        values,
        segments.point_indices_buffer(),
        segments.cell_starts_buffer(),
        segments.cell_count(),
        segments.point_count(),
    )
}

/// Uploads `f32` values and averages them within GPU-resident voxel segments.
pub fn reduce_voxel_average_f32_gpu(
    runtime: &WgpuRuntime,
    values: &[f32],
    segments: &GpuVoxelSegments,
) -> SpatialResult<Vec<f32>> {
    if segments.cell_count() == 0 {
        return Ok(Vec::new());
    }
    if values.len() != segments.point_count() as usize {
        return Err(SpatialError::BufferLengthMismatch {
            expected: segments.point_count() as usize,
            found: values.len(),
        });
    }

    let device = runtime.device();
    let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-values-upload"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    reduce_voxel_average_f32_gpu_buffers(runtime, &values_buffer, segments)
}

fn dispatch_voxel_reduce_f32(
    runtime: &WgpuRuntime,
    values: &wgpu::Buffer,
    point_indices: &wgpu::Buffer,
    cell_starts: &wgpu::Buffer,
    cell_count: u32,
    point_count: u32,
) -> SpatialResult<Vec<f32>> {
    if cell_count == 0 {
        return Ok(Vec::new());
    }

    let device = runtime.device();
    let queue = runtime.queue();

    let uniform = ReduceUniform {
        cell_count,
        point_count,
        _pad0: 0,
        _pad1: 0,
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let output_len = cell_count as usize * std::mem::size_of::<f32>();
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-output"),
        size: output_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-staging"),
        size: output_len as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pipelines = runtime.pipelines();

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-reduce-bind-group"),
        layout: &pipelines.voxel_reduce.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: point_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cell_starts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-reduce-encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-reduce-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_reduce.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_len as u64);
    queue.submit(Some(encoder.finish()));

    read_staging_f32(device, &staging_buffer, cell_count as usize)
}

/// Averages xyz positions within each voxel cell on the GPU.
pub fn reduce_voxel_centroids_xyz(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    segments: &VoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if x.len() != y.len() || x.len() != z.len() {
        return Err(SpatialError::BufferLengthMismatch {
            expected: x.len(),
            found: y.len(),
        });
    }

    let out_x = reduce_voxel_average_f32(runtime, x, segments)?;
    let out_y = reduce_voxel_average_f32(runtime, y, segments)?;
    let out_z = reduce_voxel_average_f32(runtime, z, segments)?;
    Ok((out_x, out_y, out_z))
}

/// Averages xyz positions using GPU-resident buffers.
pub fn reduce_voxel_centroids_xyz_gpu_buffers(
    runtime: &WgpuRuntime,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z: &wgpu::Buffer,
    segments: &GpuVoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
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
    let pipelines = runtime.pipelines();

    let uniform = ReduceUniform {
        cell_count,
        point_count,
        _pad0: 0,
        _pad1: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-xyz-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-reduce-xyz-bind-group"),
        layout: &pipelines.voxel_reduce.xyz_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: point_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cell_starts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: values_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: values_y.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: values_z.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: output_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: output_y.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: output_z.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-reduce-xyz-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-reduce-xyz-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_reduce.xyz_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_x, 0, &staging_buffer, 0, channel_len as u64);
    encoder.copy_buffer_to_buffer(&output_y, 0, &staging_buffer, channel_len as u64, channel_len as u64);
    encoder.copy_buffer_to_buffer(
        &output_z,
        0,
        &staging_buffer,
        (channel_len * 2) as u64,
        channel_len as u64,
    );
    queue.submit(Some(encoder.finish()));

    let flat = read_staging_f32(device, &staging_buffer, cell_count as usize * 3)?;
    let cells = cell_count as usize;
    Ok((
        flat[..cells].to_vec(),
        flat[cells..cells * 2].to_vec(),
        flat[cells * 2..cells * 3].to_vec(),
    ))
}

fn read_staging_f32(device: &wgpu::Device, staging_buffer: &wgpu::Buffer, len: usize) -> SpatialResult<Vec<f32>> {
    let slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| SpatialError::InvalidArgument("failed to receive wgpu map result".to_owned()))?
        .map_err(|error| SpatialError::InvalidArgument(format!("failed to map wgpu buffer: {error}")))?;

    let data = slice.get_mapped_range();
    let values: Vec<f32> = bytemuck::cast_slice(&data)[..len].to_vec();
    drop(data);
    staging_buffer.unmap();
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::{reduce_voxel_average_f32, reduce_voxel_centroids_xyz};
    use crate::kernels::voxel_segments::build_voxel_segments;
    use crate::runtime::WgpuRuntime;

    #[test]
    fn gpu_centroid_reduction_matches_cpu_reference() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let x = [0.0_f32, 0.1, 1.0, 1.1];
        let y = [0.0_f32, 0.0, 0.0, 0.0];
        let z = [0.0_f32, 0.0, 0.0, 0.0];
        let keys = vec![(0, 0, 0), (0, 0, 0), (2, 0, 0), (2, 0, 0)];
        let segments = build_voxel_segments(&keys);

        let (gpu_x, gpu_y, gpu_z) =
            reduce_voxel_centroids_xyz(&runtime, &x, &y, &z, &segments).expect("gpu reduce");

        assert!((gpu_x[0] - 0.05).abs() < 1e-5);
        assert!((gpu_x[1] - 1.05).abs() < 1e-5);
        assert_eq!(gpu_y, vec![0.0, 0.0]);
        assert_eq!(gpu_z, vec![0.0, 0.0]);

        let intensity = [0.2_f32, 0.8, 10.0, 20.0];
        let gpu_i = reduce_voxel_average_f32(&runtime, &intensity, &segments).expect("gpu average");
        assert!((gpu_i[0] - 0.5).abs() < 1e-5);
        assert!((gpu_i[1] - 15.0).abs() < 1e-5);
    }
}
