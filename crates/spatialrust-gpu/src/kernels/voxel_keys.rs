use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use wgpu::util::DeviceExt;

use crate::runtime::WgpuRuntime;

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
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(crate) struct VoxelKeyOutput {
    pub(crate) ix: i32,
    pub(crate) iy: i32,
    pub(crate) iz: i32,
    pub(crate) _pad: i32,
}

/// GPU buffers for per-point positions and computed voxel keys.
pub struct GpuVoxelKeyBuffers {
    x: wgpu::Buffer,
    y: wgpu::Buffer,
    z: wgpu::Buffer,
    keys: wgpu::Buffer,
    point_count: u32,
}

impl GpuVoxelKeyBuffers {
    /// Returns the number of source points.
    #[must_use]
    pub fn point_count(&self) -> u32 {
        self.point_count
    }

    /// Returns the GPU buffer of x coordinates.
    #[must_use]
    pub fn x_buffer(&self) -> &wgpu::Buffer {
        &self.x
    }

    /// Returns the GPU buffer of y coordinates.
    #[must_use]
    pub fn y_buffer(&self) -> &wgpu::Buffer {
        &self.y
    }

    /// Returns the GPU buffer of z coordinates.
    #[must_use]
    pub fn z_buffer(&self) -> &wgpu::Buffer {
        &self.z
    }

    /// Returns the GPU buffer of computed voxel keys.
    #[must_use]
    pub fn keys_buffer(&self) -> &wgpu::Buffer {
        &self.keys
    }

    /// Returns position/key GPU buffers to the runtime upload pool.
    pub fn recycle(self, runtime: &WgpuRuntime) {
        runtime.recycle_storage(self.x.size(), self.x);
        runtime.recycle_storage(self.y.size(), self.y);
        runtime.recycle_storage(self.z.size(), self.z);
        runtime.recycle_storage(self.keys.size(), self.keys);
    }
}

/// Computes per-point voxel grid keys on the GPU.
pub fn compute_voxel_keys(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: [f32; 3],
    inv_leaf: f32,
) -> SpatialResult<Vec<(i64, i64, i64)>> {
    if x.len() != y.len() || x.len() != z.len() {
        return Err(SpatialError::BufferLengthMismatch {
            expected: x.len(),
            found: y.len(),
        });
    }
    if x.is_empty() {
        return Ok(Vec::new());
    }

    let buffers = compute_voxel_keys_gpu_buffers(runtime, x, y, z, origin, inv_leaf)?;
    read_voxel_keys(runtime, &buffers)
}

/// Uploads positions and computes per-point voxel keys, keeping data on the GPU.
pub fn compute_voxel_keys_gpu_buffers(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: [f32; 3],
    inv_leaf: f32,
) -> SpatialResult<GpuVoxelKeyBuffers> {
    if x.len() != y.len() || x.len() != z.len() {
        return Err(SpatialError::BufferLengthMismatch {
            expected: x.len(),
            found: y.len(),
        });
    }
    if x.is_empty() {
        return Err(SpatialError::InvalidArgument(
            "cannot compute voxel keys for an empty point cloud".to_owned(),
        ));
    }

    let point_count = x.len() as u32;

    let x_buffer = runtime.upload_f32_storage("voxel-key-x", x)?;
    let y_buffer = runtime.upload_f32_storage("voxel-key-y", y)?;
    let z_buffer = runtime.upload_f32_storage("voxel-key-z", z)?;

    let keys_buffer = dispatch_voxel_keys(
        runtime,
        &x_buffer,
        &y_buffer,
        &z_buffer,
        origin,
        inv_leaf,
        point_count,
    )?;

    Ok(GpuVoxelKeyBuffers {
        x: x_buffer,
        y: y_buffer,
        z: z_buffer,
        keys: keys_buffer,
        point_count,
    })
}

fn read_voxel_keys(runtime: &WgpuRuntime, buffers: &GpuVoxelKeyBuffers) -> SpatialResult<Vec<(i64, i64, i64)>> {
    let device = runtime.device();
    let queue = runtime.queue();
    let point_count = buffers.point_count() as usize;
    let output_len = point_count * std::mem::size_of::<VoxelKeyOutput>();

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-key-staging"),
        size: output_len as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-key-readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(buffers.keys_buffer(), 0, &staging_buffer, 0, output_len as u64);
    queue.submit(Some(encoder.finish()));

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
    let outputs: &[VoxelKeyOutput] = bytemuck::cast_slice(&data);
    let keys = outputs
        .iter()
        .map(|key| (i64::from(key.ix), i64::from(key.iy), i64::from(key.iz)))
        .collect();
    drop(data);
    staging_buffer.unmap();

    Ok(keys)
}

fn dispatch_voxel_keys(
    runtime: &WgpuRuntime,
    x_buffer: &wgpu::Buffer,
    y_buffer: &wgpu::Buffer,
    z_buffer: &wgpu::Buffer,
    origin: [f32; 3],
    inv_leaf: f32,
    point_count: u32,
) -> SpatialResult<wgpu::Buffer> {
    let device = runtime.device();
    let queue = runtime.queue();

    let uniform = VoxelKeyUniform {
        origin: [origin[0], origin[1], origin[2], 0.0],
        inv_leaf,
        point_count,
        _pad0: 0,
        _pad1: 0,
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-key-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let output_len = point_count as usize * std::mem::size_of::<VoxelKeyOutput>();
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-key-output"),
        size: output_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let pipelines = runtime.pipelines();

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-key-bind-group"),
        layout: &pipelines.voxel_keys.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: y_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: z_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-key-encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-key-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_keys.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = point_count.div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    Ok(output_buffer)
}

#[cfg(test)]
mod tests {
    use super::compute_voxel_keys;
    use crate::runtime::WgpuRuntime;

    #[test]
    fn gpu_voxel_keys_match_cpu_reference() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let x = [0.0_f32, 0.1, 1.0, 1.1];
        let y = [0.0_f32, 0.0, 0.0, 0.0];
        let z = [0.0_f32, 0.0, 0.0, 0.0];
        let origin = [0.0_f32, 0.0, 0.0];
        let inv_leaf = 2.0_f32;

        let gpu_keys =
            compute_voxel_keys(&runtime, &x, &y, &z, origin, inv_leaf).expect("gpu keys");

        let cpu_keys: Vec<(i64, i64, i64)> = x
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

        assert_eq!(gpu_keys, cpu_keys);
    }
}
