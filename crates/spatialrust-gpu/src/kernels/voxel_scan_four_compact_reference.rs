// Complete immediate-parent compactor call/helpers; only the scan cache is external.
use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use wgpu::util::DeviceExt;

use crate::kernels::gpu_segments::GpuVoxelSegments;
use crate::runtime::WgpuRuntime;

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CompactParams {
    point_count: u32,
    scan_stride: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct VoxelKeyOutput {
    ix: i32,
    iy: i32,
    iz: i32,
    _pad: i32,
}

pub(super) fn previous(
    cache: &super::old::ParentScanPipelines,
    runtime: &WgpuRuntime,
    entries_buffer: &wgpu::Buffer,
    point_count: u32,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
) -> SpatialResult<GpuVoxelSegments> {
    if point_count == 0 {
        return Ok(GpuVoxelSegments::new(
            0,
            0,
            empty_storage_buffer(runtime)?,
            empty_storage_buffer(runtime)?,
            empty_storage_buffer(runtime)?,
            0,
        ));
    }

    let device = runtime.device();
    let buffer_len = point_count as u64;

    let mut temporaries = crate::scratch::ScratchBuffers::new(runtime);
    let flags_buffer = temporaries.acquire("voxel-compact-flags", buffer_len * 4);
    let inclusive_buffer = temporaries.acquire("voxel-compact-inclusive", buffer_len * 4);
    let scan_scratch_buffer = temporaries.acquire("voxel-compact-unused-scan-output", 4);
    let keys_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-keys"),
        size: buffer_len * std::mem::size_of::<VoxelKeyOutput>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let starts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-starts"),
        size: buffer_len * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let counts_buffer = temporaries.acquire("voxel-compact-counts", buffer_len * 4);
    let indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-indices"),
        size: buffer_len * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let pipelines = runtime.pipelines();
    let layout = &pipelines.voxel_compact.bind_group_layout;

    let mark_params = create_params_buffer(device, point_count, 0);
    let mark_resources = CompactBindResources {
        params: &mark_params,
        entries: entries_buffer,
        flags: &flags_buffer,
        prefix: &inclusive_buffer,
        keys: &keys_buffer,
        starts: &starts_buffer,
        counts: &counts_buffer,
        indices: &indices_buffer,
        scan_out: &scan_scratch_buffer,
    };
    let mark_bind_group = create_compact_bind_group(device, layout, &mark_resources);

    let dispatch = point_count.div_ceil(WORKGROUP_SIZE);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-compact-batched-encoder"),
    });
    encoder.clear_buffer(&counts_buffer, 0, None);

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-compact-mark-pass"),
            timestamp_writes: crate::profiling::timestamp_writes(
                &mut profiler,
                "voxel-compact-mark",
            ),
        });
        pass.set_pipeline(&pipelines.voxel_compact.mark);
        pass.set_bind_group(0, &mark_bind_group, &[]);
        pass.dispatch_workgroups(dispatch, 1, 1);
    }

    super::old::encode_inclusive_scan(
        cache,
        runtime,
        &mut encoder,
        &flags_buffer,
        &inclusive_buffer,
        point_count,
        &mut profiler,
        &mut temporaries,
    );

    let write_params = create_params_buffer(device, point_count, 0);
    let write_resources = CompactBindResources {
        params: &write_params,
        entries: entries_buffer,
        flags: &flags_buffer,
        prefix: &inclusive_buffer,
        keys: &keys_buffer,
        starts: &starts_buffer,
        counts: &counts_buffer,
        indices: &indices_buffer,
        scan_out: &scan_scratch_buffer,
    };
    let write_bind_group = create_compact_bind_group(device, layout, &write_resources);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-compact-write-pass"),
            timestamp_writes: crate::profiling::timestamp_writes(
                &mut profiler,
                "voxel-compact-write",
            ),
        });
        pass.set_pipeline(&pipelines.voxel_compact.write);
        pass.set_bind_group(0, &write_bind_group, &[]);
        pass.dispatch_workgroups(dispatch, 1, 1);
    }

    // Copy the scalar before releasing the prefix lease. Another thread may
    // immediately acquire it after submission, but queue order protects this copy.
    let count_staging =
        encode_last_inclusive_count(device, &mut encoder, &inclusive_buffer, point_count);
    temporaries.submit_and_recycle(encoder);
    let num_cells = read_staging_u32(device, &count_staging, 1)?[0] as usize;
    let readback_bytes = count_staging.size();
    if num_cells > point_count as usize {
        return Err(SpatialError::InvalidArgument(format!(
            "GPU compact produced {num_cells} cells for {point_count} points"
        )));
    }

    Ok(GpuVoxelSegments::new(
        num_cells as u32,
        point_count,
        keys_buffer,
        indices_buffer,
        starts_buffer,
        readback_bytes,
    ))
}

struct CompactBindResources<'a> {
    params: &'a wgpu::Buffer,
    entries: &'a wgpu::Buffer,
    flags: &'a wgpu::Buffer,
    prefix: &'a wgpu::Buffer,
    keys: &'a wgpu::Buffer,
    starts: &'a wgpu::Buffer,
    counts: &'a wgpu::Buffer,
    indices: &'a wgpu::Buffer,
    scan_out: &'a wgpu::Buffer,
}

fn create_compact_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    resources: &CompactBindResources<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-compact-bind-group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: resources.params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: resources.entries.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: resources.flags.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: resources.prefix.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: resources.keys.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: resources.starts.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: resources.counts.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: resources.indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: resources.scan_out.as_entire_binding() },
        ],
    })
}

fn empty_storage_buffer(runtime: &WgpuRuntime) -> SpatialResult<wgpu::Buffer> {
    Ok(runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-empty"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    }))
}

fn create_params_buffer(device: &wgpu::Device, point_count: u32, scan_stride: u32) -> wgpu::Buffer {
    let params = CompactParams { point_count, scan_stride, _pad0: 0, _pad1: 0 };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-compact-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn encode_last_inclusive_count(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    inclusive_buffer: &wgpu::Buffer,
    point_count: u32,
) -> wgpu::Buffer {
    let offset = ((point_count as u64).saturating_sub(1)) * std::mem::size_of::<u32>() as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-inclusive-staging"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(inclusive_buffer, offset, &staging, 0, staging.size());
    staging
}

fn read_staging_u32(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    len: usize,
) -> SpatialResult<Vec<u32>> {
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

    let data = slice.get_mapped_range();
    let values = bytemuck::cast_slice(&data)[..len].to_vec();
    drop(data);
    staging_buffer.unmap();
    Ok(values)
}
