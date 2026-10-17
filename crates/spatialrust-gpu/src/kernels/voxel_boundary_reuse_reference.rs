// Complete immediate-parent compactor host and pipeline constructor; native tests only.
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
    cache: &ParentCompactPipelines,
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
    // Counts are derived from starts on readback; retain only the unused ABI binding.
    let counts_buffer = temporaries.acquire("voxel-compact-counts", 4);
    let indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-indices"),
        size: buffer_len * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let layout = &cache.bind_group_layout;

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

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-compact-mark-pass"),
            timestamp_writes: crate::profiling::timestamp_writes(
                &mut profiler,
                "voxel-compact-mark",
            ),
        });
        pass.set_pipeline(&cache.mark);
        pass.set_bind_group(0, &mark_bind_group, &[]);
        pass.dispatch_workgroups(dispatch, 1, 1);
    }

    crate::kernels::voxel_scan::encode_inclusive_scan(
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
        pass.set_pipeline(&cache.write);
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

#[allow(dead_code)] // Frozen constructor also creates the original legacy scan entrypoints.
pub(super) struct ParentCompactPipelines {
    /// Bind group layout for segment compact dispatch.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Marks segment boundaries.
    pub mark: wgpu::ComputePipeline,
    /// Initializes inclusive scan input.
    pub init: wgpu::ComputePipeline,
    /// Runs one inclusive-scan step.
    pub scan: wgpu::ComputePipeline,
    /// Writes compact segment metadata.
    pub write: wgpu::ComputePipeline,
    _pipeline_layout: wgpu::PipelineLayout,
    _shader: wgpu::ShaderModule,
}

impl ParentCompactPipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-compact-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
            ],
        });
        let shader = load_shader(
            device,
            "voxel-compact-shader",
            include_str!("voxel_boundary_reuse_reference.wgsl"),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-compact-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        Self {
            mark: build_compute_pipeline(
                device,
                &pipeline_layout,
                &shader,
                "voxel-compact-mark",
                "mark_boundaries",
            ),
            init: build_compute_pipeline(
                device,
                &pipeline_layout,
                &shader,
                "voxel-compact-init",
                "init_inclusive",
            ),
            scan: build_compute_pipeline(
                device,
                &pipeline_layout,
                &shader,
                "voxel-compact-scan",
                "scan_step",
            ),
            write: build_compute_pipeline(
                device,
                &pipeline_layout,
                &shader,
                "voxel-compact-write",
                "write_segments",
            ),
            bind_group_layout,
            _pipeline_layout: pipeline_layout,
            _shader: shader,
        }
    }
}

fn load_shader(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

fn build_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
