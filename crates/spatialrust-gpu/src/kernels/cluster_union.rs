use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult, TransferDirection, TransferStats};
use wgpu::util::DeviceExt;

use super::{build_voxel_segments_gpu_from_keys_buffer, compute_voxel_keys_gpu_buffers};
use crate::{scratch::ScratchBuffers, WgpuRuntime};

const SOURCE: &str = include_str!("../shaders/euclidean_cluster.wgsl");

pub(crate) struct ClusterPipelines {
    layout: wgpu::BindGroupLayout,
    initialize: wgpu::ComputePipeline,
    connect: wgpu::ComputePipeline,
    compress: wgpu::ComputePipeline,
}

impl ClusterPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..8)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: binding != 7 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cluster-union-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cluster-union-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cluster-union-shader"),
            source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
        });
        let pipeline = |name| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cluster-union-pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            initialize: pipeline("initialize"),
            connect: pipeline("connect"),
            compress: pipeline("compress"),
            layout,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    points: u32,
    cells: u32,
    radius_sq: f32,
    all_connected: u32,
}

fn plan(points: u64, unbounded: bool, limits: &wgpu::Limits) -> SpatialResult<(u32, u64)> {
    let bytes = points.checked_mul(4).ok_or_else(invalid)?;
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    if points == 0
        || points > u64::from(u32::MAX - 255)
        || bytes > bound
        || bound < 16
        || bytes > isize::MAX as u64
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
        || points.div_ceil(256) > u64::from(limits.max_compute_workgroups_per_dimension)
        || limits.max_storage_buffers_per_shader_stage < 7
        || limits.max_bindings_per_bind_group < 8
        || limits.max_bind_groups == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < 16
        || limits.max_buffer_size < 16
        || (!unbounded
            && (points * 16 > bound
                || bound < 32
                || limits.max_uniform_buffer_binding_size < 32
                || limits.max_compute_workgroup_storage_size < 2052))
    {
        return Err(invalid());
    }
    Ok((points as u32, bytes))
}

fn invalid() -> SpatialError {
    SpatialError::InvalidArgument("GPU clustering requires finite XYZ, a positive supported tolerance and bounded buffer/compute limits".to_owned())
}

pub(super) fn run(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    tolerance: f32,
) -> SpatialResult<(Vec<u32>, TransferStats)> {
    let unbounded = tolerance == f32::INFINITY;
    let (points, bytes) = plan(x.len() as u64, unbounded, &runtime.device().limits())?;
    if x.iter().chain(y).chain(z).any(|v| !v.is_finite()) {
        return Err(invalid());
    }
    let inverse = 1.0 / tolerance;
    if !unbounded && !inverse.is_finite() {
        return Err(invalid());
    }

    let mut transfers = TransferStats::default();
    let positions = if unbounded {
        None
    } else {
        let origin = [
            x.iter().copied().fold(f32::INFINITY, f32::min),
            y.iter().copied().fold(f32::INFINITY, f32::min),
            z.iter().copied().fold(f32::INFINITY, f32::min),
        ];
        let positions = compute_voxel_keys_gpu_buffers(runtime, x, y, z, origin, inverse)?;
        transfers.record(TransferDirection::HostToDevice, u64::from(points) * 12);
        Some(positions)
    };
    let segments = if let Some(positions) = &positions {
        let segments = build_voxel_segments_gpu_from_keys_buffer(
            runtime,
            positions.keys_buffer(),
            points,
            points,
        )?;
        transfers.record(TransferDirection::DeviceToHost, segments.construction_readback_bytes());
        Some(segments)
    } else {
        None
    };

    let cache = runtime.cluster_pipelines.get_or_init(|| ClusterPipelines::new(runtime.device()));
    let mut scratch = ScratchBuffers::new(runtime);
    let parent = scratch.acquire("cluster-parents", bytes);
    // The infinite-radius fast path only initializes parents, but the shared
    // explicit layout still needs valid minimum-sized unused bindings.
    let placeholder = if unbounded { Some(scratch.acquire("cluster-unused", 16)) } else { None };
    let dummy = placeholder.as_ref();
    let buffers = [
        positions.as_ref().map(|p| p.x_buffer()).or(dummy).unwrap(),
        positions.as_ref().map(|p| p.y_buffer()).or(dummy).unwrap(),
        positions.as_ref().map(|p| p.z_buffer()).or(dummy).unwrap(),
        segments.as_ref().map(|p| p.keys_buffer()).or(dummy).unwrap(),
        segments.as_ref().map(|p| p.point_indices_buffer()).or(dummy).unwrap(),
        segments.as_ref().map(|p| p.cell_starts_buffer()).or(dummy).unwrap(),
        &parent,
    ];
    let params = Params {
        points,
        cells: segments.as_ref().map_or(0, |s| s.cell_count()),
        radius_sq: tolerance * tolerance,
        all_connected: u32::from(unbounded),
    };
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cluster-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut entries =
        vec![wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }];
    entries.extend(buffers.iter().enumerate().map(|(i, buffer)| wgpu::BindGroupEntry {
        binding: i as u32 + 1,
        resource: buffer.as_entire_binding(),
    }));
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cluster-bindings"),
        layout: &cache.layout,
        entries: &entries,
    });
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("cluster-root-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    for (pipeline, label) in [
        (&cache.initialize, "cluster-initialize"),
        (&cache.connect, "cluster-connect"),
        (&cache.compress, "cluster-compress"),
    ] {
        if unbounded && label != "cluster-initialize" {
            continue;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(points.div_ceil(256), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&parent, 0, &staging, 0, bytes);
    scratch.submit_and_recycle(encoder);
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    runtime.device().poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|e| SpatialError::InvalidArgument(e.to_string()))?
        .map_err(|e| SpatialError::InvalidArgument(e.to_string()))?;
    let mapped = slice.get_mapped_range();
    let roots = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    if let Some(positions) = positions {
        positions.recycle(runtime);
    }
    transfers.record(TransferDirection::DeviceToHost, bytes);
    Ok((roots, transfers))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "cluster_union_tests.rs"]
mod tests;
