// Complete RANSAC planner/pipeline/API from merged PR 256; native tests only.
use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

use crate::runtime::WgpuRuntime;

const WORKGROUP_SIZE: u32 = 256;
const POINT_PARALLEL_MIN: u32 = 1024;

struct Plan {
    point_count: u32,
    hypothesis_count: u32,
    groups: u32,
    parallel: bool,
    output_bytes: u64,
}

fn plan(points: usize, hypotheses: usize, limits: &wgpu::Limits) -> SpatialResult<Plan> {
    let invalid = || {
        SpatialError::InvalidArgument("RANSAC scoring requires non-empty u32-sized arrays and supported buffer/compute limits".to_owned())
    };
    let point_count = u32::try_from(points).map_err(|_| invalid())?;
    let hypothesis_count = u32::try_from(hypotheses).map_err(|_| invalid())?;
    let parallel = point_count >= POINT_PARALLEL_MIN;
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    let output_bytes = u64::from(hypothesis_count) * std::mem::size_of::<GpuPlaneScore>() as u64;
    if points == 0
        || hypotheses == 0
        || u64::from(point_count) * 4 > bound
        || u64::from(hypothesis_count) * 16 > bound
        || output_bytes > bound
        || limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE
        || limits.max_compute_workgroup_size_x < WORKGROUP_SIZE
        || limits.max_compute_workgroups_per_dimension == 0
        || (parallel && limits.max_compute_workgroup_storage_size < 2048)
        || limits.max_storage_buffers_per_shader_stage < 5
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < std::mem::size_of::<RansacPlaneUniform>() as u32
    {
        return Err(invalid());
    }
    let groups =
        if parallel { hypothesis_count } else { hypothesis_count.div_ceil(WORKGROUP_SIZE) }
            .min(256)
            .min(limits.max_compute_workgroups_per_dimension);
    Ok(Plan { point_count, hypothesis_count, groups, parallel, output_bytes })
}

pub(super) struct ParentPipelines {
    layout: wgpu::BindGroupLayout,
    pipelines: [OnceLock<wgpu::ComputePipeline>; 2],
}

impl ParentPipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let entries: [wgpu::BindGroupLayoutEntry; 6] =
            std::array::from_fn(|binding| wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: binding != 5 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        Self {
            layout: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ransac-plane-layout"),
                entries: &entries,
            }),
            pipelines: std::array::from_fn(|_| OnceLock::new()),
        }
    }

    fn pipeline(&self, device: &wgpu::Device, parallel: bool) -> &wgpu::ComputePipeline {
        self.pipelines[usize::from(parallel)].get_or_init(|| {
            let source = if parallel {
                include_str!("ransac_counts_parallel_reference.wgsl")
            } else {
                include_str!("../shaders/ransac_plane_serial.wgsl")
            };
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ransac-plane-shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ransac-plane-pipeline-layout"),
                bind_group_layouts: &[&self.layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ransac-plane-pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RansacPlaneUniform {
    point_count: u32,
    hypothesis_count: u32,
    distance_threshold: f32,
    groups: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RansacHypothesisPod {
    i0: u32,
    i1: u32,
    i2: u32,
    _pad: u32,
}

/// GPU score for one RANSAC plane hypothesis.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuPlaneScore {
    /// Number of inlier points for this hypothesis.
    pub inlier_count: u32,
    /// Unit plane normal `(nx, ny, nz)`.
    pub normal: [f32; 3],
    /// Plane offset term in Hessian form.
    pub d: f32,
}

/// Scores RANSAC plane hypotheses in parallel on the GPU.
///
/// Uploads XYZ and hypotheses and explicitly reads back one score per hypothesis.
/// Large inputs also partition points across workgroup lanes, reducing integer
/// counts while retaining each point's f32 distance arithmetic. Invalid indices,
/// empty point arrays and unsupported sizes/compute limits fail before GPU work.
pub(super) fn previous(
    pipelines: &ParentPipelines,
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    hypotheses: &[[u32; 3]],
    distance_threshold: f32,
) -> SpatialResult<Vec<GpuPlaneScore>> {
    if x.len() != y.len() || x.len() != z.len() {
        return Err(SpatialError::InvalidArgument("xyz arrays must have equal length".to_owned()));
    }
    if hypotheses.is_empty() {
        return Ok(Vec::new());
    }

    let plan = plan(x.len(), hypotheses.len(), &runtime.device().limits())?;
    if hypotheses.iter().flatten().any(|&index| index >= plan.point_count) {
        return Err(SpatialError::InvalidArgument(
            "RANSAC hypothesis index exceeds point array".to_owned(),
        ));
    }
    let device = runtime.device();
    let queue = runtime.queue();
    let pipeline = pipelines.pipeline(device, plan.parallel);

    let x_buffer = runtime.upload_f32_storage("ransac-plane-x", x)?;
    let y_buffer = runtime.upload_f32_storage("ransac-plane-y", y)?;
    let z_buffer = runtime.upload_f32_storage("ransac-plane-z", z)?;

    let hypothesis_pods: Vec<RansacHypothesisPod> = hypotheses
        .iter()
        .map(|indices| RansacHypothesisPod {
            i0: indices[0],
            i1: indices[1],
            i2: indices[2],
            _pad: 0,
        })
        .collect();
    let hypothesis_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ransac-plane-hypotheses"),
        contents: bytemuck::cast_slice(&hypothesis_pods),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_len = plan.output_bytes;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ransac-plane-scores"),
        size: output_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let uniform = RansacPlaneUniform {
        point_count: plan.point_count,
        hypothesis_count: plan.hypothesis_count,
        distance_threshold,
        groups: plan.groups,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ransac-plane-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ransac-plane-bind-group"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: z_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: hypothesis_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output_buffer.as_entire_binding() },
        ],
    });

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ransac-plane-staging"),
        size: output_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ransac-plane") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ransac-plane-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(plan.groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_len);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
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
    let scores: Vec<GpuPlaneScore> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    drop(bind_group);
    runtime.recycle_storage(std::mem::size_of_val(x) as u64, x_buffer);
    runtime.recycle_storage(std::mem::size_of_val(y) as u64, y_buffer);
    runtime.recycle_storage(std::mem::size_of_val(z) as u64, z_buffer);

    Ok(scores)
}
