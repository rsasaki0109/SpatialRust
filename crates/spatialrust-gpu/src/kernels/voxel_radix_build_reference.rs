#![allow(dead_code)]
// Frozen immediate-parent radix; only cache injection and module paths differ.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::runtime::WgpuRuntime;

const WORKGROUP_SIZE: u32 = 256;
const RADIX: u64 = 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RadixParams {
    point_count: u32,
    group_count: u32,
    axis: u32,
    shift: u32,
}

pub(super) fn sort_entries_radix_gpu(
    cache: &ParentRadixPipelines,
    runtime: &WgpuRuntime,
    entries: wgpu::Buffer,
    point_count: u32,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
) -> wgpu::Buffer {
    if point_count <= 1 {
        return entries;
    }

    let device = runtime.device();
    let pipelines = cache;
    let group_count = point_count.div_ceil(WORKGROUP_SIZE);
    let [histogram_pipeline, prefix_pipeline, scatter_pipeline] =
        pipelines.sort_pipelines(device, group_count);
    let mut temporaries = crate::scratch::ScratchBuffers::new(runtime);
    let scratch = temporaries.acquire("voxel-radix-scratch", entries.size());
    let histograms =
        temporaries.acquire("voxel-radix-histograms", u64::from(group_count) * RADIX * 4);
    let offsets = temporaries.acquire("voxel-radix-offsets", histograms.size());
    let ranks = temporaries.acquire("voxel-radix-ranks", u64::from(point_count) * 4);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-radix-encoder"),
    });
    let mut input = &entries;
    let mut output = &scratch;

    // Entry construction supplies increasing point indices. Stable passes on
    // z, y, x preserve that order for equal keys without sorting a fourth word.
    for axis in (0..3).rev() {
        for shift in (0..32).step_by(4) {
            let params = RadixParams { point_count, group_count, axis, shift };
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel-radix-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("voxel-radix-bind-group"),
                layout: &pipelines.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: histograms.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: offsets.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: ranks.as_entire_binding() },
                ],
            });
            // Separate dispatches provide device-wide producer/consumer
            // ordering. All 24 digits are submitted together without readback.
            for (pipeline, groups, stage) in [
                (histogram_pipeline, group_count, "voxel-radix-histogram"),
                (prefix_pipeline, RADIX as u32, "voxel-radix-prefix"),
                (scatter_pipeline, group_count, "voxel-radix-scatter"),
            ] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("voxel-radix-pass"),
                    timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, stage),
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            std::mem::swap(&mut input, &mut output);
        }
    }
    temporaries.submit_and_recycle(encoder);

    // Twenty-four ping-pong passes finish in the original allocation.
    entries
}

pub struct ParentRadixPipelines {
    /// Bind group layout shared by radix dispatches.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Counts digits and computes stable workgroup-local ranks.
    pub histogram: wgpu::ComputePipeline,
    /// Computes exclusive digit/workgroup offsets.
    pub prefix: wgpu::ComputePipeline,
    /// Scatters entries to their stable sorted positions.
    pub scatter: wgpu::ComputePipeline,
    cached_prefix: std::sync::OnceLock<[wgpu::ComputePipeline; 2]>,
    _pipeline_layout: wgpu::PipelineLayout,
    _shader: wgpu::ShaderModule,
}
impl ParentRadixPipelines {
    // At most eight group-count items per scan lane retain the legacy kernels.
    // The extra histogram barrier pays off only for larger prefix chunks.
    pub(crate) fn sort_pipelines(
        &self,
        device: &wgpu::Device,
        group_count: u32,
    ) -> [&wgpu::ComputePipeline; 3] {
        if group_count <= 2048 {
            return [&self.histogram, &self.prefix, &self.scatter];
        }
        let cached = self.cached_prefix.get_or_init(|| {
            let constants = std::collections::HashMap::from([("cached_prefix".to_owned(), 1.0)]);
            ["histogram", "prefix"].map(|entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("voxel-radix-cached-prefix"),
                    layout: Some(&self._pipeline_layout),
                    module: &self._shader,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    cache: None,
                })
            })
        });
        [&cached[0], &cached[1], &self.scatter]
    }

    pub(super) fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-radix-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
        });
        let shader = load_shader(
            device,
            "voxel-radix-shader",
            include_str!("voxel_radix_build_reference.wgsl"),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-radix-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        Self {
            histogram: build_compute_pipeline(
                device,
                &pipeline_layout,
                &shader,
                "voxel-radix-histogram",
                "histogram",
            ),
            prefix: build_compute_pipeline(
                device,
                &pipeline_layout,
                &shader,
                "voxel-radix-prefix",
                "prefix",
            ),
            scatter: build_compute_pipeline(
                device,
                &pipeline_layout,
                &shader,
                "voxel-radix-scatter",
                "scatter",
            ),
            cached_prefix: std::sync::OnceLock::new(),
            bind_group_layout,
            _pipeline_layout: pipeline_layout,
            _shader: shader,
        }
    }
}
pub struct ParentBuildPipelines {
    /// Bind group layout for sort-entry construction.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Sort-entry construction pipeline.
    pub pipeline: wgpu::ComputePipeline,
    _pipeline_layout: wgpu::PipelineLayout,
    _shader: wgpu::ShaderModule,
}
impl ParentBuildPipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-sort-build-layout"),
            entries: &[uniform_entry(0), storage_entry(1, true), storage_entry(2, false)],
        });
        let shader = load_shader(
            device,
            "voxel-sort-build-shader",
            include_str!("voxel_radix_build_entries_reference.wgsl"),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-sort-build-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = build_compute_pipeline(
            device,
            &pipeline_layout,
            &shader,
            "voxel-sort-build-pipeline",
            "build_sort_entries",
        );
        Self { bind_group_layout, pipeline, _pipeline_layout: pipeline_layout, _shader: shader }
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
