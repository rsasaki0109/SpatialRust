//! Hierarchical work-efficient inclusive scan for voxel boundary flags.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{GpuStageProfiler, WgpuRuntime};

const BLOCK_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScanParams {
    element_count: u32,
    _pad: [u32; 3],
}

pub(crate) struct PrefixScanPipelines {
    layout: wgpu::BindGroupLayout,
    blocks: wgpu::ComputePipeline,
    offsets: wgpu::ComputePipeline,
}

impl PrefixScanPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel-inclusive-scan-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/inclusive_scan.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-inclusive-scan-layout"),
            entries: &[0, 1, 2, 3].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: binding == 1 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-inclusive-scan-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let build = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        Self { blocks: build("scan_blocks"), offsets: build("add_block_offsets"), layout }
    }
}

struct ScanLevel {
    prefix: wgpu::Buffer,
    sums: wgpu::Buffer,
    count: u32,
}

pub(crate) fn scan_pass_count(mut count: u32) -> u32 {
    if count == 0 {
        return 0;
    }
    let mut levels = 1;
    while count > BLOCK_SIZE {
        count = count.div_ceil(BLOCK_SIZE);
        levels += 1;
    }
    levels * 2 - 1
}

pub(super) fn encode_inclusive_scan(
    runtime: &WgpuRuntime,
    encoder: &mut wgpu::CommandEncoder,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    count: u32,
    profiler: &mut Option<&mut GpuStageProfiler>,
    temporaries: &mut crate::scratch::ScratchBuffers<'_>,
) {
    if count == 0 {
        return;
    }
    let pipelines =
        runtime.voxel_scan_pipelines.get_or_init(|| PrefixScanPipelines::new(runtime.device()));
    let mut levels = Vec::with_capacity(scan_pass_count(count).div_ceil(2) as usize);
    let mut source = input.clone();
    let mut prefix = output.clone();
    let mut current_count = count;
    loop {
        let groups = current_count.div_ceil(BLOCK_SIZE);
        let sums = temporaries.acquire("voxel-scan-hierarchy", u64::from(groups) * 4);
        dispatch(
            runtime,
            encoder,
            pipelines,
            &pipelines.blocks,
            &source,
            &prefix,
            &sums,
            current_count,
            profiler,
            "voxel-compact-scan-blocks",
        );
        levels.push(ScanLevel { prefix, sums, count: current_count });
        if groups == 1 {
            break;
        }
        source = levels.last().expect("scan level").sums.clone();
        prefix = temporaries.acquire("voxel-scan-hierarchy", u64::from(groups) * 4);
        current_count = groups;
    }
    for lower in (0..levels.len() - 1).rev() {
        dispatch(
            runtime,
            encoder,
            pipelines,
            &pipelines.offsets,
            &levels[lower + 1].prefix,
            &levels[lower].prefix,
            &levels[lower].sums,
            levels[lower].count,
            profiler,
            "voxel-compact-scan-offsets",
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    runtime: &WgpuRuntime,
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &PrefixScanPipelines,
    pipeline: &wgpu::ComputePipeline,
    source: &wgpu::Buffer,
    prefix: &wgpu::Buffer,
    sums: &wgpu::Buffer,
    count: u32,
    profiler: &mut Option<&mut GpuStageProfiler>,
    stage: &'static str,
) {
    let params = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-scan-params"),
        contents: bytemuck::bytes_of(&ScanParams { element_count: count, _pad: [0; 3] }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(stage),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: source.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: prefix.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: sums.as_entire_binding() },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(stage),
        timestamp_writes: crate::profiling::timestamp_writes(profiler, stage),
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(count.div_ceil(BLOCK_SIZE), 1, 1);
}

#[cfg(test)]
mod tests {
    #[test]
    fn pass_count_covers_recursive_hierarchy() {
        for (count, passes) in [
            (0, 0),
            (1, 1),
            (256, 1),
            (257, 3),
            (65_536, 3),
            (65_537, 5),
            (16_777_216, 5),
            (u32::MAX, 7),
        ] {
            assert_eq!(super::scan_pass_count(count), passes);
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_scan_four_tests.rs"]
mod four;
