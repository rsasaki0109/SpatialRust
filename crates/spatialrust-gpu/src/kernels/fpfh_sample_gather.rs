//! Integer packing from resident original-order positions and uploaded normals.
use crate::{GpuQueryBatch, WgpuRuntime};
use wgpu::util::DeviceExt;

pub(crate) struct FpfhSamplePipeline {
    pipeline: wgpu::ComputePipeline,
}

impl FpfhSamplePipeline {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fpfh-sample-gather-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/fpfh_sample_gather.wgsl").into(),
            ),
        });
        Self {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fpfh-sample-gather-pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("gather_samples"),
                compilation_options: Default::default(),
                cache: None,
            }),
        }
    }
}

// The caller validates nonempty matching finite normals and FPFH count/device
// limits before creating the resident queries or calling this private helper.
pub(super) fn upload(
    runtime: &WgpuRuntime,
    queries: &GpuQueryBatch,
    normals: &[[f32; 3]],
) -> wgpu::Buffer {
    let count = queries.query_count();
    let normals = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fpfh-gather-normals"),
        contents: bytemuck::cast_slice(normals),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let samples = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("fpfh-samples"),
        size: u64::from(count) * 32,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let params = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fpfh-gather-params"),
        contents: bytemuck::cast_slice(&[count, 0, 0, 0]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let cache =
        runtime.fpfh_sample_pipeline.get_or_init(|| FpfhSamplePipeline::new(runtime.device()));
    let buffers = [&params, queries.buffer().unwrap(), &normals, &samples];
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fpfh-sample-gather-bindings"),
        layout: &cache.pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fpfh-sample-gather"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&cache.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
    samples
}
