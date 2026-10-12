use super::*;

pub(super) fn forced_histogram(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    channel: u32,
    mut profiler: Option<&mut GpuStageProfiler>,
    pipelines: &HistogramPipelines,
) -> SpatialResult<GpuImageHistogram> {
    source.validate_runtime(runtime)?;
    if channel >= source.channels() {
        return Err(invalid("GPU histogram channel is outside the logical image channels"));
    }
    let plan = dispatch_plan(source.width(), source.height(), &runtime.device().limits())?;
    let params =
        Params { width: source.width(), height: source.height(), channel, groups: plan.groups };
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-image-histogram-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut scratch = ScratchBuffers::new(runtime);
    let partials = scratch.acquire("gpu-image-histogram-partials", plan.partial_bytes);
    let output = runtime.buffer_pool().acquire_storage(
        runtime,
        "gpu-image-histogram-output",
        HISTOGRAM_BYTES,
    );
    let source_view = source.view();
    let bind_group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-image-histogram-bind-group"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&source_view),
            },
            wgpu::BindGroupEntry { binding: 2, resource: partials.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-image-histogram-encoder"),
    });
    for (pipeline, stage, groups) in [
        (&pipelines.partials, "histogram_u8_partials_gpu", plan.groups),
        (&pipelines.reduce, "histogram_u8_reduce_gpu", 1),
    ] {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(stage),
            timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, stage),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    scratch.submit_and_recycle(encoder);
    let mut receipt = source.receipt().clone();
    receipt.record_gpu_to_gpu(plan.partial_bytes, "histogram_u8_partials_gpu");
    receipt.record_gpu_to_gpu(HISTOGRAM_BYTES, "histogram_u8_reduce_gpu");
    Ok(GpuImageHistogram {
        buffer: output,
        device_key: runtime_device_key(runtime),
        pixel_count: plan.pixels,
        channel,
        receipt,
    })
}

pub(super) fn pipelines(device: &wgpu::Device, source: &str) -> HistogramPipelines {
    let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gpu-image-histogram-layout"),
        entries: &[
            buffer(0, wgpu::BufferBindingType::Uniform),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            buffer(2, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer(3, wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gpu-image-histogram-shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gpu-image-histogram-pipeline-layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = |entry| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };
    HistogramPipelines {
        partials: pipeline("histogram_partials"),
        reduce: pipeline("histogram_reduce"),
        replicated: OnceLock::new(),
        layout,
    }
}
