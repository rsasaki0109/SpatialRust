use super::*;

pub(super) fn frozen_pipeline(runtime: &WgpuRuntime) -> BilateralPipeline {
    let shader = SHADER.replace(
        include_str!("bilateral_round.wgsl"),
        include_str!("bilateral_round_reference.wgsl"),
    );
    BilateralPipeline::with_shader(runtime.device(), &shader)
}

// PR 194 submission body with only the pipeline selected by this fixture.
pub(super) fn filter_frozen(
    plan: &GpuBilateralFilter,
    runtime: &WgpuRuntime,
    source: &GpuImage,
    cache: &BilateralPipeline,
) -> SpatialResult<GpuImage> {
    if !Arc::ptr_eq(&plan.device_key, &runtime_device_key(runtime))
        || source.channels() != plan.channels
    {
        return Err(invalid());
    }
    source.validate_runtime(runtime)?;
    let limits = runtime.device().limits();
    capabilities(plan.receipt.host_to_device_bytes(), &limits)?;
    let bytes = dimensions(source.width(), source.height(), plan.options.diameter / 2, &limits)?;
    let output = create_texture(runtime, source.width(), source.height(), "bilateral-output");
    let params = Params {
        width: source.width(),
        height: source.height(),
        channels: plan.channels,
        taps: plan.taps,
        border: u32::from(plan.options.border == GpuImageBorder::ConstantZero),
        padding: [0; 3],
    };
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bilateral-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bilateral-group"),
        layout: &cache.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&source.view()),
            },
            wgpu::BindGroupEntry { binding: 2, resource: plan.coefficients.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(
                    &output.create_view(&Default::default()),
                ),
            },
        ],
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bilateral_filter_gpu"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&cache.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(source.width().div_ceil(16), source.height().div_ceil(16), 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
    let mut receipt = source.receipt().clone();
    receipt.record_gpu_to_gpu(bytes, "bilateral_filter_gpu");
    GpuImage::from_parts(
        runtime,
        source.width(),
        source.height(),
        source.channels(),
        output,
        source.metadata(),
        receipt,
    )
}

#[path = "bilateral_candidate_benchmark.rs"]
mod benchmark;
#[path = "bilateral_candidate_proof.rs"]
mod proof;
