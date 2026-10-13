//! Actual frozen Canny shaders and their original five-buffer execution boundary.
use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FrozenLayout {
    Unpacked,
    PackedGradients,
    PackedHorizontal,
}

pub(super) fn previous_state_flags_pipelines(runtime: &WgpuRuntime) -> CannyPipelines {
    source_pipelines(runtime, include_str!("../../tests/fixtures/canny_state_flags_previous.wgsl"))
}

pub(super) fn previous_state_flags_canny(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    o: GpuCannyOptions,
    cache: &CannyPipelines,
) -> GpuImage {
    frozen_canny(runtime, source, o, cache, FrozenLayout::PackedHorizontal)
}

pub(super) fn bind_legacy(
    runtime: &WgpuRuntime,
    cache: &CannyPipelines,
    work: &Work,
    strong: &wgpu::Buffer,
    uniform: &wgpu::Buffer,
    source: &wgpu::TextureView,
    output: &wgpu::TextureView,
) -> wgpu::BindGroup {
    let buffers = [&work.gradients, &work.horizontal, &work.states, &work.parent, strong];
    let mut entries = vec![
        wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(source) },
    ];
    entries.extend(buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry {
        binding: i as u32 + 2,
        resource: b.as_entire_binding(),
    }));
    entries.push(wgpu::BindGroupEntry {
        binding: 7,
        resource: wgpu::BindingResource::TextureView(output),
    });
    runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &cache.layout,
        entries: &entries,
    })
}

// Frozen PR #180 shader from d5f73a5b263bba9262e2bb3526943168088f02ab.
pub(super) fn previous_pipelines(runtime: &WgpuRuntime) -> CannyPipelines {
    source_pipelines(runtime, include_str!("../../tests/fixtures/canny_unpacked_previous.wgsl"))
}

// Frozen PR #181: compact gradients but the original eight-byte horizontal rows.
pub(super) fn previous_horizontal_pipelines(runtime: &WgpuRuntime) -> CannyPipelines {
    source_pipelines(runtime, include_str!("../../tests/fixtures/canny_horizontal_previous.wgsl"))
}

pub(super) fn source_pipelines(runtime: &WgpuRuntime, shader: &str) -> CannyPipelines {
    let binding_layout = binding_layout(runtime.device(), true);
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("canny-frozen-shader"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let layout = runtime.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&binding_layout],
        push_constant_ranges: &[],
    });
    let pipelines = ENTRIES.map(|entry| {
        runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    });
    CannyPipelines { layout: binding_layout, pipelines }
}

// Test-only previous completed call. The control preparation, buffer/texture
// acquisition, submission, metadata and receipts match the production boundary.
pub(super) fn previous_canny(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    o: GpuCannyOptions,
    cache: &CannyPipelines,
) -> GpuImage {
    frozen_canny(runtime, source, o, cache, FrozenLayout::Unpacked)
}

pub(super) fn previous_horizontal_canny(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    o: GpuCannyOptions,
    cache: &CannyPipelines,
) -> GpuImage {
    frozen_canny(runtime, source, o, cache, FrozenLayout::PackedGradients)
}

pub(super) fn frozen_plan(p: &mut Plan, layout: FrozenLayout) {
    if layout == FrozenLayout::Unpacked {
        p.gradient_bytes *= 2;
    }
    if p.params.aperture != 3 && layout != FrozenLayout::PackedHorizontal {
        p.horizontal_bytes = u64::from(p.params.pixels) * 8;
    }
}

fn frozen_canny(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    o: GpuCannyOptions,
    cache: &CannyPipelines,
    layout: FrozenLayout,
) -> GpuImage {
    source.validate_runtime(runtime).unwrap();
    let mut p =
        plan(source.width(), source.height(), source.channels(), o, &runtime.device().limits())
            .unwrap();
    frozen_plan(&mut p, layout);
    assert!(
        p.gradient_bytes
            <= runtime
                .device()
                .limits()
                .max_buffer_size
                .min(u64::from(runtime.device().limits().max_storage_buffer_binding_size))
    );
    let mut scratch = ScratchBuffers::new(runtime);
    let work = Work::acquire(&mut scratch, &p);
    let strong = scratch.acquire("canny-strong-roots", p.pixel_bytes);
    let output = create_texture(runtime, source.width(), source.height(), "canny-previous-output");
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("canny-previous-params"),
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let group = bind_legacy(
        runtime,
        cache,
        &work,
        &strong,
        &uniform,
        &source.view(),
        &output.create_view(&Default::default()),
    );
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    let indices = pass_indices(o.aperture_size);
    encode(cache, &group, p.params.groups, indices, &mut encoder, None);
    scratch.submit_and_recycle(encoder);
    let mut receipt = source.receipt().clone();
    for &index in indices {
        let bytes = match index {
            0 | 2 => p.gradient_bytes,
            1 => p.horizontal_bytes,
            3 => 3 * p.pixel_bytes,
            5 => 2 * p.pixel_bytes,
            _ => p.pixel_bytes,
        };
        receipt.record_gpu_to_gpu(bytes, STAGES[index]);
    }
    GpuImage::from_parts(
        runtime,
        source.width(),
        source.height(),
        1,
        output,
        source.metadata(),
        receipt,
    )
    .unwrap()
}
