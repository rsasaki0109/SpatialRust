// Complete morphology reference frozen from acaaa71650374480f6cd1a843616e5d27751b067.
use super::*;
const WORKGROUP_X: u32 = 16;
const WORKGROUP_Y: u32 = 16;
const SEPARABLE_MIN_AREA: u64 = 81;
const SEPARABLE_MIN_SAMPLES: u64 = 1_048_576;
pub(crate) fn morphology_pass_count(width: u32, height: u32, kw: u32, kh: u32) -> u32 {
    let area = u64::from(kw) * u64::from(kh);
    let count = u64::from(width) * u64::from(height);
    // Also called with unvalidated profiler options: avoid overflow and division by zero.
    1 + u32::from(area >= SEPARABLE_MIN_AREA && count >= SEPARABLE_MIN_SAMPLES.div_ceil(area))
}
struct MorphologyPlan {
    params: SpatialParams,
    separable: bool,
    bytes: u64,
}
fn morphology_plan(
    width: u32,
    height: u32,
    kw: u32,
    kh: u32,
    operation: GpuMorphology,
    limits: &wgpu::Limits,
) -> SpatialResult<MorphologyPlan> {
    let invalid = || {
        SpatialError::InvalidArgument(
        "GPU morphology requires odd kernels in 1..=31 and supported texture/compute/coordinate limits".to_owned())
    };
    let count = u64::from(width) * u64::from(height);
    let bytes = count.checked_mul(4).ok_or_else(invalid)?;
    if width == 0
        || height == 0
        || kw == 0
        || kh == 0
        || kw % 2 == 0
        || kh % 2 == 0
        || kw > 31
        || kh > 31
        || width > limits.max_texture_dimension_2d
        || height > limits.max_texture_dimension_2d
        || width > i32::MAX as u32
        || height > i32::MAX as u32
        || u64::from(width - 1) + u64::from(kw / 2) > i32::MAX as u64
        || u64::from(height - 1) + u64::from(kh / 2) > i32::MAX as u64
        || limits.max_compute_invocations_per_workgroup < WORKGROUP_X * WORKGROUP_Y
        || limits.max_compute_workgroup_size_x < WORKGROUP_X
        || limits.max_compute_workgroup_size_y < WORKGROUP_Y
        || limits.max_compute_workgroup_size_z == 0
        || width.div_ceil(WORKGROUP_X) > limits.max_compute_workgroups_per_dimension
        || height.div_ceil(WORKGROUP_Y) > limits.max_compute_workgroups_per_dimension
        || limits.max_sampled_textures_per_shader_stage == 0
        || limits.max_storage_textures_per_shader_stage == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < std::mem::size_of::<SpatialParams>() as u32
        || limits.max_buffer_size < std::mem::size_of::<SpatialParams>() as u64
        || limits.max_bind_groups == 0
        || limits.max_bindings_per_bind_group < 3
    {
        return Err(invalid());
    }
    Ok(MorphologyPlan {
        params: SpatialParams {
            source_width: width,
            source_height: height,
            output_width: width,
            output_height: height,
            kernel_width: kw,
            kernel_height: kh,
            operation: u32::from(operation == GpuMorphology::Dilate),
            _pad: 0,
        },
        separable: morphology_pass_count(width, height, kw, kh) == 2,
        bytes,
    })
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpatialParams {
    source_width: u32,
    source_height: u32,
    output_width: u32,
    output_height: u32,
    kernel_width: u32,
    kernel_height: u32,
    operation: u32,
    _pad: u32,
}
const SPATIAL_WGSL: &str = r#"
struct Params {
    source_width: u32,
    source_height: u32,
    output_width: u32,
    output_height: u32,
    kernel_width: u32,
    kernel_height: u32,
    operation: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input_px: texture_2d<u32>;
@group(0) @binding(2) var output_px: texture_storage_2d<rgba8uint, write>;

fn in_output(gid: vec3<u32>) -> bool {
    return gid.x < params.output_width && gid.y < params.output_height;
}

fn gray_at(x: i32, y: i32) -> i32 {
    let sx = clamp(x, 0, i32(params.source_width) - 1);
    let sy = clamp(y, 0, i32(params.source_height) - 1);
    return i32(textureLoad(input_px, vec2<i32>(sx, sy), 0).r);
}

@compute @workgroup_size(16, 16)
fn resize_nearest(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_output(gid)) { return; }
    let sx = min((gid.x * params.source_width) / params.output_width, params.source_width - 1u);
    let sy = min((gid.y * params.source_height) / params.output_height, params.source_height - 1u);
    textureStore(output_px, vec2<i32>(gid.xy), textureLoad(input_px, vec2<i32>(i32(sx), i32(sy)), 0));
}

@compute @workgroup_size(16, 16)
fn sobel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_output(gid)) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let gx = -gray_at(x - 1, y - 1) + gray_at(x + 1, y - 1)
        - 2 * gray_at(x - 1, y) + 2 * gray_at(x + 1, y)
        - gray_at(x - 1, y + 1) + gray_at(x + 1, y + 1);
    let gy = -gray_at(x - 1, y - 1) - 2 * gray_at(x, y - 1) - gray_at(x + 1, y - 1)
        + gray_at(x - 1, y + 1) + 2 * gray_at(x, y + 1) + gray_at(x + 1, y + 1);
    let magnitude = u32(clamp(abs(gx) + abs(gy), 0, 255));
    textureStore(output_px, vec2<i32>(x, y), vec4<u32>(magnitude, 0u, 0u, 0u));
}

@compute @workgroup_size(16, 16)
fn morphology(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_output(gid)) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let rx = i32(params.kernel_width / 2u);
    let ry = i32(params.kernel_height / 2u);
    var value = select(255, 0, params.operation == 1u);
    for (var dy = -ry; dy <= ry; dy = dy + 1) {
        for (var dx = -rx; dx <= rx; dx = dx + 1) {
            let sample = gray_at(x + dx, y + dy);
            if (params.operation == 0u) { value = min(value, sample); }
            else { value = max(value, sample); }
        }
    }
    textureStore(output_px, vec2<i32>(x, y), vec4<u32>(u32(value), 0u, 0u, 0u));
}
"#;

#[allow(dead_code)]
pub(super) struct FrozenSpatialPipelines {
    layout: wgpu::BindGroupLayout,
    resize: wgpu::ComputePipeline,
    sobel: wgpu::ComputePipeline,
    morphology: wgpu::ComputePipeline,
    separable_morphology: OnceLock<[wgpu::ComputePipeline; 2]>,
}
impl FrozenSpatialPipelines {
    fn separable_morphology(&self, device: &wgpu::Device) -> &[wgpu::ComputePipeline; 2] {
        self.separable_morphology.get_or_init(|| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpu-morphology-separable-shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("morphology_separable.wgsl").into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu-morphology-separable-pl"),
                bind_group_layouts: &[&self.layout],
                push_constant_ranges: &[],
            });
            ["horizontal", "vertical"].map(|entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
        })
    }
}
pub(super) fn create_spatial_pipelines(device: &wgpu::Device) -> FrozenSpatialPipelines {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gpu-image-spatial-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gpu-image-spatial-shader"),
        source: wgpu::ShaderSource::Wgsl(SPATIAL_WGSL.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gpu-image-spatial-pl"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let create = |entry_point| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };
    FrozenSpatialPipelines {
        layout,
        resize: create("resize_nearest"),
        sobel: create("sobel"),
        morphology: create("morphology"),
        separable_morphology: OnceLock::new(),
    }
}
#[allow(clippy::too_many_arguments)]
fn encode_morphology(
    runtime: &WgpuRuntime,
    cache: &FrozenSpatialPipelines,
    p: &MorphologyPlan,
    source: &wgpu::TextureView,
    output: &wgpu::TextureView,
    intermediate: Option<&wgpu::TextureView>,
    uniform: &wgpu::Buffer,
    encoder: &mut wgpu::CommandEncoder,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
) {
    let pipelines = p.separable.then(|| cache.separable_morphology(runtime.device()));
    for index in 0..if p.separable { 2 } else { 1 } {
        let pipeline = pipelines.map_or(&cache.morphology, |pipelines| &pipelines[index]);
        let input = if p.separable && index == 1 { intermediate.unwrap() } else { source };
        let target = if p.separable && index == 0 { intermediate.unwrap() } else { output };
        let stage =
            if p.separable && index == 0 { "morphology_horizontal_gpu" } else { "morphology_gpu" };
        let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(stage),
            layout: &cache.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(target),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(stage),
            timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, stage),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(
            p.params.output_width.div_ceil(WORKGROUP_X),
            p.params.output_height.div_ceil(WORKGROUP_Y),
            1,
        );
    }
}
#[allow(clippy::too_many_arguments)]
pub(super) fn baseline_morphology(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    kernel_width: u32,
    kernel_height: u32,
    operation: GpuMorphology,
    profiler: Option<&mut crate::GpuStageProfiler>,
    cache: &FrozenSpatialPipelines,
) -> SpatialResult<GpuImage> {
    require_gray(source, "morphology_gpu")?;
    source.validate_runtime(runtime)?;
    let p = morphology_plan(
        source.width(),
        source.height(),
        kernel_width,
        kernel_height,
        operation,
        &runtime.device().limits(),
    )?;
    let output = create_texture(runtime, source.width(), source.height(), "gpu-morphology-out");
    let intermediate = p.separable.then(|| {
        create_texture(runtime, source.width(), source.height(), "gpu-morphology-horizontal")
    });
    let intermediate_view = intermediate.as_ref().map(|t| t.create_view(&Default::default()));
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-morphology-params"),
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encode_morphology(
        runtime,
        cache,
        &p,
        &source.view(),
        &output.create_view(&Default::default()),
        intermediate_view.as_ref(),
        &uniform,
        &mut encoder,
        profiler,
    );
    runtime.queue().submit(Some(encoder.finish()));
    // Only submitted scratch is returned. Queue ordering protects later reuse;
    // an abandoned encoder drops its texture instead of publishing it to the pool.
    if let Some(texture) = intermediate {
        runtime.recycle_image_texture(source.width(), source.height(), texture);
    }
    let mut receipt = source.receipt().clone();
    if p.separable {
        receipt.record_gpu_to_gpu(p.bytes, "morphology_horizontal_gpu");
    }
    receipt.record_gpu_to_gpu(p.bytes, "morphology_gpu");
    GpuImage::from_parts(
        runtime,
        source.width(),
        source.height(),
        1,
        output,
        source.metadata(),
        receipt,
    )
}
pub(super) fn baseline_plan(
    w: u32,
    h: u32,
    kw: u32,
    kh: u32,
    op: GpuMorphology,
    l: &wgpu::Limits,
) -> SpatialResult<(bool, u64, Vec<u8>)> {
    morphology_plan(w, h, kw, kh, op, l)
        .map(|p| (p.separable, p.bytes, bytemuck::bytes_of(&p.params).to_vec()))
}
