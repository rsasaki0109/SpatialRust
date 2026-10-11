//! Device-resident resize, edge, and morphology texture kernels.

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

use crate::image::gpu_image::{create_texture, GpuImage, GpuImageReceipt};
use crate::WgpuRuntime;

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

/// Single-channel morphology operation executed on a GPU texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuMorphology {
    /// Replaces each pixel with the neighborhood minimum.
    Erode,
    /// Replaces each pixel with the neighborhood maximum.
    Dilate,
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

pub(crate) struct SpatialPipelines {
    layout: wgpu::BindGroupLayout,
    resize: wgpu::ComputePipeline,
    sobel: wgpu::ComputePipeline,
    morphology: wgpu::ComputePipeline,
    separable_morphology: OnceLock<[wgpu::ComputePipeline; 2]>,
}

impl SpatialPipelines {
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

pub(crate) fn create_spatial_pipelines(device: &wgpu::Device) -> SpatialPipelines {
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
    SpatialPipelines {
        layout,
        resize: create("resize_nearest"),
        sobel: create("sobel"),
        morphology: create("morphology"),
        separable_morphology: OnceLock::new(),
    }
}

/// Resizes a device image with nearest-neighbor sampling without host transfer.
pub fn resize_nearest_gpu(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    width: u32,
    height: u32,
) -> SpatialResult<GpuImage> {
    resize_nearest_gpu_with_profiler(runtime, source, width, height, None)
}

pub(crate) fn resize_nearest_gpu_with_profiler(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    width: u32,
    height: u32,
    profiler: Option<&mut crate::GpuStageProfiler>,
) -> SpatialResult<GpuImage> {
    if width == 0 || height == 0 {
        return Err(SpatialError::InvalidArgument(
            "GPU resize output dimensions must be positive".to_owned(),
        ));
    }
    dispatch(
        runtime,
        source,
        width,
        height,
        1,
        1,
        0,
        &runtime
            .image_spatial_pipelines
            .get_or_init(|| create_spatial_pipelines(runtime.device()))
            .resize,
        "resize_nearest_gpu",
        source.channels(),
        profiler,
    )
}

/// Computes clamped L1 Sobel magnitude on a single-channel device image.
pub fn sobel_gpu(runtime: &WgpuRuntime, source: &GpuImage) -> SpatialResult<GpuImage> {
    sobel_gpu_with_profiler(runtime, source, None)
}

pub(crate) fn sobel_gpu_with_profiler(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    profiler: Option<&mut crate::GpuStageProfiler>,
) -> SpatialResult<GpuImage> {
    require_gray(source, "sobel_gpu")?;
    dispatch(
        runtime,
        source,
        source.width(),
        source.height(),
        3,
        3,
        0,
        &runtime
            .image_spatial_pipelines
            .get_or_init(|| create_spatial_pipelines(runtime.device()))
            .sobel,
        "sobel_gpu",
        1,
        profiler,
    )
}

/// Applies rectangular erosion or dilation without leaving device memory.
///
/// Odd kernels in `1..=31` use replicate borders. Boxes with area at least 81
/// and at least 1,048,576 pixel/sample visits use two exact extrema passes and
/// four scratch bytes per pixel; smaller workloads use the original single pass.
pub fn morphology_gpu(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    kernel_width: u32,
    kernel_height: u32,
    operation: GpuMorphology,
) -> SpatialResult<GpuImage> {
    morphology_gpu_with_profiler(runtime, source, kernel_width, kernel_height, operation, None)
}

pub(crate) fn morphology_gpu_with_profiler(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    kernel_width: u32,
    kernel_height: u32,
    operation: GpuMorphology,
    profiler: Option<&mut crate::GpuStageProfiler>,
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
    let cache =
        runtime.image_spatial_pipelines.get_or_init(|| create_spatial_pipelines(runtime.device()));
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

#[allow(clippy::too_many_arguments)]
fn encode_morphology(
    runtime: &WgpuRuntime,
    cache: &SpatialPipelines,
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
fn dispatch(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    output_width: u32,
    output_height: u32,
    kernel_width: u32,
    kernel_height: u32,
    operation: u32,
    pipeline: &wgpu::ComputePipeline,
    stage: &'static str,
    output_channels: u32,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
) -> SpatialResult<GpuImage> {
    source.validate_runtime(runtime)?;
    let output = create_texture(runtime, output_width, output_height, stage);
    let params = SpatialParams {
        source_width: source.width(),
        source_height: source.height(),
        output_width,
        output_height,
        kernel_width,
        kernel_height,
        operation,
        _pad: 0,
    };
    let uniform = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-image-spatial-params"),
        size: std::mem::size_of::<SpatialParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    runtime.queue().write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
    let source_view = source.view();
    let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-image-spatial-bg"),
        layout: &runtime
            .image_spatial_pipelines
            .get_or_init(|| create_spatial_pipelines(runtime.device()))
            .layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&source_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&output_view),
            },
        ],
    });
    let mut encoder = runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-image-spatial-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(stage),
            timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, stage),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            output_width.div_ceil(WORKGROUP_X),
            output_height.div_ceil(WORKGROUP_Y),
            1,
        );
    }
    runtime.queue().submit(Some(encoder.finish()));
    let output_bytes = u64::from(output_width) * u64::from(output_height) * 4;
    let mut receipt = GpuImageReceipt::default();
    receipt.merge_from(source.receipt());
    receipt.record_gpu_to_gpu(output_bytes, stage);
    GpuImage::from_parts(
        runtime,
        output_width,
        output_height,
        output_channels,
        output,
        source.metadata(),
        receipt,
    )
}

fn require_gray(source: &GpuImage, operation: &str) -> SpatialResult<()> {
    if source.channels() != 1 {
        return Err(SpatialError::InvalidArgument(format!(
            "{operation} requires a single-channel GpuImage"
        )));
    }
    Ok(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "morphology_tests.rs"]
mod morphology_tests;
