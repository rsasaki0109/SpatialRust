//! Device-resident resize, edge, and morphology texture kernels.

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};

use crate::image::gpu_image::{create_texture, GpuImage, GpuImageReceipt};
use crate::WgpuRuntime;

const WORKGROUP_X: u32 = 16;
const WORKGROUP_Y: u32 = 16;

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
    }
}

/// Resizes a device image with nearest-neighbor sampling without host transfer.
pub fn resize_nearest_gpu(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    width: u32,
    height: u32,
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
    )
}

/// Computes clamped L1 Sobel magnitude on a single-channel device image.
pub fn sobel_gpu(runtime: &WgpuRuntime, source: &GpuImage) -> SpatialResult<GpuImage> {
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
    )
}

/// Applies rectangular erosion or dilation without leaving device memory.
pub fn morphology_gpu(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    kernel_width: u32,
    kernel_height: u32,
    operation: GpuMorphology,
) -> SpatialResult<GpuImage> {
    require_gray(source, "morphology_gpu")?;
    if kernel_width == 0
        || kernel_height == 0
        || kernel_width % 2 == 0
        || kernel_height % 2 == 0
        || kernel_width > 31
        || kernel_height > 31
    {
        return Err(SpatialError::InvalidArgument(
            "GPU morphology kernels must be odd and in 1..=31".to_owned(),
        ));
    }
    dispatch(
        runtime,
        source,
        source.width(),
        source.height(),
        kernel_width,
        kernel_height,
        match operation {
            GpuMorphology::Erode => 0,
            GpuMorphology::Dilate => 1,
        },
        &runtime
            .image_spatial_pipelines
            .get_or_init(|| create_spatial_pipelines(runtime.device()))
            .morphology,
        "morphology_gpu",
        1,
    )
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
            timestamp_writes: None,
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
