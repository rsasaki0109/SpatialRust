//! Gray box blur on the GPU.

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use wgpu::util::DeviceExt;

use crate::image::gpu_image::{create_texture, GpuImage, GpuImageReceipt};
use crate::WgpuRuntime;

const WORKGROUP: u32 = 256;

/// Border extrapolation for GPU image filters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuImageBorder {
    /// Clamp coordinates to the nearest valid edge pixel (replicate).
    Replicate,
    /// Treat out-of-bounds samples as zero.
    ConstantZero,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurParams {
    width: u32,
    height: u32,
    kernel_width: u32,
    kernel_height: u32,
    border_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const BLUR_WGSL: &str = r#"
struct Params {
    width: u32,
    height: u32,
    kernel_width: u32,
    kernel_height: u32,
    border_mode: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input_px: texture_2d<u32>;
@group(0) @binding(2) var output_px: texture_storage_2d<rgba8uint, write>;

fn sample_gray(x: i32, y: i32) -> u32 {
    var sx = x;
    var sy = y;
    if (params.border_mode == 0u) {
        sx = clamp(sx, 0, i32(params.width) - 1);
        sy = clamp(sy, 0, i32(params.height) - 1);
    } else if (sx < 0 || sy < 0 || sx >= i32(params.width) || sy >= i32(params.height)) {
        return 0u;
    }
    return textureLoad(input_px, vec2<i32>(sx, sy), 0).r;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let pixel_count = params.width * params.height;
    if (index >= pixel_count) {
        return;
    }
    let x = i32(index % params.width);
    let y = i32(index / params.width);
    let radius_x = i32(params.kernel_width / 2u);
    let radius_y = i32(params.kernel_height / 2u);
    var sum: u32 = 0u;
    var count: u32 = 0u;
    for (var dy: i32 = -radius_y; dy <= radius_y; dy = dy + 1) {
        for (var dx: i32 = -radius_x; dx <= radius_x; dx = dx + 1) {
            sum = sum + sample_gray(x + dx, y + dy);
            count = count + 1u;
        }
    }
    let value = (sum + count / 2u) / count;
    textureStore(output_px, vec2<i32>(x, y), vec4<u32>(value, 0u, 0u, 0u));
}
"#;

pub(crate) struct BlurPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

pub(crate) fn create_blur_pipeline(device: &wgpu::Device) -> BlurPipeline {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gpu-image-blur-bgl"),
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
        label: Some("gpu-image-blur-shader"),
        source: wgpu::ShaderSource::Wgsl(BLUR_WGSL.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gpu-image-blur-pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gpu-image-blur-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    BlurPipeline { bind_group_layout, pipeline }
}

/// Applies a square/rectangular mean box blur to a single-channel `GpuImage`.
pub fn box_blur_gpu(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    kernel_width: u32,
    kernel_height: u32,
    border: GpuImageBorder,
) -> SpatialResult<GpuImage> {
    source.validate_runtime(runtime)?;
    if source.channels() != 1 {
        return Err(SpatialError::InvalidArgument(
            "box_blur_gpu currently supports single-channel GpuImage inputs".to_owned(),
        ));
    }
    if kernel_width == 0 || kernel_height == 0 || kernel_width % 2 == 0 || kernel_height % 2 == 0 {
        return Err(SpatialError::InvalidArgument(
            "box_blur_gpu requires positive odd kernel dimensions".to_owned(),
        ));
    }
    let pixel_count = (source.width() as usize).saturating_mul(source.height() as usize);
    let out_bytes = u64::from(source.width()) * u64::from(source.height()) * 4;
    let output = create_texture(runtime, source.width(), source.height(), "gpu-image-blur-out");
    let params = BlurParams {
        width: source.width(),
        height: source.height(),
        kernel_width,
        kernel_height,
        border_mode: match border {
            GpuImageBorder::Replicate => 0,
            GpuImageBorder::ConstantZero => 1,
        },
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-image-blur-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let pipeline =
        runtime.image_blur_pipeline.get_or_init(|| create_blur_pipeline(runtime.device()));
    let source_view = source.view();
    let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-image-blur-bg"),
        layout: &pipeline.bind_group_layout,
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
        label: Some("gpu-image-blur-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-image-blur-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(pixel_count.div_ceil(WORKGROUP as usize) as u32, 1, 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
    let mut receipt = GpuImageReceipt::default();
    receipt.merge_from(source.receipt());
    receipt.record_gpu_to_gpu(out_bytes, "box_blur_gpu");
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
