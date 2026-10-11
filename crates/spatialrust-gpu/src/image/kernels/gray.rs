//! BT.601 RGB to gray on the GPU.

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use spatialrust_image::{ColorSpace, ImageMetadata};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

use crate::image::gpu_image::{create_texture, GpuImage, GpuImageReceipt};
use crate::WgpuRuntime;

const WORKGROUP: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GrayParams {
    width: u32,
    height: u32,
    groups_per_row: u32,
    group_count: u32,
}

#[derive(Clone, Copy, Debug)]
struct GrayPlan {
    groups_x: u32,
    groups_y: u32,
    group_count: u32,
}

fn gray_plan(width: u32, height: u32, limits: &wgpu::Limits) -> SpatialResult<GrayPlan> {
    let invalid = || {
        SpatialError::InvalidArgument(
        "GPU gray requires positive dimensions, u32 pixel counts and supported texture/compute/binding limits".to_owned(),
    )
    };
    let pixels = u64::from(width) * u64::from(height);
    if width == 0
        || height == 0
        || width > limits.max_texture_dimension_2d
        || height > limits.max_texture_dimension_2d
        || width > i32::MAX as u32
        || height > i32::MAX as u32
        || pixels > u64::from(u32::MAX)
        || limits.max_compute_invocations_per_workgroup < WORKGROUP
        || limits.max_compute_workgroup_size_x < WORKGROUP
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
        || limits.max_compute_workgroups_per_dimension == 0
        || limits.max_sampled_textures_per_shader_stage == 0
        || limits.max_storage_textures_per_shader_stage == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < std::mem::size_of::<GrayParams>() as u32
        || limits.max_buffer_size < std::mem::size_of::<GrayParams>() as u64
        || limits.max_bind_groups == 0
        || limits.max_bindings_per_bind_group < 3
    {
        return Err(invalid());
    }
    let group_count = pixels.div_ceil(u64::from(WORKGROUP)) as u32;
    let groups_y = group_count.div_ceil(limits.max_compute_workgroups_per_dimension);
    if groups_y > limits.max_compute_workgroups_per_dimension {
        return Err(invalid());
    }
    let groups_x = group_count.div_ceil(groups_y);
    Ok(GrayPlan { groups_x, groups_y, group_count })
}

const GRAY_WGSL: &str = r#"
struct Params { width: u32, height: u32, pad0: u32, pad1: u32, };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input_px: texture_2d<u32>;
@group(0) @binding(2) var output_px: texture_storage_2d<rgba8uint, write>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let pixel_count = params.width * params.height;
    if (index >= pixel_count) {
        return;
    }
    let xy = vec2<i32>(i32(index % params.width), i32(index / params.width));
    let rgb = textureLoad(input_px, xy, 0).rgb;
    // Match CPU BT.601 fixed-point: (77*R + 150*G + 29*B + 128) >> 8
    let gray = (77u * rgb.r + 150u * rgb.g + 29u * rgb.b + 128u) >> 8u;
    textureStore(output_px, xy, vec4<u32>(gray, 0u, 0u, 0u));
}
"#;

pub(crate) struct GrayPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: OnceLock<wgpu::ComputePipeline>,
    grid_pipeline: OnceLock<wgpu::ComputePipeline>,
}

pub(crate) fn create_gray_pipeline(device: &wgpu::Device) -> GrayPipeline {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gpu-image-gray-bgl"),
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
    GrayPipeline { bind_group_layout, pipeline: OnceLock::new(), grid_pipeline: OnceLock::new() }
}

fn create_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    source: &str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gpu-image-gray-shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gpu-image-gray-pl"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gpu-image-gray-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Converts an RGB `GpuImage` to a gray `GpuImage` using BT.601 fixed-point luma.
///
/// Large images distribute flat workgroups across bounded dispatch rows. Runtime,
/// dimensions and device limits are checked before allocation or pipeline setup.
pub fn rgb_to_gray_gpu(runtime: &WgpuRuntime, source: &GpuImage) -> SpatialResult<GpuImage> {
    rgb_to_gray_gpu_with_profiler(runtime, source, None)
}

pub(crate) fn rgb_to_gray_gpu_with_profiler(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    profiler: Option<&mut crate::GpuStageProfiler>,
) -> SpatialResult<GpuImage> {
    source.validate_runtime(runtime)?;
    if source.channels() != 3 {
        return Err(SpatialError::InvalidArgument(
            "rgb_to_gray_gpu requires a 3-channel RGB GpuImage".to_owned(),
        ));
    }
    let plan = gray_plan(source.width(), source.height(), &runtime.device().limits())?;
    if let Some(profiler) = &profiler {
        profiler.validate(runtime, 1)?;
    }
    let cache = runtime.image_gray_pipeline.get_or_init(|| create_gray_pipeline(runtime.device()));
    let pipeline = if plan.groups_y == 1 {
        cache
            .pipeline
            .get_or_init(|| create_pipeline(runtime.device(), &cache.bind_group_layout, GRAY_WGSL))
    } else {
        cache.grid_pipeline.get_or_init(|| {
            create_pipeline(
                runtime.device(),
                &cache.bind_group_layout,
                include_str!("gray_grid.wgsl"),
            )
        })
    };
    gray_dispatch(runtime, source, plan, cache, pipeline, profiler)
}

fn gray_dispatch(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    plan: GrayPlan,
    cache: &GrayPipeline,
    pipeline: &wgpu::ComputePipeline,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
) -> SpatialResult<GpuImage> {
    let output = create_texture(runtime, source.width(), source.height(), "gpu-image-gray-out");
    let out_bytes = u64::from(source.width()) * u64::from(source.height()) * 4;
    let params = GrayParams {
        width: source.width(),
        height: source.height(),
        groups_per_row: plan.groups_x,
        group_count: plan.group_count,
    };
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-image-gray-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let source_view = source.view();
    let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-image-gray-bg"),
        layout: &cache.bind_group_layout,
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
        label: Some("gpu-image-gray-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-image-gray-pass"),
            timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, "rgb_to_gray_gpu"),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(plan.groups_x, plan.groups_y, 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
    let mut receipt = GpuImageReceipt::default();
    receipt.merge_from(source.receipt());
    receipt.record_gpu_to_gpu(out_bytes, "rgb_to_gray_gpu");
    let metadata = ImageMetadata { color_space: ColorSpace::Gray, ..source.metadata() };
    GpuImage::from_parts(runtime, source.width(), source.height(), 1, output, metadata, receipt)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "gray_tests.rs"]
mod tests;
