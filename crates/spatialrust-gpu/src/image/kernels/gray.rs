//! BT.601 RGB to gray on the GPU.

use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use spatialrust_image::{ColorSpace, ImageMetadata};
use wgpu::util::DeviceExt;

use crate::image::gpu_image::{GpuImage, GpuImageReceipt};
use crate::WgpuRuntime;

const WORKGROUP: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GrayParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

const GRAY_WGSL: &str = r#"
struct Params { width: u32, height: u32, pad0: u32, pad1: u32, };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_px: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_px: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let pixel_count = params.width * params.height;
    if (index >= pixel_count) {
        return;
    }
    let base = index * 3u;
    let r = input_px[base];
    let g = input_px[base + 1u];
    let b = input_px[base + 2u];
    // Match CPU BT.601 fixed-point: (77*R + 150*G + 29*B + 128) >> 8
    output_px[index] = (77u * r + 150u * g + 29u * b + 128u) >> 8u;
}
"#;

struct GrayPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

fn gray_pipeline(device: &wgpu::Device) -> &'static GrayPipeline {
    static PIPELINE: OnceLock<GrayPipeline> = OnceLock::new();
    // Note: OnceLock with device-specific pipeline is only correct for the shared
    // process runtime used by tests; first caller wins.
    PIPELINE.get_or_init(|| {
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-image-gray-shader"),
            source: wgpu::ShaderSource::Wgsl(GRAY_WGSL.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu-image-gray-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu-image-gray-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        GrayPipeline { bind_group_layout, pipeline }
    })
}

/// Converts an RGB `GpuImage` to a gray `GpuImage` using BT.601 fixed-point luma.
pub fn rgb_to_gray_gpu(runtime: &WgpuRuntime, source: &GpuImage) -> SpatialResult<GpuImage> {
    source.validate_runtime(runtime)?;
    if source.channels() != 3 {
        return Err(SpatialError::InvalidArgument(
            "rgb_to_gray_gpu requires a 3-channel RGB GpuImage".to_owned(),
        ));
    }
    let pixel_count = (source.width() as usize).saturating_mul(source.height() as usize);
    let out_words = pixel_count;
    let out_bytes = (out_words * std::mem::size_of::<u32>()) as u64;
    let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-image-gray-out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = GrayParams {
        width: source.width(),
        height: source.height(),
        _pad0: 0,
        _pad1: 0,
    };
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-image-gray-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let pipeline = gray_pipeline(runtime.device());
    let bind_group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-image-gray-bg"),
        layout: &pipeline.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: source.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-image-gray-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-image-gray-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(pixel_count.div_ceil(WORKGROUP as usize) as u32, 1, 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
    let mut receipt = GpuImageReceipt::default();
    receipt.merge_from(source.receipt());
    receipt.record_gpu_to_gpu(out_bytes, "rgb_to_gray_gpu");
    let metadata = ImageMetadata { color_space: ColorSpace::Gray, ..source.metadata() };
    GpuImage::from_parts(
        runtime,
        source.width(),
        source.height(),
        1,
        output,
        out_bytes,
        metadata,
        receipt,
    )
}
