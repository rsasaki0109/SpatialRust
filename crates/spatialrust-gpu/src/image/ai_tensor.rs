//! Device-resident planar floating-point tensors packed from GPU images.

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};

use super::gpu_image::{read_staging_bytes, runtime_device_key, GpuImage, GpuImageReceipt};
use crate::WgpuRuntime;

const WORKGROUP_X: u32 = 16;
const WORKGROUP_Y: u32 = 16;

const SHADER: &str = r#"
struct Params {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
    scale: f32,
    mean0: f32,
    mean1: f32,
    mean2: f32,
    mean3: f32,
    std0: f32,
    std1: f32,
    std2: f32,
    std3: f32,
    _tail0: f32,
    _tail1: f32,
    _tail2: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var source_px: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16)
fn pack_ai_chw(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let pixel = textureLoad(source_px, vec2<i32>(gid.xy), 0);
    let values = vec4<f32>(pixel);
    let means = vec4<f32>(params.mean0, params.mean1, params.mean2, params.mean3);
    let stds = vec4<f32>(params.std0, params.std1, params.std2, params.std3);
    let index = gid.y * params.width + gid.x;
    let plane = params.width * params.height;
    for (var channel = 0u; channel < params.channels; channel++) {
        output[channel * plane + index] = (values[channel] * params.scale - means[channel]) / stds[channel];
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AiPackParams {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
    scale: f32,
    mean: [f32; 4],
    std: [f32; 4],
    _tail: [f32; 3],
}

pub(crate) struct AiPackPipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

pub(crate) fn create_ai_pack_pipeline(device: &wgpu::Device) -> AiPackPipeline {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gpu-ai-pack-bgl"),
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
        label: Some("gpu-ai-pack-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gpu-ai-pack-layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gpu-ai-pack-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("pack_ai_chw"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    AiPackPipeline { layout, pipeline }
}

/// Device-resident planar `f32` tensor with explicit optional readback.
pub struct GpuAiTensor {
    buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    channels: u32,
    byte_len: u64,
    device_key: usize,
    receipt: GpuImageReceipt,
}

impl GpuAiTensor {
    /// Returns tensor dimensions as CHW.
    #[must_use]
    pub const fn shape(&self) -> [u32; 3] {
        [self.channels, self.height, self.width]
    }

    /// Returns cumulative transfer and stage accounting.
    #[must_use]
    pub const fn receipt(&self) -> &GpuImageReceipt {
        &self.receipt
    }

    /// Explicitly reads planar values back to host memory.
    pub fn readback_f32(&mut self, runtime: &WgpuRuntime) -> SpatialResult<Vec<f32>> {
        if self.device_key != runtime_device_key(runtime) {
            return Err(SpatialError::InvalidArgument(
                "GpuAiTensor belongs to a different runtime device".to_owned(),
            ));
        }
        let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-ai-tensor-readback"),
            size: self.byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu-ai-tensor-readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.byte_len);
        runtime.queue().submit(Some(encoder.finish()));
        let bytes = read_staging_bytes(runtime.device(), &staging, self.byte_len as usize)?;
        self.receipt.record_device_to_host(self.byte_len, "readback_ai_chw_f32");
        Ok(bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte f32 chunk")))
            .collect())
    }

    /// Returns tensor storage to the runtime pool.
    pub fn recycle(self, runtime: &WgpuRuntime) {
        if self.device_key == runtime_device_key(runtime) {
            runtime.recycle_storage(self.byte_len, self.buffer);
        }
    }
}

/// Packs a resident image into normalized planar CHW `f32` storage.
pub fn pack_ai_chw_gpu(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    scale: f32,
    mean: [f32; 4],
    std: [f32; 4],
) -> SpatialResult<GpuAiTensor> {
    source.validate_runtime(runtime)?;
    if !scale.is_finite()
        || mean.iter().any(|value| !value.is_finite())
        || std.iter().any(|value| !value.is_finite() || *value == 0.0)
    {
        return Err(SpatialError::InvalidArgument(
            "AI packing scale/mean/std must be finite and std non-zero".to_owned(),
        ));
    }
    let elements =
        u64::from(source.width()) * u64::from(source.height()) * u64::from(source.channels());
    let byte_len = elements * std::mem::size_of::<f32>() as u64;
    let output = runtime.buffer_pool().acquire_storage(runtime, "gpu-ai-chw-output", byte_len);
    let params = AiPackParams {
        width: source.width(),
        height: source.height(),
        channels: source.channels(),
        _pad: 0,
        scale,
        mean,
        std,
        _tail: [0.0; 3],
    };
    let uniform = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-ai-pack-params"),
        size: std::mem::size_of::<AiPackParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    runtime.queue().write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
    let pipeline =
        runtime.image_ai_pack_pipeline.get_or_init(|| create_ai_pack_pipeline(runtime.device()));
    let source_view = source.view();
    let bind_group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-ai-pack-bg"),
        layout: &pipeline.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&source_view),
            },
            wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-ai-pack-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pack_ai_chw_gpu"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            source.width().div_ceil(WORKGROUP_X),
            source.height().div_ceil(WORKGROUP_Y),
            1,
        );
    }
    runtime.queue().submit(Some(encoder.finish()));
    let mut receipt = source.receipt().clone();
    receipt.record_gpu_to_gpu(byte_len, "pack_ai_chw_gpu");
    Ok(GpuAiTensor {
        buffer: output,
        width: source.width(),
        height: source.height(),
        channels: source.channels(),
        byte_len,
        device_key: runtime_device_key(runtime),
        receipt,
    })
}
