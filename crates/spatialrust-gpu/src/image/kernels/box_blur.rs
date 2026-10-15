//! Exact integer gray box blur with bounded device-resident separable sums.

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

use crate::image::gpu_image::{create_texture, GpuImage};
use crate::{scratch::ScratchBuffers, GpuStageProfiler, WgpuRuntime};

const WORKGROUP: u32 = 256;
const SEPARABLE_MIN_AREA: u64 = 81;
const SEPARABLE_MIN_SAMPLES: u64 = 65_536;
const SHADERS: [&str; 5] = [
    include_str!("box_blur_serial.wgsl"),
    concat!(include_str!("box_blur_params.wgsl"), include_str!("box_blur_horizontal.wgsl")),
    concat!(include_str!("box_blur_params.wgsl"), include_str!("box_blur_vertical.wgsl")),
    concat!(include_str!("box_blur_params.wgsl"), include_str!("box_blur_packed_horizontal.wgsl")),
    concat!(include_str!("box_blur_params.wgsl"), include_str!("box_blur_packed_vertical.wgsl")),
];

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
    groups: u32,
    pixel_count: u32,
    _pad: u32,
}

struct Plan {
    params: BlurParams,
    separable: bool,
    bytes: u64,
    scratch_bytes: u64,
    packed: bool,
}

pub(crate) fn blur_pass_count(
    width: u32,
    height: u32,
    kernel_width: u32,
    kernel_height: u32,
) -> u32 {
    let area = u64::from(kernel_width) * u64::from(kernel_height);
    let count = u64::from(width) * u64::from(height);
    // Compare via division so even unvalidated profiler inputs cannot overflow.
    1 + u32::from(area >= SEPARABLE_MIN_AREA && count >= SEPARABLE_MIN_SAMPLES.div_ceil(area))
}

fn plan(
    width: u32,
    height: u32,
    kw: u32,
    kh: u32,
    border: GpuImageBorder,
    limits: &wgpu::Limits,
) -> SpatialResult<Plan> {
    let invalid = || {
        SpatialError::InvalidArgument(
        "GPU box blur requires positive odd kernels, supported texture/compute limits, bounded u32 pixels/sums and supported separable scratch".to_owned())
    };
    let count = u64::from(width) * u64::from(height);
    let area = u64::from(kw) * u64::from(kh);
    let separable = blur_pass_count(width, height, kw, kh) == 2;
    let bytes = count.checked_mul(4).ok_or_else(invalid)?;
    if width == 0
        || height == 0
        || kw == 0
        || kh == 0
        || kw % 2 == 0
        || kh % 2 == 0
        || count > u64::from(u32::MAX)
        || area > u64::from(u32::MAX) / 255
        || area * 255 + area / 2 > u64::from(u32::MAX)
        || width > limits.max_texture_dimension_2d
        || height > limits.max_texture_dimension_2d
        || width > i32::MAX as u32
        || height > i32::MAX as u32
        || u64::from(width - 1) + u64::from(kw / 2) > i32::MAX as u64
        || u64::from(height - 1) + u64::from(kh / 2) > i32::MAX as u64
        || limits.max_compute_invocations_per_workgroup < WORKGROUP
        || limits.max_compute_workgroup_size_x < WORKGROUP
        || limits.max_compute_workgroups_per_dimension == 0
        || limits.max_sampled_textures_per_shader_stage == 0
        || limits.max_storage_textures_per_shader_stage == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < std::mem::size_of::<BlurParams>() as u32
        || (separable
            && (limits.max_storage_buffers_per_shader_stage == 0
                || bytes
                    > limits
                        .max_buffer_size
                        .min(u64::from(limits.max_storage_buffer_binding_size))))
    {
        return Err(invalid());
    }
    let groups = (count as u32)
        .div_ceil(WORKGROUP)
        .min(4096)
        .min(limits.max_compute_workgroups_per_dimension);
    // The original checks above still validate the full uncompressed extent.
    let packed = separable && kw <= 257;
    let scratch_bytes = if packed { count.div_ceil(2) * 4 } else { bytes };
    Ok(Plan {
        params: BlurParams {
            width,
            height,
            kernel_width: kw,
            kernel_height: kh,
            border_mode: u32::from(border == GpuImageBorder::ConstantZero),
            groups,
            pixel_count: count as u32,
            _pad: 0,
        },
        separable,
        bytes,
        scratch_bytes,
        packed,
    })
}

struct Variant {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

pub(crate) struct BlurPipeline {
    variants: [OnceLock<Variant>; 5],
}

impl BlurPipeline {
    fn new() -> Self {
        Self { variants: std::array::from_fn(|_| OnceLock::new()) }
    }

    fn variant(&self, device: &wgpu::Device, index: usize) -> &Variant {
        self.variants[index].get_or_init(|| {
            let buffer = |ty| wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            };
            let texture = wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            };
            let output = wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba8Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
            };
            let types = [
                buffer(wgpu::BufferBindingType::Uniform),
                if matches!(index, 2 | 4) {
                    buffer(wgpu::BufferBindingType::Storage { read_only: true })
                } else {
                    texture
                },
                if matches!(index, 1 | 3) {
                    buffer(wgpu::BufferBindingType::Storage { read_only: false })
                } else {
                    output
                },
            ];
            let entries: Vec<_> = types
                .into_iter()
                .enumerate()
                .map(|(binding, ty)| wgpu::BindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty,
                    count: None,
                })
                .collect();
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu-blur-bgl"),
                entries: &entries,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpu-blur-shader"),
                source: wgpu::ShaderSource::Wgsl(SHADERS[index].into()),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu-blur-pl"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpu-blur-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            Variant { layout, pipeline }
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn encode(
    runtime: &WgpuRuntime,
    cache: &BlurPipeline,
    p: &Plan,
    source: &wgpu::TextureView,
    output: &wgpu::TextureView,
    uniform: &wgpu::Buffer,
    sums: Option<&wgpu::Buffer>,
    encoder: &mut wgpu::CommandEncoder,
    mut profiler: Option<&mut GpuStageProfiler>,
) {
    let indices: &[usize] = if p.packed {
        &[3, 4]
    } else if p.separable {
        &[1, 2]
    } else {
        &[0]
    };
    for &index in indices {
        let variant = cache.variant(runtime.device(), index);
        let input = if matches!(index, 2 | 4) {
            sums.unwrap().as_entire_binding()
        } else {
            wgpu::BindingResource::TextureView(source)
        };
        let target = if matches!(index, 1 | 3) {
            sums.unwrap().as_entire_binding()
        } else {
            wgpu::BindingResource::TextureView(output)
        };
        let stage = if matches!(index, 1 | 3) { "box_blur_horizontal_gpu" } else { "box_blur_gpu" };
        let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(stage),
            layout: &variant.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input },
                wgpu::BindGroupEntry { binding: 2, resource: target },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(stage),
            timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, stage),
        });
        pass.set_pipeline(&variant.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(p.params.groups, 1, 1);
    }
}

/// Applies an integer mean box blur to a single-channel `GpuImage`.
///
/// Positive odd kernels round half upward once after summing the entire box.
/// Boxes with area at least 81 and at least 65,536 total pixel/sample visits
/// use two passes. Widths up to 257 pack two exact horizontal sums per u32
/// (four scratch bytes per pixel pair); wider kernels keep four bytes per
/// pixel. Smaller workloads use one pass. Transfers stay explicit and bounded.
pub fn box_blur_gpu(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    kernel_width: u32,
    kernel_height: u32,
    border: GpuImageBorder,
) -> SpatialResult<GpuImage> {
    box_blur_gpu_with_profiler(runtime, source, kernel_width, kernel_height, border, None)
}

pub(crate) fn box_blur_gpu_with_profiler(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    kernel_width: u32,
    kernel_height: u32,
    border: GpuImageBorder,
    profiler: Option<&mut GpuStageProfiler>,
) -> SpatialResult<GpuImage> {
    source.validate_runtime(runtime)?;
    if source.channels() != 1 {
        return Err(SpatialError::InvalidArgument(
            "GPU box blur requires one logical channel".to_owned(),
        ));
    }
    let p = plan(
        source.width(),
        source.height(),
        kernel_width,
        kernel_height,
        border,
        &runtime.device().limits(),
    )?;
    let cache = runtime.image_blur_pipeline.get_or_init(BlurPipeline::new);
    let mut scratch = ScratchBuffers::new(runtime);
    let sums = p.separable.then(|| scratch.acquire("gpu-blur-horizontal-sums", p.scratch_bytes));
    let output = create_texture(runtime, source.width(), source.height(), "gpu-image-blur-out");
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-image-blur-params"),
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encode(
        runtime,
        cache,
        &p,
        &source.view(),
        &output.create_view(&Default::default()),
        &uniform,
        sums.as_ref(),
        &mut encoder,
        profiler,
    );
    scratch.submit_and_recycle(encoder);
    let mut receipt = source.receipt().clone();
    if p.separable {
        receipt.record_gpu_to_gpu(p.scratch_bytes, "box_blur_horizontal_gpu");
    }
    receipt.record_gpu_to_gpu(p.bytes, "box_blur_gpu");
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

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "box_blur_tests.rs"]
mod tests;
