//! Resident integer Canny gradients, suppression and complete 8-connected hysteresis.

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use wgpu::util::DeviceExt;

use super::gpu_image::{create_texture, GpuImage};
use crate::{scratch::ScratchBuffers, GpuStageProfiler, WgpuRuntime};

const WORKGROUP: u32 = 256;
const SHADER: &str = include_str!("canny.wgsl");
const ENTRIES: [&str; 7] = [
    "gradient_direct",
    "gradient_horizontal",
    "gradient_vertical",
    "classify",
    "connect",
    "mark_roots",
    "pack_edges",
];
const STAGES: [&str; 7] = [
    "canny_gradients_gpu",
    "canny_horizontal_gpu",
    "canny_gradients_gpu",
    "canny_classify_gpu",
    "canny_connect_gpu",
    "canny_roots_gpu",
    "canny_gpu",
];

/// Threshold and gradient options for resident grayscale Canny.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuCannyOptions {
    /// Finite nonnegative lower threshold. Reversed thresholds are swapped.
    pub low_threshold: f64,
    /// Finite nonnegative upper threshold; strong comparisons are strict.
    pub high_threshold: f64,
    /// Sobel aperture: 3, 5 or 7. Aperture 7 uses the CPU-compatible 1/16 scale.
    pub aperture_size: u32,
    /// Compare squared Euclidean magnitudes instead of L1 magnitudes.
    pub l2_gradient: bool,
}

impl Default for GpuCannyOptions {
    fn default() -> Self {
        Self { low_threshold: 100.0, high_threshold: 200.0, aperture_size: 3, l2_gradient: false }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    pixels: u32,
    groups: u32,
    aperture: u32,
    l2: u32,
    low: u32,
    high: u32,
    smooth: [[i32; 4]; 2],
    derivative: [[i32; 4]; 2],
}

struct Plan {
    params: Params,
    gradient_bytes: u64,
    pixel_bytes: u64,
    horizontal_bytes: u64,
}

fn thresholds(options: GpuCannyOptions) -> (u32, u32) {
    let (mut low, mut high) = (options.low_threshold, options.high_threshold);
    if options.aperture_size == 7 {
        low /= 16.0;
        high /= 16.0;
    }
    if low > high {
        std::mem::swap(&mut low, &mut high);
    }
    let convert = |value: f64| {
        let value = if options.l2_gradient {
            let value = value.min(32767.0);
            value.mul_add(value, 0.0)
        } else {
            value
        };
        // Every actual derivative magnitude is below this bound; larger CPU
        // i64 thresholds therefore make exactly the same comparisons false.
        value.floor().min(f64::from(u32::MAX)) as u32
    };
    (convert(low), convert(high))
}

fn plan(
    width: u32,
    height: u32,
    channels: u32,
    options: GpuCannyOptions,
    limits: &wgpu::Limits,
) -> SpatialResult<Plan> {
    let invalid = || {
        SpatialError::InvalidArgument("GPU Canny requires a supported nonempty grayscale image, 3/5/7 aperture, finite nonnegative thresholds and bounded buffer/binding/compute limits".to_owned())
    };
    let pixels = u64::from(width) * u64::from(height);
    // Both scaled signed derivatives fit i16; magnitude remains a full u32.
    let gradient_bytes = pixels.checked_mul(8).ok_or_else(invalid)?;
    let pixel_bytes = pixels.checked_mul(4).ok_or_else(invalid)?;
    // Unrounded 5/7 horizontal derivative and smoothing sums also fit i16.
    let horizontal_bytes =
        if options.aperture_size == 3 { 8 } else { pixels.checked_mul(4).ok_or_else(invalid)? };
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    if width == 0
        || height == 0
        || channels != 1
        || ![3, 5, 7].contains(&options.aperture_size)
        || !options.low_threshold.is_finite()
        || !options.high_threshold.is_finite()
        || options.low_threshold < 0.0
        || options.high_threshold < 0.0
        || pixels > u64::from(u32::MAX)
        || width > limits.max_texture_dimension_2d
        || height > limits.max_texture_dimension_2d
        || u64::from(width) + 3 > i32::MAX as u64
        || u64::from(height) + 3 > i32::MAX as u64
        || gradient_bytes > bound
        || horizontal_bytes > bound
        || pixel_bytes > bound
        || limits.max_buffer_size < std::mem::size_of::<Params>() as u64
        || limits.max_uniform_buffer_binding_size < std::mem::size_of::<Params>() as u32
        || limits.max_compute_invocations_per_workgroup < WORKGROUP
        || limits.max_compute_workgroup_size_x < WORKGROUP
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
        || limits.max_compute_workgroups_per_dimension == 0
        || limits.max_bind_groups == 0
        || limits.max_bindings_per_bind_group < 8
        || limits.max_storage_buffers_per_shader_stage < 5
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_sampled_textures_per_shader_stage == 0
        || limits.max_storage_textures_per_shader_stage == 0
    {
        return Err(invalid());
    }
    let (smooth, derivative) = match options.aperture_size {
        3 => ([1, 2, 1, 0, 0, 0, 0, 0], [-1, 0, 1, 0, 0, 0, 0, 0]),
        5 => ([1, 4, 6, 4, 1, 0, 0, 0], [-1, -2, 0, 2, 1, 0, 0, 0]),
        _ => ([1, 6, 15, 20, 15, 6, 1, 0], [-1, -4, -5, 0, 5, 4, 1, 0]),
    };
    let vectors = |v: [i32; 8]| std::array::from_fn(|i| std::array::from_fn(|j| v[4 * i + j]));
    let (low, high) = thresholds(options);
    Ok(Plan {
        params: Params {
            width,
            height,
            pixels: pixels as u32,
            groups: (pixels as u32)
                .div_ceil(WORKGROUP)
                .min(4096)
                .min(limits.max_compute_workgroups_per_dimension),
            aperture: options.aperture_size,
            l2: u32::from(options.l2_gradient),
            low,
            high,
            smooth: vectors(smooth),
            derivative: vectors(derivative),
        },
        gradient_bytes,
        pixel_bytes,
        horizontal_bytes,
    })
}

pub(crate) struct CannyPipelines {
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 7],
}

impl CannyPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let types: Vec<_> = (0..8)
            .map(|binding| match binding {
                0 => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                1 => wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                7 => wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                _ => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
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
            label: Some("canny-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("canny-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("canny-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipelines = ENTRIES.map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Self { layout, pipelines }
    }
}

struct Work {
    gradients: wgpu::Buffer,
    horizontal: wgpu::Buffer,
    states: wgpu::Buffer,
    parent: wgpu::Buffer,
    strong: wgpu::Buffer,
}

impl Work {
    fn acquire(scratch: &mut ScratchBuffers<'_>, p: &Plan) -> Self {
        Self {
            gradients: scratch.acquire("canny-gradients", p.gradient_bytes),
            horizontal: scratch.acquire("canny-horizontal", p.horizontal_bytes),
            states: scratch.acquire("canny-states", p.pixel_bytes),
            parent: scratch.acquire("canny-parents", p.pixel_bytes),
            strong: scratch.acquire("canny-strong-roots", p.pixel_bytes),
        }
    }

    fn bind(
        &self,
        runtime: &WgpuRuntime,
        cache: &CannyPipelines,
        uniform: &wgpu::Buffer,
        source: &wgpu::TextureView,
        output: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let buffers = [&self.gradients, &self.horizontal, &self.states, &self.parent, &self.strong];
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(source),
            },
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
            label: Some("canny-bindings"),
            layout: &cache.layout,
            entries: &entries,
        })
    }
}

fn pass_indices(aperture: u32) -> &'static [usize] {
    if aperture == 3 {
        &[0, 3, 4, 5, 6]
    } else {
        &[1, 2, 3, 4, 5, 6]
    }
}

fn encode(
    cache: &CannyPipelines,
    group: &wgpu::BindGroup,
    groups: u32,
    indices: &[usize],
    encoder: &mut wgpu::CommandEncoder,
    mut profiler: Option<&mut GpuStageProfiler>,
) {
    for &index in indices {
        let stage = STAGES[index];
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(stage),
            timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, stage),
        });
        pass.set_pipeline(&cache.pipelines[index]);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

/// Computes a CPU-compatible binary Canny edge map entirely on the device.
///
/// Accepts one logical u8 channel and preserves image dimensions/metadata.
/// Signed integer Sobel, asymmetric directional NMS ties and exact threshold
/// conversion match the specialized CPU route. Complete eight-neighbor weak
/// connectivity uses monotone atomic union of weak candidates and per-root
/// strong-neighbor flags, with no
/// propagation iteration cutoff. All five/six passes share one submission.
/// No image/label readback, host wait, implicit blur, CPU fallback or automatic
/// placement occurs. Buffer/workgroup/runtime limits reject before allocation.
pub fn canny_gpu(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    options: GpuCannyOptions,
) -> SpatialResult<GpuImage> {
    canny_impl(runtime, source, options, None)
}

/// Records five (aperture 3) or six (5/7) optional GPU pass timestamps.
pub fn canny_gpu_profiled(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    options: GpuCannyOptions,
    profiler: &mut GpuStageProfiler,
) -> SpatialResult<GpuImage> {
    let result = canny_impl(runtime, source, options, Some(profiler));
    profiler.finish_recording(&result);
    result
}

fn canny_impl(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    options: GpuCannyOptions,
    profiler: Option<&mut GpuStageProfiler>,
) -> SpatialResult<GpuImage> {
    source.validate_runtime(runtime)?;
    let p = plan(
        source.width(),
        source.height(),
        source.channels(),
        options,
        &runtime.device().limits(),
    )?;
    let indices = pass_indices(options.aperture_size);
    if let Some(ref profiler) = profiler {
        profiler.validate(runtime, indices.len() as u32)?;
    }
    let cache = runtime.image_canny_pipelines.get_or_init(|| CannyPipelines::new(runtime.device()));
    let mut scratch = ScratchBuffers::new(runtime);
    let work = Work::acquire(&mut scratch, &p);
    let output = create_texture(runtime, source.width(), source.height(), "canny-output");
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("canny-params"),
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let group = work.bind(
        runtime,
        cache,
        &uniform,
        &source.view(),
        &output.create_view(&Default::default()),
    );
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encode(cache, &group, p.params.groups, indices, &mut encoder, profiler);
    scratch.submit_and_recycle(encoder);
    let mut receipt = source.receipt().clone();
    // Destination footprints: in-place atomic traffic is not a byte-bandwidth metric.
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
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "canny_tests.rs"]
mod tests;
