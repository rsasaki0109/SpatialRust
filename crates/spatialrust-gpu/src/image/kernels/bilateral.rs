//! Reusable image-independent bilateral coefficients and exact resident filtering.
use super::GpuImageBorder;
use crate::{
    image::gpu_image::{create_texture, runtime_device_key, GpuImage, GpuImageReceipt},
    WgpuRuntime,
};
use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use std::sync::Arc;
use wgpu::util::DeviceExt;
const SHADER: &str = concat!(
    include_str!("../rounding_u32.wgsl"),
    "\n",
    include_str!("../binary64_positive.wgsl"),
    "\n",
    include_str!("bilateral_round.wgsl"),
    "\n",
    include_str!("bilateral.wgsl")
);
/// Image-independent parameters for exact resident u8 bilateral filtering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBilateralOptions {
    /// Diameter in 1..=7; radius is floor(diameter/2), including even diameters.
    pub diameter: u32,
    /// Finite positive color sigma for squared L1 color distance, including alpha.
    pub sigma_color: f64,
    /// Finite positive spatial sigma for the circular neighborhood.
    pub sigma_space: f64,
    /// Replicate or constant-zero boundary extrapolation.
    pub border: GpuImageBorder,
}
impl Default for GpuBilateralOptions {
    fn default() -> Self {
        Self { diameter: 5, sigma_color: 40.0, sigma_space: 3.0, border: GpuImageBorder::Replicate }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    channels: u32,
    taps: u32,
    border: u32,
    padding: [u32; 3],
}
fn invalid() -> SpatialError {
    SpatialError::InvalidArgument("GPU bilateral requires 1..=4 u8 channels, diameter 1..=7, finite positive sigmas, normal-or-zero weights and supported device limits".into())
}
fn control(
    channels: u32,
    options: GpuBilateralOptions,
    limits: &wgpu::Limits,
) -> SpatialResult<(Vec<[u32; 2]>, u32)> {
    if !(1..=4).contains(&channels)
        || !(1..=7).contains(&options.diameter)
        || !options.sigma_color.is_finite()
        || options.sigma_color <= 0.0
        || !options.sigma_space.is_finite()
        || options.sigma_space <= 0.0
    {
        return Err(invalid());
    }
    let cf = -0.5 / (options.sigma_color * options.sigma_color);
    let sf = -0.5 / (options.sigma_space * options.sigma_space);
    if !cf.is_finite() || !sf.is_finite() {
        return Err(invalid());
    }
    let radius = (options.diameter / 2) as i32;
    let mut taps = Vec::new();
    let mut distances = Vec::new();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let d = dx * dx + dy * dy;
            if d <= radius * radius {
                taps.push((dx, dy, d));
                if !distances.contains(&d) {
                    distances.push(d);
                }
            }
        }
    }
    distances.sort_unstable();
    let colors = 255 * channels + 1;
    let entries = taps.len() * 2 + distances.len() * colors as usize;
    let bytes = (entries * 8) as u64;
    capabilities(bytes, limits)?;
    let mut data = Vec::with_capacity(entries);
    for &(dx, dy, d) in &taps {
        let index = distances.iter().position(|&v| v == d).unwrap();
        data.push([dx as u32, dy as u32]);
        data.push([(taps.len() * 2 + index * colors as usize) as u32, 0]);
    }
    for d in distances {
        for color in 0..colors {
            // A center tap always has L1 distance zero; other center entries cannot be sampled.
            let w = if d == 0 {
                if color == 0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                let c = f64::from(color);
                (c.mul_add(c, 0.0).mul_add(cf, f64::from(d) * sf)).exp()
            };
            if !w.is_finite() || w < 0.0 || (w != 0.0 && !w.is_normal()) {
                return Err(invalid());
            }
            let bits = w.to_bits();
            data.push([bits as u32, (bits >> 32) as u32]);
        }
    }
    Ok((data, taps.len() as u32))
}
fn capabilities(bytes: u64, l: &wgpu::Limits) -> SpatialResult<()> {
    if bytes > l.max_buffer_size
        || bytes > u64::from(l.max_storage_buffer_binding_size)
        || l.max_storage_buffers_per_shader_stage == 0
        || l.max_uniform_buffer_binding_size < 32
        || l.max_buffer_size < 32
        || l.max_bind_groups == 0
        || l.max_bindings_per_bind_group < 4
        || l.max_uniform_buffers_per_shader_stage == 0
        || l.max_sampled_textures_per_shader_stage == 0
        || l.max_storage_textures_per_shader_stage == 0
        || l.max_compute_workgroup_size_x < 16
        || l.max_compute_workgroup_size_y < 16
        || l.max_compute_workgroup_size_z == 0
        || l.max_compute_invocations_per_workgroup < 256
    {
        return Err(invalid());
    }
    Ok(())
}
fn dimensions(w: u32, h: u32, r: u32, l: &wgpu::Limits) -> SpatialResult<u64> {
    if w == 0
        || h == 0
        || w > i32::MAX as u32
        || h > i32::MAX as u32
        || w > l.max_texture_dimension_2d
        || h > l.max_texture_dimension_2d
        || u64::from(w - 1) + u64::from(r) > i32::MAX as u64
        || u64::from(h - 1) + u64::from(r) > i32::MAX as u64
        || w.div_ceil(16) > l.max_compute_workgroups_per_dimension
        || h.div_ceil(16) > l.max_compute_workgroups_per_dimension
    {
        return Err(invalid());
    }
    Ok(u64::from(w) * u64::from(h) * 4)
}
/// Owned, runtime-bound bilateral coefficients prepared and uploaded explicitly.
/// Reuse across image sizes with the same logical channels and options. Preparation
/// rejects nonfinite factors/weights and positive subnormal coefficients. Filtering
/// uses exact integer-word binary64 accumulation and CPU-compatible byte rounding;
/// no native shader float64, image payload copy, host wait or image-sized scratch.
/// Plan control uploads are accounted separately from each image's transfer receipt.
pub struct GpuBilateralFilter {
    device_key: Arc<()>,
    channels: u32,
    options: GpuBilateralOptions,
    coefficients: wgpu::Buffer,
    taps: u32,
    receipt: GpuImageReceipt,
}
impl GpuBilateralFilter {
    /// Explicitly prepares and uploads bounded image-independent coefficients.
    /// No image is read and no image pipeline initializes during preparation.
    pub fn prepare(
        runtime: &WgpuRuntime,
        channels: u32,
        options: GpuBilateralOptions,
    ) -> SpatialResult<Self> {
        let (data, taps) = control(channels, options, &runtime.device().limits())?;
        let bytes = (data.len() * 8) as u64;
        let coefficients = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bilateral-coefficients"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut receipt = GpuImageReceipt::default();
        receipt.record_host_to_device(bytes, "prepare_bilateral_coefficients");
        Ok(Self {
            device_key: runtime_device_key(runtime),
            channels,
            options,
            coefficients,
            taps,
            receipt,
        })
    }
    /// Returns physical coefficient upload bytes and preparation stages.
    #[must_use]
    pub fn receipt(&self) -> &GpuImageReceipt {
        &self.receipt
    }
    /// Submits one device pass and returns an independently owned image.
    /// Image channels/runtime/limits reject before output or lazy pipeline allocation.
    /// The image receipt inherits its source; the reusable plan receipt stays separate.
    pub fn filter(&self, runtime: &WgpuRuntime, source: &GpuImage) -> SpatialResult<GpuImage> {
        if !Arc::ptr_eq(&self.device_key, &runtime_device_key(runtime))
            || source.channels() != self.channels
        {
            return Err(invalid());
        }
        source.validate_runtime(runtime)?;
        let limits = runtime.device().limits();
        capabilities(self.receipt.host_to_device_bytes(), &limits)?;
        let bytes =
            dimensions(source.width(), source.height(), self.options.diameter / 2, &limits)?;
        let cache = runtime
            .image_bilateral_pipeline
            .get_or_init(|| BilateralPipeline::new(runtime.device()));
        let output = create_texture(runtime, source.width(), source.height(), "bilateral-output");
        let params = Params {
            width: source.width(),
            height: source.height(),
            channels: self.channels,
            taps: self.taps,
            border: u32::from(self.options.border == GpuImageBorder::ConstantZero),
            padding: [0; 3],
        };
        let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bilateral-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bilateral-group"),
            layout: &cache.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&source.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.coefficients.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &output.create_view(&Default::default()),
                    ),
                },
            ],
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bilateral_filter_gpu"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cache.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(source.width().div_ceil(16), source.height().div_ceil(16), 1);
        }
        runtime.queue().submit(Some(encoder.finish()));
        let mut receipt = source.receipt().clone();
        receipt.record_gpu_to_gpu(bytes, "bilateral_filter_gpu");
        GpuImage::from_parts(
            runtime,
            source.width(),
            source.height(),
            source.channels(),
            output,
            source.metadata(),
            receipt,
        )
    }
}
pub(crate) struct BilateralPipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}
impl BilateralPipeline {
    fn new(device: &wgpu::Device) -> Self {
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bilateral-layout"),
            entries: &[
                buffer(0, wgpu::BufferBindingType::Uniform),
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
                buffer(2, wgpu::BufferBindingType::Storage { read_only: true }),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bilateral-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bilateral-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bilateral-pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("bilateral"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { layout, pipeline }
    }
}
#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "bilateral_tests.rs"]
mod tests;
