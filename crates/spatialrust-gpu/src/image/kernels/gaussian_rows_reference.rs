// Complete immediate-parent Gaussian planner, caches and call; native only.
use super::*;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    pixels: u32,
    groups: u32,
    kernel_width: u32,
    kernel_height: u32,
    border: u32,
    bits: u32,
    channels: u32,
    padding: [u32; 3],
    weights_x: [[u32; 4]; 2],
    weights_y: [[u32; 4]; 2],
    constant: [u32; 4],
}

struct Plan {
    params: Params,
    scratch_bytes: u64,
    packed: bool,
}

// Same host binary64 normalization and center correction as gaussian_blur_u8.
// A 7-tap axis selects Q15 for BOTH axes; other supported calls use Q8.
fn weights(size: u32, sigma: f64, bits: u32) -> [[u32; 4]; 2] {
    let scale = 1u32 << bits;
    let denominator = 2.0 * sigma * sigma;
    let center = f64::from(size / 2);
    let mut raw = [0.0; 7];
    for i in 0..size {
        let offset = f64::from(i) - center;
        raw[i as usize] = (-(offset * offset) / denominator).exp();
    }
    let sum = raw[..size as usize].iter().sum::<f64>();
    let mut fixed = [0i32; 8];
    for i in 0..size {
        fixed[i as usize] = (raw[i as usize] * f64::from(scale) / sum).round() as i32;
    }
    fixed[(size / 2) as usize] += scale as i32 - fixed.iter().sum::<i32>();
    std::array::from_fn(|i| std::array::from_fn(|j| fixed[i * 4 + j] as u32))
}

fn plan(
    width: u32,
    height: u32,
    channels: u32,
    options: GpuGaussianBlurOptions,
    limits: &wgpu::Limits,
) -> SpatialResult<Plan> {
    let invalid = || {
        SpatialError::InvalidArgument(
        "GPU Gaussian requires supported nonempty 1..=4-channel images, 3/5/7 kernels, finite positive sigmas and bounded scratch/compute limits".to_owned(),
    )
    };
    let pixels = u64::from(width) * u64::from(height);
    let components = pixels.checked_mul(u64::from(channels)).ok_or_else(invalid)?;
    let scratch_bytes = components.checked_mul(4).ok_or_else(invalid)?;
    if width == 0
        || height == 0
        || !(1..=4).contains(&channels)
        || ![3, 5, 7].contains(&options.kernel_width)
        || ![3, 5, 7].contains(&options.kernel_height)
        || !options.sigma_x.is_finite()
        || !options.sigma_y.is_finite()
        || options.sigma_x <= 0.0
        || options.sigma_y <= 0.0
        || pixels > u64::from(u32::MAX)
        || components > u64::from(u32::MAX)
        || width > limits.max_texture_dimension_2d
        || height > limits.max_texture_dimension_2d
        // Reflections use signed coordinates and a doubled dimension.
        || u64::from(width) * 2 + 3 > i32::MAX as u64
        || u64::from(height) * 2 + 3 > i32::MAX as u64
        || scratch_bytes > limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size))
        || limits.max_buffer_size < std::mem::size_of::<Params>() as u64
        || limits.max_compute_invocations_per_workgroup < WORKGROUP
        || limits.max_compute_workgroup_size_x < WORKGROUP
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
        || limits.max_compute_workgroups_per_dimension == 0
        || limits.max_bind_groups == 0
        || limits.max_bindings_per_bind_group < 3
        || limits.max_sampled_textures_per_shader_stage == 0
        || limits.max_storage_textures_per_shader_stage == 0
        || limits.max_storage_buffers_per_shader_stage == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < std::mem::size_of::<Params>() as u32
    {
        return Err(invalid());
    }
    let bits = if options.kernel_width == 7 || options.kernel_height == 7 { 15 } else { 8 };
    let (border, constant) = match options.border {
        GpuGaussianBorder::Replicate => (0, [0; 4]),
        GpuGaussianBorder::Reflect => (1, [0; 4]),
        GpuGaussianBorder::Reflect101 => (2, [0; 4]),
        GpuGaussianBorder::Wrap => (3, [0; 4]),
        GpuGaussianBorder::Constant(pixel) => (4, pixel.map(u32::from)),
    };
    // Keep the original capacity checks above, including error precedence.
    let packed = bits == 8 && channels >= 2;
    let words_per_pixel = if packed { channels.div_ceil(2) } else { channels };
    let scratch_bytes = pixels * u64::from(words_per_pixel) * 4;
    Ok(Plan {
        params: Params {
            width,
            height,
            pixels: pixels as u32,
            groups: (pixels as u32)
                .div_ceil(WORKGROUP)
                .min(4096)
                .min(limits.max_compute_workgroups_per_dimension),
            kernel_width: options.kernel_width,
            kernel_height: options.kernel_height,
            border,
            bits,
            channels,
            padding: [if packed { words_per_pixel } else { 0 }, 0, 0],
            weights_x: weights(options.kernel_width, options.sigma_x, bits),
            weights_y: weights(options.kernel_height, options.sigma_y, bits),
            constant,
        },
        scratch_bytes,
        packed,
    })
}

pub(super) struct ParentGaussianPipelines {
    pipelines: std::sync::OnceLock<[wgpu::ComputePipeline; 2]>,
    packed: std::sync::OnceLock<[wgpu::ComputePipeline; 2]>,
}

impl ParentGaussianPipelines {
    pub(super) fn new() -> Self {
        Self { pipelines: std::sync::OnceLock::new(), packed: std::sync::OnceLock::new() }
    }

    pub(super) fn pipelines(
        &self,
        device: &wgpu::Device,
        packed: bool,
    ) -> &[wgpu::ComputePipeline; 2] {
        let cache = if packed { &self.packed } else { &self.pipelines };
        cache.get_or_init(|| {
            let shaders = if packed {
                [
                    concat!(
                        include_str!("gaussian_params.wgsl"),
                        include_str!("gaussian_packed_horizontal.wgsl")
                    ),
                    concat!(
                        include_str!("gaussian_params.wgsl"),
                        include_str!("gaussian_packed_vertical.wgsl")
                    ),
                ]
            } else {
                [
                    concat!(
                        include_str!("gaussian_params.wgsl"),
                        include_str!("gaussian_horizontal.wgsl")
                    ),
                    concat!(
                        include_str!("gaussian_params.wgsl"),
                        include_str!("gaussian_vertical.wgsl")
                    ),
                ]
            };
            shaders.map(|source| {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("gpu-gaussian-shader"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("gpu-gaussian-pipeline"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
        })
    }
}

pub(super) fn previous(
    cache: &ParentGaussianPipelines,
    runtime: &WgpuRuntime,
    source: &GpuImage,
    options: GpuGaussianBlurOptions,
    mut profiler: Option<&mut GpuStageProfiler>,
) -> SpatialResult<GpuImage> {
    source.validate_runtime(runtime)?;
    let p = plan(
        source.width(),
        source.height(),
        source.channels(),
        options,
        &runtime.device().limits(),
    )?;
    if let Some(ref profiler) = profiler {
        profiler.validate(runtime, 2)?;
    }
    let mut scratch = ScratchBuffers::new(runtime);
    let sums = scratch.acquire("gpu-gaussian-horizontal", p.scratch_bytes);
    let output = create_texture(runtime, source.width(), source.height(), "gpu-gaussian-output");
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-gaussian-params"),
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let source_view = source.view();
    let output_view = output.create_view(&Default::default());
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    let pipelines = cache.pipelines(runtime.device(), p.packed);
    for (index, pipeline) in pipelines.iter().enumerate() {
        let stage = if index == 0 { "gaussian_horizontal_gpu" } else { "gaussian_blur_gpu" };
        let image = if index == 0 { &source_view } else { &output_view };
        let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(stage),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(image),
                },
                wgpu::BindGroupEntry { binding: 2, resource: sums.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(stage),
            timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, stage),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(p.params.groups, 1, 1);
    }
    scratch.submit_and_recycle(encoder);
    let mut receipt = source.receipt().clone();
    receipt.record_gpu_to_gpu(p.scratch_bytes, "gaussian_horizontal_gpu");
    receipt.record_gpu_to_gpu(u64::from(p.params.pixels) * 4, "gaussian_blur_gpu");
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
