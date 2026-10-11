//! CPU-compatible grayscale histogram equalization without intermediate downloads.

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use wgpu::util::DeviceExt;

use super::gpu_image::create_texture;
use super::{
    histogram_cdf_gpu, histogram_cdf_gpu_profiled, histogram_u8_gpu, histogram_u8_gpu_profiled,
};
use super::{GpuImage, GpuImageCdf};
use crate::{scratch::ScratchBuffers, GpuStageProfiler, WgpuRuntime};

const LOOKUP_BYTES: u64 = 1024;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    groups_x: u32,
    groups_y: u32,
}

pub(crate) struct EqualizePipelines {
    lookup_layout: wgpu::BindGroupLayout,
    lookup: wgpu::ComputePipeline,
    apply_layout: wgpu::BindGroupLayout,
    apply: wgpu::ComputePipeline,
}

impl EqualizePipelines {
    fn new(device: &wgpu::Device) -> Self {
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let lookup_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu-equalize-lookup-layout"),
            entries: &[
                buffer(0, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer(1, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let apply_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu-equalize-apply-layout"),
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
        let pipeline = |layout: &wgpu::BindGroupLayout, text: &'static str, entry: &'static str| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(text.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        Self {
            lookup: pipeline(
                &lookup_layout,
                concat!(include_str!("rounding_u32.wgsl"), include_str!("equalize_lookup.wgsl")),
                "equalize_lookup",
            ),
            apply: pipeline(&apply_layout, include_str!("equalize_apply.wgsl"), "equalize_apply"),
            lookup_layout,
            apply_layout,
        }
    }
}

fn plan(width: u32, height: u32, limits: &wgpu::Limits) -> SpatialResult<(u32, u32)> {
    super::histogram::dispatch_plan(width, height, limits)?;
    if width > limits.max_texture_dimension_2d
        || height > limits.max_texture_dimension_2d
        || width > i32::MAX as u32
        || height > i32::MAX as u32
        || limits.max_storage_textures_per_shader_stage == 0
        || limits.max_sampled_textures_per_shader_stage == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_compute_workgroup_size_y < 16
    {
        return Err(invalid("GPU equalization requires supported texture dimensions, sampled/storage textures and uniforms"));
    }
    let axis =
        |size: u32| size.div_ceil(16).min(64).min(limits.max_compute_workgroups_per_dimension);
    Ok((axis(width), axis(height)))
}

/// Equalizes a resident one-channel u8 image with CPU-compatible byte rounding.
///
/// Histogram, inclusive CDF, lookup and image application stay on the GPU.
/// Constant images preserve their original byte. Metadata is preserved. Five
/// compute passes run without payload readback or implicit host synchronization.
/// Binary64 rounding behavior of the CPU reference is emulated with u32 words;
/// no shader float64 capability is required. Input is limited to u32::MAX pixels.
pub fn equalize_histogram_gpu(runtime: &WgpuRuntime, source: &GpuImage) -> SpatialResult<GpuImage> {
    equalize_with_profiler(runtime, source, None)
}

/// Equalizes a grayscale image while recording exactly five GPU pass timestamps.
/// Requires five unused stages on the same runtime before allocating work.
pub fn equalize_histogram_gpu_profiled(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    profiler: &mut GpuStageProfiler,
) -> SpatialResult<GpuImage> {
    profiler.validate(runtime, 5)?;
    let result = equalize_with_profiler(runtime, source, Some(profiler));
    profiler.finish_recording(&result);
    result
}

fn equalize_with_profiler(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    mut profiler: Option<&mut GpuStageProfiler>,
) -> SpatialResult<GpuImage> {
    source.validate_runtime(runtime)?;
    if source.channels() != 1 {
        return Err(invalid("GPU histogram equalization requires one logical channel"));
    }
    let groups = plan(source.width(), source.height(), &runtime.device().limits())?;
    let histogram = match profiler.as_deref_mut() {
        Some(p) => histogram_u8_gpu_profiled(runtime, source, 0, p)?,
        None => histogram_u8_gpu(runtime, source, 0)?,
    };
    let cdf = match profiler.as_deref_mut() {
        Some(p) => histogram_cdf_gpu_profiled(runtime, &histogram, p)?,
        None => histogram_cdf_gpu(runtime, &histogram)?,
    };
    let result = apply_equalization(runtime, source, &cdf, groups, profiler);
    cdf.recycle(runtime);
    histogram.recycle(runtime);
    result
}

fn apply_equalization(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    cdf: &GpuImageCdf,
    groups: (u32, u32),
    mut profiler: Option<&mut GpuStageProfiler>,
) -> SpatialResult<GpuImage> {
    let pipelines =
        runtime.image_equalize_pipelines.get_or_init(|| EqualizePipelines::new(runtime.device()));
    let mut scratch = ScratchBuffers::new(runtime);
    let lookup = scratch.acquire("gpu-equalize-lookup", LOOKUP_BYTES);
    let output = create_texture(runtime, source.width(), source.height(), "gpu-equalize-output");
    let params = Params {
        width: source.width(),
        height: source.height(),
        groups_x: groups.0,
        groups_y: groups.1,
    };
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-equalize-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let lookup_group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-equalize-lookup-group"),
        layout: &pipelines.lookup_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cdf.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: lookup.as_entire_binding() },
        ],
    });
    let source_view = source.view();
    let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
    let apply_group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-equalize-apply-group"),
        layout: &pipelines.apply_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&source_view),
            },
            wgpu::BindGroupEntry { binding: 2, resource: lookup.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&output_view),
            },
        ],
    });
    let mut encoder = runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-equalize-encoder"),
    });
    for (pipeline, bind_group, stage, groups) in [
        (&pipelines.lookup, &lookup_group, "equalize_histogram_lookup_gpu", (1, 1)),
        (&pipelines.apply, &apply_group, "equalize_histogram_apply_gpu", groups),
    ] {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(stage),
            timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, stage),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }
    scratch.submit_and_recycle(encoder);
    let mut receipt = cdf.receipt().clone();
    receipt.record_gpu_to_gpu(LOOKUP_BYTES, "equalize_histogram_lookup_gpu");
    receipt.record_gpu_to_gpu(source.storage_bytes(), "equalize_histogram_apply_gpu");
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

fn invalid(message: &str) -> SpatialError {
    SpatialError::InvalidArgument(message.to_owned())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::super::gpu_image::read_staging_bytes;
    use super::*;

    #[test]
    fn equalization_plan_bounds_counters_textures_and_dispatch_before_allocation() {
        let limits = wgpu::Limits::default();
        assert_eq!(plan(1, 1, &limits).unwrap(), (1, 1));
        assert_eq!(plan(3840, 2160, &limits).unwrap(), (64, 64));
        let mut reduced = limits.clone();
        reduced.max_compute_workgroups_per_dimension = 3;
        assert_eq!(plan(3840, 2160, &reduced).unwrap(), (3, 3));
        for (width, height) in
            [(0, 1), (1, 0), (65_536, 65_536), (limits.max_texture_dimension_2d + 1, 1)]
        {
            assert!(plan(width, height, &limits).is_err());
        }
        for field in 0..5 {
            let mut invalid = limits.clone();
            match field {
                0 => invalid.max_storage_textures_per_shader_stage = 0,
                1 => invalid.max_sampled_textures_per_shader_stage = 0,
                2 => invalid.max_uniform_buffers_per_shader_stage = 0,
                3 => invalid.max_compute_invocations_per_workgroup = 255,
                _ => invalid.max_compute_workgroup_size_y = 15,
            }
            assert!(plan(17, 19, &invalid).is_err());
        }
    }

    fn rounded(value: f64) -> u32 {
        let floor = value.floor();
        let fraction = value - floor;
        (floor as u32) + u32::from(fraction > 0.5 || (fraction == 0.5 && floor as u32 % 2 != 0))
    }

    fn gcd(mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }

    #[test]
    fn gpu_lookup_matches_cpu_binary64_at_half_boundaries_and_maximum_counts() {
        let runtime = WgpuRuntime::new_headless().unwrap();
        let pipelines = EqualizePipelines::new(runtime.device());
        let mut denominators: Vec<_> = (1..=4096).collect();
        denominators.extend([
            50,
            58,
            162,
            210,
            1_000_000,
            8_294_399,
            16_777_214,
            16_777_216,
            2_147_483_646,
            u32::MAX - 1,
        ]);
        let mut seed = 19u32;
        for _ in 0..512 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            denominators.push(seed.clamp(1, u32::MAX - 1));
        }
        let mut inputs = Vec::new();
        let mut expected = Vec::new();
        for d in denominators {
            let mut numerators: Vec<_> =
                (0..=255).map(|i| (u64::from(d) * i / 255) as u32).collect();
            let step = 510 / gcd(d, 510);
            if step % 2 == 1 {
                for odd in (step..510).step_by((2 * step) as usize) {
                    let n = (u64::from(odd) * u64::from(d) / 510) as u32;
                    numerators.extend([n.saturating_sub(1), n, (n + 1).min(d)]);
                }
            }
            numerators.sort_unstable();
            numerators.dedup();
            // Each test CDF is monotone, starts with one minimum-valued pixel,
            // and ends at a valid checked total d+1. Blocks cover every case.
            for chunk in numerators.chunks(254) {
                let mut cdf = [1u32; 256];
                let mut lookup = [0u32; 256];
                for (i, &n) in chunk.iter().enumerate() {
                    cdf[i + 1] = n + 1;
                    lookup[i + 1] = rounded(f64::from(n) * (255.0 / f64::from(d)));
                }
                for i in chunk.len() + 1..256 {
                    cdf[i] = d + 1;
                    lookup[i] = 255;
                }
                inputs.extend(cdf);
                expected.extend(lookup);
            }
        }
        // Constant CDFs preserve the source byte via an identity lookup.
        inputs.extend([17u32; 256]);
        expected.extend(0..256u32);
        let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("equalize-test-cdfs"),
            contents: bytemuck::cast_slice(&inputs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("equalize-test-lookups"),
            size: (inputs.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("equalize-test-readback"),
            size: output.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        for row in 0..inputs.len() / 256 {
            fn binding(buffer: &wgpu::Buffer, row: usize) -> wgpu::BufferBinding<'_> {
                wgpu::BufferBinding {
                    buffer,
                    offset: row as u64 * LOOKUP_BYTES,
                    size: std::num::NonZeroU64::new(LOOKUP_BYTES),
                }
            }
            let bind_group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("equalize-test-group"),
                layout: &pipelines.lookup_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(binding(&input, row)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(binding(&output, row)),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipelines.lookup);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output.size());
        runtime.queue().submit(Some(encoder.finish()));
        let bytes = read_staging_bytes(runtime.device(), &staging, output.size() as usize).unwrap();
        for (i, (word, expected)) in bytes.chunks_exact(4).zip(expected).enumerate() {
            let actual = u32::from_le_bytes(word.try_into().unwrap());
            assert_eq!(
                actual,
                expected,
                "CDF row {}, bin {}, count {}",
                i / 256,
                i % 256,
                inputs[i]
            );
        }
        eprintln!(
            "Verified {} GPU lookup entries against independent CPU binary64 rounding",
            inputs.len()
        );
    }
}
