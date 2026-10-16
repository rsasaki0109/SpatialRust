// Frozen complete GPU median from merged PR 253; native tests only.
//! Exact per-channel median selection on resident integer textures.
use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use std::{fmt::Write, sync::OnceLock};
use wgpu::util::DeviceExt;

use super::GpuImageBorder;
use crate::image::gpu_image::{create_texture, GpuImage};
use crate::WgpuRuntime;

const WORKGROUP: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    border: u32,
    padding: u32,
}

fn plan(width: u32, height: u32, kernel: u32, limits: &wgpu::Limits) -> SpatialResult<u64> {
    let radius = kernel / 2;
    if !matches!(kernel, 3 | 5)
        || width == 0
        || height == 0
        || width > limits.max_texture_dimension_2d
        || height > limits.max_texture_dimension_2d
        || u64::from(width - 1) + u64::from(radius) > i32::MAX as u64
        || u64::from(height - 1) + u64::from(radius) > i32::MAX as u64
        || limits.max_compute_workgroup_size_x < WORKGROUP
        || limits.max_compute_workgroup_size_y < WORKGROUP
        || limits.max_compute_workgroup_size_z == 0
        || limits.max_compute_invocations_per_workgroup < WORKGROUP * WORKGROUP
        || width.div_ceil(WORKGROUP) > limits.max_compute_workgroups_per_dimension
        || height.div_ceil(WORKGROUP) > limits.max_compute_workgroups_per_dimension
        || limits.max_sampled_textures_per_shader_stage == 0
        || limits.max_storage_textures_per_shader_stage == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < 16
        || limits.max_buffer_size < 16
        || limits.max_bind_groups == 0
        || limits.max_bindings_per_bind_group < 3
    {
        return Err(SpatialError::InvalidArgument(
            "GPU median requires a 3x3 or 5x5 kernel and supported texture/coordinate/compute limits"
                .to_owned(),
        ));
    }
    Ok(u64::from(width) * u64::from(height) * 4)
}

#[derive(Clone, Copy)]
pub(super) struct Comparator {
    pub(super) a: usize,
    pub(super) b: usize,
    pub(super) ascending: bool,
}

// Use a short 3x3 selection network. For 5x5, prune a padded bitonic sorting
// network to the median output; padding is the largest byte in each lane.
pub(super) fn network(kernel: u32) -> Vec<Comparator> {
    if kernel == 3 {
        // Sort three triples; the median is the median of the largest minimum,
        // middle median and smallest maximum. The three groups are disjoint.
        return [
            (0, 1),
            (1, 2),
            (0, 1),
            (3, 4),
            (4, 5),
            (3, 4),
            (6, 7),
            (7, 8),
            (6, 7),
            (0, 3),
            (3, 6),
            (1, 4),
            (4, 7),
            (1, 4),
            (5, 8),
            (2, 5),
            (2, 4),
            (4, 6),
            (2, 4),
        ]
        .into_iter()
        .map(|(a, b)| Comparator { a, b, ascending: true })
        .collect();
    }
    let count = (kernel * kernel) as usize;
    let padded = count.next_power_of_two();
    let mut pairs = Vec::new();
    let mut width = 2;
    while width <= padded {
        let mut stride = width / 2;
        while stride != 0 {
            for a in 0..padded {
                let b = a ^ stride;
                if b > a {
                    pairs.push(Comparator { a, b, ascending: a & width == 0 });
                }
            }
            stride /= 2;
        }
        width *= 2;
    }
    let mut needed = vec![false; padded];
    needed[count / 2] = true;
    let mut selected = Vec::new();
    for pair in pairs.into_iter().rev() {
        if needed[pair.a] || needed[pair.b] {
            needed[pair.a] = true;
            needed[pair.b] = true;
            selected.push(pair);
        }
    }
    selected.reverse();
    selected
}

pub(super) fn shader(kernel: u32) -> String {
    let mut source = String::from(include_str!("median_params.wgsl"));
    source.push_str("@compute @workgroup_size(16, 16)\nfn median(@builtin(global_invocation_id) gid: vec3<u32>) {\nif (gid.x >= params.width || gid.y >= params.height) { return; }\nlet xy = vec2<i32>(gid.xy);\n");
    let count = kernel * kernel;
    let radius = (kernel / 2) as i32;
    for i in 0..count.next_power_of_two() {
        if i < count {
            writeln!(
                source,
                "var s{i} = sample_at(xy + vec2<i32>({}, {}));",
                (i % kernel) as i32 - radius,
                (i / kernel) as i32 - radius
            )
            .unwrap();
        } else {
            writeln!(source, "var s{i} = vec4<u32>(255u);").unwrap();
        }
    }
    for Comparator { a, b, ascending } in network(kernel) {
        let (low, high) = if ascending { (a, b) } else { (b, a) };
        writeln!(source, "{{ let old = s{a}; let other = s{b}; s{low} = min(old, other); s{high} = max(old, other); }}").unwrap();
    }
    writeln!(source, "textureStore(output_px, xy, s{});\n}}", count / 2).unwrap();
    source
}

pub(crate) struct MedianPipelines {
    layout: wgpu::BindGroupLayout,
    kernels: [OnceLock<wgpu::ComputePipeline>; 2],
}

impl MedianPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu-median-layout"),
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
        Self { layout, kernels: std::array::from_fn(|_| OnceLock::new()) }
    }

    fn kernel(&self, device: &wgpu::Device, kernel: u32) -> &wgpu::ComputePipeline {
        self.kernels[usize::from(kernel == 5)].get_or_init(|| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpu-median-selection"),
                source: wgpu::ShaderSource::Wgsl(shader(kernel).into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu-median-pipeline-layout"),
                bind_group_layouts: &[&self.layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpu-median"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("median"),
                compilation_options: Default::default(),
                cache: None,
            })
        })
    }
}

/// Selects the exact per-channel median of a 3x3 or 5x5 square neighborhood.
/// Supports all 1..=4 logical u8 channels and retains metadata. Each component,
/// including alpha, is filtered independently.
/// Replicate and constant-zero borders match the corresponding CPU median.
/// One ordered device pass returns an owned image without host transfer/wait
/// or image-sized scratch. Upload/readback remain explicit caller operations.
/// Runtime, kernel and device limits reject before output/cache allocation.
pub(super) fn previous(
    runtime: &WgpuRuntime,
    previous_cache: &OnceLock<MedianPipelines>,
    source: &GpuImage,
    kernel_size: u32,
    border: GpuImageBorder,
) -> SpatialResult<GpuImage> {
    source.validate_runtime(runtime)?;
    let bytes = plan(source.width(), source.height(), kernel_size, &runtime.device().limits())?;
    let cache = previous_cache.get_or_init(|| MedianPipelines::new(runtime.device()));
    let pipeline = cache.kernel(runtime.device(), kernel_size);
    let output = create_texture(runtime, source.width(), source.height(), "gpu-median-output");
    let params = Params {
        width: source.width(),
        height: source.height(),
        border: u32::from(border == GpuImageBorder::ConstantZero),
        padding: 0,
    };
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-median-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-median-group"),
        layout: &cache.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&source.view()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(
                    &output.create_view(&Default::default()),
                ),
            },
        ],
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("median_blur_gpu"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(
            source.width().div_ceil(WORKGROUP),
            source.height().div_ceil(WORKGROUP),
            1,
        );
    }
    runtime.queue().submit(Some(encoder.finish()));
    let mut receipt = source.receipt().clone();
    receipt.record_gpu_to_gpu(bytes, "median_blur_gpu");
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
