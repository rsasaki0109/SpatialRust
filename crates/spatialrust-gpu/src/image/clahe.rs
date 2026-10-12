//! Resident CPU-compatible contrast-limited adaptive histogram equalization.

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

use super::{gpu_image::create_texture, GpuImage};
use crate::{scratch::ScratchBuffers, GpuStageProfiler, WgpuRuntime};

#[path = "clahe_split.rs"]
mod split;

const GEOMETRY_SHADER: &str = concat!(
    include_str!("rounding_u32.wgsl"),
    include_str!("binary64_positive.wgsl"),
    include_str!("clahe_params.wgsl"),
    include_str!("clahe_geometry.wgsl")
);
const LOOKUP_SHADER: &str = concat!(
    include_str!("rounding_u32.wgsl"),
    include_str!("clahe_params.wgsl"),
    include_str!("clahe_clip_scan.wgsl"),
    include_str!("clahe_lookup.wgsl")
);
const APPLY_SHADER: &str = concat!(
    include_str!("rounding_u32.wgsl"),
    include_str!("binary64_positive.wgsl"),
    include_str!("clahe_params.wgsl"),
    include_str!("clahe_apply.wgsl")
);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    tiles_x: u32,
    tiles_y: u32,
    clip_count: u32,
    tile_count: u32,
    tile_groups: u32,
    geometry_groups: u32,
    groups_x: u32,
    groups_y: u32,
}

struct Plan {
    params: Params,
    geometry_bytes: u64,
    lookup_bytes: u64,
    split: Option<split::Plan>,
}

fn plan(
    width: u32,
    height: u32,
    clip_limit: f64,
    tiles_x: u32,
    tiles_y: u32,
    limits: &wgpu::Limits,
) -> SpatialResult<Plan> {
    let invalid = || {
        error("GPU CLAHE requires positive supported dimensions/grid, finite non-negative clip limit, bounded u32 tile counts and supported compute/buffer limits")
    };
    if width == 0
        || height == 0
        || tiles_x == 0
        || tiles_y == 0
        || !clip_limit.is_finite()
        || clip_limit < 0.0
        || width > limits.max_texture_dimension_2d
        || height > limits.max_texture_dimension_2d
        || width > i32::MAX as u32
        || height > i32::MAX as u32
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_workgroup_size_y < 16
        || limits.max_compute_workgroup_storage_size < 4096
        || limits.max_compute_workgroups_per_dimension == 0
        || limits.max_storage_buffers_per_shader_stage < 2
        || limits.max_sampled_textures_per_shader_stage == 0
        || limits.max_storage_textures_per_shader_stage == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < std::mem::size_of::<Params>() as u32
    {
        return Err(invalid());
    }
    let tile_width = width.div_ceil(tiles_x);
    let tile_height = height.div_ceil(tiles_y);
    let area = u64::from(tile_width) * u64::from(tile_height);
    let tile_count = u64::from(tiles_x) * u64::from(tiles_y);
    let geometry_count = u64::from(width) + u64::from(height);
    let geometry_bytes = geometry_count * 16;
    let lookup_bytes = tile_count.checked_mul(1024).ok_or_else(invalid)?;
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    if area > u64::from(u32::MAX)
        || tile_count > u64::from(u32::MAX)
        || geometry_count > u64::from(u32::MAX)
        || u64::from(tiles_x) * u64::from(tile_width) > u64::from(u32::MAX)
        || u64::from(tiles_y) * u64::from(tile_height) > u64::from(u32::MAX)
        || geometry_bytes > bound
        || lookup_bytes > bound
    {
        return Err(invalid());
    }
    // Equivalent to the CPU's usize saturation: counts cannot exceed tile area.
    // Huge finite clip limits may overflow the intermediate f64 product to +inf.
    let clip_count = if clip_limit > 0.0 {
        (clip_limit * area as f64 / 256.0).max(1.0).min(area as f64) as u32
    } else {
        0
    };
    let groups = |count: u32, lanes: u32, cap: u32| {
        count.div_ceil(lanes).min(cap).min(limits.max_compute_workgroups_per_dimension)
    };
    let params = Params {
        width,
        height,
        tile_width,
        tile_height,
        tiles_x,
        tiles_y,
        clip_count,
        tile_count: tile_count as u32,
        tile_groups: groups(tile_count as u32, 1, 256),
        geometry_groups: groups(geometry_count as u32, 256, 256),
        groups_x: groups(width, 16, 64),
        groups_y: groups(height, 16, 64),
    };
    Ok(Plan { split: split::plan(&params, limits), params, geometry_bytes, lookup_bytes })
}

pub(crate) struct ClahePipelines {
    layouts: [wgpu::BindGroupLayout; 3],
    pipelines: [wgpu::ComputePipeline; 3],
    split: OnceLock<split::Pipelines>,
}

impl ClahePipelines {
    fn new(device: &wgpu::Device) -> Self {
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let texture = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let uniform = buffer(0, wgpu::BufferBindingType::Uniform);
        let writable = wgpu::BufferBindingType::Storage { read_only: false };
        let readonly = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = |entries| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu-clahe-layout"),
                entries,
            })
        };
        let layouts = [
            layout(&[uniform, buffer(1, writable)]),
            layout(&[uniform, texture, buffer(2, writable)]),
            layout(&[
                uniform,
                texture,
                buffer(2, readonly),
                buffer(3, readonly),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ]),
        ];
        let pipelines = std::array::from_fn(|i| {
            let (source, entry) = [
                (GEOMETRY_SHADER, "clahe_geometry"),
                (LOOKUP_SHADER, "clahe_lookup"),
                (APPLY_SHADER, "clahe_apply"),
            ][i];
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[&layouts[i]],
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
        });
        Self { layouts, pipelines, split: OnceLock::new() }
    }
}

/// Applies CPU-compatible CLAHE to a resident one-channel u8 image.
///
/// Tile histograms use Reflect101 padding, exact clipping/redistribution and CPU
/// binary64 lookup rounding. Bilinear interpolation reproduces the CPU operation
/// order using integer-word binary64 arithmetic, without shader float64 support.
/// Metadata is preserved. Three passes use one submission, with no payload
/// readback or implicit host wait. Clip limit zero disables histogram clipping;
/// CLAHE can change constant-image bytes, just as the CPU implementation does.
/// Each tile has at most u32::MAX padded pixels; lookup and axis geometry buffers
/// must fit device binding limits. Empty images and non-grayscale input reject.
pub fn clahe_gpu(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    clip_limit: f64,
    tiles_x: u32,
    tiles_y: u32,
) -> SpatialResult<GpuImage> {
    clahe_with_profiler(runtime, source, clip_limit, tiles_x, tiles_y, None)
}

/// Applies resident CLAHE while recording three or four pass timestamps.
/// Split histograms require four unused stages on the same runtime; direct
/// histograms require three. Capacity is checked before allocating work.
pub fn clahe_gpu_profiled(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    clip_limit: f64,
    tiles_x: u32,
    tiles_y: u32,
    profiler: &mut GpuStageProfiler,
) -> SpatialResult<GpuImage> {
    let stages = plan(
        source.width(),
        source.height(),
        clip_limit,
        tiles_x,
        tiles_y,
        &runtime.device().limits(),
    )
    .map_or(3, |p| if p.split.is_some() { 4 } else { 3 });
    profiler.validate(runtime, stages)?;
    let result = clahe_with_profiler(runtime, source, clip_limit, tiles_x, tiles_y, Some(profiler));
    profiler.finish_recording(&result);
    result
}

fn clahe_with_profiler(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    clip_limit: f64,
    tiles_x: u32,
    tiles_y: u32,
    profiler: Option<&mut GpuStageProfiler>,
) -> SpatialResult<GpuImage> {
    source.validate_runtime(runtime)?;
    if source.channels() != 1 {
        return Err(error("GPU CLAHE requires one logical channel"));
    }
    let plan = plan(
        source.width(),
        source.height(),
        clip_limit,
        tiles_x,
        tiles_y,
        &runtime.device().limits(),
    )?;
    let pipelines =
        runtime.image_clahe_pipelines.get_or_init(|| ClahePipelines::new(runtime.device()));
    execute_clahe(runtime, source, &plan, pipelines, &pipelines.pipelines[1], profiler)
}

fn execute_clahe(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    plan: &Plan,
    pipelines: &ClahePipelines,
    direct_lookup: &wgpu::ComputePipeline,
    mut profiler: Option<&mut GpuStageProfiler>,
) -> SpatialResult<GpuImage> {
    let mut scratch = ScratchBuffers::new(runtime);
    let geometry = scratch.acquire("gpu-clahe-geometry", plan.geometry_bytes);
    let lookup = scratch.acquire("gpu-clahe-lookups", plan.lookup_bytes);
    let partials = plan
        .split
        .as_ref()
        .map(|p| scratch.acquire("gpu-clahe-partial-histograms", p.partial_bytes));
    let output = create_texture(runtime, source.width(), source.height(), "gpu-clahe-output");
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-clahe-params"),
        contents: bytemuck::bytes_of(&plan.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let split_uniform = plan.split.as_ref().map(|p| {
        runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu-clahe-split-params"),
            contents: bytemuck::bytes_of(&p.params),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    });
    let source_view = source.view();
    let output_view = output.create_view(&Default::default());
    fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
    }
    let entries = [
        vec![entry(0, &uniform), entry(1, &geometry)],
        vec![
            entry(0, &uniform),
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&source_view),
            },
            entry(2, &lookup),
        ],
        vec![
            entry(0, &uniform),
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&source_view),
            },
            entry(2, &geometry),
            entry(3, &lookup),
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&output_view),
            },
        ],
    ];
    let [geometry_entries, direct_entries, apply_entries] = entries;
    let mut passes = vec![(
        "clahe_geometry_gpu",
        &pipelines.layouts[0],
        &pipelines.pipelines[0],
        geometry_entries,
        plan.params.geometry_groups,
        1,
    )];
    if let Some(p) = &plan.split {
        let split_pipelines =
            pipelines.split.get_or_init(|| split::Pipelines::new(runtime.device()));
        let split_uniform = split_uniform.as_ref().unwrap();
        let partials = partials.as_ref().unwrap();
        passes.push((
            "clahe_histogram_gpu",
            &split_pipelines.layouts[0],
            &split_pipelines.pipelines[0],
            vec![
                entry(0, split_uniform),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                entry(2, partials),
            ],
            p.params.partial_groups,
            1,
        ));
        passes.push((
            "clahe_lookup_gpu",
            &split_pipelines.layouts[1],
            &split_pipelines.pipelines[1],
            vec![entry(0, split_uniform), entry(1, partials), entry(2, &lookup)],
            p.params.tile_groups,
            1,
        ));
    } else {
        passes.push((
            "clahe_lookup_gpu",
            &pipelines.layouts[1],
            direct_lookup,
            direct_entries,
            plan.params.tile_groups,
            1,
        ));
    }
    passes.push((
        "clahe_apply_gpu",
        &pipelines.layouts[2],
        &pipelines.pipelines[2],
        apply_entries,
        plan.params.groups_x,
        plan.params.groups_y,
    ));
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    for (stage, layout, pipeline, entries, x, y) in passes {
        let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(stage),
            layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(stage),
            timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, stage),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }
    scratch.submit_and_recycle(encoder);
    let mut receipt = source.receipt().clone();
    receipt.record_gpu_to_gpu(plan.geometry_bytes, "clahe_geometry_gpu");
    if let Some(p) = &plan.split {
        receipt.record_gpu_to_gpu(p.partial_bytes, "clahe_histogram_gpu");
    }
    receipt.record_gpu_to_gpu(plan.lookup_bytes, "clahe_lookup_gpu");
    receipt.record_gpu_to_gpu(source.storage_bytes(), "clahe_apply_gpu");
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

fn error(message: &str) -> SpatialError {
    SpatialError::InvalidArgument(message.to_owned())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "clahe_tests.rs"]
mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "clahe_lookup_tests.rs"]
mod lookup_tests;
