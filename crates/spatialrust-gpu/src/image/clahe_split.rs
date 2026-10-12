//! Private bounded split-tile histograms; unsupported or small work stays direct.
use super::*;

pub(super) const MIN_TILE_AREA: u64 = 65_536;
pub(super) const MAX_SPLIT_TILES: u32 = 16;
const TARGET_CHUNK_AREA: u64 = 8192;
const MAX_CHUNKS: u64 = 32;

const HISTOGRAM_SHADER: &str =
    concat!(include_str!("clahe_split_params.wgsl"), include_str!("clahe_split_histogram.wgsl"));
pub(super) const LOOKUP_SHADER: &str = concat!(
    include_str!("rounding_u32.wgsl"),
    include_str!("clahe_split_params.wgsl"),
    include_str!("clahe_clip_scan.wgsl"),
    include_str!("clahe_split_lookup.wgsl")
);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct Params {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) tile_width: u32,
    pub(super) tile_height: u32,
    pub(super) tiles_x: u32,
    pub(super) tile_count: u32,
    pub(super) chunks: u32,
    pub(super) partial_groups: u32,
    pub(super) clip_count: u32,
    pub(super) tile_groups: u32,
    pub(super) padding: [u32; 2],
}

pub(super) struct Plan {
    pub(super) params: Params,
    pub(super) partial_bytes: u64,
}

pub(super) fn plan(p: &super::Params, limits: &wgpu::Limits) -> Option<Plan> {
    let area = u64::from(p.tile_width) * u64::from(p.tile_height);
    if area < MIN_TILE_AREA || p.tile_count > MAX_SPLIT_TILES {
        return None;
    }
    let chunks = area.div_ceil(TARGET_CHUNK_AREA).min(MAX_CHUNKS);
    let partials = u64::from(p.tile_count).checked_mul(chunks)?;
    let entries = partials.checked_mul(256)?;
    let partial_bytes = entries.checked_mul(4)?;
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    if entries > u64::from(u32::MAX) || partial_bytes > bound {
        return None;
    }
    Some(Plan {
        params: Params {
            width: p.width,
            height: p.height,
            tile_width: p.tile_width,
            tile_height: p.tile_height,
            tiles_x: p.tiles_x,
            tile_count: p.tile_count,
            chunks: chunks as u32,
            partial_groups: partials
                .min(512)
                .min(u64::from(limits.max_compute_workgroups_per_dimension))
                as u32,
            clip_count: p.clip_count,
            tile_groups: p.tile_groups,
            padding: [0; 2],
        },
        partial_bytes,
    })
}

pub(super) struct Pipelines {
    pub(super) layouts: [wgpu::BindGroupLayout; 2],
    pub(super) pipelines: [wgpu::ComputePipeline; 2],
}

impl Pipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let uniform = buffer(0, wgpu::BufferBindingType::Uniform);
        let output = buffer(2, wgpu::BufferBindingType::Storage { read_only: false });
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
        let layouts = [
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("clahe-partial-histogram-layout"),
                entries: &[uniform, texture, output],
            }),
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("clahe-merged-lookup-layout"),
                entries: &[
                    uniform,
                    buffer(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    output,
                ],
            }),
        ];
        let pipelines = std::array::from_fn(|i| {
            let (source, entry) = [
                (HISTOGRAM_SHADER, "clahe_partial_histogram"),
                (LOOKUP_SHADER, "clahe_merged_lookup"),
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
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Self { layouts, pipelines }
    }
}
