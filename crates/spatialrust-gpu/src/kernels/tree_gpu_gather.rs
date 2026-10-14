//! Explicit SoA upload and integer GPU gather into existing tree point records.
use super::{buffer_bytes, check_compute, invalid, tree_staging};
use crate::WgpuRuntime;
use spatialrust_core::SpatialResult;
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

pub(crate) struct TreeGatherPipeline {
    pipelines: [OnceLock<wgpu::ComputePipeline>; 3],
}

impl TreeGatherPipeline {
    fn new() -> Self {
        Self { pipelines: std::array::from_fn(|_| OnceLock::new()) }
    }

    fn pipeline(
        &self,
        device: &wgpu::Device,
        with_queries: bool,
        with_samples: bool,
    ) -> &wgpu::ComputePipeline {
        let variant = if with_samples { 2 } else { usize::from(with_queries) };
        self.pipelines[variant].get_or_init(|| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("tree-point-gather-shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/tree_point_gather.wgsl").into(),
                ),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tree-point-gather-pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some(if with_samples {
                    "gather_with_samples"
                } else if with_queries {
                    "gather_with_queries"
                } else {
                    "gather"
                }),
                compilation_options: Default::default(),
                cache: None,
            })
        })
    }
}

// A bounded grid-stride pass also accepts trees larger than a one-axis dispatch.
fn plan(count: u64, limits: &wgpu::Limits) -> SpatialResult<(u64, u32)> {
    check_compute(limits, 0)?;
    let bytes = buffer_bytes(count, 16, limits)?;
    if limits.max_compute_workgroups_per_dimension == 0 {
        return Err(invalid());
    }
    let groups =
        count.div_ceil(64).min(u64::from(limits.max_compute_workgroups_per_dimension)).min(65_535);
    // max_storage_buffer_binding_size is u32. A 16-byte output therefore bounds
    // count + the capped stride well below u32::MAX, including the final add.
    Ok((bytes, groups as u32))
}

pub(super) fn upload(
    runtime: &WgpuRuntime,
    axes: [&[f32]; 3],
    order: &[u32],
    with_queries: bool,
) -> SpatialResult<(wgpu::Buffer, Option<wgpu::Buffer>)> {
    upload_impl(runtime, axes, order, with_queries, None)
        .map(|(points, queries, _)| (points, queries))
}

pub(super) fn upload_with_samples(
    runtime: &WgpuRuntime,
    axes: [&[f32]; 3],
    order: &[u32],
    normals: &[[f32; 3]],
) -> SpatialResult<(wgpu::Buffer, Option<wgpu::Buffer>, Option<wgpu::Buffer>)> {
    upload_impl(runtime, axes, order, true, Some(normals))
}

fn upload_impl(
    runtime: &WgpuRuntime,
    axes: [&[f32]; 3],
    order: &[u32],
    with_queries: bool,
    normals: Option<&[[f32; 3]]>,
) -> SpatialResult<(wgpu::Buffer, Option<wgpu::Buffer>, Option<wgpu::Buffer>)> {
    let count = axes[0].len();
    if axes.iter().any(|axis| axis.len() != count) || order.len() != count || count == 0 {
        return Err(invalid());
    }
    let (bytes, groups) = plan(count as u64, &runtime.device().limits())?;
    if let Some(normals) = normals {
        if normals.len() != count || !with_queries {
            return Err(invalid());
        }
        check_samples(count as u64, &runtime.device().limits())?;
    }
    tree_staging::validate(axes)?;
    let upload = |label, data: &[u8]| {
        runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    // Copy the three borrowed columns directly into one mapped SoA buffer.
    // This needs no host concatenation vector or per-point CPU gather.
    let positions = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("tree-gather-soa"),
        size: count as u64 * if normals.is_some() { 24 } else { 12 },
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: true,
    });
    {
        let mut mapped = positions.slice(..).get_mapped_range_mut();
        let axis_bytes = count * 4;
        for (axis, values) in axes.iter().enumerate() {
            mapped[axis * axis_bytes..(axis + 1) * axis_bytes]
                .copy_from_slice(bytemuck::cast_slice(values));
        }
        if let Some(normals) = normals {
            mapped[3 * axis_bytes..].copy_from_slice(bytemuck::cast_slice(normals));
        }
    }
    positions.unmap();
    let order = upload("tree-gather-order", bytemuck::cast_slice(order));
    let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("nearest-points"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let queries = with_queries.then(|| {
        runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("neighborhood-queries"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    });
    let samples = normals.map(|_| {
        runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("fpfh-samples"),
            size: count as u64 * 32,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    });
    let params = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tree-gather-params"),
        contents: bytemuck::cast_slice(&[count as u32, 0, 0, 0]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let cache = runtime.tree_gather_pipeline.get_or_init(TreeGatherPipeline::new);
    let pipeline = cache.pipeline(runtime.device(), with_queries, normals.is_some());
    let mut buffers = vec![&params, &positions, &order, &output];
    if let Some(queries) = queries.as_ref() {
        buffers.push(queries);
    }
    if let Some(samples) = samples.as_ref() {
        buffers.push(samples);
    }
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tree-gather-bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tree-point-gather"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
    Ok((output, queries, samples))
}

pub(super) fn check(count: u64, limits: &wgpu::Limits) -> SpatialResult<()> {
    plan(count, limits).map(|_| ())
}

pub(super) fn check_samples(count: u64, limits: &wgpu::Limits) -> SpatialResult<()> {
    check(count, limits)?;
    check_compute(limits, count)?;
    buffer_bytes(count, 32, limits)?;
    if limits.max_storage_buffers_per_shader_stage < 5 || limits.max_bindings_per_bind_group < 6 {
        return Err(invalid());
    }
    Ok(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "tree_gpu_gather_tests.rs"]
mod tests;
