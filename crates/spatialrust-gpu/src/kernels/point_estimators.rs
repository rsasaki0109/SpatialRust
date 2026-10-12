//! Shared bounded execution for host-returning point normal/covariance estimators.

use spatialrust_core::{SpatialError, SpatialResult};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

use crate::{scratch::ScratchBuffers, WgpuRuntime};

#[derive(Clone, Copy, Debug)]
pub(super) enum Kind {
    KnnNormals,
    RadiusNormals,
    PlaneCovariances,
}

impl Kind {
    fn index(self) -> usize {
        match self {
            Self::KnnNormals => 0,
            Self::RadiusNormals => 1,
            Self::PlaneCovariances => 2,
        }
    }

    fn uniform_bytes(self) -> u64 {
        if matches!(self, Self::KnnNormals) {
            16
        } else {
            48
        }
    }

    fn source(self) -> &'static str {
        match self {
            Self::KnnNormals => super::normals::NORMALS_WGSL,
            Self::RadiusNormals => super::normals_grid::NORMALS_GRID_WGSL,
            Self::PlaneCovariances => super::covariances_grid::COV_GRID_WGSL,
        }
    }
}

pub(crate) struct PointEstimatorPipelines {
    pipelines: [OnceLock<wgpu::ComputePipeline>; 3],
}

impl PointEstimatorPipelines {
    fn new() -> Self {
        Self { pipelines: std::array::from_fn(|_| OnceLock::new()) }
    }

    fn pipeline(&self, device: &wgpu::Device, kind: Kind) -> &wgpu::ComputePipeline {
        self.pipelines[kind.index()].get_or_init(|| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("point-estimator-shader"),
                source: wgpu::ShaderSource::Wgsl(kind.source().into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("point-estimator-pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        })
    }
}

struct Plan {
    output_bytes: u64,
    groups: u32,
}

fn plan(
    kind: Kind,
    points: u64,
    auxiliary_counts: &[u64],
    limits: &wgpu::Limits,
) -> SpatialResult<Plan> {
    let invalid = || {
        SpatialError::InvalidArgument(
        "GPU point estimation requires bounded u32 indices and supported storage/uniform/compute limits".to_owned())
    };
    let rows = if matches!(kind, Kind::PlaneCovariances) { 2 } else { 1 };
    let output_bytes = points.checked_mul(rows * 16).ok_or_else(invalid)?;
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    let expected_aux = if matches!(kind, Kind::KnnNormals) { 1 } else { 2 };
    if points == 0
        || points > u64::from(u32::MAX) / rows
        || auxiliary_counts.len() != expected_aux
        || points * 4 > bound
        || output_bytes > bound
        || auxiliary_counts.iter().any(|&n| n == 0 || n > u64::from(u32::MAX) || n * 4 > bound)
        || (matches!(kind, Kind::KnnNormals) && auxiliary_counts[0] % points != 0)
        || (!matches!(kind, Kind::KnnNormals)
            && (auxiliary_counts[0] != points || auxiliary_counts[1] < 2))
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
        || points.div_ceil(256) > u64::from(limits.max_compute_workgroups_per_dimension)
        || limits.max_storage_buffers_per_shader_stage < 4 + expected_aux as u32
        || limits.max_bind_groups == 0
        || limits.max_bindings_per_bind_group < 5 + expected_aux as u32
        || limits.max_uniform_buffers_per_shader_stage == 0
        || u64::from(limits.max_uniform_buffer_binding_size) < kind.uniform_bytes()
        || limits.max_buffer_size < kind.uniform_bytes()
    {
        return Err(invalid());
    }
    Ok(Plan { output_bytes, groups: points.div_ceil(256) as u32 })
}

pub(super) fn validate_knn(
    runtime: &WgpuRuntime,
    points: usize,
    neighbors: &[u32],
    k: u32,
) -> SpatialResult<()> {
    let expected = points
        .checked_mul(k as usize)
        .ok_or_else(|| SpatialError::InvalidArgument("point_count*k overflows usize".to_owned()))?;
    if k == 0 || neighbors.len() != expected {
        return Err(SpatialError::InvalidArgument(
            "neighbors must have point_count*k entries with indices inside the point arrays"
                .to_owned(),
        ));
    }
    plan(Kind::KnnNormals, points as u64, &[neighbors.len() as u64], &runtime.device().limits())?;
    if neighbors.iter().any(|&i| i >= points as u32) {
        return Err(SpatialError::InvalidArgument(
            "neighbor index is outside the point arrays".to_owned(),
        ));
    }
    Ok(())
}

type Grid = ([f32; 3], [u32; 3], Vec<u32>, Vec<u32>);

pub(super) fn prepare_grid(
    runtime: &WgpuRuntime,
    kind: Kind,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    radius: f32,
) -> SpatialResult<Grid> {
    let limits = runtime.device().limits();
    let n = x.len() as u64;
    // Reject point/output limits before allocating the CPU counting-sort grid.
    plan(kind, n, &[n, 2], &limits)?;
    let (origin, dims) = spatialrust_search::grid_bounds(x, y, z, radius)?;
    let mut cells = 1_u64;
    for dim in dims {
        if dim == 0 || dim > i32::MAX as u32 {
            return Err(SpatialError::InvalidArgument(
                "GPU grid dimensions must fit positive i32 coordinates".to_owned(),
            ));
        }
        cells = cells.checked_mul(u64::from(dim)).ok_or_else(|| {
            SpatialError::InvalidArgument("GPU grid cell count overflows".to_owned())
        })?;
    }
    let cell_entries = cells.checked_add(1).ok_or_else(|| {
        SpatialError::InvalidArgument("GPU grid sentinel count overflows".to_owned())
    })?;
    plan(kind, n, &[n, cell_entries], &limits)?;
    let (sorted, cell_start) = spatialrust_search::build_grid(x, y, z, origin, dims, radius);
    Ok((origin, dims, sorted, cell_start))
}

pub(super) fn run<T>(
    runtime: &WgpuRuntime,
    kind: Kind,
    points: usize,
    uniform: &[u8],
    inputs: &[&[u8]],
    decode: impl FnOnce(&[[f32; 4]]) -> Vec<T>,
) -> SpatialResult<Vec<T>> {
    let auxiliary: Vec<_> = inputs[3..].iter().map(|bytes| (bytes.len() / 4) as u64).collect();
    let p = plan(kind, points as u64, &auxiliary, &runtime.device().limits())?;
    debug_assert_eq!(uniform.len() as u64, kind.uniform_bytes());
    debug_assert!(inputs[..3].iter().all(|bytes| bytes.len() == points * 4));
    let cache = runtime.point_estimator_pipelines.get_or_init(PointEstimatorPipelines::new);
    let pipeline = cache.pipeline(runtime.device(), kind);
    let mut scratch = ScratchBuffers::new(runtime);
    let buffers: Vec<_> = inputs
        .iter()
        .map(|bytes| {
            let buffer = scratch.acquire("point-estimator-input", bytes.len() as u64);
            runtime.queue().write_buffer(&buffer, 0, bytes);
            buffer
        })
        .collect();
    let output = scratch.acquire("point-estimator-output", p.output_bytes);
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("point-estimator-params"),
        contents: uniform,
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("point-estimator-readback"),
        size: p.output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut entries =
        vec![wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }];
    for (i, buffer) in buffers.iter().chain(std::iter::once(&output)).enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: i as u32 + 1,
            resource: buffer.as_entire_binding(),
        });
    }
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("point-estimator-bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("point-estimator"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(p.groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, p.output_bytes);
    scratch.submit_and_recycle(encoder);
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    runtime.device().poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| SpatialError::InvalidArgument("failed to receive wgpu map result".to_owned()))?
        .map_err(|e| SpatialError::InvalidArgument(format!("failed to map wgpu buffer: {e}")))?;
    let mapped = slice.get_mapped_range();
    let result = decode(bytemuck::cast_slice(&mapped));
    drop(mapped);
    staging.unmap();
    Ok(result)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "point_estimators_tests.rs"]
mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "grid_build_tests.rs"]
mod grid_build_tests;
