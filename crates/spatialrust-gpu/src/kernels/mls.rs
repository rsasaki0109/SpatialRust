use crate::{scratch::ScratchBuffers, GpuKdTree, GpuNeighborhoods, GpuQueryBatch, WgpuRuntime};
use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult, TransferDirection, TransferStats};
use spatialrust_search::KdTree;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    capacity: u32,
    min_neighbors: u32,
    padding: u32,
    radius_squared: f32,
    weight_scale_squared: f32,
    unused: [f32; 2],
}

fn invalid() -> SpatialError {
    SpatialError::InvalidArgument("GPU MLS requires matching input/runtime ownership, finite XYZ/extents, positive finite nonzero radius squares, order 0..=2 and supported buffers/dispatch".to_owned())
}

fn valid_radius(radius: f32) -> bool {
    radius.is_finite() && radius > 0.0 && (radius * radius).is_finite() && radius * radius > 0.0
}

fn plan(points: u64, limits: &wgpu::Limits) -> SpatialResult<u32> {
    let bytes = points.checked_mul(192).ok_or_else(invalid)?;
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    if points > u64::from(u32::MAX)
        || bytes > bound
        || bytes > isize::MAX as u64
        || bound < 192
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
        || limits.max_compute_invocations_per_workgroup < 64
        || points.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
        || limits.max_storage_buffers_per_shader_stage < 6
        || limits.max_bindings_per_bind_group < 7
        || limits.max_bind_groups == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < 32
        || points
            .checked_mul(16)
            .and_then(|n| n.checked_add(32))
            .map_or(true, |n| n > limits.max_buffer_size || n > isize::MAX as u64)
    {
        return Err(invalid());
    }
    Ok(points as u32)
}

/// Explicitly uploaded MLS positions/tree/queries, reusable on one runtime.
/// CPU tree construction is external; subsequent search and surface fitting run on GPU.
pub struct GpuMlsInput {
    tree: GpuKdTree,
    queries: GpuQueryBatch,
    count: u32,
    identity: Arc<()>,
    input_identity: Arc<()>,
    transfers: TransferStats,
}

impl GpuMlsInput {
    /// Uploads a finite CPU tree and original-order queries. The query buffer is
    /// also the covariance position buffer: 32 bytes per point plus 32 per tree
    /// node, with no duplicate sample upload. Finite f32 extent squares required.
    pub fn upload(runtime: &WgpuRuntime, tree: &KdTree) -> SpatialResult<Self> {
        let layout = tree.layout();
        let (x, y, z) = layout.positions();
        let count = plan(x.len() as u64, &runtime.device().limits())?;
        if x.iter().chain(y).chain(z).any(|v| !v.is_finite()) {
            return Err(invalid());
        }
        if count > 0 {
            let extent: f64 = [x, y, z]
                .iter()
                .map(|axis| {
                    let lo = axis.iter().copied().fold(f32::INFINITY, f32::min);
                    let hi = axis.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    (f64::from(hi) - f64::from(lo)).powi(2)
                })
                .sum();
            if !(extent as f32).is_finite() {
                return Err(invalid());
            }
        }
        let tree = GpuKdTree::upload(runtime, tree)?;
        let queries = GpuQueryBatch::upload_soa(runtime, [x, y, z])?;
        let mut transfers = tree.upload_transfers();
        transfers.record(
            TransferDirection::HostToDevice,
            queries.upload_transfers().host_to_device_bytes(),
        );
        Ok(Self {
            tree,
            queries,
            count,
            identity: runtime.profiling_identity(),
            input_identity: Arc::new(()),
            transfers,
        })
    }

    /// Original point count.
    #[must_use]
    pub const fn point_count(&self) -> u32 {
        self.count
    }

    /// Payload explicitly uploaded at construction (zero for empty input).
    #[must_use]
    pub const fn upload_transfers(&self) -> TransferStats {
        self.transfers
    }

    /// Builds reusable radius rows tied to this input, including self. Every
    /// match is counted; fitting rejects overflow. No readback or completion wait.
    pub fn neighborhoods_resident(
        &self,
        runtime: &WgpuRuntime,
        radius: f32,
        capacity: u32,
    ) -> SpatialResult<GpuMlsNeighborhoods> {
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity()) || !valid_radius(radius) {
            return Err(invalid());
        }
        let rows = self.tree.radius_neighbors_resident(runtime, &self.queries, radius, capacity)?;
        Ok(GpuMlsNeighborhoods { rows, radius, input_identity: Arc::clone(&self.input_identity) })
    }

    /// Fits Gaussian-weighted local frames and constant/linear/quadratic surfaces
    /// on GPU, excluding the query itself after the minimum-support test. Sparse
    /// points stay unchanged; insufficient polynomial support/ill-conditioned
    /// solves use the plane on device. The normalized f32 solve uses a relative
    /// Cholesky rank threshold of 1e-6; CPU f64 uses Gaussian elimination with
    /// absolute pivot threshold 1e-12.
    /// Coordinates and fit/fallback decisions can differ near rank/eigenvalue
    /// degeneracy. Unsupported extent underflow is a numerical error. Three passes
    /// share one submission, without payload copies/host wait. Invalid rows or
    /// numerics zero ALL output; GPU consumers must check status.
    pub fn smooth_resident(
        &self,
        runtime: &WgpuRuntime,
        neighbors: &GpuMlsNeighborhoods,
        polynomial_order: u8,
        min_neighbors: u32,
    ) -> SpatialResult<GpuMlsPositions> {
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity())
            || !Arc::ptr_eq(&self.input_identity, &neighbors.input_identity)
            || polynomial_order > 2
        {
            return Err(invalid());
        }
        plan(u64::from(self.count), &runtime.device().limits())?;
        let create = |label, size| {
            runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let result = GpuMlsPositions {
            buffer: create("mls-positions", (u64::from(self.count) * 16).max(16)),
            status: create("mls-status", 32),
            count: self.count,
            identity: Arc::clone(&self.identity),
        };
        if self.count == 0 {
            return Ok(result);
        }
        let cache = runtime.mls_pipelines.get_or_init(|| MlsPipelines::new(runtime.device()));
        let params = Params {
            count: self.count,
            capacity: neighbors.rows.capacity(),
            min_neighbors,
            padding: 0,
            radius_squared: neighbors.radius * neighbors.radius,
            weight_scale_squared: (neighbors.radius * 0.5).powi(2).max(1e-12),
            unused: [0.0; 2],
        };
        let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mls-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut scratch = ScratchBuffers::new(runtime);
        let frames = scratch.acquire("mls-frames", u64::from(self.count) * 64);
        let systems = scratch.acquire("mls-systems", u64::from(self.count) * 192);
        let buffers = [
            &uniform,
            self.queries.buffer().unwrap(),
            neighbors.rows.buffer(),
            &frames,
            &systems,
            &result.buffer,
            &result.status,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mls-bindings"),
            layout: &cache.layout,
            entries: &entries,
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        for pipeline in [&cache.frames, &cache.fits[polynomial_order as usize], &cache.finalize] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mls-compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
        }
        scratch.submit_and_recycle(encoder);
        Ok(result)
    }
}

/// Radius rows tied to one immutable GPU MLS input. Capacity includes self.
pub struct GpuMlsNeighborhoods {
    rows: GpuNeighborhoods,
    radius: f32,
    input_identity: Arc<()>,
}
impl GpuMlsNeighborhoods {
    /// Underlying rows/counts for device consumers. Treat storage as immutable.
    #[must_use]
    pub fn rows(&self) -> &GpuNeighborhoods {
        &self.rows
    }
}

/// Device counts of MLS outcomes; valid only when the first five status words are zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuMlsDiagnostics {
    /// Points retained without fitting (sparse/zero support or zero extent).
    pub unchanged: u32,
    /// Points using the plane after insufficient polynomial support or rank failure.
    pub plane_fallback: u32,
    /// Points using the requested constant/linear/quadratic polynomial fit.
    pub polynomial: u32,
}

/// Per-point outcome of device MLS processing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuMlsMode {
    /// Original coordinates retained without fitting.
    Unchanged,
    /// Plane projection after insufficient support or failed rank test.
    PlaneFallback,
    /// Requested constant/linear/quadratic fit evaluated on device.
    Polynomial,
}

/// Explicit position/status download with outcome counts and payload accounting.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMlsReadback {
    /// Original-order smoothed XYZ.
    pub positions: Vec<[f32; 3]>,
    /// Per-point outcomes aligned with positions; no additional device transfer.
    pub modes: Vec<GpuMlsMode>,
    /// Device fit/fallback counts.
    pub diagnostics: GpuMlsDiagnostics,
    /// Payload copied during this explicit readback.
    pub transfers: TransferStats,
}

/// Owned resident MLS positions/modes and global status. No buffers are pooled;
/// outputs survive input drops and other calls. GPU consumers must check status.
pub struct GpuMlsPositions {
    buffer: wgpu::Buffer,
    status: wgpu::Buffer,
    count: u32,
    identity: Arc<()>,
}
impl GpuMlsPositions {
    /// Original-order vec4<f32>: XYZ and mode (0 unchanged, 1 plane, 2 polynomial).
    /// Invalid results are zeroed; empty input has 16 unused bytes.
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    /// Eight u32 words: bad rows, required capacity, malformed rows, last bad row
    /// plus one, numerical errors, unchanged/plane/polynomial counts. First five
    /// words must ALL be zero; remaining counts are diagnostics for valid output.
    #[must_use]
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }
    /// Original point/output count.
    #[must_use]
    pub const fn point_count(&self) -> u32 {
        self.count
    }
    /// Position/mode and status readback bytes including padding; empty zero.
    #[must_use]
    pub const fn readback_bytes(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            self.count as u64 * 16 + 32
        }
    }
    /// Explicitly downloads positions/modes and status in one submission, waits
    /// for completion and rejects invalid status/nonfinite rows. Payload is
    /// 16 bytes per point plus 32 status bytes, empty input zero. No graph readback.
    pub fn download(&self, runtime: &WgpuRuntime) -> SpatialResult<GpuMlsReadback> {
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity()) {
            return Err(invalid());
        }
        if self.count == 0 {
            return Ok(GpuMlsReadback {
                positions: vec![],
                modes: vec![],
                diagnostics: Default::default(),
                transfers: Default::default(),
            });
        }
        let bytes = u64::from(self.count) * 16;
        let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("mls-readback"),
            size: bytes + 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, bytes);
        encoder.copy_buffer_to_buffer(&self.status, 0, &staging, bytes, 32);
        runtime.queue().submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        runtime.device().poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|e| SpatialError::InvalidArgument(e.to_string()))?
            .map_err(|e| SpatialError::InvalidArgument(e.to_string()))?;
        let mapped = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        let status = &words[bytes as usize / 4..];
        let decoded = (|| {
            if status[..5].iter().any(|&v| v != 0)
                || u64::from(status[5]) + u64::from(status[6]) + u64::from(status[7])
                    != u64::from(self.count)
            {
                return Err(SpatialError::InvalidArgument(format!("GPU MLS rejected incomplete/malformed neighborhoods or numerical output; status={status:?}")));
            }
            let mut positions = Vec::with_capacity(self.count as usize);
            let mut modes = Vec::with_capacity(self.count as usize);
            let mut counts = [0u32; 3];
            for row in words[..bytes as usize / 4].chunks_exact(4) {
                let values: [f32; 4] = std::array::from_fn(|i| f32::from_bits(row[i]));
                if values.iter().any(|v| !v.is_finite()) || ![0.0, 1.0, 2.0].contains(&values[3]) {
                    return Err(invalid());
                }
                counts[values[3] as usize] += 1;
                modes.push(match values[3] as u32 {
                    0 => GpuMlsMode::Unchanged,
                    1 => GpuMlsMode::PlaneFallback,
                    _ => GpuMlsMode::Polynomial,
                });
                positions.push([values[0], values[1], values[2]]);
            }
            if counts != status[5..] {
                return Err(invalid());
            }
            Ok((
                positions,
                modes,
                GpuMlsDiagnostics {
                    unchanged: status[5],
                    plane_fallback: status[6],
                    polynomial: status[7],
                },
            ))
        })();
        drop(mapped);
        staging.unmap();
        let (positions, modes, diagnostics) = decoded?;
        let mut transfers = TransferStats::default();
        transfers.record(TransferDirection::DeviceToHost, bytes + 32);
        Ok(GpuMlsReadback { positions, modes, diagnostics, transfers })
    }
}

pub(crate) struct MlsPipelines {
    layout: wgpu::BindGroupLayout,
    frames: wgpu::ComputePipeline,
    fits: [wgpu::ComputePipeline; 3],
    finalize: wgpu::ComputePipeline,
}
impl MlsPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..7)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: binding < 3 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mls-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mls-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let common = include_str!("../shaders/mls_common.wgsl");
        let frame_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mls-frame-shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{common}{}", include_str!("../shaders/mls_frames.wgsl")).into(),
            ),
        });
        let create = |shader: &wgpu::ShaderModule, entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mls-pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let frames = create(&frame_shader, "prepare_frames");
        let fits = std::array::from_fn(|order| {
            let terms = [1, 3, 6][order];
            let template = if order == 0 {
                include_str!("../shaders/mls_constant.wgsl").to_owned()
            } else {
                include_str!("../shaders/mls_fit.wgsl").replace(
                    "const FIT_TERMS: u32 = 6u;",
                    &format!("const FIT_TERMS: u32 = {terms}u;"),
                )
            };
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mls-fit-shader"),
                source: wgpu::ShaderSource::Wgsl(format!("{common}{template}").into()),
            });
            create(&shader, "fit_surface")
        });
        Self { frames, fits, finalize: create(&frame_shader, "finalize"), layout }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "mls_tests.rs"]
mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "mls_soa_upload_reference.rs"]
pub(crate) mod soa_upload_reference;
