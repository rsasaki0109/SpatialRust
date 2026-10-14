use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult, TransferDirection, TransferStats};
use spatialrust_search::KdTree;
use wgpu::util::DeviceExt;

use crate::{GpuKdTree, GpuNeighborhoods, GpuQueryBatch, WgpuRuntime};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    salient_capacity: u32,
    non_max_capacity: u32,
    min_neighbors: u32,
    gamma_21: f32,
    gamma_32: f32,
    radius: f32,
    non_max_radius_squared: f32,
}

fn invalid() -> SpatialError {
    SpatialError::InvalidArgument("GPU ISS requires matching input/runtime ownership, finite XYZ/extents, positive finite nonzero radius squares, finite gamma values in [0,1] and supported buffers/dispatch".to_owned())
}

fn valid_radius(radius: f32) -> bool {
    radius.is_finite() && radius > 0.0 && (radius * radius).is_finite() && radius * radius > 0.0
}

fn plan(points: u64, limits: &wgpu::Limits) -> SpatialResult<u32> {
    let bytes = points.checked_mul(16).ok_or_else(invalid)?;
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    if points > u64::from(u32::MAX)
        || bytes > bound
        || bytes > isize::MAX as u64
        || bound < 32
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

/// Explicitly uploaded ISS positions/tree/queries, reusable on one runtime.
/// CPU tree construction is external; subsequent search and selection run on GPU.
pub struct GpuIssInput {
    tree: GpuKdTree,
    queries: GpuQueryBatch,
    count: u32,
    identity: Arc<()>,
    input_identity: Arc<()>,
    transfers: TransferStats,
}

impl GpuIssInput {
    /// Uploads a finite CPU tree and original-order queries. The query buffer is
    /// also the covariance position buffer: 32 bytes per point plus 32 per tree
    /// node, with no duplicate sample upload. Finite f32 extent squares required.
    pub fn upload(runtime: &WgpuRuntime, tree: &KdTree) -> SpatialResult<Self> {
        Self::upload_impl(runtime, tree, false)
    }

    /// Explicitly generates tree points and original-order query/position
    /// records in one integer GPU pass. Host payload is 16*N + 32*M bytes,
    /// compared with 32*N + 32*M for `upload`; device output bytes are 32*N.
    /// No readback or completion wait. CPU partitioning, validation and node
    /// staging remain. Removes 32*N host packing bytes, with 16*N transient
    /// GPU input bytes beside owned outputs. Small inputs may be slower;
    /// placement is explicit and existing ISS device requirements apply.
    pub fn upload_with_gpu_gather(runtime: &WgpuRuntime, tree: &KdTree) -> SpatialResult<Self> {
        Self::upload_impl(runtime, tree, true)
    }

    fn upload_impl(runtime: &WgpuRuntime, tree: &KdTree, gpu_gather: bool) -> SpatialResult<Self> {
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
        let (tree, queries) = if gpu_gather {
            GpuKdTree::upload_and_queries_with_gpu_gather(runtime, tree)?
        } else {
            (GpuKdTree::upload(runtime, tree)?, GpuQueryBatch::upload_soa(runtime, [x, y, z])?)
        };
        let mut transfers = tree.upload_transfers();
        transfers.record(
            TransferDirection::HostToDevice,
            queries.upload_transfers().host_to_device_bytes(),
        );
        transfers.record(
            TransferDirection::DeviceToDevice,
            queries.upload_transfers().device_to_device_bytes(),
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

    /// Payload explicitly transferred at construction (zero for empty input),
    /// including device output bytes for the GPU gather constructor.
    #[must_use]
    pub const fn upload_transfers(&self) -> TransferStats {
        self.transfers
    }

    /// Computes both reusable radius graphs, including self, without readback
    /// or a host wait. Every match is counted; selection rejects overflow.
    pub fn neighborhoods_resident(
        &self,
        runtime: &WgpuRuntime,
        salient_radius: f32,
        non_max_radius: f32,
        salient_capacity: u32,
        non_max_capacity: u32,
    ) -> SpatialResult<GpuIssNeighborhoods> {
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity())
            || !valid_radius(salient_radius)
            || !valid_radius(non_max_radius)
        {
            return Err(invalid());
        }
        let salient = self.tree.radius_neighbors_resident(
            runtime,
            &self.queries,
            salient_radius,
            salient_capacity,
        )?;
        let non_max = self.tree.radius_neighbors_resident(
            runtime,
            &self.queries,
            non_max_radius,
            non_max_capacity,
        )?;
        Ok(GpuIssNeighborhoods {
            salient,
            non_max,
            salient_radius,
            non_max_radius,
            input_identity: Arc::clone(&self.input_identity),
        })
    }

    /// Runs covariance/eigenvalue saliency then local-maximum selection in two
    /// passes in one submission. Both graphs must be complete, even for rejected
    /// candidates. Gamma values must be finite in [0,1]; min_neighbors includes
    /// self. Equal computed maxima all survive. No payload transfer or host wait.
    /// GPU f32 arithmetic can change threshold/NMS decisions versus CPU f64.
    /// Nonzero local extents whose f32 square is subnormal/zero invalidate the
    /// result with numerical status rather than evaluating underflowed covariance.
    /// Bad graph status invalidates and zeros ALL scores/flags; GPU consumers
    /// must check status and explicit download rejects invalid output.
    pub fn keypoints_resident(
        &self,
        runtime: &WgpuRuntime,
        neighbors: &GpuIssNeighborhoods,
        gamma_21: f32,
        gamma_32: f32,
        min_neighbors: u32,
    ) -> SpatialResult<GpuIssKeypoints> {
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity())
            || !Arc::ptr_eq(&self.input_identity, &neighbors.input_identity)
            || [gamma_21, gamma_32].iter().any(|g| !g.is_finite() || !(0.0..=1.0).contains(g))
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
        let result = GpuIssKeypoints {
            scores: create("iss-eigenvalues", (u64::from(self.count) * 16).max(16)),
            flags: create("iss-selection", (u64::from(self.count) * 4).max(4)),
            status: create("iss-status", 32),
            count: self.count,
            identity: Arc::clone(&self.identity),
        };
        if self.count == 0 {
            return Ok(result);
        }
        let cache = runtime.iss_pipelines.get_or_init(|| IssPipelines::new(runtime.device()));
        let params = Params {
            count: self.count,
            salient_capacity: neighbors.salient.capacity(),
            non_max_capacity: neighbors.non_max.capacity(),
            min_neighbors,
            gamma_21,
            gamma_32,
            radius: neighbors.salient_radius,
            non_max_radius_squared: neighbors.non_max_radius * neighbors.non_max_radius,
        };
        let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("iss-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [
            &uniform,
            self.queries.buffer().unwrap(),
            neighbors.salient.buffer(),
            neighbors.non_max.buffer(),
            &result.scores,
            &result.flags,
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
            label: Some("iss-bindings"),
            layout: &cache.layout,
            entries: &entries,
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        for pipeline in [&cache.saliency, &cache.selection] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("iss-compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
        }
        runtime.queue().submit(Some(encoder.finish()));
        Ok(result)
    }
}

/// Two radius graphs tied to exactly one immutable uploaded ISS input.
pub struct GpuIssNeighborhoods {
    salient: GpuNeighborhoods,
    non_max: GpuNeighborhoods,
    salient_radius: f32,
    non_max_radius: f32,
    input_identity: Arc<()>,
}
impl GpuIssNeighborhoods {
    /// Covariance radius graph. Treat exposed storage as immutable.
    #[must_use]
    pub fn salient_rows(&self) -> &GpuNeighborhoods {
        &self.salient
    }
    /// Non-maximum-suppression graph. Treat exposed storage as immutable.
    #[must_use]
    pub fn non_max_rows(&self) -> &GpuNeighborhoods {
        &self.non_max
    }
}

/// Owned resident ISS selection flags, eigenvalue/eligibility rows and validity
/// status. No result storage is pooled. Custom consumers must inspect status.
pub struct GpuIssKeypoints {
    scores: wgpu::Buffer,
    flags: wgpu::Buffer,
    status: wgpu::Buffer,
    count: u32,
    identity: Arc<()>,
}
impl GpuIssKeypoints {
    /// Original-order u32 flags (0 or 1), with 4 unused bytes for empty input.
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.flags
    }
    /// Original-order vec4<f32>: ascending eigenvalues and eligibility (0 or 1).
    /// Rejected ratio tests still expose eigenvalues; insufficient support is
    /// zero. Empty input has 16 unused bytes. Invalid graph output is all zero.
    #[must_use]
    pub fn scores_buffer(&self) -> &wgpu::Buffer {
        &self.scores
    }
    /// Eight u32 words: bad salient rows, bad NMS rows, required salient/NMS
    /// capacity, malformed rows, last bad row plus one, numerical errors, reserved.
    /// All zero means valid output. Consumers must reject any nonzero status.
    #[must_use]
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }
    /// Original point/flag count.
    #[must_use]
    pub const fn point_count(&self) -> u32 {
        self.count
    }
    /// Explicit flag/status readback payload; scores are not downloaded.
    #[must_use]
    pub const fn readback_bytes(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            self.count as u64 * 4 + 32
        }
    }
    /// Explicitly downloads flags/status and returns ascending original indices.
    /// Waits for completion; malformed/incomplete graphs reject before decoding.
    /// Payload is 4 bytes per input point plus 32 status bytes, empty input zero.
    /// CPU ordered flag compaction occurs here; no neighborhood/eigen readback.
    pub fn download(&self, runtime: &WgpuRuntime) -> SpatialResult<(Vec<u32>, TransferStats)> {
        self.readback(runtime, false, |words| {
            if words.iter().any(|&v| v > 1) {
                return Err(invalid());
            }
            Ok(words
                .iter()
                .enumerate()
                .filter_map(|(i, &flag)| (flag == 1).then_some(i as u32))
                .collect())
        })
    }

    /// Explicitly downloads eigenvalue/eligibility rows and status for diagnostics.
    /// This separate opt-in transfer costs 16 bytes per point plus 32 status
    /// bytes (empty input zero); the ordinary selection API does not call it.
    /// Waits for completion and rejects invalid status/nonfinite or unsorted rows.
    pub fn download_scores(
        &self,
        runtime: &WgpuRuntime,
    ) -> SpatialResult<(Vec<[f32; 4]>, TransferStats)> {
        self.readback(runtime, true, |words| {
            let mut rows = Vec::with_capacity(self.count as usize);
            for row in words.chunks_exact(4) {
                let values: [f32; 4] = std::array::from_fn(|i| f32::from_bits(row[i]));
                if values.iter().any(|v| !v.is_finite())
                    || values[0] > values[1]
                    || values[1] > values[2]
                    || (values[3] != 0.0 && values[3] != 1.0)
                {
                    return Err(invalid());
                }
                rows.push(values);
            }
            Ok(rows)
        })
    }

    fn readback<T: Default>(
        &self,
        runtime: &WgpuRuntime,
        scores: bool,
        decode: impl FnOnce(&[u32]) -> SpatialResult<T>,
    ) -> SpatialResult<(T, TransferStats)> {
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity()) {
            return Err(invalid());
        }
        if self.count == 0 {
            return Ok((T::default(), TransferStats::default()));
        }
        let bytes = u64::from(self.count) * if scores { 16 } else { 4 };
        let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("iss-readback"),
            size: bytes + 32,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            if scores { &self.scores } else { &self.flags },
            0,
            &staging,
            0,
            bytes,
        );
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
        let payload_words = bytes as usize / 4;
        let status = &words[payload_words..];
        let decoded = if status.iter().any(|&v| v != 0) {
            Err(SpatialError::InvalidArgument(format!("GPU ISS rejected incomplete/malformed neighborhoods or numerical output; status={status:?}")))
        } else {
            decode(&words[..payload_words])
        };
        drop(mapped);
        staging.unmap();
        let mut transfers = TransferStats::default();
        transfers.record(TransferDirection::DeviceToHost, bytes + 32);
        Ok((decoded?, transfers))
    }
}

pub(crate) struct IssPipelines {
    layout: wgpu::BindGroupLayout,
    saliency: wgpu::ComputePipeline,
    selection: wgpu::ComputePipeline,
}
impl IssPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..7)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: binding < 4 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iss-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iss-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iss-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/iss.wgsl").into()),
        });
        let create = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("iss-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self { saliency: create("build_saliency"), selection: create("select_keypoints"), layout }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "iss_tests.rs"]
mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "iss_soa_upload_reference.rs"]
pub(crate) mod soa_upload_reference;
