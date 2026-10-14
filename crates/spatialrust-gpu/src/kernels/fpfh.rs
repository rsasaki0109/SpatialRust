use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult, TransferDirection, TransferStats};
use spatialrust_search::KdTree;
use wgpu::util::DeviceExt;

use crate::{scratch::ScratchBuffers, GpuKdTree, GpuNeighborhoods, GpuQueryBatch, WgpuRuntime};

#[path = "fpfh_sample_gather.rs"]
pub(crate) mod sample_gather;

#[path = "fpfh_two_pass_reference.rs"]
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) mod two_pass_reference;

/// Three eleven-bin angular histograms per FPFH descriptor.
pub const GPU_FPFH_DESCRIPTOR_LEN: usize = 33;

// Preserve the previous two-pass large-input path: unconditional fusion
// regressed large shuffled uploads in both isolated completed-call surveys.
const FUSED_POINT_LIMIT: u32 = 1_048_576;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Sample {
    position: [f32; 4],
    normal: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    capacity: u32,
    padding: [u32; 2],
}

fn invalid() -> SpatialError {
    SpatialError::InvalidArgument("GPU FPFH requires matching input/runtime ownership, finite XYZ/unit normals, a positive finite radius square and supported buffers/dispatch".to_owned())
}

fn plan(points: u64, limits: &wgpu::Limits) -> SpatialResult<(u32, u64)> {
    let bytes = points.checked_mul(132).ok_or_else(invalid)?;
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    if points > u64::from(u32::MAX)
        || bytes > bound
        || bytes > isize::MAX as u64
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
        || limits.max_compute_invocations_per_workgroup < 64
        || points.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
        || limits.max_storage_buffers_per_shader_stage < 5
        || limits.max_bindings_per_bind_group < 6
        || limits.max_bind_groups == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < 16
        || bound < 32
        || bytes
            .checked_add(16)
            .map_or(true, |v| v > limits.max_buffer_size || v > isize::MAX as u64)
    {
        return Err(invalid());
    }
    Ok((points as u32, bytes))
}

/// Explicitly uploaded FPFH positions/unit normals and their GPU KD-tree/query
/// batch. Original point order is retained; CPU tree construction is external.
/// All subsequent neighborhood and descriptor computation stays on GPU.
pub struct GpuFpfhInput {
    samples: Option<wgpu::Buffer>,
    tree: GpuKdTree,
    queries: GpuQueryBatch,
    points: u32,
    identity: Arc<()>,
    input_identity: Arc<()>,
    transfers: TransferStats,
}

impl GpuFpfhInput {
    /// Uploads a CPU KD-tree and matching original-order finite unit normals.
    /// Payload is 64 bytes per point plus 32 per tree node (samples, original
    /// queries and KD-tree points/nodes). No readback is performed. Coordinate
    /// extents must have a finite f32 squared length, as in CPU FPFH.
    pub fn upload(
        runtime: &WgpuRuntime,
        tree: &KdTree,
        normals: &[[f32; 3]],
    ) -> SpatialResult<Self> {
        Self::upload_impl(runtime, tree, normals, false)
    }

    /// Explicitly generates tree points, original-order queries and padded
    /// position/normal samples using integer copies. Below 1,048,576 points,
    /// one pass generates all three outputs; larger inputs retain two passes.
    /// Host payload is
    /// 28*N + 32*M bytes (axes, indices, raw normals and nodes), compared with
    /// 64*N + 32*M for `upload`. Device output bytes are 64*N, with no readback
    /// or completion wait. CPU partitioning, validation and node staging remain.
    /// Removes host packing allocations totaling 64*N payload bytes over
    /// construction; fused inputs use two transient buffers totaling 28*N
    /// bytes, while the large two-pass path uses up to 28*N beside owned outputs.
    /// Small inputs may be slower than default upload; placement is
    /// explicit and existing FPFH device requirements apply.
    pub fn upload_with_gpu_gather(
        runtime: &WgpuRuntime,
        tree: &KdTree,
        normals: &[[f32; 3]],
    ) -> SpatialResult<Self> {
        Self::upload_impl(runtime, tree, normals, true)
    }

    fn upload_impl(
        runtime: &WgpuRuntime,
        tree: &KdTree,
        normals: &[[f32; 3]],
        gpu_gather: bool,
    ) -> SpatialResult<Self> {
        let layout = tree.layout();
        let (x, y, z) = layout.positions();
        let (points, _) = plan(x.len() as u64, &runtime.device().limits())?;
        if normals.len() != x.len() {
            return Err(SpatialError::BufferLengthMismatch {
                expected: x.len(),
                found: normals.len(),
            });
        }
        if !super::feature_validation::valid([x, y, z], Some(normals)) {
            return Err(invalid());
        }
        let (tree, queries, gathered_samples) = if gpu_gather {
            if points < FUSED_POINT_LIMIT {
                GpuKdTree::upload_with_gpu_samples(runtime, tree, normals)?
            } else {
                let (tree, queries) = GpuKdTree::upload_and_queries_with_gpu_gather(runtime, tree)?;
                (tree, queries, None)
            }
        } else {
            (
                GpuKdTree::upload(runtime, tree)?,
                GpuQueryBatch::upload_soa(runtime, [x, y, z])?,
                None,
            )
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
        let samples = if points == 0 {
            None
        } else if gpu_gather {
            transfers.record(TransferDirection::HostToDevice, u64::from(points) * 12);
            transfers.record(TransferDirection::DeviceToDevice, u64::from(points) * 32);
            Some(
                gathered_samples
                    .unwrap_or_else(|| sample_gather::upload(runtime, &queries, normals)),
            )
        } else {
            let samples: Vec<_> = x
                .iter()
                .zip(y)
                .zip(z)
                .zip(normals)
                .map(|(((&x, &y), &z), n)| Sample {
                    position: [x, y, z, 0.0],
                    normal: [n[0], n[1], n[2], 0.0],
                })
                .collect();
            transfers.record(TransferDirection::HostToDevice, u64::from(points) * 32);
            Some(runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fpfh-samples"),
                contents: bytemuck::cast_slice(&samples),
                usage: wgpu::BufferUsages::STORAGE,
            }))
        };
        Ok(Self {
            samples,
            tree,
            queries,
            points,
            identity: runtime.profiling_identity(),
            input_identity: Arc::new(()),
            transfers,
        })
    }

    /// Original point/descriptor count.
    #[must_use]
    pub const fn point_count(&self) -> u32 {
        self.points
    }

    /// Payload bytes explicitly transferred at construction, including device
    /// output bytes when the GPU gather constructor is selected.
    #[must_use]
    pub const fn upload_transfers(&self) -> TransferStats {
        self.transfers
    }

    /// Builds reusable GPU radius neighborhoods for this input, including self.
    /// No readback or host wait occurs. All matches are counted; capacity
    /// overflow is checked on GPU during descriptor computation and at download.
    pub fn radius_neighborhoods_resident(
        &self,
        runtime: &WgpuRuntime,
        radius: f32,
        capacity: u32,
    ) -> SpatialResult<GpuFpfhNeighborhoods> {
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity())
            || !radius.is_finite()
            || radius <= 0.0
            || !(radius * radius).is_finite()
            || radius * radius == 0.0
        {
            return Err(invalid());
        }
        let neighborhoods =
            self.tree.radius_neighbors_resident(runtime, &self.queries, radius, capacity)?;
        Ok(GpuFpfhNeighborhoods {
            neighborhoods,
            input_identity: Arc::clone(&self.input_identity),
            radius,
        })
    }

    /// Computes GPU SPFH and distance-weighted FPFH using resident neighborhoods
    /// from this exact input. Both passes share one submission; results remain
    /// on device. Overflow/malformed rows poison the entire output to zero and
    /// set status; download rejects that output, and GPU consumers must check
    /// status before using it. No automatic CPU fallback or payload copy occurs.
    /// GPU angular/bin rounding may differ from CPU near boundaries.
    pub fn descriptors_resident(
        &self,
        runtime: &WgpuRuntime,
        neighbors: &GpuFpfhNeighborhoods,
    ) -> SpatialResult<GpuFpfhDescriptors> {
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity())
            || !Arc::ptr_eq(&self.input_identity, &neighbors.input_identity)
        {
            return Err(invalid());
        }
        let (_, bytes) = plan(u64::from(self.points), &runtime.device().limits())?;
        let create = |label, size| {
            runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let output = GpuFpfhDescriptors {
            buffer: create("fpfh-descriptors", bytes.max(4)),
            status: create("fpfh-status", 16),
            points: self.points,
            bytes,
            identity: Arc::clone(&self.identity),
        };
        if self.points == 0 {
            return Ok(output);
        }
        let cache = runtime.fpfh_pipelines.get_or_init(|| FpfhPipelines::new(runtime.device()));
        let mut scratch = ScratchBuffers::new(runtime);
        let spfh = scratch.acquire("fpfh-spfh", bytes);
        let params = Params {
            count: self.points,
            capacity: neighbors.neighborhoods.capacity(),
            padding: [0; 2],
        };
        let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fpfh-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [
            &uniform,
            self.samples.as_ref().unwrap(),
            neighbors.neighborhoods.buffer(),
            &spfh,
            &output.buffer,
            &output.status,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fpfh-bindings"),
            layout: &cache.layout,
            entries: &entries,
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        for pipeline in [&cache.spfh, &cache.fpfh] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fpfh-compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(self.points.div_ceil(64), 1, 1);
        }
        scratch.submit_and_recycle(encoder);
        Ok(output)
    }
}

/// Radius rows tied to one immutable FPFH input; usable by both device phases.
pub struct GpuFpfhNeighborhoods {
    neighborhoods: GpuNeighborhoods,
    input_identity: Arc<()>,
    radius: f32,
}

impl GpuFpfhNeighborhoods {
    /// The underlying radius rows/counts. Treat storage as immutable.
    #[must_use]
    pub fn neighborhoods(&self) -> &GpuNeighborhoods {
        &self.neighborhoods
    }
    /// Radius used for this neighborhood graph.
    #[must_use]
    pub const fn radius(&self) -> f32 {
        self.radius
    }
}

/// Owned resident FPFH descriptors with GPU validity status. Nonempty outputs
/// have 33 f32 values per original point. Custom GPU consumers must reject
/// status.bad_rows>0 before accepting any descriptor; failed outputs are zero.
pub struct GpuFpfhDescriptors {
    buffer: wgpu::Buffer,
    status: wgpu::Buffer,
    points: u32,
    bytes: u64,
    identity: Arc<()>,
}

impl GpuFpfhDescriptors {
    /// Flat original-order 33-f32 descriptor rows; empty results have 4 unused bytes.
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    /// Four u32 status words: bad rows, last bad row plus one, maximum required
    /// radius capacity, malformed rows. All zero means valid descriptors.
    #[must_use]
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }
    /// Number of descriptor rows.
    #[must_use]
    pub const fn point_count(&self) -> u32 {
        self.points
    }
    /// Explicit descriptor/status readback bytes (zero for empty input).
    #[must_use]
    pub const fn readback_bytes(&self) -> u64 {
        if self.points == 0 {
            0
        } else {
            self.bytes + 16
        }
    }
    /// Explicitly downloads descriptors and GPU status in one submission. Rejects
    /// incomplete/malformed neighborhoods before decoding descriptors. Transfers
    /// include 132 bytes per point plus 16 status bytes; no neighborhood/SPFH
    /// readback occurs. This waits for completion and propagates GPU errors.
    pub fn download(
        &self,
        runtime: &WgpuRuntime,
    ) -> SpatialResult<(Vec<[f32; 33]>, TransferStats)> {
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity()) {
            return Err(invalid());
        }
        if self.points == 0 {
            return Ok((Vec::new(), TransferStats::default()));
        }
        let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("fpfh-readback"),
            size: self.bytes + 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.bytes);
        encoder.copy_buffer_to_buffer(&self.status, 0, &staging, self.bytes, 16);
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
        let status: &[u32] = bytemuck::cast_slice(&mapped[self.bytes as usize..]);
        if status[0] > 0 {
            let error=SpatialError::InvalidArgument(format!("GPU FPFH rejected {} incomplete/malformed neighborhood rows; last row {}, required capacity {}, malformed rows {}",status[0],status[1].saturating_sub(1),status[2],status[3]));
            drop(mapped);
            staging.unmap();
            return Err(error);
        }
        let descriptors = bytemuck::cast_slice::<u8, f32>(&mapped[..self.bytes as usize])
            .chunks_exact(GPU_FPFH_DESCRIPTOR_LEN)
            .map(|row| row.try_into().expect("FPFH rows contain exactly 33 values"))
            .collect();
        drop(mapped);
        staging.unmap();
        let mut transfers = TransferStats::default();
        transfers.record(TransferDirection::DeviceToHost, self.bytes + 16);
        Ok((descriptors, transfers))
    }
}

pub(crate) struct FpfhPipelines {
    layout: wgpu::BindGroupLayout,
    spfh: wgpu::ComputePipeline,
    fpfh: wgpu::ComputePipeline,
}
impl FpfhPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..6)
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
            label: Some("fpfh-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fpfh-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fpfh-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fpfh.wgsl").into()),
        });
        let create = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fpfh-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self { spfh: create("build_spfh"), fpfh: create("build_fpfh"), layout }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "fpfh_tests.rs"]
mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "fpfh_soa_upload_reference.rs"]
pub(crate) mod soa_upload_reference;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "fpfh_validation_reference.rs"]
pub(crate) mod validation_reference;
