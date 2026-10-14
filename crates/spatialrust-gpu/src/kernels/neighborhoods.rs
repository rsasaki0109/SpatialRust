use super::{buffer_bytes, check_compute, invalid, Arc, GpuKdTree, Neighbor, Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult, TransferDirection, TransferStats};
use wgpu::util::DeviceExt;

use crate::WgpuRuntime;

/// Maximum k for the bounded GPU nearest-neighbor accumulator.
pub const MAX_GPU_K_NEIGHBORS: u32 = 64;

/// Membership and ordering of a resident neighborhood result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuNeighborhoodKind {
    /// Up to k points, stably sorted by distance with CPU accumulator ties.
    NearestK,
    /// Radius membership in CPU near-first leaf traversal order.
    Radius,
}

/// Immutable finite query positions, explicitly uploaded or generated from a
/// tree on GPU. Reusable across trees and passes on the same runtime.
pub struct GpuQueryBatch {
    buffer: Option<wgpu::Buffer>,
    count: u32,
    identity: Arc<()>,
    transfers: TransferStats,
}

impl GpuQueryBatch {
    pub(super) fn from_gathered(
        runtime: &WgpuRuntime,
        buffer: Option<wgpu::Buffer>,
        count: u32,
    ) -> Self {
        let mut transfers = TransferStats::default();
        transfers.record(TransferDirection::DeviceToDevice, u64::from(count) * 16);
        Self { buffer, count, identity: runtime.profiling_identity(), transfers }
    }

    pub(crate) fn buffer(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_ref()
    }

    /// Uploads XYZ queries as padded 16-byte records. Finite XYZ and supported
    /// buffer/dispatch sizes are required; empty batches allocate no buffer.
    pub fn upload(runtime: &WgpuRuntime, queries: &[[f32; 3]]) -> SpatialResult<Self> {
        let limits = runtime.device().limits();
        let bytes = buffer_bytes(queries.len() as u64, 16, &limits)?;
        check_compute(&limits, queries.len() as u64)?;
        if queries.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid());
        }
        let mut batch = Self {
            buffer: None,
            count: queries.len() as u32,
            identity: runtime.profiling_identity(),
            transfers: TransferStats::default(),
        };
        if !queries.is_empty() {
            let packed: Vec<_> = queries.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
            batch.buffer =
                Some(runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("neighborhood-queries"),
                    contents: bytemuck::cast_slice(&packed),
                    usage: wgpu::BufferUsages::STORAGE,
                }));
            batch.transfers.record(TransferDirection::HostToDevice, bytes);
        }
        Ok(batch)
    }

    // Registration and point-feature trees own original-order SoA positions.
    // Pack directly into the existing device record, avoiding an AoS copy.
    pub(crate) fn upload_soa(runtime: &WgpuRuntime, axes: [&[f32]; 3]) -> SpatialResult<Self> {
        let count = axes[0].len();
        if axes.iter().any(|axis| axis.len() != count) {
            return Err(invalid());
        }
        let limits = runtime.device().limits();
        let bytes = buffer_bytes(count as u64, 16, &limits)?;
        check_compute(&limits, count as u64)?;
        let packed = pack_soa(axes)?;
        let mut batch = Self {
            buffer: None,
            count: count as u32,
            identity: runtime.profiling_identity(),
            transfers: TransferStats::default(),
        };
        if count != 0 {
            batch.buffer =
                Some(runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("neighborhood-queries"),
                    contents: bytemuck::cast_slice(&packed),
                    usage: wgpu::BufferUsages::STORAGE,
                }));
            batch.transfers.record(TransferDirection::HostToDevice, bytes);
        }
        Ok(batch)
    }

    /// Number of XYZ query records.
    #[must_use]
    pub const fn query_count(&self) -> u32 {
        self.count
    }

    /// Payload bytes transferred during explicit construction. GPU-generated
    /// queries report device-to-device output bytes and zero host upload bytes.
    #[must_use]
    pub const fn upload_transfers(&self) -> TransferStats {
        self.transfers
    }
}

fn pack_soa(axes: [&[f32]; 3]) -> SpatialResult<Vec<[f32; 4]>> {
    query_staging::prepare(axes)
}

#[path = "query_staging.rs"]
mod query_staging;

/// Immutable GPU-resident bounded neighborhoods. Creating this result submits
/// GPU search without payload uploads/readbacks or a completion wait. Query/target
/// buffers are explicitly uploaded beforehand. Outputs are owned, never pooled.
pub struct GpuNeighborhoods {
    buffer: wgpu::Buffer,
    queries: u32,
    capacity: u32,
    bytes: u64,
    kind: GpuNeighborhoodKind,
    identity: Arc<()>,
}

impl GpuNeighborhoods {
    /// Output storage for subsequent GPU kernels on the same runtime. Each row
    /// has `2 + 2*capacity` u32 words: stored count, total count, then pairs of
    /// original target index and f32 squared-distance bits. Nearest rows are
    /// sorted by distance; radius rows retain traversal order. Respect stored
    /// counts; unused slots are zero. Empty results have an unused 8-byte buffer.
    /// Radius consumers must handle total count > capacity before using a row.
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Number of result rows, in query input order.
    #[must_use]
    pub const fn query_count(&self) -> u32 {
        self.queries
    }

    /// Maximum stored neighbors per row; radius counts may exceed this.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Membership and ordering contract.
    #[must_use]
    pub const fn kind(&self) -> GpuNeighborhoodKind {
        self.kind
    }

    /// Payload bytes in a complete explicit readback, including row padding.
    #[must_use]
    pub const fn readback_bytes(&self) -> u64 {
        self.bytes
    }

    /// Host/device payload transfers performed by search itself. Target/query
    /// uploads are separate; uniform control data is excluded.
    #[must_use]
    pub fn execution_transfers(&self) -> TransferStats {
        TransferStats::default()
    }

    /// Explicitly downloads rows and exact counts, waiting for queue completion.
    /// Use complete_rows/into_complete_rows to reject radius overflow, or
    /// truncated_rows to intentionally inspect a bounded traversal prefix.
    pub fn download(&self, runtime: &WgpuRuntime) -> SpatialResult<GpuNeighborhoodReadback> {
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity()) {
            return Err(invalid());
        }
        if self.queries == 0 {
            return Ok(GpuNeighborhoodReadback {
                rows: Vec::new(),
                totals: Vec::new(),
                transfers: TransferStats::default(),
            });
        }
        let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("neighborhood-readback"),
            size: self.bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.bytes);
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
        let stride = 2 + 2 * self.capacity as usize;
        let mut rows = Vec::with_capacity(self.queries as usize);
        let mut totals = Vec::with_capacity(self.queries as usize);
        for row in words.chunks_exact(stride) {
            if row[0] > self.capacity
                || row[1] < row[0]
                || (self.kind == GpuNeighborhoodKind::NearestK && row[0] != row[1])
            {
                drop(mapped);
                staging.unmap();
                return Err(SpatialError::InvalidArgument(
                    "GPU neighborhood metadata is invalid".to_owned(),
                ));
            }
            rows.push(
                (0..row[0] as usize)
                    .map(|j| Neighbor {
                        index: row[2 + 2 * j] as usize,
                        distance_squared: f32::from_bits(row[3 + 2 * j]),
                    })
                    .collect(),
            );
            totals.push(row[1]);
        }
        drop(mapped);
        staging.unmap();
        let mut transfers = TransferStats::default();
        transfers.record(TransferDirection::DeviceToHost, self.bytes);
        Ok(GpuNeighborhoodReadback { rows, totals, transfers })
    }
}

/// CPU rows from an explicit readback, with radius-overflow information kept
/// separate from stored neighbors. Complete access rejects any incomplete row.
pub struct GpuNeighborhoodReadback {
    rows: Vec<Vec<Neighbor>>,
    totals: Vec<u32>,
    transfers: TransferStats,
}

impl GpuNeighborhoodReadback {
    /// True if every radius match fit its row (always true for nearest-k).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.rows.iter().zip(&self.totals).all(|(row, &total)| row.len() == total as usize)
    }

    /// Exact radius match counts, or selected counts for nearest-k.
    #[must_use]
    pub fn total_counts(&self) -> &[u32] {
        &self.totals
    }

    /// Explicit bounded row access, allowing radius truncation. This is a
    /// traversal prefix, not necessarily the closest points inside the radius.
    #[must_use]
    pub fn truncated_rows(&self) -> &[Vec<Neighbor>] {
        &self.rows
    }

    /// Complete rows, or an error identifying the first overflowed radius row.
    pub fn complete_rows(&self) -> SpatialResult<&[Vec<Neighbor>]> {
        self.check_complete()?;
        Ok(&self.rows)
    }

    /// Consumes this readback, rejecting radius overflow before returning rows.
    pub fn into_complete_rows(self) -> SpatialResult<Vec<Vec<Neighbor>>> {
        self.check_complete()?;
        Ok(self.rows)
    }

    /// Payload bytes transferred during this explicit readback.
    #[must_use]
    pub const fn transfers(&self) -> TransferStats {
        self.transfers
    }

    fn check_complete(&self) -> SpatialResult<()> {
        if let Some((i, (row, &total))) = self
            .rows
            .iter()
            .zip(&self.totals)
            .enumerate()
            .find(|(_, (row, total))| row.len() < **total as usize)
        {
            return Err(SpatialError::InvalidArgument(format!(
                "radius neighborhood row {i} contains {total} matches but stores only {}; increase capacity before consuming complete neighborhoods", row.len()
            )));
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    root: u32,
    capacity: u32,
    bound: f32,
}

pub(crate) struct NeighborhoodPipelines {
    layout: wgpu::BindGroupLayout,
    nearest: wgpu::ComputePipeline,
    radius: wgpu::ComputePipeline,
}

impl NeighborhoodPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..5)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: binding != 4 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("neighborhood-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("neighborhood-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("neighborhood-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/neighborhoods.wgsl").into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("neighborhood-pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self { nearest: pipeline("nearest_k"), radius: pipeline("radius_neighbors"), layout }
    }
}

fn plan(
    queries: u64,
    capacity: u64,
    kind: GpuNeighborhoodKind,
    limits: &wgpu::Limits,
) -> SpatialResult<u64> {
    if capacity > u64::from((u32::MAX - 2) / 2)
        || (kind == GpuNeighborhoodKind::NearestK && capacity > u64::from(MAX_GPU_K_NEIGHBORS))
    {
        return Err(invalid());
    }
    check_compute(limits, queries)?;
    buffer_bytes(queries, 16, limits)?;
    let stride = capacity.checked_mul(8).and_then(|v| v.checked_add(8)).ok_or_else(invalid)?;
    let bytes = buffer_bytes(queries, stride, limits)?;
    // Also bound shader word offsets and CPU decoding capacities before work.
    let host_slots = queries
        .checked_mul(capacity)
        .and_then(|n| n.checked_mul(std::mem::size_of::<Neighbor>() as u64))
        .ok_or_else(invalid)?;
    let host_rows =
        queries.checked_mul(std::mem::size_of::<Vec<Neighbor>>() as u64).ok_or_else(invalid)?;
    if bytes / 4 > u64::from(u32::MAX)
        || host_slots > isize::MAX as u64
        || host_rows > isize::MAX as u64
    {
        return Err(invalid());
    }
    Ok(bytes)
}

impl GpuKdTree {
    /// Submits up to k nearest neighbors for each explicitly uploaded query.
    /// Results stay on GPU, sorted by distance with the CPU KD-tree selection
    /// and stable tie order. Supports k=0..=64; includes self/duplicate points.
    /// No payload upload/readback or host wait occurs. Unsupported sizes error
    /// before allocation. Finite targets/queries are required at upload; GPU
    /// f32 rounding/underflow may differ across backends near search boundaries.
    pub fn nearest_k_resident(
        &self,
        runtime: &WgpuRuntime,
        queries: &GpuQueryBatch,
        k: u32,
    ) -> SpatialResult<GpuNeighborhoods> {
        self.neighborhoods(runtime, queries, k, f32::INFINITY, GpuNeighborhoodKind::NearestK)
    }

    /// Submits radius search (radius is not squared) with bounded row capacity.
    /// All matches are counted on GPU; the first capacity matches are stored
    /// in CPU traversal order. Capacity=0 performs count-only search. Negative/
    /// NaN radii produce empty rows; infinite/overflowing radius squares are
    /// unbounded. Self/duplicate points are included. No silent truncation:
    /// readback complete access rejects overflow, and device consumers must
    /// inspect total counts. Target/query payloads are reused without copies.
    pub fn radius_neighbors_resident(
        &self,
        runtime: &WgpuRuntime,
        queries: &GpuQueryBatch,
        radius: f32,
        capacity: u32,
    ) -> SpatialResult<GpuNeighborhoods> {
        self.neighborhoods(
            runtime,
            queries,
            capacity,
            if radius < 0.0 { f32::NAN } else { radius * radius },
            GpuNeighborhoodKind::Radius,
        )
    }

    fn neighborhoods(
        &self,
        runtime: &WgpuRuntime,
        queries: &GpuQueryBatch,
        capacity: u32,
        bound: f32,
        kind: GpuNeighborhoodKind,
    ) -> SpatialResult<GpuNeighborhoods> {
        let identity = runtime.profiling_identity();
        if !Arc::ptr_eq(&self.identity, &identity) || !Arc::ptr_eq(&queries.identity, &identity) {
            return Err(invalid());
        }
        let bytes =
            plan(u64::from(queries.count), u64::from(capacity), kind, &runtime.device().limits())?;
        let output = GpuNeighborhoods {
            buffer: runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("neighborhood-results"),
                size: bytes.max(8),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            queries: queries.count,
            capacity,
            bytes,
            kind,
            identity,
        };
        // Owned wgpu buffers start at zero, including unused slots and headers
        // on empty/invalid-gate paths. They never alias recycled scratch.
        if queries.count == 0
            || self.root.is_none()
            || bound.is_nan()
            || (kind == GpuNeighborhoodKind::NearestK && capacity == 0)
        {
            return Ok(output);
        }
        let cache = runtime
            .neighborhood_pipelines
            .get_or_init(|| NeighborhoodPipelines::new(runtime.device()));
        let params = Params { count: queries.count, root: self.root.unwrap(), capacity, bound };
        let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("neighborhood-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [
            &uniform,
            self.nodes.as_ref().unwrap(),
            self.points.as_ref().unwrap(),
            queries.buffer.as_ref().unwrap(),
            &output.buffer,
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
            label: Some("neighborhood-bindings"),
            layout: &cache.layout,
            entries: &entries,
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("neighborhood-search"),
                timestamp_writes: None,
            });
            pass.set_pipeline(if kind == GpuNeighborhoodKind::NearestK {
                &cache.nearest
            } else {
                &cache.radius
            });
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(queries.count.div_ceil(64), 1, 1);
        }
        runtime.queue().submit(Some(encoder.finish()));
        Ok(output)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "neighborhood_tests.rs"]
mod tests;
