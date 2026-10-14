use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult, TransferDirection, TransferStats};
use spatialrust_search::{KdTree, KdTreeNode, Neighbor};
use wgpu::util::DeviceExt;

use crate::{scratch::ScratchBuffers, WgpuRuntime};

#[path = "tree_staging.rs"]
mod tree_staging;

#[path = "tree_gpu_gather.rs"]
pub(crate) mod tree_gpu_gather;

#[path = "neighborhoods.rs"]
mod neighborhoods;
pub(crate) use neighborhoods::NeighborhoodPipelines;
pub use neighborhoods::{
    GpuNeighborhoodKind, GpuNeighborhoodReadback, GpuNeighborhoods, GpuQueryBatch,
    MAX_GPU_K_NEIGHBORS,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Node {
    split: f32,
    axis: u32,
    left: u32,
    right: u32,
    start: u32,
    end: u32,
    padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Point {
    xyz: [f32; 3],
    index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    root: u32,
    bound: f32,
    unused: u32,
}

pub(crate) struct NearestPipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl NearestPipeline {
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
            label: Some("nearest-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nearest-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nearest-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/nearest.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nearest-pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("nearest"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { layout, pipeline }
    }
}

fn invalid() -> SpatialError {
    SpatialError::InvalidArgument("GPU nearest search requires a matching runtime, finite target and supported buffer/dispatch limits".to_owned())
}

fn buffer_bytes(count: u64, stride: u64, limits: &wgpu::Limits) -> SpatialResult<u64> {
    let bytes = count.checked_mul(stride).ok_or_else(invalid)?;
    if count > u64::from(u32::MAX)
        || bytes > limits.max_buffer_size
        || bytes > u64::from(limits.max_storage_buffer_binding_size)
        || bytes > isize::MAX as u64
    {
        return Err(invalid());
    }
    Ok(bytes)
}

fn check_compute(limits: &wgpu::Limits, queries: u64) -> SpatialResult<()> {
    if queries.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
        || limits.max_compute_invocations_per_workgroup < 64
        || limits.max_storage_buffers_per_shader_stage < 4
        || limits.max_bindings_per_bind_group < 5
        || limits.max_bind_groups == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < 16
        || limits.max_buffer_size < 32
        || limits.max_storage_buffer_binding_size < 32
    {
        return Err(invalid());
    }
    Ok(())
}

/// Immutable runtime-owned KD-tree uploaded explicitly and reused across batches.
/// Construction/partitioning is on CPU; nearest-point traversal is on GPU.
/// No device-to-host transfer is performed by upload, and no CPU search fallback
/// is used. Public methods validate device identity before resource use.
pub struct GpuKdTree {
    nodes: Option<wgpu::Buffer>,
    points: Option<wgpu::Buffer>,
    root: Option<u32>,
    identity: Arc<()>,
    upload_transfers: TransferStats,
}

impl GpuKdTree {
    // Internal device consumer: queries/results remain private to its encoder.
    // Reuses the ordinary bounded nearest shader and tie traversal exactly.
    pub(crate) fn encode_nearest(
        &self,
        runtime: &WgpuRuntime,
        encoder: &mut wgpu::CommandEncoder,
        queries: &wgpu::Buffer,
        count: u32,
        bound: f32,
        results: &wgpu::Buffer,
    ) -> SpatialResult<()> {
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity()) || self.root.is_none() {
            return Err(invalid());
        }
        check_compute(&runtime.device().limits(), u64::from(count))?;
        let params = Params { count, root: self.root.unwrap(), bound, unused: 0 };
        let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resident-nearest-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let cache = runtime.nearest_pipeline.get_or_init(|| NearestPipeline::new(runtime.device()));
        let buffers = [
            &uniform,
            self.nodes.as_ref().unwrap(),
            self.points.as_ref().unwrap(),
            queries,
            results,
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
            label: Some("resident-nearest-bindings"),
            layout: &cache.layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resident-nearest"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&cache.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        Ok(())
    }

    /// Uploads a finite CPU KD-tree, preserving its leaf and tie traversal order.
    /// The borrowed tree may be dropped afterwards. Upload accounting includes
    /// 16 bytes per point and 32 per node, excluding uniform control data.
    pub fn upload(runtime: &WgpuRuntime, tree: &KdTree) -> SpatialResult<Self> {
        let layout = tree.layout();
        let (x, y, z) = layout.positions();
        let limits = runtime.device().limits();
        let point_bytes = buffer_bytes(x.len() as u64, 16, &limits)?;
        let node_bytes = buffer_bytes(layout.nodes().len() as u64, 32, &limits)?;
        check_compute(&limits, 0)?;
        let points = tree_staging::prepare([x, y, z], layout.point_order())?;
        let mut target = Self {
            nodes: None,
            points: None,
            root: layout.root(),
            identity: runtime.profiling_identity(),
            upload_transfers: TransferStats::default(),
        };
        if x.is_empty() {
            return Ok(target);
        }
        let nodes: Vec<_> = layout
            .nodes()
            .map(|node| {
                let mut output = Node::zeroed();
                match node {
                    KdTreeNode::Leaf { start, end } => {
                        output.axis = u32::MAX;
                        output.start = start;
                        output.end = end;
                    }
                    KdTreeNode::Branch { axis, split, left, right } => {
                        output.axis = u32::from(axis);
                        output.split = split;
                        output.left = left;
                        output.right = right;
                    }
                }
                output
            })
            .collect();
        let upload = |label, data: &[u8]| {
            runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        target.nodes = Some(upload("nearest-nodes", bytemuck::cast_slice(&nodes)));
        target.points = Some(upload("nearest-points", bytemuck::cast_slice(&points)));
        target.upload_transfers.record(TransferDirection::HostToDevice, point_bytes + node_bytes);
        Ok(target)
    }

    /// Explicitly uploads original SoA axes/indices and gathers point records on GPU.
    /// CPU partitioning, finite validation and node packing remain. Existing
    /// nearest-search device requirements apply; unsupported limits reject.
    /// The pass copies integer words, preserving coordinate bits and leaf/tie order.
    /// Host upload is 16*N + 32*M bytes; the gather records 16*N device-to-device
    /// bytes. There is no download or host completion wait.
    /// Two transient inputs use 16*N extra device bytes beside the owned output;
    /// there is no 16*N host point-packing vector. Submission retains inputs until
    /// completion, and later searches on this runtime follow queue ordering.
    pub fn upload_with_gpu_gather(runtime: &WgpuRuntime, tree: &KdTree) -> SpatialResult<Self> {
        Self::upload_gathered(runtime, tree, false).map(|(target, _)| target)
    }

    /// Uploads the tree and generates queries for every point in original input
    /// order, sharing the same integer gather pass and uploaded SoA axes/indices.
    /// Combined host upload is 16*N + 32*M bytes, compared with 32*N + 32*M
    /// for separate tree/query uploads. Each returned object records 16*N
    /// device-to-device output bytes; queries record zero host upload bytes.
    /// CPU partitioning, finite validation and node packing remain. No readback
    /// or completion wait occurs. Both outputs are owned and reusable; the CPU
    /// tree may be dropped. Two transient inputs use 16*N extra device bytes;
    /// no host point/query packing vectors are created. Device requirements
    /// match nearest search, with an additional query dispatch-size check.
    pub fn upload_and_queries_with_gpu_gather(
        runtime: &WgpuRuntime,
        tree: &KdTree,
    ) -> SpatialResult<(Self, GpuQueryBatch)> {
        let (target, queries) = Self::upload_gathered(runtime, tree, true)?;
        Ok((target, queries.unwrap()))
    }

    fn upload_gathered(
        runtime: &WgpuRuntime,
        tree: &KdTree,
        with_queries: bool,
    ) -> SpatialResult<(Self, Option<GpuQueryBatch>)> {
        Self::upload_gathered_impl(runtime, tree, with_queries, None)
            .map(|(target, queries, _)| (target, queries))
    }

    // FPFH validates unit normals first, then records their 12*N host bytes and
    // the 32*N sample output separately. Tree/query receipts keep their contract.
    pub(crate) fn upload_with_gpu_samples(
        runtime: &WgpuRuntime,
        tree: &KdTree,
        normals: &[[f32; 3]],
    ) -> SpatialResult<(Self, GpuQueryBatch, Option<wgpu::Buffer>)> {
        let (target, queries, samples) =
            Self::upload_gathered_impl(runtime, tree, true, Some(normals))?;
        Ok((target, queries.unwrap(), samples))
    }

    fn upload_gathered_impl(
        runtime: &WgpuRuntime,
        tree: &KdTree,
        with_queries: bool,
        normals: Option<&[[f32; 3]]>,
    ) -> SpatialResult<(Self, Option<GpuQueryBatch>, Option<wgpu::Buffer>)> {
        let layout = tree.layout();
        let (x, y, z) = layout.positions();
        let limits = runtime.device().limits();
        let point_bytes = buffer_bytes(x.len() as u64, 16, &limits)?;
        let node_bytes = buffer_bytes(layout.nodes().len() as u64, 32, &limits)?;
        check_compute(&limits, 0)?;
        tree_gpu_gather::check(x.len() as u64, &limits)?;
        if with_queries {
            check_compute(&limits, x.len() as u64)?;
        }
        if let Some(normals) = normals {
            if normals.len() != x.len() {
                return Err(invalid());
            }
            tree_gpu_gather::check_samples(x.len() as u64, &limits)?;
        }
        let mut target = Self {
            nodes: None,
            points: None,
            root: layout.root(),
            identity: runtime.profiling_identity(),
            upload_transfers: TransferStats::default(),
        };
        if x.is_empty() {
            return Ok((
                target,
                with_queries.then(|| GpuQueryBatch::from_gathered(runtime, None, 0)),
                None,
            ));
        }
        let (points, queries, samples) = if let Some(normals) = normals {
            tree_gpu_gather::upload_with_samples(runtime, [x, y, z], layout.point_order(), normals)?
        } else {
            let (points, queries) =
                tree_gpu_gather::upload(runtime, [x, y, z], layout.point_order(), with_queries)?;
            (points, queries, None)
        };
        target.points = Some(points);
        let queries = queries
            .map(|buffer| GpuQueryBatch::from_gathered(runtime, Some(buffer), x.len() as u32));
        let nodes: Vec<_> = layout
            .nodes()
            .map(|node| {
                let mut output = Node::zeroed();
                match node {
                    KdTreeNode::Leaf { start, end } => {
                        output.axis = u32::MAX;
                        output.start = start;
                        output.end = end;
                    }
                    KdTreeNode::Branch { axis, split, left, right } => {
                        output.axis = u32::from(axis);
                        output.split = split;
                        output.left = left;
                        output.right = right;
                    }
                }
                output
            })
            .collect();
        let upload = |label, data: &[u8]| {
            runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        target.nodes = Some(upload("nearest-nodes", bytemuck::cast_slice(&nodes)));
        target.upload_transfers.record(TransferDirection::HostToDevice, point_bytes + node_bytes);
        target.upload_transfers.record(TransferDirection::DeviceToDevice, point_bytes);
        Ok((target, queries, samples))
    }

    /// Bytes transferred when this target was explicitly uploaded.
    #[must_use]
    pub fn upload_transfers(&self) -> TransferStats {
        self.upload_transfers
    }

    /// Explicitly uploads queries and downloads one optional nearest result per
    /// query, in input order, using an inclusive squared-distance gate.
    /// Negative/NaN gates and nonfinite queries produce None as on CPU; positive
    /// infinity is unbounded. Finite target coordinates are required at upload.
    /// GPU f32 arithmetic can differ near rounding/underflow boundaries across
    /// backends. Equal computed distances retain CPU near-first traversal ties.
    /// Accounting is 16 query bytes uploaded and 8 result bytes downloaded per
    /// query (uniform controls excluded); empty/invalid-gate calls copy no data.
    /// This synchronous API waits for completion, including the readback.
    pub fn nearest_one_within_download(
        &self,
        runtime: &WgpuRuntime,
        queries: &[[f32; 3]],
        bound: f32,
    ) -> SpatialResult<(Vec<Option<Neighbor>>, TransferStats)> {
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity()) {
            return Err(invalid());
        }
        let limits = runtime.device().limits();
        let query_bytes = buffer_bytes(queries.len() as u64, 16, &limits)?;
        let result_bytes = buffer_bytes(queries.len() as u64, 8, &limits)?;
        check_compute(&limits, queries.len() as u64)?;
        let mut transfers = TransferStats::default();
        if queries.is_empty() || self.root.is_none() || bound.is_nan() || bound < 0.0 {
            return Ok((vec![None; queries.len()], transfers));
        }
        let packed: Vec<[f32; 4]> = queries.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        let mut scratch = ScratchBuffers::new(runtime);
        let query_buffer = scratch.acquire("nearest-queries", query_bytes);
        runtime.queue().write_buffer(&query_buffer, 0, bytemuck::cast_slice(&packed));
        let output = scratch.acquire("nearest-output", result_bytes);
        let params =
            Params { count: queries.len() as u32, root: self.root.unwrap(), bound, unused: 0 };
        let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nearest-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let cache = runtime.nearest_pipeline.get_or_init(|| NearestPipeline::new(runtime.device()));
        let buffers = [
            &uniform,
            self.nodes.as_ref().unwrap(),
            self.points.as_ref().unwrap(),
            &query_buffer,
            &output,
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
            label: Some("nearest-bindings"),
            layout: &cache.layout,
            entries: &entries,
        });
        let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("nearest-readback"),
            size: result_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nearest"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cache.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(params.count.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, result_bytes);
        scratch.submit_and_recycle(encoder);
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        runtime.device().poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|e| SpatialError::InvalidArgument(e.to_string()))?
            .map_err(|e| SpatialError::InvalidArgument(e.to_string()))?;
        let mapped = slice.get_mapped_range();
        let neighbors = bytemuck::cast_slice::<u8, [u32; 2]>(&mapped)
            .iter()
            .map(|word| {
                (word[0] != u32::MAX).then(|| Neighbor {
                    index: word[0] as usize,
                    distance_squared: f32::from_bits(word[1]),
                })
            })
            .collect();
        drop(mapped);
        staging.unmap();
        transfers.record(TransferDirection::HostToDevice, query_bytes);
        transfers.record(TransferDirection::DeviceToHost, result_bytes);
        Ok((neighbors, transfers))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "nearest_tests.rs"]
mod tests;
