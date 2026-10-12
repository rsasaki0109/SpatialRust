use super::{invalid, plan, GpuGicpInput};
use crate::{GpuIcpInput, GpuKdTree, WgpuRuntime, MAX_GPU_K_NEIGHBORS};
use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialResult, TransferDirection};
use spatialrust_search::KdTree;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    sources: u32,
    targets: u32,
    neighbors: u32,
    epsilon: f32,
}

fn covariance_plan(
    source: u64,
    target: u64,
    k: u32,
    epsilon: f32,
    limits: &wgpu::Limits,
) -> SpatialResult<u64> {
    if source == 0
        || target == 0
        || !(3..=MAX_GPU_K_NEIGHBORS).contains(&k)
        || !(1e-4..=1.0).contains(&epsilon)
    {
        return Err(invalid());
    }
    let total = source.checked_add(target).and_then(|v| v.checked_add(1)).ok_or_else(invalid)?;
    plan(source, total, limits)?;
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    let stride = 8 + 8 * u64::from(k);
    for count in [source, target] {
        if count > u64::from(u32::MAX)
            || count.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
            || count
                .checked_mul(stride)
                .map_or(true, |v| v > bound || v / 4 > u64::from(u32::MAX) || v > isize::MAX as u64)
        {
            return Err(invalid());
        }
    }
    if (total - 1).div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
        || total.checked_mul(8).map_or(true, |v| v > u64::from(u32::MAX))
    {
        return Err(invalid());
    }
    Ok(total * 32)
}

impl GpuGicpInput {
    /// Explicitly uploads finite source/target trees and original-order positions,
    /// then computes both k-NN plane-regularized covariances entirely on GPU.
    /// Query positions and target tree are shared with the registration passes.
    /// Payload upload is 32*(S+T) + 32*(source_nodes+target_nodes) bytes; no
    /// neighborhoods/covariances are downloaded or re-uploaded. Three submissions
    /// prepare bounded neighborhoods, covariance eigenvectors and global status,
    /// without a host wait. Prepared covariance rows remain owned by this input.
    /// Both trees must be nonempty; k=3..=64 and epsilon=1e-4..=1 are supported.
    /// Eigenvectors use centered/scaled compensated f32 covariance and the CPU
    /// 32-step/physical 1e-12 Jacobi stopping convention. f32 arithmetic, ties,
    /// repeated eigenvalues and CPU raw-product rounding can change covariances.
    /// Invalid preparation makes ALL subsequent equation downloads fail (status
    /// bit 16); no CPU fallback occurs. CPU tree construction is external.
    pub fn upload_with_gpu_covariances(
        runtime: &WgpuRuntime,
        source: &KdTree,
        target: &KdTree,
        k: u32,
        epsilon: f32,
    ) -> SpatialResult<Self> {
        let (x, y, z) = source.layout().positions();
        let targets = target.layout().positions().0.len();
        let bytes = covariance_plan(
            x.len() as u64,
            targets as u64,
            k,
            epsilon,
            &runtime.device().limits(),
        )?;
        let bound = runtime
            .device()
            .limits()
            .max_buffer_size
            .min(u64::from(runtime.device().limits().max_storage_buffer_binding_size));
        for nodes in [source.layout().nodes().len(), target.layout().nodes().len()] {
            if (nodes as u64).checked_mul(32).map_or(true, |v| v > bound) {
                return Err(invalid());
            }
        }
        let points: Vec<_> = (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect();
        let positions = GpuIcpInput::upload(runtime, &points, target)?;
        let source_tree = GpuKdTree::upload(runtime, source)?;
        let source_neighbors = source_tree.nearest_k_resident(runtime, &positions.source, k)?;
        let target_neighbors = positions.tree.nearest_k_resident(runtime, &positions.target, k)?;
        let covariances = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gicp-resident-covariances"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params =
            Params { sources: x.len() as u32, targets: targets as u32, neighbors: k, epsilon };
        encode_covariances(
            runtime,
            &positions,
            [&source_neighbors, &target_neighbors],
            &covariances,
            params,
        );
        let mut transfers = positions.upload_transfers();
        transfers.record(
            TransferDirection::HostToDevice,
            source_tree.upload_transfers().host_to_device_bytes(),
        );
        // I + (epsilon-1)nn^T has lambda_min >= epsilon up to bounded f32
        // rounding. Factor two slack covers <= 1e-6 component errors at epsilon
        // >= 1e-4. The shader additionally rejects non-SPD packed rows.
        let weight_scale = 6.0 / epsilon;
        Ok(Self { positions, covariances, weight_scale, covariance_offset: 1, transfers })
    }
}

fn encode_covariances(
    runtime: &WgpuRuntime,
    positions: &GpuIcpInput,
    neighbors: [&crate::GpuNeighborhoods; 2],
    covariances: &wgpu::Buffer,
    params: Params,
) {
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gicp-covariance-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let cache = runtime
        .gicp_covariance_pipelines
        .get_or_init(|| GicpCovariancePipelines::new(runtime.device()));
    let buffers = [
        &uniform,
        positions.source.buffer().unwrap(),
        positions.target.buffer().unwrap(),
        neighbors[0].buffer(),
        neighbors[1].buffer(),
        covariances,
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
        label: Some("gicp-covariance-bindings"),
        layout: &cache.layout,
        entries: &entries,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    for pipeline in &cache.pipelines {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gicp-covariance"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((params.sources + params.targets).div_ceil(64), 1, 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
}

pub(crate) struct GicpCovariancePipelines {
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 2],
}
impl GicpCovariancePipelines {
    fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..6)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: binding != 5 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gicp-covariance-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gicp-covariance-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gicp-covariance"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/gicp_covariances.wgsl").into(),
            ),
        });
        let pipelines = ["build_covariances", "finalize"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gicp-covariance-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Self { layout, pipelines }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "gicp_covariance_tests.rs"]
mod tests;
