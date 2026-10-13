//! Radius grid covariance preparation without a source KD-tree or neighbor rows.
use super::{invalid, plan as equation_plan, GpuGicpInput};
use crate::{
    kernels::point_estimators::{prepare_grid, Kind},
    scratch::ScratchBuffers,
    GpuIcpInput, WgpuRuntime,
};
use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult, TransferDirection};
use spatialrust_search::KdTree;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    origin: [f32; 4],
    dims: [u32; 4],
    inv_cell: f32,
    radius_sq: f32,
    epsilon: f32,
    capacity: u32,
    offset: u32,
    total: u32,
    padding: [u32; 2],
}

fn grid_plan(
    source: u64,
    target: u64,
    radius: f32,
    capacity: u32,
    epsilon: f32,
    limits: &wgpu::Limits,
) -> SpatialResult<u64> {
    let total = source.checked_add(target).ok_or_else(invalid)?;
    if source == 0
        || target == 0
        || source > 1 << 24
        || target > 1 << 24
        || capacity == 0
        || !radius.is_finite()
        || radius <= 0.0
        || !(radius * radius).is_normal()
        || !(1e-4..=1.0).contains(&epsilon)
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_invocations_per_workgroup < 256
        || total.div_ceil(256) > u64::from(limits.max_compute_workgroups_per_dimension)
        || total
            .checked_add(1)
            .and_then(|n| n.checked_mul(8))
            .map_or(true, |n| n > u64::from(u32::MAX))
    {
        return Err(invalid());
    }
    equation_plan(source, total + 1, limits)?;
    Ok((total + 1) * 32)
}

impl GpuGicpInput {
    /// Uploads original positions/target tree and prepares radius covariances
    /// directly on GPU using CPU-built counting-sort grids. No source KD-tree,
    /// neighborhood rows or covariance readback/re-upload is needed.
    /// Capacity includes self/duplicates; overflow invalidates both sets/all
    /// equations. Dense grids must fit the existing cell/device limits; there
    /// is no automatic switch to the bounded-neighborhood or CPU paths.
    /// The existing grid's raw f32 means, central sums and 16 three-pivot sweeps
    /// are retained, including sparse epsilon*I. Numerical failure rejects.
    /// Requires nonempty finite equal-length source axes/target tree, positive
    /// capacity/radius with normal finite f32 square, epsilon=1e-4..=1, and
    /// supported reduction limits. Each dataset is bounded to 2^24 points for
    /// exact f32 support counts. CPU grid/tree preparation remains explicit.
    /// H2D is 20*S+36*T+32*target_nodes+4*(source_cells+target_cells+2);
    /// there is no preparation D2H. Prepared inputs own covariance storage.
    pub fn upload_with_gpu_grid_covariances(
        runtime: &WgpuRuntime,
        source: (&[f32], &[f32], &[f32]),
        target: &KdTree,
        radius: f32,
        capacity: u32,
        epsilon: f32,
    ) -> SpatialResult<Self> {
        let (x, y, z) = source;
        for axis in [y, z] {
            if axis.len() != x.len() {
                return Err(SpatialError::BufferLengthMismatch {
                    expected: x.len(),
                    found: axis.len(),
                });
            }
        }
        let target_axes = target.layout().positions();
        let targets = target_axes.0.len();
        let bytes = grid_plan(
            x.len() as u64,
            targets as u64,
            radius,
            capacity,
            epsilon,
            &runtime.device().limits(),
        )?;
        let grids = [
            prepare_grid(runtime, Kind::PlaneCovariances, x, y, z, radius)?,
            prepare_grid(
                runtime,
                Kind::PlaneCovariances,
                target_axes.0,
                target_axes.1,
                target_axes.2,
                radius,
            )?,
        ];
        let points: Vec<_> = (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect();
        let positions = GpuIcpInput::upload(runtime, &points, target)?;
        let covariances = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gicp-grid-covariances"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let total = (x.len() + targets) as u32;
        let counts = [x.len() as u32, targets as u32];
        let offsets = [0, x.len() as u32];
        let cache = runtime
            .gicp_grid_covariance_pipelines
            .get_or_init(|| GridPipelines::new(runtime.device()));
        let mut scratch = ScratchBuffers::new(runtime);
        let mut transfers = positions.upload_transfers();
        let groups: Vec<_> = grids
            .iter()
            .enumerate()
            .map(|(i, (origin, dims, sorted, starts))| {
                let params = Params {
                    origin: [origin[0], origin[1], origin[2], 0.0],
                    dims: [dims[0], dims[1], dims[2], counts[i]],
                    inv_cell: 1.0 / radius,
                    radius_sq: radius * radius,
                    epsilon,
                    capacity,
                    offset: offsets[i],
                    total,
                    padding: [0; 2],
                };
                let uniform =
                    runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("gicp-grid-params"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                let aux = [sorted, starts].map(|words| {
                    let data = bytemuck::cast_slice(words);
                    let buffer = scratch.acquire("gicp-grid-index", data.len() as u64);
                    runtime.queue().write_buffer(&buffer, 0, data);
                    transfers.record(TransferDirection::HostToDevice, data.len() as u64);
                    buffer
                });
                let queries = if i == 0 { &positions.source } else { &positions.target };
                let buffers = [&uniform, queries.buffer().unwrap(), &aux[0], &aux[1], &covariances];
                let entries: Vec<_> = buffers
                    .iter()
                    .enumerate()
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect();
                runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gicp-grid-bindings"),
                    layout: &cache.layout,
                    entries: &entries,
                })
            })
            .collect();
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gicp-grid-build"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cache.build);
            for (i, group) in groups.iter().enumerate() {
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(counts[i].div_ceil(256), 1, 1);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gicp-grid-finalize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cache.finalize);
            pass.set_bind_group(0, &groups[0], &[]);
            pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
        }
        scratch.submit_and_recycle(encoder);
        Ok(Self {
            positions,
            covariances,
            weight_scale: 6.0 / epsilon,
            covariance_offset: 1,
            transfers,
        })
    }
}

// Adapt only storage access/output and add bounded-support checks to the one
// canonical legacy grid kernel. Its arithmetic/order is not duplicated. Exact
// anchors deliberately fail tests if the canonical interface changes.
fn shader_source() -> String {
    let old = crate::kernels::covariances_grid::COV_GRID_WGSL;
    let mut body = old[old.find("fn rotate(").expect("canonical grid rotate")..].to_owned();
    for (axis, component) in [("xs", "x"), ("ys", "y"), ("zs", "z")] {
        for index in ["i", "j"] {
            let from = format!("{axis}[{index}]");
            assert!(body.contains(&from));
            body = body.replace(&from, &format!("points[{index}].{component}"));
        }
    }
    let edits=[
        ("count = count + 1.0;", "count = count + 1.0;\n                    if u32(count) > params.capacity { atomicOr(&result.status,8u); return; }"),
        ("out_cov[i] = vec4<f32>(params.epsilon, params.epsilon, params.epsilon, 0.0);", "store_covariance(i,vec4<f32>(params.epsilon,params.epsilon,params.epsilon,0.0),vec4<f32>(0.0));"),
        ("out_cov[params.dims.w + i] = vec4<f32>(0.0);", ""),
        ("out_cov[i] = vec4<f32>(reg[0], reg[1], reg[2], reg[3]);", "store_covariance(i,vec4<f32>(reg[0],reg[1],reg[2],reg[3]),vec4<f32>(reg[4],reg[5],0.0,0.0));"),
        ("out_cov[params.dims.w + i] = vec4<f32>(reg[4], reg[5], 0.0, 0.0);", ""),
    ];
    for (from, to) in edits {
        assert_eq!(body.matches(from).count(), 1);
        body = body.replace(from, to);
    }
    for from in ["let end = cell_start[row + after_x];", "let j = sorted[s];"] {
        assert_eq!(body.matches(from).count(), 2);
        let check = if from.starts_with("let end") {
            "if begin>end || end>params.dims.w {atomicOr(&result.status,1u);return;}"
        } else {
            "if j>=params.dims.w {atomicOr(&result.status,1u);return;}"
        };
        body = body.replace(from, &format!("{from}\n                {check}"));
    }
    // The legacy recycled-output comment is irrelevant to owned packed rows.
    body =
        body.replace("// Recycled output storage is uninitialized; overwrite the second row.", "");
    format!("{}\n{body}", include_str!("../shaders/gicp_grid_covariances.wgsl"))
}

pub(crate) struct GridPipelines {
    layout: wgpu::BindGroupLayout,
    build: wgpu::ComputePipeline,
    finalize: wgpu::ComputePipeline,
}
impl GridPipelines {
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
            label: Some("gicp-grid-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gicp-grid-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gicp-grid-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source().into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gicp-grid-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self { build: pipeline("main"), finalize: pipeline("finalize"), layout }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "gicp_grid_covariance_tests.rs"]
mod tests;
