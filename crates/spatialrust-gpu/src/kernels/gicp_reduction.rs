use crate::{scratch::ScratchBuffers, GpuIcpInput, WgpuRuntime};
use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult, TransferDirection, TransferStats};
use spatialrust_math::Isometry3;
use spatialrust_search::KdTree;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const RESULT_BYTES: u64 = 256;

#[path = "gicp_upload_preparation.rs"]
mod upload_preparation;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    rows: [[f32; 4]; 3],
    anchor: [f32; 4],
    count: u32,
    targets: u32,
    groups: u32,
    unused: u32,
    bound: f32,
    max_distance: f32,
    scale: f32,
    weight_scale: f32,
}

fn invalid() -> SpatialError {
    SpatialError::InvalidArgument("GPU GICP reduction requires finite SPD covariance rows, matching counts/runtime, supported finite geometry/weight scales, a finite unit pose, positive finite nonzero distance square and supported device limits".to_owned())
}

fn plan(source: u64, covariance_count: u64, limits: &wgpu::Limits) -> SpatialResult<u32> {
    let groups = source.div_ceil(64).min(256);
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    for (count, stride) in [(source, 48), (covariance_count, 32), (groups, 240)] {
        if count > u64::from(u32::MAX)
            || count
                .checked_mul(stride)
                .map_or(true, |bytes| bytes > bound || bytes > isize::MAX as u64)
        {
            return Err(invalid());
        }
    }
    if bound < RESULT_BYTES
        || limits.max_compute_workgroups_per_dimension < 28
        || source.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
        || limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_storage_size < 1024
        || limits.max_storage_buffers_per_shader_stage < 8
        || limits.max_bindings_per_bind_group < 9
        || limits.max_bind_groups == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < 96
    {
        return Err(invalid());
    }
    Ok(groups as u32)
}

// Nearest search finishes before queries become centered position/count rows.
// The remaining inverse weights/residual use three vec4 rows per source.
fn acquire_working_buffers(
    scratch: &mut ScratchBuffers<'_>,
    count: u32,
    groups: u32,
) -> [wgpu::Buffer; 4] {
    [
        scratch.acquire("gicp-queries-centered", u64::from(count) * 16),
        scratch.acquire("gicp-matches", u64::from(count) * 8),
        scratch.acquire("gicp-weights-residual", u64::from(count) * 48),
        scratch.acquire("gicp-partials", u64::from(groups) * 240),
    ]
}

// Validate exact packed-f32 SPD data in f64 and bound combined inverse weights
// by trace(target^-1), since C_target + R C_source R^T >= C_target.
fn inverse_trace(c: &[f32; 6]) -> SpatialResult<f64> {
    if c.iter().any(|v| !v.is_finite()) || c[0] <= 0.0 {
        return Err(invalid());
    }
    let [xx, yy, zz, xy, xz, yz] = c.map(f64::from);
    let a = xx.sqrt();
    let b = xy / a;
    let d = xz / a;
    let pivot1 = yy - b * b;
    if pivot1 <= 0.0 {
        return Err(invalid());
    }
    let e = pivot1.sqrt();
    let f = (yz - d * b) / e;
    let pivot2 = zz - d * d - f * f;
    if pivot2 <= 0.0 {
        return Err(invalid());
    }
    let g = pivot2.sqrt();
    let trace = (1.0 / a).powi(2)
        + (-b / (a * e)).powi(2)
        + (1.0 / e).powi(2)
        + ((b * f / e - d) / (a * g)).powi(2)
        + (-f / (e * g)).powi(2)
        + (1.0 / g).powi(2);
    if !trace.is_finite() || trace <= 0.0 {
        return Err(invalid());
    }
    Ok(trace)
}

/// Reusable explicitly uploaded positions/tree and SPD covariances for GICP.
/// Covariances are explicitly uploaded or prepared on GPU. Position storage shares the ICP input
/// representation; every evaluation transforms immutable original source points.
pub struct GpuGicpInput {
    positions: GpuIcpInput,
    covariances: wgpu::Buffer,
    weight_scale: f32,
    covariance_offset: u32,
    transfers: TransferStats,
}

impl GpuGicpInput {
    /// Uploads positions/tree plus original-order covariance rows packed as
    /// [xx,yy,zz,xy,xz,yz]. Covariances must be finite SPD in their f32 form.
    /// H2D payload is 48*S + 64*T + 32*M bytes; covariance rows are padded to 32
    /// bytes. A normal finite target-inverse trace scale is required.
    /// Native large covariance batches use bounded CPU staging workers; each
    /// row retains its f64 SPD validation and original upload order. Small
    /// batches and WASM prepare serially, before any device upload.
    pub fn upload(
        runtime: &WgpuRuntime,
        source: &[[f32; 3]],
        source_covariances: &[[f32; 6]],
        target: &KdTree,
        target_covariances: &[[f32; 6]],
    ) -> SpatialResult<Self> {
        if source.len() != source_covariances.len()
            || target.layout().positions().0.len() != target_covariances.len()
        {
            return Err(invalid());
        }
        let total = (source.len() as u64)
            .checked_add(target_covariances.len() as u64)
            .ok_or_else(invalid)?;
        plan(source.len() as u64, total, &runtime.device().limits())?;
        let (packed, weight_scale) =
            upload_preparation::prepare(source_covariances, target_covariances)?;
        let positions = GpuIcpInput::upload(runtime, source, target)?;
        let covariances = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gicp-covariances"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut transfers = positions.upload_transfers();
        transfers.record(TransferDirection::HostToDevice, total * 32);
        Ok(Self { positions, covariances, weight_scale, covariance_offset: 0, transfers })
    }

    /// Payload copied at construction; GPU-generated covariances add no H2D.
    #[must_use]
    pub const fn upload_transfers(&self) -> TransferStats {
        self.transfers
    }

    /// Original source count.
    #[must_use]
    pub fn source_count(&self) -> u32 {
        self.positions.source_count()
    }

    /// Transforms originals, runs unbounded nearest matching followed by the
    /// finite inclusive gate, prepares rotated covariance inverse weights and
    /// reduces centered/scaled normal equations and Euclidean fitness. Five
    /// passes share one submission; no payload transfer or host wait occurs.
    /// After matching, transformed queries become centered position/count rows.
    /// Private scratch is 72*S + 240*min(ceil(S/64),256) bytes for nonempty S.
    /// GPU Cholesky pivots <=1e-6 of covariance scale reject the whole result,
    /// rather than silently dropping ill-conditioned pairs. f32 arithmetic can
    /// change weights, matches, solutions and stopping relative to CPU f64.
    pub fn equations_resident(
        &self,
        runtime: &WgpuRuntime,
        transform: Isometry3<f32>,
        max_distance: f32,
    ) -> SpatialResult<GpuGicpNormalEquations> {
        let q = transform.rotation();
        let t = transform.translation();
        let norm = f64::from(q.x).hypot(f64::from(q.y)).hypot(f64::from(q.z)).hypot(f64::from(q.w));
        let bound = max_distance * max_distance;
        if !Arc::ptr_eq(&self.positions.identity, &runtime.profiling_identity())
            || !norm.is_finite()
            || (norm - 1.0).abs() > 1e-4
            || !t.x.is_finite()
            || !t.y.is_finite()
            || !t.z.is_finite()
            || !max_distance.is_finite()
            || max_distance <= 0.0
            || !bound.is_finite()
            || bound == 0.0
        {
            return Err(invalid());
        }
        let count = self.source_count();
        let groups = plan(
            u64::from(count),
            u64::from(count) + u64::from(self.positions.target.query_count()),
            &runtime.device().limits(),
        )?;
        let output = GpuGicpNormalEquations {
            buffer: runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("gicp-equations"),
                size: RESULT_BYTES,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            source_count: count,
            coordinate_scale: self.positions.scale,
            max_distance,
            weight_scale: self.weight_scale,
            identity: Arc::clone(&self.positions.identity),
        };
        if count == 0 {
            return Ok(output);
        }
        let r = q.to_mat3();
        let params = Params {
            rows: [
                [r.m[0][0], r.m[0][1], r.m[0][2], t.x],
                [r.m[1][0], r.m[1][1], r.m[1][2], t.y],
                [r.m[2][0], r.m[2][1], r.m[2][2], t.z],
            ],
            anchor: self.positions.source_anchor,
            count,
            targets: self.positions.target.query_count(),
            groups,
            unused: self.covariance_offset,
            bound,
            max_distance,
            scale: self.positions.scale,
            weight_scale: self.weight_scale,
        };
        let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gicp-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut scratch = ScratchBuffers::new(runtime);
        let [queries, matches, pairs, partials] =
            acquire_working_buffers(&mut scratch, count, groups);
        let cache = runtime
            .gicp_reduction_pipelines
            .get_or_init(|| GicpReductionPipelines::new(runtime.device()));
        let buffers = [
            &uniform,
            self.positions.source.buffer().unwrap(),
            self.positions.target.buffer().unwrap(),
            &queries,
            &matches,
            &self.covariances,
            &pairs,
            &partials,
            &output.buffer,
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
            label: Some("gicp-bindings"),
            layout: &cache.layout,
            entries: &entries,
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        dispatch(&mut encoder, &cache.pipelines[0], &group, count.div_ceil(64), 1);
        self.positions.tree.encode_nearest(
            runtime,
            &mut encoder,
            &queries,
            count,
            f32::INFINITY,
            &matches,
        )?;
        for (index, x, y) in [(1, count.div_ceil(64), 1), (2, groups, 28), (3, 1, 1)] {
            dispatch(&mut encoder, &cache.pipelines[index], &group, x, y);
        }
        scratch.submit_and_recycle(encoder);
        Ok(output)
    }
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    x: u32,
    y: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("gicp-reduction"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(x, y, 1);
}

/// Downloaded GICP normal equations in the documented normalized parameter basis.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuGicpEquations {
    /// Gated pairs contributing to supported normal equations.
    pub correspondences: u32,
    /// Gated pairs contributing to Euclidean squared residuals.
    pub fitness_correspondences: u32,
    /// Symmetric normalized 6x6 Hessian (not world-coordinate coefficients).
    pub hessian: [[f64; 6]; 6],
    /// Normalized gradient; solve H*u=-g before converting with world_step.
    pub gradient: [f64; 6],
    /// Actual device-transformed bounding-box anchor in world coordinates.
    pub anchor: [f64; 3],
    /// Rotation scale converting solved normalized coordinates to radians.
    pub rotation_step_scale: f64,
    /// Translation scale converting solved local coordinates to physical units.
    pub translation_step_scale: f64,
    /// Physical covariance-inverse weight scale divided out of H/g.
    pub weight_scale: f64,
    /// Physical Euclidean squared-error sum (not Mahalanobis objective or mean).
    pub squared_error_sum: f64,
    /// Payload copied during this explicit download.
    pub transfers: TransferStats,
}
impl GpuGicpEquations {
    /// Converts the solved normalized step to the world [-skew(p)|I] basis:
    /// omega=u_rotation*distance/coordinate_scale, local_t=u_translation*distance,
    /// world_t=local_t-omega cross anchor. This is a LINEARIZED basis conversion;
    /// the caller's existing finite rotation/update policy remains separate.
    pub fn world_step(&self, solution: [f64; 6]) -> SpatialResult<[f64; 6]> {
        let omega: [f64; 3] = std::array::from_fn(|i| solution[i] * self.rotation_step_scale);
        let [x, y, z] = self.anchor;
        let cross =
            [omega[1] * z - omega[2] * y, omega[2] * x - omega[0] * z, omega[0] * y - omega[1] * x];
        let step = std::array::from_fn(|i| {
            if i < 3 {
                omega[i]
            } else {
                solution[i] * self.translation_step_scale - cross[i - 3]
            }
        });
        if step.iter().any(|v| !v.is_finite()) {
            return Err(invalid());
        }
        Ok(step)
    }
}

/// Owned resident fixed-size normal equations/status, independent of input/pools.
pub struct GpuGicpNormalEquations {
    buffer: wgpu::Buffer,
    source_count: u32,
    coordinate_scale: f32,
    max_distance: f32,
    weight_scale: f32,
    identity: Arc<()>,
}
impl GpuGicpNormalEquations {
    /// 256-byte storage: u32 status/H-count/fitness-count/reserved-zero, one
    /// vec4<f32> actual anchor, then 28 compensated vec2<f32> sums. Terms 0..20
    /// are Hessian upper triangle by row, 21..26 gradient and 27 Euclidean error
    /// normalized by distance squared. Status bits 1/2/4/8 indicate transform,
    /// match, nonfinite arithmetic and unsupported covariance conditioning;
    /// bit 16 indicates invalid resident covariance preparation, including
    /// radius neighborhood capacity overflow. No truncated rows are consumed.
    /// Any nonzero status invalidates ALL payload, which is zeroed on device.
    /// GPU consumers must check status and preserve immutable application data.
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Metadata required to decode the resident normalized parameter basis.
    #[must_use]
    pub fn parameter_scales(&self) -> (f64, f64, f64) {
        (
            f64::from(self.max_distance) / f64::from(self.coordinate_scale),
            f64::from(self.max_distance),
            f64::from(self.weight_scale),
        )
    }

    /// Explicit readback size, zero for empty source and otherwise 256 bytes.
    #[must_use]
    pub const fn readback_bytes(&self) -> u64 {
        if self.source_count == 0 {
            0
        } else {
            RESULT_BYTES
        }
    }

    /// Copies and waits for fixed-size equations/status, validates them before
    /// decoding hi/lo f32 sums in f64. No query/pair/neighbor readback occurs.
    pub fn download(&self, runtime: &WgpuRuntime) -> SpatialResult<GpuGicpEquations> {
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity()) {
            return Err(invalid());
        }
        let mut words = vec![0u32; 64];
        let mut transfers = TransferStats::default();
        if self.source_count > 0 {
            let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("gicp-readback"),
                size: RESULT_BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = runtime.device().create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, RESULT_BYTES);
            runtime.queue().submit(Some(encoder.finish()));
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            runtime.device().poll(wgpu::Maintain::Wait);
            rx.recv()
                .map_err(|e| SpatialError::InvalidArgument(e.to_string()))?
                .map_err(|e| SpatialError::InvalidArgument(e.to_string()))?;
            let mapped = slice.get_mapped_range();
            words.copy_from_slice(bytemuck::cast_slice(&mapped));
            drop(mapped);
            staging.unmap();
            transfers.record(TransferDirection::DeviceToHost, RESULT_BYTES);
        }
        if words[0] != 0
            || words[1] != words[2]
            || words[2] > self.source_count
            || words[3] != 0
            || (words[2] == 0 && words[4..].iter().any(|&v| v != 0))
        {
            return Err(SpatialError::InvalidArgument(format!(
                "GPU GICP reduction invalid status/counts: {:?}",
                &words[..4]
            )));
        }
        let values: Vec<_> = words[4..].iter().map(|&v| f64::from(f32::from_bits(v))).collect();
        if values.iter().any(|v| !v.is_finite()) || values[3] != 0.0 {
            return Err(invalid());
        }
        let sum = |i: usize| values[4 + i * 2] + values[5 + i * 2];
        // Row prefixes of the upper triangle are 0, 6, 11, 15, 18, 20.
        let hessian = std::array::from_fn(|a: usize| {
            std::array::from_fn(|b: usize| {
                let (lo, hi) = (a.min(b), a.max(b));
                sum(lo * (13 - lo) / 2 + hi - lo)
            })
        });
        let gradient = std::array::from_fn(|i| sum(21 + i));
        let squared_error_sum = sum(27) * f64::from(self.max_distance * self.max_distance);
        if squared_error_sum < 0.0 || (0..6).any(|i| hessian[i][i] < 0.0) {
            return Err(invalid());
        }
        let (rotation_step_scale, translation_step_scale, weight_scale) = self.parameter_scales();
        Ok(GpuGicpEquations {
            correspondences: words[1],
            fitness_correspondences: words[2],
            hessian,
            gradient,
            anchor: [values[0], values[1], values[2]],
            rotation_step_scale,
            translation_step_scale,
            weight_scale,
            squared_error_sum,
            transfers,
        })
    }
}

pub(crate) struct GicpReductionPipelines {
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 4],
}
impl GicpReductionPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..9)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: matches!(binding, 1 | 2 | 4 | 5),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gicp-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gicp-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gicp-reduction"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/gicp_reduction.wgsl").into()),
        });
        let pipelines =
            ["transform_source", "prepare_pairs", "reduce_partials", "finalize"].map(|entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("gicp-pipeline"),
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

#[path = "gicp_covariances.rs"]
pub(crate) mod resident_covariances;

#[path = "gicp_grid_covariances.rs"]
pub(crate) mod resident_grid_covariances;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "gicp_reduction_tests.rs"]
mod tests;
