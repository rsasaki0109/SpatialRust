use crate::{scratch::ScratchBuffers, GpuIcpInput, WgpuRuntime};
use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult, TransferDirection, TransferStats};
use spatialrust_math::Isometry3;
use spatialrust_search::KdTree;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[path = "point_to_plane_normal_preparation.rs"]
mod normal_preparation;

const RESULT_BYTES: u64 = 256;

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
    normal_scale: f32,
}

fn invalid() -> SpatialError {
    SpatialError::InvalidArgument("GPU point-to-plane ICP reduction requires finite raw target normals, matching counts/runtime, supported finite geometry/normal scales, a finite unit pose, positive finite nonzero distance square and supported device limits".to_owned())
}

fn plan(source: u64, normal_count: u64, limits: &wgpu::Limits) -> SpatialResult<u32> {
    let groups = source.div_ceil(64).min(256);
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    for (count, stride) in [(source, 16), (normal_count, 16), (groups, 240)] {
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

// The transformed queries become rotation Jacobian/residual rows after the
// nearest-search pass. Only the normal/count half of each pair needs a separate
// buffer; both halves are fully written before the reductions read them.
fn acquire_working_buffers(
    scratch: &mut ScratchBuffers<'_>,
    count: u32,
    groups: u32,
) -> [wgpu::Buffer; 4] {
    [
        scratch.acquire("point_to_plane-queries-rotation", u64::from(count) * 16),
        scratch.acquire("point_to_plane-matches", u64::from(count) * 8),
        scratch.acquire("point_to_plane-pair-normals", u64::from(count) * 16),
        scratch.acquire("point_to_plane-partials", u64::from(groups) * 240),
    ]
}

/// Reusable explicitly uploaded positions/tree and original-order target normals.
/// A single common normal scale preserves relative raw-normal weights; normals
/// are never normalized individually. Source positions remain immutable.
pub struct GpuPointToPlaneInput {
    positions: GpuIcpInput,
    normals: wgpu::Buffer,
    normal_scale: f32,
    transfers: TransferStats,
}

impl GpuPointToPlaneInput {
    /// Uploads original source positions, target tree and original-order raw
    /// target normals. Normals must be finite; zero rows retain their zero weight.
    /// Nonzero components must remain normal f32 after common scaling. H2D
    /// payload is 16*S + 48*T + 32*M bytes (M = target tree nodes).
    pub fn upload(
        runtime: &WgpuRuntime,
        source: &[[f32; 3]],
        target: &KdTree,
        target_normals: &[[f32; 3]],
    ) -> SpatialResult<Self> {
        Self::upload_impl(runtime, source, target, target_normals, false)
    }

    /// Explicitly uses shared GPU gathering for target tree points/queries.
    /// Host payload is 16*S + 32*T + 32*M bytes, compared with 16*S + 48*T + 32*M
    /// for `upload`; device output bytes are 32*T. Raw-normal validation/packing and common normal scale are unchanged.
    /// Source/geometry preparation remains on CPU. Two transient inputs add
    /// 16*T GPU bytes; host target packing totaling 32*T payload bytes over
    /// construction is removed. No readback or completion wait is added.
    /// Placement is explicit; small inputs may be slower and existing device
    /// requirements apply. See `GpuIcpInput::upload_with_gpu_gather`.
    pub fn upload_with_gpu_gather(
        runtime: &WgpuRuntime,
        source: &[[f32; 3]],
        target: &KdTree,
        target_normals: &[[f32; 3]],
    ) -> SpatialResult<Self> {
        Self::upload_impl(runtime, source, target, target_normals, true)
    }

    fn upload_impl(
        runtime: &WgpuRuntime,
        source: &[[f32; 3]],
        target: &KdTree,
        target_normals: &[[f32; 3]],
        gpu_gather: bool,
    ) -> SpatialResult<Self> {
        if target.layout().positions().0.len() != target_normals.len() {
            return Err(invalid());
        }
        plan(source.len() as u64, target_normals.len() as u64, &runtime.device().limits())?;
        let (packed, normal_scale) = normal_preparation::prepare(target_normals)?;
        let positions = if gpu_gather {
            GpuIcpInput::upload_with_gpu_gather(runtime, source, target)?
        } else {
            GpuIcpInput::upload(runtime, source, target)?
        };
        let normals = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("point-to-plane-normals"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut transfers = positions.upload_transfers();
        transfers.record(TransferDirection::HostToDevice, target_normals.len() as u64 * 16);
        Ok(Self { positions, normals, normal_scale, transfers })
    }

    /// All position/tree/normal payload uploaded at construction.
    #[must_use]
    pub const fn upload_transfers(&self) -> TransferStats {
        self.transfers
    }

    /// Original source count.
    #[must_use]
    pub fn source_count(&self) -> u32 {
        self.positions.source_count()
    }

    /// Transforms originals, runs unbounded nearest matching then an inclusive
    /// Euclidean distance gate, and reduces centered/scaled Hessian/gradient and
    /// raw-normal point-to-plane squared fitness. Five passes share a submission;
    /// no payload transfer or host wait occurs. Transformed query storage is
    /// reused for rotation/residual rows after matching. Private scratch is
    /// 40*S + 240*min(ceil(S/64),256) bytes.
    /// f32 operations can change matches, weights, solutions and stopping
    /// relative to the existing CPU f64 path.
    pub fn equations_resident(
        &self,
        runtime: &WgpuRuntime,
        transform: Isometry3<f32>,
        max_distance: f32,
    ) -> SpatialResult<GpuPointToPlaneNormalEquations> {
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
            u64::from(self.positions.target.query_count()),
            &runtime.device().limits(),
        )?;
        let output = GpuPointToPlaneNormalEquations {
            buffer: runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("point_to_plane-equations"),
                size: RESULT_BYTES,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            source_count: count,
            coordinate_scale: self.positions.scale,
            max_distance,
            normal_scale: self.normal_scale,
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
            unused: 0,
            bound,
            max_distance,
            scale: self.positions.scale,
            normal_scale: self.normal_scale,
        };
        let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("point_to_plane-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut scratch = ScratchBuffers::new(runtime);
        let [queries, matches, pair_normals, partials] =
            acquire_working_buffers(&mut scratch, count, groups);
        let cache = runtime
            .point_to_plane_reduction_pipelines
            .get_or_init(|| PointToPlaneReductionPipelines::new(runtime.device()));
        let buffers = [
            &uniform,
            self.positions.source.buffer().unwrap(),
            self.positions.target.buffer().unwrap(),
            &queries,
            &matches,
            &self.normals,
            &pair_normals,
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
            label: Some("point_to_plane-bindings"),
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
        label: Some("point_to_plane-reduction"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(x, y, 1);
}

/// Downloaded point-to-plane ICP normal equations in the documented normalized parameter basis.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuPointToPlaneEquations {
    /// Gated pairs contributing to supported normal equations.
    pub correspondences: u32,
    /// Gated pairs contributing to raw-normal squared plane residuals.
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
    /// Common raw-normal component scale divided out of H/g.
    pub normal_scale: f64,
    /// Physical raw-normal point-to-plane squared-error sum (not mean).
    pub squared_error_sum: f64,
    /// Payload copied during this explicit download.
    pub transfers: TransferStats,
}
impl GpuPointToPlaneEquations {
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
pub struct GpuPointToPlaneNormalEquations {
    buffer: wgpu::Buffer,
    source_count: u32,
    coordinate_scale: f32,
    max_distance: f32,
    normal_scale: f32,
    identity: Arc<()>,
}
impl GpuPointToPlaneNormalEquations {
    /// 256-byte storage: u32 status/H-count/fitness-count/reserved-zero, one
    /// vec4<f32> actual anchor, then 28 compensated vec2<f32> sums. Terms 0..20
    /// are Hessian upper triangle by row, 21..26 gradient and 27 raw-normal plane error
    /// normalized by distance squared and normal scale squared. Status bits 1/2/4/8 indicate transform,
    /// match, nonfinite arithmetic and reserved future conditions.
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
            f64::from(self.normal_scale),
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
    pub fn download(&self, runtime: &WgpuRuntime) -> SpatialResult<GpuPointToPlaneEquations> {
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity()) {
            return Err(invalid());
        }
        let mut words = vec![0u32; 64];
        let mut transfers = TransferStats::default();
        if self.source_count > 0 {
            let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("point_to_plane-readback"),
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
                "GPU point-to-plane ICP reduction invalid status/counts: {:?}",
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
        let squared_error_sum =
            sum(27) * f64::from(self.max_distance).powi(2) * f64::from(self.normal_scale).powi(2);
        if squared_error_sum < 0.0 || (0..6).any(|i| hessian[i][i] < 0.0) {
            return Err(invalid());
        }
        let (rotation_step_scale, translation_step_scale, normal_scale) = self.parameter_scales();
        Ok(GpuPointToPlaneEquations {
            correspondences: words[1],
            fitness_correspondences: words[2],
            hessian,
            gradient,
            anchor: [values[0], values[1], values[2]],
            rotation_step_scale,
            translation_step_scale,
            normal_scale,
            squared_error_sum,
            transfers,
        })
    }
}

pub(crate) struct PointToPlaneReductionPipelines {
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 4],
}
impl PointToPlaneReductionPipelines {
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
            label: Some("point_to_plane-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("point_to_plane-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point_to_plane-reduction"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/point_to_plane_reduction.wgsl").into(),
            ),
        });
        let pipelines =
            ["transform_source", "prepare_pairs", "reduce_partials", "finalize"].map(|entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("point_to_plane-pipeline"),
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
#[path = "point_to_plane_reduction_tests.rs"]
mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "point_to_plane_reduction_gather_reference.rs"]
pub(crate) mod gather_reference;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "point_to_plane_preparation_reference.rs"]
pub(crate) mod preparation_reference;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "point_to_plane_normal_upload_tests.rs"]
mod normal_upload_tests;
