use crate::{scratch::ScratchBuffers, GpuKdTree, GpuQueryBatch, WgpuRuntime};
use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult, TransferDirection, TransferStats};
use spatialrust_math::Isometry3;
use spatialrust_search::KdTree;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const RESULT_BYTES: u64 = 176;

#[path = "icp_geometry_preparation.rs"]
mod geometry_preparation;

#[path = "icp_trim.rs"]
pub(crate) mod trim;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    rows: [[f32; 4]; 3],
    source_anchor: [f32; 4],
    target_anchor: [f32; 4],
    count: u32,
    targets: u32,
    groups: u32,
    unused: u32,
    bound: f32,
    max_distance: f32,
    scale: f32,
    padding: f32,
}

fn invalid() -> SpatialError {
    SpatialError::InvalidArgument("GPU ICP reduction requires matching runtime, finite XYZ/extents, a finite unit-quaternion transform, a positive finite nonzero distance square, nonempty target and supported limits".to_owned())
}

fn plan(count: u64, limits: &wgpu::Limits) -> SpatialResult<u32> {
    let groups = count.div_ceil(64).min(256);
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    if count > u64::from(u32::MAX)
        || count.checked_mul(16).map_or(true, |n| n > bound || n > isize::MAX as u64)
        || groups * 144 > bound
        || bound < RESULT_BYTES
        || limits.max_compute_workgroups_per_dimension < 9
        || count.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
        || limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_storage_size < 768
        || limits.max_storage_buffers_per_shader_stage < 6
        || limits.max_bindings_per_bind_group < 7
        || limits.max_bind_groups == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < 112
    {
        return Err(invalid());
    }
    Ok(groups as u32)
}

fn geometry(points: &[[f32; 3]]) -> SpatialResult<([f32; 4], f64)> {
    geometry_preparation::prepare(points)
}

/// Explicitly uploaded source and target for reusable device ICP moments.
/// CPU target-tree construction is external. Original source points remain
/// immutable; every evaluation applies the absolute pose to those originals.
pub struct GpuIcpInput {
    pub(super) source: GpuQueryBatch,
    pub(super) target: GpuQueryBatch,
    pub(super) tree: GpuKdTree,
    pub(super) source_anchor: [f32; 4],
    target_anchor: [f32; 4],
    pub(super) scale: f32,
    pub(super) identity: Arc<()>,
    transfers: TransferStats,
}

impl GpuIcpInput {
    /// Uploads source XYZ and a nonempty finite target tree. Payload is 16*S
    /// source bytes, 32*T target bytes (tree order plus original order), and
    /// 32*M tree-node bytes. The combined normalization extent must have a normal
    /// finite f32 square; coincident source/target sets use scale 1.
    /// Derived zero frame anchors use +0 consistently; uploaded original XYZ
    /// bits, including signed zeros, are preserved. Nonzero anchors and scales
    /// retain their f64 preparation arithmetic.
    pub fn upload(
        runtime: &WgpuRuntime,
        source: &[[f32; 3]],
        target: &KdTree,
    ) -> SpatialResult<Self> {
        Self::upload_impl(runtime, source, target, false)
    }

    /// Explicitly gathers target tree points and original-order target queries
    /// together on GPU. Host payload is 16*S + 16*T + 32*M bytes, compared
    /// with 16*S + 32*T + 32*M for `upload`; device output bytes are 32*T.
    /// Source packing, CPU geometry validation/partitioning and node staging
    /// remain. Integer copies preserve input bits, anchors, scale and ordering.
    /// Two transient inputs add 16*T GPU bytes beside owned outputs; host
    /// target point/query packing vectors totaling 32*T payload bytes over
    /// construction are removed. This is not a peak-memory reduction claim.
    /// There is no readback or completion wait. Placement is explicit;
    /// small inputs may be slower and existing device requirements apply.
    pub fn upload_with_gpu_gather(
        runtime: &WgpuRuntime,
        source: &[[f32; 3]],
        target: &KdTree,
    ) -> SpatialResult<Self> {
        Self::upload_impl(runtime, source, target, true)
    }

    fn upload_impl(
        runtime: &WgpuRuntime,
        source: &[[f32; 3]],
        target: &KdTree,
        gpu_gather: bool,
    ) -> SpatialResult<Self> {
        plan(source.len() as u64, &runtime.device().limits())?;
        let layout = target.layout();
        let (x, y, z) = layout.positions();
        plan(x.len() as u64, &runtime.device().limits())?;
        if x.is_empty() {
            return Err(invalid());
        }
        let (source_anchor, source_extent) = geometry(source)?;
        let (target_anchor, target_extent) = geometry_preparation::prepare_soa([x, y, z])?;
        let extent = source_extent.max(target_extent);
        if extent > 0.0 && (extent as f32) < f32::MIN_POSITIVE {
            return Err(invalid());
        }
        let scale = if extent == 0.0 { 1.0 } else { extent.sqrt() as f32 };
        let (tree, gathered_target) = if gpu_gather {
            let (tree, queries) = GpuKdTree::upload_and_queries_with_gpu_gather(runtime, target)?;
            (tree, Some(queries))
        } else {
            (GpuKdTree::upload(runtime, target)?, None)
        };
        let source = GpuQueryBatch::upload(runtime, source)?;
        let target = match gathered_target {
            Some(queries) => queries,
            None => GpuQueryBatch::upload_soa(runtime, [x, y, z])?,
        };
        let mut transfers = tree.upload_transfers();
        transfers.record(
            TransferDirection::HostToDevice,
            source.upload_transfers().host_to_device_bytes(),
        );
        transfers.record(
            TransferDirection::HostToDevice,
            target.upload_transfers().host_to_device_bytes(),
        );
        transfers.record(
            TransferDirection::DeviceToDevice,
            target.upload_transfers().device_to_device_bytes(),
        );
        Ok(Self {
            source,
            target,
            tree,
            source_anchor,
            target_anchor,
            scale,
            identity: runtime.profiling_identity(),
            transfers,
        })
    }

    /// Payload transferred during explicit input upload.
    #[must_use]
    pub const fn upload_transfers(&self) -> TransferStats {
        self.transfers
    }

    /// Original source point count.
    #[must_use]
    pub fn source_count(&self) -> u32 {
        self.source.query_count()
    }

    /// Submits source transformation, bounded nearest search, compensated
    /// centroid/fitness and centered cross-covariance reductions in six passes.
    /// No payload copy/readback or host wait occurs. The fixed-size owned result
    /// can outlive inputs; consumers must check status before using its moments.
    /// GPU f32 transforms/reductions can differ from CPU f64 accumulation near
    /// match gates, degenerate geometry and convergence thresholds. No trimming
    /// or pose solve is performed here. Overflow invalidates ALL result payload.
    pub fn moments_resident(
        &self,
        runtime: &WgpuRuntime,
        transform: Isometry3<f32>,
        max_distance: f32,
    ) -> SpatialResult<GpuIcpMoments> {
        self.moments_impl(
            runtime,
            transform,
            max_distance,
            None,
            trim::GpuIcpTrimPrecision::DoubleSingle,
        )
    }

    /// Submits stable GPU selection of the lowest-distance floor(fraction*N)
    /// gated pairs, breaking distance ties by original source index. Moments
    /// use retained pairs with DoubleSingle normalized arithmetic; fitness
    /// uses ALL gated pairs. Binary64 fraction
    /// multiplication/floor is emulated with integer limbs on the device.
    /// No match/point payload readback or host wait occurs. A fraction of one
    /// delegates to the canonical untrimmed kernels; no CPU fallback occurs.
    pub fn trimmed_moments_resident(
        &self,
        runtime: &WgpuRuntime,
        transform: Isometry3<f32>,
        max_distance: f32,
        fraction: f64,
    ) -> SpatialResult<GpuIcpTrimmedMoments> {
        self.trimmed_moments_resident_with_precision(
            runtime,
            transform,
            max_distance,
            fraction,
            trim::GpuIcpTrimPrecision::DoubleSingle,
        )
    }

    /// Explicit choice of retained-moment arithmetic. DoubleSingle carries
    /// two f32 words through normalized offsets, division and products; f32
    /// transforms/search stay unchanged. CompensatedF32 retains the earlier
    /// coordinate/product precision. Fraction one uses canonical ordinary
    /// moments for either choice. Transfers/ownership/status rules are identical.
    pub fn trimmed_moments_resident_with_precision(
        &self,
        runtime: &WgpuRuntime,
        transform: Isometry3<f32>,
        max_distance: f32,
        fraction: f64,
        precision: trim::GpuIcpTrimPrecision,
    ) -> SpatialResult<GpuIcpTrimmedMoments> {
        if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
            return Err(SpatialError::InvalidArgument(
                "trim_fraction must be finite and in (0, 1]".to_owned(),
            ));
        }
        let trimmed = fraction < 1.0;
        let inner = self.moments_impl(
            runtime,
            transform,
            max_distance,
            trimmed.then_some(fraction),
            precision,
        )?;
        Ok(GpuIcpTrimmedMoments { inner, trimmed })
    }

    fn moments_impl(
        &self,
        runtime: &WgpuRuntime,
        transform: Isometry3<f32>,
        max_distance: f32,
        fraction: Option<f64>,
        precision: trim::GpuIcpTrimPrecision,
    ) -> SpatialResult<GpuIcpMoments> {
        let q = transform.rotation();
        let t = transform.translation();
        let norm = f64::from(q.x).hypot(f64::from(q.y)).hypot(f64::from(q.z)).hypot(f64::from(q.w));
        let bound = max_distance * max_distance;
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity())
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
        let groups = if fraction.is_some() {
            trim::plan_trim(u64::from(count), &runtime.device().limits())?
        } else {
            plan(u64::from(count), &runtime.device().limits())?
        };
        let output = GpuIcpMoments {
            buffer: runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("icp-moments"),
                size: RESULT_BYTES,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            count,
            scale: self.scale,
            bound,
            identity: Arc::clone(&self.identity),
        };
        if count == 0 {
            return Ok(output);
        }
        let r = transform.rotation().to_mat3();
        let params = Params {
            rows: [
                [r.m[0][0], r.m[0][1], r.m[0][2], t.x],
                [r.m[1][0], r.m[1][1], r.m[1][2], t.y],
                [r.m[2][0], r.m[2][1], r.m[2][2], t.z],
            ],
            source_anchor: self.source_anchor,
            target_anchor: self.target_anchor,
            count,
            targets: self.target.query_count(),
            groups,
            unused: 0,
            bound,
            max_distance,
            scale: self.scale,
            padding: 0.0,
        };
        let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("icp-reduction-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut scratch = ScratchBuffers::new(runtime);
        let queries = scratch.acquire("icp-transformed", u64::from(count) * 16);
        let matches = scratch.acquire("icp-matches", u64::from(count) * 8);
        let partials = scratch.acquire("icp-partials", u64::from(groups) * 144);
        let selection = fraction.map(|_| scratch.acquire("icp-trim-selection", 32));
        let cache = if fraction.is_some() {
            runtime
                .icp_trim_pipelines
                .get_or_init(|| trim::TrimPipelines::new(runtime.device()))
                .reduction(runtime.device(), precision)
        } else {
            runtime
                .icp_reduction_pipelines
                .get_or_init(|| IcpReductionPipelines::new(runtime.device()))
        };
        let mut buffers = vec![
            &uniform,
            self.source.buffer().unwrap(),
            self.target.buffer().unwrap(),
            &queries,
            &matches,
            &partials,
            &output.buffer,
        ];
        if let Some(selection) = &selection {
            buffers.push(selection);
        }
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("icp-reduction-bindings"),
            layout: &cache.layout,
            entries: &entries,
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        dispatch(&mut encoder, &cache.pipelines[0], &group, count.div_ceil(64), 1);
        self.tree.encode_nearest(runtime, &mut encoder, &queries, count, bound, &matches)?;
        if let Some(fraction) = fraction {
            trim::encode_selection(
                runtime,
                &mut encoder,
                &mut scratch,
                &params,
                fraction,
                &matches,
                selection.as_ref().unwrap(),
            );
        }
        for (index, x, y) in [(1, groups, 7), (2, 1, 1), (3, groups, 9), (4, 1, 1)] {
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
        label: Some("icp-reduction"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(x, y, 1);
}

/// Decoded device ICP moments in physical units, with explicit transfer receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuIcpStatistics {
    /// Pairs contributing to centroids/covariance; ordinary moments use every
    /// gated match. Zero yields zero centroids/covariance.
    pub correspondences: u32,
    /// Arithmetic source centroid after applying the requested transform.
    pub source_centroid: [f64; 3],
    /// Corresponding target centroid.
    pub target_centroid: [f64; 3],
    /// Unnormalized centered sum of source[i] * target[j], row-major.
    pub cross_covariance: [[f64; 3]; 3],
    /// Sum of squared matched residual lengths (not a mean). In the containing
    /// trimmed statistics, use its all-gated count as the fitness denominator.
    pub squared_error_sum: f64,
    /// Payload copied by this explicit download, independent of input upload.
    pub transfers: TransferStats,
}

/// Owned fixed-size device ICP moments. No buffer is pooled or input-borrowed.
pub struct GpuIcpMoments {
    buffer: wgpu::Buffer,
    count: u32,
    scale: f32,
    bound: f32,
    identity: Arc<()>,
}
impl GpuIcpMoments {
    /// Physical coordinate scale for the normalized centroid/covariance words.
    #[must_use]
    pub const fn normalization_scale(&self) -> f32 {
        self.scale
    }

    /// Physical squared-distance scale for normalized residual sum word 15.
    #[must_use]
    pub const fn squared_distance_scale(&self) -> f32 {
        self.bound
    }

    /// 176-byte storage: u32 status/count/two reserved zeros, two vec4<f32>
    /// anchors, then 16 vec2<f32> compensated sums. Status bits 1/2/4 denote
    /// transform overflow, malformed matches, nonfinite moments. Status must be
    /// zero. Terms 0..5 are normalized source/target centroid offsets; 6..14
    /// normalized centered cross-covariance; 15 normalized squared-error sum.
    /// Scale is the input bounding-box diagonal; fitness scale is distance^2.
    /// Upload geometry canonicalizes derived zero anchors to +0. Earlier
    /// builds could expose either signed zero from CPU min/max preparation;
    /// the arithmetic value is unchanged. Transformed anchors follow the
    /// requested GPU pose arithmetic.
    /// Invalid or zero-match results have all-zero payload (status retained).
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Explicit readback size; zero for an empty source, otherwise 176 bytes.
    #[must_use]
    pub const fn readback_bytes(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            RESULT_BYTES
        }
    }

    /// Copies the fixed-size moments/status, waits, then validates and decodes
    /// compensated f32 sums in f64. No point/query/match payload is downloaded.
    pub fn download(&self, runtime: &WgpuRuntime) -> SpatialResult<GpuIcpStatistics> {
        self.download_impl(runtime, false).map(|(statistics, _)| statistics)
    }

    fn download_impl(
        &self,
        runtime: &WgpuRuntime,
        trimmed: bool,
    ) -> SpatialResult<(GpuIcpStatistics, u32)> {
        if !Arc::ptr_eq(&self.identity, &runtime.profiling_identity()) {
            return Err(invalid());
        }
        let mut transfers = TransferStats::default();
        let mut words = vec![0u32; RESULT_BYTES as usize / 4];
        if self.count > 0 {
            let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("icp-moments-readback"),
                size: RESULT_BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = runtime.device().create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, RESULT_BYTES);
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
            words.copy_from_slice(bytemuck::cast_slice(&mapped));
            drop(mapped);
            staging.unmap();
            transfers.record(TransferDirection::DeviceToHost, RESULT_BYTES);
        }
        let evaluated = if trimmed { words[2] } else { words[1] };
        if words[0] != 0
            || words[1] > evaluated
            || evaluated > self.count
            || words[3] != 0
            || (!trimmed && words[2] != 0)
            || (evaluated == 0 && words[4..].iter().any(|&v| v != 0))
            || (words[1] == 0 && words[4..42].iter().any(|&v| v != 0))
        {
            return Err(SpatialError::InvalidArgument(format!(
                "GPU ICP reduction invalid status/count: {:?}",
                &words[..4]
            )));
        }
        let values: Vec<_> = words[4..].iter().map(|&v| f64::from(f32::from_bits(v))).collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid());
        }
        let sum = |term: usize| values[8 + term * 2] + values[9 + term * 2];
        let scale = f64::from(self.scale);
        let source_centroid =
            std::array::from_fn(|i| if words[1] == 0 { 0.0 } else { values[i] + sum(i) * scale });
        let target_centroid = std::array::from_fn(|i| {
            if words[1] == 0 {
                0.0
            } else {
                values[4 + i] + sum(3 + i) * scale
            }
        });
        let cross_covariance =
            std::array::from_fn(|i| std::array::from_fn(|j| sum(6 + i * 3 + j) * scale * scale));
        let squared_error_sum = sum(15) * f64::from(self.bound);
        if squared_error_sum < 0.0 {
            return Err(invalid());
        }
        Ok((
            GpuIcpStatistics {
                correspondences: words[1],
                source_centroid,
                target_centroid,
                cross_covariance,
                squared_error_sum,
                transfers,
            },
            evaluated,
        ))
    }
}

/// Fixed-size selected moments plus the all-gated fitness denominator.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuIcpTrimmedStatistics {
    /// Retained-pair centroids/covariance, ALL-gated residual sum, and readback receipt.
    pub moments: GpuIcpStatistics,
    /// Number of all gated matches used for fitness (before trimming).
    pub evaluated_correspondences: u32,
}

/// Owned 176-byte trimmed moments/status; no point or match readback is implicit.
pub struct GpuIcpTrimmedMoments {
    inner: GpuIcpMoments,
    trimmed: bool,
}

impl GpuIcpTrimmedMoments {
    /// Device storage layout matches ordinary moments, except word 2 is the
    /// all-gated count. Word 1 is retained count. Terms 0..14 use retained pairs;
    /// term 15 uses every gated pair. Zero retained count zeros terms 0..14 and
    /// anchors but preserves all-gated fitness. Any invalid status zeros payload.
    /// Fraction one retains ordinary moments' two reserved zeros.
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        self.inner.buffer()
    }

    /// Explicit fixed readback size, zero for empty source and 176 otherwise.
    #[must_use]
    pub const fn readback_bytes(&self) -> u64 {
        self.inner.readback_bytes()
    }

    /// Physical coordinate scale for normalized centroid/covariance words.
    #[must_use]
    pub const fn normalization_scale(&self) -> f32 {
        self.inner.normalization_scale()
    }

    /// Physical squared-distance scale for normalized all-gated residual sum.
    #[must_use]
    pub const fn squared_distance_scale(&self) -> f32 {
        self.inner.squared_distance_scale()
    }

    /// Explicitly copies/waits/validates only the fixed-size moments and counts.
    pub fn download(&self, runtime: &WgpuRuntime) -> SpatialResult<GpuIcpTrimmedStatistics> {
        let (moments, evaluated_correspondences) =
            self.inner.download_impl(runtime, self.trimmed)?;
        Ok(GpuIcpTrimmedStatistics { moments, evaluated_correspondences })
    }
}

pub(crate) struct IcpReductionPipelines {
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 5],
}
impl IcpReductionPipelines {
    fn new(device: &wgpu::Device) -> Self {
        Self::with_shader(device, include_str!("../shaders/icp_reduction.wgsl"), false)
    }

    fn with_shader(device: &wgpu::Device, shader_source: &str, trimmed: bool) -> Self {
        let entries: Vec<_> = (0..if trimmed { 8 } else { 7 })
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: matches!(binding, 1 | 2 | 4 | 7),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("icp-reduction-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("icp-reduction-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("icp-reduction-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let pipelines = [
            "transform_source",
            "centroid_partials",
            "finish_centroids",
            "covariance_partials",
            "finish_covariance",
        ]
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("icp-reduction-pipeline"),
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
#[path = "icp_reduction_tests.rs"]
mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "icp_reduction_gather_reference.rs"]
pub(crate) mod gather_reference;
