use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use spatialrust_core::{
    DType, DeviceKind, ExecutionPolicy, FieldSemantic, HasPositions3, PointBuffer, PointBufferSet,
    PointCloud, PointField, PointSchema, SpatialError, SpatialResult,
};
use spatialrust_math::Vec3;

use crate::filter::PointCloudFilter;

/// Voxel keys are small integer tuples, so a fast multiply-rotate hasher (à la
/// FxHash) beats the default SipHash by a wide margin on the cell map.
#[derive(Default)]
struct VoxelKeyHasher {
    hash: u64,
}

impl VoxelKeyHasher {
    #[inline]
    fn mix(&mut self, value: u64) {
        const K: u64 = 0x517c_c1b7_2722_0a95;
        self.hash = (self.hash.rotate_left(5) ^ value).wrapping_mul(K);
    }
}

impl Hasher for VoxelKeyHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.mix(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.mix(i);
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.mix(u64::from(b));
        }
    }
}

/// Cell map keyed by integer voxel coordinates, using the fast voxel hasher.
type VoxelCellMap = HashMap<(i64, i64, i64), VoxelCell, BuildHasherDefault<VoxelKeyHasher>>;

/// Voxel aggregation strategy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VoxelAggregationMode {
    /// Average all points in each voxel (centroid).
    #[default]
    Centroid,
    /// Keep the first point that falls into each voxel.
    ApproximateFirst,
}

/// Attribute aggregation policy for non-position fields.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttributeAggregation {
    /// Average numeric attributes within a voxel.
    #[default]
    Average,
    /// Keep the first point's attribute values.
    First,
}

/// Default minimum point count before GPU voxel downsampling is selected (centroid).
///
/// Local end-to-end filter benches show GPU centroid wins above ~500k points.
pub const DEFAULT_GPU_MIN_POINTS: usize = 500_000;

/// Default minimum point count before GPU approximate-first downsampling is selected.
///
/// Approximate-first pays a higher gather/readback cost than centroid. End-to-end
/// benches through 1M still favor CPU (~45 ms vs ~50 ms at 1M); auto-GPU is deferred
/// until a crossover is measured above that range.
pub const DEFAULT_GPU_MIN_POINTS_APPROXIMATE: usize = 2_000_000;

/// Non-position F32 attribute count at/above which approximate-first Auto uses a higher GPU threshold.
///
/// Epic 38: `point_xyzinormal` approximate-first GPU lost at all measured scales.
/// Epic 46: upload pool + zero-copy attrs restored GPU crossover at 1M+.
pub const APPROXIMATE_HEAVY_F32_ATTRIBUTE_CHANNELS: usize = 4;

/// Auto GPU threshold for approximate-first on attribute-heavy schemas (e.g. xyzinormal).
pub const DEFAULT_GPU_MIN_POINTS_APPROXIMATE_HEAVY: usize = 1_000_000;

/// Configuration for voxel-grid downsampling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelGridDownsampleConfig {
    /// Voxel edge length in meters.
    pub leaf_size: f32,
    /// Optional grid origin. Defaults to the cloud minimum corner.
    pub origin: Option<Vec3<f32>>,
    /// Position aggregation mode.
    pub mode: VoxelAggregationMode,
    /// Aggregation policy for other fields.
    pub attribute_policy: AttributeAggregation,
    /// Minimum input point count before GPU execution is considered worthwhile.
    ///
    /// `None` always uses GPU when requested. Defaults follow local bench results:
    /// centroid ~500k, approximate-first ~2M (1M end-to-end still CPU-favored).
    ///
    /// Approximate-first Auto also consults the input schema: clouds with
    /// [`APPROXIMATE_HEAVY_F32_ATTRIBUTE_CHANNELS`] or more non-position F32 fields
    /// (e.g. `point_xyzinormal`) use [`DEFAULT_GPU_MIN_POINTS_APPROXIMATE_HEAVY`].
    pub gpu_min_points: Option<usize>,
}

impl VoxelGridDownsampleConfig {
    /// Creates a centroid downsampling config with uniform leaf size.
    #[must_use]
    pub fn centroid(leaf_size: f32) -> Self {
        Self {
            leaf_size,
            origin: None,
            mode: VoxelAggregationMode::Centroid,
            attribute_policy: AttributeAggregation::Average,
            gpu_min_points: Some(DEFAULT_GPU_MIN_POINTS),
        }
    }

    /// Creates an approximate first-point downsampling config.
    #[must_use]
    pub fn approximate(leaf_size: f32) -> Self {
        Self {
            leaf_size,
            origin: None,
            mode: VoxelAggregationMode::ApproximateFirst,
            attribute_policy: AttributeAggregation::First,
            gpu_min_points: Some(DEFAULT_GPU_MIN_POINTS_APPROXIMATE),
        }
    }

    /// Disables the GPU point-count heuristic so GPU is always used when requested.
    #[must_use]
    pub const fn without_gpu_min_points(mut self) -> Self {
        self.gpu_min_points = None;
        self
    }

    /// Returns the point-count threshold used by [`ExecutionPolicy::Auto`].
    ///
    /// Approximate-first mode raises the effective threshold to
    /// [`DEFAULT_GPU_MIN_POINTS_APPROXIMATE_HEAVY`] when the schema carries many F32
    /// attributes (Epic 38 regression, Epic 46 crossover at 1M+).
    #[must_use]
    pub fn effective_gpu_min_points(&self, schema: &PointSchema) -> Option<usize> {
        let base = self.gpu_min_points?;
        if self.mode != VoxelAggregationMode::ApproximateFirst {
            return Some(base);
        }
        if count_non_position_f32_fields(schema) >= APPROXIMATE_HEAVY_F32_ATTRIBUTE_CHANNELS {
            return Some(DEFAULT_GPU_MIN_POINTS_APPROXIMATE_HEAVY);
        }
        Some(base)
    }
}

/// Voxel-grid downsampling filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelGridDownsample {
    config: VoxelGridDownsampleConfig,
}

impl VoxelGridDownsample {
    /// Creates a filter from config.
    #[must_use]
    pub const fn new(config: VoxelGridDownsampleConfig) -> Self {
        Self { config }
    }

    /// Returns the filter config.
    #[must_use]
    pub const fn config(&self) -> VoxelGridDownsampleConfig {
        self.config
    }

    /// Applies the filter using the requested execution policy.
    ///
    /// GPU execution assigns voxel keys on wgpu, builds segments with GPU sorting,
    /// and performs centroid or approximate-first aggregation on wgpu.
    ///
    /// [`ExecutionPolicy::Auto`] picks GPU only when the input meets
    /// [`VoxelGridDownsampleConfig::gpu_min_points`]. Explicit GPU requests below the
    /// threshold fall back to CPU to avoid upload/readback overhead on small clouds.
    pub fn filter_with_policy(
        &self,
        input: &PointCloud,
        policy: ExecutionPolicy,
    ) -> SpatialResult<PointCloud> {
        self.filter_internal(input, policy)
    }
}

impl PointCloudFilter for VoxelGridDownsample {
    fn name(&self) -> &'static str {
        "VoxelGridDownsample"
    }

    fn filter(&self, input: &PointCloud) -> SpatialResult<PointCloud> {
        self.filter_internal(input, ExecutionPolicy::CpuSingle)
    }
}

impl VoxelGridDownsample {
    fn filter_internal(
        &self,
        input: &PointCloud,
        policy: ExecutionPolicy,
    ) -> SpatialResult<PointCloud> {
        if input.is_empty() {
            return Ok(input.clone());
        }
        if self.config.leaf_size <= 0.0 {
            return Err(SpatialError::InvalidArgument(
                "leaf_size must be greater than zero".to_owned(),
            ));
        }

        let (x, y, z) = input.positions3()?;
        let origin = self.config.origin.unwrap_or_else(|| compute_min_corner(x, y, z));
        let inv_leaf = 1.0 / self.config.leaf_size;
        let policy = self.resolve_policy(input, policy);

        if matches!(policy, ExecutionPolicy::Gpu(DeviceKind::Wgpu)) {
            #[cfg(feature = "filter-voxel-gpu")]
            {
                return match self.config.mode {
                    VoxelAggregationMode::Centroid => filter_gpu_centroid(
                        input,
                        x,
                        y,
                        z,
                        origin,
                        inv_leaf,
                        self.config.attribute_policy,
                    ),
                    VoxelAggregationMode::ApproximateFirst => filter_gpu_approximate_first(
                        input,
                        x,
                        y,
                        z,
                        origin,
                        inv_leaf,
                        self.config.attribute_policy,
                    ),
                };
            }
            #[cfg(not(feature = "filter-voxel-gpu"))]
            {
                return Err(SpatialError::InvalidArgument(
                    "GPU voxel downsampling requires the filter-voxel-gpu feature".to_owned(),
                ));
            }
        }

        // Fast path: the common centroid + average case (the default config and
        // what PCL's VoxelGrid does) is a single pass that resolves field
        // buffers once and accumulates per-cell sums into flat arrays, avoiding a
        // per-cell index Vec and a string-keyed field lookup per point.
        if matches!(
            policy,
            ExecutionPolicy::Auto | ExecutionPolicy::CpuSingle | ExecutionPolicy::CpuParallel
        ) && self.config.mode == VoxelAggregationMode::Centroid
            && self.config.attribute_policy == AttributeAggregation::Average
        {
            return filter_cpu_centroid_fast(input, x, y, z, origin, inv_leaf);
        }

        let cells = match policy {
            ExecutionPolicy::Gpu(DeviceKind::Wgpu) => {
                build_voxel_cells_gpu(x, y, z, origin, inv_leaf)?
            }
            ExecutionPolicy::Gpu(_) => {
                return Err(SpatialError::InvalidArgument(
                    "unsupported GPU device kind for voxel downsampling".to_owned(),
                ));
            }
            ExecutionPolicy::Auto | ExecutionPolicy::CpuSingle | ExecutionPolicy::CpuParallel => {
                build_voxel_cells_cpu(x, y, z, origin, inv_leaf)
            }
        };

        let schema = input.schema().clone();
        let mut buffers = PointBufferSet::new();
        for field in schema.fields() {
            buffers
                .insert(field.name.clone(), PointBuffer::with_capacity(field.dtype, cells.len()));
        }

        let mut ordered_cells: Vec<_> = cells.into_iter().collect();
        ordered_cells.sort_by_key(|(key, _)| *key);

        for (_, cell) in ordered_cells {
            append_voxel_point(
                input,
                &mut buffers,
                schema.fields(),
                &cell,
                self.config.mode,
                self.config.attribute_policy,
            )?;
        }

        PointCloud::try_from_parts(schema, buffers, input.metadata().clone())
    }

    fn resolve_policy(&self, input: &PointCloud, policy: ExecutionPolicy) -> ExecutionPolicy {
        match policy {
            ExecutionPolicy::Auto => {
                if self.should_use_gpu(input) {
                    ExecutionPolicy::Gpu(DeviceKind::Wgpu)
                } else {
                    ExecutionPolicy::CpuSingle
                }
            }
            ExecutionPolicy::Gpu(DeviceKind::Wgpu) if !self.should_use_gpu(input) => {
                ExecutionPolicy::CpuSingle
            }
            other => other,
        }
    }

    fn should_use_gpu(&self, input: &PointCloud) -> bool {
        #[cfg(feature = "filter-voxel-gpu")]
        {
            match self.config.effective_gpu_min_points(input.schema()) {
                Some(min_points) => input.len() >= min_points,
                None => true,
            }
        }
        #[cfg(not(feature = "filter-voxel-gpu"))]
        {
            let _ = input;
            false
        }
    }
}

fn count_non_position_f32_fields(schema: &PointSchema) -> usize {
    schema
        .fields()
        .iter()
        .filter(|field| {
            !matches!(
                field.semantic,
                FieldSemantic::PositionX | FieldSemantic::PositionY | FieldSemantic::PositionZ
            ) && matches!(field.dtype, DType::F32 | DType::F16)
        })
        .count()
}

#[derive(Clone, Debug, Default)]
struct VoxelCell {
    indices: Vec<usize>,
}

/// Single-pass centroid voxel downsampling for the default (Centroid + Average)
/// case. Resolves every field's buffer once, then accumulates per-cell sums into
/// flat arrays keyed by a sequential cell id, so there is no per-cell allocation
/// and no per-point field lookup.
fn filter_cpu_centroid_fast(
    input: &PointCloud,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: Vec3<f32>,
    inv_leaf: f32,
) -> SpatialResult<PointCloud> {
    let schema = input.schema().clone();
    let fields = schema.fields();
    let n_fields = fields.len();

    // Resolve each field's backing buffer once.
    let field_buffers: Vec<&PointBuffer> =
        fields.iter().map(|f| input.field(&f.name)).collect::<SpatialResult<_>>()?;

    let mut key_to_id: HashMap<(i64, i64, i64), u32, BuildHasherDefault<VoxelKeyHasher>> =
        HashMap::default();
    let mut keys: Vec<(i64, i64, i64)> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    // Flat `cell * n_fields + field` accumulator of f64 sums.
    let mut sums: Vec<f64> = Vec::new();

    for i in 0..x.len() {
        let key = voxel_key(x[i], y[i], z[i], origin, inv_leaf);
        let id = *key_to_id.entry(key).or_insert_with(|| {
            let id = counts.len() as u32;
            counts.push(0);
            keys.push(key);
            sums.extend(std::iter::repeat(0.0).take(n_fields));
            id
        }) as usize;
        counts[id] += 1;
        let base = id * n_fields;
        for (fi, buffer) in field_buffers.iter().enumerate() {
            sums[base + fi] += f64::from(read_buffer_f32(buffer, i));
        }
    }

    // Deterministic output: emit cells in voxel-key order.
    let mut order: Vec<u32> = (0..counts.len() as u32).collect();
    order.sort_by_key(|&id| keys[id as usize]);

    let mut buffers = PointBufferSet::new();
    for field in fields {
        buffers.insert(field.name.clone(), PointBuffer::with_capacity(field.dtype, counts.len()));
    }
    for &id in &order {
        let id = id as usize;
        let inv_count = 1.0 / f64::from(counts[id]);
        let base = id * n_fields;
        for (fi, field) in fields.iter().enumerate() {
            push_field(&mut buffers, field, (sums[base + fi] * inv_count) as f32)?;
        }
    }

    PointCloud::try_from_parts(schema, buffers, input.metadata().clone())
}

/// Reads any numeric buffer column as `f32` by index.
fn read_buffer_f32(buffer: &PointBuffer, index: usize) -> f32 {
    match buffer {
        PointBuffer::F32(v) => v[index],
        PointBuffer::F64(v) => v[index] as f32,
        PointBuffer::U8(v) => f32::from(v[index]),
        PointBuffer::U16(v) => f32::from(v[index]),
        PointBuffer::U32(v) => v[index] as f32,
        PointBuffer::I32(v) => v[index] as f32,
    }
}

fn build_voxel_cells_cpu(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: Vec3<f32>,
    inv_leaf: f32,
) -> VoxelCellMap {
    let mut cells = VoxelCellMap::default();
    for index in 0..x.len() {
        let key = voxel_key(x[index], y[index], z[index], origin, inv_leaf);
        cells.entry(key).or_default().indices.push(index);
    }
    cells
}

#[cfg(feature = "filter-voxel-gpu")]
fn build_voxel_cells_gpu(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: Vec3<f32>,
    inv_leaf: f32,
) -> SpatialResult<VoxelCellMap> {
    use spatialrust_gpu::{compute_voxel_keys, WgpuRuntime};

    let runtime = WgpuRuntime::shared()?;
    let keys = compute_voxel_keys(&runtime, x, y, z, [origin.x, origin.y, origin.z], inv_leaf)?;

    let mut cells = VoxelCellMap::default();
    for (index, key) in keys.into_iter().enumerate() {
        cells.entry(key).or_default().indices.push(index);
    }
    Ok(cells)
}

#[cfg(feature = "filter-voxel-gpu")]
fn gpu_non_position_fields(schema: &PointSchema) -> Vec<PointField> {
    schema
        .fields()
        .iter()
        .filter(|field| {
            !matches!(
                field.semantic,
                FieldSemantic::PositionX | FieldSemantic::PositionY | FieldSemantic::PositionZ
            )
        })
        .cloned()
        .collect()
}

#[cfg(feature = "filter-voxel-gpu")]
fn partition_gpu_attribute_fields(fields: &[PointField]) -> (Vec<PointField>, Vec<PointField>) {
    let mut f32_fields = Vec::new();
    let mut u8_fields = Vec::new();
    for field in fields {
        if field.dtype == DType::U8 {
            u8_fields.push(field.clone());
        } else {
            f32_fields.push(field.clone());
        }
    }
    (f32_fields, u8_fields)
}

#[cfg(feature = "filter-voxel-gpu")]
fn collect_attribute_f32_sources(
    input: &PointCloud,
    fields: &[PointField],
) -> SpatialResult<Vec<Vec<f32>>> {
    let mut sources = Vec::with_capacity(fields.len());
    for field in fields {
        let mut values = Vec::with_capacity(input.len());
        for index in 0..input.len() {
            values.push(read_field_f32(input, field, index)?);
        }
        sources.push(values);
    }
    Ok(sources)
}

#[cfg(feature = "filter-voxel-gpu")]
fn borrow_attribute_f32_channels<'a>(
    input: &'a PointCloud,
    fields: &[PointField],
) -> SpatialResult<Option<Vec<&'a [f32]>>> {
    let mut channels = Vec::with_capacity(fields.len());
    for field in fields {
        if !matches!(field.dtype, DType::F32 | DType::F16) {
            return Ok(None);
        }
        channels.push(input.field(&field.name)?.as_f32()?);
    }
    Ok(Some(channels))
}

#[cfg(feature = "filter-voxel-gpu")]
fn collect_attribute_u8_sources(
    input: &PointCloud,
    fields: &[PointField],
) -> SpatialResult<Vec<Vec<u8>>> {
    let mut sources = Vec::with_capacity(fields.len());
    for field in fields {
        let buffer = input.field(&field.name)?;
        let PointBuffer::U8(values) = buffer else {
            return Err(SpatialError::UnsupportedDType(field.dtype));
        };
        sources.push(values.to_vec());
    }
    Ok(sources)
}

#[cfg(feature = "filter-voxel-gpu")]
fn assemble_gpu_voxel_output(
    input: &PointCloud,
    out_x: Vec<f32>,
    out_y: Vec<f32>,
    out_z: Vec<f32>,
    f32_attribute_fields: &[PointField],
    f32_attribute_values: Vec<Vec<f32>>,
    u8_attribute_fields: &[PointField],
    u8_attribute_values: Vec<Vec<u8>>,
) -> SpatialResult<PointCloud> {
    let schema = input.schema().clone();
    let mut buffers = PointBufferSet::new();

    let x_field = schema
        .find_semantic(FieldSemantic::PositionX)
        .ok_or_else(|| SpatialError::MissingField("x".to_owned()))?;
    let y_field = schema
        .find_semantic(FieldSemantic::PositionY)
        .ok_or_else(|| SpatialError::MissingField("y".to_owned()))?;
    let z_field = schema
        .find_semantic(FieldSemantic::PositionZ)
        .ok_or_else(|| SpatialError::MissingField("z".to_owned()))?;

    set_field_from_f32(&mut buffers, x_field, out_x)?;
    set_field_from_f32(&mut buffers, y_field, out_y)?;
    set_field_from_f32(&mut buffers, z_field, out_z)?;

    for (field, values) in f32_attribute_fields.iter().zip(f32_attribute_values) {
        set_field_from_f32(&mut buffers, field, values)?;
    }
    for (field, values) in u8_attribute_fields.iter().zip(u8_attribute_values) {
        set_field_from_u8(&mut buffers, field, values)?;
    }

    PointCloud::try_from_parts(schema, buffers, input.metadata().clone())
}

#[cfg(feature = "filter-voxel-gpu")]
fn filter_gpu_centroid(
    input: &PointCloud,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: Vec3<f32>,
    inv_leaf: f32,
    attribute_policy: AttributeAggregation,
) -> SpatialResult<PointCloud> {
    use spatialrust_gpu::{
        build_voxel_segments_gpu_from_keys_buffer, compute_voxel_keys_gpu_buffers,
        downsample_voxel_centroid_gpu, reduce_voxel_centroids_xyz_and_average_multi_gpu,
        reduce_voxel_centroids_xyz_and_gather_first_multi_gpu, WgpuRuntime,
    };

    let attribute_fields = gpu_non_position_fields(input.schema());
    if attribute_fields.is_empty() {
        let runtime = WgpuRuntime::shared()?;
        let pipeline = downsample_voxel_centroid_gpu(
            &runtime,
            x,
            y,
            z,
            [origin.x, origin.y, origin.z],
            inv_leaf,
        )?;
        return assemble_gpu_voxel_output(
            input,
            pipeline.out_x,
            pipeline.out_y,
            pipeline.out_z,
            &[],
            Vec::new(),
            &[],
            Vec::new(),
        );
    }

    let (f32_fields, u8_fields) = partition_gpu_attribute_fields(&attribute_fields);
    let runtime = WgpuRuntime::shared()?;
    let positions = compute_voxel_keys_gpu_buffers(
        &runtime,
        x,
        y,
        z,
        [origin.x, origin.y, origin.z],
        inv_leaf,
    )?;
    let point_count = positions.point_count();
    let segments = build_voxel_segments_gpu_from_keys_buffer(
        &runtime,
        positions.keys_buffer(),
        point_count,
        point_count.next_power_of_two(),
    )?;
    let owned_f32_sources;
    let f32_refs: Vec<&[f32]> =
        if let Some(borrowed) = borrow_attribute_f32_channels(input, &f32_fields)? {
            borrowed
        } else {
            owned_f32_sources = collect_attribute_f32_sources(input, &f32_fields)?;
            owned_f32_sources.iter().map(Vec::as_slice).collect()
        };
    let u8_sources = collect_attribute_u8_sources(input, &u8_fields)?;
    let u8_refs: Vec<&[u8]> = u8_sources.iter().map(Vec::as_slice).collect();
    let (out_x, out_y, out_z, f32_values, u8_values) = match attribute_policy {
        AttributeAggregation::Average => reduce_voxel_centroids_xyz_and_average_multi_gpu(
            &runtime,
            positions.x_buffer(),
            positions.y_buffer(),
            positions.z_buffer(),
            &f32_refs,
            &u8_refs,
            &segments,
        )?,
        AttributeAggregation::First => reduce_voxel_centroids_xyz_and_gather_first_multi_gpu(
            &runtime,
            positions.x_buffer(),
            positions.y_buffer(),
            positions.z_buffer(),
            &f32_refs,
            &u8_refs,
            &segments,
        )?,
    };

    positions.recycle(&runtime);

    assemble_gpu_voxel_output(
        input,
        out_x,
        out_y,
        out_z,
        &f32_fields,
        f32_values,
        &u8_fields,
        u8_values,
    )
}

#[cfg(feature = "filter-voxel-gpu")]
fn filter_gpu_approximate_first(
    input: &PointCloud,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: Vec3<f32>,
    inv_leaf: f32,
    attribute_policy: AttributeAggregation,
) -> SpatialResult<PointCloud> {
    use spatialrust_gpu::{
        build_voxel_segments_gpu_from_keys_buffer, compute_voxel_keys_gpu_buffers,
        downsample_voxel_approximate_first_gpu, gather_voxel_first_xyz_and_average_multi_gpu,
        gather_voxel_first_xyz_and_multi_gpu, WgpuRuntime,
    };

    let attribute_fields = gpu_non_position_fields(input.schema());
    if attribute_fields.is_empty() {
        let runtime = WgpuRuntime::shared()?;
        let pipeline = downsample_voxel_approximate_first_gpu(
            &runtime,
            x,
            y,
            z,
            [origin.x, origin.y, origin.z],
            inv_leaf,
        )?;
        return assemble_gpu_voxel_output(
            input,
            pipeline.out_x,
            pipeline.out_y,
            pipeline.out_z,
            &[],
            Vec::new(),
            &[],
            Vec::new(),
        );
    }

    let (f32_fields, u8_fields) = partition_gpu_attribute_fields(&attribute_fields);
    let runtime = WgpuRuntime::shared()?;
    let positions = compute_voxel_keys_gpu_buffers(
        &runtime,
        x,
        y,
        z,
        [origin.x, origin.y, origin.z],
        inv_leaf,
    )?;
    let point_count = positions.point_count();
    let segments = build_voxel_segments_gpu_from_keys_buffer(
        &runtime,
        positions.keys_buffer(),
        point_count,
        point_count.next_power_of_two(),
    )?;
    let owned_f32_sources;
    let f32_refs: Vec<&[f32]> =
        if let Some(borrowed) = borrow_attribute_f32_channels(input, &f32_fields)? {
            borrowed
        } else {
            owned_f32_sources = collect_attribute_f32_sources(input, &f32_fields)?;
            owned_f32_sources.iter().map(Vec::as_slice).collect()
        };
    let u8_sources = collect_attribute_u8_sources(input, &u8_fields)?;
    let u8_refs: Vec<&[u8]> = u8_sources.iter().map(Vec::as_slice).collect();
    let (out_x, out_y, out_z, f32_values, u8_values) = match attribute_policy {
        AttributeAggregation::Average => gather_voxel_first_xyz_and_average_multi_gpu(
            &runtime,
            positions.x_buffer(),
            positions.y_buffer(),
            positions.z_buffer(),
            &f32_refs,
            &u8_refs,
            &segments,
        )?,
        AttributeAggregation::First => gather_voxel_first_xyz_and_multi_gpu(
            &runtime,
            positions.x_buffer(),
            positions.y_buffer(),
            positions.z_buffer(),
            &f32_refs,
            &u8_refs,
            &segments,
        )?,
    };

    positions.recycle(&runtime);

    assemble_gpu_voxel_output(
        input,
        out_x,
        out_y,
        out_z,
        &f32_fields,
        f32_values,
        &u8_fields,
        u8_values,
    )
}

#[cfg(not(feature = "filter-voxel-gpu"))]
fn build_voxel_cells_gpu(
    _x: &[f32],
    _y: &[f32],
    _z: &[f32],
    _origin: Vec3<f32>,
    _inv_leaf: f32,
) -> SpatialResult<VoxelCellMap> {
    Err(SpatialError::InvalidArgument(
        "GPU voxel downsampling requires the filter-voxel-gpu feature".to_owned(),
    ))
}

fn compute_min_corner(x: &[f32], y: &[f32], z: &[f32]) -> Vec3<f32> {
    let mut min = Vec3::new(x[0], y[0], z[0]);
    for index in 1..x.len() {
        min.x = min.x.min(x[index]);
        min.y = min.y.min(y[index]);
        min.z = min.z.min(z[index]);
    }
    min
}

fn voxel_key(x: f32, y: f32, z: f32, origin: Vec3<f32>, inv_leaf: f32) -> (i64, i64, i64) {
    let ix = ((x - origin.x) * inv_leaf).floor() as i64;
    let iy = ((y - origin.y) * inv_leaf).floor() as i64;
    let iz = ((z - origin.z) * inv_leaf).floor() as i64;
    (ix, iy, iz)
}

fn append_voxel_point(
    input: &PointCloud,
    buffers: &mut PointBufferSet,
    fields: &[PointField],
    cell: &VoxelCell,
    mode: VoxelAggregationMode,
    attribute_policy: AttributeAggregation,
) -> SpatialResult<()> {
    let representative = cell.indices[0];
    let average_positions = mode == VoxelAggregationMode::Centroid;

    for field in fields {
        let value = match field.semantic {
            FieldSemantic::PositionX | FieldSemantic::PositionY | FieldSemantic::PositionZ => {
                if average_positions {
                    average_field(input, field, &cell.indices)?
                } else {
                    read_field_f32(input, field, representative)?
                }
            }
            _ => match (mode, attribute_policy) {
                (VoxelAggregationMode::ApproximateFirst, _) => {
                    read_field_f32(input, field, representative)?
                }
                (_, AttributeAggregation::First) => read_field_f32(input, field, representative)?,
                (_, AttributeAggregation::Average) => average_field(input, field, &cell.indices)?,
            },
        };
        push_field(buffers, field, value)?;
    }
    Ok(())
}

fn average_field(input: &PointCloud, field: &PointField, indices: &[usize]) -> SpatialResult<f32> {
    if indices.is_empty() {
        return Err(SpatialError::InvalidArgument("cannot average an empty voxel cell".to_owned()));
    }
    let mut sum = 0.0_f64;
    for &index in indices {
        sum += f64::from(read_field_f32(input, field, index)?);
    }
    Ok((sum / indices.len() as f64) as f32)
}

fn read_field_f32(input: &PointCloud, field: &PointField, index: usize) -> SpatialResult<f32> {
    let buffer = input.field(&field.name)?;
    match field.dtype {
        DType::F32 | DType::F16 => Ok(buffer.as_f32()?[index]),
        DType::F64 => {
            let PointBuffer::F64(values) = buffer else {
                return Err(SpatialError::UnsupportedDType(field.dtype));
            };
            Ok(values[index] as f32)
        }
        DType::U8 => {
            let PointBuffer::U8(values) = buffer else {
                return Err(SpatialError::UnsupportedDType(field.dtype));
            };
            Ok(f32::from(values[index]))
        }
        DType::U16 => {
            let PointBuffer::U16(values) = buffer else {
                return Err(SpatialError::UnsupportedDType(field.dtype));
            };
            Ok(f32::from(values[index]))
        }
        DType::I32 => {
            let PointBuffer::I32(values) = buffer else {
                return Err(SpatialError::UnsupportedDType(field.dtype));
            };
            Ok(values[index] as f32)
        }
        DType::U32 => {
            let PointBuffer::U32(values) = buffer else {
                return Err(SpatialError::UnsupportedDType(field.dtype));
            };
            Ok(values[index] as f32)
        }
    }
}

#[cfg(feature = "filter-voxel-gpu")]
fn set_field_from_f32(
    buffers: &mut PointBufferSet,
    field: &PointField,
    values: Vec<f32>,
) -> SpatialResult<()> {
    let buffer = match field.dtype {
        DType::F32 | DType::F16 => PointBuffer::from_f32(values),
        DType::F64 => PointBuffer::F64(values.into_iter().map(f64::from).collect()),
        DType::U8 => PointBuffer::U8(values.into_iter().map(|value| value.round() as u8).collect()),
        DType::U16 => {
            PointBuffer::U16(values.into_iter().map(|value| value.round() as u16).collect())
        }
        DType::I32 => {
            PointBuffer::I32(values.into_iter().map(|value| value.round() as i32).collect())
        }
        DType::U32 => {
            PointBuffer::U32(values.into_iter().map(|value| value.round() as u32).collect())
        }
    };
    buffers.insert(field.name.clone(), buffer);
    Ok(())
}

#[cfg(feature = "filter-voxel-gpu")]
fn set_field_from_u8(
    buffers: &mut PointBufferSet,
    field: &PointField,
    values: Vec<u8>,
) -> SpatialResult<()> {
    if field.dtype != DType::U8 {
        return Err(SpatialError::UnsupportedDType(field.dtype));
    }
    buffers.insert(field.name.clone(), PointBuffer::U8(values));
    Ok(())
}

fn push_field(buffers: &mut PointBufferSet, field: &PointField, value: f32) -> SpatialResult<()> {
    let buffer = buffers
        .get_mut(&field.name)
        .ok_or_else(|| SpatialError::MissingField(field.name.clone()))?;
    match field.dtype {
        DType::F32 | DType::F16 => buffer.push_f32(value),
        DType::F64 => buffer.push_f64(f64::from(value)),
        DType::U8 => buffer.push_u8(value.round() as u8),
        DType::U16 => buffer.push_u16(value.round() as u16),
        DType::I32 => buffer.push_i32(value.round() as i32),
        DType::U32 => {
            let PointBuffer::U32(values) = buffer else {
                return Err(SpatialError::UnsupportedDType(field.dtype));
            };
            values.push(value.round() as u32);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{VoxelGridDownsample, VoxelGridDownsampleConfig};
    use crate::PointCloudFilter;
    #[cfg(feature = "filter-voxel-gpu")]
    use spatialrust_core::HasNormals3;
    use spatialrust_core::{HasIntensity, HasPositions3, PointCloudBuilder, StandardSchemas};

    #[test]
    fn centroid_downsample_reduces_points() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        builder.push_point([0.1, 0.0, 0.0]).unwrap();
        builder.push_point([1.0, 0.0, 0.0]).unwrap();
        builder.push_point([1.1, 0.0, 0.0]).unwrap();
        let input = builder.build().unwrap();

        let filter = VoxelGridDownsample::new(
            VoxelGridDownsampleConfig::centroid(0.5).without_gpu_min_points(),
        );
        let output = filter.filter(&input).unwrap();
        assert_eq!(output.len(), 2);

        let (x, _, _) = output.positions3().unwrap();
        assert!((x[0] - 0.05).abs() < 1e-5);
        assert!((x[1] - 1.05).abs() < 1e-5);
    }

    #[test]
    fn approximate_keeps_first_point_in_voxel() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzi());
        builder.push_point([0.0, 0.0, 0.0, 0.2]).unwrap();
        builder.push_point([0.1, 0.0, 0.0, 0.9]).unwrap();
        let input = builder.build().unwrap();

        let filter = VoxelGridDownsample::new(VoxelGridDownsampleConfig::approximate(1.0));
        let output = filter.filter(&input).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output.intensity().unwrap()[0], 0.2);
    }

    #[test]
    fn average_intensity_in_centroid_mode() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzi());
        builder.push_point([0.0, 0.0, 0.0, 0.2]).unwrap();
        builder.push_point([0.1, 0.0, 0.0, 0.8]).unwrap();
        let input = builder.build().unwrap();

        let filter = VoxelGridDownsample::new(
            VoxelGridDownsampleConfig::centroid(1.0).without_gpu_min_points(),
        );
        let output = filter.filter(&input).unwrap();
        assert_eq!(output.len(), 1);
        assert!((output.intensity().unwrap()[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn rejects_non_positive_leaf_size() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        let input = builder.build().unwrap();
        let filter = VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(0.0));
        assert!(filter.filter(&input).is_err());
    }

    #[cfg(feature = "filter-voxel-gpu")]
    #[test]
    fn gpu_policy_matches_cpu_downsample() {
        use spatialrust_core::ExecutionPolicy;

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        builder.push_point([0.1, 0.0, 0.0]).unwrap();
        builder.push_point([1.0, 0.0, 0.0]).unwrap();
        builder.push_point([1.1, 0.0, 0.0]).unwrap();
        let input = builder.build().unwrap();

        let filter = VoxelGridDownsample::new(
            VoxelGridDownsampleConfig::centroid(0.5).without_gpu_min_points(),
        );
        let cpu = filter.filter(&input).unwrap();
        let gpu = filter
            .filter_with_policy(&input, ExecutionPolicy::Gpu(spatialrust_core::DeviceKind::Wgpu))
            .unwrap();

        assert_eq!(cpu.len(), gpu.len());
        let (cpu_x, _, _) = cpu.positions3().unwrap();
        let (gpu_x, _, _) = gpu.positions3().unwrap();
        assert!((cpu_x[0] - gpu_x[0]).abs() < 1e-5);
        assert!((cpu_x[1] - gpu_x[1]).abs() < 1e-5);
    }

    #[cfg(feature = "filter-voxel-gpu")]
    #[test]
    fn gpu_policy_averages_attributes_on_gpu() {
        use spatialrust_core::ExecutionPolicy;

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzi());
        builder.push_point([0.0, 0.0, 0.0, 0.2]).unwrap();
        builder.push_point([0.1, 0.0, 0.0, 0.8]).unwrap();
        let input = builder.build().unwrap();

        let filter = VoxelGridDownsample::new(
            VoxelGridDownsampleConfig::centroid(1.0).without_gpu_min_points(),
        );
        let cpu = filter.filter(&input).unwrap();
        let gpu = filter
            .filter_with_policy(&input, ExecutionPolicy::Gpu(spatialrust_core::DeviceKind::Wgpu))
            .unwrap();

        assert_eq!(cpu.len(), gpu.len());
        assert!((cpu.intensity().unwrap()[0] - gpu.intensity().unwrap()[0]).abs() < 1e-5);
    }

    #[cfg(feature = "filter-voxel-gpu")]
    #[test]
    fn gpu_policy_averages_u8_rgb_on_gpu() {
        use spatialrust_core::{ExecutionPolicy, PointBuffer};

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzrgb());
        builder.push_point([0.0, 0.0, 0.0, 10.0, 20.0, 30.0]).unwrap();
        builder.push_point([0.1, 0.0, 0.0, 30.0, 40.0, 50.0]).unwrap();
        let input = builder.build().unwrap();

        let filter = VoxelGridDownsample::new(
            VoxelGridDownsampleConfig::centroid(1.0).without_gpu_min_points(),
        );
        let cpu = filter.filter(&input).unwrap();
        let gpu = filter
            .filter_with_policy(&input, ExecutionPolicy::Gpu(spatialrust_core::DeviceKind::Wgpu))
            .unwrap();

        assert_eq!(cpu.len(), gpu.len());
        for channel in ["r", "g", "b"] {
            let PointBuffer::U8(cpu_values) = cpu.field(channel).unwrap() else {
                panic!("expected u8 channel");
            };
            let PointBuffer::U8(gpu_values) = gpu.field(channel).unwrap() else {
                panic!("expected u8 channel");
            };
            assert_eq!(cpu_values, gpu_values);
        }
    }

    #[cfg(feature = "filter-voxel-gpu")]
    #[test]
    fn gpu_approximate_first_matches_cpu_downsample() {
        use spatialrust_core::ExecutionPolicy;

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzi());
        builder.push_point([0.0, 0.0, 0.0, 0.2]).unwrap();
        builder.push_point([0.1, 0.0, 0.0, 0.9]).unwrap();
        builder.push_point([1.0, 0.0, 0.0, 10.0]).unwrap();
        builder.push_point([1.1, 0.0, 0.0, 20.0]).unwrap();
        let input = builder.build().unwrap();

        let filter = VoxelGridDownsample::new(
            VoxelGridDownsampleConfig::approximate(0.5).without_gpu_min_points(),
        );
        let cpu = filter.filter(&input).unwrap();
        let gpu = filter
            .filter_with_policy(&input, ExecutionPolicy::Gpu(spatialrust_core::DeviceKind::Wgpu))
            .unwrap();

        assert_eq!(cpu.len(), gpu.len());
        let (cpu_x, _, _) = cpu.positions3().unwrap();
        let (gpu_x, _, _) = gpu.positions3().unwrap();
        for index in 0..cpu.len() {
            assert!((cpu_x[index] - gpu_x[index]).abs() < 1e-5);
        }
        assert_eq!(cpu.intensity().unwrap(), gpu.intensity().unwrap());
    }

    #[cfg(feature = "filter-voxel-gpu")]
    #[test]
    fn gpu_approximate_first_xyzinormal_matches_cpu_downsample() {
        use spatialrust_core::ExecutionPolicy;

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzinormal());
        builder.push_point([0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 1.0]).unwrap();
        builder.push_point([0.1, 0.0, 0.0, 0.9, 0.1, 0.0, 1.0]).unwrap();
        builder.push_point([1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0]).unwrap();
        builder.push_point([1.1, 0.0, 0.0, 20.0, 0.0, 0.0, 1.0]).unwrap();
        let input = builder.build().unwrap();

        let filter = VoxelGridDownsample::new(
            VoxelGridDownsampleConfig::approximate(0.5).without_gpu_min_points(),
        );
        let cpu = filter.filter(&input).unwrap();
        let gpu = filter
            .filter_with_policy(&input, ExecutionPolicy::Gpu(spatialrust_core::DeviceKind::Wgpu))
            .unwrap();

        assert_eq!(cpu.len(), gpu.len());
        let (cpu_x, cpu_y, cpu_z) = cpu.positions3().unwrap();
        let (gpu_x, gpu_y, gpu_z) = gpu.positions3().unwrap();
        for index in 0..cpu.len() {
            assert!((cpu_x[index] - gpu_x[index]).abs() < 1e-5);
            assert!((cpu_y[index] - gpu_y[index]).abs() < 1e-5);
            assert!((cpu_z[index] - gpu_z[index]).abs() < 1e-5);
        }
        assert_eq!(cpu.intensity().unwrap(), gpu.intensity().unwrap());
        let (cpu_nx, cpu_ny, cpu_nz) = cpu.normals3().unwrap();
        let (gpu_nx, gpu_ny, gpu_nz) = gpu.normals3().unwrap();
        assert_eq!(cpu_nx, gpu_nx);
        assert_eq!(cpu_ny, gpu_ny);
        assert_eq!(cpu_nz, gpu_nz);
    }

    #[cfg(feature = "filter-voxel-gpu")]
    #[test]
    fn gpu_policy_falls_back_to_cpu_below_threshold() {
        use spatialrust_core::ExecutionPolicy;

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        builder.push_point([0.1, 0.0, 0.0]).unwrap();
        let input = builder.build().unwrap();

        let filter = VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(0.5));
        let cpu = filter.filter(&input).unwrap();
        let gpu = filter
            .filter_with_policy(&input, ExecutionPolicy::Gpu(spatialrust_core::DeviceKind::Wgpu))
            .unwrap();

        assert_eq!(cpu.len(), gpu.len());
        let (cpu_x, _, _) = cpu.positions3().unwrap();
        let (gpu_x, _, _) = gpu.positions3().unwrap();
        assert!((cpu_x[0] - gpu_x[0]).abs() < 1e-5);
    }

    #[cfg(feature = "filter-voxel-gpu")]
    #[test]
    fn auto_policy_uses_cpu_for_small_clouds() {
        use spatialrust_core::ExecutionPolicy;

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        builder.push_point([0.1, 0.0, 0.0]).unwrap();
        let input = builder.build().unwrap();

        let filter = VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(0.5));
        let cpu = filter.filter(&input).unwrap();
        let auto = filter.filter_with_policy(&input, ExecutionPolicy::Auto).unwrap();

        assert_eq!(cpu.len(), auto.len());
        let (cpu_x, _, _) = cpu.positions3().unwrap();
        let (auto_x, _, _) = auto.positions3().unwrap();
        assert!((cpu_x[0] - auto_x[0]).abs() < 1e-5);
    }

    #[cfg(feature = "filter-voxel-gpu")]
    #[test]
    fn approximate_default_gpu_threshold_is_higher_than_centroid() {
        use super::{
            VoxelGridDownsampleConfig, DEFAULT_GPU_MIN_POINTS, DEFAULT_GPU_MIN_POINTS_APPROXIMATE,
        };

        #[allow(clippy::assertions_on_constants)]
        {
            assert!(DEFAULT_GPU_MIN_POINTS_APPROXIMATE > DEFAULT_GPU_MIN_POINTS);
        }

        let centroid = VoxelGridDownsampleConfig::centroid(0.5);
        let approximate = VoxelGridDownsampleConfig::approximate(0.5);
        assert_eq!(centroid.gpu_min_points, Some(DEFAULT_GPU_MIN_POINTS));
        assert_eq!(approximate.gpu_min_points, Some(DEFAULT_GPU_MIN_POINTS_APPROXIMATE));
    }

    #[test]
    fn effective_gpu_min_points_blocks_heavy_approximate_schema() {
        use super::{
            VoxelGridDownsampleConfig, APPROXIMATE_HEAVY_F32_ATTRIBUTE_CHANNELS,
            DEFAULT_GPU_MIN_POINTS_APPROXIMATE, DEFAULT_GPU_MIN_POINTS_APPROXIMATE_HEAVY,
        };

        let approximate = VoxelGridDownsampleConfig::approximate(1.0);
        assert_eq!(
            approximate.effective_gpu_min_points(&StandardSchemas::point_xyz()),
            Some(DEFAULT_GPU_MIN_POINTS_APPROXIMATE)
        );
        assert_eq!(
            approximate.effective_gpu_min_points(&StandardSchemas::point_xyzinormal()),
            Some(DEFAULT_GPU_MIN_POINTS_APPROXIMATE_HEAVY)
        );
        assert!(
            super::count_non_position_f32_fields(&StandardSchemas::point_xyzinormal())
                >= APPROXIMATE_HEAVY_F32_ATTRIBUTE_CHANNELS
        );
    }

    #[cfg(feature = "filter-voxel-gpu")]
    #[test]
    fn auto_approximate_first_uses_cpu_for_xyzinormal() {
        use spatialrust_core::ExecutionPolicy;

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzinormal());
        for index in 0..128 {
            builder
                .push_point([
                    (index % 16) as f32 * 0.1,
                    (index / 16) as f32 * 0.1,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    1.0,
                ])
                .unwrap();
        }
        let input = builder.build().unwrap();

        let mut config = VoxelGridDownsampleConfig::approximate(0.5);
        config.gpu_min_points = Some(10);
        let filter = VoxelGridDownsample::new(config);
        let cpu = filter.filter(&input).unwrap();
        let auto = filter.filter_with_policy(&input, ExecutionPolicy::Auto).unwrap();

        assert_eq!(cpu.len(), auto.len());
        let (cpu_x, _, _) = cpu.positions3().unwrap();
        let (auto_x, _, _) = auto.positions3().unwrap();
        for index in 0..cpu.len() {
            assert!((cpu_x[index] - auto_x[index]).abs() < 1e-5);
        }
    }

    #[cfg(feature = "filter-voxel-gpu")]
    fn synthetic_xyzinormal_plane(point_count: usize) -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzinormal());
        for index in 0..point_count {
            let x = (index % 256) as f32 * 0.1;
            let y = ((index / 256) % 256) as f32 * 0.1;
            let intensity = (index % 256) as f32;
            builder.push_point([x, y, 0.0, intensity, 0.0, 0.0, 1.0]).unwrap();
        }
        builder.build().unwrap()
    }

    #[cfg(feature = "filter-voxel-gpu")]
    #[test]
    fn auto_approximate_first_uses_cpu_below_heavy_threshold() {
        use spatialrust_core::ExecutionPolicy;

        const POINT_COUNT: usize = 500_000;
        let input = synthetic_xyzinormal_plane(POINT_COUNT);
        let filter = VoxelGridDownsample::new(VoxelGridDownsampleConfig::approximate(4.0));
        let cpu = filter.filter(&input).unwrap();
        let auto = filter.filter_with_policy(&input, ExecutionPolicy::Auto).unwrap();

        assert_eq!(cpu.len(), auto.len());
        let (cpu_x, cpu_y, cpu_z) = cpu.positions3().unwrap();
        let (auto_x, auto_y, auto_z) = auto.positions3().unwrap();
        for index in 0..cpu.len() {
            assert!((cpu_x[index] - auto_x[index]).abs() < 1e-4);
            assert!((cpu_y[index] - auto_y[index]).abs() < 1e-4);
            assert!((cpu_z[index] - auto_z[index]).abs() < 1e-4);
        }
    }

    #[cfg(feature = "filter-voxel-gpu")]
    #[test]
    fn auto_approximate_first_uses_gpu_at_heavy_threshold() {
        use spatialrust_core::{DeviceKind, ExecutionPolicy};

        const POINT_COUNT: usize = 1_000_000;
        let input = synthetic_xyzinormal_plane(POINT_COUNT);
        let filter = VoxelGridDownsample::new(VoxelGridDownsampleConfig::approximate(4.0));
        let gpu =
            filter.filter_with_policy(&input, ExecutionPolicy::Gpu(DeviceKind::Wgpu)).unwrap();
        let auto = filter.filter_with_policy(&input, ExecutionPolicy::Auto).unwrap();

        assert_eq!(gpu.len(), auto.len());
        let (gpu_x, gpu_y, gpu_z) = gpu.positions3().unwrap();
        let (auto_x, auto_y, auto_z) = auto.positions3().unwrap();
        for index in 0..gpu.len() {
            assert!((gpu_x[index] - auto_x[index]).abs() < 1e-4);
            assert!((gpu_y[index] - auto_y[index]).abs() < 1e-4);
            assert!((gpu_z[index] - auto_z[index]).abs() < 1e-4);
        }
    }
}
