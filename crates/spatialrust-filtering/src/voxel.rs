use std::collections::HashMap;

use spatialrust_core::{
    DType, DeviceKind, ExecutionPolicy, FieldSemantic, HasPositions3, PointBuffer, PointBufferSet,
    PointCloud, PointField, SpatialError, SpatialResult,
};
use spatialrust_math::Vec3;

use crate::filter::PointCloudFilter;

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
/// benches at 500k still favor CPU (~23 ms vs ~37 ms); GPU wins from ~750k upward.
pub const DEFAULT_GPU_MIN_POINTS_APPROXIMATE: usize = 750_000;

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
    /// centroid ~500k, approximate-first ~750k.
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

        let cells = match policy {
            ExecutionPolicy::Gpu(DeviceKind::Wgpu) => build_voxel_cells_gpu(x, y, z, origin, inv_leaf)?,
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
            match self.config.gpu_min_points {
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

#[derive(Clone, Debug, Default)]
struct VoxelCell {
    indices: Vec<usize>,
}

fn build_voxel_cells_cpu(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: Vec3<f32>,
    inv_leaf: f32,
) -> HashMap<(i64, i64, i64), VoxelCell> {
    let mut cells: HashMap<(i64, i64, i64), VoxelCell> = HashMap::new();
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
) -> SpatialResult<HashMap<(i64, i64, i64), VoxelCell>> {
    use spatialrust_gpu::{compute_voxel_keys, WgpuRuntime};

    let runtime = WgpuRuntime::shared()?;
    let keys = compute_voxel_keys(
        &runtime,
        x,
        y,
        z,
        [origin.x, origin.y, origin.z],
        inv_leaf,
    )?;

    let mut cells: HashMap<(i64, i64, i64), VoxelCell> = HashMap::new();
    for (index, key) in keys.into_iter().enumerate() {
        cells.entry(key).or_default().indices.push(index);
    }
    Ok(cells)
}

#[cfg(feature = "filter-voxel-gpu")]
fn gpu_aggregate_attribute_fields(
    runtime: &spatialrust_gpu::WgpuRuntime,
    input: &PointCloud,
    fields: &[&PointField],
    segments: &spatialrust_gpu::GpuVoxelSegments,
    policy: AttributeAggregation,
) -> SpatialResult<Vec<Vec<f32>>> {
    use spatialrust_gpu::{gather_voxel_first_f32_multi_gpu, reduce_voxel_average_f32_multi_gpu};

    if fields.is_empty() {
        return Ok(Vec::new());
    }

    match policy {
        AttributeAggregation::First => {
            let mut sources = Vec::with_capacity(fields.len());
            for field in fields {
                let mut source_values = Vec::with_capacity(input.len());
                for index in 0..input.len() {
                    source_values.push(read_field_f32(input, field, index)?);
                }
                sources.push(source_values);
            }

            let refs: Vec<&[f32]> = sources.iter().map(Vec::as_slice).collect();
            gather_voxel_first_f32_multi_gpu(runtime, &refs, segments)
        }
        AttributeAggregation::Average => {
            let mut sources = Vec::with_capacity(fields.len());
            for field in fields {
                let mut source_values = Vec::with_capacity(input.len());
                for index in 0..input.len() {
                    source_values.push(read_field_f32(input, field, index)?);
                }
                sources.push(source_values);
            }

            let refs: Vec<&[f32]> = sources.iter().map(Vec::as_slice).collect();
            reduce_voxel_average_f32_multi_gpu(runtime, &refs, segments)
        }
    }
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
    use spatialrust_gpu::{downsample_voxel_centroid_gpu, WgpuRuntime};

    let runtime = WgpuRuntime::shared()?;
    let pipeline = downsample_voxel_centroid_gpu(
        &runtime,
        x,
        y,
        z,
        [origin.x, origin.y, origin.z],
        inv_leaf,
    )?;
    let segments = &pipeline.segments;
    let (out_x, out_y, out_z) = (pipeline.out_x, pipeline.out_y, pipeline.out_z);

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

    let attribute_fields: Vec<_> = schema
        .fields()
        .iter()
        .filter(|field| {
            !matches!(
                field.semantic,
                FieldSemantic::PositionX | FieldSemantic::PositionY | FieldSemantic::PositionZ
            )
        })
        .collect();
    let attribute_refs: Vec<_> = attribute_fields.to_vec();
    let attribute_values = gpu_aggregate_attribute_fields(
        &runtime,
        input,
        &attribute_refs,
        segments,
        attribute_policy,
    )?;
    for (field, values) in attribute_fields.iter().zip(attribute_values) {
        set_field_from_f32(&mut buffers, field, values)?;
    }

    PointCloud::try_from_parts(schema, buffers, input.metadata().clone())
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
    use spatialrust_gpu::{downsample_voxel_approximate_first_gpu, WgpuRuntime};

    let runtime = WgpuRuntime::shared()?;
    let pipeline = downsample_voxel_approximate_first_gpu(
        &runtime,
        x,
        y,
        z,
        [origin.x, origin.y, origin.z],
        inv_leaf,
    )?;
    let segments = &pipeline.segments;
    let (out_x, out_y, out_z) = (pipeline.out_x, pipeline.out_y, pipeline.out_z);

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

    let attribute_fields: Vec<_> = schema
        .fields()
        .iter()
        .filter(|field| {
            !matches!(
                field.semantic,
                FieldSemantic::PositionX | FieldSemantic::PositionY | FieldSemantic::PositionZ
            )
        })
        .collect();
    let attribute_refs: Vec<_> = attribute_fields.to_vec();
    let attribute_values = gpu_aggregate_attribute_fields(
        &runtime,
        input,
        &attribute_refs,
        segments,
        attribute_policy,
    )?;
    for (field, values) in attribute_fields.iter().zip(attribute_values) {
        set_field_from_f32(&mut buffers, field, values)?;
    }

    PointCloud::try_from_parts(schema, buffers, input.metadata().clone())
}

#[cfg(not(feature = "filter-voxel-gpu"))]
fn build_voxel_cells_gpu(
    _x: &[f32],
    _y: &[f32],
    _z: &[f32],
    _origin: Vec3<f32>,
    _inv_leaf: f32,
) -> SpatialResult<HashMap<(i64, i64, i64), VoxelCell>> {
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

fn set_field_from_f32(
    buffers: &mut PointBufferSet,
    field: &PointField,
    values: Vec<f32>,
) -> SpatialResult<()> {
    let buffer = match field.dtype {
        DType::F32 | DType::F16 => PointBuffer::from_f32(values),
        DType::F64 => PointBuffer::F64(values.into_iter().map(f64::from).collect()),
        DType::U8 => PointBuffer::U8(values.into_iter().map(|value| value.round() as u8).collect()),
        DType::U16 => PointBuffer::U16(values.into_iter().map(|value| value.round() as u16).collect()),
        DType::I32 => PointBuffer::I32(values.into_iter().map(|value| value.round() as i32).collect()),
        DType::U32 => PointBuffer::U32(values.into_iter().map(|value| value.round() as u32).collect()),
    };
    buffers.insert(field.name.clone(), buffer);
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
    use spatialrust_core::{HasIntensity, HasPositions3, PointCloudBuilder, StandardSchemas};

    #[test]
    fn centroid_downsample_reduces_points() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        builder.push_point([0.1, 0.0, 0.0]).unwrap();
        builder.push_point([1.0, 0.0, 0.0]).unwrap();
        builder.push_point([1.1, 0.0, 0.0]).unwrap();
        let input = builder.build().unwrap();

        let filter = VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(0.5).without_gpu_min_points());
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

        let filter = VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(1.0).without_gpu_min_points());
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

        let filter = VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(0.5).without_gpu_min_points());
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

        let filter = VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(1.0).without_gpu_min_points());
        let cpu = filter.filter(&input).unwrap();
        let gpu = filter
            .filter_with_policy(&input, ExecutionPolicy::Gpu(spatialrust_core::DeviceKind::Wgpu))
            .unwrap();

        assert_eq!(cpu.len(), gpu.len());
        assert!((cpu.intensity().unwrap()[0] - gpu.intensity().unwrap()[0]).abs() < 1e-5);
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

        let filter = VoxelGridDownsample::new(VoxelGridDownsampleConfig::approximate(0.5).without_gpu_min_points());
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
        let auto = filter
            .filter_with_policy(&input, ExecutionPolicy::Auto)
            .unwrap();

        assert_eq!(cpu.len(), auto.len());
        let (cpu_x, _, _) = cpu.positions3().unwrap();
        let (auto_x, _, _) = auto.positions3().unwrap();
        assert!((cpu_x[0] - auto_x[0]).abs() < 1e-5);
    }

    #[cfg(feature = "filter-voxel-gpu")]
    #[test]
    fn approximate_default_gpu_threshold_is_higher_than_centroid() {
        use super::{
            DEFAULT_GPU_MIN_POINTS, DEFAULT_GPU_MIN_POINTS_APPROXIMATE, VoxelGridDownsampleConfig,
        };

        assert!(DEFAULT_GPU_MIN_POINTS_APPROXIMATE > DEFAULT_GPU_MIN_POINTS);

        let centroid = VoxelGridDownsampleConfig::centroid(0.5);
        let approximate = VoxelGridDownsampleConfig::approximate(0.5);
        assert_eq!(centroid.gpu_min_points, Some(DEFAULT_GPU_MIN_POINTS));
        assert_eq!(
            approximate.gpu_min_points,
            Some(DEFAULT_GPU_MIN_POINTS_APPROXIMATE)
        );
    }
}
