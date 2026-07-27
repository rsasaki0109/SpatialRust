//! Bounded chunk operations and deterministic external voxel aggregation.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use spatialrust_core::{
    DType, FieldSemantic, PointBuffer, PointBufferSet, PointCloud, PointSchema, SpatialMetadata,
};
use spatialrust_io::{BoundedSpool, SpoolOptions};
use spatialrust_math::{Mat4, Vec3};
use spatialrust_records::{
    BoundedSpatialRecordSource, CancellationToken, ChunkIdentity, MemoryReservation, MemoryTracker,
    RecordsError, RecordsResult, SchemaDescriptor, SpatialRecord, SpatialRecordChunk,
    StreamOptions,
};

/// Per-chunk operation that preserves the input schema.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChunkMapOperation {
    /// Keep points inside inclusive bounds, or outside them when `invert` is true.
    Crop {
        /// Inclusive XYZ minimum.
        min: [f32; 3],
        /// Inclusive XYZ maximum.
        max: [f32; 3],
        /// Select the complement of the box.
        invert: bool,
    },
    /// Apply an affine transform to positions and its linear part to normals.
    Transform(Mat4<f32>),
}

/// Bounded source adapter for chunk-local crop and affine transform operations.
pub struct ChunkMapSource<S> {
    source: S,
    operation: ChunkMapOperation,
    schema: SchemaDescriptor,
    options: StreamOptions,
    tracker: MemoryTracker,
    cancellation: CancellationToken,
    max_chunk_bytes: u64,
    next_sequence: u64,
    next_point_offset: u64,
}

impl<S: BoundedSpatialRecordSource> ChunkMapSource<S> {
    /// Creates a crop adapter and validates ordered finite bounds.
    pub fn crop(source: S, min: [f32; 3], max: [f32; 3], invert: bool) -> RecordsResult<Self> {
        if min.into_iter().chain(max).any(|value| !value.is_finite())
            || (0..3).any(|axis| min[axis] > max[axis])
        {
            return Err(RecordsError::InvalidConfiguration(
                "crop bounds must be finite and ordered".into(),
            ));
        }
        Self::try_new(source, ChunkMapOperation::Crop { min, max, invert })
    }

    /// Creates an affine transform adapter.
    pub fn transform(source: S, transform: Mat4<f32>) -> RecordsResult<Self> {
        Self::try_new(source, ChunkMapOperation::Transform(transform))
    }

    fn try_new(source: S, operation: ChunkMapOperation) -> RecordsResult<Self> {
        let schema = source.schema().clone();
        let options = source.options().clone();
        let tracker = source.memory_tracker().clone();
        let cancellation = source.cancellation_token();
        let max_chunk_bytes = schema_bytes(&schema, options.chunk_points())?;
        Ok(Self {
            source,
            operation,
            schema,
            options,
            tracker,
            cancellation,
            max_chunk_bytes,
            next_sequence: 0,
            next_point_offset: 0,
        })
    }

    /// Consumes the adapter and returns the upstream source.
    #[must_use]
    pub fn into_inner(self) -> S {
        self.source
    }

    fn map_cloud(&self, input: &PointCloud) -> RecordsResult<PointCloud> {
        match self.operation {
            ChunkMapOperation::Crop { min, max, invert } => crop_cloud(input, min, max, invert),
            ChunkMapOperation::Transform(transform) => transform_cloud(input, transform),
        }
    }
}

impl<S: BoundedSpatialRecordSource> BoundedSpatialRecordSource for ChunkMapSource<S> {
    fn schema(&self) -> &SchemaDescriptor {
        &self.schema
    }

    fn options(&self) -> &StreamOptions {
        &self.options
    }

    fn memory_tracker(&self) -> &MemoryTracker {
        &self.tracker
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    fn max_chunk_bytes(&self) -> u64 {
        self.max_chunk_bytes
    }

    fn next_chunk(&mut self) -> Option<RecordsResult<SpatialRecordChunk>> {
        loop {
            if let Err(error) = self.cancellation.check() {
                return Some(Err(error));
            }
            let input = match self.source.next_chunk()? {
                Ok(chunk) => chunk,
                Err(error) => return Some(Err(error)),
            };
            let input_points = input.record().cloud().len();
            let reservation =
                match self.tracker.try_reserve(match schema_bytes(&self.schema, input_points) {
                    Ok(bytes) => bytes,
                    Err(error) => return Some(Err(error)),
                }) {
                    Ok(reservation) => reservation,
                    Err(error) => return Some(Err(error)),
                };
            let cloud = match self.map_cloud(input.record().cloud()) {
                Ok(cloud) => cloud,
                Err(error) => return Some(Err(error)),
            };
            drop(input);
            if cloud.is_empty() {
                continue;
            }
            let point_count = match usize_u64(cloud.len(), "chunk map point count") {
                Ok(value) => value,
                Err(error) => return Some(Err(error)),
            };
            let identity = ChunkIdentity {
                sequence: self.next_sequence,
                point_offset: self.next_point_offset,
            };
            let record = match SpatialRecord::try_new(self.schema.clone(), cloud) {
                Ok(record) => record,
                Err(error) => return Some(Err(error)),
            };
            let next_sequence = match self.next_sequence.checked_add(1) {
                Some(value) => value,
                None => {
                    return Some(Err(RecordsError::ReceiptOverflow("chunk map sequence".into())));
                }
            };
            let next_point_offset = match self.next_point_offset.checked_add(point_count) {
                Some(value) => value,
                None => {
                    return Some(Err(RecordsError::ReceiptOverflow(
                        "chunk map point offset".into(),
                    )));
                }
            };
            let chunk = SpatialRecordChunk::try_from_reserved(identity, record, reservation);
            if chunk.is_ok() {
                self.next_sequence = next_sequence;
                self.next_point_offset = next_point_offset;
            }
            return Some(chunk);
        }
    }
}

/// Deterministic global position reduction over a bounded source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionReduction {
    /// Number of points observed, including non-finite positions.
    pub point_count: u64,
    /// Number of positions whose three components are finite.
    pub finite_point_count: u64,
    /// Finite inclusive bounds, or `None` when no finite positions exist.
    pub bounds: Option<([f64; 3], [f64; 3])>,
    /// Mean of finite positions, or `None` when no finite positions exist.
    pub centroid: Option<[f64; 3]>,
}

/// Consumes a source and computes bounds and centroid without retaining chunks.
pub fn reduce_positions(
    source: &mut impl BoundedSpatialRecordSource,
) -> RecordsResult<PositionReduction> {
    let mut point_count = 0_u64;
    let mut finite_count = 0_u64;
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut sums = [CompensatedSum::default(); 3];
    while let Some(chunk) = source.next_chunk() {
        let chunk = chunk?;
        let cloud = chunk.record().cloud();
        let (x, y, z) = positions(cloud)?;
        point_count = point_count
            .checked_add(usize_u64(cloud.len(), "reduction chunk point count")?)
            .ok_or_else(|| RecordsError::ReceiptOverflow("reduction point count".into()))?;
        for index in 0..cloud.len() {
            let point = [f64::from(x[index]), f64::from(y[index]), f64::from(z[index])];
            if point.iter().any(|value| !value.is_finite()) {
                continue;
            }
            finite_count = finite_count
                .checked_add(1)
                .ok_or_else(|| RecordsError::ReceiptOverflow("finite point count".into()))?;
            for axis in 0..3 {
                min[axis] = min[axis].min(point[axis]);
                max[axis] = max[axis].max(point[axis]);
                sums[axis].add(point[axis]);
            }
        }
    }
    Ok(PositionReduction {
        point_count,
        finite_point_count: finite_count,
        bounds: (finite_count > 0).then_some((min, max)),
        centroid: (finite_count > 0).then(|| {
            [
                sums[0].total() / finite_count as f64,
                sums[1].total() / finite_count as f64,
                sums[2].total() / finite_count as f64,
            ]
        }),
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let corrected = value - self.correction;
        let next = self.sum + corrected;
        self.correction = (next - self.sum) - corrected;
        self.sum = next;
    }

    fn total(self) -> f64 {
        self.sum
    }
}

fn crop_cloud(
    input: &PointCloud,
    min: [f32; 3],
    max: [f32; 3],
    invert: bool,
) -> RecordsResult<PointCloud> {
    let (x, y, z) = positions(input)?;
    let mut buffers = empty_buffers(input.schema(), input.len());
    for field in input.schema().fields() {
        let source = input.field(&field.name)?;
        let target = buffers
            .get_mut(&field.name)
            .ok_or_else(|| RecordsError::InvalidChunk("crop output buffer missing".into()))?;
        for index in 0..input.len() {
            let inside = x[index] >= min[0]
                && x[index] <= max[0]
                && y[index] >= min[1]
                && y[index] <= max[1]
                && z[index] >= min[2]
                && z[index] <= max[2];
            if inside ^ invert {
                push_scalar(target, scalar_at(source, index)?)?;
            }
        }
    }
    PointCloud::try_from_parts(input.schema().clone(), buffers, input.metadata().clone())
        .map_err(Into::into)
}

fn transform_cloud(input: &PointCloud, transform: Mat4<f32>) -> RecordsResult<PointCloud> {
    let (x, y, z) = positions(input)?;
    let normals = normal_columns(input)?;
    let mut buffers = empty_buffers(input.schema(), input.len());
    for field in input.schema().fields() {
        let source = input.field(&field.name)?;
        let target = buffers
            .get_mut(&field.name)
            .ok_or_else(|| RecordsError::InvalidChunk("transform output buffer missing".into()))?;
        for index in 0..input.len() {
            let value = match field.semantic {
                FieldSemantic::PositionX | FieldSemantic::PositionY | FieldSemantic::PositionZ => {
                    let point = transform.transform_point(Vec3::new(x[index], y[index], z[index]));
                    match field.semantic {
                        FieldSemantic::PositionX => f64::from(point.x),
                        FieldSemantic::PositionY => f64::from(point.y),
                        _ => f64::from(point.z),
                    }
                }
                FieldSemantic::NormalX | FieldSemantic::NormalY | FieldSemantic::NormalZ => {
                    let (nx, ny, nz) = normals.ok_or_else(|| {
                        RecordsError::InvalidChunk("incomplete normal columns".into())
                    })?;
                    let normal = transform
                        .transform_vector(Vec3::new(nx[index], ny[index], nz[index]))
                        .normalize();
                    match field.semantic {
                        FieldSemantic::NormalX => f64::from(normal.x),
                        FieldSemantic::NormalY => f64::from(normal.y),
                        _ => f64::from(normal.z),
                    }
                }
                _ => scalar_at(source, index)?,
            };
            push_scalar(target, value)?;
        }
    }
    PointCloud::try_from_parts(input.schema().clone(), buffers, input.metadata().clone())
        .map_err(Into::into)
}

/// Configuration for deterministic, spill-backed global voxel centroids.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamingVoxelConfig {
    leaf_size_bits: u32,
    run_points: usize,
    max_runs: usize,
    spool: SpoolOptions,
}

impl StreamingVoxelConfig {
    /// Creates a configuration with positive leaf size, run capacity, and run limit.
    pub fn new(
        leaf_size: f32,
        run_points: usize,
        max_runs: usize,
        spool: SpoolOptions,
    ) -> RecordsResult<Self> {
        if !leaf_size.is_finite() || leaf_size <= 0.0 {
            return Err(RecordsError::InvalidConfiguration(
                "streaming voxel leaf size must be positive and finite".into(),
            ));
        }
        if run_points == 0 || max_runs == 0 {
            return Err(RecordsError::InvalidConfiguration(
                "streaming voxel run_points and max_runs must be positive".into(),
            ));
        }
        Ok(Self { leaf_size_bits: leaf_size.to_bits(), run_points, max_runs, spool })
    }

    /// Returns the voxel edge length.
    #[must_use]
    pub fn leaf_size(&self) -> f32 {
        f32::from_bits(self.leaf_size_bits)
    }

    /// Returns the maximum points sorted in one in-memory run.
    #[must_use]
    pub const fn run_points(&self) -> usize {
        self.run_points
    }

    /// Returns the maximum number of external merge runs.
    #[must_use]
    pub const fn max_runs(&self) -> usize {
        self.max_runs
    }

    /// Returns the global disk spool contract.
    #[must_use]
    pub const fn spool(&self) -> &SpoolOptions {
        &self.spool
    }
}

/// Bounded source of globally aggregated voxel centroids.
///
/// Construction consumes the upstream stream into sorted fixed-width runs.
/// Output is then produced by a deterministic k-way merge ordered by
/// `(voxel key, source point offset)`.
pub struct StreamingVoxelSource {
    schema: SchemaDescriptor,
    options: StreamOptions,
    tracker: MemoryTracker,
    cancellation: CancellationToken,
    max_chunk_bytes: u64,
    metadata: SpatialMetadata,
    _spool: BoundedSpool,
    _merge_reservation: MemoryReservation,
    runs: Vec<RunCursor>,
    heap: BinaryHeap<HeapRecord>,
    pending: Option<VoxelAccumulator>,
    next_sequence: u64,
    next_point_offset: u64,
    finished: bool,
}

impl StreamingVoxelSource {
    /// Builds sorted runs from `source` under shared memory and disk budgets.
    pub fn try_build<S: BoundedSpatialRecordSource>(
        mut source: S,
        config: StreamingVoxelConfig,
    ) -> RecordsResult<Self> {
        let schema = source.schema().clone();
        let options = source.options().clone();
        let tracker = source.memory_tracker().clone();
        let cancellation = source.cancellation_token();
        let max_chunk_bytes = schema_bytes(&schema, options.chunk_points())?;
        let field_count = schema.point_schema().fields().len();
        let run_bytes = run_memory_bytes(config.run_points, field_count, config.max_runs)?;
        let run_reservation = tracker.try_reserve(run_bytes)?;
        let mut run = RunBuffer::with_capacity(config.run_points, field_count);
        let mut spool = BoundedSpool::create(config.spool(), "voxel-runs")
            .map_err(|error| RecordsError::InvalidConfiguration(error.to_string()))?;
        let mut run_metas = Vec::with_capacity(config.max_runs);
        let mut metadata = None;
        let leaf = f64::from(config.leaf_size());
        let mut expected_sequence = 0_u64;
        let mut expected_point_offset = 0_u64;

        while let Some(chunk) = source.next_chunk() {
            cancellation.check()?;
            let chunk = chunk?;
            let cloud = chunk.record().cloud();
            if chunk.identity().sequence != expected_sequence
                || chunk.identity().point_offset != expected_point_offset
            {
                return Err(RecordsError::InvalidChunk(format!(
                    "voxel input identity discontinuity: expected ({expected_sequence}, \
                     {expected_point_offset}), found ({}, {})",
                    chunk.identity().sequence,
                    chunk.identity().point_offset
                )));
            }
            expected_sequence = expected_sequence
                .checked_add(1)
                .ok_or_else(|| RecordsError::ReceiptOverflow("voxel input sequence".into()))?;
            expected_point_offset = expected_point_offset
                .checked_add(usize_u64(cloud.len(), "voxel input chunk point count")?)
                .ok_or_else(|| RecordsError::ReceiptOverflow("voxel input point offset".into()))?;
            if metadata.is_none() {
                metadata = Some(cloud.metadata().clone());
            }
            let (x, y, z) = positions(cloud)?;
            for local_index in 0..cloud.len() {
                if run.len() == config.run_points {
                    flush_run(&mut spool, &mut run, &mut run_metas, config.max_runs)?;
                }
                let point = [
                    f64::from(x[local_index]),
                    f64::from(y[local_index]),
                    f64::from(z[local_index]),
                ];
                if point.iter().any(|value| !value.is_finite()) {
                    continue;
                }
                let key = [
                    voxel_coordinate(point[0], leaf)?,
                    voxel_coordinate(point[1], leaf)?,
                    voxel_coordinate(point[2], leaf)?,
                ];
                let source_index = chunk
                    .identity()
                    .point_offset
                    .checked_add(usize_u64(local_index, "voxel local point index")?)
                    .ok_or_else(|| RecordsError::ReceiptOverflow("voxel source offset".into()))?;
                run.push(key, source_index, cloud, local_index)?;
            }
        }
        if !run.is_empty() {
            flush_run(&mut spool, &mut run, &mut run_metas, config.max_runs)?;
        }
        drop(run_reservation);
        spool.flush().map_err(|error| RecordsError::InvalidConfiguration(error.to_string()))?;

        let merge_bytes = merge_memory_bytes(run_metas.len(), field_count)?;
        let merge_reservation = tracker.try_reserve(merge_bytes)?;
        let mut runs = Vec::with_capacity(run_metas.len());
        let mut heap = BinaryHeap::new();
        for (run_index, meta) in run_metas.into_iter().enumerate() {
            let mut cursor = RunCursor::open(spool.path(), meta, field_count)?;
            if let Some(record) = cursor.next_record()? {
                heap.push(HeapRecord { run_index, record });
            }
            runs.push(cursor);
        }
        Ok(Self {
            schema,
            options,
            tracker,
            cancellation,
            max_chunk_bytes,
            metadata: metadata.unwrap_or_default(),
            _spool: spool,
            _merge_reservation: merge_reservation,
            runs,
            heap,
            pending: None,
            next_sequence: 0,
            next_point_offset: 0,
            finished: false,
        })
    }

    /// Returns the fixed-width temporary spill extent.
    #[must_use]
    pub const fn spool_bytes(&self) -> u64 {
        self._spool.extent_bytes()
    }

    /// Returns the number of sorted runs participating in the merge.
    #[must_use]
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    fn pop_record(&mut self) -> RecordsResult<Option<SpillRecord>> {
        let Some(item) = self.heap.pop() else {
            return Ok(None);
        };
        if let Some(next) = self.runs[item.run_index].next_record()? {
            self.heap.push(HeapRecord { run_index: item.run_index, record: next });
        }
        Ok(Some(item.record))
    }

    fn next_accumulator(&mut self) -> RecordsResult<Option<VoxelAccumulator>> {
        loop {
            let Some(record) = self.pop_record()? else {
                return Ok(self.pending.take());
            };
            match &mut self.pending {
                Some(pending) if pending.key == record.key => pending.add(&record.values),
                Some(_) => {
                    let completed = self.pending.replace(VoxelAccumulator::from_record(record));
                    return Ok(completed);
                }
                None => self.pending = Some(VoxelAccumulator::from_record(record)),
            }
        }
    }
}

impl BoundedSpatialRecordSource for StreamingVoxelSource {
    fn schema(&self) -> &SchemaDescriptor {
        &self.schema
    }

    fn options(&self) -> &StreamOptions {
        &self.options
    }

    fn memory_tracker(&self) -> &MemoryTracker {
        &self.tracker
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    fn max_chunk_bytes(&self) -> u64 {
        self.max_chunk_bytes
    }

    fn next_chunk(&mut self) -> Option<RecordsResult<SpatialRecordChunk>> {
        if self.finished {
            return None;
        }
        if let Err(error) = self.cancellation.check() {
            return Some(Err(error));
        }
        let reservation = match self.tracker.try_reserve(self.max_chunk_bytes) {
            Ok(reservation) => reservation,
            Err(error) => return Some(Err(error)),
        };
        let mut buffers = empty_buffers(self.schema.point_schema(), self.options.chunk_points());
        let mut output_points = 0;
        while output_points < self.options.chunk_points() {
            let accumulator = match self.next_accumulator() {
                Ok(Some(accumulator)) => accumulator,
                Ok(None) => {
                    self.finished = true;
                    break;
                }
                Err(error) => return Some(Err(error)),
            };
            for (field_index, field) in self.schema.point_schema().fields().iter().enumerate() {
                let value = accumulator.sums[field_index].total() / accumulator.count as f64;
                let Some(buffer) = buffers.get_mut(&field.name) else {
                    return Some(Err(RecordsError::InvalidChunk(format!(
                        "voxel output buffer missing for field `{}`",
                        field.name
                    ))));
                };
                if let Err(error) = push_scalar(buffer, value) {
                    return Some(Err(error));
                }
            }
            output_points += 1;
        }
        if output_points == 0 {
            return None;
        }
        let cloud = match PointCloud::try_from_parts(
            self.schema.point_schema().clone(),
            buffers,
            self.metadata.clone(),
        ) {
            Ok(cloud) => cloud,
            Err(error) => return Some(Err(error.into())),
        };
        let identity =
            ChunkIdentity { sequence: self.next_sequence, point_offset: self.next_point_offset };
        let record = match SpatialRecord::try_new(self.schema.clone(), cloud) {
            Ok(record) => record,
            Err(error) => return Some(Err(error)),
        };
        let next_sequence = match self.next_sequence.checked_add(1) {
            Some(value) => value,
            None => {
                return Some(Err(RecordsError::ReceiptOverflow("voxel output sequence".into())));
            }
        };
        let output_points_u64 = match usize_u64(output_points, "voxel output chunk point count") {
            Ok(value) => value,
            Err(error) => return Some(Err(error)),
        };
        let next_point_offset = match self.next_point_offset.checked_add(output_points_u64) {
            Some(value) => value,
            None => {
                return Some(Err(RecordsError::ReceiptOverflow(
                    "voxel output point offset".into(),
                )));
            }
        };
        let chunk = SpatialRecordChunk::try_from_reserved(identity, record, reservation);
        if chunk.is_ok() {
            self.next_sequence = next_sequence;
            self.next_point_offset = next_point_offset;
        }
        Some(chunk)
    }
}

#[derive(Debug)]
struct RunBuffer {
    keys: Vec<[i64; 3]>,
    source_indices: Vec<u64>,
    values: Vec<f64>,
    order: Vec<usize>,
    field_count: usize,
}

impl RunBuffer {
    fn with_capacity(points: usize, field_count: usize) -> Self {
        Self {
            keys: Vec::with_capacity(points),
            source_indices: Vec::with_capacity(points),
            values: Vec::with_capacity(points.saturating_mul(field_count)),
            order: Vec::with_capacity(points),
            field_count,
        }
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    fn push(
        &mut self,
        key: [i64; 3],
        source_index: u64,
        cloud: &PointCloud,
        point_index: usize,
    ) -> RecordsResult<()> {
        self.keys.push(key);
        self.source_indices.push(source_index);
        self.order.push(self.order.len());
        for field in cloud.schema().fields() {
            self.values.push(scalar_at(cloud.field(&field.name)?, point_index)?);
        }
        Ok(())
    }

    fn clear(&mut self) {
        self.keys.clear();
        self.source_indices.clear();
        self.values.clear();
        self.order.clear();
    }
}

#[derive(Clone, Copy, Debug)]
struct RunMeta {
    data_offset: u64,
    records: u64,
}

fn flush_run(
    spool: &mut BoundedSpool,
    run: &mut RunBuffer,
    metas: &mut Vec<RunMeta>,
    max_runs: usize,
) -> RecordsResult<()> {
    if metas.len() == max_runs {
        return Err(RecordsError::InvalidConfiguration(format!(
            "streaming voxel run limit {max_runs} exceeded"
        )));
    }
    run.order.sort_unstable_by_key(|index| (run.keys[*index], run.source_indices[*index]));
    let records = usize_u64(run.len(), "voxel run record count")?;
    spool.write_all(&records.to_le_bytes()).map_err(spool_error)?;
    let data_offset = spool.stream_position().map_err(spool_error)?;
    for &index in &run.order {
        for coordinate in run.keys[index] {
            spool.write_all(&coordinate.to_le_bytes()).map_err(spool_error)?;
        }
        spool.write_all(&run.source_indices[index].to_le_bytes()).map_err(spool_error)?;
        let start = index * run.field_count;
        for value in &run.values[start..start + run.field_count] {
            spool.write_all(&value.to_le_bytes()).map_err(spool_error)?;
        }
    }
    metas.push(RunMeta { data_offset, records });
    run.clear();
    Ok(())
}

#[derive(Debug)]
struct RunCursor {
    file: File,
    remaining: u64,
    field_count: usize,
}

impl RunCursor {
    fn open(path: &std::path::Path, meta: RunMeta, field_count: usize) -> RecordsResult<Self> {
        let mut file = File::open(path).map_err(spool_error)?;
        file.seek(SeekFrom::Start(meta.data_offset)).map_err(spool_error)?;
        Ok(Self { file, remaining: meta.records, field_count })
    }

    fn next_record(&mut self) -> RecordsResult<Option<SpillRecord>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let mut key = [0_i64; 3];
        for coordinate in &mut key {
            *coordinate = read_i64(&mut self.file)?;
        }
        let source_index = read_u64(&mut self.file)?;
        let mut values = Vec::with_capacity(self.field_count);
        for _ in 0..self.field_count {
            values.push(read_f64(&mut self.file)?);
        }
        self.remaining -= 1;
        Ok(Some(SpillRecord { key, source_index, values }))
    }
}

#[derive(Debug)]
struct SpillRecord {
    key: [i64; 3],
    source_index: u64,
    values: Vec<f64>,
}

#[derive(Debug)]
struct HeapRecord {
    run_index: usize,
    record: SpillRecord,
}

impl PartialEq for HeapRecord {
    fn eq(&self, other: &Self) -> bool {
        (self.record.key, self.record.source_index, self.run_index)
            == (other.record.key, other.record.source_index, other.run_index)
    }
}

impl Eq for HeapRecord {}

impl PartialOrd for HeapRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapRecord {
    fn cmp(&self, other: &Self) -> Ordering {
        (other.record.key, other.record.source_index, other.run_index).cmp(&(
            self.record.key,
            self.record.source_index,
            self.run_index,
        ))
    }
}

#[derive(Debug)]
struct VoxelAccumulator {
    key: [i64; 3],
    count: u64,
    sums: Vec<CompensatedSum>,
}

impl VoxelAccumulator {
    fn from_record(record: SpillRecord) -> Self {
        let mut sums = vec![CompensatedSum::default(); record.values.len()];
        for (sum, value) in sums.iter_mut().zip(record.values) {
            sum.add(value);
        }
        Self { key: record.key, count: 1, sums }
    }

    fn add(&mut self, values: &[f64]) {
        self.count += 1;
        for (sum, value) in self.sums.iter_mut().zip(values) {
            sum.add(*value);
        }
    }
}

fn positions(cloud: &PointCloud) -> RecordsResult<(&[f32], &[f32], &[f32])> {
    let x = cloud
        .schema()
        .find_semantic(FieldSemantic::PositionX)
        .ok_or_else(|| RecordsError::MissingField("PositionX".into()))?;
    let y = cloud
        .schema()
        .find_semantic(FieldSemantic::PositionY)
        .ok_or_else(|| RecordsError::MissingField("PositionY".into()))?;
    let z = cloud
        .schema()
        .find_semantic(FieldSemantic::PositionZ)
        .ok_or_else(|| RecordsError::MissingField("PositionZ".into()))?;
    Ok((
        cloud.field(&x.name)?.as_f32()?,
        cloud.field(&y.name)?.as_f32()?,
        cloud.field(&z.name)?.as_f32()?,
    ))
}

type NormalColumns<'a> = Option<(&'a [f32], &'a [f32], &'a [f32])>;

fn normal_columns(cloud: &PointCloud) -> RecordsResult<NormalColumns<'_>> {
    let fields = [
        cloud.schema().find_semantic(FieldSemantic::NormalX),
        cloud.schema().find_semantic(FieldSemantic::NormalY),
        cloud.schema().find_semantic(FieldSemantic::NormalZ),
    ];
    if fields.iter().all(Option::is_none) {
        return Ok(None);
    }
    let [Some(nx), Some(ny), Some(nz)] = fields else {
        return Err(RecordsError::InvalidChunk(
            "normal fields must be present as a complete XYZ triplet".into(),
        ));
    };
    Ok(Some((
        cloud.field(&nx.name)?.as_f32()?,
        cloud.field(&ny.name)?.as_f32()?,
        cloud.field(&nz.name)?.as_f32()?,
    )))
}

fn empty_buffers(schema: &PointSchema, capacity: usize) -> PointBufferSet {
    let mut buffers = PointBufferSet::new();
    for field in schema.fields() {
        buffers.insert(field.name.clone(), PointBuffer::with_capacity(field.dtype, capacity));
    }
    buffers
}

fn scalar_at(buffer: &PointBuffer, index: usize) -> RecordsResult<f64> {
    Ok(match buffer {
        PointBuffer::F32(values) => f64::from(values[index]),
        PointBuffer::F64(values) => values[index],
        PointBuffer::U8(values) => f64::from(values[index]),
        PointBuffer::U16(values) => f64::from(values[index]),
        PointBuffer::U32(values) => f64::from(values[index]),
        PointBuffer::I32(values) => f64::from(values[index]),
    })
}

fn push_scalar(buffer: &mut PointBuffer, value: f64) -> RecordsResult<()> {
    match buffer {
        PointBuffer::F32(values) => values.push(value as f32),
        PointBuffer::F64(values) => values.push(value),
        PointBuffer::U8(values) => values.push(value.round().clamp(0.0, u8::MAX as f64) as u8),
        PointBuffer::U16(values) => {
            values.push(value.round().clamp(0.0, u16::MAX as f64) as u16);
        }
        PointBuffer::U32(values) => {
            values.push(value.round().clamp(0.0, u32::MAX as f64) as u32);
        }
        PointBuffer::I32(values) => {
            values.push(value.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32);
        }
    }
    Ok(())
}

fn schema_bytes(schema: &SchemaDescriptor, points: usize) -> RecordsResult<u64> {
    let bytes_per_point = schema.point_schema().fields().iter().try_fold(0_u64, |sum, field| {
        sum.checked_add(match field.dtype {
            DType::F32 | DType::F16 | DType::U32 | DType::I32 => 4,
            DType::F64 => 8,
            DType::U8 => 1,
            DType::U16 => 2,
        })
        .ok_or_else(|| RecordsError::InvalidConfiguration("schema byte width overflow".into()))
    })?;
    bytes_per_point
        .checked_mul(usize_u64(points, "schema point capacity")?)
        .ok_or_else(|| RecordsError::InvalidConfiguration("chunk byte size overflow".into()))
}

fn run_memory_bytes(points: usize, fields: usize, max_runs: usize) -> RecordsResult<u64> {
    let point_buffers = fields
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(40))
        .and_then(|bytes| bytes.checked_mul(points))
        .ok_or_else(|| RecordsError::InvalidConfiguration("voxel run memory overflow".into()))?;
    let bytes = max_runs
        .checked_mul(std::mem::size_of::<RunMeta>())
        .and_then(|run_meta_bytes| run_meta_bytes.checked_add(point_buffers))
        .ok_or_else(|| RecordsError::InvalidConfiguration("voxel run memory overflow".into()))?;
    usize_u64(bytes, "voxel run memory bytes")
}

fn merge_memory_bytes(runs: usize, fields: usize) -> RecordsResult<u64> {
    let record_bytes = fields
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(96))
        .ok_or_else(|| RecordsError::InvalidConfiguration("voxel merge memory overflow".into()))?;
    let heap_bytes = runs
        .checked_add(2)
        .and_then(|records| records.checked_mul(record_bytes))
        .and_then(|bytes| bytes.checked_add(fields.saturating_mul(16)))
        .ok_or_else(|| RecordsError::InvalidConfiguration("voxel merge memory overflow".into()))?;
    usize_u64(heap_bytes, "voxel merge memory bytes")
}

fn usize_u64(value: usize, context: &str) -> RecordsResult<u64> {
    u64::try_from(value)
        .map_err(|_| RecordsError::InvalidConfiguration(format!("{context} does not fit u64")))
}

fn voxel_coordinate(value: f64, leaf: f64) -> RecordsResult<i64> {
    let coordinate = (value / leaf).floor();
    if coordinate < i64::MIN as f64 || coordinate > i64::MAX as f64 {
        return Err(RecordsError::InvalidChunk(
            "voxel coordinate exceeds signed 64-bit range".into(),
        ));
    }
    Ok(coordinate as i64)
}

fn spool_error(error: std::io::Error) -> RecordsError {
    RecordsError::InvalidChunk(format!("voxel spool I/O failed: {error}"))
}

fn read_i64(reader: &mut impl Read) -> RecordsResult<i64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes).map_err(spool_error)?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> RecordsResult<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes).map_err(spool_error)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f64(reader: &mut impl Read) -> RecordsResult<f64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes).map_err(spool_error)?;
    Ok(f64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::{reduce_positions, ChunkMapSource, StreamingVoxelConfig, StreamingVoxelSource};
    use spatialrust_core::{
        HasIntensity, HasPositions3, PointCloud, PointCloudBuilder, StandardSchemas,
    };
    use spatialrust_io::SpoolOptions;
    use spatialrust_math::{Mat3, Mat4, Vec3};
    use spatialrust_records::{
        BoundedSpatialRecordSource, CancellationToken, MemoryBudget, RecyclingMemoryChunkSource,
        SchemaDescriptor, SchemaVersion, StreamOptions,
    };

    fn cloud(points: &[[f32; 3]]) -> PointCloud {
        let mut builder = PointCloudBuilder::xyz();
        for point in points {
            builder.push_point(*point).unwrap();
        }
        builder.build().unwrap()
    }

    fn source(cloud: PointCloud, chunk_points: usize) -> RecyclingMemoryChunkSource {
        let schema = SchemaDescriptor::try_new(
            "stream.xyz",
            SchemaVersion::new(1, 0),
            cloud.schema().clone(),
        )
        .unwrap();
        let options =
            StreamOptions::new(chunk_points, MemoryBudget::new(16 * 1024).unwrap()).unwrap();
        RecyclingMemoryChunkSource::try_new(schema, cloud, options, CancellationToken::default())
            .unwrap()
    }

    fn collect_positions(source: &mut impl BoundedSpatialRecordSource) -> Vec<[f32; 3]> {
        let mut output = Vec::new();
        while let Some(chunk) = source.next_chunk() {
            let chunk = chunk.unwrap();
            let (x, y, z) = chunk.record().cloud().positions3().unwrap();
            output.extend((0..x.len()).map(|index| [x[index], y[index], z[index]]));
        }
        output
    }

    #[test]
    fn crop_and_transform_recompute_contiguous_chunk_identity() {
        let input = cloud(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]);
        let cropped =
            ChunkMapSource::crop(source(input, 2), [0.5, -1.0, -1.0], [2.5, 1.0, 1.0], false)
                .unwrap();
        let transform = Mat4::<f32>::from_rotation_translation(
            Mat3::<f32>::identity(),
            Vec3::new(10.0, 0.0, 0.0),
        );
        let mut transformed = ChunkMapSource::transform(cropped, transform).unwrap();
        let first = transformed.next_chunk().unwrap().unwrap();
        assert_eq!(first.identity().sequence, 0);
        assert_eq!(first.identity().point_offset, 0);
        assert_eq!(first.record().cloud().positions3().unwrap().0, &[11.0]);
        drop(first);
        let second = transformed.next_chunk().unwrap().unwrap();
        assert_eq!(second.identity().sequence, 1);
        assert_eq!(second.identity().point_offset, 1);
        assert_eq!(second.record().cloud().positions3().unwrap().0, &[12.0]);
        drop(second);
        assert!(transformed.next_chunk().is_none());
        assert!(transformed.memory_tracker().snapshot().peak_bytes <= 16 * 1024);
    }

    #[test]
    fn global_reduction_ignores_non_finite_positions() {
        let input = cloud(&[[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [f32::NAN, 0.0, 0.0]]);
        let mut source = source(input, 1);
        let reduction = reduce_positions(&mut source).unwrap();
        assert_eq!(reduction.point_count, 3);
        assert_eq!(reduction.finite_point_count, 2);
        assert_eq!(reduction.bounds, Some(([0.0, 1.0, 2.0], [2.0, 3.0, 4.0])));
        assert_eq!(reduction.centroid, Some([1.0, 2.0, 3.0]));
    }

    #[test]
    fn voxel_output_is_identical_across_input_chunk_and_run_sizes() {
        let input = cloud(&[
            [1.3, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [2.8, 0.0, 0.0],
        ]);
        let config_a = StreamingVoxelConfig::new(
            1.0,
            2,
            8,
            SpoolOptions::new(std::env::temp_dir(), 16 * 1024).unwrap(),
        )
        .unwrap();
        let config_b = StreamingVoxelConfig::new(
            1.0,
            3,
            8,
            SpoolOptions::new(std::env::temp_dir(), 16 * 1024).unwrap(),
        )
        .unwrap();
        let mut output_a =
            StreamingVoxelSource::try_build(source(input.clone(), 1), config_a).unwrap();
        let mut output_b = StreamingVoxelSource::try_build(source(input, 4), config_b).unwrap();
        let points_a = collect_positions(&mut output_a);
        let points_b = collect_positions(&mut output_b);
        assert!(output_a.spool_bytes() > 0);
        assert!(output_a.run_count() > 1);
        assert_eq!(points_a, points_b);
        assert_eq!(points_a, vec![[0.15, 0.0, 0.0], [1.2, 0.0, 0.0], [2.8, 0.0, 0.0]]);
    }

    #[test]
    fn voxel_centroid_averages_attributes_across_runs() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzi());
        builder.push_point([0.1, 0.0, 0.0, 10.0]).unwrap();
        builder.push_point([0.2, 0.0, 0.0, 20.0]).unwrap();
        builder.push_point([1.1, 0.0, 0.0, 50.0]).unwrap();
        let input = builder.build().unwrap();
        let config = StreamingVoxelConfig::new(
            1.0,
            1,
            4,
            SpoolOptions::new(std::env::temp_dir(), 4096).unwrap(),
        )
        .unwrap();
        let mut output = StreamingVoxelSource::try_build(source(input, 2), config).unwrap();
        let first = output.next_chunk().unwrap().unwrap();
        assert_eq!(first.record().cloud().intensity().unwrap(), &[15.0, 50.0]);
    }

    #[test]
    fn voxel_spool_and_run_limits_fail_closed() {
        let input = cloud(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let tiny_spool = StreamingVoxelConfig::new(
            0.1,
            2,
            2,
            SpoolOptions::new(std::env::temp_dir(), 8).unwrap(),
        )
        .unwrap();
        assert!(StreamingVoxelSource::try_build(source(input.clone(), 2), tiny_spool).is_err());

        let one_run = StreamingVoxelConfig::new(
            0.1,
            1,
            1,
            SpoolOptions::new(std::env::temp_dir(), 4096).unwrap(),
        )
        .unwrap();
        assert!(StreamingVoxelSource::try_build(source(input, 2), one_run).is_err());
    }

    #[test]
    fn cancelled_voxel_build_releases_all_tracked_memory() {
        let input = cloud(&[[0.0, 0.0, 0.0]]);
        let schema = SchemaDescriptor::try_new(
            "stream.xyz",
            SchemaVersion::new(1, 0),
            input.schema().clone(),
        )
        .unwrap();
        let options = StreamOptions::new(1, MemoryBudget::new(4096).unwrap()).unwrap();
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let source =
            RecyclingMemoryChunkSource::try_new(schema, input, options, cancellation).unwrap();
        let tracker = source.memory_tracker().clone();
        let config = StreamingVoxelConfig::new(
            1.0,
            1,
            2,
            SpoolOptions::new(std::env::temp_dir(), 4096).unwrap(),
        )
        .unwrap();
        assert!(StreamingVoxelSource::try_build(source, config).is_err());
        assert_eq!(tracker.snapshot().current_bytes, 0);
    }
}
