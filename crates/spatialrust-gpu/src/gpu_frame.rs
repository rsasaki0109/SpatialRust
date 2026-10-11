//! GPU-resident spatial frame ownership and chained execution.

use spatialrust_core::{
    PointSchema, SpatialError, SpatialResult, SpatialTensor, TransferDirection, TransferStats,
};

use crate::aoso_staging::runtime_device_key;
use crate::{
    reduce_voxel_attributes_aoso_chunks, upload_spatial_tensor_xyz_chunks,
    AoSoAAttributeAggregation, AoSoAAttributeReduction, GpuAoSoAttributeChunk, GpuAoSoNormals,
    GpuAoSoRadiusGrid, GpuAoSoXyzBuffer, GpuVoxelSegments, WgpuRuntime,
};

/// GPU frame capabilities available to downstream algorithms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuFrameCapability {
    /// Interleaved XYZ positions.
    Positions,
    /// Per-point normals and curvature.
    Normals,
    /// Interleaved point attributes.
    Attributes,
    /// Voxel partition metadata.
    VoxelSegments,
    /// Sparse uniform radius grid.
    RadiusGrid,
}

/// Transfer and logical-stage receipt for one GPU frame.
///
/// Counts point/attribute payload copies and scalar metadata readbacks performed
/// by frame methods, including construction. Kernel parameter uniforms and
/// shader storage accesses are excluded. Transfers performed outside frame
/// methods (for example, uploads before attaching attributes or direct buffer
/// readbacks) remain the caller's responsibility.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GpuExecutionReceipt {
    transfers: TransferStats,
    stages: Vec<&'static str>,
}

impl GpuExecutionReceipt {
    /// Returns the common transfer accounting for this execution.
    #[must_use]
    pub const fn transfer_stats(&self) -> TransferStats {
        self.transfers
    }

    /// Returns bytes uploaded from host memory.
    #[must_use]
    pub const fn host_to_device_bytes(&self) -> u64 {
        self.transfers.host_to_device_bytes()
    }

    /// Returns bytes copied between GPU buffers.
    #[must_use]
    pub const fn gpu_to_gpu_bytes(&self) -> u64 {
        self.transfers.device_to_device_bytes()
    }

    /// Returns payload and scalar metadata bytes read back to host memory.
    #[must_use]
    pub const fn device_to_host_bytes(&self) -> u64 {
        self.transfers.device_to_host_bytes()
    }

    /// Returns the logical GPU stages recorded by the high-level pipeline.
    #[must_use]
    pub fn stages(&self) -> &[&'static str] {
        &self.stages
    }
}

/// Owned GPU-resident point frame with explicit schema and device identity.
pub struct GpuSpatialFrame {
    schema: PointSchema,
    device_key: usize,
    positions: GpuAoSoXyzBuffer,
    normals: Option<GpuAoSoNormals>,
    attributes: Vec<GpuAoSoAttributeChunk>,
    voxel_segments: Option<GpuVoxelSegments>,
    radius_grid: Option<GpuAoSoRadiusGrid>,
    receipt: GpuExecutionReceipt,
}

impl GpuSpatialFrame {
    /// Creates a frame owning interleaved positions on `runtime`.
    pub fn new(
        runtime: &WgpuRuntime,
        schema: PointSchema,
        positions: GpuAoSoXyzBuffer,
    ) -> SpatialResult<Self> {
        let device_key = runtime_device_key(runtime);
        if positions.device_key() != device_key {
            return Err(SpatialError::InvalidArgument(
                "position buffer belongs to a different runtime device".to_owned(),
            ));
        }
        Ok(Self {
            schema,
            device_key,
            positions,
            normals: None,
            attributes: Vec::new(),
            voxel_segments: None,
            radius_grid: None,
            receipt: GpuExecutionReceipt::default(),
        })
    }

    /// Returns the source point schema.
    #[must_use]
    pub const fn schema(&self) -> &PointSchema {
        &self.schema
    }

    /// Returns the number of source points.
    #[must_use]
    pub const fn point_count(&self) -> usize {
        self.positions.point_count()
    }

    /// Returns retained interleaved positions.
    #[must_use]
    pub const fn positions(&self) -> &GpuAoSoXyzBuffer {
        &self.positions
    }

    /// Returns retained normals when attached.
    #[must_use]
    pub const fn normals(&self) -> Option<&GpuAoSoNormals> {
        self.normals.as_ref()
    }

    /// Returns retained voxel segments when attached.
    #[must_use]
    pub const fn voxel_segments(&self) -> Option<&GpuVoxelSegments> {
        self.voxel_segments.as_ref()
    }

    /// Returns retained radius grid when attached.
    #[must_use]
    pub const fn radius_grid(&self) -> Option<&GpuAoSoRadiusGrid> {
        self.radius_grid.as_ref()
    }

    /// Returns execution and transfer accounting.
    #[must_use]
    pub const fn receipt(&self) -> &GpuExecutionReceipt {
        &self.receipt
    }

    /// Returns whether a capability is currently attached.
    #[must_use]
    pub fn has_capability(&self, capability: GpuFrameCapability) -> bool {
        match capability {
            GpuFrameCapability::Positions => true,
            GpuFrameCapability::Normals => self.normals.is_some(),
            GpuFrameCapability::Attributes => !self.attributes.is_empty(),
            GpuFrameCapability::VoxelSegments => self.voxel_segments.is_some(),
            GpuFrameCapability::RadiusGrid => self.radius_grid.is_some(),
        }
    }

    /// Verifies that `runtime` owns the frame's buffers.
    pub fn validate_runtime(&self, runtime: &WgpuRuntime) -> SpatialResult<()> {
        if self.device_key != runtime_device_key(runtime) {
            return Err(SpatialError::InvalidArgument(
                "GPU frame belongs to a different runtime device".to_owned(),
            ));
        }
        Ok(())
    }

    /// Attaches per-point normal output after validating length and device.
    pub fn attach_normals(
        &mut self,
        runtime: &WgpuRuntime,
        normals: GpuAoSoNormals,
    ) -> SpatialResult<()> {
        self.validate_runtime(runtime)?;
        if normals.device_key() != self.device_key {
            return Err(SpatialError::InvalidArgument(
                "normal buffer belongs to a different runtime device".to_owned(),
            ));
        }
        if normals.point_count() != self.point_count() {
            return Err(SpatialError::BufferLengthMismatch {
                expected: self.point_count(),
                found: normals.point_count(),
            });
        }
        if let Some(previous) = self.normals.replace(normals) {
            previous.recycle(runtime);
        }
        Ok(())
    }

    /// Attaches voxel segments after validating their source point count.
    pub fn attach_voxel_segments(&mut self, segments: GpuVoxelSegments) -> SpatialResult<()> {
        if segments.point_count() as usize != self.point_count() {
            return Err(SpatialError::BufferLengthMismatch {
                expected: self.point_count(),
                found: segments.point_count() as usize,
            });
        }
        self.voxel_segments = Some(segments);
        Ok(())
    }

    /// Attaches a sparse radius grid after validating its point count.
    pub fn attach_radius_grid(&mut self, grid: GpuAoSoRadiusGrid) -> SpatialResult<()> {
        if grid.segments().point_count() as usize != self.point_count() {
            return Err(SpatialError::BufferLengthMismatch {
                expected: self.point_count(),
                found: grid.segments().point_count() as usize,
            });
        }
        self.radius_grid = Some(grid);
        Ok(())
    }

    /// Attaches interleaved attribute chunks after validating total length.
    pub fn attach_attributes(
        &mut self,
        runtime: &WgpuRuntime,
        attributes: Vec<GpuAoSoAttributeChunk>,
    ) -> SpatialResult<()> {
        self.validate_runtime(runtime)?;
        if attributes.iter().any(|attribute| attribute.device_key() != self.device_key) {
            return Err(SpatialError::InvalidArgument(
                "attribute buffer belongs to a different runtime device".to_owned(),
            ));
        }
        let found = attributes.iter().map(GpuAoSoAttributeChunk::point_count).sum();
        if found != self.point_count() {
            return Err(SpatialError::BufferLengthMismatch { expected: self.point_count(), found });
        }
        for previous in std::mem::replace(&mut self.attributes, attributes) {
            previous.recycle(runtime);
        }
        Ok(())
    }

    /// Rebuilds the sparse radius grid from retained positions.
    pub fn rebuild_radius_grid(&mut self, runtime: &WgpuRuntime, radius: f32) -> SpatialResult<()> {
        self.rebuild_radius_grid_with_profiler(runtime, radius, None)
    }

    fn rebuild_radius_grid_with_profiler(
        &mut self,
        runtime: &WgpuRuntime,
        radius: f32,
        profiler: Option<&mut crate::GpuStageProfiler>,
    ) -> SpatialResult<()> {
        self.validate_runtime(runtime)?;
        let grid = crate::aoso_staging::build_radius_grid_aoso_gpu_with_profiler(
            runtime,
            &self.positions,
            radius,
            profiler,
        )?;
        self.receipt
            .transfers
            .record(TransferDirection::DeviceToHost, grid.segments().construction_readback_bytes());
        self.radius_grid = Some(grid);
        self.receipt.stages.push("radius-grid");
        Ok(())
    }

    /// Estimates normals using a cached matching grid or rebuilds it first.
    pub fn estimate_normals(&mut self, runtime: &WgpuRuntime, radius: f32) -> SpatialResult<()> {
        self.estimate_normals_with_profiler(runtime, radius, None)
    }

    fn estimate_normals_with_profiler(
        &mut self,
        runtime: &WgpuRuntime,
        radius: f32,
        mut profiler: Option<&mut crate::GpuStageProfiler>,
    ) -> SpatialResult<()> {
        self.validate_runtime(runtime)?;
        let matches = self.radius_grid.as_ref().is_some_and(|grid| grid.radius() == radius);
        if !matches {
            self.rebuild_radius_grid_with_profiler(runtime, radius, profiler.as_deref_mut())?;
        }
        let grid = self.radius_grid.as_ref().ok_or_else(|| {
            SpatialError::InvalidArgument("radius grid is unavailable".to_owned())
        })?;
        let normals = crate::aoso_staging::estimate_normals_radius_grid_aoso_gpu_with_profiler(
            runtime,
            &self.positions,
            grid,
            profiler,
        )?;
        self.attach_normals(runtime, normals)?;
        self.receipt.stages.push("radius-normals");
        Ok(())
    }

    /// Reduces attached interleaved attributes using retained voxel segments.
    pub fn reduce_attributes(
        &mut self,
        runtime: &WgpuRuntime,
        aggregation: AoSoAAttributeAggregation,
    ) -> SpatialResult<AoSoAAttributeReduction> {
        self.validate_runtime(runtime)?;
        if self.attributes.is_empty() {
            return Err(SpatialError::InvalidArgument(
                "GPU frame has no attached attributes".to_owned(),
            ));
        }
        let segments = self.voxel_segments.as_ref().ok_or_else(|| {
            SpatialError::InvalidArgument("GPU frame has no voxel segments".to_owned())
        })?;
        let reduced =
            reduce_voxel_attributes_aoso_chunks(runtime, &self.attributes, segments, aggregation)?;
        if !reduced.is_empty() {
            self.receipt.transfers.record(
                TransferDirection::DeviceToDevice,
                self.attributes.iter().map(GpuAoSoAttributeChunk::byte_len).sum(),
            );
        }
        self.receipt.transfers.record(
            TransferDirection::DeviceToHost,
            (reduced.len() * reduced.layout().stride_f32() * std::mem::size_of::<f32>()) as u64,
        );
        self.receipt.stages.push("attribute-reduce");
        Ok(reduced)
    }

    /// Explicitly reads interleaved positions back to CPU memory.
    pub fn readback_positions(&mut self, runtime: &WgpuRuntime) -> SpatialResult<Vec<[f32; 3]>> {
        self.validate_runtime(runtime)?;
        let byte_len = self.positions.point_count() * 3 * std::mem::size_of::<f32>();
        if byte_len == 0 {
            return Ok(Vec::new());
        }
        let device = runtime.device();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-frame-position-readback"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu-frame-position-readback-encoder"),
        });
        encoder.copy_buffer_to_buffer(self.positions.buffer(), 0, &staging, 0, byte_len as u64);
        runtime.queue().submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| SpatialError::InvalidArgument("failed to receive map result".to_owned()))?
            .map_err(|error| SpatialError::InvalidArgument(format!("map failed: {error}")))?;
        let mapped = slice.get_mapped_range();
        let values: &[f32] = bytemuck::cast_slice(&mapped);
        let positions = values.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        drop(mapped);
        staging.unmap();
        self.receipt.transfers.record(TransferDirection::DeviceToHost, byte_len as u64);
        Ok(positions)
    }

    /// Recycles pooled frame buffers after validating the runtime device.
    pub fn recycle(self, runtime: &WgpuRuntime) -> SpatialResult<()> {
        self.validate_runtime(runtime)?;
        self.positions.recycle(runtime);
        if let Some(normals) = self.normals {
            normals.recycle(runtime);
        }
        for attribute in self.attributes {
            attribute.recycle(runtime);
        }
        Ok(())
    }
}

/// Uploads a tensor and chains global voxel partitioning and radius normals.
///
/// Retains every source point, without computing or reading back voxel centroids.
/// A nonempty frame synchronizes two four-byte cell counts during partition/grid
/// construction; these metadata transfers are included in its receipt. Position
/// and normal arrays remain on the device until explicitly read back.
pub fn run_aoso_voxel_normal_frame(
    runtime: &WgpuRuntime,
    tensor: &SpatialTensor<'_>,
    origin: [f32; 3],
    inv_leaf: f32,
    normal_radius: f32,
) -> SpatialResult<GpuSpatialFrame> {
    run_aoso_voxel_normal_frame_with_profiler(
        runtime,
        tensor,
        origin,
        inv_leaf,
        normal_radius,
        None,
    )
}

/// Runs a frame with pass timestamps for both partitions and radius normals.
///
/// Query readback requires an explicit profiler collection and is separate
/// from the frame's transfer receipt. Capacity is checked before any upload.
#[allow(clippy::too_many_arguments)]
pub fn run_aoso_voxel_normal_frame_profiled(
    runtime: &WgpuRuntime,
    tensor: &SpatialTensor<'_>,
    origin: [f32; 3],
    inv_leaf: f32,
    normal_radius: f32,
    profiler: &mut crate::GpuStageProfiler,
) -> SpatialResult<GpuSpatialFrame> {
    let n = u32::try_from(tensor.len()).map_err(|_| {
        SpatialError::InvalidArgument("AoSoA positions exceed the GPU point limit".to_owned())
    })?;
    let stages = if n == 0 {
        0
    } else {
        // Keys + entry build + radix (24 digits * 3 phases) + compaction
        // (mark/init/write and ceil(log2(n)) scan passes), for two grids.
        let scan = u32::BITS - (n - 1).leading_zeros();
        2 * (5 + if n > 1 { 72 } else { 0 } + scan) + 1
    };
    profiler.validate(runtime, stages)?;
    let result = run_aoso_voxel_normal_frame_with_profiler(
        runtime,
        tensor,
        origin,
        inv_leaf,
        normal_radius,
        Some(profiler),
    );
    profiler.finish_recording(&result);
    result
}

pub(crate) fn run_aoso_voxel_normal_frame_with_profiler(
    runtime: &WgpuRuntime,
    tensor: &SpatialTensor<'_>,
    origin: [f32; 3],
    inv_leaf: f32,
    normal_radius: f32,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
) -> SpatialResult<GpuSpatialFrame> {
    let chunks = upload_spatial_tensor_xyz_chunks(runtime, tensor)?;
    let upload_bytes = chunks
        .iter()
        .map(|chunk| (chunk.point_count() * 3 * std::mem::size_of::<f32>()) as u64)
        .sum();
    let partition = crate::aoso_staging::build_voxel_segments_aoso_chunks_with_profiler(
        runtime,
        &chunks,
        origin,
        inv_leaf,
        profiler.as_deref_mut(),
    );
    for chunk in chunks {
        chunk.recycle(runtime);
    }
    let (positions, segments) = partition?;
    let mut frame = GpuSpatialFrame::new(runtime, tensor.schema().clone(), positions)?;
    frame
        .receipt
        .transfers
        .record(TransferDirection::DeviceToHost, segments.construction_readback_bytes());
    frame.attach_voxel_segments(segments)?;
    frame.receipt.transfers.record(TransferDirection::HostToDevice, upload_bytes);
    frame.receipt.transfers.record(TransferDirection::DeviceToDevice, upload_bytes);
    frame.receipt.stages = vec!["upload", "voxel-segments"];
    frame.estimate_normals_with_profiler(runtime, normal_radius, profiler)?;
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::{run_aoso_voxel_normal_frame, GpuFrameCapability};
    use crate::{
        build_voxel_segments, upload_spatial_tensor_attribute_chunks, AoSoAAttributeAggregation,
        WgpuRuntime,
    };
    use spatialrust_core::{
        AoSoAAttributeLayout, PointBuffer, PointBufferSet, PointCloud, PointCloudBuilder,
        SpatialTensor, StandardSchemas,
    };

    #[test]
    fn chained_frame_owns_capabilities_and_tracks_readback() {
        let mut builder = PointCloudBuilder::xyz();
        let mut expected = Vec::new();
        for row in 0..8 {
            for column in 0..8 {
                let point = [column as f32 * 0.1, row as f32 * 0.1, 0.0];
                builder.push_point(point).unwrap();
                expected.push(point);
            }
        }
        let cloud = builder.build().unwrap();
        let tensor = SpatialTensor::new(&cloud, 13).unwrap();
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");

        let mut frame =
            run_aoso_voxel_normal_frame(&runtime, &tensor, [0.0; 3], 4.0, 0.25).unwrap();
        assert_eq!(frame.point_count(), expected.len());
        assert!(frame.has_capability(GpuFrameCapability::Positions));
        assert!(frame.has_capability(GpuFrameCapability::Normals));
        assert!(frame.has_capability(GpuFrameCapability::VoxelSegments));
        assert!(frame.has_capability(GpuFrameCapability::RadiusGrid));
        assert!(!frame.has_capability(GpuFrameCapability::Attributes));
        assert_eq!(frame.normals().unwrap().point_count(), expected.len());
        assert_eq!(
            frame.receipt().stages(),
            &["upload", "voxel-segments", "radius-grid", "radius-normals"]
        );
        assert_eq!(frame.receipt().host_to_device_bytes(), (expected.len() * 3 * 4) as u64);
        assert_eq!(frame.receipt().gpu_to_gpu_bytes(), (expected.len() * 3 * 4) as u64);
        // Only the voxel and radius-grid cell counts cross back during construction.
        assert_eq!(frame.receipt().device_to_host_bytes(), 8);
        assert!(frame.voxel_segments().unwrap().cell_count() < expected.len() as u32);

        let normals = frame.normals().unwrap().readback(&runtime).unwrap();
        for normal in normals {
            assert!(normal.normal[0].abs() < 1e-5);
            assert!(normal.normal[1].abs() < 1e-5);
            assert!((normal.normal[2].abs() - 1.0).abs() < 1e-5);
            assert!(normal.curvature.abs() < 1e-5);
        }

        let actual = frame.readback_positions(&runtime).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(frame.receipt().device_to_host_bytes(), 8 + (expected.len() * 3 * 4) as u64);
        assert!(frame.reduce_attributes(&runtime, AoSoAAttributeAggregation::Average).is_err());
        frame.recycle(&runtime).unwrap();
    }

    #[test]
    fn frame_partition_preserves_source_points_across_chunk_sizes() {
        let points = [
            [1.3, 0.0, 0.0],
            [-0.4, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [-1.3, 0.0, 0.0],
            [2.1, 0.0, 0.0],
            [1.4, 0.0, 0.0],
        ];
        let origin = [-0.5, 0.0, 0.0];
        let mut builder = PointCloudBuilder::xyz();
        for point in points {
            builder.push_point(point).unwrap();
        }
        let cloud = builder.build().unwrap();
        let keys = points.map(|p| ((p[0] - origin[0]).floor() as i64, 0, 0));
        let expected = build_voxel_segments(&keys);
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        for chunk_size in [1, 2, 64] {
            let tensor = SpatialTensor::new(&cloud, chunk_size).unwrap();
            let mut frame =
                run_aoso_voxel_normal_frame(&runtime, &tensor, origin, 1.0, 0.5).unwrap();
            assert_eq!(frame.point_count(), points.len());
            assert_eq!(frame.normals().unwrap().point_count(), points.len());
            assert_eq!(
                frame.voxel_segments().unwrap().to_voxel_segments(&runtime).unwrap(),
                expected
            );
            assert_eq!(frame.receipt().device_to_host_bytes(), 8);
            assert_eq!(frame.readback_positions(&runtime).unwrap(), points);
            assert_eq!(frame.receipt().host_to_device_bytes(), 7 * 12);
            assert_eq!(frame.receipt().gpu_to_gpu_bytes(), 7 * 12);
            assert_eq!(frame.receipt().device_to_host_bytes(), 8 + 7 * 12);
            frame.recycle(&runtime).unwrap();
        }
    }

    #[test]
    fn empty_frame_has_no_payload_or_metadata_transfers() {
        let mut buffers = PointBufferSet::new();
        for field in ["x", "y", "z"] {
            buffers.insert(field, PointBuffer::F32(Vec::new()));
        }
        let cloud =
            PointCloud::try_from_parts(StandardSchemas::point_xyz(), buffers, Default::default())
                .unwrap();
        let tensor = SpatialTensor::new(&cloud, 2).unwrap();
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let mut frame = run_aoso_voxel_normal_frame(&runtime, &tensor, [0.0; 3], 1.0, 0.5).unwrap();
        assert_eq!(frame.point_count(), 0);
        assert_eq!(frame.voxel_segments().unwrap().construction_readback_bytes(), 0);
        assert_eq!(frame.radius_grid().unwrap().segments().construction_readback_bytes(), 0);
        frame.estimate_normals(&runtime, 1.0).unwrap();
        assert!(frame.readback_positions(&runtime).unwrap().is_empty());
        assert_eq!(frame.receipt().host_to_device_bytes(), 0);
        assert_eq!(frame.receipt().gpu_to_gpu_bytes(), 0);
        assert_eq!(frame.receipt().device_to_host_bytes(), 0);
        frame.recycle(&runtime).unwrap();
    }

    #[test]
    fn frame_rejects_a_different_runtime() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        let cloud = builder.build().unwrap();
        let tensor = SpatialTensor::new(&cloud, 1).unwrap();
        let runtime = WgpuRuntime::new_headless().expect("first runtime");
        let other = WgpuRuntime::new_headless().expect("second runtime");
        let frame = run_aoso_voxel_normal_frame(&runtime, &tensor, [0.0; 3], 1.0, 1.0).unwrap();

        assert!(frame.validate_runtime(&other).is_err());
        frame.recycle(&runtime).unwrap();
    }

    #[test]
    fn frame_native_normals_and_attributes_replace_safely() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzinormal());
        for point in [
            [0.1, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0],
            [1.1, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0],
            [1.3, 0.0, 0.0, 8.0, 0.0, 0.0, 1.0],
            [2.1, 0.0, 0.0, 6.0, 0.0, 0.0, 1.0],
        ] {
            builder.push_point(point).unwrap();
        }
        let cloud = builder.build().unwrap();
        let tensor = SpatialTensor::new(&cloud, 2).unwrap();
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let attributes = upload_spatial_tensor_attribute_chunks(
            &runtime,
            &tensor,
            AoSoAAttributeLayout::XYZ_INTENSITY_NORMALS,
        )
        .unwrap();
        let mut frame = run_aoso_voxel_normal_frame(&runtime, &tensor, [0.0; 3], 1.0, 0.5).unwrap();
        frame.attach_attributes(&runtime, attributes).unwrap();

        let reduced =
            frame.reduce_attributes(&runtime, AoSoAAttributeAggregation::Average).unwrap();
        assert_eq!(reduced.len(), 3);
        assert_eq!(reduced.as_slice()[10], 6.0);
        assert!(frame.has_capability(GpuFrameCapability::Attributes));
        // Attribute uploads took place outside frame methods; reduction copies
        // the attached records on-device and reads back the reduced records.
        assert_eq!(frame.receipt().host_to_device_bytes(), 4 * 3 * 4);
        assert_eq!(frame.receipt().gpu_to_gpu_bytes(), 4 * 3 * 4 + 4 * 7 * 4);
        assert_eq!(frame.receipt().device_to_host_bytes(), 8 + 3 * 7 * 4);

        frame.estimate_normals(&runtime, 0.5).unwrap();
        assert_eq!(frame.radius_grid().unwrap().radius(), 0.5);
        assert_eq!(frame.receipt().device_to_host_bytes(), 8 + 3 * 7 * 4);
        let before_invalid_radius = frame.receipt().clone();
        assert!(frame.estimate_normals(&runtime, 0.0).is_err());
        assert_eq!(frame.receipt(), &before_invalid_radius);
        frame.estimate_normals(&runtime, 1.0).unwrap();
        assert_eq!(frame.radius_grid().unwrap().radius(), 1.0);
        assert_eq!(frame.normals().unwrap().point_count(), 4);
        assert_eq!(frame.receipt().device_to_host_bytes(), 12 + 3 * 7 * 4);
        assert!(frame.receipt().stages().ends_with(&[
            "radius-normals",
            "radius-grid",
            "radius-normals",
        ]));
        frame.recycle(&runtime).unwrap();
    }
}
