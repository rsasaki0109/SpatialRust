//! GPU-resident spatial frame ownership and chained execution.

use spatialrust_core::{PointSchema, SpatialError, SpatialResult, SpatialTensor};

use crate::aoso_staging::runtime_device_key;
use crate::{
    build_radius_grid_aoso_gpu, downsample_voxel_centroid_aoso_chunks,
    estimate_normals_radius_grid_aoso_gpu, reduce_voxel_attributes_aoso_chunks,
    upload_spatial_tensor_xyz_chunks, AoSoAAttributeAggregation, AoSoAAttributeReduction,
    AoSoAVoxelCentroidResult, GpuAoSoAttributeChunk, GpuAoSoNormals, GpuAoSoRadiusGrid,
    GpuAoSoXyzBuffer, GpuVoxelSegments, WgpuRuntime,
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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GpuExecutionReceipt {
    host_to_device_bytes: u64,
    gpu_to_gpu_bytes: u64,
    device_to_host_bytes: u64,
    stages: Vec<&'static str>,
}

impl GpuExecutionReceipt {
    /// Returns bytes uploaded from host memory.
    #[must_use]
    pub const fn host_to_device_bytes(&self) -> u64 {
        self.host_to_device_bytes
    }

    /// Returns bytes copied between GPU buffers.
    #[must_use]
    pub const fn gpu_to_gpu_bytes(&self) -> u64 {
        self.gpu_to_gpu_bytes
    }

    /// Returns bytes explicitly read back to host memory.
    #[must_use]
    pub const fn device_to_host_bytes(&self) -> u64 {
        self.device_to_host_bytes
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
        self.validate_runtime(runtime)?;
        let grid = build_radius_grid_aoso_gpu(runtime, &self.positions, radius)?;
        self.radius_grid = Some(grid);
        self.receipt.stages.push("radius-grid");
        Ok(())
    }

    /// Estimates normals using a cached matching grid or rebuilds it first.
    pub fn estimate_normals(&mut self, runtime: &WgpuRuntime, radius: f32) -> SpatialResult<()> {
        self.validate_runtime(runtime)?;
        let matches = self.radius_grid.as_ref().is_some_and(|grid| grid.radius() == radius);
        if !matches {
            self.rebuild_radius_grid(runtime, radius)?;
        }
        let grid = self.radius_grid.as_ref().ok_or_else(|| {
            SpatialError::InvalidArgument("radius grid is unavailable".to_owned())
        })?;
        let normals = estimate_normals_radius_grid_aoso_gpu(runtime, &self.positions, grid)?;
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
        self.receipt.device_to_host_bytes +=
            (reduced.len() * reduced.layout().stride_f32() * std::mem::size_of::<f32>()) as u64;
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
        self.receipt.device_to_host_bytes += byte_len as u64;
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
pub fn run_aoso_voxel_normal_frame(
    runtime: &WgpuRuntime,
    tensor: &SpatialTensor<'_>,
    origin: [f32; 3],
    inv_leaf: f32,
    normal_radius: f32,
) -> SpatialResult<GpuSpatialFrame> {
    let chunks = upload_spatial_tensor_xyz_chunks(runtime, tensor)?;
    let upload_bytes = chunks.iter().map(|chunk| chunk.byte_len()).sum();
    let voxel = downsample_voxel_centroid_aoso_chunks(runtime, &chunks, origin, inv_leaf)?;
    for chunk in chunks {
        chunk.recycle(runtime);
    }
    let AoSoAVoxelCentroidResult { segments, positions, .. } = voxel;
    let grid = build_radius_grid_aoso_gpu(runtime, &positions, normal_radius)?;
    let normals = estimate_normals_radius_grid_aoso_gpu(runtime, &positions, &grid)?;
    let mut frame = GpuSpatialFrame::new(runtime, tensor.schema().clone(), positions)?;
    frame.attach_voxel_segments(segments)?;
    frame.attach_radius_grid(grid)?;
    frame.attach_normals(runtime, normals)?;
    frame.receipt.host_to_device_bytes = upload_bytes;
    frame.receipt.gpu_to_gpu_bytes = upload_bytes;
    frame.receipt.stages = vec!["upload", "voxel-segments", "radius-grid", "radius-normals"];
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::{run_aoso_voxel_normal_frame, GpuFrameCapability};
    use crate::{upload_spatial_tensor_attribute_chunks, AoSoAAttributeAggregation, WgpuRuntime};
    use spatialrust_core::{
        AoSoAAttributeLayout, PointCloudBuilder, SpatialTensor, StandardSchemas,
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
            run_aoso_voxel_normal_frame(&runtime, &tensor, [0.0; 3], 20.0, 0.25).unwrap();
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

        let actual = frame.readback_positions(&runtime).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(frame.receipt().device_to_host_bytes(), (expected.len() * 3 * 4) as u64);
        assert!(frame.reduce_attributes(&runtime, AoSoAAttributeAggregation::Average).is_err());
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

        frame.estimate_normals(&runtime, 0.5).unwrap();
        assert_eq!(frame.radius_grid().unwrap().radius(), 0.5);
        frame.estimate_normals(&runtime, 1.0).unwrap();
        assert_eq!(frame.radius_grid().unwrap().radius(), 1.0);
        assert_eq!(frame.normals().unwrap().point_count(), 4);
        assert!(frame.receipt().stages().ends_with(&[
            "radius-normals",
            "radius-grid",
            "radius-normals",
        ]));
        frame.recycle(&runtime).unwrap();
    }
}
