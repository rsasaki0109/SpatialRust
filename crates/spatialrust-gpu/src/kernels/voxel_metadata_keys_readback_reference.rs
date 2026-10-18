//! One bounded staging allocation for an explicit segment metadata readback.

use crate::kernels::voxel_compact::{
    GpuVoxelSegments, SpatialError, SpatialResult, VoxelKeyOutput, WgpuRuntime,
};

// Batching retains all three planes until decoding. Bound that staging request;
// larger payloads keep the original serial allocations and their lower peak.
const MAX_BATCH_BYTES: u64 = 64 * 1024 * 1024;
type Metadata = (Vec<(i64, i64, i64)>, Vec<u32>, Vec<u32>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Layout {
    starts_offset: u64,
    indices_offset: u64,
    bytes: u64,
}

fn layout(cell_count: usize, point_count: usize, max_buffer_size: u64) -> Option<Layout> {
    if cell_count == 0 || cell_count > point_count {
        return None;
    }
    let cells = u64::try_from(cell_count).ok()?;
    let points = u64::try_from(point_count).ok()?;
    let starts_offset = cells.checked_mul(std::mem::size_of::<VoxelKeyOutput>() as u64)?;
    let indices_offset = starts_offset.checked_add(cells.checked_mul(4)?)?;
    let bytes = indices_offset.checked_add(points.checked_mul(4)?)?;
    if bytes > max_buffer_size.min(MAX_BATCH_BYTES).min(isize::MAX as u64) {
        return None;
    }
    // Every offset is a multiple of COPY_BUFFER_ALIGNMENT; MAP_ALIGNMENT applies
    // to the mapped range's zero start, not the internal plane boundaries.
    Some(Layout { starts_offset, indices_offset, bytes })
}

pub(super) fn try_read(
    runtime: &WgpuRuntime,
    segments: &GpuVoxelSegments,
    cell_count: usize,
) -> Option<SpatialResult<Metadata>> {
    let plan = layout(
        cell_count,
        segments.point_count() as usize,
        runtime.device().limits().max_buffer_size,
    )?;
    Some(read(runtime, segments, plan))
}

fn read(
    runtime: &WgpuRuntime,
    segments: &GpuVoxelSegments,
    plan: Layout,
) -> SpatialResult<Metadata> {
    let device = runtime.device();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-metadata-staging"),
        size: plan.bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-metadata-read-encoder"),
    });
    encoder.copy_buffer_to_buffer(segments.keys_buffer(), 0, &staging, 0, plan.starts_offset);
    encoder.copy_buffer_to_buffer(
        segments.cell_starts_buffer(),
        0,
        &staging,
        plan.starts_offset,
        plan.indices_offset - plan.starts_offset,
    );
    encoder.copy_buffer_to_buffer(
        segments.point_indices_buffer(),
        0,
        &staging,
        plan.indices_offset,
        plan.bytes - plan.indices_offset,
    );
    runtime.queue().submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| SpatialError::InvalidArgument("failed to receive wgpu map result".to_owned()))?
        .map_err(|error| {
            SpatialError::InvalidArgument(format!("failed to map wgpu buffer: {error}"))
        })?;

    let _unmap = StagingUnmap(&staging);
    let data = slice.get_mapped_range();
    let starts = plan.starts_offset as usize;
    let indices = plan.indices_offset as usize;
    let keys = bytemuck::cast_slice::<u8, VoxelKeyOutput>(&data[..starts])
        .iter()
        .map(|key| (i64::from(key.ix), i64::from(key.iy), i64::from(key.iz)))
        .collect();
    let cell_starts = bytemuck::cast_slice::<u8, u32>(&data[starts..indices]).to_vec();
    let point_indices = bytemuck::cast_slice::<u8, u32>(&data[indices..]).to_vec();
    Ok((keys, cell_starts, point_indices))
}

// Declared before the mapped view so unwinding also drops the view first.
struct StagingUnmap<'a>(&'a wgpu::Buffer);

impl Drop for StagingUnmap<'_> {
    fn drop(&mut self) {
        self.0.unmap();
    }
}
