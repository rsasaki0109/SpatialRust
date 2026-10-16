use spatialrust_core::SpatialResult;
use wgpu::util::DeviceExt;

use crate::{GpuVoxelSegments, WgpuRuntime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WideU8Plan {
    pub channel_bytes: u64,
    pub output_bytes: u64,
}

pub(crate) fn plan_wide_voxel_u8(
    limits: &wgpu::Limits,
    cells: u32,
    channels: usize,
    target_bytes: u64,
    offset: u64,
    host_limit: u64,
) -> Option<WideU8Plan> {
    if channels <= 4
        || offset % 4 != 0
        || limits.max_bind_groups == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < 16
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
        || target_bytes > limits.max_buffer_size
        || target_bytes > host_limit
    {
        return None;
    }
    let batch = super::voxel_u8_batch::plan_batched_voxel_u8(limits, cells, 4)?;
    let channel_bytes = u64::from(cells).div_ceil(4).checked_mul(4)?;
    let attribute_bytes = channel_bytes.checked_mul(u64::try_from(channels).ok()?)?;
    if offset.checked_add(attribute_bytes)? > target_bytes {
        return None;
    }
    Some(WideU8Plan { channel_bytes, output_bytes: batch.output_bytes })
}

pub(crate) fn try_record_wide_voxel_u8_to_staging(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    channels: &[&[u8]],
    segments: &GpuVoxelSegments,
    staging: &wgpu::Buffer,
    offset: u64,
    average: bool,
) -> SpatialResult<bool> {
    if channels.len() <= 4 {
        return Ok(false);
    }
    let Some(plan) = plan_wide_voxel_u8(
        &runtime.device().limits(),
        segments.cell_count(),
        channels.len(),
        staging.size(),
        offset,
        isize::MAX as u64,
    ) else {
        return Ok(false);
    };
    let device = runtime.device();
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-u8-wide-output"),
        size: plan.output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut controls = super::voxel_u8_batch::BatchedU8Controls::default();
    for (batch, chunk) in channels.chunks(4).enumerate() {
        let values: Vec<_> = chunk
            .iter()
            .map(|channel| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(if average {
                        "voxel-reduce-xyz-u8-values"
                    } else {
                        "voxel-gather-xyz-u8-values"
                    }),
                    contents: &crate::readback::u8_storage_upload_contents(channel),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .collect();
        if chunk.len() == 1 {
            let record = if average {
                super::voxel_reduce::record_voxel_reduce_u8_packed_pass
            } else {
                super::voxel_gather::record_voxel_gather_u8_packed_pass
            };
            record(
                encoder,
                runtime,
                &values[0],
                segments.point_indices_buffer(),
                segments.cell_starts_buffer(),
                segments.cell_count(),
                segments.point_count(),
                &output,
            )?;
        } else {
            super::voxel_u8_batch::record_voxel_u8_batch_pass_with_controls(
                encoder,
                runtime,
                &values,
                segments,
                &output,
                average,
                &mut controls,
            );
        }
        // Preserve packed tails and copy each batch before reusing the output.
        encoder.copy_buffer_to_buffer(
            &output,
            0,
            staging,
            offset + plan.channel_bytes * (batch * 4) as u64,
            plan.channel_bytes * chunk.len() as u64,
        );
    }
    Ok(true)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_u8_wide_tests.rs"]
mod tests;
