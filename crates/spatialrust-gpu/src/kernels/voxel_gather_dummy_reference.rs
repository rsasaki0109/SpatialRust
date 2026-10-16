// Original active/inactive four-channel recording block from merged PR 244.
use super::super::*;

pub(super) fn baseline_record_voxel_gather_multi4_active_outputs(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    value_buffers: &[wgpu::Buffer],
    segments: &GpuVoxelSegments,
    output_buffers: &[wgpu::Buffer],
    channel_len: u64,
) -> SpatialResult<()> {
    let device = runtime.device();
    let channel_count = output_buffers.len();
    let empty = empty_storage_buffer(device)?;
    let dummy_outputs: [wgpu::Buffer; MULTI4_CHANNELS] = std::array::from_fn(|_| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-xyz-attrs-dummy-output"),
            size: channel_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    });
    let values: [&wgpu::Buffer; MULTI4_CHANNELS] =
        std::array::from_fn(
            |index| {
                if index < channel_count {
                    &value_buffers[index]
                } else {
                    &empty
                }
            },
        );
    let outputs: [&wgpu::Buffer; MULTI4_CHANNELS] = std::array::from_fn(|index| {
        if index < channel_count {
            &output_buffers[index]
        } else {
            &dummy_outputs[index]
        }
    });
    record_voxel_gather_multi4_pass(
        encoder,
        runtime,
        &values,
        segments,
        &outputs,
        channel_count as u32,
    )?;
    Ok(())
}
