// Complete standalone parents and their serial readback frozen from 0027d5d5fa6a7cfbfe9b46d70bf6dcc6f10ede85.
use super::*;
use crate::kernels::voxel_reduce::record_voxel_reduce_f32_pass;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
const WORKGROUP_SIZE: u32 = 256;
const MULTI2_CHANNELS: usize = 2;
const MULTI4_CHANNELS: usize = 4;
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GatherUniform {
    cell_count: u32,
    point_count: u32,
    channel_count: u32,
    _pad: u32,
}
struct StagingUnmap<'a>(&'a wgpu::Buffer);
impl Drop for StagingUnmap<'_> {
    fn drop(&mut self) {
        self.0.unmap();
    }
}
pub(super) fn read_staging_f32_channels(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    channels: usize,
    cells: usize,
) -> SpatialResult<Vec<Vec<f32>>> {
    let slice = staging_buffer.slice(..);
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

    let _unmap = StagingUnmap(staging_buffer);
    let data = slice.get_mapped_range();
    let values: &[f32] = bytemuck::cast_slice(&data);
    let values = &values[..cells * channels];
    Ok((0..channels)
        .map(|channel| values[channel * cells..(channel + 1) * cells].to_vec())
        .collect())
}
pub(super) mod voxel_standalone_packed {
    use spatialrust_core::SpatialResult;
    use wgpu::util::DeviceExt;

    use crate::{GpuVoxelSegments, WgpuRuntime};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct PackedSubmissionPlan {
        pub width: usize,
        pub channels_per_submission: usize,
        pub channel_bytes: u64,
    }

    pub(crate) fn plan_packed_voxel_submission(
        limits: &wgpu::Limits,
        cells: u32,
        channels: usize,
        width: usize,
        host_limit: u64,
    ) -> Option<PackedSubmissionPlan> {
        if cells == 0
            || channels == 0
            || ![2, 4].contains(&width)
            || limits.max_storage_buffers_per_shader_stage < 7
            || limits.max_bindings_per_bind_group < 8
            || limits.max_storage_buffers_per_shader_stage < (2 + 2 * width) as u32
            || limits.max_bindings_per_bind_group < (3 + 2 * width) as u32
            || limits.max_compute_workgroup_size_x < 256
            || limits.max_compute_invocations_per_workgroup < 256
            || cells.div_ceil(256) > limits.max_compute_workgroups_per_dimension
        {
            return None;
        }
        let channels_per_submission = if channels > width {
            // Preserve the original coalesced policy and its acceptance limits.
            super::voxel_gather_submit::plan_voxel_gather_submission_with_host_limit(
                limits, cells, channels, width, host_limit,
            )?
            .channels_per_submission
        } else {
            channels
        };
        let channel_bytes = u64::from(cells).checked_mul(4)?;
        let output_words = u64::from(cells).checked_mul(channels.min(width) as u64)?;
        let output_bytes = output_words.checked_mul(4)?;
        let staging_bytes = channel_bytes.checked_mul(channels_per_submission as u64)?;
        if output_words > u64::from(u32::MAX)
            || output_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || staging_bytes > limits.max_buffer_size
            || staging_bytes > host_limit
        {
            return None;
        }
        Some(PackedSubmissionPlan { width, channels_per_submission, channel_bytes })
    }

    pub(crate) fn try_gather_voxel_channels_packed(
        runtime: &WgpuRuntime,
        channels: &[&[f32]],
        segments: &GpuVoxelSegments,
    ) -> Option<SpatialResult<Vec<Vec<f32>>>> {
        if channels.is_empty() {
            return None;
        }
        let plan = plan_packed_voxel_submission(
            &runtime.device().limits(),
            segments.cell_count(),
            channels.len(),
            runtime.max_gather_channels() as usize,
            isize::MAX as u64,
        )?;
        Some(gather_voxel_channels_packed_with_plan(runtime, channels, segments, plan))
    }

    pub(crate) fn gather_voxel_channels_packed_with_plan(
        runtime: &WgpuRuntime,
        channels: &[&[f32]],
        segments: &GpuVoxelSegments,
        plan: PackedSubmissionPlan,
    ) -> SpatialResult<Vec<Vec<f32>>> {
        let device = runtime.device();
        let mut gathered = Vec::with_capacity(channels.len());
        // At most two original batches retain inputs/outputs through each wait.
        for window in channels.chunks(plan.channels_per_submission) {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-gather-packed-staging"),
                size: plan.channel_bytes * window.len() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxel-gather-packed-encoder"),
            });
            record_packed_voxel_window(&mut encoder, runtime, window, segments, &staging, 0, plan);
            runtime.queue().submit(Some(encoder.finish()));
            gathered.extend(super::read_staging_f32_channels(
                device,
                &staging,
                window.len(),
                segments.cell_count() as usize,
            )?);
        }
        Ok(gathered)
    }

    pub(crate) fn record_packed_voxel_window(
        encoder: &mut wgpu::CommandEncoder,
        runtime: &WgpuRuntime,
        window: &[&[f32]],
        segments: &GpuVoxelSegments,
        staging: &wgpu::Buffer,
        offset: u64,
        plan: PackedSubmissionPlan,
    ) {
        let device = runtime.device();
        for (batch, chunk) in window.chunks(plan.width).enumerate() {
            let values: Vec<_> = chunk
                .iter()
                .map(|channel| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("voxel-gather-multi-values"),
                        contents: bytemuck::cast_slice(channel),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
                })
                .collect();
            let bytes = plan.channel_bytes * chunk.len() as u64;
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-gather-packed-output"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            // Standalone one-channel tails also gather point zero in empty cells.
            crate::kernels::voxel_gather_packed::record_packed_voxel_gather(
                encoder, runtime, &values, segments, &output,
            );
            encoder.copy_buffer_to_buffer(
                &output,
                0,
                staging,
                offset + plan.channel_bytes * (batch * plan.width) as u64,
                bytes,
            );
        }
    }
}
pub(super) mod voxel_gather_submit {
    use spatialrust_core::SpatialResult;
    use wgpu::util::DeviceExt;

    use crate::{GpuVoxelSegments, WgpuRuntime};

    use crate::kernels::voxel_gather::{
        record_voxel_gather_multi2_pass, record_voxel_gather_multi4_pass,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct GatherSubmissionPlan {
        pub batch_width: usize,
        pub channels_per_submission: usize,
        pub channel_bytes: u64,
    }

    pub(crate) fn plan_voxel_gather_submission(
        limits: &wgpu::Limits,
        cells: u32,
        channels: usize,
        batch_width: usize,
    ) -> Option<GatherSubmissionPlan> {
        plan_voxel_gather_submission_with_host_limit(
            limits,
            cells,
            channels,
            batch_width,
            isize::MAX as u64,
        )
    }

    pub(crate) fn plan_voxel_gather_submission_with_host_limit(
        limits: &wgpu::Limits,
        cells: u32,
        channels: usize,
        batch_width: usize,
        host_staging_limit: u64,
    ) -> Option<GatherSubmissionPlan> {
        if cells == 0
            || ![2, 4].contains(&batch_width)
            || channels <= batch_width
            || limits.max_storage_buffers_per_shader_stage < (2 + 2 * batch_width) as u32
            || limits.max_bindings_per_bind_group < (3 + 2 * batch_width) as u32
            || limits.max_compute_workgroup_size_x < 256
            || limits.max_compute_invocations_per_workgroup < 256
            || cells.div_ceil(256) > limits.max_compute_workgroups_per_dimension
        {
            return None;
        }
        let channel_bytes = u64::from(cells).checked_mul(4)?;
        let channels_per_submission = channels.min(2 * batch_width);
        let staging_bytes = channel_bytes.checked_mul(channels_per_submission as u64)?;
        if channel_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || staging_bytes > limits.max_buffer_size
            || staging_bytes > host_staging_limit
        {
            return None;
        }
        Some(GatherSubmissionPlan { batch_width, channels_per_submission, channel_bytes })
    }

    pub(crate) fn try_gather_voxel_channels_coalesced(
        runtime: &WgpuRuntime,
        channels: &[&[f32]],
        segments: &GpuVoxelSegments,
    ) -> Option<SpatialResult<Vec<Vec<f32>>>> {
        let batch_width = runtime.max_gather_channels() as usize;
        if channels.len() <= batch_width {
            return None;
        }
        let plan = plan_voxel_gather_submission(
            &runtime.device().limits(),
            segments.cell_count(),
            channels.len(),
            batch_width,
        )?;
        Some(gather_voxel_channels_with_plan(runtime, channels, segments, plan))
    }

    pub(crate) fn gather_voxel_channels_with_plan(
        runtime: &WgpuRuntime,
        channels: &[&[f32]],
        segments: &GpuVoxelSegments,
        plan: GatherSubmissionPlan,
    ) -> SpatialResult<Vec<Vec<f32>>> {
        let device = runtime.device();
        let empty = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-empty"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mut gathered = Vec::with_capacity(channels.len());
        // At most two original batches retain device inputs/outputs until each wait.
        for window in channels.chunks(plan.channels_per_submission) {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-gather-coalesced-staging"),
                size: plan.channel_bytes * window.len() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxel-gather-coalesced-encoder"),
            });
            record_window(&mut encoder, runtime, window, segments, &empty, &staging, plan)?;
            runtime.queue().submit(Some(encoder.finish()));
            gathered.extend(super::read_staging_f32_channels(
                device,
                &staging,
                window.len(),
                segments.cell_count() as usize,
            )?);
        }
        Ok(gathered)
    }

    fn record_window(
        encoder: &mut wgpu::CommandEncoder,
        runtime: &WgpuRuntime,
        channels: &[&[f32]],
        segments: &GpuVoxelSegments,
        empty: &wgpu::Buffer,
        staging: &wgpu::Buffer,
        plan: GatherSubmissionPlan,
    ) -> SpatialResult<()> {
        let device = runtime.device();
        for (batch, chunk) in channels.chunks(plan.batch_width).enumerate() {
            let values: Vec<_> = chunk
                .iter()
                .map(|channel| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("voxel-gather-multi-values"),
                        contents: bytemuck::cast_slice(channel),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
                })
                .collect();
            // Preserve the original two/four-output pipeline, including unused slots.
            let width = if chunk.len() > 2 { 4 } else { 2 };
            let outputs: Vec<_> = (0..width)
                .map(|_| {
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(if width == 4 {
                            "voxel-gather-multi4-output"
                        } else {
                            "voxel-gather-multi-output"
                        }),
                        size: plan.channel_bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    })
                })
                .collect();
            if width == 4 {
                let inputs = std::array::from_fn(|i| values.get(i).unwrap_or(empty));
                let output_refs = std::array::from_fn(|i| &outputs[i]);
                record_voxel_gather_multi4_pass(
                    encoder,
                    runtime,
                    &inputs,
                    segments,
                    &output_refs,
                    chunk.len() as u32,
                )?;
            } else {
                let inputs = std::array::from_fn(|i| values.get(i).unwrap_or(empty));
                let output_refs = std::array::from_fn(|i| &outputs[i]);
                record_voxel_gather_multi2_pass(
                    encoder,
                    runtime,
                    &inputs,
                    segments,
                    &output_refs,
                    chunk.len() as u32,
                )?;
            }
            for (index, output) in outputs.iter().take(chunk.len()).enumerate() {
                let channel = batch * plan.batch_width + index;
                encoder.copy_buffer_to_buffer(
                    output,
                    0,
                    staging,
                    plan.channel_bytes * channel as u64,
                    plan.channel_bytes,
                );
            }
        }
        Ok(())
    }
}
pub(super) mod voxel_average_batch {
    use spatialrust_core::SpatialResult;
    use wgpu::util::DeviceExt;

    use crate::{GpuVoxelSegments, WgpuRuntime};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct AverageBatchPlan {
        pub channel_bytes: u64,
        pub staging_bytes: u64,
    }

    pub(crate) fn plan_voxel_average_batch(
        limits: &wgpu::Limits,
        cells: u32,
        channels: usize,
    ) -> Option<AverageBatchPlan> {
        plan_voxel_average_batch_with_host_limit(limits, cells, channels, isize::MAX as u64)
    }

    pub(crate) fn plan_voxel_average_batch_with_host_limit(
        limits: &wgpu::Limits,
        cells: u32,
        channels: usize,
        host_limit: u64,
    ) -> Option<AverageBatchPlan> {
        if cells == 0
            || channels < 2
            || limits.max_storage_buffers_per_shader_stage < 7
            || limits.max_bindings_per_bind_group < 8
            || limits.max_compute_workgroup_size_x < 256
            || limits.max_compute_invocations_per_workgroup < 256
            || cells.div_ceil(256) > limits.max_compute_workgroups_per_dimension
        {
            return None;
        }
        let words = u64::from(cells).checked_mul(channels.min(4) as u64)?;
        let channel_bytes = u64::from(cells).checked_mul(4)?;
        let output_bytes = words.checked_mul(4)?;
        let staging_bytes = channel_bytes.checked_mul(u64::try_from(channels).ok()?)?;
        if words > u64::from(u32::MAX)
            || output_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || staging_bytes > limits.max_buffer_size
            || staging_bytes > host_limit
        {
            return None;
        }
        Some(AverageBatchPlan { channel_bytes, staging_bytes })
    }

    pub(crate) fn try_reduce_voxel_average_batched(
        runtime: &WgpuRuntime,
        channels: &[&[f32]],
        segments: &GpuVoxelSegments,
    ) -> Option<SpatialResult<Vec<Vec<f32>>>> {
        if channels.len() < 2 {
            return None;
        }
        let plan = plan_voxel_average_batch(
            &runtime.device().limits(),
            segments.cell_count(),
            channels.len(),
        )?;
        Some(reduce_batched(runtime, channels, segments, plan))
    }

    fn reduce_batched(
        runtime: &WgpuRuntime,
        channels: &[&[f32]],
        segments: &GpuVoxelSegments,
        plan: AverageBatchPlan,
    ) -> SpatialResult<Vec<Vec<f32>>> {
        let device = runtime.device();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-reduce-multi-staging"),
            size: plan.staging_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("voxel-reduce-multi-encoder"),
        });
        for (batch, chunk) in channels.chunks(4).enumerate() {
            let values: Vec<_> = chunk
                .iter()
                .map(|channel| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("voxel-reduce-multi-values"),
                        contents: bytemuck::cast_slice(channel),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
                })
                .collect();
            let bytes = plan.channel_bytes * chunk.len() as u64;
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-reduce-multi-packed-output"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            if chunk.len() == 1 {
                crate::kernels::voxel_reduce::record_voxel_reduce_f32_pass(
                    &mut encoder,
                    runtime,
                    &values[0],
                    segments.point_indices_buffer(),
                    segments.cell_starts_buffer(),
                    segments.cell_count(),
                    segments.point_count(),
                    &output,
                )?;
            } else {
                record_voxel_average_batch(&mut encoder, runtime, &values, segments, &output);
            }
            encoder.copy_buffer_to_buffer(
                &output,
                0,
                &staging,
                plan.channel_bytes * (batch * 4) as u64,
                bytes,
            );
        }
        runtime.queue().submit(Some(encoder.finish()));
        super::read_staging_f32_channels(
            device,
            &staging,
            channels.len(),
            segments.cell_count() as usize,
        )
    }

    use crate::kernels::voxel_average_batch::record_voxel_average_batch;
}
pub(super) fn baseline_gather_voxel_first_f32_multi_gpu(
    runtime: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
) -> SpatialResult<Vec<Vec<f32>>> {
    if channels.is_empty() {
        return Ok(Vec::new());
    }
    if segments.cell_count() == 0 {
        return Ok(vec![Vec::new(); channels.len()]);
    }

    let point_count = segments.point_count() as usize;
    for channel in channels {
        if channel.len() != point_count {
            return Err(SpatialError::BufferLengthMismatch {
                expected: point_count,
                found: channel.len(),
            });
        }
    }

    if let Some(result) =
        voxel_standalone_packed::try_gather_voxel_channels_packed(runtime, channels, segments)
    {
        return result;
    }

    if let Some(result) =
        voxel_gather_submit::try_gather_voxel_channels_coalesced(runtime, channels, segments)
    {
        return result;
    }

    let max_channels = runtime.max_gather_channels() as usize;
    let device = runtime.device();
    let empty = empty_storage_buffer(device)?;
    let mut gathered = Vec::with_capacity(channels.len());

    for chunk in channels.chunks(max_channels) {
        let mut value_buffers = [None, None, None, None];
        for (index, channel) in chunk.iter().enumerate() {
            value_buffers[index] =
                Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel-gather-multi-values"),
                    contents: bytemuck::cast_slice(channel),
                    usage: wgpu::BufferUsages::STORAGE,
                }));
        }

        let value_refs: [&wgpu::Buffer; MULTI4_CHANNELS] = std::array::from_fn(|index| {
            if index < chunk.len() {
                value_buffers[index].as_ref().expect("value buffer")
            } else {
                &empty
            }
        });

        gathered.extend(dispatch_voxel_gather_multi_gpu_buffers(
            runtime,
            &value_refs,
            segments,
            chunk.len() as u32,
        )?);
    }

    Ok(gathered)
}
pub(super) fn baseline_reduce_voxel_average_f32_multi_gpu(
    runtime: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
) -> SpatialResult<Vec<Vec<f32>>> {
    if channels.is_empty() {
        return Ok(Vec::new());
    }
    if segments.cell_count() == 0 {
        return Ok(vec![Vec::new(); channels.len()]);
    }

    let point_count = segments.point_count() as usize;
    for channel in channels {
        if channel.len() != point_count {
            return Err(SpatialError::BufferLengthMismatch {
                expected: point_count,
                found: channel.len(),
            });
        }
    }

    if let Some(result) =
        voxel_average_batch::try_reduce_voxel_average_batched(runtime, channels, segments)
    {
        return result;
    }

    let device = runtime.device();
    let queue = runtime.queue();
    let cell_count = segments.cell_count();
    let cells = cell_count as usize;
    let channel_len = cells * std::mem::size_of::<f32>();
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-multi-staging"),
        size: (channel_len * channels.len()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-reduce-multi-encoder"),
    });

    for (channel_index, channel) in channels.iter().enumerate() {
        let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-reduce-multi-values"),
            contents: bytemuck::cast_slice(channel),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-reduce-multi-output"),
            size: channel_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        record_voxel_reduce_f32_pass(
            &mut encoder,
            runtime,
            &values_buffer,
            segments.point_indices_buffer(),
            segments.cell_starts_buffer(),
            cell_count,
            segments.point_count(),
            &output_buffer,
        )?;

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            (channel_len * channel_index) as u64,
            channel_len as u64,
        );
    }

    queue.submit(Some(encoder.finish()));

    read_staging_f32_channels(device, &staging_buffer, channels.len(), cells)
}
fn dispatch_voxel_gather_multi_gpu_buffers(
    runtime: &WgpuRuntime,
    values: &[&wgpu::Buffer; MULTI4_CHANNELS],
    segments: &GpuVoxelSegments,
    channel_count: u32,
) -> SpatialResult<Vec<Vec<f32>>> {
    let pipelines = runtime.pipelines();
    if channel_count > 2 && pipelines.voxel_gather.multi4_pipeline.is_some() {
        dispatch_voxel_gather_multi4_gpu_buffers(runtime, values, segments, channel_count)
    } else {
        if channel_count > MULTI2_CHANNELS as u32 {
            return Err(SpatialError::InvalidArgument(format!(
                "gpu adapter supports only {} channels per gather dispatch",
                pipelines.voxel_gather.multi_max_channels
            )));
        }
        dispatch_voxel_gather_multi2_gpu_buffers(runtime, values, segments, channel_count)
    }
}
fn dispatch_voxel_gather_multi2_gpu_buffers(
    runtime: &WgpuRuntime,
    values: &[&wgpu::Buffer; MULTI4_CHANNELS],
    segments: &GpuVoxelSegments,
    channel_count: u32,
) -> SpatialResult<Vec<Vec<f32>>> {
    let device = runtime.device();
    let queue = runtime.queue();
    let pipelines = runtime.pipelines();
    let cell_count = segments.cell_count();
    let point_count = segments.point_count();
    let channels = channel_count as usize;

    let uniform = GatherUniform { cell_count, point_count, channel_count, _pad: 0 };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-multi-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let channel_len = cell_count as usize * std::mem::size_of::<f32>();
    let mut outputs = [None, None];
    for output in outputs.iter_mut() {
        *output = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-multi-output"),
            size: channel_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }
    let output_refs: [&wgpu::Buffer; MULTI2_CHANNELS] =
        std::array::from_fn(|index| outputs[index].as_ref().expect("output buffer"));

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-multi-staging"),
        size: (channel_len * channels) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-gather-multi-bind-group"),
        layout: &pipelines.voxel_gather.multi_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: segments.point_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: segments.cell_starts_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry { binding: 3, resource: values[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: values[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output_refs[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: output_refs[1].as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-gather-multi-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-gather-multi-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_gather.multi_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }

    for (index, output) in output_refs.iter().take(channels).enumerate() {
        let offset = (channel_len * index) as u64;
        encoder.copy_buffer_to_buffer(output, 0, &staging_buffer, offset, channel_len as u64);
    }
    queue.submit(Some(encoder.finish()));

    read_staging_f32_channels(device, &staging_buffer, channels, cell_count as usize)
}
fn dispatch_voxel_gather_multi4_gpu_buffers(
    runtime: &WgpuRuntime,
    values: &[&wgpu::Buffer; MULTI4_CHANNELS],
    segments: &GpuVoxelSegments,
    channel_count: u32,
) -> SpatialResult<Vec<Vec<f32>>> {
    let device = runtime.device();
    let queue = runtime.queue();
    let pipelines = runtime.pipelines();
    let multi4_pipeline = pipelines.voxel_gather.multi4_pipeline.as_ref().ok_or_else(|| {
        SpatialError::InvalidArgument(
            "4-channel gather pipeline is unavailable on this gpu adapter".to_owned(),
        )
    })?;
    let multi4_layout =
        pipelines.voxel_gather.multi4_bind_group_layout.as_ref().expect("multi4 layout");

    let cell_count = segments.cell_count();
    let point_count = segments.point_count();
    let channels = channel_count as usize;

    let uniform = GatherUniform { cell_count, point_count, channel_count, _pad: 0 };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-multi4-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let channel_len = cell_count as usize * std::mem::size_of::<f32>();
    let mut outputs = [None, None, None, None];
    for output in outputs.iter_mut() {
        *output = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-multi4-output"),
            size: channel_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }
    let output_refs: [&wgpu::Buffer; MULTI4_CHANNELS] =
        std::array::from_fn(|index| outputs[index].as_ref().expect("output buffer"));

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-multi4-staging"),
        size: (channel_len * channels) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-gather-multi4-bind-group"),
        layout: multi4_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: segments.point_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: segments.cell_starts_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry { binding: 3, resource: values[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: values[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: values[2].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: values[3].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: output_refs[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: output_refs[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: output_refs[2].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: output_refs[3].as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-gather-multi4-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-gather-multi4-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(multi4_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }

    for (index, output) in output_refs.iter().take(channels).enumerate() {
        let offset = (channel_len * index) as u64;
        encoder.copy_buffer_to_buffer(output, 0, &staging_buffer, offset, channel_len as u64);
    }
    queue.submit(Some(encoder.finish()));

    read_staging_f32_channels(device, &staging_buffer, channels, cell_count as usize)
}
fn empty_storage_buffer(device: &wgpu::Device) -> SpatialResult<wgpu::Buffer> {
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-empty"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    }))
}
