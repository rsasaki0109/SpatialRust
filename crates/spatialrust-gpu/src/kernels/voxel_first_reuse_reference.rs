// Complete public parents and original packed/window/wide recorders frozen from 7533330a52d81695ff0a9408c9011ec4ff41ab8a.
use super::*;
use crate::kernels::voxel_centroid_first_fused::record_fused_voxel_centroid_first;
use crate::kernels::voxel_first_gather_fused::record_fused_voxel_first_gather;
use crate::kernels::voxel_gather::{
    record_voxel_gather_f32_pass, record_voxel_gather_multi2_pass, record_voxel_gather_multi4_pass,
    record_voxel_gather_u8_packed_pass, record_voxel_gather_xyz_and_attrs4_packed_pass,
    record_voxel_gather_xyz_pass,
};
use crate::kernels::voxel_reduce::record_voxel_reduce_xyz_pass;
use crate::kernels::voxel_reduce_fused::plan_fused_voxel_average;
use crate::kernels::{
    voxel_centroid_first_fused, voxel_first_gather_fused, voxel_gather, voxel_gather_submit,
    voxel_reduce_fused, voxel_xyz_packed,
};
use crate::readback::{
    packed_u8_output_staging_bytes, read_staging_f32_channels,
    read_staging_xyz_and_packed_attributes, u8_storage_upload_contents,
};
use bytemuck::{Pod, Zeroable};
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

pub(super) mod voxel_gather_packed {
    use bytemuck::{Pod, Zeroable};
    use spatialrust_core::SpatialResult;
    use wgpu::util::DeviceExt;

    use crate::{GpuVoxelSegments, WgpuRuntime};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct PackedGatherPlan {
        pub width: usize,
        pub channel_bytes: u64,
        pub offset: u64,
    }

    pub(crate) fn plan_packed_voxel_attributes(
        limits: &wgpu::Limits,
        cells: u32,
        channels: usize,
        width: usize,
        target_bytes: u64,
        offset: u64,
        host_limit: u64,
    ) -> Option<PackedGatherPlan> {
        if cells == 0
            || channels < 2
            || ![2, 4].contains(&width)
            || limits.max_storage_buffers_per_shader_stage < 7
            || limits.max_bindings_per_bind_group < 8
            || limits.max_compute_workgroup_size_x < 256
            || limits.max_compute_invocations_per_workgroup < 256
            || cells.div_ceil(256) > limits.max_compute_workgroups_per_dimension
            || offset % 4 != 0
            || target_bytes > limits.max_buffer_size
            || target_bytes > host_limit
        {
            return None;
        }
        let words = u64::from(cells).checked_mul(channels.min(width) as u64)?;
        let channel_bytes = u64::from(cells).checked_mul(4)?;
        let output_bytes = words.checked_mul(4)?;
        let attribute_bytes = channel_bytes.checked_mul(u64::try_from(channels).ok()?)?;
        if words > u64::from(u32::MAX)
            || output_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || offset.checked_add(attribute_bytes)? > target_bytes
        {
            return None;
        }
        Some(PackedGatherPlan { width, channel_bytes, offset })
    }

    pub(crate) fn try_record_packed_voxel_attributes(
        encoder: &mut wgpu::CommandEncoder,
        runtime: &WgpuRuntime,
        channels: &[&[f32]],
        segments: &GpuVoxelSegments,
        staging: &wgpu::Buffer,
        offset: u64,
        upload_recycle: &mut Vec<wgpu::Buffer>,
    ) -> SpatialResult<bool> {
        if channels.len() < 2 {
            return Ok(false);
        }
        let Some(plan) = plan_packed_voxel_attributes(
            &runtime.device().limits(),
            segments.cell_count(),
            channels.len(),
            runtime.max_gather_channels() as usize,
            staging.size(),
            offset,
            isize::MAX as u64,
        ) else {
            return Ok(false);
        };
        record_packed_voxel_attributes(
            encoder,
            runtime,
            channels,
            segments,
            staging,
            upload_recycle,
            plan,
        )?;
        Ok(true)
    }

    pub(crate) fn record_packed_voxel_attributes(
        encoder: &mut wgpu::CommandEncoder,
        runtime: &WgpuRuntime,
        channels: &[&[f32]],
        segments: &GpuVoxelSegments,
        staging: &wgpu::Buffer,
        upload_recycle: &mut Vec<wgpu::Buffer>,
        plan: PackedGatherPlan,
    ) -> SpatialResult<()> {
        let device = runtime.device();
        // Keep the original batch width: a scalar tail returns +0 for empty cells,
        // whereas the original multi-channel kernels gather input point zero.
        for (batch, chunk) in channels.chunks(plan.width).enumerate() {
            let first_upload = upload_recycle.len();
            for channel in chunk {
                upload_recycle
                    .push(runtime.upload_f32_storage("voxel-gather-xyz-attrs-values", channel)?);
            }
            let values = &upload_recycle[first_upload..];
            let bytes = plan.channel_bytes * chunk.len() as u64;
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-gather-xyz-attrs-packed-output"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            if chunk.len() == 1 {
                super::voxel_gather::record_voxel_gather_f32_pass(
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
                record_packed_voxel_gather(encoder, runtime, values, segments, &output);
            }
            encoder.copy_buffer_to_buffer(
                &output,
                0,
                staging,
                plan.offset + plan.channel_bytes * (batch * plan.width) as u64,
                bytes,
            );
        }
        Ok(())
    }

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct Uniform {
        cells: u32,
        points: u32,
        channels: u32,
        _pad: u32,
    }

    pub(crate) fn record_packed_voxel_gather<B: std::borrow::Borrow<wgpu::Buffer>>(
        encoder: &mut wgpu::CommandEncoder,
        runtime: &WgpuRuntime,
        values: &[B],
        segments: &GpuVoxelSegments,
        output: &wgpu::Buffer,
    ) {
        let device = runtime.device();
        let (layout, pipeline) =
            runtime.pipelines().voxel_gather.packed_attributes_pipeline(device);
        let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-gather-attrs-packed-uniform"),
            contents: bytemuck::bytes_of(&Uniform {
                cells: segments.cell_count(),
                points: segments.point_count(),
                channels: values.len() as u32,
                _pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // Inactive read-only slots alias the first input and are never read.
        let value = |i| values.get(i).unwrap_or(&values[0]).borrow();
        let buffers = [
            &control,
            segments.point_indices_buffer(),
            segments.cell_starts_buffer(),
            value(0),
            value(1),
            value(2),
            value(3),
            output,
        ];
        let entries: [_; 8] = std::array::from_fn(|binding| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffers[binding].as_entire_binding(),
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel-gather-attrs-packed-group"),
            layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-gather-attrs-packed-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(segments.cell_count().div_ceil(256), 1, 1);
    }
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
            gathered.extend(crate::readback::read_staging_f32_channels(
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
            super::voxel_gather_packed::record_packed_voxel_gather(
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

pub(super) mod voxel_wide_first {
    use spatialrust_core::SpatialResult;

    use super::{
        voxel_centroid_first_fused::record_fused_voxel_centroid_first,
        voxel_first_gather_fused::record_fused_voxel_first_gather,
        voxel_gather_packed::{
            plan_packed_voxel_attributes, record_packed_voxel_attributes, PackedGatherPlan,
        },
        voxel_reduce_fused::plan_fused_voxel_average,
        voxel_xyz_packed::XyzMode,
    };
    use crate::{GpuVoxelSegments, WgpuRuntime};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct WideFirstPlan {
        pub prefix_bytes: u64,
        pub tail: PackedGatherPlan,
    }

    pub(crate) fn plan_wide_voxel_first(
        limits: &wgpu::Limits,
        cells: u32,
        attributes: usize,
        width: usize,
        target_bytes: u64,
        host_limit: u64,
    ) -> Option<WideFirstPlan> {
        if attributes <= 4
            || limits.max_bind_groups == 0
            || limits.max_uniform_buffers_per_shader_stage == 0
            || limits.max_uniform_buffer_binding_size < 16
            || limits.max_buffer_size < 16
            || limits.max_compute_workgroup_size_y == 0
            || limits.max_compute_workgroup_size_z == 0
        {
            return None;
        }
        let prefix = plan_fused_voxel_average(limits, cells, 4)?;
        let channel_bytes = u64::from(cells).checked_mul(4)?;
        let attributes_plan = plan_packed_voxel_attributes(
            limits,
            cells,
            attributes,
            width,
            target_bytes,
            channel_bytes.checked_mul(3)?,
            host_limit,
        )?;
        // Four prefix attributes preserve the original two-/four-wide batch boundaries.
        // The original full attribute plan bounds every remaining batch, including a scalar tail.
        Some(WideFirstPlan {
            prefix_bytes: prefix.output_bytes,
            tail: PackedGatherPlan { offset: prefix.output_bytes, ..attributes_plan },
        })
    }

    pub(crate) fn try_record_wide_voxel_first(
        encoder: &mut wgpu::CommandEncoder,
        runtime: &WgpuRuntime,
        xyz: [&wgpu::Buffer; 3],
        attributes: &[&[f32]],
        segments: &GpuVoxelSegments,
        staging: &wgpu::Buffer,
        mode: XyzMode,
        upload_recycle: &mut Vec<wgpu::Buffer>,
    ) -> SpatialResult<bool> {
        let Some(plan) = plan_wide_voxel_first(
            &runtime.device().limits(),
            segments.cell_count(),
            attributes.len(),
            runtime.max_gather_channels() as usize,
            staging.size(),
            isize::MAX as u64,
        ) else {
            return Ok(false);
        };
        record_wide_voxel_first(
            encoder,
            runtime,
            xyz,
            attributes,
            segments,
            staging,
            mode,
            upload_recycle,
            plan,
        )?;
        Ok(true)
    }

    // Keep borrowed inputs, destinations and the checked plan explicit at this recording boundary.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_wide_voxel_first(
        encoder: &mut wgpu::CommandEncoder,
        runtime: &WgpuRuntime,
        xyz: [&wgpu::Buffer; 3],
        attributes: &[&[f32]],
        segments: &GpuVoxelSegments,
        staging: &wgpu::Buffer,
        mode: XyzMode,
        upload_recycle: &mut Vec<wgpu::Buffer>,
        plan: WideFirstPlan,
    ) -> SpatialResult<()> {
        let first_upload = upload_recycle.len();
        for channel in &attributes[..4] {
            upload_recycle
                .push(runtime.upload_f32_storage("voxel-gather-xyz-attrs-values", channel)?);
        }
        let values = &upload_recycle[first_upload..first_upload + 4];
        let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-wide-first-prefix-output"),
            size: plan.prefix_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        match mode {
            XyzMode::First => {
                record_fused_voxel_first_gather(encoder, runtime, xyz, values, segments, &output);
            }
            XyzMode::Centroid => {
                record_fused_voxel_centroid_first(encoder, runtime, xyz, values, segments, &output);
            }
        }
        encoder.copy_buffer_to_buffer(&output, 0, staging, 0, plan.prefix_bytes);
        record_packed_voxel_attributes(
            encoder,
            runtime,
            &attributes[4..],
            segments,
            staging,
            upload_recycle,
            plan.tail,
        )
    }
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

    if let Some(result) = crate::kernels::voxel_gather_submit::try_gather_voxel_channels_coalesced(
        runtime, channels, segments,
    ) {
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

pub(super) fn baseline_gather_voxel_first_xyz_and_multi_gpu(
    runtime: &WgpuRuntime,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z: &wgpu::Buffer,
    attribute_channels: &[&[f32]],
    u8_attribute_channels: &[&[u8]],
    segments: &GpuVoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<u8>>)> {
    let attribute_count = attribute_channels.len();
    let u8_attribute_count = u8_attribute_channels.len();
    if segments.cell_count() == 0 {
        return Ok((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![Vec::new(); attribute_count],
            vec![Vec::new(); u8_attribute_count],
        ));
    }

    let point_count = segments.point_count() as usize;
    for channel in attribute_channels {
        if channel.len() != point_count {
            return Err(SpatialError::BufferLengthMismatch {
                expected: point_count,
                found: channel.len(),
            });
        }
    }
    for channel in u8_attribute_channels {
        if channel.len() != point_count {
            return Err(SpatialError::BufferLengthMismatch {
                expected: point_count,
                found: channel.len(),
            });
        }
    }

    let device = runtime.device();
    let queue = runtime.queue();
    let cell_count = segments.cell_count();
    let cells = cell_count as usize;
    let channel_len = cells * std::mem::size_of::<f32>();
    let f32_channel_count = 3 + attribute_count;
    let u8_staging_len = packed_u8_output_staging_bytes(cells, u8_attribute_count);
    let u8_channel_len = packed_u8_output_staging_bytes(cells, 1);
    let fused_plan = plan_fused_voxel_average(&device.limits(), cell_count, attribute_count);
    let staging_size = channel_len * f32_channel_count + u8_staging_len;
    let fused_xyz_attrs4 = attribute_count == 4
        && u8_attribute_count == 0
        && runtime.pipelines().voxel_gather.xyz_attrs4_pipeline.is_some();

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-xyz-attrs-staging"),
        size: staging_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-gather-xyz-attrs-encoder"),
    });
    let mut upload_recycle = Vec::new();

    if fused_xyz_attrs4 {
        for channel in attribute_channels {
            upload_recycle
                .push(runtime.upload_f32_storage("voxel-gather-xyz-attrs4-values", channel)?);
        }
        let attr_refs: [&wgpu::Buffer; MULTI4_CHANNELS] =
            std::array::from_fn(|index| &upload_recycle[index]);
        let packed_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-xyz-attrs4-packed-output"),
            size: (channel_len * f32_channel_count) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        record_voxel_gather_xyz_and_attrs4_packed_pass(
            &mut encoder,
            runtime,
            x,
            y,
            z,
            &attr_refs,
            segments,
            &packed_output,
        )?;
        encoder.copy_buffer_to_buffer(
            &packed_output,
            0,
            &staging_buffer,
            0,
            (channel_len * f32_channel_count) as u64,
        );
    } else if let Some(plan) = fused_plan {
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-fused-first-gather-output"),
            size: plan.output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        for channel in attribute_channels {
            upload_recycle
                .push(runtime.upload_f32_storage("voxel-gather-xyz-attrs-values", channel)?);
        }
        record_fused_voxel_first_gather(
            &mut encoder,
            runtime,
            [x, y, z],
            &upload_recycle,
            segments,
            &output,
        );
        encoder.copy_buffer_to_buffer(&output, 0, &staging_buffer, 0, plan.output_bytes);
    } else if !voxel_wide_first::try_record_wide_voxel_first(
        &mut encoder,
        runtime,
        [x, y, z],
        attribute_channels,
        segments,
        &staging_buffer,
        crate::kernels::voxel_xyz_packed::XyzMode::First,
        &mut upload_recycle,
    )? {
        if !crate::kernels::voxel_xyz_packed::try_record_packed_xyz_to_staging(
            &mut encoder,
            runtime,
            [x, y, z],
            segments,
            &staging_buffer,
            0,
            crate::kernels::voxel_xyz_packed::XyzMode::First,
        ) {
            let output_x = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-gather-xyz-out-x"),
                size: channel_len as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let output_y = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-gather-xyz-out-y"),
                size: channel_len as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let output_z = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-gather-xyz-out-z"),
                size: channel_len as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            record_voxel_gather_xyz_pass(
                &mut encoder,
                runtime,
                x,
                y,
                z,
                segments,
                &output_x,
                &output_y,
                &output_z,
            )?;

            encoder.copy_buffer_to_buffer(&output_x, 0, &staging_buffer, 0, channel_len as u64);
            encoder.copy_buffer_to_buffer(
                &output_y,
                0,
                &staging_buffer,
                channel_len as u64,
                channel_len as u64,
            );
            encoder.copy_buffer_to_buffer(
                &output_z,
                0,
                &staging_buffer,
                (channel_len * 2) as u64,
                channel_len as u64,
            );
        }

        record_gather_f32_attribute_channels_to_staging(
            &mut encoder,
            runtime,
            attribute_channels,
            segments,
            &staging_buffer,
            channel_len as u64,
            &mut upload_recycle,
        )?;
    }

    let u8_region_offset = (channel_len * f32_channel_count) as u64;
    if !crate::kernels::voxel_u8_batch::try_record_voxel_u8_batch_to_staging(
        &mut encoder,
        runtime,
        u8_attribute_channels,
        segments,
        &staging_buffer,
        u8_region_offset,
        false,
    ) {
        for (attribute_index, channel) in u8_attribute_channels.iter().enumerate() {
            let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel-gather-xyz-u8-values"),
                contents: &u8_storage_upload_contents(channel),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-gather-xyz-u8-output"),
                size: u8_channel_len as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            record_voxel_gather_u8_packed_pass(
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
                u8_region_offset + attribute_index as u64 * u8_channel_len as u64,
                u8_channel_len as u64,
            );
        }
    }

    queue.submit(Some(encoder.finish()));

    let output = read_staging_xyz_and_packed_attributes(
        device,
        &staging_buffer,
        cells,
        attribute_count,
        u8_attribute_count,
    )?;
    for buffer in upload_recycle {
        runtime.recycle_storage(buffer.size(), buffer);
    }
    Ok(output)
}

pub(super) fn baseline_reduce_voxel_centroids_xyz_and_gather_first_multi_gpu(
    runtime: &WgpuRuntime,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z: &wgpu::Buffer,
    attribute_channels: &[&[f32]],
    u8_attribute_channels: &[&[u8]],
    segments: &GpuVoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<u8>>)> {
    use crate::kernels::voxel_gather::record_gather_f32_attribute_channels_to_staging;
    use crate::kernels::voxel_gather::record_voxel_gather_u8_packed_pass;

    let attribute_count = attribute_channels.len();
    let u8_attribute_count = u8_attribute_channels.len();
    if segments.cell_count() == 0 {
        return Ok((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![Vec::new(); attribute_count],
            vec![Vec::new(); u8_attribute_count],
        ));
    }

    let point_count = segments.point_count() as usize;
    for channel in attribute_channels {
        if channel.len() != point_count {
            return Err(SpatialError::BufferLengthMismatch {
                expected: point_count,
                found: channel.len(),
            });
        }
    }
    for channel in u8_attribute_channels {
        if channel.len() != point_count {
            return Err(SpatialError::BufferLengthMismatch {
                expected: point_count,
                found: channel.len(),
            });
        }
    }

    let device = runtime.device();
    let queue = runtime.queue();
    let cell_count = segments.cell_count();
    let cells = cell_count as usize;
    let channel_len = cells * std::mem::size_of::<f32>();
    let f32_channel_count = 3 + attribute_count;
    let u8_staging_len = packed_u8_output_staging_bytes(cells, u8_attribute_count);
    let u8_channel_len = packed_u8_output_staging_bytes(cells, 1);
    let fused_plan = plan_fused_voxel_average(&device.limits(), cell_count, attribute_count);
    let staging_size = channel_len * f32_channel_count + u8_staging_len;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-xyz-gather-attrs-staging"),
        size: staging_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-reduce-xyz-gather-attrs-encoder"),
    });
    let mut upload_recycle = Vec::new();
    if let Some(plan) = fused_plan {
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-fused-centroid-first-output"),
            size: plan.output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        for channel in attribute_channels {
            upload_recycle
                .push(runtime.upload_f32_storage("voxel-gather-xyz-attrs-values", channel)?);
        }
        record_fused_voxel_centroid_first(
            &mut encoder,
            runtime,
            [x, y, z],
            &upload_recycle,
            segments,
            &output,
        );
        encoder.copy_buffer_to_buffer(&output, 0, &staging_buffer, 0, plan.output_bytes);
    } else if !voxel_wide_first::try_record_wide_voxel_first(
        &mut encoder,
        runtime,
        [x, y, z],
        attribute_channels,
        segments,
        &staging_buffer,
        crate::kernels::voxel_xyz_packed::XyzMode::Centroid,
        &mut upload_recycle,
    )? {
        if !crate::kernels::voxel_xyz_packed::try_record_packed_xyz_to_staging(
            &mut encoder,
            runtime,
            [x, y, z],
            segments,
            &staging_buffer,
            0,
            crate::kernels::voxel_xyz_packed::XyzMode::Centroid,
        ) {
            let output_x = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-reduce-xyz-out-x"),
                size: channel_len as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let output_y = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-reduce-xyz-out-y"),
                size: channel_len as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let output_z = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-reduce-xyz-out-z"),
                size: channel_len as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            record_voxel_reduce_xyz_pass(
                &mut encoder,
                runtime,
                x,
                y,
                z,
                segments.point_indices_buffer(),
                segments.cell_starts_buffer(),
                cell_count,
                segments.point_count(),
                &output_x,
                &output_y,
                &output_z,
            )?;
            encoder.copy_buffer_to_buffer(&output_x, 0, &staging_buffer, 0, channel_len as u64);
            encoder.copy_buffer_to_buffer(
                &output_y,
                0,
                &staging_buffer,
                channel_len as u64,
                channel_len as u64,
            );
            encoder.copy_buffer_to_buffer(
                &output_z,
                0,
                &staging_buffer,
                (channel_len * 2) as u64,
                channel_len as u64,
            );
        }

        record_gather_f32_attribute_channels_to_staging(
            &mut encoder,
            runtime,
            attribute_channels,
            segments,
            &staging_buffer,
            channel_len as u64,
            &mut upload_recycle,
        )?;
    }

    let u8_region_offset = (channel_len * f32_channel_count) as u64;
    if !crate::kernels::voxel_u8_batch::try_record_voxel_u8_batch_to_staging(
        &mut encoder,
        runtime,
        u8_attribute_channels,
        segments,
        &staging_buffer,
        u8_region_offset,
        false,
    ) {
        for (attribute_index, channel) in u8_attribute_channels.iter().enumerate() {
            let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel-gather-xyz-u8-values"),
                contents: &u8_storage_upload_contents(channel),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-gather-xyz-u8-output"),
                size: u8_channel_len as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            record_voxel_gather_u8_packed_pass(
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
                u8_region_offset + attribute_index as u64 * u8_channel_len as u64,
                u8_channel_len as u64,
            );
        }
    }

    queue.submit(Some(encoder.finish()));
    let output = read_staging_xyz_and_packed_attributes(
        device,
        &staging_buffer,
        cells,
        attribute_count,
        u8_attribute_count,
    )?;
    for buffer in upload_recycle {
        runtime.recycle_storage(buffer.size(), buffer);
    }
    Ok(output)
}

pub(crate) fn record_gather_f32_attribute_channels_to_staging(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    attribute_channels: &[&[f32]],
    segments: &GpuVoxelSegments,
    staging_buffer: &wgpu::Buffer,
    channel_len: u64,
    upload_recycle: &mut Vec<wgpu::Buffer>,
) -> SpatialResult<()> {
    if attribute_channels.is_empty() {
        return Ok(());
    }

    if voxel_gather_packed::try_record_packed_voxel_attributes(
        encoder,
        runtime,
        attribute_channels,
        segments,
        staging_buffer,
        channel_len * 3,
        upload_recycle,
    )? {
        return Ok(());
    }

    let device = runtime.device();
    let max_channels = runtime.max_gather_channels().max(1) as usize;

    for chunk_start in (0..attribute_channels.len()).step_by(max_channels) {
        let chunk_end = (chunk_start + max_channels).min(attribute_channels.len());
        let chunk = &attribute_channels[chunk_start..chunk_end];
        let channels_in_chunk = chunk.len();

        let chunk_upload_start = upload_recycle.len();
        for channel in chunk {
            upload_recycle
                .push(runtime.upload_f32_storage("voxel-gather-xyz-attrs-values", channel)?);
        }
        let value_buffers = &upload_recycle[chunk_upload_start..];

        let output_buffers: Vec<wgpu::Buffer> = (0..channels_in_chunk)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("voxel-gather-xyz-attrs-output"),
                    size: channel_len,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect();

        match channels_in_chunk {
            1 => record_voxel_gather_f32_pass(
                encoder,
                runtime,
                &value_buffers[0],
                segments.point_indices_buffer(),
                segments.cell_starts_buffer(),
                segments.cell_count(),
                segments.point_count(),
                &output_buffers[0],
            )?,
            2 => {
                let values: [&wgpu::Buffer; MULTI2_CHANNELS] =
                    [&value_buffers[0], &value_buffers[1]];
                let outputs: [&wgpu::Buffer; MULTI2_CHANNELS] =
                    [&output_buffers[0], &output_buffers[1]];
                record_voxel_gather_multi2_pass(encoder, runtime, &values, segments, &outputs, 2)?;
            }
            channel_count => {
                if runtime.pipelines().voxel_gather.multi4_pipeline.is_some() {
                    let empty = empty_storage_buffer(device)?;
                    let dummy_outputs: [wgpu::Buffer; MULTI4_CHANNELS] =
                        std::array::from_fn(|_| {
                            device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("voxel-gather-xyz-attrs-dummy-output"),
                                size: channel_len,
                                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                                mapped_at_creation: false,
                            })
                        });
                    let values: [&wgpu::Buffer; MULTI4_CHANNELS] = std::array::from_fn(|index| {
                        if index < channel_count {
                            &value_buffers[index]
                        } else {
                            &empty
                        }
                    });
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
                } else {
                    for (local_index, channel) in chunk.iter().enumerate() {
                        let chunk_upload_start = upload_recycle.len();
                        upload_recycle.push(
                            runtime.upload_f32_storage("voxel-gather-xyz-attrs-values", channel)?,
                        );
                        record_voxel_gather_f32_pass(
                            encoder,
                            runtime,
                            &upload_recycle[chunk_upload_start],
                            segments.point_indices_buffer(),
                            segments.cell_starts_buffer(),
                            segments.cell_count(),
                            segments.point_count(),
                            &output_buffers[local_index],
                        )?;
                    }
                }
            }
        }

        for (local_index, output_buffer) in output_buffers.iter().enumerate() {
            encoder.copy_buffer_to_buffer(
                output_buffer,
                0,
                staging_buffer,
                channel_len * (3 + chunk_start + local_index) as u64,
                channel_len,
            );
        }
    }

    Ok(())
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
