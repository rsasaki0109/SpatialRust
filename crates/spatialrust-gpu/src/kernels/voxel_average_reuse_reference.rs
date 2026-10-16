// Complete parents and both batch recorders frozen from 1a6b61551fecd4c8eca4b63c6e82c4a7dfe65028.
use super::*;
use crate::kernels::voxel_first_average_fused::record_fused_voxel_first_average;
use crate::kernels::voxel_gather::record_voxel_gather_xyz_pass;
use crate::kernels::voxel_reduce::{
    record_voxel_reduce_f32_pass, record_voxel_reduce_u8_packed_pass, record_voxel_reduce_xyz_pass,
};
use crate::kernels::voxel_reduce_fused::{plan_fused_voxel_average, record_fused_voxel_average};
use crate::kernels::{
    voxel_first_average_fused, voxel_reduce, voxel_reduce_fused, voxel_xyz_packed,
};
use crate::readback::{
    packed_u8_output_staging_bytes, read_staging_f32_channels,
    read_staging_xyz_and_packed_attributes, u8_storage_upload_contents,
};

pub(super) mod voxel_average_batch {
    use bytemuck::{Pod, Zeroable};
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
                super::voxel_reduce::record_voxel_reduce_f32_pass(
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
        crate::readback::read_staging_f32_channels(
            device,
            &staging,
            channels.len(),
            segments.cell_count() as usize,
        )
    }

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct Uniform {
        cells: u32,
        points: u32,
        channels: u32,
        _pad: u32,
    }

    pub(crate) fn record_voxel_average_batch(
        encoder: &mut wgpu::CommandEncoder,
        runtime: &WgpuRuntime,
        values: &[wgpu::Buffer],
        segments: &GpuVoxelSegments,
        output: &wgpu::Buffer,
    ) {
        let device = runtime.device();
        let (layout, pipeline) = runtime.pipelines().voxel_reduce.multi_average_pipeline(device);
        let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-reduce-multi-packed-uniform"),
            contents: bytemuck::bytes_of(&Uniform {
                cells: segments.cell_count(),
                points: segments.point_count(),
                channels: values.len() as u32,
                _pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // Inactive read-only slots alias the first input; the shader skips them.
        let value = |i| values.get(i).unwrap_or(&values[0]);
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
            label: Some("voxel-reduce-multi-packed-group"),
            layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-reduce-multi-packed-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(segments.cell_count().div_ceil(256), 1, 1);
    }

    pub(crate) fn plan_voxel_attribute_average_batch(
        limits: &wgpu::Limits,
        cells: u32,
        channels: usize,
        target_bytes: u64,
        offset: u64,
    ) -> Option<AverageBatchPlan> {
        plan_voxel_attribute_average_batch_with_host_limit(
            limits,
            cells,
            channels,
            target_bytes,
            offset,
            isize::MAX as u64,
        )
    }

    pub(crate) fn plan_voxel_attribute_average_batch_with_host_limit(
        limits: &wgpu::Limits,
        cells: u32,
        channels: usize,
        target_bytes: u64,
        offset: u64,
        host_limit: u64,
    ) -> Option<AverageBatchPlan> {
        let plan = plan_voxel_average_batch_with_host_limit(limits, cells, channels, host_limit)?;
        if offset % 4 != 0
            || target_bytes > limits.max_buffer_size
            || target_bytes > host_limit
            || offset.checked_add(plan.staging_bytes)? > target_bytes
        {
            return None;
        }
        Some(plan)
    }

    pub(crate) fn try_record_voxel_attribute_averages_to_staging(
        encoder: &mut wgpu::CommandEncoder,
        runtime: &WgpuRuntime,
        channels: &[&[f32]],
        segments: &GpuVoxelSegments,
        staging: &wgpu::Buffer,
        offset: u64,
    ) -> SpatialResult<bool> {
        if channels.len() < 2 {
            return Ok(false);
        }
        let Some(plan) = plan_voxel_attribute_average_batch(
            &runtime.device().limits(),
            segments.cell_count(),
            channels.len(),
            staging.size(),
            offset,
        ) else {
            return Ok(false);
        };
        record_voxel_attribute_averages_to_staging(
            encoder,
            runtime,
            channels,
            segments,
            staging,
            offset,
            plan.channel_bytes,
        )?;
        Ok(true)
    }

    pub(crate) fn record_voxel_attribute_averages_to_staging(
        encoder: &mut wgpu::CommandEncoder,
        runtime: &WgpuRuntime,
        channels: &[&[f32]],
        segments: &GpuVoxelSegments,
        staging: &wgpu::Buffer,
        offset: u64,
        channel_bytes: u64,
    ) -> SpatialResult<()> {
        let device = runtime.device();
        for (batch, chunk) in channels.chunks(4).enumerate() {
            let values: Vec<_> = chunk
                .iter()
                .map(|channel| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("voxel-reduce-xyz-attrs-values"),
                        contents: bytemuck::cast_slice(channel),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
                })
                .collect();
            let bytes = channel_bytes * chunk.len() as u64;
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-reduce-xyz-attrs-packed-output"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            if chunk.len() == 1 {
                super::voxel_reduce::record_voxel_reduce_f32_pass(
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
                record_voxel_average_batch(encoder, runtime, &values, segments, &output);
            }
            encoder.copy_buffer_to_buffer(
                &output,
                0,
                staging,
                offset + channel_bytes * (batch * 4) as u64,
                bytes,
            );
        }
        Ok(())
    }
}

mod voxel_wide_average {
    use spatialrust_core::SpatialResult;
    use wgpu::util::DeviceExt;

    use super::{
        voxel_average_batch::{
            plan_voxel_attribute_average_batch_with_host_limit,
            record_voxel_attribute_averages_to_staging,
        },
        voxel_first_average_fused::record_fused_voxel_first_average,
        voxel_reduce_fused::{plan_fused_voxel_average, record_fused_voxel_average},
        voxel_xyz_packed::XyzMode,
    };
    use crate::{GpuVoxelSegments, WgpuRuntime};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct WideAveragePlan {
        pub prefix_bytes: u64,
        pub channel_bytes: u64,
    }

    pub(crate) fn plan_wide_voxel_average(
        limits: &wgpu::Limits,
        cells: u32,
        attributes: usize,
        target_bytes: u64,
        host_limit: u64,
    ) -> Option<WideAveragePlan> {
        if attributes <= 4
            || limits.max_bind_groups == 0
            || limits.max_uniform_buffers_per_shader_stage == 0
            || limits.max_uniform_buffer_binding_size < 16
            || limits.max_buffer_size < 16
            || limits.max_compute_workgroup_size_y == 0
            || limits.max_compute_workgroup_size_z == 0
            || target_bytes > limits.max_buffer_size
            || target_bytes > host_limit
        {
            return None;
        }
        let prefix = plan_fused_voxel_average(limits, cells, 4)?;
        let channel_bytes = u64::from(cells).checked_mul(4)?;
        let float_bytes =
            channel_bytes.checked_mul(u64::try_from(attributes).ok()?.checked_add(3)?)?;
        if float_bytes > target_bytes || prefix.output_bytes > host_limit {
            return None;
        }
        if attributes > 5 {
            plan_voxel_attribute_average_batch_with_host_limit(
                limits,
                cells,
                attributes - 4,
                target_bytes,
                prefix.output_bytes,
                host_limit,
            )?;
        }
        Some(WideAveragePlan { prefix_bytes: prefix.output_bytes, channel_bytes })
    }

    pub(crate) fn try_record_wide_voxel_average(
        encoder: &mut wgpu::CommandEncoder,
        runtime: &WgpuRuntime,
        xyz: [&wgpu::Buffer; 3],
        attributes: &[&[f32]],
        segments: &GpuVoxelSegments,
        staging: &wgpu::Buffer,
        mode: XyzMode,
    ) -> SpatialResult<bool> {
        let Some(plan) = plan_wide_voxel_average(
            &runtime.device().limits(),
            segments.cell_count(),
            attributes.len(),
            staging.size(),
            isize::MAX as u64,
        ) else {
            return Ok(false);
        };
        let device = runtime.device();
        let values: Vec<_> = attributes[..4]
            .iter()
            .map(|channel| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel-reduce-xyz-attrs-values"),
                    contents: bytemuck::cast_slice(channel),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .collect();
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-wide-average-prefix-output"),
            size: plan.prefix_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        match mode {
            XyzMode::First => {
                record_fused_voxel_first_average(encoder, runtime, xyz, &values, segments, &output);
            }
            XyzMode::Centroid => {
                record_fused_voxel_average(encoder, runtime, xyz, &values, segments, &output);
            }
        }
        encoder.copy_buffer_to_buffer(&output, 0, staging, 0, plan.prefix_bytes);
        record_voxel_attribute_averages_to_staging(
            encoder,
            runtime,
            &attributes[4..],
            segments,
            staging,
            plan.prefix_bytes,
            plan.channel_bytes,
        )?;
        Ok(true)
    }
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

pub(super) fn baseline_reduce_voxel_centroids_xyz_and_average_multi_gpu(
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

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-xyz-attrs-staging"),
        size: staging_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-reduce-xyz-attrs-encoder"),
    });

    if let Some(plan) = fused_plan {
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-fused-average-output"),
            size: plan.output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let attributes: Vec<_> = attribute_channels
            .iter()
            .map(|channel| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel-reduce-xyz-attrs-values"),
                    contents: bytemuck::cast_slice(channel),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .collect();
        record_fused_voxel_average(
            &mut encoder,
            runtime,
            [x, y, z],
            &attributes,
            segments,
            &output,
        );
        encoder.copy_buffer_to_buffer(&output, 0, &staging_buffer, 0, plan.output_bytes);
    } else if !voxel_wide_average::try_record_wide_voxel_average(
        &mut encoder,
        runtime,
        [x, y, z],
        attribute_channels,
        segments,
        &staging_buffer,
        crate::kernels::voxel_xyz_packed::XyzMode::Centroid,
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

        if !voxel_average_batch::try_record_voxel_attribute_averages_to_staging(
            &mut encoder,
            runtime,
            attribute_channels,
            segments,
            &staging_buffer,
            (channel_len * 3) as u64,
        )? {
            for (attribute_index, channel) in attribute_channels.iter().enumerate() {
                let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel-reduce-xyz-attrs-values"),
                    contents: bytemuck::cast_slice(channel),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("voxel-reduce-xyz-attrs-output"),
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
                    (channel_len * (3 + attribute_index)) as u64,
                    channel_len as u64,
                );
            }
        }
    }

    let u8_region_offset = (channel_len * f32_channel_count) as u64;
    if !crate::kernels::voxel_u8_batch::try_record_voxel_u8_batch_to_staging(
        &mut encoder,
        runtime,
        u8_attribute_channels,
        segments,
        &staging_buffer,
        u8_region_offset,
        true,
    ) {
        for (attribute_index, channel) in u8_attribute_channels.iter().enumerate() {
            let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel-reduce-xyz-u8-values"),
                contents: &u8_storage_upload_contents(channel),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-reduce-xyz-u8-output"),
                size: u8_channel_len as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            record_voxel_reduce_u8_packed_pass(
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
    Ok(output)
}

pub(super) fn baseline_gather_voxel_first_xyz_and_average_multi_gpu(
    runtime: &WgpuRuntime,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z: &wgpu::Buffer,
    attribute_channels: &[&[f32]],
    u8_attribute_channels: &[&[u8]],
    segments: &GpuVoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<u8>>)> {
    use crate::kernels::voxel_reduce::record_voxel_reduce_f32_pass;
    use crate::kernels::voxel_reduce::record_voxel_reduce_u8_packed_pass;

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
        label: Some("voxel-gather-xyz-reduce-attrs-staging"),
        size: staging_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-gather-xyz-reduce-attrs-encoder"),
    });
    if let Some(plan) = fused_plan {
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-fused-first-average-output"),
            size: plan.output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let attributes: Vec<_> = attribute_channels
            .iter()
            .map(|channel| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel-reduce-xyz-attrs-values"),
                    contents: bytemuck::cast_slice(channel),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .collect();
        record_fused_voxel_first_average(
            &mut encoder,
            runtime,
            [x, y, z],
            &attributes,
            segments,
            &output,
        );
        encoder.copy_buffer_to_buffer(&output, 0, &staging_buffer, 0, plan.output_bytes);
    } else if !voxel_wide_average::try_record_wide_voxel_average(
        &mut encoder,
        runtime,
        [x, y, z],
        attribute_channels,
        segments,
        &staging_buffer,
        crate::kernels::voxel_xyz_packed::XyzMode::First,
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

        if !voxel_average_batch::try_record_voxel_attribute_averages_to_staging(
            &mut encoder,
            runtime,
            attribute_channels,
            segments,
            &staging_buffer,
            (channel_len * 3) as u64,
        )? {
            for (attribute_index, channel) in attribute_channels.iter().enumerate() {
                let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel-reduce-xyz-attrs-values"),
                    contents: bytemuck::cast_slice(channel),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("voxel-reduce-xyz-attrs-output"),
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
                    (channel_len * (3 + attribute_index)) as u64,
                    channel_len as u64,
                );
            }
        }
    }

    let u8_region_offset = (channel_len * f32_channel_count) as u64;
    if !crate::kernels::voxel_u8_batch::try_record_voxel_u8_batch_to_staging(
        &mut encoder,
        runtime,
        u8_attribute_channels,
        segments,
        &staging_buffer,
        u8_region_offset,
        true,
    ) {
        for (attribute_index, channel) in u8_attribute_channels.iter().enumerate() {
            let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel-reduce-xyz-u8-values"),
                contents: &u8_storage_upload_contents(channel),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-reduce-xyz-u8-output"),
                size: u8_channel_len as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            record_voxel_reduce_u8_packed_pass(
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
    Ok(output)
}
