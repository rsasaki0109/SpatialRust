// Complete immediate-parent sort call; only the pipeline selection is external.
use super::*;

pub(super) fn previous(
    prior: &ParentPipelines,
    runtime: &WgpuRuntime,
    entries: wgpu::Buffer,
    point_count: u32,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
) -> wgpu::Buffer {
    if point_count <= 1 {
        return entries;
    }

    let device = runtime.device();
    let pipelines = &runtime.pipelines().voxel_radix;
    let group_count = point_count.div_ceil(WORKGROUP_SIZE);
    let [histogram_pipeline, prefix_pipeline, scatter_pipeline] = prior.for_groups(group_count);
    let mut temporaries = crate::scratch::ScratchBuffers::new(runtime);
    let scratch = temporaries.acquire("voxel-radix-scratch", entries.size());
    let histograms =
        temporaries.acquire("voxel-radix-histograms", u64::from(group_count) * RADIX * 4);
    let offsets = temporaries.acquire("voxel-radix-offsets", histograms.size());
    let ranks = temporaries.acquire("voxel-radix-ranks", u64::from(point_count) * 4);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-radix-encoder"),
    });
    let mut input = &entries;
    let mut output = &scratch;

    // Entry construction supplies increasing point indices. Stable passes on
    // z, y, x preserve that order for equal keys without sorting a fourth word.
    for axis in (0..3).rev() {
        for shift in (0..32).step_by(4) {
            let params = RadixParams { point_count, group_count, axis, shift };
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel-radix-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("voxel-radix-bind-group"),
                layout: &pipelines.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: histograms.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: offsets.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: ranks.as_entire_binding() },
                ],
            });
            // Separate dispatches provide device-wide producer/consumer
            // ordering. All 24 digits are submitted together without readback.
            for (pipeline, groups, stage) in [
                (histogram_pipeline, group_count, "voxel-radix-histogram"),
                (prefix_pipeline, RADIX as u32, "voxel-radix-prefix"),
                (scatter_pipeline, group_count, "voxel-radix-scatter"),
            ] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("voxel-radix-pass"),
                    timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, stage),
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            std::mem::swap(&mut input, &mut output);
        }
    }
    temporaries.submit_and_recycle(encoder);

    // Twenty-four ping-pong passes finish in the original allocation.
    entries
}
