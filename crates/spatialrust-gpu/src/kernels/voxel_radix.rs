//! Stable LSD radix sorting of signed, lexicographically ordered voxel keys.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::runtime::WgpuRuntime;

const WORKGROUP_SIZE: u32 = 256;
const RADIX: u64 = 16;
const ALL_DIGITS: u32 = (1 << 24) - 1;

#[path = "voxel_radix_controls.rs"]
mod controls;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RadixParams {
    point_count: u32,
    group_count: u32,
    axis: u32,
    shift: u32,
}

// Sort owned entries whose source point indices are already initialized.
pub(super) fn sort_entries_radix_gpu(
    runtime: &WgpuRuntime,
    entries: wgpu::Buffer,
    point_count: u32,
    profiler: Option<&mut crate::GpuStageProfiler>,
) -> wgpu::Buffer {
    sort_radix_gpu(runtime, entries, point_count, None, profiler, ALL_DIGITS)
}

// CPU-derived masks include every varying nibble and an optional constant
// nibble to keep ping-pong parity even. The uploaded entries remain the output.
pub(super) fn sort_entries_radix_gpu_with_digits(
    runtime: &WgpuRuntime,
    entries: wgpu::Buffer,
    point_count: u32,
    digits: u32,
    profiler: Option<&mut crate::GpuStageProfiler>,
) -> wgpu::Buffer {
    debug_assert_eq!(digits & !ALL_DIGITS, 0);
    debug_assert_eq!(digits.count_ones() % 2, 0);
    if digits == ALL_DIGITS {
        return sort_entries_radix_gpu(runtime, entries, point_count, profiler);
    }
    sort_radix_gpu(runtime, entries, point_count, None, profiler, digits)
}

/// Caller validates a nonempty key buffer and handles the singleton separately.
pub(super) fn sort_keys_radix_gpu(
    runtime: &WgpuRuntime,
    keys: &wgpu::Buffer,
    point_count: u32,
    profiler: Option<&mut crate::GpuStageProfiler>,
) -> wgpu::Buffer {
    debug_assert!(point_count > 1);
    let entries = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-sort-entries-build"),
        size: u64::from(point_count) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    sort_radix_gpu(runtime, entries, point_count, Some(keys), profiler, ALL_DIGITS)
}

fn sort_radix_gpu(
    runtime: &WgpuRuntime,
    entries: wgpu::Buffer,
    point_count: u32,
    keys: Option<&wgpu::Buffer>,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
    digits: u32,
) -> wgpu::Buffer {
    debug_assert!(keys.is_none() || digits == ALL_DIGITS);
    if point_count <= 1 || digits == 0 {
        return entries;
    }

    let device = runtime.device();
    let pipelines = &runtime.pipelines().voxel_radix;
    let group_count = point_count.div_ceil(WORKGROUP_SIZE);
    let [histogram_pipeline, prefix_pipeline, scatter_pipeline] =
        pipelines.sort_pipelines(device, group_count);
    let mut temporaries = crate::scratch::ScratchBuffers::new(runtime);
    let scratch = temporaries.acquire("voxel-radix-scratch", entries.size());
    let histograms =
        temporaries.acquire("voxel-radix-histograms", u64::from(group_count) * RADIX * 4);
    let offsets = temporaries.acquire("voxel-radix-offsets", histograms.size());
    let ranks = temporaries.acquire("voxel-radix-ranks", u64::from(point_count) * 4);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-radix-encoder"),
    });
    let mut input = keys.unwrap_or(&entries);
    let mut output = &scratch;
    let controls = controls::RadixControls::new(device, point_count, group_count);

    // Entry construction supplies increasing point indices. Stable passes on
    // z, y, x preserve that order for equal keys without sorting a fourth word.
    for axis in (0..3).rev() {
        for shift in (0..32).step_by(4) {
            if digits & (1 << ((2 - axis) * 8 + shift / 4)) == 0 {
                continue;
            }
            let params = RadixParams { point_count, group_count, axis, shift };
            let uniform = controls.is_none().then(|| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel-radix-params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            });
            let control = match &controls {
                Some(batch) => batch.binding(axis, shift),
                None => uniform.as_ref().expect("individual radix control").as_entire_binding(),
            };
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("voxel-radix-bind-group"),
                layout: &pipelines.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: control },
                    wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: histograms.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: offsets.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: ranks.as_entire_binding() },
                ],
            });
            // The first digit reads keys directly. Its scatter replaces the
            // padding word with the increasing source point index.
            let scatter = if keys.is_some() && axis == 2 && shift == 0 {
                pipelines.build_scatter(device)
            } else {
                scatter_pipeline
            };
            // Separate dispatches provide device-wide producer/consumer
            // ordering. All selected digits are submitted without readback.
            for (pipeline, groups, stage) in [
                (histogram_pipeline, group_count, "voxel-radix-histogram"),
                (prefix_pipeline, RADIX as u32, "voxel-radix-prefix"),
                (scatter, group_count, "voxel-radix-scatter"),
            ] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("voxel-radix-pass"),
                    timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, stage),
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            if axis == 2 && shift == 0 {
                // The borrowed keys never become a writable ping-pong target.
                input = &scratch;
                output = &entries;
            } else {
                std::mem::swap(&mut input, &mut output);
            }
        }
    }
    temporaries.submit_and_recycle(encoder);

    // An even number of selected digits finishes in the original allocation.
    entries
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_radix_tests.rs"]
mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_radix_build_tests.rs"]
mod build;
