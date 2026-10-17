use super::*;
use crate::{GpuStageProfiler, WgpuPowerPreference};

type Entry = [u32; 4];

fn buffer(runtime: &WgpuRuntime, words: &[u32]) -> wgpu::Buffer {
    runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("radix-proof-buffer"),
        contents: bytemuck::cast_slice(words),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn read_words(runtime: &WgpuRuntime, buffer: &wgpu::Buffer) -> Vec<u32> {
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    runtime.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    runtime.wait_idle();
    receiver.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let words = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    words
}

fn group(
    runtime: &WgpuRuntime,
    params: RadixParams,
    buffers: [&wgpu::Buffer; 5],
) -> wgpu::BindGroup {
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut entries =
        vec![wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }];
    entries.extend(buffers.into_iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry {
        binding: i as u32 + 1,
        resource: b.as_entire_binding(),
    }));
    runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &runtime.pipelines().voxel_radix.bind_group_layout,
        entries: &entries,
    })
}

struct Work {
    entries: [wgpu::Buffer; 2],
    histogram: wgpu::Buffer,
    offsets: wgpu::Buffer,
    ranks: wgpu::Buffer,
    groups: Vec<wgpu::BindGroup>,
    count: u32,
}

impl Work {
    fn new(runtime: &WgpuRuntime, input: &[Entry]) -> Self {
        let count = input.len() as u32;
        let group_count = count.div_ceil(256);
        let entries = std::array::from_fn(|_| buffer(runtime, bytemuck::cast_slice(input)));
        // Dirty storage makes missing histogram/offset/rank writes observable.
        let histogram = buffer(runtime, &vec![u32::MAX; group_count as usize * 16]);
        let offsets = buffer(runtime, &vec![u32::MAX; group_count as usize * 16]);
        let ranks = buffer(runtime, &vec![u32::MAX; count as usize]);
        let groups = (0..24)
            .map(|pass| {
                let axis = 2 - pass / 8;
                let shift = pass % 8 * 4;
                group(
                    runtime,
                    RadixParams { point_count: count, group_count, axis, shift },
                    [
                        &entries[pass as usize % 2],
                        &entries[(pass as usize + 1) % 2],
                        &histogram,
                        &offsets,
                        &ranks,
                    ],
                )
            })
            .collect();
        Self { entries, histogram, offsets, ranks, groups, count }
    }

    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: [&wgpu::ComputePipeline; 3],
        profiler: &mut Option<&mut GpuStageProfiler>,
    ) {
        for group in &self.groups {
            for (pipeline, count, label) in [
                (pipelines[0], self.count.div_ceil(256), "radix_histogram"),
                (pipelines[1], 16, "radix_prefix"),
                (pipelines[2], self.count.div_ceil(256), "radix_scatter"),
            ] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: crate::profiling::timestamp_writes(profiler, label),
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(count, 1, 1);
            }
        }
    }
}

fn inputs(count: usize, pattern: usize) -> Vec<Entry> {
    let mut seed = 0x27182818_u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    (0..count)
        .map(|i| {
            let coords = match pattern {
                0 => {
                    if i % 7 == 0 {
                        [u32::MAX, 0, 0x80000000]
                    } else {
                        [next(), next(), next()]
                    }
                }
                1 => [u32::MAX, 7, i32::MAX as u32],
                _ => [((count - i) % 4096) as u32, (i % 17) as u32, (i % 7) as u32],
            };
            [coords[0], coords[1], coords[2], i as u32]
        })
        .collect()
}

fn sorted(input: &[Entry]) -> Vec<Entry> {
    let mut expected = input.to_vec();
    expected.sort_unstable_by_key(|e| (e[0] as i32, e[1] as i32, e[2] as i32, e[3]));
    expected
}

#[test]
fn radix_histogram_publishes_exact_local_earlier_digit_counts_and_ranks() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let input = inputs(65_537, 0);
    let work = Work::new(&runtime, &input);
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        // Force the optimized specialization to inspect its intermediate data.
        pass.set_pipeline(
            runtime.pipelines().voxel_radix.sort_pipelines(runtime.device(), u32::MAX)[0],
        );
        pass.set_bind_group(0, &work.groups[0], &[]);
        pass.dispatch_workgroups(work.count.div_ceil(256), 1, 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
    let groups = input.len().div_ceil(256);
    let mut counts = vec![0; groups * 16];
    let mut earlier = vec![0; groups * 16];
    let mut ranks = Vec::new();
    for (g, chunk) in input.chunks(256).enumerate() {
        let mut local = [0; 16];
        for e in chunk {
            let digit = ((e[2] ^ 0x80000000) & 15) as usize;
            ranks.push(local[digit]);
            local[digit] += 1;
        }
        let mut prefix = 0;
        for (b, n) in local.into_iter().enumerate() {
            counts[b * groups + g] = n;
            earlier[b * groups + g] = prefix;
            prefix += n;
        }
        assert_eq!(prefix, chunk.len() as u32);
    }
    assert_eq!(read_words(&runtime, &work.histogram), counts);
    assert_eq!(read_words(&runtime, &work.offsets), earlier);
    assert_eq!(read_words(&runtime, &work.ranks), ranks);
}

#[test]
fn radix_prefix_matches_independent_offsets_through_maximum_u32_mass() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let dummy: [wgpu::Buffer; 3] = std::array::from_fn(|_| buffer(&runtime, &[0; 4]));
    for groups in [1, 3, 255, 256, 257, 513, 3907, 32769] {
        for pattern in 0..4 {
            let mut histogram = vec![0; groups * 16];
            let last = histogram.len() - 1;
            match pattern {
                0 => histogram[0] = u32::MAX,
                1 => histogram[last] = u32::MAX,
                2 => {
                    let base = u32::MAX / histogram.len() as u32;
                    histogram.fill(base);
                    histogram[last] += u32::MAX % histogram.len() as u32;
                }
                _ => {
                    let mut remaining = u32::MAX;
                    for (i, n) in histogram.iter_mut().enumerate() {
                        *n = remaining.min((i.wrapping_mul(982_451_653) % 65536) as u32);
                        remaining -= *n;
                    }
                    histogram[last] += remaining;
                }
            }
            let mut earlier = vec![0; histogram.len()];
            for g in 0..groups {
                let mut total = 0_u64;
                for b in 0..16 {
                    earlier[b * groups + g] = total as u32;
                    total += u64::from(histogram[b * groups + g]);
                }
            }
            let mut total = 0_u64;
            let expected: Vec<u32> = histogram
                .iter()
                .map(|&n| {
                    let offset = total as u32;
                    total += u64::from(n);
                    offset
                })
                .collect();
            assert_eq!(total, u64::from(u32::MAX));
            let histogram = buffer(&runtime, &histogram);
            let offsets = buffer(&runtime, &earlier);
            let group = group(
                &runtime,
                RadixParams { point_count: 1, group_count: groups as u32, axis: 0, shift: 0 },
                [&dummy[0], &dummy[1], &histogram, &offsets, &dummy[2]],
            );
            let mut encoder = runtime.device().create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(
                    runtime.pipelines().voxel_radix.sort_pipelines(runtime.device(), u32::MAX)[1],
                );
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(16, 1, 1);
            }
            runtime.queue().submit(Some(encoder.finish()));
            assert_eq!(
                read_words(&runtime, &offsets),
                expected,
                "groups {groups}, pattern {pattern}"
            );
        }
    }
}

#[test]
fn radix_sort_preserves_device_order_with_stale_scratch_and_live_outputs() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut retained = Vec::new();
    for count in [257, 65_537, 524_288, 524_289, 1_000_003] {
        for pattern in 0..3 {
            let input = inputs(count, pattern);
            let expected = sorted(&input);
            let entries = sort_entries_radix_gpu(
                &runtime,
                buffer(&runtime, bytemuck::cast_slice(&input)),
                count as u32,
                None,
            );
            assert_eq!(
                read_words(&runtime, &entries),
                bytemuck::cast_slice::<Entry, u32>(&expected)
            );
            retained.push((entries, expected));
        }
    }
    let before = runtime.scratch_buffer_pool_stats().allocation_count;
    let input = inputs(1_000_003, 0);
    let result = sort_entries_radix_gpu(
        &runtime,
        buffer(&runtime, bytemuck::cast_slice(&input)),
        input.len() as u32,
        None,
    );
    assert_eq!(read_words(&runtime, &result), bytemuck::cast_slice::<Entry, u32>(&sorted(&input)));
    assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count, before);
    for (output, expected) in retained {
        assert_eq!(read_words(&runtime, &output), bytemuck::cast_slice::<Entry, u32>(&expected));
    }
}

#[path = "voxel_radix_benchmark.rs"]
mod benchmark;

#[path = "voxel_rank_tests.rs"]
mod rank_tests;

#[test]
fn cached_radix_pipelines_initialize_once_across_concurrent_sorts() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let radix = &runtime.pipelines().voxel_radix;
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|worker| {
                let runtime = &runtime;
                scope.spawn(move || {
                    // Force the private specialization to race its first creation.
                    let pipelines = radix.sort_pipelines(runtime.device(), u32::MAX);
                    let input = inputs(65_537, worker % 3);
                    let work = Work::new(runtime, &input);
                    let mut encoder = runtime.device().create_command_encoder(&Default::default());
                    work.encode(&mut encoder, pipelines, &mut None);
                    runtime.queue().submit(Some(encoder.finish()));
                    assert_eq!(
                        read_words(runtime, &work.entries[0]),
                        bytemuck::cast_slice::<Entry, u32>(&sorted(&input))
                    );
                    [pipelines[0] as *const _ as usize, pipelines[1] as *const _ as usize]
                })
            })
            .collect();
        let addresses: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
        assert!(addresses.iter().all(|&a| a == addresses[0]));
    });
}

#[path = "voxel_radix_four_tests.rs"]
mod four;
