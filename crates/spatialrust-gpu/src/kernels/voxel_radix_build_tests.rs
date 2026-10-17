use super::*;
use crate::{GpuStageProfiler, WgpuPowerPreference};

#[path = "voxel_radix_build_reference.rs"]
mod old_radix;
#[path = "voxel_radix_build_sort_reference.rs"]
mod old_sort;
use crate::kernels::voxel_sort::{
    build_voxel_segments_gpu, build_voxel_segments_gpu_from_keys_buffer,
    build_voxel_segments_gpu_from_keys_buffer_with_profiler,
};

const POISON: u32 = 0xdead_beef;
type Entry = [u32; 4];

fn buffer(r: &WgpuRuntime, words: &[u32]) -> wgpu::Buffer {
    r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(words),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn read_words(r: &WgpuRuntime, source: &wgpu::Buffer) -> Vec<u32> {
    let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = r.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    r.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    r.wait_idle();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

fn keys(n: usize, pattern: usize) -> Vec<Entry> {
    let mut seed = 0x2718_2818u32;
    (0..n)
        .map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let xyz = match pattern {
                0 => [seed, seed.rotate_left(11), seed.rotate_left(23)],
                1 => [i32::MIN as u32, u32::MAX, i32::MAX as u32],
                _ => [((n - i) % 257) as u32, (i % 7) as u32, (i % 3) as u32],
            };
            // Key padding must never be mistaken for a source point index.
            [xyz[0], xyz[1], xyz[2], seed ^ POISON]
        })
        .collect()
}

fn expected_entries(keys: &[Entry]) -> Vec<Entry> {
    let mut expected: Vec<_> =
        keys.iter().enumerate().map(|(i, e)| [e[0], e[1], e[2], i as u32]).collect();
    expected.sort_unstable_by_key(|e| (e[0] as i32, e[1] as i32, e[2] as i32, e[3]));
    expected
}

fn cpu_segments(keys: &[Entry]) -> crate::VoxelSegments {
    let coords: Vec<_> = keys
        .iter()
        .map(|e| (i64::from(e[0] as i32), i64::from(e[1] as i32), i64::from(e[2] as i32)))
        .collect();
    crate::kernels::voxel_segments::build_voxel_segments(&coords)
}

fn group(
    r: &WgpuRuntime,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let mut entries =
        vec![wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }];
    entries.extend(buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry {
        binding: i as u32 + 1,
        resource: b.as_entire_binding(),
    }));
    r.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &entries,
    })
}

struct Raw {
    input: wgpu::Buffer,
    entries: wgpu::Buffer,
    unused: wgpu::Buffer,
    histograms: wgpu::Buffer,
    offsets: wgpu::Buffer,
    ranks: wgpu::Buffer,
    build: wgpu::BindGroup,
    histogram: wgpu::BindGroup,
}

fn raw(
    r: &WgpuRuntime,
    radix: &old_radix::ParentRadixPipelines,
    build: &old_radix::ParentBuildPipelines,
    data: &[Entry],
    fused: bool,
) -> Raw {
    let n = data.len();
    let mut source = bytemuck::cast_slice(data).to_vec();
    source.extend([POISON; 8]);
    let input = buffer(r, &source);
    let entries = buffer(r, &vec![POISON; n * 4 + 8]);
    let unused = buffer(r, &vec![POISON; n * 4 + 8]);
    let histograms = buffer(r, &vec![POISON; n.div_ceil(256) * 16 + 8]);
    let offsets = buffer(r, &vec![POISON; n.div_ceil(256) * 16 + 8]);
    let ranks = buffer(r, &vec![POISON; n + 8]);
    let params = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&RadixParams {
            point_count: n as u32,
            group_count: n.div_ceil(256) as u32,
            axis: 2,
            shift: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let build_params = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[n as u32, n as u32, 0, 0]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let build_group = group(r, &build.bind_group_layout, &build_params, &[&input, &entries]);
    let histogram = group(
        r,
        if fused { &r.pipelines().voxel_radix.bind_group_layout } else { &radix.bind_group_layout },
        &params,
        &[if fused { &input } else { &entries }, &unused, &histograms, &offsets, &ranks],
    );
    Raw { input, entries, unused, histograms, offsets, ranks, build: build_group, histogram }
}

#[allow(clippy::too_many_arguments)]
fn encode_raw(
    r: &WgpuRuntime,
    radix: &old_radix::ParentRadixPipelines,
    build: &old_radix::ParentBuildPipelines,
    raw: &Raw,
    n: usize,
    cached: bool,
    fused: bool,
    encoder: &mut wgpu::CommandEncoder,
    profiler: &mut Option<&mut GpuStageProfiler>,
) {
    if n == 0 {
        return;
    }
    let groups = n.div_ceil(256) as u32;
    let selector = if cached { u32::MAX } else { 1 };
    if !fused {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: crate::profiling::timestamp_writes(profiler, "voxel-sort-build"),
        });
        pass.set_pipeline(&build.pipeline);
        pass.set_bind_group(0, &raw.build, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    let pipelines = radix.sort_pipelines(r.device(), selector);
    let current = r.pipelines().voxel_radix.sort_pipelines(r.device(), selector);
    for (pipeline, groups, stage) in [
        (if fused { current[0] } else { pipelines[0] }, groups, "voxel-radix-histogram"),
        (if fused { current[1] } else { pipelines[1] }, 16, "voxel-radix-prefix"),
        (
            if fused { r.pipelines().voxel_radix.build_scatter(r.device()) } else { pipelines[2] },
            groups,
            "voxel-radix-scatter",
        ),
    ] {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: crate::profiling::timestamp_writes(profiler, stage),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &raw.histogram, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

fn verify_raw(r: &WgpuRuntime, raw: &Raw, data: &[Entry], fused: bool) -> usize {
    let mut source = bytemuck::cast_slice(data).to_vec();
    source.extend([POISON; 8]);
    assert_eq!(read_words(r, &raw.input), source);
    let mut expected: Vec<u32> =
        data.iter().enumerate().flat_map(|(i, e)| [e[0], e[1], e[2], i as u32]).collect();
    expected.extend([POISON; 8]);
    assert_eq!(
        read_words(r, &raw.entries),
        if fused { vec![POISON; expected.len()] } else { expected.clone() }
    );
    let mut scattered: Vec<Entry> =
        data.iter().enumerate().map(|(i, e)| [e[0], e[1], e[2], i as u32]).collect();
    scattered.sort_by_key(|e| e[2] & 15);
    let mut scattered: Vec<u32> = bytemuck::cast_slice(&scattered).to_vec();
    scattered.extend([POISON; 8]);
    assert_eq!(read_words(r, &raw.unused), scattered);
    let groups = data.len().div_ceil(256);
    let mut histogram = vec![POISON; groups * 16 + 8];
    let mut offsets = histogram.clone();
    let mut ranks = Vec::new();
    for (g, chunk) in data.chunks(256).enumerate() {
        let mut counts = [0; 16];
        for e in chunk {
            let bucket = (e[2] & 15) as usize;
            ranks.push(counts[bucket]);
            counts[bucket] += 1;
        }
        for (bucket, count) in counts.into_iter().enumerate() {
            histogram[bucket * groups + g] = count;
        }
    }
    let mut total = 0u64;
    for (offset, &count) in offsets[..groups * 16].iter_mut().zip(&histogram[..groups * 16]) {
        *offset = u32::try_from(total).unwrap();
        total += u64::from(count);
    }
    assert_eq!(total, data.len() as u64);
    ranks.extend([POISON; 8]);
    assert_eq!(read_words(r, &raw.histograms), histogram);
    assert_eq!(read_words(r, &raw.offsets), offsets);
    assert_eq!(read_words(r, &raw.ranks), ranks);
    source.len() + expected.len() + scattered.len() + histogram.len() + offsets.len() + ranks.len()
}

#[test]
fn voxel_radix_build_first_digit_matches_frozen_parent_and_cpu_in_both_specializations() {
    let r = WgpuRuntime::new_headless().unwrap();
    let radix = old_radix::ParentRadixPipelines::new(r.device());
    let build = old_radix::ParentBuildPipelines::new(r.device());
    let mut checked = 0;
    let mut pairs = 0;
    for n in [1, 2, 3, 255, 256, 257, 511, 65_537, 524_288, 524_289, 1_000_003] {
        for pattern in 0..3 {
            let data = keys(n, pattern);
            for cached in [false, true] {
                for fused in [false, true] {
                    let raw = raw(&r, &radix, &build, &data, fused);
                    let mut encoder = r.device().create_command_encoder(&Default::default());
                    encode_raw(&r, &radix, &build, &raw, n, cached, fused, &mut encoder, &mut None);
                    r.queue().submit(Some(encoder.finish()));
                    checked += verify_raw(&r, &raw, &data, fused);
                }
                pairs += 1;
            }
        }
    }
    println!("VOXEL_RADIX_BUILD_RAW checked {checked} entry/first-scatter/histogram/rank/offset/source/guard words across {pairs} paired fixtures; both cached and original rank paths.");
}

#[test]
fn voxel_radix_build_complete_sort_preserves_raw_stable_order_and_owned_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    let radix = old_radix::ParentRadixPipelines::new(r.device());
    let build = old_radix::ParentBuildPipelines::new(r.device());
    let mut retained = Vec::new();
    let mut checked = 0;
    let mut pairs = 0;
    for n in [0, 1, 2, 255, 256, 257, 65_537, 524_288, 524_289, 1_000_003] {
        for pattern in 0..3 {
            let data = keys(n, pattern);
            let source =
                buffer(&r, if n == 0 { &[POISON; 4] } else { bytemuck::cast_slice(&data) });
            let expected = expected_entries(&data);
            if n > 1 {
                let parent = old_sort::build_sort_entries_from_keys_gpu(
                    &radix, &build, &r, &source, n as u32, None,
                );
                let parent = old_radix::sort_entries_radix_gpu(&radix, &r, parent, n as u32, None);
                let current = sort_keys_radix_gpu(&r, &source, n as u32, None);
                assert_eq!(read_words(&r, &current), bytemuck::cast_slice::<Entry, u32>(&expected));
                assert_eq!(read_words(&r, &parent), read_words(&r, &current));
                retained.push((current, expected.clone()));
                checked += n * 8;
            }
            let before = old_sort::build_voxel_segments_gpu_from_keys_buffer(
                &radix,
                &build,
                &r,
                &source,
                n as u32,
                (n as u32).next_power_of_two(),
            )
            .unwrap();
            let after = build_voxel_segments_gpu_from_keys_buffer(
                &r,
                &source,
                n as u32,
                (n as u32).next_power_of_two(),
            )
            .unwrap();
            assert_eq!(before.to_voxel_segments(&r).unwrap(), cpu_segments(&data));
            assert_eq!(after.to_voxel_segments(&r).unwrap(), cpu_segments(&data));
            if n != 0 {
                let raw_indices: Vec<u32> = expected.iter().map(|e| e[3]).collect();
                assert_eq!(read_words(&r, before.point_indices_buffer()), raw_indices);
                assert_eq!(read_words(&r, after.point_indices_buffer()), raw_indices);
            }
            assert_eq!(before.construction_readback_bytes(), if n == 0 { 0 } else { 4 });
            assert_eq!(before.construction_readback_bytes(), after.construction_readback_bytes());
            assert_eq!(
                read_words(&r, &source),
                if n == 0 {
                    vec![POISON; 4]
                } else {
                    bytemuck::cast_slice::<Entry, u32>(&data).to_vec()
                }
            );
            pairs += 1;
        }
    }
    r.clear_scratch_buffer_pool();
    for (output, expected) in retained {
        assert_eq!(read_words(&r, &output), bytemuck::cast_slice::<Entry, u32>(&expected));
        checked += expected.len() * 4;
    }
    let extreme = [(i64::MIN, -1, i64::MAX), (i64::MAX, 0, i64::MIN), (i64::MIN, -1, i64::MAX)];
    assert_eq!(
        build_voxel_segments_gpu(&r, &extreme).unwrap(),
        old_sort::build_voxel_segments_gpu(&radix, &build, &r, &extreme).unwrap()
    );
    println!("VOXEL_RADIX_BUILD_FULL checked {checked} raw stable sort words and {pairs} complete parent/CPU API pairs; signed extremes, padding, singleton/empty and retained outputs verified.");
}

#[test]
fn voxel_radix_build_profiles_storage_reuse_validation_and_zero_budgets() {
    let r =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    let radix = old_radix::ParentRadixPipelines::new(r.device());
    let build = old_radix::ParentBuildPipelines::new(r.device());
    let mut removed = 0;
    for n in [0, 1, 2, 257, 524_289] {
        let data = keys(n, 2);
        let source = buffer(&r, if n == 0 { &[0; 4] } else { bytemuck::cast_slice(&data) });
        let capacity = if n == 0 {
            0
        } else {
            1 + if n > 1 { 72 } else { 1 } + crate::kernels::voxel_scan::scan_pass_count(n as u32)
        };
        let mut reports = Vec::new();
        let mut stats = Vec::new();
        for current in [false, true] {
            r.clear_scratch_buffer_pool();
            let mut profiler = r.timestamp_queries_enabled().then(|| {
                GpuStageProfiler::new(&r, (capacity + u32::from(!current && n > 1)).max(1)).unwrap()
            });
            let output = if current {
                build_voxel_segments_gpu_from_keys_buffer_with_profiler(
                    &r,
                    &source,
                    n as u32,
                    n as u32,
                    profiler.as_mut(),
                )
                .unwrap()
            } else {
                old_sort::build_voxel_segments_gpu_from_keys_buffer_with_profiler(
                    &radix,
                    &build,
                    &r,
                    &source,
                    n as u32,
                    n as u32,
                    profiler.as_mut(),
                )
                .unwrap()
            };
            stats.push(r.scratch_buffer_pool_stats());
            reports.push(profiler.map(|p| p.collect(&r).unwrap()));
            assert_eq!(output.to_voxel_segments(&r).unwrap(), cpu_segments(&data));
        }
        assert_eq!(stats[0].cached_bytes, stats[1].cached_bytes);
        assert_eq!(stats[0].cached_resources, stats[1].cached_resources);
        if let [Some(before), Some(after)] = reports.as_slice() {
            let extra = usize::from(n > 1);
            assert_eq!(before.stages.len(), capacity as usize + extra);
            assert_eq!(after.stages.len(), capacity as usize);
            assert_eq!(before.device_to_host_bytes, after.device_to_host_bytes + extra as u64 * 16);
            assert_eq!(
                before
                    .stages
                    .iter()
                    .map(|s| s.stage)
                    .filter(|&s| n <= 1 || s != "voxel-sort-build")
                    .collect::<Vec<_>>(),
                after.stages.iter().map(|s| s.stage).collect::<Vec<_>>()
            );
            removed += extra * 16;
        }
        let warm = r.scratch_buffer_pool_stats().allocation_count;
        let _ = build_voxel_segments_gpu_from_keys_buffer(&r, &source, n as u32, n as u32).unwrap();
        assert_eq!(warm, r.scratch_buffer_pool_stats().allocation_count);
        for budget in [4, 0] {
            r.set_scratch_buffer_pool_budget(budget);
            let output =
                build_voxel_segments_gpu_from_keys_buffer(&r, &source, n as u32, n as u32).unwrap();
            assert_eq!(output.to_voxel_segments(&r).unwrap(), cpu_segments(&data));
            assert!(r.scratch_buffer_pool_stats().cached_bytes <= budget);
        }
        r.set_scratch_buffer_pool_budget(crate::resource_cache::DEFAULT_GPU_POOL_BUDGET_BYTES);
    }
    let source = buffer(&r, &[POISON; 8]);
    let invalid_usage = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    for (input, count, padded) in
        [(&source, 2, 1), (&source, 3, 3), (&invalid_usage, 2, 2), (&source, u32::MAX, u32::MAX)]
    {
        let stats = r.scratch_buffer_pool_stats();
        let before = old_sort::build_voxel_segments_gpu_from_keys_buffer(
            &radix, &build, &r, input, count, padded,
        )
        .err()
        .unwrap()
        .to_string();
        let after = build_voxel_segments_gpu_from_keys_buffer(&r, input, count, padded)
            .err()
            .unwrap()
            .to_string();
        assert_eq!(before, after);
        assert_eq!(stats.allocation_count, r.scratch_buffer_pool_stats().allocation_count);
        assert_eq!(stats.reuse_count, r.scratch_buffer_pool_stats().reuse_count);
    }
    println!("VOXEL_RADIX_BUILD_STORAGE verified {removed} optional query bytes removed, identical scratch storage, warmed reuse, zero budgets and unchanged preflight errors.");
}

#[test]
fn voxel_radix_build_pipeline_variants_initialize_once_under_concurrent_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|worker| {
                let r = &r;
                scope.spawn(move || {
                    let pipelines = &r.pipelines().voxel_radix;
                    let addresses =
                        [0, 1].map(|_| pipelines.build_scatter(r.device()) as *const _ as usize);
                    let data = keys(65_537, worker % 3);
                    let source = buffer(r, bytemuck::cast_slice(&data));
                    let output = sort_keys_radix_gpu(r, &source, data.len() as u32, None);
                    assert_eq!(
                        read_words(r, &output),
                        bytemuck::cast_slice::<Entry, u32>(&expected_entries(&data))
                    );
                    addresses
                })
            })
            .collect();
        let addresses: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
        assert!(addresses.iter().all(|a| *a == addresses[0]));
        assert_eq!(addresses[0][0], addresses[0][1]);
    });
    println!("VOXEL_RADIX_BUILD_CACHE verified one immutable scatter specialization and four concurrent full sorts.");
}

#[path = "voxel_radix_build_benchmark.rs"]
mod benchmark;
