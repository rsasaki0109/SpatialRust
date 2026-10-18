use super::*;
use crate::kernels::voxel_radix::{sort_entries_radix_gpu, sort_keys_radix_gpu};
use crate::kernels::voxel_sort::{
    build_voxel_segments_gpu, build_voxel_segments_gpu_from_keys_buffer,
    build_voxel_segments_gpu_from_keys_buffer_with_profiler,
};
use crate::{GpuStageProfiler, VoxelSegments, WgpuPowerPreference, WgpuRuntime};

#[path = "voxel_radix_controls_reference.rs"]
mod old_radix;
#[path = "voxel_radix_controls_sort_reference.rs"]
mod old_sort;

type Entry = [u32; 4];
const POISON: u32 = 0xdead_beef;

fn buffer(r: &WgpuRuntime, words: &[u32]) -> wgpu::Buffer {
    r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
    let mut seed = 0x3141_5926u32;
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
            [xyz[0], xyz[1], xyz[2], seed ^ POISON]
        })
        .collect()
}

fn cpu_keys(data: &[Entry]) -> Vec<(i64, i64, i64)> {
    data.iter()
        .map(|e| (i64::from(e[0] as i32), i64::from(e[1] as i32), i64::from(e[2] as i32)))
        .collect()
}

fn expected_entries(data: &[Entry]) -> Vec<Entry> {
    let mut rows: Vec<_> =
        data.iter().enumerate().map(|(i, e)| [e[0], e[1], e[2], i as u32]).collect();
    rows.sort_unstable_by_key(|e| (e[0] as i32, e[1] as i32, e[2] as i32, e[3]));
    rows
}

fn expected(data: &[Entry]) -> VoxelSegments {
    crate::kernels::voxel_segments::build_voxel_segments(&cpu_keys(data))
}

#[test]
fn radix_controls_layout_preserves_all_words_padding_limits_and_fallbacks() {
    assert_eq!(PARAM_BYTES, 16);
    let mut checked = 0;
    for alignment in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
        let limits =
            wgpu::Limits { min_uniform_buffer_offset_alignment: alignment, ..Default::default() };
        let plan = layout(&limits).unwrap();
        assert_eq!(plan.stride % u64::from(alignment), 0);
        assert_eq!(plan.bytes as u64, plan.stride * 23 + 16);
        assert!(plan.bytes as u64 <= MAX_BATCH_BYTES);
        for (n, groups) in [(0, 0), (1, 1), (257, 2), (u32::MAX, u32::MAX.div_ceil(256))] {
            let bytes = packed(plan, n, groups);
            for (digit, chunk) in bytes.chunks(plan.stride as usize).enumerate() {
                let expected = [n, groups, 2 - digit as u32 / 8, digit as u32 % 8 * 4];
                assert_eq!(&chunk[..16], bytemuck::cast_slice(&expected));
                assert!(chunk[16..].iter().all(|&byte| byte == 0));
                checked += 4;
            }
        }
        let bounded = wgpu::Limits { max_buffer_size: plan.bytes as u64, ..limits.clone() };
        assert_eq!(layout(&bounded), Some(plan));
        assert!(
            layout(&wgpu::Limits { max_buffer_size: plan.bytes as u64 - 1, ..bounded }).is_none()
        );
        assert!(layout(&wgpu::Limits { max_uniform_buffer_binding_size: 15, ..limits }).is_none());
    }
    for alignment in [0, 3, 4096, 1 << 31, u32::MAX] {
        assert!(layout(&wgpu::Limits {
            min_uniform_buffer_offset_alignment: alignment,
            max_buffer_size: u64::MAX,
            ..Default::default()
        })
        .is_none());
    }
    println!("RADIX_CONTROLS_LAYOUT verified {checked} independent control words, zero padding, exact buffer/binding/alignment boundaries and bounded fallbacks.");
}

#[test]
fn radix_controls_gpu_reads_every_immutable_slice_after_owner_drop() {
    let r = WgpuRuntime::new_headless().unwrap();
    let shader = r.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<uniform> p:vec4<u32>;
             @group(0) @binding(1) var<storage,read_write> words:array<vec4<u32>>;
             @compute @workgroup_size(1) fn main() {
               let digit=(2u-p.z)*8u+p.w/4u; words[digit]=p;
             }"
            .into(),
        ),
    });
    let pipeline = r.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut checked = 0;
    for (n, groups) in [(2, 1), (257, 2), (1_000_003, 3907), (u32::MAX, 16_777_216)] {
        for current in [false, true] {
            let output = buffer(&r, &[POISON; 100]);
            let batch = current.then(|| RadixControls::new(r.device(), n, groups).unwrap());
            let mut encoder = r.device().create_command_encoder(&Default::default());
            for digit in 0..24u32 {
                let params = super::super::RadixParams {
                    point_count: n,
                    group_count: groups,
                    axis: 2 - digit / 8,
                    shift: digit % 8 * 4,
                };
                let uniform = (!current).then(|| {
                    r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
                });
                let control = match &batch {
                    Some(batch) => batch.binding(params.axis, params.shift),
                    None => uniform.as_ref().unwrap().as_entire_binding(),
                };
                let group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: control },
                        wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            drop(batch);
            r.queue().submit(Some(encoder.finish()));
            let mut expected: Vec<_> =
                (0..24).flat_map(|digit| [n, groups, 2 - digit / 8, digit % 8 * 4]).collect();
            expected.extend([POISON; 4]);
            assert_eq!(read_words(&r, &output), expected);
            checked += expected.len();
        }
    }
    println!("RADIX_CONTROLS_BINDINGS verified {checked} parent/current GPU control/guard words with all owners dropped before submission.");
}

#[test]
fn radix_controls_complete_parent_cpu_and_raw_sorts_preserve_stable_owned_words() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut retained = Vec::new();
    let mut checked = 0;
    let mut cases = 0;
    for n in [0, 1, 2, 255, 256, 257, 65_537, 524_288, 524_289, 1_000_003] {
        for pattern in 0..3 {
            let data = keys(n, pattern);
            let mut words = bytemuck::cast_slice::<Entry, u32>(&data).to_vec();
            words.extend([POISON; 4]);
            let source = buffer(&r, &words);
            let sorted = expected_entries(&data);
            if n > 1 {
                let parent = old_radix::sort_keys_radix_gpu(&r, &source, n as u32, None);
                let current = sort_keys_radix_gpu(&r, &source, n as u32, None);
                assert_eq!(read_words(&r, &parent), bytemuck::cast_slice::<Entry, u32>(&sorted));
                assert_eq!(read_words(&r, &current), read_words(&r, &parent));
                let mut entries: Vec<u32> = data
                    .iter()
                    .enumerate()
                    .flat_map(|(i, e)| [e[0], e[1], e[2], i as u32])
                    .collect();
                entries.extend([POISON; 4]);
                let mut oracle = bytemuck::cast_slice::<Entry, u32>(&sorted).to_vec();
                oracle.extend([POISON; 4]);
                for active in [false, true] {
                    let input = buffer(&r, &entries);
                    let output = if active {
                        sort_entries_radix_gpu(&r, input, n as u32, None)
                    } else {
                        old_radix::sort_entries_radix_gpu(&r, input, n as u32, None)
                    };
                    assert_eq!(read_words(&r, &output), oracle);
                    checked += oracle.len();
                }
                checked += n * 8;
                if n <= 65_537 {
                    retained.push((current, sorted));
                }
            }
            let cpu = expected(&data);
            let parent = old_sort::build_voxel_segments_gpu_from_keys_buffer(
                &r,
                &source,
                n as u32,
                (n as u32).next_power_of_two(),
            )
            .unwrap();
            let current = build_voxel_segments_gpu_from_keys_buffer(
                &r,
                &source,
                n as u32,
                (n as u32).next_power_of_two(),
            )
            .unwrap();
            assert_eq!(parent.to_voxel_segments(&r).unwrap(), cpu);
            assert_eq!(current.to_voxel_segments(&r).unwrap(), cpu);
            assert_eq!(parent.construction_readback_bytes(), if n == 0 { 0 } else { 4 });
            assert_eq!(current.construction_readback_bytes(), parent.construction_readback_bytes());
            if n != 0 {
                let indices: Vec<_> = expected_entries(&data).iter().map(|e| e[3]).collect();
                assert_eq!(read_words(&r, current.point_indices_buffer()), indices);
                assert_eq!(read_words(&r, parent.point_indices_buffer()), indices);
                checked += n * 2;
            }
            let coords = cpu_keys(&data);
            assert_eq!(old_sort::build_voxel_segments_gpu(&r, &coords).unwrap(), cpu);
            assert_eq!(build_voxel_segments_gpu(&r, &coords).unwrap(), cpu);
            assert_eq!(read_words(&r, &source), words);
            cases += 1;
        }
    }
    r.clear_scratch_buffer_pool();
    for (output, oracle) in retained {
        assert_eq!(read_words(&r, &output), bytemuck::cast_slice::<Entry, u32>(&oracle));
        checked += oracle.len() * 4;
    }
    let extremes = [(i64::MIN, 0, i64::MAX), (i64::MAX, -1, i64::MIN), (i64::MIN, 0, i64::MAX)];
    assert_eq!(
        build_voxel_segments_gpu(&r, &extremes).unwrap(),
        old_sort::build_voxel_segments_gpu(&r, &extremes).unwrap()
    );
    println!("RADIX_CONTROLS_FULL verified {checked} raw parent/current/oracle stable words across {cases} complete CPU/device-key cases, both prefix families, untouched borrowed inputs and retained outputs.");
}

#[test]
fn radix_controls_profiles_pools_budgets_and_preflight_match_complete_parent() {
    let r =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    for n in [0, 1, 2, 257, 524_289] {
        let data = keys(n, 2);
        let source = buffer(&r, if n == 0 { &[0; 4] } else { bytemuck::cast_slice(&data) });
        let capacity = if n == 0 {
            0
        } else {
            1 + if n == 1 { 1 } else { 72 } + crate::kernels::voxel_scan::scan_pass_count(n as u32)
        };
        let mut reports = Vec::new();
        let mut stats = Vec::new();
        for current in [false, true] {
            r.clear_scratch_buffer_pool();
            let before = r.scratch_buffer_pool_stats();
            let mut profiler = GpuStageProfiler::new(&r, capacity.max(1)).unwrap();
            let output = if current {
                build_voxel_segments_gpu_from_keys_buffer_with_profiler(
                    &r,
                    &source,
                    n as u32,
                    n as u32,
                    Some(&mut profiler),
                )
            } else {
                old_sort::build_voxel_segments_gpu_from_keys_buffer_with_profiler(
                    &r,
                    &source,
                    n as u32,
                    n as u32,
                    Some(&mut profiler),
                )
            }
            .unwrap();
            let after = r.scratch_buffer_pool_stats();
            stats.push((
                after.cached_bytes,
                after.cached_resources,
                after.allocation_count - before.allocation_count,
            ));
            reports.push(profiler.collect(&r).unwrap());
            assert_eq!(output.to_voxel_segments(&r).unwrap(), expected(&data));
        }
        assert_eq!(stats[0], stats[1]);
        assert_eq!(reports[0].device_to_host_bytes, u64::from(capacity) * 16);
        assert_eq!(reports[0].device_to_host_bytes, reports[1].device_to_host_bytes);
        assert_eq!(
            reports[0].stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
            reports[1].stages.iter().map(|s| s.stage).collect::<Vec<_>>()
        );
        assert_eq!(reports[0].stages.len(), capacity as usize);
        let warm = r.scratch_buffer_pool_stats().allocation_count;
        let _ = build_voxel_segments_gpu_from_keys_buffer(&r, &source, n as u32, n as u32).unwrap();
        assert_eq!(warm, r.scratch_buffer_pool_stats().allocation_count);
        for budget in [4, 0] {
            r.set_scratch_buffer_pool_budget(budget);
            let output =
                build_voxel_segments_gpu_from_keys_buffer(&r, &source, n as u32, n as u32).unwrap();
            assert_eq!(output.to_voxel_segments(&r).unwrap(), expected(&data));
            assert!(r.scratch_buffer_pool_stats().cached_bytes <= budget);
        }
        r.set_scratch_buffer_pool_budget(crate::resource_cache::DEFAULT_GPU_POOL_BUDGET_BYTES);
    }
    let source = buffer(&r, &[POISON; 8]);
    let wrong = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    for (source, n, padded) in
        [(&source, 2, 1), (&source, 3, 3), (&wrong, 2, 2), (&source, u32::MAX, u32::MAX)]
    {
        let before = r.scratch_buffer_pool_stats();
        let old = old_sort::build_voxel_segments_gpu_from_keys_buffer(&r, source, n, padded)
            .err()
            .unwrap()
            .to_string();
        let new = build_voxel_segments_gpu_from_keys_buffer(&r, source, n, padded)
            .err()
            .unwrap()
            .to_string();
        assert_eq!(old, new);
        assert_eq!(before.allocation_count, r.scratch_buffer_pool_stats().allocation_count);
        assert_eq!(before.reuse_count, r.scratch_buffer_pool_stats().reuse_count);
    }
    println!("RADIX_CONTROLS_RESOURCES verified identical pass/query counts, scratch descriptors/allocation deltas, warm reuse, zero budgets and pre-allocation rejection.");
}

#[test]
fn radix_controls_concurrent_different_counts_keep_per_call_immutable_values() {
    let r = WgpuRuntime::new_headless().unwrap();
    let barrier = std::sync::Barrier::new(4);
    let retained = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..4)
            .map(|worker| {
                let r = &r;
                let barrier = &barrier;
                scope.spawn(move || {
                    let n = 65_537 + worker * 257;
                    let data = keys(n, worker % 3);
                    let input = buffer(r, bytemuck::cast_slice(&data));
                    barrier.wait();
                    let output = sort_keys_radix_gpu(r, &input, n as u32, None);
                    (output, expected_entries(&data))
                })
            })
            .collect();
        jobs.into_iter().map(|job| job.join().unwrap()).collect::<Vec<_>>()
    });
    r.clear_scratch_buffer_pool();
    let changed = keys(66_665, 1);
    let source = buffer(&r, bytemuck::cast_slice(&changed));
    let _ = sort_keys_radix_gpu(&r, &source, changed.len() as u32, None);
    for (output, oracle) in retained {
        assert_eq!(read_words(&r, &output), bytemuck::cast_slice::<Entry, u32>(&oracle));
    }
    println!("RADIX_CONTROLS_CONCURRENT verified four concurrent counts/patterns and independent live outputs after clear and changed-input reuse.");
}

#[path = "voxel_radix_controls_benchmark.rs"]
mod benchmark;
