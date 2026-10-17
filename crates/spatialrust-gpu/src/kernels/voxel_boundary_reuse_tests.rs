use super::*;
use crate::{kernels::voxel_segments::build_voxel_segments, GpuStageProfiler};

#[path = "voxel_boundary_reuse_reference.rs"]
mod old;

fn buffer(r: &WgpuRuntime, values: &[u32]) -> wgpu::Buffer {
    r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn read_words(r: &WgpuRuntime, source: &wgpu::Buffer) -> Vec<u32> {
    read_u32_buffer(r.device(), r.queue(), source, source.size() as usize / 4).unwrap()
}

fn entries(n: usize, run: usize) -> (Vec<[u32; 4]>, crate::VoxelSegments) {
    let mut data = Vec::with_capacity(n);
    let mut cell = 0usize;
    while data.len() < n {
        let length = if run == 0 { [1, 2, 3, 255, 256, 257, 1023, 4097][cell % 8] } else { run };
        let length = length.min(n - data.len());
        for _ in 0..length {
            data.push([
                (cell as i32 - 100) as u32,
                if cell % 2 == 0 { i32::MIN as u32 } else { i32::MAX as u32 },
                (-3i32) as u32,
                (n - data.len() - 1) as u32,
            ]);
        }
        cell += 1;
    }
    let keys: Vec<_> = data
        .iter()
        .map(|e| (i64::from(e[0] as i32), i64::from(e[1] as i32), i64::from(e[2] as i32)))
        .collect();
    let mut expected = build_voxel_segments(&keys);
    expected.point_indices = data.iter().map(|e| e[3]).collect();
    for (&start, &length) in expected.cell_starts.iter().zip(&expected.cell_counts) {
        expected.point_indices[start as usize..(start + length) as usize].sort_unstable();
    }
    (data, expected)
}

fn boundary_flags(data: &[[u32; 4]]) -> Vec<u32> {
    (0..data.len()).map(|i| u32::from(i == 0 || data[i][..3] != data[i - 1][..3])).collect()
}

fn prefix_oracle(data: &[[u32; 4]]) -> Vec<u32> {
    let mut total = 0u64;
    boundary_flags(data)
        .into_iter()
        .map(|v| {
            total += u64::from(v);
            u32::try_from(total).unwrap()
        })
        .collect()
}

struct RawDispatch {
    mark_group: wgpu::BindGroup,
    group: wgpu::BindGroup,
    keys: wgpu::Buffer,
    starts: wgpu::Buffer,
    indices: wgpu::Buffer,
    counts: wgpu::Buffer,
    flags: wgpu::Buffer,
    prefix: wgpu::Buffer,
    unused: wgpu::Buffer,
}

fn raw_dispatch(
    r: &WgpuRuntime,
    layout: &wgpu::BindGroupLayout,
    data: &[[u32; 4]],
    expected: &crate::VoxelSegments,
    current: bool,
    prepared: bool,
) -> RawDispatch {
    let n = data.len();
    let mut payload: Vec<u32> = bytemuck::cast_slice(data).to_vec();
    payload.extend([0xdead_beef; 8]);
    let input = buffer(r, &payload);
    let flags = if current {
        buffer(r, &[0xdead_beef])
    } else {
        let mut values = vec![0xdead_beef; n + 8];
        if prepared {
            values[..n].copy_from_slice(&boundary_flags(data));
        }
        buffer(r, &values)
    };
    let mut values = vec![0xdead_beef; n + 8];
    if prepared {
        values[..n].copy_from_slice(&prefix_oracle(data));
    }
    let prefix = buffer(r, &values);
    let keys = buffer(r, &vec![0xdead_beef; expected.keys.len() * 4 + 8]);
    let starts = buffer(r, &vec![0xdead_beef; expected.keys.len() + 8]);
    let indices = buffer(r, &vec![0xdead_beef; n + 8]);
    let counts = buffer(r, &[0xdead_beef]);
    let unused = buffer(r, &[0xdead_beef; 8]);
    let uniform = create_params_buffer(r.device(), n as u32, 0);
    let mark_group = create_compact_bind_group(
        r.device(),
        layout,
        &CompactBindResources {
            params: &uniform,
            entries: &input,
            flags: if current { &indices } else { &flags },
            prefix: &prefix,
            keys: &keys,
            starts: &starts,
            counts: &counts,
            indices: if current { &flags } else { &indices },
            scan_out: &unused,
        },
    );
    let group = create_compact_bind_group(
        r.device(),
        layout,
        &CompactBindResources {
            params: &uniform,
            entries: &input,
            flags: &flags,
            prefix: &prefix,
            keys: &keys,
            starts: &starts,
            counts: &counts,
            indices: &indices,
            scan_out: &unused,
        },
    );
    RawDispatch { mark_group, group, keys, starts, indices, counts, flags, prefix, unused }
}

fn assert_prefix(r: &WgpuRuntime, raw: &RawDispatch, data: &[[u32; 4]]) {
    let values = read_words(r, &raw.prefix);
    assert_eq!(&values[..data.len()], prefix_oracle(data));
    assert_eq!(&values[data.len()..], &[0xdead_beef; 8]);
}

fn checked_words(raw: &RawDispatch) -> usize {
    [&raw.keys, &raw.starts, &raw.indices, &raw.counts, &raw.flags, &raw.prefix, &raw.unused]
        .iter()
        .map(|b| b.size() as usize / 4)
        .sum()
}

fn verify_intermediate(
    r: &WgpuRuntime,
    raw: &RawDispatch,
    data: &[[u32; 4]],
    current: bool,
) -> usize {
    let n = data.len();
    let flags = read_words(r, if current { &raw.indices } else { &raw.flags });
    assert_eq!(&flags[..n], boundary_flags(data));
    assert_eq!(&flags[n..], &[0xdead_beef; 8]);
    if current {
        assert_eq!(raw.flags.size(), 4);
        assert_eq!(read_words(r, &raw.flags), [0xdead_beef]);
    } else {
        assert!(read_words(r, &raw.indices).iter().all(|&v| v == 0xdead_beef));
    }
    assert_prefix(r, raw, data);
    assert!(read_words(r, &raw.keys).iter().all(|&v| v == 0xdead_beef));
    assert!(read_words(r, &raw.starts).iter().all(|&v| v == 0xdead_beef));
    assert_eq!(read_words(r, &raw.counts), [0xdead_beef]);
    assert_eq!(read_words(r, &raw.unused), [0xdead_beef; 8]);
    checked_words(raw)
}

fn verify_raw(
    r: &WgpuRuntime,
    raw: &RawDispatch,
    data: &[[u32; 4]],
    expected: &crate::VoxelSegments,
    current: bool,
) -> usize {
    let n = data.len();
    let cells = expected.keys.len();
    let keys = read_words(r, &raw.keys);
    let expected_keys: Vec<u32> = expected
        .keys
        .iter()
        .flat_map(|&(x, y, z)| [x as i32 as u32, y as i32 as u32, z as i32 as u32, 0])
        .collect();
    assert!(keys[..cells * 4] == expected_keys);
    assert_eq!(&keys[cells * 4..], &[0xdead_beef; 8]);
    let starts = read_words(r, &raw.starts);
    assert_eq!(&starts[..cells], expected.cell_starts);
    assert_eq!(&starts[cells..], &[0xdead_beef; 8]);
    let indices = read_words(r, &raw.indices);
    assert!(indices[..n] == data.iter().map(|e| e[3]).collect::<Vec<_>>());
    assert_eq!(&indices[n..], &[0xdead_beef; 8]);
    if current {
        assert_eq!(raw.flags.size(), 4);
        assert_eq!(read_words(r, &raw.flags), [0xdead_beef]);
    } else {
        let flags = read_words(r, &raw.flags);
        assert_eq!(&flags[..n], boundary_flags(data));
        assert_eq!(&flags[n..], &[0xdead_beef; 8]);
    }
    assert_prefix(r, raw, data);
    assert_eq!(read_words(r, &raw.counts), [0xdead_beef]);
    assert_eq!(read_words(r, &raw.unused), [0xdead_beef; 8]);
    checked_words(raw)
}

#[test]
fn voxel_boundary_reuse_raw_mark_scan_write_match_parent_cpu_and_guards() {
    let r = WgpuRuntime::new_headless().unwrap();
    let prior = old::ParentCompactPipelines::new(r.device());
    let current = &r.pipelines().voxel_compact;
    let mut words = 0;
    let mut cases = 0;
    for n in [0, 1, 2, 3, 255, 256, 257, 511, 1025, 65537, 1_000_003] {
        for run in [1, 4, 257, n.max(1), 0] {
            let (data, expected) = entries(n, run);
            for (mark, write, layout, new) in [
                (&prior.mark, &prior.write, &prior.bind_group_layout, false),
                (&current.mark, &current.write, &current.bind_group_layout, true),
            ] {
                let raw = raw_dispatch(&r, layout, &data, &expected, new, false);
                let mut encoder = r.device().create_command_encoder(&Default::default());
                let mut temporaries = crate::scratch::ScratchBuffers::new(&r);
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(mark);
                    pass.set_bind_group(0, &raw.mark_group, &[]);
                    pass.dispatch_workgroups((n as u32).div_ceil(256).max(1), 1, 1);
                }
                crate::kernels::voxel_scan::encode_inclusive_scan(
                    &r,
                    &mut encoder,
                    if new { &raw.indices } else { &raw.flags },
                    &raw.prefix,
                    n as u32,
                    &mut None,
                    &mut temporaries,
                );
                temporaries.submit_and_recycle(encoder);
                words += verify_intermediate(&r, &raw, &data, new);
                let mut encoder = r.device().create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(write);
                    pass.set_bind_group(0, &raw.group, &[]);
                    pass.dispatch_workgroups((n as u32).div_ceil(256).max(1), 1, 1);
                }
                r.queue().submit(Some(encoder.finish()));
                words += verify_raw(&r, &raw, &data, &expected, new);
            }
            cases += 1;
        }
    }
    println!("VOXEL_BOUNDARY_REUSE_RAW verified {words} phased parent/CPU words and guards across {cases} paired fixtures; intermediate index flags/prefix, final indices and four-byte inactive poison agree.");
}

#[test]
fn voxel_boundary_reuse_complete_parent_cpu_preserve_indices_metadata_and_retained_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    let prior = old::ParentCompactPipelines::new(r.device());
    let mut words = 0;
    let mut retained = None;
    for n in [0, 1, 257, 65537, 1_000_003] {
        for run in [1, 4, 257, n.max(1), 0] {
            let (mut data, expected) = entries(n, run);
            data.push([0xdead_beef; 4]);
            let input = buffer(&r, bytemuck::cast_slice(&data));
            let before = old::previous(&prior, &r, &input, n as u32, None).unwrap();
            let after = compact_voxel_segments_gpu_buffers(&r, &input, n as u32).unwrap();
            assert_eq!(before.point_count(), after.point_count());
            assert_eq!(before.cell_count(), after.cell_count());
            assert_eq!(before.construction_readback_bytes(), after.construction_readback_bytes());
            assert_eq!(after.construction_readback_bytes(), if n == 0 { 0 } else { 4 });
            for output in [&before, &after] {
                if n > 0 {
                    assert!(
                        read_words(&r, output.point_indices_buffer())
                            == data[..n].iter().map(|e| e[3]).collect::<Vec<_>>()
                    );
                }
                assert!(output.to_voxel_segments(&r).unwrap() == expected);
            }
            words += 2 * (expected.point_indices.len() + expected.keys.len() * 5);
            if n == 257 && run == 257 {
                retained = Some((after, expected));
            }
            if let Some((ref output, ref expected)) = retained {
                assert!(output.to_voxel_segments(&r).unwrap() == *expected);
            }
        }
    }
    r.clear_scratch_buffer_pool();
    if let Some((output, expected)) = retained {
        assert!(output.to_voxel_segments(&r).unwrap() == expected);
    }
    println!("VOXEL_BOUNDARY_REUSE_PUBLIC verified {words} complete parent/CPU logical values across 25 paired fixtures, unchanged scalar receipts and retained outputs.");
}

#[test]
fn voxel_boundary_reuse_pool_bytes_profiling_zero_budget_and_stale_bindings() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    let prior = old::ParentCompactPipelines::new(r.device());
    let mut saved = 0u64;
    for n in [1, 257, 65537] {
        let (data, expected) = entries(n, 0);
        let input = buffer(&r, bytemuck::cast_slice(&data));
        let capacity = 2 + crate::kernels::voxel_scan::scan_pass_count(n as u32);
        r.clear_scratch_buffer_pool();
        let mut before_profiler =
            r.timestamp_queries_enabled().then(|| GpuStageProfiler::new(&r, capacity).unwrap());
        let before = old::previous(&prior, &r, &input, n as u32, before_profiler.as_mut()).unwrap();
        let before_stats = r.scratch_buffer_pool_stats();
        let before_report = before_profiler.map(|profiler| profiler.collect(&r).unwrap());
        r.clear_scratch_buffer_pool();
        let mut after_profiler =
            r.timestamp_queries_enabled().then(|| GpuStageProfiler::new(&r, capacity).unwrap());
        let after = compact_voxel_segments_gpu_buffers_with_profiler(
            &r,
            &input,
            n as u32,
            after_profiler.as_mut(),
        )
        .unwrap();
        let after_stats = r.scratch_buffer_pool_stats();
        let after_report = after_profiler.map(|profiler| profiler.collect(&r).unwrap());
        assert_eq!(before_stats.cached_bytes - after_stats.cached_bytes, 4 * n as u64 - 4);
        assert_eq!(before_stats.cached_resources, after_stats.cached_resources);
        if let (Some(before_report), Some(after_report)) = (before_report, after_report) {
            assert_eq!(before_report.device_to_host_bytes, after_report.device_to_host_bytes);
            assert_eq!(before_report.device_to_host_bytes, u64::from(capacity) * 16);
            assert_eq!(
                before_report.stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
                after_report.stages.iter().map(|s| s.stage).collect::<Vec<_>>()
            );
        }
        assert!(before.to_voxel_segments(&r).unwrap() == expected);
        assert!(after.to_voxel_segments(&r).unwrap() == expected);
        let warmed = r.scratch_buffer_pool_stats();
        let repeated = compact_voxel_segments_gpu_buffers(&r, &input, n as u32).unwrap();
        assert_eq!(r.scratch_buffer_pool_stats().allocation_count, warmed.allocation_count);
        assert!(repeated.to_voxel_segments(&r).unwrap() == expected);
        for budget in [4, 0] {
            r.set_scratch_buffer_pool_budget(budget);
            let result = compact_voxel_segments_gpu_buffers(&r, &input, n as u32).unwrap();
            assert!(result.to_voxel_segments(&r).unwrap() == expected);
            assert!(r.scratch_buffer_pool_stats().cached_bytes <= budget);
        }
        r.set_scratch_buffer_pool_budget(crate::resource_cache::DEFAULT_GPU_POOL_BUDGET_BYTES);
        saved += 4 * n as u64 - 4;
    }
    println!("VOXEL_BOUNDARY_REUSE_STORAGE verified {saved} aggregate requested/retained bytes removed on fresh pools, unchanged stage names/query bytes, warm reuse and zero budgets.");
}

#[path = "voxel_boundary_reuse_benchmark.rs"]
mod benchmark;
