use super::*;
use crate::{kernels::voxel_segments::build_voxel_segments, GpuStageProfiler};

#[path = "voxel_boundary_fusion_reference.rs"]
mod old;
#[path = "voxel_boundary_fusion_scan_reference.rs"]
mod old_scan;
use crate::kernels::voxel_compact::{
    compact_voxel_segments_gpu_buffers, compact_voxel_segments_gpu_buffers_with_profiler,
};

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

struct ScanFixture {
    input: wgpu::Buffer,
    flags: wgpu::Buffer,
    prefix: wgpu::Buffer,
    sums: wgpu::Buffer,
    inactive: [wgpu::Buffer; 5],
    mark_group: wgpu::BindGroup,
    scan_group: wgpu::BindGroup,
}

fn fixture(
    r: &WgpuRuntime,
    data: &[[u32; 4]],
    compact: &old::ParentCompactPipelines,
    layout: &wgpu::BindGroupLayout,
    fused: bool,
) -> ScanFixture {
    let n = data.len();
    let mut payload: Vec<u32> = bytemuck::cast_slice(data).to_vec();
    payload.extend([0xdead_beef; 8]);
    let input = buffer(r, &payload);
    let flags = buffer(r, &vec![0xdead_beef; n + 8]);
    let prefix = buffer(r, &vec![0xdead_beef; n + 8]);
    let sums = buffer(r, &vec![0xdead_beef; n.div_ceil(256).max(1) + 8]);
    let inactive =
        std::array::from_fn(|i| buffer(r, &vec![0xdead_beef; if i == 0 { 4 } else { 1 }]));
    let params = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&ScanParams { element_count: n as u32, _pad: [0; 3] }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mark_buffers = [
        &input,
        &flags,
        &prefix,
        &inactive[0],
        &inactive[1],
        &inactive[2],
        &inactive[3],
        &inactive[4],
    ];
    let mut bindings =
        vec![wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() }];
    bindings.extend(mark_buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry {
        binding: i as u32 + 1,
        resource: b.as_entire_binding(),
    }));
    let mark_group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &compact.bind_group_layout,
        entries: &bindings,
    });
    let scan_group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: if fused { &input } else { &flags }.as_entire_binding(),
            },
            wgpu::BindGroupEntry { binding: 2, resource: prefix.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: sums.as_entire_binding() },
        ],
    });
    ScanFixture { input, flags, prefix, sums, inactive, mark_group, scan_group }
}

fn encode_mark(
    encoder: &mut wgpu::CommandEncoder,
    compact: &old::ParentCompactPipelines,
    raw: &ScanFixture,
    n: u32,
    profiler: &mut Option<&mut GpuStageProfiler>,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: crate::profiling::timestamp_writes(profiler, "voxel-compact-mark"),
    });
    pass.set_pipeline(&compact.mark);
    pass.set_bind_group(0, &raw.mark_group, &[]);
    pass.dispatch_workgroups(n.div_ceil(256).max(1), 1, 1);
}

#[allow(clippy::too_many_arguments)]
fn encode_hierarchy(
    r: &WgpuRuntime,
    compact: &old::ParentCompactPipelines,
    scan: &old_scan::ParentScanPipelines,
    raw: &ScanFixture,
    n: u32,
    fused: bool,
    encoder: &mut wgpu::CommandEncoder,
    profiler: &mut Option<&mut GpuStageProfiler>,
    temporaries: &mut crate::scratch::ScratchBuffers<'_>,
) {
    if fused {
        encode_boundary_inclusive_scan(
            r,
            encoder,
            &raw.input,
            &raw.prefix,
            n,
            profiler,
            temporaries,
        );
    } else {
        if n > 0 {
            encode_mark(encoder, compact, raw, n, profiler);
        }
        old_scan::encode_inclusive_scan(
            scan,
            r,
            encoder,
            &raw.flags,
            &raw.prefix,
            n,
            profiler,
            temporaries,
        );
    }
}

fn verify_scan(
    r: &WgpuRuntime,
    raw: &ScanFixture,
    data: &[[u32; 4]],
    expected: &[u32],
    fused: bool,
) -> usize {
    let n = data.len();
    let actual = read_words(r, &raw.prefix);
    assert_eq!(&actual[..n], expected);
    assert_eq!(&actual[n..], &[0xdead_beef; 8]);
    let flags = read_words(r, &raw.flags);
    if fused {
        assert!(flags.iter().all(|&v| v == 0xdead_beef));
    } else {
        assert_eq!(&flags[..n], boundary_flags(data));
        assert_eq!(&flags[n..], &[0xdead_beef; 8]);
    }
    for inactive in &raw.inactive {
        assert!(read_words(r, inactive).iter().all(|&v| v == 0xdead_beef));
    }
    let input = read_words(r, &raw.input);
    assert_eq!(&input[..n * 4], bytemuck::cast_slice::<_, u32>(data));
    assert_eq!(&input[n * 4..], &[0xdead_beef; 8]);
    actual.len()
        + flags.len()
        + input.len()
        + raw.inactive.iter().map(|b| b.size() as usize / 4).sum::<usize>()
}

#[test]
fn voxel_boundary_fusion_raw_blocks_match_parent_u64_totals_and_guards() {
    let r = WgpuRuntime::new_headless().unwrap();
    let compact = old::ParentCompactPipelines::new(r.device());
    let scan = old_scan::ParentScanPipelines::new(r.device());
    let current = PrefixScanPipelines::new(r.device());
    let mut words = 0;
    let mut cases = 0;
    for n in [0, 1, 2, 3, 255, 256, 257, 511, 1025, 65536, 65537, 1_000_003] {
        for run in [1, 4, 257, n.max(1), 0] {
            let (data, _) = entries(n, run);
            let flags = boundary_flags(&data);
            let expected: Vec<u32> = flags
                .chunks(256)
                .flat_map(|chunk| {
                    let mut sum = 0u64;
                    chunk.iter().map(move |&v| {
                        sum += u64::from(v);
                        u32::try_from(sum).unwrap()
                    })
                })
                .collect();
            let mut sums: Vec<u32> = flags
                .chunks(256)
                .map(|chunk| {
                    u32::try_from(chunk.iter().map(|&v| u64::from(v)).sum::<u64>()).unwrap()
                })
                .collect();
            if n == 0 {
                sums.push(0);
            }
            for fused in [false, true] {
                let raw = fixture(
                    &r,
                    &data,
                    &compact,
                    if fused { &current.layout } else { &scan.layout },
                    fused,
                );
                let mut encoder = r.device().create_command_encoder(&Default::default());
                if !fused {
                    encode_mark(&mut encoder, &compact, &raw, n as u32, &mut None);
                }
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(if fused { &current.boundaries } else { &scan.blocks });
                    pass.set_bind_group(0, &raw.scan_group, &[]);
                    pass.dispatch_workgroups((n as u32).div_ceil(256).max(1), 1, 1);
                }
                r.queue().submit(Some(encoder.finish()));
                words += verify_scan(&r, &raw, &data, &expected, fused);
                let actual = read_words(&r, &raw.sums);
                assert_eq!(&actual[..sums.len()], sums);
                assert_eq!(&actual[sums.len()..], &[0xdead_beef; 8]);
                words += actual.len();
            }
            cases += 1;
        }
    }
    println!("VOXEL_BOUNDARY_FUSION_RAW verified {words} parent/u64 block prefix, total, source/unused words and guards across {cases} paired fixtures; empty raw dispatch participates in every barrier.");
}

#[test]
fn voxel_boundary_fusion_complete_hierarchy_matches_parent_u64_and_zero_noop() {
    let r = WgpuRuntime::new_headless().unwrap();
    let compact = old::ParentCompactPipelines::new(r.device());
    let scan = old_scan::ParentScanPipelines::new(r.device());
    let current = PrefixScanPipelines::new(r.device());
    let mut words = 0;
    let mut cases = 0;
    for n in [0, 1, 3, 255, 256, 257, 511, 65536, 65537, 1_000_003] {
        for run in [1, 4, 257, n.max(1), 0] {
            let (data, _) = entries(n, run);
            let expected = prefix_oracle(&data);
            for fused in [false, true] {
                let raw = fixture(
                    &r,
                    &data,
                    &compact,
                    if fused { &current.layout } else { &scan.layout },
                    fused,
                );
                let mut encoder = r.device().create_command_encoder(&Default::default());
                let before = r.scratch_buffer_pool_stats();
                let mut temporaries = crate::scratch::ScratchBuffers::new(&r);
                encode_hierarchy(
                    &r,
                    &compact,
                    &scan,
                    &raw,
                    n as u32,
                    fused,
                    &mut encoder,
                    &mut None,
                    &mut temporaries,
                );
                temporaries.submit_and_recycle(encoder);
                if n == 0 {
                    assert_eq!(
                        before.allocation_count,
                        r.scratch_buffer_pool_stats().allocation_count
                    );
                }
                words += verify_scan(&r, &raw, &data, &expected, fused);
                assert!(read_words(&r, &raw.sums).iter().all(|&v| v == 0xdead_beef));
            }
            cases += 1;
        }
    }
    println!("VOXEL_BOUNDARY_FUSION_HIERARCHY verified {words} complete parent/u64 prefix, source/unused words and guards across {cases} paired fixtures; no scratch requests on zero input.");
}

#[test]
fn voxel_boundary_fusion_complete_parent_cpu_preserve_indices_metadata_and_retained_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    let prior = old::ParentCompactPipelines::new(r.device());
    let scan = old_scan::ParentScanPipelines::new(r.device());
    let mut words = 0;
    let mut retained = None;
    for n in [0, 1, 257, 65537, 1_000_003] {
        for run in [1, 4, 257, n.max(1), 0] {
            let (mut data, expected) = entries(n, run);
            data.push([0xdead_beef; 4]);
            let input = buffer(&r, bytemuck::cast_slice(&data));
            let before = old::previous(&prior, &scan, &r, &input, n as u32, None).unwrap();
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
    println!("VOXEL_BOUNDARY_FUSION_PUBLIC verified {words} complete parent/CPU logical values across 25 paired fixtures, unchanged scalar receipts and retained outputs.");
}

#[test]
fn voxel_boundary_fusion_pool_bytes_profiling_zero_budget_and_stale_bindings() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    let prior = old::ParentCompactPipelines::new(r.device());
    let scan = old_scan::ParentScanPipelines::new(r.device());
    let mut saved = 0u64;
    for n in [1, 257, 65537] {
        let (data, expected) = entries(n, 0);
        let input = buffer(&r, bytemuck::cast_slice(&data));
        let capacity = 2 + crate::kernels::voxel_scan::scan_pass_count(n as u32);
        r.clear_scratch_buffer_pool();
        let mut before_profiler =
            r.timestamp_queries_enabled().then(|| GpuStageProfiler::new(&r, capacity).unwrap());
        let before =
            old::previous(&prior, &scan, &r, &input, n as u32, before_profiler.as_mut()).unwrap();
        let before_stats = r.scratch_buffer_pool_stats();
        let before_report = before_profiler.map(|profiler| profiler.collect(&r).unwrap());
        r.clear_scratch_buffer_pool();
        let mut after_profiler =
            r.timestamp_queries_enabled().then(|| GpuStageProfiler::new(&r, capacity - 1).unwrap());
        let after = compact_voxel_segments_gpu_buffers_with_profiler(
            &r,
            &input,
            n as u32,
            after_profiler.as_mut(),
        )
        .unwrap();
        let after_stats = r.scratch_buffer_pool_stats();
        let after_report = after_profiler.map(|profiler| profiler.collect(&r).unwrap());
        assert_eq!(before_stats.cached_bytes, after_stats.cached_bytes);
        assert_eq!(before_stats.cached_resources, after_stats.cached_resources);
        if let (Some(before_report), Some(after_report)) = (before_report, after_report) {
            assert_eq!(before_report.device_to_host_bytes, after_report.device_to_host_bytes + 16);
            assert_eq!(after_report.device_to_host_bytes, u64::from(capacity - 1) * 16);
            assert_eq!(before_report.device_to_host_bytes, u64::from(capacity) * 16);
            assert_eq!(
                before_report
                    .stages
                    .iter()
                    .map(|s| s.stage)
                    .filter(|&s| s != "voxel-compact-mark")
                    .collect::<Vec<_>>(),
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
        saved += 16;
    }
    println!("VOXEL_BOUNDARY_FUSION_STORAGE verified {saved} optional query bytes removed across three fresh-pool comparisons; identical retained storage and one fewer mark stage, warm reuse and zero budgets.");
}

#[cfg(feature = "gpu-aoso-staging")]
#[test]
fn voxel_boundary_fusion_profiled_frame_accepts_exact_capacity_and_rejects_before_upload() {
    use spatialrust_core::{
        PointBuffer, PointBufferSet, PointCloud, SpatialTensor, StandardSchemas,
    };
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    if !r.timestamp_queries_enabled() {
        return;
    }
    for (n, required) in [(0, 0), (1, 9), (2, 153), (257, 157)] {
        let points: Vec<[f32; 3]> =
            (0..n).map(|i| [(i % 17) as f32 * 0.1, (i / 17) as f32 * 0.1, 0.0]).collect();
        let mut buffers = PointBufferSet::new();
        for (axis, field) in ["x", "y", "z"].iter().enumerate() {
            buffers.insert(*field, PointBuffer::F32(points.iter().map(|p| p[axis]).collect()));
        }
        let cloud =
            PointCloud::try_from_parts(StandardSchemas::point_xyz(), buffers, Default::default())
                .unwrap();
        let tensor = SpatialTensor::new(&cloud, 13).unwrap();
        let mut plain =
            crate::run_aoso_voxel_normal_frame(&r, &tensor, [0.0; 3], 4.0, 0.25).unwrap();
        if required > 0 {
            let mut small = GpuStageProfiler::new(&r, required - 1).unwrap();
            let before = r.scratch_buffer_pool_stats();
            assert!(crate::run_aoso_voxel_normal_frame_profiled(
                &r, &tensor, [0.0; 3], 4.0, 0.25, &mut small
            )
            .is_err());
            assert_eq!(before.allocation_count, r.scratch_buffer_pool_stats().allocation_count);
            assert_eq!(before.reuse_count, r.scratch_buffer_pool_stats().reuse_count);
            assert!(small.collect(&r).unwrap().stages.is_empty());
        }
        let mut profiler = GpuStageProfiler::new(&r, required.max(1)).unwrap();
        let mut current = crate::run_aoso_voxel_normal_frame_profiled(
            &r,
            &tensor,
            [0.0; 3],
            4.0,
            0.25,
            &mut profiler,
        )
        .unwrap();
        let report = profiler.collect(&r).unwrap();
        assert_eq!(report.stages.len(), required as usize);
        assert_eq!(report.device_to_host_bytes, u64::from(required) * 16);
        assert!(report.stages.iter().all(|s| s.stage != "voxel-compact-mark"));
        assert_eq!(
            plain.receipt().host_to_device_bytes(),
            current.receipt().host_to_device_bytes()
        );
        assert_eq!(plain.receipt().gpu_to_gpu_bytes(), current.receipt().gpu_to_gpu_bytes());
        assert_eq!(
            plain.receipt().device_to_host_bytes(),
            current.receipt().device_to_host_bytes()
        );
        assert_eq!(
            plain.voxel_segments().unwrap().to_voxel_segments(&r).unwrap(),
            current.voxel_segments().unwrap().to_voxel_segments(&r).unwrap()
        );
        assert_eq!(current.normals().unwrap().point_count(), n);
        assert_eq!(plain.readback_positions(&r).unwrap(), points);
        assert_eq!(current.readback_positions(&r).unwrap(), points);
        plain.recycle(&r).unwrap();
        current.recycle(&r).unwrap();
    }
    println!("VOXEL_BOUNDARY_FUSION_FRAME verified exact 0/9/153/157 stage capacities and receipts for empty/singleton/two/257-point frames; undersized capacity is rejected before scratch/upload work.");
}

#[path = "voxel_boundary_fusion_benchmark.rs"]
mod benchmark;
