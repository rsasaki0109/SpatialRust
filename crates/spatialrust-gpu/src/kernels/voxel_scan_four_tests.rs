use super::*;
use crate::{kernels::voxel_segments::build_voxel_segments, scratch::ScratchBuffers};

#[path = "voxel_scan_four_reference.rs"]
mod old;
#[path = "voxel_scan_four_compact_reference.rs"]
mod old_compact;

fn buffer(r: &WgpuRuntime, values: &[u32]) -> wgpu::Buffer {
    r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(values),
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

fn values(n: usize, pattern: u32) -> Vec<u32> {
    if n == 0 {
        return Vec::new();
    }
    match pattern {
        0 => vec![0; n],
        1 => (0..n).map(|i| u32::from(i.wrapping_mul(37) % 113 < 47)).collect(),
        2 => (0..n)
            .map(|i| u32::MAX / n as u32 + u32::from((i as u32) < u32::MAX % n as u32))
            .collect(),
        3 => {
            let mut out = vec![0; n];
            out[n / 2] = u32::MAX;
            out
        }
        _ => {
            let weights: Vec<u64> = (0..n)
                .map(|i| {
                    u64::from(
                        (i as u32)
                            .wrapping_mul(747_796_405)
                            .wrapping_add(2_891_336_453)
                            .rotate_left(13)
                            & 65535,
                    ) + 1
                })
                .collect();
            let mass: u64 = weights.iter().sum();
            let mut out: Vec<u32> =
                weights.iter().map(|w| (w * u64::from(u32::MAX) / mass) as u32).collect();
            let used: u64 = out.iter().map(|&v| u64::from(v)).sum();
            out[n / 3] += (u64::from(u32::MAX) - used) as u32;
            out
        }
    }
}

fn oracle(input: &[u32]) -> Vec<u32> {
    let mut total = 0u64;
    input
        .iter()
        .map(|&v| {
            total += u64::from(v);
            u32::try_from(total).unwrap()
        })
        .collect()
}

#[test]
fn voxel_scan_four_raw_blocks_match_parent_u64_prefixes_sums_and_guards() {
    let r = WgpuRuntime::new_headless().unwrap();
    let previous = old::ParentScanPipelines::new(r.device());
    let current = PrefixScanPipelines::new(r.device());
    let mut words = 0;
    for n in
        [1, 3, 4, 63, 64, 65, 255, 256, 257, 511, 1023, 1024, 1025, 65535, 65536, 65537, 1_000_003]
    {
        for pattern in 0..5 {
            let data = values(n, pattern);
            let expected: Vec<u32> = data.chunks(256).flat_map(oracle).collect();
            let sums: Vec<u32> = data
                .chunks(256)
                .map(|chunk| {
                    u32::try_from(chunk.iter().map(|&v| u64::from(v)).sum::<u64>()).unwrap()
                })
                .collect();
            let mut payload = data;
            payload.extend_from_slice(&[0xdead_beef; 8]);
            let input = buffer(&r, &payload);
            for (layout, pipeline) in
                [(&previous.layout, &previous.blocks), (&current.layout, &current.blocks)]
            {
                let prefix = buffer(&r, &vec![0xdead_beef; n + 8]);
                let totals = buffer(&r, &vec![0xdead_beef; sums.len() + 8]);
                let uniform = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&ScanParams {
                        element_count: n as u32,
                        _pad: [0; 3],
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: prefix.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: totals.as_entire_binding() },
                    ],
                });
                let mut encoder = r.device().create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
                }
                r.queue().submit(Some(encoder.finish()));
                let actual = read_words(&r, &prefix);
                let actual_sums = read_words(&r, &totals);
                assert_eq!(&actual[..n], expected);
                assert_eq!(&actual[n..], &[0xdead_beef; 8]);
                assert_eq!(&actual_sums[..sums.len()], sums);
                assert_eq!(&actual_sums[sums.len()..], &[0xdead_beef; 8]);
                words += n + sums.len();
            }
        }
    }
    eprintln!("VOXEL_SCAN_FOUR_RAW verified {words} exact parent/u64 block prefix and total words, max-u32 mass, partial blocks and poisoned tails");
}

#[test]
fn voxel_scan_four_complete_hierarchy_matches_parent_u64_and_zero_noop() {
    let r = WgpuRuntime::new_headless().unwrap();
    let prior = old::ParentScanPipelines::new(r.device());
    let mut words = 0;
    for n in [
        0, 1, 3, 4, 63, 64, 65, 255, 256, 257, 511, 1023, 1024, 1025, 65535, 65536, 65537,
        1_000_003,
    ] {
        for pattern in 0..5 {
            let data = values(n, pattern);
            let expected = oracle(&data);
            let mut payload = data;
            payload.extend_from_slice(&[0xdead_beef; 8]);
            let input = buffer(&r, &payload);
            for current in [false, true] {
                let output = buffer(&r, &vec![0xdead_beef; n + 8]);
                let mut encoder = r.device().create_command_encoder(&Default::default());
                let mut temporaries = ScratchBuffers::new(&r);
                if current {
                    encode_inclusive_scan(
                        &r,
                        &mut encoder,
                        &input,
                        &output,
                        n as u32,
                        &mut None,
                        &mut temporaries,
                    );
                } else {
                    old::encode_inclusive_scan(
                        &prior,
                        &r,
                        &mut encoder,
                        &input,
                        &output,
                        n as u32,
                        &mut None,
                        &mut temporaries,
                    );
                }
                temporaries.submit_and_recycle(encoder);
                let actual = read_words(&r, &output);
                assert_eq!(&actual[..n], expected, "n={n},pattern={pattern},current={current}");
                assert_eq!(&actual[n..], &[0xdead_beef; 8]);
                words += n;
            }
        }
    }
    eprintln!("VOXEL_SCAN_FOUR_HIERARCHY verified {words} complete parent/u64 prefix words through three hierarchy levels, reused scratch and zero no-op");
}

fn entries(n: usize, run: usize) -> (Vec<[u32; 4]>, crate::VoxelSegments) {
    let entries: Vec<_> = (0..n)
        .map(|i| {
            [((i / run) as i32 - 100) as u32, (-3i32) as u32, i32::MAX as u32, (n - i - 1) as u32]
        })
        .collect();
    let keys: Vec<_> = entries
        .iter()
        .map(|p| (i64::from(p[0] as i32), i64::from(p[1] as i32), i64::from(p[2] as i32)))
        .collect();
    let mut expected = build_voxel_segments(&keys);
    expected.point_indices = entries.iter().map(|e| e[3]).collect();
    for (&start, &len) in expected.cell_starts.iter().zip(&expected.cell_counts) {
        expected.point_indices[start as usize..(start + len) as usize].sort_unstable();
    }
    (entries, expected)
}

#[test]
fn voxel_scan_four_complete_compactor_keeps_parent_cpu_indices_counts_and_owned_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    let prior = old::ParentScanPipelines::new(r.device());
    let mut words = 0;
    let mut retained = None;
    for n in [0, 1, 257, 65537, 1_000_003] {
        for run in [1, 4, 257, n.max(1)] {
            let (mut data, expected) = entries(n, run);
            data.push([0xdead_beef; 4]);
            let input = buffer(&r, bytemuck::cast_slice(&data));
            let before = old_compact::previous(&prior, &r, &input, n as u32, None).unwrap();
            let after = super::super::voxel_compact::compact_voxel_segments_gpu_buffers(
                &r, &input, n as u32,
            )
            .unwrap();
            assert_eq!(before.construction_readback_bytes(), after.construction_readback_bytes());
            assert_eq!(after.construction_readback_bytes(), if n == 0 { 0 } else { 4 });
            assert_eq!(before.cell_count(), after.cell_count());
            assert_eq!(before.point_count(), after.point_count());
            if n > 0 {
                for output in [&before, &after] {
                    assert_eq!(
                        &read_words(&r, output.point_indices_buffer())[..n],
                        data[..n].iter().map(|v| v[3]).collect::<Vec<_>>()
                    );
                }
            }
            assert!(before.to_voxel_segments(&r).unwrap() == expected);
            assert!(after.to_voxel_segments(&r).unwrap() == expected);
            words += 2 * (expected.point_indices.len() + expected.keys.len() * 5);
            if n == 257 && run == 257 {
                retained = Some((after, expected));
            }
            if let Some((ref output, ref expected)) = retained {
                assert!(output.to_voxel_segments(&r).unwrap() == *expected);
            }
        }
    }
    eprintln!("VOXEL_SCAN_FOUR_PUBLIC verified {words} parent/CPU logical segment values across 20 complete compactor fixtures, raw index order, scalar receipts and retained outputs");
}

#[path = "voxel_scan_four_benchmark.rs"]
mod benchmark;
