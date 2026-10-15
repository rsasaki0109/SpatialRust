use super::*;
use crate::kernels::voxel_average_batch::{
    plan_voxel_average_batch, plan_voxel_average_batch_with_host_limit, record_voxel_average_batch,
};

#[path = "voxel_average_batch_reference.rs"]
mod baseline;

fn baseline_call(r: &WgpuRuntime, f: &Fixture) -> Channels {
    baseline::baseline_reduce_voxel_average_f32_multi_gpu(
        r,
        &f.source.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &f.segments,
    )
    .unwrap()
}

#[test]
fn voxel_average_batch_planner_checks_exact_device_host_and_index_bounds() {
    let limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 7,
        max_bindings_per_bind_group: 8,
        max_compute_workgroup_size_x: 256,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroups_per_dimension: 2,
        max_storage_buffer_binding_size: 4112,
        max_buffer_size: 7196,
        ..wgpu::Limits::downlevel_defaults()
    };
    let plan = plan_voxel_average_batch(&limits, 257, 7).unwrap();
    assert_eq!(plan.channel_bytes, 1028);
    assert_eq!(plan.staging_bytes, 7196);
    assert_eq!(plan_voxel_average_batch_with_host_limit(&limits, 257, 7, 7196), Some(plan));
    assert!(plan_voxel_average_batch_with_host_limit(&limits, 257, 7, 7195).is_none());
    for boundary in 0..7 {
        let mut l = limits.clone();
        match boundary {
            0 => l.max_storage_buffers_per_shader_stage -= 1,
            1 => l.max_bindings_per_bind_group -= 1,
            2 => l.max_compute_workgroup_size_x -= 1,
            3 => l.max_compute_invocations_per_workgroup -= 1,
            4 => l.max_compute_workgroups_per_dimension -= 1,
            5 => l.max_storage_buffer_binding_size -= 1,
            6 => l.max_buffer_size -= 1,
            _ => unreachable!(),
        }
        assert!(plan_voxel_average_batch(&l, 257, 7).is_none(), "boundary={boundary}");
    }
    for channels in [0, 1, usize::MAX] {
        assert!(plan_voxel_average_batch(&limits, 257, channels).is_none());
    }
    assert!(plan_voxel_average_batch(&limits, 0, 4).is_none());
    for channels in [2, 3, 4] {
        assert_eq!(
            plan_voxel_average_batch(&limits, 257, channels).unwrap().staging_bytes,
            1028 * channels as u64
        );
    }
    let wide = wgpu::Limits {
        max_storage_buffer_binding_size: u32::MAX,
        max_buffer_size: u64::MAX,
        max_compute_workgroups_per_dimension: u32::MAX,
        ..limits
    };
    assert!(plan_voxel_average_batch(&wide, u32::MAX, 4).is_none());
    let cells = (i32::MAX as u64 / 16) as u32;
    assert!(plan_voxel_average_batch_with_host_limit(&wide, cells, 4, i32::MAX as u64).is_some());
    assert!(
        plan_voxel_average_batch_with_host_limit(&wide, cells + 1, 4, i32::MAX as u64).is_none()
    );
}

fn read_words(r: &WgpuRuntime, buffer: &wgpu::Buffer) -> Vec<u32> {
    let slice = buffer.slice(..);
    let (send, recv) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| send.send(result).unwrap());
    r.device().poll(wgpu::Maintain::Wait);
    recv.recv().unwrap().unwrap();
    let view = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    buffer.unmap();
    result
}

fn compare_kernel(
    r: &WgpuRuntime,
    source: &Channels,
    indices: &[u32],
    starts: &[u32],
    oracle: bool,
) -> usize {
    let cells = starts.len();
    let channels = source.len();
    let upload = |label, data: &[u32]| {
        r.upload_u32_storage(label, if data.is_empty() { &[0] } else { data }).unwrap()
    };
    let segments = GpuVoxelSegments::new(
        cells as u32,
        indices.len() as u32,
        upload("average-batch-keys", &[0]),
        upload("average-batch-indices", indices),
        upload("average-batch-starts", starts),
        0,
    );
    let values: Vec<_> = source
        .iter()
        .map(|v| {
            r.upload_f32_storage("average-batch-values", if v.is_empty() { &[0.0] } else { v })
                .unwrap()
        })
        .collect();
    let output = upload("average-batch-output", &vec![0xdead_beef; cells * channels + 8]);
    let old: Vec<_> = (0..channels)
        .map(|_| upload("average-batch-parent-output", &vec![0xdead_beef; cells + 8]))
        .collect();
    let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("average-batch-staging"),
        size: output.size() + old.iter().map(wgpu::Buffer::size).sum::<u64>(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = r.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    record_voxel_average_batch(&mut encoder, r, &values, &segments, &output);
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output.size());
    let mut offset = output.size();
    for (value, old) in values.iter().zip(&old) {
        record_voxel_reduce_f32_pass(
            &mut encoder,
            r,
            value,
            segments.point_indices_buffer(),
            segments.cell_starts_buffer(),
            cells as u32,
            indices.len() as u32,
            old,
        )
        .unwrap();
        encoder.copy_buffer_to_buffer(old, 0, &staging, offset, old.size());
        offset += old.size();
    }
    r.queue().submit(Some(encoder.finish()));
    let words = read_words(r, &staging);
    assert_eq!(&words[cells * channels..cells * channels + 8], &[0xdead_beef; 8]);
    for (c, values) in source.iter().enumerate() {
        let old_offset = (output.size() / 4) as usize + c * (cells + 8);
        assert_eq!(&words[c * cells..(c + 1) * cells], &words[old_offset..old_offset + cells]);
        assert_eq!(&words[old_offset + cells..old_offset + cells + 8], &[0xdead_beef; 8]);
        for cell in 0..cells {
            let start = starts[cell] as usize;
            let end = starts.get(cell + 1).map_or(indices.len(), |&n| n as usize);
            let value = f32::from_bits(words[c * cells + cell]);
            if end == start {
                assert_eq!(value.to_bits(), 0);
            } else if oracle {
                let sum: f64 =
                    indices[start..end].iter().map(|&i| f64::from(values[i as usize])).sum();
                let expected = sum / (end - start) as f64;
                let tolerance = 4.0 * f64::from(f32::EPSILON) * expected.abs().max(1.0);
                assert!(
                    (f64::from(value) - expected).abs() <= tolerance,
                    "channel={c} cell={cell} actual={value} expected={expected}"
                );
            }
        }
    }
    cells * channels
}

#[test]
fn voxel_average_batch_kernels_match_original_math_empty_cells_nonfinite_values_and_guards() {
    let r = WgpuRuntime::new_headless().unwrap();
    if plan_voxel_average_batch(&r.device().limits(), 65_537, 4).is_none() {
        eprintln!("VOXEL_AVERAGE_BATCH_KERNEL unavailable: device limits");
        return;
    }
    let mut compared = 0;
    for cells in [1, 3, 17, 255, 256, 257, 1025, 65_537] {
        let mut points = 0;
        let starts: Vec<_> = (0..cells)
            .map(|i| {
                let start = points;
                points += [0, 1, 2, 3, 4, 7, 17][i % 7];
                start as u32
            })
            .collect();
        let indices: Vec<_> = (0..points).rev().map(|i| i as u32).collect();
        for channels in 2..=4 {
            for variant in 0..3 {
                let source: Channels = (0..channels)
                    .map(|c| {
                        (0..points)
                            .map(|i| match variant {
                                0 => ((i * 17 + c * 29) % 4096) as f32 * 0.25 - 511.5,
                                1 => match (i + c * 7) % 19 {
                                    0 => f32::from_bits(0x7fc0_1234),
                                    1 => f32::from_bits(0x7f80_0001),
                                    2 => f32::INFINITY,
                                    3 => f32::NEG_INFINITY,
                                    4 => -0.0,
                                    5 => f32::from_bits(1),
                                    _ => 1.25,
                                },
                                _ => [16_777_216.0, 1.0, -16_777_216.0, 3.0][(i + c) % 4],
                            })
                            .collect()
                    })
                    .collect();
                compared += compare_kernel(&r, &source, &indices, &starts, variant == 0);
            }
        }
    }
    let points = 262_145;
    let source: Channels = (0..4)
        .map(|c| {
            (0..points).map(|i| [16_777_216.0, 1.0, -16_777_216.0, 3.0][(i + c) % 4]).collect()
        })
        .collect();
    compared += compare_kernel(&r, &source, &(0..points as u32).collect::<Vec<_>>(), &[0], false);
    eprintln!("VOXEL_AVERAGE_BATCH_KERNEL verified {compared} exact original GPU words; empty/non-power/nonfinite/cancellation/long-cell/guard fixtures");
}

#[test]
fn public_voxel_average_batch_matches_complete_parent_cpu_lengths_and_partial_batches() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    for cells in [0, 1, 3, 17, 255, 256, 257, 65_537, 1_048_577] {
        for channels in [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 17] {
            if cells > 65_537 && ![2, 3, 4, 5, 7].contains(&channels) {
                continue;
            }
            let occupancy = if cells > 65_537 { 1 } else { 4 };
            let mut f = fixture(&r, cells, channels, occupancy);
            if cells == 257 {
                f.segments = crate::kernels::voxel_sort::build_voxel_segments_gpu_from_keys(
                    &r,
                    &(0..cells * occupancy).map(|i| ((i % cells) as i64, 0, 0)).collect::<Vec<_>>(),
                )
                .unwrap();
            }
            let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
            let stats = r.buffer_pool().stats();
            let old = baseline_call(&r, &f);
            let live = reduce_voxel_average_f32_multi_gpu(&r, &refs, &f.segments).unwrap();
            assert_bits(&live, &old);
            assert_bits(&live, &f.mean);
            assert_eq!(capacity(&live), cells * channels);
            assert_eq!(r.buffer_pool().stats(), stats);
            drop(refs);
            drop(f);
            r.clear_buffer_pool();
            assert_bits(&live, &old);
            compared += cells * channels * 2;
        }
    }
    let f = fixture(&r, 17, 11, 4);
    let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
    for (bad_index, found) in [(0, 67), (4, 66), (8, 65), (10, 64)] {
        let mut bad = refs.clone();
        bad[bad_index] = &f.source[bad_index][..found];
        if bad_index == 4 {
            bad[10] = &f.source[10][..63];
        }
        let stats = r.buffer_pool().stats();
        let error = reduce_voxel_average_f32_multi_gpu(&r, &bad, &f.segments).unwrap_err();
        assert_eq!(error, SpatialError::BufferLengthMismatch { expected: 68, found });
        assert_eq!(
            error,
            baseline::baseline_reduce_voxel_average_f32_multi_gpu(&r, &bad, &f.segments)
                .unwrap_err()
        );
        assert_eq!(r.buffer_pool().stats(), stats);
    }
    let empty = fixture(&r, 0, 0, 1);
    assert_eq!(
        reduce_voxel_average_f32_multi_gpu(&r, &[&[123.0][..]; 11], &empty.segments).unwrap(),
        vec![Vec::<f32>::new(); 11]
    );
    eprintln!(
        "VOXEL_AVERAGE_BATCH_PUBLIC verified {compared} complete parent/CPU words; supported={}",
        plan_voxel_average_batch(&r.device().limits(), 257, 4).is_some()
    );
}

#[test]
fn voxel_average_batch_reuploads_changed_host_values_and_keeps_owned_results() {
    let r = WgpuRuntime::new_headless().unwrap();
    for channels in [2, 3, 4, 5, 7, 9, 11] {
        let mut f = fixture(&r, 257, channels, 4);
        let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
        let old = reduce_voxel_average_f32_multi_gpu(&r, &refs, &f.segments).unwrap();
        assert_bits(&old, &f.mean);
        drop(refs);
        for source in &mut f.source {
            source.fill(64.25);
        }
        let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
        let live = reduce_voxel_average_f32_multi_gpu(&r, &refs, &f.segments).unwrap();
        assert_bits(&live, &baseline_call(&r, &f));
        assert!(live.iter().flatten().all(|&v| v == 64.25));
        let expected = f.mean.clone();
        drop(refs);
        drop(f);
        r.clear_buffer_pool();
        assert_bits(&old, &expected);
        assert!(live.iter().flatten().all(|&v| v == 64.25));
    }
}

#[test]
fn voxel_average_batch_concurrent_cold_calls_share_one_lazy_pipeline() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    if plan_voxel_average_batch(&r.device().limits(), 257, 4).is_none() {
        return;
    }
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let handles: Vec<_> = [2, 4, 7]
        .into_iter()
        .map(|channels| {
            let r = r.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let f = fixture(&r, 257, channels, 4);
                let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
                barrier.wait();
                let live = reduce_voxel_average_f32_multi_gpu(&r, &refs, &f.segments).unwrap();
                assert_bits(&live, &f.mean);
                assert_bits(&live, &baseline_call(&r, &f));
                r.pipelines().voxel_reduce.multi_average_pipeline(r.device()) as *const _ as usize
            })
        })
        .collect();
    let pointers: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(pointers.windows(2).all(|p| p[0] == p[1]));
}

#[path = "voxel_average_batch_benchmark.rs"]
mod benchmark;
