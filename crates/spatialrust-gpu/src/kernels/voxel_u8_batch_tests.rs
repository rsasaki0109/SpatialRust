use super::*;
use crate::kernels::voxel_u8_batch::{plan_batched_voxel_u8, record_voxel_u8_batch_pass};
use wgpu::util::DeviceExt;

#[path = "voxel_u8_batch_reference.rs"]
mod baseline;

fn call_baseline(r: &WgpuRuntime, f: &Fixture, mode: usize) -> Output {
    let function = match mode {
        0 => baseline::baseline_gather_voxel_first_xyz_and_multi_gpu,
        1 => baseline::baseline_gather_voxel_first_xyz_and_average_multi_gpu,
        2 => baseline::baseline_reduce_voxel_centroids_xyz_and_average_multi_gpu,
        3 => baseline::baseline_reduce_voxel_centroids_xyz_and_gather_first_multi_gpu,
        _ => unreachable!(),
    };
    function(
        r,
        &f.xyz[0],
        &f.xyz[1],
        &f.xyz[2],
        &f.floats[3..].iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &f.bytes.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &f.segments,
    )
    .unwrap()
}

#[test]
fn voxel_u8_batch_planner_checks_exact_and_one_short_limits() {
    let limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 7,
        max_bindings_per_bind_group: 8,
        max_compute_workgroup_size_x: 256,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroup_storage_size: 4096,
        max_compute_workgroups_per_dimension: 2,
        max_storage_buffer_binding_size: 1040,
        max_buffer_size: 1040,
        ..wgpu::Limits::downlevel_defaults()
    };
    assert_eq!(plan_batched_voxel_u8(&limits, 257, 4).unwrap().output_bytes, 1040);
    for u in [0, 1, 5, 7, usize::MAX] {
        assert!(plan_batched_voxel_u8(&limits, 257, u).is_none());
    }
    assert!(plan_batched_voxel_u8(&limits, 0, 3).is_none());
    for bound in 0..8 {
        let mut l = limits.clone();
        match bound {
            0 => l.max_storage_buffers_per_shader_stage -= 1,
            1 => l.max_bindings_per_bind_group -= 1,
            2 => l.max_compute_workgroup_size_x -= 1,
            3 => l.max_compute_invocations_per_workgroup -= 1,
            4 => l.max_compute_workgroup_storage_size -= 1,
            5 => l.max_compute_workgroups_per_dimension -= 1,
            6 => l.max_storage_buffer_binding_size -= 1,
            7 => l.max_buffer_size -= 1,
            _ => unreachable!(),
        }
        assert!(plan_batched_voxel_u8(&l, 257, 4).is_none(), "bound={bound}");
    }
    let large = wgpu::Limits {
        max_compute_workgroups_per_dimension: u32::MAX,
        max_storage_buffer_binding_size: u32::MAX,
        max_buffer_size: u64::MAX,
        ..limits
    };
    assert!(plan_batched_voxel_u8(&large, u32::MAX, 4).is_none());
    assert!(plan_batched_voxel_u8(&large, u32::MAX, 2).is_none());
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

fn compare_kernels(
    r: &WgpuRuntime,
    channels: &[Vec<u8>],
    indices: &[u32],
    starts: &[u32],
    cpu: bool,
) -> usize {
    let cells = starts.len();
    let upload = |label, data: &[u32]| {
        r.upload_u32_storage(label, if data.is_empty() { &[0] } else { data }).unwrap()
    };
    let segments = GpuVoxelSegments::new(
        cells as u32,
        indices.len() as u32,
        upload("byte-batch-keys", &[0]),
        upload("byte-batch-indices", indices),
        upload("byte-batch-starts", starts),
        0,
    );
    let inputs: Vec<_> = channels
        .iter()
        .map(|values| {
            r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("byte-batch-test-input"),
                contents: &crate::readback::pad_u8_for_gpu_storage(if values.is_empty() {
                    &[0]
                } else {
                    values
                }),
                usage: wgpu::BufferUsages::STORAGE,
            })
        })
        .collect();
    let stride = cells.div_ceil(4);
    for average in [false, true] {
        let output = upload("byte-batch-output", &vec![0xdead_beef; stride * channels.len() + 8]);
        let old: Vec<_> = channels
            .iter()
            .map(|_| upload("byte-batch-parent-output", &vec![0xdead_beef; stride + 8]))
            .collect();
        let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("byte-batch-test-staging"),
            size: output.size() + old.iter().map(wgpu::Buffer::size).sum::<u64>(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            r.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        record_voxel_u8_batch_pass(&mut encoder, r, &inputs, &segments, &output, average);
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output.size());
        let record = if average {
            crate::kernels::voxel_reduce::record_voxel_reduce_u8_packed_pass
        } else {
            crate::kernels::voxel_gather::record_voxel_gather_u8_packed_pass
        };
        let mut offset = output.size();
        for (input, old) in inputs.iter().zip(&old) {
            record(
                &mut encoder,
                r,
                input,
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
        assert_eq!(&words[stride * channels.len()..stride * channels.len() + 8], &[0xdead_beef; 8]);
        for (c, values) in channels.iter().enumerate() {
            let old_offset = (output.size() / 4) as usize + c * (stride + 8);
            let packed = &words[c * stride..(c + 1) * stride];
            assert_eq!(packed, &words[old_offset..old_offset + stride]);
            assert_eq!(&words[old_offset + stride..old_offset + stride + 8], &[0xdead_beef; 8]);
            if cells % 4 != 0 {
                assert_eq!(packed[stride - 1] >> (cells % 4 * 8), 0);
            }
            if cpu {
                for cell in 0..cells {
                    let start = starts[cell] as usize;
                    let end = starts.get(cell + 1).map_or(indices.len(), |&n| n as usize);
                    let value = (packed[cell / 4] >> (cell % 4 * 8)) & 255;
                    if start == end {
                        assert_eq!(value, 0);
                    } else if !average {
                        assert_eq!(value, u32::from(values[indices[start] as usize]));
                    } else {
                        let sum: u32 = indices[start..end]
                            .iter()
                            .map(|&i| u32::from(values[i as usize]))
                            .sum();
                        let count = (end - start) as u32;
                        let (q, rem) = (sum / count, sum % count);
                        if rem * 2 == count {
                            assert!(value == q || value == q + 1);
                        } else {
                            assert_eq!(value, q + u32::from(rem * 2 > count));
                        }
                    }
                }
            }
        }
    }
    cells * channels.len() * 2
}

#[test]
fn voxel_u8_batch_kernels_match_parent_cpu_ties_tail_padding_and_guards() {
    let r = WgpuRuntime::new_headless().unwrap();
    if plan_batched_voxel_u8(&r.device().limits(), 65_537, 4).is_none() {
        eprintln!("VOXEL_U8_BATCH_KERNEL unavailable: device limits");
        return;
    }
    let mut compared = 0;
    for cells in [1, 2, 3, 4, 5, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 65_537] {
        let mut points = 0;
        let starts: Vec<_> = (0..cells)
            .map(|i| {
                let start = points;
                points += [0, 1, 2, 3, 4, 7, 17][i % 7];
                start as u32
            })
            .collect();
        let indices: Vec<_> = (0..points).rev().map(|i| i as u32).collect();
        for u in 2..=4 {
            let channels: Vec<Vec<_>> = (0..u)
                .map(|c| (0..points).map(|i| ((i * 37 + c * 71) % 256) as u8).collect())
                .collect();
            compared += compare_kernels(&r, &channels, &indices, &starts, true);
        }
    }
    // Every ordered two-byte half tie, including both ends of the byte range.
    let mut ties = Vec::new();
    for a in 0..=255u8 {
        for b in 0..=255u8 {
            if (u32::from(a) + u32::from(b)) % 2 == 1 {
                ties.extend([a, b]);
            }
        }
    }
    let channels =
        vec![ties.clone(), ties.iter().map(|v| 255 - v).collect(), ties.clone(), ties.clone()];
    compared += compare_kernels(
        &r,
        &channels,
        &(0..ties.len() as u32).collect::<Vec<_>>(),
        &(0..ties.len() as u32).step_by(2).collect::<Vec<_>>(),
        true,
    );
    // Sequential f32 accumulation beyond 2^24 must match the original GPU, not an integer sum.
    let n = 262_145;
    let channels: Vec<Vec<_>> =
        (0..4).map(|c| (0..n).map(|i| ((i * 37 + c * 71) % 256) as u8).collect()).collect();
    compared +=
        compare_kernels(&r, &channels, &(0..n as u32).rev().collect::<Vec<_>>(), &[0], false);
    eprintln!("VOXEL_U8_BATCH_KERNEL verified {compared} cell results; all 32768 ordered half ties, byte codes, guards and long-cell f32 sums");
}

#[test]
fn public_voxel_u8_batch_matches_complete_parent_cpu_validation_and_fallbacks() {
    let r = WgpuRuntime::new_headless().unwrap();
    let supported = plan_batched_voxel_u8(&r.device().limits(), 257, 4).is_some();
    let mut compared = 0;
    for cells in [0, 1, 3, 4, 5, 17, 257, 65_537, 1_048_577] {
        for u in [0, 1, 2, 3, 4, 5, 7] {
            if cells > 65_537 && (u == 0 || u == 7) {
                continue;
            }
            for attrs in [0, 4] {
                let occupancy = if cells > 65_537 { 1 } else { 4 };
                let mut f = fixture(&r, cells, attrs, u, occupancy);
                if cells == 257 {
                    f.segments = crate::kernels::voxel_sort::build_voxel_segments_gpu_from_keys(
                        &r,
                        &(0..cells * occupancy)
                            .map(|i| ((i % cells) as i64, 0, 0))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                }
                for mode in 0..4 {
                    let old = call_baseline(&r, &f, mode);
                    let live = call(&r, &f, mode, false);
                    assert_output(&live, &old);
                    assert_output(&live, &expected(&f, mode));
                    assert_capacity(&live, cells, attrs, u, false);
                    compared += cells * (3 + attrs + u) * 2;
                }
            }
        }
    }
    for attrs in [1, 3, 5] {
        let f = fixture(&r, 17, attrs, 3, 4);
        for mode in 0..4 {
            assert_output(&call(&r, &f, mode, false), &call_baseline(&r, &f, mode));
        }
    }
    let f = fixture(&r, 17, 4, 3, 4);
    let attrs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
    let bytes: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
    for mode in 0..4 {
        let mut bad_attrs = attrs.clone();
        bad_attrs[2] = &f.floats[5][..67];
        let mut bad_bytes = bytes.clone();
        bad_bytes[0] = &f.bytes[0][..66];
        for (a, b, found) in [(&bad_attrs, &bad_bytes, 67), (&attrs, &bad_bytes, 66)] {
            let stats = r.buffer_pool().stats();
            assert_eq!(
                invoke(&r, &f, a, b, mode, false).unwrap_err(),
                SpatialError::BufferLengthMismatch { expected: 68, found }
            );
            assert_eq!(stats, r.buffer_pool().stats());
        }
    }
    eprintln!("VOXEL_U8_BATCH_PUBLIC verified {compared} complete parent/CPU elements; fused_supported={supported}");
}

#[test]
fn voxel_u8_batch_uploads_changed_host_bytes_and_returns_owned_results() {
    let r = WgpuRuntime::new_headless().unwrap();
    for u in 2..=4 {
        for mode in 0..4 {
            let mut f = fixture(&r, 257, 4, u, 4);
            let old = call(&r, &f, mode, false);
            let snapshot = old.clone();
            for bytes in &mut f.bytes {
                bytes.fill(231);
            }
            let changed = call(&r, &f, mode, false);
            assert_output(&changed, &call_baseline(&r, &f, mode));
            assert!(changed.4.iter().flatten().all(|&b| b == 231));
            drop(f);
            r.clear_buffer_pool();
            assert_output(&old, &snapshot);
            assert!(changed.4.iter().flatten().all(|&b| b == 231));
        }
    }
}

#[test]
fn voxel_u8_batch_lazy_caches_are_shared_by_concurrent_cold_calls() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    if plan_batched_voxel_u8(&r.device().limits(), 257, 4).is_none() {
        return;
    }
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let handles: Vec<_> = (2..=4)
        .map(|u| {
            let r = r.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let f = fixture(&r, 257, 4, u, 4);
                barrier.wait();
                for mode in 0..4 {
                    assert_output(&call(&r, &f, mode, false), &expected(&f, mode));
                }
                [
                    r.pipelines().voxel_gather.batched_u8_pipeline(r.device()) as *const _ as usize,
                    r.pipelines().voxel_reduce.batched_u8_pipeline(r.device()) as *const _ as usize,
                ]
            })
        })
        .collect();
    let pointers: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(pointers.windows(2).all(|p| p[0] == p[1]));
}

#[path = "voxel_u8_batch_benchmark.rs"]
mod benchmark;
