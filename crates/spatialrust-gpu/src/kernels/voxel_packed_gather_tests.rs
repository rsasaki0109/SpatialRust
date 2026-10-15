use super::*;
use crate::kernels::voxel_gather::{
    record_voxel_gather_f32_pass, record_voxel_gather_multi2_pass, record_voxel_gather_multi4_pass,
};
use crate::kernels::voxel_gather_packed::{
    plan_packed_voxel_attributes, record_packed_voxel_attributes,
    try_record_packed_voxel_attributes,
};
#[path = "voxel_packed_gather_reference.rs"]
mod baseline;

#[test]
fn packed_voxel_gather_planner_checks_width_output_region_dispatch_and_host_limits() {
    for width in [2, 4] {
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 7,
            max_bindings_per_bind_group: 8,
            max_compute_workgroup_size_x: 256,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroups_per_dimension: 2,
            max_storage_buffer_binding_size: 1028 * width as u32,
            max_buffer_size: 11060,
            ..wgpu::Limits::downlevel_defaults()
        };
        let plan =
            plan_packed_voxel_attributes(&limits, 257, 7, width, 11060, 3084, 11060).unwrap();
        assert_eq!(plan.width, width);
        assert_eq!(plan.channel_bytes, 1028);
        assert_eq!(plan.offset, 3084);
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
            assert!(plan_packed_voxel_attributes(&l, 257, 7, width, 11060, 3084, 11060).is_none());
        }
        assert!(plan_packed_voxel_attributes(&limits, 257, 7, width, 11060, 3084, 11059).is_none());
        assert!(plan_packed_voxel_attributes(&limits, 257, 7, width, 10279, 3084, 11060).is_none());
        for offset in [1, 2, 3, 3085, u64::MAX - 3] {
            assert!(plan_packed_voxel_attributes(&limits, 257, 7, width, 11060, offset, 11060)
                .is_none());
        }
        for channels in [0, 1, usize::MAX] {
            assert!(plan_packed_voxel_attributes(
                &limits, 257, channels, width, 11060, 3084, 11060
            )
            .is_none());
        }
        assert!(plan_packed_voxel_attributes(&limits, 0, 7, width, 11060, 3084, 11060).is_none());
        let wide = wgpu::Limits {
            max_buffer_size: u64::MAX,
            max_storage_buffer_binding_size: u32::MAX,
            max_compute_workgroups_per_dimension: u32::MAX,
            ..limits
        };
        let cells = (i32::MAX as u64 / 40) as u32;
        for (n, accepted) in [(cells, true), (cells + 1, false)] {
            assert_eq!(
                plan_packed_voxel_attributes(
                    &wide,
                    n,
                    7,
                    width,
                    u64::from(n) * 40,
                    u64::from(n) * 12,
                    i32::MAX as u64
                )
                .is_some(),
                accepted
            );
        }
        assert!(plan_packed_voxel_attributes(&wide, u32::MAX, 7, width, u64::MAX, 0, u64::MAX)
            .is_none());
        assert!(plan_packed_voxel_attributes(&wide, 257, usize::MAX, width, u64::MAX, 0, u64::MAX)
            .is_none());
    }
    for width in [0, 1, 3, 5, usize::MAX] {
        assert!(plan_packed_voxel_attributes(&wgpu::Limits::default(), 1, 7, width, 40, 12, 40)
            .is_none());
    }
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

#[test]
fn packed_voxel_gather_matches_original_two_four_and_scalar_words_with_offset_guards() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    if plan_packed_voxel_attributes(&r.device().limits(), 257, 7, 4, 11060, 3084, isize::MAX as u64)
        .is_none()
    {
        eprintln!("VOXEL_PACKED_GATHER_RECORDER unavailable: device limits");
        return;
    }
    for cells in [1, 3, 17, 257, 65_537] {
        let mut points = 1;
        let starts: Vec<_> = (0..cells)
            .map(|i| {
                let start = points;
                points += [0, 1, 2, 3, 4, 7, 17][i % 7];
                start as u32
            })
            .collect();
        let indices: Vec<_> = (0..points).rev().map(|i| i as u32).collect();
        let upload = |name, data: &[u32]| r.upload_u32_storage(name, data).unwrap();
        let segments = GpuVoxelSegments::new(
            cells as u32,
            points as u32,
            upload("packed-gather-keys", &[0]),
            upload("packed-gather-indices", &indices),
            upload("packed-gather-starts", &starts),
            0,
        );
        for channels in [2, 3, 4, 5, 7, 8, 9, 11, 17] {
            let source: Vec<Vec<f32>> = (0..channels)
                .map(|c| {
                    (0..points)
                        .map(|i| {
                            f32::from_bits(
                                [
                                    0x3fa0_0000,
                                    0x7fc0_1234,
                                    0x7f80_0001,
                                    0x7f80_0000,
                                    0xff80_0000,
                                    0x8000_0000,
                                    1,
                                    0x007f_ffff,
                                    0x0080_0000,
                                    0x7f7f_ffff,
                                    0xffc0_9876,
                                    0x4248_0000,
                                    0xff7f_ffff,
                                ][(i * 17 + c * 7) % 13],
                            )
                        })
                        .collect()
                })
                .collect();
            let values: Vec<_> = source
                .iter()
                .map(|v| r.upload_f32_storage("packed-gather-parent-values", v).unwrap())
                .collect();
            let refs: Vec<_> = source.iter().map(Vec::as_slice).collect();
            for width in [2, 4] {
                let old: Vec<_> = (0..channels + 4)
                    .map(|_| upload("packed-gather-parent-output", &vec![0xdead_beef; cells + 8]))
                    .collect();
                let region_end = 5 + cells * channels;
                let parent_base = region_end + 8;
                let words = vec![0xdead_beef_u32; parent_base + channels * (cells + 8)];
                let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("packed-gather-offset-staging"),
                    size: (words.len() * 4) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                r.queue().write_buffer(&staging, 0, bytemuck::cast_slice(&words));
                let mut encoder =
                    r.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                let plan = plan_packed_voxel_attributes(
                    &r.device().limits(),
                    cells as u32,
                    channels,
                    width,
                    staging.size(),
                    20,
                    isize::MAX as u64,
                )
                .unwrap();
                let mut recycle = Vec::new();
                record_packed_voxel_attributes(
                    &mut encoder,
                    &r,
                    &refs,
                    &segments,
                    &staging,
                    &mut recycle,
                    plan,
                )
                .unwrap();
                assert_eq!(recycle.len(), channels);
                for start in (0..channels).step_by(width) {
                    let count = (channels - start).min(width);
                    if count == 1 {
                        record_voxel_gather_f32_pass(
                            &mut encoder,
                            &r,
                            &values[start],
                            segments.point_indices_buffer(),
                            segments.cell_starts_buffer(),
                            cells as u32,
                            points as u32,
                            &old[start],
                        )
                        .unwrap();
                    } else if count == 2 {
                        let inputs = std::array::from_fn(|i| &values[start + i]);
                        let outputs = std::array::from_fn(|i| &old[start + i]);
                        record_voxel_gather_multi2_pass(
                            &mut encoder,
                            &r,
                            &inputs,
                            &segments,
                            &outputs,
                            count as u32,
                        )
                        .unwrap();
                    } else {
                        let inputs =
                            std::array::from_fn(|i| values.get(start + i).unwrap_or(&values[0]));
                        let outputs = std::array::from_fn(|i| &old[start + i]);
                        record_voxel_gather_multi4_pass(
                            &mut encoder,
                            &r,
                            &inputs,
                            &segments,
                            &outputs,
                            count as u32,
                        )
                        .unwrap();
                    }
                }
                for (c, output) in old.iter().take(channels).enumerate() {
                    encoder.copy_buffer_to_buffer(
                        output,
                        0,
                        &staging,
                        ((parent_base + c * (cells + 8)) * 4) as u64,
                        output.size(),
                    );
                }
                r.queue().submit(Some(encoder.finish()));
                let result = read_words(&r, &staging);
                assert_eq!(&result[..5], &[0xdead_beef; 5]);
                assert_eq!(&result[region_end..parent_base], &[0xdead_beef; 8]);
                for c in 0..channels {
                    let parent = parent_base + c * (cells + 8);
                    assert_eq!(
                        &result[5 + c * cells..5 + (c + 1) * cells],
                        &result[parent..parent + cells],
                        "channels={channels} width={width} channel={c}"
                    );
                    assert_eq!(&result[parent + cells..parent + cells + 8], &[0xdead_beef; 8]);
                    for cell in 0..cells {
                        let start = starts[cell] as usize;
                        let end = starts.get(cell + 1).copied().unwrap_or(points as u32) as usize;
                        let expected = if end <= start {
                            if channels % width == 1 && c == channels - 1 {
                                0
                            } else {
                                source[c][0].to_bits()
                            }
                        } else {
                            source[c][indices[start] as usize].to_bits()
                        };
                        assert_eq!(result[5+c*cells+cell],expected,"independent word: channels={channels} width={width} channel={c} cell={cell}");
                    }
                }
                let stats = r.buffer_pool().stats();
                let mut encoder =
                    r.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                for offset in [2, u64::MAX - 3, staging.size()] {
                    assert!(!try_record_packed_voxel_attributes(
                        &mut encoder,
                        &r,
                        &refs,
                        &segments,
                        &staging,
                        offset,
                        &mut recycle
                    )
                    .unwrap());
                }
                assert_eq!(recycle.len(), channels);
                assert_eq!(r.buffer_pool().stats(), stats);
                r.queue().submit(Some(encoder.finish()));
                assert_eq!(read_words(&r, &staging), result);
                for buffer in recycle {
                    r.recycle_storage(buffer.size(), buffer);
                }
                compared += cells * channels * 2;
            }
        }
    }
    eprintln!("VOXEL_PACKED_GATHER_RECORDER verified {compared} parent/independent words; widths 2/4, empty/scalar tails, nonfinite bits and staging/output guards");
}
fn baseline_call(r: &WgpuRuntime, f: &Fixture, mode: usize) -> Output {
    let function = if mode == 0 {
        baseline::baseline_gather_voxel_first_xyz_and_multi_gpu
    } else {
        baseline::baseline_reduce_voxel_centroids_xyz_and_gather_first_multi_gpu
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

#[path = "voxel_packed_gather_benchmark.rs"]
mod benchmark;
#[test]
fn public_packed_voxel_gathers_match_complete_parents_cpu_and_ordered_errors() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    for cells in [0, 1, 3, 17, 257, 65_537, 1_048_577] {
        for attributes in [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 17] {
            for bytes in [0, 1, 3, 5] {
                if cells > 65_537
                    && (![5, 7, 8, 9].contains(&attributes) || ![0, 3].contains(&bytes))
                {
                    continue;
                }
                let occupancy = if cells > 65_537 { 1 } else { 4 };
                let mut f = fixture(&r, cells, attributes, bytes, occupancy);
                if cells == 257 {
                    f.segments = crate::kernels::voxel_sort::build_voxel_segments_gpu_from_keys(
                        &r,
                        &(0..cells * occupancy)
                            .map(|i| ((i % cells) as i64, 0, 0))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                    assert_eq!(f.segments.construction_readback_bytes(), 4);
                }
                for mode in [0, 3] {
                    let old = baseline_call(&r, &f, mode);
                    let stats = r.buffer_pool().stats();
                    let live = call(&r, &f, mode, false);
                    assert_output(&live, &old);
                    assert_output(&live, &expected(&f, mode));
                    assert_capacity(&live, cells, attributes, bytes, false);
                    let live_stats = r.buffer_pool().stats();
                    assert_output(&baseline_call(&r, &f, mode), &live);
                    let parent_stats = r.buffer_pool().stats();
                    assert_eq!(live_stats.cached_bytes, parent_stats.cached_bytes);
                    assert_eq!(live_stats.cached_resources, parent_stats.cached_resources);
                    assert_eq!(live_stats.budget_bytes, parent_stats.budget_bytes);
                    assert_eq!(
                        live_stats.allocation_count - stats.allocation_count,
                        parent_stats.allocation_count - live_stats.allocation_count
                    );
                    assert_eq!(
                        live_stats.reuse_count - stats.reuse_count,
                        parent_stats.reuse_count - live_stats.reuse_count
                    );
                    assert_eq!(
                        live_stats.discarded_count - stats.discarded_count,
                        parent_stats.discarded_count - live_stats.discarded_count
                    );
                    compared += cells * (3 + attributes + bytes) * 2;
                }
            }
        }
    }
    let f = fixture(&r, 17, 11, 3, 4);
    let attrs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
    let bytes: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
    for mode in [0, 3] {
        for bad_index in [0, 4, 8, 10] {
            let mut bad = attrs.clone();
            bad[bad_index] = &f.floats[3 + bad_index][..67];
            let mut bad_bytes = bytes.clone();
            bad_bytes[0] = &f.bytes[0][..66];
            let stats = r.buffer_pool().stats();
            assert_eq!(
                invoke(&r, &f, &bad, &bad_bytes, mode, false).unwrap_err(),
                SpatialError::BufferLengthMismatch { expected: 68, found: 67 }
            );
            assert_eq!(
                invoke(&r, &f, &attrs, &bad_bytes, mode, false).unwrap_err(),
                SpatialError::BufferLengthMismatch { expected: 68, found: 66 }
            );
            assert_eq!(r.buffer_pool().stats(), stats);
        }
    }
    let empty = fixture(&r, 0, 11, 3, 1);
    for mode in [0, 3] {
        assert_capacity(&call(&r, &empty, mode, false), 0, 11, 3, false);
    }
    eprintln!("VOXEL_PACKED_GATHER_PUBLIC verified {compared} complete parent/CPU elements; batch_supported={}",plan_packed_voxel_attributes(&r.device().limits(),257,7,r.max_gather_channels() as usize,11060,3084,isize::MAX as u64).is_some());
}

#[test]
fn packed_voxel_gather_calls_reupload_changed_attributes_and_bytes_keep_owned_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    for attributes in [5, 7, 9, 11] {
        for mode in [0, 3] {
            let mut f = fixture(&r, 257, attributes, 3, 4);
            let old = call(&r, &f, mode, false);
            let expected = old.clone();
            for values in &mut f.floats[3..] {
                values.fill(64.25);
            }
            for values in &mut f.bytes {
                values.fill(231);
            }
            let live = call(&r, &f, mode, false);
            assert_output(&live, &baseline_call(&r, &f, mode));
            assert!(live.3.iter().flatten().all(|&v| v == 64.25));
            assert!(live.4.iter().flatten().all(|&v| v == 231));
            drop(f);
            r.clear_buffer_pool();
            assert_output(&old, &expected);
            assert!(live.3.iter().flatten().all(|&v| v == 64.25));
            assert!(live.4.iter().flatten().all(|&v| v == 231));
        }
    }
}

#[test]
fn packed_voxel_gather_concurrent_cold_calls_reuse_the_existing_lazy_pipeline() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    if plan_packed_voxel_attributes(
        &r.device().limits(),
        257,
        7,
        r.max_gather_channels() as usize,
        11060,
        3084,
        isize::MAX as u64,
    )
    .is_none()
    {
        return;
    }
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let handles: Vec<_> = [5, 7, 9]
        .into_iter()
        .map(|attributes| {
            let r = r.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let f = fixture(&r, 257, attributes, 3, 4);
                barrier.wait();
                for mode in [0, 3] {
                    let live = call(&r, &f, mode, false);
                    assert_output(&live, &expected(&f, mode));
                    assert_output(&live, &baseline_call(&r, &f, mode));
                }
                r.pipelines().voxel_gather.packed_attributes_pipeline(r.device()) as *const _
                    as usize
            })
        })
        .collect();
    let pointers: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(pointers.windows(2).all(|p| p[0] == p[1]));
}
