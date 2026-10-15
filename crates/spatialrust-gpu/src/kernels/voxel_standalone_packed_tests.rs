use super::*;
use crate::kernels::voxel_standalone_packed::record_packed_voxel_window;
use crate::kernels::voxel_standalone_packed::{
    gather_voxel_channels_packed_with_plan, plan_packed_voxel_submission,
};
#[path = "voxel_standalone_packed_reference.rs"]
mod baseline;
fn baseline_call(r: &WgpuRuntime, f: &Fixture) -> Channels {
    baseline::baseline_gather_voxel_first_f32_multi_gpu(
        r,
        &f.source.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &f.segments,
    )
    .unwrap()
}

#[test]
fn standalone_packed_gather_planner_preserves_width_and_two_batch_bounds() {
    for width in [2, 4] {
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 7.max((2 + 2 * width) as u32),
            max_bindings_per_bind_group: 8.max((3 + 2 * width) as u32),
            max_compute_workgroup_size_x: 256,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroups_per_dimension: 2,
            max_storage_buffer_binding_size: 1028 * width as u32,
            max_buffer_size: 1028 * (2 * width) as u64,
            ..wgpu::Limits::downlevel_defaults()
        };
        for channels in [1, 2, 3, 4, 5, 7, 8, 9, 11, 17, usize::MAX] {
            let p = plan_packed_voxel_submission(&limits, 257, channels, width, u64::MAX).unwrap();
            let window = if channels > width { channels.min(2 * width) } else { channels };
            assert_eq!(p.width, width);
            assert_eq!(p.channels_per_submission, window);
            assert_eq!(p.channel_bytes, 1028);
            let exact = p.channel_bytes * window as u64;
            assert_eq!(plan_packed_voxel_submission(&limits, 257, channels, width, exact), Some(p));
            assert!(
                plan_packed_voxel_submission(&limits, 257, channels, width, exact - 1).is_none()
            );
            let mut small = limits.clone();
            small.max_storage_buffer_binding_size = 1028 * channels.min(width) as u32 - 1;
            assert!(plan_packed_voxel_submission(&small, 257, channels, width, u64::MAX).is_none());
        }
        for boundary in 0..7 {
            let mut l = limits.clone();
            match boundary {
                0 => l.max_storage_buffers_per_shader_stage -= 1,
                1 => l.max_bindings_per_bind_group -= 1,
                2 => l.max_compute_workgroup_size_x -= 1,
                3 => l.max_compute_invocations_per_workgroup -= 1,
                4 => l.max_compute_workgroups_per_dimension -= 1,
                5 => l.max_storage_buffer_binding_size -= 1,
                _ => l.max_buffer_size -= 1,
            }
            assert!(plan_packed_voxel_submission(&l, 257, 2 * width + 1, width, u64::MAX).is_none());
        }
        let mut output_limited = limits.clone();
        output_limited.max_storage_buffer_binding_size = 1028 * width as u32 - 1;
        assert!(crate::kernels::voxel_gather_submit::plan_voxel_gather_submission(
            &output_limited,
            257,
            2 * width + 1,
            width
        )
        .is_some());
        assert!(plan_packed_voxel_submission(&output_limited, 257, 2 * width + 1, width, u64::MAX)
            .is_none());
        assert!(plan_packed_voxel_submission(&limits, 0, 7, width, u64::MAX).is_none());
        assert!(plan_packed_voxel_submission(&limits, 257, 0, width, u64::MAX).is_none());
        let wide = wgpu::Limits {
            max_buffer_size: u64::MAX,
            max_storage_buffer_binding_size: u32::MAX,
            max_compute_workgroups_per_dimension: u32::MAX,
            ..limits
        };
        let cells = (i32::MAX as u64 / (8 * width) as u64) as u32;
        assert!(plan_packed_voxel_submission(&wide, cells, 2 * width + 1, width, i32::MAX as u64)
            .is_some());
        assert!(plan_packed_voxel_submission(
            &wide,
            cells + 1,
            2 * width + 1,
            width,
            i32::MAX as u64
        )
        .is_none());
        assert!(plan_packed_voxel_submission(&wide, u32::MAX, 7, width, u64::MAX).is_none());
    }
    for width in [0, 1, 3, 5, usize::MAX] {
        assert!(plan_packed_voxel_submission(&wgpu::Limits::default(), 257, 7, width, u64::MAX)
            .is_none());
    }
}

fn read_words(r: &WgpuRuntime, b: &wgpu::Buffer) -> Vec<u32> {
    let slice = b.slice(..);
    let (send, recv) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |x| send.send(x).unwrap());
    r.device().poll(wgpu::Maintain::Wait);
    recv.recv().unwrap().unwrap();
    let view = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    b.unmap();
    result
}

#[test]
fn standalone_packed_gather_forced_widths_keep_single_tails_raw_bits_and_staging_guards() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    for width in [2, 4] {
        if plan_packed_voxel_submission(&r.device().limits(), 257, 7, width, isize::MAX as u64)
            .is_none()
        {
            continue;
        }
        for cells in [1, 3, 257, 65_537] {
            for channels in 1..=2 * width {
                let mut f = fixture(&r, cells, channels, 1);
                for (c, v) in f.source.iter_mut().enumerate() {
                    *v = words(cells + c * 7).into_iter().skip(c * 7).map(f32::from_bits).collect();
                    v[0] =
                        f32::from_bits([0x3fa0_0000, 0x8000_0000, 0x7fc0_1234, 0x7f80_0001][c % 4]);
                }
                let starts: Vec<_> = (0..cells)
                    .map(|i| match i % 5 {
                        0 | 1 => 0,
                        2 => (cells - 1) as u32,
                        3 => (cells / 2) as u32,
                        _ => i as u32,
                    })
                    .collect();
                let indices: Vec<_> = (0..cells).rev().map(|i| i as u32).collect();
                let upload = |label, data: &[u32]| r.upload_u32_storage(label, data).unwrap();
                f.segments = GpuVoxelSegments::new(
                    cells as u32,
                    cells as u32,
                    upload("standalone-packed-keys", &[0]),
                    upload("standalone-packed-indices", &indices),
                    upload("standalone-packed-starts", &starts),
                    0,
                );
                let expected: Channels = f
                    .source
                    .iter()
                    .map(|v| {
                        (0..cells)
                            .map(|i| {
                                let start = starts[i];
                                let end = starts.get(i + 1).copied().unwrap_or(cells as u32);
                                v[if end <= start { 0 } else { indices[start as usize] as usize }]
                            })
                            .collect()
                    })
                    .collect();
                let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
                let plan = plan_packed_voxel_submission(
                    &r.device().limits(),
                    cells as u32,
                    channels,
                    width,
                    isize::MAX as u64,
                )
                .unwrap();
                let region_end = 5 + cells * channels;
                let sentinel = vec![0xdead_beef_u32; region_end + 8];
                let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("standalone-packed-offset-guard"),
                    size: (sentinel.len() * 4) as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                r.queue().write_buffer(&staging, 0, bytemuck::cast_slice(&sentinel));
                let stats = r.buffer_pool().stats();
                let mut encoder = r.device().create_command_encoder(&Default::default());
                record_packed_voxel_window(
                    &mut encoder,
                    &r,
                    &refs,
                    &f.segments,
                    &staging,
                    20,
                    plan,
                );
                r.queue().submit(Some(encoder.finish()));
                let got = read_words(&r, &staging);
                assert_eq!(&got[..5], &[0xdead_beef; 5]);
                assert_eq!(&got[region_end..], &[0xdead_beef; 8]);
                for c in 0..channels {
                    for cell in 0..cells {
                        assert_eq!(got[5 + c * cells + cell], expected[c][cell].to_bits());
                    }
                }
                let live =
                    gather_voxel_channels_packed_with_plan(&r, &refs, &f.segments, plan).unwrap();
                assert_bits(&live, &expected);
                assert_bits(&live, &baseline_call(&r, &f));
                assert_eq!(capacity(&live), cells * channels);
                assert_eq!(r.buffer_pool().stats(), stats);
                drop(refs);
                drop(f);
                r.clear_buffer_pool();
                assert_bits(&live, &expected);
                compared += cells * channels * 3;
            }
        }
    }
    eprintln!("VOXEL_STANDALONE_PACKED_WIDTHS verified {compared} raw/parent/CPU words with widths 2/4; empty single tails, nonfinite bits, prefix/suffix guards and pool ownership");
}
#[test]
fn public_standalone_packed_gather_matches_complete_parent_cpu_and_ordered_validation() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    for cells in [0, 1, 3, 17, 255, 256, 257, 65_537, 1_048_577] {
        for channels in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16, 17] {
            if cells > 65_537 && ![5, 7, 8, 9].contains(&channels) {
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
                assert_eq!(f.segments.construction_readback_bytes(), 4);
            }
            let old = baseline_call(&r, &f);
            let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
            let live = gather_voxel_first_f32_multi_gpu(&r, &refs, &f.segments).unwrap();
            assert_bits(&live, &old);
            assert_bits(&live, &f.first);
            assert_eq!(capacity(&live), cells * channels);
            assert_eq!(capacity(&old), cells * channels);
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
        let before = r.buffer_pool().stats();
        let error = gather_voxel_first_f32_multi_gpu(&r, &bad, &f.segments).unwrap_err();
        assert_eq!(error, SpatialError::BufferLengthMismatch { expected: 68, found });
        assert_eq!(
            error,
            baseline::baseline_gather_voxel_first_f32_multi_gpu(&r, &bad, &f.segments).unwrap_err()
        );
        assert_eq!(r.buffer_pool().stats(), before);
    }
    let empty = fixture(&r, 0, 0, 1);
    assert_eq!(
        gather_voxel_first_f32_multi_gpu(&r, &[&[123.0][..]; 11], &empty.segments).unwrap(),
        vec![Vec::<f32>::new(); 11]
    );
    eprintln!(
        "VOXEL_STANDALONE_PACKED_PUBLIC verified {compared} complete parent/CPU words; batch_width={}",
        r.max_gather_channels()
    );
}

#[test]
fn standalone_packed_gather_preserves_sparse_descending_and_raw_float_payloads() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    for channels in [1, 2, 3, 4, 5, 7, 8, 9, 11, 17] {
        let mut f = fixture(&r, 1025, channels, 1);
        for (channel, source) in f.source.iter_mut().enumerate() {
            *source = words(1025 + channel * 7)
                .into_iter()
                .skip(channel * 7)
                .map(f32::from_bits)
                .collect();
        }
        // Empty and descending pairs keep the original multi-gather point-zero rule.
        let starts: Vec<_> = (0..1025)
            .map(|i| match i % 5 {
                0 => 0,
                1 => 0,
                2 => 7,
                3 => 3,
                _ => 17,
            })
            .collect();
        let indices: Vec<_> = (0..1025).rev().map(|i| i as u32).collect();
        let upload = |label, data: &[u32]| r.upload_u32_storage(label, data).unwrap();
        f.segments = GpuVoxelSegments::new(
            1025,
            1025,
            upload("submit-sparse-keys", &[0]),
            upload("submit-sparse-indices", &indices),
            upload("submit-sparse-starts", &starts),
            0,
        );
        let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
        let live = gather_voxel_first_f32_multi_gpu(&r, &refs, &f.segments).unwrap();
        assert_bits(&live, &baseline_call(&r, &f));
        let expected: Channels = f
            .source
            .iter()
            .map(|source| {
                (0..1025)
                    .map(|cell| {
                        let start = starts[cell];
                        let end = starts.get(cell + 1).copied().unwrap_or(1025);
                        source[if end <= start { 0 } else { indices[start as usize] as usize }]
                    })
                    .collect()
            })
            .collect();
        assert_bits(&live, &expected);
        compared += 1025 * channels * 2;
    }
    eprintln!("VOXEL_STANDALONE_PACKED_SPARSE verified {compared} raw parent/CPU words including signed zeros and NaNs");
}

#[test]
fn standalone_packed_gather_concurrent_calls_keep_changed_inputs_and_owned_outputs_independent() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let handles: Vec<_> = [5, 8, 11]
        .into_iter()
        .map(|channels| {
            let r = r.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let mut f = fixture(&r, 16_385, channels, 4);
                barrier.wait();
                let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
                let old = gather_voxel_first_f32_multi_gpu(&r, &refs, &f.segments).unwrap();
                assert_bits(&old, &f.first);
                drop(refs);
                for source in &mut f.source {
                    source.fill(-0.0);
                }
                let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
                let changed = gather_voxel_first_f32_multi_gpu(&r, &refs, &f.segments).unwrap();
                assert!(changed.iter().flatten().all(|v| v.to_bits() == 0x8000_0000));
                let expected = f.first.clone();
                drop(refs);
                drop(f);
                r.clear_buffer_pool();
                assert_bits(&old, &expected);
                assert!(changed.iter().flatten().all(|v| v.to_bits() == 0x8000_0000));
                r.pipelines().voxel_gather.packed_attributes_pipeline(r.device()) as *const _
                    as usize
            })
        })
        .collect();
    let pointers: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(pointers.windows(2).all(|p| p[0] == p[1]));
}
#[path = "voxel_standalone_packed_benchmark.rs"]
mod benchmark;
