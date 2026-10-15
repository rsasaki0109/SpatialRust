use super::*;
use crate::kernels::voxel_gather_submit::{
    gather_voxel_channels_with_plan, plan_voxel_gather_submission,
    plan_voxel_gather_submission_with_host_limit,
};

#[path = "voxel_gather_submit_reference.rs"]
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
fn voxel_gather_submission_planner_bounds_two_batches_and_exact_device_limits() {
    for width in [2, 4] {
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: (2 + 2 * width) as u32,
            max_bindings_per_bind_group: (3 + 2 * width) as u32,
            max_compute_workgroup_size_x: 256,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroups_per_dimension: 2,
            max_storage_buffer_binding_size: 1028,
            max_buffer_size: 1028 * (2 * width) as u64,
            ..wgpu::Limits::downlevel_defaults()
        };
        let plan = plan_voxel_gather_submission(&limits, 257, 2 * width + 1, width).unwrap();
        assert_eq!(plan.channels_per_submission, 2 * width);
        assert_eq!(plan.channel_bytes, 1028);
        assert_eq!(plan.batch_width, width);
        let exact_host_limit = plan.channel_bytes * plan.channels_per_submission as u64;
        assert_eq!(
            plan_voxel_gather_submission_with_host_limit(
                &limits,
                257,
                2 * width + 1,
                width,
                exact_host_limit,
            ),
            Some(plan)
        );
        assert!(plan_voxel_gather_submission_with_host_limit(
            &limits,
            257,
            2 * width + 1,
            width,
            exact_host_limit - 1,
        )
        .is_none());
        assert_eq!(plan_voxel_gather_submission(&limits, 257, usize::MAX, width), Some(plan));
        assert_eq!(
            plan_voxel_gather_submission(&limits, 257, width + 1, width)
                .unwrap()
                .channels_per_submission,
            width + 1
        );
        for channels in 0..=width {
            assert!(plan_voxel_gather_submission(&limits, 257, channels, width).is_none());
        }
        assert!(plan_voxel_gather_submission(&limits, 0, 7, width).is_none());
        for bound in 0..7 {
            let mut l = limits.clone();
            match bound {
                0 => l.max_storage_buffers_per_shader_stage -= 1,
                1 => l.max_bindings_per_bind_group -= 1,
                2 => l.max_compute_workgroup_size_x -= 1,
                3 => l.max_compute_invocations_per_workgroup -= 1,
                4 => l.max_compute_workgroups_per_dimension -= 1,
                5 => l.max_storage_buffer_binding_size -= 1,
                6 => l.max_buffer_size -= 1,
                _ => unreachable!(),
            }
            assert!(
                plan_voxel_gather_submission(&l, 257, 2 * width + 1, width).is_none(),
                "width={width} bound={bound}"
            );
        }
        assert!(plan_voxel_gather_submission(&limits, u32::MAX, 11, width).is_none());
        let large_limits = wgpu::Limits {
            max_buffer_size: u64::MAX,
            max_storage_buffer_binding_size: u32::MAX,
            max_compute_workgroups_per_dimension: u32::MAX,
            ..limits
        };
        let cells = (i32::MAX as u64 / (8 * width) as u64) as u32;
        assert!(plan_voxel_gather_submission_with_host_limit(
            &large_limits,
            cells,
            2 * width + 1,
            width,
            i32::MAX as u64,
        )
        .is_some());
        assert!(plan_voxel_gather_submission_with_host_limit(
            &large_limits,
            cells + 1,
            2 * width + 1,
            width,
            i32::MAX as u64,
        )
        .is_none());
    }
    for width in [0, 1, 3, 5, usize::MAX] {
        assert!(plan_voxel_gather_submission(&wgpu::Limits::default(), 257, 11, width).is_none());
    }
}

#[test]
fn public_voxel_gather_submission_matches_complete_parent_cpu_and_ordered_validation() {
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
        "VOXEL_GATHER_SUBMIT_PUBLIC verified {compared} complete parent/CPU words; batch_width={}",
        r.max_gather_channels()
    );
}

#[test]
fn voxel_gather_submission_preserves_sparse_descending_and_raw_float_payloads() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    for channels in [5, 7, 8, 9, 11, 17] {
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
    eprintln!("VOXEL_GATHER_SUBMIT_SPARSE verified {compared} raw parent/CPU words including signed zeros and NaNs");
}

#[test]
fn voxel_gather_submission_executes_two_and_four_channel_batches_and_partial_windows() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    for width in [2, 4] {
        if width > r.max_gather_channels() as usize {
            continue;
        }
        for channels in [3, 5, 7, 8, 9, 11, 16, 17] {
            if channels <= width {
                continue;
            }
            let f = fixture(&r, 257, channels, 3);
            let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
            let plan =
                plan_voxel_gather_submission(&r.device().limits(), 257, channels, width).unwrap();
            let live = gather_voxel_channels_with_plan(&r, &refs, &f.segments, plan).unwrap();
            assert_bits(&live, &f.first);
            assert_bits(&live, &baseline_call(&r, &f));
            assert_eq!(capacity(&live), 257 * channels);
            assert_eq!(
                channels.div_ceil(plan.channels_per_submission),
                channels.div_ceil(2 * width)
            );
            compared += 257 * channels * 2;
        }
    }
    eprintln!("VOXEL_GATHER_SUBMIT_WIDTHS verified {compared} parent/CPU words with forced existing batch widths");
}

#[test]
fn voxel_gather_submission_concurrent_calls_keep_changed_inputs_and_owned_outputs_independent() {
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
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[path = "voxel_gather_submit_benchmark.rs"]
mod benchmark;
