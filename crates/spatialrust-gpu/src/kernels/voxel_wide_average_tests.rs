use super::*;
use crate::kernels::{
    voxel_wide_average::{plan_wide_voxel_average, try_record_wide_voxel_average, WideAveragePlan},
    voxel_xyz_packed::XyzMode,
};

#[path = "voxel_wide_average_reference.rs"]
mod baseline;

fn invoke_current_parent(
    r: &WgpuRuntime,
    f: &Fixture,
    attrs: &[&[f32]],
    bytes: &[&[u8]],
    mode: usize,
    old: bool,
) -> SpatialResult<Output> {
    let function = match (mode, old) {
        (1, false) => gather_voxel_first_xyz_and_average_multi_gpu,
        (1, true) => baseline::baseline_gather_voxel_first_xyz_and_average_multi_gpu,
        (2, false) => reduce_voxel_centroids_xyz_and_average_multi_gpu,
        (2, true) => baseline::baseline_reduce_voxel_centroids_xyz_and_average_multi_gpu,
        _ => unreachable!(),
    };
    function(r, &f.xyz[0], &f.xyz[1], &f.xyz[2], attrs, bytes, &f.segments)
}

fn complete_call(r: &WgpuRuntime, f: &Fixture, mode: usize, old: bool) -> Output {
    invoke_current_parent(
        r,
        f,
        &f.floats[3..].iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &f.bytes.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        mode,
        old,
    )
    .unwrap()
}

fn read_words(r: &WgpuRuntime, buffer: &wgpu::Buffer) -> Vec<u32> {
    let slice = buffer.slice(..);
    let (send, recv) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |v| send.send(v).unwrap());
    r.device().poll(wgpu::Maintain::Wait);
    recv.recv().unwrap().unwrap();
    let view = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    buffer.unmap();
    result
}

#[test]
fn wide_voxel_average_planner_checks_prefix_tail_target_host_and_complete_fallback() {
    let limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 10,
        max_bindings_per_bind_group: 11,
        max_storage_buffer_binding_size: 257 * 28,
        max_buffer_size: 257 * 100,
        max_compute_workgroup_size_x: 256,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroups_per_dimension: 2,
        ..wgpu::Limits::downlevel_defaults()
    };
    let mut checked = 0;
    for channels in 0..=21 {
        for cells in [0, 1, 3, 255, 256, 257, 512, 513] {
            let bytes = u64::from(cells) * 4 * (3 + channels as u64);
            let want = channels > 4 && cells != 0 && cells <= 257;
            let plan = plan_wide_voxel_average(&limits, cells, channels, bytes, bytes);
            assert_eq!(plan.is_some(), want);
            if want {
                assert_eq!(
                    plan,
                    Some(WideAveragePlan {
                        prefix_bytes: u64::from(cells) * 28,
                        channel_bytes: u64::from(cells) * 4
                    })
                );
                assert!(
                    plan_wide_voxel_average(&limits, cells, channels, bytes - 1, bytes).is_none()
                );
                assert!(
                    plan_wide_voxel_average(&limits, cells, channels, bytes, bytes - 1).is_none()
                );
                checked += 3;
            } else {
                checked += 1;
            }
        }
    }
    for field in 0..13 {
        let mut l = limits.clone();
        match field {
            0 => l.max_storage_buffers_per_shader_stage = 9,
            1 => l.max_bindings_per_bind_group = 10,
            2 => l.max_bind_groups = 0,
            3 => l.max_uniform_buffers_per_shader_stage = 0,
            4 => l.max_uniform_buffer_binding_size = 15,
            5 => l.max_buffer_size = 15,
            6 => l.max_compute_workgroup_size_x = 255,
            7 => l.max_compute_workgroup_size_y = 0,
            8 => l.max_compute_workgroup_size_z = 0,
            9 => l.max_compute_invocations_per_workgroup = 255,
            10 => l.max_compute_workgroups_per_dimension = 1,
            11 => l.max_storage_buffer_binding_size -= 1,
            _ => l.max_buffer_size = 257 * 40 - 1,
        }
        assert!(plan_wide_voxel_average(&l, 257, 7, 257 * 40, u64::MAX).is_none());
        checked += 1;
    }
    let wide = wgpu::Limits {
        max_buffer_size: u64::MAX,
        max_storage_buffer_binding_size: u32::MAX,
        max_compute_workgroups_per_dimension: u32::MAX,
        ..limits
    };
    assert!(plan_wide_voxel_average(&wide, 1, usize::MAX, u64::MAX, u64::MAX).is_none());
    assert!(plan_wide_voxel_average(&wide, u32::MAX, 7, u64::MAX, u64::MAX).is_none());
    assert!(plan_wide_voxel_average(&wide, 257, 7, 257 * 40 + 3, 257 * 40 + 3).is_some());
    let cells = (i32::MAX as u64 / 40) as u32;
    assert!(
        plan_wide_voxel_average(&wide, cells, 7, u64::from(cells) * 40, i32::MAX as u64).is_some()
    );
    assert!(plan_wide_voxel_average(
        &wide,
        cells + 1,
        7,
        u64::from(cells + 1) * 40,
        i32::MAX as u64
    )
    .is_none());
    checked += 5;
    eprintln!("VOXEL_WIDE_AVERAGE_PLAN verified {checked} checked prefix/tail, device/dispatch, complete target, host and overflow plans");
}

#[test]
fn wide_voxel_average_recorder_preserves_raw_math_empty_cells_and_adjacent_bytes() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for cells in [1, 3, 257, 65_537] {
        for attributes in [5, 7, 8, 9, 11] {
            let mut f = fixture(&r, cells, attributes, 0, 4);
            let mut points = 1;
            let starts: Vec<_> = (0..cells)
                .map(|i| {
                    let start = points;
                    points += [0, 1, 3, 5, 7][i % 5];
                    start as u32
                })
                .collect();
            let indices: Vec<_> = (0..points).rev().map(|i| i as u32).collect();
            f.segments = GpuVoxelSegments::new(
                cells as u32,
                points as u32,
                r.upload_u32_storage("wide-average-keys", &[0]).unwrap(),
                r.upload_u32_storage("wide-average-indices", &indices).unwrap(),
                r.upload_u32_storage("wide-average-starts", &starts).unwrap(),
                0,
            );
            let special = words(12);
            f.floats = (0..3 + attributes)
                .map(|c| {
                    (0..points)
                        .map(|i| match c % 4 {
                            0 => ((i * 17 + c * 29) % 4096) as f32 * 0.25 - 511.5,
                            1 => f32::from_bits(special[(i + c) % 12]),
                            2 => [16_777_216.0, 1.0, -16_777_216.0, 3.0][i % 4],
                            _ => -0.0,
                        })
                        .collect()
                })
                .collect();
            f.xyz = std::array::from_fn(|c| {
                r.upload_f32_storage("wide-average-xyz", &f.floats[c]).unwrap()
            });
            let attrs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
            for mode in [1, 2] {
                let expected = complete_call(&r, &f, mode, true);
                let end = cells * (3 + attributes);
                let poison = vec![0xdead_beef_u32; end + 24];
                let target = staging(&r, &poison);
                let stats = r.buffer_pool().stats();
                let mut encoder = r.device().create_command_encoder(&Default::default());
                let supported = try_record_wide_voxel_average(
                    &mut encoder,
                    &r,
                    [&f.xyz[0], &f.xyz[1], &f.xyz[2]],
                    &attrs,
                    &f.segments,
                    &target,
                    if mode == 1 { XyzMode::First } else { XyzMode::Centroid },
                )
                .unwrap();
                r.queue().submit(Some(encoder.finish()));
                if !supported {
                    continue;
                }
                let got = read_words(&r, &target);
                for (c, plane) in floats(&expected).enumerate() {
                    assert_eq!(
                        &got[c * cells..(c + 1) * cells],
                        &plane.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
                assert!(got[end..].iter().all(|v| *v == 0xdead_beef));
                assert_eq!(r.buffer_pool().stats(), stats);
                for cell in 0..cells {
                    let start = starts[cell];
                    let end = starts.get(cell + 1).copied().unwrap_or(points as u32);
                    if end <= start {
                        for c in 0..3 + attributes {
                            let word = got[c * cells + cell];
                            assert_eq!(
                                word,
                                if mode == 1 && c < 3 { f.floats[c][0].to_bits() } else { 0 }
                            );
                        }
                    }
                }
                let short = staging(&r, &vec![0xdead_beef; end - 1]);
                let mut encoder = r.device().create_command_encoder(&Default::default());
                assert!(!try_record_wide_voxel_average(
                    &mut encoder,
                    &r,
                    [&f.xyz[0], &f.xyz[1], &f.xyz[2]],
                    &attrs,
                    &f.segments,
                    &short,
                    XyzMode::First
                )
                .unwrap());
                assert!(!try_record_wide_voxel_average(
                    &mut encoder,
                    &r,
                    [&f.xyz[0], &f.xyz[1], &f.xyz[2]],
                    &attrs[..4],
                    &f.segments,
                    &target,
                    XyzMode::First
                )
                .unwrap());
                r.queue().submit(Some(encoder.finish()));
                assert!(read_words(&r, &short).iter().all(|v| *v == 0xdead_beef));
                assert_eq!(read_words(&r, &target), got);
                assert_eq!(r.buffer_pool().stats(), stats);
                checked += cells * (3 + attributes);
            }
        }
    }
    eprintln!("VOXEL_WIDE_AVERAGE_RAW verified {checked} current-parent words, scalar tails, empty cells, nonfinite/cancellation math, guarded neighbors and rejected no-op recordings");
}

#[test]
fn wide_voxel_average_complete_calls_match_current_parents_cpu_and_owned_capacities() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for cells in [0, 1, 3, 17, 255, 256, 257, 65_537] {
        for (attributes, bytes) in
            [(0, 0), (1, 3), (4, 0), (4, 3), (5, 0), (7, 3), (8, 1), (9, 5), (11, 3), (17, 1)]
        {
            for occupancy in [1, 4] {
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
                for mode in [1, 2] {
                    let live = complete_call(&r, &f, mode, false);
                    let old = complete_call(&r, &f, mode, true);
                    assert_output(&live, &old);
                    assert_output(&live, &expected(&f, mode));
                    assert_capacity(&live, cells, attributes, bytes, false);
                    assert_capacity(&old, cells, attributes, bytes, false);
                    checked += cells * (3 + attributes + bytes) * 2;
                }
            }
        }
    }
    for (attributes, bytes, mode) in [(5, 3, 1), (7, 0, 2), (9, 5, 1), (17, 3, 2)] {
        let f = fixture(&r, 1_048_577, attributes, bytes, 4);
        let live = complete_call(&r, &f, mode, false);
        let old = complete_call(&r, &f, mode, true);
        let cpu = expected(&f, mode);
        assert_output(&live, &old);
        assert_output(&live, &cpu);
        assert_capacity(&live, f.cells, attributes, bytes, false);
        drop(f);
        r.clear_buffer_pool();
        assert_output(&live, &cpu);
        checked += 1_048_577 * (3 + attributes + bytes) * 2;
    }
    eprintln!("VOXEL_WIDE_AVERAGE_PUBLIC verified {checked} complete current-parent/CPU elements, fused controls, byte tails, sorted segments and million-cell ownership");
}

#[test]
fn wide_voxel_average_keeps_original_empty_and_float_before_byte_error_order() {
    let r = WgpuRuntime::new_headless().unwrap();
    let f = fixture(&r, 17, 9, 3, 4);
    let attrs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
    let bytes: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
    for mode in [1, 2] {
        for (index, found) in [(0, 67), (4, 66), (8, 65)] {
            let mut bad = attrs.clone();
            bad[index] = &f.floats[3 + index][..found];
            let mut bad_bytes = bytes.clone();
            bad_bytes[0] = &f.bytes[0][..64];
            let stats = r.buffer_pool().stats();
            let live = invoke_current_parent(&r, &f, &bad, &bad_bytes, mode, false).unwrap_err();
            assert_eq!(
                live,
                invoke_current_parent(&r, &f, &bad, &bad_bytes, mode, true).unwrap_err()
            );
            assert_eq!(live, SpatialError::BufferLengthMismatch { expected: 68, found });
            assert_eq!(
                invoke_current_parent(&r, &f, &attrs, &bad_bytes, mode, false).unwrap_err(),
                SpatialError::BufferLengthMismatch { expected: 68, found: 64 }
            );
            assert_eq!(r.buffer_pool().stats(), stats);
        }
        let empty = fixture(&r, 0, 9, 3, 1);
        let stats = r.buffer_pool().stats();
        let ignored_attrs = vec![&[1.0][..]; 9];
        let ignored_bytes = [&[1_u8][..]];
        let live =
            invoke_current_parent(&r, &empty, &ignored_attrs, &ignored_bytes, mode, false).unwrap();
        assert_output(
            &live,
            &invoke_current_parent(&r, &empty, &ignored_attrs, &ignored_bytes, mode, true).unwrap(),
        );
        assert_capacity(&live, 0, 9, 1, false);
        assert_eq!(r.buffer_pool().stats(), stats);
    }
}

#[test]
fn wide_voxel_average_cold_concurrent_changed_inputs_keep_owned_results() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    let gate = std::sync::Arc::new(std::sync::Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|worker| {
            let r = r.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                let mode = 1 + worker % 2;
                let attributes = [5, 7, 9, 17][worker];
                let mut f = fixture(&r, 16_385, attributes, 3, 4);
                gate.wait();
                let live = complete_call(&r, &f, mode, false);
                let original = expected(&f, mode);
                assert_output(&live, &original);
                for axis in 0..3 {
                    f.floats[axis].fill([7.0, 11.0, -19.0][axis]);
                    r.queue().write_buffer(&f.xyz[axis], 0, bytemuck::cast_slice(&f.floats[axis]));
                }
                for source in &mut f.floats[3..] {
                    source.fill(123.0);
                }
                for source in &mut f.bytes {
                    source.fill(231);
                }
                let changed = complete_call(&r, &f, mode, false);
                let cpu = expected(&f, mode);
                assert_output(&changed, &cpu);
                assert_capacity(&changed, f.cells, attributes, 3, false);
                drop(f);
                r.clear_buffer_pool();
                assert_output(&live, &original);
                assert_output(&changed, &cpu);
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[path = "voxel_wide_average_benchmark.rs"]
mod benchmark;
