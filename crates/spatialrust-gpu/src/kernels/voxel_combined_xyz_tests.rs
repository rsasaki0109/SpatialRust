use super::*;
use crate::kernels::voxel_xyz_packed::{
    plan_packed_xyz, plan_packed_xyz_to_staging, try_record_packed_xyz_to_staging, XyzMode,
};
use wgpu::util::DeviceExt;

#[path = "voxel_combined_xyz_reference.rs"]
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
        (0, false) => gather_voxel_first_xyz_and_multi_gpu,
        (0, true) => baseline::baseline_gather_voxel_first_xyz_and_multi_gpu,
        (1, false) => gather_voxel_first_xyz_and_average_multi_gpu,
        (1, true) => baseline::baseline_gather_voxel_first_xyz_and_average_multi_gpu,
        (2, false) => reduce_voxel_centroids_xyz_and_average_multi_gpu,
        (2, true) => baseline::baseline_reduce_voxel_centroids_xyz_and_average_multi_gpu,
        (3, false) => reduce_voxel_centroids_xyz_and_gather_first_multi_gpu,
        (3, true) => baseline::baseline_reduce_voxel_centroids_xyz_and_gather_first_multi_gpu,
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

#[test]
fn combined_packed_xyz_planner_checks_existing_limits_and_complete_aligned_target() {
    let limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        max_bindings_per_bind_group: 9,
        max_compute_workgroup_size_x: 256,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroups_per_dimension: 2,
        max_storage_buffer_binding_size: 257 * 12,
        max_buffer_size: 11_060,
        ..wgpu::Limits::downlevel_defaults()
    };
    let bytes = 257 * 12;
    let mut checked = 0;
    for (offset, target) in [(0, bytes), (20, bytes + 20), (20, 11_060)] {
        assert_eq!(
            plan_packed_xyz_to_staging(&limits, 257, offset, target, target),
            plan_packed_xyz(&limits, 257, target)
        );
        checked += 1;
        assert!(plan_packed_xyz_to_staging(&limits, 257, offset, target, target - 1).is_none());
        checked += 1;
    }
    for (offset, target) in [
        (1, 11_060),
        (2, 11_060),
        (3, 11_060),
        (20, bytes + 19),
        (11_060, 11_060),
        (u64::MAX - 3, u64::MAX),
        (0, 11_061),
    ] {
        assert!(plan_packed_xyz_to_staging(&limits, 257, offset, target, u64::MAX).is_none());
        checked += 1;
    }
    for field in 0..13 {
        let mut l = limits.clone();
        match field {
            0 => l.max_storage_buffers_per_shader_stage = 7,
            1 => l.max_bindings_per_bind_group = 8,
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
            _ => l.max_buffer_size = 11_059,
        }
        assert!(plan_packed_xyz_to_staging(&l, 257, 20, 11_060, u64::MAX).is_none());
        checked += 1;
    }
    let mut small = limits.clone();
    small.max_storage_buffer_binding_size = 257 * 4;
    assert!(plan_packed_xyz_to_staging(&small, 257, 0, bytes, u64::MAX).is_none());
    checked += 1;
    let wide = wgpu::Limits {
        max_buffer_size: u64::MAX,
        max_storage_buffer_binding_size: u32::MAX,
        max_compute_workgroups_per_dimension: u32::MAX,
        ..limits
    };
    let cells = (i32::MAX as u64 / 12) as u32;
    assert!(
        plan_packed_xyz_to_staging(&wide, cells, 0, cells as u64 * 12, i32::MAX as u64).is_some()
    );
    assert!(
        plan_packed_xyz_to_staging(&wide, cells, 0, i32::MAX as u64 + 1, i32::MAX as u64).is_none()
    );
    assert!(plan_packed_xyz_to_staging(
        &wide,
        cells + 1,
        0,
        (cells as u64 + 1) * 12,
        i32::MAX as u64
    )
    .is_none());
    assert!(plan_packed_xyz_to_staging(&wide, 0, 0, 64, u64::MAX).is_none());
    assert!(plan_packed_xyz_to_staging(&wide, u32::MAX, 0, u64::MAX, u64::MAX).is_none());
    checked += 5;
    eprintln!("VOXEL_COMBINED_XYZ_PLAN verified {checked} exact/one-short target, offset, host and unchanged XYZ capability bounds");
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
fn combined_packed_xyz_recorder_keeps_raw_math_prefix_suffix_and_neighbor_regions_exact() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for cells in [1, 3, 257, 65_537] {
        let mut points = 1;
        let starts: Vec<_> = (0..cells)
            .map(|i| {
                let start = points;
                points += [0, 1, 3, 5, 7][i % 5];
                start as u32
            })
            .collect();
        let indices: Vec<_> = (0..points).rev().map(|i| i as u32).collect();
        let upload = |label, data: &[u32]| r.upload_u32_storage(label, data).unwrap();
        let segments = GpuVoxelSegments::new(
            cells as u32,
            points as u32,
            upload("combined-xyz-keys", &[0]),
            upload("combined-xyz-indices", &indices),
            upload("combined-xyz-starts", &starts),
            0,
        );
        let special = words(12);
        let source: [Vec<f32>; 3] = std::array::from_fn(|axis| {
            (0..points)
                .map(|i| {
                    if i == 0 {
                        [1.25, -0.0, f32::from_bits(0x7fc0_1234)][axis]
                    } else {
                        match axis {
                            0 => ((i * 17) % 4096) as f32 * 0.25 - 511.5,
                            1 => f32::from_bits(special[i % 12]),
                            _ => [16_777_216.0, 1.0, -16_777_216.0, 3.0][i % 4],
                        }
                    }
                })
                .collect()
        });
        let xyz: [wgpu::Buffer; 3] = std::array::from_fn(|axis| {
            r.upload_f32_storage("combined-xyz-source", &source[axis]).unwrap()
        });
        for mode in [XyzMode::First, XyzMode::Centroid] {
            let old: [wgpu::Buffer; 3] = std::array::from_fn(|_| {
                r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("combined-xyz-old-output"),
                    contents: bytemuck::cast_slice(&vec![0xdead_beef_u32; cells + 8]),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
            });
            let region_end = 5 + cells * 3;
            let old_base = region_end + 8;
            let poison = vec![0xdead_beef_u32; old_base + 3 * (cells + 8) + 8];
            let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("combined-xyz-offset-guarded-staging"),
                size: (poison.len() * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            r.queue().write_buffer(&staging, 0, bytemuck::cast_slice(&poison));
            let stats = r.buffer_pool().stats();
            let mut encoder = r.device().create_command_encoder(&Default::default());
            assert!(try_record_packed_xyz_to_staging(
                &mut encoder,
                &r,
                [&xyz[0], &xyz[1], &xyz[2]],
                &segments,
                &staging,
                20,
                mode
            ));
            if mode == XyzMode::First {
                crate::kernels::voxel_gather::record_voxel_gather_xyz_pass(
                    &mut encoder,
                    &r,
                    &xyz[0],
                    &xyz[1],
                    &xyz[2],
                    &segments,
                    &old[0],
                    &old[1],
                    &old[2],
                )
                .unwrap();
            } else {
                crate::kernels::voxel_reduce::record_voxel_reduce_xyz_pass(
                    &mut encoder,
                    &r,
                    &xyz[0],
                    &xyz[1],
                    &xyz[2],
                    segments.point_indices_buffer(),
                    segments.cell_starts_buffer(),
                    cells as u32,
                    points as u32,
                    &old[0],
                    &old[1],
                    &old[2],
                )
                .unwrap();
            }
            for (axis, output) in old.iter().enumerate() {
                encoder.copy_buffer_to_buffer(
                    output,
                    0,
                    &staging,
                    ((old_base + axis * (cells + 8)) * 4) as u64,
                    output.size(),
                );
            }
            r.queue().submit(Some(encoder.finish()));
            let got = read_words(&r, &staging);
            assert_eq!(r.buffer_pool().stats(), stats);
            assert!(got[..5]
                .iter()
                .chain(&got[region_end..old_base])
                .chain(&got[poison.len() - 8..])
                .all(|v| *v == 0xdead_beef));
            for axis in 0..3 {
                let start = old_base + axis * (cells + 8);
                assert_eq!(
                    &got[5 + axis * cells..5 + (axis + 1) * cells],
                    &got[start..start + cells]
                );
                assert!(got[start + cells..start + cells + 8].iter().all(|v| *v == 0xdead_beef));
                for cell in 0..cells {
                    let first = starts[cell];
                    let end = starts.get(cell + 1).copied().unwrap_or(points as u32);
                    if mode == XyzMode::First {
                        let index = if end <= first { 0 } else { indices[first as usize] as usize };
                        assert_eq!(got[5 + axis * cells + cell], source[axis][index].to_bits());
                    } else if first == end {
                        assert_eq!(got[5 + axis * cells + cell], 0);
                    }
                }
            }
            let mut encoder = r.device().create_command_encoder(&Default::default());
            for offset in [1, 2, staging.size() - 4, u64::MAX - 3] {
                assert!(!try_record_packed_xyz_to_staging(
                    &mut encoder,
                    &r,
                    [&xyz[0], &xyz[1], &xyz[2]],
                    &segments,
                    &staging,
                    offset,
                    mode
                ));
            }
            r.queue().submit(Some(encoder.finish()));
            assert_eq!(read_words(&r, &staging), got);
            assert_eq!(r.buffer_pool().stats(), stats);
            checked += cells * 3 * 2;
        }
    }
    eprintln!("VOXEL_COMBINED_XYZ_RAW verified {checked} old GPU/packed words, raw first-point CPU values, unchanged adjacent regions and rejected no-op recordings");
}

#[test]
fn combined_packed_xyz_complete_calls_match_four_current_parents_cpu_and_owned_capacities() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for cells in [0, 1, 3, 17, 255, 256, 257, 65_537] {
        for (attributes, bytes) in
            [(0, 0), (0, 3), (1, 0), (4, 0), (4, 3), (5, 0), (7, 3), (9, 5), (17, 1)]
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
                for mode in 0..4 {
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
    for (attributes, bytes, mode) in [(0, 3, 0), (7, 3, 1), (7, 0, 2), (5, 3, 3)] {
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
    eprintln!("VOXEL_COMBINED_XYZ_PUBLIC verified {checked} complete four-mode parent/CPU elements; fused controls, byte tails, sorted segments, exact capacities and million-cell ownership");
}

#[test]
fn combined_packed_xyz_keeps_float_before_byte_errors_and_original_empty_order() {
    let r = WgpuRuntime::new_headless().unwrap();
    let f = fixture(&r, 17, 9, 3, 4);
    let attrs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
    let bytes: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
    for mode in 0..4 {
        for (index, found) in [(0, 67), (4, 66), (8, 65)] {
            let mut bad = attrs.clone();
            bad[index] = &f.floats[3 + index][..found];
            let mut bad_bytes = bytes.clone();
            bad_bytes[0] = &f.bytes[0][..64];
            let stats = r.buffer_pool().stats();
            let current = invoke_current_parent(&r, &f, &bad, &bad_bytes, mode, false).unwrap_err();
            let previous = invoke_current_parent(&r, &f, &bad, &bad_bytes, mode, true).unwrap_err();
            assert_eq!(current, previous);
            assert_eq!(current, SpatialError::BufferLengthMismatch { expected: 68, found });
            assert_eq!(
                invoke_current_parent(&r, &f, &attrs, &bad_bytes, mode, false).unwrap_err(),
                SpatialError::BufferLengthMismatch { expected: 68, found: 64 }
            );
            assert_eq!(r.buffer_pool().stats(), stats);
        }
        let empty = fixture(&r, 0, 9, 3, 1);
        let stats = r.buffer_pool().stats();
        let ignored_attrs = [&[1.0][..]];
        let ignored_bytes = [&[1_u8][..]];
        let current =
            invoke_current_parent(&r, &empty, &ignored_attrs, &ignored_bytes, mode, false).unwrap();
        let previous =
            invoke_current_parent(&r, &empty, &ignored_attrs, &ignored_bytes, mode, true).unwrap();
        assert_output(&current, &previous);
        assert_capacity(&current, 0, 1, 1, false);
        assert_eq!(r.buffer_pool().stats(), stats);
    }
}

#[test]
fn combined_packed_xyz_concurrent_changed_inputs_keep_old_results_after_pool_recycling() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    let gate = std::sync::Arc::new(std::sync::Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|mode| {
            let r = r.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                let mut f = fixture(&r, 16_385, 7, 3, 4);
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
                assert_capacity(&changed, f.cells, 7, 3, false);
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

#[path = "voxel_combined_xyz_benchmark.rs"]
mod benchmark;
