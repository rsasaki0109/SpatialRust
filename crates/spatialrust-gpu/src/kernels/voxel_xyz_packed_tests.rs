use super::*;
use crate::kernels::voxel_xyz_packed::{plan_packed_xyz, record_packed_xyz, XyzMode};
use wgpu::util::DeviceExt;

#[path = "voxel_xyz_packed_reference.rs"]
mod baseline;

fn invoke(r: &WgpuRuntime, f: &Fixture, mode: XyzMode, old: bool) -> SpatialResult<Xyz> {
    let [x, y, z] = &f.buffers;
    match (mode, old) {
        (XyzMode::First, false) => gather_voxel_first_xyz_gpu_buffers(r, x, y, z, &f.segments),
        (XyzMode::First, true) => {
            baseline::baseline_gather_voxel_first_xyz_gpu_buffers(r, x, y, z, &f.segments)
        }
        (XyzMode::Centroid, false) => {
            reduce_voxel_centroids_xyz_gpu_buffers(r, x, y, z, &f.segments)
        }
        (XyzMode::Centroid, true) => {
            baseline::baseline_reduce_voxel_centroids_xyz_gpu_buffers(r, x, y, z, &f.segments)
        }
    }
}

fn capacities(v: &Xyz) -> [usize; 3] {
    [v.0.capacity(), v.1.capacity(), v.2.capacity()]
}

#[test]
fn packed_xyz_planner_checks_complete_dispatch_binding_host_and_original_capabilities() {
    let limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        max_bindings_per_bind_group: 9,
        max_compute_workgroup_size_x: 256,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroups_per_dimension: 2,
        max_storage_buffer_binding_size: 257 * 12,
        max_buffer_size: 257 * 12,
        ..wgpu::Limits::downlevel_defaults()
    };
    let p = plan_packed_xyz(&limits, 257, 257 * 12).unwrap();
    assert_eq!(p.bytes, 257 * 12);
    let mut checked = 0;
    for field in 0..14 {
        let mut l = limits.clone();
        let mut host = 257 * 12;
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
            12 => l.max_buffer_size -= 1,
            _ => host -= 1,
        }
        assert!(plan_packed_xyz(&l, 257, host).is_none(), "boundary {field}");
        checked += 1;
    }
    let mut plane_only = limits.clone();
    plane_only.max_storage_buffer_binding_size = 257 * 4;
    assert!(plan_packed_xyz(&plane_only, 257, u64::MAX).is_none());
    assert!(
        plane_only.max_storage_buffer_binding_size >= 257 * 4
            && plane_only.max_buffer_size >= 257 * 12
    );
    let wide = wgpu::Limits {
        max_storage_buffer_binding_size: u32::MAX,
        max_buffer_size: u64::MAX,
        max_compute_workgroups_per_dimension: u32::MAX,
        ..limits
    };
    for cells in [1, 3, 17, 255, 256, 257, 65_537, 1_048_577] {
        let exact = u64::from(cells) * 12;
        assert_eq!(plan_packed_xyz(&wide, cells, exact).unwrap().bytes, exact);
        assert!(plan_packed_xyz(&wide, cells, exact - 1).is_none());
        checked += 2;
    }
    let cells = (i32::MAX as u64 / 12) as u32;
    assert!(plan_packed_xyz(&wide, cells, i32::MAX as u64).is_some());
    assert!(plan_packed_xyz(&wide, cells + 1, i32::MAX as u64).is_none());
    assert!(plan_packed_xyz(&wide, 0, u64::MAX).is_none());
    assert!(plan_packed_xyz(&wide, u32::MAX / 3 + 1, u64::MAX).is_none());
    assert!(plan_packed_xyz(&wide, u32::MAX, u64::MAX).is_none());
    eprintln!("VOXEL_XYZ_PACKED_PLAN verified {} exact/one-short device/host plans and unchanged separate-plane fallback envelope",checked+6);
}

#[test]
fn packed_xyz_complete_public_calls_match_current_parent_cpu_and_exact_owned_capacities() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for cells in [0, 1, 3, 17, 255, 256, 257, 65_537, 1_048_577] {
        for occupancy in [1, 3, 5, 7] {
            if cells > 65_537 && occupancy != 1 {
                continue;
            }
            let mut f = fixture(&r, cells, occupancy);
            if cells == 257 {
                f.segments = crate::kernels::voxel_sort::build_voxel_segments_gpu_from_keys(
                    &r,
                    &(0..cells * occupancy).map(|i| ((i % cells) as i64, 0, 0)).collect::<Vec<_>>(),
                )
                .unwrap();
                assert_eq!(f.segments.construction_readback_bytes(), 4);
            }
            for mode in [XyzMode::First, XyzMode::Centroid] {
                let expected = if mode == XyzMode::First { &f.first } else { &f.mean };
                let stats = r.buffer_pool().stats();
                let mut live = invoke(&r, &f, mode, false).unwrap();
                let old = invoke(&r, &f, mode, true).unwrap();
                assert_bits(&live, &old);
                assert_bits(&live, expected);
                assert_eq!(capacities(&live), [cells; 3]);
                assert_eq!(capacities(&old), [cells; 3]);
                assert_eq!(r.buffer_pool().stats(), stats);
                if cells > 0 {
                    live.0[0] = f32::from_bits(0xdead_beef);
                }
                let next = invoke(&r, &f, mode, false).unwrap();
                r.clear_buffer_pool();
                assert_bits(&next, expected);
                if cells > 0 {
                    assert_eq!(live.0[0].to_bits(), 0xdead_beef);
                }
                checked += cells * 9;
            }
            let first = invoke(&r, &f, XyzMode::First, false).unwrap();
            let mean = invoke(&r, &f, XyzMode::Centroid, false).unwrap();
            let expected = (f.first.clone(), f.mean.clone());
            drop(f);
            r.clear_buffer_pool();
            assert_bits(&first, &expected.0);
            assert_bits(&mean, &expected.1);
        }
    }
    eprintln!("VOXEL_XYZ_PACKED_PUBLIC verified {checked} complete current-parent/CPU/repeated words, exact capacities and owned lifetimes");
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

fn sparse_fixture(r: &WgpuRuntime, cells: usize, descending: bool) -> Fixture {
    let points = cells * 3;
    let source: [Vec<f32>; 3] = std::array::from_fn(|axis| {
        let mut values: Vec<_> =
            words(points + axis * 7).into_iter().skip(axis * 7).map(f32::from_bits).collect();
        values[0] = f32::from_bits([0x3fa0_0000, 0x8000_0000, 0x7fc0_1234][axis]);
        values
    });
    let starts: Vec<u32> = (0..cells)
        .map(|i| {
            if descending {
                match i % 5 {
                    0 | 1 => 0,
                    2 => (points - 1) as u32,
                    3 => (points / 2) as u32,
                    _ => (i * 3) as u32,
                }
            } else {
                (i / 2 * 6) as u32
            }
        })
        .collect();
    let indices: Vec<_> = (0..points).rev().map(|i| i as u32).collect();
    let first: [Vec<f32>; 3] = std::array::from_fn(|axis| {
        (0..cells)
            .map(|cell| {
                let start = starts[cell];
                let end = starts.get(cell + 1).copied().unwrap_or(points as u32);
                source[axis][if end <= start { 0 } else { indices[start as usize] as usize }]
            })
            .collect()
    });
    let buffers = std::array::from_fn(|axis| {
        r.upload_f32_storage("xyz-packed-raw-source", &source[axis]).unwrap()
    });
    let upload = |name, data: &[u32]| r.upload_u32_storage(name, data).unwrap();
    let segments = GpuVoxelSegments::new(
        cells as u32,
        points as u32,
        upload("xyz-packed-raw-keys", &[0]),
        upload("xyz-packed-raw-indices", &indices),
        upload("xyz-packed-raw-starts", &starts),
        0,
    );
    let [x, y, z] = first;
    Fixture { buffers, segments, first: (x, y, z), mean: (Vec::new(), Vec::new(), Vec::new()) }
}

#[test]
fn packed_xyz_raw_outputs_keep_nonfinite_bits_zero_cells_and_poisoned_output_staging_guards() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for cells in [1, 3, 257, 65_537] {
        let f = sparse_fixture(&r, cells, false);
        let p = plan_packed_xyz(&r.device().limits(), cells as u32, isize::MAX as u64).unwrap();
        assert_eq!(p.bytes, cells as u64 * 12);
        for mode in [XyzMode::First, XyzMode::Centroid] {
            let expected = invoke(&r, &f, mode, true).unwrap();
            let poison = vec![0xdead_beef_u32; cells * 3 + 8];
            let output = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("xyz-packed-poisoned-output"),
                contents: bytemuck::cast_slice(&poison),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("xyz-packed-staging-guards"),
                size: ((poison.len() + 13) * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            r.queue().write_buffer(
                &staging,
                0,
                bytemuck::cast_slice(&vec![0xdead_beef_u32; poison.len() + 13]),
            );
            let stats = r.buffer_pool().stats();
            let mut encoder = r.device().create_command_encoder(&Default::default());
            record_packed_xyz(
                &mut encoder,
                &r,
                [&f.buffers[0], &f.buffers[1], &f.buffers[2]],
                &f.segments,
                mode,
                &output,
            );
            encoder.copy_buffer_to_buffer(&output, 0, &staging, 20, (poison.len() * 4) as u64);
            r.queue().submit(Some(encoder.finish()));
            let raw = read_words(&r, &staging);
            assert!(raw[..5].iter().all(|v| *v == 0xdead_beef));
            assert!(raw[5 + cells * 3..].iter().all(|v| *v == 0xdead_beef));
            for (actual, want) in
                raw[5..5 + cells * 3].iter().zip(axes(&expected).into_iter().flatten())
            {
                assert_eq!(*actual, want.to_bits(), "{mode:?}");
            }
            let live = invoke(&r, &f, mode, false).unwrap();
            assert_bits(&live, &expected);
            assert_eq!(capacities(&live), [cells; 3]);
            assert_eq!(r.buffer_pool().stats(), stats);
            if mode == XyzMode::First {
                assert_bits(&live, &f.first);
            } else {
                for cell in (0..cells.saturating_sub(1)).step_by(2) {
                    for axis in axes(&live) {
                        assert_eq!(axis[cell].to_bits(), 0);
                    }
                }
            }
            checked += cells * 6;
        }
    }
    eprintln!("VOXEL_XYZ_PACKED_RAW verified {checked} frozen/public/raw words including NaNs/zeros, empty cells and output/staging guards");
}

#[test]
fn packed_xyz_gather_keeps_descending_point_zero_rules_and_empty_calls_keep_pools_unchanged() {
    let r = WgpuRuntime::new_headless().unwrap();
    let f = fixture(&r, 0, 1);
    let stats = r.buffer_pool().stats();
    for mode in [XyzMode::First, XyzMode::Centroid] {
        assert_bits(&invoke(&r, &f, mode, false).unwrap(), &(Vec::new(), Vec::new(), Vec::new()));
    }
    assert_eq!(r.buffer_pool().stats(), stats);
    let f = sparse_fixture(&r, 1025, true);
    let live = invoke(&r, &f, XyzMode::First, false).unwrap();
    let old = invoke(&r, &f, XyzMode::First, true).unwrap();
    assert_bits(&live, &old);
    assert_bits(&live, &f.first);
    let expected = f.first.clone();
    drop(f);
    r.clear_buffer_pool();
    assert_bits(&live, &expected);
}

#[test]
fn packed_xyz_cold_concurrent_caches_and_changed_resident_inputs_keep_old_outputs_owned() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    let gate = std::sync::Arc::new(std::sync::Barrier::new(4));
    let handles: Vec<_> =
        [(XyzMode::First, 3), (XyzMode::Centroid, 3), (XyzMode::First, 5), (XyzMode::Centroid, 5)]
            .into_iter()
            .map(|(mode, occupancy)| {
                let r = r.clone();
                let gate = gate.clone();
                std::thread::spawn(move || {
                    let f = fixture(&r, 16_385, occupancy);
                    gate.wait();
                    let old = invoke(&r, &f, mode, false).unwrap();
                    let expected =
                        if mode == XyzMode::First { f.first.clone() } else { f.mean.clone() };
                    assert_bits(&old, &expected);
                    for source in &f.buffers {
                        r.queue().write_buffer(
                            source,
                            0,
                            bytemuck::cast_slice(&vec![0x8000_0000_u32; 16_385 * occupancy]),
                        );
                    }
                    let changed = invoke(&r, &f, mode, false).unwrap();
                    let zero = if mode == XyzMode::First { 0x8000_0000 } else { 0 };
                    assert!(axes(&changed).into_iter().flatten().all(|v| v.to_bits() == zero));
                    let pointer = if mode == XyzMode::First {
                        r.pipelines().voxel_gather.packed_attributes_pipeline(r.device())
                            as *const _ as usize
                    } else {
                        r.pipelines().voxel_reduce.packed_xyz_pipeline(r.device()) as *const _
                            as usize
                    };
                    drop(f);
                    r.clear_buffer_pool();
                    assert_bits(&old, &expected);
                    assert!(axes(&changed).into_iter().flatten().all(|v| v.to_bits() == zero));
                    (mode, pointer)
                })
            })
            .collect();
    let pointers: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(pointers[0], pointers[2]);
    assert_eq!(pointers[1], pointers[3]);
    assert_ne!(pointers[0].1, pointers[1].1);
}

#[path = "voxel_xyz_packed_benchmark.rs"]
mod benchmark;
