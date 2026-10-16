use super::*;
use crate::readback_planes::{copy_owned_xyz_planes, copy_xyz_with_workers, plan_plane_copy};
#[path = "voxel_xyz_parallel_reference.rs"]
mod baseline;

fn invoke(r: &WgpuRuntime, f: &Fixture, reduce: bool, old: bool) -> SpatialResult<Xyz> {
    let function = match (reduce, old) {
        (false, false) => gather_voxel_first_xyz_gpu_buffers,
        (false, true) => baseline::baseline_gather_voxel_first_xyz_gpu_buffers,
        (true, false) => reduce_voxel_centroids_xyz_gpu_buffers,
        (true, true) => baseline::baseline_reduce_voxel_centroids_xyz_gpu_buffers,
    };
    function(r, &f.buffers[0], &f.buffers[1], &f.buffers[2], &f.segments)
}

fn assert_capacity(out: &Xyz, cells: usize) {
    for plane in [&out.0, &out.1, &out.2] {
        assert_eq!(plane.len(), cells);
        assert_eq!(plane.capacity(), cells);
    }
}

fn assert_raw(out: &Xyz, input: &[u32], cells: usize) {
    assert_capacity(out, cells);
    for (axis, plane) in axes(out).into_iter().enumerate() {
        assert_eq!(
            bytemuck::cast_slice::<f32, u32>(plane),
            &input[axis * cells..(axis + 1) * cells]
        );
    }
}

#[test]
fn parallel_xyz_host_copies_keep_raw_bits_capacities_caller_worker_bounds_and_independence() {
    let mut checked = 0;
    for cells in [0_usize, 17, 1_048_575, 1_048_576, 1_048_577, 2_097_153] {
        let input = words(cells * 3 + 16);
        let values: &[f32] = bytemuck::cast_slice(&input);
        let old = (
            values[..cells].to_vec(),
            values[cells..cells * 2].to_vec(),
            values[cells * 2..cells * 3].to_vec(),
        );
        let mut outputs = vec![copy_owned_xyz_planes(values, cells)];
        for workers in [2, 3] {
            outputs.push(copy_xyz_with_workers(values, cells, workers));
        }
        for mut out in outputs {
            assert_bits(&out, &old);
            assert_raw(&out, &input, cells);
            if cells > 0 {
                out.0[0] = f32::from_bits(0xdead_beef);
                assert_eq!(input[0], 0);
                assert_eq!(out.1[0].to_bits(), input[cells]);
                assert_eq!(old.0[0].to_bits(), 0);
            }
            checked += cells * 3 * 2;
        }
        for available in [0, 1, 2, 3, 4, usize::MAX] {
            let plan = plan_plane_copy(cells, 3, available);
            assert_eq!(plan.is_some(), cells >= 1_048_576 && available >= 2);
            if let Some(plan) = plan {
                assert_eq!(plan.workers, available.min(3));
                assert!((2..=3).contains(&plan.workers));
            }
        }
    }
    for workers in [0, 1, 4, usize::MAX] {
        assert!(std::panic::catch_unwind(|| copy_xyz_with_workers(&[], 0, workers)).is_err());
    }
    assert!(std::panic::catch_unwind(|| copy_xyz_with_workers(&[0.0], 1_048_576, 2)).is_err());
    assert!(std::panic::catch_unwind(|| copy_owned_xyz_planes(&[], usize::MAX)).is_err());
    eprintln!("VOXEL_XYZ_PARALLEL_HOST verified {checked} serial/automatic/two/three-worker raw words, exact capacities, caller-inclusive caps, complete source bounds and independent outputs");
}

#[test]
fn parallel_xyz_mapped_readback_keeps_serial_cpu_bits_panic_remap_and_changed_staging_ownership() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for cells in [0, 17, 1_048_575, 1_048_576, 1_048_577] {
        let mut input = words(cells * 3 + 16);
        let buffer = staging(&r, &input);
        let live = crate::readback::read_staging_xyz(r.device(), &buffer, cells).unwrap();
        let old = baseline::read_staging_xyz(r.device(), &buffer, cells).unwrap();
        assert_bits(&live, &old);
        assert_raw(&live, &input, cells);
        if cells == 1_048_576 {
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::readback::read_staging_xyz(r.device(), &buffer, cells + 7).unwrap()
            }));
            assert!(panic.is_err());
            assert_bits(
                &crate::readback::read_staging_xyz(r.device(), &buffer, cells).unwrap(),
                &old,
            );
        }
        input.fill(123.0_f32.to_bits());
        r.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(&input));
        r.queue().submit([]);
        let changed = crate::readback::read_staging_xyz(r.device(), &buffer, cells).unwrap();
        assert_capacity(&changed, cells);
        assert!(axes(&changed).into_iter().flatten().all(|v| v.to_bits() == 123.0_f32.to_bits()));
        drop(buffer);
        assert_bits(&live, &old);
        checked += cells * 3 * 2;
    }
    eprintln!("VOXEL_XYZ_PARALLEL_RAW verified {checked} mapped serial-parent/CPU words, cutoff controls, panic/remap, staging mutation and owned lifetimes");
}

#[test]
fn parallel_xyz_complete_public_calls_match_full_current_parents_cpu_and_empty_pool_behavior() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for (cells, occupancy) in [
        (0, 1),
        (17, 4),
        (257, 4),
        (65_537, 4),
        (1_048_575, 1),
        (1_048_576, 4),
        (1_048_577, 1),
        (2_097_153, 1),
    ] {
        let mut f = fixture(&r, cells, occupancy);
        if cells == 257 {
            f.segments = crate::kernels::voxel_sort::build_voxel_segments_gpu_from_keys(
                &r,
                &(0..cells * occupancy).map(|i| ((i % cells) as i64, 0, 0)).collect::<Vec<_>>(),
            )
            .unwrap();
            assert_eq!(f.segments.construction_readback_bytes(), 4);
        }
        let pool = r.buffer_pool().stats();
        let mut retained = Vec::new();
        for reduce in [false, true] {
            let live = invoke(&r, &f, reduce, false).unwrap();
            let old = invoke(&r, &f, reduce, true).unwrap();
            let cpu = if reduce { &f.mean } else { &f.first };
            assert_bits(&live, &old);
            assert_bits(&live, cpu);
            assert_capacity(&live, cells);
            assert_capacity(&old, cells);
            retained.push((live, cpu.clone()));
            checked += cells * 3 * 2;
        }
        assert_eq!(r.buffer_pool().stats(), pool);
        drop(f);
        r.clear_buffer_pool();
        for (live, cpu) in retained {
            assert_bits(&live, &cpu);
        }
    }
    eprintln!("VOXEL_XYZ_PARALLEL_PUBLIC verified {checked} complete two-mode current-parent/CPU words, cutoff/large/sorted cases, exact capacities, empty returns and owned destruction");
}

#[test]
fn parallel_xyz_packed_and_split_gpu_layouts_keep_sparse_nonfinite_first_bits_and_tail_guards() {
    use crate::kernels::voxel_xyz_packed::{plan_packed_xyz, record_packed_xyz, XyzMode};
    let r = WgpuRuntime::new_headless().unwrap();
    let cells = 1_048_577;
    let input = words(cells * 3);
    let source: &[f32] = bytemuck::cast_slice(&input);
    let buffers: [wgpu::Buffer; 3] = std::array::from_fn(|axis| {
        r.upload_f32_storage("parallel-xyz-raw-input", &source[axis * cells..(axis + 1) * cells])
            .unwrap()
    });
    let indices: Vec<_> = (0..cells).rev().map(|i| i as u32).collect();
    let starts: Vec<_> = (0..cells).map(|i| (i / 2) as u32).collect();
    let upload = |label, data: &[u32]| r.upload_u32_storage(label, data).unwrap();
    let segments = GpuVoxelSegments::new(
        cells as u32,
        cells as u32,
        upload("parallel-xyz-keys", &[0]),
        upload("parallel-xyz-indices", &indices),
        upload("parallel-xyz-starts", &starts),
        0,
    );
    let expected: [Vec<f32>; 3] = std::array::from_fn(|axis| {
        (0..cells)
            .map(|cell| {
                let start = starts[cell];
                let end = starts.get(cell + 1).copied().unwrap_or(cells as u32);
                source
                    [axis * cells + if end <= start { 0 } else { indices[start as usize] as usize }]
            })
            .collect()
    });
    let [x, y, z] = expected;
    let expected = (x, y, z);
    let mut checked = 0;
    for packed in [false, true] {
        if packed
            && plan_packed_xyz(&r.device().limits(), cells as u32, isize::MAX as u64).is_none()
        {
            continue;
        }
        let target = staging(&r, &vec![0xdead_beef; cells * 3 + 16]);
        let mut encoder = r.device().create_command_encoder(&Default::default());
        let output = |len| {
            r.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("parallel-xyz-layout-output"),
                size: len,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        if packed {
            let out = output((cells * 12) as u64);
            record_packed_xyz(
                &mut encoder,
                &r,
                [&buffers[0], &buffers[1], &buffers[2]],
                &segments,
                XyzMode::First,
                &out,
            );
            encoder.copy_buffer_to_buffer(&out, 0, &target, 0, (cells * 12) as u64);
        } else {
            let out: [wgpu::Buffer; 3] = std::array::from_fn(|_| output((cells * 4) as u64));
            record_voxel_gather_xyz_pass(
                &mut encoder,
                &r,
                &buffers[0],
                &buffers[1],
                &buffers[2],
                &segments,
                &out[0],
                &out[1],
                &out[2],
            )
            .unwrap();
            for (axis, out) in out.iter().enumerate() {
                encoder.copy_buffer_to_buffer(
                    out,
                    0,
                    &target,
                    (axis * cells * 4) as u64,
                    (cells * 4) as u64,
                );
            }
        }
        r.queue().submit(Some(encoder.finish()));
        let live = crate::readback::read_staging_xyz(r.device(), &target, cells).unwrap();
        let old = baseline::read_staging_xyz(r.device(), &target, cells).unwrap();
        assert_bits(&live, &old);
        assert_bits(&live, &expected);
        assert_capacity(&live, cells);
        let raw = crate::readback::read_staging_f32(r.device(), &target, cells * 3 + 16).unwrap();
        assert!(raw[cells * 3..].iter().all(|v| v.to_bits() == 0xdead_beef));
        checked += cells * 3 * 2;
    }
    eprintln!("VOXEL_XYZ_PARALLEL_LAYOUT verified {checked} original packed/split GPU serial-parent/CPU words, repeated empty starts, reversed indices, raw nonfinite first values and untouched tail guards");
}

#[test]
fn parallel_xyz_concurrent_public_calls_keep_prior_outputs_after_resident_mutation_and_drop() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    let gate = std::sync::Arc::new(std::sync::Barrier::new(2));
    let handles: Vec<_> = (0..2)
        .map(|mode| {
            let r = r.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                let cells = 1_048_577;
                let f = fixture(&r, cells, 1);
                gate.wait();
                let live = invoke(&r, &f, mode == 1, false).unwrap();
                let old = invoke(&r, &f, mode == 1, true).unwrap();
                assert_bits(&live, &old);
                assert_capacity(&live, cells);
                assert_bits(&live, if mode == 1 { &f.mean } else { &f.first });
                let changed_input = vec![123.0_f32; cells];
                for buffer in &f.buffers {
                    r.queue().write_buffer(buffer, 0, bytemuck::cast_slice(&changed_input));
                }
                let changed = invoke(&r, &f, mode == 1, false).unwrap();
                assert_capacity(&changed, cells);
                assert!(axes(&changed)
                    .into_iter()
                    .flatten()
                    .all(|v| v.to_bits() == 123.0_f32.to_bits()));
                drop(f);
                assert_bits(&live, &old);
                (live, old)
            })
        })
        .collect();
    let retained: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    r.clear_buffer_pool();
    for (live, old) in retained {
        assert_bits(&live, &old);
    }
}

#[path = "voxel_xyz_parallel_benchmark.rs"]
mod benchmark;
