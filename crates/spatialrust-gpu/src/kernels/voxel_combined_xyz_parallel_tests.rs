use super::*;
#[path = "voxel_combined_xyz_parallel_reference.rs"]
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

fn assert_owned(out: &Output, cells: usize, attrs: usize, bytes: usize) {
    assert_capacity(out, cells, attrs, bytes, false);
    for plane in floats(out) {
        assert_eq!(plane.len(), cells);
        assert_eq!(plane.capacity(), cells);
    }
    for plane in &out.4 {
        assert_eq!(plane.len(), cells);
        assert_eq!(plane.capacity(), cells);
    }
}

#[test]
fn combined_parallel_xyz_mapped_planes_keep_serial_cpu_bits_byte_tails_and_panic_remap() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for (cells, attrs, bytes) in [
        (0_usize, 0, 0),
        (17, 1, 3),
        (1_048_575, 4, 1),
        (1_048_576, 0, 0),
        (1_048_576, 1, 3),
        (1_048_576, 4, 0),
        (1_048_577, 7, 3),
        (1_048_577, 0, 5),
        (2_097_153, 0, 1),
    ] {
        let float_words = cells * (3 + attrs);
        let byte_words = cells.div_ceil(4) * bytes;
        let input = words(float_words + byte_words + 16);
        let buffer = staging(&r, &input);
        let live = crate::readback::read_staging_xyz_and_packed_attributes(
            r.device(),
            &buffer,
            cells,
            attrs,
            bytes,
        )
        .unwrap();
        let old = baseline::baseline_read_staging_xyz_and_packed_attributes(
            r.device(),
            &buffer,
            cells,
            attrs,
            bytes,
        )
        .unwrap();
        assert_output(&live, &old);
        assert_owned(&live, cells, attrs, bytes);
        for (c, plane) in floats(&live).enumerate() {
            assert_eq!(bytemuck::cast_slice::<f32, u32>(plane), &input[c * cells..(c + 1) * cells]);
        }
        for (c, plane) in live.4.iter().enumerate() {
            for (cell, &value) in plane.iter().enumerate() {
                assert_eq!(
                    value,
                    (input[float_words + c * cells.div_ceil(4) + cell / 4] >> ((cell % 4) * 8))
                        as u8
                );
            }
        }
        if cells == 1_048_576 && attrs == 1 {
            for (bad_attrs, bad_bytes) in [(attrs + 20, bytes), (attrs, bytes + 20)] {
                let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::readback::read_staging_xyz_and_packed_attributes(
                        r.device(),
                        &buffer,
                        cells,
                        bad_attrs,
                        bad_bytes,
                    )
                    .unwrap()
                }));
                assert!(panic.is_err());
                let next = crate::readback::read_staging_xyz_and_packed_attributes(
                    r.device(),
                    &buffer,
                    cells,
                    attrs,
                    bytes,
                )
                .unwrap();
                assert_output(&next, &old);
            }
        }
        drop(buffer);
        assert_output(&live, &old);
        checked += cells * (3 + attrs + bytes) * 2;
    }
    eprintln!("VOXEL_COMBINED_XYZ_PARALLEL_RAW verified {checked} mapped serial-XYZ-parent/CPU elements, cutoff/zero/single/many attribute controls, raw nonfinite bits, byte tails, panic/remap and owned destruction");
}

#[test]
fn combined_parallel_xyz_all_four_public_calls_match_complete_current_parents_cpu_and_capacities() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for (cells, attrs, bytes, occupancy) in [
        (0, 7, 3, 4),
        (16, 0, 0, 4),
        (257, 7, 3, 4),
        (65_537, 1, 1, 4),
        (1_048_575, 4, 3, 1),
        (1_048_576, 0, 0, 1),
        (1_048_576, 1, 3, 4),
        (1_048_577, 7, 3, 1),
        (1_048_577, 4, 5, 1),
        (2_097_153, 0, 1, 1),
    ] {
        let mut f = fixture(&r, cells, attrs, bytes, occupancy);
        if cells == 257 {
            f.segments = crate::kernels::voxel_sort::build_voxel_segments_gpu_from_keys(
                &r,
                &(0..cells * occupancy).map(|i| ((i % cells) as i64, 0, 0)).collect::<Vec<_>>(),
            )
            .unwrap();
            assert_eq!(f.segments.construction_readback_bytes(), 4);
        }
        let mut retained = Vec::new();
        for mode in 0..4 {
            let live = complete_call(&r, &f, mode, false);
            let old = complete_call(&r, &f, mode, true);
            let cpu = expected(&f, mode);
            assert_output(&live, &old);
            assert_output(&live, &cpu);
            assert_owned(&live, cells, attrs, bytes);
            assert_owned(&old, cells, attrs, bytes);
            retained.push((live, cpu));
            checked += cells * (3 + attrs + bytes) * 2;
        }
        drop(f);
        r.clear_buffer_pool();
        for (live, cpu) in retained {
            assert_output(&live, &cpu);
        }
    }
    eprintln!("VOXEL_COMBINED_XYZ_PARALLEL_PUBLIC verified {checked} complete four-mode current-parent/CPU elements, serial/cutoff/large/sorted cases, zero/single/wide float and byte paths, exact capacities and owned destruction");
}

#[test]
fn combined_parallel_xyz_public_validation_preserves_float_then_byte_order_and_empty_returns() {
    let r = WgpuRuntime::new_headless().unwrap();
    let f = fixture(&r, 17, 7, 3, 4);
    let attrs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
    let bytes: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
    let empty = fixture(&r, 0, 7, 3, 1);
    for mode in 0..4 {
        for (index, found) in [(0, 67), (3, 66), (6, 65)] {
            let mut bad = attrs.clone();
            bad[index] = &f.floats[index + 3][..found];
            if index < 6 {
                bad[6] = &f.floats[9][..1];
            }
            let mut bad_bytes = bytes.clone();
            bad_bytes[0] = &f.bytes[0][..1];
            let live = invoke_current_parent(&r, &f, &bad, &bad_bytes, mode, false).unwrap_err();
            assert_eq!(
                live,
                invoke_current_parent(&r, &f, &bad, &bad_bytes, mode, true).unwrap_err()
            );
            assert_eq!(live, SpatialError::BufferLengthMismatch { expected: 68, found });
        }
        for (index, found) in [(0, 67), (1, 66), (2, 65)] {
            let mut bad = bytes.clone();
            bad[index] = &f.bytes[index][..found];
            if index < 2 {
                bad[2] = &f.bytes[2][..1];
            }
            let live = invoke_current_parent(&r, &f, &attrs, &bad, mode, false).unwrap_err();
            assert_eq!(live, invoke_current_parent(&r, &f, &attrs, &bad, mode, true).unwrap_err());
            assert_eq!(live, SpatialError::BufferLengthMismatch { expected: 68, found });
        }
        let stats = r.buffer_pool().stats();
        let live = invoke_current_parent(&r, &empty, &[&[1.0]], &[&[1_u8]], mode, false).unwrap();
        assert_output(
            &live,
            &invoke_current_parent(&r, &empty, &[&[1.0]], &[&[1_u8]], mode, true).unwrap(),
        );
        assert_owned(&live, 0, 1, 1);
        assert_eq!(r.buffer_pool().stats(), stats);
    }
}

#[test]
fn combined_parallel_xyz_concurrent_four_modes_keep_prior_outputs_after_gpu_and_host_mutation() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    let gate = std::sync::Arc::new(std::sync::Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|mode| {
            let r = r.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                let cells = 1_048_576 + mode % 2;
                let attrs = mode;
                let bytes = [1, 3, 0, 5][mode];
                let mut f = fixture(&r, cells, attrs, bytes, 1);
                gate.wait();
                let live = complete_call(&r, &f, mode, false);
                let old = complete_call(&r, &f, mode, true);
                assert_output(&live, &old);
                assert_output(&live, &expected(&f, mode));
                assert_owned(&live, cells, attrs, bytes);
                for plane in &mut f.floats {
                    plane.fill(123.0);
                }
                for plane in &mut f.bytes {
                    plane.fill(77);
                }
                for (axis, buffer) in f.xyz.iter().enumerate() {
                    r.queue().write_buffer(buffer, 0, bytemuck::cast_slice(&f.floats[axis]));
                }
                let changed = complete_call(&r, &f, mode, false);
                assert_output(&changed, &expected(&f, mode));
                assert_owned(&changed, cells, attrs, bytes);
                assert!(floats(&changed).flatten().all(|v| v.to_bits() == 123.0_f32.to_bits()));
                assert!(changed.4.iter().flatten().all(|&b| b == 77));
                drop(f);
                assert_output(&live, &old);
                (live, old)
            })
        })
        .collect();
    let retained: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    r.clear_buffer_pool();
    for (live, old) in retained {
        assert_output(&live, &old);
    }
}

#[path = "voxel_combined_xyz_parallel_benchmark.rs"]
mod benchmark;
