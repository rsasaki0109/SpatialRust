use super::*;
use crate::readback_planes::{copy_owned_f32_planes, copy_with_plan, plan_plane_copy};
#[path = "voxel_parallel_readback_reference.rs"]
mod baseline;

#[test]
fn parallel_voxel_plane_plans_bound_workers_and_reject_small_or_overflow_extents() {
    let mut checked = 0;
    for cells in [0, 1, 1_048_575, 1_048_576, 1_048_577, 2_097_153] {
        for channels in [0, 1, 2, 3, 4, 5, 7, 17] {
            for available in [0, 1, 2, 3, 4, 8, usize::MAX] {
                let plan = plan_plane_copy(cells, channels, available);
                let eligible = cells >= 1_048_576 && channels >= 2 && available >= 2;
                assert_eq!(plan.is_some(), eligible);
                if let Some(plan) = plan {
                    assert!(
                        plan.workers <= 4 && plan.workers <= channels && plan.workers <= available
                    );
                    assert!(channels.div_ceil(plan.channels_per_worker) <= plan.workers);
                    assert!(plan.channels_per_worker * cells >= 1_048_576);
                    let covered: usize = (0..channels)
                        .step_by(plan.channels_per_worker)
                        .map(|start| (channels - start).min(plan.channels_per_worker))
                        .sum();
                    assert_eq!(covered, channels);
                }
                checked += 1;
            }
        }
    }
    for (cells, channels) in
        [(usize::MAX, 2), (1_048_576, usize::MAX), (isize::MAX as usize / 8 + 1, 2)]
    {
        assert!(plan_plane_copy(cells, channels, 4).is_none());
        checked += 1;
    }
    eprintln!("VOXEL_PARALLEL_READBACK_PLAN verified {checked} cutoff, worker/plane caps, exact coverage and checked host extents");
}

#[test]
fn parallel_voxel_host_planes_keep_raw_bits_exact_capacities_and_independent_ownership() {
    let mut checked = 0;
    for cells in [0, 17, 1_048_575, 1_048_576, 1_048_577] {
        for channels in [0, 1, 2, 3, 7] {
            let input = words(cells * channels + 16);
            let values: &[f32] = bytemuck::cast_slice(&input);
            let mut variants = vec![copy_owned_f32_planes(values, cells, channels)];
            for available in [2, 3, 4] {
                if let Some(plan) = plan_plane_copy(cells, channels, available) {
                    variants.push(copy_with_plan(values, cells, channels, plan));
                }
            }
            for mut output in variants {
                assert_eq!(output.len(), channels);
                for (c, plane) in output.iter().enumerate() {
                    assert_eq!(plane.len(), cells);
                    assert_eq!(plane.capacity(), cells);
                    assert_eq!(
                        bytemuck::cast_slice::<f32, u32>(plane),
                        &input[c * cells..(c + 1) * cells]
                    );
                }
                if cells > 0 && channels > 0 {
                    output[0][0] = f32::from_bits(0xdead_beef);
                    assert_eq!(input[0], 0);
                    if channels > 1 {
                        assert_eq!(output[1][0].to_bits(), input[cells]);
                    }
                }
                checked += cells * channels;
            }
        }
    }
    let plan = plan_plane_copy(1_048_576, 2, 4).unwrap();
    assert!(std::panic::catch_unwind(|| copy_with_plan(&[0.0], 1_048_576, 2, plan)).is_err());
    assert!(std::panic::catch_unwind(|| copy_owned_f32_planes(&[], usize::MAX, 2)).is_err());
    eprintln!("VOXEL_PARALLEL_READBACK_HOST verified {checked} serial/automatic/forced-worker raw words, capacities and source/output independence");
}

#[test]
fn parallel_voxel_mapped_planes_match_serial_cpu_bits_byte_tails_and_unmap_on_unwind() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for (cells, attributes, bytes) in [
        (0_usize, 0, 0),
        (17, 7, 3),
        (1_048_575, 2, 1),
        (1_048_576, 2, 3),
        (1_048_577, 7, 5),
        (1_048_577, 1, 3),
    ] {
        let float_words = cells * (3 + attributes);
        let byte_words = cells.div_ceil(4) * bytes;
        let input = words(float_words + byte_words + 16);
        let buffer = staging(&r, &input);
        let live = crate::readback::read_staging_xyz_and_packed_attributes(
            r.device(),
            &buffer,
            cells,
            attributes,
            bytes,
        )
        .unwrap();
        let old = baseline::baseline_read_staging_xyz_and_packed_attributes(
            r.device(),
            &buffer,
            cells,
            attributes,
            bytes,
        )
        .unwrap();
        assert_output(&live, &old);
        assert_capacity(&live, cells, attributes, bytes, false);
        for (c, plane) in floats(&live).enumerate() {
            assert_eq!(bytemuck::cast_slice::<f32, u32>(plane), &input[c * cells..(c + 1) * cells]);
        }
        for (c, plane) in live.4.iter().enumerate() {
            for (cell, &byte) in plane.iter().enumerate() {
                assert_eq!(
                    byte,
                    ((input[float_words + c * cells.div_ceil(4) + cell / 4] >> ((cell % 4) * 8))
                        & 255) as u8
                );
            }
        }
        if cells == 1_048_576 {
            for (bad_cells, bad_attrs, bad_bytes) in
                [(cells, attributes + 20, bytes), (cells, attributes, bytes + 20)]
            {
                let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::readback::read_staging_xyz_and_packed_attributes(
                        r.device(),
                        &buffer,
                        bad_cells,
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
                    attributes,
                    bytes,
                )
                .unwrap();
                assert_output(&next, &old);
            }
        }
        drop(buffer);
        r.clear_buffer_pool();
        assert_output(&live, &old);
        checked += cells * (3 + attributes + bytes) * 2;
    }
    eprintln!("VOXEL_PARALLEL_READBACK_RAW verified {checked} exact serial/CPU float and packed-byte elements, cutoff controls, ownership and panic/remap bounds");
}

#[test]
fn parallel_voxel_complete_four_modes_keep_serial_parent_cpu_lifetimes_and_error_order() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for (cells, attributes, bytes, mode) in [
        (16, 4, 0, 0),
        (4096, 7, 3, 1),
        (1_048_575, 2, 0, 2),
        (1_048_576, 2, 3, 0),
        (1_048_577, 7, 3, 1),
        (1_048_576, 4, 0, 2),
        (1_048_577, 4, 5, 3),
        (1_048_577, 1, 1, 0),
    ] {
        let f = fixture(&r, cells, attributes, bytes, 4);
        let live = complete_call(&r, &f, mode, false);
        let old = complete_call(&r, &f, mode, true);
        let cpu = expected(&f, mode);
        assert_output(&live, &old);
        assert_output(&live, &cpu);
        assert_capacity(&live, cells, attributes, bytes, false);
        drop(f);
        r.clear_buffer_pool();
        assert_output(&live, &cpu);
        checked += cells * (3 + attributes + bytes) * 2;
    }
    let f = fixture(&r, 17, 7, 3, 4);
    let attrs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
    let bytes: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
    for mode in 0..4 {
        for (index, found) in [(0, 67), (3, 66), (6, 65)] {
            let mut bad = attrs.clone();
            bad[index] = &f.floats[index + 3][..found];
            let mut bad_bytes = bytes.clone();
            bad_bytes[0] = &f.bytes[0][..64];
            assert_eq!(
                invoke_current_parent(&r, &f, &bad, &bad_bytes, mode, false).unwrap_err(),
                invoke_current_parent(&r, &f, &bad, &bad_bytes, mode, true).unwrap_err()
            );
            assert_eq!(
                invoke_current_parent(&r, &f, &bad, &bad_bytes, mode, false).unwrap_err(),
                SpatialError::BufferLengthMismatch { expected: 68, found }
            );
        }
        let empty = fixture(&r, 0, 7, 3, 1);
        let stats = r.buffer_pool().stats();
        let live = invoke_current_parent(&r, &empty, &[&[1.0]], &[&[1_u8]], mode, false).unwrap();
        assert_capacity(&live, 0, 1, 1, false);
        assert_eq!(r.buffer_pool().stats(), stats);
    }
    eprintln!("VOXEL_PARALLEL_READBACK_PUBLIC verified {checked} complete four-mode current-parent/CPU elements, cutoff/one-attribute controls, ownership and ordered validation");
}

#[test]
fn parallel_voxel_concurrent_mapped_copies_keep_prior_outputs_after_buffer_mutation_and_drop() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    let gate = std::sync::Arc::new(std::sync::Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|worker| {
            let r = r.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                let cells: usize = 1_048_577;
                let attributes = [2, 3, 4, 7][worker];
                let bytes = 3;
                let float_words = cells * (3 + attributes);
                let byte_words = cells.div_ceil(4) * bytes;
                let mut input = words(float_words + byte_words);
                let buffer = staging(&r, &input);
                gate.wait();
                let live = crate::readback::read_staging_xyz_and_packed_attributes(
                    r.device(),
                    &buffer,
                    cells,
                    attributes,
                    bytes,
                )
                .unwrap();
                let old = baseline::baseline_read_staging_xyz_and_packed_attributes(
                    r.device(),
                    &buffer,
                    cells,
                    attributes,
                    bytes,
                )
                .unwrap();
                assert_output(&live, &old);
                input[..float_words].fill(123.0_f32.to_bits());
                input[float_words..].fill(0xe7e7_e7e7);
                r.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(&input));
                r.queue().submit([]);
                let changed = crate::readback::read_staging_xyz_and_packed_attributes(
                    r.device(),
                    &buffer,
                    cells,
                    attributes,
                    bytes,
                )
                .unwrap();
                assert_capacity(&changed, cells, attributes, bytes, false);
                assert!(floats(&changed).flatten().all(|v| v.to_bits() == 123.0_f32.to_bits()));
                assert!(changed.4.iter().flatten().all(|v| *v == 231));
                drop(buffer);
                drop(input);
                r.clear_buffer_pool();
                assert_output(&live, &old);
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[path = "voxel_parallel_readback_benchmark.rs"]
mod benchmark;
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
