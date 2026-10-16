use super::*;
use crate::readback_planes::{
    copy_owned_packed_u8_planes, copy_packed_u8_with_plan, plan_packed_u8_copy,
};
#[path = "voxel_u8_parallel_reference.rs"]
mod baseline;

#[test]
fn parallel_u8_plans_bound_workers_packed_extents_and_four_mib_per_channel() {
    let mut checked = 0;
    for cells in [0_usize, 1, 17, 4_194_303, 4_194_304, 4_194_305, 4_194_306, 4_194_307, 8_388_609]
    {
        for channels in [0, 1, 2, 3, 4, 5, 7, 17] {
            for available in [0, 1, 2, 3, 4, 8, usize::MAX] {
                let plan = plan_packed_u8_copy(cells, channels, available);
                assert_eq!(plan.is_some(), cells >= 4_194_304 && channels >= 2 && available >= 2);
                if let Some(plan) = plan {
                    assert!(
                        plan.workers <= 4 && plan.workers <= channels && plan.workers <= available
                    );
                    assert_eq!(plan.channels_per_worker, channels.div_ceil(plan.workers));
                    assert!(channels.div_ceil(plan.channels_per_worker) <= plan.workers);
                    let covered: usize = (0..channels)
                        .step_by(plan.channels_per_worker)
                        .map(|first| (channels - first).min(plan.channels_per_worker))
                        .sum();
                    assert_eq!(covered, channels);
                    assert!(cells >= 4_194_304);
                }
                checked += 1;
            }
        }
    }
    for (cells, channels) in
        [(usize::MAX, 2), (4_194_304, usize::MAX), (isize::MAX as usize / 2 + 1, 2)]
    {
        assert!(plan_packed_u8_copy(cells, channels, 4).is_none());
        checked += 1;
    }
    eprintln!("VOXEL_U8_PARALLEL_PLAN verified {checked} byte threshold, bounded workers, exact bands and packed overflow guards");
}

#[test]
fn parallel_u8_host_forced_workers_keep_all_remainders_independent_bits_and_owned_capacities() {
    let mut checked = 0;
    for (cells, channels) in [
        (0_usize, 0),
        (0, 3),
        (1, 3),
        (2, 5),
        (3, 7),
        (4, 2),
        (17, 7),
        (4_194_303, 3),
        (4_194_304, 0),
        (4_194_304, 1),
        (4_194_304, 2),
        (4_194_305, 3),
        (4_194_306, 5),
        (4_194_307, 7),
        (8_388_609, 2),
    ] {
        let words_per_channel = cells.div_ceil(4);
        let input = words(words_per_channel * channels + 16);
        let mut variants = vec![copy_owned_packed_u8_planes(&input, cells, channels)];
        for available in [2, 3, 4] {
            if let Some(plan) = plan_packed_u8_copy(cells, channels, available) {
                variants.push(copy_packed_u8_with_plan(&input, cells, channels, plan));
            }
        }
        for mut output in variants {
            assert_eq!(output.len(), channels);
            for (channel, plane) in output.iter().enumerate() {
                assert_eq!(plane.len(), cells);
                assert_eq!(plane.capacity(), cells);
                for (cell, &value) in plane.iter().enumerate() {
                    let word = input[channel * words_per_channel + cell / 4];
                    assert_eq!(value, ((word >> ((cell % 4) * 8)) & 255) as u8);
                }
            }
            if cells > 0 && channels > 0 {
                output[0][0] ^= 255;
                assert_eq!(input[0], 0);
                if channels > 1 {
                    assert_eq!(output[1][0], (input[words_per_channel] & 255) as u8);
                }
            }
            checked += cells * channels;
        }
    }
    let plan = plan_packed_u8_copy(4_194_304, 2, 4).unwrap();
    assert!(
        std::panic::catch_unwind(|| copy_packed_u8_with_plan(&[0], 4_194_304, 2, plan)).is_err()
    );
    assert!(std::panic::catch_unwind(|| copy_owned_packed_u8_planes(&[], 17, 2)).is_err());
    assert!(std::panic::catch_unwind(|| copy_owned_packed_u8_planes(&[], usize::MAX, usize::MAX))
        .is_err());
    eprintln!("VOXEL_U8_PARALLEL_HOST verified {checked} automatic/forced two/three/four-worker bytes, all tails, exact capacities and independent ownership");
}

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
fn parallel_u8_mapped_planes_keep_serial_cpu_bits_byte_tails_and_panic_remap() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for (cells, attrs, bytes) in [
        (0_usize, 0, 0),
        (17, 1, 3),
        (4_194_303, 0, 3),
        (4_194_304, 0, 0),
        (4_194_304, 0, 1),
        (4_194_304, 1, 2),
        (4_194_305, 0, 3),
        (4_194_306, 0, 5),
        (4_194_307, 2, 7),
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
        if cells == 4_194_304 && attrs == 1 {
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
    eprintln!("VOXEL_U8_PARALLEL_RAW verified {checked} mapped serial-byte-parent/CPU elements, cutoff/zero/single/many attribute controls, raw nonfinite bits, byte tails, panic/remap and owned destruction");
}

#[test]
fn parallel_u8_all_four_public_calls_match_complete_current_parents_cpu_and_capacities() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for (cells, attrs, bytes, occupancy) in [
        (0, 7, 3, 4),
        (16, 0, 0, 4),
        (257, 7, 3, 4),
        (65_537, 1, 1, 4),
        (4_194_303, 0, 3, 1),
        (4_194_304, 0, 0, 1),
        (4_194_304, 0, 1, 1),
        (4_194_304, 1, 2, 1),
        (4_194_305, 0, 3, 1),
        (4_194_306, 0, 5, 1),
        (4_194_307, 2, 7, 1),
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
    eprintln!("VOXEL_U8_PARALLEL_PUBLIC verified {checked} complete four-mode current-parent/CPU elements, serial/cutoff/large/sorted cases, zero/single/wide float and byte paths, exact capacities and owned destruction");
}

#[test]
fn parallel_u8_public_validation_preserves_float_then_byte_order_and_empty_returns() {
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
fn parallel_u8_concurrent_four_modes_keep_prior_outputs_after_gpu_and_host_mutation() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    let gate = std::sync::Arc::new(std::sync::Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|mode| {
            let r = r.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                let cells = 4_194_304 + mode;
                let attrs = mode % 2;
                let bytes = [2, 3, 5, 7][mode];
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

#[path = "voxel_u8_parallel_benchmark.rs"]
mod benchmark;
