use super::*;
#[path = "voxel_channel_parallel_reference.rs"]
mod baseline;

fn invoke(
    r: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
    reduce: bool,
    old: bool,
) -> SpatialResult<Channels> {
    match (reduce, old) {
        (false, false) => gather_voxel_first_f32_multi_gpu(r, channels, segments),
        (false, true) => baseline::baseline_gather_voxel_first_f32_multi_gpu(r, channels, segments),
        (true, false) => reduce_voxel_average_f32_multi_gpu(r, channels, segments),
        (true, true) => {
            baseline::baseline_reduce_voxel_average_f32_multi_gpu(r, channels, segments)
        }
    }
}

fn assert_capacity(output: &[Vec<f32>], cells: usize, channels: usize) {
    assert_eq!(output.len(), channels);
    for plane in output {
        assert_eq!(plane.len(), cells);
        assert_eq!(plane.capacity(), cells);
    }
}

#[test]
fn parallel_channel_mapped_readback_keeps_raw_bits_cutoff_capacities_and_unmap_on_panic() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for (cells, channels) in [
        (0_usize, 0),
        (0, 7),
        (17, 7),
        (1_048_575, 2),
        (1_048_576, 2),
        (1_048_577, 7),
        (1_048_577, 1),
    ] {
        let input = words(cells * channels + 16);
        let buffer = staging(&r, &input);
        let live = crate::readback::read_staging_f32_channels(r.device(), &buffer, channels, cells)
            .unwrap();
        let old =
            baseline::read_staging_f32_channels(r.device(), &buffer, channels, cells).unwrap();
        assert_bits(&live, &old);
        assert_capacity(&live, cells, channels);
        for (c, plane) in live.iter().enumerate() {
            assert_eq!(bytemuck::cast_slice::<f32, u32>(plane), &input[c * cells..(c + 1) * cells]);
        }
        if cells == 1_048_576 {
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::readback::read_staging_f32_channels(r.device(), &buffer, channels + 1, cells)
                    .unwrap()
            }));
            assert!(panic.is_err());
            let next =
                crate::readback::read_staging_f32_channels(r.device(), &buffer, channels, cells)
                    .unwrap();
            assert_bits(&next, &old);
        }
        drop(buffer);
        assert_bits(&live, &old);
        checked += cells * channels * 2;
    }
    eprintln!("VOXEL_CHANNEL_PARALLEL_RAW verified {checked} serial-parent/CPU words, exact capacities, cutoff controls and panic/remap lifetime");
}

#[test]
fn parallel_channel_public_calls_match_complete_current_parent_cpu_and_owned_lifetimes() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for (cells, channels, occupancy) in [
        (16, 4, 4),
        (257, 7, 4),
        (1_048_575, 2, 1),
        (1_048_576, 1, 4),
        (1_048_576, 2, 4),
        (1_048_577, 7, 1),
        (1_048_577, 9, 1),
        (1_048_577, 17, 1),
    ] {
        let mut f = fixture(&r, cells, channels, occupancy);
        if cells == 257 {
            f.segments = crate::kernels::voxel_sort::build_voxel_segments_gpu_from_keys(
                &r,
                &(0..cells * occupancy).map(|i| ((i % cells) as i64, 0, 0)).collect::<Vec<_>>(),
            )
            .unwrap();
            assert_eq!(f.segments.construction_readback_bytes(), 4);
        }
        let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
        let stats = r.buffer_pool().stats();
        let mut retained = Vec::new();
        for reduce in [false, true] {
            let live = invoke(&r, &refs, &f.segments, reduce, false).unwrap();
            let old = invoke(&r, &refs, &f.segments, reduce, true).unwrap();
            let cpu = if reduce { &f.mean } else { &f.first };
            assert_bits(&live, &old);
            assert_bits(&live, cpu);
            assert_capacity(&live, cells, channels);
            assert_capacity(&old, cells, channels);
            retained.push(live);
            checked += cells * channels * 2;
        }
        assert_eq!(r.buffer_pool().stats(), stats);
        let expected = [f.first.clone(), f.mean.clone()];
        drop(refs);
        drop(f);
        r.clear_buffer_pool();
        for (live, cpu) in retained.iter().zip(&expected) {
            assert_bits(live, cpu);
        }
    }
    eprintln!("VOXEL_CHANNEL_PARALLEL_PUBLIC verified {checked} complete two-mode current-parent/CPU elements across cutoff, scalar/multibatch/multiwindow paths and input destruction");
}

#[test]
fn parallel_channel_forced_packed_and_coalesced_widths_keep_sparse_raw_single_tails() {
    use crate::kernels::{voxel_gather_submit as coalesced, voxel_standalone_packed as packed};
    let r = WgpuRuntime::new_headless().unwrap();
    let cells = 1_048_577;
    let mut checked = 0;
    let mut supported = 0;
    for width in [2, 4] {
        let channels = 2 * width + 1;
        let Some(plan) = packed::plan_packed_voxel_submission(
            &r.device().limits(),
            cells as u32,
            channels,
            width,
            isize::MAX as u64,
        ) else {
            continue;
        };
        supported += 1;
        let mut f = fixture(&r, cells, channels, 1);
        for (c, source) in f.source.iter_mut().enumerate() {
            *source = words(cells + c * 7).into_iter().skip(c * 7).map(f32::from_bits).collect();
            source[0] = f32::from_bits([0x8000_0000, 0x7fc0_1234, 0x7f80_0001][c % 3]);
        }
        let indices: Vec<_> = (0..cells).rev().map(|i| i as u32).collect();
        let starts: Vec<_> = (0..cells).map(|i| (i / 2) as u32).collect();
        let upload = |label, data: &[u32]| r.upload_u32_storage(label, data).unwrap();
        f.segments = GpuVoxelSegments::new(
            cells as u32,
            cells as u32,
            upload("parallel-channel-keys", &[0]),
            upload("parallel-channel-indices", &indices),
            upload("parallel-channel-starts", &starts),
            0,
        );
        let expected: Channels = f
            .source
            .iter()
            .map(|source| {
                (0..cells)
                    .map(|i| {
                        let start = starts[i];
                        let end = starts.get(i + 1).copied().unwrap_or(cells as u32);
                        source[if end <= start { 0 } else { indices[start as usize] as usize }]
                    })
                    .collect()
            })
            .collect();
        let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
        let live =
            packed::gather_voxel_channels_packed_with_plan(&r, &refs, &f.segments, plan).unwrap();
        let old_plan = baseline::voxel_standalone_packed::plan_packed_voxel_submission(
            &r.device().limits(),
            cells as u32,
            channels,
            width,
            isize::MAX as u64,
        )
        .unwrap();
        let old = baseline::voxel_standalone_packed::gather_voxel_channels_packed_with_plan(
            &r,
            &refs,
            &f.segments,
            old_plan,
        )
        .unwrap();
        assert_bits(&live, &old);
        assert_bits(&live, &expected);
        assert_capacity(&live, cells, channels);
        let plan = coalesced::plan_voxel_gather_submission(
            &r.device().limits(),
            cells as u32,
            channels,
            width,
        )
        .unwrap();
        let old_plan = baseline::voxel_gather_submit::plan_voxel_gather_submission(
            &r.device().limits(),
            cells as u32,
            channels,
            width,
        )
        .unwrap();
        let live =
            coalesced::gather_voxel_channels_with_plan(&r, &refs, &f.segments, plan).unwrap();
        let old = baseline::voxel_gather_submit::gather_voxel_channels_with_plan(
            &r,
            &refs,
            &f.segments,
            old_plan,
        )
        .unwrap();
        assert_bits(&live, &old);
        assert_bits(&live, &expected);
        assert_capacity(&live, cells, channels);
        checked += cells * channels * 4;
    }
    // A backend below the packed binding envelope still exercises the complete public fallback test.
    assert!(supported > 0 || r.device().limits().max_storage_buffers_per_shader_stage < 7);
    eprintln!("VOXEL_CHANNEL_PARALLEL_WIDTHS verified {checked} packed/coalesced serial-parent/CPU words at forced widths 2/4, empty cells, nonfinite bits and one-channel final windows");
}

#[test]
fn parallel_channel_public_non_power_averages_and_validation_keep_parent_error_order() {
    let r = WgpuRuntime::new_headless().unwrap();
    for channels in [1, 2, 5, 7, 9] {
        let f = fixture(&r, 257, channels, 3);
        let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
        let cpu: Vec<Vec<f64>> = f
            .source
            .iter()
            .map(|source| {
                (0..257)
                    .map(|cell| {
                        let sum: f64 =
                            (0..3).map(|lane| f64::from(source[lane * 257 + cell])).sum();
                        sum / 3.0
                    })
                    .collect()
            })
            .collect();
        let live = invoke(&r, &refs, &f.segments, true, false).unwrap();
        assert_bits(&live, &invoke(&r, &refs, &f.segments, true, true).unwrap());
        // Division by three may differ by an ULP between GPU and CPU backends.
        // Keep the GPU parent bit comparison exact, and use the existing
        // average-kernel error bound for the independent f64 reference.
        for (plane, expected) in live.iter().zip(&cpu) {
            for (&value, &expected) in plane.iter().zip(expected) {
                let tolerance = 4.0 * f64::from(f32::EPSILON) * expected.abs().max(1.0);
                assert!((f64::from(value) - expected).abs() <= tolerance);
            }
        }
    }
    let f = fixture(&r, 17, 7, 4);
    let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
    let empty = fixture(&r, 0, 7, 1);
    for reduce in [false, true] {
        for (index, found) in [(0, 67), (3, 66), (6, 65)] {
            let mut bad = refs.clone();
            bad[index] = &f.source[index][..found];
            if index < 6 {
                bad[6] = &f.source[6][..1];
            }
            let live = invoke(&r, &bad, &f.segments, reduce, false).unwrap_err();
            assert_eq!(live, invoke(&r, &bad, &f.segments, reduce, true).unwrap_err());
            assert_eq!(live, SpatialError::BufferLengthMismatch { expected: 68, found });
        }
        let stats = r.buffer_pool().stats();
        assert!(invoke(&r, &[], &f.segments, reduce, false).unwrap().is_empty());
        let live = invoke(&r, &[&[1.0], &[]], &empty.segments, reduce, false).unwrap();
        assert_capacity(&live, 0, 2);
        assert_bits(&live, &invoke(&r, &[&[1.0], &[]], &empty.segments, reduce, true).unwrap());
        assert_eq!(r.buffer_pool().stats(), stats);
    }
}

#[test]
fn parallel_channel_concurrent_public_calls_keep_outputs_after_source_mutation_and_drop() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    let gate = std::sync::Arc::new(std::sync::Barrier::new(2));
    let handles: Vec<_> = (0..2)
        .map(|worker| {
            let r = r.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                let cells = 1_048_577;
                let channels = worker + 2;
                let reduce = worker == 1;
                let mut f = fixture(&r, cells, channels, 1);
                let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
                gate.wait();
                let live = invoke(&r, &refs, &f.segments, reduce, false).unwrap();
                let old = invoke(&r, &refs, &f.segments, reduce, true).unwrap();
                assert_bits(&live, &old);
                assert_bits(&live, if reduce { &f.mean } else { &f.first });
                assert_capacity(&live, cells, channels);
                drop(refs);
                for source in &mut f.source {
                    source.fill(123.0);
                }
                let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
                let changed = invoke(&r, &refs, &f.segments, reduce, false).unwrap();
                assert!(changed.iter().flatten().all(|v| v.to_bits() == 123.0_f32.to_bits()));
                assert_capacity(&changed, cells, channels);
                drop(refs);
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

#[path = "voxel_channel_parallel_benchmark.rs"]
mod benchmark;
