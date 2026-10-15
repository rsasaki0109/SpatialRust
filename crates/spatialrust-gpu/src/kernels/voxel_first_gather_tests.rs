use super::*;
use crate::kernels::voxel_first_gather_fused::record_fused_voxel_first_gather;
use crate::kernels::voxel_gather::record_voxel_gather_xyz_pass;
use crate::kernels::voxel_gather::{
    record_voxel_gather_f32_pass, record_voxel_gather_multi2_pass, record_voxel_gather_multi4_pass,
};
use crate::kernels::voxel_reduce_fused::plan_fused_voxel_average;

#[path = "voxel_first_gather_reference.rs"]
mod baseline;

fn call_baseline(r: &WgpuRuntime, f: &Fixture, attrs: &[&[f32]], bytes: &[&[u8]]) -> Output {
    baseline::baseline_gather_voxel_first_xyz_and_multi_gpu(
        r,
        &f.xyz[0],
        &f.xyz[1],
        &f.xyz[2],
        attrs,
        bytes,
        &f.segments,
    )
    .unwrap()
}

fn expected_first_gather(f: &Fixture) -> Output {
    expected(f, 0)
}

fn assert_analytic(actual: &Output, expected: &Output) {
    assert_output(actual, expected);
}

#[test]
fn public_fused_voxel_first_gather_matches_parent_cpu_validation_and_fallbacks() {
    let r = WgpuRuntime::new_headless().unwrap();
    let supported = plan_fused_voxel_average(&r.device().limits(), 257, 4).is_some();
    let mut compared = 0;
    for (cells, occupancy) in
        [(0, 1), (1, 1), (17, 4), (257, 3), (1_025, 7), (65_537, 4), (1_048_577, 1)]
    {
        for attrs in [0, 1, 2, 3, 4, 5, 7] {
            for bytes in [0, 3] {
                let mut f = fixture(&r, cells, attrs, bytes, occupancy);
                if cells == 257 {
                    let keys: Vec<_> =
                        (0..cells * occupancy).map(|i| ((i % cells) as i64, 0, 0)).collect();
                    f.segments =
                        super::super::voxel_sort::build_voxel_segments_gpu_from_keys(&r, &keys)
                            .unwrap();
                    assert_eq!(f.segments.construction_readback_bytes(), 4);
                }
                let refs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
                let byte_refs: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
                let old = call_baseline(&r, &f, &refs, &byte_refs);
                let live = invoke(&r, &f, &refs, &byte_refs, 0, false).unwrap();
                assert_output(&live, &old);
                assert_analytic(&live, &expected_first_gather(&f));
                assert_capacity(&live, cells, attrs, bytes, false);
                let expected = old.clone();
                drop(refs);
                drop(byte_refs);
                drop(f);
                r.clear_buffer_pool();
                assert_output(&live, &expected);
                compared += cells * (3 + attrs + bytes) * 2;
            }
        }
    }
    let f = fixture(&r, 17, 4, 3, 4);
    let attrs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
    let bytes: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
    let mut bad = attrs.clone();
    bad[2] = &f.floats[5][..67];
    let mut bad_bytes = bytes.clone();
    bad_bytes[0] = &f.bytes[0][..66];
    for (af, bu, found) in [(&bad, &bad_bytes, 67), (&attrs, &bad_bytes, 66)] {
        let pool = r.buffer_pool().stats();
        let error = invoke(&r, &f, af, bu, 0, false).unwrap_err();
        assert_eq!(error, SpatialError::BufferLengthMismatch { expected: 68, found });
        assert_eq!(
            error,
            baseline::baseline_gather_voxel_first_xyz_and_multi_gpu(
                &r,
                &f.xyz[0],
                &f.xyz[1],
                &f.xyz[2],
                af,
                bu,
                &f.segments
            )
            .unwrap_err()
        );
        assert_eq!(r.buffer_pool().stats(), pool);
    }
    eprintln!(
        "VOXEL_FIRST_GATHER_PUBLIC verified {compared} parent/CPU elements; fused_supported={supported}"
    );
}

#[test]
fn fused_voxel_first_gather_kernel_preserves_separate_math_empty_cells_and_output_guards() {
    let r = WgpuRuntime::new_headless().unwrap();
    let Some(_) = plan_fused_voxel_average(&r.device().limits(), 65_537, 4) else {
        eprintln!("VOXEL_FIRST_GATHER_KERNEL unavailable: device limits");
        return;
    };
    let mut compared = 0;
    for cells in [1, 3, 17, 255, 256, 257, 1_025, 65_537] {
        let counts: Vec<_> = (0..cells).map(|i| [0, 1, 3, 4, 7, 17][i % 6]).collect();
        let mut points = 0;
        let starts: Vec<_> = counts
            .iter()
            .map(|&count| {
                let start = points;
                points += count;
                start as u32
            })
            .collect();
        let indices: Vec<_> = (0..points).rev().map(|i| i as u32).collect();
        let upload = |name, data: &[u32]| {
            r.upload_u32_storage(name, if data.is_empty() { &[0] } else { data }).unwrap()
        };
        let segments = GpuVoxelSegments::new(
            cells as u32,
            points as u32,
            upload("fused-keys", &[0]),
            upload("fused-indices", &indices),
            upload("fused-starts", &starts),
            0,
        );
        for attrs in 1..=4 {
            let source: Vec<Vec<f32>> = (0..3 + attrs)
                .map(|c| {
                    words(points + c * 7).into_iter().skip(c * 7).map(f32::from_bits).collect()
                })
                .collect();
            let values: Vec<_> = source
                .iter()
                .map(|v| {
                    r.upload_f32_storage("fused-values", if v.is_empty() { &[-0.0] } else { v })
                        .unwrap()
                })
                .collect();
            let plan = plan_fused_voxel_average(&r.device().limits(), cells as u32, attrs).unwrap();
            let fused = r
                .upload_u32_storage("fused-output", &vec![0xdead_beef; cells * (3 + attrs) + 8])
                .unwrap();
            // Upload storage buffers support copies, including sentinel readback.
            let mut encoder =
                r.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            record_fused_voxel_first_gather(
                &mut encoder,
                &r,
                [&values[0], &values[1], &values[2]],
                &values[3..],
                &segments,
                &fused,
            );
            let old: Vec<_> = (0..3 + attrs)
                .map(|_| {
                    r.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("fused-legacy-output"),
                        size: (cells * 4) as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    })
                })
                .collect();
            record_voxel_gather_xyz_pass(
                &mut encoder,
                &r,
                &values[0],
                &values[1],
                &values[2],
                &segments,
                &old[0],
                &old[1],
                &old[2],
            )
            .unwrap();
            match attrs {
                1 => record_voxel_gather_f32_pass(
                    &mut encoder,
                    &r,
                    &values[3],
                    segments.point_indices_buffer(),
                    segments.cell_starts_buffer(),
                    cells as u32,
                    points as u32,
                    &old[3],
                )
                .unwrap(),
                2 => record_voxel_gather_multi2_pass(
                    &mut encoder,
                    &r,
                    &[&values[3], &values[4]],
                    &segments,
                    &[&old[3], &old[4]],
                    2,
                )
                .unwrap(),
                _ => {
                    let dummy = r.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("centroid-first-dummy-output"),
                        size: (cells * 4) as u64,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    });
                    let inputs = std::array::from_fn(|i| values.get(3 + i).unwrap_or(&values[3]));
                    let outputs = std::array::from_fn(|i| old.get(3 + i).unwrap_or(&dummy));
                    record_voxel_gather_multi4_pass(
                        &mut encoder,
                        &r,
                        &inputs,
                        &segments,
                        &outputs,
                        attrs as u32,
                    )
                    .unwrap();
                }
            }
            let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("fused-kernel-readback"),
                size: plan.output_bytes * 2 + 32,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&fused, 0, &staging, 0, plan.output_bytes + 32);
            for (c, buffer) in old.iter().enumerate() {
                encoder.copy_buffer_to_buffer(
                    buffer,
                    0,
                    &staging,
                    plan.output_bytes + 32 + (c * cells * 4) as u64,
                    (cells * 4) as u64,
                );
            }
            r.queue().submit(Some(encoder.finish()));
            let result = crate::readback::read_staging_f32(
                r.device(),
                &staging,
                cells * (3 + attrs) * 2 + 8,
            )
            .unwrap();
            let len = cells * (3 + attrs);
            assert!(result[len..len + 8].iter().all(|v| v.to_bits() == 0xdead_beef));
            for (new, old) in result[..len].iter().zip(&result[len + 8..]) {
                assert_eq!(new.to_bits(), old.to_bits());
            }
            compared += len;
        }
    }
    assert!(compared > 0);
    eprintln!("VOXEL_FIRST_GATHER_KERNEL verified {compared} exact words including NaNs; empty cells and guards");
}

#[test]
fn concurrent_fused_voxel_first_gather_calls_share_lazy_pipeline_and_owned_results() {
    let r = WgpuRuntime::new_headless().unwrap();
    if plan_fused_voxel_average(&r.device().limits(), 17, 4).is_none() {
        return;
    }
    let barrier = std::sync::Barrier::new(3);
    std::thread::scope(|scope| {
        let mut jobs = Vec::new();
        for attrs in [1, 3, 4] {
            let (r, barrier) = (&r, &barrier);
            jobs.push(scope.spawn(move || {
                let f = fixture(r, 16_385, attrs, 3, 4);
                barrier.wait();
                let live = call(r, &f, 0, false);
                assert_analytic(&live, &expected_first_gather(&f));
                let expected = live.clone();
                drop(f);
                r.clear_buffer_pool();
                assert_output(&live, &expected);
                r.pipelines().voxel_gather.first_xyz_attributes_pipeline(r.device()) as *const _
                    as usize
            }));
        }
        let addresses: Vec<_> = jobs.into_iter().map(|j| j.join().unwrap()).collect();
        assert!(addresses.iter().all(|&a| a == addresses[0]));
    });
}

#[path = "voxel_first_gather_benchmark.rs"]
mod benchmark;

#[test]
fn fused_voxel_first_gather_reuses_uploads_with_changed_host_values_and_owned_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    if plan_fused_voxel_average(&r.device().limits(), 1_025, 4).is_none() {
        return;
    }
    for attrs in 1..=4 {
        let mut f = fixture(&r, 1_025, attrs, 3, 3);
        r.clear_buffer_pool();
        let mut live = call(&r, &f, 0, false);
        let expected_live = expected_first_gather(&f);
        assert_analytic(&live, &expected_live);
        let warm = r.buffer_pool().stats();
        assert_eq!(warm.cached_resources, attrs);
        for (c, channel) in f.floats[3..].iter_mut().enumerate() {
            for (i, value) in channel.iter_mut().enumerate() {
                *value = f32::from_bits(
                    [0x7fc0_1234, 0x8000_0000, 0x7f80_0000, 1, 0x3f80_0000][(i + c) % 5],
                );
            }
        }
        for channel in &mut f.bytes {
            channel.fill(231);
        }
        let next = call(&r, &f, 0, false);
        let reused = r.buffer_pool().stats();
        assert_eq!(reused.allocation_count, warm.allocation_count);
        assert_eq!(reused.reuse_count - warm.reuse_count, attrs as u64);
        assert_eq!(reused.cached_resources, attrs);
        assert_analytic(&next, &expected_first_gather(&f));
        let refs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
        let bytes: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
        assert_output(&next, &call_baseline(&r, &f, &refs, &bytes));
        let expected_next = next.clone();
        live.0[0] = f32::from_bits(0xdead_beef);
        live.3[0][0] = 999.0;
        drop(refs);
        drop(bytes);
        drop(f);
        r.clear_buffer_pool();
        assert_eq!(live.0[0].to_bits(), 0xdead_beef);
        assert_eq!(live.3[0][0], 999.0);
        assert_output(&next, &expected_next);
    }
}

#[test]
fn public_fused_voxel_first_gather_sparse_empty_cells_preserve_all_word_and_byte_rules() {
    let r = WgpuRuntime::new_headless().unwrap();
    if plan_fused_voxel_average(&r.device().limits(), 17, 4).is_none() {
        return;
    }
    for attrs in 1..=4 {
        for bytes in [0, 3] {
            let mut f = fixture(&r, 17, attrs, bytes, 1);
            for (c, plane) in f.floats.iter_mut().enumerate() {
                *plane = words(17 + c * 7).into_iter().skip(c * 7).map(f32::from_bits).collect();
                if c < 3 {
                    r.queue().write_buffer(&f.xyz[c], 0, bytemuck::cast_slice(plane));
                }
            }
            let mut starts: Vec<u32> = (0..17).map(|i| i / 2).collect();
            // Also retain legacy end <= start behavior for a descending pair.
            starts[3] = 0;
            let indices: Vec<u32> = (0..17).rev().collect();
            let upload = |name, values: &[u32]| r.upload_u32_storage(name, values).unwrap();
            f.segments = GpuVoxelSegments::new(
                17,
                17,
                upload("sparse-keys", &[0]),
                upload("sparse-indices", &indices),
                upload("sparse-starts", &starts),
                0,
            );
            let refs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
            let byte_refs: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
            let live = invoke(&r, &f, &refs, &byte_refs, 0, false).unwrap();
            assert_output(&live, &call_baseline(&r, &f, &refs, &byte_refs));
            let mut oracle: Output = (
                vec![0.0; 17],
                vec![0.0; 17],
                vec![0.0; 17],
                vec![vec![0.0; 17]; attrs],
                vec![vec![0; 17]; bytes],
            );
            for cell in 0..17 {
                let end = starts.get(cell + 1).copied().unwrap_or(17);
                let empty = end <= starts[cell];
                let index = if empty { 0 } else { indices[starts[cell] as usize] as usize };
                oracle.0[cell] = f.floats[0][index];
                oracle.1[cell] = f.floats[1][index];
                oracle.2[cell] = f.floats[2][index];
                for c in 0..attrs {
                    oracle.3[c][cell] =
                        if empty && attrs == 1 { 0.0 } else { f.floats[3 + c][index] };
                }
                for c in 0..bytes {
                    oracle.4[c][cell] = if empty { 0 } else { f.bytes[c][index] };
                }
            }
            assert_output(&live, &oracle);
            assert_capacity(&live, 17, attrs, bytes, false);
        }
    }
}
