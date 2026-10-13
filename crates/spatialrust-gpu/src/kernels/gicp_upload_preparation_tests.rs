//! Frozen serial staging/upload from b56246c (PR #188), plus independent cases.
use super::super::{plan, GpuGicpInput, GpuIcpInput, WgpuRuntime};
use super::*;
use spatialrust_core::{TransferDirection, TransferStats};
use spatialrust_math::Isometry3;
use spatialrust_search::KdTree;
use wgpu::util::DeviceExt;

fn previous_prepare(
    source: &[[f32; 6]],
    target: &[[f32; 6]],
) -> SpatialResult<(Vec<[f32; 8]>, f32)> {
    for covariance in source {
        inverse_trace(covariance)?;
    }
    let mut bound = 0.0_f64;
    for covariance in target {
        bound = bound.max(inverse_trace(covariance)?);
    }
    let weight_scale = bound as f32;
    if !weight_scale.is_finite() || weight_scale < f32::MIN_POSITIVE {
        return Err(invalid());
    }
    let packed: Vec<[f32; 8]> = source
        .iter()
        .chain(target)
        .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5], 0.0, 0.0])
        .collect();
    Ok((packed, weight_scale))
}

// Frozen upload ordering: SPD validation, positions upload, then serial packing.
fn previous_upload(
    runtime: &WgpuRuntime,
    source: &[[f32; 3]],
    source_covariances: &[[f32; 6]],
    target: &KdTree,
    target_covariances: &[[f32; 6]],
) -> SpatialResult<GpuGicpInput> {
    if source.len() != source_covariances.len()
        || target.layout().positions().0.len() != target_covariances.len()
    {
        return Err(invalid());
    }
    let total =
        (source.len() as u64).checked_add(target_covariances.len() as u64).ok_or_else(invalid)?;
    plan(source.len() as u64, total, &runtime.device().limits())?;
    for covariance in source_covariances {
        inverse_trace(covariance)?;
    }
    let mut bound = 0.0_f64;
    for covariance in target_covariances {
        bound = bound.max(inverse_trace(covariance)?);
    }
    let weight_scale = bound as f32;
    if !weight_scale.is_finite() || weight_scale < f32::MIN_POSITIVE {
        return Err(invalid());
    }
    let positions = GpuIcpInput::upload(runtime, source, target)?;
    let packed: Vec<[f32; 8]> = source_covariances
        .iter()
        .chain(target_covariances)
        .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5], 0.0, 0.0])
        .collect();
    let covariances = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gicp-covariances"),
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut transfers = positions.upload_transfers();
    transfers.record(TransferDirection::HostToDevice, total * 32);
    Ok(GpuGicpInput { positions, covariances, weight_scale, covariance_offset: 0, transfers })
}

fn rows(n: usize, phase: usize) -> Vec<[f32; 6]> {
    (0..n)
        .map(|i| {
            let scale = [1e-8, 0.25, 1.0, 1e8][(i + phase) % 4];
            let c = if (i + phase) % 3 == 0 {
                [1.0, 2.0, 4.0, -0.0, 0.0, -0.0]
            } else {
                [2.0, 3.0, 4.0, 0.25, -0.5, 0.125]
            };
            c.map(|v| v * scale)
        })
        .collect()
}

fn compare(source: &[[f32; 6]], target: &[[f32; 6]], available: usize) {
    let old = previous_prepare(source, target);
    let new = prepare_with_workers(source, target, available);
    match (old, new) {
        (Ok((a, scale_a)), Ok((b, scale_b))) => {
            assert_eq!(scale_a.to_bits(), scale_b.to_bits());
            assert_eq!(bytemuck::cast_slice::<_, u32>(&a), bytemuck::cast_slice::<_, u32>(&b));
        }
        (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
        _ => panic!("serial/parallel validation outcome differs"),
    }
}

#[test]
fn exact_rows_bounds_worker_splits_and_concurrent_preparation() {
    let mut compared = 0;
    for n in [0, 1, 257, 16_383, 16_384, 32_767, 32_768, 32_769, 65_537, 262_145] {
        for targets in [1, 257, n.max(1)] {
            let source = rows(n, 0);
            let target = rows(targets, 1);
            for workers in [0, 1, 2, 6] {
                compare(&source, &target, workers);
                compared += source.len() + target.len();
            }
            let public = prepare(&source, &target).unwrap();
            let old = previous_prepare(&source, &target).unwrap();
            assert_eq!(public.1.to_bits(), old.1.to_bits());
            assert_eq!(
                bytemuck::cast_slice::<_, u32>(&public.0),
                bytemuck::cast_slice::<_, u32>(&old.0)
            );
            compared += source.len() + target.len();
        }
    }
    let source = rows(65_537, 0);
    let target = rows(32_769, 1);
    std::thread::scope(|scope| {
        for workers in [1, 2, 6, 12] {
            let (source, target) = (&source, &target);
            scope.spawn(move || compare(source, target, workers));
        }
    });
    println!("exact_upload_rows={compared} concurrent_rows={}", 4 * (source.len() + target.len()));
    for available in [0, 1, 2, 6, 12, usize::MAX] {
        for n in [0, 16_384, 32_767, 32_768, 98_304, usize::MAX] {
            assert_eq!(worker_count(n, available), available.max(1).min((n / 16_384).max(1)));
        }
    }
}

#[test]
fn independent_scale_padding_and_invalid_chunk_boundaries() {
    let (packed, scale) =
        prepare(&[[2.0, 4.0, 8.0, -0.0, 0.0, -0.0]], &[[1.0, 2.0, 4.0, 0.0, 0.0, 0.0]]).unwrap();
    assert_eq!(scale, 1.75);
    assert_eq!(
        packed[0].map(f32::to_bits),
        [2.0, 4.0, 8.0, -0.0, 0.0, -0.0, 0.0, 0.0].map(f32::to_bits)
    );
    let mut bad = vec![[0.0; 6], [1.0, 1.0, 1.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0, 2.0]];
    for axis in 0..6 {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut row = [1.0, 2.0, 4.0, 0.0, 0.0, 0.0];
            row[axis] = value;
            bad.push(row);
        }
    }
    for row in bad {
        for (is_target, index) in [
            (false, 0),
            (false, 16_383),
            (false, 32_768),
            (true, 0),
            (true, 16_384),
            (true, 32_768),
        ] {
            let mut source = rows(32_769, 0);
            let mut target = rows(32_769, 1);
            if is_target {
                target[index] = row;
            } else {
                source[index] = row;
            }
            for workers in [1, 2, 6] {
                compare(&source, &target, workers);
            }
        }
    }
    for scale in [f32::from_bits(1), f32::MIN_POSITIVE, f32::MAX] {
        let target = [[scale, scale, scale, 0.0, 0.0, 0.0]];
        compare(&rows(32_769, 0), &target, 6);
    }
    compare(&[], &[], 6);
    compare(&rows(32_769, 0), &[], 6);
}

fn tree(points: &[[f32; 3]]) -> KdTree {
    let columns: [Vec<_>; 3] = std::array::from_fn(|axis| points.iter().map(|p| p[axis]).collect());
    KdTree::from_slices(&columns[0], &columns[1], &columns[2])
}

type Fixture = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 6]>, Vec<[f32; 6]>);

fn fixture(n: usize, targets: usize) -> Fixture {
    let points = |len| {
        (0..len)
            .map(|i| [(i % 7) as f32 * 0.01, (i % 11) as f32 * 0.01, (i % 13) as f32 * 0.01])
            .collect()
    };
    // Well-conditioned varied SPD rows avoid shader conditioning rejection.
    let covariance = |len| {
        (0..len).map(|i| [2.0 + (i % 5) as f32 * 0.25, 3.0, 4.0, 0.25, -0.5, 0.125]).collect()
    };
    (points(n), points(targets), covariance(n), covariance(targets))
}

#[test]
#[ignore = "requires native GPU adapter"]
fn uploaded_equations_receipts_and_live_results_match_previous() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut cases = 0;
    for n in [0, 1, 32_767, 32_768, 32_769, 65_537, 262_145] {
        for targets in [1, 257] {
            let (source, target, sc, tc) = fixture(n, targets);
            let target = tree(&target);
            let old = previous_upload(&runtime, &source, &sc, &target, &tc).unwrap();
            let new = GpuGicpInput::upload(&runtime, &source, &sc, &target, &tc).unwrap();
            assert_eq!(old.weight_scale.to_bits(), new.weight_scale.to_bits());
            assert_eq!(old.upload_transfers(), new.upload_transfers());
            let mut expected = TransferStats::default();
            expected.record(
                TransferDirection::HostToDevice,
                (48 * n + 64 * targets + 32 * target.layout().nodes().len()) as u64,
            );
            assert_eq!(new.upload_transfers(), expected);
            let pairs: Vec<_> = [0.001, 0.025, 1.0]
                .map(|distance| {
                    (
                        old.equations_resident(&runtime, Isometry3::identity(), distance).unwrap(),
                        new.equations_resident(&runtime, Isometry3::identity(), distance).unwrap(),
                    )
                })
                .into();
            drop(old);
            drop(new);
            for (a, b) in pairs {
                assert_eq!(a.download(&runtime).unwrap(), b.download(&runtime).unwrap());
                let read = |output: &super::super::GpuGicpNormalEquations| {
                    read_words(&runtime, &output.buffer)
                };
                assert_eq!(read(&a), read(&b));
                cases += 1;
            }
        }
    }
    println!("exact_upload_equation_cases={cases} words={}", cases * 64);
}

fn read_words(runtime: &WgpuRuntime, buffer: &wgpu::Buffer) -> Vec<u32> {
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, 256);
    runtime.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    runtime.wait_idle();
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let words = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    words
}

#[path = "gicp_upload_preparation_benchmark.rs"]
mod benchmark;
