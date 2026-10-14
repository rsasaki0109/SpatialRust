//! Original-order SoA preparation against the merged f3c5032 upload.
use super::super::{geometry, plan, GpuIcpInput, GpuKdTree, GpuQueryBatch, WgpuRuntime};
use super::*;
use spatialrust_core::TransferDirection;
use spatialrust_math::Isometry3;
use spatialrust_search::KdTree;

fn previous_upload(
    runtime: &WgpuRuntime,
    source: &[[f32; 3]],
    target: &KdTree,
) -> SpatialResult<GpuIcpInput> {
    plan(source.len() as u64, &runtime.device().limits())?;
    let layout = target.layout();
    let (x, y, z) = layout.positions();
    plan(x.len() as u64, &runtime.device().limits())?;
    if x.is_empty() {
        return Err(invalid());
    }
    let target_positions: Vec<_> = (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect();
    let (source_anchor, source_extent) = geometry(source)?;
    let (target_anchor, target_extent) = geometry(&target_positions)?;
    let extent = source_extent.max(target_extent);
    if extent > 0.0 && (extent as f32) < f32::MIN_POSITIVE {
        return Err(invalid());
    }
    let scale = if extent == 0.0 { 1.0 } else { extent.sqrt() as f32 };
    let tree = GpuKdTree::upload(runtime, target)?;
    let source = GpuQueryBatch::upload(runtime, source)?;
    let target = GpuQueryBatch::upload(runtime, &target_positions)?;
    let mut transfers = tree.upload_transfers();
    transfers
        .record(TransferDirection::HostToDevice, source.upload_transfers().host_to_device_bytes());
    transfers
        .record(TransferDirection::HostToDevice, target.upload_transfers().host_to_device_bytes());
    Ok(GpuIcpInput {
        source,
        target,
        tree,
        source_anchor,
        target_anchor,
        scale,
        identity: runtime.profiling_identity(),
        transfers,
    })
}

fn columns(points: &[[f32; 3]]) -> [Vec<f32>; 3] {
    std::array::from_fn(|axis| points.iter().map(|p| p[axis]).collect())
}

fn slices(axes: &[Vec<f32>; 3]) -> [&[f32]; 3] {
    [&axes[0], &axes[1], &axes[2]]
}

fn fixture(n: usize, profile: usize) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| {
            let j = (n - i - 1) as u32;
            let hash = j.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
            let zero = f32::from_bits((j % 2) << 31);
            match profile {
                0 => [
                    ((hash & 1023) as f32 - 512.0) / 32.0,
                    (((hash >> 10) & 1023) as f32 - 512.0) / 32.0,
                    (((hash >> 20) & 1023) as f32 - 512.0) / 32.0,
                ],
                1 => [(j % 31) as f32 * 0.125, (j % 17) as f32 * 0.125, 0.0],
                2 => [(j % 31) as f32 * 0.125, zero, -zero],
                3 => [0.0; 3],
                4 => [zero, -zero, zero],
                5 => [1e10; 3],
                6 => [f32::from_bits(j % 17), -f32::from_bits(j % 23), zero],
                _ => [-0.0; 3],
            }
        })
        .collect()
}

fn compare_geometry(points: &[[f32; 3]], available: usize) {
    let axes = columns(points);
    match (
        prepare_with_workers(points, available),
        prepare_soa_with_workers(slices(&axes), available),
    ) {
        (Ok((a, e)), Ok((b, f))) => {
            assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
            assert_eq!(e.to_bits(), f.to_bits());
        }
        (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
        _ => panic!("SoA geometry validation differs"),
    }
}

#[test]
fn soa_geometry_exact_at_worker_boundaries_and_concurrently() {
    let mut points_checked = 0;
    for n in [
        0, 1, 3, 257, 65_535, 65_536, 65_537, 131_071, 131_072, 131_073, 262_143, 262_144, 262_145,
        524_289,
    ] {
        for profile in 0..8 {
            let points = fixture(n, profile);
            let axes = columns(&points);
            for available in [0, 1, 2, 6] {
                compare_geometry(&points, available);
                points_checked += n;
            }
            let a = prepare(&points).unwrap();
            let b = prepare_soa(slices(&axes)).unwrap();
            assert_eq!(a.0.map(f32::to_bits), b.0.map(f32::to_bits));
            assert_eq!(a.1.to_bits(), b.1.to_bits());
            points_checked += n;
        }
    }
    let points = fixture(262_145, 4);
    std::thread::scope(|scope| {
        for workers in [1, 2, 6, 12] {
            let points = &points;
            scope.spawn(move || compare_geometry(points, workers));
        }
    });
    println!("soa_geometry_points={points_checked} concurrent_points={}", 4 * points.len());
}

#[test]
fn soa_geometry_rejects_shapes_nonfinite_and_extent_overflow() {
    for axes in [[&[][..], &[1.0][..], &[][..]], [&[1.0][..], &[][..], &[1.0][..]]] {
        for workers in [0, 1, 6] {
            assert!(prepare_soa_with_workers(axes, workers).is_err());
        }
        assert!(prepare_soa(axes).is_err());
    }
    for axis in 0..3 {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for index in [0, 65_536, 131_072, 262_144] {
                let mut points = fixture(262_145, 0);
                points[index][axis] = value;
                for workers in [1, 2, 6] {
                    compare_geometry(&points, workers);
                }
                assert!(prepare_soa(slices(&columns(&points))).is_err());
            }
        }
    }
    for points in [
        vec![[f32::MAX; 3]; 262_145],
        vec![[-f32::MAX; 3], [f32::MAX; 3]],
        vec![[f32::from_bits(1); 3], [f32::from_bits(2); 3]],
        vec![[-1e20; 3], [1e20; 3]],
    ] {
        for workers in [1, 2, 6] {
            compare_geometry(&points, workers);
        }
    }
    let axes: [Vec<_>; 3] = std::array::from_fn(|_| vec![f32::MAX; 262_145]);
    assert_eq!(prepare_soa(slices(&axes)).unwrap(), ([f32::MAX, f32::MAX, f32::MAX, 0.0], 0.0));
}

fn tree(points: &[[f32; 3]]) -> KdTree {
    let axes = columns(points);
    KdTree::from_slices(&axes[0], &axes[1], &axes[2])
}

#[test]
#[ignore = "requires native GPU adapter"]
fn soa_upload_preserves_all_moment_words_receipts_and_owned_results() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let mut cases = 0;
    for targets in [1, 17, 257, 65_537, 262_143, 262_144, 262_145, 524_289] {
        for profile in [0, 1, 2, 3, 4, 7] {
            let target_points = fixture(targets, profile);
            let source: Vec<_> =
                target_points.iter().step_by((targets / 33).max(1)).copied().collect();
            let target = tree(&target_points);
            let old = previous_upload(&runtime, &source, &target).unwrap();
            let new = GpuIcpInput::upload(&runtime, &source, &target).unwrap();
            assert_eq!(old.source_anchor.map(f32::to_bits), new.source_anchor.map(f32::to_bits));
            assert_eq!(old.target_anchor.map(f32::to_bits), new.target_anchor.map(f32::to_bits));
            assert_eq!(old.scale.to_bits(), new.scale.to_bits());
            assert_eq!(old.upload_transfers(), new.upload_transfers());
            let expected_bytes = 16 * source.len() as u64
                + 32 * targets as u64
                + 32 * target.layout().nodes().len() as u64;
            assert_eq!(new.upload_transfers().host_to_device_bytes(), expected_bytes);
            assert_eq!(new.upload_transfers().device_to_host_bytes(), 0);
            assert!(new.moments_resident(&other, Isometry3::identity(), 1.0).is_err());
            let results = [0.001, 1.0, 100.0].map(|distance| {
                (
                    old.moments_resident(&runtime, Isometry3::identity(), distance).unwrap(),
                    new.moments_resident(&runtime, Isometry3::identity(), distance).unwrap(),
                )
            });
            drop(old);
            drop(new);
            for (a, b) in results {
                assert_eq!(a.download(&runtime).unwrap(), b.download(&runtime).unwrap());
                assert_eq!(read_words(&runtime, &a.buffer), read_words(&runtime, &b.buffer));
                cases += 1;
            }
        }
    }
    let target = tree(&fixture(257, 0));
    let input = GpuIcpInput::upload(&runtime, &[], &target).unwrap();
    assert_eq!(input.source_count(), 0);
    let old = previous_upload(&runtime, &[], &target).unwrap();
    assert_eq!(
        old.moments_resident(&runtime, Isometry3::identity(), 1.0)
            .unwrap()
            .download(&runtime)
            .unwrap(),
        input
            .moments_resident(&runtime, Isometry3::identity(), 1.0)
            .unwrap()
            .download(&runtime)
            .unwrap()
    );
    for target_points in [
        vec![],
        vec![[f32::from_bits(1); 3], [f32::from_bits(2); 3]],
        vec![[-f32::MAX; 3], [f32::MAX; 3]],
    ] {
        let target = tree(&target_points);
        assert_eq!(
            previous_upload(&runtime, &[], &target).err().unwrap().to_string(),
            GpuIcpInput::upload(&runtime, &[], &target).err().unwrap().to_string()
        );
    }
    println!("soa_moment_cases={cases} exact_words={}", cases * 44);
}

fn read_words(runtime: &WgpuRuntime, buffer: &wgpu::Buffer) -> Vec<u32> {
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 176,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, 176);
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

#[path = "icp_soa_upload_benchmark.rs"]
mod benchmark;
