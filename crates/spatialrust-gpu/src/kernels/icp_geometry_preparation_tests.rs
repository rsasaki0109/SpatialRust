//! Frozen geometry/upload from b1f58d5 (PR #189); independent numeric cases.
use super::super::{plan, GpuIcpInput, GpuKdTree, GpuQueryBatch, WgpuRuntime};
use super::*;
use spatialrust_core::TransferDirection;
use spatialrust_math::Isometry3;
use spatialrust_search::KdTree;

fn previous_geometry(points: &[[f32; 3]]) -> SpatialResult<([f32; 4], f64)> {
    if points.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid());
    }
    if points.is_empty() {
        return Ok(([0.0; 4], 0.0));
    }
    let mut anchor = [0.0; 4];
    let mut extent_squared = 0.0;
    for axis in 0..3 {
        let lo = points.iter().map(|p| p[axis]).fold(f32::INFINITY, f32::min);
        let hi = points.iter().map(|p| p[axis]).fold(f32::NEG_INFINITY, f32::max);
        anchor[axis] = ((f64::from(lo) + f64::from(hi)) * 0.5) as f32;
        extent_squared += (f64::from(hi) - f64::from(lo)).powi(2);
    }
    if !(extent_squared as f32).is_finite() {
        return Err(invalid());
    }
    Ok((anchor, extent_squared))
}

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
    let (source_anchor, source_extent) = previous_geometry(source)?;
    let (target_anchor, target_extent) = previous_geometry(&target_positions)?;
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

fn fixture(n: usize, profile: usize) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| {
            let j = i as u32;
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

// This is the sole intended representation change: derived zero anchors
// become +0. Nonzero anchors, f64 extents and uploaded point bits stay exact.
fn canonical_bits(v: f32) -> u32 {
    if v == 0.0 {
        0
    } else {
        v.to_bits()
    }
}

fn compare(points: &[[f32; 3]], workers: usize) {
    match (previous_geometry(points), prepare_with_workers(points, workers)) {
        (Ok((a, e)), Ok((b, f))) => {
            assert_eq!(a.map(canonical_bits), b.map(f32::to_bits));
            assert_eq!(e.to_bits(), f.to_bits());
        }
        (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
        _ => panic!("geometry validation differs"),
    }
}

#[test]
fn geometry_matches_frozen_bits_at_worker_boundaries_and_concurrently() {
    let mut points_checked = 0;
    for n in [
        0, 1, 3, 257, 16_383, 16_384, 32_767, 32_768, 32_769, 65_537, 131_071, 131_072, 131_073,
        262_143, 262_144, 262_145, 524_289,
    ] {
        for profile in 0..8 {
            let points = fixture(n, profile);
            for workers in [0, 1, 2, 6] {
                compare(&points, workers);
                points_checked += n;
            }
            let old = previous_geometry(&points).unwrap();
            let new = prepare(&points).unwrap();
            assert_eq!(old.0.map(canonical_bits), new.0.map(f32::to_bits));
            assert_eq!(old.1.to_bits(), new.1.to_bits());
            points_checked += n;
        }
    }
    let points = fixture(262_145, 2);
    std::thread::scope(|scope| {
        for workers in [1, 2, 6, 12] {
            let points = &points;
            scope.spawn(move || compare(points, workers));
        }
    });
    for len in [0, 65_536, 131_071, 131_072, 262_143, 262_144, 393_216, usize::MAX] {
        for available in [0, 1, 2, 6, 12, usize::MAX] {
            assert_eq!(
                worker_count(len, available),
                if len < 262_144 { 1 } else { available.max(1).min((len / 65_536).max(1)) }
            );
        }
    }
    println!("exact_geometry_points={points_checked} concurrent_points={}", 4 * points.len());
}

#[test]
fn signed_zero_patterns_analytic_bounds_and_invalid_rows_are_exact() {
    // Every eight-point sign pattern repeated over actual parallel chunks.
    for mask in 0..256 {
        let points: Vec<_> = (0..262_145)
            .map(|i| {
                let z = f32::from_bits(((mask >> (i % 8)) & 1) << 31);
                [z, -z, (i % 17) as f32]
            })
            .collect();
        compare(&points, 2);
        compare(&points, 6);
    }
    for workers in [1, 2, 6] {
        let input = vec![[-0.0; 3]; 262_145];
        assert_eq!(prepare_with_workers(&input, workers).unwrap().0.map(f32::to_bits), [0; 4]);
        assert!(input.iter().flatten().all(|v| v.to_bits() == (-0.0_f32).to_bits()));
    }
    assert_eq!(
        prepare(&[[-f32::from_bits(1), 0.0, 0.0], [0.0; 3]]).unwrap().0.map(f32::to_bits),
        [0; 4]
    );
    let points = [[-2.0, -4.0, -8.0], [6.0, 8.0, 10.0]];
    let (anchor, extent) = prepare(&points).unwrap();
    assert_eq!(anchor, [2.0, 2.0, 1.0, 0.0]);
    assert_eq!(extent, 8.0_f64.powi(2) + 12.0_f64.powi(2) + 18.0_f64.powi(2));
    for axis in 0..3 {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for index in [0, 65_536, 65_537, 131_072, 131_073, 262_144] {
                let mut points = fixture(262_145, 0);
                points[index][axis] = value;
                for workers in [1, 2, 6] {
                    compare(&points, workers);
                }
            }
        }
    }
    for value in [f32::MAX, -f32::MAX, 1e20, -1e20] {
        let mut points = fixture(262_145, 0);
        points[16_384][0] = value;
        for workers in [1, 2, 6] {
            compare(&points, workers);
        }
    }
    let points = vec![[f32::MAX; 3]; 262_145];
    compare(&points, 6);
    assert_eq!(
        prepare_with_workers(&points, 6).unwrap(),
        ([f32::MAX, f32::MAX, f32::MAX, 0.0], 0.0)
    );
}

fn tree(points: &[[f32; 3]]) -> KdTree {
    let columns: [Vec<_>; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).collect());
    KdTree::from_slices(&columns[0], &columns[1], &columns[2])
}

#[test]
#[ignore = "requires native GPU adapter"]
fn uploaded_moments_anchors_scales_receipts_and_owned_results_are_exact() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut cases = 0;
    let mut changed_zero_anchors = 0;
    for n in [
        0, 1, 32_767, 32_768, 32_769, 65_537, 131_071, 131_072, 131_073, 262_143, 262_144, 262_145,
        524_289,
    ] {
        for profile in [0, 1, 2, 3, 4, 7] {
            let points = fixture(n, profile);
            let target =
                tree(&fixture(if n == 65_537 && profile == 0 { 65_537 } else { 17 }, profile));
            let old = previous_upload(&runtime, &points, &target).unwrap();
            let new = GpuIcpInput::upload(&runtime, &points, &target).unwrap();
            assert_eq!(old.source_anchor.map(canonical_bits), new.source_anchor.map(f32::to_bits));
            assert_eq!(old.target_anchor.map(canonical_bits), new.target_anchor.map(f32::to_bits));
            assert_eq!(old.scale.to_bits(), new.scale.to_bits());
            assert_eq!(old.upload_transfers(), new.upload_transfers());
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
                let old = read_words(&runtime, &a.buffer);
                let new = read_words(&runtime, &b.buffer);
                // Status/count/reserved words and all 16 compensated terms
                // remain bit-exact. Only the two exposed frame anchors may
                // change the sign of a zero component; no epsilon is used.
                assert_eq!(&old[..4], &new[..4]);
                assert_eq!(&old[12..], &new[12..]);
                for (&a, &b) in old[4..12].iter().zip(&new[4..12]) {
                    assert_eq!(
                        canonical_bits(f32::from_bits(a)),
                        canonical_bits(f32::from_bits(b))
                    );
                    changed_zero_anchors += usize::from(a != b);
                }
                cases += 1;
            }
        }
    }
    let mut oversized = vec![[f32::MAX, 0.0, 0.0]; 131_073];
    oversized[0] = [0.0; 3];
    for points in [fixture(131_073, 6), oversized] {
        let target = tree(&[[0.0; 3]]);
        assert_eq!(
            previous_upload(&runtime, &points, &target).err().unwrap().to_string(),
            GpuIcpInput::upload(&runtime, &points, &target).err().unwrap().to_string()
        );
    }
    println!(
        "geometry_moment_cases={cases} words={} changed_zero_anchor_words={changed_zero_anchors}",
        cases * 44
    );
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

#[path = "icp_geometry_preparation_benchmark.rs"]
mod benchmark;
