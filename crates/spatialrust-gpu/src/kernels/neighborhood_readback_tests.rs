use super::super::{
    invalid, Arc, GpuKdTree, GpuNeighborhoodReadback, GpuNeighborhoods, GpuQueryBatch,
};
use super::*;
use crate::WgpuRuntime;
use spatialrust_core::{TransferDirection, TransferStats};
use spatialrust_search::{KdTree, NearestNeighborIndex, RadiusSearchIndex};
use wgpu::util::DeviceExt;

#[path = "neighborhood_readback_reference.rs"]
mod parent;

const BITS: [u32; 12] = [
    0,
    0x8000_0000,
    1,
    0x8000_0001,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_1234,
    0xffc0_5678,
    0x7f80_1234,
    0xff80_5678,
];

fn packed(n: usize, capacity: u32, stored: u32, kind: GpuNeighborhoodKind) -> Vec<u32> {
    let stride = 2 + 2 * capacity as usize;
    let mut words = vec![0xdead_beef; n * stride];
    for (i, row) in words.chunks_exact_mut(stride).enumerate() {
        row[0] = stored;
        row[1] = stored + if kind == GpuNeighborhoodKind::Radius { (i % 5) as u32 } else { 0 };
        for j in 0..stored as usize {
            row[2 + 2 * j] = match j % 5 {
                0 => u32::MAX,
                1 => 0,
                _ => (i as u32).wrapping_mul(0x9e37_79b9).wrapping_add(j as u32),
            };
            row[3 + 2 * j] = BITS[(i + j) % BITS.len()];
        }
    }
    words
}

fn distribution(words: &mut [u32], capacity: u32, pattern: usize) {
    let stride = 2 + 2 * capacity as usize;
    let n = words.len() / stride;
    for (i, row) in words.chunks_exact_mut(stride).enumerate() {
        match pattern {
            1 => row[0] = 8 + (i % 9) as u32,
            2 => row[0] = if i < n / 4 { capacity } else { 0 },
            _ => continue,
        }
        row[1] = row[0] + (i % 5) as u32;
    }
}

fn check_packed(words: &[u32], stride: usize, result: &Decoded) {
    assert_eq!(result.0.len(), words.len() / stride);
    assert_eq!(result.1.len(), result.0.len());
    for ((source, row), total) in words.chunks_exact(stride).zip(&result.0).zip(&result.1) {
        assert_eq!(*total, source[1]);
        assert_eq!(row.len(), source[0] as usize);
        for (j, neighbor) in row.iter().enumerate() {
            assert_eq!(neighbor.index, source[2 + 2 * j] as usize);
            assert_eq!(neighbor.distance_squared.to_bits(), source[3 + 2 * j]);
        }
    }
}

fn compare(a: &GpuNeighborhoodReadback, b: &GpuNeighborhoodReadback) {
    assert_eq!(a.totals, b.totals);
    assert_eq!(a.transfers, b.transfers);
    assert_eq!(a.is_complete(), b.is_complete());
    assert_eq!(a.rows.len(), b.rows.len());
    for (a, b) in a.rows.iter().zip(&b.rows) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert_eq!(a.index, b.index);
            assert_eq!(a.distance_squared.to_bits(), b.distance_squared.to_bits());
        }
    }
    assert_eq!(
        a.complete_rows().err().map(|e| e.to_string()),
        b.complete_rows().err().map(|e| e.to_string())
    );
}

fn gpu_words(
    r: &WgpuRuntime,
    words: &[u32],
    capacity: u32,
    kind: GpuNeighborhoodKind,
) -> GpuNeighborhoods {
    GpuNeighborhoods {
        buffer: r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("synthetic-neighborhood-readback"),
            contents: bytemuck::cast_slice(words),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        }),
        queries: (words.len() / (2 + 2 * capacity as usize)) as u32,
        capacity,
        bytes: std::mem::size_of_val(words) as u64,
        kind,
        identity: r.profiling_identity(),
    }
}

#[test]
fn decode_policy_bounds_probe_workers_and_serial_controls() {
    for n in [0, 1, MIN_ROWS - 1, MIN_ROWS, MIN_ROWS + 1, MIN_NEIGHBORS - 1, MIN_NEIGHBORS] {
        for (capacity, stored) in [(0, 0), (1, 1), (8, 0), (8, 1), (8, 8)] {
            let words = packed(n, capacity, stored, GpuNeighborhoodKind::NearestK);
            let expected =
                n >= MIN_ROWS && stored != 0 && stored as usize >= MIN_NEIGHBORS.div_ceil(n.max(1));
            assert_eq!(eligible(&words, 2 + 2 * capacity as usize, capacity), expected);
        }
    }
    for n in [0, 1, MIN_ROWS - 1, MIN_ROWS, MIN_ROWS + 1, usize::MAX] {
        for available in [0, 1, 2, 3, 4, 12, usize::MAX] {
            let count = workers(n, available);
            assert!((1..=4).contains(&count));
            assert!(count <= available.max(1));
            assert!(n == 0 || n.div_ceil(count) > 0);
        }
    }
    // Unsampled corruption can affect policy without bypassing validation.
    let mut words = packed(MIN_ROWS + 1, 8, 8, GpuNeighborhoodKind::NearestK);
    words[17 * 18] = 9;
    assert!(eligible(&words, 18, 8));
    assert!(decode(&words, 18, 8, GpuNeighborhoodKind::NearestK).is_err());
    for (capacity, pattern, expected) in [(16, 1, true), (64, 2, false)] {
        let mut words = packed(MIN_ROWS + 1, capacity, capacity, GpuNeighborhoodKind::Radius);
        distribution(&mut words, capacity, pattern);
        let stride = 2 + 2 * capacity as usize;
        assert_eq!(eligible(&words, stride, capacity), expected);
        for available in [1, 2, 4] {
            check_packed(
                &words,
                stride,
                &decode_with_workers(
                    &words,
                    stride,
                    capacity,
                    GpuNeighborhoodKind::Radius,
                    available,
                )
                .unwrap(),
            );
        }
        check_packed(
            &words,
            stride,
            &decode(&words, stride, capacity, GpuNeighborhoodKind::Radius).unwrap(),
        );
    }
}

#[test]
fn decode_preserves_all_bits_padding_counts_order_and_worker_bands() {
    for (n, capacity, stored) in [
        (0, 0, 0),
        (1, 0, 0),
        (65, 1, 1),
        (1025, 65, 3),
        (MIN_ROWS - 1, 8, 8),
        (MIN_ROWS, 8, 8),
        (MIN_ROWS + 1, 8, 8),
        (MIN_ROWS + 3, 65, 9),
        (MIN_NEIGHBORS + 1, 1, 1),
    ] {
        for kind in [GpuNeighborhoodKind::NearestK, GpuNeighborhoodKind::Radius] {
            let words = packed(n, capacity, stored, kind);
            let stride = 2 + 2 * capacity as usize;
            check_packed(&words, stride, &decode(&words, stride, capacity, kind).unwrap());
            for available in [1, 2, 3, 4, 99] {
                check_packed(
                    &words,
                    stride,
                    &decode_with_workers(&words, stride, capacity, kind, available).unwrap(),
                );
            }
        }
    }
}

#[test]
fn invalid_headers_in_every_band_fail_with_original_error_and_release_maps() {
    let r = WgpuRuntime::new_headless().unwrap();
    let n = MIN_ROWS + 1;
    let mut words = packed(n, 8, 8, GpuNeighborhoodKind::NearestK);
    let gpu = gpu_words(&r, &words, 8, GpuNeighborhoodKind::NearestK);
    for index in [0, n.div_ceil(4) - 1, n.div_ceil(4), n / 2 + 1, 3 * n / 4 + 1, n - 1] {
        for header in [[9, 9], [8, 7], [7, 8]] {
            words[index * 18..index * 18 + 2].copy_from_slice(&header);
            let original = parent::download;
            for available in [1, 2, 4] {
                let error =
                    decode_with_workers(&words, 18, 8, GpuNeighborhoodKind::NearestK, available)
                        .err()
                        .unwrap();
                assert!(error.to_string().contains("GPU neighborhood metadata is invalid"));
            }
            r.queue().write_buffer(
                &gpu.buffer,
                (index * 18 * 4) as u64,
                bytemuck::cast_slice(&header),
            );
            for _ in 0..2 {
                assert_eq!(
                    gpu.download(&r).err().unwrap().to_string(),
                    original(&gpu, &r).err().unwrap().to_string()
                );
            }
            words[index * 18..index * 18 + 2].copy_from_slice(&[8, 8]);
            r.queue().write_buffer(
                &gpu.buffer,
                (index * 18 * 4) as u64,
                bytemuck::cast_slice(&[8u32, 8]),
            );
            compare(&gpu.download(&r).unwrap(), &parent::download(&gpu, &r).unwrap());
        }
    }
}

#[test]
fn explicit_maps_match_frozen_parent_preserve_receipts_and_concurrent_ownership() {
    let r = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    for (capacity, stored, kind) in [
        (8, 8, GpuNeighborhoodKind::NearestK),
        (65, 9, GpuNeighborhoodKind::Radius),
        (0, 0, GpuNeighborhoodKind::Radius),
    ] {
        let words = packed(MIN_ROWS + 1, capacity, stored, kind);
        let gpu = gpu_words(&r, &words, capacity, kind);
        let expected = parent::download(&gpu, &r).unwrap();
        assert!(gpu.download(&other).is_err());
        std::thread::scope(|scope| {
            for _ in 0..3 {
                let (r, gpu, expected) = (&r, &gpu, &expected);
                scope.spawn(move || {
                    let result = gpu.download(r).unwrap();
                    compare(&result, expected);
                    assert_eq!(result.transfers.device_to_host_bytes(), gpu.bytes);
                    assert_eq!(result.transfers.host_to_device_bytes(), 0);
                    assert_eq!(result.transfers.device_to_device_bytes(), 0);
                });
            }
        });
        let owned = gpu.download(&r).unwrap();
        drop(gpu);
        drop(words);
        r.clear_buffer_pool();
        compare(&owned, &expected);
    }
}

fn tree_fixture() -> (KdTree, Vec<[f32; 3]>) {
    let points: Vec<_> =
        (0..17).map(|i| [i as f32 * 0.5, ((i * 7) % 11) as f32 * 0.5, 0.]).collect();
    let axes: [Vec<_>; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).collect());
    (KdTree::from_slices(&axes[0], &axes[1], &axes[2]), points)
}

#[test]
fn searched_rows_match_cpu_at_parallel_cutoff_with_overflow_and_empty_queries() {
    let r = WgpuRuntime::new_headless().unwrap();
    let (tree, points) = tree_fixture();
    let target = GpuKdTree::upload(&r, &tree).unwrap();
    for n in [0, 65, MIN_ROWS - 1, MIN_ROWS, MIN_ROWS + 1] {
        let queries: Vec<_> = (0..n).map(|i| points[i % points.len()]).collect();
        let batch = GpuQueryBatch::upload(&r, &queries).unwrap();
        for radius in [false, true] {
            let output = if radius {
                target.radius_neighbors_resident(&r, &batch, f32::INFINITY, 8)
            } else {
                target.nearest_k_resident(&r, &batch, 8)
            }
            .unwrap();
            let readback = output.download(&r).unwrap();
            compare(&readback, &parent::download(&output, &r).unwrap());
            let expected: Vec<_> = points
                .iter()
                .map(|&p| {
                    if radius {
                        let mut row = tree.radius_search(p[0], p[1], p[2], f32::INFINITY);
                        row.truncate(8);
                        row
                    } else {
                        tree.nearest_k(p[0], p[1], p[2], 8)
                    }
                })
                .collect();
            for (i, row) in readback.rows.iter().enumerate() {
                assert_eq!(row, &expected[i % points.len()]);
                assert_eq!(readback.totals[i], if radius { 17 } else { 8 });
            }
            assert_eq!(
                readback.transfers.device_to_host_bytes(),
                if n == 0 { 0 } else { n as u64 * 72 }
            );
            assert_eq!(readback.is_complete(), !radius || n == 0);
            assert_eq!(readback.into_complete_rows().is_ok(), !radius || n == 0);
        }
    }
}

#[path = "neighborhood_readback_benchmark.rs"]
mod benchmark;
