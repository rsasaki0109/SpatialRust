use super::*;
use crate::kernels::voxel_segments::{build_voxel_segments, VoxelSegments};
use crate::kernels::voxel_sort::{
    build_voxel_segments_gpu, build_voxel_segments_gpu_from_keys,
    build_voxel_segments_gpu_from_keys_buffer,
};
use wgpu::util::DeviceExt;

#[path = "voxel_metadata_readback_reference.rs"]
mod parent;

const POISON: u32 = 0xdead_beef;

fn buffer(r: &WgpuRuntime, words: &[u32]) -> wgpu::Buffer {
    r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn parent_segments(r: &WgpuRuntime, segments: &GpuVoxelSegments) -> VoxelSegments {
    if segments.cell_count() == 0 {
        return VoxelSegments {
            keys: vec![],
            point_indices: vec![],
            cell_starts: vec![],
            cell_counts: vec![],
        };
    }
    let (keys, starts, counts, indices) =
        parent::read_segment_metadata(r, segments, segments.cell_count() as usize).unwrap();
    parent::finalize_segments_from_readback(keys, indices, starts, counts)
}

fn fixture(r: &WgpuRuntime, cells: usize, points: usize) -> (GpuVoxelSegments, VoxelSegments) {
    assert!(cells > 0 && cells <= points);
    let keys: Vec<_> = (0..cells)
        .map(|cell| (i64::from(i32::MIN) + cell as i64, i64::from(i32::MAX) - cell as i64, -1))
        .collect();
    let starts: Vec<u32> = (0..cells).map(|cell| (cell * points / cells) as u32).collect();
    let counts: Vec<u32> = (0..cells)
        .map(|cell| {
            let end = if cell + 1 == cells { points as u32 } else { starts[cell + 1] };
            end - starts[cell]
        })
        .collect();
    let indices: Vec<_> = (0..points as u32).rev().collect();
    let mut key_words: Vec<_> =
        keys.iter().flat_map(|&(x, y, z)| [x as u32, y as u32, z as u32, POISON]).collect();
    key_words.extend([POISON; 12]);
    let mut start_words = starts.clone();
    start_words.extend([POISON; 7]);
    let mut index_words = indices.clone();
    index_words.extend([POISON; 9]);
    let segments = GpuVoxelSegments::new(
        cells as u32,
        points as u32,
        buffer(r, &key_words),
        buffer(r, &index_words),
        buffer(r, &start_words),
        4,
    );
    (
        segments,
        VoxelSegments { keys, point_indices: indices, cell_starts: starts, cell_counts: counts },
    )
}

fn assert_raw(r: &WgpuRuntime, segments: &GpuVoxelSegments, expected: &VoxelSegments) {
    let cells = segments.cell_count() as usize;
    let (keys, starts, indices) = try_read(r, segments, cells).unwrap().unwrap();
    assert_eq!(keys, expected.keys);
    assert_eq!(starts, expected.cell_starts);
    assert_eq!(indices, expected.point_indices);
    let old = parent::read_segment_metadata(r, segments, cells).unwrap();
    assert_eq!((keys, starts, expected.cell_counts.clone(), indices), old);
}

#[test]
fn metadata_layout_checks_exact_ranges_caps_alignment_and_overflow() {
    assert_eq!(std::mem::size_of::<VoxelKeyOutput>(), 16);
    let mut checked = 0;
    for points in [1usize, 2, 3, 255, 256, 257, 65_537, 1_000_003, 16_777_216] {
        for cells in [1, points.div_ceil(4), points] {
            let bytes = cells as u64 * 20 + points as u64 * 4;
            let plan = layout(cells, points, u64::MAX);
            if bytes > MAX_BATCH_BYTES {
                assert!(plan.is_none());
                continue;
            }
            let plan = plan.unwrap();
            assert_eq!(
                plan,
                Layout {
                    starts_offset: cells as u64 * 16,
                    indices_offset: cells as u64 * 20,
                    bytes,
                }
            );
            assert!(plan.starts_offset < plan.indices_offset && plan.indices_offset < plan.bytes);
            for offset in [plan.starts_offset, plan.indices_offset, plan.bytes] {
                assert_eq!(offset % wgpu::COPY_BUFFER_ALIGNMENT, 0);
            }
            assert_eq!(layout(cells, points, bytes), Some(plan));
            assert!(layout(cells, points, bytes - 1).is_none());
            let parent_peak = (cells as u64 * 16).max(points as u64 * 4);
            assert!(bytes >= parent_peak && bytes * 4 <= parent_peak * 9);
            checked += 1;
        }
    }
    // The internal starts plane can end at 4 mod 8. Only the entire map starts
    // at MAP_ALIGNMENT; copy offsets and lengths still obey their own alignment.
    assert_eq!(layout(1, 1, 24).unwrap().indices_offset % wgpu::MAP_ALIGNMENT, 4);
    assert_eq!(
        layout(1, (MAX_BATCH_BYTES as usize - 20) / 4, u64::MAX).unwrap().bytes,
        MAX_BATCH_BYTES
    );
    assert!(layout(1, (MAX_BATCH_BYTES as usize - 20) / 4 + 1, u64::MAX).is_none());
    for (cells, points) in
        [(0, 0), (0, 1), (1, 0), (2, 1), (1, usize::MAX), (usize::MAX, usize::MAX)]
    {
        assert!(layout(cells, points, u64::MAX).is_none());
    }
    println!("VOXEL_METADATA_LAYOUT verified {checked} exact nonoverlapping ranges, copy/map alignments, device/cap boundaries, arithmetic failures and source staging peak up to 2.25x parent.");
}

#[test]
fn metadata_gpu_raw_planes_match_parent_with_signed_keys_padding_and_partial_cells() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for points in [1usize, 2, 3, 255, 256, 257, 65_535, 65_536, 65_537, 1_000_003] {
        for cells in [1, points.div_ceil(4), points] {
            let (segments, expected) = fixture(&r, cells, points);
            assert_raw(&r, &segments, &expected);
            assert_eq!(segments.construction_readback_bytes(), 4);
            let mut finalized = expected.clone();
            for cell in 0..cells {
                let start = finalized.cell_starts[cell] as usize;
                let end = start + finalized.cell_counts[cell] as usize;
                finalized.point_indices[start..end].reverse();
            }
            assert_eq!(segments.to_voxel_segments(&r).unwrap(), finalized);
            checked += 1;
        }
    }
    println!("VOXEL_METADATA_RAW verified {checked} padded GPU fixtures, three exact planes, independent counts, ignored key padding, signed endpoints, parent and final per-cell ordering.");
}

fn input_keys(n: usize, pattern: usize) -> Vec<(i64, i64, i64)> {
    (0..n)
        .map(|i| match pattern {
            0 => (i as i64 * 17 % 31 - 15, i as i64 * 13 % 7 - 3, -1),
            1 => (i64::MIN, i64::MAX, -1),
            _ => (n as i64 - i as i64, i64::from(i32::MIN), i64::from(i32::MAX)),
        })
        .collect()
}

fn clamped(keys: &[(i64, i64, i64)]) -> Vec<(i64, i64, i64)> {
    let bound = |x: i64| x.clamp(i64::from(i32::MIN), i64::from(i32::MAX));
    keys.iter().map(|&(x, y, z)| (bound(x), bound(y), bound(z))).collect()
}

#[test]
fn metadata_public_cpu_and_gpu_key_routes_keep_exact_outputs_and_receipts() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for n in [0, 1, 2, 255, 256, 257, 65_537] {
        for pattern in 0..3 {
            let keys = input_keys(n, pattern);
            let normalized = clamped(&keys);
            let expected = build_voxel_segments(&normalized);
            let segments = build_voxel_segments_gpu_from_keys(&r, &keys).unwrap();
            assert_eq!(segments.construction_readback_bytes(), if n == 0 { 0 } else { 4 });
            assert_eq!(segments.to_voxel_segments(&r).unwrap(), expected);
            assert_eq!(parent_segments(&r, &segments), expected);
            assert_eq!(build_voxel_segments_gpu(&r, &keys).unwrap(), expected);
            let words: Vec<_> = normalized
                .iter()
                .flat_map(|&(x, y, z)| [x as u32, y as u32, z as u32, POISON])
                .collect();
            let source = buffer(&r, if words.is_empty() { &[POISON; 4] } else { &words });
            let gpu =
                build_voxel_segments_gpu_from_keys_buffer(&r, &source, n as u32, n as u32).unwrap();
            assert_eq!(gpu.to_voxel_segments(&r).unwrap(), expected);
            assert_eq!(parent_segments(&r, &gpu), expected);
            checked += 1;
        }
    }
    println!("VOXEL_METADATA_PUBLIC verified {checked} complete CPU-key and GPU-key builds, empty/singleton, clamp endpoints, duplicates, dispatch boundaries and unchanged construction scalar receipts.");
}

#[test]
fn metadata_large_payload_uses_serial_fallback_and_matches_parent() {
    let r = WgpuRuntime::new_headless().unwrap();
    let n = (MAX_BATCH_BYTES / 24) as usize + 1;
    let (segments, expected) = fixture(&r, n, n);
    assert!(try_read(&r, &segments, n).is_none());
    // Singleton cells already have final index order.
    let actual = segments.to_voxel_segments(&r).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual, parent_segments(&r, &segments));
    assert_eq!(segments.construction_readback_bytes(), 4);
    println!("VOXEL_METADATA_FALLBACK verified actual serial readback at {n} cells: combined payload exceeds 64MiB while each original plane fits, all output bytes match parent.");
}

#[test]
fn metadata_concurrent_readers_preserve_sources_and_live_results_after_pool_clear() {
    let r = WgpuRuntime::new_headless().unwrap();
    let (segments, expected) = fixture(&r, 257, 65_537);
    let retained = segments.to_voxel_segments(&r).unwrap();
    let barrier = std::sync::Barrier::new(4);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let (r, segments, expected, barrier) = (&r, &segments, &expected, &barrier);
            scope.spawn(move || {
                barrier.wait();
                for _ in 0..3 {
                    assert_raw(r, segments, expected);
                }
            });
        }
    });
    r.clear_buffer_pool();
    r.clear_scratch_buffer_pool();
    r.set_scratch_buffer_pool_budget(0);
    assert_raw(&r, &segments, &expected);
    assert_eq!(retained, segments.to_voxel_segments(&r).unwrap());
    assert_eq!(retained, parent_segments(&r, &segments));
    println!("VOXEL_METADATA_OWNERSHIP verified twelve concurrent reads on shared live source buffers, repeated maps, owned retained results and zero-budget/pool-clear independence.");
}

#[path = "voxel_metadata_readback_benchmark.rs"]
mod benchmark;
