use super::*;
use crate::kernels::gpu_segments::GpuVoxelSegments;
use crate::kernels::voxel_segments::build_voxel_segments;
use crate::kernels::voxel_sort::{
    build_voxel_segments_gpu, build_voxel_segments_gpu_from_keys,
    build_voxel_segments_gpu_from_keys_buffer,
};
use crate::{VoxelSegments, WgpuRuntime};
use wgpu::util::DeviceExt;

#[path = "voxel_metadata_keys_compact_reference.rs"]
mod parent_compact;
#[path = "voxel_metadata_keys_readback_reference.rs"]
mod parent_readback;

const POISON: u32 = 0xdead_beef;

fn parent_decode(keys: &[VoxelKeyOutput]) -> Vec<(i64, i64, i64)> {
    keys.iter().map(|key| (i64::from(key.ix), i64::from(key.iy), i64::from(key.iz))).collect()
}

fn raw_keys(n: usize) -> Vec<VoxelKeyOutput> {
    const SPECIAL: [i32; 6] = [i32::MIN, i32::MAX, -1, 0, 1, -65_537];
    (0..n)
        .map(|i| VoxelKeyOutput {
            ix: SPECIAL[i % SPECIAL.len()],
            iy: (i as u32).wrapping_mul(0x9e37_79b9) as i32,
            iz: SPECIAL[(i + 3) % SPECIAL.len()],
            _pad: POISON as i32,
        })
        .collect()
}

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
        parent_compact::read_segment_metadata(r, segments, segments.cell_count() as usize).unwrap();
    parent_compact::finalize_segments_from_readback(keys, indices, starts, counts)
}

fn fixture(r: &WgpuRuntime, cells: usize, points: usize) -> (GpuVoxelSegments, VoxelSegments) {
    let keys: Vec<_> = (0..cells)
        .map(|i| (i64::from(i32::MIN) + i as i64, i64::from(i32::MAX) - i as i64, -1))
        .collect();
    let starts: Vec<_> = (0..cells).map(|i| (i * points / cells) as u32).collect();
    let counts: Vec<_> = (0..cells)
        .map(|i| (if i + 1 == cells { points as u32 } else { starts[i + 1] }) - starts[i])
        .collect();
    let mut indices: Vec<_> = (0..points as u32).rev().collect();
    let mut key_words: Vec<_> =
        keys.iter().flat_map(|&(x, y, z)| [x as u32, y as u32, z as u32, POISON]).collect();
    key_words.extend([POISON; 12]);
    let mut start_words = starts.clone();
    start_words.extend([POISON; 7]);
    let mut index_words = indices.clone();
    index_words.extend([POISON; 9]);
    let gpu = GpuVoxelSegments::new(
        cells as u32,
        points as u32,
        buffer(r, &key_words),
        buffer(r, &index_words),
        buffer(r, &start_words),
        4,
    );
    for i in 0..cells {
        let start = starts[i] as usize;
        let end = start + counts[i] as usize;
        indices[start..end].reverse();
    }
    (gpu, VoxelSegments { keys, point_indices: indices, cell_starts: starts, cell_counts: counts })
}

#[test]
fn voxel_metadata_decode_policy_bounds_workers_bands_and_small_fallbacks() {
    let mut checked = 0;
    for len in [
        0,
        1,
        65_537,
        MIN_PARALLEL_CELLS - 1,
        MIN_PARALLEL_CELLS,
        MIN_PARALLEL_CELLS + 1,
        1_000_003,
        2_000_003,
        usize::MAX,
    ] {
        for available in [0, 1, 2, 3, 4, 12, usize::MAX] {
            let count = workers(len, available);
            assert!((1..=4).contains(&count));
            assert!(count <= available.max(1));
            if len < MIN_PARALLEL_CELLS || available < 2 {
                assert_eq!(count, 1);
            } else {
                assert_eq!(count, available.min(4));
                let size = len.div_ceil(count);
                let actual = len.div_ceil(size);
                assert_eq!(actual, count);
                assert!(len - (actual - 1) * size > 0);
            }
            checked += 1;
        }
    }
    println!("VOXEL_DECODE_POLICY verified {checked} threshold/availability/maximum extents, bounded nonempty bands, caller participation and small/one-worker fallbacks.");
}

#[test]
fn voxel_metadata_decode_workers_preserve_signed_words_order_guards_and_capacity() {
    let mut checked = 0;
    for n in [
        0,
        1,
        3,
        65_537,
        MIN_PARALLEL_CELLS - 1,
        MIN_PARALLEL_CELLS,
        MIN_PARALLEL_CELLS + 1,
        1_000_003,
        2_000_003,
    ] {
        let input = raw_keys(n + 18);
        let before = bytemuck::cast_slice::<_, u32>(&input).to_vec();
        let keys = &input[7..7 + n];
        let expected: Vec<_> =
            keys.iter().map(|v| (v.ix as i64, v.iy as i64, v.iz as i64)).collect();
        assert_eq!(parent_decode(keys), expected);
        for available in [0, 1, 2, 3, 4, 12] {
            let out = decode_with_workers(keys, available);
            assert_eq!(out, expected);
            assert_eq!(out.len(), n);
            assert_eq!(out.capacity(), n);
            checked += out.len() * 3;
        }
        assert_eq!(decode(keys), expected);
        assert_eq!(bytemuck::cast_slice::<_, u32>(&input), before);
    }
    println!("VOXEL_DECODE_WORDS verified {checked} independent signed coordinates across explicit 0/1/2/3/4/12 worker budgets, dirty padding/guards, exact capacity and original order.");
}

#[test]
fn voxel_metadata_decode_mapped_ranges_match_parent_counts_indices_and_cap_fallback() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for (c, n) in [
        (1, 1),
        (257, 65_537),
        (MIN_PARALLEL_CELLS - 1, MIN_PARALLEL_CELLS + 31),
        (MIN_PARALLEL_CELLS, MIN_PARALLEL_CELLS + 31),
        (MIN_PARALLEL_CELLS + 1, MIN_PARALLEL_CELLS + 37),
        (1_000_003, 1_000_003),
        (2_000_003, 2_000_003),
        (2_796_203, 2_796_203),
    ] {
        let (segments, expected) = fixture(&r, c, n);
        let current = segments.to_voxel_segments(&r).unwrap();
        assert_eq!(current, expected);
        assert_eq!(current, parent_segments(&r, &segments));
        assert_eq!(segments.construction_readback_bytes(), 4);
        assert_eq!(
            super::super::layout(c, n, r.device().limits().max_buffer_size).is_some(),
            c != 2_796_203
        );
        checked += 1;
    }
    println!("VOXEL_DECODE_MAPPED verified {checked} padded real GPU fixtures, threshold tails, parent/independent keys/counts/indices/final ordering and actual >64MiB serial fallback.");
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
fn voxel_metadata_decode_complete_cpu_device_key_calls_match_parent_and_cpu() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for n in
        [0, 1, 257, MIN_PARALLEL_CELLS - 1, MIN_PARALLEL_CELLS, MIN_PARALLEL_CELLS + 1, 1_000_003]
    {
        for pattern in 0..3 {
            let keys = input_keys(n, pattern);
            let normalized = clamped(&keys);
            let expected = build_voxel_segments(&normalized);
            let gpu = build_voxel_segments_gpu_from_keys(&r, &keys).unwrap();
            assert_eq!(gpu.to_voxel_segments(&r).unwrap(), expected);
            assert_eq!(parent_segments(&r, &gpu), expected);
            assert_eq!(build_voxel_segments_gpu(&r, &keys).unwrap(), expected);
            assert_eq!(gpu.construction_readback_bytes(), if n == 0 { 0 } else { 4 });
            let words: Vec<_> = normalized
                .iter()
                .flat_map(|&(x, y, z)| [x as u32, y as u32, z as u32, POISON])
                .collect();
            let source = buffer(&r, if words.is_empty() { &[POISON; 4] } else { &words });
            let borrowed =
                build_voxel_segments_gpu_from_keys_buffer(&r, &source, n as u32, n as u32).unwrap();
            assert_eq!(borrowed.to_voxel_segments(&r).unwrap(), expected);
            assert_eq!(parent_segments(&r, &borrowed), expected);
            checked += 1;
        }
    }
    println!("VOXEL_DECODE_PUBLIC verified {checked} complete CPU/device-key builds, actual occupied-cell policy, empty/singleton/clamping/duplicates/thresholds and unchanged scalar receipts.");
}

#[test]
fn voxel_metadata_decode_concurrent_mapped_borrows_keep_owned_results_after_clear() {
    let r = WgpuRuntime::new_headless().unwrap();
    let (segments, expected) = fixture(&r, MIN_PARALLEL_CELLS + 1, MIN_PARALLEL_CELLS + 37);
    let retained = segments.to_voxel_segments(&r).unwrap();
    let barrier = std::sync::Barrier::new(4);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let (r, segments, expected, barrier) = (&r, &segments, &expected, &barrier);
            scope.spawn(move || {
                barrier.wait();
                for _ in 0..2 {
                    assert_eq!(segments.to_voxel_segments(r).unwrap(), *expected);
                }
            });
        }
    });
    r.clear_buffer_pool();
    r.clear_scratch_buffer_pool();
    r.set_scratch_buffer_pool_budget(0);
    assert_eq!(retained, expected);
    assert_eq!(segments.to_voxel_segments(&r).unwrap(), retained);
    assert_eq!(parent_segments(&r, &segments), retained);
    println!("VOXEL_DECODE_OWNERSHIP verified eight concurrent parallel-key reads of live shared source buffers, scope-complete maps, retained host results and pool-clear/zero-budget independence.");
}

#[path = "voxel_metadata_keys_benchmark.rs"]
mod benchmark;
