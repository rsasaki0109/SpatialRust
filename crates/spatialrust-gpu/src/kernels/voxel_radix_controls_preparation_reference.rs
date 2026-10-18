//! Bounded, original-order host entries before an explicit GPU upload.

use super::VoxelSortEntry;

const PARALLEL_MIN_POINTS: usize = 524_288;
const POINTS_PER_WORKER: usize = 131_072;
const MAX_WORKERS: usize = 4;

pub(super) fn prepare(keys: &[(i64, i64, i64)]) -> Vec<VoxelSortEntry> {
    // Keep small calls independent of CPU-count queries and thread startup.
    if keys.len() < PARALLEL_MIN_POINTS {
        return serial(keys);
    }
    #[cfg(not(target_arch = "wasm32"))]
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    #[cfg(target_arch = "wasm32")]
    let available = 1;
    prepare_with_workers(keys, available)
}

fn workers(len: usize, available: usize) -> usize {
    if cfg!(target_arch = "wasm32") || len < PARALLEL_MIN_POINTS {
        1
    } else {
        available.clamp(1, MAX_WORKERS).min((len / POINTS_PER_WORKER).max(1))
    }
}

fn prepare_with_workers(keys: &[(i64, i64, i64)], available: usize) -> Vec<VoxelSortEntry> {
    let count = workers(keys.len(), available);
    #[cfg(not(target_arch = "wasm32"))]
    if count > 1 {
        // One exact-sized, safely initialized payload; no per-worker point vectors.
        let mut output = bytemuck::allocation::zeroed_vec(keys.len());
        let chunk = keys.len().div_ceil(count);
        std::thread::scope(|scope| {
            let mut bands = output.chunks_mut(chunk);
            let first = bands.next().expect("parallel voxel input is nonempty");
            for (index, rows) in bands.enumerate() {
                let base = (index + 1) * chunk;
                let source = &keys[base..base + rows.len()];
                scope.spawn(move || fill(source, rows, base));
            }
            // Count the caller as one worker, spawning at most MAX_WORKERS - 1.
            fill(&keys[..first.len()], first, 0);
        });
        return output;
    }
    let _ = count;
    serial(keys)
}

#[cfg(not(target_arch = "wasm32"))]
fn fill(keys: &[(i64, i64, i64)], output: &mut [VoxelSortEntry], base: usize) {
    // The public API checks u32/device limits before any preparation. Each
    // disjoint band adds its original global base to retain stable tie order.
    for (index, (&(ix, iy, iz), row)) in keys.iter().zip(output).enumerate() {
        *row = VoxelSortEntry {
            ix: ix.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            iy: iy.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            iz: iz.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            point_index: (base + index) as u32,
        };
    }
}

// Original exact-capacity collector for small, one-worker and WASM calls.
fn serial(keys: &[(i64, i64, i64)]) -> Vec<VoxelSortEntry> {
    keys.iter()
        .enumerate()
        .map(|(index, &(ix, iy, iz))| VoxelSortEntry {
            ix: ix.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            iy: iy.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            iz: iz.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            point_index: index as u32,
        })
        .collect()
}
