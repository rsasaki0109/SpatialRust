//! Bounded native CPU conversion of keys borrowed from a mapped GPU readback.

use super::VoxelKeyOutput;

const MIN_PARALLEL_CELLS: usize = 524_288;
const CELLS_PER_WORKER: usize = 131_072;
const MAX_WORKERS: usize = 4;

pub(super) fn decode(keys: &[VoxelKeyOutput]) -> Vec<(i64, i64, i64)> {
    // Small and WASM calls retain the original collector and avoid CPU queries.
    if keys.len() < MIN_PARALLEL_CELLS {
        return serial(keys);
    }
    #[cfg(not(target_arch = "wasm32"))]
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    #[cfg(target_arch = "wasm32")]
    let available = 1;
    decode_with_workers(keys, available)
}

fn workers(len: usize, available: usize) -> usize {
    if cfg!(target_arch = "wasm32") || len < MIN_PARALLEL_CELLS {
        1
    } else {
        available.clamp(1, MAX_WORKERS).min((len / CELLS_PER_WORKER).max(1))
    }
}

fn decode_with_workers(keys: &[VoxelKeyOutput], available: usize) -> Vec<(i64, i64, i64)> {
    let count = workers(keys.len(), available);
    #[cfg(not(target_arch = "wasm32"))]
    if count > 1 {
        // One safely initialized final vector. Initialization adds host writes;
        // there are no per-worker key vectors or point-sized intermediate copies.
        let mut output = vec![(0i64, 0i64, 0i64); keys.len()];
        let size = keys.len().div_ceil(count);
        std::thread::scope(|scope| {
            let mut bands = output.chunks_mut(size);
            let first = bands.next().expect("parallel key input is nonempty");
            for (index, rows) in bands.enumerate() {
                let base = (index + 1) * size;
                let source = &keys[base..base + rows.len()];
                scope.spawn(move || fill(source, rows));
            }
            // Include the caller in the bound; all borrowed maps outlive this scope.
            fill(&keys[..first.len()], first);
        });
        return output;
    }
    let _ = count;
    serial(keys)
}

#[cfg(not(target_arch = "wasm32"))]
fn fill(keys: &[VoxelKeyOutput], output: &mut [(i64, i64, i64)]) {
    for (key, row) in keys.iter().zip(output) {
        *row = (i64::from(key.ix), i64::from(key.iy), i64::from(key.iz));
    }
}

fn serial(keys: &[VoxelKeyOutput]) -> Vec<(i64, i64, i64)> {
    keys.iter().map(|key| (i64::from(key.ix), i64::from(key.iy), i64::from(key.iz))).collect()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_metadata_keys_tests.rs"]
mod tests;
