//! Exact varying-nibble summaries of already-clamped CPU sort entries.

use super::VoxelSortEntry;

const ALL_DIGITS: u32 = (1 << 24) - 1;
const PROBE_POINTS: usize = 256;
const PARALLEL_MIN_POINTS: usize = 524_288;
const POINTS_PER_WORKER: usize = 131_072;
const MAX_WORKERS: usize = 4;

pub(super) fn plan(entries: &[VoxelSortEntry]) -> u32 {
    let Some(first) = entries.first() else { return 0 };
    let prefix = differences(&entries[..entries.len().min(PROBE_POINTS)], first);
    let mask = digits(prefix);
    // Full-width inputs usually establish all varying digits in this bounded
    // probe, avoiding an extra full scan and CPU-count query for that case.
    if mask == ALL_DIGITS || entries.len() <= PROBE_POINTS {
        return even(mask);
    }
    #[cfg(not(target_arch = "wasm32"))]
    let available = if entries.len() >= PARALLEL_MIN_POINTS {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        1
    };
    #[cfg(target_arch = "wasm32")]
    let available = 1;
    plan_with_workers(entries, available)
}

fn workers(len: usize, available: usize) -> usize {
    if cfg!(target_arch = "wasm32") || len < PARALLEL_MIN_POINTS {
        1
    } else {
        available.clamp(1, MAX_WORKERS).min((len / POINTS_PER_WORKER).max(1))
    }
}

fn plan_with_workers(entries: &[VoxelSortEntry], available: usize) -> u32 {
    let Some(first) = entries.first() else { return 0 };
    let count = workers(entries.len(), available);
    #[cfg(not(target_arch = "wasm32"))]
    if count > 1 {
        let chunk = entries.len().div_ceil(count);
        let result = std::thread::scope(|scope| {
            let mut bands = entries.chunks(chunk);
            let caller = bands.next().expect("nonempty parallel digit summary");
            let handles: Vec<_> =
                bands.map(|band| scope.spawn(move || differences(band, first))).collect();
            let mut result = differences(caller, first);
            for handle in handles {
                let other = handle.join().expect("voxel digit summary worker panicked");
                for axis in 0..3 {
                    result[axis] |= other[axis];
                }
            }
            result
        });
        return even(digits(result));
    }
    let _ = count;
    even(digits(differences(entries, first)))
}

fn differences(entries: &[VoxelSortEntry], first: &VoxelSortEntry) -> [u32; 3] {
    let mut result = [0; 3];
    for entry in entries {
        result[0] |= (entry.ix as u32) ^ (first.ix as u32);
        result[1] |= (entry.iy as u32) ^ (first.iy as u32);
        result[2] |= (entry.iz as u32) ^ (first.iz as u32);
    }
    result
}

fn digits(differences: [u32; 3]) -> u32 {
    let mut mask = 0;
    for (axis, value) in differences.into_iter().enumerate() {
        for nibble in 0..8 {
            if (value >> (4 * nibble)) & 15 != 0 {
                mask |= 1 << ((2 - axis) * 8 + nibble);
            }
        }
    }
    mask
}

fn even(mask: u32) -> u32 {
    if mask.count_ones() % 2 == 0 {
        mask
    } else {
        // At most 23 varying digits means at least one constant digit exists.
        mask | (1 << ((!mask & ALL_DIGITS).trailing_zeros()))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_radix_digits_tests.rs"]
mod tests;
