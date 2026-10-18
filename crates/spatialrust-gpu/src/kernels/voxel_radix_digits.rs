//! Exact varying-nibble summaries within each sign of clamped CPU sort entries.

use super::VoxelSortEntry;

const ALL_DIGITS: u32 = (1 << 24) - 1;
const PROBE_POINTS: usize = 256;
const PARALLEL_MIN_POINTS: usize = 524_288;
const POINTS_PER_WORKER: usize = 131_072;
const MAX_WORKERS: usize = 4;

#[derive(Clone, Copy)]
struct Class {
    all: u32,
    any: u32,
}

impl Default for Class {
    fn default() -> Self {
        Self { all: u32::MAX, any: 0 }
    }
}

impl Class {
    fn seen(self) -> bool {
        // A populated class has all bits contained in any; the empty identity
        // (MAX, 0) does not. AND/OR merging preserves both properties.
        self.all <= self.any
    }
}

type Summary = [[Class; 2]; 3];

pub(super) fn plan(entries: &[VoxelSortEntry]) -> u32 {
    let prefix = summarize(&entries[..entries.len().min(PROBE_POINTS)]);
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
    let global =
        prefix.map(|classes| (classes[0].any | classes[1].any) ^ (classes[0].all & classes[1].all));
    if varying_digits(global) == mask {
        let first = &entries[0];
        let differing = fold_with_workers(
            entries,
            available,
            |band| {
                let mut result = [0; 3];
                for entry in band {
                    result[0] |= (entry.ix as u32) ^ (first.ix as u32);
                    result[1] |= (entry.iy as u32) ^ (first.iy as u32);
                    result[2] |= (entry.iz as u32) ^ (first.iz as u32);
                }
                result
            },
            |result, other| {
                for axis in 0..3 {
                    result[axis] |= other[axis];
                }
            },
        );
        let complete = varying_digits(differing);
        // Prefix class digits are a lower bound; complete global digits are
        // an upper bound. Equal masks prove the complete class mask exactly.
        // The original one-sign proof also remains valid if later digits vary.
        if complete == mask || differing.iter().all(|value| value & 0x8000_0000 == 0) {
            return even(complete);
        }
    }
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
    if entries.is_empty() {
        return 0;
    }
    even(digits(fold_with_workers(entries, available, summarize, |result, other| {
        for axis in 0..3 {
            for sign in 0..2 {
                result[axis][sign].all &= other[axis][sign].all;
                result[axis][sign].any |= other[axis][sign].any;
            }
        }
    })))
}

fn fold_with_workers<S: Send>(
    entries: &[VoxelSortEntry],
    available: usize,
    summarize: impl Fn(&[VoxelSortEntry]) -> S + Sync,
    merge: impl Fn(&mut S, S),
) -> S {
    let count = workers(entries.len(), available);
    #[cfg(not(target_arch = "wasm32"))]
    if count > 1 {
        let chunk = entries.len().div_ceil(count);
        let result = std::thread::scope(|scope| {
            let mut bands = entries.chunks(chunk);
            let caller = bands.next().expect("nonempty parallel digit summary");
            let summarize = &summarize;
            let handles: Vec<_> = bands.map(|band| scope.spawn(move || summarize(band))).collect();
            let mut result = summarize(caller);
            for handle in handles {
                let other = handle.join().expect("voxel digit summary worker panicked");
                merge(&mut result, other);
            }
            result
        });
        return result;
    }
    let _ = (count, merge);
    summarize(entries)
}

fn summarize(entries: &[VoxelSortEntry]) -> Summary {
    let mut result = [[Class::default(); 2]; 3];
    for entry in entries {
        for (axis, value) in
            [entry.ix as u32, entry.iy as u32, entry.iz as u32].into_iter().enumerate()
        {
            let class = &mut result[axis][(value >> 31) as usize];
            class.all &= value;
            class.any |= value;
        }
    }
    result
}

fn digits(summary: Summary) -> u32 {
    let differences = summary
        .map(|classes| classes.iter().filter(|c| c.seen()).fold(0, |v, c| v | (c.any ^ c.all)));
    let mut mask = varying_digits(differences);
    for (axis, classes) in summary.into_iter().enumerate() {
        // The final digit separates signs even if lower digits are constant
        // within each sign. Its existing sign-bit flip orders negatives first.
        if classes.iter().all(|c| c.seen()) {
            mask |= 1 << ((2 - axis) * 8 + 7);
        }
    }
    mask
}

fn varying_digits(differences: [u32; 3]) -> u32 {
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
        // An omitted digit is constant within each sign. Retaining one cannot
        // affect the final order because the sign-separating digit is retained.
        mask | (1 << ((!mask & ALL_DIGITS).trailing_zeros()))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_radix_digits_tests.rs"]
mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_sign_digits_tests.rs"]
mod sign_tests;
