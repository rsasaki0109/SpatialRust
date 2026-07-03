//! Shared parallel index-range staging for per-point algorithms.
//!
//! Aligns worker counts and chunk boundaries with [`SpatialTensor`](spatialrust_core::SpatialTensor)
//! defaults so feature estimation and filtering reuse the same staging policy.

use std::ops::Range;

use spatialrust_core::DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE;

/// Minimum point count before multi-threaded index staging is used.
pub const PARALLEL_STAGING_MIN_POINTS: usize = 4_096;

/// Returns the number of worker threads to use for `point_count` points.
#[must_use]
pub fn parallel_worker_count(point_count: usize) -> usize {
    parallel_worker_count_with_chunk(point_count, DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE)
}

/// Returns the number of worker threads for an explicit staging chunk size.
#[must_use]
pub fn parallel_worker_count_with_chunk(point_count: usize, chunk_size: usize) -> usize {
    if point_count < PARALLEL_STAGING_MIN_POINTS || chunk_size == 0 {
        return 1;
    }
    let available = std::thread::available_parallelism().map_or(1, |count| count.get());
    let useful = (point_count / chunk_size).max(1);
    available.min(useful)
}

/// Returns disjoint half-open index ranges covering `[0, point_count)`.
pub fn parallel_index_ranges(
    point_count: usize,
    worker_count: usize,
) -> impl Iterator<Item = Range<usize>> {
    let worker_count = worker_count.max(1);
    let chunk_size = point_count.div_ceil(worker_count);
    (0..point_count).step_by(chunk_size).map(move |start| {
        let end = (start + chunk_size).min(point_count);
        start..end
    })
}

/// Runs `work` over disjoint index ranges using [`parallel_worker_count`].
pub fn parallel_index_for_each<W>(point_count: usize, work: W)
where
    W: Fn(Range<usize>) + Send + Sync,
{
    let worker_count = parallel_worker_count(point_count);
    if worker_count == 1 {
        work(0..point_count);
        return;
    }

    std::thread::scope(|scope| {
        for range in parallel_index_ranges(point_count, worker_count) {
            scope.spawn(|| work(range));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{parallel_index_ranges, parallel_worker_count, PARALLEL_STAGING_MIN_POINTS};

    #[test]
    fn single_worker_below_threshold() {
        assert_eq!(parallel_worker_count(PARALLEL_STAGING_MIN_POINTS - 1), 1);
    }

    #[test]
    fn ranges_cover_all_indices() {
        let point_count = 50_000;
        let workers = parallel_worker_count(point_count);
        assert!(workers > 1);
        let ranges: Vec<_> = parallel_index_ranges(point_count, workers).collect();
        let covered: usize = ranges.iter().map(|range| range.len()).sum();
        assert_eq!(covered, point_count);
        for window in ranges.windows(2) {
            assert_eq!(window[0].end, window[1].start);
        }
    }
}
