//! Shared deterministic CPU dispatch policy for vision kernels.

/// Minimum component count for lightweight row-parallel kernels.
pub(crate) const LIGHT_PARALLEL_COMPONENTS: usize = 100_000;
/// Minimum component count for row kernels with per-worker scratch.
pub(crate) const ROW_PARALLEL_COMPONENTS: usize = 256 * 1024;
/// Minimum component count for large row/band/tile kernels.
pub(crate) const LARGE_PARALLEL_COMPONENTS: usize = 1_000_000;
/// Tall outputs are grouped into bounded row tiles to amortize scheduling.
pub(crate) const TALL_IMAGE_ROWS: usize = 2_000;
/// Rows processed by each tall-image tile.
pub(crate) const ROWS_PER_TILE: usize = 8;

/// Returns whether a kernel should leave its scalar path.
#[inline]
pub(crate) const fn should_parallelize(
    components: usize,
    independent_items: usize,
    threshold: usize,
) -> bool {
    components >= threshold && independent_items > 1
}

/// Bounds workers by both the runtime pool and independent work.
#[inline]
pub(crate) const fn bounded_workers(
    components: usize,
    independent_items: usize,
    threshold: usize,
    available_workers: usize,
) -> usize {
    if should_parallelize(components, independent_items, threshold) {
        let available_workers = if available_workers == 0 { 1 } else { available_workers };
        let workers = if available_workers < independent_items {
            available_workers
        } else {
            independent_items
        };
        if workers == 0 {
            1
        } else {
            workers
        }
    } else {
        1
    }
}

/// Returns a non-zero contiguous item count for each bounded worker.
#[inline]
pub(crate) const fn items_per_worker(independent_items: usize, workers: usize) -> usize {
    let workers = if workers == 0 { 1 } else { workers };
    let items = independent_items.div_ceil(workers);
    if items == 0 {
        1
    } else {
        items
    }
}

/// Selects packed one/three-channel `u8` execution without weakening fallbacks.
#[inline]
pub(crate) fn is_packed_u8_fast_path<const CHANNELS: usize>(
    input_width: usize,
    input_stride: usize,
    output_width: usize,
    output_stride: usize,
) -> bool {
    matches!(CHANNELS, 1 | 3)
        && input_width.checked_mul(CHANNELS) == Some(input_stride)
        && output_width.checked_mul(CHANNELS) == Some(output_stride)
}

#[cfg(test)]
mod tests {
    use super::{
        bounded_workers, is_packed_u8_fast_path, items_per_worker, should_parallelize,
        LARGE_PARALLEL_COMPONENTS, ROW_PARALLEL_COMPONENTS,
    };

    #[test]
    fn thresholds_are_exact_and_require_independent_work() {
        assert!(!should_parallelize(ROW_PARALLEL_COMPONENTS - 1, 8, ROW_PARALLEL_COMPONENTS));
        assert!(should_parallelize(ROW_PARALLEL_COMPONENTS, 8, ROW_PARALLEL_COMPONENTS));
        assert!(!should_parallelize(LARGE_PARALLEL_COMPONENTS, 1, LARGE_PARALLEL_COMPONENTS));
    }

    #[test]
    fn worker_count_is_bounded_and_never_zero() {
        assert_eq!(bounded_workers(10, 100, ROW_PARALLEL_COMPONENTS, 16), 1);
        assert_eq!(bounded_workers(ROW_PARALLEL_COMPONENTS, 3, ROW_PARALLEL_COMPONENTS, 16), 3);
        assert_eq!(bounded_workers(ROW_PARALLEL_COMPONENTS, 20, ROW_PARALLEL_COMPONENTS, 4), 4);
        assert_eq!(bounded_workers(ROW_PARALLEL_COMPONENTS, 20, ROW_PARALLEL_COMPONENTS, 0), 1);
        assert_eq!(items_per_worker(10, 3), 4);
        assert_eq!(items_per_worker(0, 0), 1);
    }

    #[test]
    fn packed_u8_selector_preserves_channel_and_stride_fallbacks() {
        assert!(is_packed_u8_fast_path::<1>(32, 32, 16, 16));
        assert!(is_packed_u8_fast_path::<3>(32, 96, 16, 48));
        assert!(!is_packed_u8_fast_path::<4>(32, 128, 16, 64));
        assert!(!is_packed_u8_fast_path::<3>(32, 100, 16, 48));
        assert!(!is_packed_u8_fast_path::<3>(32, 96, 16, 52));
        assert!(!is_packed_u8_fast_path::<3>(usize::MAX, 0, 16, 48));
    }
}
