//! Parallel dispatch for chunked spatial queries (`parallel` feature).
//!
//! Contiguous groups of original [`SpatialTensor`] chunks share scoped workers,
//! bounded by available CPU parallelism. Results are appended in chunk order;
//! worker-local result buffers need no locks. Small inputs and one-worker jobs
//! run on the caller. Result storage still scales with the number of neighbors.

use std::ops::Range;

use spatialrust_core::SpatialTensor;

use crate::chunked::{ChunkedNearestNeighborIndex, ChunkedRadiusSearchIndex};
use crate::Neighbor;

/// Minimum point count before threaded chunk dispatch is used.
pub const PARALLEL_CHUNK_QUERY_MIN_POINTS: usize = 4_096;

/// Parallel radius search over [`SpatialTensor`] chunks.
pub fn radius_search_spatial_tensor_parallel<I: ChunkedRadiusSearchIndex + Sync>(
    index: &I,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    tensor: &SpatialTensor<'_>,
    radius: f32,
) -> Vec<(usize, Neighbor)> {
    let mut out = Vec::new();
    radius_search_spatial_tensor_parallel_into(index, x, y, z, tensor, radius, &mut out);
    out
}

/// Parallel radius search, appending `(query_index, neighbor)` pairs to `out`.
///
/// Retains original chunk and neighbor order. Scoped workers per call are capped
/// by available CPU parallelism and chunk count; small or one-worker jobs run
/// on the caller. Neighbor storage scales with result count.
pub fn radius_search_spatial_tensor_parallel_into<I: ChunkedRadiusSearchIndex + Sync>(
    index: &I,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    tensor: &SpatialTensor<'_>,
    radius: f32,
    out: &mut Vec<(usize, Neighbor)>,
) {
    dispatch_chunks(tensor, available_workers(), out, |range, local| {
        index.radius_search_chunk_into(x, y, z, range, radius, local);
    });
}

/// Parallel k-NN search over [`SpatialTensor`] chunks.
pub fn nearest_k_spatial_tensor_parallel<I: ChunkedNearestNeighborIndex + Sync>(
    index: &I,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    tensor: &SpatialTensor<'_>,
    k: usize,
) -> Vec<(usize, Neighbor)> {
    let mut out = Vec::new();
    nearest_k_spatial_tensor_parallel_into(index, x, y, z, tensor, k, &mut out);
    out
}

/// Parallel k-NN search, appending `(query_index, neighbor)` pairs to `out`.
///
/// Uses the same worker bounds and ordered append contract as
/// [`radius_search_spatial_tensor_parallel_into`]. Empty input or `k == 0`
/// leaves `out` untouched.
pub fn nearest_k_spatial_tensor_parallel_into<I: ChunkedNearestNeighborIndex + Sync>(
    index: &I,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    tensor: &SpatialTensor<'_>,
    k: usize,
    out: &mut Vec<(usize, Neighbor)>,
) {
    if tensor.is_empty() || k == 0 {
        return;
    }
    dispatch_chunks(tensor, available_workers(), out, |range, local| {
        index.nearest_k_chunk_into(x, y, z, range, k, local);
    });
}

fn available_workers() -> usize {
    std::thread::available_parallelism().map_or(1, |count| count.get())
}

fn chunk_worker_count(point_count: usize, chunk_size: usize, available: usize) -> usize {
    if point_count < PARALLEL_CHUNK_QUERY_MIN_POINTS {
        return 1;
    }
    available.max(1).min(point_count.div_ceil(chunk_size))
}

fn dispatch_chunks<Q>(
    tensor: &SpatialTensor<'_>,
    available: usize,
    out: &mut Vec<(usize, Neighbor)>,
    query: Q,
) where
    Q: Fn(Range<usize>, &mut Vec<(usize, Neighbor)>) + Sync,
{
    let point_count = tensor.len();
    if point_count == 0 {
        return;
    }
    let chunk_size = tensor.chunk_size();
    let chunk_count = point_count.div_ceil(chunk_size);
    let workers = chunk_worker_count(point_count, chunk_size, available);
    // Preserve each backend callback's original chunk boundary. Compute ranges
    // lazily, avoiding an O(chunk_count) range/handle allocation for tiny chunks.
    let query_band = |chunks: Range<usize>, local: &mut Vec<(usize, Neighbor)>| {
        for chunk in chunks {
            let start = chunk * chunk_size; // chunk < ceil(point_count/chunk_size)
            let end = start + chunk_size.min(point_count - start);
            query(start..end, local);
        }
    };
    if workers == 1 {
        query_band(0..chunk_count, out);
        return;
    }

    let chunks_per_worker = chunk_count / workers;
    let extra_chunks = chunk_count % workers;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            let start = worker * chunks_per_worker + worker.min(extra_chunks);
            let end = start + chunks_per_worker + usize::from(worker < extra_chunks);
            let query_band = &query_band;
            handles.push(scope.spawn(move || {
                let mut local = Vec::new();
                query_band(start..end, &mut local);
                local
            }));
        }

        for handle in handles {
            out.extend(handle.join().expect("bounded chunk query worker panicked"));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NearestNeighborIndex, RadiusSearchIndex, SpatialIndex};
    use spatialrust_core::{PointCloud, PointCloudBuilder};
    use std::{collections::HashSet, sync::Mutex, thread::ThreadId};

    fn cloud(point_count: usize) -> PointCloud {
        if point_count == 0 {
            return PointCloud::xyz();
        }
        let mut builder = PointCloudBuilder::xyz();
        for index in 0..point_count {
            builder.push_point([index as f32, 0.0, 0.0]).unwrap();
        }
        builder.build().unwrap()
    }

    #[derive(Default)]
    struct RecordingIndex {
        calls: Mutex<Vec<(ThreadId, Range<usize>)>>,
    }

    impl RecordingIndex {
        fn record(&self, range: Range<usize>, out: &mut Vec<(usize, Neighbor)>) {
            self.calls.lock().unwrap().push((std::thread::current().id(), range.clone()));
            // Two results in deliberately descending neighbor order per chunk.
            // Encoding the chunk end also detects accidental boundary merging.
            for query in range.clone() {
                for index in [range.end, range.start] {
                    out.push((query, Neighbor { index, distance_squared: query as f32 }));
                }
            }
        }
    }

    impl SpatialIndex for RecordingIndex {
        fn len(&self) -> usize {
            0
        }
    }
    impl RadiusSearchIndex for RecordingIndex {
        fn radius_search(&self, _: f32, _: f32, _: f32, _: f32) -> Vec<Neighbor> {
            unreachable!("dispatch must use original chunk callbacks")
        }
    }
    impl NearestNeighborIndex for RecordingIndex {
        fn nearest_one(&self, _: f32, _: f32, _: f32) -> Option<Neighbor> {
            unreachable!()
        }
        fn nearest_k(&self, _: f32, _: f32, _: f32, _: usize) -> Vec<Neighbor> {
            unreachable!()
        }
    }
    impl ChunkedRadiusSearchIndex for RecordingIndex {
        fn radius_search_chunk_into(
            &self,
            _: &[f32],
            _: &[f32],
            _: &[f32],
            chunk: Range<usize>,
            _: f32,
            out: &mut Vec<(usize, Neighbor)>,
        ) {
            self.record(chunk, out);
        }
    }
    impl ChunkedNearestNeighborIndex for RecordingIndex {
        fn nearest_k_chunk_into(
            &self,
            _: &[f32],
            _: &[f32],
            _: &[f32],
            chunk: Range<usize>,
            _: usize,
            out: &mut Vec<(usize, Neighbor)>,
        ) {
            self.record(chunk, out);
        }
    }

    fn prefix() -> Vec<(usize, Neighbor)> {
        vec![(usize::MAX, Neighbor { index: usize::MAX, distance_squared: -1.0 })]
    }

    #[test]
    fn worker_policy_bounds_threads_including_tiny_and_partial_chunks() {
        for available in [0, 1, 2, 6, usize::MAX] {
            let cap = available.max(1);
            assert_eq!(chunk_worker_count(0, 1, available), 1);
            assert_eq!(chunk_worker_count(4095, 1, available), 1);
            assert_eq!(chunk_worker_count(4096, 1, available), cap.min(4096));
            assert_eq!(chunk_worker_count(4097, 4096, available), cap.min(2));
            assert_eq!(chunk_worker_count(4096, usize::MAX, available), 1);
            assert_eq!(chunk_worker_count(usize::MAX, usize::MAX, available), 1);
            assert_eq!(
                chunk_worker_count(usize::MAX, 2, available),
                cap.min(usize::MAX.div_ceil(2))
            );
        }
    }

    #[test]
    fn bounded_dispatch_preserves_callbacks_order_and_append_prefix() {
        let cloud = cloud(4103);
        for chunk_size in [1, 7, 512, 4103, usize::MAX] {
            let tensor = cloud.spatial_tensor_chunks(chunk_size).unwrap();
            for available in [1, 2, 6] {
                let index = RecordingIndex::default();
                let mut actual = prefix();
                dispatch_chunks(&tensor, available, &mut actual, |range, out| {
                    index.record(range, out)
                });
                let mut expected = prefix();
                let mut expected_ranges = Vec::new();
                for start in (0..cloud.len()).step_by(chunk_size) {
                    let range = start..start + chunk_size.min(cloud.len() - start);
                    expected_ranges.push(range.clone());
                    RecordingIndex::default().record(range, &mut expected);
                }
                assert_eq!(actual, expected, "chunk={chunk_size}, workers={available}");
                let calls = index.calls.lock().unwrap();
                let mut ranges: Vec<_> = calls.iter().map(|(_, range)| range.clone()).collect();
                ranges.sort_by_key(|range| range.start);
                assert_eq!(ranges, expected_ranges);
                let threads: HashSet<_> = calls.iter().map(|(id, _)| *id).collect();
                let workers = chunk_worker_count(cloud.len(), chunk_size, available);
                assert_eq!(threads.len(), workers);
                let counts: Vec<_> = threads
                    .iter()
                    .map(|id| {
                        let ranges: Vec<_> = calls
                            .iter()
                            .filter(|(thread, _)| thread == id)
                            .map(|(_, range)| range)
                            .collect();
                        for pair in ranges.windows(2) {
                            assert_eq!(pair[0].end, pair[1].start);
                        }
                        ranges.len()
                    })
                    .collect();
                assert!(counts.iter().max().unwrap() - counts.iter().min().unwrap() <= 1);
                if workers == 1 {
                    assert!(threads.contains(&std::thread::current().id()));
                } else {
                    assert!(!threads.contains(&std::thread::current().id()));
                }
            }
        }
    }

    #[test]
    fn public_radius_and_knn_append_unsorted_results_for_generic_backend() {
        let cloud = cloud(4103);
        let tensor = cloud.spatial_tensor_chunks(1).unwrap();
        let expected_index = RecordingIndex::default();
        let mut expected = prefix();
        crate::radius_search_spatial_tensor(
            &expected_index,
            &[],
            &[],
            &[],
            &tensor,
            2.0,
            &mut expected,
        );
        let index = RecordingIndex::default();
        let mut radius = prefix();
        radius_search_spatial_tensor_parallel_into(
            &index,
            &[],
            &[],
            &[],
            &tensor,
            2.0,
            &mut radius,
        );
        assert_eq!(radius, expected);
        let mut knn = prefix();
        nearest_k_spatial_tensor_parallel_into(&index, &[], &[], &[], &tensor, 2, &mut knn);
        assert_eq!(knn, expected);
        assert_eq!(
            radius_search_spatial_tensor_parallel(&index, &[], &[], &[], &tensor, 2.0),
            expected[1..]
        );
        assert_eq!(
            nearest_k_spatial_tensor_parallel(&index, &[], &[], &[], &tensor, 2),
            expected[1..]
        );
        let calls = index.calls.lock().unwrap();
        let threads: HashSet<_> = calls.iter().map(|(id, _)| *id).collect();
        // Four independent API calls may create different thread identities.
        assert!(threads.len() <= 4 * available_workers().min(tensor.len()));
    }

    #[test]
    fn small_single_chunk_empty_and_zero_k_keep_caller_or_noop_behavior() {
        for (point_count, chunk_size) in [(0, 1), (4095, 1), (4096, 4096)] {
            let cloud = cloud(point_count);
            let tensor = cloud.spatial_tensor_chunks(chunk_size).unwrap();
            let index = RecordingIndex::default();
            let mut out = prefix();
            radius_search_spatial_tensor_parallel_into(
                &index,
                &[],
                &[],
                &[],
                &tensor,
                1.0,
                &mut out,
            );
            nearest_k_spatial_tensor_parallel_into(&index, &[], &[], &[], &tensor, 1, &mut out);
            let calls = index.calls.lock().unwrap();
            assert!(calls.iter().all(|(id, _)| *id == std::thread::current().id()));
            if point_count == 0 {
                assert_eq!(out, prefix());
            }
        }
        let cloud = cloud(4103);
        let tensor = cloud.spatial_tensor_chunks(1).unwrap();
        let index = RecordingIndex::default();
        let mut out = prefix();
        nearest_k_spatial_tensor_parallel_into(&index, &[], &[], &[], &tensor, 0, &mut out);
        assert_eq!(out, prefix());
        assert!(index.calls.lock().unwrap().is_empty());
    }

    #[cfg(feature = "search-kdtree")]
    #[test]
    fn kdtree_radius_and_knn_match_sequential_without_sorting() {
        let mut builder = PointCloudBuilder::xyz();
        for i in 0..4103 {
            // Negative coordinates, duplicates and equal-distance neighbors.
            builder
                .push_point([(i % 67) as f32 - 33.0, ((i / 67) % 61) as f32 - 30.0, 0.0])
                .unwrap();
        }
        let cloud = builder.build().unwrap();
        let x = cloud.field("x").unwrap().as_f32().unwrap();
        let y = cloud.field("y").unwrap().as_f32().unwrap();
        let z = cloud.field("z").unwrap().as_f32().unwrap();
        let tree = crate::KdTree::from_slices(x, y, z);
        for chunk_size in [1, 7, 512, 4103] {
            let tensor = cloud.spatial_tensor_chunks(chunk_size).unwrap();
            let mut radius = prefix();
            crate::radius_search_spatial_tensor(&tree, x, y, z, &tensor, 1.5, &mut radius);
            let mut knn = prefix();
            crate::nearest_k_spatial_tensor(&tree, x, y, z, &tensor, 8, &mut knn);
            for workers in [1, 2, 6] {
                let mut actual = prefix();
                dispatch_chunks(&tensor, workers, &mut actual, |range, out| {
                    tree.radius_search_chunk_into(x, y, z, range, 1.5, out);
                });
                assert_eq!(actual, radius);
                let mut actual = prefix();
                dispatch_chunks(&tensor, workers, &mut actual, |range, out| {
                    tree.nearest_k_chunk_into(x, y, z, range, 8, out);
                });
                assert_eq!(actual, knn);
            }
            let mut actual = prefix();
            radius_search_spatial_tensor_parallel_into(&tree, x, y, z, &tensor, 1.5, &mut actual);
            assert_eq!(actual, radius);
            let mut actual = prefix();
            nearest_k_spatial_tensor_parallel_into(&tree, x, y, z, &tensor, 8, &mut actual);
            assert_eq!(actual, knn);
        }
    }
}
