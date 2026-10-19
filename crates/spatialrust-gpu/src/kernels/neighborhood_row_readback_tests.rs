use super::*;

#[path = "neighborhood_row_decode_candidate.rs"]
mod candidate_decode;
#[path = "neighborhood_row_download_candidate.rs"]
mod candidate_download;
use candidate_decode::{decode as candidate_rows, decode_with_workers as candidate_with_workers};

#[path = "neighborhood_row_decode_reference.rs"]
mod parent_decode;
#[path = "neighborhood_row_download_reference.rs"]
mod parent_download;

#[test]
fn direct_parallel_rows_keep_bits_counts_order_padding_and_worker_edges() {
    for (n, capacity, stored) in [
        (0, 0, 0),
        (65, 1, 1),
        (MIN_ROWS - 1, 8, 8),
        (MIN_ROWS, 8, 8),
        (MIN_ROWS + 1, 8, 8),
        (MIN_ROWS + 3, 16, 16),
        (MIN_ROWS + 1, 64, 64),
        (MIN_ROWS + 3, 65, 9),
        (MIN_ROWS * 2 + 1, 8, 8),
    ] {
        for kind in [GpuNeighborhoodKind::NearestK, GpuNeighborhoodKind::Radius] {
            let words = packed(n, capacity, stored, kind);
            let stride = 2 + 2 * capacity as usize;
            let expected = parent_decode::decode(&words, stride, capacity, kind).unwrap();
            check_packed(&words, stride, &expected);
            check_packed(&words, stride, &candidate_rows(&words, stride, capacity, kind).unwrap());
            for available in [0, 1, 2, 3, 4, 99] {
                check_packed(
                    &words,
                    stride,
                    &candidate_with_workers(&words, stride, capacity, kind, available).unwrap(),
                );
            }
        }
    }
    for (capacity, pattern) in [(16, 1), (64, 2)] {
        let mut words = packed(MIN_ROWS + 1, capacity, capacity, GpuNeighborhoodKind::Radius);
        distribution(&mut words, capacity, pattern);
        check_packed(
            &words,
            2 + 2 * capacity as usize,
            &candidate_rows(
                &words,
                2 + 2 * capacity as usize,
                capacity,
                GpuNeighborhoodKind::Radius,
            )
            .unwrap(),
        );
    }
}

#[test]
fn direct_parallel_rows_reject_headers_in_each_band_and_unmap_before_error_return() {
    let r = WgpuRuntime::new_headless().unwrap();
    let n = MIN_ROWS + 3;
    for kind in [GpuNeighborhoodKind::NearestK, GpuNeighborhoodKind::Radius] {
        let mut words = packed(n, 8, 8, kind);
        let gpu = gpu_words(&r, &words, 8, kind);
        for index in [0, n.div_ceil(4) - 1, n.div_ceil(4), n / 2, n * 3 / 4, n - 1] {
            for header in [[9, 9], [8, 7], [7, 8]] {
                if kind == GpuNeighborhoodKind::Radius && header == [7, 8] {
                    continue;
                }
                let original = [words[index * 18], words[index * 18 + 1]];
                words[index * 18..index * 18 + 2].copy_from_slice(&header);
                for available in [1, 2, 3, 4] {
                    let error =
                        candidate_with_workers(&words, 18, 8, kind, available).err().unwrap();
                    assert_eq!(error, parent_decode::decode(&words, 18, 8, kind).err().unwrap());
                }
                r.queue().write_buffer(
                    &gpu.buffer,
                    (index * 18 * 4) as u64,
                    bytemuck::cast_slice(&header),
                );
                assert_eq!(
                    candidate_download::download(&gpu, &r).err().unwrap(),
                    parent_download::download(&gpu, &r).err().unwrap()
                );
                words[index * 18..index * 18 + 2].copy_from_slice(&original);
                r.queue().write_buffer(
                    &gpu.buffer,
                    (index * 18 * 4) as u64,
                    bytemuck::cast_slice(&original),
                );
                compare(
                    &candidate_download::download(&gpu, &r).unwrap(),
                    &parent_download::download(&gpu, &r).unwrap(),
                );
            }
        }
    }
}

#[test]
fn direct_parallel_maps_keep_ownership_receipts_and_concurrent_results() {
    let r = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    for (capacity, stored, kind) in [
        (8, 8, GpuNeighborhoodKind::NearestK),
        (16, 16, GpuNeighborhoodKind::NearestK),
        (65, 9, GpuNeighborhoodKind::Radius),
    ] {
        let words = packed(MIN_ROWS + 1, capacity, stored, kind);
        let gpu = gpu_words(&r, &words, capacity, kind);
        assert!(candidate_download::download(&gpu, &other).is_err());
        let expected = parent_download::download(&gpu, &r).unwrap();
        compare(&gpu.download(&r).unwrap(), &expected);
        std::thread::scope(|scope| {
            for _ in 0..3 {
                let (r, gpu, expected) = (&r, &gpu, &expected);
                scope.spawn(move || {
                    compare(&candidate_download::download(gpu, r).unwrap(), expected)
                });
            }
        });
        let current = candidate_download::download(&gpu, &r).unwrap();
        assert_eq!(current.transfers.device_to_host_bytes(), gpu.bytes);
        assert_eq!(current.transfers.host_to_device_bytes(), 0);
        assert_eq!(current.transfers.device_to_device_bytes(), 0);
        drop(gpu);
        drop(words);
        r.clear_buffer_pool();
        compare(&current, &expected);
    }
}

fn search_fixture(n: usize) -> (KdTree, Vec<[f32; 3]>) {
    let points: Vec<_> = (0..n)
        .map(|i| [(i % 97) as f32 * 0.5, (i / 97 % 89) as f32 * 0.5, (i / 8633) as f32 * 0.5])
        .collect();
    let axes: [Vec<_>; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).collect());
    (KdTree::from_slices(&axes[0], &axes[1], &axes[2]), points)
}

#[path = "neighborhood_row_readback_benchmark.rs"]
mod benchmark;
