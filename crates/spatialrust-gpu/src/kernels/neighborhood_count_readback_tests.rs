use super::*;

#[path = "neighborhood_count_decode_reference.rs"]
mod parent_decode;
#[path = "neighborhood_count_download_reference.rs"]
mod parent_download;

fn count_words(n: usize, kind: GpuNeighborhoodKind) -> Vec<u32> {
    (0..n)
        .flat_map(|i| {
            [
                0,
                if kind == GpuNeighborhoodKind::NearestK {
                    0
                } else {
                    match i % 5 {
                        0 => 0,
                        1 => u32::MAX,
                        _ => (i as u32).wrapping_mul(0x9e37_79b9),
                    }
                },
            ]
        })
        .collect()
}

#[test]
fn count_only_decode_keeps_every_total_and_cutoff_shape() {
    for n in
        [0, 1, 65, 4096, MIN_COUNT_ROWS - 1, MIN_COUNT_ROWS, MIN_COUNT_ROWS + 1, 131_073, 524_289]
    {
        for kind in [GpuNeighborhoodKind::NearestK, GpuNeighborhoodKind::Radius] {
            let words = count_words(n, kind);
            let current = decode(&words, 2, 0, kind).unwrap();
            let parent = parent_decode::decode(&words, 2, 0, kind).unwrap();
            assert_eq!(current, parent);
            check_packed(&words, 2, &current);
            assert_eq!(current.0.len(), n);
            assert!(current.0.iter().all(Vec::is_empty));
        }
    }
}

#[test]
fn count_only_or_validation_rejects_each_bit_and_releases_explicit_maps() {
    let r = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let n = MIN_COUNT_ROWS + 1;
    for kind in [GpuNeighborhoodKind::NearestK, GpuNeighborhoodKind::Radius] {
        let mut words = count_words(n, kind);
        let gpu = gpu_words(&r, &words, 0, kind);
        assert!(gpu.download(&other).is_err());
        compare(&gpu.download(&r).unwrap(), &parent_download::download(&gpu, &r).unwrap());
        for index in [0, 1, 63, n / 2, n - 1] {
            for bit in 0..32 {
                for field in 0..if kind == GpuNeighborhoodKind::NearestK { 2 } else { 1 } {
                    let offset = index * 2 + field;
                    let original = words[offset];
                    words[offset] = 1 << bit;
                    let error = decode(&words, 2, 0, kind).err().unwrap();
                    assert_eq!(error, parent_decode::decode(&words, 2, 0, kind).err().unwrap());
                    words[offset] = original;
                }
            }
        }
        // Repeat actual failed/successful maps at the first, middle and last row.
        for index in [0, n / 2, n - 1] {
            let offset = index * 8;
            r.queue().write_buffer(&gpu.buffer, offset as u64, bytemuck::cast_slice(&[u32::MAX]));
            for _ in 0..2 {
                assert_eq!(
                    gpu.download(&r).err().unwrap(),
                    parent_download::download(&gpu, &r).err().unwrap()
                );
            }
            r.queue().write_buffer(&gpu.buffer, offset as u64, bytemuck::cast_slice(&[0u32]));
            compare(&gpu.download(&r).unwrap(), &parent_download::download(&gpu, &r).unwrap());
        }
    }
}

fn search_fixture(n: usize) -> (KdTree, Vec<[f32; 3]>) {
    let points: Vec<_> = (0..n)
        .map(|i| [(i % 97) as f32 * 0.5, (i / 97 % 89) as f32 * 0.5, (i / 8633) as f32 * 0.5])
        .collect();
    let axes: [Vec<_>; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).collect());
    (KdTree::from_slices(&axes[0], &axes[1], &axes[2]), points)
}

#[test]
fn count_only_gpu_search_keeps_cpu_counts_overflow_receipts_and_owned_results() {
    let r = WgpuRuntime::new_headless().unwrap();
    let (tree, points) = tree_fixture();
    let target = GpuKdTree::upload(&r, &tree).unwrap();
    let mut live = None;
    for n in [0, 65, MIN_COUNT_ROWS - 1, MIN_COUNT_ROWS, MIN_COUNT_ROWS + 1] {
        let qs: Vec<_> = (0..n).map(|i| points[i % points.len()]).collect();
        let batch = GpuQueryBatch::upload(&r, &qs).unwrap();
        for (radius, bound) in [(false, 0.), (true, 0.), (true, 0.5), (true, f32::INFINITY)] {
            let output = if radius {
                target.radius_neighbors_resident(&r, &batch, bound, 0)
            } else {
                target.nearest_k_resident(&r, &batch, 0)
            }
            .unwrap();
            let current = output.download(&r).unwrap();
            compare(&current, &parent_download::download(&output, &r).unwrap());
            let expected: Vec<_> = points
                .iter()
                .map(|p| {
                    if radius {
                        tree.radius_search(p[0], p[1], p[2], bound).len() as u32
                    } else {
                        0
                    }
                })
                .collect();
            for (i, &total) in current.total_counts().iter().enumerate() {
                assert_eq!(total, expected[i % points.len()]);
                assert!(current.rows[i].is_empty());
            }
            assert_eq!(current.transfers.device_to_host_bytes(), n as u64 * 8);
            assert_eq!(current.transfers.host_to_device_bytes(), 0);
            assert_eq!(current.transfers.device_to_device_bytes(), 0);
            assert_eq!(output.execution_transfers(), TransferStats::default());
            assert_eq!(current.is_complete(), !radius || n == 0);
            assert_eq!(current.into_complete_rows().is_ok(), !radius || n == 0);
            if n == MIN_COUNT_ROWS + 1 && radius && bound == f32::INFINITY {
                live = Some(output);
            }
        }
    }
    drop(target);
    drop(tree);
    drop(points);
    r.clear_buffer_pool();
    let live = live.unwrap();
    std::thread::scope(|scope| {
        for _ in 0..3 {
            let (live, r) = (&live, &r);
            scope.spawn(move || {
                let current = live.download(r).unwrap();
                compare(&current, &parent_download::download(live, r).unwrap());
                assert_eq!(current.total_counts(), vec![17; MIN_COUNT_ROWS + 1]);
            });
        }
    });
}

#[path = "neighborhood_count_readback_benchmark.rs"]
mod benchmark;
