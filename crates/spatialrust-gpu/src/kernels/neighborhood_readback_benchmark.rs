use super::*;
use std::{hint::black_box, time::Instant};

fn previous_decode(
    words: &[u32],
    stride: usize,
    capacity: u32,
    kind: GpuNeighborhoodKind,
) -> SpatialResult<Decoded> {
    let mut rows = Vec::with_capacity(words.len() / stride);
    let mut totals = Vec::with_capacity(words.len() / stride);
    for row in words.chunks_exact(stride) {
        if row[0] > capacity
            || row[1] < row[0]
            || (kind == GpuNeighborhoodKind::NearestK && row[0] != row[1])
        {
            return Err(SpatialError::InvalidArgument(
                "GPU neighborhood metadata is invalid".to_owned(),
            ));
        }
        rows.push(
            (0..row[0] as usize)
                .map(|j| Neighbor {
                    index: row[2 + 2 * j] as usize,
                    distance_squared: f32::from_bits(row[3 + 2 * j]),
                })
                .collect(),
        );
        totals.push(row[1]);
    }
    Ok((rows, totals))
}

#[test]
#[ignore = "paired CPU conversion and complete explicit maps; run alone in release mode"]
fn benchmark_neighborhood_readback_decode_and_download() {
    let r = WgpuRuntime::new_headless().unwrap();
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("NEIGHBOR_READBACK_ENV backend={} device_type={} available_workers={available} max_workers=4 probe_rows=64 warmups=3 pairs=30 checks_drops_excluded=true worker_spawn_and_join_included=true",r.adapter_info().backend,r.adapter_info().device_type);
    for (n, capacity, stored, radius, pattern) in [
        (1025, 8, 8, false, 0),
        (MIN_ROWS - 1, 8, 8, false, 0),
        (MIN_ROWS, 8, 8, false, 0),
        (MIN_ROWS + 1, 8, 8, false, 0),
        (MIN_ROWS + 1, 16, 16, false, 0),
        (MIN_ROWS + 1, 64, 64, false, 0),
        (MIN_ROWS * 2 + 1, 8, 8, false, 0),
        (MIN_NEIGHBORS, 1, 1, false, 0),
        (MIN_ROWS + 1, 8, 0, true, 0),
        (MIN_ROWS + 1, 8, 1, true, 0),
        (MIN_ROWS + 1, 0, 0, true, 0),
        (MIN_ROWS + 1, 65, 9, true, 0),
        (MIN_ROWS + 1, 65, 1, true, 0),
        (MIN_ROWS + 1, 65, 0, true, 0),
        (MIN_ROWS + 1, 65, 65, true, 0),
        (MIN_ROWS + 1, 16, 16, true, 1),
        (MIN_ROWS + 1, 64, 64, true, 2),
    ] {
        let kind = if radius { GpuNeighborhoodKind::Radius } else { GpuNeighborhoodKind::NearestK };
        let mut words = packed(n, capacity, stored, kind);
        distribution(&mut words, capacity, pattern);
        let stride = 2 + 2 * capacity as usize;
        let gpu = gpu_words(&r, &words, capacity, kind);
        let expected = parent::download(&gpu, &r).unwrap();
        let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..4 {
                let variant = offset ^ (round % 4);
                let start = Instant::now();
                let result = match variant {
                    0 => previous_decode(black_box(&words), stride, capacity, kind).unwrap(),
                    1 => decode(black_box(&words), stride, capacity, kind).unwrap(),
                    2 => {
                        let result = parent::download(black_box(&gpu), &r).unwrap();
                        (result.rows, result.totals)
                    }
                    _ => {
                        let result = gpu.download(black_box(&r)).unwrap();
                        (result.rows, result.totals)
                    }
                };
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                black_box(&result);
                check_packed(&words, stride, &result);
                if variant >= 2 {
                    assert_eq!(result.1, expected.totals);
                }
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        let selected = eligible(&words, stride, capacity);
        let threads = if selected { workers(n, available) } else { 1 };
        println!("NEIGHBOR_READBACK_RECORD {{\"queries\":{n},\"capacity\":{capacity},\"stored\":{stored},\"distribution\":{pattern},\"radius\":{radius},\"selected\":{selected},\"workers\":{threads},\"cpu_old_ms\":{:?},\"cpu_ms\":{:?},\"download_old_ms\":{:?},\"download_ms\":{:?}}}",samples[0],samples[1],samples[2],samples[3]);
    }
}

#[test]
#[ignore = "paired complete tree/query upload, search and explicit download; run alone in release mode"]
fn benchmark_neighborhood_readback_full_search() {
    let r = WgpuRuntime::new_headless().unwrap();
    let (tree, points) = tree_fixture();
    println!("NEIGHBOR_READBACK_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 tree_build_excluded=true uploads_search_map_decode_included=true checks_drops_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (n, capacity, radius, bound) in [
        (1025, 8, false, 0.),
        (MIN_ROWS - 1, 8, false, 0.),
        (MIN_ROWS, 8, false, 0.),
        (MIN_ROWS + 1, 8, false, 0.),
        (MIN_ROWS + 1, 16, false, 0.),
        (MIN_ROWS * 2 + 1, 8, false, 0.),
        (MIN_ROWS + 1, 8, true, f32::INFINITY),
        (MIN_ROWS + 1, 8, true, 0.),
        (MIN_ROWS + 1, 0, true, f32::INFINITY),
    ] {
        let qs: Vec<_> = (0..n).map(|i| points[i % points.len()]).collect();
        let uploaded = GpuKdTree::upload(&r, &tree).unwrap();
        let batch = GpuQueryBatch::upload(&r, &qs).unwrap();
        let expected_gpu = if radius {
            uploaded.radius_neighbors_resident(&r, &batch, bound, capacity)
        } else {
            uploaded.nearest_k_resident(&r, &batch, capacity)
        }
        .unwrap();
        let expected = parent::download(&expected_gpu, &r).unwrap();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                let uploaded = GpuKdTree::upload(black_box(&r), black_box(&tree)).unwrap();
                let batch = GpuQueryBatch::upload(&r, black_box(&qs)).unwrap();
                let gpu = if radius {
                    uploaded.radius_neighbors_resident(&r, &batch, bound, capacity)
                } else {
                    uploaded.nearest_k_resident(&r, &batch, capacity)
                }
                .unwrap();
                let result =
                    if variant == 0 { parent::download(&gpu, &r) } else { gpu.download(&r) }
                        .unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                black_box(&result);
                compare(&result, &expected);
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("NEIGHBOR_READBACK_FULL {{\"points\":17,\"queries\":{n},\"capacity\":{capacity},\"radius\":{radius},\"zero_radius\":{},\"full_old_ms\":{:?},\"full_ms\":{:?}}}",radius&&bound==0.,samples[0],samples[1]);
    }
}
