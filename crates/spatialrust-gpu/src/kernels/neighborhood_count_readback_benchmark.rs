use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "paired count-only CPU/decode and complete explicit readback; run alone in release mode"]
fn benchmark_neighborhood_count_decode_and_download() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("NEIGHBOR_COUNT_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 worker_creation_unchanged=true checks_drops_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (n, capacity, stored, radius) in [
        (1025, 0, 0, false),
        (4096, 0, 0, false),
        (65535, 0, 0, false),
        (65536, 0, 0, false),
        (65537, 0, 0, false),
        (524289, 0, 0, false),
        (1048577, 0, 0, false),
        (1025, 0, 0, true),
        (4096, 0, 0, true),
        (65535, 0, 0, true),
        (65536, 0, 0, true),
        (65537, 0, 0, true),
        (524289, 0, 0, true),
        (1048577, 0, 0, true),
        (65537, 8, 0, true),
        (65537, 8, 1, true),
        (65537, 8, 8, false),
        (65537, 65, 0, true),
    ] {
        let kind = if radius { GpuNeighborhoodKind::Radius } else { GpuNeighborhoodKind::NearestK };
        let words =
            if capacity == 0 { count_words(n, kind) } else { packed(n, capacity, stored, kind) };
        let stride = 2 + 2 * capacity as usize;
        let gpu = gpu_words(&r, &words, capacity, kind);
        let expected = parent_download::download(&gpu, &r).unwrap();
        let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..4 {
                let variant = offset ^ (round % 4);
                let start = Instant::now();
                let result = match variant {
                    0 => parent_decode::decode(black_box(&words), stride, capacity, kind).unwrap(),
                    1 => decode(black_box(&words), stride, capacity, kind).unwrap(),
                    2 => {
                        let result = parent_download::download(black_box(&gpu), &r).unwrap();
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
                assert_eq!(result.1, expected.totals);
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("NEIGHBOR_COUNT_RECORD {{\"queries\":{n},\"capacity\":{capacity},\"stored\":{stored},\"radius\":{radius},\"selected\":{},\"cpu_old_ms\":{:?},\"cpu_ms\":{:?},\"download_old_ms\":{:?},\"download_ms\":{:?}}}",capacity==0&&n>=MIN_COUNT_ROWS,samples[0],samples[1],samples[2],samples[3]);
    }
}

#[test]
#[ignore = "paired complete uploads, GPU search and explicit download; run alone in release mode"]
fn benchmark_neighborhood_count_full_search() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("NEIGHBOR_COUNT_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 tree_build_excluded=true uploads_search_map_decode_included=true checks_drops_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (n, q, capacity, radius, bound) in [
        (17, 65535, 0, true, 0.),
        (17, 65536, 0, true, 0.),
        (17, 65537, 0, true, f32::INFINITY),
        (17, 524289, 0, true, 0.),
        (4097, 65537, 0, true, 0.),
        (4097, 131073, 0, true, 0.),
        (17, 65537, 0, false, 0.),
        (17, 65537, 8, false, 0.),
        (4097, 65537, 8, false, 0.),
        (17, 4096, 0, true, f32::INFINITY),
    ] {
        let (tree, points) = search_fixture(n);
        let qs: Vec<_> = (0..q).map(|i| points[i % points.len()]).collect();
        let uploaded = GpuKdTree::upload(&r, &tree).unwrap();
        let batch = GpuQueryBatch::upload(&r, &qs).unwrap();
        let out = if radius {
            uploaded.radius_neighbors_resident(&r, &batch, bound, capacity)
        } else {
            uploaded.nearest_k_resident(&r, &batch, capacity)
        }
        .unwrap();
        let expected = parent_download::download(&out, &r).unwrap();
        if capacity == 0 {
            let count = if !radius {
                0
            } else if bound == 0. {
                1
            } else {
                n as u32
            };
            assert_eq!(expected.total_counts(), vec![count; q]);
        }
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                let uploaded = GpuKdTree::upload(black_box(&r), black_box(&tree)).unwrap();
                let batch = GpuQueryBatch::upload(&r, black_box(&qs)).unwrap();
                let out = if radius {
                    uploaded.radius_neighbors_resident(&r, &batch, bound, capacity)
                } else {
                    uploaded.nearest_k_resident(&r, &batch, capacity)
                }
                .unwrap();
                let current = if variant == 0 {
                    parent_download::download(&out, &r)
                } else {
                    out.download(&r)
                }
                .unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                black_box(&current);
                compare(&current, &expected);
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("NEIGHBOR_COUNT_FULL {{\"points\":{n},\"queries\":{q},\"capacity\":{capacity},\"radius\":{radius},\"zero_radius\":{},\"selected\":{},\"full_old_ms\":{:?},\"full_ms\":{:?}}}",radius&&bound==0.,capacity==0&&q>=MIN_COUNT_ROWS,samples[0],samples[1]);
    }
}
