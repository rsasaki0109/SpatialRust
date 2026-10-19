use super::*;
use std::{hint::black_box, time::Instant};
type Job = (usize, usize, u32, usize, bool);

fn execute(
    r: &WgpuRuntime,
    cache: &OnceLock<parent::ParentPipelines>,
    tree: &GpuKdTree,
    batch: &GpuQueryBatch,
    k: u32,
    radius: bool,
    current: bool,
) -> GpuNeighborhoods {
    if current {
        if radius {
            tree.radius_neighbors_resident(r, batch, 0.5, k)
        } else {
            tree.nearest_k_resident(r, batch, k)
        }
    } else {
        parent::previous(
            cache,
            tree,
            r,
            batch,
            k,
            if radius { 0.25 } else { f32::INFINITY },
            if radius { GpuNeighborhoodKind::Radius } else { GpuNeighborhoodKind::NearestK },
        )
    }
    .unwrap()
}

fn gpu_passes(
    r: &WgpuRuntime,
    previous: &OnceLock<parent::ParentPipelines>,
    tree: &GpuKdTree,
    batch: &GpuQueryBatch,
    k: u32,
    radius: bool,
) -> [Vec<f64>; 2] {
    let kind = if radius { GpuNeighborhoodKind::Radius } else { GpuNeighborhoodKind::NearestK };
    let old = previous.get().unwrap();
    let new = r.neighborhood_pipelines.get().unwrap();
    let pipelines = [
        if radius { &old.radius } else { &old.nearest },
        new.pipeline(r.device(), kind, k, batch.count),
    ];
    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    let mut expected = None;
    for round in 0..33 {
        for offset in 0..2 {
            let variant = (round + offset) % 2;
            let mut profiler = crate::GpuStageProfiler::new(r, 1).unwrap();
            let out = raw(r, tree, batch, k, kind, 0.5, pipelines[variant], Some(&mut profiler));
            let report = profiler.collect(r).unwrap();
            assert_eq!(report.stages.len(), 1);
            assert_eq!(report.device_to_host_bytes, 16);
            assert_eq!(report.stages[0].stage, "neighborhood-search");
            if round >= 3 {
                samples[variant].push(report.stages[0].elapsed_ns / 1_000_000.);
            }
            let words = read_words(r, &out);
            if let Some(expected) = &expected {
                assert_eq!(expected, &words);
            } else {
                expected = Some(words);
            }
        }
    }
    samples
}

fn measure(r: &WgpuRuntime, previous: &OnceLock<parent::ParentPipelines>, job: Job) {
    let (n, q, k, pattern, radius) = job;
    let (cpu, points) = fixture(n, pattern);
    let qs = queries(&points, q);
    let target = GpuKdTree::upload(r, &cpu).unwrap();
    let batch = GpuQueryBatch::upload(r, &qs).unwrap();
    let expected = oracle(
        &cpu,
        &qs,
        k,
        if radius { GpuNeighborhoodKind::Radius } else { GpuNeighborhoodKind::NearestK },
        0.5,
    );
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    r.wait_idle();
    for round in 0..33 {
        for offset in 0..4 {
            let variant = offset ^ (round % 4);
            let full = variant >= 2;
            let current = variant % 2 == 1;
            let start = Instant::now();
            let uploaded = full.then(|| {
                (
                    GpuKdTree::upload(black_box(r), black_box(&cpu)).unwrap(),
                    GpuQueryBatch::upload(r, black_box(&qs)).unwrap(),
                )
            });
            let (tree, queries) =
                uploaded.as_ref().map_or((&target, &batch), |(tree, queries)| (tree, queries));
            let out = execute(
                black_box(r),
                previous,
                black_box(tree),
                black_box(queries),
                k,
                radius,
                current,
            );
            let downloaded = if full {
                Some(out.download(r).unwrap())
            } else {
                r.wait_idle();
                None
            };
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            let downloaded = downloaded.unwrap_or_else(|| out.download(r).unwrap());
            assert_eq!(downloaded.truncated_rows(), expected.0);
            assert_eq!(downloaded.total_counts(), expected.1);
            assert_eq!(out.execution_transfers(), TransferStats::default());
            assert_eq!(
                downloaded.transfers().device_to_host_bytes(),
                q as u64 * (8 + 8 * u64::from(k))
            );
            if round >= 3 {
                samples[variant].push(elapsed);
            }
        }
    }
    let pass = gpu_passes(r, previous, &target, &batch, k, radius);
    println!("NEIGHBOR_CAPACITY_RECORD {{\"points\":{n},\"queries\":{q},\"k\":{k},\"pattern\":{pattern},\"radius\":{radius},\"changed\":{},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?},\"pass_old_ms\":{:?},\"pass_ms\":{:?}}}",!radius && k<=16 && q>=1024,samples[0],samples[1],samples[2],samples[3],pass[0],pass[1]);
}

#[test]
#[ignore = "paired raw-kernel/resident/tree-and-query-upload-search-download calls; run alone in release mode"]
fn benchmark_complete_neighborhood_capacity_calls() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    let previous = OnceLock::new();
    println!("NEIGHBOR_CAPACITY_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 routes=kernel_resident_full wall_order=balanced kernel_order=alternating checks_drops_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for job in [
        (257, 1, 1, 0, false),
        (4097, 65, 1, 0, false),
        (4097, 65, 8, 0, false),
        (4097, 65, 16, 0, false),
        (4097, 65, 17, 0, false),
        (65537, 1023, 8, 0, false),
        (65537, 1024, 8, 0, false),
        (65537, 4097, 1, 0, false),
        (65537, 4097, 2, 0, false),
        (65537, 4097, 8, 0, false),
        (65537, 4097, 16, 0, false),
        (65537, 4097, 64, 0, false),
        (262144, 16384, 1, 0, false),
        (262144, 16384, 8, 0, false),
        (262144, 16384, 16, 0, false),
        (65537, 4097, 8, 1, false),
        (65537, 4097, 8, 2, false),
        (65537, 4097, 8, 3, false),
        (4097, 65, 8, 0, true),
        (65537, 4097, 65, 0, true),
    ] {
        measure(&r, &previous, job);
    }
}

#[test]
#[ignore = "fresh application pipeline caches on a retained device; run alone in release mode"]
fn benchmark_cold_neighborhood_capacity_calls() {
    let mut r = WgpuRuntime::new_headless().unwrap();
    let (cpu, points) = fixture(4097, 0);
    let qs = queries(&points, 1025);
    let target = GpuKdTree::upload(&r, &cpu).unwrap();
    let batch = GpuQueryBatch::upload(&r, &qs).unwrap();
    println!("NEIGHBOR_CAPACITY_COLD_ENV backend={} device_type={} pairs=20 warmups=0 driver_cache_reset=false runtime_creation_and_upload_excluded=true cache_initialization_included_both=true checks_drops_excluded=true",r.adapter_info().backend,r.adapter_info().device_type);
    for (name, jobs) in [
        ("nearest_one", vec![(1, false)]),
        ("nearest_eight", vec![(8, false)]),
        ("nearest_wide", vec![(64, false)]),
        ("radius", vec![(8, true)]),
        ("mixed_nearest", vec![(1, false), (8, false), (16, false), (17, false)]),
        ("all_programs", vec![(1, false), (8, false), (16, false), (64, false), (8, true)]),
    ] {
        let expected: Vec<_> = jobs
            .iter()
            .map(|&(k, radius)| {
                oracle(
                    &cpu,
                    &qs,
                    k,
                    if radius {
                        GpuNeighborhoodKind::Radius
                    } else {
                        GpuNeighborhoodKind::NearestK
                    },
                    0.5,
                )
            })
            .collect();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        let mut counts = [0; 2];
        for round in 0..20 {
            for offset in 0..2 {
                let current = (round + offset) % 2 == 1;
                r.wait_idle();
                r.neighborhood_pipelines.take();
                let previous = OnceLock::new();
                let mut outputs = Vec::with_capacity(jobs.len());
                let start = Instant::now();
                for &(k, radius) in &jobs {
                    outputs.push(execute(&r, &previous, &target, &batch, k, radius, current));
                    r.wait_idle();
                }
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                counts[usize::from(current)] =
                    if current { populated(r.neighborhood_pipelines.get().unwrap()) } else { 2 };
                for (out, expected) in outputs.iter().zip(&expected) {
                    check_readback(out, &r, expected);
                }
                samples[usize::from(current)].push(elapsed);
            }
        }
        assert_eq!(
            counts,
            match name {
                "mixed_nearest" => [2, 4],
                "all_programs" => [2, 5],
                _ => [2, 1],
            }
        );
        println!("NEIGHBOR_CAPACITY_COLD_RECORD {{\"scenario\":\"{name}\",\"pipeline_counts\":{counts:?},\"old_ms\":{:?},\"ms\":{:?}}}",samples[0],samples[1]);
    }
}
