use super::*;
use spatialrust_search::{KdTree, NearestNeighborIndex, RadiusSearchIndex};

#[path = "neighborhood_capacity_reference.rs"]
mod parent;

fn fixture(n: usize, kind: usize) -> (KdTree, Vec<[f32; 3]>) {
    let points: Vec<_> = (0..n)
        .map(|i| {
            let j = n - i - 1;
            match kind {
                0 => [(j % 97) as f32 * 0.5, (j / 97 % 89) as f32 * 0.5, (j / 8633) as f32 * 0.5],
                1 => [j as f32 * 0.5, 0.0, 0.0],
                2 => [0.0; 3],
                _ => [((j % 5) as f32 - 2.0) * 0.5, ((j / 5 % 5) as f32 - 2.0) * 0.5, 0.0],
            }
        })
        .collect();
    let axes: [Vec<_>; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).collect());
    (KdTree::from_slices(&axes[0], &axes[1], &axes[2]), points)
}

fn queries(points: &[[f32; 3]], count: usize) -> Vec<[f32; 3]> {
    if points.is_empty() {
        return Vec::new();
    }
    (0..count)
        .map(|i| {
            let p = points[(i * 7919) % points.len()];
            if i % 3 == 0 {
                p
            } else {
                [p[0] + 0.125, p[1] - 0.0625, p[2]]
            }
        })
        .collect()
}

fn populated(cache: &NeighborhoodPipelines) -> usize {
    cache.nearest.iter().filter(|p| p.get().is_some()).count()
        + usize::from(cache.radius.get().is_some())
}

fn read_words(r: &WgpuRuntime, source: &wgpu::Buffer) -> Vec<u32> {
    let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = r.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    r.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    r.wait_idle();
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let out = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    out
}

fn oracle(
    tree: &KdTree,
    queries: &[[f32; 3]],
    capacity: u32,
    kind: GpuNeighborhoodKind,
    radius: f32,
) -> (Vec<Vec<Neighbor>>, Vec<u32>) {
    let rows: Vec<_> = queries
        .iter()
        .map(|q| {
            if kind == GpuNeighborhoodKind::NearestK {
                tree.nearest_k(q[0], q[1], q[2], capacity as usize)
            } else {
                tree.radius_search(q[0], q[1], q[2], radius)
            }
        })
        .collect();
    let totals = rows.iter().map(|row| row.len() as u32).collect();
    (
        rows.into_iter()
            .map(|mut row| {
                row.truncate(capacity as usize);
                row
            })
            .collect(),
        totals,
    )
}

fn check_readback(
    out: &GpuNeighborhoods,
    r: &WgpuRuntime,
    expected: &(Vec<Vec<Neighbor>>, Vec<u32>),
) {
    assert_eq!(out.execution_transfers(), TransferStats::default());
    let downloaded = out.download(r).unwrap();
    assert_eq!(downloaded.truncated_rows(), expected.0);
    assert_eq!(downloaded.total_counts(), expected.1);
    assert_eq!(downloaded.transfers().device_to_host_bytes(), out.readback_bytes());
    assert_eq!(
        downloaded.is_complete(),
        expected.0.iter().zip(&expected.1).all(|(a, &b)| a.len() == b as usize)
    );
}

#[allow(clippy::too_many_arguments)]
fn raw(
    r: &WgpuRuntime,
    tree: &GpuKdTree,
    batch: &GpuQueryBatch,
    capacity: u32,
    kind: GpuNeighborhoodKind,
    radius: f32,
    pipeline: &wgpu::ComputePipeline,
    profiler: Option<&mut crate::GpuStageProfiler>,
) -> wgpu::Buffer {
    let params = Params {
        count: batch.count,
        root: tree.root.unwrap(),
        capacity,
        bound: if kind == GpuNeighborhoodKind::NearestK { f32::INFINITY } else { radius * radius },
    };
    let words = batch.count as usize * (2 + 2 * capacity as usize);
    let mut data = vec![0u32; words];
    data.extend([0xdead_beef; 8]);
    let output = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let uniform = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let buffers = [
        &uniform,
        tree.nodes.as_ref().unwrap(),
        tree.points.as_ref().unwrap(),
        batch.buffer.as_ref().unwrap(),
        &output,
    ];
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
        .collect();
    let group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut encoder = r.device().create_command_encoder(&Default::default());
    let idle = u32::from(profiler.is_none());
    let mut profiler = profiler;
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("neighborhood-search"),
            timestamp_writes: crate::profiling::timestamp_writes(
                &mut profiler,
                "neighborhood-search",
            ),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        // Include an idle workgroup to probe writes beyond partial dispatches.
        pass.dispatch_workgroups(batch.count.div_ceil(64) + idle, 1, 1);
    }
    r.queue().submit(Some(encoder.finish()));
    output
}

#[test]
fn specialized_neighborhood_array_extents_cover_every_admitted_capacity() {
    let limits = wgpu::Limits::default();
    for k in 0..=64 {
        let index = nearest_variant(k, MIN_SPECIALIZED_QUERIES);
        let extent = NEAREST_CAPACITIES[index];
        assert!(extent >= k && extent <= 64);
        assert!(NEAREST_CAPACITIES[..index].iter().all(|&n| n < k));
        assert_eq!(
            plan(65, u64::from(k), GpuNeighborhoodKind::NearestK, &limits).unwrap(),
            65 * (8 + 8 * u64::from(k))
        );
    }
    for queries in [0, 1, 65, 1023, 1024, 1025, u32::MAX] {
        for k in 1..=64 {
            let index = nearest_variant(k, queries);
            assert!(NEAREST_CAPACITIES[index] >= k);
            if queries < 1024 {
                assert_eq!(index, 3);
            } else {
                assert!(NEAREST_CAPACITIES[..index].iter().all(|&n| n < k));
            }
        }
    }
    assert!(plan(1, 65, GpuNeighborhoodKind::NearestK, &limits).is_err());
    for k in [0, 1, 16, 64, 65, 1024] {
        assert!(plan(1, k, GpuNeighborhoodKind::Radius, &limits).is_ok());
    }
}

#[test]
fn specialized_neighborhood_raw_parent_cpu_words_ties_and_canaries_are_exact() {
    let r = WgpuRuntime::new_headless().unwrap();
    let current = NeighborhoodPipelines::new(r.device());
    let previous = parent::ParentPipelines::new(r.device());
    let mut checked = 0;
    let mut cases = 0;
    for n in [1, 17, 65, 513] {
        for pattern in 0..4 {
            let (cpu, points) = fixture(n, pattern);
            let qs = queries(&points, 65);
            let tree = GpuKdTree::upload(&r, &cpu).unwrap();
            let batch = GpuQueryBatch::upload(&r, &qs).unwrap();
            let jobs = (1..=64).map(|k| (k, GpuNeighborhoodKind::NearestK, 0.5)).chain(
                [0, 1, 8, 16, 65, 513].into_iter().flat_map(|k| {
                    [0.0, 0.5, f32::INFINITY]
                        .into_iter()
                        .map(move |radius| (k, GpuNeighborhoodKind::Radius, radius))
                }),
            );
            for (k, kind, radius) in jobs {
                let expected = oracle(&cpu, &qs, k, kind, radius);
                let pipes = [
                    if kind == GpuNeighborhoodKind::NearestK {
                        &previous.nearest
                    } else {
                        &previous.radius
                    },
                    current.pipeline(r.device(), kind, k, MIN_SPECIALIZED_QUERIES),
                ];
                let mut results = Vec::new();
                for pipeline in pipes {
                    let out = raw(&r, &tree, &batch, k, kind, radius, pipeline, None);
                    results.push(read_words(&r, &out));
                }
                assert_eq!(results[0], results[1], "n={n},pattern={pattern},k={k},kind={kind:?}");
                let stride = 2 + 2 * k as usize;
                assert_eq!(&results[1][65 * stride..], &[0xdead_beef; 8]);
                for ((row, expected), &total) in
                    results[1][..65 * stride].chunks_exact(stride).zip(&expected.0).zip(&expected.1)
                {
                    assert_eq!(&row[..2], &[expected.len() as u32, total]);
                    for (pair, neighbor) in row[2..].chunks_exact(2).zip(expected) {
                        assert_eq!(
                            pair,
                            [neighbor.index as u32, neighbor.distance_squared.to_bits()]
                        );
                    }
                    assert!(row[2 + 2 * expected.len()..].iter().all(|&v| v == 0));
                }
                checked += results[1].len() * 2;
                cases += 1;
            }
        }
    }
    println!("NEIGHBOR_CAPACITY_RAW checked {checked} complete parent/current/CPU u32 words in {cases} cases; every k, all array boundaries, stable ties, radius overflow/count-only, zero padding, guards and idle workgroups.");
}

#[test]
fn specialized_neighborhood_public_calls_errors_and_lazy_cache_are_exact() {
    let r = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let (cpu, points) = fixture(513, 0);
    let qs = queries(&points, 1025);
    let tree = GpuKdTree::upload(&r, &cpu).unwrap();
    let batch = GpuQueryBatch::upload(&r, &qs).unwrap();
    let empty = GpuQueryBatch::upload(&r, &[]).unwrap();
    let previous = OnceLock::new();
    let stats = r.scratch_buffer_pool_stats();
    assert!(tree.nearest_k_resident(&r, &batch, 65).is_err());
    assert!(parent::previous(
        &previous,
        &tree,
        &r,
        &batch,
        65,
        f32::INFINITY,
        GpuNeighborhoodKind::NearestK
    )
    .is_err());
    assert!(tree.nearest_k_resident(&other, &batch, 1).is_err());
    assert!(parent::previous(
        &previous,
        &tree,
        &other,
        &batch,
        1,
        f32::INFINITY,
        GpuNeighborhoodKind::NearestK
    )
    .is_err());
    for output in [
        tree.nearest_k_resident(&r, &empty, 1).unwrap(),
        tree.nearest_k_resident(&r, &batch, 0).unwrap(),
        tree.radius_neighbors_resident(&r, &batch, -1.0, 8).unwrap(),
    ] {
        assert!(output.download(&r).unwrap().truncated_rows().iter().all(Vec::is_empty));
    }
    assert!(r.neighborhood_pipelines.get().is_none());
    assert!(previous.get().is_none());
    for count in [1023usize, 1024] {
        let fresh = WgpuRuntime::new_headless().unwrap();
        let target = GpuKdTree::upload(&fresh, &cpu).unwrap();
        let qs = queries(&points, count);
        let batch = GpuQueryBatch::upload(&fresh, &qs).unwrap();
        let output = target.nearest_k_resident(&fresh, &batch, 8).unwrap();
        check_readback(&output, &fresh, &oracle(&cpu, &qs, 8, GpuNeighborhoodKind::NearestK, 0.0));
        let cache = fresh.neighborhood_pipelines.get().unwrap();
        assert_eq!(populated(cache), 1);
        assert!(cache.nearest[if count < 1024 { 3 } else { 1 }].get().is_some());
    }
    let mut rows = 0;
    for (k, count) in [(1, 1), (2, 2), (8, 2), (9, 3), (16, 3), (17, 4), (32, 4), (64, 4)] {
        let output = tree.nearest_k_resident(&r, &batch, k).unwrap();
        assert_eq!(populated(r.neighborhood_pipelines.get().unwrap()), count);
        let old = parent::previous(
            &previous,
            &tree,
            &r,
            &batch,
            k,
            f32::INFINITY,
            GpuNeighborhoodKind::NearestK,
        )
        .unwrap();
        let expected = oracle(&cpu, &qs, k, GpuNeighborhoodKind::NearestK, 0.0);
        check_readback(&output, &r, &expected);
        check_readback(&old, &r, &expected);
        assert_eq!(read_words(&r, output.buffer()), read_words(&r, old.buffer()));
        rows += qs.len();
    }
    for radius in [0.0, 0.5, f32::INFINITY] {
        for k in [0, 1, 8, 65] {
            let output = tree.radius_neighbors_resident(&r, &batch, radius, k).unwrap();
            let old = parent::previous(
                &previous,
                &tree,
                &r,
                &batch,
                k,
                radius * radius,
                GpuNeighborhoodKind::Radius,
            )
            .unwrap();
            let expected = oracle(&cpu, &qs, k, GpuNeighborhoodKind::Radius, radius);
            check_readback(&output, &r, &expected);
            check_readback(&old, &r, &expected);
            assert_eq!(read_words(&r, output.buffer()), read_words(&r, old.buffer()));
            rows += qs.len();
        }
    }
    assert_eq!(populated(r.neighborhood_pipelines.get().unwrap()), 5);
    assert_eq!(r.scratch_buffer_pool_stats(), stats);
    let extreme = KdTree::from_slices(
        &[f32::MAX, -f32::MAX, 0.0, -0.0, f32::from_bits(1)],
        &[0.0; 5],
        &[0.0; 5],
    );
    let target = GpuKdTree::upload(&r, &extreme).unwrap();
    let queries =
        GpuQueryBatch::upload(&r, &[[0.0; 3], [f32::MAX, 0.0, 0.0], [-f32::MAX, 0.0, 0.0]])
            .unwrap();
    for k in [1, 2, 8, 9, 16, 17, 64] {
        let out = target.nearest_k_resident(&r, &queries, k).unwrap();
        let old = parent::previous(
            &previous,
            &target,
            &r,
            &queries,
            k,
            f32::INFINITY,
            GpuNeighborhoodKind::NearestK,
        )
        .unwrap();
        assert_eq!(read_words(&r, out.buffer()), read_words(&r, old.buffer()));
    }
    println!("NEIGHBOR_CAPACITY_PUBLIC checked {rows} complete parent/CPU rows, exact raw buffers, receipts, empty/error-before-cache behavior, all five lazy programs and overflow/subnormal/signed-zero inputs.");
}

#[test]
fn specialized_neighborhood_concurrent_first_use_and_owned_results_survive_reuse() {
    let r = WgpuRuntime::new_headless().unwrap();
    let (cpu, points) = fixture(513, 0);
    let qs = queries(&points, 1025);
    let target = GpuKdTree::upload(&r, &cpu).unwrap();
    let batch = GpuQueryBatch::upload(&r, &qs).unwrap();
    let barrier = std::sync::Barrier::new(5);
    let outputs = std::thread::scope(|scope| {
        let jobs: Vec<_> = [1, 8, 16, 64, 0]
            .into_iter()
            .map(|k| {
                let (r, target, batch, barrier) = (&r, &target, &batch, &barrier);
                scope.spawn(move || {
                    barrier.wait();
                    (
                        k,
                        if k == 0 {
                            target.radius_neighbors_resident(r, batch, 0.5, 3)
                        } else {
                            target.nearest_k_resident(r, batch, k)
                        }
                        .unwrap(),
                    )
                })
            })
            .collect();
        jobs.into_iter().map(|t| t.join().unwrap()).collect::<Vec<_>>()
    });
    let cache = r.neighborhood_pipelines.get().unwrap();
    assert_eq!(populated(cache), 5);
    let identities: Vec<_> = cache
        .nearest
        .iter()
        .map(|p| p.get().unwrap() as *const _ as usize)
        .chain([cache.radius.get().unwrap() as *const _ as usize])
        .collect();
    drop(target);
    drop(batch);
    for budget in [crate::resource_cache::DEFAULT_GPU_POOL_BUDGET_BYTES, 4, 0] {
        r.set_scratch_buffer_pool_budget(budget);
        r.clear_scratch_buffer_pool();
        let (changed, _) = fixture(65, 2);
        let target = GpuKdTree::upload(&r, &changed).unwrap();
        let batch = GpuQueryBatch::upload(&r, &[[9999.0; 3]]).unwrap();
        for k in [1, 8, 16, 64] {
            target.nearest_k_resident(&r, &batch, k).unwrap().download(&r).unwrap();
        }
        for (k, out) in &outputs {
            check_readback(
                out,
                &r,
                &oracle(
                    &cpu,
                    &qs,
                    if *k == 0 { 3 } else { *k },
                    if *k == 0 {
                        GpuNeighborhoodKind::Radius
                    } else {
                        GpuNeighborhoodKind::NearestK
                    },
                    0.5,
                ),
            );
        }
    }
    assert_eq!(
        identities,
        cache
            .nearest
            .iter()
            .map(|p| p.get().unwrap() as *const _ as usize)
            .chain([cache.radius.get().unwrap() as *const _ as usize])
            .collect::<Vec<_>>()
    );
}

#[path = "neighborhood_capacity_benchmark.rs"]
mod benchmark;
