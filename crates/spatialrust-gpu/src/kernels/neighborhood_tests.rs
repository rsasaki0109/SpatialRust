use super::*;
use spatialrust_search::{KdTree, NearestNeighborIndex, RadiusSearchIndex};

fn fixture(n: usize, kind: usize) -> (KdTree, Vec<[f32; 3]>) {
    let points: Vec<_> = (0..n)
        .map(|i| {
            let j = n - i - 1;
            match kind {
                0 => [(j % 19) as f32 * 0.5, (j / 19 % 19) as f32 * 0.5, (j / 361) as f32 * 0.5],
                1 => [j as f32 * 0.5, 0.0, 0.0],
                2 => [0.0; 3],
                _ => [((j % 5) as f32 - 2.0) * 0.5, ((j / 5 % 5) as f32 - 2.0) * 0.5, 0.0],
            }
        })
        .collect();
    let xyz: [Vec<_>; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).collect());
    (KdTree::from_slices(&xyz[0], &xyz[1], &xyz[2]), points)
}

#[test]
fn neighborhood_plan_bounds_capacity_buffers_host_and_dispatch() {
    let limits = wgpu::Limits::default();
    assert_eq!(plan(0, 64, GpuNeighborhoodKind::NearestK, &limits).unwrap(), 0);
    assert_eq!(plan(1, 64, GpuNeighborhoodKind::NearestK, &limits).unwrap(), 520);
    assert!(plan(1, 65, GpuNeighborhoodKind::NearestK, &limits).is_err());
    assert_eq!(plan(1, 65, GpuNeighborhoodKind::Radius, &limits).unwrap(), 528);
    assert!(plan(0, u64::from(u32::MAX), GpuNeighborhoodKind::Radius, &limits).is_err());
    assert!(plan(u64::MAX, 1, GpuNeighborhoodKind::Radius, &limits).is_err());
    let max_queries = u64::from(limits.max_compute_workgroups_per_dimension) * 64;
    assert!(plan(max_queries, 0, GpuNeighborhoodKind::Radius, &limits).is_ok());
    assert!(plan(max_queries + 1, 0, GpuNeighborhoodKind::Radius, &limits).is_err());
    let rows = u64::from(limits.max_storage_buffer_binding_size) / 520;
    assert!(plan(rows, 64, GpuNeighborhoodKind::NearestK, &limits).is_ok());
    assert!(plan(rows + 1, 64, GpuNeighborhoodKind::NearestK, &limits).is_err());
    let mut restricted = limits.clone();
    restricted.max_compute_workgroup_size_x = 63;
    assert!(plan(1, 1, GpuNeighborhoodKind::NearestK, &restricted).is_err());
    restricted = limits.clone();
    restricted.max_storage_buffers_per_shader_stage = 3;
    assert!(plan(1, 1, GpuNeighborhoodKind::NearestK, &restricted).is_err());
}

#[test]
fn gpu_knn_retains_cpu_membership_stable_ties_and_independent_distances() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut verified = 0;
    for n in [1, 16, 17, 63, 64, 65, 255, 257, 513] {
        for kind in 0..4 {
            let (tree, points) = fixture(n, kind);
            let target = GpuKdTree::upload(&runtime, &tree).unwrap();
            let queries: Vec<_> = points
                .iter()
                .step_by((n / 33).max(1))
                .flat_map(|&p| [p, [p[0] + 0.125, p[1] - 0.0625, p[2]], [p[0] + 0.25, p[1], p[2]]])
                .collect();
            let batch = GpuQueryBatch::upload(&runtime, &queries).unwrap();
            assert_eq!(batch.upload_transfers().host_to_device_bytes(), queries.len() as u64 * 16);
            assert_eq!(batch.upload_transfers().device_to_host_bytes(), 0);
            for k in [0, 1, 2, 8, 16, 32, 64] {
                let expected: Vec<_> =
                    queries.iter().map(|q| tree.nearest_k(q[0], q[1], q[2], k)).collect();
                let output = target.nearest_k_resident(&runtime, &batch, k as u32).unwrap();
                assert_eq!(output.execution_transfers(), TransferStats::default());
                assert_eq!(output.kind(), GpuNeighborhoodKind::NearestK);
                let downloaded = output.download(&runtime).unwrap();
                assert_eq!(
                    downloaded.complete_rows().unwrap(),
                    expected,
                    "n={n}, kind={kind}, k={k}"
                );
                assert_eq!(
                    downloaded.transfers().device_to_host_bytes(),
                    queries.len() as u64 * (8 + 8 * k as u64)
                );
                for (q, row) in queries.iter().zip(downloaded.complete_rows().unwrap()) {
                    let mut distances: Vec<_> = points
                        .iter()
                        .map(|p| {
                            let dx = p[0] - q[0];
                            let dy = p[1] - q[1];
                            let dz = p[2] - q[2];
                            dx * dx + dy * dy + dz * dz
                        })
                        .collect();
                    distances.sort_by(f32::total_cmp);
                    distances.truncate(k);
                    assert_eq!(
                        row.iter().map(|r| r.distance_squared).collect::<Vec<_>>(),
                        distances
                    );
                    verified += row.len();
                }
            }
        }
    }
    eprintln!("GPU_NEIGHBORHOOD verified {verified} selected neighbors against CPU ordering and brute distances");
}

#[test]
fn gpu_radius_retains_cpu_order_exact_counts_and_explicit_overflow() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut verified = 0;
    for n in [1, 17, 65, 257, 513] {
        for kind in 0..4 {
            let (tree, points) = fixture(n, kind);
            let target = GpuKdTree::upload(&runtime, &tree).unwrap();
            let queries: Vec<_> =
                points.iter().step_by((n / 17).max(1)).copied().chain([[500.0; 3]]).collect();
            let batch = GpuQueryBatch::upload(&runtime, &queries).unwrap();
            for radius in [
                -1.0,
                f32::NAN,
                0.0,
                f32::from_bits(1.0_f32.to_bits() - 1),
                1.0,
                f32::from_bits(1.0_f32.to_bits() + 1),
                f32::INFINITY,
            ] {
                let expected: Vec<_> =
                    queries.iter().map(|q| tree.radius_search(q[0], q[1], q[2], radius)).collect();
                for capacity in [0, 3, 64, n] {
                    let output = target
                        .radius_neighbors_resident(&runtime, &batch, radius, capacity as u32)
                        .unwrap();
                    let downloaded = output.download(&runtime).unwrap();
                    let truncated: Vec<_> = expected
                        .iter()
                        .map(|row| row[..row.len().min(capacity)].to_vec())
                        .collect();
                    assert_eq!(
                        downloaded.truncated_rows(),
                        truncated,
                        "n={n}, kind={kind}, radius={radius}, cap={capacity}"
                    );
                    assert_eq!(
                        downloaded.total_counts(),
                        expected.iter().map(|r| r.len() as u32).collect::<Vec<_>>()
                    );
                    let complete = expected.iter().all(|r| r.len() <= capacity);
                    assert_eq!(downloaded.is_complete(), complete);
                    assert_eq!(downloaded.complete_rows().is_ok(), complete);
                    assert_eq!(downloaded.into_complete_rows().is_ok(), complete);
                    for (q, row) in queries.iter().zip(&expected) {
                        let count = points
                            .iter()
                            .filter(|p| {
                                let dx = p[0] - q[0];
                                let dy = p[1] - q[1];
                                let dz = p[2] - q[2];
                                radius >= 0.0 && dx * dx + dy * dy + dz * dz <= radius * radius
                            })
                            .count();
                        assert_eq!(row.len(), count);
                        verified += row.len().min(capacity);
                    }
                }
            }
        }
    }
    eprintln!(
        "GPU_NEIGHBORHOOD verified {verified} radius entries with exact counts and overflow guards"
    );
}

#[test]
fn gpu_neighborhood_owns_live_outputs_and_reuses_queries_across_callers() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let (tree, points) = fixture(4097, 0);
    let target = GpuKdTree::upload(&runtime, &tree).unwrap();
    let batch = GpuQueryBatch::upload(&runtime, &points).unwrap();
    let expected: Vec<_> = points.iter().map(|q| tree.nearest_k(q[0], q[1], q[2], 16)).collect();
    let live = target.nearest_k_resident(&runtime, &batch, 16).unwrap();
    drop(tree);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let (runtime, target, batch, expected) = (&runtime, &target, &batch, &expected);
            scope.spawn(move || {
                for _ in 0..2 {
                    let out = target.nearest_k_resident(runtime, batch, 16).unwrap();
                    assert_eq!(out.download(runtime).unwrap().complete_rows().unwrap(), expected);
                }
            });
        }
    });
    let other = target.radius_neighbors_resident(&runtime, &batch, 0.0, 1).unwrap();
    assert_ne!(live.buffer(), other.buffer());
    drop(target);
    drop(batch);
    assert_eq!(live.download(&runtime).unwrap().complete_rows().unwrap(), expected);
    assert_eq!(other.download(&runtime).unwrap().total_counts(), vec![1; points.len()]);
}

#[test]
fn gpu_neighborhood_validates_empty_inputs_runtime_and_metadata() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let tree = KdTree::from_slices(&[f32::MAX, -f32::MAX], &[0.0; 2], &[0.0; 2]);
    let target = GpuKdTree::upload(&runtime, &tree).unwrap();
    let batch = GpuQueryBatch::upload(&runtime, &[[0.0; 3]]).unwrap();
    for kind in [GpuNeighborhoodKind::NearestK, GpuNeighborhoodKind::Radius] {
        let output = if kind == GpuNeighborhoodKind::NearestK {
            target.nearest_k_resident(&runtime, &batch, 64)
        } else {
            target.radius_neighbors_resident(&runtime, &batch, f32::INFINITY, 64)
        }
        .unwrap();
        let rows = output.download(&runtime).unwrap().into_complete_rows().unwrap();
        let expected = if kind == GpuNeighborhoodKind::NearestK {
            tree.nearest_k(0.0, 0.0, 0.0, 64)
        } else {
            tree.radius_search(0.0, 0.0, 0.0, f32::INFINITY)
        };
        assert_eq!(rows, vec![expected]);
    }
    let empty = GpuQueryBatch::upload(&runtime, &[]).unwrap();
    let output = target.nearest_k_resident(&runtime, &empty, 64).unwrap();
    assert_eq!(output.readback_bytes(), 0);
    assert_eq!(output.download(&runtime).unwrap().transfers(), TransferStats::default());
    let empty_target = GpuKdTree::upload(&runtime, &KdTree::from_slices(&[], &[], &[])).unwrap();
    let empty_rows = empty_target
        .radius_neighbors_resident(&runtime, &batch, 1.0, 0)
        .unwrap()
        .download(&runtime)
        .unwrap();
    assert!(empty_rows.is_complete());
    assert_eq!(empty_rows.total_counts(), [0]);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(GpuQueryBatch::upload(&runtime, &[[value, 0.0, 0.0]]).is_err());
    }
    assert!(target.nearest_k_resident(&runtime, &empty, 65).is_err());
    let other = WgpuRuntime::new_headless().unwrap();
    let other_queries = GpuQueryBatch::upload(&other, &[]).unwrap();
    assert!(target.nearest_k_resident(&runtime, &other_queries, 0).is_err());
    assert!(target.radius_neighbors_resident(&other, &empty, 0.0, 0).is_err());
    assert!(output.download(&other).is_err());
    // A downstream kernel may corrupt externally exposed storage. Reject bad
    // headers instead of trusting them for host allocation/indexing.
    let malformed = GpuNeighborhoods {
        buffer: runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[9_u32, 1, 0, 0]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        }),
        queries: 1,
        capacity: 1,
        bytes: 16,
        kind: GpuNeighborhoodKind::NearestK,
        identity: runtime.profiling_identity(),
    };
    assert!(malformed.download(&runtime).is_err());
}

#[test]
fn downstream_gpu_kernel_consumes_resident_counts_and_indices_without_readback() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let (tree, points) = fixture(257, 2);
    let target = GpuKdTree::upload(&runtime, &tree).unwrap();
    let queries = GpuQueryBatch::upload(&runtime, &points[..65]).unwrap();
    let neighborhoods = target.radius_neighbors_resident(&runtime, &queries, 0.0, 3).unwrap();
    assert_eq!(neighborhoods.execution_transfers(), TransferStats::default());
    let shader = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("neighborhood-consumer"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
            struct Params { count: u32, capacity: u32, pad0: u32, pad1: u32 }
            @group(0) @binding(0) var<uniform> params: Params;
            @group(0) @binding(1) var<storage, read> neighbors: array<u32>;
            @group(0) @binding(2) var<storage, read_write> output: array<u32>;
            @compute @workgroup_size(64)
            fn consume(@builtin(global_invocation_id) id: vec3<u32>) {
                if id.x >= params.count { return; }
                let base = id.x * (2u + 2u * params.capacity);
                output[3u * id.x] = neighbors[base];
                output[3u * id.x + 1u] = neighbors[base + 1u];
                output[3u * id.x + 2u] = neighbors[base + 2u];
            }
        "#
            .into(),
        ),
    });
    let pipeline = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("neighborhood-consumer"),
        layout: None,
        module: &shader,
        entry_point: Some("consume"),
        compilation_options: Default::default(),
        cache: None,
    });
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[65_u32, 3, 0, 0]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 65 * 12,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: neighborhoods.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
        ],
    });
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 65 * 12,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(2, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 65 * 12);
    runtime.queue().submit(Some(encoder.finish()));
    drop(neighborhoods);
    drop(queries);
    drop(target);
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    runtime.device().poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&mapped);
    let first = tree.radius_search(0.0, 0.0, 0.0, 0.0)[0].index as u32;
    for row in words.chunks_exact(3) {
        assert_eq!(row, [3, 257, first]);
    }
    drop(mapped);
    staging.unmap();
}

#[test]
fn large_partial_dispatch_preserves_original_indices_and_zero_padding() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let (tree, points) = fixture(65537, 0);
    let target = GpuKdTree::upload(&runtime, &tree).unwrap();
    let queries = GpuQueryBatch::upload(&runtime, &points).unwrap();
    for kind in [GpuNeighborhoodKind::NearestK, GpuNeighborhoodKind::Radius] {
        let output = if kind == GpuNeighborhoodKind::NearestK {
            target.nearest_k_resident(&runtime, &queries, 1)
        } else {
            target.radius_neighbors_resident(&runtime, &queries, 0.0, 1)
        }
        .unwrap();
        let readback = output.download(&runtime).unwrap();
        assert_eq!(readback.total_counts(), vec![1; points.len()]);
        for (i, row) in readback.complete_rows().unwrap().iter().enumerate() {
            assert_eq!(row, &[Neighbor { index: i, distance_squared: 0.0 }]);
        }
    }
    // Target smaller than k: unused result slots must remain initialized zero.
    let tiny = GpuKdTree::upload(&runtime, &KdTree::from_slices(&[0.0], &[0.0], &[0.0])).unwrap();
    let query = GpuQueryBatch::upload(&runtime, &[[0.0; 3]]).unwrap();
    let output = tiny.nearest_k_resident(&runtime, &query, 64).unwrap();
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output.bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(output.buffer(), 0, &staging, 0, output.bytes);
    runtime.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    runtime.device().poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&mapped);
    assert_eq!(&words[..4], &[1, 1, 0, 0]);
    assert!(words[4..].iter().all(|&word| word == 0));
    drop(mapped);
    staging.unmap();
}
