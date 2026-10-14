use super::super::{GpuKdTree, GpuQueryBatch, KdTree, KdTreeNode, WgpuRuntime};
use super::*;
use std::{hint::black_box, time::Instant};

#[test]
fn sample_gather_plan_reuses_feature_limits_and_checks_each_output() {
    let limits = wgpu::Limits::default();
    let max = (u64::from(limits.max_storage_buffer_binding_size) / 32)
        .min(limits.max_buffer_size / 32)
        .min(u64::from(limits.max_compute_workgroups_per_dimension) * 64);
    assert!(check_samples(0, &limits).is_ok());
    assert!(check_samples(max, &limits).is_ok());
    assert!(check_samples(max + 1, &limits).is_err());
    assert!(check_samples(u64::MAX, &limits).is_err());
    let mut narrow = limits.clone();
    narrow.max_compute_workgroups_per_dimension = 1;
    assert!(check(65, &narrow).is_ok());
    assert!(check_samples(65, &narrow).is_err());
    narrow = limits.clone();
    narrow.max_storage_buffers_per_shader_stage = 4;
    assert!(check(1, &narrow).is_ok());
    assert!(check_samples(1, &narrow).is_err());
    narrow = limits.clone();
    narrow.max_bindings_per_bind_group = 5;
    assert!(check(1, &narrow).is_ok());
    assert!(check_samples(1, &narrow).is_err());
    narrow = limits;
    narrow.max_buffer_size = 64;
    narrow.max_storage_buffer_binding_size = 64;
    assert!(check_samples(2, &narrow).is_ok());
    assert!(check(3, &narrow).is_ok());
    assert!(check_samples(3, &narrow).is_err());
}

#[test]
#[ignore = "requires native GPU adapter"]
fn fused_sample_words_keep_tree_nodes_original_queries_and_lazy_variant() {
    let r = WgpuRuntime::new_headless().unwrap();
    let (empty, queries, samples) =
        GpuKdTree::upload_with_gpu_samples(&r, &fixture(0, 0), &[]).unwrap();
    assert!(empty.points.is_none() && queries.buffer().is_none() && samples.is_none());
    assert!(r.tree_gather_pipeline.get().is_none());
    assert!(GpuKdTree::upload_with_gpu_samples(&r, &fixture(0, 0), &[[0.0, 0.0, 1.0]]).is_err());
    for (axes, order, normals) in [
        ([&[0.0][..], &[][..], &[0.0][..]], &[0][..], &[[0.0, 0.0, 1.0]][..]),
        ([&[0.0][..]; 3], &[][..], &[[0.0, 0.0, 1.0]][..]),
        ([&[0.0][..]; 3], &[0][..], &[][..]),
    ] {
        assert!(upload_with_samples(&r, axes, order, normals).is_err());
    }
    assert!(r.tree_gather_pipeline.get().is_none());
    let probe = Probe::new(&r);
    let mut words = 0;
    for n in [1, 65, 65_537, 1_048_577] {
        for profile in 0..4 {
            let tree = fixture(n, profile);
            let (x, y, z) = tree.layout().positions();
            let normals: Vec<_> =
                (0..n).map(|i| [0.6, 0.8, f32::from_bits((i as u32 & 1) << 31)]).collect();
            let expected_query: Vec<_> =
                (0..n).flat_map(|i| [x[i].to_bits(), y[i].to_bits(), z[i].to_bits(), 0]).collect();
            let expected_sample: Vec<_> = (0..n)
                .flat_map(|i| {
                    [
                        x[i].to_bits(),
                        y[i].to_bits(),
                        z[i].to_bits(),
                        0,
                        normals[i][0].to_bits(),
                        normals[i][1].to_bits(),
                        normals[i][2].to_bits(),
                        0,
                    ]
                })
                .collect();
            let old = GpuKdTree::upload(&r, &tree).unwrap();
            let (new, queries, samples) =
                GpuKdTree::upload_with_gpu_samples(&r, &tree, &normals).unwrap();
            assert_eq!(new.root, old.root);
            assert_eq!(
                new.upload_transfers().host_to_device_bytes(),
                old.upload_transfers().host_to_device_bytes()
            );
            assert_eq!(new.upload_transfers().device_to_device_bytes(), n as u64 * 16);
            assert_eq!(queries.upload_transfers().device_to_device_bytes(), n as u64 * 16);
            drop(tree);
            drop(normals);
            assert_eq!(
                probe.read(&r, new.points.as_ref().unwrap()),
                probe.read(&r, old.points.as_ref().unwrap())
            );
            assert_eq!(
                probe.read(&r, new.nodes.as_ref().unwrap()),
                probe.read(&r, old.nodes.as_ref().unwrap())
            );
            assert_eq!(probe.read(&r, queries.buffer().unwrap()), expected_query);
            assert_eq!(probe.read(&r, samples.as_ref().unwrap()), expected_sample);
            words += n * 16 + new.nodes.as_ref().unwrap().size() as usize / 4;
        }
    }
    let cache = r.tree_gather_pipeline.get().unwrap();
    assert!(cache.pipelines[0].get().is_none() && cache.pipelines[1].get().is_none());
    assert!(cache.pipelines[2].get().is_some());
    println!("FPFH_FUSED_TREE_WORDS {words}");
}

#[test]
fn gather_plan_checks_capabilities_bytes_and_capped_grid_stride() {
    let limits = wgpu::Limits::default();
    assert_eq!(plan(0, &limits).unwrap(), (0, 0));
    assert_eq!(plan(1, &limits).unwrap(), (16, 1));
    assert_eq!(plan(65, &limits).unwrap(), (1040, 2));
    let mut narrow = limits.clone();
    narrow.max_compute_workgroups_per_dimension = 1;
    assert_eq!(plan(100_003, &narrow).unwrap(), (1_600_048, 1));
    let max = u64::from(limits.max_storage_buffer_binding_size) / 16;
    assert!(plan(max, &limits).is_ok());
    assert!(plan(max + 1, &limits).is_err());
    assert!(plan(u64::MAX, &limits).is_err());
    for (field, value) in [(0, 3), (1, 4), (2, 0), (3, 63), (4, 63), (5, 0), (6, 15), (7, 0)] {
        let mut bad = limits.clone();
        match field {
            0 => bad.max_storage_buffers_per_shader_stage = value,
            1 => bad.max_bindings_per_bind_group = value,
            2 => bad.max_compute_workgroups_per_dimension = value,
            3 => bad.max_compute_workgroup_size_x = value,
            4 => bad.max_compute_invocations_per_workgroup = value,
            5 => bad.max_uniform_buffers_per_shader_stage = value,
            6 => bad.max_uniform_buffer_binding_size = value,
            _ => bad.max_bind_groups = value,
        }
        assert!(plan(1, &bad).is_err());
    }
}

fn fixture(n: usize, profile: usize) -> KdTree {
    let axes: [Vec<_>; 3] = std::array::from_fn(|axis| {
        (0..n)
            .map(|i| match profile {
                0 => ((i * 17 + axis * 13) % 100_003) as f32 * 0.125 - 6000.0,
                1 => f32::from_bits(((i + axis) as u32 & 1) << 31),
                2 => f32::from_bits((i as u32 % 23) | (((i + axis) as u32 & 1) << 31)),
                _ => f32::from_bits(
                    (i as u32).wrapping_mul(0x9e37_79b9).rotate_left(axis as u32 * 7) & 0xff7f_ffff,
                ),
            })
            .collect()
    });
    KdTree::from_slices(&axes[0], &axes[1], &axes[2])
}

#[test]
#[ignore = "requires isolated native release measurement"]
fn completed_cpu_and_gpu_tree_uploads() {
    let r = WgpuRuntime::new_headless().unwrap();
    let info = r.adapter_info();
    println!(
        "TREE_GPU_ENV backend={} device_type={} available_workers={}",
        info.backend,
        info.device_type,
        std::thread::available_parallelism().map_or(1, |n| n.get())
    );
    for (n, profile) in [
        (1, 0),
        (4097, 0),
        (65_537, 0),
        (262_145, 0),
        (524_289, 0),
        (1_048_577, 0),
        (1_048_577, 1),
        (1_048_577, 2),
    ] {
        let tree = fixture(n, profile);
        let nodes = tree.layout().nodes().len();
        let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..4 {
                let route = (round + offset) % 4;
                let start = Instant::now();
                let (target, queries) = match route {
                    0 => (GpuKdTree::upload(&r, black_box(&tree)).unwrap(), None),
                    1 => (GpuKdTree::upload_with_gpu_gather(&r, black_box(&tree)).unwrap(), None),
                    2 => {
                        let target = GpuKdTree::upload(&r, black_box(&tree)).unwrap();
                        let (x, y, z) = tree.layout().positions();
                        (target, Some(GpuQueryBatch::upload_soa(&r, [x, y, z]).unwrap()))
                    }
                    _ => {
                        let (target, queries) =
                            GpuKdTree::upload_and_queries_with_gpu_gather(&r, black_box(&tree))
                                .unwrap();
                        (target, Some(queries))
                    }
                };
                r.wait_idle();
                let receipt = target.upload_transfers();
                let query_receipt =
                    queries.as_ref().map(|q| q.upload_transfers()).unwrap_or_default();
                drop(target);
                drop(queries);
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(receipt.host_to_device_bytes(), n as u64 * 16 + nodes as u64 * 32);
                assert_eq!(receipt.device_to_host_bytes(), 0);
                assert_eq!(
                    receipt.device_to_device_bytes(),
                    if route % 2 == 0 { 0 } else { n as u64 * 16 }
                );
                assert_eq!(
                    query_receipt.host_to_device_bytes(),
                    if route == 2 { n as u64 * 16 } else { 0 }
                );
                assert_eq!(
                    query_receipt.device_to_device_bytes(),
                    if route == 3 { n as u64 * 16 } else { 0 }
                );
                assert_eq!(query_receipt.device_to_host_bytes(), 0);
                if round >= 3 {
                    samples[route].push(ms);
                }
            }
        }
        println!("TREE_GPU_RECORD {{\"points\":{n},\"profile\":{profile},\"nodes\":{nodes},\"tree_old_ms\":{:?},\"tree_gpu_ms\":{:?},\"pair_old_ms\":{:?},\"pair_gpu_ms\":{:?}}}",samples[0],samples[1],samples[2],samples[3]);
    }
}

struct Probe {
    pipeline: wgpu::ComputePipeline,
}
impl Probe {
    fn new(r: &WgpuRuntime) -> Self {
        let shader = r.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("feature-upload-word-probe"),
            source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage,read> src:array<u32>; @group(0) @binding(1) var<storage,read_write> dst:array<u32>; @compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>){for(var i=id.x;i<arrayLength(&src);i+=groups.x*64u){dst[i]=src[i];}}".into()),
        });
        Self {
            pipeline: r.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("probe"),
                compilation_options: Default::default(),
                cache: None,
            }),
        }
    }
    fn read(&self, r: &WgpuRuntime, source: &wgpu::Buffer) -> Vec<u32> {
        let bytes = source.size();
        let output = r.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: source.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
            ],
        });
        let mut encoder = r.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((bytes as u32 / 4).div_ceil(64).min(256), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, bytes);
        r.queue().submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        r.device().poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let words = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        words
    }
}

#[test]
#[ignore = "requires native GPU adapter"]
fn integer_gather_preserves_all_point_node_words_receipts_and_cpu_drop() {
    let r = WgpuRuntime::new_headless().unwrap();
    assert!(r.tree_gather_pipeline.get().is_none());
    let empty = GpuKdTree::upload_with_gpu_gather(&r, &fixture(0, 0)).unwrap();
    assert!(empty.points.is_none() && empty.nodes.is_none());
    assert_eq!(empty.upload_transfers(), spatialrust_core::TransferStats::default());
    assert!(r.tree_gather_pipeline.get().is_none());
    let probe = Probe::new(&r);
    let mut words = 0;
    for n in [1, 17, 63, 64, 65, 257, 65_537, 524_287, 524_288, 524_289, 1_048_577, 4_194_241] {
        for profile in 0..4 {
            // One point beyond the capped 65,535*64 first wave exercises the
            // second grid-stride iteration on the actual production dispatch.
            if n == 4_194_241 && profile != 0 {
                continue;
            }
            let tree = fixture(n, profile);
            let layout = tree.layout();
            let (x, y, z) = layout.positions();
            let expected: Vec<u32> = layout
                .point_order()
                .iter()
                .flat_map(|&index| {
                    let i = index as usize;
                    [x[i].to_bits(), y[i].to_bits(), z[i].to_bits(), index]
                })
                .collect();
            let nodes: Vec<u32> = layout
                .nodes()
                .flat_map(|node| match node {
                    KdTreeNode::Leaf { start, end } => [0, u32::MAX, 0, 0, start, end, 0, 0],
                    KdTreeNode::Branch { axis, split, left, right } => {
                        [split.to_bits(), u32::from(axis), left, right, 0, 0, 0, 0]
                    }
                })
                .collect();
            let old = GpuKdTree::upload(&r, &tree).unwrap();
            let new = GpuKdTree::upload_with_gpu_gather(&r, &tree).unwrap();
            assert_eq!(old.root, new.root);
            assert_eq!(new.root, layout.root());
            assert_eq!(
                new.upload_transfers().host_to_device_bytes(),
                old.upload_transfers().host_to_device_bytes()
            );
            assert_eq!(
                new.upload_transfers().host_to_device_bytes(),
                n as u64 * 16 + nodes.len() as u64 * 4
            );
            assert_eq!(new.upload_transfers().device_to_device_bytes(), n as u64 * 16);
            assert_eq!(new.upload_transfers().device_to_host_bytes(), 0);
            drop(tree);
            assert_eq!(probe.read(&r, old.points.as_ref().unwrap()), expected);
            assert_eq!(probe.read(&r, new.points.as_ref().unwrap()), expected);
            assert_eq!(probe.read(&r, old.nodes.as_ref().unwrap()), nodes);
            assert_eq!(probe.read(&r, new.nodes.as_ref().unwrap()), nodes);
            words += expected.len() + nodes.len();
        }
    }
    println!("TREE_GPU_WORDS {words}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn gather_searches_keep_ties_radius_order_runtime_checks_and_owned_results() {
    let r = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let mut words = 0;
    for profile in 0..3 {
        let tree = fixture(65_537, profile);
        let layout = tree.layout();
        let (x, y, z) = layout.positions();
        let positions: Vec<_> =
            [0, 1, 16, 32768, 65536].iter().map(|&i| [x[i], y[i], z[i]]).collect();
        let old = GpuKdTree::upload(&r, &tree).unwrap();
        let new = GpuKdTree::upload_with_gpu_gather(&r, &tree).unwrap();
        drop(tree);
        let queries = GpuQueryBatch::upload(&r, &positions).unwrap();
        assert!(new.nearest_k_resident(&other, &queries, 1).is_err());
        let mut results = Vec::new();
        for k in [1, 8, 64] {
            results.push((
                old.nearest_k_resident(&r, &queries, k).unwrap(),
                new.nearest_k_resident(&r, &queries, k).unwrap(),
            ));
        }
        for radius in [0.0, 0.125, f32::MAX] {
            results.push((
                old.radius_neighbors_resident(&r, &queries, radius, 16).unwrap(),
                new.radius_neighbors_resident(&r, &queries, radius, 16).unwrap(),
            ));
        }
        drop(old);
        drop(new);
        drop(queries);
        for (a, b) in results {
            let av = probe.read(&r, a.buffer());
            let bv = probe.read(&r, b.buffer());
            assert_eq!(av, bv);
            words += av.len();
            assert_eq!(a.execution_transfers(), b.execution_transfers());
            let ar = a.download(&r).unwrap();
            let br = b.download(&r).unwrap();
            assert_eq!(ar.truncated_rows(), br.truncated_rows());
            assert_eq!(ar.total_counts(), br.total_counts());
            assert_eq!(ar.complete_rows().is_ok(), br.complete_rows().is_ok());
            assert_eq!(ar.transfers(), br.transfers());
        }
    }
    println!("TREE_GPU_RESULT_WORDS {words}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn finite_errors_reject_before_pipeline_creation_and_queued_cold_uploads_are_independent() {
    let r = WgpuRuntime::new_headless().unwrap();
    r.device().push_error_scope(wgpu::ErrorFilter::Validation);
    for axis in 0..3 {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut axes = [vec![0.0; 524_289], vec![0.0; 524_289], vec![0.0; 524_289]];
            axes[axis][524_288] = bad;
            let tree = KdTree::from_slices(&axes[0], &axes[1], &axes[2]);
            assert_eq!(
                GpuKdTree::upload(&r, &tree).err().unwrap(),
                GpuKdTree::upload_with_gpu_gather(&r, &tree).err().unwrap()
            );
            assert_eq!(
                GpuKdTree::upload(&r, &tree).err().unwrap(),
                GpuKdTree::upload_and_queries_with_gpu_gather(&r, &tree).err().unwrap()
            );
            assert!(r.tree_gather_pipeline.get().is_none());
        }
    }
    for with_queries in [false, true] {
        assert!(upload(&r, [&[0.0], &[], &[0.0]], &[0], with_queries).is_err());
        assert!(upload(&r, [&[0.0], &[0.0], &[0.0]], &[], with_queries).is_err());
    }
    r.wait_idle();
    assert!(pollster::block_on(r.device().pop_error_scope()).is_none());
    let trees = [fixture(4097, 0), fixture(4097, 1), fixture(4097, 2)];
    let targets = std::thread::scope(|scope| {
        let runtime = &r;
        trees
            .iter()
            .map(|tree| {
                scope.spawn(move || {
                    GpuKdTree::upload_and_queries_with_gpu_gather(runtime, tree).unwrap()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    let probe = Probe::new(&r);
    for (tree, (target, queries)) in trees.iter().zip(targets) {
        let old = GpuKdTree::upload(&r, tree).unwrap();
        assert_eq!(
            probe.read(&r, old.points.as_ref().unwrap()),
            probe.read(&r, target.points.as_ref().unwrap())
        );
        let (x, y, z) = tree.layout().positions();
        let expected: Vec<_> = (0..x.len())
            .flat_map(|i| [x[i].to_bits(), y[i].to_bits(), z[i].to_bits(), 0])
            .collect();
        assert_eq!(probe.read(&r, queries.buffer().unwrap()), expected);
    }
    let cache = r.tree_gather_pipeline.get().unwrap();
    assert!(cache.pipelines[0].get().is_none());
    assert!(cache.pipelines[1].get().is_some());
}

#[test]
#[ignore = "requires native GPU adapter"]
fn shared_upload_preserves_original_query_bits_counts_and_empty_ownership() {
    let r = WgpuRuntime::new_headless().unwrap();
    let (empty, queries) =
        GpuKdTree::upload_and_queries_with_gpu_gather(&r, &fixture(0, 0)).unwrap();
    assert!(empty.points.is_none() && queries.buffer().is_none());
    assert_eq!(queries.query_count(), 0);
    assert_eq!(empty.upload_transfers(), queries.upload_transfers());
    assert!(r.tree_gather_pipeline.get().is_none());
    let probe = Probe::new(&r);
    let mut words = 0;
    for n in [1, 63, 64, 65, 257, 65_537, 524_289, 1_048_577] {
        for profile in 0..4 {
            let tree = fixture(n, profile);
            let (x, y, z) = tree.layout().positions();
            let expected: Vec<_> =
                (0..n).flat_map(|i| [x[i].to_bits(), y[i].to_bits(), z[i].to_bits(), 0]).collect();
            let old_queries = GpuQueryBatch::upload_soa(&r, [x, y, z]).unwrap();
            let old = GpuKdTree::upload(&r, &tree).unwrap();
            let (new, queries) = GpuKdTree::upload_and_queries_with_gpu_gather(&r, &tree).unwrap();
            assert_eq!(queries.query_count(), n as u32);
            assert_eq!(queries.upload_transfers().host_to_device_bytes(), 0);
            assert_eq!(queries.upload_transfers().device_to_device_bytes(), n as u64 * 16);
            assert_eq!(queries.upload_transfers().device_to_host_bytes(), 0);
            assert_eq!(
                new.upload_transfers().host_to_device_bytes(),
                old.upload_transfers().host_to_device_bytes()
            );
            assert_eq!(new.upload_transfers().device_to_device_bytes(), n as u64 * 16);
            drop(tree);
            assert_eq!(probe.read(&r, queries.buffer().unwrap()), expected);
            assert_eq!(probe.read(&r, old_queries.buffer().unwrap()), expected);
            assert_eq!(
                probe.read(&r, new.points.as_ref().unwrap()),
                probe.read(&r, old.points.as_ref().unwrap())
            );
            assert_eq!(
                probe.read(&r, new.nodes.as_ref().unwrap()),
                probe.read(&r, old.nodes.as_ref().unwrap())
            );
            words += expected.len();
        }
    }
    println!("TREE_GPU_QUERY_WORDS {words}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn shared_queries_keep_neighborhood_results_runtime_identity_and_drop_safety() {
    let r = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let mut words = 0;
    for profile in 0..3 {
        let tree = fixture(257, profile);
        let (x, y, z) = tree.layout().positions();
        let old_queries = GpuQueryBatch::upload_soa(&r, [x, y, z]).unwrap();
        let old = GpuKdTree::upload(&r, &tree).unwrap();
        let (new, queries) = GpuKdTree::upload_and_queries_with_gpu_gather(&r, &tree).unwrap();
        drop(tree);
        assert!(new.nearest_k_resident(&other, &queries, 1).is_err());
        let foreign = GpuQueryBatch::upload(&other, &[[0.0; 3]]).unwrap();
        assert!(new.nearest_k_resident(&r, &foreign, 1).is_err());
        let mut results = Vec::new();
        for k in [1, 8, 64] {
            results.push((
                old.nearest_k_resident(&r, &old_queries, k).unwrap(),
                new.nearest_k_resident(&r, &queries, k).unwrap(),
            ));
        }
        for radius in [0.0, 0.125, f32::MAX] {
            results.push((
                old.radius_neighbors_resident(&r, &old_queries, radius, 16).unwrap(),
                new.radius_neighbors_resident(&r, &queries, radius, 16).unwrap(),
            ));
        }
        drop(old);
        drop(old_queries);
        drop(new);
        drop(queries);
        for (a, b) in results {
            let av = probe.read(&r, a.buffer());
            assert_eq!(av, probe.read(&r, b.buffer()));
            words += av.len();
            let ar = a.download(&r).unwrap();
            let br = b.download(&r).unwrap();
            assert_eq!(ar.truncated_rows(), br.truncated_rows());
            assert_eq!(ar.total_counts(), br.total_counts());
            assert_eq!(ar.complete_rows().is_ok(), br.complete_rows().is_ok());
            assert_eq!(ar.transfers(), br.transfers());
        }
    }
    println!("TREE_GPU_SHARED_RESULT_WORDS {words}");
}
