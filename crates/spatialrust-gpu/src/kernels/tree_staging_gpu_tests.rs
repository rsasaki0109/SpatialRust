// Frozen upload from main d6b7558; method and concrete constructor names only.
use super::super::super::*;
use super::{fixture, previous_prepare};

pub(super) fn previous_upload(runtime: &WgpuRuntime, tree: &KdTree) -> SpatialResult<GpuKdTree> {
    let layout = tree.layout();
    let (x, y, z) = layout.positions();
    let limits = runtime.device().limits();
    let point_bytes = buffer_bytes(x.len() as u64, 16, &limits)?;
    let node_bytes = buffer_bytes(layout.nodes().len() as u64, 32, &limits)?;
    check_compute(&limits, 0)?;
    if x.iter().chain(y).chain(z).any(|v| !v.is_finite()) {
        return Err(invalid());
    }
    let mut target = GpuKdTree {
        nodes: None,
        points: None,
        root: layout.root(),
        identity: runtime.profiling_identity(),
        upload_transfers: TransferStats::default(),
    };
    if x.is_empty() {
        return Ok(target);
    }
    let nodes: Vec<_> = layout
        .nodes()
        .map(|node| {
            let mut output = Node::zeroed();
            match node {
                KdTreeNode::Leaf { start, end } => {
                    output.axis = u32::MAX;
                    output.start = start;
                    output.end = end;
                }
                KdTreeNode::Branch { axis, split, left, right } => {
                    output.axis = u32::from(axis);
                    output.split = split;
                    output.left = left;
                    output.right = right;
                }
            }
            output
        })
        .collect();
    let points: Vec<_> = layout
        .point_order()
        .iter()
        .map(|&index| {
            let i = index as usize;
            Point { xyz: [x[i], y[i], z[i]], index }
        })
        .collect();
    let upload = |label, data: &[u8]| {
        runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    target.nodes = Some(upload("nearest-nodes", bytemuck::cast_slice(&nodes)));
    target.points = Some(upload("nearest-points", bytemuck::cast_slice(&points)));
    target.upload_transfers.record(TransferDirection::HostToDevice, point_bytes + node_bytes);
    Ok(target)
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

fn node_words(tree: &KdTree) -> Vec<u32> {
    tree.layout()
        .nodes()
        .flat_map(|node| match node {
            KdTreeNode::Leaf { start, end } => [0, u32::MAX, 0, 0, start, end, 0, 0],
            KdTreeNode::Branch { axis, split, left, right } => {
                [split.to_bits(), u32::from(axis), left, right, 0, 0, 0, 0]
            }
        })
        .collect()
}

#[test]
#[ignore = "requires native GPU adapter"]
fn uploaded_tree_points_nodes_receipts_and_cpu_drop_ownership_match() {
    let r = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let mut words = 0;
    for n in [0, 1, 17, 262_143, 262_144, 262_145, 524_287, 524_288, 524_289, 1_048_577] {
        for profile in 0..4 {
            let data = fixture(n, profile);
            let tree = KdTree::from_slices(&data[0], &data[1], &data[2]);
            let layout = tree.layout();
            let (x, y, z) = layout.positions();
            let expected_points = previous_prepare([x, y, z], layout.point_order()).unwrap();
            let expected_nodes = node_words(&tree);
            let old = previous_upload(&r, &tree).unwrap();
            let new = GpuKdTree::upload(&r, &tree).unwrap();
            assert_eq!(old.root, new.root);
            assert_eq!(new.root, layout.root());
            assert_eq!(old.upload_transfers(), new.upload_transfers());
            assert_eq!(
                new.upload_transfers().host_to_device_bytes(),
                n as u64 * 16 + expected_nodes.len() as u64 * 4
            );
            assert_eq!(new.upload_transfers().device_to_host_bytes(), 0);
            drop(tree);
            drop(data);
            if n == 0 {
                assert!(old.points.is_none() && new.points.is_none());
                assert!(old.nodes.is_none() && new.nodes.is_none());
            } else {
                let expected_points: &[u32] = bytemuck::cast_slice(&expected_points);
                assert_eq!(probe.read(&r, old.points.as_ref().unwrap()), expected_points);
                assert_eq!(probe.read(&r, new.points.as_ref().unwrap()), expected_points);
                assert_eq!(probe.read(&r, old.nodes.as_ref().unwrap()), expected_nodes);
                assert_eq!(probe.read(&r, new.nodes.as_ref().unwrap()), expected_nodes);
                words += expected_points.len() + expected_nodes.len();
            }
        }
    }
    println!("TREE_STAGING_GPU_WORDS {words}");
    r.device().push_error_scope(wgpu::ErrorFilter::Validation);
    for axis in 0..3 {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut data = fixture(524_289, 0);
            data[axis][524_288] = bad;
            let tree = KdTree::from_slices(&data[0], &data[1], &data[2]);
            assert_eq!(
                previous_upload(&r, &tree).err().unwrap(),
                GpuKdTree::upload(&r, &tree).err().unwrap()
            );
        }
    }
    r.wait_idle();
    assert!(pollster::block_on(r.device().pop_error_scope()).is_none());
}

#[test]
#[ignore = "requires native GPU adapter"]
fn uploaded_tree_preserves_large_neighborhood_ties_radius_order_and_runtime_guards() {
    let r = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    for profile in 0..3 {
        let data = fixture(524_289, profile);
        let tree = KdTree::from_slices(&data[0], &data[1], &data[2]);
        let indices = [0, 1, 16, 131_071, 131_072, 262_144];
        let queries: Vec<_> =
            indices.iter().map(|&i| [data[0][i], data[1][i], data[2][i]]).collect();
        let old = previous_upload(&r, &tree).unwrap();
        let new = GpuKdTree::upload(&r, &tree).unwrap();
        drop(tree);
        drop(data);
        let batch = GpuQueryBatch::upload(&r, &queries).unwrap();
        assert!(new.nearest_k_resident(&other, &batch, 1).is_err());
        for k in [1, 8, 64] {
            let a = old.nearest_k_resident(&r, &batch, k).unwrap();
            let b = new.nearest_k_resident(&r, &batch, k).unwrap();
            assert_eq!(a.execution_transfers(), b.execution_transfers());
            let a = a.download(&r).unwrap();
            let b = b.download(&r).unwrap();
            assert_eq!(a.complete_rows().unwrap(), b.complete_rows().unwrap());
            assert_eq!(a.transfers(), b.transfers());
            compared += queries.len();
        }
        for radius in [0.0, 0.125, f32::MAX] {
            let a = old.radius_neighbors_resident(&r, &batch, radius, 16).unwrap();
            let b = new.radius_neighbors_resident(&r, &batch, radius, 16).unwrap();
            assert_eq!(a.execution_transfers(), b.execution_transfers());
            let a = a.download(&r).unwrap();
            let b = b.download(&r).unwrap();
            assert_eq!(a.truncated_rows(), b.truncated_rows());
            assert_eq!(a.total_counts(), b.total_counts());
            assert_eq!(a.complete_rows().is_ok(), b.complete_rows().is_ok());
            assert_eq!(a.transfers(), b.transfers());
            compared += queries.len();
        }
    }
    println!("TREE_STAGING_NEIGHBORHOOD_ROWS {compared}");
}
