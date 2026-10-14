use super::*;
use crate::kernels::{
    gicp_reduction as gicp, icp_reduction as icp, point_to_plane_reduction as plane,
};
use crate::{GpuGicpInput, GpuIcpInput, GpuPointToPlaneInput};
use spatialrust_math::{Isometry3, Quat, Vec3};

fn fixture(n: usize, profile: usize) -> (Vec<[f32; 3]>, KdTree, Vec<[f32; 3]>) {
    let points: Vec<_> = (0..n)
        .map(|i| {
            let zero = f32::from_bits((i as u32 & 1) << 31);
            match profile {
                1 => [zero, -zero, zero],
                2 => [
                    f32::from_bits(i as u32 % 23),
                    if i == 0 { 1.0 } else { zero },
                    -f32::from_bits(i as u32 % 17),
                ],
                3 => [f32::MAX; 3],
                _ => [
                    (i % 17) as f32 * 0.125,
                    (i / 17 % 17) as f32 * 0.125,
                    (i / 289) as f32 * 0.001,
                ],
            }
        })
        .collect();
    let axes: [Vec<_>; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).collect());
    let tree = KdTree::from_slices(&axes[0], &axes[1], &axes[2]);
    let source = points.iter().map(|p| [p[0], p[1], p[2]]).collect();
    (source, tree, points)
}

fn normals(n: usize) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| if i % 7 == 0 { [0.0, -0.0, 0.0] } else { [1.0, (i % 3) as f32 * 0.25, -0.0] })
        .collect()
}

fn covariances(n: usize) -> Vec<[f32; 6]> {
    (0..n).map(|i| [1.0 + (i % 3) as f32 * 0.125, 2.0, 3.0, -0.0, 0.0, 0.125]).collect()
}

fn compare_positions(
    probe: &Probe,
    r: &WgpuRuntime,
    old: &GpuIcpInput,
    new: &GpuIcpInput,
    source: &[[f32; 3]],
    points: &[[f32; 3]],
) -> usize {
    assert_eq!(icp::gather_reference::metadata(old), icp::gather_reference::metadata(new));
    assert_eq!(old.source_count(), new.source_count());
    assert_eq!(old.tree.root, new.tree.root);
    let mut words = 0;
    for (a, b, expected) in [
        (old.source.buffer(), new.source.buffer(), source),
        (old.target.buffer(), new.target.buffer(), points),
    ] {
        if expected.is_empty() {
            assert!(a.is_none() && b.is_none());
            continue;
        }
        let old = probe.read(r, a.unwrap());
        let new = probe.read(r, b.unwrap());
        assert_eq!(old, new);
        for (row, point) in new.chunks_exact(4).zip(expected) {
            assert_eq!(row, [point[0].to_bits(), point[1].to_bits(), point[2].to_bits(), 0]);
        }
        words += new.len();
    }
    for (a, b) in [
        (old.tree.points.as_ref().unwrap(), new.tree.points.as_ref().unwrap()),
        (old.tree.nodes.as_ref().unwrap(), new.tree.nodes.as_ref().unwrap()),
    ] {
        let old = probe.read(r, a);
        let new = probe.read(r, b);
        assert_eq!(old, new);
        words += new.len();
    }
    words
}

fn receipts(
    old: TransferStats,
    new: TransferStats,
    source: usize,
    targets: usize,
    nodes: usize,
    family: usize,
) {
    let (s, t) = match family {
        1 => (48, 64),
        2 => (16, 48),
        _ => (16, 32),
    };
    assert_eq!(
        old.host_to_device_bytes(),
        source as u64 * s + targets as u64 * t + nodes as u64 * 32
    );
    assert_eq!(new.host_to_device_bytes(), old.host_to_device_bytes() - targets as u64 * 16);
    assert_eq!(old.device_to_device_bytes(), 0);
    assert_eq!(new.device_to_device_bytes(), targets as u64 * 32);
    assert_eq!(old.device_to_host_bytes(), 0);
    assert_eq!(new.device_to_host_bytes(), 0);
}

#[test]
#[ignore = "requires native GPU adapter"]
fn registration_gather_preserves_input_words_metadata_and_receipts() {
    let r = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let mut words = 0;
    for (n, profile) in [
        (1, 0),
        (17, 0),
        (257, 0),
        (65_537, 0),
        (262_145, 0),
        (524_289, 0),
        (1_048_577, 0),
        (524_289, 1),
        (524_289, 2),
        (17, 3),
    ] {
        let (source, tree, points) = fixture(n, profile);
        let nodes = tree.layout().nodes().len();
        let normals = normals(n);
        let covariances = covariances(n);
        let old = icp::gather_reference::previous_upload(&r, &source, &tree).unwrap();
        let new = GpuIcpInput::upload_with_gpu_gather(&r, &source, &tree).unwrap();
        receipts(old.upload_transfers(), new.upload_transfers(), n, n, nodes, 0);
        words += compare_positions(&probe, &r, &old, &new, &source, &points);
        let old =
            gicp::gather_reference::previous_upload(&r, &source, &covariances, &tree, &covariances)
                .unwrap();
        let new =
            GpuGicpInput::upload_with_gpu_gather(&r, &source, &covariances, &tree, &covariances)
                .unwrap();
        receipts(old.upload_transfers(), new.upload_transfers(), n, n, nodes, 1);
        let a = gicp::gather_reference::parts(&old);
        let b = gicp::gather_reference::parts(&new);
        words += compare_positions(&probe, &r, a.0, b.0, &source, &points);
        assert_eq!((a.2, a.3), (b.2, b.3));
        let old_words = probe.read(&r, a.1);
        let new_words = probe.read(&r, b.1);
        assert_eq!(old_words, new_words);
        words += new_words.len();
        let old = plane::gather_reference::previous_upload(&r, &source, &tree, &normals).unwrap();
        let new =
            GpuPointToPlaneInput::upload_with_gpu_gather(&r, &source, &tree, &normals).unwrap();
        receipts(old.upload_transfers(), new.upload_transfers(), n, n, nodes, 2);
        let a = plane::gather_reference::parts(&old);
        let b = plane::gather_reference::parts(&new);
        words += compare_positions(&probe, &r, a.0, b.0, &source, &points);
        assert_eq!(a.2, b.2);
        let old_words = probe.read(&r, a.1);
        let new_words = probe.read(&r, b.1);
        assert_eq!(old_words, new_words);
        words += new_words.len();
    }
    let (_, tree, points) = fixture(17, 0);
    let old = GpuIcpInput::upload(&r, &[], &tree).unwrap();
    let new = GpuIcpInput::upload_with_gpu_gather(&r, &[], &tree).unwrap();
    receipts(old.upload_transfers(), new.upload_transfers(), 0, 17, tree.layout().nodes().len(), 0);
    words += compare_positions(&probe, &r, &old, &new, &[], &points);
    println!("REGISTRATION_GATHER_INPUT words={words}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn registration_gather_preserves_owned_outputs_trimmed_moments_and_status() {
    let r = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let other = WgpuRuntime::new_headless().unwrap();
    let mut words = 0;
    for (n, profile) in [(1, 0), (17, 0), (65, 0), (257, 0), (65, 1), (65, 2), (65, 3)] {
        let (source, tree, _) = fixture(n, profile);
        let normals = normals(n);
        let cov = covariances(n);
        let old = GpuIcpInput::upload(&r, &source, &tree).unwrap();
        let new = GpuIcpInput::upload_with_gpu_gather(&r, &source, &tree).unwrap();
        let og = GpuGicpInput::upload(&r, &source, &cov, &tree, &cov).unwrap();
        let ng = GpuGicpInput::upload_with_gpu_gather(&r, &source, &cov, &tree, &cov).unwrap();
        let op = GpuPointToPlaneInput::upload(&r, &source, &tree, &normals).unwrap();
        let np =
            GpuPointToPlaneInput::upload_with_gpu_gather(&r, &source, &tree, &normals).unwrap();
        drop(tree);
        drop(source);
        drop(normals);
        drop(cov);
        for pose in [
            Isometry3::identity(),
            Isometry3::new(Quat::<f32>::identity(), Vec3::new(0.01, -0.02, 0.005)),
            Isometry3::new(
                Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.03),
                Vec3::new(0.01, 0.0, 0.0),
            ),
        ] {
            let a = old.moments_resident(&r, pose, 0.5).unwrap();
            let b = new.moments_resident(&r, pose, 0.5).unwrap();
            let aw = probe.read(&r, a.buffer());
            let bw = probe.read(&r, b.buffer());
            assert_eq!(aw, bw);
            words += bw.len();
            assert_eq!(a.download(&r), b.download(&r));
            assert!(b.download(&other).is_err());
            for precision in [
                icp::trim::GpuIcpTrimPrecision::DoubleSingle,
                icp::trim::GpuIcpTrimPrecision::CompensatedF32,
            ] {
                let a = old
                    .trimmed_moments_resident_with_precision(&r, pose, 0.5, 0.75, precision)
                    .unwrap();
                let b = new
                    .trimmed_moments_resident_with_precision(&r, pose, 0.5, 0.75, precision)
                    .unwrap();
                let aw = probe.read(&r, a.buffer());
                let bw = probe.read(&r, b.buffer());
                assert_eq!(aw, bw);
                words += bw.len();
                assert_eq!(a.download(&r), b.download(&r));
            }
            let a = og.equations_resident(&r, pose, 0.5).unwrap();
            let b = ng.equations_resident(&r, pose, 0.5).unwrap();
            let aw = probe.read(&r, a.buffer());
            let bw = probe.read(&r, b.buffer());
            assert_eq!(aw, bw);
            words += bw.len();
            assert_eq!(a.download(&r), b.download(&r));
            let a = op.equations_resident(&r, pose, 0.5).unwrap();
            let b = np.equations_resident(&r, pose, 0.5).unwrap();
            let aw = probe.read(&r, a.buffer());
            let bw = probe.read(&r, b.buffer());
            assert_eq!(aw, bw);
            words += bw.len();
            assert_eq!(a.download(&r), b.download(&r));
        }
        let a = new.moments_resident(&r, Isometry3::identity(), 0.5).unwrap();
        let b = ng.equations_resident(&r, Isometry3::identity(), 0.5).unwrap();
        let c = np.equations_resident(&r, Isometry3::identity(), 0.5).unwrap();
        drop(old);
        drop(new);
        drop(og);
        drop(ng);
        drop(op);
        drop(np);
        words += probe.read(&r, a.buffer()).len()
            + probe.read(&r, b.buffer()).len()
            + probe.read(&r, c.buffer()).len();
    }
    println!("REGISTRATION_GATHER_OUTPUT words={words}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn registration_gather_rejections_retain_precedence_before_pipeline_initialization() {
    let r = WgpuRuntime::new_headless().unwrap();
    let (source, tree, _) = fixture(17, 0);
    let covariance = covariances(17);
    let normals = normals(17);
    let mut errors = 0;
    for (s, t) in [
        (Vec::new(), KdTree::from_slices(&[], &[], &[])),
        (vec![[f32::NAN, 0.0, 0.0]], fixture(17, 0).1),
        (source.clone(), KdTree::from_slices(&[f32::INFINITY], &[0.0], &[0.0])),
        (vec![[0.0; 3], [f32::MAX, 0.0, 0.0]], fixture(17, 0).1),
        (
            vec![[0.0; 3], [f32::from_bits(1), 0.0, 0.0]],
            KdTree::from_slices(&[0.0], &[0.0], &[0.0]),
        ),
    ] {
        assert_eq!(
            icp::gather_reference::previous_upload(&r, &s, &t).err(),
            GpuIcpInput::upload_with_gpu_gather(&r, &s, &t).err()
        );
        errors += 1;
    }
    for bad in [
        vec![],
        vec![[0.0; 6]; 17],
        vec![[f32::NAN; 6]; 17],
        vec![[1.0, 1.0, 1.0, 2.0, 0.0, 0.0]; 17],
    ] {
        assert_eq!(
            gicp::gather_reference::previous_upload(&r, &source, &bad, &tree, &covariance).err(),
            GpuGicpInput::upload_with_gpu_gather(&r, &source, &bad, &tree, &covariance).err()
        );
        errors += 1;
    }
    for bad in [
        vec![],
        vec![[f32::NAN; 3]; 17],
        vec![[f32::from_bits(1), 0.0, 0.0]; 17],
        vec![[f32::MAX, f32::MIN_POSITIVE, 0.0]; 17],
    ] {
        assert_eq!(
            plane::gather_reference::previous_upload(&r, &source, &tree, &bad).err(),
            GpuPointToPlaneInput::upload_with_gpu_gather(&r, &source, &tree, &bad).err()
        );
        errors += 1;
    }
    assert!(r.tree_gather_pipeline.get().is_none());
    let other = WgpuRuntime::new_headless().unwrap();
    let input = GpuPointToPlaneInput::upload_with_gpu_gather(&r, &source, &tree, &normals).unwrap();
    assert!(input.equations_resident(&other, Isometry3::identity(), 0.5).is_err());
    println!("REGISTRATION_GATHER_ERRORS errors={errors}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn registration_gather_cold_concurrent_constructors_share_paired_pipeline() {
    let r = WgpuRuntime::new_headless().unwrap();
    let (source, tree, _) = fixture(65, 0);
    let cov = covariances(65);
    let normals = normals(65);
    std::thread::scope(|scope| {
        let a = scope.spawn(|| {
            GpuIcpInput::upload_with_gpu_gather(&r, &source, &tree)
                .unwrap()
                .moments_resident(&r, Isometry3::identity(), 0.5)
                .unwrap()
        });
        let b = scope.spawn(|| {
            GpuGicpInput::upload_with_gpu_gather(&r, &source, &cov, &tree, &cov)
                .unwrap()
                .equations_resident(&r, Isometry3::identity(), 0.5)
                .unwrap()
        });
        let c = scope.spawn(|| {
            GpuPointToPlaneInput::upload_with_gpu_gather(&r, &source, &tree, &normals)
                .unwrap()
                .equations_resident(&r, Isometry3::identity(), 0.5)
                .unwrap()
        });
        assert_eq!(a.join().unwrap().download(&r).unwrap().correspondences, 65);
        assert_eq!(b.join().unwrap().download(&r).unwrap().correspondences, 65);
        assert_eq!(c.join().unwrap().download(&r).unwrap().correspondences, 65);
    });
}

#[path = "registration_gather_upload_benchmark.rs"]
mod benchmark;

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
