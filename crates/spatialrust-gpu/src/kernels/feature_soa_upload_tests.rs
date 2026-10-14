use super::{fpfh, iss, mls};
use crate::{GpuFpfhInput, GpuIssInput, GpuMlsInput, WgpuRuntime};
use spatialrust_core::TransferStats;
use spatialrust_search::KdTree;

fn fixture(n: usize, profile: usize) -> (KdTree, Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let points: Vec<_> = (0..n)
        .map(|i| {
            let zero = f32::from_bits((i as u32 & 1) << 31);
            match profile {
                0 => [(i % 32) as f32 * 0.125, (i / 32) as f32 * 0.125, (i % 7) as f32 * 0.001],
                1 => [zero, -zero, zero],
                2 => [f32::from_bits(i as u32 % 23), -f32::from_bits(i as u32 % 17), zero],
                _ => [f32::MAX; 3],
            }
        })
        .collect();
    let axes: [Vec<_>; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).collect());
    let tree = KdTree::from_slices(&axes[0], &axes[1], &axes[2]);
    let normals = (0..n).map(|i| [0.0, f32::from_bits((i as u32 & 1) << 31), 1.0]).collect();
    (tree, points, normals)
}

// Read STORAGE-only production inputs through an integer compute probe. This
// adds no COPY_SRC usage, payload download or pipeline to the production API.
struct Probe {
    pipeline: wgpu::ComputePipeline,
}
impl Probe {
    fn new(r: &WgpuRuntime) -> Self {
        let shader = r.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("feature-upload-word-probe"),
            source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage,read> src:array<u32>; @group(0) @binding(1) var<storage,read_write> dst:array<u32>; @compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>){if id.x<arrayLength(&src){dst[id.x]=src[id.x];}}".into()),
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
            pass.dispatch_workgroups((bytes as u32 / 4).div_ceil(64), 1, 1);
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

fn compare_query(
    probe: &Probe,
    r: &WgpuRuntime,
    a: Option<&wgpu::Buffer>,
    b: Option<&wgpu::Buffer>,
    points: &[[f32; 3]],
) {
    if points.is_empty() {
        assert!(a.is_none() && b.is_none());
        return;
    }
    let a = probe.read(r, a.unwrap());
    let b = probe.read(r, b.unwrap());
    assert_eq!(a, b);
    for (row, p) in b.chunks_exact(4).zip(points) {
        assert_eq!(row, [p[0].to_bits(), p[1].to_bits(), p[2].to_bits(), 0]);
    }
}

#[test]
#[ignore = "requires native GPU adapter"]
fn direct_soa_preserves_uploaded_words_and_receipts() {
    let r = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let mut cases = 0;
    let mut words = 0;
    for n in [0, 1, 17, 257, 65_537, 262_145] {
        for profile in 0..4 {
            let (tree, points, normals) = fixture(n, profile);
            let fo = fpfh::soa_upload_reference::previous_upload(&r, &tree, &normals).unwrap();
            let f = GpuFpfhInput::upload(&r, &tree, &normals).unwrap();
            let io = iss::soa_upload_reference::previous_upload(&r, &tree).unwrap();
            let i = GpuIssInput::upload(&r, &tree).unwrap();
            let mo = mls::soa_upload_reference::previous_upload(&r, &tree).unwrap();
            let m = GpuMlsInput::upload(&r, &tree).unwrap();
            let nodes = tree.layout().nodes().len() as u64;
            drop(tree);
            assert_eq!(fo.upload_transfers(), f.upload_transfers());
            assert_eq!(io.upload_transfers(), i.upload_transfers());
            assert_eq!(mo.upload_transfers(), m.upload_transfers());
            if n == 0 {
                assert_eq!(f.upload_transfers(), TransferStats::default());
                assert_eq!(i.upload_transfers(), TransferStats::default());
                assert_eq!(m.upload_transfers(), TransferStats::default());
            } else {
                assert_eq!(f.upload_transfers().host_to_device_bytes(), n as u64 * 64 + nodes * 32);
                assert_eq!(i.upload_transfers().host_to_device_bytes(), n as u64 * 32 + nodes * 32);
                assert_eq!(m.upload_transfers().host_to_device_bytes(), n as u64 * 32 + nodes * 32);
            }
            compare_query(
                &probe,
                &r,
                fpfh::soa_upload_reference::query_buffer(&fo),
                fpfh::soa_upload_reference::query_buffer(&f),
                &points,
            );
            compare_query(
                &probe,
                &r,
                iss::soa_upload_reference::query_buffer(&io),
                iss::soa_upload_reference::query_buffer(&i),
                &points,
            );
            compare_query(
                &probe,
                &r,
                mls::soa_upload_reference::query_buffer(&mo),
                mls::soa_upload_reference::query_buffer(&m),
                &points,
            );
            if n > 0 {
                let a = probe.read(&r, fpfh::soa_upload_reference::sample_buffer(&fo).unwrap());
                let b = probe.read(&r, fpfh::soa_upload_reference::sample_buffer(&f).unwrap());
                assert_eq!(a, b);
                for ((row, p), normal) in b.chunks_exact(8).zip(&points).zip(&normals) {
                    assert_eq!(
                        row,
                        [
                            p[0].to_bits(),
                            p[1].to_bits(),
                            p[2].to_bits(),
                            0,
                            normal[0].to_bits(),
                            normal[1].to_bits(),
                            normal[2].to_bits(),
                            0
                        ]
                    );
                }
            }
            words += n * 20;
            cases += 1;
        }
    }
    println!("FEATURE_SOA_UPLOAD_WORDS cases={cases} exact_words={words}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn direct_soa_rejections_match_frozen_constructors() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut cases = 0;
    for axis in 0..3 {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut axes = [[0.0; 3]; 3];
            axes[axis][1] = value;
            let tree = KdTree::from_slices(&axes[0], &axes[1], &axes[2]);
            assert_eq!(
                fpfh::soa_upload_reference::previous_upload(&r, &tree, &[[0.0, 0.0, 1.0]; 3])
                    .err()
                    .unwrap()
                    .to_string(),
                GpuFpfhInput::upload(&r, &tree, &[[0.0, 0.0, 1.0]; 3]).err().unwrap().to_string()
            );
            assert_eq!(
                iss::soa_upload_reference::previous_upload(&r, &tree).err().unwrap().to_string(),
                GpuIssInput::upload(&r, &tree).err().unwrap().to_string()
            );
            assert_eq!(
                mls::soa_upload_reference::previous_upload(&r, &tree).err().unwrap().to_string(),
                GpuMlsInput::upload(&r, &tree).err().unwrap().to_string()
            );
            cases += 3;
        }
    }
    let huge = KdTree::from_slices(&[-f32::MAX, f32::MAX], &[0.0; 2], &[0.0; 2]);
    assert_eq!(
        fpfh::soa_upload_reference::previous_upload(&r, &huge, &[[0.0, 0.0, 1.0]; 2])
            .err()
            .unwrap()
            .to_string(),
        GpuFpfhInput::upload(&r, &huge, &[[0.0, 0.0, 1.0]; 2]).err().unwrap().to_string()
    );
    assert_eq!(
        iss::soa_upload_reference::previous_upload(&r, &huge).err().unwrap().to_string(),
        GpuIssInput::upload(&r, &huge).err().unwrap().to_string()
    );
    assert_eq!(
        mls::soa_upload_reference::previous_upload(&r, &huge).err().unwrap().to_string(),
        GpuMlsInput::upload(&r, &huge).err().unwrap().to_string()
    );
    cases += 3;
    let tree = KdTree::from_slices(&[0.0], &[-0.0], &[0.0]);
    for normals in [
        vec![],
        vec![[0.0; 3]],
        vec![[0.0, 0.0, 2.0]],
        vec![[f32::NAN, 0.0, 1.0]],
        vec![[f32::INFINITY, 0.0, 1.0]],
    ] {
        assert_eq!(
            fpfh::soa_upload_reference::previous_upload(&r, &tree, &normals)
                .err()
                .unwrap()
                .to_string(),
            GpuFpfhInput::upload(&r, &tree, &normals).err().unwrap().to_string()
        );
        cases += 1;
    }
    println!("FEATURE_SOA_REJECTIONS exact_errors={cases}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn direct_soa_preserves_resident_outputs_status_and_input_drop_ownership() {
    let r = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let mut calls = 0;
    let mut words = 0;
    for n in [1, 17, 65, 257] {
        for profile in 0..3 {
            let (tree, _points, normals) = fixture(n, profile);
            let old = fpfh::soa_upload_reference::previous_upload(&r, &tree, &normals).unwrap();
            let new = GpuFpfhInput::upload(&r, &tree, &normals).unwrap();
            let a = old.radius_neighborhoods_resident(&r, 0.35, 64).unwrap();
            let b = new.radius_neighborhoods_resident(&r, 0.35, 64).unwrap();
            assert!(new.descriptors_resident(&r, &a).is_err());
            assert!(new.descriptors_resident(&other, &b).is_err());
            let ao = old.descriptors_resident(&r, &a).unwrap();
            let bo = new.descriptors_resident(&r, &b).unwrap();
            drop(old);
            drop(new);
            assert_eq!(probe.read(&r, ao.buffer()), probe.read(&r, bo.buffer()));
            assert_eq!(probe.read(&r, ao.status_buffer()), probe.read(&r, bo.status_buffer()));
            assert_eq!(
                ao.download(&r)
                    .map(|v| (
                        v.0.into_iter().map(|row| row.map(f32::to_bits)).collect::<Vec<_>>(),
                        v.1
                    ))
                    .map_err(|e| e.to_string()),
                bo.download(&r)
                    .map(|v| (
                        v.0.into_iter().map(|row| row.map(f32::to_bits)).collect::<Vec<_>>(),
                        v.1
                    ))
                    .map_err(|e| e.to_string())
            );
            calls += 1;
            words += n * 33 + 4;
            let old = iss::soa_upload_reference::previous_upload(&r, &tree).unwrap();
            let new = GpuIssInput::upload(&r, &tree).unwrap();
            let a = old.neighborhoods_resident(&r, 0.35, 0.3, 64, 64).unwrap();
            let b = new.neighborhoods_resident(&r, 0.35, 0.3, 64, 64).unwrap();
            assert!(new.keypoints_resident(&r, &a, 0.975, 0.975, 5).is_err());
            assert!(new.keypoints_resident(&other, &b, 0.975, 0.975, 5).is_err());
            let ao = old.keypoints_resident(&r, &a, 0.975, 0.975, 5).unwrap();
            let bo = new.keypoints_resident(&r, &b, 0.975, 0.975, 5).unwrap();
            drop(old);
            drop(new);
            assert_eq!(probe.read(&r, ao.buffer()), probe.read(&r, bo.buffer()));
            assert_eq!(probe.read(&r, ao.scores_buffer()), probe.read(&r, bo.scores_buffer()));
            assert_eq!(probe.read(&r, ao.status_buffer()), probe.read(&r, bo.status_buffer()));
            assert_eq!(
                ao.download(&r).map_err(|e| e.to_string()),
                bo.download(&r).map_err(|e| e.to_string())
            );
            calls += 1;
            words += n * 5 + 8;
            let old = mls::soa_upload_reference::previous_upload(&r, &tree).unwrap();
            let new = GpuMlsInput::upload(&r, &tree).unwrap();
            drop(tree);
            let a = old.neighborhoods_resident(&r, 0.35, 64).unwrap();
            let b = new.neighborhoods_resident(&r, 0.35, 64).unwrap();
            assert!(new.smooth_resident(&r, &a, 0, 5).is_err());
            assert!(new.smooth_resident(&other, &b, 0, 5).is_err());
            let mut owned = Vec::new();
            for order in 0..=2 {
                owned.push((
                    old.smooth_resident(&r, &a, order, 5).unwrap(),
                    new.smooth_resident(&r, &b, order, 5).unwrap(),
                ));
            }
            drop(old);
            drop(new);
            for (ao, bo) in owned {
                assert_eq!(probe.read(&r, ao.buffer()), probe.read(&r, bo.buffer()));
                assert_eq!(probe.read(&r, ao.status_buffer()), probe.read(&r, bo.status_buffer()));
                match (ao.download(&r), bo.download(&r)) {
                    (Ok(a), Ok(b)) => {
                        assert_eq!(
                            a.positions
                                .into_iter()
                                .map(|p| p.map(f32::to_bits))
                                .collect::<Vec<_>>(),
                            b.positions
                                .into_iter()
                                .map(|p| p.map(f32::to_bits))
                                .collect::<Vec<_>>()
                        );
                        assert_eq!(a.modes, b.modes);
                        assert_eq!(a.diagnostics, b.diagnostics);
                        assert_eq!(a.transfers, b.transfers);
                    }
                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                    _ => panic!("MLS validity changed"),
                }
                calls += 1;
                words += n * 4 + 8;
            }
        }
    }
    println!("FEATURE_SOA_OUTPUTS calls={calls} exact_words={words}");
}

#[path = "feature_soa_upload_benchmark.rs"]
mod benchmark;
