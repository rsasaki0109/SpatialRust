use super::*;

fn fixture(n: usize) -> (Vec<[f32; 3]>, KdTree) {
    let source: Vec<_> =
        (0..n.min(65)).map(|i| [(i % 17) as f32 * 0.125, (i / 17) as f32 * 0.125, 0.0]).collect();
    let axes: [Vec<_>; 3] = std::array::from_fn(|a| {
        (0..n)
            .map(|i| match a {
                0 => (i % 17) as f32 * 0.125,
                1 => (i / 17 % 17) as f32 * 0.125,
                _ => (i / 289) as f32 * 0.001,
            })
            .collect()
    });
    (source, KdTree::from_slices(&axes[0], &axes[1], &axes[2]))
}

fn normals(n: usize, profile: usize) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| {
            let zero = f32::from_bits((i as u32 & 1) << 31);
            match profile {
                1 => [zero, -zero, zero],
                2 => [f32::MIN_POSITIVE, -f32::MIN_POSITIVE, zero],
                3 => [f32::MAX, -f32::MAX, zero],
                _ => [1.0 + (i % 3) as f32 * 0.25, -0.5, zero],
            }
        })
        .collect()
}

#[test]
#[ignore = "requires native GPU adapter"]
fn parallel_normal_upload_preserves_input_words_scale_and_both_route_receipts() {
    let r = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let mut words = 0;
    for (n, profile) in [
        (1, 0),
        (257, 0),
        (262_143, 0),
        (262_144, 0),
        (262_145, 0),
        (524_289, 0),
        (1_048_577, 0),
        (262_145, 1),
        (262_145, 2),
        (262_145, 3),
    ] {
        let (source, tree) = fixture(n);
        let normals = normals(n, profile);
        for gpu in [false, true] {
            let old =
                preparation_reference::previous_upload(&r, &source, &tree, &normals, gpu).unwrap();
            let new = if gpu {
                GpuPointToPlaneInput::upload_with_gpu_gather(&r, &source, &tree, &normals)
            } else {
                GpuPointToPlaneInput::upload(&r, &source, &tree, &normals)
            }
            .unwrap();
            assert_eq!(old.normal_scale.to_bits(), new.normal_scale.to_bits());
            assert_eq!(old.upload_transfers(), new.upload_transfers());
            assert_eq!(old.source_count(), new.source_count());
            assert_eq!(
                crate::kernels::icp_reduction::gather_reference::metadata(&old.positions),
                crate::kernels::icp_reduction::gather_reference::metadata(&new.positions)
            );
            let a = probe.read(&r, &old.normals);
            let b = probe.read(&r, &new.normals);
            assert_eq!(a, b);
            let (expected, scale) =
                preparation_reference::previous_normal_preparation(&normals).unwrap();
            assert_eq!(scale.to_bits(), new.normal_scale.to_bits());
            for (row, expected) in b.chunks_exact(4).zip(expected) {
                assert_eq!(row, expected.map(f32::to_bits));
            }
            words += b.len();
        }
    }
    println!("PLANE_STAGING_INPUT words={words}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn parallel_normal_preparation_preserves_owned_equations_errors_and_runtime() {
    let r = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let mut words = 0;
    for n in [1, 257, 262_145] {
        let (source, tree) = fixture(n);
        for factor in [1e-8, 1.0, 1e8] {
            let normals: Vec<_> =
                normals(n, 0).into_iter().map(|row| row.map(|v| v * factor)).collect();
            for gpu in [false, true] {
                let old = preparation_reference::previous_upload(&r, &source, &tree, &normals, gpu)
                    .unwrap();
                let new = if gpu {
                    GpuPointToPlaneInput::upload_with_gpu_gather(&r, &source, &tree, &normals)
                } else {
                    GpuPointToPlaneInput::upload(&r, &source, &tree, &normals)
                }
                .unwrap();
                for pose in [
                    Isometry3::identity(),
                    Isometry3::new(
                        spatialrust_math::Quat::<f32>::identity(),
                        spatialrust_math::Vec3::new(0.01, -0.02, 0.005),
                    ),
                ] {
                    let a = old.equations_resident(&r, pose, 0.5).unwrap();
                    let b = new.equations_resident(&r, pose, 0.5).unwrap();
                    let aw = probe.read(&r, a.buffer());
                    let bw = probe.read(&r, b.buffer());
                    assert_eq!(aw, bw);
                    words += bw.len();
                    assert_eq!(a.download(&r), b.download(&r));
                    assert!(b.download(&other).is_err());
                }
                let a = old.equations_resident(&r, Isometry3::identity(), 0.5).unwrap();
                let b = new.equations_resident(&r, Isometry3::identity(), 0.5).unwrap();
                drop(old);
                drop(new);
                assert_eq!(a.download(&r), b.download(&r));
            }
        }
    }
    let (mut source, tree) = fixture(262_145);
    source[0][0] = f32::NAN;
    let mut rows = normals(262_145, 0);
    let mut errors = 0;
    for index in [0, 65_537, 131_074, 196_611, 262_144] {
        for value in [f32::NAN, f32::INFINITY, f32::from_bits(1)] {
            rows[index][0] = value;
            for gpu in [false, true] {
                let a =
                    preparation_reference::previous_upload(&r, &source, &tree, &rows, gpu).err();
                let b = if gpu {
                    GpuPointToPlaneInput::upload_with_gpu_gather(&r, &source, &tree, &rows)
                } else {
                    GpuPointToPlaneInput::upload(&r, &source, &tree, &rows)
                }
                .err();
                assert!(a.is_some());
                assert_eq!(a, b);
                errors += 1;
            }
            rows[index] = [1.0, -0.5, 0.0];
        }
    }
    println!("PLANE_STAGING_OUTPUT words={words} errors={errors}");
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
