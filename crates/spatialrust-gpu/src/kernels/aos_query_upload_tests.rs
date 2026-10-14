use super::*;
use spatialrust_math::Isometry3;
use spatialrust_search::KdTree;

fn fixture(n: usize, profile: usize) -> Vec<[f32; 3]> {
    let mut state = 0x6a13_d492_u32;
    (0..n)
        .map(|i| {
            std::array::from_fn(|a| match profile {
                1 => f32::from_bits(((i + a) as u32 & 1) << 31),
                2 => f32::from_bits((i as u32 % 31) | (((i + a) as u32 & 1) << 31)),
                3 => {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    f32::from_bits(if state & 0x7f80_0000 == 0x7f80_0000 {
                        state ^ 0x0080_0000
                    } else {
                        state
                    })
                }
                _ => [
                    (i % 17) as f32 * 0.125,
                    (i / 17 % 17) as f32 * 0.125,
                    (i / 289) as f32 * 0.001,
                ][a],
            })
        })
        .collect()
}

#[test]
#[ignore = "requires native GPU adapter"]
fn public_aos_upload_preserves_exact_gpu_words_receipts_and_input_drop() {
    let r = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let mut words = 0;
    for n in [0, 1, 17, 65_537, 524_287, 524_288, 524_289, 1_048_577] {
        for profile in 0..4 {
            let input = fixture(n, profile);
            let old = aos_upload_reference::previous_upload(&r, &input).unwrap();
            let new = GpuQueryBatch::upload(&r, &input).unwrap();
            assert_eq!(old.upload_transfers(), new.upload_transfers());
            assert_eq!(new.upload_transfers().host_to_device_bytes(), n as u64 * 16);
            assert_eq!(new.upload_transfers().device_to_device_bytes(), 0);
            assert_eq!(new.upload_transfers().device_to_host_bytes(), 0);
            assert_eq!(new.query_count(), n as u32);
            let expected: Vec<_> = input
                .iter()
                .flat_map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits(), 0])
                .collect();
            drop(input);
            if n == 0 {
                assert!(new.buffer.is_none() && old.buffer.is_none());
                continue;
            }
            let a = probe.read(&r, old.buffer.as_ref().unwrap());
            let b = probe.read(&r, new.buffer.as_ref().unwrap());
            assert_eq!(a, b);
            assert_eq!(b, expected);
            words += b.len();
        }
    }
    println!("AOS_QUERY_GPU words={words}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn staged_queries_keep_search_rows_radius_counts_and_owned_icp_moments() {
    let r = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let mut words = 0;
    let x = [0.0, 0.0, 0.5, 1.0];
    let y = [0.0; 4];
    let z = [0.0; 4];
    let cpu = KdTree::from_slices(&x, &y, &z);
    let tree = GpuKdTree::upload(&r, &cpu).unwrap();
    for n in [0, 65, 524_289] {
        let points: Vec<_> = (0..n).map(|i| [(i % 5) as f32 * 0.25, 0.0, -0.0]).collect();
        let old = aos_upload_reference::previous_upload(&r, &points).unwrap();
        let new = GpuQueryBatch::upload(&r, &points).unwrap();
        drop(points);
        for radius in [0.0, 0.25, 1.0] {
            let a = tree.nearest_k_resident(&r, &old, 3).unwrap();
            let b = tree.nearest_k_resident(&r, &new, 3).unwrap();
            let aw = probe.read(&r, a.buffer());
            let bw = probe.read(&r, b.buffer());
            assert_eq!(aw, bw);
            words += bw.len();
            let ar = a.download(&r).unwrap();
            let br = b.download(&r).unwrap();
            assert_eq!(ar.truncated_rows(), br.truncated_rows());
            assert_eq!(ar.total_counts(), br.total_counts());
            assert_eq!(ar.is_complete(), br.is_complete());
            let a = tree.radius_neighbors_resident(&r, &old, radius, 2).unwrap();
            let b = tree.radius_neighbors_resident(&r, &new, radius, 2).unwrap();
            let aw = probe.read(&r, a.buffer());
            let bw = probe.read(&r, b.buffer());
            assert_eq!(aw, bw);
            words += bw.len();
            let ar = a.download(&r).unwrap();
            let br = b.download(&r).unwrap();
            assert_eq!(ar.truncated_rows(), br.truncated_rows());
            assert_eq!(ar.total_counts(), br.total_counts());
            assert_eq!(ar.is_complete(), br.is_complete());
            assert!(b.download(&other).is_err());
        }
    }
    let source = fixture(524_289, 0);
    let target = KdTree::from_slices(&x, &y, &z);
    let input = crate::GpuIcpInput::upload(&r, &source, &target).unwrap();
    let mut frozen = crate::GpuIcpInput::upload(&r, &source, &target).unwrap();
    frozen.source = aos_upload_reference::previous_upload(&r, &source).unwrap();
    let a = input.moments_resident(&r, Isometry3::identity(), 0.5).unwrap();
    let b = frozen.moments_resident(&r, Isometry3::identity(), 0.5).unwrap();
    drop(source);
    drop(input);
    drop(frozen);
    let aw = probe.read(&r, a.buffer());
    let bw = probe.read(&r, b.buffer());
    assert_eq!(aw, bw);
    words += bw.len();
    assert_eq!(a.download(&r), b.download(&r));
    println!("AOS_QUERY_OUTPUT words={words}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn nonfinite_queries_reject_before_resource_use_and_cold_uploads_are_safe() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut input = fixture(524_289, 0);
    let mut errors = 0;
    for index in [0, 262_144, 262_145, input.len() - 1] {
        for a in 0..3 {
            let old = input[index][a];
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                input[index][a] = value;
                assert_eq!(
                    aos_upload_reference::previous_upload(&r, &input).err(),
                    GpuQueryBatch::upload(&r, &input).err()
                );
                errors += 1;
                input[index][a] = old;
            }
        }
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> =
            (0..3).map(|_| scope.spawn(|| GpuQueryBatch::upload(&r, &input).unwrap())).collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap().query_count(), input.len() as u32);
        }
    });
    println!("AOS_QUERY_GPU_ERRORS errors={errors}");
}

struct Probe {
    pipeline: wgpu::ComputePipeline,
}
impl Probe {
    fn new(r: &WgpuRuntime) -> Self {
        let shader = r.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aos-query-word-probe"),
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
