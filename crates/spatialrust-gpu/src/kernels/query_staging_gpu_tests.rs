// Frozen query-upload constructor from main 3345396; only the serial packer
// and concrete return/constructor names differ. Integer probe is test-only.
use super::super::super::*;
use super::{fixture, previous_prepare};

pub(super) fn previous_upload(
    runtime: &WgpuRuntime,
    axes: [&[f32]; 3],
) -> SpatialResult<GpuQueryBatch> {
    let count = axes[0].len();
    if axes.iter().any(|axis| axis.len() != count) {
        return Err(invalid());
    }
    let limits = runtime.device().limits();
    let bytes = buffer_bytes(count as u64, 16, &limits)?;
    check_compute(&limits, count as u64)?;
    let packed = previous_prepare(axes)?;
    let mut batch = GpuQueryBatch {
        buffer: None,
        count: count as u32,
        identity: runtime.profiling_identity(),
        transfers: TransferStats::default(),
    };
    if count != 0 {
        batch.buffer =
            Some(runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("neighborhood-queries"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE,
            }));
        batch.transfers.record(TransferDirection::HostToDevice, bytes);
    }
    Ok(batch)
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
fn uploaded_parallel_queries_preserve_all_words_receipts_and_cpu_drop_ownership() {
    let r = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let mut words = 0;
    for n in [0, 1, 17, 524_287, 524_288, 524_289, 1_048_577] {
        for profile in 0..4 {
            let data = fixture(n, profile);
            let axes = [&data[0][..], &data[1][..], &data[2][..]];
            let expected = previous_prepare(axes).unwrap();
            let old = previous_upload(&r, axes).unwrap();
            let new = GpuQueryBatch::upload_soa(&r, axes).unwrap();
            drop(data);
            assert_eq!(old.query_count(), new.query_count());
            assert_eq!(new.query_count(), n as u32);
            assert_eq!(old.upload_transfers(), new.upload_transfers());
            assert_eq!(new.upload_transfers().host_to_device_bytes(), n as u64 * 16);
            assert_eq!(new.upload_transfers().device_to_host_bytes(), 0);
            if n == 0 {
                assert!(old.buffer().is_none() && new.buffer().is_none());
            } else {
                let a = probe.read(&r, old.buffer().unwrap());
                let b = probe.read(&r, new.buffer().unwrap());
                let expected: &[u32] = bytemuck::cast_slice(&expected);
                assert_eq!(a, expected);
                assert_eq!(b, expected);
                words += expected.len();
            }
        }
    }
    println!("QUERY_STAGING_GPU_WORDS {words}");
    r.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut data = fixture(524_289, 0);
    for axis in 0..3 {
        let last = data[axis].len() - 1;
        let original = data[axis][last];
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            data[axis][last] = bad;
            let axes = [&data[0][..], &data[1][..], &data[2][..]];
            let old = previous_upload(&r, axes).err().unwrap();
            let new = GpuQueryBatch::upload_soa(&r, axes).err().unwrap();
            assert_eq!(old, new);
        }
        data[axis][last] = original;
    }
    assert!(GpuQueryBatch::upload_soa(&r, [&data[0], &data[1][1..], &data[2]]).is_err());
    r.wait_idle();
    assert!(pollster::block_on(r.device().pop_error_scope()).is_none());
}
