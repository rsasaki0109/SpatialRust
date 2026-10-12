use super::*;

fn read_words(runtime: &WgpuRuntime, buffer: &wgpu::Buffer, bytes: u64) -> Vec<u32> {
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
    runtime.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    runtime.device().poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let words = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    words
}

#[test]
fn fpfh_plan_rejects_buffer_dispatch_and_binding_boundaries() {
    let limits = wgpu::Limits::default();
    assert!(plan(u64::MAX, &limits).is_err());
    assert!(plan(u64::from(u32::MAX) + 1, &limits).is_err());
    let max = u64::from(limits.max_storage_buffer_binding_size) / 132;
    assert!(plan(max, &limits).is_ok());
    assert!(plan(max + 1, &limits).is_err());
    let mut restricted = limits.clone();
    restricted.max_compute_workgroups_per_dimension = 1;
    assert!(plan(64, &restricted).is_ok());
    assert!(plan(65, &restricted).is_err());
    for field in 0..5 {
        restricted = limits.clone();
        match field {
            0 => restricted.max_compute_workgroup_size_x = 63,
            1 => restricted.max_storage_buffers_per_shader_stage = 4,
            2 => restricted.max_bindings_per_bind_group = 5,
            3 => restricted.max_uniform_buffer_binding_size = 15,
            _ => restricted.max_buffer_size = 31,
        }
        assert!(plan(1, &restricted).is_err());
    }
}

#[test]
fn fpfh_upload_validates_normals_extent_and_empty_receipts() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let tree = KdTree::from_slices(&[0.0], &[0.0], &[0.0]);
    for normal in [[0.0; 3], [0.0, 0.0, 2.0], [f32::NAN, 0.0, 1.0], [f32::INFINITY, 0.0, 1.0]] {
        assert!(GpuFpfhInput::upload(&runtime, &tree, &[normal]).is_err());
    }
    assert!(GpuFpfhInput::upload(&runtime, &tree, &[]).is_err());
    let huge = KdTree::from_slices(&[f32::MAX, -f32::MAX], &[0.0; 2], &[0.0; 2]);
    assert!(GpuFpfhInput::upload(&runtime, &huge, &[[0.0, 0.0, 1.0]; 2]).is_err());
    let empty = GpuFpfhInput::upload(&runtime, &KdTree::from_slices(&[], &[], &[]), &[]).unwrap();
    assert_eq!(empty.upload_transfers(), TransferStats::default());
    let neighbors = empty.radius_neighborhoods_resident(&runtime, 1.0, 0).unwrap();
    let output = empty.descriptors_resident(&runtime, &neighbors).unwrap();
    assert_eq!(output.readback_bytes(), 0);
    assert_eq!(output.download(&runtime).unwrap().1, TransferStats::default());
}

#[test]
fn overflow_and_malformed_device_neighbors_poison_all_descriptors() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let x: Vec<_> = (0..65).map(|i| i as f32 * 0.125).collect();
    let zero = vec![0.0; 65];
    let tree = KdTree::from_slices(&x, &zero, &zero);
    let input = GpuFpfhInput::upload(&runtime, &tree, &[[0.0, 0.0, 1.0]; 65]).unwrap();
    let short = input.radius_neighborhoods_resident(&runtime, 100.0, 3).unwrap();
    let bad = input.descriptors_resident(&runtime, &short).unwrap();
    assert_eq!(read_words(&runtime, bad.status_buffer(), 16), [65, 65, 65, 0]);
    assert!(read_words(&runtime, bad.buffer(), 65 * 132).iter().all(|&v| v == 0));
    assert!(bad.download(&runtime).is_err());
    let full = input.radius_neighborhoods_resident(&runtime, 0.2, 3).unwrap();
    let shader=runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {label:None,source:wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage,read_write> words:array<u32>; @compute @workgroup_size(1) fn poison(){words[2]=0xffffffffu;}".into())});
    let pipeline = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("poison"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: full.neighborhoods().buffer().as_entire_binding(),
        }],
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
    let poisoned = input.descriptors_resident(&runtime, &full).unwrap();
    assert_eq!(read_words(&runtime, poisoned.status_buffer(), 16), [1, 1, 2, 1]);
    assert!(read_words(&runtime, poisoned.buffer(), 65 * 132).iter().all(|&v| v == 0));
    assert!(poisoned.download(&runtime).is_err());
    let fresh = input.radius_neighborhoods_resident(&runtime, 0.2, 3).unwrap();
    let valid = input.descriptors_resident(&runtime, &fresh).unwrap();
    assert_eq!(read_words(&runtime, valid.status_buffer(), 16), [0; 4]);
    assert!(valid.download(&runtime).unwrap().0.iter().flatten().any(|&v| v > 0.0));
}
