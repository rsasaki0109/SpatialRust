use super::*;

fn read_words(runtime: &WgpuRuntime, source: &wgpu::Buffer, bytes: u64) -> Vec<u32> {
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
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

fn poison(runtime: &WgpuRuntime, buffer: &wgpu::Buffer) {
    let shader=runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label:None,source:wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage,read_write> words:array<u32>; @compute @workgroup_size(1) fn poison(){words[2]=0xffffffffu;}".into()),
    });
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
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
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
}

#[test]
fn mls_plan_checks_counts_storage_bindings_and_dispatch() {
    let limits = wgpu::Limits::default();
    assert!(plan(u64::MAX, &limits).is_err());
    assert!(plan(u64::from(u32::MAX) + 1, &limits).is_err());
    let maximum = (u64::from(limits.max_storage_buffer_binding_size) / 192)
        .min(u64::from(limits.max_compute_workgroups_per_dimension) * 64);
    assert!(plan(maximum, &limits).is_ok());
    assert!(plan(maximum + 1, &limits).is_err());
    for case in 0..6 {
        let mut restricted = limits.clone();
        match case {
            0 => restricted.max_compute_workgroup_size_x = 63,
            1 => restricted.max_storage_buffers_per_shader_stage = 5,
            2 => restricted.max_bindings_per_bind_group = 6,
            3 => restricted.max_uniform_buffer_binding_size = 31,
            4 => restricted.max_buffer_size = 191,
            _ => restricted.max_compute_workgroups_per_dimension = 0,
        }
        assert!(plan(1, &restricted).is_err());
    }
}

#[test]
fn mls_upload_parameters_and_empty_receipts_are_checked() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for x in [vec![f32::NAN], vec![f32::MAX, -f32::MAX]] {
        let zero = vec![0.0; x.len()];
        assert!(GpuMlsInput::upload(&runtime, &KdTree::from_slices(&x, &zero, &zero)).is_err());
    }
    let input = GpuMlsInput::upload(&runtime, &KdTree::from_slices(&[], &[], &[])).unwrap();
    assert_eq!(input.upload_transfers(), TransferStats::default());
    for radius in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX, f32::MIN_POSITIVE] {
        assert!(input.neighborhoods_resident(&runtime, radius, 0).is_err());
    }
    let rows = input.neighborhoods_resident(&runtime, 1.0, 0).unwrap();
    assert!(input.smooth_resident(&runtime, &rows, 3, 0).is_err());
    for order in 0..=2 {
        let output = input.smooth_resident(&runtime, &rows, order, 0).unwrap();
        assert_eq!(output.readback_bytes(), 0);
        let data = output.download(&runtime).unwrap();
        assert!(data.positions.is_empty());
        assert_eq!(data.diagnostics, GpuMlsDiagnostics::default());
        assert_eq!(data.transfers, TransferStats::default());
    }
}

#[test]
fn overflow_poison_and_extent_underflow_zero_every_output() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let x: Vec<_> = (0..65).map(|i| i as f32 * 0.125).collect();
    let zero = vec![0.0; 65];
    let input = GpuMlsInput::upload(&runtime, &KdTree::from_slices(&x, &zero, &zero)).unwrap();
    let rows = input.neighborhoods_resident(&runtime, 100.0, 3).unwrap();
    let bad = input.smooth_resident(&runtime, &rows, 2, 0).unwrap();
    assert_eq!(read_words(&runtime, bad.status_buffer(), 32), [65, 65, 0, 65, 0, 0, 0, 0]);
    assert!(read_words(&runtime, bad.buffer(), 65 * 16).iter().all(|&v| v == 0));
    assert!(bad.download(&runtime).is_err());
    let rows = input.neighborhoods_resident(&runtime, 0.2, 3).unwrap();
    poison(&runtime, rows.rows().buffer());
    let bad = input.smooth_resident(&runtime, &rows, 2, 6).unwrap();
    assert_eq!(&read_words(&runtime, bad.status_buffer(), 32)[..5], [1, 2, 1, 1, 0]);
    assert!(read_words(&runtime, bad.buffer(), 65 * 16).iter().all(|&v| v == 0));
    assert!(bad.download(&runtime).is_err());
    let tiny = GpuMlsInput::upload(
        &runtime,
        &KdTree::from_slices(&[0.0, 1e-22, 2e-22], &[0.0; 3], &[0.0; 3]),
    )
    .unwrap();
    let rows = tiny.neighborhoods_resident(&runtime, 1.0, 3).unwrap();
    let bad = tiny.smooth_resident(&runtime, &rows, 1, 0).unwrap();
    assert_eq!(read_words(&runtime, bad.status_buffer(), 32)[4], 3);
    assert!(read_words(&runtime, bad.buffer(), 3 * 16).iter().all(|&v| v == 0));
    assert!(bad.download(&runtime).is_err());
    let rows = input.neighborhoods_resident(&runtime, 0.2, 3).unwrap();
    let good = input.smooth_resident(&runtime, &rows, 2, 6).unwrap().download(&runtime).unwrap();
    assert_eq!(good.diagnostics.unchanged, 65);
    assert_eq!(good.transfers.device_to_host_bytes(), 65 * 16 + 32);
}

#[test]
fn analytic_quadratic_fit_removes_the_query_spike_and_preserves_orders() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let (mut x, mut y, mut z) = (vec![], vec![], vec![]);
    for a in -2..=2 {
        for b in -2..=2 {
            let px = a as f32 * 0.05;
            let py = b as f32 * 0.05;
            x.push(px);
            y.push(py);
            z.push(if a == 0 && b == 0 { 0.02 } else { 0.5 * (px * px + py * py) });
        }
    }
    let input = GpuMlsInput::upload(&runtime, &KdTree::from_slices(&x, &y, &z)).unwrap();
    let rows = input.neighborhoods_resident(&runtime, 0.3, 25).unwrap();
    let linear = input.smooth_resident(&runtime, &rows, 1, 6).unwrap().download(&runtime).unwrap();
    let quad = input.smooth_resident(&runtime, &rows, 2, 6).unwrap().download(&runtime).unwrap();
    assert!(linear.positions[12][2] > 0.002);
    assert!(quad.positions[12][2].abs() < 1e-6, "quadratic center {:?}", quad.positions[12]);
    assert!(quad.diagnostics.polynomial > 0);
    assert_eq!(
        quad.diagnostics.unchanged + quad.diagnostics.plane_fallback + quad.diagnostics.polynomial,
        25
    );
    let sparse =
        input.smooth_resident(&runtime, &rows, 2, u32::MAX).unwrap().download(&runtime).unwrap();
    assert_eq!(sparse.diagnostics.unchanged, 25);
    for (i, p) in sparse.positions.iter().enumerate() {
        assert_eq!(p.map(f32::to_bits), [x[i].to_bits(), y[i].to_bits(), z[i].to_bits()]);
    }
}
