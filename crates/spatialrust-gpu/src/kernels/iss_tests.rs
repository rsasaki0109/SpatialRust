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

#[test]
fn iss_plan_checks_counts_storage_bindings_and_dispatch() {
    let limits = wgpu::Limits::default();
    assert!(plan(u64::MAX, &limits).is_err());
    assert!(plan(u64::from(u32::MAX) + 1, &limits).is_err());
    let maximum = (u64::from(limits.max_storage_buffer_binding_size) / 16)
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
            4 => restricted.max_buffer_size = 31,
            _ => restricted.max_compute_workgroups_per_dimension = 0,
        }
        assert!(plan(1, &restricted).is_err());
    }
}

#[test]
fn iss_validates_upload_configuration_and_empty_receipts() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for x in [vec![f32::NAN], vec![f32::MAX, -f32::MAX]] {
        let zero = vec![0.0; x.len()];
        assert!(GpuIssInput::upload(&runtime, &KdTree::from_slices(&x, &zero, &zero)).is_err());
    }
    let input = GpuIssInput::upload(&runtime, &KdTree::from_slices(&[], &[], &[])).unwrap();
    assert_eq!(input.upload_transfers(), TransferStats::default());
    for radius in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX, f32::MIN_POSITIVE] {
        assert!(input.neighborhoods_resident(&runtime, radius, 1.0, 0, 0).is_err());
        assert!(input.neighborhoods_resident(&runtime, 1.0, radius, 0, 0).is_err());
    }
    let rows = input.neighborhoods_resident(&runtime, 1.0, 1.0, 0, 0).unwrap();
    for gamma in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
        assert!(input.keypoints_resident(&runtime, &rows, gamma, 0.9, 0).is_err());
        assert!(input.keypoints_resident(&runtime, &rows, 0.9, gamma, 0).is_err());
    }
    let output = input.keypoints_resident(&runtime, &rows, 0.9, 0.9, 0).unwrap();
    assert_eq!(output.point_count(), 0);
    assert_eq!(output.readback_bytes(), 0);
    assert_eq!(output.download(&runtime).unwrap(), (vec![], TransferStats::default()));
    assert_eq!(output.download_scores(&runtime).unwrap(), (vec![], TransferStats::default()));
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
fn incomplete_or_poisoned_graphs_zero_every_flag_and_score() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let x: Vec<_> = (0..65).map(|i| i as f32 * 0.125).collect();
    let zero = vec![0.0; 65];
    let input = GpuIssInput::upload(&runtime, &KdTree::from_slices(&x, &zero, &zero)).unwrap();
    for second in [false, true] {
        let rows = if second {
            input.neighborhoods_resident(&runtime, 0.2, 100.0, 3, 3).unwrap()
        } else {
            input.neighborhoods_resident(&runtime, 100.0, 0.01, 3, 1).unwrap()
        };
        let output = input.keypoints_resident(&runtime, &rows, 0.975, 0.975, 3).unwrap();
        assert_eq!(
            read_words(&runtime, output.status_buffer(), 32),
            if second { [0, 65, 0, 65, 0, 65, 0, 0] } else { [65, 0, 65, 0, 0, 65, 0, 0] }
        );
        assert!(read_words(&runtime, output.buffer(), 65 * 4).iter().all(|&v| v == 0));
        assert!(read_words(&runtime, output.scores_buffer(), 65 * 16).iter().all(|&v| v == 0));
        assert!(output.download(&runtime).is_err());
        assert!(output.download_scores(&runtime).is_err());
        let rows = input.neighborhoods_resident(&runtime, 0.2, 0.01, 3, 1).unwrap();
        poison(
            &runtime,
            if second { rows.non_max_rows().buffer() } else { rows.salient_rows().buffer() },
        );
        let output = input.keypoints_resident(&runtime, &rows, 0.975, 0.975, 3).unwrap();
        assert_eq!(
            read_words(&runtime, output.status_buffer(), 32),
            if second { [0, 1, 0, 1, 1, 1, 0, 0] } else { [1, 0, 2, 0, 1, 1, 0, 0] }
        );
        assert!(read_words(&runtime, output.buffer(), 65 * 4).iter().all(|&v| v == 0));
        assert!(read_words(&runtime, output.scores_buffer(), 65 * 16).iter().all(|&v| v == 0));
        assert!(output.download(&runtime).is_err());
    }
    let rows = input.neighborhoods_resident(&runtime, 0.2, 0.01, 3, 1).unwrap();
    let good = input.keypoints_resident(&runtime, &rows, 0.975, 0.975, 3).unwrap();
    assert_eq!(read_words(&runtime, good.status_buffer(), 32), [0; 8]);
    assert!(good.download(&runtime).unwrap().0.is_empty());
}

#[test]
fn analytic_box_eigenvalues_and_equal_maxima_survive() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let (mut x, mut y, mut z) = (vec![], vec![], vec![]);
    for px in [-0.06, 0.06] {
        for py in [-0.04, 0.04] {
            for pz in [-0.02, 0.02] {
                x.push(px);
                y.push(py);
                z.push(pz);
            }
        }
    }
    x.push(100.0);
    y.push(100.0);
    z.push(100.0);
    let tree = KdTree::from_slices(&x, &y, &z);
    let input = GpuIssInput::upload(&runtime, &tree).unwrap();
    let rows = input.neighborhoods_resident(&runtime, 0.2, 1.0, 9, 9).unwrap();
    let output = input.keypoints_resident(&runtime, &rows, 0.975, 0.975, 5).unwrap();
    assert_eq!(output.download(&runtime).unwrap().0, (0..8).collect::<Vec<_>>());
    let scores = read_words(&runtime, output.scores_buffer(), 9 * 16);
    for row in scores[..32].chunks_exact(4) {
        for (&bits, expected) in row[..3].iter().zip([0.0004, 0.0016, 0.0036]) {
            assert!((f32::from_bits(bits) - expected).abs() < 2e-9);
        }
        assert_eq!(f32::from_bits(row[3]), 1.0);
    }
    assert_eq!(scores[32..], [0; 4]);
    for (gamma, min_neighbors) in [(0.4, 5), (0.975, 9)] {
        assert!(input
            .keypoints_resident(&runtime, &rows, gamma, 0.975, min_neighbors)
            .unwrap()
            .download(&runtime)
            .unwrap()
            .0
            .is_empty());
    }
}
