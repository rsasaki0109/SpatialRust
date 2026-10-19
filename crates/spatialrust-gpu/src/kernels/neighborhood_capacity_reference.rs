// Frozen complete parent pipeline and submitted search, native tests only.
use super::*;
pub(crate) struct ParentPipelines {
    pub(super) layout: wgpu::BindGroupLayout,
    pub(super) nearest: wgpu::ComputePipeline,
    pub(super) radius: wgpu::ComputePipeline,
}

impl ParentPipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..5)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: binding != 4 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("neighborhood-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("neighborhood-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("neighborhood-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("neighborhood_capacity_reference.wgsl").into(),
            ),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("neighborhood-pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self { nearest: pipeline("nearest_k"), radius: pipeline("radius_neighbors"), layout }
    }
}

pub(super) fn previous(
    cache: &OnceLock<ParentPipelines>,
    tree: &GpuKdTree,
    runtime: &WgpuRuntime,
    queries: &GpuQueryBatch,
    capacity: u32,
    bound: f32,
    kind: GpuNeighborhoodKind,
) -> SpatialResult<GpuNeighborhoods> {
    let identity = runtime.profiling_identity();
    if !Arc::ptr_eq(&tree.identity, &identity) || !Arc::ptr_eq(&queries.identity, &identity) {
        return Err(invalid());
    }
    let bytes =
        plan(u64::from(queries.count), u64::from(capacity), kind, &runtime.device().limits())?;
    let output = GpuNeighborhoods {
        buffer: runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("neighborhood-results"),
            size: bytes.max(8),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        queries: queries.count,
        capacity,
        bytes,
        kind,
        identity,
    };
    // Owned wgpu buffers start at zero, including unused slots and headers
    // on empty/invalid-gate paths. They never alias recycled scratch.
    if queries.count == 0
        || tree.root.is_none()
        || bound.is_nan()
        || (kind == GpuNeighborhoodKind::NearestK && capacity == 0)
    {
        return Ok(output);
    }
    let cache = cache.get_or_init(|| ParentPipelines::new(runtime.device()));
    let params = Params { count: queries.count, root: tree.root.unwrap(), capacity, bound };
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("neighborhood-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let buffers = [
        &uniform,
        tree.nodes.as_ref().unwrap(),
        tree.points.as_ref().unwrap(),
        queries.buffer.as_ref().unwrap(),
        &output.buffer,
    ];
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
        .collect();
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("neighborhood-bindings"),
        layout: &cache.layout,
        entries: &entries,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("neighborhood-search"),
            timestamp_writes: None,
        });
        pass.set_pipeline(if kind == GpuNeighborhoodKind::NearestK {
            &cache.nearest
        } else {
            &cache.radius
        });
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(queries.count.div_ceil(64), 1, 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
    Ok(output)
}
